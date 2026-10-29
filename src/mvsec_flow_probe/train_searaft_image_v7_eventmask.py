"""v7 single-controlled EVENT-ACTIVE training-mask experiment (not paper replication).

A frozen; B 10% labeled full fine-tune; C full full fine-tune; D stage2-only.
No test-set checkpoint selection; eval is optional and is final-only.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dsec_flow_probe.train_raw_finetune_formal import build_stage1, build_stage2
from mvsec_flow_probe.searaft_image_dataset_v6 import MVSECImageTimeFlowDataset as MVSECSEAFlowDataset
from mvsec_flow_probe.searaft_dataset import flow_to_model_coords
from mvsec_flow_probe.observed_dataset import to_native_flow
from mvsec_flow_probe.searaft_model import (EventJEPASEARAFT,add_searaft_core,
                                             training_policy)
from mvsec_flow_probe.scaleevent_flow_metrics import ScaleEventFlowMeter

TEST_SEQS=('indoor_flying1','indoor_flying2','indoor_flying3')


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def module_sha256(module):
    """Audit every module's actual state including trainable buffers."""
    h=hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        x=tensor.detach().cpu().contiguous()
        h.update(name.encode());h.update(str(tuple(x.shape)).encode())
        h.update(str(x.dtype).encode());h.update(x.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda:f.read(16*1024*1024),b''):
            h.update(x)
    return h.hexdigest()


def choose_ft10(n,seed):
    k=max(1,round(n*0.1))
    return sorted(map(int,np.random.default_rng(seed).choice(n,size=k,replace=False)))


def trainable_counts(model):
    return {k:{'total':sum(x.numel() for x in getattr(model,k).parameters()),
               'trainable':sum(x.numel() for x in getattr(model,k).parameters() if x.requires_grad)}
            for k in ('stage1','stage2','fmap_adapter','decoder')}


def initialize_searaft_decoder(decoder, checkpoint):
    """Transfer all architecture-compatible SEA-RAFT decoder weights.

    We cannot reuse RGB ResNet-FPN context/fnet layers in a ViT adapter.
    Fail loudly if any of the transferable decoder keys do not match.
    """
    path=Path(checkpoint)
    if not path.is_file(): raise FileNotFoundError(path)
    if path.suffix=='.safetensors':
        from safetensors.torch import load_file
        upstream=load_file(str(path),device='cpu')
    else:
        upstream=torch.load(path,map_location='cpu',weights_only=True)
        if not isinstance(upstream,dict):raise TypeError('Decoder weights are not a state dict')
        for key in ('state_dict','model_state_dict','model'):
            if key in upstream and isinstance(upstream[key],dict):
                upstream=upstream[key];break
    normalized={}
    for k,v in upstream.items():
        if not isinstance(v,torch.Tensor):continue
        k=k.removeprefix('module.').removeprefix('model.')
        k=k.removeprefix('decoder.')
        normalized[k]=v
    target=decoder.state_dict()
    required={k:v for k,v in target.items() if not k.startswith('cnet_adapter.')}
    loaded={k:v for k,v in normalized.items() if k in required and v.shape==required[k].shape}
    missing=sorted(set(required)-set(loaded))
    if missing:
        raise RuntimeError(f'SEA-RAFT pretrained decoder incompatible: '
                           f'{len(missing)} keys not found/mismatched; example={missing[:12]}. '
                           'Do not silently train from random initialization.')
    decoder.load_state_dict(loaded,strict=False)
    report={'path':str(path.resolve()),'sha256':sha256_file(path),
            'copied_tensors':len(loaded),'total_transferable_tensors':len(required),
            'random_initialized_adapters':[k for k in target if k.startswith('cnet_adapter.')],
            'note':'RGB backbone cannot be transplanted to ViT feature adapter'}
    print('[SEA-RAFT TRANSFER]',json.dumps(report),flush=True)
    return report


def optimizer_for(model, args):
    groups=[]
    for name,lr in [('stage1',args.lr_stage1),('stage2',args.lr_stage2),
                    ('fmap_adapter',args.lr_decoder),('decoder',args.lr_decoder)]:
        params=[p for p in getattr(model,name).parameters() if p.requires_grad]
        if params: groups.append({'params':params,'lr':lr,'name':name})
    return torch.optim.AdamW(groups,weight_decay=args.weight_decay)


def require_provenance(root,stage1, *, need_test):
    """Require native GT labels recomputed for consecutive image timestamps."""
    sequences=('outdoor_day2','indoor_flying1','indoor_flying2','indoor_flying3') if need_test else ('outdoor_day2',)
    summary={}
    for seq in sequences:
        meta=root/'processed_image_time_v6'/seq/'metadata.json'
        manifest=root/'processed_image_time_v6'/seq/'manifest.csv'
        if not meta.is_file() or not manifest.is_file():
            raise FileNotFoundError(f'Missing v6 data: {meta} or {manifest}')
        d=json.loads(meta.read_text())
        if d.get('candidate_mode')!='image_time_v6':
            raise RuntimeError(f'{seq}: wrong protocol metadata')
        if d.get('context')!=4:
            raise RuntimeError(f'{seq}: Stage2 checkpoint needs Tc=4')
        if d.get('flow_gt')!='composed from native 50ms EV-FlowNet GT in pixel units':
            raise RuntimeError(f'{seq}: unexpected flow label producer')
        with manifest.open(newline='') as f:
            actual=sum(1 for _ in csv.DictReader(f))
        if actual!=int(d['selected']):
            raise RuntimeError(f'{seq}: manifest length disagrees with metadata')
        summary[seq]={'samples':actual,'metadata':d}
    return summary


@torch.inference_mode()
def evaluate(model,root,batch_size,workers,device):
    model.eval()
    results={}
    for seq in TEST_SEQS:
        ds=MVSECSEAFlowDataset(root,seq)
        loader=DataLoader(ds,batch_size=batch_size,shuffle=False,
                          num_workers=workers,pin_memory=True,persistent_workers=workers>0)
        meter=ScaleEventFlowMeter()
        for batch in tqdm(loader,desc=f'FINAL {seq}'):
            events=batch['events'].to(device,non_blocking=True)
            gt=batch['flow_gt'].to(device,non_blocking=True)
            mask=batch['gt_valid'].to(device,non_blocking=True)
            active=batch['event_active'].to(device,non_blocking=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                pred224=model(events)['final']
            pred=to_native_flow(pred224.float(),height=gt.shape[-2],width=gt.shape[-1])
            meter.update(pred,gt,mask,active)
        results[seq]=meter.compute()
        print(f'[TEST] {seq}: {json.dumps(results[seq])}',flush=True)
    return results


def save_last(path,model,optimizer,scheduler,epoch,meta):
    obj={'model':model.state_dict(),'optimizer':optimizer.state_dict(),
         'scheduler':scheduler.state_dict(),'epoch':epoch,'metadata':meta}
    temp=path.with_suffix('.tmp')
    torch.save(obj,temp)
    os.replace(temp,path)
    print(f'[CHECKPOINT] {path} epoch={epoch}',flush=True)


def make_parser():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage1-ckpt',type=Path,required=True)
    p.add_argument('--stage2-ckpt',type=Path,required=True)
    p.add_argument('--decoder-init-ckpt',type=Path,default=None,help='SEA-RAFT upstream pretrained .safetensors/.pth; required for fresh training')
    p.add_argument('--sea-core',type=Path,default=Path('third_party/ScaleEvent/sea_raft/core'))
    p.add_argument('--protocol',choices=['frozen','ft10','full','stage2-only','stage1-only'],required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--grad-accum',type=int,default=8)
    p.add_argument('--num-workers',type=int,default=4)
    p.add_argument('--lr-stage1',type=float,default=1e-6)
    p.add_argument('--lr-stage2',type=float,default=1e-5)
    p.add_argument('--lr-decoder',type=float,default=2e-4)
    p.add_argument('--weight-decay',type=float,default=1e-4)
    p.add_argument('--gamma',type=float,default=0.85)
    p.add_argument('--clip',type=float,default=0.1)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--save-every',type=int,default=5)
    p.add_argument('--max-train-samples',type=int,default=None,help='DEBUG only; must omit for formal')
    p.add_argument('--disable-grad-checkpoint',action='store_true')
    p.add_argument('--skip-test',action='store_true',help='Train now, evaluate in a separate final-only run')
    p.add_argument('--eval-only',action='store_true')
    p.add_argument('--resume',type=Path)
    return p


def main():
    a=make_parser().parse_args()
    if min(a.epochs,a.batch_size,a.grad_accum)<1:
        raise ValueError('epochs, batch-size, grad-accum must be positive')
    if a.max_train_samples and not a.skip_test:
        raise ValueError('DEBUG --max-train-samples must use --skip-test')
    if a.eval_only and not a.resume:
        raise ValueError('--eval-only requires --resume path/to/last.pt')
    if not torch.cuda.is_available():
        raise RuntimeError('This formal SEA-RAFT script requires CUDA (A100 recommended)')
    device=torch.device('cuda:0')
    seed_all(a.seed)
    a.root=a.root.resolve()
    a.output_dir.mkdir(parents=True,exist_ok=True)
    add_searaft_core(a.sea_core)
    from loss import sequence_loss  # real SEA-RAFT Laplace-mixture loss
    source_meta=require_provenance(a.root,a.stage1_ckpt,need_test=(a.eval_only or not a.skip_test))
    for seq,record in source_meta.items():
        meta=record['metadata']
        if record['samples'] != int(meta['context_compatible']) and not a.max_train_samples:
            raise RuntimeError(f'{seq} incomplete v6 preprocessing: '
                               f'{record["samples"]}/{meta["context_compatible"]} labels. '
                               'Rerun pre_mvsec_image_time_v6.py without --max-samples.')
    if not a.max_train_samples and source_meta['outdoor_day2']['samples']<20000:
        raise RuntimeError('Training data suspiciously small; require 20k+ image-time flow pairs')
    for ckpt in (a.stage1_ckpt,a.stage2_ckpt):
        if not ckpt.is_file():raise FileNotFoundError(ckpt)
    print('[LOAD] Stage-1, Stage-2 strict pretrained checkpoints',flush=True)
    stage1=build_stage1(a.stage1_ckpt)
    stage2, stage2_cfg=build_stage2(a.stage2_ckpt)
    model=EventJEPASEARAFT(stage1,stage2,a.protocol,
            checkpoint_encoder=not a.disable_grad_checkpoint).to(device)
    if not a.resume and not a.eval_only and a.decoder_init_ckpt is None:
        raise ValueError('Fresh v6 training must provide --decoder-init-ckpt. '
                         'Pretrained SEA-RAFT decoder initialization is part of the experiment protocol.')
    init_report=(initialize_searaft_decoder(model.decoder,a.decoder_init_ckpt)
                 if a.decoder_init_ckpt and not a.resume else None)
    if init_report is not None:
        (a.output_dir/'decoder_init_report.json').write_text(json.dumps(init_report,indent=2))
    print('[POLICY]',a.protocol,training_policy(a.protocol),flush=True)
    count=trainable_counts(model)
    print('[PARAMS]',json.dumps(count),flush=True)
    if count['fmap_adapter']['trainable']==0 or count['decoder']['trainable']==0:
        raise RuntimeError('SEA-RAFT adapter/decoder must always be trainable')
    optimizer=optimizer_for(model,a)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.9)
    starts=0
    if a.resume:
        st=torch.load(a.resume,map_location='cpu',weights_only=False)
        prior=st['metadata']
        if prior.get('data_protocol')!='mvsec_image_timestamp_v6_candidate_not_official_replication':
            raise ValueError('resume must use image timestamp v6 checkpoint, not an old observed-v1 checkpoint')
        if prior['protocol']!=a.protocol or prior['seed']!=a.seed:
            raise ValueError('resume checkpoint protocol/seed mismatch')
        model.load_state_dict(st['model'],strict=True)
        if not a.eval_only:
            optimizer.load_state_dict(st['optimizer'])
            scheduler.load_state_dict(st['scheduler'])
            starts=int(st['epoch'])
        print(f'[RESUME] loaded {a.resume}, next epoch {starts+1}',flush=True)
    hashes_before={name:module_sha256(getattr(model,name))
        for name in ('stage1','stage2','fmap_adapter','decoder')}
    cfg={'protocol':a.protocol,'seed':a.seed,'epochs':a.epochs,
         'stage1_ckpt_sha256':sha256_file(a.stage1_ckpt),
         'stage2_ckpt_sha256':sha256_file(a.stage2_ckpt),
         'checkpoint_code_commit':'upstream_snapshot_c78a0f6_based',
         'train_sequence':'outdoor_day2',
         'eval_sequences':list(TEST_SEQS),
         'decoder':'ScaleEvent sea_raft/core update+loss with Event-JEPA 1/8 feature adapter',
         'representation':'GEP Stage-1-only, direct tokens' if a.protocol=='stage1-only' else 'Event-JEPA Stage1+Stage2',
         'event_input':'four consecutive image-time event intervals, latest matches GT; outdoor ~45Hz, indoor ~31Hz',
         'input_resolution':[224,224],'flow_gt_resolution':[260,346],
         'data_protocol':'mvsec_image_timestamp_v6_candidate_not_official_replication',
         'loss_mask_protocol':'v7_gt_valid_and_target_interval_event_active_nn224',
         'data_provenance':source_meta,
         'sea_raft_pretrained_decoder':init_report if init_report is not None else ('resumed_checkpoint' if a.resume else None),

         'input_note':'non-isotropic full-FoV resizing; upstream GT integrated at image timestamps; NOT proven official ScaleEvent sample filtering',
         'hyperparameters':{'lr_stage1':a.lr_stage1,'lr_stage2':a.lr_stage2,
                            'lr_decoder':a.lr_decoder,'batch_size':a.batch_size,
                            'grad_accum':a.grad_accum,'gamma':a.gamma,
                            'weight_decay':a.weight_decay,'gradient_clip':a.clip},
         'stage1_params_trainable':count['stage1']['trainable'],
         'stage2_params_trainable':count['stage2']['trainable']}
    # The dataset checks all required event input PNG and masks are available.
    indices=None
    all_rows_file=a.root/'processed_image_time_v6'/'outdoor_day2'/'manifest.csv'
    import csv
    with all_rows_file.open(newline='') as f:
        n=len(list(csv.DictReader(f)))
    if a.protocol=='ft10':indices=choose_ft10(n,a.seed)
    elif a.max_train_samples: indices=list(range(min(n,a.max_train_samples)))
    if a.max_train_samples and indices is not None:
        indices=indices[:a.max_train_samples]
    cfg['full_train_samples']=n
    cfg['train_samples']=len(indices) if indices is not None else n
    cfg['label_fraction']=0.1 if a.protocol=='ft10' else 1.0
    cfg['ft10_indices_sha256']=(hashlib.sha256(np.asarray(indices,dtype=np.int64).tobytes()).hexdigest()
                                  if a.protocol=='ft10' else None)
    cfg['official_final']=not bool(a.max_train_samples)
    print('[DATA]',n,'full labels;',cfg['train_samples'],'selected;',
          'test deferred' if a.skip_test else 'test after training',flush=True)
    (a.output_dir/'run_config.json').write_text(json.dumps(cfg,indent=2,ensure_ascii=False))
    if not a.eval_only:
        dataset=MVSECSEAFlowDataset(a.root,'outdoor_day2',indices=indices)
        loader=DataLoader(dataset,batch_size=a.batch_size,shuffle=True,
                num_workers=a.num_workers,pin_memory=True,
                persistent_workers=a.num_workers>0,drop_last=False)
        model.train()
        hist_path=a.output_dir/'train_history.json'
        trainlog=json.loads(hist_path.read_text()) if a.resume and hist_path.is_file() else []
        start=time.time()
        for epoch in range(starts,a.epochs):
            loss_sum=0.; n_obs=0
            total_batches=len(loader)
            optimizer.zero_grad(set_to_none=True)
            for i,batch in enumerate(tqdm(loader,desc=f'{a.protocol} {epoch+1}/{a.epochs}')):
                events=batch['events'].to(device,non_blocking=True)
                gtn=batch['flow_gt'].to(device,non_blocking=True)
                maskn=batch['gt_valid'].to(device,non_blocking=True)
                gt,valid=flow_to_model_coords(gtn,maskn)
                # v7 controlled diagnostic: train on the same spatial EVENT-ACTIVE
                # definition used in v6 evaluation. Keep all other choices fixed.
                # The v6 source remains untouched, and the v7 output is separate.
                active_n=batch['event_active'].to(device,non_blocking=True)
                active_224=torch.nn.functional.interpolate(
                    active_n[:,None].float(),size=valid.shape[-2:],mode='nearest')[:,0]>0.5
                valid=valid & active_224
                if not valid.any():
                    raise RuntimeError('This training batch has zero event-active GT-valid pixels. '
                                       'Do not silently switch masks: inspect dataset and sampling.')
                if not (valid & (torch.linalg.vector_norm(gt,dim=1)<400)).any():
                    raise RuntimeError('training batch has no valid GT pixels')
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    output=model(events,flow_gt=gt)
                    raw_loss=sequence_loss(output,gt,valid.float(),gamma=a.gamma)
                if not torch.isfinite(raw_loss):
                    raise RuntimeError(f'nonfinite loss at epoch {epoch} batch {i}')
                # Correct gradient accumulation even at the last partial group
                group_start=(i//a.grad_accum)*a.grad_accum
                group_size=min(a.grad_accum,total_batches-group_start)
                (raw_loss/group_size).backward()
                loss_sum+=float(raw_loss.detach().item())*events.shape[0]
                n_obs+=events.shape[0]
                if (i+1)%a.grad_accum==0 or i+1==total_batches:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],a.clip)
                    optimizer.step();optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            info={'epoch':epoch+1,'train_loss':loss_sum/max(n_obs,1),
                  'observations':n_obs,'elapsed_sec':time.time()-start,
                  'max_reserved_GB':torch.cuda.max_memory_reserved()/1e9}
            trainlog.append(info)
            (a.output_dir/'train_history.json').write_text(json.dumps(trainlog,indent=2))
            print('[EPOCH]',json.dumps(info),flush=True)
            if (epoch+1)%a.save_every==0 or epoch+1==a.epochs:
                save_last(a.output_dir/'last.pt',model,optimizer,scheduler,epoch+1,cfg)
        hashes_after={name:module_sha256(getattr(model,name))
            for name in ('stage1','stage2','fmap_adapter','decoder')}
        expected=training_policy(a.protocol)
        expected_change={'stage1':expected['stage1'],'stage2':expected['stage2'],
                         'fmap_adapter':True,'decoder':True}
        actual_change={name:hashes_after[name]!=hashes_before[name]
                       for name in expected_change}
        audit={'before':hashes_before,'after':hashes_after,
               'changed':actual_change,'expected_changed':expected_change}
        (a.output_dir/'parameter_audit.json').write_text(json.dumps(audit,indent=2))
        print('[PARAMETER AUDIT]',json.dumps(actual_change),flush=True)
        if actual_change!=expected_change:
            raise RuntimeError(f'parameter audit mismatch: {audit}')
        print('[TRAIN DONE] output:',a.output_dir,flush=True)
    if a.skip_test:
        print('[TRAIN ONLY] Indoor test not touched. Run --eval-only after indoor data are ready.',flush=True)
        return
    if a.eval_only:
        print('[FINAL EVAL] loaded checkpoint:',a.resume,flush=True)
    scores=evaluate(model,a.root,batch_size=a.batch_size,workers=a.num_workers,device=device)
    (a.output_dir/'metrics_final.json').write_text(json.dumps({'config':cfg,'results':scores},indent=2))
    print('[FINAL REPORT]',a.output_dir/'metrics_final.json',flush=True)

if __name__=='__main__':
    main()
