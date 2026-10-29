"""Frozen Stage-1 vs Event-JEPA comparison on observed-interval MVSEC.

An intentionally controlled *lightweight probe*, NOT ScaleEvent/SEA-RAFT
full-supervised reproduction. Train outdoor_day2, final-only evaluation on
indoor_flying1/2/3. No checkpoint selection on any indoor test sequence.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dsec_flow_probe.backbone import FrozenFlowBackbone
from dsec_flow_probe.head import PatchLinearFlowHead
from mvsec_flow_probe.observed_dataset import MVSECObservedFlowDataset, to_native_flow
from mvsec_flow_probe.scaleevent_flow_metrics import ScaleEventFlowMeter


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    sha=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''): sha.update(b)
    return sha.hexdigest()


def masked_epe_loss(pred, gt, valid):
    err = torch.linalg.vector_norm(pred.float()-gt.float(),dim=1)
    return (err*valid.float()).sum()/valid.float().sum().clamp_min(1)


@torch.no_grad()
def run_eval(loader, backbone, head, device, seq):
    backbone.eval();head.eval()
    meter=ScaleEventFlowMeter()
    for batch in tqdm(loader,desc=f'TEST {seq}'):
        ctx=batch['context'].to(device,non_blocking=True)
        gt=batch['flow_gt'].to(device,non_blocking=True)
        gt_valid=batch['gt_valid'].to(device,non_blocking=True)
        active=batch['event_active'].to(device,non_blocking=True)
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=(device.type=='cuda')):
            z=backbone(ctx)
        native=to_native_flow(head(z.float()),height=gt.shape[-2],width=gt.shape[-1])
        meter.update(native,gt,gt_valid,active)
    result=meter.compute()
    print(seq, json.dumps(result,ensure_ascii=False))
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--mode',choices=['stage1','jepa'],required=True)
    p.add_argument('--checkpoint',type=Path,default=None,help='JEPA Stage2 step checkpoint, not stage1')
    p.add_argument('--stage1-ckpt',type=Path,required=True,help='Preprocessing provenance; checked and logged')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--num-workers',type=int,default=4)
    p.add_argument('--lr',type=float,default=5e-4)
    p.add_argument('--weight-decay',type=float,default=1e-4)
    p.add_argument('--max-train-samples',type=int,default=None,help='SMOKE ONLY; omit for formal')
    p.add_argument('--max-test-samples',type=int,default=None,help='SMOKE ONLY; omit for formal')
    p.add_argument('--skip-test',action='store_true',help='Training-only smoke without indoor data')
    p.add_argument('--eval-only',action='store_true')
    p.add_argument('--head-checkpoint',type=Path,default=None)
    a=p.parse_args()
    if a.mode=='jepa' and a.checkpoint is None: p.error('--checkpoint required for jepa')
    if a.eval_only and a.head_checkpoint is None: p.error('--head-checkpoint needed for --eval-only')
    if a.epochs<=0 or a.batch_size<=0: p.error('epochs and batch size must be positive')
    seed_all(a.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    if not a.stage1_ckpt.is_file(): raise FileNotFoundError(a.stage1_ckpt)
    target_stage1_sha = sha256_file(a.stage1_ckpt)
    to_check = ['outdoor_day2'] if a.skip_test and not a.eval_only else ['outdoor_day2','indoor_flying1','indoor_flying2','indoor_flying3']
    for seq in to_check:
        meta_path = a.root / 'processed_observed_v1' / seq / 'metadata.json'
        if not meta_path.is_file():
            raise FileNotFoundError(f'Missing preprocessing metadata: {meta_path}')
        meta = json.loads(meta_path.read_text())
        if meta.get('stage1_sha256') not in (None, target_stage1_sha):
            raise RuntimeError(f'Mismatched Stage-1 checkpoint in {seq} preprocessing')
    backbone=FrozenFlowBackbone(mode=a.mode,checkpoint=a.checkpoint).to(device)
    backbone.requires_grad_(False);backbone.eval()
    head=PatchLinearFlowHead().to(device)
    if a.eval_only:
        obj=torch.load(a.head_checkpoint,map_location='cpu',weights_only=False)
        head.load_state_dict(obj['head'],strict=True)
    else:
        train_ds=MVSECObservedFlowDataset(a.root,'outdoor_day2',max_samples=a.max_train_samples)
        loader=DataLoader(train_ds,batch_size=a.batch_size,shuffle=True,num_workers=a.num_workers,
                          pin_memory=(device.type=='cuda'),drop_last=False)
        optim=torch.optim.AdamW(head.parameters(),lr=a.lr,weight_decay=a.weight_decay)
        scheduler=torch.optim.lr_scheduler.StepLR(optim,step_size=5,gamma=0.9)
        t_start=time.time()
        for epoch in range(a.epochs):
            head.train(); loss_numer=0.;n=0
            for batch in tqdm(loader,desc=f'TRAIN {a.mode} epoch {epoch+1}/{a.epochs}'):
                ctx=batch['context'].to(device,non_blocking=True)
                gt=batch['flow_gt'].to(device,non_blocking=True)
                mask=batch['gt_valid'].to(device,non_blocking=True)
                with torch.no_grad():
                    with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=(device.type=='cuda')):
                        features=backbone(ctx)
                pred=to_native_flow(head(features.float()),height=gt.shape[-2],width=gt.shape[-1])
                loss=masked_epe_loss(pred,gt,mask)
                if not bool(torch.isfinite(loss).item()):
                    raise RuntimeError(f'nonfinite loss at epoch={epoch} - inspect flow target and predictions')
                optim.zero_grad(set_to_none=True)
                loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),0.1)
                optim.step()
                batch_n=ctx.size(0);loss_numer+=float(loss.item())*batch_n;n+=batch_n
            scheduler.step()
            print(f'epoch={epoch+1} mean_train_EPE_loss={loss_numer/max(n,1):.6f} lr={scheduler.get_last_lr()[0]:.3e}')
        obj={'head':head.state_dict(),'method':a.mode,'seed':a.seed,'epochs':a.epochs,
             'checkpoint':str(a.checkpoint) if a.checkpoint else None,
             'checkpoint_sha256':sha256_file(a.checkpoint) if a.checkpoint else None,
             'stage1_sha256':sha256_file(a.stage1_ckpt),
             'preprocessing':'processed_observed_v1 / full-FoV resize 224x224',
             'training_samples':len(train_ds),'duration_s':time.time()-t_start,
             'formal': a.max_train_samples is None and a.max_test_samples is None and not a.skip_test}
        torch.save(obj,a.output_dir/'head_final.pt')
        print('[SAVED]',a.output_dir/'head_final.pt')
    if a.skip_test:
        print('[TRAIN ONLY] --skip-test: no evaluation; NOT a formal result')
        return
    results={}
    for seq in ('indoor_flying1','indoor_flying2','indoor_flying3'):
        ds=MVSECObservedFlowDataset(a.root,seq,max_samples=a.max_test_samples)
        eval_loader=DataLoader(ds,batch_size=a.batch_size,shuffle=False,num_workers=a.num_workers,
                               pin_memory=(device.type=='cuda'))
        results[seq]=run_eval(eval_loader,backbone,head,device,seq)
    summary={'mode':a.mode,'source_commit':'c78a0f6','protocol':'observed-interval MVSEC full-FoV-resized224 frozen linear head',
             'test_only_at_final':True,'result_status':'SMOKE' if a.max_train_samples is not None or a.max_test_samples is not None else 'COMPLETE_LIGHTWEIGHT_PROBE',
             'results':results}
    (a.output_dir/'metrics.json').write_text(json.dumps(summary,indent=2))
    print('\n[RESULT] ',a.output_dir/'metrics.json')

if __name__=='__main__': main()
