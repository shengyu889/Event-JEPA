#!/usr/bin/env python3
"""Full-pretrained SEA-RAFT event-domain baseline and add-on GEP/JEPA fusion.

This is a *new, isolated engineering baseline*, not official ScaleEvent replication.
Original SEA-RAFT weights, fnet and cnet are strictly loaded (no partial-load).
Paired eventRGB inputs and native GT share MVSEC processed_image_time_v6.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

NATIVE_H, NATIVE_W = 260, 346
TEST_SEQS = ('indoor_flying1', 'indoor_flying2', 'indoor_flying3')
DSEC_MEAN = (0.8993729784963826, 0.7969581014619264, 0.8928228776286392)
DSEC_STD = (0.2726268701553345, 0.2706460952758789, 0.2812058925628662)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for buf in iter(lambda: f.read(8 << 20), b''):
            h.update(buf)
    return h.hexdigest()


def load_native_rgb(base, interval_index):
    p = base/'eventRGB_native'/f'{interval_index:06d}.png'
    with Image.open(p) as im:
        rgb = np.array(im.convert('RGB'), dtype=np.uint8, copy=True)
    if rgb.shape != (NATIVE_H, NATIVE_W, 3):
        raise ValueError(f'{p}: expected 260x346 event RGB; got {rgb.shape}')
    return torch.from_numpy(rgb).permute(2, 0, 1).contiguous()


class ImageTimeV6Pairs(Dataset):
    def __init__(self, root, seq, *, limit=None, contexts=False):
        self.base = Path(root)/'processed_image_time_v6'/seq
        meta = json.loads((self.base/'metadata.json').read_text())
        if meta.get('candidate_mode') != 'image_time_v6':
            raise RuntimeError(f'{seq}: unexpected GT protocol')
        with (self.base/'manifest.csv').open(newline='') as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise RuntimeError(f'{seq}: empty v6 manifest')
        if limit is not None and limit > 0 and len(rows) > limit:
            ids = np.linspace(0, len(rows)-1, int(limit), dtype=np.int64)
            rows = [rows[int(i)] for i in ids]
        self.rows = rows
        self.contexts = contexts
        for r in self.rows:
            i = int(r['pair_index'])
            c = [int(r[f'context_{k}']) for k in range(4)]
            if c != [i-2, i-1, i, i+1]:
                raise RuntimeError(f'Invalid v6 temporal alignment: {seq}:{i}, {c}')
            if not (self.base/r['flow_file']).is_file():
                raise FileNotFoundError(self.base/r['flow_file'])
        needed = set()
        for r in self.rows:
            inds = range(4) if contexts else (2,3)
            needed.update(int(r[f'context_{k}']) for k in inds)
        for i in needed:
            p=self.base/'eventRGB_native'/f'{i:06d}.png'
            if not p.is_file(): raise FileNotFoundError(p)

    def __len__(self): return len(self.rows)

    def __getitem__(self, ix):
        row = self.rows[ix]
        c = [int(row[f'context_{k}']) for k in range(4)]
        frames = [load_native_rgb(self.base, j) for j in c] if self.contexts else [
            load_native_rgb(self.base, c[2]), load_native_rgb(self.base, c[3])]
        with np.load(self.base/row['flow_file'], allow_pickle=False) as d:
            gt = d['flow'].astype(np.float32, copy=True)
            valid = d['valid'].astype(bool, copy=True)
            active = d['event_active'].astype(bool, copy=True)
        if gt.shape != (2, NATIVE_H, NATIVE_W) or valid.shape != (NATIVE_H, NATIVE_W) or active.shape != valid.shape:
            raise ValueError('Native flow/mask dimension mismatch')
        if not np.isfinite(gt).all():raise ValueError('Non-finite GT')
        return {
            'pair':torch.stack(frames[-2:]),
            'context':torch.stack(frames) if self.contexts else torch.empty(0, dtype=torch.uint8),
            'flow':torch.from_numpy(gt),
            'valid':torch.from_numpy(valid),
            'active':torch.from_numpy(active),
            'pair_index':int(row['pair_index']),
        }


def load_upstream_searaft(core, config):
    core = Path(core).resolve()
    for fn in ('raft.py','extractor.py','update.py','layer.py','corr.py','loss.py','utils/utils.py'):
        if not (core/fn).is_file(): raise FileNotFoundError(core/fn)
    # Critically: SEA-RAFT expects 'utils' as a PACKAGE. Do not import
    # Event-JEPA src/utils.py prior to this call.
    preloaded = sys.modules.get('utils')
    if preloaded is not None and not hasattr(preloaded, '__path__'):
        raise RuntimeError('Import conflict: src/utils.py already imported as utils. '
                           'Start fresh process with SEA-RAFT core loaded first.')
    sys.path.insert(0, str(core))
    from raft import RAFT
    from loss import sequence_loss
    cfg = json.loads(Path(config).read_text())
    if cfg.get('dim') != 128 or cfg.get('iters') != 4:
        raise ValueError('v8 requires upstream spring-M config (dim=128, iters=4)')
    return RAFT, sequence_loss, SimpleNamespace(**cfg)


def restore_shared_batchnorm_aliases(raft, weights):
    """Restore *identical* state_dict aliases removed by safetensors deduplication.

    SEA-RAFT BasicBlock registers one BN as both `bn3` and `downsample[1]`.
    Some safetensors model exports store only `bn3.*`, while PyTorch's
    state_dict lists both paths. This helper ONLY adds provable aliases.
    Genuine missing tensors remain missing and strict=True still rejects them.
    """
    expected = raft.state_dict()
    weights = dict(weights)
    recovered = []
    for path, block in raft.named_modules():
        bn = getattr(block, 'bn3', None)
        downsample = getattr(block, 'downsample', None)
        if not isinstance(bn, nn.BatchNorm2d):
            continue
        if not isinstance(downsample, nn.Sequential) or len(downsample) <= 1:
            continue
        if downsample[1] is not bn:
            continue
        for field in bn.state_dict().keys():
            src_key = f'{path}.bn3.{field}'
            dst_key = f'{path}.downsample.1.{field}'
            if dst_key not in expected or dst_key in weights:
                continue
            if src_key not in weights:
                raise RuntimeError(
                    f'Cannot restore shared BN alias {dst_key}: '
                    f'checkpoint also lacks its actual source {src_key}'
                )
            src_tensor = weights[src_key]
            expected_tensor = expected[dst_key]
            if (src_tensor.shape != expected_tensor.shape
                    or src_tensor.dtype != expected_tensor.dtype):
                raise RuntimeError(
                    f'Alias source tensor mismatch: {src_key} -> {dst_key}; '
                    f'checkpoint {tuple(src_tensor.shape)}/{src_tensor.dtype}, '
                    f'model {tuple(expected_tensor.shape)}/{expected_tensor.dtype}'
                )
            # Check model really shares the same storage for both key paths.
            if src_key not in expected or (
                    expected[src_key].data_ptr() != expected_tensor.data_ptr()):
                raise RuntimeError(
                    f'Unsafe alias recovery refused: {src_key} and '
                    f'{dst_key} are not the same model tensor'
                )
            weights[dst_key] = src_tensor
            recovered.append(dst_key)
    return weights, recovered


def load_full_weights(raft, ckpt):
    ckpt=Path(ckpt)
    if not ckpt.is_file(): raise FileNotFoundError(ckpt)
    if ckpt.suffix == '.safetensors':
        from safetensors.torch import load_file
        weights = load_file(str(ckpt), device='cpu')
    else:
        weights = torch.load(str(ckpt), map_location='cpu', weights_only=True)
        for field in ('state_dict','model_state_dict','model'):
            if isinstance(weights,dict) and isinstance(weights.get(field),dict):
                weights=weights[field];break
    if not isinstance(weights, dict): raise TypeError('Invalid checkpoint contents')
    weights={k.removeprefix('module.').removeprefix('model.'):v for k,v in weights.items() if torch.is_tensor(v)}
    weights, restored_aliases = restore_shared_batchnorm_aliases(raft, weights)
    if restored_aliases:
        print('[RESTORED SAFETENSORS SHARED BN ALIASES]',
              json.dumps(restored_aliases), flush=True)
    incompatible=raft.load_state_dict(weights, strict=True)
    keys=set(raft.state_dict())
    if not any(k.startswith('fnet.') for k in keys) or not any(k.startswith('cnet.') for k in keys):
        raise RuntimeError('A pretrained SEA-RAFT must include BOTH fnet and cnet')
    report={'loaded_state_tensors':len(keys),
            'fnet_tensors':len([k for k in keys if k.startswith('fnet.')]),
            'cnet_tensors':len([k for k in keys if k.startswith('cnet.')]),
            'missing':list(incompatible.missing_keys),
            'unexpected':list(incompatible.unexpected_keys),
            'sha256':sha256(ckpt),'path':str(ckpt.resolve()),
            'restored_shared_aliases': restored_aliases}
    print('[STRICT FULL RAFT LOAD]', json.dumps(report), flush=True)
    return report


def load_event_pretrained(stage1_ckpt, stage2_ckpt, mode):
    from dinov2.models.vision_transformer import vit_small
    from event_jepa.encoders import EventTokenEncoder
    w=torch.load(stage1_ckpt, map_location='cpu', weights_only=False)
    stage1=vit_small(patch_size=14,img_size=518,block_chunks=0,init_values=1e-6,num_register_tokens=0)
    stage1.load_state_dict(w['event_encoder'], strict=True)
    stage2=None
    if mode=='jepa':
        w2=torch.load(stage2_ckpt,map_location='cpu',weights_only=False)
        c=w2['config']
        if (int(c['context_frames']),int(c['n_tokens']),int(c['embed_dim'])) != (4,256,384):
            raise RuntimeError('JEPA checkpoint must be Tc4,N256,D384')
        stage2=EventTokenEncoder(embed_dim=int(c['embed_dim']),num_heads=int(c['num_heads']),
             num_layers=int(c['encoder_layers']),n_tokens=int(c['n_tokens']),
             max_positions=int(c['max_positions']),max_context_frames=int(c['max_context_frames']))
        stage2.load_state_dict(w2['online_encoder'],strict=True)
    return stage1,stage2


class ResidualEventFusion(nn.Module):
    """Keep every pretrained SEA-RAFT parameter and inject Event tokens via hooks.

    fnet is shared and called twice. cnet uses a 6-channel image pair.
    Fusion heads project 16x16 tokens to their EXISTING 1/8 spatial grid;
    they do not replace feature maps and cannot be mistaken for full SEA-RAFT init.
    """
    def __init__(self, raft, stage1, stage2, mode='jepa', train_encoders=False):
        super().__init__()
        if mode not in ('gep','jepa'):raise ValueError(mode)
        self.raft=raft
        self.stage1=stage1
        self.stage2=stage2
        self.mode=mode
        self.train_encoders=train_encoders
        self.stage1.requires_grad_(train_encoders)
        if self.stage2 is not None:self.stage2.requires_grad_(train_encoders)
        feature_ch=raft.args.dim*2
        self.fusion_fnet=nn.Sequential(nn.Conv2d(384,feature_ch,1),nn.GELU(),nn.Conv2d(feature_ch,feature_ch,3,padding=1))
        self.fusion_cnet=nn.Sequential(nn.Conv2d(768,feature_ch,1),nn.GELU(),nn.Conv2d(feature_ch,feature_ch,3,padding=1))
        # Small but nonzero gate allows gradients through adapters on step one.
        self.gate_fnet=nn.Parameter(torch.tensor(0.05))
        self.gate_cnet=nn.Parameter(torch.tensor(0.05))

    def train(self, mode=True):
        super().train(mode)
        if not self.train_encoders:
            self.stage1.eval()
            if self.stage2 is not None:self.stage2.eval()
        return self

    def event_tokens(self, context):
        # context: B,T,3,260,346 uint8. Stage1 expects DSEC-normalized 224x224.
        b,t,c,h,w=context.shape
        if (t,c,h,w)!=(4,3,260,346):raise ValueError('Expected four native event windows')
        inp=context.reshape(b*t,3,h,w).float()/255.0
        inp=F.interpolate(inp,size=(224,224),mode='bilinear',align_corners=False)
        mean=inp.new_tensor(DSEC_MEAN).view(1,3,1,1)
        std=inp.new_tensor(DSEC_STD).view(1,3,1,1)
        inp=(inp-mean)/std
        def encode(x):return self.stage1.forward_features(x)['x_norm_patchtokens']
        if self.train_encoders:
            from torch.utils.checkpoint import checkpoint
            z=checkpoint(encode,inp,use_reentrant=False)
        else:
            with torch.no_grad():z=encode(inp)
        z=z.reshape(b,4,256,384)
        if self.mode=='jepa':
            if self.train_encoders:
                z=self.stage2(z)
            else:
                with torch.no_grad():z=self.stage2(z)
            if z.shape!=(b,1024,384):raise RuntimeError(f'Invalid JEPA z={z.shape}')
            z=z.reshape(b,4,256,384)
        return z[:,-2:].reshape(b,2,16,16,384).permute(0,1,4,2,3)

    def forward(self, pair, *, context, flow_gt=None, test_mode=False):
        tokens=self.event_tokens(context)
        b=pair.shape[0]
        fmap_input=tokens.reshape(b*2,384,16,16)
        match=self.fusion_fnet(fmap_input)
        match=match.reshape(b,2,match.shape[1],16,16)
        cc=self.fusion_cnet(torch.cat([tokens[:,0],tokens[:,1]],dim=1))
        called=[0]
        def hook_fnet(_module,_inputs,output):
            idx=called[0];called[0]+=1
            if idx not in (0,1):raise RuntimeError('SEA-RAFT fnet called >2 times')
            residual=F.interpolate(match[:,idx],size=output.shape[-2:],mode='bilinear',align_corners=False)
            return output + self.gate_fnet*residual
        def hook_cnet(_module,_inputs,output):
            residual=F.interpolate(cc,size=output.shape[-2:],mode='bilinear',align_corners=False)
            return output + self.gate_cnet*residual
        h1=self.raft.fnet.register_forward_hook(hook_fnet)
        h2=self.raft.cnet.register_forward_hook(hook_cnet)
        try:
            out=self.raft(pair[:,0].float(),pair[:,1].float(),flow_gt=flow_gt,test_mode=test_mode)
            if called[0]!=2:raise RuntimeError(f'Expected fnet twice, got {called[0]}')
            return out
        finally:
            h1.remove();h2.remove()


def freeze_full_raft(model):
    """Freeze all SEA-RAFT weights AND BN running statistics.

    Call after every recursive `model.train()` so RAFT remains in eval mode.
    Gradient still propagates through RAFT operations to event fusion adapters.
    """
    if not isinstance(model, ResidualEventFusion):
        raise TypeError('Frozen SEA-RAFT mode requires a ResidualEventFusion')
    model.raft.requires_grad_(False)
    model.raft.eval()


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def create_model(args):
    RAFT,loss_fn,config=load_upstream_searaft(args.sea_core,args.sea_config)
    core=RAFT(config)
    report=load_full_weights(core,args.raft_ckpt)
    if args.base_ckpt:
        parent=torch.load(args.base_ckpt,map_location='cpu',weights_only=False)
        if parent.get('config',{}).get('fusion')!='none':
            raise RuntimeError('--base-ckpt must refer to a trained FULL SEA-RAFT baseline (fusion=none)')
        core.load_state_dict(parent['model'],strict=True)
        report['finetuned_base_checkpoint']=str(Path(args.base_ckpt).resolve())
        report['finetuned_base_sha256']=sha256(args.base_ckpt)
        print('[RESTORED DOMAIN-ADAPTED FULL BASE]',args.base_ckpt,flush=True)
    if args.fusion=='none':return core,loss_fn,report
    if not args.stage1_ckpt:raise ValueError('--stage1-ckpt required for GEP/JEPA fusion')
    if args.fusion=='jepa' and not args.stage2_ckpt:raise ValueError('--stage2-ckpt required for JEPA fusion')
    stage1,stage2=load_event_pretrained(args.stage1_ckpt,args.stage2_ckpt,args.fusion)
    model=ResidualEventFusion(core,stage1,stage2,args.fusion,train_encoders=args.finetune_event_encoders)
    return model,loss_fn,report


def forward_model(model, batch, device, fusion, *, flow_gt=None, test_mode=False):
    pair=batch['pair'].to(device,non_blocking=True)
    if fusion=='none':
        return model(pair[:,0].float(),pair[:,1].float(),flow_gt=flow_gt,test_mode=test_mode)
    ctx=batch['context'].to(device,non_blocking=True)
    return model(pair,context=ctx,flow_gt=flow_gt,test_mode=test_mode)


def eval_sequence(model, seq, root, device, fusion, limit=None, workers=2):
    from mvsec_flow_probe.scaleevent_flow_metrics import ScaleEventFlowMeter
    ds=ImageTimeV6Pairs(root,seq,limit=limit,contexts=fusion!='none')
    loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=workers,pin_memory=True)
    model.eval()
    meter=ScaleEventFlowMeter()
    zero_sum=0.;num=0;pred_mag_sum=0.;cos_sum=0.;cos_n=0
    for batch in loader:
        gt=batch['flow'].to(device)
        valid=batch['valid'].to(device)
        active=batch['active'].to(device)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.type=='cuda'):
            out=forward_model(model,batch,device,fusion,test_mode=True)
        pred=out['final'].float()
        if pred.shape!=gt.shape:raise RuntimeError(f'Prediction {pred.shape} != native GT {gt.shape}')
        meter.update(pred,gt,valid,active)
        mask=(valid & active)
        if bool(mask.any()):
            z=torch.linalg.vector_norm(gt.float(),dim=1)
            p=torch.linalg.vector_norm(pred,dim=1)
            zero_sum+=z[mask].sum().item();pred_mag_sum+=p[mask].sum().item();num+=mask.sum().item()
            cos=F.cosine_similarity(pred.float(),gt.float(),dim=1,eps=1e-6)
            cvalid=mask&(z>1e-3)&(p>1e-3)
            cos_sum+=cos[cvalid].sum().item();cos_n+=cvalid.sum().item()
    result=meter.compute()
    result.update({'zero_flow_EPE_same_mask':zero_sum/max(num,1),
                   'pred_magnitude_mean':pred_mag_sum/max(num,1),
                   'mean_cosine_nonzero':cos_sum/max(cos_n,1),
                   'joint_valid_pixels_diagnostic':int(num),
                   'eval_samples':len(ds)})
    print(f'[EVAL] {seq} {json.dumps(result,ensure_ascii=False)}',flush=True)
    return result


def save_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False));os.replace(tmp,path)


def run(args):
    if not torch.cuda.is_available():raise RuntimeError('This command requires an NVIDIA GPU')
    seed_all(args.seed)
    device=torch.device('cuda:0')
    if args.mode=='eval-pretrained' and args.fusion!='none':
        raise ValueError('eval-pretrained only supports the unchanged original SEA-RAFT (fusion=none)')
    if args.mode=='train' and args.fusion!='none' and not args.base_ckpt and not args.freeze_raft:
        raise ValueError('Fusion training without --base-ckpt is allowed only with --freeze-raft, starting from STRICT pretrained SEA-RAFT')
    if args.freeze_raft and (args.mode != 'train' or args.fusion == 'none'):
        raise ValueError('--freeze-raft is only valid in fusion training (GEP/JEPA)')
    model,loss_fn,init_report=create_model(args)
    model=model.to(device)
    if args.freeze_raft:
        # Pretrained RAFT is a fixed, shared zero-shot teacher/base. All RAFT
        # parameters and BN running statistics must remain exactly unchanged.
        freeze_full_raft(model)
        print('[FROZEN FULL RAFT] params frozen; BatchNorm stats frozen; '
              'trained parameters are fusion adapters/gates only',flush=True)
    run_cfg={'mode':args.mode,'fusion':args.fusion,'seed':args.seed,
        'data_protocol':'mvsec_image_timestamp_v6_candidate_not_official_replication',
        'model':'Full original SEA-RAFT backbone+decoder' if args.fusion=='none' else f'Full SEA-RAFT + residual {args.fusion.upper()} fusion',
        'input':'two adjacent native 260x346 eventRGB event windows; NOT true image frames',
        'pretrained_full_raft':init_report,'epochs':args.epochs,
        'lr_raft':args.lr_raft,'lr_fusion':args.lr_fusion,'lr_stage1':args.lr_stage1,
        'lr_stage2':args.lr_stage2,'supervision':'gt_valid (match v6); eval joint gt_valid & event_active',
        'freeze_raft':args.freeze_raft, 'batch_size':args.batch_size,
        'grad_accum':args.grad_accum, 'base_ckpt':str(args.base_ckpt) if args.base_ckpt else None,
        'finetune_event_encoders':args.finetune_event_encoders,
        'note':'ENGINEERING baseline and fusion. Not official ScaleEvent replication.'}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.mode=='train' and (args.output_dir/'last.pt').exists():
        raise FileExistsError(f'Existing checkpoint would be overwritten: {args.output_dir / 'last.pt'}')
    if args.mode=='eval-pretrained' and (args.output_dir/'last.pt').exists():
        raise FileExistsError('Evaluation output directory contains a training checkpoint')
    save_json(args.output_dir/'run_config.json',run_cfg)
    if args.mode=='train':
        ds=ImageTimeV6Pairs(args.root,'outdoor_day2',limit=args.max_train_samples,contexts=args.fusion!='none')
        loader=DataLoader(ds,batch_size=args.batch_size,shuffle=True,num_workers=args.workers,
                    persistent_workers=args.workers>0,pin_memory=True)
        if args.fusion=='none':
            groups=[{'params':model.parameters(),'lr':args.lr_raft}]
        else:
            groups=[{'params':model.raft.parameters(),'lr':args.lr_raft},
                    {'params':list(model.fusion_fnet.parameters())+list(model.fusion_cnet.parameters())+
                    [model.gate_fnet,model.gate_cnet], 'lr':args.lr_fusion}]
            if args.finetune_event_encoders:
                groups.append({'params':model.stage1.parameters(),'lr':args.lr_stage1})
                if model.stage2 is not None:groups.append({'params':model.stage2.parameters(),'lr':args.lr_stage2})
        opt=torch.optim.AdamW(groups,weight_decay=1e-4)
        schedule=torch.optim.lr_scheduler.StepLR(opt,step_size=5,gamma=0.9)
        history=[]
        for ep in range(args.epochs):
            model.train()
            if args.freeze_raft:
                # Recursive model.train() above would otherwise re-enable all
                # RAFT BatchNorm updates. This is essential for fixed-base study.
                freeze_full_raft(model)
            total_loss=0.;n=0;norm_epe=0.;zero_sum=0.;n_pix=0
            opt.zero_grad(set_to_none=True)
            n_batches=len(loader)
            for bi,b in enumerate(loader):
                gt=b['flow'].to(device,non_blocking=True)
                valid=b['valid'].to(device,non_blocking=True)
                if not valid.any():raise RuntimeError('No valid GT in training batch')
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    out=forward_model(model,b,device,args.fusion,flow_gt=gt,test_mode=False)
                    loss=loss_fn(out,gt,valid.float(),gamma=0.85)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                gstart=(bi//args.grad_accum)*args.grad_accum
                denom=min(args.grad_accum,n_batches-gstart)
                (loss/denom).backward()
                total_loss+=float(loss.detach())*gt.shape[0];n+=gt.shape[0]
                with torch.no_grad():
                    active=(valid&b['active'].to(device))
                    if bool(active.any()):
                        epe=torch.linalg.vector_norm(out['final'].float()-gt.float(),dim=1)
                        norm_epe+=epe[active].sum().item()
                        zero_sum+=torch.linalg.vector_norm(gt.float(),dim=1)[active].sum().item()
                        n_pix+=active.sum().item()
                if (bi+1)%args.grad_accum==0 or bi+1==n_batches:
                    torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),0.1)
                    opt.step();opt.zero_grad(set_to_none=True)
            schedule.step()
            row={'epoch':ep+1,'train_loss':total_loss/max(n,1),
                 'train_event_valid_epe':norm_epe/max(n_pix,1),
                 'train_zero_flow_epe_same_mask':zero_sum/max(n_pix,1),
                 'train_samples':n,'event_valid_pixels':int(n_pix)}
            history.append(row)
            save_json(args.output_dir/'train_history.json',history)
            print('[EPOCH]',json.dumps(row),flush=True)
            if (ep+1)%args.save_every==0 or ep+1==args.epochs:
                p=args.output_dir/'last.pt';tmp=args.output_dir/'last.pt.tmp'
                torch.save({'model':model.state_dict(),'epoch':ep+1,'config':run_cfg},tmp);os.replace(tmp,p)
        print('[DONE] checkpoint:',args.output_dir/'last.pt',flush=True)
    else:
        if args.mode=='eval-trained':
            if not args.resume:raise ValueError('--resume is required for eval-trained')
            ck=torch.load(args.resume,map_location='cpu',weights_only=False)
            if ck.get('config',{}).get('fusion')!=args.fusion:raise RuntimeError('Checkpoint fusion mode mismatch')
            model.load_state_dict(ck['model'],strict=True)
            print('[TRAINED CHECKPOINT]',args.resume,flush=True)
        results={seq:eval_sequence(model,seq,args.root,device,args.fusion,
                         limit=args.max_eval_samples,workers=args.workers) for seq in TEST_SEQS}
        save_json(args.output_dir/'metrics_final.json',{'config':run_cfg,'results':results})
        print('[RESULTS]',args.output_dir/'metrics_final.json',flush=True)


def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('datasets/MVSEC'))
    p.add_argument('--sea-core',type=Path,default=Path('third_party/ScaleEvent/sea_raft/core'))
    p.add_argument('--sea-config',type=Path,default=Path('third_party/ScaleEvent/sea_raft/config/eval/spring-M.json'))
    p.add_argument('--raft-ckpt',type=Path,default=Path('pretrained/searaft_M/model.safetensors'))
    p.add_argument('--base-ckpt',type=Path,help='Train GEP and JEPA from the SAME domain-finetuned full RAFT checkpoint')
    p.add_argument('--mode',choices=('eval-pretrained','train','eval-trained'),required=True)
    p.add_argument('--fusion',choices=('none','gep','jepa'),default='none')
    p.add_argument('--stage1-ckpt',type=Path,default=Path('checkpoints/stage1/gep_stage1_small.pt'))
    p.add_argument('--stage2-ckpt',type=Path,default=Path('runs/event_jepa_vits14_formal_seed0/step_00030000.pt'))
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--resume',type=Path)
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--grad-accum',type=int,default=8)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--lr-raft',type=float,default=2e-5)
    p.add_argument('--lr-fusion',type=float,default=2e-4)
    p.add_argument('--lr-stage1',type=float,default=1e-6)
    p.add_argument('--lr-stage2',type=float,default=1e-5)
    p.add_argument('--finetune-event-encoders',action='store_true')
    p.add_argument('--freeze-raft',action='store_true',
       help='Use full original pretrained SEA-RAFT as a fixed base, including frozen BN; '
            'for GEP/JEPA train without --base-ckpt')
    p.add_argument('--max-eval-samples',type=int)
    p.add_argument('--max-train-samples',type=int)
    p.add_argument('--save-every',type=int,default=5)
    p.add_argument('--seed',type=int,default=0)
    args=p.parse_args()
    if args.batch_size<1 or args.grad_accum<1 or args.epochs<1:raise ValueError('Invalid training hyperparameters')
    if args.freeze_raft and args.finetune_event_encoders:
        raise ValueError('v8.2 frozen-RAFT pilot requires FROZEN GEP/JEPA encoders; do not set --finetune-event-encoders')
    if args.freeze_raft and args.base_ckpt is not None:
        raise ValueError('v8.2 frozen-RAFT pilot must start at ORIGINAL pretrained SEA-RAFT; omit --base-ckpt')
    if args.mode!='train' and args.max_train_samples:raise ValueError('--max-train-samples only in train')
    if args.mode=='eval-pretrained' and args.resume:raise ValueError('eval-pretrained must not resume a finetuned model')
    run(args)

if __name__=='__main__':cli()
