#!/usr/bin/env python3
"""Read-only MVSEC v6 GT zero-flow baseline + model movement diagnostic.

Designed for Event-JEPA v6 layout, does NOT modify checkpoints or datasets.
Compares trained predictions with zero-flow ON THE SAME sampled frames and mask.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np

SEQS = ('indoor_flying1', 'indoor_flying2', 'indoor_flying3')


def picks(n: int, samples: int) -> list[int]:
    if n <= 0:
        raise ValueError('Empty dataset')
    if samples == 0 or samples >= n:
        return list(range(n))
    return np.linspace(0, n-1, samples, dtype=np.int64).tolist()


def load_rows(root, seq):
    base = root / 'processed_image_time_v6' / seq
    with (base / 'manifest.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f'Empty manifest: {base}')
    return base, rows


def read_label(base, row):
    with np.load(base / row['flow_file'], allow_pickle=False) as d:
        flow = d['flow'].astype(np.float32, copy=True)
        valid = d['valid'].astype(bool, copy=True)
        active = d['event_active'].astype(bool, copy=True)
    if flow.shape != (2, 260, 346) or valid.shape != (260,346) or active.shape != (260,346):
        raise ValueError('v6 GT shape mismatch')
    return flow, valid, active


class Stats:
    def __init__(self):
        self.frames=0
        self.total=0
        self.gt_valid=0
        self.event_active=0
        self.joint=0
        self.gt_mag_sum=0.
        self.zero_epe_sum=0.
        self.zero_outlier=0
        self.pred_epe_sum=0.
        self.pred_outlier=0
        self.pred_mag_sum=0.
        self.pred_near_zero=0
        self.direction_dot=0.
        self.direction_count=0
        self.better_count=0
        self.gt_gt1_count=0
        self.pred_gt1_epe_sum=0.
        self.zero_gt1_epe_sum=0.

    def update(self, flow, valid, active, pred=None):
        mask=(valid & active)
        n=int(mask.sum())
        self.frames+=1
        self.total+=valid.size
        self.gt_valid+=int(valid.sum())
        self.event_active+=int(active.sum())
        self.joint+=n
        if n == 0:
            return
        g=flow[:,mask].astype(np.float64, copy=False)
        gm=np.linalg.norm(g,axis=0)
        self.gt_mag_sum+=float(gm.sum())
        self.zero_epe_sum+=float(gm.sum())
        self.zero_outlier+=int(((gm>3)&(gm>0.05*gm)).sum())
        if pred is not None:
            p=pred[:,mask].astype(np.float64,copy=False)
            pm=np.linalg.norm(p,axis=0)
            e=np.linalg.norm(p-g,axis=0)
            self.pred_epe_sum+=float(e.sum())
            self.pred_mag_sum+=float(pm.sum())
            self.pred_near_zero+=int((pm<0.25).sum())
            self.pred_outlier+=int(((e>3)&(e>0.05*gm)).sum())
            self.better_count+=int((e<gm).sum())
            moving=gm>1.0
            self.gt_gt1_count+=int(moving.sum())
            self.pred_gt1_epe_sum+=float(e[moving].sum())
            self.zero_gt1_epe_sum+=float(gm[moving].sum())
            d=gm*pm
            informative=d>1e-6
            self.direction_dot+=float(((p*g).sum(axis=0)[informative]/d[informative]).sum())
            self.direction_count+=int(informative.sum())

    def report(self, with_model=False):
        if not self.joint:
            raise RuntimeError('No pixels satisfy GT-valid AND event-active')
        out={
            'frames':self.frames,
            'native_pixels':self.total,
            'gt_valid_fraction':self.gt_valid/self.total,
            'event_active_fraction':self.event_active/self.total,
            'joint_eval_fraction':self.joint/self.total,
            'joint_eval_pixels':self.joint,
            'zero_epe':self.zero_epe_sum/self.joint,
            'zero_outlier_pct':100*self.zero_outlier/self.joint,
        }
        if with_model:
            out.update({
                'pred_epe':self.pred_epe_sum/self.joint,
                'pred_outlier_pct':100*self.pred_outlier/self.joint,
                'relative_epe_gain_over_zero_pct':100*(self.zero_epe_sum-self.pred_epe_sum)/self.zero_epe_sum,
                'gt_magnitude_mean':self.gt_mag_sum/self.joint,
                'pred_magnitude_mean':self.pred_mag_sum/self.joint,
                'pred_mag_lt_0_25_pct':100*self.pred_near_zero/self.joint,
                'pixels_pred_better_than_zero_pct':100*self.better_count/self.joint,
                'mean_cosine_pred_gt_on_nonzero':self.direction_dot/max(self.direction_count,1),
                'mean_pred_epe_gtmag_gt1':self.pred_gt1_epe_sum/max(self.gt_gt1_count,1),
                'mean_zero_epe_gtmag_gt1':self.zero_gt1_epe_sum/max(self.gt_gt1_count,1),
            })
        return out


def prepare_model(args):
    import torch
    from dsec_flow_probe.train_raw_finetune_formal import build_stage1, build_stage2
    from mvsec_flow_probe.searaft_model import add_searaft_core, EventJEPASEARAFT
    add_searaft_core(args.sea_core)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU unavailable, omit --model to compute the exact zero-flow baseline on CPU')
    stage1 = build_stage1(args.stage1_ckpt)
    stage2, _ = build_stage2(args.stage2_ckpt)
    model = EventJEPASEARAFT(stage1,stage2,args.model,checkpoint_encoder=False).cuda().eval()
    ckpt=Path(f'runs/searaft_image_v6_{args.model}_seed0/last.pt') if args.checkpoint is None else args.checkpoint
    state=torch.load(ckpt,map_location='cpu',weights_only=False)
    if state.get('metadata',{}).get('protocol')!=args.model:
        raise ValueError('Checkpoint protocol mismatch')
    model.load_state_dict(state['model'],strict=True)
    return model


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('datasets/MVSEC'))
    p.add_argument('--model',choices=['frozen','ft10','full','stage2-only','stage1-only'])
    p.add_argument('--samples',type=int,default=64,help='samples per sequence (0=entire split)')
    p.add_argument('--stage1-ckpt',type=Path,default=Path('checkpoints/stage1/gep_stage1_small.pt'))
    p.add_argument('--stage2-ckpt',type=Path,default=Path('runs/event_jepa_vits14_formal_seed0/step_00030000.pt'))
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--sea-core',type=Path,default=Path('third_party/ScaleEvent/sea_raft/core'))
    p.add_argument('--output',type=Path,default=Path('results/searaft_v6_collapse_audit.json'))
    args=p.parse_args()
    if args.samples < 0:raise ValueError('--samples must be >=0')
    root=args.root.resolve()
    model=prepare_model(args) if args.model else None
    report={}
    for seq in SEQS:
        base,rows=load_rows(root,seq)
        chosen=picks(len(rows),args.samples)
        stats=Stats()
        for count, idx in enumerate(chosen,1):
            row=rows[idx]
            flow, valid, active=read_label(base,row)
            if model is None:
                stats.update(flow,valid,active)
            else:
                # Use the SAME Event-JEPA preprocessing and flow-coordinate restoration
                # as the original official v6 evaluation routine.
                import torch
                from PIL import Image
                from pre_mvsec_observed import preprocess_rgb_native_to_224
                from mvsec_flow_probe.observed_dataset import to_native_flow
                js=[int(row[f'context_{i}']) for i in range(4)]
                ev=[]
                for j in js:
                    with Image.open(base/'eventRGB_native'/f'{j:06d}.png') as im:
                        rgb=np.asarray(im.convert('RGB'),dtype=np.uint8)
                    ev.append(preprocess_rgb_native_to_224(rgb))
                x=torch.stack(ev,0).unsqueeze(0).cuda()
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    pred224=model(x)['final']
                pred=to_native_flow(pred224.float(),height=260,width=346)[0].cpu().numpy()
                stats.update(flow,valid,active,pred=pred)
            if count%100==0:print(seq, count,'/',len(chosen),flush=True)
        report[seq]=stats.report(with_model=model is not None)
        vals=report[seq]
        if model is None:
            print(f'{seq} n={vals["frames"]} zero EPE={vals["zero_epe"]:.5f} zero Outlier={vals["zero_outlier_pct"]:.3f}% mask={vals["joint_eval_fraction"]:.3%}',flush=True)
        else:
            print(f'{seq} n={vals["frames"]} zero EPE={vals["zero_epe"]:.5f} pred EPE={vals["pred_epe"]:.5f} gain={vals["relative_epe_gain_over_zero_pct"]:+.3f}%  pred_mag={vals["pred_magnitude_mean"]:.4f}  near_zero={vals["pred_mag_lt_0_25_pct"]:.1f}% ',flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    obj={'protocol':'mvsec_image_timestamp_v6_candidate_not_official_replication',
         'mode':args.model or 'zero-only', 'samples_per_sequence':args.samples,
         'comment':'Subsampled model and baseline are ALWAYS evaluated on identical frames & masks; samples=0 is full split.',
         'results':report}
    args.output.write_text(json.dumps(obj,indent=2))
    print('WROTE',args.output,flush=True)


if __name__=='__main__':main()
