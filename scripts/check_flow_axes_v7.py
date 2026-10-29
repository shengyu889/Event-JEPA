#!/usr/bin/env python3
"""Read-only coordinate-axis/sign diagnostic over *identical* MVSEC test pixels.

These eight transformations are debugging hypotheses, NOT fair trained models.
Never use the minimum of the eight as a publication result.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

SEQS=('indoor_flying1','indoor_flying2','indoor_flying3')

def vector_views(p):
    x,y=p[0],p[1]
    return {
      'identity':(x,y), 'negate_both':(-x,-y),
      'negate_x':(-x,y), 'negate_y':(x,-y),
      'swap_xy':(y,x), 'swap_negate_both':(-y,-x),
      'rotate_cw':(y,-x), 'rotate_ccw':(-y,x),
    }

def get_args():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('datasets/MVSEC'))
    p.add_argument('--model',default='full',choices=['full','stage1-only','stage2-only','frozen','ft10'])
    p.add_argument('--samples',type=int,default=24)
    p.add_argument('--stage1-ckpt',type=Path,default=Path('checkpoints/stage1/gep_stage1_small.pt'))
    p.add_argument('--stage2-ckpt',type=Path,default=Path('runs/event_jepa_vits14_formal_seed0/step_00030000.pt'))
    p.add_argument('--sea-core',type=Path,default=Path('third_party/ScaleEvent/sea_raft/core'))
    p.add_argument('--checkpoint',type=Path,default=None)
    p.add_argument('--output',type=Path,default=Path('results/axes_full_v7.json'))
    return p.parse_args()

def main():
    a=get_args()
    if a.samples < 1:raise ValueError('Use --samples >=1 for an inexpensive check')
    # Use exact same model reconstruction and flow restoration as v6 diagnostic.
    from audit_flow_collapse_v6 import picks,load_rows,read_label,prepare_model
    from pre_mvsec_observed import preprocess_rgb_native_to_224
    from mvsec_flow_probe.observed_dataset import to_native_flow
    from PIL import Image
    import torch
    model=prepare_model(a)
    report={}
    for seq in SEQS:
        base,rows=load_rows(a.root,seq)
        accum={k:{'error':0.,'dot':0.,'count':0} for k in vector_views(np.zeros((2,1),dtype=np.float32))}
        zero_error=0.; n_pixels=0
        for idx in picks(len(rows),a.samples):
            row=rows[idx]
            gt,valid,active=read_label(base,row)
            mask=valid & active
            if not mask.any():continue
            events=[]
            for i in range(4):
                j=int(row[f'context_{i}'])
                with Image.open(base/'eventRGB_native'/f'{j:06d}.png') as im:
                    events.append(preprocess_rgb_native_to_224(np.asarray(im.convert('RGB'),dtype=np.uint8)))
            x=torch.stack(events).unsqueeze(0).cuda()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                pred224=model(x)['final']
            pred=to_native_flow(pred224.float(),height=260,width=346)[0].cpu().numpy()
            g=gt[:,mask].astype(np.float64)
            p=pred[:,mask].astype(np.float64)
            gn=np.linalg.norm(g,axis=0)
            zero_error+=gn.sum();n_pixels+=int(mask.sum())
            for name,(u,v) in vector_views(p).items():
                pred_v=np.stack((u,v),axis=0)
                e=np.linalg.norm(pred_v-g,axis=0)
                acc=accum[name];acc['error']+=float(e.sum())
                pn=np.linalg.norm(pred_v,axis=0)
                nz=(pn>1e-3)&(gn>1)
                cos=((pred_v[:,nz]*g[:,nz]).sum(axis=0) / (pn[nz]*gn[nz]))
                acc['dot']+=float(cos.sum());acc['count']+=int(nz.sum())
        if not n_pixels:raise RuntimeError(f'{seq}: no valid event pixels')
        vals={k:{'epe':v['error']/n_pixels,
                'cos_on_gt_mag_gt1':v['dot']/max(1,v['count'])}
              for k,v in accum.items()}
        ordered=sorted(vals,key=lambda k:vals[k]['epe'])
        report[seq]={'zero_epe':zero_error/n_pixels,'pixels':n_pixels,
                     'transforms':vals,'best_transform':ordered[0]}
        print(f'[{seq}] zero={zero_error/n_pixels:.4f} '
              + ' '.join(f'{name}:EPE={vals[name]["epe"]:.3f},cos={vals[name]["cos_on_gt_mag_gt1"]:+.3f}'
                         for name in ordered[:4]),flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'note':'Debug-only oracle over 8 transforms; not model comparison',
                                    'model':a.model,'n_sampled_frames':a.samples,'results':report},indent=2))
    print('SAVED:',a.output)
if __name__=='__main__':main()
