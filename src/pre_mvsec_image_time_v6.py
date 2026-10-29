#!/usr/bin/env python3
"""Create true image-timestamp MVSEC flow via motion composition.

Source labels: EV-FlowNet/MVSEC 50ms x_flow_dist/y_flow_dist, pixel displacement.
Target: consecutive left-image timestamp flow at ~45Hz outdoor/~31Hz indoor.
THIS DOES NOT ASSERT THAT THE EXACT ScaleEvent SAMPLE FILTER IS KNOWN.

Output is isolated at processed_image_time_v6; source files never changed.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
from collections import defaultdict

import h5py
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

from pre_mvsec_flow import ensure_npz_cache, event_index_at_time, make_event_rgb

H, W = 260, 346
SEQS = ('outdoor_day2', 'indoor_flying1', 'indoor_flying2', 'indoor_flying3')


def intervals(image_ts, gt_ts, context=4, strict_context_in_gt=True):
    """Return (image_pair_index, four interval ids) for every covered pair.

    A pair index i corresponds to image_ts[i] -> image_ts[i+1].
    Its event token id is i+1 and context interval ids are i-2..i+1.
    Requiring context inside GT range is OPTIONAL and labelled, not official.
    """
    im=np.asarray(image_ts,dtype=np.float64)
    gt=np.asarray(gt_ts,dtype=np.float64)
    if len(im)<context+1 or len(gt)<2 or not (np.diff(im)>0).all() or not (np.diff(gt)>0).all():
        raise ValueError('timestamps must be strictly increasing and long enough')
    rows=[]
    for i in range(context-1,len(im)-1):
        start=float(im[i]); end=float(im[i+1])
        if start<gt[0] or end>gt[-1] or start>=end:continue
        ctx=tuple(range(i-context+2,i+2))
        if ctx[0]<1:continue
        if strict_context_in_gt and im[ctx[0]-1]<gt[0]:continue
        rows.append((i,ctx))
    return rows


def compose_gt(gt_ts, u, v, start, end, *, invalid_zero=True):
    """Compose consecutive 50ms displacement maps to exact image interval.

    Maps source pixel coordinates through sub-interval motion, using nearest
    sampling like EV-FlowNet prop_flow. Handles crossing GT boundaries EVEN
    when total duration is <50ms (upstream shortcut does not cover this case).
    Returns flow [2,H,W] float32, valid[H,W] boolean.
    """
    gt_ts=np.asarray(gt_ts,dtype=np.float64)
    start=float(start);end=float(end)
    if not (np.isfinite(start) and np.isfinite(end) and gt_ts[0]<=start<end<=gt_ts[-1]):
        raise ValueError(f'GT coverage violation: {start}..{end} not within {gt_ts[0]}..{gt_ts[-1]}')
    if u.shape[0]<len(gt_ts)-1 or v.shape[0]<len(gt_ts)-1:
        raise ValueError('insufficient GT flow maps')
    hh,ww=u.shape[-2:]
    if (hh,ww)!=(H,W) or v.shape[-2:]!=(H,W):raise ValueError('MVSEC must be 260x346 native')
    xx,yy=np.meshgrid(np.arange(ww,dtype=np.float32), np.arange(hh,dtype=np.float32))
    cur_x=xx.copy();cur_y=yy.copy();valid=np.ones((hh,ww),dtype=bool)
    k=int(np.searchsorted(gt_ts,start,side='right')-1)
    k=min(k,len(gt_ts)-2)
    count=0
    while k<len(gt_ts)-1 and gt_ts[k]<end:
        lo=max(start,float(gt_ts[k]));hi=min(end,float(gt_ts[k+1]))
        if hi>lo:
            frac=np.float32((hi-lo)/(gt_ts[k+1]-gt_ts[k]))
            uu=np.asarray(u[k],dtype=np.float32)
            vv=np.asarray(v[k],dtype=np.float32)
            source_valid=np.isfinite(uu)&np.isfinite(vv)
            if invalid_zero:source_valid &= (uu!=0)|(vv!=0)
            uu=np.nan_to_num(uu,nan=0,posinf=0,neginf=0)
            vv=np.nan_to_num(vv,nan=0,posinf=0,neginf=0)
            # Always sample at advected coordinates, including when the pair
            # straddles two 50ms bins (nearest matching upstream EV-FlowNet).
            su=cv2.remap(uu,cur_x,cur_y,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
            sv=cv2.remap(vv,cur_x,cur_y,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
            good=cv2.remap(source_valid.astype(np.uint8),cur_x,cur_y,cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT,borderValue=0)>0
            valid &= good
            cur_x += frac*su
            cur_y += frac*sv
            valid &= (cur_x>=0)&(cur_x<ww)&(cur_y>=0)&(cur_y<hh)
            count+=1
        k+=1
    if not count:raise RuntimeError('zero GT segments covered')
    flow=np.stack((cur_x-xx,cur_y-yy),axis=0).astype(np.float32)
    flow[:,~valid]=0
    return flow,valid


def event_active(raw, a, b):
    e=np.asarray(raw[a:b])
    mask=np.zeros((H,W),dtype=np.uint8)
    if len(e):
        x=e[:,0].astype(np.int64);y=e[:,1].astype(np.int64)
        in_bounds=(0<=x)&(x<W)&(0<=y)&(y<H)
        mask[y[in_bounds],x[in_bounds]]=1
    return mask


def atomic_npz(dst: Path, **kwargs):
    tmp=dst.with_name(dst.name+'.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**kwargs)
    os.replace(tmp,dst)


def atomic_png(dst:Path, rgb):
    tmp=dst.with_name(dst.name+'.tmp')
    Image.fromarray(rgb).save(tmp,format='PNG')
    os.replace(tmp,dst)


def process(root, seq, *, max_samples=None, context=4,
            strict_context_in_gt=True, percentile=90., overwrite=False,
            metadata_only=False):
    fam='outdoor_day' 
    raw=root/'raw'/fam/f'{seq}_data.hdf5'
    orig=root/'flow_gt'/f'{seq}_gt_flow_dist.npz'
    if not raw.is_file():raise FileNotFoundError(raw)
    if not orig.is_file():raise FileNotFoundError(orig)
    cache=ensure_npz_cache(orig,root/'flow_gt'/'.npy_cache'/seq)
    gt_ts=np.load(cache['timestamps'],mmap_mode='r')
    with h5py.File(raw,'r') as f:
        im=np.asarray(f['davis/left/image_raw_ts'][:],dtype=np.float64)
        image_event_inds=f['davis/left/image_raw_event_inds'][:]
        if len(im)!=len(image_event_inds):raise ValueError('image/event indices length mismatch')
        rows=intervals(im,gt_ts,context,strict_context_in_gt)
        all_count=len(rows)
        if max_samples is not None:rows=rows[:max_samples]
        print(f'[{seq}] in-GT image pairs={int(((im[:-1]>=gt_ts[0])&(im[1:]<=gt_ts[-1])).sum())}, '
              f'context-compatible={all_count}, selected={len(rows)}, dt_ms={np.median(np.diff(im))*1000:.3f}',flush=True)
        summary={'sequence':seq,'raw_file':str(raw),'GT_label_file':str(orig),
                 'image_pairs_covered':int(((im[:-1]>=gt_ts[0])&(im[1:]<=gt_ts[-1])).sum()),
                 'context_compatible':all_count,'selected':len(rows),'context':context,
                 'strict_context_in_gt':strict_context_in_gt,
                 'flow_gt':'composed from native 50ms EV-FlowNet GT in pixel units',
                 'timebase':'HDF5 image_raw_ts exact pairs','input':'GEP Stage1-compatible percentile RGB',
                 'paper_split_verification':'NOT VERIFIED; no arbitrary count trimming',
                 'GT_zero_policy':'zero-vector treated as invalid to match EV-FlowNet flow_error_dense',
                 'outdoor_hood':'rows >= 190 excluded (EV-FlowNet evaluation reference)' if seq.startswith('outdoor') else 'not applicable',
                 'percentile':percentile,'candidate_mode':'image_time_v6'}
        if metadata_only:return summary
        proc=root/'processed_image_time_v6'/seq
        flow_dir=proc/'flow'; event_dir=proc/'eventRGB_native'
        flow_dir.mkdir(parents=True,exist_ok=True);event_dir.mkdir(parents=True,exist_ok=True)
        u=np.load(cache['x_flow_dist'],mmap_mode='r');v=np.load(cache['y_flow_dist'],mmap_mode='r')
        events=f['davis/left/events']
        # Four context intervals per row, only compute each interval once.
        needed=set(j for _,ctx in rows for j in ctx)
        for j in tqdm(sorted(needed),desc=f'{seq}: event RGB'):
            dst=event_dir/f'{j:06d}.png'
            if dst.exists() and not overwrite:continue
            a=event_index_at_time(events,im,image_event_inds,float(im[j-1]),side='right')
            b=event_index_at_time(events,im,image_event_inds,float(im[j]),side='right')
            rgb,_=make_event_rgb(events,a,b,percentile)
            atomic_png(dst,rgb)
        manifest=[]
        for i,ctx in tqdm(rows,desc=f'{seq}: GT compose'):
            label_path=flow_dir/f'{i:06d}.npz'
            if not label_path.exists() or overwrite:
                flow,valid=compose_gt(gt_ts,u,v,im[i],im[i+1])
                # EV-FlowNet evaluation excludes a car hood in outdoor sequences.
                if seq.startswith('outdoor'):
                    valid[190:,:]=False
                    flow[:,190:,:]=0
                a=event_index_at_time(events,im,image_event_inds,float(im[i]),side='right')
                b=event_index_at_time(events,im,image_event_inds,float(im[i+1]),side='right')
                activity=event_active(events,a,b)
                atomic_npz(label_path,flow=flow,valid=valid.astype(np.uint8),event_active=activity)
            manifest.append({'pair_index':i,'start_s':repr(float(im[i])),'end_s':repr(float(im[i+1])),
                   'context_0':ctx[0],'context_1':ctx[1],'context_2':ctx[2],'context_3':ctx[3],
                   'flow_file':f'flow/{i:06d}.npz'})
        if not manifest:raise RuntimeError(f'{seq}: no candidate samples')
        tmp=proc/'manifest.csv.tmp'
        with tmp.open('w',newline='') as fp:
            wr=csv.DictWriter(fp,fieldnames=manifest[0].keys());wr.writeheader();wr.writerows(manifest)
        os.replace(tmp,proc/'manifest.csv')
        (proc/'metadata.json').write_text(json.dumps(summary,indent=2))
        print(f'[PASS] {seq}: saved {len(manifest)} GT records in {proc}',flush=True)
        return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--sequences',nargs='+',default=list(SEQS),choices=SEQS)
    p.add_argument('--context',type=int,default=4)
    p.add_argument('--max-samples',type=int,default=None)
    p.add_argument('--metadata-only',action='store_true')
    p.add_argument('--allow-history-before-gt',action='store_true')
    p.add_argument('--percentile',type=float,default=90.)
    p.add_argument('--overwrite',action='store_true')
    args=p.parse_args()
    root=args.root.resolve()
    out={}
    for seq in args.sequences:
        out[seq]=process(root,seq,max_samples=args.max_samples,context=args.context,
                   strict_context_in_gt=not args.allow_history_before_gt,
                   percentile=args.percentile,overwrite=args.overwrite,
                   metadata_only=args.metadata_only)
    if args.metadata_only:
        print(json.dumps(out,indent=2))

if __name__=='__main__':main()
