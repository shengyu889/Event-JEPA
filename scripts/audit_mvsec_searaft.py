#!/usr/bin/env python3
"""Read-only diagnostic for the 20-Hz observed-interval MVSEC protocol.

Computes a zero-flow benchmark with EXACTLY the same GT-valid & event-active
pixel mask as metrics_final.json. Does not modify any training artifacts.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch

SEQS = ('outdoor_day2', 'indoor_flying1', 'indoor_flying2', 'indoor_flying3')
MODES = ('frozen', 'ft10', 'full', 'stage2-only', 'stage1-only')


def sample_indices(n, count):
    if count <= 0 or count >= n:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, num=count, dtype=np.int64))


def inspect_sequence(root, seq, count):
    processed = root / 'processed_observed_v1' / seq
    manifest = processed / 'manifest.csv'
    if not manifest.is_file():
        return {'sequence': seq, 'error': f'MISSING {manifest}'}
    with manifest.open(newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {'sequence':seq, 'error':'Empty manifest'}
    t0 = np.array([float(r['t0']) for r in rows])
    t1 = np.array([float(r['t1']) for r in rows])
    dt = t1 - t0
    if not np.isfinite(dt).all() or not (dt > 0).all():
        raise RuntimeError(f'{seq}: invalid GT time intervals')
    cache = root / 'flow_gt' / '.npy_cache' / seq
    for n in ('x_flow_dist.npy','y_flow_dist.npy'):
        if not (cache / n).is_file():
            raise FileNotFoundError(f'{cache / n}: run original preprocessing to build npy cache')
    u = np.load(cache/'x_flow_dist.npy', mmap_mode='r')
    v = np.load(cache/'y_flow_dist.npy', mmap_mode='r')
    out = {'sequence':seq, 'manifest_samples':len(rows),
           'median_dt_ms':float(np.median(dt)*1000),
           'p05_dt_ms':float(np.percentile(dt,5)*1000),
           'p95_dt_ms':float(np.percentile(dt,95)*1000),
           'approx_frequency_hz':float(1/np.median(dt)),
           'checked_samples':0}
    gt_valid_total = valid_event_total = pixels_total = 0
    epe_zero_sum = outlier_zero_total = 0
    mag_samples = []
    mag_above60 = 0
    for ind in sample_indices(len(rows),count):
        r=rows[int(ind)]; i=int(r['flow_index'])
        uu=np.asarray(u[i],dtype=np.float32)
        vv=np.asarray(v[i],dtype=np.float32)
        if uu.shape != (260,346) or vv.shape != (260,346):
            raise RuntimeError(f'GT shape mismatch: {seq}, {uu.shape}, {vv.shape}')
        valid=np.isfinite(uu)&np.isfinite(vv)&((uu!=0)|(vv!=0))
        if seq.startswith('outdoor_day'):
            valid=valid.copy(); valid[193:]=False
        mask_path=processed/'event_active'/f'{i+1:06d}.pt'
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)
        active=torch.load(mask_path,map_location='cpu',weights_only=True).numpy().astype(bool)
        assert active.shape==uu.shape
        vm=valid & active
        mag=np.hypot(uu,vv)
        values=mag[vm]
        valid_event_total+=int(vm.sum())
        gt_valid_total+=int(valid.sum())
        pixels_total+=valid.size
        mag_above60+=int((values>=60).sum())
        epe_zero_sum+=float(values.sum(dtype=np.float64))
        outlier_zero_total+=int((values>3).sum())  # zero error EPE=GT mag
        if values.size:
            # Per-frame representative values; weighted quantiles require full vector accumulation
            mag_samples.append(np.percentile(values,[50,90,99]).tolist())
    if valid_event_total == 0:
        out['warning']='No jointly valid pixels'
    else:
        out.update({
           'checked_samples':int(len(sample_indices(len(rows),count))),
           'gt_valid_pixels':gt_valid_total,
           'gt_and_event_pixels':valid_event_total,
           'gt_valid_fraction_of_all':gt_valid_total/pixels_total,
           'event_active_fraction_among_gt_valid':valid_event_total/gt_valid_total if gt_valid_total else None,
           'ZERO_FLOW_EPE':epe_zero_sum/valid_event_total,
           'ZERO_FLOW_Outlier_percent':100*outlier_zero_total/valid_event_total,
           'gt_mag_over_60_count':mag_above60,
           'average_perframe_gt_mag_percentiles_50_90_99':np.mean(mag_samples,axis=0).tolist(),
        })
    return out


def inspect_runs(root):
    info={}
    for mode in MODES:
        directory=root/f'searaft_{mode}_seed0'
        if not directory.exists():continue
        row={'path':str(directory)}
        for name in ('run_config.json','train_history.json','parameter_audit.json'):
            p=directory/name
            if p.exists():
                try: row[name]=json.loads(p.read_text())
                except Exception as e:row[name+'_error']=str(e)
        info[mode]=row
    return info


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('datasets/MVSEC'))
    p.add_argument('--runs',type=Path,default=Path('runs'))
    p.add_argument('--sample-count',type=int,default=128,
                   help='evenly spaced samples per seq, 0 means full evaluation')
    p.add_argument('--output',type=Path,default=Path('results/mvsec_searaft_protocol_audit.json'))
    a=p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    sequences=[inspect_sequence(a.root,s,a.sample_count) for s in SEQS]
    runs=inspect_runs(a.runs)
    report={'sequences':sequences,'runs':runs,
            'note':'ZERO_FLOW_* is a baseline on the same dataset mask, not predicted model results. sampled if sample-count>0.'}
    a.output.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print('MVSEC SEA-RAFT data/zero-flow audit:')
    for row in sequences:
        if 'error' in row:
            print(f"{row['sequence']:18s} ERROR {row['error']}");continue
        print(f"{row['sequence']:18s} n={row['manifest_samples']:5d} "
              f"dt={row['median_dt_ms']:7.2f}ms "
              f"zeroEPE={row.get('ZERO_FLOW_EPE',float('nan')):7.3f} "
              f"zeroOut={row.get('ZERO_FLOW_Outlier_percent',float('nan')):7.2f}% "
              f"active/GT={row.get('event_active_fraction_among_gt_valid',float('nan')):.2%}")
    for name,v in runs.items():
        cfg=v.get('run_config.json',{})
        history=v.get('train_history.json',[])
        audit=v.get('parameter_audit.json',{})
        print('RUN',name,'epochs=',len(history),'samples=',cfg.get('train_samples'),
              'initial_loss=',round(history[0]['train_loss'],4) if history else None,
              'last_loss=',round(history[-1]['train_loss'],4) if history else None,
              'changed=',audit.get('changed'))
    print('Saved',a.output)

if __name__=='__main__':main()
