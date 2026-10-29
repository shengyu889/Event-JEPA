"""Lossless native-size event-RGB image cache for online fine-tuning.

Reuses timestamps, masks and manifests from pre_mvsec_observed.py.
Does NOT read/overwrite cached Stage-1 tokens or change earlier experiments.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

from pre_mvsec_flow import (ensure_npz_cache, event_index_at_time, make_event_rgb,
                            RAW_H, RAW_W)


def prepare_sequence(root: Path, sequence: str, *, percentile=90.0,
                     overwrite=False, max_intervals=None):
    p = root / 'processed_observed_v1' / sequence
    manifest = p / 'manifest.csv'
    if not manifest.is_file():
        raise FileNotFoundError(f'{manifest} -- run pre_mvsec_observed.py first')
    with manifest.open(newline='') as f:
        rows = list(csv.DictReader(f))
    if max_intervals is not None:
        rows = rows[:max_intervals]
    if not rows:
        raise RuntimeError(f'empty manifest for {sequence}')
    meta_path = p / 'metadata.json'
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if float(meta.get('percentile', percentile)) != float(percentile):
            raise ValueError('percentile differs from observed input metadata')
    needed = sorted({int(r[f'context_{k}']) for r in rows for k in range(4)})
    fam = 'outdoor_day'
    raw = root / 'raw' / fam / f'{sequence}_data.hdf5'
    gt = root / 'flow_gt' / f'{sequence}_gt_flow_dist.npz'
    if not raw.is_file() or not gt.is_file():
        raise FileNotFoundError(f'missing {raw} or {gt}')
    cache = ensure_npz_cache(gt, root/'flow_gt'/'.npy_cache'/sequence)
    ts = np.load(cache['timestamps'], mmap_mode='r')
    save_dir = p / 'eventRGB_native'
    save_dir.mkdir(parents=True, exist_ok=True)
    made = 0
    with h5py.File(raw, 'r') as f:
        events = f['davis/left/events']
        image_ts = np.asarray(f['davis/left/image_raw_ts'][:], dtype=np.float64)
        image_idx = np.asarray(f['davis/left/image_raw_event_inds'][:], dtype=np.int64)
        for j in tqdm(needed, desc=f'{sequence}: eventRGB'):
            target = save_dir / f'{j:06d}.png'
            if target.is_file() and not overwrite:
                continue
            if not (0 < j < len(ts)):
                raise ValueError(f'{sequence} invalid interval ID {j}')
            t0, t1 = float(ts[j-1]), float(ts[j])
            if not t0 < t1:
                raise ValueError(f'{sequence} invalid t0/t1 for id {j}')
            a = event_index_at_time(events,image_ts,image_idx,t0,side='right')
            b = event_index_at_time(events,image_ts,image_idx,t1,side='right')
            rgb, _ = make_event_rgb(events,a,b,percentile)
            if rgb.shape != (RAW_H, RAW_W, 3):
                raise ValueError(f'bad event RGB dimensions: {rgb.shape}')
            tmp = target.with_suffix('.png.tmp')
            Image.fromarray(rgb).save(tmp, format='PNG')
            tmp.replace(target)
            made += 1
    for j in needed:
        if not (save_dir / f'{j:06d}.png').is_file():
            raise FileNotFoundError(f'missing native event image {j}')
    print(f'[PASS] {sequence}: {len(needed)} intervals; wrote {made}; cache {save_dir}')


def main():
    pa=argparse.ArgumentParser()
    pa.add_argument('--root',type=Path,required=True)
    pa.add_argument('--sequences',nargs='+',default=['outdoor_day2'])
    pa.add_argument('--percentile',type=float,default=90.0)
    pa.add_argument('--max-intervals',type=int)
    pa.add_argument('--overwrite',action='store_true')
    a=pa.parse_args()
    for seq in a.sequences:
        prepare_sequence(a.root,seq,percentile=a.percentile,
                         max_intervals=a.max_intervals,overwrite=a.overwrite)

if __name__=='__main__':
    main()
