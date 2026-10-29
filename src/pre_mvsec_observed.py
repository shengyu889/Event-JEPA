"""Build observed-interval Stage-1 tokens and native event masks for MVSEC.

Kept completely separate from the old Future Flow preprocessing.
Supports raw/outdoor_day and raw/indoor_flying HDF5 paths.
Outputs one z_j from events (t_(j-1), t_j], plus native GT-aligned mask.
"""
from __future__ import annotations
import argparse
import csv
import json
import hashlib
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from pre_mvsec_flow import (
    DSEC_EVENT_MEAN, DSEC_EVENT_STD, RAW_H, RAW_W,
    ensure_npz_cache, event_index_at_time, make_event_rgb,
    load_stage1_encoder,
)
from mvsec_flow_probe.scaleevent_flow_metrics import event_activity_mask


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fp:
        for chunk in iter(lambda: fp.read(16 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def preprocess_rgb_native_to_224(event_rgb: np.ndarray) -> torch.Tensor:
    if event_rgb.shape != (RAW_H, RAW_W, 3):
        raise ValueError(f'event RGB wrong shape: {event_rgb.shape}')
    x = torch.from_numpy(event_rgb.copy()).permute(2, 0, 1).float() / 255.0
    # Preserve full FoV; model input is a declared 224x224 non-isotropic adapter.
    x = F.interpolate(x[None], size=(224, 224), mode='bilinear', align_corners=False)[0]
    return ((x - DSEC_EVENT_MEAN) / DSEC_EVENT_STD).contiguous()


def generate_sequence(root: Path, seq: str, ckpt: Path, device: str, batch_size: int,
                      max_intervals: int | None, percentile: float, overwrite: bool):
    family = 'outdoor_day' 
    data_path = root / 'raw' / family / f'{seq}_data.hdf5'
    gt_path = root / 'flow_gt' / f'{seq}_gt_flow_dist.npz'
    if not data_path.is_file() or not gt_path.is_file():
        raise FileNotFoundError(f'Missing {data_path} or {gt_path}. See README.')
    cache = ensure_npz_cache(gt_path, root / 'flow_gt' / '.npy_cache' / seq)
    stamps = np.load(cache['timestamps'], mmap_mode='r')
    with h5py.File(data_path, 'r') as h5:
        events = h5['davis/left/events']
        image_ts = np.asarray(h5['davis/left/image_raw_ts'][:], dtype=np.float64)
        image_event_inds = np.asarray(h5['davis/left/image_raw_event_inds'][:], dtype=np.int64)
        n = min(len(stamps), len(np.load(cache['x_flow_dist'], mmap_mode='r')),
                len(np.load(cache['y_flow_dist'], mmap_mode='r')))
        if n < 6:
            raise ValueError(f'{seq}: too few flow timestamps: {n}')
        first_i, end_i = 3, n - 1  # i in [3,n-2], observed interval is (t_i,t_(i+1)]
        if max_intervals is not None:
            end_i = min(end_i, first_i + max_intervals)
        indexes = list(range(first_i, end_i))
        token_ids = range(indexes[0] - 2, indexes[-1] + 2)
        out = root / 'processed_observed_v1' / seq
        token_dir, mask_dir = out / 'eventToken', out / 'event_active'
        token_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        ckpt_sha = sha256_file(ckpt)
        old_metadata = out / 'metadata.json'
        if old_metadata.is_file() and not overwrite:
            old = json.loads(old_metadata.read_text())
            if old.get('stage1_sha256') not in (None, ckpt_sha):
                raise RuntimeError('Existing tokens were made using a different Stage1 checkpoint. '
                                   'Use a different output root or --overwrite.')
            if old.get('percentile') not in (None, percentile):
                raise RuntimeError('Existing tokens use a different event RGB percentile. '
                                   'Use --overwrite to rebuild all tokens.')
        print(f'[{seq}] GT intervals {len(indexes)}, token ids {token_ids.start}..{token_ids.stop-1}')
        encoder = load_stage1_encoder(ckpt, torch.device(device))
        encoder.eval()
        pending, pending_ids = [], []

        def flush():
            if not pending:
                return
            with torch.inference_mode():
                tokens = encoder.forward_features(torch.stack(pending).to(device))['x_norm_patchtokens']
            if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (256, 384):
                raise RuntimeError(f'bad Stage-1 token shape: {tokens.shape}')
            for j, z in zip(pending_ids, tokens):
                torch.save(z.detach().half().cpu().contiguous(), token_dir / f'{j:06d}.pt')
            pending.clear()
            pending_ids.clear()

        for j in tqdm(token_ids, desc=f'{seq} tokenize'):
            token_file, mask_file = token_dir / f'{j:06d}.pt', mask_dir / f'{j:06d}.pt'
            if not overwrite and token_file.is_file() and mask_file.is_file():
                continue
            t0, t1 = float(stamps[j-1]), float(stamps[j])
            if not (t0 < t1):
                raise ValueError(f'{seq}: non-increasing GT timestamps at {j}: {t0}, {t1}')
            start = event_index_at_time(events, image_ts, image_event_inds, t0, side='right')
            stop = event_index_at_time(events, image_ts, image_event_inds, t1, side='right')
            e = np.asarray(events[start:stop])
            activity = event_activity_mask(e, t0, t1, RAW_H, RAW_W)
            # This is a pixel mask in the native GT space, not the resized encoder input.
            torch.save(torch.from_numpy(activity), mask_file)
            if overwrite or not token_file.is_file():
                rgb, _ = make_event_rgb(events, start, stop, percentile)
                pending.append(preprocess_rgb_native_to_224(rgb))
                pending_ids.append(j)
                if len(pending) >= batch_size:
                    flush()
        flush()

    rows = []
    for i in indexes:
        t0, t1 = float(stamps[i]), float(stamps[i+1])
        if not all((token_dir / f'{j:06d}.pt').is_file() for j in [i-2,i-1,i,i+1]):
            raise FileNotFoundError(f'missing context token for sample {i}')
        if not (mask_dir / f'{i+1:06d}.pt').is_file():
            raise FileNotFoundError(f'missing event mask for sample {i}')
        rows.append(dict(sequence=seq,flow_index=i,t0=f'{t0:.9f}',t1=f'{t1:.9f}',
                         context_0=i-2,context_1=i-1,context_2=i,context_3=i+1))
    with (out / 'manifest.csv').open('w', newline='') as fp:
        w=csv.DictWriter(fp,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
    (out/'metadata.json').write_text(json.dumps({
        'protocol':'observed_interval_scaleevent_style_v1', 'sequence':seq,
        'num_samples':len(rows),'input_event_intervals':'z_j = (t_(j-1),t_j]',
        'target_flow':'flow_i = t_i -> t_(i+1)',
        'context_for_i':'z_(i-2), z_(i-1), z_i, z_(i+1)',
        'native_resolution':[RAW_H,RAW_W],'input_adapter':'resize full 346x260 to 224x224',
        'GT_valid':'finite AND nonzero, outdoor hood row>=193 masked',
        'stage1_checkpoint':str(ckpt), 'stage1_sha256': ckpt_sha, 'percentile':percentile,
        'NOTE':'Not a bitwise or protocol-exact ScaleEvent reproduction; common observed-interval baseline.'
    },indent=2))
    print(f'[PASS] {seq}: manifest {out/"manifest.csv"} / samples {len(rows)}')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--sequences',nargs='+',default=['outdoor_day2','indoor_flying1','indoor_flying2','indoor_flying3'])
    p.add_argument('--event-encoder-ckpt',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--max-intervals',type=int,default=None,help='Smoke only; omit for full processing.')
    p.add_argument('--percentile',type=float,default=90.0)
    p.add_argument('--overwrite',action='store_true')
    a=p.parse_args()
    if a.max_intervals is not None and a.max_intervals <= 0:
        p.error('--max-intervals must be positive')
    for seq in a.sequences:
        generate_sequence(a.root,seq,a.event_encoder_ckpt,a.device,a.batch_size,a.max_intervals,a.percentile,a.overwrite)

if __name__=='__main__': main()
