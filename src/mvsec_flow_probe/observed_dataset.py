"""MVSEC observed-interval flow transfer dataset.

Uses the same gt flow .npz convention as src/pre_mvsec_flow.py, without
cropping the native 260x346 GT.  Tokens z_(i+1) are constructed from events
(t_i,t_(i+1)] -- exactly the optical-flow GT interval.
"""
from __future__ import annotations

import csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from pre_mvsec_flow import ensure_npz_cache

NATIVE_H, NATIVE_W = 260, 346


def to_native_flow(pred224: torch.Tensor, height=NATIVE_H, width=NATIVE_W) -> torch.Tensor:
    """[B,2,224,224] displacement in model pixels -> native GT pixel units."""
    if pred224.ndim != 4 or pred224.shape[1] != 2:
        raise ValueError(f'bad predicted flow shape: {pred224.shape}')
    h, w = pred224.shape[-2:]
    out = torch.nn.functional.interpolate(pred224, size=(height, width), mode='bilinear', align_corners=False)
    scale = pred224.new_tensor([width / w, height / h]).view(1, 2, 1, 1)
    return out * scale


def flow_validity(u: np.ndarray, v: np.ndarray, sequence: str) -> np.ndarray:
    """MVSEC/E-RAFT-derived validity approximation; zero-vector ambiguity documented."""
    valid = np.isfinite(u) & np.isfinite(v) & ((u != 0) | (v != 0))
    if sequence.startswith('outdoor_day'):
        valid = valid.copy()
        valid[193:] = False  # matches original project's E-RAFT outdoor hood mask
    return valid


class MVSECObservedFlowDataset(Dataset):
    def __init__(self, root: str | Path, sequence: str, *, max_samples: int | None = None):
        self.root = Path(root)
        self.sequence = str(sequence)
        self.processed = self.root / 'processed_observed_v1' / sequence
        manifest = self.processed / 'manifest.csv'
        if not manifest.is_file():
            raise FileNotFoundError(f'{manifest}: first run pre_mvsec_observed.py')
        with manifest.open(newline='') as fp:
            self.rows = list(csv.DictReader(fp))
        if max_samples is not None:
            self.rows = self.rows[:max_samples]
        if not self.rows:
            raise RuntimeError(f'no rows in {manifest}')
        npz = self.root / 'flow_gt' / f'{sequence}_gt_flow_dist.npz'
        if not npz.is_file():
            raise FileNotFoundError(npz)
        cache = ensure_npz_cache(npz, self.root / 'flow_gt' / '.npy_cache' / sequence)
        self.u_path = str(cache['x_flow_dist'])
        self.v_path = str(cache['y_flow_dist'])
        self.u = None
        self.v = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if self.u is None:
            self.u = np.load(self.u_path, mmap_mode='r')
            self.v = np.load(self.v_path, mmap_mode='r')
        r = self.rows[index]
        i = int(r['flow_index'])
        indexes = [int(r[f'context_{k}']) for k in range(4)]
        if indexes != [i - 2, i - 1, i, i + 1]:
            raise RuntimeError(f'not an observed-interval manifest row: {i}, {indexes}')
        tokens = []
        for j in indexes:
            p = self.processed / 'eventToken' / f'{j:06d}.pt'
            z = torch.load(p, map_location='cpu', weights_only=True)
            if tuple(z.shape) != (256, 384):
                raise ValueError(f'bad tokens: {p}, {z.shape}')
            tokens.append(z.float())
        active = torch.load(self.processed / 'event_active' / f'{i+1:06d}.pt', map_location='cpu', weights_only=True)
        u = np.asarray(self.u[i], dtype=np.float32).copy()
        v = np.asarray(self.v[i], dtype=np.float32).copy()
        if u.shape != (NATIVE_H, NATIVE_W) or v.shape != (NATIVE_H, NATIVE_W):
            raise ValueError(f'unexpected MVSEC native flow: {u.shape}, {v.shape}')
        valid = flow_validity(u, v, self.sequence)
        gt = np.stack([np.nan_to_num(u, nan=0, posinf=0, neginf=0),
                       np.nan_to_num(v, nan=0, posinf=0, neginf=0)], axis=0)
        if tuple(active.shape) != (NATIVE_H, NATIVE_W):
            raise ValueError(f'event activity mask wrong geometry: {active.shape}')
        if int((valid & active.numpy()).sum()) == 0:
            raise RuntimeError(f'no event-conditioned valid GT pixels at {self.sequence} index {i}')
        return {
            'context': torch.stack(tokens, dim=0),
            'flow_gt': torch.from_numpy(gt),
            'gt_valid': torch.from_numpy(valid.copy()),
            'event_active': active.bool(),
            'flow_index': i,
            't0': float(r['t0']),
            't1': float(r['t1']),
        }
