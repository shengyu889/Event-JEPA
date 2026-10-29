"""MVSEC native GT + lossless observed-interval event RGB for SEA-RAFT FT."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from pre_mvsec_flow import ensure_npz_cache
from pre_mvsec_observed import preprocess_rgb_native_to_224
from mvsec_flow_probe.observed_dataset import flow_validity, NATIVE_H, NATIVE_W


def flow_to_model_coords(gt: torch.Tensor, valid: torch.Tensor,
                         height=224, width=224):
    """Native GT displacements -> model-resolution displacements and mask.

    gt: B,2,260,346; valid: B,260,346 (bool). Do not interpolate masks bilinearly.
    """
    if gt.ndim != 4 or gt.shape[1] != 2 or valid.shape != gt.shape[:1]+gt.shape[2:]:
        raise ValueError(f'wrong shapes gt={gt.shape}, valid={valid.shape}')
    h, w = gt.shape[-2:]
    # In native GT, invalid flow entries may be zero-filled. Bilinear resizing
    # must not let invalid vectors dilute valid motion near mask boundaries.
    coverage = F.interpolate(valid[:,None].float(), size=(height,width),
                             mode='bilinear', align_corners=False)
    numer = F.interpolate(gt * valid[:,None].to(gt.dtype), size=(height,width),
                          mode='bilinear', align_corners=False)
    small = numer / coverage.clamp_min(1e-6)
    small = small * gt.new_tensor([width/w, height/h]).view(1,2,1,1)
    mask = F.interpolate(valid[:,None].float(), size=(height,width),
                         mode='nearest')[:,0] > .5
    return small,mask


class MVSECSEAFlowDataset(Dataset):
    def __init__(self, root: str | Path, sequence: str, indices=None):
        self.root=Path(root); self.seq=str(sequence)
        self.processed=self.root/'processed_observed_v1'/self.seq
        manifest=self.processed/'manifest.csv'
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        with manifest.open(newline='') as f:
            all_rows=list(csv.DictReader(f))
        self.rows=all_rows if indices is None else [all_rows[i] for i in indices]
        if not self.rows: raise RuntimeError(f'{sequence}: empty training dataset')
        cache = ensure_npz_cache(
            self.root/'flow_gt'/f'{self.seq}_gt_flow_dist.npz',
            self.root/'flow_gt'/'.npy_cache'/self.seq)
        self.u_path=str(cache['x_flow_dist']);self.v_path=str(cache['y_flow_dist'])
        self.u=None;self.v=None
        # Fail before hours of training if inputs are missing
        relevant={int(r[f'context_{k}']) for r in self.rows for k in range(4)}
        for j in relevant:
            fp=self.processed/'eventRGB_native'/f'{j:06d}.png'
            if not fp.is_file():
                raise FileNotFoundError(f'{fp} -- run pre_mvsec_searaft.py')
        for r in self.rows:
            j=int(r['context_3']); i=int(r['flow_index'])
            if j != i+1 or [int(r[f'context_{k}']) for k in range(4)] != [i-2,i-1,i,i+1]:
                raise ValueError(f'not observed-aligned: {self.seq}:{i}')
            m=self.processed/'event_active'/f'{j:06d}.pt'
            if not m.is_file():
                raise FileNotFoundError(m)

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        if self.u is None:
            self.u=np.load(self.u_path,mmap_mode='r')
            self.v=np.load(self.v_path,mmap_mode='r')
        r=self.rows[index]; i=int(r['flow_index'])
        js=[int(r[f'context_{k}']) for k in range(4)]
        imgs=[]
        for j in js:
            with Image.open(self.processed/'eventRGB_native'/f'{j:06d}.png') as f:
                rgb=np.asarray(f.convert('RGB'),dtype=np.uint8)
            imgs.append(preprocess_rgb_native_to_224(rgb))
        u=np.asarray(self.u[i],dtype=np.float32).copy()
        v=np.asarray(self.v[i],dtype=np.float32).copy()
        if u.shape != (NATIVE_H,NATIVE_W) or v.shape!=(NATIVE_H,NATIVE_W):
            raise ValueError(f'wrong GT dims for {self.seq}:{i}: {u.shape}')
        valid=flow_validity(u,v,self.seq)
        gt=np.stack((np.nan_to_num(u,nan=0,posinf=0,neginf=0),
                     np.nan_to_num(v,nan=0,posinf=0,neginf=0)),axis=0)
        active=torch.load(self.processed/'event_active'/f'{i+1:06d}.pt',
                          map_location='cpu',weights_only=True).bool()
        if tuple(active.shape) != (NATIVE_H,NATIVE_W):
            raise ValueError(f'invalid activity shape {active.shape}')
        return {'events':torch.stack(imgs,dim=0), 'flow_gt':torch.from_numpy(gt),
                'gt_valid':torch.from_numpy(valid.copy()),'event_active':active,
                'index':i}
