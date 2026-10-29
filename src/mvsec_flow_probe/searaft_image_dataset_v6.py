"""SEA-RAFT fine-tuning dataset based on 22/32ms per-image GT labels."""
from __future__ import annotations
from pathlib import Path
import csv
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from pre_mvsec_observed import preprocess_rgb_native_to_224
from mvsec_flow_probe.searaft_dataset import flow_to_model_coords

class MVSECImageTimeFlowDataset(Dataset):
    def __init__(self,root,seq,indices=None):
        self.root=Path(root);self.seq=seq
        self.base=self.root/'processed_image_time_v6'/seq
        manifest=self.base/'manifest.csv'
        if not manifest.is_file():raise FileNotFoundError(manifest)
        with manifest.open(newline='') as f: rows=list(csv.DictReader(f))
        self.rows=rows if indices is None else [rows[k] for k in indices]
        if not self.rows:raise ValueError(f'{seq}: no rows')
        for row in self.rows:
            js=[int(row[f'context_{i}']) for i in range(4)]
            if not all(js[i]+1==js[i+1] for i in range(3)) or js[-1]!=int(row['pair_index'])+1:
                raise ValueError(f'Invalid context indices {row}')
            for j in js:
                p=self.base/'eventRGB_native'/f'{j:06d}.png'
                if not p.is_file():raise FileNotFoundError(p)
            p=self.base/row['flow_file']
            if not p.is_file():raise FileNotFoundError(p)

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index]
        js=[int(row[f'context_{i}']) for i in range(4)]
        imgs=[]
        for j in js:
            with Image.open(self.base/'eventRGB_native'/f'{j:06d}.png') as f:
                rgb=np.asarray(f.convert('RGB'),dtype=np.uint8)
            imgs.append(preprocess_rgb_native_to_224(rgb))
        with np.load(self.base/row['flow_file'],allow_pickle=False) as d:
            gt=d['flow'].astype(np.float32,copy=True)
            valid=d['valid'].astype(bool,copy=True)
            active=d['event_active'].astype(bool,copy=True)
        if gt.shape!=(2,260,346) or valid.shape!=(260,346) or active.shape!=(260,346):
            raise ValueError('native GT or masks wrong dimensions')
        return {'events':torch.stack(imgs),'flow_gt':torch.from_numpy(gt),
                'gt_valid':torch.from_numpy(valid),'event_active':torch.from_numpy(active),
                'index':int(row['pair_index'])}
