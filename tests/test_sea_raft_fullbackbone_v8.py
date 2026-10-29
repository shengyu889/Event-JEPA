import csv
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
import torch.nn.functional as F

SRC=Path(__file__).resolve().parents[1]/'scripts'/'sea_raft_fullbackbone_v8.py'
spec=importlib.util.spec_from_file_location('sea_raft_fullbackbone_v8', SRC)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_v6_pair_indices_and_native_flow(tmp_path):
    root=tmp_path/'MVSEC'
    base=root/'processed_image_time_v6'/'indoor_flying1'
    (base/'eventRGB_native').mkdir(parents=True)
    (base/'flow').mkdir(parents=True)
    (base/'metadata.json').write_text('{"candidate_mode":"image_time_v6"}')
    for i in range(1,5):
        img=np.ones((260,346,3),np.uint8)*i
        Image.fromarray(img).save(base/'eventRGB_native'/f'{i:06d}.png')
    np.savez_compressed(base/'flow'/'000003.npz',flow=np.ones((2,260,346),np.float32),
                        valid=np.ones((260,346),np.uint8),event_active=np.ones((260,346),np.uint8))
    with (base/'manifest.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['pair_index','context_0','context_1','context_2','context_3','flow_file'])
        w.writeheader();w.writerow({'pair_index':3,'context_0':1,'context_1':2,'context_2':3,
                                     'context_3':4,'flow_file':'flow/000003.npz'})
    ds=mod.ImageTimeV6Pairs(root,'indoor_flying1',contexts=False)
    x=ds[0]
    assert x['pair'].shape==(2,3,260,346)
    assert int(x['pair'][0,0,0,0])==3
    assert int(x['pair'][1,0,0,0])==4
    assert x['flow'].shape==(2,260,346)
    assert x['context'].numel()==0
    ds2=mod.ImageTimeV6Pairs(root,'indoor_flying1',contexts=True)
    assert ds2[0]['context'].shape==(4,3,260,346)
    assert int(ds2[0]['context'][0,0,0,0])==1


def test_strict_raft_load_includes_fnet_cnet(tmp_path):
    class FakeFull(nn.Module):
        def __init__(self):
            super().__init__();self.fnet=nn.Conv2d(3,2,1);self.cnet=nn.Conv2d(6,2,1)
    src=FakeFull();dst=FakeFull()
    p=tmp_path/'upstream.pt'
    torch.save(src.state_dict(),p)
    d=mod.load_full_weights(dst,p)
    assert d['fnet_tensors']==2 and d['cnet_tensors']==2
    assert not d['missing'] and not d['unexpected']
    missing=src.state_dict().copy();del missing['fnet.weight']
    bad=tmp_path/'bad.pt';torch.save(missing,bad)
    with pytest.raises(RuntimeError):mod.load_full_weights(dst,bad)


class FakeStage1(nn.Module):
    def __init__(self):super().__init__();self.scale=nn.Parameter(torch.tensor(1.0))
    def forward_features(self,x):
        return {'x_norm_patchtokens':x.mean(dim=(1,2,3))[:,None,None].expand(-1,256,384)*self.scale}

class FakeStage2(nn.Module):
    def __init__(self):super().__init__();self.scale=nn.Parameter(torch.tensor(1.0))
    def forward(self,x):return x.reshape(x.shape[0],1024,384)*self.scale

class FakeRAFT(nn.Module):
    def __init__(self):
        super().__init__();self.args=SimpleNamespace(dim=128)
        self.fnet=nn.Conv2d(3,256,1)
        self.cnet=nn.Conv2d(6,256,1)
    def forward(self,a,b,flow_gt=None,test_mode=False):
        x=self.fnet(F.avg_pool2d(a,8))
        y=self.fnet(F.avg_pool2d(b,8))
        c=self.cnet(F.avg_pool2d(torch.cat([a,b],dim=1),8))
        z=(x+y+c).mean(dim=1,keepdim=True)
        flow=torch.cat([z,z],dim=1)
        return {'final':flow,'flow':[flow],'nf':[]}


def test_residual_fusion_does_not_replace_fnet_cnet_and_backprops():
    raft=FakeRAFT(); stage1=FakeStage1();stage2=FakeStage2()
    model=mod.ResidualEventFusion(raft,stage1,stage2,mode='jepa',train_encoders=False)
    model.train()
    assert not any(p.requires_grad for p in model.stage1.parameters())
    assert not any(p.requires_grad for p in model.stage2.parameters())
    assert model.raft.fnet.weight.requires_grad and model.raft.cnet.weight.requires_grad
    native=torch.randint(0,255,(1,4,3,260,346),dtype=torch.uint8)
    out=model(native[:,-2:],context=native)
    assert out['final'].ndim==4
    out['final'].mean().backward()
    assert raft.fnet.weight.grad is not None and raft.cnet.weight.grad is not None
    assert model.fusion_fnet[0].weight.grad is not None
    assert model.fusion_cnet[0].weight.grad is not None
    assert model.stage1.scale.grad is None
    assert model.stage2.scale.grad is None
    assert not raft.fnet._forward_hooks and not raft.cnet._forward_hooks


def test_gep_branch_skips_stage2():
    model=mod.ResidualEventFusion(FakeRAFT(),FakeStage1(),None,mode='gep',train_encoders=False)
    native=torch.randint(0,255,(1,4,3,260,346),dtype=torch.uint8)
    o=model(native[:,-2:],context=native)
    assert o['final'].shape[-2:]==(32,43)
