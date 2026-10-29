"""Core baseline tests; no real SEA-RAFT code or GPU needed."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from torch import nn
import torch.nn.functional as F
from unittest.mock import patch
import mvsec_flow_probe.searaft_model as m


class FakeStage1(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj=nn.Linear(3,384)
    def forward_features(self,x):
        z=F.adaptive_avg_pool2d(x,(16,16)).permute(0,2,3,1).reshape(x.shape[0],256,3)
        return {'x_norm_patchtokens':self.proj(z)}


class ForbiddenStage2(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=nn.Parameter(torch.ones(1))
        self.called=False
    def forward(self,tokens):
        self.called=True
        raise AssertionError('GEP baseline must never use JEPA Stage2')


class FakeDecoder(nn.Module):
    def __init__(self,*args,**kwargs):
        super().__init__()
        self.scale=nn.Parameter(torch.ones(1))
    def forward(self,first,second,flow_gt=None):
        pred=F.interpolate((first[:,:2]+second[:,:2])*self.scale,size=(224,224),mode='bilinear',align_corners=False)
        return {'final':pred,'flow':[pred],'nf':[]}


def test_gep_stage1_only_policy():
    policy=m.training_policy('stage1-only')
    assert policy=={'stage1':True,'stage2':False,'decoder':True}
    assert m.training_policy('full')['stage2'] is True


def test_gep_stage1_only_bypasses_stage2_and_backprop():
    torch.manual_seed(7)
    with patch.object(m,'SEAEventDecoder',FakeDecoder):
        s1=FakeStage1();s2=ForbiddenStage2()
        model=m.EventJEPASEARAFT(s1,s2,'stage1-only',checkpoint_encoder=False)
        model.train()
        x=torch.randn(1,4,3,224,224)
        output=model(x)
        assert tuple(output['final'].shape)==(1,2,224,224)
        output['final'].square().mean().backward()
        assert not s2.called
        assert all(not p.requires_grad for p in s2.parameters())
        assert all(p.grad is None for p in s2.parameters())
        assert s1.proj.weight.grad is not None
        assert float(s1.proj.weight.grad.abs().sum())>0
        assert model.fmap_adapter[0].weight.grad is not None
        assert float(model.fmap_adapter[0].weight.grad.abs().sum())>0
        assert model.decoder.scale.grad is not None
