"""Deterministic CPU-only contract tests; do not substitute for real A100 integration."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
import pytest
from mvsec_flow_probe.searaft_corr import CorrBlock
from mvsec_flow_probe.searaft_model import training_policy, coords_grid, SEAEventDecoder
from types import SimpleNamespace

@pytest.mark.parametrize('mode,s1,s2',[
    ('frozen',False,False),('ft10',True,True),('full',True,True),('stage2-only',False,True)])
def test_protocol(mode,s1,s2):
    p=training_policy(mode)
    assert p['stage1']==s1 and p['stage2']==s2 and p['decoder']

def test_flow_coords():
    pos=coords_grid(2,3,4,'cpu')
    assert pos.shape==(2,2,3,4)
    assert pos[0,0,2,3]==3 and pos[1,1,2,3]==2

def test_corr_gradients():
    a=torch.randn(1,8,8,8,requires_grad=True)
    b=torch.randn(1,8,8,8,requires_grad=True)
    cfg=SimpleNamespace(corr_levels=2,corr_radius=2)
    blk=CorrBlock(a,b,cfg)
    out=blk(coords_grid(1,8,8,'cpu'))
    assert out.shape==(1,50,8,8)
    out.square().mean().backward()
    assert a.grad is not None and b.grad is not None


def test_nf_loss_finite():
    gt=torch.zeros(1,2,8,8)
    pred=torch.ones_like(gt)
    info=torch.zeros(1,4,8,8)
    out=SEAEventDecoder._nf_loss(gt,pred,info)
    assert out.shape==(1,2,8,8)
    assert out.isfinite().all()
