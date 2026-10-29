"""Unit test the non-negotiable RAFT freeze rule using only a CPU model."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import torch
from torch import nn

SCRIPT = Path(__file__).parents[1] / 'scripts' / 'sea_raft_fullbackbone_v8_2.py'
spec = importlib.util.spec_from_file_location('eventjepa_searaft_v8_2', SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

class FakeRAFT(nn.Module):
    def __init__(self):
        super().__init__()
        self.args = type('Args', (), {'dim':128})()
        self.fnet = nn.Sequential(nn.Conv2d(3, 3, 1), nn.BatchNorm2d(3))
        self.cnet = nn.Sequential(nn.Conv2d(6, 6, 1), nn.BatchNorm2d(6))

class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

def test_full_raft_freeze_survives_train_and_bn_stats_unchanged():
    model = mod.ResidualEventFusion(FakeRAFT(), FakeEncoder(), None, mode='gep', train_encoders=False)
    model.train()
    mod.freeze_full_raft(model)
    assert model.raft.training is False
    assert all(not p.requires_grad for p in model.raft.parameters())
    assert all(p.requires_grad for p in model.fusion_fnet.parameters())
    assert all(not p.requires_grad for p in model.stage1.parameters())
    bn = model.raft.fnet[1]
    before_mean = bn.running_mean.detach().clone()
    before_count = bn.num_batches_tracked.item()
    model.raft.fnet(torch.randn(4,3,8,8))
    assert torch.equal(bn.running_mean,before_mean)
    assert bn.num_batches_tracked.item()==before_count
    # Simulates start of second training epoch.
    model.train()
    mod.freeze_full_raft(model)
    assert not model.raft.cnet[1].training

def test_freeze_refuses_nonfusion_model():
    try:
        mod.freeze_full_raft(FakeRAFT())
    except TypeError:
        pass
    else:
        raise AssertionError('Expected non-fusion input to be rejected')


def test_fusion_adapters_receive_gradient_through_fixed_network():
    model = mod.ResidualEventFusion(FakeRAFT(), FakeEncoder(), None,mode='gep',train_encoders=False)
    mod.freeze_full_raft(model)
    # The frozen convolution must transmit gradient to the adapter input.
    x = torch.randn(2,3,8,8,requires_grad=True)
    y = model.raft.fnet(x).square().mean()
    y.backward()
    assert x.grad is not None and x.grad.abs().sum()>0
    assert all(p.grad is None for p in model.raft.parameters())
