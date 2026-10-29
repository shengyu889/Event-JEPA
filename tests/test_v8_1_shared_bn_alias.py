import importlib.util
from pathlib import Path
import pytest
import torch
from torch import nn

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/sea_raft_fullbackbone_v8.py'
spec = importlib.util.spec_from_file_location('fullbackbone_v8_1', SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

class SharedBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn3 = nn.BatchNorm2d(8)
        self.downsample = nn.Sequential(nn.Conv2d(8,8,1), self.bn3)

class ToyRaft(nn.Module):
    def __init__(self):
        super().__init__()
        self.fnet=nn.Module()
        self.cnet=nn.Module()
        for root in (self.fnet,self.cnet):
            root.layer2=nn.Sequential(SharedBlock())
            root.layer3=nn.Sequential(SharedBlock())

def test_missing_aliases_reconstructed_losslessly():
    a=ToyRaft()
    full = a.state_dict()
    sparse={k:v for k,v in full.items() if '.downsample.1.' not in k}
    fixed, keys=mod.restore_shared_batchnorm_aliases(a,sparse)
    assert len(keys)==20  # 4 blocks, 5 BN entries including num_batches_tracked
    assert all('.downsample.1.' in k for k in keys)
    a.load_state_dict(fixed,strict=True)
    for key in keys:
        assert torch.equal(fixed[key],full[key])

def test_real_missing_tensor_is_not_hidden():
    a=ToyRaft()
    full=a.state_dict()
    sparse={k:v for k,v in full.items() if '.downsample.1.' not in k}
    sparse.pop('fnet.layer2.0.bn3.weight')
    with pytest.raises(RuntimeError,match='lacks its actual source'):
        mod.restore_shared_batchnorm_aliases(a,sparse)

def test_other_missing_weight_keeps_strict_load_failure():
    a=ToyRaft()
    sparse={k:v for k,v in a.state_dict().items() if '.downsample.1.' not in k}
    sparse.pop('fnet.layer2.0.downsample.0.weight')
    fixed,_=mod.restore_shared_batchnorm_aliases(a,sparse)
    with pytest.raises(RuntimeError,match='Missing key'):
        a.load_state_dict(fixed,strict=True)

def test_no_mutation_of_checkpoint_mapping():
    a=ToyRaft()
    sparse={k:v for k,v in a.state_dict().items() if '.downsample.1.' not in k}
    before=set(sparse)
    _,keys=mod.restore_shared_batchnorm_aliases(a,sparse)
    assert set(sparse)==before
    assert len(keys)==20
