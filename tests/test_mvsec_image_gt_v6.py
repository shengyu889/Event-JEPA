import sys
import types
from pathlib import Path
import numpy as np
import pytest

# Import the standalone geometry module without pulling in DINOv2, PyTorch,
# stage1 checkpoints, or raw HDF5 in unit tests.
if 'pre_mvsec_flow' not in sys.modules:
    stub=types.ModuleType('pre_mvsec_flow')
    stub.ensure_npz_cache=lambda *_:None
    stub.event_index_at_time=lambda *_1,**_2:None
    stub.make_event_rgb=lambda *_1,**_2:None
    sys.modules['pre_mvsec_flow']=stub

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'))
from pre_mvsec_image_time_v6 import compose_gt, intervals


def fields(value1, value2=None):
    u=np.full((2,260,346),value1,dtype=np.float32)
    v=np.zeros_like(u)
    if value2 is not None:u[1]=value2
    return u,v


def test_single_bin_displacement_scale():
    u,v=fields(4.)
    flow,mask=compose_gt(np.array([0.,.05,.10]),u,v,.005,.025)
    assert np.allclose(flow[0,:,0:345],1.6,atol=1e-4)
    assert np.all(mask[:,0:343])


def test_short_interval_crossing_gt_boundary():
    # A .02s image pair straddles 50ms and needs TWO distinct GT maps.
    u,v=fields(4.,8.)
    flow,mask=compose_gt(np.array([0.,.05,.10]),u,v,.040,.060)
    # First .01s * (4/.05) = .8; second .01s * (8/.05)=1.6.
    assert np.isclose(flow[0,30,30],2.4,atol=1e-5)
    assert mask[30,30]


def test_zero_and_out_of_bounds_invalid():
    u,v=fields(1000.)
    flow,mask=compose_gt(np.array([0.,.05,.10]),u,v,.005,.045)
    assert not np.any(mask)
    assert not np.any(flow)
    u[:]=0
    _,mask=compose_gt(np.array([0.,.05,.10]),u,v,.005,.025)
    assert not mask.any()


def test_context_window_is_observed_interval():
    im=np.arange(9,dtype=np.float64)*.02
    gt=np.arange(5,dtype=np.float64)*.05
    rows=intervals(im,gt,context=4,strict_context_in_gt=True)
    assert rows[0]==(3,(1,2,3,4))
    assert rows[-1]==(7,(5,6,7,8))
    # Strict context in GT can discard the first 3 intervals if GT starts later.
    rows=intervals(im,gt_ts=np.array([.06,.10,.15,.20]),context=4,strict_context_in_gt=True)
    assert rows[0][0]==6


def test_no_unsupported_gt_range():
    u,v=fields(2)
    with pytest.raises(ValueError):
        compose_gt(np.array([0,.05,.1]),u,v,.08,.11)
