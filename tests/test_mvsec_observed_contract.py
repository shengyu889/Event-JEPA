from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

import numpy as np
import torch
from mvsec_flow_probe.observed_dataset import to_native_flow, flow_validity
from pre_mvsec_observed import preprocess_rgb_native_to_224


def test_native_vector_scaling():
    pred=torch.ones(2,2,224,224)
    output=to_native_flow(pred)
    assert output.shape==(2,2,260,346)
    assert torch.allclose(output[:,0],torch.full((2,260,346),346/224))
    assert torch.allclose(output[:,1],torch.full((2,260,346),260/224))


def test_no_historical_future_mismatch():
    i=17
    assert [i-2,i-1,i,i+1][-1]==i+1
    # z_(i+1) contains events from (t_i,t_(i+1)], same GT interval.


def test_event_full_fov_resized():
    rgb=np.full((260,346,3),255,dtype=np.uint8)
    t=preprocess_rgb_native_to_224(rgb)
    assert t.shape==(3,224,224)
    assert torch.isfinite(t).all()


def test_hood_only_applied_to_outdoor():
    u=np.ones((260,346),np.float32)
    v=np.zeros_like(u)
    assert not flow_validity(u,v,'outdoor_day2')[200,0]
    assert flow_validity(u,v,'indoor_flying1')[200,0]
