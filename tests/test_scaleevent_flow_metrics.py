import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scaleevent_flow_metrics import ScaleEventFlowMeter, event_activity_mask


def test_3pe_not_necessarily_outlier():
    # Two GT-valid event-active pixels, one with large GT motion.
    # err=4 for both, GT magnitudes=100 vs 10 => only second is Outlier.
    gt = torch.tensor([[[[100.0, 10.0]], [[0.0, 0.0]]]])
    pred = gt.clone()
    pred[:, 0] += 4
    mask = torch.ones(1, 1, 2, dtype=torch.bool)
    m = ScaleEventFlowMeter()
    m.update(pred, gt, mask, mask)
    x = m.compute()
    assert x['EPE'] == 4
    assert x['3PE%'] == 100.0
    assert x['Outlier%'] == 50.0
    assert x['3PE_not_Outlier_pixel_count'] == 1


def test_event_mask_enforced_and_micro_average():
    gt = torch.zeros(2, 2, 1, 2)
    pred = gt.clone()
    pred[0, 0, 0] = torch.tensor([1., 100.])
    pred[1, 0, 0] = torch.tensor([3., 100.])
    gv = torch.ones(2, 1, 2, dtype=torch.bool)
    ev = torch.zeros_like(gv)
    ev[:, 0, 0] = True
    m = ScaleEventFlowMeter()
    m.update(pred, gt, gv, ev)
    x = m.compute()
    assert x['valid_event_gt_pixels'] == 2
    assert x['gt_valid_without_event'] == 2
    assert x['EPE'] == 2.0
    assert x['Outlier%'] == 0.0  # strict >3


def test_event_window_and_coordinates():
    events = np.array([[2, 1, 0.10, 1], [0, 0, 0.15, -1], [3, 1, .20, 1], [2,1,.201,1], [99,3,.18,1]])
    a = event_activity_mask(events, 0.10, 0.20, height=3, width=4)
    assert a.sum() == 2
    assert a[0, 0] and a[1, 3] and not a[1, 2]


def test_invalid_dimensions():
    m = ScaleEventFlowMeter()
    try:
        m.update(torch.zeros(1,2,3,4), torch.zeros(1,2,3,4), torch.ones(1,3,4), torch.ones(1,4,4))
    except ValueError:
        pass
    else:
        raise AssertionError('expected shape mismatch to raise')
