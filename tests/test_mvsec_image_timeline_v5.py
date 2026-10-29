import importlib.util
from pathlib import Path
import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1]/'scripts'/'audit_mvsec_image_timeline_v5.py'
spec = importlib.util.spec_from_file_location('audit_timeline',SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def test_counts_and_overlap():
    im=np.arange(0,1.01,0.025)
    gt=np.arange(0,1.01,0.05)
    r=mod.audit_timeline(im,gt,approximate_offset_seconds=0.1)
    assert r['image_consecutive_pairs']==40
    assert r['candidate_image_pairs_within_gt_range']==40
    assert r['candidate_image_pairs_with_approx_cutoff']==36
    assert r['candidate_image_pairs_with_both']==36
    assert abs(r['image_dt']['median_ms']-25)<1e-9
    assert abs(r['gt_dt']['median_ms']-50)<1e-9

def test_gt_overlap():
    im=np.array([-0.05,0,0.05,0.1,0.15])
    gt=np.array([0,0.05,0.1])
    r=mod.audit_timeline(im,gt,approximate_offset_seconds=0)
    assert r['candidate_image_pairs_within_gt_range']==2

def test_reject_nonmonotonic():
    with pytest.raises(ValueError,match='non-strictly'):
        mod.audit_timeline([0,1,0.5],[0,1],approximate_offset_seconds=0)
