import ast
from pathlib import Path
import numpy as np

def test_training_v7_event_mask_in_computation():
    p=Path(__file__).resolve().parents[1]/'src/mvsec_flow_probe/train_searaft_image_v7_eventmask.py'
    s=p.read_text()
    ast.parse(s)
    assert 'valid=valid & active_224' in s
    assert 'loss_mask_protocol' in s
    assert 'stage2, stage2_cfg=build_stage2(a.stage2_ckpt)' in s

def test_all_8_axis_conventions():
    import importlib.util
    p=Path(__file__).resolve().parents[1]/'scripts/check_flow_axes_v7.py'
    spec=importlib.util.spec_from_file_location('axes_v7',str(p))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    p=np.array([[1.,2.,-3.],[4.,5.,6.]])
    v=m.vector_views(p)
    assert len(v)==8
    np.testing.assert_allclose(v['swap_xy'][0],p[1])
    np.testing.assert_allclose(v['negate_both'][1],-p[1])
    np.testing.assert_allclose(v['identity'][0],p[0])

def test_launcher_separate_output():
    root=Path(__file__).resolve().parents[1]
    s=(root/'scripts/run_searaft_image_v7_eventmask_full.sh').read_text()
    assert '--protocol full' in s
    assert 'runs/searaft_image_v7_eventmask_full_seed0' in s
    assert 'train_searaft_image_v7_eventmask.py' in s
