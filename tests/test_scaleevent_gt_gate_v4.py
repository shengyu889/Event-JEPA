from pathlib import Path
import importlib.util
import numpy as np
import h5py
import pytest

P=Path(__file__).resolve().parents[1]/'scripts'/'scaleevent_gt_gate_v4.py'
spec=importlib.util.spec_from_file_location('scaleevent_gate_v4',P)
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)


def create(tmp_path,seq,n=9):
    d=tmp_path/'candidate'
    d.mkdir(exist_ok=True)
    np.savez_compressed(d/f'{seq}_gt_flow_dist.npz',timestamps=np.arange(n,dtype=np.float64)*.05+100.0,
                        x_flow_dist=np.ones((n,260,346),np.float32),y_flow_dist=np.zeros((n,260,346),np.float32))
    fam='outdoor_day' if seq.startswith('outdoor') else 'indoor_flying'
    raw=tmp_path/'original'/'raw'/fam/f'{seq}_data.hdf5'
    raw.parent.mkdir(parents=True,exist_ok=True)
    with h5py.File(raw,'w') as f:
        ev=np.array([[4,4,99.9,1],[4,4,100.5,-1]],dtype=np.float64)
        f['davis/left/events']=ev
        f['davis/left/image_raw_ts']=np.array([100,100.4])
        f['davis/left/image_raw_event_inds']=np.array([0,1])
    return d,tmp_path/'original'


def test_count_and_time_header(tmp_path):
    d,r=create(tmp_path,'outdoor_day2',n=9)
    info=g.check_gt(d/'outdoor_day2_gt_flow_dist.npz')
    assert info['ok'] and info['n_available_pairs']==8
    assert info['n_observed_tc4_compatible']==5
    assert abs(info['dt_ms']['median']-50)<1e-6
    assert g.check_h5(r/'raw/outdoor_day/outdoor_day2_data.hdf5',info)['ok']


def test_missing_gt(tmp_path):
    r=g.check_gt(tmp_path/'missing.npz')
    assert not r['ok']


def test_stage_denied_on_count_mismatch(tmp_path):
    d,r=create(tmp_path,'outdoor_day2')
    for s in g.SEQUENCES[1:]: create(tmp_path,s)
    report=g.make_report(d,r)
    assert not report['errors']
    assert report['published_counts_match'] is False
    with pytest.raises(RuntimeError):g.stage_sources(report,tmp_path/'staged')
    staged=g.stage_sources(report,tmp_path/'staged',allow_count_mismatch=True)
    assert (staged/'flow_gt/outdoor_day2_gt_flow_dist.npz').is_symlink()
    assert 'CANDIDATE' in (staged/'source_provenance_candidate.json').read_text()


def test_reject_broken_hdf5(tmp_path):
    d,r=create(tmp_path,'indoor_flying2')
    p=r/'raw/indoor_flying/indoor_flying2_data.hdf5'
    p.write_bytes(p.read_bytes()[:200])
    assert not g.check_h5(p,g.check_gt(d/'indoor_flying2_gt_flow_dist.npz'))['ok']


def test_compare_identical_timelines(tmp_path):
    d,r=create(tmp_path,'outdoor_day2')
    for seq in g.SEQUENCES[1:]:create(tmp_path,seq)
    report=g.make_report(d,r)
    g.compare_source_timelines(report,d)
    assert report['sequences']['outdoor_day2']['timeline_comparison']['match_1us_fraction']==1.0
