#!/usr/bin/env python3
"""Inspect an MVSEC-Flow GT release without decompressing its huge flow arrays.

The declared ScaleEvent sequence totals alone are *not* proof of the same GT
protocol. Does not modify old data, caches, model checkpoints, or training code.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import zipfile
from pathlib import Path

import h5py
import numpy as np
from numpy.lib import format as npfmt

SEQUENCES = ('outdoor_day2', 'indoor_flying1', 'indoor_flying2', 'indoor_flying3')
PUBLISHED_TRAIN = 26677
PUBLISHED_TEST = 7775
FLOW_NAME = '{seq}_gt_flow_dist.npz'
RAW_H, RAW_W = 260, 346


def npy_meta(f):
    """Read just .npy header (stream and file objects accepted)."""
    major, minor = npfmt.read_magic(f)
    if major == 1:
        shape, order, dtype = npfmt.read_array_header_1_0(f)
    else:
        shape, order, dtype = npfmt.read_array_header_2_0(f)
    return dict(shape=list(shape), dtype=str(dtype), fortran=bool(order), version=f'{major}.{minor}')


def check_gt(path: Path) -> dict:
    r = {'path': str(path), 'exists': path.is_file(), 'ok': False}
    if not path.is_file():
        r['error'] = 'GT NPZ file not found'
        return r
    r['file_bytes'] = path.stat().st_size
    try:
        with zipfile.ZipFile(path) as z:
            files = set(z.namelist())
            names = ['timestamps.npy', 'x_flow_dist.npy', 'y_flow_dist.npy']
            r['members'] = sorted(files)
            if not set(names).issubset(files):
                r['error'] = f'missing arrays: {sorted(set(names)-files)}'
                return r
            with z.open('timestamps.npy') as f:
                ts = np.load(f, allow_pickle=False).astype(np.float64)
            r['arrays'] = {}
            for name in names:
                with z.open(name) as f:
                    r['arrays'][name.removesuffix('.npy')] = npy_meta(f)
            u = r['arrays']['x_flow_dist']['shape']
            v = r['arrays']['y_flow_dist']['shape']
            if u != v or len(u) != 3 or u[1:] != [RAW_H, RAW_W]:
                r['error'] = f'GT shape mismatch or non-native dimensions: u={u}, v={v}'
                return r
            if ts.ndim != 1 or len(ts) < 6 or not np.isfinite(ts).all() or not np.all(np.diff(ts)>0):
                r['error'] = 'timestamps must be a finite strictly increasing 1D vector with >=6 entries'
                return r
            r['n_timestamps'] = int(len(ts))
            r['n_flow_arrays'] = int(u[0])
            r['n_available_pairs'] = int(min(len(ts)-1, u[0]))
            r['n_observed_tc4_compatible'] = max(0, r['n_available_pairs'] - 3)
            # These are separate counters: ScaleEvent has not released a per-index list.
            r['timestamp_start'] = float(ts[0])
            r['timestamp_end'] = float(ts[-1])
            dt = np.diff(ts) * 1000
            r['dt_ms'] = {k: round(float(val), 6) for k,val in (
                ('min',dt.min()),('median',np.median(dt)),('p05',np.percentile(dt,5)),
                ('p95',np.percentile(dt,95)),('max',dt.max()))}
            r['flow_interval_semantics'] = 'NOT PROVEN by NPZ headers alone'
            r['ok'] = True
    except Exception as exc:
        r['error'] = f'{type(exc).__name__}: {exc}'
    return r


def check_h5(path: Path, gt: dict) -> dict:
    d={'path':str(path), 'exists':path.is_file(),'ok':False}
    if not path.is_file():
        d['error']='raw HDF5 file not found'; return d
    d['file_bytes']=path.stat().st_size
    try:
        with h5py.File(path,'r') as h5:
            ev=h5['davis/left/events']
            its=h5['davis/left/image_raw_ts']
            inds=h5['davis/left/image_raw_event_inds']
            if len(ev)==0 or len(its)==0 or len(inds)!=len(its):
                raise ValueError('empty/misaligned event/image index arrays')
            d['events']=list(ev.shape)
            d['image_timestamps']=int(len(its))
            d['event_first_s']=float(ev[0,2]); d['event_last_s']=float(ev[-1,2])
            d['image_first_s']=float(its[0]); d['image_last_s']=float(its[-1])
            # Reject if interval clocks not aligned; permit one timestamp bucket of slack.
            if gt.get('ok'):
                t0,t1=gt['timestamp_start'],gt['timestamp_end']
                dt=max(0.001,gt['dt_ms']['median']/1000)
                d['gt_within_event_range']=(t0 >= d['event_first_s']-dt and t1<=d['event_last_s']+dt)
                if not d['gt_within_event_range']:
                    d['warning']='GT timestamps not enclosed by raw event clocks (check sensor offset/unit)'
            d['ok']=True
    except Exception as exc:
        d['error']=f'{type(exc).__name__}: {exc}'
    return d


def find_gt(labels: Path, seq: str):
    direct=[labels/f'{seq}_gt_flow_dist.npz', labels/'flow_gt'/f'{seq}_gt_flow_dist.npz']
    matches=[p for p in direct if p.is_file()]
    if not matches:
        matches=list(labels.rglob(f'{seq}_gt_flow_dist.npz'))
    if len(matches)==1: return matches[0], None
    if len(matches)>1: return None, f'Ambiguous GT: {matches[:8]}'
    return direct[0], None


def make_report(labels: Path, raw_root: Path) -> dict:
    report={'candidate_labels':str(labels.resolve()), 'raw_dataset':str(raw_root.resolve()),
            'paper_counts':{'train':PUBLISHED_TRAIN,'test_total':PUBLISHED_TEST},
            'sequences':{},'warnings':[], 'errors':[], 'strict_scaleevent_match':False}
    for seq in SEQUENCES:
        path,ambiguous=find_gt(labels,seq)
        gt=check_gt(path) if path is not None else {'ok':False,'error':ambiguous}
        family='outdoor_day'
        raw=raw_root/'raw'/family/f'{seq}_data.hdf5'
        h5=check_h5(raw,gt)
        report['sequences'][seq]={'gt':gt,'raw':h5}
        if not gt.get('ok') or not h5.get('ok'):
            report['errors'].append(f'{seq}: {gt.get("error","")} {h5.get("error","")}'.strip())
        if h5.get('ok') and h5.get('gt_within_event_range') is False:
            report['errors'].append(f'{seq}: mismatched event/GT time range')
    good=not report['errors']
    if good:
        counts={}
        matched=[]
        for metric in ['n_available_pairs','n_observed_tc4_compatible']:
            tr=report['sequences']['outdoor_day2']['gt'][metric]
            te=sum(report['sequences'][s]['gt'][metric] for s in SEQUENCES[1:])
            counts[metric]={'train':tr,'test_total':te}
            if tr==PUBLISHED_TRAIN and te==PUBLISHED_TEST:
                matched.append(metric)
        report['observed_counts']=counts
        report['count_matching_conventions']=matched
        report['published_counts_match']=bool(matched)
        if not matched:
            report['warnings'].append('Candidate label counts differ from ScaleEvent reported sample numbers under both raw-pair and Tc4-compatible conventions.')
        else:
            report['warnings'].append('Counts match under a possible convention, but GT provenance, directions, masks, sampling and voxel implementation remain unverified.')
    report['warnings'].append('Do not claim strict ScaleEvent replication from count matching alone; the released SEA-RAFT code lacks its MVSEC-specific flow trainer.')
    return report



def load_timestamps_only(path: Path) -> np.ndarray:
    with zipfile.ZipFile(path) as z, z.open('timestamps.npy') as f:
        ts=np.load(f,allow_pickle=False)
    return np.asarray(ts,dtype=np.float64)


def compare_source_timelines(report: dict, reference_labels: Path) -> None:
    """Compare new candidate GT time grids to the old MVSEC GT release."""
    report['timeline_reference']=str(reference_labels.resolve())
    for seq in SEQUENCES:
        target_path=report['sequences'][seq]['gt'].get('path')
        ref_path,ambiguous=find_gt(reference_labels,seq)
        if not target_path or ref_path is None or not Path(target_path).exists() or not ref_path.exists():
            report['sequences'][seq]['timeline_comparison']={'error':ambiguous or 'missing candidate or reference'}
            continue
        try:
            a=load_timestamps_only(Path(target_path))
            b=load_timestamps_only(ref_path)
            left=np.searchsorted(b,a,side='left')
            hi=b[np.clip(left,0,len(b)-1)]
            lo=b[np.clip(left-1,0,len(b)-1)]
            dist=np.minimum(abs(hi-a),abs(lo-a))
            report['sequences'][seq]['timeline_comparison']={
                'candidate_count':int(len(a)), 'reference_count':int(len(b)),
                'candidate_dt_ms':float(np.median(np.diff(a))*1000),
                'reference_dt_ms':float(np.median(np.diff(b))*1000),
                'match_1us_fraction':round(float(np.mean(dist<=1e-6)),6),
                'match_1ms_fraction':round(float(np.mean(dist<=1e-3)),6),
                'nearest_difference_ms_p95':round(float(np.percentile(dist,95)*1000),6),
            }
        except Exception as exc:
            report['sequences'][seq]['timeline_comparison']={'error':f'{type(exc).__name__}: {exc}'}

def stage_sources(report:dict, root:Path, *,allow_count_mismatch=False):
    if report['errors']:
        raise RuntimeError('Source files did not pass validation; staging refused.')
    if not report.get('published_counts_match') and not allow_count_mismatch:
        raise RuntimeError('Label counts do not match paper. Refusing to stage by default; use --allow-count-mismatch for diagnostic staging ONLY.')
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f'Target staging directory is not empty: {root}; refusing to overwrite.')
    for seq in SEQUENCES:
        fam='indoor_flying' if seq.startswith('indoor_flying') else 'outdoor_day'
        record=report['sequences'][seq]
        for source,dest in [(Path(record['gt']['path']),root/'flow_gt'/f'{seq}_gt_flow_dist.npz'),
                            (Path(record['raw']['path']),root/'raw'/fam/f'{seq}_data.hdf5')]:
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.symlink_to(source.resolve())
    gate={**report,'staged_at':str(root.resolve()),'stage_status':'CANDIDATE ONLY; NOT SCALEEVENT VERIFIED'}
    (root/'source_provenance_candidate.json').write_text(json.dumps(gate,indent=2,ensure_ascii=False))
    return root


def main():
    pa=argparse.ArgumentParser(description=__doc__)
    pa.add_argument('--candidate-labels',type=Path,required=True,help='Directory holding downloaded MVSEC-Flow *_gt_flow_dist.npz')
    pa.add_argument('--raw-root',type=Path,default=Path('datasets/MVSEC'))
    pa.add_argument('--reference-labels',type=Path,help='OPTIONAL: compare timestamps against existing 20Hz flow labels')
    pa.add_argument('--output',type=Path,default=Path('results/scaleevent_gt_gate_v4.json'))
    pa.add_argument('--stage-root',type=Path,help='OPTIONAL: new separate root containing only symlinks to validated raw files')
    pa.add_argument('--allow-count-mismatch',action='store_true',help='Stage mismatched data for diagnostics only; NEVER for paper claim')
    a=pa.parse_args()
    report=make_report(a.candidate_labels,a.raw_root)
    if a.reference_labels:
        compare_source_timelines(report,a.reference_labels)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print('MVSEC-FLOW SOURCE PROVENANCE GATE v4')
    print(f'{"Sequence":20s} {"GT pairs":>10} {"Tc4-compatible":>16} {"median dt(ms)":>16} {"raw OK":>8}')
    for seq,x in report['sequences'].items():
        g=x['gt'];h=x['raw'];n=g.get('n_available_pairs','-')
        print(f'{seq:20s} {str(n):>10s} {str(g.get("n_observed_tc4_compatible","-")):>16s} '
              f'{str(g.get("dt_ms",{}).get("median","-")):>16s} {str(h.get("ok")):>8s}')
    print('Published counts: 26677 train / 7775 test')
    print('Observed:',report.get('observed_counts','INCOMPLETE'))
    if a.reference_labels:
        for seq,data in report['sequences'].items():
            print('TIMELINE',seq,data.get('timeline_comparison'))
    print('Count match:',report.get('published_counts_match',False))
    if report['errors']:
        print('ERRORS:');[print('  -',v) for v in report['errors']]
    for msg in report['warnings']:print('CAUTION:',msg)
    print('Saved:',a.output)
    if a.stage_root:
        try:
            out=stage_sources(report,a.stage_root,allow_count_mismatch=a.allow_count_mismatch)
            print('DIAGNOSTIC STAGING READY:',out)
        except RuntimeError as exc:
            print('STAGING BLOCKED:',exc)
            return 2
    return 0 if not report['errors'] else 2

if __name__=='__main__':raise SystemExit(main())
