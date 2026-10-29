#!/usr/bin/env python3
"""Read-only MVSEC image/GT timeline audit for a ScaleEvent-like flow protocol.

Does not generate flow labels and DOES NOT prove the ScaleEvent Table 6 split.
The EV-FlowNet cutoffs below are bag-relative in upstream code. Here they are
used ONLY as a first-image-relative diagnostic because HDF5 bag-start may differ.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import h5py

SEQUENCES = ('outdoor_day2', 'indoor_flying1', 'indoor_flying2', 'indoor_flying3')
EVFLOW_BAG_OFFSETS = dict(outdoor_day2=45.0, indoor_flying1=4.0, indoor_flying2=9.0, indoor_flying3=7.0)
PAPER_TRAIN, PAPER_TEST = 26677, 7775


def summarize_deltas(ts):
    dif = np.diff(ts)
    if not len(dif):
        return None
    return {'min_ms': float(np.min(dif)*1000), 'median_ms':float(np.median(dif)*1000),
            'p05_ms':float(np.percentile(dif,5)*1000),
            'p95_ms':float(np.percentile(dif,95)*1000),
            'max_ms':float(np.max(dif)*1000)}


def audit_timeline(image_ts, gt_ts, *, approximate_offset_seconds):
    im = np.asarray(image_ts, dtype=np.float64)
    gt = np.asarray(gt_ts, dtype=np.float64)
    if im.ndim != 1 or gt.ndim != 1 or len(im)<2 or len(gt)<2:
        raise ValueError('image_ts and gt_ts must be 1D vectors with >=2 entries')
    if not np.isfinite(im).all() or not np.isfinite(gt).all():
        raise ValueError('non-finite timestamps')
    if not np.all(np.diff(im)>0) or not np.all(np.diff(gt)>0):
        raise ValueError('non-strictly increasing timestamps')
    # Consecutive image intervals, not pseudo GT. The GT may span a 50 ms
    # period, while image sampling occurs more frequently (e.g., 45 Hz).
    pair_start, pair_end = im[:-1], im[1:]
    covered = (pair_start >= gt[0]) & (pair_end <= gt[-1])
    cutoff = im[0] + approximate_offset_seconds
    approx_trim = pair_start >= cutoff
    return {
        'images':int(im.size),
        'image_consecutive_pairs':int(im.size-1),
        'image_dt':summarize_deltas(im),
        'image_first_s':float(im[0]), 'image_last_s':float(im[-1]),
        'gt_timestamps':int(gt.size), 'gt_nominal_pairs':int(gt.size-1),
        'gt_dt':summarize_deltas(gt),
        'gt_first_s':float(gt[0]), 'gt_last_s':float(gt[-1]),
        'candidate_image_pairs_within_gt_range':int(covered.sum()),
        'candidate_image_pairs_with_approx_cutoff':int(approx_trim.sum()),
        'candidate_image_pairs_with_both':int((covered & approx_trim).sum()),
        'approx_cutoff_seconds_relative_to_FIRST_IMAGE':float(approximate_offset_seconds),
        'WARNING':'Candidate image intervals are NOT valid GT automatically. Optical flow must be integrated/composed between exact image timestamps; the EV-FlowNet upstream cutoff refers to BAG start, not necessarily first image.'
    }


def load_gt_timestamps(root, seq):
    # Prefer already-created v4/v1 cache: never read dense u/v arrays.
    cached = root/'flow_gt'/'.npy_cache'/seq/'timestamps.npy'
    if cached.is_file():
        a = np.load(cached, mmap_mode='r', allow_pickle=False)
        return np.asarray(a, dtype=np.float64)
    original = root/'flow_gt'/f'{seq}_gt_flow_dist.npz'
    if not original.is_file():
        raise FileNotFoundError(f'neither cached timestamps nor NPZ exists for {seq}: {original}')
    import zipfile
    with zipfile.ZipFile(original) as z, z.open('timestamps.npy') as fp:
        return np.load(fp, allow_pickle=False).astype(np.float64)


def run(root):
    report={'protocol':'timeline-evidence-only','root':str(root.resolve()),
            'paper_reported_counts':{'train':PAPER_TRAIN,'test_total':PAPER_TEST},
            'sequences':{},'errors':[]}
    for seq in SEQUENCES:
        family = 'outdoor_day' 
        h5_path = root/'raw'/family/f'{seq}_data.hdf5'
        try:
            if not h5_path.is_file():
                raise FileNotFoundError(h5_path)
            with h5py.File(h5_path,'r') as f:
                image_ts = f['davis/left/image_raw_ts'][:]
                image_event_indices = f['davis/left/image_raw_event_inds']
                if len(image_ts) != len(image_event_indices):
                    raise ValueError('image_ts and image_event_inds length differ')
                event_count = int(len(f['davis/left/events']))
            gt_ts = load_gt_timestamps(root,seq)
            data = audit_timeline(image_ts,gt_ts,approximate_offset_seconds=EVFLOW_BAG_OFFSETS[seq])
            data['hdf5_path']=str(h5_path)
            data['raw_events']=event_count
            report['sequences'][seq]=data
        except Exception as e:
            report['errors'].append(f'{seq}: {type(e).__name__}: {e}')
            report['sequences'][seq]={'ERROR':str(e)}
    if not report['errors']:
        ks=('image_consecutive_pairs','candidate_image_pairs_within_gt_range',
            'candidate_image_pairs_with_approx_cutoff','candidate_image_pairs_with_both')
        report['counts_by_definition']={k:{
            'outdoor_day2':report['sequences']['outdoor_day2'][k],
            'indoor_sum':sum(report['sequences'][s][k] for s in SEQUENCES[1:])
        } for k in ks}
        report['count_matches_paper_exactly']={k:(v['outdoor_day2']==PAPER_TRAIN and v['indoor_sum']==PAPER_TEST) for k,v in report['counts_by_definition'].items()}
    report['warnings']=[
        'A matching image-frame count does not establish ScaleEvent annotation sampling or metric protocol.',
        'Do not generate GT by repeating 50ms labels or simple time scaling for nonconstant motion.',
        'EV-FlowNet upstream start_time values are relative to ROS bag start, so approximate HDF5 first-image cutoffs are merely diagnostic.',
        'ScaleEvent reports initializing downstream decoders with released pretrained weights; verify actual SEA-RAFT initialization before comparing.'
    ]
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('datasets/MVSEC'))
    p.add_argument('--output',type=Path,default=Path('results/mvsec_image_timeline_v5.json'))
    args=p.parse_args()
    d=run(args.root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(d,indent=2,ensure_ascii=False))
    print('MVSEC image-vs-GT timeline audit v5')
    print(f"{'Sequence':<20} {'images':>9} {'image pairs':>12} {'in GT range':>12} {'img ms':>11} {'GT ms':>11}")
    for seq,info in d['sequences'].items():
        if 'ERROR' in info:
            print(seq, 'ERROR',info['ERROR']);continue
        print(f"{seq:<20} {info['images']:>9} {info['image_consecutive_pairs']:>12} {info['candidate_image_pairs_within_gt_range']:>12} {info['image_dt']['median_ms']:>11.3f} {info['gt_dt']['median_ms']:>11.3f}")
    if 'counts_by_definition' in d:
        print('Published train/test:',PAPER_TRAIN,PAPER_TEST)
        for k,v in d['counts_by_definition'].items():
            print(f'{k}: {v["outdoor_day2"]} train / {v["indoor_sum"]} indoor, exact-match={d["count_matches_paper_exactly"][k]}')
    for err in d['errors']: print('ERROR:',err)
    print('JSON:',args.output)
    print('DISCLAIMER: No new GT was generated. This does not validate ScaleEvent Table 6 comparability.')
    if d['errors']: raise SystemExit(2)

if __name__ == '__main__':main()
