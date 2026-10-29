#!/usr/bin/env python3
"""Audit v8.2 frozen-RAFT GEP/JEPA fusion without changing checkpoints.

Checks: exact original SEA-RAFT state vs trained fusion ckpt, strict checkpoint
load, zero-gate prediction parity, and actual-fusion EPE on SAME MVSEC v6 samples.
This diagnostic never trains and never modifies existing checkpoints.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def import_v82(path: Path):
    spec = importlib.util.spec_from_file_location('eventjepa_v82_audited', str(path.resolve()))
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import v8.2 file: {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare_module_tensors(expected: dict, fusion_ckpt: dict, prefix: str):
    """Require every frozen module tensor to match the pretrained state bit-for-bit."""
    original_keys = set(expected)
    ckpt_module = {k[len(prefix):]: value for k, value in fusion_ckpt.items()
                   if k.startswith(prefix)}
    missing = sorted(original_keys - set(ckpt_module))
    extra = sorted(set(ckpt_module) - original_keys)
    different = []
    for name in sorted(original_keys & set(ckpt_module)):
        a, b = expected[name], ckpt_module[name]
        if (a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a, b)):
            different.append(name)
    return {'expected_tensors': len(original_keys), 'checkpoint_tensors': len(ckpt_module),
            'missing': missing[:20], 'extra': extra[:20],
            'changed': different[:20], 'changed_count': len(different),
            'exact_match': not (missing or extra or different)}


def compare_base_tensors(expected: dict, fusion_ckpt: dict):
    return compare_module_tensors(expected, fusion_ckpt, 'raft.')


def flow_statistics(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor):
    if not bool(mask.any()):
        return (0., 0., 0., 0., 0)
    epe = torch.linalg.vector_norm(pred.float() - gt.float(), dim=1)
    mag = torch.linalg.vector_norm(pred.float(), dim=1)
    gt_mag = torch.linalg.vector_norm(gt.float(), dim=1)
    cos = F.cosine_similarity(pred.float(), gt.float(), dim=1, eps=1e-6)
    nonzero = mask & (mag > 1e-3) & (gt_mag > 1e-3)
    return (epe[mask].sum().item(), mag[mask].sum().item(),
            cos[nonzero].sum().item(), int(nonzero.sum().item()),
            int(mask.sum().item()))


def audited_predictions(model, pair, context, gt):
    """Same model and inputs: original RAFT; gate=0; trained gates."""
    model.eval()
    with torch.inference_mode():
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=pair.device.type == 'cuda'):
            raw = model.raft(pair[:, 0].float(), pair[:, 1].float(), test_mode=True)['final'].float()
            before_fnet = model.gate_fnet.detach().clone()
            before_cnet = model.gate_cnet.detach().clone()
            try:
                model.gate_fnet.zero_()
                model.gate_cnet.zero_()
                zero = model(pair, context=context, test_mode=True)['final'].float()
            finally:
                model.gate_fnet.copy_(before_fnet)
                model.gate_cnet.copy_(before_cnet)
            actual = model(pair, context=context, test_mode=True)['final'].float()
    return raw, zero, actual


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError('NVIDIA GPU needed for real audit; use --help or CPU unit tests otherwise')
    if args.mode not in ('gep', 'jepa'):
        raise ValueError('Only frozen v8.2 GEP/JEPA runs are supported')
    mod = import_v82(args.v82_script)
    cfg = SimpleNamespace(
        sea_core=args.sea_core, sea_config=args.sea_config,
        raft_ckpt=args.raft_ckpt, base_ckpt=None,
        fusion=args.mode, stage1_ckpt=args.stage1_ckpt,
        stage2_ckpt=args.stage2_ckpt,
        finetune_event_encoders=False,
    )
    print('[AUDIT] instantiate strict original pretrained SEA-RAFT + event encoder', flush=True)
    model, _, load_report = mod.create_model(cfg)
    original_state = model.raft.state_dict()  # still CPU, prior to checkpoint load
    original_stage1 = model.stage1.state_dict()
    original_stage2 = model.stage2.state_dict() if model.stage2 is not None else None
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if not isinstance(ckpt, dict) or 'model' not in ckpt or 'config' not in ckpt:
        raise RuntimeError('Not a v8.2 training checkpoint')
    saved = ckpt['config']
    if saved.get('fusion') != args.mode:
        raise RuntimeError(f"Expected {args.mode} checkpoint; got {saved.get('fusion')}")
    if saved.get('freeze_raft') is not True or saved.get('finetune_event_encoders') is not False:
        raise RuntimeError('This run does not claim Frozen RAFT + Frozen event encoder')
    if saved.get('base_ckpt') not in (None, ''):
        raise RuntimeError('The checkpoint is not a RAW (original) SEA-RAFT fusion run')
    print('[AUDIT] compare frozen RAFT with initial pretrained state (all params and buffers)', flush=True)
    comparison = compare_base_tensors(original_state, ckpt['model'])
    print('[BASE EXACT MATCH]', json.dumps(comparison, ensure_ascii=False), flush=True)
    if not comparison['exact_match']:
        raise RuntimeError('RAFT WEIGHTS OR BN BUFFERS CHANGED; refusing to proceed')
    stage1_cmp = compare_module_tensors(original_stage1, ckpt['model'], 'stage1.')
    stage2_cmp = (compare_module_tensors(original_stage2, ckpt['model'], 'stage2.')
                  if original_stage2 is not None else None)
    print('[EVENT ENCODER EXACT MATCH]', json.dumps(
        {'stage1': stage1_cmp, 'stage2': stage2_cmp},ensure_ascii=False),flush=True)
    if not stage1_cmp['exact_match'] or (stage2_cmp is not None and not stage2_cmp['exact_match']):
        raise RuntimeError('Frozen Stage1/Stage2 encoder weights changed; refusing to proceed')
    model.load_state_dict(ckpt['model'], strict=True)
    device = torch.device('cuda:0')
    model.to(device).eval()
    model.raft.eval()
    gates = {'fnet': float(model.gate_fnet.item()), 'cnet': float(model.gate_cnet.item())}
    print('[TRAINED GATES]', gates, flush=True)
    results = {}
    for seq in mod.TEST_SEQS:
        dataset = mod.ImageTimeV6Pairs(args.root, seq, limit=args.samples, contexts=True)
        loader = DataLoader(dataset, batch_size=1, num_workers=args.workers,
                            pin_memory=True, shuffle=False)
        sums = {k: {'epe': 0., 'pred_mag': 0., 'cos': 0., 'cos_n': 0, 'pixel_n': 0}
                for k in ('raw', 'zero_gate', 'fusion')}
        max_zero_diff = 0.
        abs_zero_sum = 0.
        max_fusion_diff = 0.
        abs_fusion_sum = 0.
        pixels = 0
        for batch in loader:
            pair = batch['pair'].to(device, non_blocking=True)
            context = batch['context'].to(device, non_blocking=True)
            gt = batch['flow'].to(device, non_blocking=True).float()
            mask = (batch['valid'] & batch['active']).to(device, non_blocking=True)
            raw, zero, fusion = audited_predictions(model, pair, context, gt)
            z_diff = (zero - raw).abs()
            f_diff = (fusion - raw).abs()
            max_zero_diff = max(max_zero_diff, float(z_diff.max().item()))
            max_fusion_diff = max(max_fusion_diff, float(f_diff.max().item()))
            if bool(mask.any()):
                abs_zero_sum += float(z_diff.mean(dim=1)[mask].sum().item())
                abs_fusion_sum += float(f_diff.mean(dim=1)[mask].sum().item())
                pixels += int(mask.sum().item())
            for name, pred in (('raw', raw), ('zero_gate', zero), ('fusion', fusion)):
                e, p, c, cn, nn = flow_statistics(pred, gt, mask)
                s = sums[name]
                s['epe'] += e; s['pred_mag'] += p; s['cos'] += c
                s['cos_n'] += cn; s['pixel_n'] += nn
        metrics = {}
        for name, stat in sums.items():
            n = stat['pixel_n']; cn = stat['cos_n']
            metrics[name] = {
                'EPE': stat['epe'] / max(n, 1),
                'pred_mag': stat['pred_mag'] / max(n, 1),
                'cosine': stat['cos'] / max(cn, 1),
                'event_valid_pixels': n,
            }
        result = {
            'samples': len(dataset), 'metrics': metrics,
            'max_abs_zero_vs_raw': max_zero_diff,
            'mean_abs_zero_vs_raw_eventpixels': abs_zero_sum / max(pixels, 1),
            'max_abs_fusion_vs_raw': max_fusion_diff,
            'mean_abs_fusion_vs_raw_eventpixels': abs_fusion_sum / max(pixels, 1),
            'delta_EPE_fusion_minus_raw': metrics['fusion']['EPE'] - metrics['raw']['EPE'],
        }
        results[seq] = result
        print('[SEQ]', seq, json.dumps(result, ensure_ascii=False), flush=True)
        if max_zero_diff > args.tolerance:
            raise RuntimeError(f'Zero-gate RAFT parity FAILED on {seq}: max diff={max_zero_diff:.8g}')
    report = {
        'model': args.mode, 'fusion_checkpoint': str(args.checkpoint.resolve()),
        'raw_checkpoint': str(args.raft_ckpt.resolve()),
        'strict_pretrain': load_report,
        'base_exact': comparison, 'stage1_exact': stage1_cmp,
        'stage2_exact': stage2_cmp, 'trained_gate_values': gates,
        'same_sample_parity_tolerance': args.tolerance,
        'results': results,
        'interpretation': 'audit only, no training; MVSEC v6 candidate protocol',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print('[AUDIT PASS] RAFT and event encoders match original weights; zero gates recover raw RAFT',flush=True)
    print('[WROTE]', args.output, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('gep','jepa'), required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--root',type=Path,default=Path('datasets/MVSEC'))
    p.add_argument('--v82-script',type=Path,default=Path('scripts/sea_raft_fullbackbone_v8_2.py'))
    p.add_argument('--raft-ckpt',type=Path,default=Path('pretrained/searaft_M/model.safetensors'))
    p.add_argument('--sea-core',type=Path,default=Path('third_party/ScaleEvent/sea_raft/core'))
    p.add_argument('--sea-config',type=Path,default=Path('third_party/ScaleEvent/sea_raft/config/eval/spring-M.json'))
    p.add_argument('--stage1-ckpt',type=Path,default=Path('checkpoints/stage1/gep_stage1_small.pt'))
    p.add_argument('--stage2-ckpt',type=Path,default=Path('runs/event_jepa_vits14_formal_seed0/step_00030000.pt'))
    p.add_argument('--samples',type=int,default=8,help='Uniformly subsampled frames per indoor sequence')
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--tolerance',type=float,default=1e-5)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    if args.samples<1: p.error('--samples must be positive')
    run(args)


if __name__ == '__main__':
    main()
