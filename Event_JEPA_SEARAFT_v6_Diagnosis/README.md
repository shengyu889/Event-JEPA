# Event-JEPA SEA-RAFT v6: Zero-flow and motion-collapse audit

This is a **read-only diagnostic**, not an experiment configuration change and not a retrain.

Tested via Python compilation and synthetic array checks. **Not** tested with a real GPU, full MVSEC dataset, or saved server checkpoint.

Copy `audit_flow_collapse_v6.py` into your project `scripts/` directory and set `PYTHONPATH` as in previous Event-JEPA v6 instructions.

## 1. Exact full-test zero-flow baseline (CPU, no GPU needed)

```bash
python scripts/audit_flow_collapse_v6.py --root datasets/MVSEC --samples 0 \
  --output results/zero_v6_exact.json
```

## 2. Identical-sample comparison with a real v6 Full FT checkpoint (A100)

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/audit_flow_collapse_v6.py \
  --root datasets/MVSEC --model full --samples 64 \
  --output results/full_v6_motion_64.json
```

For comparison to GEP Stage1-only:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/audit_flow_collapse_v6.py \
  --root datasets/MVSEC --model stage1-only --samples 64 \
  --output results/gep_v6_motion_64.json
```

The two model runs choose **exactly the same 64 frames per sequence**, compare GT&event masks identical to v6 metrics and print model/zero EPE and model magnitude. Move to --samples 0 only if necessary.

## How to interpret

- `relative_epe_gain_over_zero_pct <= 0`: learned prediction is no better than always outputting zero on the same pixels.
- `pred_magnitude_mean ~0`, `pred_mag_lt_0_25_pct` high: possible low-motion output collapse.
- `pred_magnitude_mean` nonzero but EPE ~ zero: model predicts nonzero flow but often in the wrong direction or at the wrong scale. Inspect `mean_cosine_pred_gt_on_nonzero`.
- Remember SEA-RAFT's mixture-Laplace NLL can decrease from learning uncertainty (`info`) without lower final EPE.
- `joint_eval_fraction` far below `gt_valid_fraction`: evaluation differs strongly from the loss mask. This isn't automatically a bug, but is a concrete objective mismatch to investigate.
- 20 epochs training losses decreasing through the end do **not** demonstrate convergence, and EPE requires same-frame validation rather than only loss.

Do not claim strict ScaleEvent Table 6 replication from v6; its exact event representation, decoder adapter and sample filtering are not proven equivalent.
