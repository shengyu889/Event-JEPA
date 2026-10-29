# Event-JEPA / GEP Stage1-only + Full FT / SEA-RAFT v2

This **incremental patch** adds a fifth **main experiment** to the completed A/B/C/D SEA-RAFT suite. It does not retrain Stage1/Stage2 and does not modify the old A/B/C/D checkpoint files. All experiment labels and evaluation settings remain unchanged.

## What this patch changes

Files overwritten from the original `Event-JEPA_SEARAFT_ABCD_v1.zip`:

- `src/mvsec_flow_probe/searaft_model.py`: adds a `stage1-only` protocol. GEP Stage1 encodes the two observed event intervals ending at the start/end of optical flow; **Stage2 is entirely bypassed**, not just frozen. The shared `fmap_adapter` and authentic `SEAEventDecoder` are identical to A/B/C/D. Stage1 gradients are enabled. Only two Stage1 forwards are executed, saving compute.
- `src/mvsec_flow_probe/train_searaft_formal.py`: adds `stage1-only` in argument choices, exactly the same 20-epoch training and loss/optimizer/scheduler/augmentation setup as the `full` protocol, and records representation metadata. Stage2 checkpoint must be loadable for provenance/identical decoder RNG init, but its parameters are untouched and excluded from the optimizer.

New files:

- `scripts/run_searaft_gep_stage1_full.sh` — run E on one idle A100.
- `scripts/eval_searaft_gep_stage1_full.sh` — evaluate E after training.
- `scripts/audit_mvsec_searaft.py` — **read-only** diagnostic for GT dt, sample counts, event-mask fractions, and a zero-flow EPE/Outlier benchmark; also prints train loss curves and module change audits.
- `tests/test_gep_stage1only_contract.py` — verifies Stage2 never executes and Stage1/adapter/head receive gradients using a lightweight mock decoder.

Note: GEP-only receives two individual event representations, whereas JEPA Stage2 contextualizes four frames. This measures benefit of using the JEPA representation as a module in this optical-flow architecture, **not** an architecture-/parameter-matched causal attribution solely to pretraining.

## Deploy

```bash
cd /mnt/data2/gsy/Event-JEPA
conda activate eventjepa
unzip -o Event-JEPA_GEP_Stage1Only_SEARAFT_FullFT_v2.zip -d .
export PYTHONPATH="$PWD/dinov2:$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q tests/test_gep_stage1only_contract.py tests/test_searaft_contract.py
```

Do not overwrite previous results or delete old runs.

## Audit existing A/B/C/D and MVSEC protocol immediately (no GPU)

```bash
python scripts/audit_mvsec_searaft.py \
  --root datasets/MVSEC \
  --runs runs \
  --sample-count 128
```

Outputs `results/mvsec_searaft_protocol_audit.json`.

- `median_dt_ms`: inspect actual GT interval length. Around 50ms implies ~20Hz.
- `ZERO_FLOW_EPE`: mean GT displacement magnitude on the **same GT-valid & event-active mask** as your evaluation, using zero predicted flow. If the trained EPE exceeds the zero-flow benchmark, the model has not learned a useful predictor under the protocol.
- `event_active_fraction_among_gt_valid`: sanity check on sparse evaluation pixels.
- `train_history`: inspect train loss trend; a decreasing loss does not prove held-out generalization.
- `parameter_audit`: verify which parameters really changed.

To compute **exact** (rather than sampled) zero-flow metrics on all available frames:

```bash
python scripts/audit_mvsec_searaft.py --root datasets/MVSEC --sample-count 0
```

This is slower and requires the preprocessed `event_active` masks plus `.npy_cache` GT arrays. `--sample-count 128` computes only an indicative estimate; do not quote it as a paper result.

## Run fifth main experiment E on idle GPU 0

```bash
GPU=0 bash scripts/run_searaft_gep_stage1_full.sh
```

To keep the process alive after SSH disconnect use tmux:

```bash
tmux new -s gep_searaft_full
GPU=0 bash scripts/run_searaft_gep_stage1_full.sh
```

On another shell:

```bash
tail -f logs/searaft/stage1-only_seed0.log
```

Check expected protocol change audit in `runs/searaft_stage1-only_seed0/parameter_audit.json`:

```json
{"changed": {"stage1": true, "stage2": false, "fmap_adapter": true, "decoder": true}}
```

**Do not call this success until real A100 training produces checkpoints and the audit reports the expected changes.**

## Evaluate once after E's last epoch

```bash
GPU=0 bash scripts/eval_searaft_gep_stage1_full.sh
```

It writes `runs/searaft_stage1-only_seed0/metrics_final.json` with EPE, Outlier%, 1PE/2PE/3PE for indoor_flying1/2/3. A/B/C/D metrics remain untouched.

## Known serious comparability limitations (do not conceal in a paper)

- Current 20Hz-ish training sample count `12,193` differs from ScaleEvent's `26,677`; test total `4,952` differs from ScaleEvent's `7,775`. First check the actual `median_dt_ms` and whether ScaleEvent uses 45Hz dt=1. The count mismatch alone does not prove ScaleEvent's exact dt setting.
- The cached `eventRGB_native/*.png` **losslessly stores an already-aggregated, percentile-clipped 8-bit image**, which is **not a lossless representation of raw events**; fine temporal information is not recoverable. The previous v1 README's wording `lossless raw native-size event RGB` was misleading.
- Native MVSEC `346×260` event images are non-isotropically resized to `224×224`; Stage1's `16×16` patch features are bilinearly upsampled to a `28×28` SEA-RAFT grid. Fine correspondence detail is lost; SEA-RAFT ResNet-FPN (and its pretraining) is not fully restored.
- A/B/C/D and E share the same adapted decoder and split, enabling internal comparison, but are **not** a strict ScaleEvent apples-to-apples benchmark.
- GT-valid mask is currently `finite AND nonzero` (outdoor additionally hood-masked), not yet demonstrated identical to ScaleEvent's original validity mask; zero-flow pixels may be excluded erroneously.
- Do **not** reinterpret ScaleEvent's 0.29/0.38/0.40 EPE as directly comparable until GT frame rate, dt parameter, valid mask and evaluation samples are synchronized.

## Validation performed here

- Python source syntax compiled.
- `tests/test_gep_stage1only_contract.py`: 2 tests passed on CPU with mock SEA-RAFT decoder (verifies Stage2 bypass and Stage1 gradient propagation).
- Audit `ZERO_FLOW_EPE` tested against a synthetic HDF5-independent GT+event-mask example.
- Did **not** run real pretrained checkpoints, A100, or the real adapted SEA-RAFT module in this local container; server integration must be checked via the training run.
