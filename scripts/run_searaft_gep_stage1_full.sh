#!/usr/bin/env bash
# Main-experiment E: GEP Stage1-only + fully supervised SEA-RAFT, use ONE idle GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/dinov2:$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
ROOT="$PWD/datasets/MVSEC"
S1="$PWD/checkpoints/stage1/gep_stage1_small.pt"
S2="$PWD/runs/event_jepa_vits14_formal_seed0/step_00030000.pt"
CORE="$PWD/third_party/ScaleEvent/sea_raft/core"
for x in "$S1" "$S2" "$CORE/update.py" "$CORE/loss.py" "$ROOT/processed_observed_v1/outdoor_day2/manifest.csv"; do
  [[ -s "$x" ]] || { echo "MISSING $x" >&2; exit 1; }
done
mkdir -p logs/searaft
CUDA_VISIBLE_DEVICES="${GPU:-0}" python -u src/mvsec_flow_probe/train_searaft_formal.py \
  --root "$ROOT" --stage1-ckpt "$S1" --stage2-ckpt "$S2" --sea-core "$CORE" \
  --protocol stage1-only --output-dir runs/searaft_stage1-only_seed0 \
  --epochs 20 --batch-size 1 --grad-accum 8 --seed 0 --skip-test \
  2>&1 | tee logs/searaft/stage1-only_seed0.log
