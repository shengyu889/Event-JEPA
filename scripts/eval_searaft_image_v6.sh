#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/dinov2:$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
ROOT="$PWD/datasets/MVSEC"
STAGE1="$PWD/checkpoints/stage1/gep_stage1_small.pt"
JEPA="$PWD/runs/event_jepa_vits14_formal_seed0/step_00030000.pt"
for mode in frozen ft10 full stage2-only stage1-only; do
  run="runs/searaft_image_v6_${mode}_seed0"
  [[ -s "$run/last.pt" ]] || { echo "MISSING: $run/last.pt (skip)"; continue; }
  echo "Evaluating $mode"
  CUDA_VISIBLE_DEVICES="${GPU:-0}" python -u src/mvsec_flow_probe/train_searaft_image_v6.py \
    --root "$ROOT" --stage1-ckpt "$STAGE1" --stage2-ckpt "$JEPA" \
    --protocol "$mode" --seed 0 --output-dir "$run" --eval-only \
    --resume "$run/last.pt" \
    2>&1 | tee "logs_searaft_image_v6_${mode}_eval.log"
done
