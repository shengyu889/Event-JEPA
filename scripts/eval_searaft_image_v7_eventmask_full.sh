#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/dinov2:$PWD/src:$PWD/scripts:$PWD${PYTHONPATH:+:$PYTHONPATH}"
GPU="${GPU:-0}"
RUN="$PWD/runs/searaft_image_v7_eventmask_full_seed0"
[[ -s "$RUN/last.pt" ]] || { echo "Missing: $RUN/last.pt"; exit 1; }
CUDA_VISIBLE_DEVICES="$GPU" python -u src/mvsec_flow_probe/train_searaft_image_v7_eventmask.py \
  --root "$PWD/datasets/MVSEC" \
  --stage1-ckpt "$PWD/checkpoints/stage1/gep_stage1_small.pt" \
  --stage2-ckpt "$PWD/runs/event_jepa_vits14_formal_seed0/step_00030000.pt" \
  --protocol full --seed 0 --output-dir "$RUN" \
  --eval-only --resume "$RUN/last.pt" \
  2>&1 | tee logs/searaft_image_v7/eventmask_full_eval.log
