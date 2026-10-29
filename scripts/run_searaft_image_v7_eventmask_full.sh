#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/dinov2:$PWD/src:$PWD/scripts:$PWD${PYTHONPATH:+:$PYTHONPATH}"
GPU="${GPU:-0}"
DECODER_INIT_CKPT="${DECODER_INIT_CKPT:-$PWD/pretrained/searaft_M/model.safetensors}"
[[ -s "$DECODER_INIT_CKPT" ]] || { echo "Missing SEA-RAFT checkpoint $DECODER_INIT_CKPT"; exit 1; }
mkdir -p logs/searaft_image_v7
CUDA_VISIBLE_DEVICES="$GPU" python -u src/mvsec_flow_probe/train_searaft_image_v7_eventmask.py \
  --root "$PWD/datasets/MVSEC" \
  --stage1-ckpt "$PWD/checkpoints/stage1/gep_stage1_small.pt" \
  --stage2-ckpt "$PWD/runs/event_jepa_vits14_formal_seed0/step_00030000.pt" \
  --decoder-init-ckpt "$DECODER_INIT_CKPT" \
  --protocol full --epochs 20 --batch-size 32 --grad-accum 1 --seed 0 \
  --skip-test --output-dir runs/searaft_image_v7_eventmask_full_seed0 \
  2>&1 | tee logs/searaft_image_v7/eventmask_full_seed0.log
