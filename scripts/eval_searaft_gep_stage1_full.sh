#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/dinov2:$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
ROOT="$PWD/datasets/MVSEC"
S1="$PWD/checkpoints/stage1/gep_stage1_small.pt"
S2="$PWD/runs/event_jepa_vits14_formal_seed0/step_00030000.pt"
CORE="$PWD/third_party/ScaleEvent/sea_raft/core"
DIR="runs/searaft_stage1-only_seed0"
[[ -s "$DIR/last.pt" ]] || { echo "MISSING: $DIR/last.pt" >&2; exit 1; }
mkdir -p logs/searaft
CUDA_VISIBLE_DEVICES="${GPU:-0}" python -u src/mvsec_flow_probe/train_searaft_formal.py \
  --root "$ROOT" --stage1-ckpt "$S1" --stage2-ckpt "$S2" --sea-core "$CORE" \
  --protocol stage1-only --output-dir "$DIR" \
  --epochs 20 --batch-size 1 --grad-accum 8 --seed 0 \
  --eval-only --resume "$DIR/last.pt" \
  2>&1 | tee logs/searaft/stage1-only_final_eval.log
