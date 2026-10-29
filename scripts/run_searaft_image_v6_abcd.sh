#!/usr/bin/env bash
# Four independent image-timestamp SEA-RAFT FT experiments; no test-set selection.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/dinov2:$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
ROOT="$PWD/datasets/MVSEC"
STAGE1="$PWD/checkpoints/stage1/gep_stage1_small.pt"
JEPA="$PWD/runs/event_jepa_vits14_formal_seed0/step_00030000.pt"
mkdir -p logs/searaft_image_v6
DECODER_INIT_CKPT="${DECODER_INIT_CKPT:-$PWD/pretrained/searaft_M/model.safetensors}"
[[ -s "$DECODER_INIT_CKPT" ]] || { echo "Missing pretrained SEA-RAFT decoder: $DECODER_INIT_CKPT"; exit 1; }
for f in "$STAGE1" "$JEPA"; do test -f "$f" || { echo "Missing $f"; exit 1; };done
pids=(); names=()
for task in "0 frozen" "1 ft10" "2 full" "3 stage2-only"; do
  read -r gpu mode <<< "$task"
  logfile="logs/searaft_image_v6/${mode}_seed0.log"
  echo "GPU $gpu -> $mode, log $logfile"
  CUDA_VISIBLE_DEVICES="$gpu" python -u src/mvsec_flow_probe/train_searaft_image_v6.py \
    --root "$ROOT" --stage1-ckpt "$STAGE1" --stage2-ckpt "$JEPA" \
    --decoder-init-ckpt "$DECODER_INIT_CKPT" \
    --protocol "$mode" --epochs 20 --batch-size 32 --grad-accum 1 \
    --seed 0 --skip-test --output-dir "runs/searaft_image_v6_${mode}_seed0" \
    > "$logfile" 2>&1 &
  pids+=("$!"); names+=("$mode")
done
failed=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "[OK] ${names[$i]}";
  else echo "[FAIL] ${names[$i]} -- examine logs";failed=1;fi
done
exit "$failed"
