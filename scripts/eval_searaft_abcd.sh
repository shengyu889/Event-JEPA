#!/usr/bin/env bash
# Run only AFTER indoor_flying1/2/3 full HDF5 repair + eventRGB caching.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/dinov2:$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
ROOT="$PWD/datasets/MVSEC"
STAGE1="$PWD/checkpoints/stage1/gep_stage1_small.pt"
STAGE2="$PWD/runs/event_jepa_vits14_formal_seed0/step_00030000.pt"
CORE="$PWD/third_party/ScaleEvent/sea_raft/core"
mkdir -p logs/searaft
pids=()
for entry in '0 frozen' '1 ft10' '2 full' '3 stage2-only'; do
  read -r GPU MODE <<<"$entry"
  DIR="runs/searaft_${MODE}_seed0"
  [[ -s "$DIR/last.pt" ]] || { echo "[MISSING] $DIR/last.pt" >&2; exit 1; }
  CUDA_VISIBLE_DEVICES="$GPU" python -u src/mvsec_flow_probe/train_searaft_formal.py \
    --root "$ROOT" --stage1-ckpt "$STAGE1" --stage2-ckpt "$STAGE2" \
    --sea-core "$CORE" --protocol "$MODE" --output-dir "$DIR" \
    --epochs 20 --batch-size 16 --grad-accum 1 --seed 0 \
    --eval-only --resume "$DIR/last.pt" \
    > "logs/searaft/${MODE}_final_eval.log" 2>&1 &
  pids+=("$!")
done
failed=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || failed=1
done
exit "$failed"
