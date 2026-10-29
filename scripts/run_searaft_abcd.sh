#!/usr/bin/env bash
# Four independent single-GPU formal experiments. Each sees only one A100.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/dinov2:$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
ROOT="$PWD/datasets/MVSEC"
STAGE1="$PWD/checkpoints/stage1/gep_stage1_small.pt"
STAGE2="$PWD/runs/event_jepa_vits14_formal_seed0/step_00030000.pt"
CORE="$PWD/third_party/ScaleEvent/sea_raft/core"
for p in "$STAGE1" "$STAGE2" "$CORE/corr.py" "$CORE/update.py" "$CORE/loss.py" "$ROOT/processed_observed_v1/outdoor_day2/manifest.csv"; do
  [[ -e "$p" ]] || { echo "[MISSING] $p" >&2; exit 1; }
done
mkdir -p logs/searaft
pids=()
for entry in '0 frozen' '1 ft10' '2 full' '3 stage2-only'; do
  read -r GPU PROTOCOL <<<"$entry"
  echo "[LAUNCH] GPU=$GPU protocol=$PROTOCOL"
  CUDA_VISIBLE_DEVICES="$GPU" python -u src/mvsec_flow_probe/train_searaft_formal.py \
    --root "$ROOT" --stage1-ckpt "$STAGE1" --stage2-ckpt "$STAGE2" \
    --sea-core "$CORE" --protocol "$PROTOCOL" \
    --output-dir "runs/searaft_${PROTOCOL}_seed0" \
    --epochs 20 --batch-size 16 --grad-accum 1 --seed 0 --skip-test \
    >"logs/searaft/${PROTOCOL}_seed0.log" 2>&1 &
  pids+=("$!")
done
failed=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[DONE] process $i pid ${pids[$i]}"
  else
    echo "[FAILED] process $i pid ${pids[$i]}" >&2
    failed=1
  fi
done
exit "$failed"
