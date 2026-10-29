#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/dinov2:$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
for p in \
  third_party/ScaleEvent/sea_raft/core/raft.py \
  third_party/ScaleEvent/sea_raft/core/extractor.py \
  third_party/ScaleEvent/sea_raft/config/eval/spring-M.json \
  pretrained/searaft_M/model.safetensors \
  datasets/MVSEC/processed_image_time_v6/outdoor_day2/manifest.csv \
  datasets/MVSEC/processed_image_time_v6/indoor_flying1/manifest.csv; do
  if [[ ! -s "$p" ]];then printf 'MISSING %s\n' "$p";exit 1;fi
  printf 'OK %s\n' "$p"
done
python -m pytest -q tests/test_sea_raft_fullbackbone_v8.py
mkdir -p logs
printf 'READY: run the 64-sample eval-pretrained command from README\n'
