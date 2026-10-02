#!/usr/bin/env bash
set -u
set -o pipefail

REPO=/home/tom/event-jepa/gep-event-jepa
ROOT=/home/tom/event-jepa/datasets/DSEC
LIST="$REPO/scripts/dsec_train_sequences.txt"
CKPT=/home/tom/event-jepa/checkpoints/gep_stage1_small.pt
LOGDIR="$REPO/logs/dsec_stage1"

mkdir -p "$LOGDIR"

cd "$REPO" || exit 1

total=$(grep -cve '^[[:space:]]*$' "$LIST")
idx=0

while read -r seq; do
    [ -z "$seq" ] && continue

    idx=$((idx + 1))

    IMGDIR="$ROOT/train_images/$seq/images/left/rectified"
    TOKDIR="$ROOT/train_images/$seq/images/left/eventToken"

    expected=$(find "$IMGDIR" -maxdepth 1 -type f -name '*.png' | wc -l)
    existing=0

    if [ -d "$TOKDIR" ]; then
        existing=$(find "$TOKDIR" -maxdepth 1 -type f -name '*.pt' | wc -l)
    fi

    echo
    echo "============================================================"
    echo "[$idx/$total] $seq"
    echo "images=$expected existing_tokens=$existing"
    echo "============================================================"

    # 已经完整生成，例如之前的 zurich_city_00_a
    if [ "$expected" -gt 0 ] && [ "$existing" -eq "$expected" ]; then
        echo "[SKIP] already complete"
        continue
    fi

    # 如果之前中断过，清掉 partial tokens，重新生成该序列
    rm -rf "$TOKDIR"

    {
        /usr/bin/time -v \
        env PYTHONPATH=src \
        python -u src/pre_dse_stream.py \
          --root "$ROOT" \
          --split train \
          --sequence "$seq" \
          --event-encoder-ckpt "$CKPT" \
          --prepare-pairs \
          --event-only \
          --device cuda:0 \
          --workers 1
    } 2>&1 | tee "$LOGDIR/$seq.log"

    status=${PIPESTATUS[0]}

    if [ "$status" -ne 0 ]; then
        echo "[FAILED] $seq"
        echo "$seq" >> "$LOGDIR/FAILED.txt"
        exit "$status"
    fi

    produced=$(find "$TOKDIR" -maxdepth 1 -type f -name '*.pt' | wc -l)

    echo "[CHECK] images=$expected tokens=$produced"

    if [ "$produced" -ne "$expected" ]; then
        echo "[FAILED] token/image count mismatch: $seq"
        echo "$seq" >> "$LOGDIR/FAILED.txt"
        exit 1
    fi

    echo "[OK] $seq"

    # Event-JEPA Stage-2 only needs eventToken.
    # Remove derived temporary pair images after successful tokenization.
    rm -rf \
      "$ROOT/train_images/$seq/images/left/eventImage" \
      "$ROOT/train_images/$seq/images/left/warpped" \
      "$ROOT/train_images/$seq/images/left/vis"

    echo "[CLEAN] removed temporary eventImage/warpped/vis"
done < "$LIST"

echo
echo "============================================================"
echo "ALL DSEC TRAIN EVENT TOKENS FINISHED"
echo "============================================================"
