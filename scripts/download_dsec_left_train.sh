#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/tom/event-jepa/datasets/DSEC"
CACHE="/home/tom/event-jepa/datasets/DSEC_downloads"
SEQ_FILE="/home/tom/event-jepa/gep-event-jepa/scripts/dsec_train_sequences.txt"

BASE="https://download.ifi.uzh.ch/rpg/DSEC/train"
CAL_URL="https://download.ifi.uzh.ch/rpg/DSEC/train_coarse/train_calibration.zip"

mkdir -p "$ROOT" "$CACHE"

echo "============================================================"
echo "DSEC left-only downloader"
echo "ROOT : $ROOT"
echo "CACHE: $CACHE"
echo "============================================================"

# ------------------------------------------------------------
# 1. Download all training calibrations once.
# ------------------------------------------------------------
CAL_ZIP="$CACHE/train_calibration.zip"
CAL_TMP="$CACHE/_train_calibration_extract"

if [ ! -f "$CAL_ZIP" ]; then
    echo "[calib] downloading train_calibration.zip"
    wget -c "$CAL_URL" -O "$CAL_ZIP"
else
    echo "[calib] archive already exists"
fi

if [ ! -d "$CAL_TMP" ]; then
    echo "[calib] extracting..."
    mkdir -p "$CAL_TMP"
    unzip -q "$CAL_ZIP" -d "$CAL_TMP"
fi

# ------------------------------------------------------------
# 2. Each training sequence.
# ------------------------------------------------------------
while read -r SEQ; do
    [ -z "$SEQ" ] && continue

    echo
    echo "============================================================"
    echo "Sequence: $SEQ"
    echo "============================================================"

    SEQ_CACHE="$CACHE/$SEQ"

    EVENT_DST="$ROOT/train_events/$SEQ/events/left"
    IMAGE_DST="$ROOT/train_images/$SEQ/images/left/rectified"
    TS_DST="$ROOT/train_images/$SEQ/images/timestamps.txt"
    CAL_DST="$ROOT/train_calibration/$SEQ/calibration"

    mkdir -p \
        "$SEQ_CACHE" \
        "$EVENT_DST" \
        "$IMAGE_DST" \
        "$(dirname "$TS_DST")" \
        "$CAL_DST"

    # --------------------------------------------------------
    # Events left
    # --------------------------------------------------------
    if [ ! -s "$EVENT_DST/events.h5" ]; then
        ARCHIVE="${SEQ}_events_left.zip"
        URL="$BASE/$SEQ/$ARCHIVE"

        echo "[events] downloading $ARCHIVE"
        (
            cd "$SEQ_CACHE"
            wget -c "$URL"
        )

        echo "[events] extracting"
        unzip -q -o "$SEQ_CACHE/$ARCHIVE" -d "$EVENT_DST"

        test -s "$EVENT_DST/events.h5"
        rm -f "$SEQ_CACHE/$ARCHIVE"
    else
        echo "[events] already prepared"
    fi

    # --------------------------------------------------------
    # Rectified left RGB images
    # --------------------------------------------------------
    if ! find "$IMAGE_DST" -maxdepth 1 -name '*.png' -print -quit | grep -q .; then
        ARCHIVE="${SEQ}_images_rectified_left.zip"
        URL="$BASE/$SEQ/$ARCHIVE"

        echo "[images] downloading $ARCHIVE"
        (
            cd "$SEQ_CACHE"
            wget -c "$URL"
        )

        echo "[images] extracting"
        unzip -q -o "$SEQ_CACHE/$ARCHIVE" -d "$IMAGE_DST"

        rm -f "$SEQ_CACHE/$ARCHIVE"
    else
        echo "[images] already prepared"
    fi

    # --------------------------------------------------------
    # Image timestamps
    # --------------------------------------------------------
    if [ ! -s "$TS_DST" ]; then
        SRC_TS="$SEQ_CACHE/${SEQ}_image_timestamps.txt"

        echo "[timestamps] downloading"
        (
            cd "$SEQ_CACHE"
            wget -c "$BASE/$SEQ/${SEQ}_image_timestamps.txt"
        )

        cp "$SRC_TS" "$TS_DST"
    else
        echo "[timestamps] already prepared"
    fi

    # --------------------------------------------------------
    # Calibration
    # Robust against extra archive directory levels.
    # --------------------------------------------------------
    if [ ! -s "$CAL_DST/cam_to_cam.yaml" ]; then
        CAM_FILE="$(
            find "$CAL_TMP" \
                -type f \
                -path "*${SEQ}*" \
                -name 'cam_to_cam.yaml' \
                -print -quit
        )"

        if [ -z "$CAM_FILE" ]; then
            echo "ERROR: calibration not found for $SEQ"
            exit 1
        fi

        cp "$CAM_FILE" "$CAL_DST/cam_to_cam.yaml"

        LIDAR_FILE="$(dirname "$CAM_FILE")/cam_to_lidar.yaml"
        if [ -f "$LIDAR_FILE" ]; then
            cp "$LIDAR_FILE" "$CAL_DST/cam_to_lidar.yaml"
        fi
    else
        echo "[calib] already prepared"
    fi

    # --------------------------------------------------------
    # Integrity check
    # --------------------------------------------------------
    N_IMAGES=$(
        find "$IMAGE_DST" -maxdepth 1 -name '*.png' | wc -l
    )
    N_TIMESTAMPS=$(
        wc -l < "$TS_DST"
    )

    echo "[check] images     = $N_IMAGES"
    echo "[check] timestamps = $N_TIMESTAMPS"

    if [ "$N_IMAGES" -ne "$N_TIMESTAMPS" ]; then
        echo "ERROR: image/timestamp mismatch for $SEQ"
        exit 1
    fi

    echo "[OK] $SEQ"

done < "$SEQ_FILE"

echo
echo "============================================================"
echo "ALL AVAILABLE DSEC TRAIN SEQUENCES PREPARED"
echo "============================================================"
