from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


def make_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def link_dir(src: Path, dst: Path):
    if not src.is_dir():
        raise FileNotFoundError(src)

    make_parent(dst)

    if dst.exists() or dst.is_symlink():
        raise FileExistsError(dst)

    dst.symlink_to(
        src.resolve(),
        target_is_directory=True,
    )


def link_file(src: Path, dst: Path):
    if not src.is_file():
        raise FileNotFoundError(src)

    make_parent(dst)

    if dst.exists() or dst.is_symlink():
        raise FileExistsError(dst)

    dst.symlink_to(src.resolve())


def build_dsec(
    dsec_root: Path,
    out_root: Path,
):
    train_root = dsec_root / "train_images"

    sequences = sorted(
        p for p in train_root.iterdir()
        if p.is_dir()
    )

    if len(sequences) != 41:
        raise RuntimeError(
            f"Expected 41 DSEC train sequences, "
            f"found {len(sequences)}"
        )

    total_tokens = 0

    for seq in sequences:
        src = (
            seq
            / "images"
            / "left"
            / "eventToken_vits14"
        )

        if not src.is_dir():
            raise FileNotFoundError(src)

        n = len(list(src.glob("*.pt")))

        if n < 8:
            raise RuntimeError(
                f"{seq.name}: only {n} tokens"
            )

        dst = (
            out_root
            / "train_images"
            / f"dsec__{seq.name}"
            / "images"
            / "left"
            / "eventToken"
        )

        link_dir(src, dst)

        total_tokens += n

        print(
            f"[DSEC] {seq.name:24s} "
            f"{n:6d} frames"
        )

    return {
        "sequences": len(sequences),
        "frames": total_tokens,
    }


def read_mvsec_rows(
    mvsec_root: Path,
):
    index_path = mvsec_root / "index.csv"

    with index_path.open(
        "r",
        newline="",
    ) as f:
        rows = list(csv.DictReader(f))

    if len(rows) < 16:
        raise RuntimeError(
            "MVSEC sequence is too short"
        )

    # Convert true MVSEC depth timestamps
    # from seconds -> integer microseconds.
    previous = None

    for row in rows:
        ts_s = float(
            row["depth_timestamp"]
        )

        ts_us = int(round(ts_s * 1e6))

        if previous is not None:
            if ts_us <= previous:
                raise RuntimeError(
                    "MVSEC timestamps are not "
                    "strictly increasing after "
                    "microsecond conversion"
                )

        row["_timestamp_us"] = ts_us
        previous = ts_us

    return rows


def build_mvsec_part(
    rows,
    mvsec_root: Path,
    out_root: Path,
    split: str,
    sequence_name: str,
):
    token_root = (
        mvsec_root
        / "eventToken_vits14"
    )

    dst_root = (
        out_root
        / split
        / sequence_name
        / "images"
        / "left"
        / "eventToken"
    )

    dst_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    for row in rows:
        frame_id = int(
            row["frame_id"]
        )

        source = (
            token_root
            / f"{frame_id:06d}.pt"
        )

        timestamp_us = int(
            row["_timestamp_us"]
        )

        destination = (
            dst_root
            / f"{timestamp_us}.pt"
        )

        link_file(
            source,
            destination,
        )

    print(
        f"[MVSEC] {sequence_name}: "
        f"{len(rows)} frames"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dsec-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--mvsec-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--mvsec-train-ratio",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--rebuild",
        action="store_true",
    )

    args = parser.parse_args()

    dsec_root = args.dsec_root.resolve()
    mvsec_root = args.mvsec_root.resolve()
    out_root = args.output_root.resolve()

    if not 0.5 < args.mvsec_train_ratio < 1.0:
        raise ValueError(
            "MVSEC train ratio must be "
            "between 0.5 and 1.0"
        )

    if out_root.exists():
        if not args.rebuild:
            raise FileExistsError(
                f"{out_root} already exists. "
                f"Use --rebuild."
            )

        shutil.rmtree(out_root)

    out_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("BUILD DSEC VIEW")
    print("=" * 80)

    dsec_info = build_dsec(
        dsec_root,
        out_root,
    )

    print()
    print("=" * 80)
    print("BUILD MVSEC VIEW")
    print("=" * 80)

    rows = read_mvsec_rows(
        mvsec_root
    )

    cut = int(
        len(rows)
        * args.mvsec_train_ratio
    )

    train_rows = rows[:cut]
    val_rows = rows[cut:]

    if len(train_rows) < 8:
        raise RuntimeError(
            "MVSEC train split too short"
        )

    if len(val_rows) < 8:
        raise RuntimeError(
            "MVSEC val split too short"
        )

    build_mvsec_part(
        train_rows,
        mvsec_root,
        out_root,
        "train_images",
        "mvsec__outdoor_day2_train",
    )

    build_mvsec_part(
        val_rows,
        mvsec_root,
        out_root,
        "test_images",
        "mvsec__outdoor_day2_val",
    )

    metadata = {
        "backbone": "GEP ViT-S/14",
        "token_shape": [256, 384],
        "token_dtype": "float32",
        "context_frames": 4,
        "horizons": [1, 2, 4],
        "timestamp_unit": "microseconds",
        "dsec_train_sequences":
            dsec_info["sequences"],
        "dsec_train_frames":
            dsec_info["frames"],
        "mvsec_sequence":
            "outdoor_day2",
        "mvsec_total_frames":
            len(rows),
        "mvsec_train_frames":
            len(train_rows),
        "mvsec_val_frames":
            len(val_rows),
        "mvsec_train_ratio":
            args.mvsec_train_ratio,
    }

    with (
        out_root / "protocol.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print()
    print("=" * 80)
    print("RESULT: PASS")
    print(json.dumps(
        metadata,
        indent=2,
    ))
    print("=" * 80)


if __name__ == "__main__":
    main()
