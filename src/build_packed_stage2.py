from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap
from tqdm import tqdm


N_TOKENS = 256
EMBED_DIM = 384
WINDOW = 8


@dataclass
class SequenceInfo:
    name: str
    paths: list[Path]
    timestamps: np.ndarray
    frame_offset: int = 0
    window_offset: int = 0


def timestamp(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError as exc:
        raise ValueError(
            f"non-numeric token filename: {path}"
        ) from exc


def scan_split(
    root: Path,
    image_split: str,
) -> list[SequenceInfo]:

    split_root = root / image_split

    if not split_root.is_dir():
        raise FileNotFoundError(split_root)

    sequences = []

    for seq in sorted(
        p for p in split_root.iterdir()
        if p.is_dir()
    ):
        token_dir = (
            seq
            / "images"
            / "left"
            / "eventToken"
        )

        if not token_dir.is_dir():
            continue

        paths = sorted(
            token_dir.glob("*.pt"),
            key=timestamp,
        )

        if len(paths) < WINDOW:
            raise RuntimeError(
                f"{seq.name}: only {len(paths)} frames"
            )

        ts = np.asarray(
            [timestamp(p) for p in paths],
            dtype=np.int64,
        )

        if not np.all(
            np.diff(ts) > 0
        ):
            raise RuntimeError(
                f"{seq.name}: timestamps not increasing"
            )

        sequences.append(
            SequenceInfo(
                name=seq.name,
                paths=paths,
                timestamps=ts,
            )
        )

    if not sequences:
        raise RuntimeError(
            f"no sequences found under {split_root}"
        )

    return sequences


def load_one(path: Path) -> np.ndarray:
    x = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(x, torch.Tensor):
        raise TypeError(
            f"{path}: not a tensor"
        )

    if tuple(x.shape) != (
        N_TOKENS,
        EMBED_DIM,
    ):
        raise ValueError(
            f"{path}: bad shape {tuple(x.shape)}"
        )

    x = (
        x.detach()
        .float()
        .cpu()
        .contiguous()
    )

    if not torch.isfinite(x).all():
        raise ValueError(
            f"{path}: NaN/Inf"
        )

    return x.numpy()


def build_one_split(
    root: Path,
    packed_root: Path,
    name: str,
    image_split: str,
    workers: int,
    chunk_size: int,
):

    seqs = scan_split(
        root,
        image_split,
    )

    total_frames = sum(
        len(s.paths)
        for s in seqs
    )

    total_windows = sum(
        len(s.paths) - 7
        for s in seqs
    )

    print()
    print("=" * 80)
    print(
        f"{name.upper()}: "
        f"{len(seqs)} sequences, "
        f"{total_frames} frames, "
        f"{total_windows} windows"
    )
    print("=" * 80)

    token_path = (
        packed_root
        / f"{name}_tokens.npy"
    )

    timestamp_path = (
        packed_root
        / f"{name}_timestamps.npy"
    )

    starts_path = (
        packed_root
        / f"{name}_window_starts.npy"
    )

    tokens = open_memmap(
        token_path,
        mode="w+",
        dtype=np.float32,
        shape=(
            total_frames,
            N_TOKENS,
            EMBED_DIM,
        ),
    )

    timestamps = open_memmap(
        timestamp_path,
        mode="w+",
        dtype=np.int64,
        shape=(total_frames,),
    )

    starts = open_memmap(
        starts_path,
        mode="w+",
        dtype=np.int64,
        shape=(total_windows,),
    )

    frame_cursor = 0
    window_cursor = 0

    manifest_sequences = []

    with ThreadPoolExecutor(
        max_workers=workers
    ) as executor:

        for seq in seqs:
            n = len(seq.paths)
            nw = n - 7

            seq.frame_offset = frame_cursor
            seq.window_offset = window_cursor

            timestamps[
                frame_cursor:
                frame_cursor + n
            ] = seq.timestamps

            starts[
                window_cursor:
                window_cursor + nw
            ] = (
                frame_cursor
                + np.arange(
                    nw,
                    dtype=np.int64,
                )
            )

            for lo in tqdm(
                range(0, n, chunk_size),
                desc=f"{name}:{seq.name}",
                leave=False,
            ):
                hi = min(
                    lo + chunk_size,
                    n,
                )

                arrays = list(
                    executor.map(
                        load_one,
                        seq.paths[lo:hi],
                    )
                )

                block = np.stack(
                    arrays,
                    axis=0,
                )

                tokens[
                    frame_cursor + lo:
                    frame_cursor + hi
                ] = block

            manifest_sequences.append(
                {
                    "name": seq.name,
                    "frames": n,
                    "windows": nw,
                    "frame_start":
                        frame_cursor,
                    "frame_stop":
                        frame_cursor + n,
                    "window_start":
                        window_cursor,
                    "window_stop":
                        window_cursor + nw,
                }
            )

            print(
                f"{seq.name:34s} "
                f"frames={n:6d} "
                f"windows={nw:6d}"
            )

            frame_cursor += n
            window_cursor += nw

    tokens.flush()
    timestamps.flush()
    starts.flush()

    del tokens
    del timestamps
    del starts

    # -------------------------------------------------------------
    # Exactness audit
    # -------------------------------------------------------------

    packed_tokens = np.load(
        token_path,
        mmap_mode="r",
    )

    audit = [
        (seqs[0], 0),
        (
            seqs[len(seqs) // 2],
            len(
                seqs[len(seqs) // 2].paths
            ) // 2,
        ),
        (
            seqs[-1],
            len(seqs[-1].paths) - 1,
        ),
    ]

    max_error = 0.0

    for seq, local_idx in audit:
        original = load_one(
            seq.paths[local_idx]
        )

        global_idx = (
            seq.frame_offset
            + local_idx
        )

        packed = np.asarray(
            packed_tokens[global_idx],
            dtype=np.float32,
        )

        err = float(
            np.max(
                np.abs(
                    original - packed
                )
            )
        )

        max_error = max(
            max_error,
            err,
        )

        print(
            "[AUDIT]",
            seq.name,
            local_idx,
            "max_abs_err=",
            err,
        )

    if max_error != 0.0:
        raise RuntimeError(
            f"packing changed tokens: "
            f"max_abs_err={max_error}"
        )

    print(
        f"[AUDIT] {name}: "
        f"max_abs_err={max_error}"
    )

    return {
        "sequences":
            manifest_sequences,
        "num_sequences":
            len(seqs),
        "num_frames":
            total_frames,
        "num_windows":
            total_windows,
        "max_abs_err":
            max_error,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--rebuild",
        action="store_true",
    )

    args = parser.parse_args()

    root = args.root.resolve()
    packed = root / "packed"

    if packed.exists():
        if not args.rebuild:
            raise FileExistsError(
                f"{packed} exists; "
                f"use --rebuild"
            )

        import shutil
        shutil.rmtree(packed)

    packed.mkdir(
        parents=True,
        exist_ok=True,
    )

    train = build_one_split(
        root,
        packed,
        "train",
        "train_images",
        args.workers,
        args.chunk_size,
    )

    val = build_one_split(
        root,
        packed,
        "val",
        "test_images",
        args.workers,
        args.chunk_size,
    )

    manifest = {
        "format_version": 2,
        "token_shape": [
            N_TOKENS,
            EMBED_DIM,
        ],
        "dtype": "float32",
        "window_frames": 8,
        "context_frames": 4,
        "horizons": [1, 2, 4],
        "timestamp_unit":
            "microseconds",
        "train": train,
        "val": val,
    }

    with (
        packed / "manifest.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    print()
    print("=" * 80)
    print("PACKING COMPLETE")
    print(
        "train windows:",
        train["num_windows"],
    )
    print(
        "val windows  :",
        val["num_windows"],
    )
    print(
        "train exactness:",
        train["max_abs_err"],
    )
    print(
        "val exactness  :",
        val["max_abs_err"],
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
