from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import sys
sys.path.append("dinov2")

from dinov2.models.vision_transformer import vit_small
from pre_dse_cli import load_encoder_state_dict
from utils import accumulate_to_rgb


# ---------------------------------------------------------------------
# Must match the DSEC Stage-1 token preprocessing.
# ---------------------------------------------------------------------

DSEC_EVENT_MEAN = torch.tensor(
    [
        0.8993729784963826,
        0.7969581014619264,
        0.8928228776286392,
    ],
    dtype=torch.float32,
).view(3, 1, 1)

DSEC_EVENT_STD = torch.tensor(
    [
        0.2726268701553345,
        0.2706460952758789,
        0.2812058925628662,
    ],
    dtype=torch.float32,
).view(3, 1, 1)

RAW_H = 260
RAW_W = 346

CROP_H = 224
CROP_W = 224

CROP_TOP = (RAW_H - CROP_H) // 2
CROP_LEFT = (RAW_W - CROP_W) // 2

# E-RAFT's dense MVSEC protocol masks the vehicle hood.
MVSEC_HOOD_ROW = 193


def center_crop_np(x: np.ndarray) -> np.ndarray:
    return x[
        ...,
        CROP_TOP:CROP_TOP + CROP_H,
        CROP_LEFT:CROP_LEFT + CROP_W,
    ]


def event_rgb_to_tensor(event_rgb: np.ndarray) -> torch.Tensor:
    if event_rgb.shape != (RAW_H, RAW_W, 3):
        raise ValueError(
            f"unexpected event RGB shape {event_rgb.shape}"
        )

    x = torch.from_numpy(
        np.asarray(event_rgb)
    ).permute(2, 0, 1).float() / 255.0

    x = (
        x
        - DSEC_EVENT_MEAN
    ) / DSEC_EVENT_STD

    x = x[
        :,
        CROP_TOP:CROP_TOP + CROP_H,
        CROP_LEFT:CROP_LEFT + CROP_W,
    ].contiguous()

    if x.shape != (3, 224, 224):
        raise RuntimeError(
            f"bad cropped tensor shape {tuple(x.shape)}"
        )

    return x


def load_stage1_encoder(
    checkpoint_path: Path,
    device: torch.device,
):
    state = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    encoder = vit_small(
        patch_size=14,
        img_size=518,
        block_chunks=0,
        init_values=1e-6,
    )

    load_encoder_state_dict(
        encoder,
        state,
        "event_encoder",
    )

    encoder.to(device)
    encoder.eval()
    encoder.requires_grad_(False)

    return encoder


# ---------------------------------------------------------------------
# NPZ -> NPY cache
#
# MVSEC flow arrays are very large. Do NOT decompress x/y flow into
# normal RAM. Extract their underlying .npy members once, then memmap.
# ---------------------------------------------------------------------

def ensure_npz_cache(
    npz_path: Path,
    cache_dir: Path,
) -> dict[str, Path]:
    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    wanted = {
        "timestamps.npy",
        "x_flow_dist.npy",
        "y_flow_dist.npy",
    }

    outputs = {
        name[:-4]:
            cache_dir / name
        for name in wanted
    }

    if all(p.is_file() for p in outputs.values()):
        return outputs

    print(
        f"[cache] extracting {npz_path.name} "
        f"to {cache_dir}"
    )

    with zipfile.ZipFile(
        npz_path,
        "r",
    ) as zf:
        members = set(zf.namelist())

        missing = wanted - members
        if missing:
            raise KeyError(
                f"{npz_path}: missing NPZ members "
                f"{sorted(missing)}; have "
                f"{sorted(members)}"
            )

        for member in sorted(wanted):
            dst = cache_dir / member

            if dst.is_file():
                continue

            tmp = dst.with_suffix(
                dst.suffix + ".tmp"
            )

            with zf.open(member) as src, \
                    tmp.open("wb") as out:
                shutil.copyfileobj(
                    src,
                    out,
                    length=16 * 1024 * 1024,
                )

            tmp.replace(dst)

    return outputs


# ---------------------------------------------------------------------
# Fast timestamp -> raw-event index.
#
# MVSEC supplies image_raw_event_inds, so first bracket using nearby
# image timestamps. Fall back to binary search on HDF5 if necessary.
# ---------------------------------------------------------------------

def h5_searchsorted(
    events,
    timestamp: float,
    *,
    side: str,
    lo: int = 0,
    hi: int | None = None,
) -> int:
    if side not in {"left", "right"}:
        raise ValueError(side)

    if hi is None:
        hi = len(events)

    lo = int(max(0, lo))
    hi = int(min(len(events), hi))

    while lo < hi:
        mid = (lo + hi) // 2

        value = float(
            events[mid, 2]
        )

        if (
            value < timestamp
            or (
                side == "right"
                and value <= timestamp
            )
        ):
            lo = mid + 1
        else:
            hi = mid

    return lo


def event_index_at_time(
    events,
    image_ts: np.ndarray,
    image_event_inds: np.ndarray,
    timestamp: float,
    *,
    side: str,
) -> int:
    j = int(
        np.searchsorted(
            image_ts,
            timestamp,
            side="left",
        )
    )

    n_events = len(events)

    if 0 < j < len(image_ts):
        lo = int(
            image_event_inds[j - 1]
        )
        hi = int(
            image_event_inds[j]
        )

        if lo >= 0 and hi >= lo:
            # Small safety margin around the bracket.
            lo = max(
                0,
                lo - 16,
            )
            hi = min(
                n_events,
                hi + 16,
            )

            if lo < hi:
                t = np.asarray(
                    events[
                        lo:hi,
                        2,
                    ],
                    dtype=np.float64,
                )

                local = int(
                    np.searchsorted(
                        t,
                        timestamp,
                        side=side,
                    )
                )

                idx = lo + local

                # Verify the bracket result. If it lies at
                # the boundary, use exact HDF5 binary search.
                if 0 < local < len(t):
                    return idx

    return h5_searchsorted(
        events,
        timestamp,
        side=side,
    )


def make_event_rgb(
    events,
    start_idx: int,
    end_idx: int,
    percentile: float,
) -> tuple[np.ndarray, int]:
    if end_idx < start_idx:
        raise ValueError(
            f"event index reversed: "
            f"{start_idx}>{end_idx}"
        )

    e = np.asarray(
        events[
            start_idx:end_idx
        ]
    )

    if e.size == 0:
        # Same white background convention used by
        # the existing MVSEC preprocessor.
        rgb = np.full(
            (RAW_H, RAW_W, 3),
            255,
            dtype=np.uint8,
        )
        return rgb, 0

    x = e[:, 0].astype(
        np.int32,
        copy=False,
    )
    y = e[:, 1].astype(
        np.int32,
        copy=False,
    )

    # MVSEC polarity is -1/+1.
    p = e[:, 3] > 0

    in_bounds = (
        (x >= 0)
        & (x < RAW_W)
        & (y >= 0)
        & (y < RAW_H)
    )

    x = x[in_bounds]
    y = y[in_bounds]
    p = p[in_bounds]

    rgb = accumulate_to_rgb(
        x,
        y,
        p,
        (RAW_H, RAW_W),
        pct=percentile,
    )

    return rgb, int(len(x))


def make_flow_target(
    u: np.ndarray,
    v: np.ndarray,
) -> torch.Tensor:
    u = np.asarray(
        u,
        dtype=np.float32,
    )
    v = np.asarray(
        v,
        dtype=np.float32,
    )

    if (
        u.shape != (RAW_H, RAW_W)
        or v.shape != (RAW_H, RAW_W)
    ):
        raise ValueError(
            f"unexpected flow shapes "
            f"{u.shape}, {v.shape}"
        )

    if not (
        np.isfinite(u).all()
        and np.isfinite(v).all()
    ):
        raise ValueError(
            "non-finite MVSEC flow"
        )

    # Match E-RAFT dense MVSEC validity:
    # either x or y flow must be non-zero.
    valid = (
        (u != 0)
        | (v != 0)
    )

    # E-RAFT excludes the vehicle hood.
    valid[
        MVSEC_HOOD_ROW:,
        :
    ] = False

    flow = np.stack(
        [
            u,
            v,
            valid.astype(
                np.float32
            ),
        ],
        axis=0,
    )

    flow = center_crop_np(
        flow
    ).copy()

    out = torch.from_numpy(
        flow
    ).float()

    if out.shape != (
        3,
        224,
        224,
    ):
        raise RuntimeError(
            f"bad flow shape "
            f"{tuple(out.shape)}"
        )

    return out


def process_sequence(
    *,
    root: Path,
    sequence: str,
    encoder,
    device: torch.device,
    batch_size: int,
    percentile: float,
    start_index: int,
    max_samples: int | None,
    overwrite: bool,
):
    data_path = (
        root
        / "raw"
        / f"{sequence}_data.hdf5"
    )

    npz_path = (
        root
        / "flow_gt"
        / f"{sequence}_gt_flow_dist.npz"
    )

    if not data_path.is_file():
        raise FileNotFoundError(
            data_path
        )

    if not npz_path.is_file():
        raise FileNotFoundError(
            npz_path
        )

    cache = ensure_npz_cache(
        npz_path,
        root
        / "flow_gt"
        / ".npy_cache"
        / sequence,
    )

    timestamps = np.load(
        cache["timestamps"],
        mmap_mode="r",
    )

    x_flow = np.load(
        cache["x_flow_dist"],
        mmap_mode="r",
    )

    y_flow = np.load(
        cache["y_flow_dist"],
        mmap_mode="r",
    )

    n = min(
        len(timestamps),
        len(x_flow),
        len(y_flow),
    )

    if n < 6:
        raise RuntimeError(
            f"{sequence}: too few "
            f"flow frames: {n}"
        )

    # i is flow index:
    # Flow[i] = t_i -> t_(i+1)
    #
    # context:
    # z_(i-3), ..., z_i
    #
    # z_j uses events:
    # (t_(j-1), t_j]
    #
    # Therefore i>=4.
    first_i = max(
        4,
        int(start_index),
    )

    last_exclusive = n - 1

    if max_samples is not None:
        last_exclusive = min(
            last_exclusive,
            first_i
            + int(max_samples),
        )

    sample_indices = list(
        range(
            first_i,
            last_exclusive,
        )
    )

    if not sample_indices:
        raise RuntimeError(
            f"{sequence}: no samples"
        )

    needed_token_indices = list(
        range(
            sample_indices[0] - 3,
            sample_indices[-1] + 1,
        )
    )

    out_root = (
        root
        / "processed"
        / sequence
    )

    token_dir = (
        out_root
        / "eventToken"
    )

    flow_dir = (
        out_root
        / "flow"
    )

    debug_dir = (
        out_root
        / "debug_event_rgb"
    )

    for p in (
        token_dir,
        flow_dir,
        debug_dir,
    ):
        p.mkdir(
            parents=True,
            exist_ok=True,
        )

    print()
    print("=" * 80)
    print("MVSEC FLOW PREPROCESS")
    print("=" * 80)
    print("sequence       :", sequence)
    print("data           :", data_path)
    print("flow npz       :", npz_path)
    print("flow frames    :", n)
    print(
        "flow dt median :",
        float(
            np.median(
                np.diff(
                    np.asarray(
                        timestamps
                    )
                )
            )
        ),
    )
    print(
        "sample range   :",
        sample_indices[0],
        "..",
        sample_indices[-1],
    )
    print(
        "num samples    :",
        len(sample_indices),
    )
    print(
        "token range    :",
        needed_token_indices[0],
        "..",
        needed_token_indices[-1],
    )
    print(
        "num tokens     :",
        len(
            needed_token_indices
        ),
    )

    with h5py.File(
        data_path,
        "r",
    ) as f:
        events = f[
            "davis/left/events"
        ]

        image_ts = np.asarray(
            f[
                "davis/left/"
                "image_raw_ts"
            ][:],
            dtype=np.float64,
        )

        image_event_inds = np.asarray(
            f[
                "davis/left/"
                "image_raw_event_inds"
            ][:],
            dtype=np.int64,
        )

        if (
            image_ts.ndim != 1
            or image_event_inds.ndim
            != 1
        ):
            raise RuntimeError(
                "bad MVSEC image index arrays"
            )

        if len(image_ts) != len(
            image_event_inds
        ):
            raise RuntimeError(
                "image timestamp/index "
                "length mismatch"
            )

        # --------------------------------------------------
        # Generate Stage-1 tokens.
        # --------------------------------------------------

        pending_tensors = []
        pending_meta = []

        token_records = {}

        def flush_batch():
            nonlocal pending_tensors
            nonlocal pending_meta

            if not pending_tensors:
                return

            batch = torch.stack(
                pending_tensors,
                dim=0,
            ).to(
                device,
                non_blocking=True,
            )

            with torch.inference_mode():
                z = (
                    encoder.forward_features(
                        batch
                    )[
                        "x_norm_patchtokens"
                    ]
                )

            z = z.float().cpu()

            if (
                z.ndim != 3
                or z.shape[1:]
                != (256, 384)
            ):
                raise RuntimeError(
                    "unexpected Stage-1 "
                    f"token shape "
                    f"{tuple(z.shape)}"
                )

            for k, meta in enumerate(
                pending_meta
            ):
                token_index = meta[
                    "token_index"
                ]

                path = (
                    token_dir
                    / f"{token_index:06d}.pt"
                )

                torch.save(
                    z[k].contiguous(),
                    path,
                )

                token_records[
                    token_index
                ] = meta

            pending_tensors = []
            pending_meta = []

        for token_index in tqdm(
            needed_token_indices,
            desc=f"{sequence}: tokens",
        ):
            token_path = (
                token_dir
                / f"{token_index:06d}.pt"
            )

            t_prev = float(
                timestamps[
                    token_index - 1
                ]
            )

            t_curr = float(
                timestamps[
                    token_index
                ]
            )

            if (
                token_path.is_file()
                and not overwrite
            ):
                token_records[
                    token_index
                ] = {
                    "token_index":
                        token_index,
                    "timestamp":
                        t_curr,
                    "event_start":
                        t_prev,
                    "event_end":
                        t_curr,
                    "event_count":
                        -1,
                }
                continue

            # Events in:
            # (t_prev, t_curr]
            event_start_idx = (
                event_index_at_time(
                    events,
                    image_ts,
                    image_event_inds,
                    t_prev,
                    side="right",
                )
            )

            event_end_idx = (
                event_index_at_time(
                    events,
                    image_ts,
                    image_event_inds,
                    t_curr,
                    side="right",
                )
            )

            event_rgb, event_count = (
                make_event_rgb(
                    events,
                    event_start_idx,
                    event_end_idx,
                    percentile,
                )
            )

            # Save a few visual sanity checks.
            if (
                token_index
                < needed_token_indices[0]
                + 8
            ):
                Image.fromarray(
                    event_rgb
                ).save(
                    debug_dir
                    / f"{token_index:06d}.png"
                )

            x = event_rgb_to_tensor(
                event_rgb
            )

            pending_tensors.append(
                x
            )

            pending_meta.append(
                {
                    "token_index":
                        token_index,
                    "timestamp":
                        t_curr,
                    "event_start":
                        t_prev,
                    "event_end":
                        t_curr,
                    "event_count":
                        event_count,
                }
            )

            if (
                len(pending_tensors)
                >= batch_size
            ):
                flush_batch()

        flush_batch()

    # ------------------------------------------------------
    # Save flow targets + manifest.
    # ------------------------------------------------------

    manifest_rows = []

    for i in tqdm(
        sample_indices,
        desc=f"{sequence}: flow",
    ):
        flow_path = (
            flow_dir
            / f"{i:06d}.pt"
        )

        if (
            overwrite
            or not flow_path.is_file()
        ):
            target = make_flow_target(
                x_flow[i],
                y_flow[i],
            )

            torch.save(
                target.contiguous(),
                flow_path,
            )

        context_indices = [
            i - 3,
            i - 2,
            i - 1,
            i,
        ]

        for j in context_indices:
            path = (
                token_dir
                / f"{j:06d}.pt"
            )

            if not path.is_file():
                raise RuntimeError(
                    f"missing context token "
                    f"{path}"
                )

        target = torch.load(
            flow_path,
            map_location="cpu",
            weights_only=True,
        )

        valid_fraction = float(
            target[2].mean().item()
        )

        manifest_rows.append(
            {
                "sequence":
                    sequence,
                "flow_index":
                    i,
                "flow_start_timestamp":
                    float(timestamps[i]),
                "flow_end_timestamp":
                    float(
                        timestamps[i + 1]
                    ),
                "context_0":
                    context_indices[0],
                "context_1":
                    context_indices[1],
                "context_2":
                    context_indices[2],
                "context_3":
                    context_indices[3],
                "flow_file":
                    str(
                        flow_path.relative_to(
                            out_root
                        )
                    ),
                "valid_fraction":
                    valid_fraction,
            }
        )

    manifest_path = (
        out_root
        / "manifest.csv"
    )

    with manifest_path.open(
        "w",
        newline="",
    ) as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=list(
                manifest_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            manifest_rows
        )

    metadata = {
        "sequence":
            sequence,
        "protocol":
            "mvsec_20hz_causal_flow_transfer",
        "raw_resolution":
            [RAW_H, RAW_W],
        "crop_resolution":
            [CROP_H, CROP_W],
        "crop_top":
            CROP_TOP,
        "crop_left":
            CROP_LEFT,
        "event_interval":
            "(t_(i-1), t_i]",
        "target_flow_interval":
            "t_i -> t_(i+1)",
        "context_frames":
            4,
        "percentile":
            percentile,
        "stage1_token_shape":
            [256, 384],
        "hood_mask_raw_row_start":
            MVSEC_HOOD_ROW,
        "num_samples":
            len(manifest_rows),
    }

    (
        out_root
        / "metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    print()
    print(
        f"[PASS] {sequence}: "
        f"{len(manifest_rows)} samples"
    )
    print(
        "manifest:",
        manifest_path,
    )


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--root",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--sequences",
        nargs="+",
        default=[
            "outdoor_day1",
            "outdoor_day2",
        ],
    )

    p.add_argument(
        "--event-encoder-ckpt",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--device",
        default="cuda:0",
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    p.add_argument(
        "--percentile",
        type=float,
        default=90.0,
    )

    p.add_argument(
        "--start-index",
        type=int,
        default=4,
    )

    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
    )

    p.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = p.parse_args()

    if args.batch_size < 1:
        raise ValueError(
            "--batch-size must be >=1"
        )

    root = args.root.resolve()

    device = torch.device(
        args.device
    )

    print(
        "loading Stage-1 event encoder..."
    )

    encoder = load_stage1_encoder(
        args.event_encoder_ckpt,
        device,
    )

    print(
        "Stage-1 encoder loaded on",
        device,
    )

    for sequence in args.sequences:
        process_sequence(
            root=root,
            sequence=sequence,
            encoder=encoder,
            device=device,
            batch_size=args.batch_size,
            percentile=args.percentile,
            start_index=args.start_index,
            max_samples=args.max_samples,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
