from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

# Make the bundled DINOv2 submodule importable regardless of
# whether this script is launched from src/ or repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
DINOV2_ROOT = REPO_ROOT / "dinov2"

if not DINOV2_ROOT.is_dir():
    raise FileNotFoundError(
        f"DINOv2 submodule not found: {DINOV2_ROOT}"
    )

sys.path.insert(0, str(DINOV2_ROOT))

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from dinov2.models.vision_transformer import vit_small

from dataset import (
    CenterCrop,
    Normalize,
    PadToMinSide,
    PairedProcessor,
    ToTensor,
)

from pre_dse_cli import load_encoder_state_dict
from utils import accumulate_to_rgb


# ============================================================
# Formal Experiment v1.0
# EventScape preprocessing constants from GEP.
# ============================================================

SCAP_ME = [
    0.9888106104297153,
    0.9747728761936781,
    0.9859595939484498,
]

SCAP_SE = [
    0.06632214632296055,
    0.09744895769725151,
    0.0735958979181792,
]

SCAP_MI = [
    0.3691855686450492,
    0.372362750445305,
    0.38055244521714615,
]

SCAP_SI = [
    0.21236170116948833,
    0.20754919382931097,
    0.21394057988512902,
]

CROP_SIZE = 224

# Normal EventScape packet:
# ~40 ms according to audited raw timestamps.
MIN_DURATION_US = 35000
MAX_DURATION_US = 45000

# Consecutive packets in the audited sequence differ
# by only a few microseconds. 1 ms is deliberately
# conservative while still detecting real discontinuities.
MAX_INTERFRAME_GAP_US = 1000


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as fp:
        while True:
            block = fp.read(1024 * 1024)

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def frame_id_from_name(path: Path) -> int:
    """
    Examples:
        01_000_0000_events.npz
        01_000_0000_image.png

    -> 0
    """
    parts = path.stem.split("_")

    if len(parts) < 2:
        raise ValueError(
            f"cannot parse frame id: {path.name}"
        )

    return int(parts[-2])


def build_event_encoder(
    checkpoint_path: Path,
    device: torch.device,
):
    checkpoint = torch.load(
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
        checkpoint,
        "event_encoder",
    )

    encoder.to(device)
    encoder.eval()
    encoder.requires_grad_(False)

    return encoder


def build_preprocessor():
    """
    Deterministic EventScape token preprocessing.

    This deliberately removes the random crop / flip used
    for Stage-1 training.

    It matches the deterministic feature extraction path:
        ToTensor
        Normalize(EventScape stats)
        PadToMinSide(224)
        CenterCrop(224)
    """
    return PairedProcessor(
        [
            ToTensor(type="EI"),

            Normalize(
                SCAP_ME,
                SCAP_SE,
                SCAP_MI,
                SCAP_SI,
                type="EI",
            ),

            PadToMinSide(
                target=(224, 224),
                pad_x1=0,
                pad_x2=0,
            ),

            CenterCrop(
                (224, 224)
            ),
        ]
    )


def collect_pairs(
    sequence_root: Path,
):
    event_dir = (
        sequence_root
        / "events"
        / "data"
    )

    rgb_dir = (
        sequence_root
        / "rgb"
        / "data"
    )

    if not event_dir.is_dir():
        raise FileNotFoundError(event_dir)

    if not rgb_dir.is_dir():
        raise FileNotFoundError(rgb_dir)

    event_files = sorted(
        event_dir.glob("*.npz")
    )

    rgb_files = sorted(
        rgb_dir.glob("*.png")
    )

    event_map = {
        frame_id_from_name(path):
            path
        for path in event_files
    }

    rgb_map = {
        frame_id_from_name(path):
            path
        for path in rgb_files
    }

    if set(event_map) != set(rgb_map):
        only_event = sorted(
            set(event_map)
            - set(rgb_map)
        )

        only_rgb = sorted(
            set(rgb_map)
            - set(event_map)
        )

        raise RuntimeError(
            "event/RGB mismatch\n"
            f"event only={only_event[:10]}\n"
            f"rgb only={only_rgb[:10]}"
        )

    frame_ids = sorted(
        event_map
    )

    if not frame_ids:
        raise RuntimeError(
            f"no frames: {sequence_root}"
        )

    return [
        (
            frame_id,
            event_map[frame_id],
            rgb_map[frame_id],
        )
        for frame_id in frame_ids
    ]


def count_formal_windows(
    rows,
    context_frames=4,
    max_horizon=4,
):
    """
    Tc=4 and max horizon +4 means every JEPA sample
    lives inside an 8-frame temporal span.

    A window is accepted only when every included frame
    is temporal_valid and frame IDs remain consecutive.
    """

    span = (
        context_frames
        + max_horizon
    )

    total = 0

    for start in range(
        0,
        len(rows) - span + 1,
    ):
        window = rows[
            start:
            start + span
        ]

        ids = [
            int(row["frame_id"])
            for row in window
        ]

        expected = list(
            range(
                ids[0],
                ids[0] + span,
            )
        )

        if ids != expected:
            continue

        if not all(
            int(row["temporal_valid"])
            == 1
            for row in window
        ):
            continue

        total += 1

    return total


@torch.no_grad()
def process_sequence(args):
    root = Path(
        args.root
    ).resolve()

    ckpt = Path(
        args.event_encoder_ckpt
    ).resolve()

    if not ckpt.is_file():
        raise FileNotFoundError(
            ckpt
        )

    sequence_root = (
        root
        / args.split
        / args.town
        / args.sequence
    )

    if not sequence_root.is_dir():
        raise FileNotFoundError(
            sequence_root
        )

    output_root = (
        Path(args.output_root)
        .resolve()
    )

    output_sequence = (
        output_root
        / args.split
        / args.town
        / args.sequence
    )

    token_dir = (
        output_sequence
        / "eventToken"
    )

    debug_dir = (
        output_sequence
        / "debug_event_rgb"
    )

    token_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    debug_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pairs = collect_pairs(
        sequence_root
    )

    if (
        args.max_frames
        is not None
    ):
        pairs = pairs[
            :args.max_frames
        ]

    device = torch.device(
        args.device
        if (
            args.device.startswith(
                "cuda"
            )
            and torch.cuda.is_available()
        )
        else "cpu"
    )

    print("=" * 88)
    print(
        "EVENTSCAPE FORMAL V1 TOKENIZER"
    )
    print("=" * 88)

    print(
        "raw sequence       :",
        sequence_root,
    )

    print(
        "output sequence    :",
        output_sequence,
    )

    print(
        "frames             :",
        len(pairs),
    )

    print(
        "event encoder ckpt :",
        ckpt,
    )

    print(
        "ckpt sha256        :",
        sha256_file(ckpt),
    )

    print(
        "device             :",
        device,
    )

    print(
        "batch size         :",
        args.batch_size,
    )

    print(
        "event percentile   :",
        args.percentile,
    )

    print(
        "crop               :",
        "deterministic 224x224 center",
    )

    encoder = build_event_encoder(
        ckpt,
        device,
    )

    preprocessor = (
        build_preprocessor()
    )

    rows = []

    batch_tensors = []
    batch_rows = []

    previous_frame_id = None
    previous_end_us = None

    saved_debug = 0

    def flush():
        nonlocal batch_tensors
        nonlocal batch_rows

        if not batch_tensors:
            return

        batch = torch.stack(
            batch_tensors,
            dim=0,
        ).to(
            device,
            non_blocking=True,
        )

        features = (
            encoder.forward_features(
                batch
            )[
                "x_norm_patchtokens"
            ]
        )

        if (
            features.ndim != 3
            or features.shape[1:] != (
                256,
                384,
            )
        ):
            raise RuntimeError(
                "unexpected token shape: "
                f"{tuple(features.shape)}"
            )

        features = (
            features
            .float()
            .cpu()
            .contiguous()
        )

        if not torch.isfinite(
            features
        ).all():
            raise RuntimeError(
                "non-finite Stage-1 tokens"
            )

        for token, row in zip(
            features,
            batch_rows,
        ):
            out_path = (
                token_dir
                / (
                    f"{int(row['frame_id']):06d}"
                    ".pt"
                )
            )

            if (
                out_path.exists()
                and not args.overwrite
            ):
                old = torch.load(
                    out_path,
                    map_location="cpu",
                    weights_only=True,
                )

                if tuple(
                    old.shape
                ) != (
                    256,
                    384,
                ):
                    raise RuntimeError(
                        f"existing bad token: "
                        f"{out_path}"
                    )

            else:
                torch.save(
                    token,
                    out_path,
                )

            row[
                "token_file"
            ] = str(
                out_path.relative_to(
                    output_sequence
                )
            )

            rows.append(
                row
            )

        batch_tensors = []
        batch_rows = []

    for (
        frame_id,
        event_path,
        rgb_path,
    ) in tqdm(
        pairs,
        desc="EventScape tokenization",
    ):
        with np.load(
            event_path,
            allow_pickle=False,
        ) as event_data:
            required = {
                "x",
                "y",
                "p",
                "t",
            }

            if not required.issubset(
                event_data.files
            ):
                raise RuntimeError(
                    f"{event_path}: "
                    "missing x/y/p/t"
                )

            x = np.asarray(
                event_data["x"]
            )

            y = np.asarray(
                event_data["y"]
            )

            p = np.asarray(
                event_data["p"]
            )

            t = np.asarray(
                event_data["t"]
            ).reshape(-1)

        if (
            len(x) != len(y)
            or len(x) != len(p)
            or len(x) != len(t)
        ):
            raise RuntimeError(
                f"event length mismatch: "
                f"{event_path}"
            )

        if len(t) == 0:
            raise RuntimeError(
                f"empty event frame: "
                f"{event_path}"
            )

        t64 = t.astype(
            np.int64,
            copy=False,
        )

        if not np.all(
            np.diff(t64) >= 0
        ):
            raise RuntimeError(
                f"non-monotonic events: "
                f"{event_path}"
            )

        start_us = int(
            t64[0]
        )

        end_us = int(
            t64[-1]
        )

        duration_us = (
            end_us
            - start_us
        )

        with Image.open(
            rgb_path
        ) as rgb:
            rgb = rgb.convert(
                "RGB"
            )

            width, height = (
                rgb.size
            )

        event_rgb_np = (
            accumulate_to_rgb(
                x.astype(
                    np.int32,
                    copy=False,
                ),
                y.astype(
                    np.int32,
                    copy=False,
                ),
                p,
                (height, width),
                pct=args.percentile,
            )
        )

        if (
            event_rgb_np.shape
            != (
                height,
                width,
                3,
            )
        ):
            raise RuntimeError(
                "bad event RGB shape "
                f"{event_rgb_np.shape}"
            )

        event_rgb = (
            Image.fromarray(
                event_rgb_np,
                mode="RGB",
            )
        )

        if (
            saved_debug
            < args.debug_frames
        ):
            event_rgb.save(
                debug_dir
                / f"{frame_id:06d}.png"
            )

            saved_debug += 1

        event_tensor, _ = (
            preprocessor(
                event_rgb,
                None,
            )
        )

        if tuple(
            event_tensor.shape
        ) != (
            3,
            224,
            224,
        ):
            raise RuntimeError(
                "bad preprocessed shape: "
                f"{tuple(event_tensor.shape)}"
            )

        if not torch.isfinite(
            event_tensor
        ).all():
            raise RuntimeError(
                "non-finite input tensor"
            )

        if previous_end_us is None:
            prev_gap_us = None
        else:
            prev_gap_us = (
                start_us
                - previous_end_us
            )

        frame_consecutive = (
            previous_frame_id
            is None
            or frame_id
            == previous_frame_id + 1
        )

        duration_valid = (
            MIN_DURATION_US
            <= duration_us
            <= MAX_DURATION_US
        )

        gap_valid = (
            previous_end_us
            is None
            or (
                0
                <= prev_gap_us
                <= MAX_INTERFRAME_GAP_US
            )
        )

        temporal_valid = (
            duration_valid
            and frame_consecutive
            and gap_valid
        )

        row = {
            "dataset":
                "EventScape",

            "split":
                args.split,

            "town":
                args.town,

            "sequence":
                args.sequence,

            "frame_id":
                frame_id,

            "event_file":
                str(
                    event_path.relative_to(
                        sequence_root
                    )
                ),

            "rgb_file":
                str(
                    rgb_path.relative_to(
                        sequence_root
                    )
                ),

            "event_count":
                len(t64),

            "timestamp_start_us":
                start_us,

            "timestamp_end_us":
                end_us,

            # This is the timestamp used by
            # future temporal dataset code.
            "timestamp_us":
                end_us,

            "timestamp_s":
                end_us * 1e-6,

            "duration_us":
                duration_us,

            "prev_gap_us":
                (
                    ""
                    if prev_gap_us
                    is None
                    else prev_gap_us
                ),

            "frame_consecutive":
                int(
                    frame_consecutive
                ),

            "duration_valid":
                int(
                    duration_valid
                ),

            "gap_valid":
                int(
                    gap_valid
                ),

            "temporal_valid":
                int(
                    temporal_valid
                ),

            "token_file":
                "",
        }

        batch_tensors.append(
            event_tensor
        )

        batch_rows.append(
            row
        )

        if (
            len(batch_tensors)
            >= args.batch_size
        ):
            flush()

        previous_frame_id = (
            frame_id
        )

        previous_end_us = (
            end_us
        )

    flush()

    if len(rows) != len(pairs):
        raise RuntimeError(
            "output row count mismatch"
        )

    manifest_path = (
        output_sequence
        / "manifest.csv"
    )

    fieldnames = [
        "dataset",
        "split",
        "town",
        "sequence",
        "frame_id",
        "event_file",
        "rgb_file",
        "event_count",
        "timestamp_start_us",
        "timestamp_end_us",
        "timestamp_us",
        "timestamp_s",
        "duration_us",
        "prev_gap_us",
        "frame_consecutive",
        "duration_valid",
        "gap_valid",
        "temporal_valid",
        "token_file",
    ]

    with manifest_path.open(
        "w",
        newline="",
    ) as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    token_files = sorted(
        token_dir.glob(
            "*.pt"
        )
    )

    if len(token_files) != len(rows):
        raise RuntimeError(
            "token count mismatch: "
            f"{len(token_files)} "
            f"vs {len(rows)}"
        )

    valid_frames = sum(
        int(
            row["temporal_valid"]
        )
        for row in rows
    )

    formal_windows = (
        count_formal_windows(
            rows,
            context_frames=4,
            max_horizon=4,
        )
    )

    metadata = {
        "format":
            "eventscape_formal_v1",

        "dataset":
            "EventScape",

        "split":
            args.split,

        "town":
            args.town,

        "sequence":
            args.sequence,

        "num_frames":
            len(rows),

        "num_temporal_valid_frames":
            valid_frames,

        "num_tc4_h124_windows":
            formal_windows,

        "timestamp_unit":
            "microseconds",

        "timestamp_scale_to_seconds":
            1e-6,

        "normal_duration_us":
            [
                MIN_DURATION_US,
                MAX_DURATION_US,
            ],

        "max_interframe_gap_us":
            MAX_INTERFRAME_GAP_US,

        "event_representation":
            "accumulate_to_rgb",

        "event_percentile":
            args.percentile,

        "event_mean":
            SCAP_ME,

        "event_std":
            SCAP_SE,

        "preprocessing":
            [
                "ToTensor",
                "Normalize(EventScape)",
                "PadToMinSide(224,224)",
                "CenterCrop(224,224)",
            ],

        "random_augmentation":
            False,

        "event_encoder":
            "GEP Event Encoder Small",

        "event_encoder_checkpoint":
            str(ckpt),

        "event_encoder_checkpoint_sha256":
            sha256_file(ckpt),

        "token_shape":
            [
                256,
                384,
            ],

        "token_dtype":
            "float32",
    }

    metadata_path = (
        output_sequence
        / "metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    print()
    print("=" * 88)
    print(
        "EVENTSCAPE FORMAL TOKENIZATION COMPLETE"
    )
    print("=" * 88)

    print(
        "frames                 :",
        len(rows),
    )

    print(
        "temporal-valid frames  :",
        valid_frames,
    )

    print(
        "invalid frames         :",
        len(rows)
        - valid_frames,
    )

    print(
        "Tc4/H[1,2,4] windows   :",
        formal_windows,
    )

    print(
        "matched 8-frame windows:",
        formal_windows,
    )

    print(
        "tokens                 :",
        len(token_files),
    )

    print(
        "manifest               :",
        manifest_path,
    )

    print(
        "metadata               :",
        metadata_path,
    )


def build_parser():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--root",
        required=True,
    )

    p.add_argument(
        "--output-root",
        required=True,
    )

    p.add_argument(
        "--split",
        choices=[
            "train",
            "valid",
        ],
        required=True,
    )

    p.add_argument(
        "--town",
        required=True,
    )

    p.add_argument(
        "--sequence",
        required=True,
    )

    p.add_argument(
        "--event-encoder-ckpt",
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
        default=99.0,
    )

    p.add_argument(
        "--debug-frames",
        type=int,
        default=4,
    )

    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
    )

    p.add_argument(
        "--overwrite",
        action="store_true",
    )

    return p


def main():
    args = (
        build_parser()
        .parse_args()
    )

    process_sequence(
        args
    )


if __name__ == "__main__":
    main()
