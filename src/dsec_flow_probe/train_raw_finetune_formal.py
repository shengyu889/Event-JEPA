from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader, Subset

from dinov2.models.vision_transformer import vit_small

from dsec_flow_probe.dataset import (
    decode_dsec_flow,
    center_crop_flow,
)
from dsec_flow_probe.head import (
    PatchLinearFlowHead,
    masked_flow_l1,
)
from dsec_flow_probe.backbone import module_sha256
from event_jepa.encoders import EventTokenEncoder
from utils import get_lr


# ---------------------------------------------------------------------
# Exact GEP DSEC event normalization used for official Stage-1 tokens.
# ---------------------------------------------------------------------

DSEC_ME = torch.tensor(
    [
        0.8993729784963826,
        0.7969581014619264,
        0.8928228776286392,
    ],
    dtype=torch.float32,
).view(3, 1, 1)

DSEC_SE = torch.tensor(
    [
        0.2726268701553345,
        0.2706460952758789,
        0.2812058925628662,
    ],
    dtype=torch.float32,
).view(3, 1, 1)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_names(path: str | Path):
    path = Path(path)

    names = [
        x.strip()
        for x in path.read_text().splitlines()
        if x.strip()
    ]

    if not names:
        raise RuntimeError(
            f"empty sequence file: {path}"
        )

    if len(names) != len(set(names)):
        raise RuntimeError(
            f"duplicate names: {path}"
        )

    return sorted(names)


# ---------------------------------------------------------------------
# EXACT deterministic Stage-1 preprocessing used when generating
# eventToken_vits14:
#
# ToTensor
# Normalize(DSEC)
# PadToMinSide(224,224)
# CenterCrop(224,224)
# ---------------------------------------------------------------------

def preprocess_event_image(
    path: str | Path,
) -> torch.Tensor:

    image = Image.open(path).convert("RGB")

    x = torch.from_numpy(
        np.array(image, copy=True)
    )

    x = (
        x.permute(2, 0, 1)
        .float()
        .div_(255.0)
    )

    x = (
        x - DSEC_ME
    ) / DSEC_SE

    _, h, w = x.shape

    pad_bottom = max(
        0,
        224 - h,
    )

    pad_right = max(
        0,
        224 - w,
    )

    if (
        pad_bottom > 0
        or pad_right > 0
    ):
        x = F.pad(
            x,
            (
                0,
                pad_right,
                0,
                pad_bottom,
            ),
            mode="constant",
            value=0.0,
        )

    _, h, w = x.shape

    if h < 224 or w < 224:
        raise RuntimeError(
            f"event image too small after pad: "
            f"{path}: {(h, w)}"
        )

    top = (
        h - 224
    ) // 2

    left = (
        w - 224
    ) // 2

    x = x[
        :,
        top : top + 224,
        left : left + 224,
    ]

    if tuple(x.shape) != (
        3,
        224,
        224,
    ):
        raise RuntimeError(
            f"bad event shape: "
            f"{tuple(x.shape)}"
        )

    return x.contiguous()


class DSECRawSequenceFlowDataset(
    Dataset
):
    """
    Raw-event counterpart of the existing
    DSECSequenceFlowDataset.

    Input:
        events [Tc,3,224,224]

    Target:
        flow [3,224,224]

    Context is causal and ends exactly at
    flow FROM timestamp.

    No future frame.
    No sequence boundary crossing.
    """

    def __init__(
        self,
        root: str | Path,
        sequences,
        context_frames: int = 4,
        max_frame_gap_us: int = 75000,
    ):
        super().__init__()

        self.root = Path(root)
        self.sequences = list(
            sequences
        )

        self.context_frames = int(
            context_frames
        )

        self.max_frame_gap_us = int(
            max_frame_gap_us
        )

        if self.context_frames != 4:
            raise ValueError(
                "formal Event-JEPA flow "
                "requires Tc=4"
            )

        self.samples = []

        self.skipped_missing_source = 0
        self.skipped_short_context = 0
        self.skipped_context_gap = 0

        for sequence in self.sequences:
            self._add_sequence(
                sequence
            )

        if not self.samples:
            raise RuntimeError(
                "no valid raw DSEC flow samples"
            )

    def _add_sequence(
        self,
        sequence: str,
    ):
        flow_root = (
            self.root
            / "train_optical_flow"
            / sequence
            / "flow"
        )

        forward_dir = (
            flow_root
            / "forward"
        )

        timestamps_file = (
            flow_root
            / "forward_timestamps.txt"
        )

        event_dir = (
            self.root
            / "train_images"
            / sequence
            / "images"
            / "left"
            / "eventImage"
        )

        for path in (
            forward_dir,
            timestamps_file,
            event_dir,
        ):
            if not path.exists():
                raise FileNotFoundError(
                    path
                )

        flow_paths = sorted(
            forward_dir.glob("*.png"),
            key=lambda p: int(p.stem),
        )

        lines = [
            line.strip()
            for line in (
                timestamps_file
                .read_text()
                .splitlines()
            )
            if line.strip()
        ]

        # first row is:
        # # from_timestamp_us, to_timestamp_us
        rows = lines[1:]

        if len(flow_paths) != len(rows):
            raise RuntimeError(
                f"{sequence}: flow/timestamp "
                f"mismatch: "
                f"{len(flow_paths)} vs "
                f"{len(rows)}"
            )

        event_paths = sorted(
            event_dir.glob("*.png"),
            key=lambda p: int(p.stem),
        )

        event_ts = [
            int(p.stem)
            for p in event_paths
        ]

        event_index = {
            ts: i
            for i, ts in enumerate(
                event_ts
            )
        }

        for flow_path, row in zip(
            flow_paths,
            rows,
        ):
            fields = [
                x.strip()
                for x in row.split(",")
            ]

            from_ts = int(
                fields[0]
            )

            to_ts = int(
                fields[1]
            )

            if from_ts not in event_index:
                self.skipped_missing_source += 1
                continue

            source_idx = event_index[
                from_ts
            ]

            start_idx = (
                source_idx
                - self.context_frames
                + 1
            )

            if start_idx < 0:
                self.skipped_short_context += 1
                continue

            context_paths = (
                event_paths[
                    start_idx:
                    source_idx + 1
                ]
            )

            context_ts = (
                event_ts[
                    start_idx:
                    source_idx + 1
                ]
            )

            if (
                len(context_paths)
                != self.context_frames
            ):
                continue

            gaps = np.diff(
                np.asarray(
                    context_ts,
                    dtype=np.int64,
                )
            )

            if (
                (gaps <= 0).any()
                or (
                    gaps
                    > self.max_frame_gap_us
                ).any()
            ):
                self.skipped_context_gap += 1
                continue

            self.samples.append(
                {
                    "sequence":
                        sequence,

                    "flow_path":
                        flow_path,

                    "from_ts":
                        from_ts,

                    "to_ts":
                        to_ts,

                    "event_paths":
                        context_paths,

                    "context_ts":
                        context_ts,
                }
            )

    def __len__(self):
        return len(
            self.samples
        )

    def __getitem__(
        self,
        index: int,
    ):
        sample = self.samples[
            index
        ]

        events = torch.stack(
            [
                preprocess_event_image(
                    p
                )
                for p in
                sample["event_paths"]
            ],
            dim=0,
        )

        flow = decode_dsec_flow(
            sample["flow_path"]
        )

        flow = center_crop_flow(
            flow,
            crop_h=224,
            crop_w=224,
        )

        return {
            "events":
                events,

            "flow":
                flow,

            "sequence":
                sample["sequence"],

            "from_ts":
                sample["from_ts"],

            "to_ts":
                sample["to_ts"],
        }


def build_stage1(
    checkpoint: str | Path,
):
    state = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    if (
        not isinstance(state, dict)
        or "event_encoder"
        not in state
    ):
        raise KeyError(
            "Stage-1 checkpoint must "
            "contain event_encoder"
        )

    encoder = vit_small(
        patch_size=14,
        img_size=518,
        block_chunks=0,
        init_values=1e-6,
        num_register_tokens=0,
    )

    result = encoder.load_state_dict(
        state["event_encoder"],
        strict=True,
    )

    print(
        "Stage-1 strict load:",
        result,
    )

    return encoder


def build_stage2(
    checkpoint: str | Path,
):
    state = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    if "online_encoder" not in state:
        raise KeyError(
            "Event-JEPA checkpoint missing "
            "online_encoder"
        )

    if "config" not in state:
        raise KeyError(
            "Event-JEPA checkpoint missing "
            "config"
        )

    cfg = dict(
        state["config"]
    )

    required = [
        "embed_dim",
        "num_heads",
        "encoder_layers",
        "n_tokens",
        "max_positions",
        "max_context_frames",
        "context_frames",
    ]

    missing = [
        key
        for key in required
        if key not in cfg
    ]

    if missing:
        raise KeyError(
            f"checkpoint config missing "
            f"{missing}"
        )

    if int(
        cfg["context_frames"]
    ) != 4:
        raise RuntimeError(
            "formal downstream requires "
            "Event-JEPA Tc=4"
        )

    if int(
        cfg["n_tokens"]
    ) != 256:
        raise RuntimeError(
            "formal downstream expects "
            "256 Stage-1 tokens"
        )

    if int(
        cfg["embed_dim"]
    ) != 384:
        raise RuntimeError(
            "formal ViT-S/14 expects "
            "D=384"
        )

    encoder = EventTokenEncoder(
        embed_dim=int(
            cfg["embed_dim"]
        ),
        num_heads=int(
            cfg["num_heads"]
        ),
        num_layers=int(
            cfg["encoder_layers"]
        ),
        n_tokens=int(
            cfg["n_tokens"]
        ),
        max_positions=int(
            cfg["max_positions"]
        ),
        max_context_frames=int(
            cfg["max_context_frames"]
        ),
    )

    result = encoder.load_state_dict(
        state["online_encoder"],
        strict=True,
    )

    print(
        "Stage-2 strict load:",
        result,
    )

    return encoder, cfg


def build_ft10_indices(
    dataset,
    seed: int,
):
    """
    10% labels, independently sampled
    inside every training sequence.

    Fixed seed makes subset exactly
    reproducible and preserves all
    training sequences.
    """

    groups = defaultdict(list)

    for i, sample in enumerate(
        dataset.samples
    ):
        groups[
            sample["sequence"]
        ].append(i)

    rng = np.random.default_rng(
        seed
    )

    chosen = []
    details = {}

    for sequence in sorted(groups):
        indices = np.asarray(
            groups[sequence],
            dtype=np.int64,
        )

        k = max(
            1,
            int(
                round(
                    len(indices)
                    * 0.10
                )
            ),
        )

        selected = rng.choice(
            indices,
            size=k,
            replace=False,
        )

        selected = sorted(
            int(x)
            for x in selected
        )

        chosen.extend(
            selected
        )

        details[sequence] = {
            "available":
                len(indices),

            "selected":
                len(selected),
        }

    return (
        sorted(chosen),
        details,
    )


@torch.no_grad()
def audit_raw_vs_cache(
    stage1,
    dataset,
    device,
    num_samples: int,
    tolerance: float,
):
    """
    Critical correctness test:

    raw eventImage
        -> exact Stage-1 preprocessing
        -> official GEP Stage-1
    must reproduce eventToken_vits14.

    Run in FP32.
    """

    stage1.eval()

    raw_batch = []
    cached_batch = []

    frames_checked = 0

    for sample in (
        dataset.samples[
            :num_samples
        ]
    ):
        for event_path in (
            sample["event_paths"]
        ):
            raw_batch.append(
                preprocess_event_image(
                    event_path
                )
            )

            cache_path = (
                event_path.parent.parent
                / "eventToken_vits14"
                / (
                    event_path.stem
                    + ".pt"
                )
            )

            if not cache_path.is_file():
                raise FileNotFoundError(
                    cache_path
                )

            cached = torch.load(
                cache_path,
                map_location="cpu",
                weights_only=True,
            ).float()

            cached_batch.append(
                cached
            )

            frames_checked += 1

    x = torch.stack(
        raw_batch
    ).to(device)

    cached = torch.stack(
        cached_batch
    ).to(device)

    with torch.autocast(
        device_type="cuda",
        enabled=False,
    ):
        produced = (
            stage1
            .forward_features(
                x.float()
            )[
                "x_norm_patchtokens"
            ]
            .float()
        )

    if produced.shape != cached.shape:
        raise RuntimeError(
            f"audit shape mismatch: "
            f"{tuple(produced.shape)} vs "
            f"{tuple(cached.shape)}"
        )

    diff = (
        produced - cached
    ).abs()

    max_abs = float(
        diff.max().item()
    )

    mean_abs = float(
        diff.mean().item()
    )

    print()
    print("=" * 80)
    print(
        "RAW -> STAGE1 -> CACHE AUDIT"
    )
    print("=" * 80)

    print(
        "frames checked :",
        frames_checked,
    )

    print(
        "max abs error  :",
        max_abs,
    )

    print(
        "mean abs error :",
        mean_abs,
    )

    print(
        "tolerance      :",
        tolerance,
    )

    if max_abs > tolerance:
        raise RuntimeError(
            "raw Stage-1 pipeline does "
            "NOT reproduce "
            "eventToken_vits14"
        )

    print(
        "AUDIT: PASS"
    )
    print()


def encode_stage1(
    stage1,
    events,
    trainable: bool,
):
    """
    events: [B,T,3,224,224]
    returns [B,T,256,384]
    """

    B, T, C, H, W = (
        events.shape
    )

    flat = events.reshape(
        B * T,
        C,
        H,
        W,
    )

    if trainable:
        tokens = (
            stage1.forward_features(
                flat
            )[
                "x_norm_patchtokens"
            ]
        )
    else:
        with torch.no_grad():
            # Keep frozen Stage-1 in FP32,
            # matching cached token semantics.
            with torch.autocast(
                device_type="cuda",
                enabled=False,
            ):
                tokens = (
                    stage1.forward_features(
                        flat.float()
                    )[
                        "x_norm_patchtokens"
                    ]
                )

    if tuple(
        tokens.shape[1:]
    ) != (
        256,
        384,
    ):
        raise RuntimeError(
            "bad Stage-1 token shape: "
            f"{tuple(tokens.shape)}"
        )

    return tokens.reshape(
        B,
        T,
        256,
        384,
    )


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--root",
        required=True,
    )

    p.add_argument(
        "--stage1-checkpoint",
        required=True,
    )

    p.add_argument(
        "--stage2-checkpoint",
        required=True,
    )

    p.add_argument(
        "--train-sequences-file",
        required=True,
    )

    p.add_argument(
        "--final-sequences-file",
        required=True,
    )

    p.add_argument(
        "--output-dir",
        required=True,
    )

    p.add_argument(
        "--protocol",
        choices=[
            "ft10",
            "full",
            "stage2-only",
            "stage1-only",
        ],
        required=True,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    p.add_argument(
        "--total-steps",
        type=int,
        default=5000,
    )

    p.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    p.add_argument(
        "--grad-accum",
        type=int,
        default=1,
    )

    p.add_argument(
        "--num-workers",
        type=int,
        default=8,
    )

    # GEP downstream LR ratios:
    # head        = 1e-3
    # Stage-2     = 0.1 * head = 1e-4
    # Stage-1     = 0.01 * head = 1e-5

    p.add_argument(
        "--head-lr",
        type=float,
        default=1e-3,
    )

    p.add_argument(
        "--stage2-lr",
        type=float,
        default=1e-4,
    )

    p.add_argument(
        "--stage1-lr",
        type=float,
        default=1e-5,
    )

    p.add_argument(
        "--precision",
        choices=[
            "fp32",
            "bf16",
        ],
        default="bf16",
    )

    p.add_argument(
        "--log-every",
        type=int,
        default=50,
    )

    p.add_argument(
        "--audit-samples",
        type=int,
        default=2,
    )

    p.add_argument(
        "--audit-tol",
        type=float,
        default=5e-4,
    )

    p.add_argument(
        "--audit-only",
        action="store_true",
    )

    args = p.parse_args()

    if args.grad_accum < 1:
        raise ValueError(
            "--grad-accum must be >= 1"
        )

    if (
        args.warmup_steps < 0
        or args.warmup_steps
        >= args.total_steps
    ):
        raise ValueError(
            "invalid warmup steps"
        )

    seed_everything(
        args.seed
    )

    device = torch.device(
        "cuda"
    )

    root = Path(
        args.root
    )

    train_names = read_names(
        args.train_sequences_file
    )

    final_names = read_names(
        args.final_sequences_file
    )

    # ----------------------------------------------------------
    # Formal split safety:
    # FINAL is NEVER instantiated.
    # ----------------------------------------------------------

    overlap = (
        set(train_names)
        & set(final_names)
    )

    if overlap:
        raise RuntimeError(
            f"FINAL leakage: "
            f"{sorted(overlap)}"
        )

    all_flow_sequences = sorted(
        p.name
        for p in (
            root
            / "train_optical_flow"
        ).iterdir()
        if p.is_dir()
    )

    if (
        set(train_names)
        | set(final_names)
    ) != set(
        all_flow_sequences
    ):
        raise RuntimeError(
            "formal train + FINAL "
            "does not exactly cover "
            "the 18 DSEC flow sequences"
        )

    base_dataset = (
        DSECRawSequenceFlowDataset(
            root=root,
            sequences=train_names,
            context_frames=4,
        )
    )

    print(
        "raw train samples:",
        len(base_dataset),
    )

    if len(base_dataset) != 6056:
        raise RuntimeError(
            "raw dataset does not match "
            "frozen formal Flow-A sample "
            f"count: got {len(base_dataset)}, "
            "expected 6056"
        )

    stage1 = build_stage1(
        args.stage1_checkpoint
    ).to(device)

    stage2, stage2_cfg = (
        build_stage2(
            args.stage2_checkpoint
        )
    )

    stage2 = stage2.to(
        device
    )

    # ----------------------------------------------------------
    # Before ANY fine-tuning, prove that raw path is identical
    # to eventToken_vits14 path.
    # ----------------------------------------------------------

    audit_raw_vs_cache(
        stage1=stage1,
        dataset=base_dataset,
        device=device,
        num_samples=args.audit_samples,
        tolerance=args.audit_tol,
    )

    if args.audit_only:
        print(
            "AUDIT-ONLY COMPLETE"
        )
        return

    # ----------------------------------------------------------
    # Protocol.
    # ----------------------------------------------------------

    stage1_trainable = (
        args.protocol
        in {
            "ft10",
            "full",
            "stage1-only",
        }
    )

    stage2_trainable = (
        args.protocol != "stage1-only"
    )

    if args.protocol == "ft10":
        selected_indices, subset_info = (
            build_ft10_indices(
                base_dataset,
                seed=args.seed,
            )
        )

        train_dataset = Subset(
            base_dataset,
            selected_indices,
        )

        label_fraction = 0.10

    else:
        selected_indices = list(
            range(
                len(base_dataset)
            )
        )

        subset_info = None

        train_dataset = (
            base_dataset
        )

        label_fraction = 1.0

    # Freeze/unfreeze.
    stage1.requires_grad_(
        stage1_trainable
    )

    stage2.requires_grad_(
        stage2_trainable
    )

    if stage1_trainable:
        stage1.train()
    else:
        stage1.eval()

    if stage2_trainable:
        stage2.train()
    else:
        stage2.eval()

    # Identical head initialization
    # across B/C/D with same seed.
    torch.manual_seed(
        args.seed
    )
    torch.cuda.manual_seed_all(
        args.seed
    )

    head = PatchLinearFlowHead(
        embed_dim=384,
        grid_size=16,
        patch_size=14,
    ).to(device)

    stage1_hash_before = (
        module_sha256(
            stage1
        )
    )

    stage2_hash_before = (
        module_sha256(
            stage2
        )
    )

    head_hash_before = (
        module_sha256(
            head
        )
    )

    param_groups = []

    if stage1_trainable:
        param_groups.append(
            {
                "params":
                    stage1.parameters(),

                "lr":
                    args.stage1_lr,

                "weight_decay":
                    0.0,

                "name":
                    "stage1",
            }
        )

    if stage2_trainable:
        param_groups.append(
            {
                "params":
                    stage2.parameters(),

                "lr":
                    args.stage2_lr,

                "weight_decay":
                    0.0,

                "name":
                    "stage2",
            }
        )

    param_groups.append(
        {
            "params":
                head.parameters(),

            "lr":
                args.head_lr,

            "weight_decay":
                0.0,

            "name":
                "head",
        }
    )

    optimizer = torch.optim.AdamW(
        param_groups
    )

    generator = (
        torch.Generator()
    )

    generator.manual_seed(
        args.seed
    )

    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        generator=generator,
        persistent_workers=(
            args.num_workers > 0
        ),
    )

    out = Path(
        args.output_dir
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    subset_manifest = {
        "protocol":
            args.protocol,

        "seed":
            args.seed,

        "label_fraction":
            label_fraction,

        "base_train_samples":
            len(base_dataset),

        "selected_train_samples":
            len(train_dataset),

        "selected_indices":
            selected_indices,

        "per_sequence":
            subset_info,

        "FINAL_sequences_declared_only":
            final_names,
    }

    (
        out
        / "subset_manifest.json"
    ).write_text(
        json.dumps(
            subset_manifest,
            indent=2,
        )
    )

    print()
    print("=" * 80)
    print(
        "FORMAL RAW-EVENT DSEC FLOW"
    )
    print("=" * 80)

    print(
        "protocol              :",
        args.protocol,
    )

    print(
        "GEP Stage-1           :",
        (
            "TRAINABLE"
            if stage1_trainable
            else "FROZEN"
        ),
    )

    print(
        "Event-JEPA Stage-2    :",
        (
            "TRAINABLE"
            if stage2_trainable
            else "BYPASSED / FROZEN"
        ),
    )

    print(
        "flow head             :",
        "TRAINABLE",
    )

    print(
        "label fraction        :",
        label_fraction,
    )

    print(
        "base train samples    :",
        len(base_dataset),
    )

    print(
        "selected samples      :",
        len(train_dataset),
    )

    print(
        "FINAL sequences       :",
        len(final_names),
        "(DECLARED ONLY; NOT LOADED)",
    )

    print(
        "micro batch           :",
        args.batch_size,
    )

    print(
        "grad accum            :",
        args.grad_accum,
    )

    print(
        "effective batch       :",
        args.batch_size
        * args.grad_accum,
    )

    print(
        "optimizer steps       :",
        args.total_steps,
    )

    print(
        "Stage-1 LR            :",
        (
            args.stage1_lr
            if stage1_trainable
            else 0.0
        ),
    )

    print(
        "Stage-2 LR            :",
        (
            args.stage2_lr
            if stage2_trainable
            else 0.0
        ),
    )

    print(
        "head LR               :",
        args.head_lr,
    )

    print(
        "precision             :",
        args.precision,
    )

    train_iter = iter(
        loader
    )

    running_loss = 0.0
    running_count = 0

    start = time.time()

    trainable_params = [
        p
        for group in
        optimizer.param_groups
        for p in group["params"]
        if p.requires_grad
    ]

    use_amp = (
        args.precision
        == "bf16"
    )

    for step in range(
        args.total_steps
    ):
        optimizer.zero_grad(
            set_to_none=True
        )

        micro_loss_sum = 0.0

        for micro in range(
            args.grad_accum
        ):
            try:
                batch = next(
                    train_iter
                )
            except StopIteration:
                train_iter = iter(
                    loader
                )
                batch = next(
                    train_iter
                )

            events = batch[
                "events"
            ].to(
                device,
                non_blocking=True,
            )

            target = batch[
                "flow"
            ].to(
                device,
                non_blocking=True,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                # For B/C:
                # gradient goes raw event
                # -> GEP Stage-1.
                #
                # For D:
                # Stage-1 is frozen FP32.
                tokens = encode_stage1(
                    stage1,
                    events,
                    trainable=(
                        stage1_trainable
                    ),
                )

                if stage2_trainable:
                    memory = stage2(
                        tokens
                    )
                else:
                    memory = None

            if stage2_trainable:

                expected = (
                    events.shape[0],
                    4 * 256,
                    384,
                )

                if tuple(
                    memory.shape
                ) != expected:
                    raise RuntimeError(
                        f"bad Stage-2 shape: "
                        f"{tuple(memory.shape)}, "
                        f"expected {expected}"
                    )

                # Event-JEPA representation
                # for current/latest frame.
                z_last = memory[
                    :,
                    -256:,
                    :,
                ]

            else:

                # Strict Stage1-only control.
                #
                # tokens:
                # [B,4,256,384]
                #
                # Same Tc=4 sample pool is retained,
                # but only the current event frame
                # representation is exposed to the
                # SAME downstream Flow head.
                expected = (
                    events.shape[0],
                    4,
                    256,
                    384,
                )

                if tuple(
                    tokens.shape
                ) != expected:
                    raise RuntimeError(
                        f"bad Stage-1 temporal shape: "
                        f"{tuple(tokens.shape)}, "
                        f"expected {expected}"
                    )

                z_last = tokens[
                    :,
                    -1,
                    :,
                    :,
                ]

            pred = head(
                z_last.float()
            )

            raw_loss = (
                masked_flow_l1(
                    pred,
                    target,
                )
            )

            if not torch.isfinite(
                raw_loss
            ):
                raise RuntimeError(
                    f"non-finite loss "
                    f"at step {step + 1}"
                )

            loss = (
                raw_loss
                / args.grad_accum
            )

            loss.backward()

            micro_loss_sum += float(
                raw_loss.item()
            )

        # LR schedule is defined in
        # optimizer-update steps.
        stage1_lr = get_lr(
            step,
            args.warmup_steps,
            args.stage1_lr,
            args.total_steps,
            0.0,
        )

        stage2_lr = get_lr(
            step,
            args.warmup_steps,
            args.stage2_lr,
            args.total_steps,
            0.0,
        )

        head_lr = get_lr(
            step,
            args.warmup_steps,
            args.head_lr,
            args.total_steps,
            0.0,
        )

        for group in (
            optimizer.param_groups
        ):
            if (
                group["name"]
                == "stage1"
            ):
                group["lr"] = (
                    stage1_lr
                )

            elif (
                group["name"]
                == "stage2"
            ):
                group["lr"] = (
                    stage2_lr
                )

            elif (
                group["name"]
                == "head"
            ):
                group["lr"] = (
                    head_lr
                )

        grad_norm = clip_grad_norm_(
            trainable_params,
            max_norm=1.0,
        )

        optimizer.step()

        step_loss = (
            micro_loss_sum
            / args.grad_accum
        )

        running_loss += (
            step_loss
        )

        running_count += 1

        if (
            step == 0
            or (
                step + 1
            ) % args.log_every == 0
            or (
                step + 1
                == args.total_steps
            )
        ):
            elapsed = (
                time.time()
                - start
            )

            print(
                f"step="
                f"{step + 1:05d}/"
                f"{args.total_steps} "
                f"loss={step_loss:.5f} "
                f"grad={float(grad_norm):.4f} "
                f"s1lr="
                f"{stage1_lr if stage1_trainable else 0.0:.8f} "
                f"s2lr={stage2_lr:.8f} "
                f"headlr={head_lr:.8f} "
                f"time={elapsed:.1f}s"
            )

    # ----------------------------------------------------------
    # Strict post-training state audit.
    # ----------------------------------------------------------

    stage1_hash_after = (
        module_sha256(
            stage1
        )
    )

    stage2_hash_after = (
        module_sha256(
            stage2
        )
    )

    head_hash_after = (
        module_sha256(
            head
        )
    )

    stage1_changed = (
        stage1_hash_before
        != stage1_hash_after
    )

    stage2_changed = (
        stage2_hash_before
        != stage2_hash_after
    )

    head_changed = (
        head_hash_before
        != head_hash_after
    )

    expected_changes = {
        "ft10": {
            "stage1": True,
            "stage2": True,
            "head": True,
        },
        "full": {
            "stage1": True,
            "stage2": True,
            "head": True,
        },
        "stage2-only": {
            "stage1": False,
            "stage2": True,
            "head": True,
        },
        "stage1-only": {
            "stage1": True,
            "stage2": False,
            "head": True,
        },
    }[args.protocol]

    actual_changes = {
        "stage1": stage1_changed,
        "stage2": stage2_changed,
        "head": head_changed,
    }

    if actual_changes != expected_changes:
        raise RuntimeError(
            "parameter audit failed: "
            f"actual={actual_changes}, "
            f"expected={expected_changes}"
        )

    wallclock = (
        time.time()
        - start
    )

    summary = {
        **subset_manifest,

        "source_stage1_checkpoint":
            str(
                Path(
                    args.stage1_checkpoint
                ).resolve()
            ),

        "source_stage2_checkpoint":
            str(
                Path(
                    args.stage2_checkpoint
                ).resolve()
            ),

        "context_frames":
            4,

        "total_steps":
            args.total_steps,

        "warmup_steps":
            args.warmup_steps,

        "micro_batch":
            args.batch_size,

        "grad_accum":
            args.grad_accum,

        "effective_batch":
            args.batch_size
            * args.grad_accum,

        "stage1_lr":
            (
                args.stage1_lr
                if stage1_trainable
                else 0.0
            ),

        "stage2_lr":
            (
                args.stage2_lr
                if stage2_trainable
                else 0.0
            ),

        "head_lr":
            args.head_lr,

        "precision":
            args.precision,

        "train_mean_loss":
            running_loss
            / max(
                running_count,
                1,
            ),

        "stage1_changed":
            stage1_changed,

        "stage2_changed":
            stage2_changed,

        "head_changed":
            head_changed,

        "stage1_sha_before":
            stage1_hash_before,

        "stage1_sha_after":
            stage1_hash_after,

        "stage2_sha_before":
            stage2_hash_before,

        "stage2_sha_after":
            stage2_hash_after,

        "head_sha_before":
            head_hash_before,

        "head_sha_after":
            head_hash_after,

        "FINAL_evaluated":
            False,

        "wallclock_seconds":
            wallclock,
    }

    # Save complete downstream state.
    torch.save(
        {
            "event_encoder":
                stage1.state_dict(),

            "online_encoder":
                stage2.state_dict(),

            "head":
                head.state_dict(),

            "source_stage2_config":
                stage2_cfg,

            "summary":
                summary,
        },
        out
        / "final.pt",
    )

    (
        out
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print()
    print("=" * 80)
    print(
        "FORMAL RAW-EVENT FLOW "
        "TRAIN-ONLY: PASS"
    )
    print("=" * 80)

    print(
        "protocol          :",
        args.protocol,
    )

    print(
        "train samples     :",
        len(train_dataset),
    )

    print(
        "train mean loss   :",
        summary[
            "train_mean_loss"
        ],
    )

    print(
        "Stage-1 changed   :",
        stage1_changed,
    )

    print(
        "Stage-2 changed   :",
        stage2_changed,
    )

    print(
        "head changed      :",
        head_changed,
    )

    print(
        "FINAL evaluated   :",
        False,
    )

    print(
        "wallclock seconds :",
        round(
            wallclock,
            2,
        ),
    )


if __name__ == "__main__":
    main()
