from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    Subset,
)
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from dinov2.models.vision_transformer import vit_small

from event_jepa.encoders import EventTokenEncoder


# ============================================================
# Formal protocol constants
# ============================================================

H = 280
W = 350

PATCH = 14

GRID_H = H // PATCH       # 20
GRID_W = W // PATCH       # 25

D = 384
TC = 4

TILE_H = 16
TILE_W = 16

TRAIN_MIN_DEPTH = 0.1
TRAIN_MAX_DEPTH = 30.0

MVSEC_MEAN = (
    0.8993729784963826,
    0.7969581014619264,
    0.8928228776286392,
)

MVSEC_STD = (
    0.22043360769748688,
    0.2921656668186188,
    0.22049927711486816,
)

EVAL_SEQUENCES = (
    "outdoor_day1",
    "outdoor_night1",
    "outdoor_night2",
    "outdoor_night3",
)


# ============================================================
# Utility
# ============================================================

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def module_sha256(module: nn.Module) -> str:

    h = hashlib.sha256()

    for name, tensor in sorted(
        module.state_dict().items()
    ):
        x = (
            tensor.detach()
            .cpu()
            .contiguous()
        )

        h.update(name.encode())
        h.update(
            str(tuple(x.shape)).encode()
        )
        h.update(
            str(x.dtype).encode()
        )

        # Convert to FP32 for portable hashing.
        h.update(
            x.float()
            .numpy()
            .tobytes()
        )

    return h.hexdigest()


def get_lr(
    step: int,
    warmup_steps: int,
    peak_lr: float,
    decay_steps: int,
):

    if step < warmup_steps:
        return (
            peak_lr
            * step
            / warmup_steps
        )

    ratio = (
        step - warmup_steps
    ) / max(
        1,
        decay_steps - warmup_steps
    )

    ratio = min(
        max(ratio, 0.0),
        1.0,
    )

    coeff = (
        0.5
        * (
            1.0
            + math.cos(
                math.pi * ratio
            )
        )
    )

    return peak_lr * coeff


# ============================================================
# Dataset
# ============================================================

def read_sequence_index(
    sequence_root: Path,
):

    csv_path = (
        sequence_root
        / "index.csv"
    )

    if not csv_path.is_file():
        raise FileNotFoundError(
            csv_path
        )

    with csv_path.open(
        "r",
        newline="",
    ) as f:

        rows = list(
            csv.DictReader(f)
        )

    valid = []

    for row in rows:

        event_path = (
            sequence_root
            / row["event_path"]
        )

        depth_path = (
            sequence_root
            / row["depth_path"]
        )

        if (
            event_path.is_file()
            and depth_path.is_file()
        ):

            r = dict(row)

            r["_event_path"] = str(
                event_path
            )

            r["_depth_path"] = str(
                depth_path
            )

            valid.append(r)

    if len(valid) < TC:
        raise RuntimeError(
            f"{sequence_root.name}: "
            f"only {len(valid)} valid rows"
        )

    return valid


class MVSECTemporalDepthDataset(
    Dataset
):

    def __init__(
        self,
        root: Path,
        sequences,
        *,
        train: bool,
    ):

        super().__init__()

        self.root = Path(root)
        self.train = bool(train)

        self.samples = []
        self.sequence_names = []

        for sequence in sequences:

            sequence_root = (
                self.root
                / sequence
            )

            if not sequence_root.is_dir():
                raise FileNotFoundError(
                    sequence_root
                )

            rows = read_sequence_index(
                sequence_root
            )

            # strictly causal:
            # [t-3,t-2,t-1,t] -> depth(t)
            for i in range(
                TC - 1,
                len(rows),
            ):

                history = rows[
                    i - TC + 1:
                    i + 1
                ]

                sample = {
                    "sequence":
                        sequence,

                    "events":
                        [
                            x["_event_path"]
                            for x
                            in history
                        ],

                    "depth":
                        rows[i][
                            "_depth_path"
                        ],

                    "frame_id":
                        rows[i].get(
                            "frame_id",
                            str(i),
                        ),
                }

                self.samples.append(
                    sample
                )

                self.sequence_names.append(
                    sequence
                )

        if not self.samples:
            raise RuntimeError(
                "No temporal MVSEC "
                "depth samples found"
            )

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def load_event(
        path: str,
    ):

        image = (
            Image.open(path)
            .convert("RGB")
        )

        x = np.array(
            image,
            copy=True,
        )

        return (
            torch.from_numpy(x)
            .permute(2, 0, 1)
            .float()
            / 255.0
        )

    def train_transform(
        self,
        events,
        depth,
        mask,
    ):

        # ---------------------------------
        # SAME crop for all 4 frames
        # and target depth.
        # ---------------------------------

        _, _, h0, w0 = (
            events.shape
        )

        area = h0 * w0

        log_ratio = (
            math.log(0.9),
            math.log(1.1),
        )

        crop = None

        for _ in range(10):

            target_area = (
                area
                * random.uniform(
                    0.5,
                    1.0,
                )
            )

            aspect = math.exp(
                random.uniform(
                    *log_ratio
                )
            )

            crop_w = int(
                round(
                    math.sqrt(
                        target_area
                        * aspect
                    )
                )
            )

            crop_h = int(
                round(
                    math.sqrt(
                        target_area
                        / aspect
                    )
                )
            )

            if (
                0 < crop_w <= w0
                and
                0 < crop_h <= h0
            ):

                top = random.randint(
                    0,
                    h0 - crop_h,
                )

                left = random.randint(
                    0,
                    w0 - crop_w,
                )

                crop = (
                    top,
                    left,
                    crop_h,
                    crop_w,
                )

                break

        if crop is None:

            in_ratio = (
                float(w0)
                / float(h0)
            )

            if in_ratio < 0.9:

                crop_w = w0
                crop_h = int(
                    round(
                        crop_w
                        / 0.9
                    )
                )

            elif in_ratio > 1.1:

                crop_h = h0
                crop_w = int(
                    round(
                        crop_h
                        * 1.1
                    )
                )

            else:

                crop_h = h0
                crop_w = w0

            top = (
                h0 - crop_h
            ) // 2

            left = (
                w0 - crop_w
            ) // 2

            crop = (
                top,
                left,
                crop_h,
                crop_w,
            )

        (
            top,
            left,
            crop_h,
            crop_w,
        ) = crop

        # Event history:
        # [T,C,H,W] -> [T*C,H,W]
        flat = events.reshape(
            TC * 3,
            h0,
            w0,
        )

        flat = TF.resized_crop(
            flat,
            top,
            left,
            crop_h,
            crop_w,
            (H, W),
            interpolation=
                InterpolationMode.BILINEAR,
            antialias=True,
        )

        events = flat.reshape(
            TC,
            3,
            H,
            W,
        )

        depth = TF.resized_crop(
            depth.unsqueeze(0),
            top,
            left,
            crop_h,
            crop_w,
            (H, W),
            interpolation=
                InterpolationMode.BILINEAR,
            antialias=True,
        ).squeeze(0)

        mask = TF.resized_crop(
            mask.unsqueeze(0),
            top,
            left,
            crop_h,
            crop_w,
            (H, W),
            interpolation=
                InterpolationMode.NEAREST,
            antialias=False,
        ).squeeze(0)

        # horizontal flip
        if random.random() < 0.5:

            events = torch.flip(
                events,
                dims=[-1],
            )

            depth = torch.flip(
                depth,
                dims=[-1],
            )

            mask = torch.flip(
                mask,
                dims=[-1],
            )

        # GEP depth event-noise
        # augmentation.
        if random.random() < 0.5:

            noise = (
                torch.randn_like(
                    events
                )
                * 0.03
            )

            events = (
                events + noise
            ).clamp(
                0.0,
                1.0,
            )

        return (
            events,
            depth,
            mask,
        )

    @staticmethod
    def eval_transform(
        events,
        depth,
        mask,
    ):

        T, C, h0, w0 = (
            events.shape
        )

        flat = events.reshape(
            T * C,
            h0,
            w0,
        )

        flat = TF.resize(
            flat,
            (H, W),
            interpolation=
                InterpolationMode.BILINEAR,
            antialias=True,
        )

        events = flat.reshape(
            T,
            C,
            H,
            W,
        )

        depth = TF.resize(
            depth.unsqueeze(0),
            (H, W),
            interpolation=
                InterpolationMode.BILINEAR,
            antialias=True,
        ).squeeze(0)

        mask = TF.resize(
            mask.unsqueeze(0),
            (H, W),
            interpolation=
                InterpolationMode.NEAREST,
            antialias=False,
        ).squeeze(0)

        return (
            events,
            depth,
            mask,
        )

    def __getitem__(
        self,
        idx,
    ):

        sample = self.samples[idx]

        events = torch.stack(
            [
                self.load_event(p)
                for p
                in sample["events"]
            ],
            dim=0,
        )

        depth_raw = np.asarray(
            Image.open(
                sample["depth"]
            ),
            dtype=np.float32,
        )

        # pre_mvs:
        # meters * 100 -> uint16
        # therefore read-back:
        # uint16 * 0.01 -> meters
        depth = torch.from_numpy(
            depth_raw
            * 0.01
        ).float()

        mask = torch.from_numpy(
            (
                depth_raw > 0
            ).astype(
                np.float32
            )
        )

        if self.train:

            (
                events,
                depth,
                mask,
            ) = self.train_transform(
                events,
                depth,
                mask,
            )

            # Training supervision range.
            valid = (
                (mask > 0.5)
                &
                torch.isfinite(depth)
                &
                (
                    depth
                    >= TRAIN_MIN_DEPTH
                )
                &
                (
                    depth
                    <= TRAIN_MAX_DEPTH
                )
            )

        else:

            (
                events,
                depth,
                mask,
            ) = self.eval_transform(
                events,
                depth,
                mask,
            )

            # IMPORTANT:
            # GEP paper evaluates WITHOUT
            # limiting maximum GT depth.
            valid = (
                (mask > 0.5)
                &
                torch.isfinite(depth)
                &
                (
                    depth
                    >= TRAIN_MIN_DEPTH
                )
            )

        mask = valid.float()

        depth = torch.where(
            valid,
            depth,
            torch.zeros_like(
                depth
            ),
        )

        # Released GEP MVSEC stats.
        mean = torch.tensor(
            MVSEC_MEAN,
            dtype=events.dtype,
        ).view(
            1,
            3,
            1,
            1,
        )

        std = torch.tensor(
            MVSEC_STD,
            dtype=events.dtype,
        ).view(
            1,
            3,
            1,
            1,
        )

        events = (
            events - mean
        ) / std

        return {
            "events":
                events.contiguous(),

            "depth":
                depth.contiguous(),

            "mask":
                mask.contiguous(),

            "sequence":
                sample["sequence"],

            "frame_id":
                str(
                    sample[
                        "frame_id"
                    ]
                ),
        }


# ============================================================
# GEP depth loss
# ============================================================

def masked_mean(
    x,
    mask,
    eps=1e-6,
):

    if x.shape != mask.shape:
        mask = mask.expand_as(x)

    denom = mask.sum()

    if denom <= eps:
        return x.new_zeros(())

    return (
        x * mask
    ).sum() / denom


def silog_loss(
    pred,
    target,
    mask,
    lam=0.85,
    eps=1e-6,
):

    valid = mask > 0.5

    if not valid.any():
        return pred.new_zeros(())

    d = (
        torch.log(
            pred[valid] + eps
        )
        -
        torch.log(
            target[valid] + eps
        )
    )

    variance = (
        (d ** 2).mean()
        -
        lam
        * (d.mean() ** 2)
    )

    return torch.sqrt(
        variance.clamp_min(
            0.0
        )
    )


def gradient_loss(
    pred,
    target,
    mask,
):

    pred_dx = (
        pred[..., :, 1:]
        -
        pred[..., :, :-1]
    )

    pred_dy = (
        pred[..., 1:, :]
        -
        pred[..., :-1, :]
    )

    target_dx = (
        target[..., :, 1:]
        -
        target[..., :, :-1]
    )

    target_dy = (
        target[..., 1:, :]
        -
        target[..., :-1, :]
    )

    mask_dx = (
        mask[..., :, 1:]
        *
        mask[..., :, :-1]
    )

    mask_dy = (
        mask[..., 1:, :]
        *
        mask[..., :-1, :]
    )

    return (
        masked_mean(
            (
                pred_dx
                - target_dx
            ).abs(),
            mask_dx,
        )
        +
        masked_mean(
            (
                pred_dy
                - target_dy
            ).abs(),
            mask_dy,
        )
    )


def multi_scale_si_gradient_loss(
    pred,
    target,
    mask,
    scales=(1, 2, 4, 8),
):

    total = pred.new_zeros(())
    count = 0

    for scale in scales:

        if scale == 1:

            p = pred
            t = target
            m = mask

        else:

            p = F.avg_pool2d(
                pred.unsqueeze(1),
                scale,
                scale,
            ).squeeze(1)

            t = F.avg_pool2d(
                target.unsqueeze(1),
                scale,
                scale,
            ).squeeze(1)

            m = F.avg_pool2d(
                mask.unsqueeze(1),
                scale,
                scale,
            ).squeeze(1)

            m = (
                m > 0.5
            ).float()

        if m.sum() <= 0:
            continue

        lp = torch.log(
            p.clamp_min(1e-6)
        )

        lt = torch.log(
            t.clamp_min(1e-6)
        )

        total = (
            total
            +
            gradient_loss(
                lp,
                lt,
                m,
            )
        )

        count += 1

    if count == 0:
        return pred.new_zeros(())

    return total / count


def depth_loss(
    pred,
    target,
    mask,
):

    pred = pred.squeeze(1)

    silog = silog_loss(
        pred,
        target,
        mask,
    )

    msgrad = (
        multi_scale_si_gradient_loss(
            pred,
            target,
            mask,
        )
    )

    total = (
        silog
        +
        0.25 * msgrad
    )

    return (
        total,
        silog,
        msgrad,
    )


# ============================================================
# GEP-style depth decoder
# ============================================================

class DecoderBlock(nn.Module):

    def __init__(
        self,
        in_ch,
        out_ch,
    ):

        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                3,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(
                out_ch,
                out_ch,
                3,
                padding=1,
            ),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class MultiScaleDepthDecoder(
    nn.Module
):

    def __init__(
        self,
        stage_dims=(
            256,
            128,
            64,
        ),
    ):

        super().__init__()

        self.stage_dims = (
            stage_dims
        )

        self.proj = nn.Linear(
            D,
            stage_dims[0],
        )

        stages = []

        for idx, out_dim in enumerate(
            stage_dims
        ):

            in_dim = (
                stage_dims[0]
                if idx == 0
                else
                stage_dims[idx - 1]
            )

            stages.append(
                DecoderBlock(
                    in_dim,
                    out_dim,
                )
            )

        self.stages = nn.ModuleList(
            stages
        )

        fusion_in = sum(
            stage_dims
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(
                fusion_in,
                128,
                3,
                padding=1,
            ),
            nn.GELU(),

            nn.Conv2d(
                128,
                64,
                3,
                padding=1,
            ),
            nn.GELU(),

            nn.Conv2d(
                64,
                1,
                1,
            ),
        )

    def forward(
        self,
        tokens,
    ):

        B, N, _ = tokens.shape

        if N != GRID_H * GRID_W:
            raise RuntimeError(
                f"Depth decoder "
                f"expected 500 tokens, "
                f"got {N}"
            )

        x = self.proj(
            tokens
        )

        x = (
            x.transpose(1, 2)
            .reshape(
                B,
                self.stage_dims[0],
                GRID_H,
                GRID_W,
            )
        )

        targets = [
            (
                GRID_H,
                GRID_W,
            ),
            (
                GRID_H * 2,
                GRID_W * 2,
            ),
            (
                H,
                W,
            ),
        ]

        features = []

        for stage, target_hw in zip(
            self.stages,
            targets,
        ):

            x = stage(x)

            if (
                x.shape[-2:]
                != target_hw
            ):

                x = F.interpolate(
                    x,
                    size=target_hw,
                    mode="bilinear",
                    align_corners=False,
                )

            features.append(x)

        fused = torch.cat(
            [
                F.interpolate(
                    feature,
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                )
                if
                feature.shape[-2:]
                != (H, W)
                else feature

                for feature
                in features
            ],
            dim=1,
        )

        # Return raw logits. Depth conversion is performed exactly once
        # in EventJEPADepth.forward().
        return self.fusion(fused)


# ============================================================
# Stage-1
# ============================================================

def build_stage1(
    checkpoint: Path,
):

    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    if (
        isinstance(payload, dict)
        and
        "event_encoder"
        in payload
    ):
        state = payload[
            "event_encoder"
        ]

    else:
        state = payload

    model = vit_small(
        patch_size=14,
        img_size=518,
        block_chunks=0,
        init_values=1e-6,
        num_register_tokens=0,
    )

    result = model.load_state_dict(
        state,
        strict=True,
    )

    if (
        result.missing_keys
        or
        result.unexpected_keys
    ):
        raise RuntimeError(
            "Stage-1 strict load failed"
        )

    return model


def encode_stage1(
    stage1,
    events,
    *,
    trainable,
):

    # events:
    # [B,4,3,280,350]

    B, T, C, HH, WW = (
        events.shape
    )

    flat = events.reshape(
        B * T,
        C,
        HH,
        WW,
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

            tokens = (
                stage1.forward_features(
                    flat
                )[
                    "x_norm_patchtokens"
                ]
            )

    expected = (
        GRID_H * GRID_W,
        D,
    )

    if (
        tuple(
            tokens.shape[1:]
        )
        != expected
    ):
        raise RuntimeError(
            "Unexpected Stage-1 "
            f"shape {tokens.shape}; "
            f"expected [*,500,384]"
        )

    return tokens.reshape(
        B,
        T,
        GRID_H * GRID_W,
        D,
    )


# ============================================================
# Event-JEPA Stage-2
# ============================================================

def build_stage2(
    checkpoint: Path,
):

    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    if "online_encoder" not in payload:
        raise KeyError(
            "Raw Event-JEPA "
            "checkpoint required"
        )

    cfg = dict(
        payload.get(
            "config",
            {},
        )
    )

    required = [
        "embed_dim",
        "num_heads",
        "encoder_layers",
        "n_tokens",
        "max_positions",
        "max_context_frames",
    ]

    missing = [
        k
        for k in required
        if k not in cfg
    ]

    if missing:
        raise KeyError(
            "Stage-2 config "
            f"missing {missing}"
        )

    model = EventTokenEncoder(
        embed_dim=int(
            cfg["embed_dim"]
        ),
        num_heads=int(
            cfg["num_heads"]
        ),
        num_layers=int(
            cfg[
                "encoder_layers"
            ]
        ),
        n_tokens=int(
            cfg["n_tokens"]
        ),
        max_positions=int(
            cfg[
                "max_positions"
            ]
        ),
        max_context_frames=int(
            cfg[
                "max_context_frames"
            ]
        ),
    )

    model.load_state_dict(
        payload[
            "online_encoder"
        ],
        strict=True,
    )

    if (
        model.n_tokens != 256
        or
        model.embed_dim != 384
    ):
        raise RuntimeError(
            "Formal Stage-2 "
            "must be N=256 D=384"
        )

    return model


def covering_starts(
    length,
    window,
):

    if length < window:
        raise ValueError(
            (
                length,
                window,
            )
        )

    if length == window:
        return [0]

    n_windows = int(
        math.ceil(
            length / window
        )
    )

    max_start = (
        length - window
    )

    starts = [
        int(
            round(
                i
                * max_start
                / (
                    n_windows - 1
                )
            )
        )
        for i
        in range(
            n_windows
        )
    ]

    return list(
        dict.fromkeys(
            starts
        )
    )


def forward_stage2_tiled(
    stage2,
    tokens,
):

    # tokens:
    # [B,4,500,384]
    #
    # grid:
    # [B,4,20,25,384]
    #
    # pretrained Stage-2:
    # [*,4,16,16,384]

    B, T, N, DD = (
        tokens.shape
    )

    if (
        T != TC
        or
        N != GRID_H * GRID_W
        or
        DD != D
    ):
        raise RuntimeError(
            "Invalid Event-JEPA "
            f"input {tokens.shape}"
        )

    grid = tokens.reshape(
        B,
        T,
        GRID_H,
        GRID_W,
        D,
    )

    y_starts = covering_starts(
        GRID_H,
        TILE_H,
    )

    x_starts = covering_starts(
        GRID_W,
        TILE_W,
    )

    tiles = []
    coordinates = []

    for y0 in y_starts:

        for x0 in x_starts:

            tile = grid[
                :,
                :,
                y0:
                    y0
                    + TILE_H,
                x0:
                    x0
                    + TILE_W,
                :,
            ]

            tile = tile.reshape(
                B,
                T,
                256,
                D,
            )

            tiles.append(
                tile
            )

            coordinates.append(
                (
                    y0,
                    x0,
                )
            )

    tiled = torch.stack(
        tiles,
        dim=1,
    )

    K = tiled.shape[1]

    # [B,K,T,256,D]
    # →
    # [B*K,T,256,D]

    tiled = tiled.reshape(
        B * K,
        T,
        256,
        D,
    )

    contextualized = stage2(
        tiled
    )

    # [B*K,T*256,D]
    contextualized = (
        contextualized.reshape(
            B,
            K,
            T,
            256,
            D,
        )
        [
            :,
            :,
            -1,
        ]
    )

    # Only current frame is
    # transferred to depth head.

    canvas = (
        contextualized.new_zeros(
            (
                B,
                D,
                GRID_H,
                GRID_W,
            )
        )
    )

    weight = (
        contextualized.new_zeros(
            (
                1,
                1,
                GRID_H,
                GRID_W,
            )
        )
    )

    ones = (
        contextualized.new_ones(
            (
                1,
                1,
                TILE_H,
                TILE_W,
            )
        )
    )

    for k, (
        y0,
        x0,
    ) in enumerate(
        coordinates
    ):

        tile = (
            contextualized[
                :,
                k,
            ]
            .transpose(
                1,
                2,
            )
            .reshape(
                B,
                D,
                TILE_H,
                TILE_W,
            )
        )

        pad = (
            x0,
            GRID_W
                - x0
                - TILE_W,

            y0,
            GRID_H
                - y0
                - TILE_H,
        )

        canvas = (
            canvas
            +
            F.pad(
                tile,
                pad,
            )
        )

        weight = (
            weight
            +
            F.pad(
                ones,
                pad,
            )
        )

    if torch.any(
        weight == 0
    ):
        raise RuntimeError(
            "Uncovered positions "
            "in Stage-2 tiling"
        )

    output = (
        canvas / weight
    )

    output = (
        output.flatten(2)
        .transpose(1, 2)
    )

    # [B,500,384]
    return output


# ============================================================
# Complete Event-JEPA Depth model
# ============================================================

class EventJEPADepth(
    nn.Module
):

    def __init__(
        self,
        stage1_checkpoint,
        stage2_checkpoint,
        protocol,
    ):

        super().__init__()

        self.protocol = protocol

        self.stage1 = build_stage1(
            stage1_checkpoint
        )

        self.stage2 = build_stage2(
            stage2_checkpoint
        )

        self.head = (
            MultiScaleDepthDecoder()
        )

        if protocol == "frozen":

            self.stage1.requires_grad_(
                False
            )

            self.stage2.requires_grad_(
                False
            )

        elif protocol in {
            "ft10",
            "full",
        }:

            pass

        elif (
            protocol
            == "stage2-only"
        ):

            self.stage1.requires_grad_(
                False
            )

        elif (
            protocol
            == "stage1-only"
        ):

            # Control:
            # GEP Stage-1 + depth head.
            # Event-JEPA Stage-2 is loaded only for checkpoint
            # consistency but is frozen and completely bypassed.
            self.stage2.requires_grad_(
                False
            )

        else:
            raise ValueError(
                protocol
            )

    def train(
        self,
        mode=True,
    ):

        super().train(mode)

        if self.protocol in {
            "frozen",
            "stage2-only",
        }:
            self.stage1.eval()

        if self.protocol in {
            "frozen",
            "stage1-only",
        }:
            self.stage2.eval()

        return self

    def forward(
        self,
        events,
    ):

        stage1_trainable = any(
            p.requires_grad
            for p
            in self.stage1.parameters()
        )

        stage2_trainable = any(
            p.requires_grad
            for p
            in self.stage2.parameters()
        )

        z1 = encode_stage1(
            self.stage1,
            events,
            trainable=
                stage1_trainable,
        )

        if self.protocol == "stage1-only":

            # Current-frame Stage-1 representation only.
            # No Stage-2 forward, no temporal mixing.
            z2 = z1[:, -1]

        elif stage2_trainable:

            z2 = (
                forward_stage2_tiled(
                    self.stage2,
                    z1,
                )
            )

        else:

            with torch.no_grad():

                z2 = (
                    forward_stage2_tiled(
                        self.stage2,
                        z1,
                    )
                )

        norm_log_depth = (
            torch.sigmoid(
                self.head(
                    z2
                )
            )
        )

        log_min = math.log(
            TRAIN_MIN_DEPTH
        )

        log_max = math.log(
            TRAIN_MAX_DEPTH
        )

        prediction = torch.exp(
            norm_log_depth
            * (
                log_max
                - log_min
            )
            +
            log_min
        )

        return prediction


# ============================================================
# Optimizer
# ============================================================

def build_optimizer(
    model,
):

    groups = []

    stage1_params = [
        p
        for p
        in model.stage1.parameters()
        if p.requires_grad
    ]

    stage2_params = [
        p
        for p
        in model.stage2.parameters()
        if p.requires_grad
    ]

    head_params = [
        p
        for p
        in model.head.parameters()
        if p.requires_grad
    ]

    if stage1_params:

        groups.append(
            {
                "params":
                    stage1_params,

                "lr_mult":
                    0.01,

                "weight_decay":
                    0.0,
            }
        )

    if stage2_params:

        groups.append(
            {
                "params":
                    stage2_params,

                "lr_mult":
                    0.05,

                "weight_decay":
                    0.0,
            }
        )

    groups.append(
        {
            "params":
                head_params,

            "lr_mult":
                1.0,

            "weight_decay":
                1e-5,
        }
    )

    return torch.optim.AdamW(
        groups,
        lr=5e-4,
    )


# ============================================================
# 10% labels
# ============================================================

def make_10pct_subset(
    dataset,
    seed,
):

    n = len(dataset)

    k = max(
        1,
        int(
            round(
                n * 0.10
            )
        ),
    )

    rng = np.random.default_rng(
        seed
    )

    indices = (
        rng.choice(
            n,
            size=k,
            replace=False,
        )
    )

    indices = sorted(
        int(x)
        for x
        in indices
    )

    return Subset(
        dataset,
        indices,
    )


# ============================================================
# Global-pixel MVSEC metrics
# ============================================================

@torch.inference_mode()
def evaluate(
    model,
    loader,
    device,
):

    model.eval()

    abs_sum = 0.0
    squared_sum = 0.0
    log_squared_sum = 0.0

    delta1_count = 0
    delta2_count = 0
    delta3_count = 0

    valid_count = 0

    for batch in loader:

        events = (
            batch["events"]
            .to(
                device,
                non_blocking=True,
            )
        )

        target = (
            batch["depth"]
            .to(
                device,
                non_blocking=True,
            )
        )

        mask = (
            batch["mask"]
            .to(
                device,
                non_blocking=True,
            )
            > 0.5
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):

            prediction = model(
                events
            ).squeeze(1)

        valid = (
            mask
            &
            torch.isfinite(
                prediction
            )
            &
            torch.isfinite(
                target
            )
            &
            (
                target
                >= TRAIN_MIN_DEPTH
            )
        )

        if not valid.any():
            continue

        # Prediction naturally lies
        # in 0.1..30 m.
        #
        # IMPORTANT:
        # Target is NOT clamped at 30 m
        # during formal evaluation.

        p = (
            prediction[
                valid
            ]
            .float()
            .clamp_min(
                TRAIN_MIN_DEPTH
            )
        )

        t = (
            target[
                valid
            ]
            .float()
        )

        error = p - t

        abs_sum += float(
            error.abs()
            .double()
            .sum()
            .item()
        )

        squared_sum += float(
            error.double()
            .square()
            .sum()
            .item()
        )

        log_squared_sum += float(
            (
                torch.log1p(
                    p.double()
                )
                -
                torch.log1p(
                    t.double()
                )
            )
            .square()
            .sum()
            .item()
        )

        ratio = torch.maximum(
            p
            /
            t.clamp_min(
                1e-6
            ),

            t
            /
            p.clamp_min(
                1e-6
            ),
        )

        delta1_count += int(
            (
                ratio < 1.25
            ).sum().item()
        )

        delta2_count += int(
            (
                ratio
                < 1.25 ** 2
            ).sum().item()
        )

        delta3_count += int(
            (
                ratio
                < 1.25 ** 3
            ).sum().item()
        )

        valid_count += int(
            valid.sum().item()
        )

    if valid_count == 0:
        raise RuntimeError(
            "No valid depth pixels"
        )

    return {
        "abs":
            abs_sum
            / valid_count,

        "rms":
            math.sqrt(
                squared_sum
                / valid_count
            ),

        "rmse_log":
            math.sqrt(
                log_squared_sum
                / valid_count
            ),

        "delta1":
            delta1_count
            / valid_count,

        "delta2":
            delta2_count
            / valid_count,

        "delta3":
            delta3_count
            / valid_count,

        "valid_pixels":
            valid_count,
    }


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--stage1",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--stage2",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--protocol",
        required=True,
        choices=[
            "frozen",
            "ft10",
            "full",
            "stage2-only",
            "stage1-only",
        ],
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--schedule-steps",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--micro-batch",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--grad-accum",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    args = parser.parse_args()

    effective_batch = (
        args.micro_batch
        * args.grad_accum
    )

    if effective_batch != 8:
        raise ValueError(
            "Formal effective batch "
            "must remain 8"
        )

    if (
        args.schedule_steps
        < args.steps
    ):
        raise ValueError(
            "schedule-steps must "
            "be >= run steps"
        )

    seed_everything(
        args.seed
    )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
    )

    # ------------------------------------
    # datasets
    # ------------------------------------

    train_full = (
        MVSECTemporalDepthDataset(
            args.root,
            [
                "outdoor_day2"
            ],
            train=True,
        )
    )

    if args.protocol == "ft10":

        train_dataset = (
            make_10pct_subset(
                train_full,
                args.seed,
            )
        )

        label_fraction = 0.10

    else:

        train_dataset = (
            train_full
        )

        label_fraction = 1.00

    eval_datasets = {
        sequence:
            MVSECTemporalDepthDataset(
                args.root,
                [sequence],
                train=False,
            )

        for sequence
        in EVAL_SEQUENCES
    }

    generator = (
        torch.Generator()
    )

    generator.manual_seed(
        args.seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=
            args.micro_batch,
        shuffle=True,
        generator=generator,
        num_workers=
            args.workers,
        pin_memory=True,
        persistent_workers=
            args.workers > 0,
        drop_last=True,
    )

    # ------------------------------------
    # model
    # ------------------------------------

    model = EventJEPADepth(
        args.stage1,
        args.stage2,
        args.protocol,
    ).to(
        device
    )

    optimizer = (
        build_optimizer(
            model
        )
    )

    trainable = {
        "stage1":
            sum(
                p.numel()
                for p
                in model.stage1.parameters()
                if p.requires_grad
            ),

        "stage2":
            sum(
                p.numel()
                for p
                in model.stage2.parameters()
                if p.requires_grad
            ),

        "head":
            sum(
                p.numel()
                for p
                in model.head.parameters()
                if p.requires_grad
            ),
    }

    print()
    print("=" * 100)
    print(
        "EVENT-JEPA MVSEC DEPTH — FORMAL"
    )
    print("=" * 100)

    print(
        "protocol            :",
        args.protocol,
    )

    print(
        "train sequence      :",
        "outdoor_day2",
    )

    print(
        "base train windows  :",
        len(train_full),
    )

    print(
        "selected windows    :",
        len(train_dataset),
    )

    print(
        "label fraction      :",
        label_fraction,
    )

    print(
        "context frames      :",
        TC,
    )

    print(
        "input               :",
        f"{H}x{W}",
    )

    print(
        "Stage-1 grid        :",
        f"{GRID_H}x{GRID_W}"
        f"={GRID_H*GRID_W}",
    )

    print(
        "Stage-2 tile        :",
        "16x16=256",
    )

    print(
        "tile starts y       :",
        covering_starts(
            GRID_H,
            TILE_H,
        ),
    )

    print(
        "tile starts x       :",
        covering_starts(
            GRID_W,
            TILE_W,
        ),
    )

    print(
        "micro batch         :",
        args.micro_batch,
    )

    print(
        "grad accum          :",
        args.grad_accum,
    )

    print(
        "effective batch     :",
        effective_batch,
    )

    print(
        "optimizer steps     :",
        args.steps,
    )

    print(
        "lr schedule steps   :",
        args.schedule_steps,
    )

    print(
        "peak LR head        :",
        "5e-4",
    )

    print(
        "peak LR Stage-2     :",
        "2.5e-5",
    )

    print(
        "peak LR Stage-1     :",
        "5e-6",
    )

    print(
        "trainable params    :",
        trainable,
    )

    print(
        "FINAL sequences     :",
        EVAL_SEQUENCES,
    )

    print(
        "periodic eval       :",
        "DISABLED",
    )

    print("=" * 100)

    # ------------------------------------
    # parameter integrity
    # ------------------------------------

    before = {
        "stage1":
            module_sha256(
                model.stage1
            ),

        "stage2":
            module_sha256(
                model.stage2
            ),

        "head":
            module_sha256(
                model.head
            ),
    }

    # ------------------------------------
    # train
    # ------------------------------------

    train_iter = iter(
        train_loader
    )

    wall_start = time.time()

    for optimizer_step in range(
        args.steps
    ):

        current_lr = get_lr(
            optimizer_step,
            100,
            5e-4,
            args.schedule_steps,
        )

        for group in (
            optimizer.param_groups
        ):

            group["lr"] = (
                current_lr
                * group.get(
                    "lr_mult",
                    1.0,
                )
            )

        optimizer.zero_grad(
            set_to_none=True
        )

        total_log = 0.0
        silog_log = 0.0
        msgrad_log = 0.0

        for micro in range(
            args.grad_accum
        ):

            try:

                batch = next(
                    train_iter
                )

            except StopIteration:

                train_iter = iter(
                    train_loader
                )

                batch = next(
                    train_iter
                )

            events = (
                batch["events"]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            depth = (
                batch["depth"]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            mask = (
                batch["mask"]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            model.train()

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):

                prediction = model(
                    events
                )

                (
                    loss,
                    silog,
                    msgrad,
                ) = depth_loss(
                    prediction,
                    depth,
                    mask,
                )

                backward_loss = (
                    loss
                    / args.grad_accum
                )

            backward_loss.backward()

            total_log += float(
                loss.detach()
            )

            silog_log += float(
                silog.detach()
            )

            msgrad_log += float(
                msgrad.detach()
            )

        params = [
            p
            for p
            in model.parameters()
            if p.requires_grad
        ]

        grad_norm = (
            torch.nn.utils
            .clip_grad_norm_(
                params,
                1.0,
            )
        )

        optimizer.step()

        step_number = (
            optimizer_step + 1
        )

        if (
            step_number == 1
            or
            step_number
            % args.log_every
            == 0
            or
            step_number
            == args.steps
        ):

            elapsed = (
                time.time()
                - wall_start
            )

            print(
                f"step "
                f"{step_number:6d}/"
                f"{args.steps} | "
                f"lr "
                f"{current_lr:.6e} | "
                f"loss "
                f"{total_log/args.grad_accum:.5f} | "
                f"silog "
                f"{silog_log/args.grad_accum:.5f} | "
                f"msgrad "
                f"{msgrad_log/args.grad_accum:.5f} | "
                f"grad "
                f"{float(grad_norm):.4f} | "
                f"elapsed "
                f"{elapsed:.1f}s"
            )

    # ------------------------------------
    # audit
    # ------------------------------------

    after = {
        "stage1":
            module_sha256(
                model.stage1
            ),

        "stage2":
            module_sha256(
                model.stage2
            ),

        "head":
            module_sha256(
                model.head
            ),
    }

    changed = {
        key:
            before[key]
            != after[key]

        for key
        in before
    }

    expected = {
        "frozen": {
            "stage1": False,
            "stage2": False,
            "head": True,
        },

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
    }[
        args.protocol
    ]

    if changed != expected:
        raise RuntimeError(
            "Parameter audit failed: "
            f"got={changed}, "
            f"expected={expected}"
        )

    print()
    print(
        "PARAMETER AUDIT:",
        changed,
    )

    # ------------------------------------
    # save before FINAL evaluation
    # ------------------------------------

    checkpoint = {
        "protocol":
            args.protocol,

        "steps":
            args.steps,

        "schedule_steps":
            args.schedule_steps,

        "seed":
            args.seed,

        "base_train_windows":
            len(train_full),

        "selected_train_windows":
            len(train_dataset),

        "label_fraction":
            label_fraction,

        "parameter_changed":
            changed,

        "stage1":
            model.stage1.state_dict(),

        "stage2":
            model.stage2.state_dict(),

        "head":
            model.head.state_dict(),
    }

    checkpoint_path = (
        args.output
        /
        (
            "smoke.pt"
            if args.smoke
            else
            "final.pt"
        )
    )

    torch.save(
        checkpoint,
        checkpoint_path,
    )

    print(
        "checkpoint:",
        checkpoint_path,
    )

    if args.smoke:

        print()
        print(
            "MVSEC DEPTH SMOKE: PASS"
        )

        return

    # ------------------------------------
    # ONE FINAL EVALUATION
    # ------------------------------------

    results = {}

    for sequence in (
        EVAL_SEQUENCES
    ):

        dataset = (
            eval_datasets[
                sequence
            ]
        )

        loader = DataLoader(
            dataset,
            batch_size=
                args.micro_batch,
            shuffle=False,
            num_workers=
                args.workers,
            pin_memory=True,
            persistent_workers=
                args.workers > 0,
            drop_last=False,
        )

        print()
        print(
            f"[FINAL] "
            f"{sequence} | "
            f"{len(dataset)} windows"
        )

        result = evaluate(
            model,
            loader,
            device,
        )

        results[
            sequence
        ] = result

        print(
            json.dumps(
                result,
                indent=2,
            )
        )

    # Pixel-weighted aggregate
    # over all four sequences.
    combined = ConcatDataset(
        list(
            eval_datasets.values()
        )
    )

    combined_loader = DataLoader(
        combined,
        batch_size=
            args.micro_batch,
        shuffle=False,
        num_workers=
            args.workers,
        pin_memory=True,
        persistent_workers=
            args.workers > 0,
        drop_last=False,
    )

    aggregate = evaluate(
        model,
        combined_loader,
        device,
    )

    results[
        "aggregate"
    ] = aggregate

    metrics_path = (
        args.output
        / "final_metrics.json"
    )

    metrics_path.write_text(
        json.dumps(
            results,
            indent=2,
        )
    )

    print()
    print("=" * 100)
    print(
        "MVSEC DEPTH FINAL RESULT"
    )
    print("=" * 100)

    print(
        f"{args.protocol:<14} "
        f"Abs={aggregate['abs']:.4f} m | "
        f"RMS={aggregate['rms']:.4f} m | "
        f"d1={aggregate['delta1']:.4f} | "
        f"d2={aggregate['delta2']:.4f} | "
        f"d3={aggregate['delta3']:.4f} | "
        f"RMSElog="
        f"{aggregate['rmse_log']:.4f}"
    )

    print(
        "metrics:",
        metrics_path,
    )

    print()
    print(
        "FORMAL TRAIN + FINAL EVAL: PASS"
    )


if __name__ == "__main__":
    main()
