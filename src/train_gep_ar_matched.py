from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import Block
from utils import spatiotemporal_aggregate, get_lr
from event_jepa.packed_stage2 import PackedStage2Dataset


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Sequence-aware 8-frame DSEC token dataset
# ============================================================

class DSEC8FrameTokenDataset(Dataset):
    """
    Sequence-aware, frame-aligned GEP Stage-2 dataset.

    Each sample:
        [8, 256, 384]

    Never crosses DSEC sequence boundaries.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        frames: int = 8,
        n_tokens: int = 256,
        embed_dim: int = 384,
    ):
        super().__init__()

        self.root = Path(root)
        self.split = split
        self.frames = int(frames)
        self.n_tokens = int(n_tokens)
        self.embed_dim = int(embed_dim)

        split_root = (
            self.root
            / f"{split}_images"
        )

        if not split_root.is_dir():
            raise FileNotFoundError(
                split_root
            )

        self.samples = []

        for seq_dir in sorted(
            p for p in split_root.iterdir()
            if p.is_dir()
        ):
            token_dir = (
                seq_dir
                / "images"
                / "left"
                / "eventToken"
            )

            if not token_dir.is_dir():
                continue

            paths = sorted(
                token_dir.glob("*.pt"),
                key=lambda p: int(p.stem),
            )

            if len(paths) < self.frames:
                continue

            # Sliding frame-aligned windows.
            for start in range(
                len(paths)
                - self.frames
                + 1
            ):
                window = tuple(
                    paths[
                        start:
                        start + self.frames
                    ]
                )

                self.samples.append(
                    (
                        seq_dir.name,
                        window,
                    )
                )

        if not self.samples:
            raise RuntimeError(
                f"no valid samples in {split_root}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sequence, paths = self.samples[idx]

        frames = []

        for path in paths:
            x = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )

            if not isinstance(
                x,
                torch.Tensor,
            ):
                raise TypeError(
                    f"not Tensor: {path}"
                )

            if tuple(x.shape) != (
                self.n_tokens,
                self.embed_dim,
            ):
                raise ValueError(
                    f"{path}: "
                    f"{tuple(x.shape)}"
                )

            if not torch.isfinite(x).all():
                raise ValueError(
                    f"non-finite token: {path}"
                )

            frames.append(
                x.float()
            )

        return {
            "tokens":
                torch.stack(
                    frames,
                    dim=0,
                ),

            "sequence":
                sequence,
        }


class PackedGEP8FrameDataset(Dataset):
    """
    Thin adapter around the shared packed Stage-2 dataset.

    Returns the exact interface expected by Matched GEP-AR:
        tokens: [8, 256, 384]
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        n_tokens: int = 256,
        embed_dim: int = 384,
    ):
        super().__init__()

        self.base = PackedStage2Dataset(
            root=root,
            split=split,
            n_tokens=n_tokens,
            embed_dim=embed_dim,
        )

        self.split = split

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]

        return {
            "tokens": sample["window"],
            "sequence": "outdoor_day2",
        }


# ============================================================
# GEP autoregressive Stage-2 core
# ============================================================

class MatchedGEPAR(nn.Module):
    """
    Keeps the important original GEP Stage-2 components:

      8-frame spatiotemporal_aggregate
      modality embedding
      positional embedding
      12 causal Transformer blocks
      Linear D->D prediction head
      next-token MSE

    Visualization-only Rec decoder is intentionally omitted.
    """

    def __init__(
        self,
        embed_dim=384,
        num_heads=6,
        num_layers=12,
        n_tokens=256,
        window_size=4096,
        images_per_group=8,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.n_tokens = n_tokens
        self.images_per_group = (
            images_per_group
        )

        cfg = SimpleNamespace(
            n_embed=embed_dim,
            n_head=num_heads,
        )

        self.transformer = nn.ModuleDict({
            "modality_embed":
                nn.Embedding(
                    5,
                    embed_dim,
                ),

            "pos_embed":
                nn.Embedding(
                    window_size,
                    embed_dim,
                ),

            "blocks":
                nn.ModuleList([
                    Block(cfg)
                    for _ in range(
                        num_layers
                    )
                ]),

            "norm":
                nn.LayerNorm(
                    embed_dim
                ),
        })

        self.head = nn.Linear(
            embed_dim,
            embed_dim,
        )

    def aggregate(
        self,
        frames: torch.Tensor,
    ):
        """
        frames:
            [B,T,256,384]
        """

        B, T, N, D = (
            frames.shape
        )

        if N != self.n_tokens:
            raise ValueError(
                f"expected {self.n_tokens} "
                f"tokens, got {N}"
            )

        if D != self.embed_dim:
            raise ValueError(
                f"expected D={self.embed_dim}, "
                f"got {D}"
            )

        slot = frames.reshape(
            B,
            T * N,
            D,
        )

        # DSEC event modality id = 2.
        ids = torch.full(
            (
                B,
                T * N,
            ),
            2,
            dtype=torch.long,
            device=frames.device,
        )

        slot, ids = (
            spatiotemporal_aggregate(
                slot,
                ids,
                tokens_per_image=(
                    self.n_tokens
                ),
                images_per_group=(
                    self.images_per_group
                ),
            )
        )

        return slot, ids

    def encode(
        self,
        frames: torch.Tensor,
    ):
        x, ids = self.aggregate(
            frames
        )

        B, L, D = x.shape

        pos = torch.arange(
            L,
            dtype=torch.long,
            device=x.device,
        )

        x = (
            x
            + self.transformer
                .modality_embed(ids)
            + self.transformer
                .pos_embed(pos)[None]
        )

        for block in (
            self.transformer.blocks
        ):
            x = block(
                x,
                is_causal=True,
            )

        return (
            self.transformer
            .norm(x)
        )

    def forward(
        self,
        frames: torch.Tensor,
    ):
        slot, ids = self.aggregate(
            frames
        )

        # Exact GEP-style next-token shift
        x = slot[:, :-1]
        target = slot[:, 1:]

        ids = ids[:, :-1]

        B, L, D = x.shape

        pos = torch.arange(
            L,
            dtype=torch.long,
            device=x.device,
        )

        x = (
            x
            + self.transformer
                .modality_embed(ids)
            + self.transformer
                .pos_embed(pos)[None]
        )

        for block in (
            self.transformer.blocks
        ):
            x = block(
                x,
                is_causal=True,
            )

        x = (
            self.transformer
            .norm(x)
        )

        pred = self.head(x)

        loss = F.mse_loss(
            pred,
            target,
        )

        return {
            "loss": loss,
            "prediction": pred,
            "target": target,
        }


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    device,
    precision,
):
    model.eval()

    total = 0.0
    count = 0

    use_amp = (
        device.type == "cuda"
        and precision == "bf16"
    )

    for batch in loader:
        x = batch[
            "tokens"
        ].to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            out = model(x)

        total += (
            float(out["loss"])
            * x.shape[0]
        )

        count += x.shape[0]

    return total / max(
        count,
        1,
    )


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--root",
        required=True,
    )

    p.add_argument(
        "--output-dir",
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
        default=30000,
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
        "--num-workers",
        type=int,
        default=4,
    )

    p.add_argument(
        "--grad-accum-steps",
        type=int,
        default=2,
    )

    p.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    p.add_argument(
        "--weight-decay",
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
        default=100,
    )

    p.add_argument(
        "--val-every",
        type=int,
        default=2500,
    )

    args = p.parse_args()

    seed_everything(
        args.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    packed_root = (
        Path(args.root) / "packed"
    )

    packed_available = all(
        (
            packed_root / name
        ).is_file()
        for name in [
            "train_tokens.npy",
            "train_timestamps.npy",
            "val_tokens.npy",
            "val_timestamps.npy",
        ]
    )

    if packed_available:
        train_dataset = (
            PackedGEP8FrameDataset(
                args.root,
                split="train",
            )
        )

        val_dataset = (
            PackedGEP8FrameDataset(
                args.root,
                split="val",
            )
        )

        print(
            "[DATA] GEP packed backend "
            f"train={len(train_dataset)} "
            f"val={len(val_dataset)}"
        )

    else:
        train_dataset = (
            DSEC8FrameTokenDataset(
                args.root,
                split="train",
            )
        )

        val_dataset = (
            DSEC8FrameTokenDataset(
                args.root,
                split="test",
            )
        )

        print(
            "[DATA] GEP pt backend "
            f"train={len(train_dataset)} "
            f"val={len(val_dataset)}"
        )

    generator = (
        torch.Generator()
    )
    generator.manual_seed(
        args.seed
    )

    train_loader = DataLoader(
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

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(
            args.num_workers > 0
        ),
    )

    model = MatchedGEPAR(
        embed_dim=384,
        num_heads=6,
        num_layers=12,
        n_tokens=256,
        window_size=4096,
        images_per_group=8,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=(
            args.weight_decay
        ),
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = {
        "method":
            "gep_ar_8f_matched",

        "seed":
            args.seed,

        "frames":
            8,

        "n_tokens":
            256,

        "embed_dim":
            384,

        "num_heads":
            6,

        "num_layers":
            12,

        "images_per_group":
            8,

        "lr":
            args.lr,

        "weight_decay":
            args.weight_decay,

        "warmup_steps":
            args.warmup_steps,

        "total_steps":
            args.total_steps,

        "batch_size":
            args.batch_size,

        "grad_accum_steps":
            args.grad_accum_steps,

        "effective_batch_size":
            args.batch_size
            * args.grad_accum_steps,
    }

    (
        output_dir
        / "config.json"
    ).write_text(
        json.dumps(
            config,
            indent=2,
        )
    )

    print("=" * 80)
    print(
        "MATCHED GEP-AR STAGE-2"
    )
    print("=" * 80)

    print(
        "device         :",
        device,
    )

    print(
        "train samples  :",
        len(train_dataset),
    )

    print(
        "val samples    :",
        len(val_dataset),
    )

    print(
        "trainable params:",
        sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        ),
    )

    train_iter = iter(
        train_loader
    )

    start_time = time.time()

    if args.grad_accum_steps < 1:
        raise ValueError(
            "grad_accum_steps must be >= 1"
        )

    print(
        "micro batch     :",
        args.batch_size,
    )

    print(
        "grad accum      :",
        args.grad_accum_steps,
    )

    print(
        "effective batch :",
        args.batch_size
        * args.grad_accum_steps,
    )

    model.train()

    for step in range(
        1,
        args.total_steps + 1,
    ):
        lr = get_lr(
            step - 1,
            args.warmup_steps,
            args.lr,
            args.total_steps,
            0.0,
        )

        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(
            set_to_none=True
        )

        loss_sum = 0.0

        for micro_step in range(
            args.grad_accum_steps
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

            x = batch[
                "tokens"
            ].to(
                device,
                non_blocking=True,
            )

            use_amp = (
                device.type == "cuda"
                and args.precision
                == "bf16"
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                out = model(x)

                raw_loss = out["loss"]

                loss = (
                    raw_loss
                    / args.grad_accum_steps
                )

            loss.backward()

            loss_sum += float(
                raw_loss.detach()
            )

        grad_norm = (
            torch.nn.utils
            .clip_grad_norm_(
                model.parameters(),
                1.0,
            )
        )

        optimizer.step()

        mean_train_loss = (
            loss_sum
            / args.grad_accum_steps
        )

        if (
            step
            % args.log_every
            == 0
        ):
            print(
                f"step={step:6d} "
                f"loss={mean_train_loss:.6f} "
                f"lr={lr:.8f} "
                f"grad={float(grad_norm):.4f}"
            )

        if (
            step
            % args.val_every
            == 0
            or step
            == args.total_steps
        ):
            val_loss = validate(
                model,
                val_loader,
                device,
                args.precision,
            )

            print(
                f"[VAL] "
                f"step={step:6d} "
                f"loss={val_loss:.6f}"
            )

            state = {
                "format_version": 1,
                "method":
                    "gep_ar_8f_matched",

                "step":
                    step,

                "transformer":
                    model.transformer
                    .state_dict(),

                "head":
                    model.head
                    .state_dict(),

                "optimizer":
                    optimizer
                    .state_dict(),

                "config":
                    config,
            }

            # Always refresh latest.
            torch.save(
                state,
                output_dir
                / "latest.pt",
            )

            # Keep only important milestones
            # to avoid many ~260 MB checkpoints.
            if step in {
                10000,
                20000,
                30000,
            }:
                torch.save(
                    state,
                    output_dir
                    / f"step_{step:08d}.pt",
                )

            model.train()

    print("=" * 80)
    print(
        "wallclock seconds:",
        time.time()
        - start_time,
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
