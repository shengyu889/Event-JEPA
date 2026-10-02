from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from dsec_flow_probe.dataset import DSECSequenceFlowDataset
from dsec_flow_probe.backbone import (
    FrozenFlowBackbone,
    module_sha256,
)
from dsec_flow_probe.head import masked_flow_l1
from utils import FlowMetrics, get_lr


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(module):
    total = sum(
        p.numel()
        for p in module.parameters()
    )
    trainable = sum(
        p.numel()
        for p in module.parameters()
        if p.requires_grad
    )
    return total, trainable


class TemporalPatchLinearFlowHead(nn.Module):
    """
    Strict full-memory linear probe.

    Input encoder memory:
        [B, T*N, D]

    Rearrange by spatial patch:
        [B, T, N, D]
        -> [B, N, T*D]

    One linear layer per spatial patch:
        T*D -> 2*14*14

    Output:
        [B, 2, 224, 224]

    No temporal Transformer / MLP / attention is added.
    """

    def __init__(
        self,
        context_frames: int = 4,
        embed_dim: int = 384,
        grid_size: int = 16,
        patch_size: int = 14,
    ):
        super().__init__()

        self.context_frames = context_frames
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.patch_size = patch_size

        self.n_tokens = (
            grid_size * grid_size
        )

        self.linear = nn.Linear(
            context_frames * embed_dim,
            2 * patch_size * patch_size,
        )

    def forward(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        z: [B,N,T*D]
        """

        if z.ndim != 3:
            raise ValueError(
                f"expected [B,N,TD], "
                f"got {tuple(z.shape)}"
            )

        B, N, TD = z.shape

        expected_td = (
            self.context_frames
            * self.embed_dim
        )

        if N != self.n_tokens:
            raise ValueError(
                f"expected {self.n_tokens} tokens, "
                f"got {N}"
            )

        if TD != expected_td:
            raise ValueError(
                f"expected feature dim "
                f"{expected_td}, got {TD}"
            )

        x = self.linear(z)

        # [B, 256, 392]
        x = x.reshape(
            B,
            self.grid_size,
            self.grid_size,
            2,
            self.patch_size,
            self.patch_size,
        )

        # [B,2,16,14,16,14]
        x = x.permute(
            0, 3, 1, 4, 2, 5
        ).contiguous()

        # [B,2,224,224]
        x = x.reshape(
            B,
            2,
            self.grid_size * self.patch_size,
            self.grid_size * self.patch_size,
        )

        return x


@torch.no_grad()
def encode_full_memory(
    backbone,
    context,
):
    """
    context:
        [B,4,N,D]

    returns:
        [B,N,4D]

    The same physical spatial patch across
    temporal slots is concatenated.
    """

    if backbone.mode != "jepa":
        raise ValueError(
            "full-memory probe currently "
            "requires mode=jepa"
        )

    T = backbone.context_frames

    if T != 4:
        raise ValueError(
            f"matched experiment requires "
            f"Tc=4, checkpoint has Tc={T}"
        )

    if context.shape[1] < T:
        raise ValueError(
            "insufficient context frames"
        )

    x = context[:, -T:]

    B, _, N, D = x.shape

    z = backbone.encoder(x)

    expected = (
        B,
        T * backbone.n_tokens,
        backbone.embed_dim,
    )

    if tuple(z.shape) != expected:
        raise RuntimeError(
            f"unexpected encoder output "
            f"{tuple(z.shape)}, "
            f"expected {expected}"
        )

    z = z.reshape(
        B,
        T,
        N,
        D,
    )

    # group the SAME spatial patch over time:
    #
    # [B,T,N,D]
    # -> [B,N,T,D]
    # -> [B,N,T*D]
    z = (
        z.permute(
            0, 2, 1, 3
        )
        .contiguous()
        .reshape(
            B,
            N,
            T * D,
        )
    )

    return z.float()


@torch.no_grad()
def validate(
    backbone,
    head,
    loader,
    device,
    precision,
):
    backbone.eval()
    head.eval()

    metrics = FlowMetrics(
        n_vals=[1, 2, 3],
        device=torch.device("cpu"),
    )

    abs_sum = 0.0
    abs_count = 0.0
    samples = 0

    for batch in tqdm(
        loader,
        desc="val",
    ):
        context = batch[
            "context"
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

        use_amp = (
            device.type == "cuda"
            and precision == "bf16"
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            z = encode_full_memory(
                backbone,
                context,
            )

        pred = head(
            z.float()
        )

        mask = target[:, 2:3]
        gt = target[:, :2]

        abs_sum += float(
            (
                (pred - gt).abs()
                * mask
            ).sum().item()
        )

        abs_count += float(
            mask.sum().item()
            * 2.0
        )

        metrics.update(
            pred.detach(),
            target,
        )

        samples += target.shape[0]

    flow_metrics = metrics.compute()

    return {
        "masked_l1":
            abs_sum
            / max(
                abs_count,
                1.0,
            ),

        "epe":
            flow_metrics["epe"],

        "ae":
            flow_metrics["ae"],

        "1pe":
            flow_metrics["1pe"],

        "2pe":
            flow_metrics["2pe"],

        "3pe":
            flow_metrics["3pe"],

        "samples":
            samples,
    }


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--root",
        required=True,
    )

    p.add_argument(
        "--checkpoint",
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
        "--num-workers",
        type=int,
        default=4,
    )

    p.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    p.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
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
        "--val-sequences",
        nargs="+",
        default=[
            "thun_00_a",
            "zurich_city_02_d",
        ],
    )

    args = p.parse_args()

    if not (
        0 <= args.warmup_steps
        < args.total_steps
    ):
        raise ValueError(
            "require 0 <= warmup < total_steps"
        )

    seed_everything(
        args.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    root = Path(
        args.root
    )

    flow_root = (
        root
        / "train_optical_flow"
    )

    all_sequences = sorted(
        x.name
        for x in flow_root.iterdir()
        if x.is_dir()
    )

    val_sequences = sorted(
        args.val_sequences
    )

    train_sequences = [
        x
        for x in all_sequences
        if x not in set(
            val_sequences
        )
    ]

    #
    # EXACT same sample pool as current-slot probe.
    #
    train_dataset = (
        DSECSequenceFlowDataset(
            root,
            train_sequences,
            context_frames=4,
        )
    )

    val_dataset = (
        DSECSequenceFlowDataset(
            root,
            val_sequences,
            context_frames=4,
        )
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

    backbone = FrozenFlowBackbone(
        mode="jepa",
        checkpoint=args.checkpoint,
    ).to(device)

    if backbone.context_frames != 4:
        raise RuntimeError(
            "this matched full-memory "
            "probe requires Tc=4"
        )

    backbone.requires_grad_(
        False
    )

    backbone.eval()

    backbone_total, backbone_trainable = (
        count_params(
            backbone
        )
    )

    if backbone_trainable != 0:
        raise RuntimeError(
            "backbone is not frozen"
        )

    backbone_hash_before = (
        module_sha256(
            backbone.encoder
        )
    )

    #
    # CRITICAL FAIRNESS:
    # reset identical RNG immediately
    # before head initialization.
    #
    torch.manual_seed(
        args.seed
    )

    torch.cuda.manual_seed_all(
        args.seed
    )

    head = (
        TemporalPatchLinearFlowHead(
            context_frames=4,
            embed_dim=
                backbone.embed_dim,
        )
        .to(device)
    )

    head_hash_before = (
        module_sha256(head)
    )

    head_total, head_trainable = (
        count_params(head)
    )

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.lr,
        weight_decay=
            args.weight_decay,
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print(
        "STRICT DSEC FULL-MEMORY "
        "TEMPORAL LINEAR FLOW PROBE"
    )
    print("=" * 80)

    print(
        "device               :",
        device,
    )
    print(
        "checkpoint           :",
        args.checkpoint,
    )
    print(
        "checkpoint horizons  :",
        backbone.config[
            "horizons"
        ],
    )
    print(
        "model context        :",
        backbone.context_frames,
    )
    print(
        "feature shape        :",
        "[B,256,1536]",
    )
    print(
        "readout              :",
        "Linear(1536,392)",
    )
    print(
        "train sequences       :",
        len(train_sequences),
    )
    print(
        "val sequences         :",
        val_sequences,
    )
    print(
        "train samples         :",
        len(train_dataset),
    )
    print(
        "val samples           :",
        len(val_dataset),
    )
    print(
        "backbone params       :",
        backbone_total,
    )
    print(
        "backbone trainable    :",
        backbone_trainable,
    )
    print(
        "head params           :",
        head_total,
    )
    print(
        "head trainable        :",
        head_trainable,
    )
    print(
        "head SHA256 before    :",
        head_hash_before,
    )
    print(
        "backbone SHA before   :",
        backbone_hash_before,
    )
    print(
        "total steps           :",
        args.total_steps,
    )
    print(
        "warmup steps          :",
        args.warmup_steps,
    )
    print(
        "batch size            :",
        args.batch_size,
    )
    print(
        "precision             :",
        args.precision,
    )

    train_iter = iter(
        train_loader
    )

    running_loss = 0.0
    running_count = 0

    start_time = time.time()

    for step in range(
        args.total_steps
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

        context = batch[
            "context"
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

        lr = get_lr(
            step,
            args.warmup_steps,
            args.lr,
            args.total_steps,
            0.0,
        )

        for group in (
            optimizer.param_groups
        ):
            group["lr"] = lr

        optimizer.zero_grad(
            set_to_none=True
        )

        use_amp = (
            device.type == "cuda"
            and args.precision
            == "bf16"
        )

        with torch.no_grad():
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                z = (
                    encode_full_memory(
                        backbone,
                        context,
                    )
                )

        pred = head(
            z.float()
        )

        loss = masked_flow_l1(
            pred,
            target,
        )

        if not torch.isfinite(
            loss
        ):
            raise RuntimeError(
                f"non-finite loss "
                f"at step {step + 1}"
            )

        loss.backward()

        grad_norm = (
            clip_grad_norm_(
                head.parameters(),
                max_norm=1.0,
            )
        )

        optimizer.step()

        running_loss += float(
            loss.item()
        )

        running_count += 1

        if (
            step == 0
            or (step + 1)
            % args.log_every
            == 0
            or (step + 1)
            == args.total_steps
        ):
            elapsed = (
                time.time()
                - start_time
            )

            print(
                f"step="
                f"{step + 1:05d}/"
                f"{args.total_steps} "
                f"lr={lr:.8f} "
                f"loss="
                f"{loss.item():.4f} "
                f"grad="
                f"{float(grad_norm):.4f} "
                f"time="
                f"{elapsed:.1f}s"
            )

    print()
    print(
        "Running complete "
        "held-out validation..."
    )

    val_result = validate(
        backbone,
        head,
        val_loader,
        device,
        args.precision,
    )

    head_hash_after = (
        module_sha256(head)
    )

    backbone_hash_after = (
        module_sha256(
            backbone.encoder
        )
    )

    backbone_unchanged = (
        backbone_hash_before
        == backbone_hash_after
    )

    head_changed = (
        head_hash_before
        != head_hash_after
    )

    if not backbone_unchanged:
        raise RuntimeError(
            "frozen backbone changed"
        )

    if not head_changed:
        raise RuntimeError(
            "flow head did not update"
        )

    wallclock = (
        time.time()
        - start_time
    )

    summary = {
        "protocol":
            "strict_full_memory_"
            "temporal_linear_flow_probe",

        "checkpoint":
            args.checkpoint,

        "seed":
            args.seed,

        "feature":
            "[B,256,1536]",

        "readout":
            "single_linear",

        "train_sequences":
            train_sequences,

        "val_sequences":
            val_sequences,

        "train_samples":
            len(train_dataset),

        "val_samples":
            len(val_dataset),

        "context_frames":
            4,

        "total_steps":
            args.total_steps,

        "warmup_steps":
            args.warmup_steps,

        "batch_size":
            args.batch_size,

        "learning_rate":
            args.lr,

        "weight_decay":
            args.weight_decay,

        "precision":
            args.precision,

        "backbone_params":
            backbone_total,

        "backbone_trainable":
            backbone_trainable,

        "head_params":
            head_total,

        "head_trainable":
            head_trainable,

        "backbone_sha256_before":
            backbone_hash_before,

        "backbone_sha256_after":
            backbone_hash_after,

        "backbone_unchanged":
            backbone_unchanged,

        "head_sha256_before":
            head_hash_before,

        "head_sha256_after":
            head_hash_after,

        "head_changed":
            head_changed,

        "train_mean_loss":
            running_loss
            / max(
                running_count,
                1,
            ),

        "val":
            val_result,

        "wallclock_seconds":
            wallclock,
    }

    torch.save(
        {
            "head":
                head.state_dict(),

            "summary":
                summary,

            "args":
                vars(args),
        },
        output_dir
        / "final.pt",
    )

    (
        output_dir
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
        "FULL-MEMORY FINAL RESULT"
    )
    print("=" * 80)

    print(
        f"masked L1 : "
        f"{val_result['masked_l1']:.5f}"
    )

    print(
        f"EPE ↓     : "
        f"{val_result['epe']:.5f}"
    )

    print(
        f"AE ↓      : "
        f"{val_result['ae']:.5f}"
    )

    print(
        f"1PE ↓     : "
        f"{val_result['1pe']:.5f}"
    )

    print(
        f"2PE ↓     : "
        f"{val_result['2pe']:.5f}"
    )

    print(
        f"3PE ↓     : "
        f"{val_result['3pe']:.5f}"
    )

    print(
        "backbone unchanged:",
        backbone_unchanged,
    )

    print(
        "head changed      :",
        head_changed,
    )

    print(
        "head SHA before   :",
        head_hash_before,
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
