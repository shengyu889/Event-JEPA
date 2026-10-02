from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from dsec_flow_probe.dataset import DSECSequenceFlowDataset
from dsec_flow_probe.backbone import (
    FrozenFlowBackbone,
    module_sha256,
)
from dsec_flow_probe.head import masked_flow_l1
from dsec_flow_probe.train_full_memory import (
    TemporalPatchLinearFlowHead,
)
from utils import FlowMetrics, get_lr


def seed_everything(seed):
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


@torch.no_grad()
def make_features(
    representation,
    context,
    backbone,
):
    """
    Always return:
        [B,256,1536]

    stage1-full:
        raw Stage-1 tokens from four real frames.

    jepa-repeat-last:
        obtain contextualized current slot z_t,
        then repeat z_t four times.

    Therefore both controls use exactly the
    same 602,504-param linear readout as
    the previous full-memory experiment.
    """

    B, T, N, D = context.shape

    if T != 4:
        raise ValueError(
            f"expected Tc=4, got {T}"
        )

    if (N, D) != (256, 384):
        raise ValueError(
            f"expected [*,4,256,384], "
            f"got {tuple(context.shape)}"
        )

    if representation == "stage1-full":
        # Raw Stage-1 temporal information.
        #
        # [B,T,N,D]
        # -> [B,N,T,D]
        # -> [B,N,T*D]
        return (
            context
            .permute(0, 2, 1, 3)
            .contiguous()
            .reshape(
                B,
                N,
                T * D,
            )
            .float()
        )

    if representation == "jepa-repeat-last":
        if backbone is None:
            raise RuntimeError(
                "JEPA backbone missing"
            )

        # Existing FrozenFlowBackbone.forward()
        # returns contextualized current slot:
        # [B,256,384].
        z_t = backbone(context)

        if tuple(z_t.shape) != (
            B,
            N,
            D,
        ):
            raise RuntimeError(
                f"bad JEPA current shape: "
                f"{tuple(z_t.shape)}"
            )

        # Capacity control:
        # same information four times,
        # same dimensionality as full memory.
        return (
            z_t[:, :, None, :]
            .expand(
                -1,
                -1,
                4,
                -1,
            )
            .contiguous()
            .reshape(
                B,
                N,
                4 * D,
            )
            .float()
        )

    raise ValueError(
        representation
    )


@torch.no_grad()
def validate(
    representation,
    backbone,
    head,
    loader,
    device,
    precision,
):
    if backbone is not None:
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
            z = make_features(
                representation,
                context,
                backbone,
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

    fm = metrics.compute()

    return {
        "masked_l1":
            abs_sum
            / max(abs_count, 1.0),
        "epe": fm["epe"],
        "ae": fm["ae"],
        "1pe": fm["1pe"],
        "2pe": fm["2pe"],
        "3pe": fm["3pe"],
        "samples": samples,
    }


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--root",
        required=True,
    )

    p.add_argument(
        "--representation",
        required=True,
        choices=[
            "stage1-full",
            "jepa-repeat-last",
        ],
    )

    p.add_argument(
        "--checkpoint",
        default=None,
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
        choices=["fp32", "bf16"],
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

    if (
        args.representation
        == "jepa-repeat-last"
        and args.checkpoint is None
    ):
        raise ValueError(
            "--checkpoint is required "
            "for jepa-repeat-last"
        )

    seed_everything(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    root = Path(args.root)

    all_sequences = sorted(
        x.name
        for x in (
            root / "train_optical_flow"
        ).iterdir()
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

    train_dataset = DSECSequenceFlowDataset(
        root,
        train_sequences,
        context_frames=4,
    )

    val_dataset = DSECSequenceFlowDataset(
        root,
        val_sequences,
        context_frames=4,
    )

    generator = torch.Generator()
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

    backbone = None

    if (
        args.representation
        == "jepa-repeat-last"
    ):
        backbone = FrozenFlowBackbone(
            mode="jepa",
            checkpoint=args.checkpoint,
        ).to(device)

        if backbone.context_frames != 4:
            raise RuntimeError(
                "matched experiment "
                "requires JEPA Tc=4"
            )

        backbone.requires_grad_(
            False
        )

        backbone.eval()

        _, trainable = count_params(
            backbone
        )

        if trainable != 0:
            raise RuntimeError(
                "backbone is not frozen"
            )

        backbone_hash_before = (
            module_sha256(
                backbone.encoder
            )
        )

    else:
        backbone_hash_before = None

    #
    # SAME RNG point as all previous
    # temporal heads.
    #
    torch.manual_seed(
        args.seed
    )

    torch.cuda.manual_seed_all(
        args.seed
    )

    head = TemporalPatchLinearFlowHead(
        context_frames=4,
        embed_dim=384,
    ).to(device)

    head_hash_before = (
        module_sha256(head)
    )

    head_total, head_trainable = (
        count_params(head)
    )

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    out = Path(
        args.output_dir
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print(
        "DSEC MEMORY CONTROL FLOW PROBE"
    )
    print("=" * 80)

    print(
        "representation        :",
        args.representation,
    )
    print(
        "checkpoint            :",
        args.checkpoint,
    )
    print(
        "feature shape         :",
        "[B,256,1536]",
    )
    print(
        "readout               :",
        "Linear(1536,392)",
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

    train_iter = iter(
        train_loader
    )

    running_loss = 0.0
    running_count = 0

    start = time.time()

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
                z = make_features(
                    args.representation,
                    context,
                    backbone,
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
                f"non-finite loss at "
                f"step {step + 1}"
            )

        loss.backward()

        grad_norm = clip_grad_norm_(
            head.parameters(),
            max_norm=1.0,
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
            print(
                f"step={step+1:05d}/"
                f"{args.total_steps} "
                f"lr={lr:.8f} "
                f"loss={loss.item():.4f} "
                f"grad="
                f"{float(grad_norm):.4f} "
                f"time="
                f"{time.time()-start:.1f}s"
            )

    print()
    print(
        "Running held-out validation..."
    )

    result = validate(
        args.representation,
        backbone,
        head,
        val_loader,
        device,
        args.precision,
    )

    head_hash_after = (
        module_sha256(head)
    )

    if backbone is not None:
        backbone_hash_after = (
            module_sha256(
                backbone.encoder
            )
        )

        backbone_unchanged = (
            backbone_hash_before
            == backbone_hash_after
        )

    else:
        backbone_hash_after = None
        backbone_unchanged = True

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
            "head did not update"
        )

    wallclock = (
        time.time()
        - start
    )

    summary = {
        "representation":
            args.representation,

        "checkpoint":
            args.checkpoint,

        "seed":
            args.seed,

        "train_samples":
            len(train_dataset),

        "val_samples":
            len(val_dataset),

        "feature_shape":
            "[B,256,1536]",

        "head_params":
            head_total,

        "head_sha256_before":
            head_hash_before,

        "head_sha256_after":
            head_hash_after,

        "backbone_sha256_before":
            backbone_hash_before,

        "backbone_sha256_after":
            backbone_hash_after,

        "backbone_unchanged":
            backbone_unchanged,

        "head_changed":
            head_changed,

        "train_mean_loss":
            running_loss
            / max(
                running_count,
                1,
            ),

        "val":
            result,

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
        out / "final.pt",
    )

    (
        out / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print()
    print("=" * 80)
    print("CONTROL FINAL RESULT")
    print("=" * 80)

    print(
        "representation:",
        args.representation,
    )
    print(
        f"masked L1 : "
        f"{result['masked_l1']:.5f}"
    )
    print(
        f"EPE ↓     : "
        f"{result['epe']:.5f}"
    )
    print(
        f"AE ↓      : "
        f"{result['ae']:.5f}"
    )
    print(
        f"1PE ↓     : "
        f"{result['1pe']:.5f}"
    )
    print(
        f"2PE ↓     : "
        f"{result['2pe']:.5f}"
    )
    print(
        f"3PE ↓     : "
        f"{result['3pe']:.5f}"
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
