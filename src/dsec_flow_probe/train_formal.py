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

from dsec_flow_probe.dataset import DSECSequenceFlowDataset
from dsec_flow_probe.backbone import (
    FrozenFlowBackbone,
    module_sha256,
)
from dsec_flow_probe.head import (
    PatchLinearFlowHead,
    masked_flow_l1,
)
from utils import get_lr


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
            f"duplicate sequence in {path}"
        )

    return sorted(names)


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


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--root",
        required=True,
    )

    p.add_argument(
        "--mode",
        required=True,
        choices=[
            "stage1",
            "stage1-mean4",
            "gep-opt-style",
            "jepa",
        ],
    )

    p.add_argument(
        "--checkpoint",
        default=None,
    )

    p.add_argument(
        "--train-sequences-file",
        required=True,
    )

    # Safety only:
    # these names are read as strings,
    # NEVER instantiated as a Dataset.
    p.add_argument(
        "--final-sequences-file",
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
        choices=["fp32", "bf16"],
        default="bf16",
    )

    p.add_argument(
        "--log-every",
        type=int,
        default=50,
    )

    args = p.parse_args()

    if (
        args.mode
        in {"gep-opt-style", "jepa"}
        and args.checkpoint is None
    ):
        raise ValueError(
            f"--checkpoint required for {args.mode}"
        )

    if (
        args.mode
        in {"stage1", "stage1-mean4"}
        and args.checkpoint is not None
    ):
        raise ValueError(
            f"{args.mode} must not receive checkpoint"
        )

    if not (
        0 <= args.warmup_steps
        < args.total_steps
    ):
        raise ValueError(
            "require 0 <= warmup < total_steps"
        )

    seed_everything(args.seed)

    root = Path(args.root)

    train_sequences = read_names(
        args.train_sequences_file
    )

    final_sequences = read_names(
        args.final_sequences_file
    )

    # --------------------------------------------------
    # Formal split safety audit.
    # --------------------------------------------------

    overlap = (
        set(train_sequences)
        & set(final_sequences)
    )

    if overlap:
        raise RuntimeError(
            f"FINAL leakage into training: "
            f"{sorted(overlap)}"
        )

    flow_root = (
        root / "train_optical_flow"
    )

    all_flow_sequences = sorted(
        p.name
        for p in flow_root.iterdir()
        if p.is_dir()
    )

    unknown_train = (
        set(train_sequences)
        - set(all_flow_sequences)
    )

    unknown_final = (
        set(final_sequences)
        - set(all_flow_sequences)
    )

    if unknown_train:
        raise RuntimeError(
            f"unknown train sequences: "
            f"{sorted(unknown_train)}"
        )

    if unknown_final:
        raise RuntimeError(
            f"unknown FINAL sequences: "
            f"{sorted(unknown_final)}"
        )

    # For Formal v1 we expect the 18 DSEC flow
    # sequences to be exactly 14 train + 4 FINAL.
    combined = (
        set(train_sequences)
        | set(final_sequences)
    )

    if combined != set(all_flow_sequences):
        missing = (
            set(all_flow_sequences)
            - combined
        )
        extra = (
            combined
            - set(all_flow_sequences)
        )

        raise RuntimeError(
            "formal split does not exactly cover "
            f"all flow sequences; "
            f"missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )

    # --------------------------------------------------
    # CRITICAL:
    # Only TRAIN sequences are ever instantiated.
    #
    # FINAL sequences are not loaded by this script.
    # --------------------------------------------------

    train_dataset = DSECSequenceFlowDataset(
        root,
        train_sequences,
        context_frames=4,
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

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

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    backbone = FrozenFlowBackbone(
        mode=args.mode,
        checkpoint=args.checkpoint,
    ).to(device)

    backbone.requires_grad_(False)
    backbone.eval()

    backbone_total, backbone_trainable = (
        count_params(backbone)
    )

    if backbone_trainable != 0:
        raise RuntimeError(
            "backbone is not frozen"
        )

    if backbone.encoder is not None:
        backbone_hash_before = (
            module_sha256(
                backbone.encoder
            )
        )
    else:
        backbone_hash_before = None

    # --------------------------------------------------
    # Reset RNG immediately before head creation.
    #
    # All four methods therefore receive the exact
    # same initial flow-head weights.
    # --------------------------------------------------

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    head = PatchLinearFlowHead().to(
        device
    )

    head_hash_before = module_sha256(
        head
    )

    head_total, head_trainable = (
        count_params(head)
    )

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("FORMAL V1 DSEC FLOW HEAD — TRAIN ONLY")
    print("=" * 80)

    print("device               :", device)
    print("mode                 :", args.mode)
    print("checkpoint           :", args.checkpoint)

    print(
        "required context     :",
        backbone.context_frames,
    )

    print(
        "train sequences       :",
        len(train_sequences),
    )

    print(
        "FINAL sequences       :",
        len(final_sequences),
        "(DECLARED ONLY; NOT LOADED)",
    )

    print(
        "train samples         :",
        len(train_dataset),
    )

    print(
        "common input context  : Tc=4"
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

    print(
        "FINAL evaluation      : DISABLED"
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
            and args.precision == "bf16"
        )

        with torch.no_grad():
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                z = backbone(
                    context
                )

        if tuple(z.shape[1:]) != (
            256,
            384,
        ):
            raise RuntimeError(
                f"bad representation shape: "
                f"{tuple(z.shape)}"
            )

        if not torch.isfinite(z).all():
            raise RuntimeError(
                "non-finite representation"
            )

        pred = head(
            z.float()
        )

        loss = masked_flow_l1(
            pred,
            target,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"non-finite loss "
                f"at step {step + 1}"
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
            % args.log_every == 0
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

    # --------------------------------------------------
    # Post-training integrity check.
    #
    # NO validation Dataset is constructed here.
    # --------------------------------------------------

    head_hash_after = (
        module_sha256(head)
    )

    if backbone.encoder is not None:
        backbone_hash_after = (
            module_sha256(
                backbone.encoder
            )
        )
    else:
        backbone_hash_after = None

    if backbone_hash_before is not None:
        backbone_unchanged = (
            backbone_hash_before
            == backbone_hash_after
        )
    else:
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
            "flow head did not update"
        )

    wallclock = (
        time.time()
        - start_time
    )

    summary = {
        "protocol":
            "formal_v1_dsec_flow_train_only",

        "mode":
            args.mode,

        "checkpoint":
            args.checkpoint,

        "seed":
            args.seed,

        "train_sequences_file":
            str(
                Path(
                    args.train_sequences_file
                ).resolve()
            ),

        "final_sequences_file":
            str(
                Path(
                    args.final_sequences_file
                ).resolve()
            ),

        "train_sequences":
            train_sequences,

        "final_sequences_declared":
            final_sequences,

        "final_dataset_constructed":
            False,

        "final_evaluated":
            False,

        "train_samples":
            len(train_dataset),

        "common_dataset_context_frames":
            4,

        "model_context_frames":
            backbone.context_frames,

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
            (
                running_loss
                / max(
                    running_count,
                    1,
                )
            ),

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
        output_dir / "final.pt",
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
        "FORMAL FLOW HEAD TRAIN-ONLY: PASS"
    )
    print("=" * 80)

    print(
        "train samples        :",
        len(train_dataset),
    )

    print(
        "train mean loss      :",
        summary["train_mean_loss"],
    )

    print(
        "backbone unchanged   :",
        backbone_unchanged,
    )

    print(
        "head changed         :",
        head_changed,
    )

    print(
        "FINAL evaluated      :",
        False,
    )

    print(
        "wallclock seconds    :",
        round(
            wallclock,
            2,
        ),
    )


if __name__ == "__main__":
    main()
