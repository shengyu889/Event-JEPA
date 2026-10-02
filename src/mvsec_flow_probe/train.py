from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import (
    clip_grad_norm_,
)
from torch.utils.data import (
    DataLoader,
)
from tqdm import tqdm

from mvsec_flow_probe.dataset import (
    MVSECFlowProbeDataset,
)

from dsec_flow_probe.backbone import (
    FrozenFlowBackbone,
    module_sha256,
)

from dsec_flow_probe.head import (
    PatchLinearFlowHead,
    masked_flow_l1,
)

from utils import (
    FlowMetrics,
    get_lr,
)


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


@torch.no_grad()
def evaluate(
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
        device=torch.device(
            "cpu"
        ),
    )

    abs_sum = 0.0
    abs_count = 0.0
    samples = 0

    for batch in tqdm(
        loader,
        desc="MVSEC outdoor_day1 TEST",
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
            z = backbone(
                context
            )

        pred = head(
            z.float()
        )

        mask = target[:, 2:3]
        gt = target[:, :2]

        abs_sum += float(
            (
                (
                    pred - gt
                ).abs()
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

        samples += int(
            target.shape[0]
        )

    m = metrics.compute()

    return {
        "masked_l1":
            abs_sum
            / max(
                abs_count,
                1.0,
            ),

        "epe":
            m["epe"],

        "ae":
            m["ae"],

        "1pe":
            m["1pe"],

        "2pe":
            m["2pe"],

        "3pe":
            m["3pe"],

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
        "--mode",
        required=True,
        choices=[
            "stage1",
            "jepa",
            "gep-opt-style",
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

    args = p.parse_args()

    if (
        args.mode
        != "stage1"
        and args.checkpoint
        is None
    ):
        raise ValueError(
            "--checkpoint is required "
            f"for {args.mode}"
        )

    if not (
        0
        <= args.warmup_steps
        < args.total_steps
    ):
        raise ValueError(
            "require "
            "0 <= warmup < total"
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

    # --------------------------------------------------
    # Fixed protocol.
    #
    # outdoor_day2:
    #     train ONLY linear head
    #
    # outdoor_day1:
    #     test ONLY once after fixed 5000 steps
    # --------------------------------------------------

    train_dataset = (
        MVSECFlowProbeDataset(
            root,
            "outdoor_day2",
        )
    )

    test_dataset = (
        MVSECFlowProbeDataset(
            root,
            "outdoor_day1",
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

    test_loader = DataLoader(
        test_dataset,
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
        mode=args.mode,
        checkpoint=args.checkpoint,
    ).to(device)

    backbone.requires_grad_(
        False
    )

    backbone.eval()

    backbone_total, \
    backbone_trainable = (
        count_params(
            backbone
        )
    )

    if backbone_trainable != 0:
        raise RuntimeError(
            "backbone is not frozen"
        )

    if (
        getattr(
            backbone,
            "encoder",
            None,
        )
        is not None
    ):
        backbone_hash_before = (
            module_sha256(
                backbone.encoder
            )
        )
    else:
        backbone_hash_before = None

    # --------------------------------------------------
    # Reset RNG immediately before creating head.
    #
    # Every representation gets IDENTICAL decoder init.
    # --------------------------------------------------

    torch.manual_seed(
        args.seed
    )

    torch.cuda.manual_seed_all(
        args.seed
    )

    head = (
        PatchLinearFlowHead()
        .to(device)
    )

    head_hash_before = (
        module_sha256(head)
    )

    head_total, \
    head_trainable = (
        count_params(head)
    )

    optimizer = (
        torch.optim.AdamW(
            head.parameters(),
            lr=args.lr,
            weight_decay=
                args.weight_decay,
        )
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
        "STRICT MVSEC "
        "CROSS-DATASET FLOW PROBE"
    )
    print("=" * 80)

    print(
        "device               :",
        device,
    )

    print(
        "mode                 :",
        args.mode,
    )

    print(
        "checkpoint           :",
        args.checkpoint,
    )

    print(
        "pretraining domain   :",
        "DSEC",
    )

    print(
        "head train sequence  :",
        "outdoor_day2",
    )

    print(
        "final test sequence  :",
        "outdoor_day1",
    )

    print(
        "train samples        :",
        len(train_dataset),
    )

    print(
        "test samples         :",
        len(test_dataset),
    )

    print(
        "common input context :",
        "Tc=4",
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
        "batch size            :",
        args.batch_size,
    )

    print(
        "lr                    :",
        args.lr,
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

    head.train()

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

        # Representation is STRICTLY frozen.
        with torch.no_grad():
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                z = backbone(
                    context
                )

        if tuple(
            z.shape[1:]
        ) != (
            256,
            384,
        ):
            raise RuntimeError(
                "unexpected backbone "
                f"output {tuple(z.shape)}"
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
                "non-finite loss at "
                f"step {step + 1}"
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
            or (
                step + 1
            )
            % args.log_every
            == 0
            or (
                step + 1
            )
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
    # OD1 has not been queried before this point.
    # No checkpoint selection / early stopping on OD1.
    # --------------------------------------------------

    print()
    print(
        "Training complete."
    )

    print(
        "Running outdoor_day1 "
        "FINAL TEST exactly once..."
    )

    test_result = evaluate(
        backbone,
        head,
        test_loader,
        device,
        args.precision,
    )

    head_hash_after = (
        module_sha256(
            head
        )
    )

    if (
        getattr(
            backbone,
            "encoder",
            None,
        )
        is not None
    ):
        backbone_hash_after = (
            module_sha256(
                backbone.encoder
            )
        )
    else:
        backbone_hash_after = None

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
            "flow head did not change"
        )

    wallclock = (
        time.time()
        - start_time
    )

    summary = {
        "protocol":
            "DSEC-pretrained -> "
            "MVSEC OD2-head-train -> "
            "OD1-test",

        "mode":
            args.mode,

        "checkpoint":
            args.checkpoint,

        "seed":
            args.seed,

        "train_sequence":
            "outdoor_day2",

        "test_sequence":
            "outdoor_day1",

        "train_samples":
            len(train_dataset),

        "test_samples":
            len(test_dataset),

        "total_steps":
            args.total_steps,

        "batch_size":
            args.batch_size,

        "lr":
            args.lr,

        "weight_decay":
            args.weight_decay,

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

        "test":
            test_result,

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
        "FINAL TEST RESULT "
        "(outdoor_day1)"
    )
    print("=" * 80)

    print(
        f"masked L1 : "
        f"{test_result['masked_l1']:.5f}"
    )

    print(
        f"EPE ↓     : "
        f"{test_result['epe']:.5f}"
    )

    print(
        f"AE ↓      : "
        f"{test_result['ae']:.5f}"
    )

    print(
        f"1PE ↓     : "
        f"{test_result['1pe']:.5f}"
    )

    print(
        f"2PE ↓     : "
        f"{test_result['2pe']:.5f}"
    )

    print(
        f"3PE ↓     : "
        f"{test_result['3pe']:.5f}"
    )

    print(
        "test samples       :",
        test_result[
            "samples"
        ],
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
        "wallclock seconds :",
        round(
            wallclock,
            2,
        ),
    )


if __name__ == "__main__":
    main()
