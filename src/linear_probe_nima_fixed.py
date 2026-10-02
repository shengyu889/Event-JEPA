import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import RandomSwapEventRedBlue
from linear_probe_nima import (
    NIMAEventPNGDataset,
    FrozenRepresentation,
    NIMA_MEAN,
    NIMA_STD,
    accuracy,
    evaluate,
)
from utils import get_lr


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def module_sha256(module):
    h = hashlib.sha256()

    for name, tensor in sorted(module.state_dict().items()):
        x = tensor.detach().cpu().contiguous()

        h.update(name.encode())
        h.update(str(tuple(x.shape)).encode())
        h.update(str(x.dtype).encode())
        h.update(
            x.view(torch.uint8)
             .numpy()
             .tobytes()
        )

    return h.hexdigest()


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

    p.add_argument("--data-root", required=True)
    p.add_argument("--stage1-checkpoint", required=True)
    p.add_argument("--stage2-export", default=None)
    p.add_argument("--output-dir", required=True)

    p.add_argument("--seed", type=int, default=0)

    p.add_argument(
        "--total-steps",
        type=int,
        default=2500,
    )

    p.add_argument(
        "--warmup-steps",
        type=int,
        default=200,
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
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
        "--min-lr",
        type=float,
        default=0.0,
    )

    p.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
    )

    p.add_argument(
        "--modality-id",
        type=int,
        default=2,
    )

    p.add_argument(
        "--log-every",
        type=int,
        default=100,
    )

    args = p.parse_args()

    if args.total_steps <= 0:
        raise ValueError("total_steps must be > 0")

    if not (
        0 <= args.warmup_steps
        < args.total_steps
    ):
        raise ValueError(
            "require 0 <= warmup_steps < total_steps"
        )

    seed_everything(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_transform = transforms.Compose([
        transforms.ToTensor(),

        transforms.Normalize(
            NIMA_MEAN,
            NIMA_STD,
        ),

        RandomSwapEventRedBlue(
            p=0.5,
            type="EL",
        ),

        transforms.RandomHorizontalFlip(
            p=0.5,
        ),

        transforms.RandomResizedCrop(
            (224, 224),
            scale=(0.5, 1.0),
            interpolation=
                transforms.InterpolationMode.BICUBIC,
        ),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),

        transforms.Normalize(
            NIMA_MEAN,
            NIMA_STD,
        ),

        transforms.Resize(
            (224, 224),
            interpolation=
                transforms.InterpolationMode.BICUBIC,
        ),
    ])

    train_dataset = NIMAEventPNGDataset(
        args.data_root,
        "train",
        train_transform,
    )

    val_dataset = NIMAEventPNGDataset(
        args.data_root,
        "val",
        val_transform,
        class_names=
            train_dataset.class_names,
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    backbone = FrozenRepresentation(
        args.stage1_checkpoint,
        args.stage2_export,
        args.modality_id,
    ).to(device)

    backbone.eval()
    backbone.requires_grad_(False)

    # -----------------------------------------
    # CRITICAL:
    # Reset RNG immediately before head creation.
    #
    # Stage-2 construction consumes RNG, so
    # without this reset Stage-1 / JEPA heads
    # would not start from identical weights.
    # -----------------------------------------
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    head = nn.Linear(
        384,
        len(train_dataset.class_names),
    ).to(device)

    backbone_total, backbone_trainable = (
        count_params(backbone)
    )

    head_total, head_trainable = (
        count_params(head)
    )

    if backbone_trainable != 0:
        raise RuntimeError(
            "backbone is not fully frozen"
        )

    if head_trainable != head_total:
        raise RuntimeError(
            "classifier head is not fully trainable"
        )

    backbone_hash_before = module_sha256(
        backbone
    )

    head_hash_before = module_sha256(
        head
    )

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    criterion = nn.CrossEntropyLoss()

    print("=" * 72)
    print("STRICT N-IMAGENET LINEAR PROBE")
    print("=" * 72)
    print("device                  :", device)
    print("classes                 :", len(train_dataset.class_names))
    print("train samples           :", len(train_dataset))
    print("val samples             :", len(val_dataset))
    print("stage2                  :", args.stage2_export)
    print("seed                    :", args.seed)
    print("total steps             :", args.total_steps)
    print("warmup steps            :", args.warmup_steps)
    print("batch size              :", args.batch_size)
    print("backbone params         :", backbone_total)
    print("backbone trainable      :", backbone_trainable)
    print("head params             :", head_total)
    print("head trainable          :", head_trainable)
    print(
        "backbone SHA256 before :",
        backbone_hash_before,
    )
    print(
        "head SHA256 before     :",
        head_hash_before,
    )

    step = 0
    train_total = 0
    train_loss_sum = 0.0
    train_top1 = 0.0
    train_top5 = 0.0

    epoch = 0

    while step < args.total_steps:
        epoch += 1

        for images, labels in tqdm(
            train_loader,
            desc=f"train epoch {epoch}",
        ):
            if step >= args.total_steps:
                break

            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            current_lr = get_lr(
                step,
                args.warmup_steps,
                args.lr,
                args.total_steps,
                args.min_lr,
            )

            for group in optimizer.param_groups:
                group["lr"] = current_lr

            optimizer.zero_grad(
                set_to_none=True
            )

            # Frozen backbone: no graph is created.
            with torch.no_grad():
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=(
                        device.type == "cuda"
                    ),
                ):
                    features = backbone(images)

            # Linear head is trained in fp32.
            logits = head(
                features.float()
            )

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()
            optimizer.step()

            n = labels.shape[0]

            scores = accuracy(
                logits.detach(),
                labels,
            )

            train_total += n
            train_loss_sum += (
                loss.item() * n
            )

            train_top1 += scores[1]
            train_top5 += scores[5]

            step += 1

            if (
                step == 1
                or step % args.log_every == 0
                or step == args.total_steps
            ):
                print(
                    f"step={step:05d}/"
                    f"{args.total_steps} "
                    f"lr={current_lr:.8f} "
                    f"loss={loss.item():.4f}"
                )

    print()
    print("Running full validation...")

    val_metrics = evaluate(
        backbone,
        head,
        val_loader,
        device,
        max_batches=None,
    )

    backbone_hash_after = module_sha256(
        backbone
    )

    head_hash_after = module_sha256(
        head
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
            "STRICT PROBE FAILED: "
            "frozen backbone changed"
        )

    if not head_changed:
        raise RuntimeError(
            "STRICT PROBE FAILED: "
            "classifier head did not change"
        )

    train_metrics = {
        "loss":
            train_loss_sum / train_total,

        "top1":
            100.0
            * train_top1
            / train_total,

        "top5":
            100.0
            * train_top5
            / train_total,
    }

    summary = {
        "protocol":
            "strict_fixed_step_linear_probe",

        "seed":
            args.seed,

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

        "num_classes":
            len(train_dataset.class_names),

        "train_samples":
            len(train_dataset),

        "val_samples":
            len(val_dataset),

        "stage1_checkpoint":
            args.stage1_checkpoint,

        "stage2_export":
            args.stage2_export,

        "modality_id":
            args.modality_id,

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

        "train":
            train_metrics,

        "val":
            val_metrics,
    }

    torch.save(
        {
            "head": head.state_dict(),
            "args": vars(args),
            "classes":
                train_dataset.class_names,
            "summary": summary,
        },
        output_dir / "final.pt",
    )

    (
        output_dir / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print()
    print("=" * 72)
    print("FINAL RESULT")
    print("=" * 72)

    print(
        f"train Top-1 : "
        f"{train_metrics['top1']:.3f}"
    )

    print(
        f"train Top-5 : "
        f"{train_metrics['top5']:.3f}"
    )

    print(
        f"val Top-1   : "
        f"{val_metrics['top1']:.3f}"
    )

    print(
        f"val Top-5   : "
        f"{val_metrics['top5']:.3f}"
    )

    print(
        "backbone unchanged:",
        backbone_unchanged,
    )

    print(
        "head changed      :",
        head_changed,
    )

    print()
    print(json.dumps(
        summary,
        indent=2,
    ))


if __name__ == "__main__":
    main()
