import argparse
import csv
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

if "dinov2" not in sys.path:
    sys.path.append("dinov2")

from dinov2.models.vision_transformer import vit_small
from event_jepa.checkpoint import load_gep_transformer_export
from model import Block
from dataset import RandomSwapEventRedBlue
from pre_dse_cli import load_encoder_state_dict


NIMA_MEAN = [
    0.9673029496145361,
    0.929740832760733,
    0.9624378831461544,
]

NIMA_STD = [
    0.12036860792540319,
    0.1674319634709885,
    0.12649585555644863,
]


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class NIMAEventPNGDataset(Dataset):
    """
    Event-only classification dataset.

    Expected:
      root/
        extracted_train/
          nXXXXXXXX/
            *.png
        extracted_val/
          nXXXXXXXX/
            *.png

    Unlike upstream NIMAClsDataset, this does not require paired JPEG files.
    """

    def __init__(
        self,
        root,
        split,
        transform,
        class_names=None,
    ):
        if split not in {"train", "val"}:
            raise ValueError("split must be train or val")

        self.root = Path(root)
        self.split = split
        self.transform = transform

        split_root = self.root / f"extracted_{split}"

        if not split_root.is_dir():
            raise FileNotFoundError(split_root)

        available = sorted(
            p.name
            for p in split_root.iterdir()
            if p.is_dir()
        )

        if class_names is None:
            self.class_names = available
        else:
            self.class_names = list(class_names)

            missing = sorted(
                set(self.class_names) - set(available)
            )

            if missing:
                raise ValueError(
                    f"{split}: missing classes: {missing[:10]}"
                )

        self.class_to_idx = {
            name: i
            for i, name in enumerate(self.class_names)
        }

        self.samples = []

        for class_name in self.class_names:
            class_dir = split_root / class_name

            files = sorted(class_dir.glob("*.png"))

            for path in files:
                self.samples.append(
                    (path, self.class_to_idx[class_name])
                )

        if not self.samples:
            raise RuntimeError(
                f"no event PNG samples found under {split_root}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]

        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class FrozenRepresentation(nn.Module):
    """
    Frozen GEP Stage-1 encoder with optional frozen Stage-2 Transformer.
    """

    def __init__(
        self,
        stage1_checkpoint,
        stage2_export=None,
        modality_id=2,
    ):
        super().__init__()

        self.modality_id = int(modality_id)

        # IMPORTANT:
        # Match the exact ViT-S construction used by our Stage-1
        # DSEC token extraction. Do not add register tokens here.
        self.encoder = vit_small(
            patch_size=14,
            img_size=518,
            block_chunks=0,
            init_values=1e-6,
        )

        payload = torch.load(
            stage1_checkpoint,
            map_location="cpu",
            weights_only=True,
        )

        load_encoder_state_dict(
            self.encoder,
            payload,
            "event_encoder",
        )

        self.encoder.requires_grad_(False)
        self.encoder.eval()

        self.transformer = None
        self.stage2_metadata = None

        if stage2_export is not None:
            state_dict, metadata = (
                load_gep_transformer_export(stage2_export)
            )

            embed_dim = int(
                metadata.get("embed_dim", 384)
            )

            num_heads = int(
                metadata.get("num_heads", 6)
            )

            num_layers = int(
                metadata.get("encoder_layers", 12)
            )

            max_positions = int(
                metadata.get("max_positions", 4096)
            )

            if embed_dim != 384:
                raise ValueError(
                    f"expected embed_dim=384, got {embed_dim}"
                )

            block_cfg = SimpleNamespace(
                n_embed=embed_dim,
                n_head=num_heads,
            )

            self.transformer = nn.ModuleDict({
                "modality_embed": nn.Embedding(
                    5,
                    embed_dim,
                ),
                "pos_embed": nn.Embedding(
                    max_positions,
                    embed_dim,
                ),
                "blocks": nn.ModuleList([
                    Block(block_cfg)
                    for _ in range(num_layers)
                ]),
                "norm": nn.LayerNorm(embed_dim),
            })

            self.transformer.load_state_dict(
                state_dict,
                strict=True,
            )

            self.transformer.requires_grad_(False)
            self.transformer.eval()

            self.stage2_metadata = metadata

    @torch.no_grad()
    def forward(self, x):
        out = self.encoder.forward_features(x)

        patch = out["x_norm_patchtokens"]
        cls = out["x_norm_clstoken"]

        if self.transformer is None:
            return cls.float()

        B, N, D = patch.shape

        positions = torch.arange(
            N,
            device=patch.device,
        )

        modality = torch.full(
            (B, N),
            self.modality_id,
            dtype=torch.long,
            device=patch.device,
        )

        z = (
            patch
            + self.transformer.pos_embed(
                positions
            )[None, :, :]
            + self.transformer.modality_embed(
                modality
            )
        )

        for block in self.transformer.blocks:
            z = block(
                z,
                is_causal=False,
            )

        z = self.transformer.norm(z)

        # Match GEP downstream representation contract.
        return (
            z.mean(dim=1) + cls
        ).float()


def accuracy(logits, target, topk=(1, 5)):
    maxk = min(max(topk), logits.shape[1])

    _, pred = logits.topk(
        maxk,
        dim=1,
        largest=True,
        sorted=True,
    )

    correct = pred.eq(
        target[:, None]
    )

    result = {}

    for k in topk:
        kk = min(k, logits.shape[1])

        result[k] = (
            correct[:, :kk]
            .any(dim=1)
            .float()
            .sum()
            .item()
        )

    return result


@torch.no_grad()
def evaluate(
    backbone,
    head,
    loader,
    device,
    max_batches=None,
):
    backbone.eval()
    head.eval()

    total = 0
    loss_sum = 0.0
    top1 = 0.0
    top5 = 0.0

    criterion = nn.CrossEntropyLoss()

    for batch_idx, (images, labels) in enumerate(
        tqdm(loader, desc="val")
    ):
        if (
            max_batches is not None
            and batch_idx >= max_batches
        ):
            break

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            features = backbone(images)
            logits = head(features)
            loss = criterion(logits, labels)

        n = labels.shape[0]

        scores = accuracy(
            logits.float(),
            labels,
        )

        total += n
        loss_sum += loss.item() * n
        top1 += scores[1]
        top5 += scores[5]

    if total == 0:
        raise RuntimeError("validation loader produced no samples")

    return {
        "loss": loss_sum / total,
        "top1": 100.0 * top1 / total,
        "top5": 100.0 * top5 / total,
        "samples": total,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        required=True,
    )

    parser.add_argument(
        "--stage1-checkpoint",
        required=True,
    )

    parser.add_argument(
        "--stage2-export",
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--modality-id",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    seed_everything(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
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
            p=0.5
        ),
        transforms.RandomResizedCrop(
            (224, 224),
            scale=(0.5, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC,
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
            interpolation=transforms.InterpolationMode.BICUBIC,
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
        class_names=train_dataset.class_names,
    )

    num_classes = len(
        train_dataset.class_names
    )

    print("device        :", device)
    print("classes       :", num_classes)
    print("train samples :", len(train_dataset))
    print("val samples   :", len(val_dataset))
    print("stage2        :", args.stage2_export)
    print("seed          :", args.seed)

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    backbone = FrozenRepresentation(
        args.stage1_checkpoint,
        args.stage2_export,
        args.modality_id,
    ).to(device)

    backbone.eval()

    for p in backbone.parameters():
        p.requires_grad = False

    head = nn.Linear(
        384,
        num_classes,
    ).to(device)

    # STRICT LINEAR PROBE:
    # optimizer sees ONLY the classifier head.
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
    )

    criterion = nn.CrossEntropyLoss()

    history_path = output_dir / "metrics.csv"

    with history_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "lr",
                "train_loss",
                "train_top1",
                "train_top5",
                "val_loss",
                "val_top1",
                "val_top5",
            ],
        )

        writer.writeheader()

        best_top1 = -1.0

        for epoch in range(args.epochs):

            backbone.eval()
            head.train()

            total = 0
            loss_sum = 0.0
            top1 = 0.0
            top5 = 0.0

            for batch_idx, (
                images,
                labels,
            ) in enumerate(
                tqdm(
                    train_loader,
                    desc=f"train {epoch+1}",
                )
            ):
                if (
                    args.max_train_batches
                    is not None
                    and batch_idx
                    >= args.max_train_batches
                ):
                    break

                images = images.to(
                    device,
                    non_blocking=True,
                )

                labels = labels.to(
                    device,
                    non_blocking=True,
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                with torch.no_grad():
                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.bfloat16,
                        enabled=device.type == "cuda",
                    ):
                        features = backbone(
                            images
                        )

                # Head runs fp32.
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

                total += n
                loss_sum += (
                    loss.item() * n
                )
                top1 += scores[1]
                top5 += scores[5]

            train_metrics = {
                "loss": loss_sum / total,
                "top1": 100.0
                * top1 / total,
                "top5": 100.0
                * top5 / total,
            }

            val_metrics = evaluate(
                backbone,
                head,
                val_loader,
                device,
                args.max_val_batches,
            )

            lr = optimizer.param_groups[0]["lr"]

            row = {
                "epoch": epoch + 1,
                "lr": lr,
                "train_loss": train_metrics["loss"],
                "train_top1": train_metrics["top1"],
                "train_top5": train_metrics["top5"],
                "val_loss": val_metrics["loss"],
                "val_top1": val_metrics["top1"],
                "val_top5": val_metrics["top5"],
            }

            writer.writerow(row)
            f.flush()

            print(
                f"epoch={epoch+1:03d} "
                f"train_top1={train_metrics['top1']:.3f} "
                f"val_top1={val_metrics['top1']:.3f} "
                f"val_top5={val_metrics['top5']:.3f}"
            )

            checkpoint = {
                "epoch": epoch + 1,
                "head": head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
                "classes": train_dataset.class_names,
                "val": val_metrics,
            }

            torch.save(
                checkpoint,
                output_dir / "last.pt",
            )

            if (
                val_metrics["top1"]
                > best_top1
            ):
                best_top1 = (
                    val_metrics["top1"]
                )

                torch.save(
                    checkpoint,
                    output_dir / "best.pt",
                )

            scheduler.step()

    result = {
        "best_val_top1": best_top1,
        "num_classes": num_classes,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "seed": args.seed,
        "stage1_checkpoint": args.stage1_checkpoint,
        "stage2_export": args.stage2_export,
        "modality_id": args.modality_id,
    }

    (
        output_dir / "summary.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
        )
    )

    print()
    print(json.dumps(
        result,
        indent=2,
    ))


if __name__ == "__main__":
    main()
