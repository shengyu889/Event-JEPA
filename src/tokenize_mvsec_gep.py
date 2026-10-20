from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from dinov2.models.vision_transformer import vit_small


MVSEC_EVENT_MEAN = [
    0.8993729784963826,
    0.7969581014619264,
    0.8928228776286392,
]

MVSEC_EVENT_STD = [
    0.22043360769748688,
    0.2921656668186188,
    0.22049927711486816,
]

IMAGE_SIZE = 224
N_TOKENS = 256
EMBED_DIM = 384


def load_gep_vits14(
    checkpoint_path: Path,
    device: torch.device,
):
    ckpt = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if "event_encoder" not in ckpt:
        raise KeyError(
            "checkpoint has no 'event_encoder'"
        )

    state = ckpt["event_encoder"]

    expected = (384, 3, 14, 14)
    actual = tuple(
        state["patch_embed.proj.weight"].shape
    )

    if actual != expected:
        raise ValueError(
            f"patch_embed shape={actual}, "
            f"expected={expected}"
        )

    model = vit_small(
        patch_size=14,
        img_size=518,
        block_chunks=0,
        init_values=1e-6,
    )

    msg = model.load_state_dict(
        state,
        strict=True,
    )

    print("[MODEL]", msg)
    print("[MODEL] GEP ViT-S/14")
    print("[MODEL] input=3ch")
    print("[MODEL] output=[256,384]")

    model.to(device)
    model.eval()

    return model


def preprocess(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")

    arr = np.asarray(image).copy()

    x = (
        torch.from_numpy(arr)
        .permute(2, 0, 1)
        .float()
        / 255.0
    )

    x = TF.normalize(
        x,
        MVSEC_EVENT_MEAN,
        MVSEC_EVENT_STD,
    )

    _, h, w = x.shape

    # Same deterministic policy as formal DSEC cache:
    # pad if necessary, then center crop 224x224.
    pad_bottom = max(0, IMAGE_SIZE - h)
    pad_right = max(0, IMAGE_SIZE - w)

    if pad_bottom or pad_right:
        x = F.pad(
            x,
            (0, pad_right, 0, pad_bottom),
            mode="constant",
            value=0.0,
        )

    _, h, w = x.shape

    top = (h - IMAGE_SIZE) // 2
    left = (w - IMAGE_SIZE) // 2

    x = x[
        :,
        top:top + IMAGE_SIZE,
        left:left + IMAGE_SIZE,
    ]

    if tuple(x.shape) != (3, 224, 224):
        raise RuntimeError(
            f"bad preprocessing shape: "
            f"{tuple(x.shape)}"
        )

    return x.contiguous()


class MVSECEventDataset(Dataset):
    def __init__(
        self,
        sequence_root: Path,
        overwrite: bool,
    ):
        self.sequence_root = sequence_root

        self.event_dir = (
            sequence_root / "events"
        )

        self.output_dir = (
            sequence_root
            / "eventToken_vits14"
        )

        self.index_path = (
            sequence_root / "index.csv"
        )

        if not self.event_dir.is_dir():
            raise FileNotFoundError(
                self.event_dir
            )

        if not self.index_path.is_file():
            raise FileNotFoundError(
                self.index_path
            )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Read index.csv so we guarantee that every
        # event frame belongs to the official exported
        # MVSEC timeline.
        with self.index_path.open(
            "r",
            newline="",
        ) as f:
            rows = list(
                csv.DictReader(f)
            )

        self.jobs = []

        skipped = 0

        for row in rows:
            frame_id = int(row["frame_id"])

            stem = f"{frame_id:06d}"

            event_path = (
                self.event_dir
                / f"{stem}.png"
            )

            output_path = (
                self.output_dir
                / f"{stem}.pt"
            )

            if not event_path.is_file():
                raise FileNotFoundError(
                    event_path
                )

            if (
                output_path.is_file()
                and not overwrite
            ):
                try:
                    x = torch.load(
                        output_path,
                        map_location="cpu",
                        weights_only=True,
                    )

                    if (
                        isinstance(x, torch.Tensor)
                        and tuple(x.shape)
                        == (N_TOKENS, EMBED_DIM)
                    ):
                        skipped += 1
                        continue

                except Exception:
                    pass

            self.jobs.append(
                (
                    event_path,
                    output_path,
                )
            )

        print(
            f"[DATA] index rows={len(rows)}"
        )
        print(
            f"[DATA] valid existing={skipped}"
        )
        print(
            f"[DATA] todo={len(self.jobs)}"
        )

    def __len__(self):
        return len(self.jobs)

    def __getitem__(self, i):
        input_path, output_path = (
            self.jobs[i]
        )

        with Image.open(input_path) as im:
            x = preprocess(im)

        return {
            "image": x,
            "output_path": str(output_path),
        }


def atomic_save(
    tensor: torch.Tensor,
    path: Path,
):
    tmp = path.with_suffix(
        ".pt.tmp"
    )

    torch.save(
        tensor,
        tmp,
    )

    tmp.replace(path)


@torch.inference_mode()
def tokenize(
    model,
    loader,
    device,
):
    n = 0

    bar = tqdm(
        loader,
        desc="MVSEC GEP-S",
    )

    for batch in bar:
        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        out = model.forward_features(
            images
        )

        tokens = out[
            "x_norm_patchtokens"
        ]

        expected = (
            images.shape[0],
            256,
            384,
        )

        if tuple(tokens.shape) != expected:
            raise RuntimeError(
                f"expected {expected}, "
                f"got {tuple(tokens.shape)}"
            )

        tokens = (
            tokens
            .float()
            .cpu()
            .contiguous()
        )

        for token, path in zip(
            tokens,
            batch["output_path"],
        ):
            if (
                tuple(token.shape)
                != (256, 384)
            ):
                raise RuntimeError(
                    tuple(token.shape)
                )

            if not torch.isfinite(
                token
            ).all():
                raise RuntimeError(
                    f"NaN/Inf: {path}"
                )

            atomic_save(
                token,
                Path(path),
            )

            n += 1

        bar.set_postfix(
            frames=n
        )


def verify(sequence_root: Path):
    index_path = (
        sequence_root / "index.csv"
    )

    event_dir = (
        sequence_root / "events"
    )

    token_dir = (
        sequence_root
        / "eventToken_vits14"
    )

    with index_path.open(
        "r",
        newline="",
    ) as f:
        rows = list(
            csv.DictReader(f)
        )

    errors = []

    for row in rows:
        stem = (
            f"{int(row['frame_id']):06d}"
        )

        ep = event_dir / f"{stem}.png"
        tp = token_dir / f"{stem}.pt"

        if not ep.is_file():
            errors.append(
                f"missing event: {stem}"
            )
            continue

        if not tp.is_file():
            errors.append(
                f"missing token: {stem}"
            )
            continue

        try:
            x = torch.load(
                tp,
                map_location="cpu",
                weights_only=True,
            )

            if tuple(x.shape) != (
                256,
                384,
            ):
                errors.append(
                    f"{stem}: "
                    f"{tuple(x.shape)}"
                )

            if x.dtype != torch.float32:
                errors.append(
                    f"{stem}: {x.dtype}"
                )

            if not torch.isfinite(x).all():
                errors.append(
                    f"{stem}: NaN/Inf"
                )

        except Exception as e:
            errors.append(
                f"{stem}: {e}"
            )

    print("=" * 70)
    print(
        "index rows :",
        len(rows),
    )

    print(
        "event PNGs :",
        len(list(event_dir.glob("*.png"))),
    )

    print(
        "token files:",
        len(list(token_dir.glob("*.pt"))),
    )

    if errors:
        print("RESULT: FAIL")

        for e in errors[:30]:
            print("[ERROR]", e)

        raise RuntimeError(
            f"{len(errors)} errors"
        )

    print("RESULT: PASS")
    print(
        "MVSEC outdoor_day2 "
        "GEP-S cache ready: "
        "FP32 [256,384]"
    )
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sequence-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
    )

    args = parser.parse_args()

    sequence_root = (
        args.sequence_root.resolve()
    )

    if args.verify_only:
        verify(sequence_root)
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA required"
        )

    device = torch.device(
        args.device
    )

    model = load_gep_vits14(
        args.checkpoint.resolve(),
        device,
    )

    dataset = MVSECEventDataset(
        sequence_root,
        overwrite=args.overwrite,
    )

    if len(dataset):
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=(
                args.workers > 0
            ),
        )

        tokenize(
            model,
            loader,
            device,
        )

    verify(sequence_root)


if __name__ == "__main__":
    main()
