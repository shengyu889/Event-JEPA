from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dinov2.models.vision_transformer import vit_small


# ---------------------------------------------------------------------
# GEP DSEC preprocessing constants
#
# These match GEP src/pre_dse.py for the released 3-channel Stage-1
# ViT-S/14 event encoder.
# ---------------------------------------------------------------------

DSEC_EVENT_MEAN = [
    0.8993729784963826,
    0.7969581014619264,
    0.8928228776286392,
]

DSEC_EVENT_STD = [
    0.2726268701553345,
    0.2706460952758789,
    0.2812058925628662,
]

IMAGE_SIZE = 224
N_TOKENS = 256
EMBED_DIM = 384


def load_gep_vits14(
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    """
    Load the released GEP Stage-1 ViT-S/14 event encoder.

    Expected checkpoint:
        {
            "event_encoder": state_dict,
            "optimizer": ...,
            "epoch": ...
        }

    Expected patch embedding:
        [384, 3, 14, 14]
    """

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Expected checkpoint dict, got {type(checkpoint)}"
        )

    if "event_encoder" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain 'event_encoder'"
        )

    state_dict = checkpoint["event_encoder"]

    patch_key = "patch_embed.proj.weight"
    if patch_key not in state_dict:
        raise KeyError(
            f"Missing {patch_key} in event encoder checkpoint"
        )

    patch_shape = tuple(state_dict[patch_key].shape)

    expected_patch_shape = (384, 3, 14, 14)

    if patch_shape != expected_patch_shape:
        raise ValueError(
            "Unexpected GEP ViT-S patch embedding shape: "
            f"{patch_shape}; expected {expected_patch_shape}"
        )

    # Released GEP-S checkpoint has no register tokens.
    if "register_tokens" in state_dict:
        raise ValueError(
            "This tokenizer expects the released GEP-S checkpoint "
            "without register tokens."
        )

    model = vit_small(
        patch_size=14,
        img_size=518,
        block_chunks=0,
        init_values=1e-6,
    )

    msg = model.load_state_dict(
        state_dict,
        strict=True,
    )

    print(f"[MODEL] checkpoint: {checkpoint_path}")
    print(f"[MODEL] load: {msg}")
    print(f"[MODEL] patch_embed: {patch_shape}")
    print("[MODEL] backbone: ViT-S/14")
    print("[MODEL] token dim: 384")
    print("[MODEL] register tokens: none")

    model.to(device)
    model.eval()

    return model


def preprocess_gep_dsec(
    image: Image.Image,
) -> torch.Tensor:
    """
    Reproduce the deterministic tokenization preprocessing used by
    GEP pre_dse.py:

        PIL RGB
          -> ToTensor [0,1]
          -> DSEC event normalization
          -> PadToMinSide(224)
          -> CenterCrop(224,224)

    No random augmentation is used when caching Stage-2 tokens.
    """

    image = image.convert("RGB")

    array = np.asarray(image)

    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(
            f"Expected RGB image, got shape {array.shape}"
        )

    # Copy avoids non-writable numpy warnings.
    tensor = torch.from_numpy(
        array.copy()
    ).permute(2, 0, 1).float() / 255.0

    tensor = TF.normalize(
        tensor,
        DSEC_EVENT_MEAN,
        DSEC_EVENT_STD,
    )

    _, h, w = tensor.shape

    pad_bottom = max(0, IMAGE_SIZE - h)
    pad_right = max(0, IMAGE_SIZE - w)

    if pad_bottom > 0 or pad_right > 0:
        # Same direction as GEP PadToMinSide:
        # left=0, right=pad_right, top=0, bottom=pad_bottom
        tensor = F.pad(
            tensor,
            (0, pad_right, 0, pad_bottom),
            mode="constant",
            value=0,
        )

    _, h, w = tensor.shape

    if h < IMAGE_SIZE or w < IMAGE_SIZE:
        raise RuntimeError(
            f"Unable to obtain >=224 image: {tensor.shape}"
        )

    top = (h - IMAGE_SIZE) // 2
    left = (w - IMAGE_SIZE) // 2

    tensor = tensor[
        :,
        top : top + IMAGE_SIZE,
        left : left + IMAGE_SIZE,
    ]

    if tuple(tensor.shape) != (3, 224, 224):
        raise RuntimeError(
            f"Invalid preprocessed shape: {tuple(tensor.shape)}"
        )

    return tensor.contiguous()


class DSECEventImageDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        sequences: list[str] | None,
        output_subdir: str,
        overwrite: bool,
    ) -> None:
        self.root = root
        self.split = split
        self.output_subdir = output_subdir
        self.overwrite = overwrite

        image_root = root / f"{split}_images"

        if not image_root.is_dir():
            raise FileNotFoundError(
                f"DSEC image root does not exist: {image_root}"
            )

        available = sorted(
            p.name
            for p in image_root.iterdir()
            if p.is_dir()
        )

        if sequences:
            missing = [
                s for s in sequences
                if s not in available
            ]

            if missing:
                raise FileNotFoundError(
                    f"Missing requested sequence(s): {missing}"
                )

            selected = sequences
        else:
            selected = available

        self.jobs: list[tuple[Path, Path, str]] = []

        print(f"[DATA] split={split}")
        print(f"[DATA] sequences={len(selected)}")

        for sequence in selected:
            left_root = (
                image_root
                / sequence
                / "images"
                / "left"
            )

            event_dir = left_root / "eventImage"
            output_dir = left_root / output_subdir

            if not event_dir.is_dir():
                raise FileNotFoundError(
                    f"eventImage not found: {event_dir}"
                )

            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            images = sorted(
                event_dir.glob("*.png"),
                key=lambda p: int(p.stem),
            )

            if not images:
                raise FileNotFoundError(
                    f"No eventImage PNGs found: {event_dir}"
                )

            sequence_jobs = 0
            skipped = 0

            for input_path in images:
                output_path = (
                    output_dir
                    / f"{input_path.stem}.pt"
                )

                if (
                    output_path.is_file()
                    and not overwrite
                ):
                    try:
                        token = torch.load(
                            output_path,
                            map_location="cpu",
                            weights_only=True,
                        )

                        if (
                            isinstance(token, torch.Tensor)
                            and tuple(token.shape)
                            == (N_TOKENS, EMBED_DIM)
                        ):
                            skipped += 1
                            continue

                    except Exception:
                        pass

                self.jobs.append(
                    (
                        input_path,
                        output_path,
                        sequence,
                    )
                )

                sequence_jobs += 1

            print(
                f"[DATA] {sequence}: "
                f"total={len(images)} "
                f"todo={sequence_jobs} "
                f"valid_existing={skipped}"
            )

        print(
            f"[DATA] total frames to tokenize: "
            f"{len(self.jobs)}"
        )

    def __len__(self) -> int:
        return len(self.jobs)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, object]:

        input_path, output_path, sequence = (
            self.jobs[index]
        )

        with Image.open(input_path) as image:
            tensor = preprocess_gep_dsec(image)

        return {
            "image": tensor,
            "output_path": str(output_path),
            "input_path": str(input_path),
            "sequence": sequence,
        }


def atomic_save_token(
    token: torch.Tensor,
    output_path: Path,
) -> None:
    """
    Avoid leaving a corrupted .pt when tokenization is interrupted.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    torch.save(token, tmp_path)
    tmp_path.replace(output_path)


@torch.inference_mode()
def tokenize(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> None:

    processed = 0

    progress = tqdm(
        loader,
        total=len(loader),
        desc="GEP-S tokenization",
    )

    for batch in progress:
        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        # Important:
        # Token extraction is deliberately FP32.
        # Formal Stage-2 training can still use BF16.
        outputs = model.forward_features(images)

        if "x_norm_patchtokens" not in outputs:
            raise KeyError(
                "DINOv2 forward_features did not return "
                "'x_norm_patchtokens'"
            )

        tokens = outputs[
            "x_norm_patchtokens"
        ]

        expected = (
            images.shape[0],
            N_TOKENS,
            EMBED_DIM,
        )

        if tuple(tokens.shape) != expected:
            raise RuntimeError(
                f"Token shape mismatch: "
                f"expected {expected}, "
                f"got {tuple(tokens.shape)}"
            )

        # Formal cache format is FP32 on CPU.
        tokens = (
            tokens
            .detach()
            .float()
            .cpu()
            .contiguous()
        )

        output_paths = batch["output_path"]

        for token, output_path_str in zip(
            tokens,
            output_paths,
        ):
            if tuple(token.shape) != (
                N_TOKENS,
                EMBED_DIM,
            ):
                raise RuntimeError(
                    f"Single token shape mismatch: "
                    f"{tuple(token.shape)}"
                )

            if not torch.isfinite(token).all():
                raise RuntimeError(
                    f"NaN/Inf detected for "
                    f"{output_path_str}"
                )

            atomic_save_token(
                token,
                Path(output_path_str),
            )

            processed += 1

        progress.set_postfix(
            frames=processed
        )


def verify_cache(
    root: Path,
    split: str,
    sequences: list[str] | None,
    output_subdir: str,
) -> None:

    image_root = root / f"{split}_images"

    available = sorted(
        p.name
        for p in image_root.iterdir()
        if p.is_dir()
    )

    selected = sequences or available

    total_images = 0
    total_tokens = 0
    bad = []

    print()
    print("=" * 80)
    print("VERIFY FORMAL GEP-S CACHE")
    print("=" * 80)

    for sequence in selected:
        left_root = (
            image_root
            / sequence
            / "images"
            / "left"
        )

        event_dir = left_root / "eventImage"
        token_dir = left_root / output_subdir

        images = sorted(
            event_dir.glob("*.png")
        )

        tokens = sorted(
            token_dir.glob("*.pt")
        )

        image_stems = {
            p.stem for p in images
        }

        token_stems = {
            p.stem for p in tokens
        }

        missing = sorted(
            image_stems - token_stems
        )

        extra = sorted(
            token_stems - image_stems
        )

        total_images += len(images)
        total_tokens += len(tokens)

        print(
            f"{sequence:24s} "
            f"images={len(images):6d} "
            f"tokens={len(tokens):6d} "
            f"missing={len(missing):4d} "
            f"extra={len(extra):4d}"
        )

        if missing:
            bad.append(
                f"{sequence}: missing "
                f"{len(missing)} token(s)"
            )

        if extra:
            bad.append(
                f"{sequence}: extra "
                f"{len(extra)} token(s)"
            )

        # Validate all token shapes.
        for token_path in tokens:
            try:
                token = torch.load(
                    token_path,
                    map_location="cpu",
                    weights_only=True,
                )

                if not isinstance(
                    token,
                    torch.Tensor,
                ):
                    bad.append(
                        f"{token_path}: not Tensor"
                    )
                    continue

                if tuple(token.shape) != (
                    N_TOKENS,
                    EMBED_DIM,
                ):
                    bad.append(
                        f"{token_path}: "
                        f"shape={tuple(token.shape)}"
                    )

                if token.dtype != torch.float32:
                    bad.append(
                        f"{token_path}: "
                        f"dtype={token.dtype}"
                    )

                if not torch.isfinite(
                    token
                ).all():
                    bad.append(
                        f"{token_path}: NaN/Inf"
                    )

            except Exception as exc:
                bad.append(
                    f"{token_path}: {exc}"
                )

    print("-" * 80)
    print(
        f"TOTAL images={total_images} "
        f"tokens={total_tokens}"
    )

    if bad:
        print("RESULT: FAIL")

        for message in bad[:50]:
            print("[ERROR]", message)

        if len(bad) > 50:
            print(
                f"... plus "
                f"{len(bad) - 50} more errors"
            )

        raise RuntimeError(
            f"Cache verification failed "
            f"with {len(bad)} error(s)"
        )

    print("RESULT: PASS")
    print(
        "All formal GEP-S tokens are "
        "FP32 [256,384]."
    )
    print("=" * 80)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Formal GEP ViT-S/14 Stage-1 tokenization "
            "for Event-JEPA Stage-2."
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="DSEC root",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="GEP Stage-1 ViT-S checkpoint",
    )

    parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="train",
    )

    parser.add_argument(
        "--sequence",
        action="append",
        default=None,
        help=(
            "Sequence to tokenize. Repeat for multiple "
            "sequences. Omit to process all sequences."
        ),
    )

    parser.add_argument(
        "--output-subdir",
        default="eventToken_vits14",
        help=(
            "Output directory name next to eventImage. "
            "Default: eventToken_vits14"
        ),
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
        help=(
            "Recompute existing tokens. "
            "Without this flag valid existing "
            "tokens are skipped."
        ),
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    args.root = args.root.resolve()
    args.checkpoint = args.checkpoint.resolve()

    if not args.root.is_dir():
        raise FileNotFoundError(
            f"DSEC root not found: {args.root}"
        )

    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{args.checkpoint}"
        )

    if args.batch_size < 1:
        raise ValueError(
            "--batch-size must be >= 1"
        )

    if args.workers < 0:
        raise ValueError(
            "--workers must be >= 0"
        )

    if args.verify_only:
        verify_cache(
            args.root,
            args.split,
            args.sequence,
            args.output_subdir,
        )
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Formal GEP tokenization requires CUDA"
        )

    device = torch.device(args.device)

    model = load_gep_vits14(
        args.checkpoint,
        device,
    )

    dataset = DSECEventImageDataset(
        root=args.root,
        split=args.split,
        sequences=args.sequence,
        output_subdir=args.output_subdir,
        overwrite=args.overwrite,
    )

    if len(dataset) > 0:
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=(
                args.workers > 0
            ),
        )

        tokenize(
            model,
            loader,
            device,
        )
    else:
        print(
            "[DATA] Nothing to tokenize; "
            "all valid tokens already exist."
        )

    verify_cache(
        args.root,
        args.split,
        args.sequence,
        args.output_subdir,
    )


if __name__ == "__main__":
    main()
