"""Dependency-light CLI helpers for DSEC pair and token preprocessing."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any


@dataclass(frozen=True)
class TokenJob:
    name: str
    event_input: Path
    event_output: Path
    image_input: Path | None
    image_output: Path | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare DSEC event/image pairs and encode ViT patch tokens."
    )
    parser.add_argument("--root", type=Path, required=True, help="DSEC dataset root")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument(
        "--sequence",
        action="append",
        default=None,
        help="sequence to process; repeat the option to select multiple sequences",
    )
    parser.add_argument(
        "--event-encoder-ckpt",
        type=Path,
        required=True,
        help="raw or event_encoder-wrapped GEP Stage-1 checkpoint",
    )
    parser.add_argument(
        "--image-encoder-ckpt",
        type=Path,
        default=None,
        help="DINOv2 image checkpoint; required unless --event-only is used",
    )
    parser.add_argument(
        "--prepare-pairs",
        action="store_true",
        help="generate rectified eventImage/warpped PNG pairs before tokenization",
    )
    parser.add_argument(
        "--event-only",
        action="store_true",
        help="write only eventToken and do not load or modify imageToken",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", default=1, type=int)
    parser.add_argument(
        "--stats-out",
        "--stats_out",
        dest="stats_out",
        default=None,
        type=Path,
        help="optional YAML path for RGB statistics",
    )
    return parser


def extract_encoder_state_dict(checkpoint: Any, key: str) -> Mapping[str, Any]:
    """Return a raw state dict from either raw or training-checkpoint storage."""
    if not isinstance(checkpoint, Mapping) or not checkpoint:
        raise ValueError("checkpoint must contain a non-empty state dictionary")

    if key in checkpoint:
        state_dict = checkpoint[key]
        if not isinstance(state_dict, Mapping) or not state_dict:
            raise ValueError(f"checkpoint field {key!r} is not a state dictionary")
        return state_dict

    training_fields = {"optimizer", "epoch", "scheduler", "scaler"}
    if training_fields.intersection(checkpoint):
        raise ValueError(f"training checkpoint does not contain {key!r}")

    return checkpoint


def load_encoder_state_dict(model: Any, checkpoint: Any, key: str) -> Any:
    """Strictly load a raw or wrapped encoder checkpoint into ``model``."""
    state_dict = extract_encoder_state_dict(checkpoint, key)
    model.load_state_dict(state_dict, strict=True)
    return model


def select_sequences(image_root: Path, requested: list[str] | None) -> list[str]:
    image_root = Path(image_root)
    if not image_root.is_dir():
        raise FileNotFoundError(f"DSEC image split directory not found: {image_root}")

    available = sorted(path.name for path in image_root.iterdir() if path.is_dir())
    if requested is None:
        if not available:
            raise FileNotFoundError(f"no DSEC sequences found under: {image_root}")
        return available

    selected = list(dict.fromkeys(requested))
    missing = [name for name in selected if name not in available]
    if missing:
        raise FileNotFoundError(
            f"requested DSEC sequence(s) not found under {image_root}: "
            + ", ".join(missing)
        )
    return selected


def prepare_token_jobs(save_root: Path, event_only: bool) -> list[TokenJob]:
    """Validate inputs, reset requested outputs, and describe per-frame work."""
    save_root = Path(save_root)
    event_image_dir = save_root / "eventImage"
    if not event_image_dir.is_dir():
        raise FileNotFoundError(f"eventImage directory not found: {event_image_dir}")

    names = sorted(
        path.name
        for path in event_image_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    )
    if not names:
        raise FileNotFoundError(f"no event PNG files found under: {event_image_dir}")

    warped_dir = save_root / "warpped"
    if not event_only:
        missing = [name for name in names if not (warped_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"warpped image missing for event frame(s): {', '.join(missing)}"
            )

    event_token_dir = save_root / "eventToken"
    shutil.rmtree(event_token_dir, ignore_errors=True)
    event_token_dir.mkdir(parents=True)

    image_token_dir: Path | None = None
    if not event_only:
        image_token_dir = save_root / "imageToken"
        shutil.rmtree(image_token_dir, ignore_errors=True)
        image_token_dir.mkdir(parents=True)

    return [
        TokenJob(
            name=name,
            event_input=event_image_dir / name,
            event_output=event_token_dir / f"{Path(name).stem}.pt",
            image_input=None if event_only else warped_dir / name,
            image_output=(
                None
                if image_token_dir is None
                else image_token_dir / f"{Path(name).stem}.pt"
            ),
        )
        for name in names
    ]


def validate_args(args: argparse.Namespace) -> None:
    args.root = Path(args.root)
    args.event_encoder_ckpt = Path(args.event_encoder_ckpt)
    if args.image_encoder_ckpt is not None:
        args.image_encoder_ckpt = Path(args.image_encoder_ckpt)

    if not args.root.is_dir():
        raise FileNotFoundError(f"DSEC root not found: {args.root}")
    if not args.event_encoder_ckpt.is_file():
        raise FileNotFoundError(
            f"event encoder checkpoint not found: {args.event_encoder_ckpt}"
        )
    if not args.event_only and args.image_encoder_ckpt is None:
        raise ValueError("--image-encoder-ckpt is required unless --event-only is used")
    if args.image_encoder_ckpt is not None and not args.image_encoder_ckpt.is_file():
        raise FileNotFoundError(
            f"image encoder checkpoint not found: {args.image_encoder_ckpt}"
        )
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")


def run_from_args(
    args: argparse.Namespace,
    processor_factory: Callable[[argparse.Namespace], Any],
) -> list[str]:
    validate_args(args)
    selected = select_sequences(args.root / f"{args.split}_images", args.sequence)
    processor = processor_factory(args)

    if args.prepare_pairs:
        processor.run(subfolders=selected, n_workers=args.workers)

    processor.process_tokens(
        subfolders=selected,
        workers=args.workers,
        event_encoder_ckpt=args.event_encoder_ckpt,
        image_encoder_ckpt=args.image_encoder_ckpt,
        event_only=args.event_only,
    )

    if args.stats_out is not None:
        processor.compute_rgb_stats(
            subfolders=selected,
            save_to=str(args.stats_out),
            workers=args.workers,
        )
    return selected
