from __future__ import annotations

import argparse
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from config import Config
from dataset import (
    DSECPairedDataset,
    MVSECPairedDataset,
    NIMAPairedDataset,
    Normalize,
    PairedProcessor,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomSwapEventRedBlue,
    ToTensor,
)
from gra import Gra
from utils import get_lr, get_param_groups


# PyTorch 2.7 CUDA does not provide a strictly deterministic backward
# implementation for bicubic interpolation. DINOv2 interpolates positional
# embeddings for non-native image sizes such as DSEC 448x644.
#
# Keep deterministic algorithms wherever available, but allow this exact
# unsupported operation to proceed with a warning rather than changing the
# model or interpolation used by GEP.
torch.use_deterministic_algorithms(
    True,
    warn_only=True,
)


# ---------------------------------------------------------------------
# DDP utilities
# ---------------------------------------------------------------------

def init_distributed():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for formal Stage-1 training")

    torch.cuda.set_device(local_rank)

    if world_size > 1:
        dist.init_process_group(backend="nccl")

    device = torch.device("cuda", local_rank)

    return rank, world_size, local_rank, device


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def seed_everything(seed: int, rank: int):
    # Keep the domain schedule identical across ranks, while augmentation /
    # sampler randomness remains rank-specific.
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed(seed + rank)


# ---------------------------------------------------------------------
# Formal dataset schedule
# ---------------------------------------------------------------------

def select_domain(step: int, seed: int) -> str:
    """
    Deterministic across all ranks.

    One optimizer step uses exactly one domain, matching the semantics of
    the original GEP get_batch() implementation.
    """
    rng = random.Random(seed + step)
    x = rng.random()

    if x < 0.70:
        return "NIMA"
    if x < 0.90:
        return "DSEC"
    return "MVSEC"


class InfiniteDistributedLoader:
    def __init__(
        self,
        dataset,
        *,
        batch_size: int,
        rank: int,
        world_size: int,
        workers: int,
        seed: int,
    ):
        self.sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
            drop_last=False,
        )

        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=self.sampler,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=(workers > 0),
        )

        self.epoch = 0
        self.sampler.set_epoch(self.epoch)
        self.iterator = iter(self.loader)

    def next(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.epoch += 1
            self.sampler.set_epoch(self.epoch)
            self.iterator = iter(self.loader)
            return next(self.iterator)


# ---------------------------------------------------------------------
# Synthetic DDP smoke dataset
# ---------------------------------------------------------------------

class SyntheticPairDataset(Dataset):
    def __init__(self, length=128, height=448, width=644):
        self.length = int(length)
        self.height = int(height)
        self.width = int(width)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        generator = torch.Generator().manual_seed(idx)

        event = torch.randn(
            20,
            self.height,
            self.width,
            generator=generator,
        )

        image = torch.randn(
            3,
            self.height,
            self.width,
            generator=generator,
        )

        return event, image


# ---------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------

def build_model_config(args, device):
    ckpt_path = Path(args.dino_ckpt)

    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"DINOv2 checkpoint not found: {ckpt_path}"
        )

    weights = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=True,
    )

    return SimpleNamespace(
        device=str(device),

        vit="small",
        vit_backbone="dinov2",
        event_backbone="vit",
        teacher_vit="small",
        swin_project=False,

        event_channels=20,

        loss_types={
            "mse": 1.0,
            "mse_patch": 1.0,
        },

        loss_grad_ratio=None,
        loss_grad_eps=1e-8,

        image_encoder_weight=weights,
        event_encoder_weight=weights,
        restore_ckpt=None,

        n_embed=384,
        n_head=6,
        n_layer=12,
        window_size=4096,

        wd=args.weight_decay,
    )


# ---------------------------------------------------------------------
# Real formal datasets
# ---------------------------------------------------------------------

def build_real_loaders(args, rank, world_size):
    stats = Config()
    pair_type = "EI"

    # --------------------------------------------------------------
    # DSEC — preserve GEP formal Stage-1 transform.
    # --------------------------------------------------------------
    dsec_transform = PairedProcessor([
        ToTensor(type=pair_type),

        Normalize(
            stats.DSEC_ME,
            stats.DSEC_SE,
            stats.DSEC_MI,
            stats.DSEC_SI,
            type=pair_type,
        ),

        # For 20-channel tensors RandomSwapEventRedBlue is a no-op;
        # retained to match GEP transform definition.
        RandomSwapEventRedBlue(type=pair_type),

        RandomHorizontalFlip(p=0.5),

        RandomResizedCrop(
            size=(448, 644),
            scale=(0.25, 1.0),
            type=pair_type,
            interpolation="bicubic",
        ),
    ])

    dsec = DSECPairedDataset(
        root_dir=args.dsec_root,
        split="train",
        transform=dsec_transform,
        event_channels=20,
        event_subdir="eventTensor_ecddp_way",
    )

    # --------------------------------------------------------------
    # N-ImageNet — preserve current GEP Stage-1 transform.
    # --------------------------------------------------------------
    nima_transform = PairedProcessor([
        ToTensor(type=pair_type),

        Normalize(
            stats.NIMA_ME,
            stats.NIMA_SE,
            stats.NIMA_MI,
            stats.NIMA_SI,
            type=pair_type,
        ),

        RandomSwapEventRedBlue(type=pair_type),

        RandomHorizontalFlip(p=0.5),

        RandomResizedCrop(
            size=(224, 224),
            scale=(0.5, 1.0),
            type=pair_type,
            interpolation="bicubic",
        ),
    ])

    nima = NIMAPairedDataset(
        root_dir=args.nima_root,
        split="train",
        transform=nima_transform,
        event_channels=20,
    )

    # --------------------------------------------------------------
    # MVSEC extension.
    #
    # Only outdoor_day2 participates in pretraining.
    # day1 + night1/2/3 remain evaluation-only.
    # --------------------------------------------------------------
    mvsec_transform = PairedProcessor([
        ToTensor(type=pair_type),

        Normalize(
            stats.MVSEC_EVENT_ME,
            stats.MVSEC_EVENT_SE,
            stats.MVSEC_IMAGE_ME,
            stats.MVSEC_IMAGE_SE,
            type=pair_type,
        ),

        RandomSwapEventRedBlue(type=pair_type),

        RandomHorizontalFlip(p=0.5),

        RandomResizedCrop(
            size=(224, 224),
            scale=(0.5, 1.0),
            type=pair_type,
            interpolation="bicubic",
        ),
    ])

    mvsec = MVSECPairedDataset(
        root_dir=args.mvsec_root,
        sequences=("outdoor_day2",),
        transform=mvsec_transform,
        event_subdir="eventTensor_ecddp",
        event_channels=20,
    )

    if is_main(rank):
        print("=" * 80)
        print("FORMAL STAGE-1 DATASETS")
        print(f"N-ImageNet : {len(nima)}")
        print(f"DSEC       : {len(dsec)}")
        print(f"MVSEC od2  : {len(mvsec)}")
        print("mixture    : NIMA 0.70 / DSEC 0.20 / MVSEC 0.10")
        print("=" * 80)

    loaders = {
        "NIMA": InfiniteDistributedLoader(
            nima,
            batch_size=args.batch_size,
            rank=rank,
            world_size=world_size,
            workers=args.workers,
            seed=args.seed + 100,
        ),

        "DSEC": InfiniteDistributedLoader(
            dsec,
            batch_size=args.batch_size,
            rank=rank,
            world_size=world_size,
            workers=args.workers,
            seed=args.seed + 200,
        ),

        "MVSEC": InfiniteDistributedLoader(
            mvsec,
            batch_size=args.batch_size,
            rank=rank,
            world_size=world_size,
            workers=args.workers,
            seed=args.seed + 300,
        ),
    }

    return loaders


def build_smoke_loader(args, rank, world_size):
    dataset = SyntheticPairDataset(
        length=max(64, args.batch_size * world_size * 8),
        height=args.smoke_height,
        width=args.smoke_width,
    )

    return InfiniteDistributedLoader(
        dataset,
        batch_size=args.batch_size,
        rank=rank,
        world_size=world_size,
        workers=0,
        seed=args.seed,
    )


# ---------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------

def save_checkpoint(
    path,
    *,
    model,
    optimizer,
    scaler,
    step,
    args,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    raw_model = model.module if isinstance(model, DDP) else model

    state = {
        "event_encoder": raw_model.event_encoder.state_dict(),
        "image_encoder": raw_model.image_encoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,

        # Keep both fields for compatibility with old training utilities.
        "step": int(step),
        "epoch": int(step - 1),

        "formal_stage1": {
            "event_channels": 20,
            "event_representation":
                "10 temporal chunks x 2 polarity histogram channels",
            "backbone": "dinov2_vits14_reg4",
            "loss_types": {
                "mse": 1.0,
                "mse_patch": 1.0,
            },
            "mixture": {
                "NIMA": 0.70,
                "DSEC": 0.20,
                "MVSEC": 0.10,
            },
            "mvsec_pretrain_sequences": [
                "outdoor_day2",
            ],
            "batch_per_gpu": args.batch_size,
            "grad_accum_steps": args.grad_accum,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "precision": args.precision,
        },
    }

    torch.save(state, path)


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train(args):
    rank, world_size, local_rank, device = init_distributed()

    try:
        seed_everything(args.seed, rank)

        model_cfg = build_model_config(args, device)

        model = Gra(
            model_cfg,
            build_training_state=False,
        ).to(device)

        # DINOv2 mask_token is only used when masks are supplied.
        # GEP Stage-1 performs ordinary unmasked forward passes, therefore
        # mask_token is structurally unused and receives no gradient.
        # Freeze it explicitly so DDP with find_unused_parameters=False
        # preserves the original GEP behaviour without extra graph traversal.
        event_encoder = model.event_encoder
        if hasattr(event_encoder, "encoder"):
            event_encoder = event_encoder.encoder

        if hasattr(event_encoder, "mask_token"):
            event_encoder.mask_token.requires_grad_(False)

        optimizer = torch.optim.AdamW(
            get_param_groups(
                model,
                args.weight_decay,
            )
        )

        start_step = 0

        use_scaler = args.precision == "fp16"

        scaler = (
            torch.amp.GradScaler("cuda")
            if use_scaler
            else None
        )

        if args.resume is not None:
            ckpt = torch.load(
                args.resume,
                map_location="cpu",
                weights_only=False,
            )

            model.event_encoder.load_state_dict(
                ckpt["event_encoder"],
                strict=True,
            )

            optimizer.load_state_dict(
                ckpt["optimizer"],
            )

            if scaler is not None and ckpt.get("scaler") is not None:
                scaler.load_state_dict(
                    ckpt["scaler"]
                )

            start_step = int(
                ckpt.get(
                    "step",
                    ckpt.get("epoch", -1) + 1,
                )
            )

        if world_size > 1:
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )

        if args.synthetic_smoke:
            smoke_loader = build_smoke_loader(
                args,
                rank,
                world_size,
            )

            loaders = None

            if is_main(rank):
                print(
                    "SYNTHETIC SMOKE: "
                    f"[20,{args.smoke_height},{args.smoke_width}]"
                )
        else:
            loaders = build_real_loaders(
                args,
                rank,
                world_size,
            )
            smoke_loader = None

        effective_batch = (
            args.batch_size
            * world_size
            * args.grad_accum
        )

        if is_main(rank):
            print("=" * 80)
            print("EVENT-JEPA FORMAL STAGE-1")
            print(f"world size       : {world_size}")
            print(f"batch / GPU      : {args.batch_size}")
            print(f"grad accumulation: {args.grad_accum}")
            print(f"effective batch  : {effective_batch}")
            print(f"precision        : {args.precision}")
            print(f"steps            : {args.steps}")
            print(f"base LR          : {args.lr}")
            print("=" * 80)

        optimizer.zero_grad(set_to_none=True)

        autocast_dtype = (
            torch.float16
            if args.precision == "fp16"
            else torch.bfloat16
        )

        for global_step in range(
            start_step,
            args.steps,
        ):
            step_start = time.time()

            current_lr = get_lr(
                global_step,
                args.warmup_steps,
                args.lr,
                args.steps,
                args.min_lr,
            )

            for group in optimizer.param_groups:
                group["lr"] = (
                    current_lr
                    * group.get("lr_mult", 1.0)
                )

            # ------------------------------------------------------
            # IMPORTANT:
            # one optimizer step == one dataset domain.
            # All grad-accum microbatches use the same domain.
            # ------------------------------------------------------
            if args.synthetic_smoke:
                domain = "SMOKE"
            else:
                domain = select_domain(
                    global_step,
                    args.seed,
                )

            local_loss_sum = 0.0

            for micro_step in range(args.grad_accum):
                if args.synthetic_smoke:
                    event, image = smoke_loader.next()
                else:
                    event, image = loaders[domain].next()

                event = event.to(
                    device,
                    non_blocking=True,
                )

                image = image.to(
                    device,
                    non_blocking=True,
                )

                sync_gradients = (
                    micro_step
                    == args.grad_accum - 1
                )

                sync_context = (
                    nullcontext()
                    if sync_gradients
                    else model.no_sync()
                )

                with sync_context:
                    with torch.autocast(
                        device_type="cuda",
                        dtype=autocast_dtype,
                    ):
                        _, _, loss, _ = model(
                            event,
                            image,
                            dataset=domain,
                        )

                        backward_loss = (
                            loss / args.grad_accum
                        )

                    if scaler is None:
                        backward_loss.backward()
                    else:
                        scaler.scale(
                            backward_loss
                        ).backward()

                local_loss_sum += float(
                    loss.detach().item()
                )

            if scaler is not None:
                scaler.unscale_(optimizer)

            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            nn.utils.clip_grad_value_(
                model.parameters(),
                clip_value=2.0,
            )

            if scaler is None:
                optimizer.step()
            else:
                scaler.step(optimizer)
                scaler.update()

            optimizer.zero_grad(
                set_to_none=True
            )

            # ------------------------------------------------------
            # Logging
            # ------------------------------------------------------
            should_log = (
                (global_step + 1) % args.log_every == 0
                or global_step == start_step
                or args.synthetic_smoke
            )

            if should_log:
                mean_loss = torch.tensor(
                    local_loss_sum / args.grad_accum,
                    device=device,
                    dtype=torch.float32,
                )

                if world_size > 1:
                    dist.all_reduce(
                        mean_loss,
                        op=dist.ReduceOp.SUM,
                    )
                    mean_loss /= world_size

                if is_main(rank):
                    elapsed = time.time() - step_start

                    max_mem = (
                        torch.cuda.max_memory_allocated(device)
                        / 1024**3
                    )

                    print(
                        f"step={global_step + 1:06d} "
                        f"domain={domain:5s} "
                        f"loss={mean_loss.item():.6f} "
                        f"lr={current_lr:.8e} "
                        f"grad_norm={float(grad_norm):.4f} "
                        f"time={elapsed:.2f}s "
                        f"max_mem={max_mem:.2f}GB",
                        flush=True,
                    )

            # ------------------------------------------------------
            # Checkpoint
            # ------------------------------------------------------
            should_save = (
                not args.synthetic_smoke
                and (
                    (global_step + 1)
                    % args.save_every
                    == 0
                )
            )

            if should_save and is_main(rank):
                output_dir = Path(
                    args.output_dir
                )

                save_checkpoint(
                    output_dir
                    / f"step_{global_step + 1:08d}.pt",
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=global_step + 1,
                    args=args,
                )

        if not args.synthetic_smoke and is_main(rank):
            output_dir = Path(args.output_dir)

            save_checkpoint(
                output_dir / "stage1_final.pt",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                step=args.steps,
                args=args,
            )

        if world_size > 1:
            dist.barrier(
                device_ids=[local_rank]
            )

        raw_model = (
            model.module
            if isinstance(model, DDP)
            else model
        )

        raw_model.close()

    finally:
        cleanup_distributed()


def parse_args():
    project_root = Path(
        "/mnt/data2/gsy/Event-JEPA"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Formal 4-GPU GEP-style Stage-1 "
            "alignment for Event-JEPA."
        )
    )

    parser.add_argument(
        "--dino-ckpt",
        type=str,
        default=str(
            project_root
            / "checkpoints"
            / "dinov2"
            / "dinov2_vits14_reg4_pretrain.pth"
        ),
    )

    parser.add_argument(
        "--dsec-root",
        type=str,
        default=str(
            project_root
            / "datasets"
            / "DSEC"
        ),
    )

    parser.add_argument(
        "--nima-root",
        type=str,
        default=str(
            project_root
            / "datasets"
            / "N_ImageNet"
        ),
    )

    parser.add_argument(
        "--mvsec-root",
        type=str,
        default=str(
            project_root
            / "datasets"
            / "MVSEC"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(
            project_root
            / "runs"
            / "stage1_formal"
        ),
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Per-GPU microbatch size.",
    )

    parser.add_argument(
        "--grad-accum",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--min-lr",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=240000,
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--precision",
        choices=("fp16", "bf16"),
        default="bf16",
    )

    parser.add_argument(
        "--synthetic-smoke",
        action="store_true",
    )

    parser.add_argument(
        "--smoke-height",
        type=int,
        default=448,
    )

    parser.add_argument(
        "--smoke-width",
        type=int,
        default=644,
    )

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
