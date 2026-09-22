import argparse
import os
import random
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from event_jepa.checkpoint import load_checkpoint, save_checkpoint
from event_jepa.config import EventJEPAConfig, load_config
from event_jepa.dataset import EventJEPADataset
from event_jepa.engine import (
    cosine_learning_rate,
    ema_momentum,
    train_micro_step,
    validate,
)
from event_jepa.model import EventJEPA


def distributed_context() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("distributed Event-JEPA training requires CUDA")
        torch.distributed.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )
    return rank, world_size, local_rank, device


def build_model(config: EventJEPAConfig) -> EventJEPA:
    return EventJEPA(
        config.embed_dim,
        config.num_heads,
        config.encoder_layers,
        config.predictor_layers,
        config.n_tokens,
        config.max_positions,
        config.max_context_frames,
        max_horizons=len(config.horizons),
    )


def _dataset(config: EventJEPAConfig, split: str) -> EventJEPADataset:
    return EventJEPADataset(
        config.data_root,
        split,
        config.context_frames,
        config.horizons,
        config.n_tokens,
        config.embed_dim,
        config.timestamp_scale,
    )


def build_loaders(config: EventJEPAConfig, rank: int, world_size: int):
    train_dataset = _dataset(config, "train")
    valid_dataset = _dataset(config, "test")
    if len(train_dataset) == 0 or len(valid_dataset) == 0:
        raise ValueError("no valid temporal samples found")
    train_sampler = None
    valid_sampler = None
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=config.seed,
            drop_last=False,
        )
    generator = torch.Generator().manual_seed(config.seed + rank)
    common = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
        "generator": generator,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        **common,
    )
    valid_loader = DataLoader(
        valid_dataset,
        shuffle=False,
        sampler=valid_sampler,
        **common,
    )
    return train_loader, valid_loader, train_sampler


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _core_model(model):
    return model.module if hasattr(model, "module") else model


def run_training(
    config_path,
    resume=None,
    data_root=None,
    max_steps=None,
) -> int:
    config = load_config(config_path)
    if data_root is not None:
        config = replace(config, data_root=Path(data_root))
    rank, world_size, local_rank, device = distributed_context()
    _seed_everything(config.seed + rank)
    writer = None
    try:
        train_loader, valid_loader, train_sampler = build_loaders(
            config,
            rank,
            world_size,
        )
        model = build_model(config).to(device)
        if world_size > 1:
            model = DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
        core = _core_model(model)
        trainable = list(core.online_encoder.parameters()) + list(
            core.predictor.parameters()
        )
        optimizer = torch.optim.AdamW(
            trainable,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scaler = None
        if device.type == "cuda" and config.precision == "fp16":
            scaler = torch.amp.GradScaler("cuda")

        step = 0
        if resume is not None:
            state = load_checkpoint(resume, core, optimizer, scaler)
            step = int(state["step"])
        target_steps = (
            config.total_steps if max_steps is None else int(max_steps)
        )
        if target_steps < step:
            raise ValueError("max_steps cannot be smaller than resumed step")

        if rank == 0:
            config.output_dir.mkdir(parents=True, exist_ok=True)
            if SummaryWriter is not None:
                writer = SummaryWriter(
                    log_dir=str(config.output_dir / "tensorboard")
                )
        optimizer.zero_grad(set_to_none=True)
        micro_step = 0
        epoch = 0
        last_validation_step = -1
        last_save_step = -1
        while step < target_steps:
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            for batch in train_loader:
                should_step = (
                    (micro_step + 1) % config.grad_accum_steps == 0
                )
                lr = cosine_learning_rate(
                    step,
                    config.warmup_steps,
                    config.total_steps,
                    config.learning_rate,
                    config.min_learning_rate,
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr
                sync_context = nullcontext()
                if world_size > 1 and not should_step:
                    sync_context = model.no_sync()
                with sync_context:
                    metrics = train_micro_step(
                        model,
                        batch,
                        optimizer,
                        scaler,
                        device,
                        config.precision,
                        config.grad_clip_norm,
                        ema_momentum(
                            step,
                            config.total_steps,
                            config.ema_start,
                            config.ema_end,
                        ),
                        config.grad_accum_steps,
                        should_step,
                    )
                micro_step += 1
                if metrics["optimizer_stepped"]:
                    step += 1
                if (
                    rank == 0
                    and metrics["optimizer_stepped"]
                    and step % config.log_every == 0
                ):
                    for key, value in metrics.items():
                        if key != "optimizer_stepped":
                            if writer is not None:
                                writer.add_scalar(
                                    f"train/{key}", value, step
                                )
                    if writer is not None:
                        writer.add_scalar("train/learning_rate", lr, step)
                        writer.add_scalar(
                            "train/mean_delta_t",
                            float(batch["delta_t"].mean()),
                            step,
                        )
                if (
                    rank == 0
                    and step > 0
                    and step % config.validate_every == 0
                    and step != last_validation_step
                ):
                    validation = validate(
                        core,
                        valid_loader,
                        device,
                        config.precision,
                    )
                    for key, value in validation.items():
                        if writer is not None:
                            writer.add_scalar(f"valid/{key}", value, step)
                    last_validation_step = step
                if (
                    rank == 0
                    and step > 0
                    and step % config.save_every == 0
                    and step != last_save_step
                ):
                    save_checkpoint(
                        config.output_dir / f"step_{step:08d}.pt",
                        core,
                        optimizer,
                        scaler,
                        config,
                        step,
                    )
                    save_checkpoint(
                        config.output_dir / "latest.pt",
                        core,
                        optimizer,
                        scaler,
                        config,
                        step,
                    )
                    last_save_step = step
                if step >= target_steps:
                    break
            epoch += 1
        if rank == 0 and step != last_save_step:
            save_checkpoint(
                config.output_dir / "latest.pt",
                core,
                optimizer,
                scaler,
                config,
                step,
            )
        return step
    finally:
        if writer is not None:
            writer.close()
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            torch.distributed.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Event-JEPA on precomputed GEP event tokens."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--data-root")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    run_training(args.config, args.resume, args.data_root, args.max_steps)


if __name__ == "__main__":
    main()
