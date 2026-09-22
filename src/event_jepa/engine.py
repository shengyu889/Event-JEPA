import math
from contextlib import nullcontext

import torch


def ema_momentum(
    step: int,
    total_steps: int,
    start: float,
    end: float,
) -> float:
    progress = min(max(step / max(total_steps, 1), 0.0), 1.0)
    return end - (end - start) * (
        math.cos(math.pi * progress) + 1.0
    ) / 2.0


def _autocast(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def train_micro_step(
    model,
    batch,
    optimizer,
    scaler,
    device,
    precision,
    grad_clip_norm,
    ema_value,
    loss_divisor,
    should_step,
):
    context = batch["context"].to(device, non_blocking=True)
    target = batch["target"].to(device, non_blocking=True)
    delta_t = batch["delta_t"].to(device, non_blocking=True)
    with _autocast(device, precision):
        output = model(context, target, delta_t)
        scaled_loss = output["loss"] / loss_divisor

    if scaler is None:
        scaled_loss.backward()
    else:
        scaler.scale(scaled_loss).backward()

    optimizer_stepped = False
    grad_norm = 0.0
    if should_step:
        trainable_parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        if scaler is None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                grad_clip_norm,
            ).item()
            optimizer.step()
            optimizer_stepped = True
        else:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                grad_clip_norm,
            ).item()
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer_stepped = scaler.get_scale() >= previous_scale
        optimizer.zero_grad(set_to_none=True)
        if optimizer_stepped:
            core_model = model.module if hasattr(model, "module") else model
            core_model.update_target(ema_value)

    return {
        "loss": float(output["loss"].detach()),
        "representation_std": float(output["representation_std"]),
        "mean_cosine": float(output["mean_cosine"]),
        "grad_norm": grad_norm,
        "optimizer_stepped": optimizer_stepped,
    }
