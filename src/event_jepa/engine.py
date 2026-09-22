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


def cosine_learning_rate(
    step: int,
    warmup_steps: int,
    total_steps: int,
    peak: float,
    minimum: float,
) -> float:
    if step < warmup_steps:
        return peak * step / max(warmup_steps, 1)
    progress = min(
        (step - warmup_steps) / max(total_steps - warmup_steps, 1),
        1.0,
    )
    return minimum + 0.5 * (peak - minimum) * (
        1.0 + math.cos(math.pi * progress)
    )


@torch.no_grad()
def validate(model, loader, device, precision, max_batches=None):
    model.eval()
    totals = {
        "loss": 0.0,
        "representation_std": 0.0,
        "mean_cosine": 0.0,
    }
    count = 0
    for batch in loader:
        with _autocast(device, precision):
            output = model(
                batch["context"].to(device),
                batch["target"].to(device),
                batch["delta_t"].to(device),
            )
        for key in totals:
            totals[key] += float(output[key])
        count += 1
        if max_batches is not None and count >= max_batches:
            break
    model.train()
    return {key: value / max(count, 1) for key, value in totals.items()}
