import math

import torch

from event_jepa.config import load_config
from event_jepa.dataset import EventJEPADataset
from train_jepa import build_model


CFG = "configs/event_jepa_v2_full_smoke.yaml"


def grad_norm(parameters):
    total = 0.0

    for p in parameters:
        if p.grad is None:
            continue

        g = p.grad.detach().float()

        total += float(
            g.pow(2).sum()
        )

    return math.sqrt(total)


cfg = load_config(CFG)

ds = EventJEPADataset(
    cfg.data_root,
    "train",
    cfg.context_frames,
    cfg.horizons,
    cfg.n_tokens,
    cfg.embed_dim,
    cfg.timestamp_scale,
)

samples = [
    ds[0],
    ds[1],
]

context = torch.stack(
    [x["context"] for x in samples]
).cuda()

target = torch.stack(
    [x["target"] for x in samples]
).cuda()

delta_t = torch.stack(
    [x["delta_t"] for x in samples]
).cuda()


def make_model():
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(
        cfg.seed
    )

    return build_model(
        cfg
    ).cuda().train()


objectives = [
    ("future", "future_loss", 1.0),
    (
        "residual_raw",
        "residual_loss",
        1.0,
    ),
    (
        "residual_weighted",
        "residual_loss",
        cfg.residual_weight,
    ),
    (
        "order_raw",
        "order_loss",
        1.0,
    ),
    (
        "order_weighted",
        "order_loss",
        cfg.order_weight,
    ),
    ("total", "loss", 1.0),
]


print("=" * 92)
print("V2 GRADIENT CONTRIBUTION AUDIT")
print("=" * 92)

print(
    f"{'objective':20s}"
    f"{'loss':>14s}"
    f"{'enc_grad':>16s}"
    f"{'pred_grad':>16s}"
)

print("-" * 92)


for name, key, weight in objectives:

    model = make_model()

    with torch.autocast(
        "cuda",
        dtype=torch.bfloat16,
    ):
        out = model(
            context,
            target,
            delta_t,
        )

        loss = (
            out[key]
            * weight
        )

    model.zero_grad(
        set_to_none=True
    )

    loss.backward()

    enc_norm = grad_norm(
        model.online_encoder.parameters()
    )

    pred_norm = grad_norm(
        model.predictor.parameters()
    )

    print(
        f"{name:20s}"
        f"{float(loss):14.6f}"
        f"{enc_norm:16.6f}"
        f"{pred_norm:16.6f}"
    )


print()
print("=" * 92)
print("RAW LOSS VALUES")
print("=" * 92)

model = make_model()

with torch.no_grad():
    with torch.autocast(
        "cuda",
        dtype=torch.bfloat16,
    ):
        out = model(
            context,
            target,
            delta_t,
        )

for key in [
    "loss",
    "future_loss",
    "residual_loss",
    "order_loss",
    "order_gap",
]:
    print(
        f"{key:20s}",
        float(out[key])
    )

print()
print(
    "V2 GRADIENT BALANCE AUDIT: PASS"
)
