from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from event_jepa.config import EventJEPAConfig
from event_jepa.dataset import EventJEPADataset
from event_jepa.model import EventJEPA


CKPT = Path(
    "runs/event_jepa_dsec_seed0/"
    "step_00100000.pt"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MAX_BATCHES = 100
BATCH_SIZE = 16


def cosine(a, b):
    return F.cosine_similarity(
        a.float(),
        b.float(),
        dim=-1,
    ).mean().item()


state = torch.load(
    CKPT,
    map_location="cpu",
    weights_only=False,
)

cfg = EventJEPAConfig(**state["config"])

model = EventJEPA(
    cfg.embed_dim,
    cfg.num_heads,
    cfg.encoder_layers,
    cfg.predictor_layers,
    cfg.n_tokens,
    cfg.max_positions,
    cfg.max_context_frames,
    len(cfg.horizons),
)

model.online_encoder.load_state_dict(
    state["online_encoder"],
    strict=True,
)

model.target_encoder.load_state_dict(
    state["target_encoder"],
    strict=True,
)

model.predictor.load_state_dict(
    state["predictor"],
    strict=True,
)

model.to(DEVICE)
model.eval()


dataset = EventJEPADataset(
    cfg.data_root,
    "test",
    context_frames=cfg.context_frames,
    horizons=tuple(cfg.horizons),
    n_tokens=cfg.n_tokens,
    embed_dim=cfg.embed_dim,
    timestamp_scale=cfg.timestamp_scale,
)

generator = torch.Generator()
generator.manual_seed(0)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    generator=generator,
)


metrics = {
    "normal_pred_future": [],
    "pred_vs_last": [],
    "last_vs_future": [],
    "wrong_target": [],
    "repeat_last_context": [],
    "reverse_context": [],
    "zero_delta_t": [],
    "x4_delta_t": [],
}


@torch.no_grad()
def encode_target(x):
    # x: [B,K,N,D]
    B, K, N, D = x.shape

    z = model.target_encoder(
        x.reshape(B * K, 1, N, D)
    )

    return z.reshape(B, K, N, D)


@torch.no_grad()
def predict(context, delta_t):
    memory = model.online_encoder(context)

    return model.predictor(
        memory,
        delta_t,
    )


with torch.no_grad():

    for batch_idx, batch in enumerate(loader):

        if batch_idx >= MAX_BATCHES:
            break

        context = batch["context"].to(
            DEVICE,
            non_blocking=True,
        )

        target = batch["target"].to(
            DEVICE,
            non_blocking=True,
        )

        delta_t = batch["delta_t"].to(
            DEVICE,
            non_blocking=True,
        )

        # --------------------------------------------------
        # Normal prediction
        # --------------------------------------------------
        pred = predict(
            context,
            delta_t,
        )

        z_future = encode_target(
            target
        )

        metrics[
            "normal_pred_future"
        ].append(
            cosine(pred, z_future)
        )

        # --------------------------------------------------
        # Is prediction just the last observed frame?
        # --------------------------------------------------
        last_frame = context[:, -1:].contiguous()

        z_last = encode_target(
            last_frame
        )

        metrics[
            "pred_vs_last"
        ].append(
            cosine(pred, z_last)
        )

        metrics[
            "last_vs_future"
        ].append(
            cosine(z_last, z_future)
        )

        # --------------------------------------------------
        # Wrong-target negative control
        # --------------------------------------------------
        if context.shape[0] > 1:

            wrong_future = torch.roll(
                z_future,
                shifts=1,
                dims=0,
            )

            metrics[
                "wrong_target"
            ].append(
                cosine(pred, wrong_future)
            )

        # --------------------------------------------------
        # Repeat last context frame Tc times
        # --------------------------------------------------
        repeated = (
            context[:, -1:]
            .expand_as(context)
            .contiguous()
        )

        pred_repeat = predict(
            repeated,
            delta_t,
        )

        metrics[
            "repeat_last_context"
        ].append(
            cosine(
                pred_repeat,
                z_future,
            )
        )

        # --------------------------------------------------
        # Reverse temporal order
        # --------------------------------------------------
        reversed_context = torch.flip(
            context,
            dims=[1],
        )

        pred_reverse = predict(
            reversed_context,
            delta_t,
        )

        metrics[
            "reverse_context"
        ].append(
            cosine(
                pred_reverse,
                z_future,
            )
        )

        # --------------------------------------------------
        # Remove timing information
        # --------------------------------------------------
        pred_zero_dt = predict(
            context,
            torch.zeros_like(delta_t),
        )

        metrics[
            "zero_delta_t"
        ].append(
            cosine(
                pred_zero_dt,
                z_future,
            )
        )

        # --------------------------------------------------
        # Wrong time: pretend target is 4x farther away
        # --------------------------------------------------
        pred_x4_dt = predict(
            context,
            delta_t * 4.0,
        )

        metrics[
            "x4_delta_t"
        ].append(
            cosine(
                pred_x4_dt,
                z_future,
            )
        )


print()
print("=" * 72)
print("EVENT-JEPA SHORTCUT AUDIT")
print("=" * 72)

for name, values in metrics.items():

    if not values:
        continue

    x = torch.tensor(values)

    print(
        f"{name:24s} "
        f"mean={x.mean().item():.6f} "
        f"std={x.std().item():.6f}"
    )

print()
print("Done.")
