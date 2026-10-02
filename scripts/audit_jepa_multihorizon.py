from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from event_jepa.config import EventJEPAConfig
from event_jepa.dataset import EventJEPADataset
from event_jepa.model import EventJEPA


CKPT = Path(
    "runs/event_jepa_dsec_mh148_seed0/"
    "step_00030000.pt"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 16
MAX_BATCHES = 100


state = torch.load(
    CKPT,
    map_location="cpu",
    weights_only=False,
)

cfg = EventJEPAConfig(**state["config"])

print("checkpoint:", CKPT)
print("horizons :", cfg.horizons)
print("device   :", DEVICE)


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


@torch.no_grad()
def encode_targets(target):
    # [B,K,N,D] -> [B,K,N,D]
    B, K, N, D = target.shape

    z = model.target_encoder(
        target.reshape(
            B * K,
            1,
            N,
            D,
        )
    )

    return z.reshape(
        B,
        K,
        N,
        D,
    )


@torch.no_grad()
def predict(context, delta_t):
    memory = model.online_encoder(context)
    return model.predictor(memory, delta_t)


def horizon_cosines(a, b):
    """
    a,b: [B,K,N,D]
    return [K]
    """
    c = F.cosine_similarity(
        a.float(),
        b.float(),
        dim=-1,
    )

    # [B,K,N] -> [K]
    return c.mean(dim=(0, 2)).cpu()


metrics = defaultdict(list)
delta_values = []


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

        delta_values.append(
            delta_t.mean(dim=0).cpu()
        )

        # ==========================================
        # Correct normal prediction
        # ==========================================
        z_future = encode_targets(target)

        pred_normal = predict(
            context,
            delta_t,
        )

        metrics["normal"].append(
            horizon_cosines(
                pred_normal,
                z_future,
            )
        )

        # ==========================================
        # Last observed frame in target latent space
        # ==========================================
        last = context[:, -1:]

        z_last = model.target_encoder(last)

        # [B,1,N,D] -> [B,K,N,D]
        z_last_expanded = z_last[:, None] if z_last.ndim == 3 else z_last

        if z_last_expanded.ndim == 4 and z_last_expanded.shape[1] == 1:
            z_last_expanded = z_last_expanded.expand(
                -1,
                len(cfg.horizons),
                -1,
                -1,
            )

        metrics["pred_vs_last"].append(
            horizon_cosines(
                pred_normal,
                z_last_expanded,
            )
        )

        metrics["last_vs_future"].append(
            horizon_cosines(
                z_last_expanded,
                z_future,
            )
        )

        # ==========================================
        # Repeat last frame Tc times
        # ==========================================
        repeat_last = (
            context[:, -1:]
            .expand_as(context)
            .contiguous()
        )

        pred_repeat = predict(
            repeat_last,
            delta_t,
        )

        metrics["repeat_last"].append(
            horizon_cosines(
                pred_repeat,
                z_future,
            )
        )

        # ==========================================
        # Reverse context temporal order
        # ==========================================
        reversed_context = torch.flip(
            context,
            dims=[1],
        )

        pred_reverse = predict(
            reversed_context,
            delta_t,
        )

        metrics["reverse_context"].append(
            horizon_cosines(
                pred_reverse,
                z_future,
            )
        )

        # ==========================================
        # Remove continuous-time value
        # horizon-rank embedding remains intact
        # ==========================================
        pred_zero_dt = predict(
            context,
            torch.zeros_like(delta_t),
        )

        metrics["zero_dt"].append(
            horizon_cosines(
                pred_zero_dt,
                z_future,
            )
        )

        # ==========================================
        # Give every horizon the same 50ms-ish time
        # horizon-rank embedding remains intact
        # ==========================================
        same_dt = (
            delta_t[:, :1]
            .expand_as(delta_t)
            .contiguous()
        )

        pred_same_dt = predict(
            context,
            same_dt,
        )

        metrics["same_dt"].append(
            horizon_cosines(
                pred_same_dt,
                z_future,
            )
        )

        # ==========================================
        # Reverse continuous times:
        # [50,200,400] -> [400,200,50]
        #
        # IMPORTANT:
        # horizon-rank embeddings stay unchanged.
        # Thus this specifically tests delta_t.
        # ==========================================
        reversed_dt = torch.flip(
            delta_t,
            dims=[1],
        )

        pred_reversed_dt = predict(
            context,
            reversed_dt,
        )

        metrics["reversed_dt"].append(
            horizon_cosines(
                pred_reversed_dt,
                z_future,
            )
        )

        # ==========================================
        # Wrong sample target
        # ==========================================
        if context.shape[0] > 1:
            wrong_future = torch.roll(
                z_future,
                shifts=1,
                dims=0,
            )

            metrics["wrong_target"].append(
                horizon_cosines(
                    pred_normal,
                    wrong_future,
                )
            )


means = {}

for name, rows in metrics.items():
    x = torch.stack(rows)
    means[name] = x.mean(dim=0)


dt = torch.stack(delta_values).mean(dim=0)


print()
print("=" * 92)
print("EVENT-JEPA MULTI-HORIZON SHORTCUT AUDIT")
print("=" * 92)

header = f"{'metric':20s}"

for h, t in zip(cfg.horizons, dt):
    header += f" | h={h:<2d} ({t.item()*1000:6.1f}ms)"

print(header)
print("-" * len(header))

order = [
    "normal",
    "pred_vs_last",
    "last_vs_future",
    "repeat_last",
    "reverse_context",
    "zero_dt",
    "same_dt",
    "reversed_dt",
    "wrong_target",
]

for name in order:
    if name not in means:
        continue

    line = f"{name:20s}"

    for value in means[name]:
        line += f" | {value.item():14.6f}"

    print(line)


print()
print("=" * 92)
print("DROP RELATIVE TO NORMAL")
print("=" * 92)

for name in [
    "repeat_last",
    "reverse_context",
    "zero_dt",
    "same_dt",
    "reversed_dt",
    "wrong_target",
]:
    if name not in means:
        continue

    drop = means["normal"] - means[name]

    line = f"{name:20s}"

    for value in drop:
        line += f" | {value.item():14.6f}"

    print(line)

print()
print("Audit complete.")
