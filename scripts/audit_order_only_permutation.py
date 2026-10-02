from itertools import permutations
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from event_jepa.config import EventJEPAConfig
from event_jepa.dataset import EventJEPADataset
from train_jepa import build_model


CHECKPOINTS = {
    "V1-MH124": Path(
        "runs/event_jepa_v1_mh124_seed0_30k/latest.pt"
    ),
    "Order-Only-MH124": Path(
        "runs/event_jepa_v2_order_mh124_seed0_30k/latest.pt"
    ),
}

TRAINED_NEGATIVE = (1, 2, 0)
IDENTITY = (0, 1, 2)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def cosine_distance(
    prediction,
    target,
):
    """
    [B,K,N,D] -> [B]
    """
    return (
        1.0
        - F.cosine_similarity(
            prediction.float(),
            target.float(),
            dim=-1,
        )
    ).mean(dim=(1, 2))


for model_name, ckpt_path in CHECKPOINTS.items():

    print()
    print("=" * 110)
    print(model_name)
    print("checkpoint:", ckpt_path)
    print("=" * 110)

    state = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )

    cfg = EventJEPAConfig(
        **state["config"]
    )

    assert cfg.context_frames == 4
    assert tuple(cfg.horizons) == (1, 2, 4)

    model = build_model(
        cfg
    ).to(device)

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

    model.eval()

    dataset = EventJEPADataset(
        cfg.data_root,
        "test",
        cfg.context_frames,
        cfg.horizons,
        cfg.n_tokens,
        cfg.embed_dim,
        cfg.timestamp_scale,
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    perms = list(
        permutations(range(3))
    )

    distances = {
        p: []
        for p in perms
    }

    with torch.no_grad():

        for batch in tqdm(loader):

            context = batch[
                "context"
            ].to(
                device,
                non_blocking=True,
            )

            target = batch[
                "target"
            ].to(
                device,
                non_blocking=True,
            )

            delta_t = batch[
                "delta_t"
            ].to(
                device,
                non_blocking=True,
            )

            B, K, N, D = target.shape

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=(
                    device.type == "cuda"
                ),
            ):

                encoded_target = (
                    model.target_encoder(
                        target.reshape(
                            B * K,
                            1,
                            N,
                            D,
                        )
                    )
                    .reshape(
                        B,
                        K,
                        N,
                        D,
                    )
                )

                for perm in perms:

                    index = list(perm) + [3]

                    corrupted = context[
                        :,
                        index,
                    ]

                    memory = (
                        model.online_encoder(
                            corrupted
                        )
                    )

                    prediction = (
                        model.predictor(
                            memory,
                            delta_t,
                        )
                    )

                    d = cosine_distance(
                        prediction,
                        encoded_target,
                    )

                    distances[
                        perm
                    ].append(
                        d.cpu()
                    )

    distances = {
        p: torch.cat(v)
        for p, v in distances.items()
    }

    correct = distances[
        IDENTITY
    ]

    print()
    print(
        f"{'permutation':16s}"
        f"{'type':20s}"
        f"{'distance':>14s}"
        f"{'gap':>14s}"
        f"{'median-gap':>14s}"
        f"{'wrong>wright %':>16s}"
    )

    print("-" * 96)

    for perm in perms:

        d = distances[perm]

        gap = d - correct

        if perm == IDENTITY:
            label = "identity"
        elif perm == TRAINED_NEGATIVE:
            label = "TRAINED NEGATIVE"
        elif perm == (2, 1, 0):
            label = "reverse"
        else:
            label = "unseen"

        if perm == IDENTITY:
            win = float("nan")
        else:
            win = (
                (gap > 0)
                .float()
                .mean()
                .item()
                * 100.0
            )

        print(
            f"{str(perm):16s}"
            f"{label:20s}"
            f"{d.mean().item():14.6f}"
            f"{gap.mean().item():14.6f}"
            f"{gap.median().item():14.6f}"
            f"{win:16.2f}"
        )

    #
    # Generalization summary:
    # exclude identity and trained negative.
    #
    unseen = [
        p
        for p in perms
        if p not in {
            IDENTITY,
            TRAINED_NEGATIVE,
        }
    ]

    unseen_gap = torch.stack(
        [
            (
                distances[p]
                - correct
            )
            for p in unseen
        ],
        dim=0,
    )

    print()
    print(
        "trained-negative mean gap:",
        float(
            (
                distances[
                    TRAINED_NEGATIVE
                ]
                - correct
            ).mean()
        )
    )

    print(
        "unseen-permutation mean gap:",
        float(
            unseen_gap.mean()
        )
    )

    print(
        "unseen positive-gap rate:",
        float(
            (
                unseen_gap > 0
            )
            .float()
            .mean()
            * 100
        ),
        "%",
    )

print()
print(
    "ORDER PERMUTATION GENERALIZATION AUDIT: PASS"
)
