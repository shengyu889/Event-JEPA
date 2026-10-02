from itertools import permutations
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dsec_flow_probe.dataset import DSECSequenceFlowDataset
from dsec_flow_probe.backbone import FrozenFlowBackbone
from dsec_flow_probe.train_full_memory import (
    TemporalPatchLinearFlowHead,
    encode_full_memory,
)
from utils import FlowMetrics


ROOT = Path(
    "/home/tom/event-jepa/datasets/DSEC"
)

VAL_SEQUENCES = [
    "thun_00_a",
    "zurich_city_02_d",
]

METHODS = {
    "V1": {
        "backbone": Path(
            "runs/event_jepa_v1_mh124_seed0_30k/latest.pt"
        ),
        "flow_head": Path(
            "runs/flow_fullmem_v1_mh124_seed0_5k/final.pt"
        ),
    },

    "Order-Only": {
        "backbone": Path(
            "runs/event_jepa_v2_order_mh124_seed0_30k/latest.pt"
        ),
        "flow_head": Path(
            "runs/flow_v2_order_fullmem_seed0_5k/final.pt"
        ),
    },
}


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

dataset = DSECSequenceFlowDataset(
    ROOT,
    VAL_SEQUENCES,
    context_frames=4,
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)


@torch.no_grad()
def evaluate(
    backbone,
    head,
    perm,
):
    metrics = FlowMetrics(
        n_vals=[1, 2, 3],
        device=torch.device("cpu"),
    )

    abs_sum = 0.0
    abs_count = 0.0

    for batch in tqdm(
        loader,
        leave=False,
    ):
        context = batch[
            "context"
        ].to(
            device,
            non_blocking=True,
        )

        target = batch[
            "flow"
        ].to(
            device,
            non_blocking=True,
        )

        # Keep current frame t fixed.
        # Only permute [t-3,t-2,t-1].
        index = list(perm) + [3]

        corrupted = context[
            :,
            index,
        ]

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=(
                device.type == "cuda"
            ),
        ):
            z = encode_full_memory(
                backbone,
                corrupted,
            )

        pred = head(
            z.float()
        )

        mask = target[:, 2:3]
        gt = target[:, :2]

        abs_sum += float(
            (
                (pred - gt).abs()
                * mask
            ).sum().item()
        )

        abs_count += float(
            mask.sum().item()
            * 2.0
        )

        metrics.update(
            pred.detach(),
            target,
        )

    fm = metrics.compute()

    return {
        "l1":
            abs_sum
            / max(abs_count, 1.0),

        "epe": fm["epe"],
        "ae": fm["ae"],
        "1pe": fm["1pe"],
        "2pe": fm["2pe"],
        "3pe": fm["3pe"],
    }


perms = list(
    permutations(range(3))
)

IDENTITY = (0, 1, 2)
TRAINED_NEG = (1, 2, 0)
REVERSE = (2, 1, 0)


for method_name, paths in METHODS.items():

    print()
    print("=" * 110)
    print(method_name)
    print("=" * 110)

    backbone = FrozenFlowBackbone(
        mode="jepa",
        checkpoint=paths["backbone"],
    ).to(device)

    backbone.eval()
    backbone.requires_grad_(False)

    head = TemporalPatchLinearFlowHead(
        context_frames=4,
        embed_dim=384,
    ).to(device)

    head_state = torch.load(
        paths["flow_head"],
        map_location="cpu",
        weights_only=False,
    )

    head.load_state_dict(
        head_state["head"],
        strict=True,
    )

    head.eval()

    results = {}

    for perm in perms:

        print(
            "evaluating",
            perm,
        )

        results[perm] = evaluate(
            backbone,
            head,
            perm,
        )

    base = results[
        IDENTITY
    ]

    print()
    print(
        f"{'permutation':16s}"
        f"{'type':18s}"
        f"{'EPE':>10s}"
        f"{'ΔEPE':>10s}"
        f"{'ΔEPE%':>10s}"
        f"{'AE':>10s}"
        f"{'1PE':>10s}"
        f"{'2PE':>10s}"
        f"{'3PE':>10s}"
    )

    print("-" * 104)

    for perm in perms:

        r = results[
            perm
        ]

        delta = (
            r["epe"]
            - base["epe"]
        )

        pct = (
            100.0
            * delta
            / base["epe"]
        )

        if perm == IDENTITY:
            label = "identity"

        elif perm == TRAINED_NEG:
            label = "trained cyclic"

        elif perm == REVERSE:
            label = "unseen reverse"

        else:
            label = "unseen"

        print(
            f"{str(perm):16s}"
            f"{label:18s}"
            f"{r['epe']:10.5f}"
            f"{delta:10.5f}"
            f"{pct:10.2f}"
            f"{r['ae']:10.5f}"
            f"{r['1pe']:10.5f}"
            f"{r['2pe']:10.5f}"
            f"{r['3pe']:10.5f}"
        )

print()
print(
    "FULL-MEMORY FLOW PERMUTATION AUDIT: PASS"
)
