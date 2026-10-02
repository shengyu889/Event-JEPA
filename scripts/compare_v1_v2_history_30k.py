from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dsec_flow_probe.dataset import DSECSequenceFlowDataset
from dsec_flow_probe.backbone import FrozenFlowBackbone


ROOT = Path(
    "/home/tom/event-jepa/datasets/DSEC"
)

VAL = [
    "thun_00_a",
    "zurich_city_02_d",
]

CHECKPOINTS = {
    "V1-MH124": Path(
        "runs/event_jepa_v1_mh124_seed0_30k/latest.pt"
    ),
    "V2-Full-MH124": Path(
        "runs/event_jepa_v2_full_mh124_seed0_30k/latest.pt"
    ),
}


ds = DSECSequenceFlowDataset(
    ROOT,
    VAL,
    context_frames=4,
)

loader = DataLoader(
    ds,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

device = torch.device("cuda")


def corrupt(x, mode):

    if mode == "repeat_last":
        return (
            x[:, -1:]
            .expand(
                -1, 4, -1, -1
            )
            .contiguous()
        )

    if mode == "reverse_past":
        return x[:, [2,1,0,3]]

    if mode == "shuffle_past":
        return x[:, [1,2,0,3]]

    if mode == "zero_past":
        y = x.clone()
        y[:, :3] = 0
        return y

    raise ValueError(mode)


variants = [
    "repeat_last",
    "reverse_past",
    "shuffle_past",
    "zero_past",
]


for model_name, ckpt in CHECKPOINTS.items():

    print()
    print("=" * 82)
    print(model_name)
    print("checkpoint:", ckpt)
    print("=" * 82)

    state = torch.load(
        ckpt,
        map_location="cpu",
        weights_only=False,
    )

    print(
        "step:",
        state["step"],
        "horizons:",
        state["config"]["horizons"],
        "residual:",
        state["config"].get(
            "residual_weight",
            0.0,
        ),
        "order:",
        state["config"].get(
            "order_weight",
            0.0,
        ),
    )

    model = FrozenFlowBackbone(
        mode="jepa",
        checkpoint=ckpt,
    ).to(device)

    model.eval()

    stats = {
        v: {
            "cos": [],
            "l2": [],
        }
        for v in variants
    }

    with torch.no_grad():

        for batch in tqdm(loader):

            x = batch[
                "context"
            ].to(
                device,
                non_blocking=True,
            )

            with torch.autocast(
                "cuda",
                dtype=torch.bfloat16,
            ):
                z0 = model(x)

                for v in variants:

                    zv = model(
                        corrupt(x, v)
                    )

                    cos = (
                        F.cosine_similarity(
                            z0.float(),
                            zv.float(),
                            dim=-1,
                        )
                        .mean(dim=-1)
                    )

                    rel_l2 = (
                        (
                            zv.float()
                            - z0.float()
                        )
                        .flatten(1)
                        .norm(dim=1)
                        /
                        (
                            z0.float()
                            .flatten(1)
                            .norm(dim=1)
                            + 1e-8
                        )
                    )

                    stats[v][
                        "cos"
                    ].append(
                        cos.cpu()
                    )

                    stats[v][
                        "l2"
                    ].append(
                        rel_l2.cpu()
                    )

    print()
    print(
        f"{'variant':18s}"
        f"{'cosine':>14s}"
        f"{'1-cos':>14s}"
        f"{'rel-L2':>14s}"
    )

    print("-" * 62)

    for v in variants:

        cos = torch.cat(
            stats[v]["cos"]
        ).mean()

        l2 = torch.cat(
            stats[v]["l2"]
        ).mean()

        print(
            f"{v:18s}"
            f"{cos.item():14.6f}"
            f"{(1-cos).item():14.6f}"
            f"{l2.item():14.6f}"
        )

print()
print(
    "V1/V2 HISTORY REPRESENTATION COMPARISON: PASS"
)
