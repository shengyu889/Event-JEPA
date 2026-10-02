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
    "V1": Path(
        "runs/event_jepa_v1_mh124_seed0_30k/latest.pt"
    ),
    "V2": Path(
        "runs/event_jepa_v2_full_mh124_seed0_30k/latest.pt"
    ),
}

# corruption permutation and its inverse.
#
# normal:
# [A,B,C,t]
#
# reverse:
# [C,B,A,t]
#
# To compare representation of the SAME physical
# frame after encoding, reverse the outputs back.
PERMS = {
    "reverse": {
        "forward": [2,1,0,3],
        "inverse": [2,1,0,3],
    },

    # [A,B,C,t] -> [B,C,A,t]
    # inverse alignment -> [A,B,C,t]
    "shuffle": {
        "forward": [1,2,0,3],
        "inverse": [2,0,1,3],
    },
}


dataset = DSECSequenceFlowDataset(
    ROOT,
    VAL,
    context_frames=4,
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

device = torch.device("cuda")


def similarity(a, b):
    """
    a,b: [B,N,D]
    """

    cos = (
        F.cosine_similarity(
            a.float(),
            b.float(),
            dim=-1,
        )
        .mean(dim=-1)
    )

    rel_l2 = (
        (a.float() - b.float())
        .flatten(1)
        .norm(dim=1)
        /
        (
            a.float()
            .flatten(1)
            .norm(dim=1)
            + 1e-8
        )
    )

    return (
        cos.cpu(),
        rel_l2.cpu(),
    )


for model_name, ckpt in CHECKPOINTS.items():

    print()
    print("=" * 100)
    print(model_name)
    print("checkpoint:", ckpt)
    print("=" * 100)

    backbone = FrozenFlowBackbone(
        mode="jepa",
        checkpoint=ckpt,
    ).to(device)

    backbone.eval()

    stats = {}

    for corruption in PERMS:
        stats[corruption] = {
            slot: {
                "cos": [],
                "l2": [],
            }
            for slot in [
                "t-3",
                "t-2",
                "t-1",
                "t",
                "ALL",
            ]
        }

    with torch.no_grad():

        for batch in tqdm(loader):

            x = batch["context"].to(
                device,
                non_blocking=True,
            )

            B, T, N, D = x.shape

            assert T == 4

            with torch.autocast(
                "cuda",
                dtype=torch.bfloat16,
            ):
                #
                # IMPORTANT:
                # use complete encoder memory,
                # not FrozenFlowBackbone.forward()
                # which returns only final slot.
                #
                z_normal = backbone.encoder(x)

            z_normal = z_normal.reshape(
                B,
                4,
                N,
                D,
            )

            for name, perm in PERMS.items():

                x_bad = x[
                    :,
                    perm["forward"],
                ]

                with torch.autocast(
                    "cuda",
                    dtype=torch.bfloat16,
                ):
                    z_bad = backbone.encoder(
                        x_bad
                    )

                z_bad = z_bad.reshape(
                    B,
                    4,
                    N,
                    D,
                )

                #
                # Re-align outputs by physical frame
                # identity, not by temporal slot.
                #
                z_bad = z_bad[
                    :,
                    perm["inverse"],
                ]

                slot_names = [
                    "t-3",
                    "t-2",
                    "t-1",
                    "t",
                ]

                for idx, slot in enumerate(
                    slot_names
                ):
                    cos, l2 = similarity(
                        z_normal[:, idx],
                        z_bad[:, idx],
                    )

                    stats[name][slot][
                        "cos"
                    ].append(cos)

                    stats[name][slot][
                        "l2"
                    ].append(l2)

                #
                # Aligned full memory.
                #
                cos, l2 = similarity(
                    z_normal.reshape(
                        B,
                        4 * N,
                        D,
                    ),
                    z_bad.reshape(
                        B,
                        4 * N,
                        D,
                    ),
                )

                stats[name]["ALL"][
                    "cos"
                ].append(cos)

                stats[name]["ALL"][
                    "l2"
                ].append(l2)

    for corruption in PERMS:

        print()
        print(
            f"{corruption.upper()} "
            "(same-frame aligned)"
        )

        print(
            f"{'slot':10s}"
            f"{'cosine':>14s}"
            f"{'1-cos':>14s}"
            f"{'rel-L2':>14s}"
        )

        print("-" * 54)

        for slot in [
            "t-3",
            "t-2",
            "t-1",
            "t",
            "ALL",
        ]:

            cos = torch.cat(
                stats[
                    corruption
                ][slot]["cos"]
            ).mean()

            l2 = torch.cat(
                stats[
                    corruption
                ][slot]["l2"]
            ).mean()

            print(
                f"{slot:10s}"
                f"{cos.item():14.6f}"
                f"{(1-cos).item():14.6f}"
                f"{l2.item():14.6f}"
            )

print()
print(
    "ALIGNED FULL-MEMORY AUDIT: PASS"
)
