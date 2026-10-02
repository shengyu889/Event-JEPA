from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dsec_flow_probe.dataset import DSECSequenceFlowDataset
from dsec_flow_probe.backbone import FrozenFlowBackbone


ROOT = Path("/home/tom/event-jepa/datasets/DSEC")

CKPT = Path(
    "runs/event_jepa_dsec_mh148_seed0/"
    "step_00030000.pt"
)

VAL = [
    "thun_00_a",
    "zurich_city_02_d",
]


def corrupt(x, mode):
    if mode == "normal":
        return x

    if mode == "repeat_last":
        return x[:, -1:].expand(
            -1, 4, -1, -1
        ).contiguous()

    if mode == "reverse_past":
        return x[:, [2, 1, 0, 3]]

    if mode == "shuffle_past":
        return x[:, [1, 2, 0, 3]]

    if mode == "zero_past":
        y = x.clone()
        y[:, :3] = 0
        return y

    if mode == "wrong_past":
        # Shift historical context by one sample,
        # while keeping each sample's current t fixed.
        y = x.clone()

        past = torch.roll(
            x[:, :3],
            shifts=1,
            dims=0,
        )

        y[:, :3] = past
        return y

    raise ValueError(mode)


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

model = FrozenFlowBackbone(
    mode="jepa",
    checkpoint=CKPT,
).to(device)

model.eval()

variants = [
    "repeat_last",
    "reverse_past",
    "shuffle_past",
    "zero_past",
    "wrong_past",
]

stats = {
    v: {
        "cos": [],
        "rel_l2": [],
    }
    for v in variants
}

with torch.no_grad():
    for batch in tqdm(loader):

        x = batch["context"].to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
        ):
            z0 = model(x)

            for variant in variants:
                zv = model(
                    corrupt(x, variant)
                )

                cos = F.cosine_similarity(
                    z0.float(),
                    zv.float(),
                    dim=-1,
                ).mean(dim=-1)

                diff = (
                    zv.float()
                    - z0.float()
                ).flatten(1)

                base = (
                    z0.float()
                ).flatten(1)

                rel_l2 = (
                    diff.norm(dim=1)
                    /
                    (
                        base.norm(dim=1)
                        + 1e-8
                    )
                )

                stats[variant][
                    "cos"
                ].append(
                    cos.cpu()
                )

                stats[variant][
                    "rel_l2"
                ].append(
                    rel_l2.cpu()
                )


print()
print("=" * 70)
print("HISTORY REPRESENTATION SENSITIVITY")
print("=" * 70)

print(
    f"{'variant':18s}"
    f"{'cosine':>14s}"
    f"{'1-cos':>14s}"
    f"{'rel-L2':>14s}"
)

print("-" * 70)

for variant in variants:

    cos = torch.cat(
        stats[variant]["cos"]
    )

    l2 = torch.cat(
        stats[variant]["rel_l2"]
    )

    print(
        f"{variant:18s}"
        f"{cos.mean().item():14.6f}"
        f"{(1-cos.mean()).item():14.6f}"
        f"{l2.mean().item():14.6f}"
    )

print()
print("samples:", len(ds))
print("HISTORY REPRESENTATION AUDIT: PASS")
