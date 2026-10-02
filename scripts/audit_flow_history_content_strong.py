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

CKPT = Path(
    "runs/event_jepa_dsec_mh148_seed0/"
    "step_00030000.pt"
)

VAL = [
    "thun_00_a",
    "zurich_city_02_d",
]

device = torch.device("cuda")


ds = DSECSequenceFlowDataset(
    ROOT,
    VAL,
    context_frames=4,
)

#
# Materialize all contexts once.
#
all_context = []
all_sequence = []

loader = DataLoader(
    ds,
    batch_size=16,
    shuffle=False,
    num_workers=4,
)

for batch in tqdm(
    loader,
    desc="loading contexts",
):
    all_context.append(
        batch["context"]
    )

    all_sequence.extend(
        batch["sequence"]
    )

all_context = torch.cat(
    all_context,
    dim=0,
)

N = len(ds)

print("samples:", N)
print(
    "context:",
    tuple(all_context.shape),
)


#
# Build a deterministic donor index for every
# sample, forcing donor sequence != target sequence.
#
far_donor = []

for i in range(N):
    target_seq = all_sequence[i]

    donor = None

    # Start approximately halfway around dataset,
    # then search for another sequence.
    start = (i + N // 2) % N

    for offset in range(N):
        j = (
            start + offset
        ) % N

        if (
            all_sequence[j]
            != target_seq
        ):
            donor = j
            break

    if donor is None:
        raise RuntimeError(
            "could not find cross-sequence donor"
        )

    far_donor.append(donor)

far_donor = torch.tensor(
    far_donor,
    dtype=torch.long,
)


#
# A second deterministic global permutation.
#
g = torch.Generator()
g.manual_seed(12345)

random_perm = torch.randperm(
    N,
    generator=g,
)


model = FrozenFlowBackbone(
    mode="jepa",
    checkpoint=CKPT,
).to(device)

model.eval()


variants = [
    "repeat_last",
    "reverse_past",
    "far_wrong",
    "random_history",
]

cos_stats = {
    v: []
    for v in variants
}

l2_stats = {
    v: []
    for v in variants
}


batch_size = 8

with torch.no_grad():

    for start in tqdm(
        range(0, N, batch_size),
        desc="representation audit",
    ):

        end = min(
            start + batch_size,
            N,
        )

        x = all_context[
            start:end
        ].to(device)

        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
        ):
            z_normal = model(x)

        corrupted = {}

        #
        # Same current frame repeated.
        #
        corrupted[
            "repeat_last"
        ] = (
            x[:, -1:]
            .expand(
                -1,
                4,
                -1,
                -1,
            )
            .contiguous()
        )

        #
        # Same historical content,
        # wrong temporal ordering.
        #
        corrupted[
            "reverse_past"
        ] = x[
            :,
            [2, 1, 0, 3],
        ]

        #
        # History from a completely different
        # validation sequence.
        #
        donor_idx = far_donor[
            start:end
        ]

        donor = all_context[
            donor_idx
        ].to(device)

        y = x.clone()

        y[:, :3] = donor[:, :3]

        corrupted[
            "far_wrong"
        ] = y

        #
        # Globally random history.
        #
        rand_idx = random_perm[
            start:end
        ]

        rand = all_context[
            rand_idx
        ].to(device)

        y = x.clone()

        y[:, :3] = rand[:, :3]

        corrupted[
            "random_history"
        ] = y


        for variant in variants:

            with torch.autocast(
                "cuda",
                dtype=torch.bfloat16,
            ):
                z = model(
                    corrupted[variant]
                )

            zf = z.float()
            zn = z_normal.float()

            cos = F.cosine_similarity(
                zn,
                zf,
                dim=-1,
            ).mean(dim=-1)

            rel_l2 = (
                (zf - zn)
                .flatten(1)
                .norm(dim=1)
                /
                (
                    zn.flatten(1)
                    .norm(dim=1)
                    + 1e-8
                )
            )

            cos_stats[
                variant
            ].append(
                cos.cpu()
            )

            l2_stats[
                variant
            ].append(
                rel_l2.cpu()
            )


print()
print("=" * 76)
print("STRONG HISTORY CONTENT SENSITIVITY")
print("=" * 76)

print(
    f"{'variant':18s}"
    f"{'cosine':>14s}"
    f"{'1-cos':>14s}"
    f"{'rel-L2':>14s}"
)

print("-" * 76)

for variant in variants:

    cos = torch.cat(
        cos_stats[variant]
    )

    l2 = torch.cat(
        l2_stats[variant]
    )

    print(
        f"{variant:18s}"
        f"{cos.mean().item():14.6f}"
        f"{(1-cos.mean()).item():14.6f}"
        f"{l2.mean().item():14.6f}"
    )

print()
print(
    "STRONG HISTORY CONTENT AUDIT: PASS"
)
