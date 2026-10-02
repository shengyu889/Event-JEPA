from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dsec_flow_probe.dataset import (
    DSECSequenceFlowDataset,
)
from dsec_flow_probe.backbone import (
    FrozenFlowBackbone,
    module_sha256,
)
from dsec_flow_probe.head import (
    PatchLinearFlowHead,
    masked_flow_l1,
)


ROOT = Path(
    "/home/tom/event-jepa/datasets/DSEC"
)

TC1_CKPT = Path(
    "runs/event_jepa_dsec_tc1_mh148_seed0/"
    "step_00030000.pt"
)

TC4_CKPT = Path(
    "runs/event_jepa_dsec_mh148_seed0/"
    "step_00030000.pt"
)


# IMPORTANT:
# Always build the common Tc=4 dataset.
# This forces all methods to see exactly
# the same samples.
val_sequences = [
    "thun_00_a",
    "zurich_city_02_d",
]

dataset = DSECSequenceFlowDataset(
    ROOT,
    val_sequences,
    context_frames=4,
)

print("=" * 80)
print("COMMON FLOW VALIDATION DATASET")
print("=" * 80)
print("sequences:", val_sequences)
print("samples  :", len(dataset))

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0,
)

batch = next(iter(loader))

context = batch["context"]
target = batch["flow"]

print(
    "context:",
    tuple(context.shape),
)

print(
    "target :",
    tuple(target.shape),
)

print(
    "delta_t:",
    batch["delta_t"].tolist(),
)

print()


models = {
    "Stage1": FrozenFlowBackbone(
        mode="stage1",
    ),

    "JEPA-MH-Tc1": FrozenFlowBackbone(
        mode="jepa",
        checkpoint=TC1_CKPT,
    ),

    "JEPA-MH-Tc4": FrozenFlowBackbone(
        mode="jepa",
        checkpoint=TC4_CKPT,
    ),
}


head_hashes = []

for name, backbone in models.items():

    print("=" * 80)
    print(name)
    print("=" * 80)

    backbone.eval()

    trainable = sum(
        p.numel()
        for p in backbone.parameters()
        if p.requires_grad
    )

    print(
        "required context:",
        backbone.context_frames,
    )

    if backbone.config is not None:
        print(
            "checkpoint horizons:",
            backbone.config["horizons"],
        )

    print(
        "backbone trainable:",
        trainable,
    )

    if backbone.encoder is not None:
        h = module_sha256(
            backbone.encoder
        )
        print(
            "encoder SHA256:",
            h,
        )
    else:
        print(
            "encoder SHA256: Stage-1 raw tokens"
        )

    with torch.no_grad():
        z = backbone(context)

    print(
        "representation:",
        tuple(z.shape),
    )

    print(
        "representation finite:",
        bool(
            torch.isfinite(z).all()
        ),
    )

    print(
        "representation mean:",
        float(z.mean()),
    )

    print(
        "representation std :",
        float(z.std()),
    )

    # Re-seed immediately before each flow head.
    # Therefore all methods start with the
    # EXACT same decoder weights.
    torch.manual_seed(0)

    head = PatchLinearFlowHead()

    head_hash = module_sha256(head)
    head_hashes.append(head_hash)

    print(
        "head SHA256:",
        head_hash,
    )

    pred = head(z)

    print(
        "prediction:",
        tuple(pred.shape),
    )

    loss = masked_flow_l1(
        pred,
        target,
    )

    print(
        "initial masked L1:",
        float(loss),
    )

    assert tuple(z.shape) == (
        2,
        256,
        384,
    )

    assert tuple(pred.shape) == (
        2,
        2,
        224,
        224,
    )

    assert torch.isfinite(z).all()
    assert torch.isfinite(pred).all()

    print()


assert len(set(head_hashes)) == 1, (
    "flow heads did not start "
    "from identical weights"
)

print("=" * 80)
print("COMMON HEAD INITIALIZATION")
print("=" * 80)
print("SHA256:", head_hashes[0])
print("identical across methods: PASS")

print()
print("DSEC FLOW BACKBONE AUDIT: PASS")
