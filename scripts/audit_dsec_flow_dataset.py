from pathlib import Path

import torch

from dsec_flow_probe.dataset import (
    DSECSequenceFlowDataset,
)


ROOT = Path(
    "/home/tom/event-jepa/datasets/DSEC"
)

FLOW_ROOT = (
    ROOT / "train_optical_flow"
)

sequences = sorted(
    p.name
    for p in FLOW_ROOT.iterdir()
    if p.is_dir()
)

print("flow sequences:", len(sequences))
print(sequences)

print()
print("=" * 80)
print("Tc=1")
print("=" * 80)

ds1 = DSECSequenceFlowDataset(
    ROOT,
    sequences,
    context_frames=1,
)

print("samples               :", len(ds1))
print(
    "skip missing source   :",
    ds1.skipped_missing_source,
)
print(
    "skip short context    :",
    ds1.skipped_short_context,
)
print(
    "skip context gap      :",
    ds1.skipped_context_gap,
)

x = ds1[0]

for k, v in x.items():
    if isinstance(v, torch.Tensor):
        print(
            k,
            tuple(v.shape),
            v.dtype,
        )
    else:
        print(k, v)

print(
    "flow u range:",
    float(x["flow"][0].min()),
    float(x["flow"][0].max()),
)

print(
    "flow v range:",
    float(x["flow"][1].min()),
    float(x["flow"][1].max()),
)

print(
    "valid ratio :",
    float(x["flow"][2].mean()),
)


print()
print("=" * 80)
print("Tc=4")
print("=" * 80)

ds4 = DSECSequenceFlowDataset(
    ROOT,
    sequences,
    context_frames=4,
)

print("samples               :", len(ds4))
print(
    "skip missing source   :",
    ds4.skipped_missing_source,
)
print(
    "skip short context    :",
    ds4.skipped_short_context,
)
print(
    "skip context gap      :",
    ds4.skipped_context_gap,
)

x = ds4[0]

for k, v in x.items():
    if isinstance(v, torch.Tensor):
        print(
            k,
            tuple(v.shape),
            v.dtype,
        )
    else:
        print(k, v)

print()
print(
    "context timestamps:",
    x["context_ts"].tolist(),
)

print(
    "context gaps us:",
    torch.diff(
        x["context_ts"]
    ).tolist(),
)

print(
    "flow delta_t:",
    x["delta_t"],
)

assert x["context"].shape == (
    4,
    256,
    384,
)

assert x["flow"].shape == (
    3,
    224,
    224,
)

assert (
    int(x["context_ts"][-1])
    == int(x["from_ts"])
)

assert (
    int(x["to_ts"])
    > int(x["from_ts"])
)

assert (
    x["flow"][2].min() >= 0
    and x["flow"][2].max() <= 1
)

print()
print("DSEC SEQUENCE FLOW DATASET: PASS")
