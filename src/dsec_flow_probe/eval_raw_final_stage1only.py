from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dsec_flow_probe.train_raw_finetune_formal import (
    DSECRawSequenceFlowDataset,
    build_stage1,
    encode_stage1,
    read_names,
)
from dsec_flow_probe.head import PatchLinearFlowHead
from dsec_flow_probe.backbone import module_sha256
from utils import FlowMetrics


ROOT = Path(
    "/mnt/data2/gsy/Event-JEPA/datasets/DSEC"
)

S1_INIT = Path(
    "/mnt/data2/gsy/Event-JEPA/"
    "checkpoints/stage1/"
    "gep_stage1_small.pt"
)

ARTIFACT = Path(
    "runs/"
    "dsec_flow_gep_stage1only_vits14_seed0_5k/"
    "final.pt"
)

FINAL_FILE = Path(
    "splits/formal_v1/"
    "dsec_flow_FINAL.txt"
)

OUTPUT = Path(
    "runs/"
    "dsec_flow_gep_stage1only_vits14_seed0_"
    "FINAL.json"
)


@torch.inference_mode()
def evaluate(
    stage1,
    head,
    dataset,
    device,
    batch_size=16,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
    )

    metrics = FlowMetrics(
        n_vals=[1, 2, 3],
        device=torch.device("cpu"),
    )

    l1_sum = 0.0
    l1_count = 0.0
    n_samples = 0

    for batch in loader:

        events = batch["events"].to(
            device,
            non_blocking=True,
        )

        target = batch["flow"].to(
            device,
            non_blocking=True,
        )

        # EXACT same raw -> Stage1 path
        # as ABCD FINAL evaluator.
        tokens = encode_stage1(
            stage1,
            events,
            trainable=False,
        )

        expected = (
            events.shape[0],
            4,
            256,
            384,
        )

        if tuple(tokens.shape) != expected:
            raise RuntimeError(
                f"bad Stage1 shape "
                f"{tuple(tokens.shape)}, "
                f"expected {expected}"
            )

        # Current frame only.
        # No Event-JEPA Stage2.
        z = tokens[:, -1]

        pred = head(
            z.float()
        )

        if not torch.isfinite(pred).all():
            raise RuntimeError(
                "non-finite flow prediction"
            )

        mask = target[:, 2:3]
        gt = target[:, :2]

        l1_sum += float(
            (
                (pred - gt).abs()
                * mask
            )
            .sum()
            .item()
        )

        l1_count += float(
            mask.sum().item()
            * 2.0
        )

        metrics.update(
            pred.detach(),
            target,
        )

        n_samples += int(
            target.shape[0]
        )

    m = metrics.compute()

    return {
        "masked_l1":
            l1_sum
            / max(
                l1_count,
                1.0,
            ),

        "epe":
            float(m["epe"]),

        "ae":
            float(m["ae"]),

        "1pe":
            float(m["1pe"]),

        "2pe":
            float(m["2pe"]),

        "3pe":
            float(m["3pe"]),

        "samples":
            n_samples,
    }


def main():

    if OUTPUT.exists():
        raise RuntimeError(
            f"Refusing accidental second FINAL reveal: "
            f"{OUTPUT}"
        )

    device = torch.device("cuda")

    artifact = torch.load(
        ARTIFACT,
        map_location="cpu",
        weights_only=False,
    )

    for key in [
        "event_encoder",
        "head",
    ]:
        if key not in artifact:
            raise KeyError(
                f"{ARTIFACT}: missing {key}"
            )

    summary = artifact.get(
        "summary",
        {}
    )

    # Training artifact sanity.
    if summary:
        expected = {
            "stage1_changed": True,
            "stage2_changed": False,
            "head_changed": True,
        }

        for key, value in expected.items():
            if summary.get(key) != value:
                raise RuntimeError(
                    f"training audit failed: "
                    f"{key}="
                    f"{summary.get(key)}, "
                    f"expected={value}"
                )

    stage1 = build_stage1(
        S1_INIT
    )

    stage1.load_state_dict(
        artifact["event_encoder"],
        strict=True,
    )

    head = PatchLinearFlowHead(
        embed_dim=384,
        grid_size=16,
        patch_size=14,
    )

    head.load_state_dict(
        artifact["head"],
        strict=True,
    )

    stage1 = stage1.to(device)
    head = head.to(device)

    stage1.requires_grad_(False)
    head.requires_grad_(False)

    stage1.eval()
    head.eval()

    s1_hash_before = module_sha256(
        stage1
    )

    head_hash_before = module_sha256(
        head
    )

    final_names = read_names(
        FINAL_FILE
    )

    if len(final_names) != 4:
        raise RuntimeError(
            f"expected 4 FINAL sequences, "
            f"got {len(final_names)}"
        )

    print("=" * 90)
    print(
        "DSEC FLOW STAGE1-ONLY FULL-FT — FINAL"
    )
    print("=" * 90)

    print(
        "FINAL sequences:",
        final_names,
    )

    print(
        "representation :",
        "fine-tuned GEP Stage-1 current frame"
    )

    print(
        "Stage-2       :",
        "NOT USED"
    )

    per_sequence = {}

    for seq in final_names:

        ds = DSECRawSequenceFlowDataset(
            root=ROOT,
            sequences=[seq],
            context_frames=4,
        )

        r = evaluate(
            stage1,
            head,
            ds,
            device,
        )

        per_sequence[seq] = r

    all_ds = DSECRawSequenceFlowDataset(
        root=ROOT,
        sequences=final_names,
        context_frames=4,
    )

    aggregate = evaluate(
        stage1,
        head,
        all_ds,
        device,
    )

    s1_hash_after = module_sha256(
        stage1
    )

    head_hash_after = module_sha256(
        head
    )

    if s1_hash_before != s1_hash_after:
        raise RuntimeError(
            "Stage1 changed during FINAL"
        )

    if head_hash_before != head_hash_after:
        raise RuntimeError(
            "head changed during FINAL"
        )

    result = {
        "protocol":
            "stage1-only-full-ft",

        "artifact":
            str(ARTIFACT),

        "final_sequences":
            final_names,

        "stage2_used":
            False,

        "per_sequence":
            per_sequence,

        "aggregate":
            aggregate,

        "stage1_unchanged_during_final":
            True,

        "head_unchanged_during_final":
            True,
    }

    OUTPUT.write_text(
        json.dumps(
            result,
            indent=2,
        )
    )

    print()
    print("=" * 90)
    print("AGGREGATE")
    print("=" * 90)

    print(
        f"Masked L1 : "
        f"{aggregate['masked_l1']:.5f}"
    )

    print(
        f"EPE ↓     : "
        f"{aggregate['epe']:.5f}"
    )

    print(
        f"AE ↓      : "
        f"{aggregate['ae']:.5f}"
    )

    print(
        f"1PE ↓     : "
        f"{aggregate['1pe']:.5f}"
    )

    print(
        f"2PE ↓     : "
        f"{aggregate['2pe']:.5f}"
    )

    print(
        f"3PE ↓     : "
        f"{aggregate['3pe']:.5f}"
    )

    print(
        f"N         : "
        f"{aggregate['samples']}"
    )

    print()
    print("=" * 90)
    print("PER-SEQUENCE EPE")
    print("=" * 90)

    for seq in final_names:
        print(
            f"{seq:<28}"
            f"{per_sequence[seq]['epe']:.5f}"
        )

    print()
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
