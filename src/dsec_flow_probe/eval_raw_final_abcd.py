from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dsec_flow_probe.train_raw_finetune_formal import (
    DSECRawSequenceFlowDataset,
    build_stage1,
    build_stage2,
    encode_stage1,
    read_names,
)
from dsec_flow_probe.head import PatchLinearFlowHead
from utils import FlowMetrics


ROOT = Path("/mnt/data2/gsy/Event-JEPA/datasets/DSEC")

S1 = Path(
    "/mnt/data2/gsy/Event-JEPA/checkpoints/stage1/"
    "gep_stage1_small.pt"
)

S2 = Path(
    "/mnt/data2/gsy/Event-JEPA/runs/"
    "event_jepa_vits14_formal_seed0/"
    "step_00030000.pt"
)

FINAL_FILE = Path(
    "splits/formal_v1/dsec_flow_FINAL.txt"
)

OUTPUT = Path(
    "runs/dsec_flow_event_jepa_vits14_30k_"
    "FINAL_ABCD.json"
)

ARTIFACTS = {
    "A_Frozen": Path(
        "runs/dsec_flow_event_jepa_vits14_30k_seed0/final.pt"
    ),
    "B_10pct_FT": Path(
        "runs/dsec_flow_event_jepa_vits14_30k_ft10_seed0/final.pt"
    ),
    "C_Full_FT": Path(
        "runs/dsec_flow_event_jepa_vits14_30k_fullft_seed0/final.pt"
    ),
    "D_Stage2Only": Path(
        "runs/dsec_flow_event_jepa_vits14_30k_stage2only_seed0/final.pt"
    ),
}


def build_method(name, artifact_path, device):
    stage1 = build_stage1(S1)
    stage2, _ = build_stage2(S2)

    artifact = torch.load(
        artifact_path,
        map_location="cpu",
        weights_only=False,
    )

    # A only trained the head.
    # B/C/D saved their complete downstream model.
    if name != "A_Frozen":
        stage1.load_state_dict(
            artifact["event_encoder"],
            strict=True,
        )
        stage2.load_state_dict(
            artifact["online_encoder"],
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
    stage2 = stage2.to(device)
    head = head.to(device)

    stage1.requires_grad_(False)
    stage2.requires_grad_(False)
    head.requires_grad_(False)

    stage1.eval()
    stage2.eval()
    head.eval()

    return stage1, stage2, head


@torch.inference_mode()
def evaluate(
    stage1,
    stage2,
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

        # Exact raw -> Stage-1 path.
        tokens = encode_stage1(
            stage1,
            events,
            trainable=False,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=True,
        ):
            memory = stage2(tokens)

        if tuple(memory.shape[1:]) != (
            4 * 256,
            384,
        ):
            raise RuntimeError(
                f"bad Stage-2 shape: {tuple(memory.shape)}"
            )

        z = memory[:, -256:, :]

        pred = head(z.float())

        mask = target[:, 2:3]
        gt = target[:, :2]

        l1_sum += float(
            ((pred - gt).abs() * mask)
            .sum()
            .item()
        )

        l1_count += float(
            mask.sum().item() * 2.0
        )

        metrics.update(
            pred,
            target,
        )

        n_samples += target.shape[0]

    m = metrics.compute()

    return {
        "masked_l1": l1_sum / max(l1_count, 1.0),
        "epe": float(m["epe"]),
        "ae": float(m["ae"]),
        "1pe": float(m["1pe"]),
        "2pe": float(m["2pe"]),
        "3pe": float(m["3pe"]),
        "samples": int(n_samples),
    }


def main():
    # -----------------------------
    # One-shot FINAL guard.
    # -----------------------------
    if OUTPUT.exists():
        raise RuntimeError(
            f"FINAL result already exists:\n{OUTPUT}\n"
            "Refusing a second reveal."
        )

    final_names = read_names(
        FINAL_FILE
    )

    if len(final_names) != 4:
        raise RuntimeError(
            f"expected 4 FINAL sequences, got {len(final_names)}"
        )

    # Preflight BEFORE reading FINAL.
    for name, path in ARTIFACTS.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"{name}: {path}"
            )

    if not S1.is_file():
        raise FileNotFoundError(S1)

    if not S2.is_file():
        raise FileNotFoundError(S2)

    print("=" * 90)
    print("DSEC FLOW — ONE-SHOT FINAL A/B/C/D")
    print("=" * 90)
    print("FINAL sequences:", final_names)
    print("Methods        :", list(ARTIFACTS))
    print()
    print("All artifacts passed preflight.")
    print("FINAL reveal starts now.")
    print()

    device = torch.device("cuda")

    results = {}

    for name, artifact_path in ARTIFACTS.items():
        print(f"[FINAL] evaluating {name} ...")

        stage1, stage2, head = build_method(
            name,
            artifact_path,
            device,
        )

        per_sequence = {}

        for seq in final_names:
            ds = DSECRawSequenceFlowDataset(
                ROOT,
                [seq],
                context_frames=4,
            )

            per_sequence[seq] = evaluate(
                stage1,
                stage2,
                head,
                ds,
                device,
            )

        # Aggregate directly over all samples,
        # not mean of 4 sequence metrics.
        all_ds = DSECRawSequenceFlowDataset(
            ROOT,
            final_names,
            context_frames=4,
        )

        aggregate = evaluate(
            stage1,
            stage2,
            head,
            all_ds,
            device,
        )

        results[name] = {
            "artifact": str(artifact_path),
            "per_sequence": per_sequence,
            "aggregate": aggregate,
        }

        del stage1, stage2, head
        torch.cuda.empty_cache()

    output = {
        "protocol": "event_jepa_dsec_flow_final_abcd_one_shot",
        "final_sequences": final_names,
        "methods": results,
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save before revealing numbers.
    OUTPUT.write_text(
        json.dumps(
            output,
            indent=2,
        )
    )

    print()
    print("=" * 100)
    print("FINAL AGGREGATE RESULTS")
    print("=" * 100)

    print(
        f"{'Protocol':<20}"
        f"{'L1':>10}"
        f"{'EPE':>10}"
        f"{'AE':>10}"
        f"{'1PE':>10}"
        f"{'2PE':>10}"
        f"{'3PE':>10}"
        f"{'N':>8}"
    )

    print("-" * 100)

    for name in ARTIFACTS:
        r = results[name]["aggregate"]

        print(
            f"{name:<20}"
            f"{r['masked_l1']:>10.5f}"
            f"{r['epe']:>10.5f}"
            f"{r['ae']:>10.5f}"
            f"{r['1pe']:>10.5f}"
            f"{r['2pe']:>10.5f}"
            f"{r['3pe']:>10.5f}"
            f"{r['samples']:>8d}"
        )

    print()
    print("PER-SEQUENCE EPE")
    print("-" * 100)

    for seq in final_names:
        print(seq)
        for name in ARTIFACTS:
            epe = results[name][
                "per_sequence"
            ][seq]["epe"]

            print(
                f"  {name:<18} "
                f"{epe:.5f}"
            )

    print()
    print("FINAL result saved to:")
    print(OUTPUT)
    print()
    print("FINAL REVEAL: COMPLETE")


if __name__ == "__main__":
    main()
