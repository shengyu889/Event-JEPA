from __future__ import annotations

import argparse
import json
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
)
from utils import FlowMetrics


def read_names(path):
    path = Path(path)

    names = [
        x.strip()
        for x in path.read_text().splitlines()
        if x.strip()
    ]

    if not names:
        raise RuntimeError(
            f"empty sequence file: {path}"
        )

    if len(names) != len(set(names)):
        raise RuntimeError(
            f"duplicate sequence: {path}"
        )

    return names


@torch.no_grad()
def evaluate_dataset(
    backbone,
    head,
    dataset,
    device,
    batch_size,
    num_workers,
    precision,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(
            num_workers > 0
        ),
    )

    backbone.eval()
    head.eval()

    metrics = FlowMetrics(
        n_vals=[1, 2, 3],
        device=torch.device("cpu"),
    )

    abs_sum = 0.0
    abs_count = 0.0
    samples = 0

    for batch in loader:
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

        use_amp = (
            device.type == "cuda"
            and precision == "bf16"
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            z = backbone(
                context
            )

        if tuple(z.shape[1:]) != (
            256,
            384,
        ):
            raise RuntimeError(
                f"bad representation shape: "
                f"{tuple(z.shape)}"
            )

        if not torch.isfinite(z).all():
            raise RuntimeError(
                "non-finite representation"
            )

        pred = head(
            z.float()
        )

        if not torch.isfinite(pred).all():
            raise RuntimeError(
                "non-finite prediction"
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

        samples += int(
            target.shape[0]
        )

    m = metrics.compute()

    return {
        "masked_l1":
            abs_sum
            / max(
                abs_count,
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
            samples,
    }


def load_method(
    name,
    mode,
    stage2_checkpoint,
    flow_head_checkpoint,
    device,
):
    backbone = FrozenFlowBackbone(
        mode=mode,
        checkpoint=stage2_checkpoint,
    ).to(device)

    backbone.requires_grad_(False)
    backbone.eval()

    if any(
        p.requires_grad
        for p in backbone.parameters()
    ):
        raise RuntimeError(
            f"{name}: backbone not frozen"
        )

    flow_state = torch.load(
        flow_head_checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    if "head" not in flow_state:
        raise KeyError(
            f"{flow_head_checkpoint}: "
            "missing head"
        )

    summary = flow_state.get(
        "summary",
        {}
    )

    if (
        summary.get(
            "final_evaluated",
            False,
        )
        is not False
    ):
        raise RuntimeError(
            f"{name}: training artifact "
            "already says FINAL evaluated"
        )

    head = PatchLinearFlowHead().to(
        device
    )

    head.load_state_dict(
        flow_state["head"],
        strict=True,
    )

    head.requires_grad_(False)
    head.eval()

    expected_head_hash = (
        summary.get(
            "head_sha256_after"
        )
    )

    actual_head_hash = (
        module_sha256(head)
    )

    if (
        expected_head_hash
        is not None
        and actual_head_hash
        != expected_head_hash
    ):
        raise RuntimeError(
            f"{name}: head hash mismatch"
        )

    backbone_hash = (
        module_sha256(
            backbone.encoder
        )
        if backbone.encoder
        is not None
        else None
    )

    return {
        "name":
            name,

        "mode":
            mode,

        "backbone":
            backbone,

        "head":
            head,

        "stage2_checkpoint":
            (
                None
                if stage2_checkpoint
                is None
                else str(
                    Path(
                        stage2_checkpoint
                    )
                )
            ),

        "flow_head_checkpoint":
            str(
                Path(
                    flow_head_checkpoint
                )
            ),

        "head_sha256":
            actual_head_hash,

        "backbone_sha256":
            backbone_hash,

        "model_context_frames":
            backbone.context_frames,
    }


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--root",
        required=True,
    )

    p.add_argument(
        "--final-sequences-file",
        required=True,
    )

    p.add_argument(
        "--output",
        required=True,
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    p.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    p.add_argument(
        "--precision",
        choices=[
            "fp32",
            "bf16",
        ],
        default="bf16",
    )

    args = p.parse_args()

    root = Path(
        args.root
    )

    final_names = read_names(
        args.final_sequences_file
    )

    # Formal v1 contract.
    if len(final_names) != 4:
        raise RuntimeError(
            f"expected 4 FINAL sequences, "
            f"got {len(final_names)}"
        )

    output = Path(
        args.output
    )

    if output.exists():
        raise RuntimeError(
            f"FINAL output already exists:\n"
            f"{output}\n"
            "Refusing accidental second reveal."
        )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    specs = [
        {
            "name":
                "Stage1",

            "mode":
                "stage1",

            "stage2":
                None,

            "head":
                "runs/"
                "formal_v1_flow_stage1_seed0_5k/"
                "final.pt",
        },

        {
            "name":
                "Stage1-Mean4",

            "mode":
                "stage1-mean4",

            "stage2":
                None,

            "head":
                "runs/"
                "formal_v1_flow_stage1_mean4_seed0_5k/"
                "final.pt",
        },

        {
            "name":
                "Matched-GEP-AR",

            "mode":
                "gep-opt-style",

            "stage2":
                "runs/"
                "formal_v1_gep_ar_seed0_30k/"
                "step_00030000.pt",

            "head":
                "runs/"
                "formal_v1_flow_gep_seed0_5k/"
                "final.pt",
        },

        {
            "name":
                "Event-JEPA",

            "mode":
                "jepa",

            "stage2":
                "runs/"
                "formal_v1_event_jepa_seed0_30k/"
                "step_00030000.pt",

            "head":
                "runs/"
                "formal_v1_flow_jepa_seed0_5k/"
                "final.pt",
        },
    ]

    methods = []

    # --------------------------------------------------
    # Preflight all methods BEFORE reading FINAL data.
    # --------------------------------------------------

    for spec in specs:
        if not Path(
            spec["head"]
        ).is_file():
            raise FileNotFoundError(
                spec["head"]
            )

        if (
            spec["stage2"]
            is not None
            and not Path(
                spec["stage2"]
            ).is_file()
        ):
            raise FileNotFoundError(
                spec["stage2"]
            )

        methods.append(
            load_method(
                spec["name"],
                spec["mode"],
                spec["stage2"],
                spec["head"],
                device,
            )
        )

    print("=" * 80)
    print("FORMAL V1 DSEC FINAL REVEAL")
    print("=" * 80)

    print(
        "FINAL sequences:",
        final_names,
    )

    print(
        "methods        :",
        [
            x["name"]
            for x in methods
        ],
    )

    print(
        "precision      :",
        args.precision,
    )

    print()
    print(
        "All artifacts passed preflight."
    )
    print(
        "FINAL evaluation starts now."
    )
    print()

    results = {}

    # --------------------------------------------------
    # Evaluate all methods.
    #
    # We store everything and print numerical results
    # ONLY after all four methods have completed.
    # --------------------------------------------------

    for method in methods:
        name = method[
            "name"
        ]

        backbone = method[
            "backbone"
        ]

        head = method[
            "head"
        ]

        before_backbone = (
            method[
                "backbone_sha256"
            ]
        )

        before_head = (
            method[
                "head_sha256"
            ]
        )

        per_sequence = {}

        for sequence in final_names:
            ds = DSECSequenceFlowDataset(
                root,
                [sequence],
                context_frames=4,
            )

            per_sequence[
                sequence
            ] = evaluate_dataset(
                backbone,
                head,
                ds,
                device,
                args.batch_size,
                args.num_workers,
                args.precision,
            )

        # Aggregate is evaluated directly over one
        # combined Dataset, not averaged from four
        # sequence-level metric values.
        all_dataset = (
            DSECSequenceFlowDataset(
                root,
                final_names,
                context_frames=4,
            )
        )

        aggregate = evaluate_dataset(
            backbone,
            head,
            all_dataset,
            device,
            args.batch_size,
            args.num_workers,
            args.precision,
        )

        after_backbone = (
            module_sha256(
                backbone.encoder
            )
            if backbone.encoder
            is not None
            else None
        )

        after_head = (
            module_sha256(
                head
            )
        )

        if (
            before_backbone
            != after_backbone
        ):
            raise RuntimeError(
                f"{name}: backbone changed "
                "during FINAL"
            )

        if (
            before_head
            != after_head
        ):
            raise RuntimeError(
                f"{name}: head changed "
                "during FINAL"
            )

        results[
            name
        ] = {
            "mode":
                method["mode"],

            "model_context_frames":
                method[
                    "model_context_frames"
                ],

            "stage2_checkpoint":
                method[
                    "stage2_checkpoint"
                ],

            "flow_head_checkpoint":
                method[
                    "flow_head_checkpoint"
                ],

            "backbone_sha256":
                before_backbone,

            "head_sha256":
                before_head,

            "backbone_unchanged":
                True,

            "head_unchanged":
                True,

            "per_sequence":
                per_sequence,

            "aggregate":
                aggregate,
        }

    artifact = {
        "protocol":
            "formal_v1_dsec_final_one_shot",

        "root":
            str(
                root.resolve()
            ),

        "final_sequences_file":
            str(
                Path(
                    args.final_sequences_file
                ).resolve()
            ),

        "final_sequences":
            final_names,

        "common_dataset_context_frames":
            4,

        "batch_size":
            args.batch_size,

        "precision":
            args.precision,

        "methods":
            results,
    }

    # --------------------------------------------------
    # Persist FINAL result BEFORE printing table.
    # --------------------------------------------------

    output.write_text(
        json.dumps(
            artifact,
            indent=2,
        )
    )

    print()
    print("=" * 96)
    print("FORMAL V1 FINAL — AGGREGATE RESULTS")
    print("=" * 96)

    print(
        f"{'Method':<20}"
        f"{'L1':>10}"
        f"{'EPE':>10}"
        f"{'AE':>10}"
        f"{'1PE':>10}"
        f"{'2PE':>10}"
        f"{'3PE':>10}"
        f"{'N':>8}"
    )

    print("-" * 96)

    for spec in specs:
        name = spec["name"]

        r = results[
            name
        ]["aggregate"]

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
    print("=" * 96)
    print("PER-SEQUENCE EPE")
    print("=" * 96)

    print(
        f"{'Sequence':<24}"
        + "".join(
            f"{spec['name']:>18}"
            for spec in specs
        )
    )

    print("-" * 96)

    for sequence in final_names:
        row = (
            f"{sequence:<24}"
        )

        for spec in specs:
            name = spec["name"]

            epe = (
                results[
                    name
                ][
                    "per_sequence"
                ][
                    sequence
                ][
                    "epe"
                ]
            )

            row += (
                f"{epe:>18.5f}"
            )

        print(row)

    print()
    print("=" * 96)
    print("FINAL INTEGRITY")
    print("=" * 96)

    for spec in specs:
        name = spec["name"]

        r = results[name]

        print(
            f"{name:<20} "
            f"backbone_unchanged="
            f"{r['backbone_unchanged']} "
            f"head_unchanged="
            f"{r['head_unchanged']}"
        )

    print()
    print(
        "FINAL results written to:"
    )
    print(
        output
    )

    print()
    print(
        "FORMAL V1 DSEC FINAL REVEAL: COMPLETE"
    )


if __name__ == "__main__":
    main()
