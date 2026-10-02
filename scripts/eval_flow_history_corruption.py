from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

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


ROOT = Path(
    "/home/tom/event-jepa/datasets/DSEC"
)

JEPA_CKPT = Path(
    "runs/event_jepa_dsec_mh148_seed0/"
    "step_00030000.pt"
)

FLOW_RUN = Path(
    "runs/flow_tc4_mh148_seed0_5k"
)

OUTPUT_JSON = (
    FLOW_RUN / "history_corruption.json"
)

OUTPUT_CSV = (
    FLOW_RUN / "history_corruption_samples.csv"
)

VAL_SEQUENCES = [
    "thun_00_a",
    "zurich_city_02_d",
]


def corrupt_context(
    context: torch.Tensor,
    variant: str,
) -> torch.Tensor:
    """
    context:
        [B,4,N,D]

    Original order:
        0 = t-3
        1 = t-2
        2 = t-1
        3 = t
    """

    if context.shape[1] != 4:
        raise ValueError(
            "history corruption requires Tc=4"
        )

    if variant == "normal":
        return context

    if variant == "repeat_last":
        return (
            context[:, -1:]
            .expand(
                -1,
                4,
                -1,
                -1,
            )
            .contiguous()
        )

    if variant == "reverse_past":
        return context[
            :,
            [2, 1, 0, 3],
            :,
            :,
        ]

    if variant == "shuffle_past":
        # Fixed deterministic permutation.
        #
        # normal:
        # [t-3,t-2,t-1,t]
        #
        # shuffled:
        # [t-2,t-1,t-3,t]
        return context[
            :,
            [1, 2, 0, 3],
            :,
            :,
        ]

    raise ValueError(
        f"unknown variant: {variant}"
    )


def sample_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
):
    """
    Single sample:
        pred   [2,H,W]
        target [3,H,W]
    """
    gt = target[:2]
    mask = target[2] > 0

    valid = int(mask.sum())

    if valid == 0:
        return {
            "epe": float("nan"),
            "ae": float("nan"),
            "1pe": float("nan"),
            "2pe": float("nan"),
            "3pe": float("nan"),
            "valid_pixels": 0,
        }

    diff = pred - gt

    epe_map = torch.sqrt(
        (diff ** 2).sum(dim=0)
    )

    epe = epe_map[mask].mean()

    result = {
        "epe":
            float(epe),

        "1pe":
            float(
                (
                    epe_map[mask] > 1
                ).float().mean()
            ),

        "2pe":
            float(
                (
                    epe_map[mask] > 2
                ).float().mean()
            ),

        "3pe":
            float(
                (
                    epe_map[mask] > 3
                ).float().mean()
            ),

        "valid_pixels":
            valid,
    }

    pu = pred[0][mask]
    pv = pred[1][mask]

    gu = gt[0][mask]
    gv = gt[1][mask]

    numerator = (
        pu * gu
        + pv * gv
        + 1
    )

    denominator = (
        torch.sqrt(
            pu ** 2 + pv ** 2 + 1
        )
        *
        torch.sqrt(
            gu ** 2 + gv ** 2 + 1
        )
    )

    cos = torch.clamp(
        numerator
        / (denominator + 1e-8),
        -1.0,
        1.0,
    )

    result["ae"] = float(
        torch.acos(cos).mean()
    )

    return result


def main():
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 86)
    print("DSEC FLOW HISTORY CORRUPTION TEST")
    print("=" * 86)

    print("device        :", device)
    print("JEPA checkpoint:", JEPA_CKPT)
    print("flow run       :", FLOW_RUN)

    #
    # Load exact same validation samples
    #
    dataset = DSECSequenceFlowDataset(
        ROOT,
        VAL_SEQUENCES,
        context_frames=4,
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(
        "val sequences :",
        VAL_SEQUENCES,
    )

    print(
        "val samples   :",
        len(dataset),
    )

    assert len(dataset) == 401

    #
    # Load exact frozen Tc4 representation
    #
    backbone = FrozenFlowBackbone(
        mode="jepa",
        checkpoint=JEPA_CKPT,
    ).to(device)

    backbone.requires_grad_(False)
    backbone.eval()

    #
    # Load exact already-trained flow head
    #
    flow_state = torch.load(
        FLOW_RUN / "final.pt",
        map_location="cpu",
        weights_only=False,
    )

    training_summary = (
        flow_state["summary"]
    )

    head = PatchLinearFlowHead()

    head.load_state_dict(
        flow_state["head"],
        strict=True,
    )

    head = head.to(device)
    head.eval()

    for p in head.parameters():
        p.requires_grad = False

    #
    # Fingerprint verification
    #
    backbone_hash = module_sha256(
        backbone.encoder
    )

    head_hash = module_sha256(
        head
    )

    print()
    print(
        "backbone SHA256:",
        backbone_hash,
    )

    print(
        "expected backbone:",
        training_summary[
            "backbone_sha256_after"
        ],
    )

    print(
        "head SHA256    :",
        head_hash,
    )

    print(
        "expected head  :",
        training_summary[
            "head_sha256_after"
        ],
    )

    assert (
        backbone_hash
        ==
        training_summary[
            "backbone_sha256_after"
        ]
    ), "backbone fingerprint mismatch"

    assert (
        head_hash
        ==
        training_summary[
            "head_sha256_after"
        ]
    ), "flow head fingerprint mismatch"

    print(
        "checkpoint/head fingerprint: PASS"
    )

    variants = [
        "normal",
        "repeat_last",
        "reverse_past",
        "shuffle_past",
    ]

    global_metrics = {
        v: FlowMetrics(
            n_vals=[1, 2, 3],
            device=torch.device("cpu"),
        )
        for v in variants
    }

    per_sequence = {
        v: defaultdict(
            lambda: FlowMetrics(
                n_vals=[1, 2, 3],
                device=torch.device("cpu"),
            )
        )
        for v in variants
    }

    rows = []

    with torch.no_grad():
        for batch in tqdm(
            loader,
            desc="history corruption",
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

            B = context.shape[0]

            #
            # Evaluate all four variants.
            #
            variant_contexts = [
                corrupt_context(
                    context,
                    v,
                )
                for v in variants
            ]

            #
            # One larger forward for efficiency:
            #
            # [4*B,4,256,384]
            #
            big_context = torch.cat(
                variant_contexts,
                dim=0,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=(
                    device.type == "cuda"
                ),
            ):
                z = backbone(
                    big_context
                )

            pred = head(
                z.float()
            )

            pred_chunks = torch.split(
                pred,
                B,
                dim=0,
            )

            for variant, pred_v in zip(
                variants,
                pred_chunks,
            ):
                #
                # Global pixel-weighted metrics
                #
                global_metrics[
                    variant
                ].update(
                    pred_v.detach().cpu(),
                    target.detach().cpu(),
                )

                #
                # Per-sample and per-sequence
                #
                for i in range(B):
                    sequence = batch[
                        "sequence"
                    ][i]

                    pred_i = (
                        pred_v[i:i+1]
                        .detach()
                        .cpu()
                    )

                    target_i = (
                        target[i:i+1]
                        .detach()
                        .cpu()
                    )

                    per_sequence[
                        variant
                    ][
                        sequence
                    ].update(
                        pred_i,
                        target_i,
                    )

                    sm = sample_metrics(
                        pred_i[0],
                        target_i[0],
                    )

                    rows.append(
                        {
                            "variant":
                                variant,

                            "sequence":
                                sequence,

                            "from_ts":
                                int(
                                    batch[
                                        "from_ts"
                                    ][i]
                                ),

                            "to_ts":
                                int(
                                    batch[
                                        "to_ts"
                                    ][i]
                                ),

                            **sm,
                        }
                    )

    #
    # Aggregate results
    #
    results = {}

    for variant in variants:
        results[variant] = (
            global_metrics[
                variant
            ].compute()
        )

        results[
            variant
        ][
            "per_sequence"
        ] = {
            seq:
                metric.compute()
            for seq, metric
            in per_sequence[
                variant
            ].items()
        }

    normal_epe = results[
        "normal"
    ]["epe"]

    for variant in variants:
        epe = results[
            variant
        ]["epe"]

        results[
            variant
        ][
            "delta_epe_vs_normal"
        ] = (
            epe - normal_epe
        )

        results[
            variant
        ][
            "relative_epe_change_pct"
        ] = (
            100.0
            * (
                epe - normal_epe
            )
            / normal_epe
        )

    payload = {
        "protocol":
            "same_checkpoint_same_head_history_corruption",

        "jepa_checkpoint":
            str(JEPA_CKPT),

        "flow_run":
            str(FLOW_RUN),

        "backbone_sha256":
            backbone_hash,

        "head_sha256":
            head_hash,

        "val_sequences":
            VAL_SEQUENCES,

        "val_samples":
            len(dataset),

        "results":
            results,
    }

    OUTPUT_JSON.write_text(
        json.dumps(
            payload,
            indent=2,
        )
    )

    with OUTPUT_CSV.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "sequence",
                "from_ts",
                "to_ts",
                "epe",
                "ae",
                "1pe",
                "2pe",
                "3pe",
                "valid_pixels",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    #
    # Print main table
    #
    print()
    print("=" * 96)
    print("AGGREGATE HISTORY CORRUPTION RESULTS")
    print("=" * 96)

    print(
        f"{'variant':18s}"
        f"{'EPE':>10s}"
        f"{'ΔEPE':>10s}"
        f"{'ΔEPE%':>10s}"
        f"{'AE':>10s}"
        f"{'1PE':>10s}"
        f"{'2PE':>10s}"
        f"{'3PE':>10s}"
    )

    print("-" * 96)

    for variant in variants:
        r = results[
            variant
        ]

        print(
            f"{variant:18s}"
            f"{r['epe']:10.4f}"
            f"{r['delta_epe_vs_normal']:10.4f}"
            f"{r['relative_epe_change_pct']:10.2f}"
            f"{r['ae']:10.4f}"
            f"{r['1pe']:10.4f}"
            f"{r['2pe']:10.4f}"
            f"{r['3pe']:10.4f}"
        )

    print()
    print("=" * 96)
    print("PER-SEQUENCE EPE")
    print("=" * 96)

    print(
        f"{'variant':18s}"
        f"{'thun_00_a':>15s}"
        f"{'zurich_city_02_d':>20s}"
    )

    print("-" * 58)

    for variant in variants:
        p = results[
            variant
        ]["per_sequence"]

        print(
            f"{variant:18s}"
            f"{p['thun_00_a']['epe']:15.4f}"
            f"{p['zurich_city_02_d']['epe']:20.4f}"
        )

    print()
    print(
        "saved:",
        OUTPUT_JSON,
    )

    print(
        "saved:",
        OUTPUT_CSV,
    )

    print()
    print(
        "HISTORY CORRUPTION EVALUATION: PASS"
    )


if __name__ == "__main__":
    main()
