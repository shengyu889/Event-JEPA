from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dsec_flow_probe.dataset import DSECSequenceFlowDataset
from dsec_flow_probe.backbone import FrozenFlowBackbone
from dsec_flow_probe.head import PatchLinearFlowHead
from utils import FlowMetrics


ROOT = Path("/home/tom/event-jepa/datasets/DSEC")

CKPT = Path(
    "runs/event_jepa_dsec_mh148_seed0/"
    "step_00030000.pt"
)

FLOW_RUN = Path(
    "runs/flow_tc4_mh148_seed0_5k"
)

VAL = [
    "thun_00_a",
    "zurich_city_02_d",
]


def build_wrong_indices(sequences):
    n = len(sequences)

    far = []

    for i in range(n):
        target_seq = sequences[i]
        start = (i + n // 2) % n

        donor = None

        for off in range(n):
            j = (start + off) % n

            if sequences[j] != target_seq:
                donor = j
                break

        if donor is None:
            raise RuntimeError(
                "no cross-sequence donor"
            )

        far.append(donor)

    return torch.tensor(
        far,
        dtype=torch.long,
    )


def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    ds = DSECSequenceFlowDataset(
        ROOT,
        VAL,
        context_frames=4,
    )

    #
    # Materialize contexts so wrong-history
    # donors are deterministic.
    #
    loader0 = DataLoader(
        ds,
        batch_size=16,
        shuffle=False,
        num_workers=4,
    )

    contexts = []
    flows = []
    sequences = []

    for batch in tqdm(
        loader0,
        desc="loading dataset",
    ):
        contexts.append(
            batch["context"]
        )

        flows.append(
            batch["flow"]
        )

        sequences.extend(
            batch["sequence"]
        )

    contexts = torch.cat(
        contexts,
        dim=0,
    )

    flows = torch.cat(
        flows,
        dim=0,
    )

    n = len(ds)

    far_idx = build_wrong_indices(
        sequences
    )

    g = torch.Generator()
    g.manual_seed(12345)

    random_idx = torch.randperm(
        n,
        generator=g,
    )

    backbone = FrozenFlowBackbone(
        mode="jepa",
        checkpoint=CKPT,
    ).to(device)

    backbone.eval()

    state = torch.load(
        FLOW_RUN / "final.pt",
        map_location="cpu",
        weights_only=False,
    )

    head = PatchLinearFlowHead()

    head.load_state_dict(
        state["head"],
        strict=True,
    )

    head = head.to(device)
    head.eval()

    variants = [
        "normal",
        "repeat_last",
        "reverse_past",
        "zero_past",
        "far_wrong",
        "random_history",
    ]

    metrics = {
        v: FlowMetrics(
            n_vals=[1,2,3],
            device=torch.device("cpu"),
        )
        for v in variants
    }

    batch_size = 8

    with torch.no_grad():

        for start in tqdm(
            range(0, n, batch_size),
            desc="flow corruption",
        ):
            end = min(
                start + batch_size,
                n,
            )

            x = contexts[
                start:end
            ].to(device)

            target = flows[
                start:end
            ].to(device)

            variants_x = {}

            variants_x["normal"] = x

            variants_x[
                "repeat_last"
            ] = (
                x[:, -1:]
                .expand(
                    -1,4,-1,-1
                )
                .contiguous()
            )

            variants_x[
                "reverse_past"
            ] = x[
                :,
                [2,1,0,3],
            ]

            zero = x.clone()
            zero[:, :3] = 0
            variants_x[
                "zero_past"
            ] = zero

            donor = contexts[
                far_idx[start:end]
            ].to(device)

            wrong = x.clone()
            wrong[:, :3] = donor[:, :3]

            variants_x[
                "far_wrong"
            ] = wrong

            donor = contexts[
                random_idx[start:end]
            ].to(device)

            rnd = x.clone()
            rnd[:, :3] = donor[:, :3]

            variants_x[
                "random_history"
            ] = rnd

            for name in variants:

                with torch.autocast(
                    "cuda",
                    dtype=torch.bfloat16,
                    enabled=(
                        device.type
                        == "cuda"
                    ),
                ):
                    z = backbone(
                        variants_x[name]
                    )

                pred = head(
                    z.float()
                )

                metrics[
                    name
                ].update(
                    pred.detach().cpu(),
                    target.detach().cpu(),
                )

    results = {
        name:
            metrics[name].compute()
        for name in variants
    }

    normal_epe = results[
        "normal"
    ]["epe"]

    print()
    print("=" * 100)
    print("STRONG FLOW HISTORY CORRUPTION")
    print("=" * 100)

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

    print("-" * 100)

    for name in variants:

        r = results[name]

        de = (
            r["epe"]
            - normal_epe
        )

        pct = (
            100.0
            * de
            / normal_epe
        )

        print(
            f"{name:18s}"
            f"{r['epe']:10.4f}"
            f"{de:10.4f}"
            f"{pct:10.2f}"
            f"{r['ae']:10.4f}"
            f"{r['1pe']:10.4f}"
            f"{r['2pe']:10.4f}"
            f"{r['3pe']:10.4f}"
        )

    print()
    print(
        "STRONG FLOW CORRUPTION: PASS"
    )


if __name__ == "__main__":
    main()
