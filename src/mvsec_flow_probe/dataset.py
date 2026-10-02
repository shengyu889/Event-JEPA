from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset


class MVSECFlowProbeDataset(Dataset):
    """
    Causal MVSEC 20-Hz flow probe.

    Each sample:
        context [4,256,384]

        z_j contains events:
            (t_(j-1), t_j]

        target flow:
            Flow[i] = t_i -> t_(i+1)

        context:
            z_(i-3), z_(i-2), z_(i-1), z_i

    Therefore the representation never sees events from
    the target flow interval (t_i, t_(i+1)].
    """

    def __init__(
        self,
        root: str | Path,
        sequence: str,
        n_tokens: int = 256,
        embed_dim: int = 384,
    ):
        super().__init__()

        self.root = Path(root)
        self.sequence = str(sequence)
        self.n_tokens = int(n_tokens)
        self.embed_dim = int(embed_dim)

        self.sequence_root = (
            self.root
            / "processed"
            / self.sequence
        )

        self.token_dir = (
            self.sequence_root
            / "eventToken"
        )

        self.flow_dir = (
            self.sequence_root
            / "flow"
        )

        manifest_path = (
            self.sequence_root
            / "manifest.csv"
        )

        if not manifest_path.is_file():
            raise FileNotFoundError(
                manifest_path
            )

        with manifest_path.open(
            newline=""
        ) as fp:
            rows = list(
                csv.DictReader(fp)
            )

        if not rows:
            raise RuntimeError(
                f"empty manifest: {manifest_path}"
            )

        self.samples = []

        previous_flow_index = None

        for row in rows:

            if row["sequence"] != self.sequence:
                raise RuntimeError(
                    "manifest sequence mismatch: "
                    f"{row['sequence']} "
                    f"vs {self.sequence}"
                )

            flow_index = int(
                row["flow_index"]
            )

            context_indices = [
                int(row[f"context_{j}"])
                for j in range(4)
            ]

            expected_context = [
                flow_index - 3,
                flow_index - 2,
                flow_index - 1,
                flow_index,
            ]

            if (
                context_indices
                != expected_context
            ):
                raise RuntimeError(
                    f"bad context for flow "
                    f"{flow_index}: "
                    f"{context_indices}"
                )

            if (
                previous_flow_index
                is not None
                and flow_index
                != previous_flow_index + 1
            ):
                raise RuntimeError(
                    "non-consecutive manifest: "
                    f"{previous_flow_index} "
                    f"-> {flow_index}"
                )

            previous_flow_index = (
                flow_index
            )

            context_paths = [
                self.token_dir
                / f"{j:06d}.pt"
                for j in context_indices
            ]

            flow_path = (
                self.sequence_root
                / row["flow_file"]
            )

            for p in context_paths:
                if not p.is_file():
                    raise FileNotFoundError(p)

            if not flow_path.is_file():
                raise FileNotFoundError(
                    flow_path
                )

            flow_start = float(
                row[
                    "flow_start_timestamp"
                ]
            )

            flow_end = float(
                row[
                    "flow_end_timestamp"
                ]
            )

            if not (
                flow_end > flow_start
            ):
                raise RuntimeError(
                    "invalid flow interval: "
                    f"{flow_start}, {flow_end}"
                )

            self.samples.append(
                {
                    "flow_index":
                        flow_index,

                    "context_indices":
                        context_indices,

                    "context_paths":
                        context_paths,

                    "flow_path":
                        flow_path,

                    "flow_start":
                        flow_start,

                    "flow_end":
                        flow_end,
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ):
        sample = self.samples[index]

        tokens = []

        for path in sample[
            "context_paths"
        ]:
            x = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            ).float()

            if tuple(x.shape) != (
                self.n_tokens,
                self.embed_dim,
            ):
                raise RuntimeError(
                    f"bad token shape "
                    f"{tuple(x.shape)}: "
                    f"{path}"
                )

            if not torch.isfinite(
                x
            ).all():
                raise RuntimeError(
                    f"non-finite token: "
                    f"{path}"
                )

            tokens.append(x)

        context = torch.stack(
            tokens,
            dim=0,
        )

        flow = torch.load(
            sample["flow_path"],
            map_location="cpu",
            weights_only=True,
        ).float()

        if tuple(flow.shape) != (
            3,
            224,
            224,
        ):
            raise RuntimeError(
                f"bad flow shape "
                f"{tuple(flow.shape)}: "
                f"{sample['flow_path']}"
            )

        if not torch.isfinite(
            flow
        ).all():
            raise RuntimeError(
                "non-finite flow: "
                f"{sample['flow_path']}"
            )

        valid = flow[2]

        if (
            valid.min() < 0
            or valid.max() > 1
        ):
            raise RuntimeError(
                "invalid flow mask"
            )

        return {
            "context":
                context,

            "flow":
                flow,

            "sequence":
                self.sequence,

            "flow_index":
                sample[
                    "flow_index"
                ],

            "context_indices":
                torch.tensor(
                    sample[
                        "context_indices"
                    ],
                    dtype=torch.int64,
                ),

            "flow_start":
                sample[
                    "flow_start"
                ],

            "flow_end":
                sample[
                    "flow_end"
                ],

            "delta_t":
                sample[
                    "flow_end"
                ]
                - sample[
                    "flow_start"
                ],
        }
