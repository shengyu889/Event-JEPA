from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def decode_dsec_flow(path: str | Path) -> torch.Tensor:
    """
    Decode official DSEC 16-bit optical-flow PNG.

    Returns:
        [3,H,W] float32
        channel 0: u
        channel 1: v
        channel 2: valid mask {0,1}
    """
    path = str(path)

    bgr = cv2.imread(
        path,
        cv2.IMREAD_UNCHANGED,
    )

    if bgr is None:
        raise RuntimeError(
            f"failed to read flow PNG: {path}"
        )

    if bgr.dtype != np.uint16:
        raise TypeError(
            f"expected uint16 flow PNG, got {bgr.dtype}: {path}"
        )

    rgb = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2RGB,
    ).astype(np.float32)

    valid = rgb[..., 2] > 0

    u = (
        rgb[..., 0] - 32768.0
    ) / 128.0

    v = (
        rgb[..., 1] - 32768.0
    ) / 128.0

    flow = np.stack(
        [
            u,
            v,
            valid.astype(np.float32),
        ],
        axis=0,
    )

    return torch.from_numpy(flow)


def center_crop_flow(
    flow: torch.Tensor,
    crop_h: int = 224,
    crop_w: int = 224,
) -> torch.Tensor:
    """
    Match the deterministic DSEC Stage-1 token preprocessing:
        PadToMinSide(224,224) -> CenterCrop(224,224)

    Since DSEC flow is larger than 224x224, no padding occurs.
    """
    if flow.ndim != 3:
        raise ValueError(
            f"expected [C,H,W], got {tuple(flow.shape)}"
        )

    _, h, w = flow.shape

    if h < crop_h or w < crop_w:
        raise ValueError(
            f"flow too small for crop: {(h,w)}"
        )

    top = (h - crop_h) // 2
    left = (w - crop_w) // 2

    return flow[
        :,
        top : top + crop_h,
        left : left + crop_w,
    ].contiguous()


class DSECSequenceFlowDataset(Dataset):
    """
    Sequence-aware DSEC flow dataset built directly from precomputed
    Stage-1 event tokens.

    Input:
        context: [Tc, 256, 384]

    Target:
        flow: [3, 224, 224]
            u, v, valid

    Critically, context ends at the FROM timestamp of the flow.
    No future token is ever included.
    """

    def __init__(
        self,
        root: str | Path,
        sequences: Iterable[str],
        context_frames: int,
        n_tokens: int = 256,
        embed_dim: int = 384,
        crop_size: int = 224,
        max_frame_gap_us: int = 75000,
    ):
        super().__init__()

        self.root = Path(root)
        self.sequences = list(sequences)
        self.context_frames = int(context_frames)
        self.n_tokens = int(n_tokens)
        self.embed_dim = int(embed_dim)
        self.crop_size = int(crop_size)
        self.max_frame_gap_us = int(
            max_frame_gap_us
        )

        if self.context_frames < 1:
            raise ValueError(
                "context_frames must be >= 1"
            )

        self.flow_root = (
            self.root / "train_optical_flow"
        )

        self.token_root = (
            self.root / "train_images"
        )

        self.samples = []
        self.skipped_missing_source = 0
        self.skipped_short_context = 0
        self.skipped_context_gap = 0

        for sequence in self.sequences:
            self._add_sequence(sequence)

        if not self.samples:
            raise RuntimeError(
                "no valid DSEC flow samples"
            )

    def _add_sequence(
        self,
        sequence: str,
    ):
        seq_flow_root = (
            self.flow_root
            / sequence
            / "flow"
        )

        forward_dir = (
            seq_flow_root / "forward"
        )

        timestamps_file = (
            seq_flow_root
            / "forward_timestamps.txt"
        )

        token_dir = (
            self.token_root
            / sequence
            / "images"
            / "left"
            / "eventToken"
        )

        if not forward_dir.is_dir():
            raise FileNotFoundError(
                forward_dir
            )

        if not timestamps_file.is_file():
            raise FileNotFoundError(
                timestamps_file
            )

        if not token_dir.is_dir():
            raise FileNotFoundError(
                token_dir
            )

        flow_paths = sorted(
            forward_dir.glob("*.png"),
            key=lambda p: int(p.stem),
        )

        lines = [
            line.strip()
            for line in
            timestamps_file.read_text().splitlines()
            if line.strip()
        ]

        if not lines:
            raise RuntimeError(
                f"empty timestamp file: {timestamps_file}"
            )

        # Skip header:
        # # from_timestamp_us, to_timestamp_us
        timestamp_rows = lines[1:]

        if len(flow_paths) != len(timestamp_rows):
            raise RuntimeError(
                f"{sequence}: flow/timestamp mismatch: "
                f"{len(flow_paths)} vs "
                f"{len(timestamp_rows)}"
            )

        token_paths = sorted(
            token_dir.glob("*.pt"),
            key=lambda p: int(p.stem),
        )

        token_timestamps = [
            int(p.stem)
            for p in token_paths
        ]

        token_index = {
            ts: i
            for i, ts in enumerate(
                token_timestamps
            )
        }

        for flow_path, row in zip(
            flow_paths,
            timestamp_rows,
        ):
            parts = [
                x.strip()
                for x in row.split(",")
            ]

            if len(parts) < 2:
                raise RuntimeError(
                    f"invalid timestamp row: {row}"
                )

            from_ts = int(parts[0])
            to_ts = int(parts[1])

            if from_ts not in token_index:
                self.skipped_missing_source += 1
                continue

            source_idx = token_index[from_ts]

            start_idx = (
                source_idx
                - self.context_frames
                + 1
            )

            if start_idx < 0:
                self.skipped_short_context += 1
                continue

            context_paths = token_paths[
                start_idx :
                source_idx + 1
            ]

            context_ts = token_timestamps[
                start_idx :
                source_idx + 1
            ]

            # Require approximately continuous ~50 ms
            # token cadence. This prevents silently spanning
            # a preprocessing gap.
            if len(context_ts) > 1:
                gaps = np.diff(
                    np.asarray(
                        context_ts,
                        dtype=np.int64,
                    )
                )

                if (
                    gaps <= 0
                ).any() or (
                    gaps > self.max_frame_gap_us
                ).any():
                    self.skipped_context_gap += 1
                    continue

            self.samples.append(
                {
                    "sequence": sequence,
                    "flow_path": flow_path,
                    "from_ts": from_ts,
                    "to_ts": to_ts,
                    "context_paths":
                        context_paths,
                    "context_ts":
                        context_ts,
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

        for path in sample["context_paths"]:
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
                    f"{tuple(x.shape)}: {path}"
                )

            if not torch.isfinite(x).all():
                raise RuntimeError(
                    f"non-finite token: {path}"
                )

            tokens.append(x)

        context = torch.stack(
            tokens,
            dim=0,
        )

        flow = decode_dsec_flow(
            sample["flow_path"]
        )

        original_hw = tuple(
            flow.shape[-2:]
        )

        flow = center_crop_flow(
            flow,
            self.crop_size,
            self.crop_size,
        )

        return {
            "context": context,
            "flow": flow,
            "sequence":
                sample["sequence"],
            "from_ts":
                sample["from_ts"],
            "to_ts":
                sample["to_ts"],
            "delta_t":
                (
                    sample["to_ts"]
                    - sample["from_ts"]
                ) / 1e6,
            "context_ts":
                torch.tensor(
                    sample["context_ts"],
                    dtype=torch.int64,
                ),
            "original_flow_hw":
                torch.tensor(
                    original_hw,
                    dtype=torch.int64,
                ),
        }
