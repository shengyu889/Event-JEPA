from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PackedStage2Dataset(Dataset):
    """
    Shared packed temporal dataset for matched Stage-2 training.

    One sample always corresponds to exactly 8 consecutive frames:

        [t-3, t-2, t-1, t, t+1, t+2, t+3, t+4]

    Returned tensors:
        window    : [8, N, D]       -- GEP-AR
        context   : [4, N, D]       -- Event-JEPA
        target    : [3, N, D]       -- horizons [1,2,4]
        delta_t   : [3] seconds
        timestamps: [8] int64 us
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        n_tokens: int = 256,
        embed_dim: int = 384,
    ) -> None:
        super().__init__()

        if split not in {"train", "val"}:
            raise ValueError(
                f"split must be train/val, got {split}"
            )

        self.root = Path(root)
        self.split = split
        self.n_tokens = int(n_tokens)
        self.embed_dim = int(embed_dim)

        packed = self.root / "packed"

        self.token_path = (
            packed / f"{split}_tokens.npy"
        )
        self.timestamp_path = (
            packed / f"{split}_timestamps.npy"
        )

        if not self.token_path.is_file():
            raise FileNotFoundError(
                self.token_path
            )

        if not self.timestamp_path.is_file():
            raise FileNotFoundError(
                self.timestamp_path
            )

        # Read only metadata here.
        tokens = np.load(
            self.token_path,
            mmap_mode="r",
        )
        timestamps = np.load(
            self.timestamp_path,
            mmap_mode="r",
        )

        expected_tail = (
            self.n_tokens,
            self.embed_dim,
        )

        if tokens.ndim != 3:
            raise ValueError(
                f"tokens must be [T,N,D], "
                f"got {tokens.shape}"
            )

        if tuple(tokens.shape[1:]) != expected_tail:
            raise ValueError(
                f"expected token shape (*,"
                f"{self.n_tokens},"
                f"{self.embed_dim}), "
                f"got {tokens.shape}"
            )

        if timestamps.ndim != 1:
            raise ValueError(
                "timestamps must be 1-D"
            )

        if len(tokens) != len(timestamps):
            raise ValueError(
                "token/timestamp count mismatch"
            )

        if len(tokens) < 8:
            raise ValueError(
                "need at least 8 frames"
            )

        gaps = np.diff(
            np.asarray(
                timestamps,
                dtype=np.int64,
            )
        )

        if not np.all(gaps > 0):
            raise ValueError(
                "timestamps must be strictly increasing"
            )

        self.num_frames = int(len(tokens))
        self.num_samples = (
            self.num_frames - 7
        )

        # Important:
        # don't keep mmap objects created in parent process.
        # Every DataLoader worker lazily opens its own mmap.
        self._tokens = None
        self._timestamps = None

    def _ensure_open(self) -> None:
        if self._tokens is None:
            self._tokens = np.load(
                self.token_path,
                mmap_mode="r",
            )

        if self._timestamps is None:
            self._timestamps = np.load(
                self.timestamp_path,
                mmap_mode="r",
            )

    def __getstate__(self):
        state = self.__dict__.copy()

        # Re-open mmap independently inside workers.
        state["_tokens"] = None
        state["_timestamps"] = None

        return state

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor]:
        if index < 0:
            index += len(self)

        if not (
            0 <= index < len(self)
        ):
            raise IndexError(index)

        self._ensure_open()

        start = index
        stop = index + 8

        # copy() deliberately detaches from read-only mmap
        # and gives torch a writable contiguous array.
        window_np = np.asarray(
            self._tokens[start:stop],
            dtype=np.float32,
        ).copy()

        ts_np = np.asarray(
            self._timestamps[start:stop],
            dtype=np.int64,
        ).copy()

        window = torch.from_numpy(
            window_np
        )

        timestamps = torch.from_numpy(
            ts_np
        )

        if tuple(window.shape) != (
            8,
            self.n_tokens,
            self.embed_dim,
        ):
            raise RuntimeError(
                f"unexpected window shape "
                f"{tuple(window.shape)}"
            )

        # Anchor t is the fourth visible context frame.
        context = window[:4]

        # Relative to window:
        # index 4 = t+1
        # index 5 = t+2
        # index 7 = t+4
        target = window[
            torch.tensor(
                [4, 5, 7],
                dtype=torch.long,
            )
        ]

        context_ts = timestamps[:4]

        target_ts = timestamps[
            torch.tensor(
                [4, 5, 7],
                dtype=torch.long,
            )
        ]

        delta_t = (
            target_ts
            - context_ts[-1]
        ).to(torch.float32) * 1e-6

        return {
            "window": window,
            "timestamps": timestamps,
            "context": context,
            "target": target,
            "delta_t": delta_t,
            "context_timestamps":
                context_ts,
            "target_timestamps":
                target_ts,
        }
