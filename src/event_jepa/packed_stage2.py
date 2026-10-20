from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PackedStage2Dataset(Dataset):
    """
    Sequence-safe mmap Stage-2 dataset.

    Storage:
        <root>/packed/
          train_tokens.npy
          train_timestamps.npy
          train_window_starts.npy

          val_tokens.npy
          val_timestamps.npy
          val_window_starts.npy

    A valid sample always contains:

      [t-3,t-2,t-1,t,t+1,t+2,t+3,t+4]

    and window_starts.npy guarantees the 8 frames
    never cross a sequence boundary.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        n_tokens: int = 256,
        embed_dim: int = 384,
    ):
        if split not in {
            "train",
            "val",
        }:
            raise ValueError(
                split
            )

        self.root = Path(root)
        self.split = split

        self.n_tokens = int(
            n_tokens
        )

        self.embed_dim = int(
            embed_dim
        )

        packed = (
            self.root / "packed"
        )

        self.token_path = (
            packed
            / f"{split}_tokens.npy"
        )

        self.timestamp_path = (
            packed
            / f"{split}_timestamps.npy"
        )

        self.starts_path = (
            packed
            / f"{split}_window_starts.npy"
        )

        for p in (
            self.token_path,
            self.timestamp_path,
            self.starts_path,
        ):
            if not p.is_file():
                raise FileNotFoundError(p)

        tokens = np.load(
            self.token_path,
            mmap_mode="r",
        )

        timestamps = np.load(
            self.timestamp_path,
            mmap_mode="r",
        )

        starts = np.load(
            self.starts_path,
            mmap_mode="r",
        )

        if tokens.ndim != 3:
            raise ValueError(
                tokens.shape
            )

        if tuple(
            tokens.shape[1:]
        ) != (
            self.n_tokens,
            self.embed_dim,
        ):
            raise ValueError(
                tokens.shape
            )

        if timestamps.ndim != 1:
            raise ValueError(
                timestamps.shape
            )

        if len(tokens) != len(
            timestamps
        ):
            raise ValueError(
                "token/timestamp mismatch"
            )

        if starts.ndim != 1:
            raise ValueError(
                starts.shape
            )

        if len(starts) == 0:
            raise ValueError(
                "no valid windows"
            )

        if (
            int(starts.min()) < 0
            or int(starts.max()) + 7
            >= len(tokens)
        ):
            raise ValueError(
                "invalid window starts"
            )

        self.num_frames = int(
            len(tokens)
        )

        self.num_samples = int(
            len(starts)
        )

        # Do not retain mmap opened in parent process.
        self._tokens = None
        self._timestamps = None
        self._starts = None

    def __getstate__(self):
        state = (
            self.__dict__.copy()
        )

        state["_tokens"] = None
        state["_timestamps"] = None
        state["_starts"] = None

        return state

    def _ensure_open(self):
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

        if self._starts is None:
            self._starts = np.load(
                self.starts_path,
                mmap_mode="r",
            )

    def __len__(self):
        return self.num_samples

    def __getitem__(
        self,
        index: int,
    ):
        if index < 0:
            index += len(self)

        if not (
            0 <= index < len(self)
        ):
            raise IndexError(index)

        self._ensure_open()

        start = int(
            self._starts[index]
        )

        stop = start + 8

        # One contiguous disk/mmap access instead of
        # seven independent torch.load calls.
        window_np = np.asarray(
            self._tokens[
                start:stop
            ],
            dtype=np.float32,
        ).copy()

        ts_np = np.asarray(
            self._timestamps[
                start:stop
            ],
            dtype=np.int64,
        ).copy()

        if not np.all(
            np.diff(ts_np) > 0
        ):
            raise RuntimeError(
                f"non-increasing timestamps "
                f"at sample {index}"
            )

        window = torch.from_numpy(
            window_np
        )

        timestamps = torch.from_numpy(
            ts_np
        )

        if tuple(
            window.shape
        ) != (
            8,
            self.n_tokens,
            self.embed_dim,
        ):
            raise RuntimeError(
                window.shape
            )

        context = window[:4]

        target = window[
            torch.tensor(
                [4, 5, 7],
                dtype=torch.long,
            )
        ]

        context_ts = (
            timestamps[:4]
        )

        target_ts = timestamps[
            torch.tensor(
                [4, 5, 7],
                dtype=torch.long,
            )
        ]

        delta_t = (
            target_ts
            - context_ts[-1]
        ).to(
            torch.float32
        ) * 1e-6

        return {
            "window":
                window,
            "timestamps":
                timestamps,
            "context":
                context,
            "target":
                target,
            "delta_t":
                delta_t,
            "context_timestamps":
                context_ts,
            "target_timestamps":
                target_ts,
        }
