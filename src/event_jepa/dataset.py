from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class TemporalSample:
    sequence: str
    context_paths: tuple[Path, ...]
    target_paths: tuple[Path, ...]
    context_timestamps: tuple[int, ...]
    target_timestamps: tuple[int, ...]


class EventJEPADataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        context_frames: int,
        horizons: tuple[int, ...],
        n_tokens: int,
        embed_dim: int,
        timestamp_scale: float,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.context_frames = context_frames
        self.horizons = tuple(horizons)
        self.n_tokens = n_tokens
        self.embed_dim = embed_dim
        self.timestamp_scale = timestamp_scale
        self.samples: list[TemporalSample] = []
        split_root = self.root / f"{split}_images"
        if not split_root.is_dir():
            raise FileNotFoundError(f"token split directory not found: {split_root}")
        for sequence_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
            token_dir = sequence_dir / "images" / "left" / "eventToken"
            if not token_dir.is_dir():
                continue
            files = sorted(token_dir.glob("*.pt"), key=self._timestamp)
            self._validate_sequence(files)
            last_horizon = max(self.horizons)
            for end in range(context_frames - 1, len(files) - last_horizon):
                context = tuple(files[end - context_frames + 1 : end + 1])
                targets = tuple(files[end + horizon] for horizon in self.horizons)
                self.samples.append(
                    TemporalSample(
                        sequence=sequence_dir.name,
                        context_paths=context,
                        target_paths=targets,
                        context_timestamps=tuple(self._timestamp(path) for path in context),
                        target_timestamps=tuple(self._timestamp(path) for path in targets),
                    )
                )

    @staticmethod
    def _timestamp(path: Path) -> int:
        try:
            return int(path.stem)
        except ValueError as error:
            raise ValueError(f"token filename stem must be numeric: {path}") from error

    def _load_token(self, path: Path) -> torch.Tensor:
        token = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(token, torch.Tensor):
            raise TypeError(f"token file must contain a Tensor: {path}")
        token = token.detach().to(dtype=torch.float32)
        expected = (self.n_tokens, self.embed_dim)
        if tuple(token.shape) != expected:
            raise ValueError(
                f"invalid token shape at {path}: expected {expected}, got {tuple(token.shape)}"
            )
        return token

    def _validate_sequence(self, files: list[Path]) -> None:
        for path in files:
            self._timestamp(path)
            self._load_token(path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        context = torch.stack([self._load_token(path) for path in sample.context_paths])
        target = torch.stack([self._load_token(path) for path in sample.target_paths])
        context_ts = torch.tensor(sample.context_timestamps, dtype=torch.int64)
        target_ts = torch.tensor(sample.target_timestamps, dtype=torch.int64)
        delta_t = (target_ts - context_ts[-1]).to(torch.float32) * self.timestamp_scale
        return {
            "context": context,
            "target": target,
            "delta_t": delta_t,
            "sequence": sample.sequence,
            "context_timestamps": context_ts,
            "target_timestamps": target_ts,
        }
