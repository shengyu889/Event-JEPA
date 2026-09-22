# Event-JEPA GEP Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a tested token-level Event-JEPA Stage-2 pretraining path to the official GEP repository while preserving a controlled GEP-vs-JEPA comparison and GEP downstream checkpoint compatibility.

**Architecture:** A frame-aware dataset reads frozen Stage-1 `eventToken/*.pt` files and returns past context tokens, future target tokens, and measured elapsed time. An online event-token Transformer encodes context, an EMA copy encodes targets, and a continuous-time cross-attention predictor predicts target latents using normalized cosine loss. The online Transformer's GEP-compatible submodule is exported for existing downstream evaluation.

**Tech Stack:** Python 3.12, PyTorch 2.6, torchvision 0.21, PyYAML, pytest, TensorBoard, DDP/NCCL.

**Spec:** `docs/superpowers/specs/2026-09-20-event-jepa-gep-integration-design.md`

## Global Constraints

- Target upstream is `uzh-rpg/generative_event_pretraining`, branch `master`, commit `b07c78c`.
- V1 consumes frozen Stage-1 event token files and does not claim end-to-end raw-event learning.
- Default GEP ViT-S token shape is `N=256`, `D=384`; every loaded tensor is validated.
- Context and targets must remain inside one sequence; numeric filename stems are mandatory timestamps.
- V1 trains only cosine joint-embedding loss; reconstruction, contrastive, rate-invariance, and motion losses are excluded.
- The EMA target encoder never receives gradients and updates only after a successful optimizer step.
- RTX 5070 Ti 16 GB is the local smoke/evaluation target; 4 x A100 80 GB is the full DDP training target.
- Existing GEP entry points remain usable; new Event-JEPA modules must not import side-effectful `src/config.py`.
- All new behavior is developed test-first and every task ends with a focused commit.

## Review Focus

- A sequence containing one token file with a mismatched `[N,D]` shape must fail during manifest construction and identify the exact file.
- A nonnumeric token filename must fail explicitly; it must never be silently converted to a frame index.
- Two adjacent sequence directories must never form one context-target sample, even when timestamps are numerically adjacent.
- Variable `Tc` and `K` within configured maxima must preserve `[B,K,N,D]`; values exceeding maxima must raise `ValueError`.
- A GradScaler-skipped optimizer step must not update EMA weights or increment the optimizer-step counter.

---

### Task 1: Side-effect-free configuration and test scaffold

**Files:**
- Create: `src/event_jepa/__init__.py`
- Create: `src/event_jepa/config.py`
- Create: `tests/test_jepa_config.py`
- Create: `pytest.ini`
- Create: `configs/event_jepa_smoke.yaml`
- Create: `configs/event_jepa_a100.yaml`
- Modify: `environment.yml`

**Interfaces:**
- Consumes: YAML mapping loaded from disk.
- Produces: `EventJEPAConfig`, `load_config(path: str | Path) -> EventJEPAConfig`, and `to_dict() -> dict`.

- [ ] **Step 1: Add the configuration tests**

```python
# tests/test_jepa_config.py
from pathlib import Path

import pytest

from event_jepa.config import EventJEPAConfig, load_config


def test_load_config_parses_horizons_and_paths(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "data_root: /tokens\noutput_dir: /runs/test\n"
        "context_frames: 2\nhorizons: [1, 3]\n"
        "n_tokens: 16\nembed_dim: 32\nnum_heads: 4\n",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.data_root == Path("/tokens")
    assert cfg.output_dir == Path("/runs/test")
    assert cfg.horizons == (1, 3)
    assert cfg.max_context_frames >= cfg.context_frames


def test_load_config_rejects_unknown_keys(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text("data_root: /tokens\nunknown_option: 4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown_option"):
        load_config(path)


def test_config_rejects_invalid_model_dimensions():
    with pytest.raises(ValueError, match="divisible"):
        EventJEPAConfig(data_root=Path("/tokens"), embed_dim=30, num_heads=8)


def test_config_rejects_nonpositive_horizon():
    with pytest.raises(ValueError, match="horizons"):
        EventJEPAConfig(data_root=Path("/tokens"), horizons=(0,))
```

- [ ] **Step 2: Run the tests and confirm the missing package failure**

Run: `PYTHONPATH=src pytest tests/test_jepa_config.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'event_jepa'`.

- [ ] **Step 3: Implement the immutable configuration**

```python
# src/event_jepa/config.py
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class EventJEPAConfig:
    data_root: Path
    output_dir: Path = Path("runs/event_jepa")
    seed: int = 0
    timestamp_scale: float = 1e-6
    context_frames: int = 4
    horizons: tuple[int, ...] = (1,)
    n_tokens: int = 256
    embed_dim: int = 384
    num_heads: int = 6
    encoder_layers: int = 12
    predictor_layers: int = 2
    max_positions: int = 4096
    max_context_frames: int = 8
    batch_size: int = 8
    grad_accum_steps: int = 1
    num_workers: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    min_learning_rate: float = 0.0
    warmup_steps: int = 1000
    total_steps: int = 100000
    ema_start: float = 0.996
    ema_end: float = 0.9999
    precision: str = "bf16"
    grad_clip_norm: float = 1.0
    log_every: int = 100
    validate_every: int = 1000
    save_every: int = 1000

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_root", Path(self.data_root))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "horizons", tuple(self.horizons))
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if self.context_frames < 1 or self.context_frames > self.max_context_frames:
            raise ValueError("context_frames must be within [1, max_context_frames]")
        if not self.horizons or any(h < 1 for h in self.horizons):
            raise ValueError("horizons must contain positive frame offsets")
        if tuple(sorted(set(self.horizons))) != self.horizons:
            raise ValueError("horizons must be unique and increasing")
        if self.n_tokens * self.context_frames > self.max_positions:
            raise ValueError("context token count exceeds max_positions")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16, or bf16")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["data_root"] = str(self.data_root)
        result["output_dir"] = str(self.output_dir)
        result["horizons"] = list(self.horizons)
        return result


def load_config(path: str | Path) -> EventJEPAConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    allowed = {field.name for field in fields(EventJEPAConfig)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown configuration keys: {', '.join(unknown)}")
    return EventJEPAConfig(**raw)
```

Export `EventJEPAConfig` and `load_config` from `src/event_jepa/__init__.py`. Add `pytest==8.3.5` and `pyyaml==6.0.2` to the pip section of `environment.yml`. Configure `pytest.ini` with `testpaths = tests` and `pythonpath = src`.

- [ ] **Step 4: Add exact smoke and full-training YAML profiles**

```yaml
# configs/event_jepa_smoke.yaml
data_root: /path/to/DSEC
output_dir: runs/event_jepa_smoke
context_frames: 2
horizons: [1]
n_tokens: 256
embed_dim: 384
num_heads: 6
encoder_layers: 2
predictor_layers: 1
max_positions: 4096
max_context_frames: 8
batch_size: 2
grad_accum_steps: 4
num_workers: 2
learning_rate: 0.0001
weight_decay: 0.00001
warmup_steps: 2
total_steps: 20
precision: bf16
log_every: 1
validate_every: 10
save_every: 10
```

```yaml
# configs/event_jepa_a100.yaml
data_root: /path/to/DSEC
output_dir: runs/event_jepa_a100
context_frames: 4
horizons: [1]
n_tokens: 256
embed_dim: 384
num_heads: 6
encoder_layers: 12
predictor_layers: 2
max_positions: 4096
max_context_frames: 8
batch_size: 16
grad_accum_steps: 1
num_workers: 8
learning_rate: 0.0001
weight_decay: 0.00001
warmup_steps: 1000
total_steps: 100000
precision: bf16
log_every: 100
validate_every: 1000
save_every: 1000
```

- [ ] **Step 5: Run tests and configuration parsing checks**

Run: `pytest tests/test_jepa_config.py -q`

Expected: `4 passed`.

Run: `python -c "from event_jepa.config import load_config; print(load_config('configs/event_jepa_smoke.yaml').to_dict())"`

Expected: one dictionary containing `context_frames: 2`, `batch_size: 2`, and `total_steps: 20`.

- [ ] **Step 6: Commit the configuration layer**

```bash
git add environment.yml pytest.ini configs src/event_jepa tests/test_jepa_config.py
git commit -m "feat: add Event-JEPA configuration profiles"
```

### Task 2: Frame-aware temporal token dataset

**Files:**
- Create: `src/event_jepa/dataset.py`
- Create: `tests/test_jepa_dataset.py`
- Modify: `src/event_jepa/__init__.py`

**Interfaces:**
- Consumes: `root`, split, `context_frames`, frame-offset `horizons`, expected token dimensions, and `timestamp_scale`.
- Produces: `EventJEPADataset`, whose item is a dictionary containing `context [Tc,N,D]`, `target [K,N,D]`, `delta_t [K]`, `sequence`, and timestamp tensors.

- [ ] **Step 1: Add reusable token-sequence fixtures and the boundary test**

```python
# tests/test_jepa_dataset.py
from pathlib import Path

import pytest
import torch

from event_jepa.dataset import EventJEPADataset


def write_sequence(root: Path, name: str, timestamps: list[int], value: float, shape=(4, 8)):
    folder = root / "train_images" / name / "images" / "left" / "eventToken"
    folder.mkdir(parents=True)
    for index, timestamp in enumerate(timestamps):
        torch.save(torch.full(shape, value + index), folder / f"{timestamp}.pt")


def test_dataset_keeps_samples_inside_sequence(tmp_path: Path):
    write_sequence(tmp_path, "seq_a", [100, 110, 120, 130], 10.0)
    write_sequence(tmp_path, "seq_b", [131, 141, 151, 161], 20.0)
    dataset = EventJEPADataset(
        tmp_path, "train", context_frames=2, horizons=(1,),
        n_tokens=4, embed_dim=8, timestamp_scale=0.1,
    )
    assert len(dataset) == 4
    for sample in dataset:
        context_family = sample["context"].floor().div(10).unique().tolist()
        target_family = sample["target"].floor().div(10).unique().tolist()
        assert context_family == target_family


def test_dataset_returns_measured_delta_t(tmp_path: Path):
    write_sequence(tmp_path, "seq", [100, 110, 145, 200], 0.0)
    dataset = EventJEPADataset(
        tmp_path, "train", context_frames=2, horizons=(1, 2),
        n_tokens=4, embed_dim=8, timestamp_scale=0.001,
    )
    sample = dataset[0]
    assert sample["context"].shape == (2, 4, 8)
    assert sample["target"].shape == (2, 4, 8)
    assert torch.allclose(sample["delta_t"], torch.tensor([0.035, 0.090]))
    assert sample["sequence"] == "seq"
```

- [ ] **Step 2: Run the focused tests and confirm the missing dataset failure**

Run: `pytest tests/test_jepa_dataset.py -q`

Expected: collection fails because `event_jepa.dataset` does not exist.

- [ ] **Step 3: Implement sequence discovery and sample indexing**

```python
# src/event_jepa/dataset.py
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
                self.samples.append(TemporalSample(
                    sequence=sequence_dir.name,
                    context_paths=context,
                    target_paths=targets,
                    context_timestamps=tuple(self._timestamp(path) for path in context),
                    target_timestamps=tuple(self._timestamp(path) for path in targets),
                ))

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
            raise ValueError(f"invalid token shape at {path}: expected {expected}, got {tuple(token.shape)}")
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
```

- [ ] **Step 4: Add malformed filename, bad shape, and insufficient-length tests**

```python
def test_dataset_rejects_nonnumeric_timestamp(tmp_path: Path):
    write_sequence(tmp_path, "seq", [100, 110, 120], 0.0)
    token_dir = tmp_path / "train_images/seq/images/left/eventToken"
    torch.save(torch.zeros(4, 8), token_dir / "bad.pt")
    with pytest.raises(ValueError, match="bad.pt"):
        EventJEPADataset(tmp_path, "train", 2, (1,), 4, 8, 0.001)


def test_dataset_rejects_shape_mismatch_with_path(tmp_path: Path):
    write_sequence(tmp_path, "seq", [100, 110], 0.0)
    bad = tmp_path / "train_images/seq/images/left/eventToken/120.pt"
    torch.save(torch.zeros(5, 8), bad)
    with pytest.raises(ValueError, match=r"120\.pt"):
        EventJEPADataset(tmp_path, "train", 2, (1,), 4, 8, 0.001)


def test_short_sequence_contributes_no_samples(tmp_path: Path):
    write_sequence(tmp_path, "seq", [100, 110], 0.0)
    dataset = EventJEPADataset(tmp_path, "train", 2, (1,), 4, 8, 0.001)
    assert len(dataset) == 0
```

- [ ] **Step 5: Run the full dataset test file**

Run: `pytest tests/test_jepa_dataset.py -q`

Expected: `5 passed`.

- [ ] **Step 6: Commit the dataset**

```bash
git add src/event_jepa/dataset.py src/event_jepa/__init__.py tests/test_jepa_dataset.py
git commit -m "feat: add frame-aware Event-JEPA token dataset"
```

### Task 3: Continuous elapsed-time embedding

**Files:**
- Create: `src/event_jepa/time_embedding.py`
- Create: `tests/test_time_embedding.py`

**Interfaces:**
- Consumes: nonnegative `delta_t` tensor with arbitrary leading shape.
- Produces: `ContinuousTimeEmbedding.forward(delta_t) -> Tensor[*delta_t.shape, embed_dim]`.

- [ ] **Step 1: Write shape, determinism, zero-time, and negative-time tests**

```python
# tests/test_time_embedding.py
import pytest
import torch

from event_jepa.time_embedding import ContinuousTimeEmbedding


def test_time_embedding_is_deterministic_and_shape_correct():
    module = ContinuousTimeEmbedding(embed_dim=32, fourier_dim=16)
    delta_t = torch.tensor([[0.01, 0.05], [0.02, 0.10]])
    first = module(delta_t)
    second = module(delta_t)
    assert first.shape == (2, 2, 32)
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()


def test_time_embedding_accepts_zero():
    output = ContinuousTimeEmbedding(16, 8)(torch.zeros(3))
    assert output.shape == (3, 16)
    assert torch.isfinite(output).all()


def test_time_embedding_rejects_negative_elapsed_time():
    with pytest.raises(ValueError, match="nonnegative"):
        ContinuousTimeEmbedding(16, 8)(torch.tensor([-0.01]))
```

- [ ] **Step 2: Run the tests and verify the module-not-found failure**

Run: `pytest tests/test_time_embedding.py -q`

Expected: collection fails because `event_jepa.time_embedding` does not exist.

- [ ] **Step 3: Implement Fourier features followed by a two-layer MLP**

```python
# src/event_jepa/time_embedding.py
import math

import torch
from torch import nn


class ContinuousTimeEmbedding(nn.Module):
    def __init__(self, embed_dim: int, fourier_dim: int = 64, max_frequency: float = 1000.0):
        super().__init__()
        if fourier_dim < 2 or fourier_dim % 2:
            raise ValueError("fourier_dim must be a positive even integer")
        frequencies = torch.logspace(0.0, math.log10(max_frequency), fourier_dim // 2)
        self.register_buffer("frequencies", frequencies, persistent=True)
        self.projection = nn.Sequential(
            nn.Linear(fourier_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        if torch.any(delta_t < 0):
            raise ValueError("delta_t must be nonnegative")
        angles = 2.0 * torch.pi * delta_t.to(torch.float32).unsqueeze(-1) * self.frequencies
        features = torch.cat((angles.sin(), angles.cos()), dim=-1)
        return self.projection(features)
```

- [ ] **Step 4: Run the time embedding tests**

Run: `pytest tests/test_time_embedding.py -q`

Expected: `3 passed`.

- [ ] **Step 5: Commit the time embedding**

```bash
git add src/event_jepa/time_embedding.py tests/test_time_embedding.py
git commit -m "feat: add continuous time embedding"
```

### Task 4: GEP-compatible context encoder and target-query predictor

**Files:**
- Create: `src/event_jepa/encoders.py`
- Create: `src/event_jepa/predictor.py`
- Create: `tests/test_jepa_components.py`

**Interfaces:**
- Consumes: context tokens `[B,T,N,D]`; predictor additionally consumes memory `[B,T*N,D]` and `delta_t [B,K]`.
- Produces: `EventTokenEncoder.forward -> [B,T*N,D]`, `EventPredictor.forward -> [B,K,N,D]`, and `EventTokenEncoder.gep_transformer_state_dict() -> dict[str,Tensor]`.

- [ ] **Step 1: Add encoder and predictor contract tests**

```python
# tests/test_jepa_components.py
import pytest
import torch

from event_jepa.encoders import EventTokenEncoder
from event_jepa.predictor import EventPredictor


def make_encoder():
    return EventTokenEncoder(
        embed_dim=32, num_heads=4, num_layers=2,
        n_tokens=4, max_positions=32, max_context_frames=3,
    )


def test_encoder_flattens_context_without_losing_embedding_dimension():
    output = make_encoder()(torch.randn(2, 3, 4, 32))
    assert output.shape == (2, 12, 32)


def test_encoder_rejects_context_beyond_configured_maximum():
    with pytest.raises(ValueError, match="max_context_frames"):
        make_encoder()(torch.randn(2, 4, 4, 32))


def test_predictor_supports_multiple_horizons():
    predictor = EventPredictor(32, 4, 2, n_tokens=4, max_horizons=3)
    prediction = predictor(torch.randn(2, 12, 32), torch.tensor([[0.01, 0.05], [0.02, 0.08]]))
    assert prediction.shape == (2, 2, 4, 32)


def test_predictor_rejects_too_many_horizons():
    predictor = EventPredictor(32, 4, 1, n_tokens=4, max_horizons=2)
    with pytest.raises(ValueError, match="max_horizons"):
        predictor(torch.randn(1, 8, 32), torch.tensor([[0.01, 0.02, 0.03]]))
```

- [ ] **Step 2: Run the tests and verify both imports fail**

Run: `pytest tests/test_jepa_components.py -q`

Expected: collection fails because the component modules do not exist.

- [ ] **Step 3: Implement the encoder with an exportable GEP transformer core**

```python
# src/event_jepa/encoders.py
from types import SimpleNamespace

import torch
from torch import nn

from model import Block


class EventTokenEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, n_tokens, max_positions, max_context_frames):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_tokens = n_tokens
        self.max_context_frames = max_context_frames
        block_config = SimpleNamespace(n_embed=embed_dim, n_head=num_heads)
        self.transformer = nn.ModuleDict({
            "modality_embed": nn.Embedding(5, embed_dim),
            "pos_embed": nn.Embedding(max_positions, embed_dim),
            "blocks": nn.ModuleList([Block(block_config) for _ in range(num_layers)]),
            "norm": nn.LayerNorm(embed_dim),
        })
        self.temporal_embed = nn.Embedding(max_context_frames, embed_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 4:
            raise ValueError("tokens must have shape [B,T,N,D]")
        batch, frames, patches, dim = tokens.shape
        if frames > self.max_context_frames:
            raise ValueError("frames exceeds max_context_frames")
        if (patches, dim) != (self.n_tokens, self.embed_dim):
            raise ValueError(f"expected patch shape {(self.n_tokens, self.embed_dim)}")
        spatial_ids = torch.arange(patches, device=tokens.device)
        temporal_ids = torch.arange(frames, device=tokens.device)
        x = tokens
        x = x + self.transformer.pos_embed(spatial_ids)[None, None, :, :]
        x = x + self.temporal_embed(temporal_ids)[None, :, None, :]
        modality_ids = torch.full((batch, frames, patches), 2, device=tokens.device, dtype=torch.long)
        x = x + self.transformer.modality_embed(modality_ids)
        x = x.reshape(batch, frames * patches, dim)
        for block in self.transformer.blocks:
            x = block(x, is_causal=False)
        return self.transformer.norm(x)

    def gep_transformer_state_dict(self) -> dict[str, torch.Tensor]:
        return {key: value.detach().cpu() for key, value in self.transformer.state_dict().items()}
```

- [ ] **Step 4: Implement target queries with continuous-time cross-attention**

```python
# src/event_jepa/predictor.py
import torch
from torch import nn

from event_jepa.time_embedding import ContinuousTimeEmbedding


class EventPredictor(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, n_tokens, max_horizons=8):
        super().__init__()
        self.n_tokens = n_tokens
        self.max_horizons = max_horizons
        self.spatial_query = nn.Parameter(torch.zeros(n_tokens, embed_dim))
        nn.init.trunc_normal_(self.spatial_query, std=0.02)
        self.horizon_embed = nn.Embedding(max_horizons, embed_dim)
        self.time_embed = ContinuousTimeEmbedding(embed_dim)
        layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers, norm=nn.LayerNorm(embed_dim))

    def forward(self, memory: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        if delta_t.ndim != 2:
            raise ValueError("delta_t must have shape [B,K]")
        batch, horizons = delta_t.shape
        if horizons > self.max_horizons:
            raise ValueError("horizons exceeds max_horizons")
        if memory.shape[0] != batch:
            raise ValueError("memory and delta_t batch sizes differ")
        horizon_ids = torch.arange(horizons, device=memory.device)
        query = self.spatial_query[None, None, :, :]
        query = query + self.time_embed(delta_t)[:, :, None, :]
        query = query + self.horizon_embed(horizon_ids)[None, :, None, :]
        query = query.reshape(batch * horizons, self.n_tokens, memory.shape[-1])
        repeated_memory = memory[:, None].expand(-1, horizons, -1, -1)
        repeated_memory = repeated_memory.reshape(batch * horizons, memory.shape[1], memory.shape[2])
        output = self.decoder(query, repeated_memory)
        return output.reshape(batch, horizons, self.n_tokens, memory.shape[-1])
```

- [ ] **Step 5: Run component tests**

Run: `pytest tests/test_jepa_components.py -q`

Expected: `4 passed`.

- [ ] **Step 6: Commit the model components**

```bash
git add src/event_jepa/encoders.py src/event_jepa/predictor.py tests/test_jepa_components.py
git commit -m "feat: add Event-JEPA encoder and predictor"
```

### Task 5: EventJEPA composition, cosine objective, and EMA update

**Files:**
- Create: `src/event_jepa/model.py`
- Create: `tests/test_jepa_model.py`
- Modify: `src/event_jepa/__init__.py`

**Interfaces:**
- Consumes: `context [B,Tc,N,D]`, `target [B,K,N,D]`, `delta_t [B,K]`.
- Produces: `EventJEPA.forward(...) -> {loss,prediction,target,representation_std,mean_cosine}` and `update_target(momentum: float) -> None`.

- [ ] **Step 1: Write loss, gradient isolation, shape, and EMA tests**

```python
# tests/test_jepa_model.py
import torch

from event_jepa.model import EventJEPA, cosine_jepa_loss


def make_model():
    return EventJEPA(
        embed_dim=32, num_heads=4, encoder_layers=2, predictor_layers=1,
        n_tokens=4, max_positions=32, max_context_frames=3, max_horizons=2,
    )


def test_cosine_loss_is_zero_for_identical_nonzero_vectors():
    target = torch.randn(2, 1, 4, 32)
    assert cosine_jepa_loss(target, target).item() < 1e-6


def test_forward_returns_patchwise_prediction_and_no_target_gradients():
    model = make_model()
    result = model(
        torch.randn(2, 2, 4, 32),
        torch.randn(2, 1, 4, 32),
        torch.tensor([[0.01], [0.02]]),
    )
    assert result["prediction"].shape == (2, 1, 4, 32)
    result["loss"].backward()
    assert any(parameter.grad is not None for parameter in model.online_encoder.parameters())
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())


def test_ema_update_matches_weighted_average():
    model = make_model()
    with torch.no_grad():
        for parameter in model.online_encoder.parameters():
            parameter.fill_(2.0)
        for parameter in model.target_encoder.parameters():
            parameter.fill_(0.0)
    model.update_target(momentum=0.75)
    for parameter in model.target_encoder.parameters():
        assert torch.allclose(parameter, torch.full_like(parameter, 0.5))


def test_train_keeps_target_encoder_in_eval_mode():
    model = make_model().train()
    assert model.online_encoder.training
    assert not model.target_encoder.training
```

- [ ] **Step 2: Run tests and confirm the missing model failure**

Run: `pytest tests/test_jepa_model.py -q`

Expected: collection fails because `event_jepa.model` does not exist.

- [ ] **Step 3: Implement the joint-embedding model**

```python
# src/event_jepa/model.py
import copy

import torch
import torch.nn.functional as F
from torch import nn

from event_jepa.encoders import EventTokenEncoder
from event_jepa.predictor import EventPredictor


def cosine_jepa_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(prediction, target.detach(), dim=-1)).mean()


class EventJEPA(nn.Module):
    def __init__(self, embed_dim, num_heads, encoder_layers, predictor_layers,
                 n_tokens, max_positions, max_context_frames, max_horizons):
        super().__init__()
        self.online_encoder = EventTokenEncoder(
            embed_dim, num_heads, encoder_layers, n_tokens,
            max_positions, max_context_frames,
        )
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_encoder.requires_grad_(False)
        self.target_encoder.eval()
        self.predictor = EventPredictor(
            embed_dim, num_heads, predictor_layers, n_tokens, max_horizons,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_encoder.eval()
        return self

    def forward(self, context, target, delta_t):
        context_latent = self.online_encoder(context)
        prediction = self.predictor(context_latent, delta_t)
        batch, horizons, patches, dim = target.shape
        with torch.no_grad():
            encoded_target = self.target_encoder(target.reshape(batch * horizons, 1, patches, dim))
            encoded_target = encoded_target.reshape(batch, horizons, patches, dim)
        loss = cosine_jepa_loss(prediction, encoded_target)
        normalized_prediction = F.normalize(prediction.detach(), dim=-1)
        normalized_target = F.normalize(encoded_target.detach(), dim=-1)
        return {
            "loss": loss,
            "prediction": prediction,
            "target": encoded_target,
            "representation_std": encoded_target.float().std(dim=(0, 1, 2)).mean(),
            "mean_cosine": (normalized_prediction * normalized_target).sum(dim=-1).mean(),
        }

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("momentum must lie within [0, 1]")
        for online, target in zip(self.online_encoder.parameters(), self.target_encoder.parameters(), strict=True):
            target.mul_(momentum).add_(online, alpha=1.0 - momentum)
```

- [ ] **Step 4: Run the model tests**

Run: `pytest tests/test_jepa_model.py -q`

Expected: `4 passed`.

- [ ] **Step 5: Run all tests accumulated so far**

Run: `pytest -q`

Expected: `20 passed`.

- [ ] **Step 6: Commit the EventJEPA model**

```bash
git add src/event_jepa/model.py src/event_jepa/__init__.py tests/test_jepa_model.py
git commit -m "feat: compose Event-JEPA with EMA targets"
```

### Task 6: Resumable checkpoints and strict GEP downstream export

**Files:**
- Create: `src/event_jepa/checkpoint.py`
- Create: `src/export_jepa.py`
- Create: `tests/test_jepa_checkpoint.py`

**Interfaces:**
- Consumes: model, optimizer, optional scaler, configuration, step, and checkpoint path.
- Produces: `save_checkpoint`, `load_checkpoint`, `export_gep_transformer`, and a CLI export command.

- [ ] **Step 1: Write round-trip and strict export tests**

```python
# tests/test_jepa_checkpoint.py
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from event_jepa.checkpoint import export_gep_transformer, load_checkpoint, save_checkpoint
from event_jepa.config import EventJEPAConfig
from event_jepa.model import EventJEPA
from model import Block


def make_model():
    return EventJEPA(32, 4, 2, 1, 4, 32, 3, 2)


def test_checkpoint_restores_model_optimizer_and_step(tmp_path: Path):
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    config = EventJEPAConfig(
        data_root=tmp_path, embed_dim=32, num_heads=4, n_tokens=4,
        encoder_layers=2, predictor_layers=1, max_positions=32,
        context_frames=2, max_context_frames=3,
    )
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, None, config, step=7)
    restored = make_model()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    state = load_checkpoint(path, restored, restored_optimizer, None)
    assert state["step"] == 7
    for left, right in zip(model.parameters(), restored.parameters(), strict=True):
        assert torch.equal(left, right)


def test_gep_export_strict_loads_into_matching_transformer(tmp_path: Path):
    model = make_model()
    path = tmp_path / "gep_transformer.pt"
    export_gep_transformer(path, model, {"max_positions": 32, "embed_dim": 32})
    exported = torch.load(path, map_location="cpu", weights_only=True)
    cfg = SimpleNamespace(n_embed=32, n_head=4)
    receiver = nn.ModuleDict({
        "modality_embed": nn.Embedding(5, 32),
        "pos_embed": nn.Embedding(32, 32),
        "blocks": nn.ModuleList([Block(cfg) for _ in range(2)]),
        "norm": nn.LayerNorm(32),
    })
    receiver.load_state_dict(exported["transformer"], strict=True)
```

- [ ] **Step 2: Run tests and confirm checkpoint imports fail**

Run: `pytest tests/test_jepa_checkpoint.py -q`

Expected: collection fails because `event_jepa.checkpoint` does not exist.

- [ ] **Step 3: Implement checkpoint save, restore, RNG preservation, and export**

```python
# src/event_jepa/checkpoint.py
import random
from pathlib import Path

import numpy as np
import torch


def _rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def save_checkpoint(path, model, optimizer, scaler, config, step):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": 1,
        "step": step,
        "online_encoder": model.online_encoder.state_dict(),
        "target_encoder": model.target_encoder.state_dict(),
        "predictor": model.predictor.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "config": config.to_dict(),
        "rng_state": _rng_state(),
    }, path)


def load_checkpoint(path, model, optimizer, scaler):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format_version") != 1:
        raise ValueError(f"unsupported checkpoint format: {state.get('format_version')}")
    model.online_encoder.load_state_dict(state["online_encoder"], strict=True)
    model.target_encoder.load_state_dict(state["target_encoder"], strict=True)
    model.predictor.load_state_dict(state["predictor"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and state["scaler"] is not None:
        scaler.load_state_dict(state["scaler"])
    random.setstate(state["rng_state"]["python"])
    np.random.set_state(state["rng_state"]["numpy"])
    torch.set_rng_state(state["rng_state"]["torch"])
    if torch.cuda.is_available() and "cuda" in state["rng_state"]:
        torch.cuda.set_rng_state_all(state["rng_state"]["cuda"])
    return state


def export_gep_transformer(path, model, metadata):
    payload = {
        "format_version": 1,
        "source": "event_jepa",
        "transformer": model.online_encoder.gep_transformer_state_dict(),
        "metadata": dict(metadata),
    }
    torch.save(payload, Path(path))
```

- [ ] **Step 4: Implement the deterministic export CLI**

```python
# src/export_jepa.py
import argparse
from pathlib import Path

import torch

from event_jepa.checkpoint import export_gep_transformer
from event_jepa.config import EventJEPAConfig
from event_jepa.model import EventJEPA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = EventJEPAConfig(**state["config"])
    model = EventJEPA(
        cfg.embed_dim, cfg.num_heads, cfg.encoder_layers, cfg.predictor_layers,
        cfg.n_tokens, cfg.max_positions, cfg.max_context_frames, len(cfg.horizons),
    )
    model.online_encoder.load_state_dict(state["online_encoder"], strict=True)
    export_gep_transformer(Path(args.output), model, cfg.to_dict())


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run checkpoint tests and an export CLI smoke command**

Run: `pytest tests/test_jepa_checkpoint.py -q`

Expected: `2 passed`.

Run after Task 7 creates a synthetic checkpoint: `PYTHONPATH=src python src/export_jepa.py --checkpoint /tmp/event_jepa_smoke/latest.pt --output /tmp/event_jepa_smoke/gep_transformer.pt`

Expected: command exits zero and `/tmp/event_jepa_smoke/gep_transformer.pt` contains keys `format_version`, `source`, `transformer`, and `metadata`.

- [ ] **Step 6: Commit checkpoint support**

```bash
git add src/event_jepa/checkpoint.py src/export_jepa.py tests/test_jepa_checkpoint.py
git commit -m "feat: add Event-JEPA checkpoints and GEP export"
```

### Task 7: Optimizer-step engine with EMA skip protection

**Files:**
- Create: `src/event_jepa/engine.py`
- Create: `tests/test_jepa_engine.py`

**Interfaces:**
- Consumes: model, one batch, optimizer, optional GradScaler, precision, gradient clipping, EMA momentum, and accumulation boundary.
- Produces: `train_micro_step(...) -> dict[str,float|bool]` and `ema_momentum(step,total,start,end) -> float`.

- [ ] **Step 1: Write successful-step, accumulation, schedule, and skipped-step tests**

```python
# tests/test_jepa_engine.py
import torch

from event_jepa.engine import ema_momentum, train_micro_step
from event_jepa.model import EventJEPA


def make_batch():
    return {
        "context": torch.randn(2, 2, 4, 32),
        "target": torch.randn(2, 1, 4, 32),
        "delta_t": torch.tensor([[0.01], [0.02]]),
    }


def make_model():
    return EventJEPA(32, 4, 1, 1, 4, 32, 3, 2)


def test_successful_optimizer_step_updates_target():
    model = make_model()
    optimizer = torch.optim.AdamW(
        list(model.online_encoder.parameters()) + list(model.predictor.parameters()), lr=1e-3,
    )
    before = [parameter.clone() for parameter in model.target_encoder.parameters()]
    metrics = train_micro_step(
        model, make_batch(), optimizer, scaler=None, device=torch.device("cpu"),
        precision="fp32", grad_clip_norm=1.0, ema_value=0.9,
        loss_divisor=1, should_step=True,
    )
    assert metrics["optimizer_stepped"] is True
    assert any(not torch.equal(left, right) for left, right in zip(before, model.target_encoder.parameters(), strict=True))


def test_accumulation_micro_step_does_not_update_target():
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = [parameter.clone() for parameter in model.target_encoder.parameters()]
    metrics = train_micro_step(
        model, make_batch(), optimizer, None, torch.device("cpu"), "fp32",
        1.0, 0.9, loss_divisor=2, should_step=False,
    )
    assert metrics["optimizer_stepped"] is False
    assert all(torch.equal(left, right) for left, right in zip(before, model.target_encoder.parameters(), strict=True))


def test_ema_schedule_reaches_endpoints():
    assert ema_momentum(0, 100, 0.996, 0.9999) == 0.996
    assert ema_momentum(100, 100, 0.996, 0.9999) == 0.9999


class SkippingScaler:
    def __init__(self):
        self.scale_value = 1024.0
    def scale(self, loss):
        return loss
    def unscale_(self, optimizer):
        return None
    def step(self, optimizer):
        return None
    def update(self):
        self.scale_value /= 2
    def get_scale(self):
        return self.scale_value


def test_skipped_scaled_step_does_not_update_ema():
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = [parameter.clone() for parameter in model.target_encoder.parameters()]
    metrics = train_micro_step(
        model, make_batch(), optimizer, SkippingScaler(), torch.device("cpu"), "fp32",
        1.0, 0.9, loss_divisor=1, should_step=True,
    )
    assert metrics["optimizer_stepped"] is False
    assert all(torch.equal(left, right) for left, right in zip(before, model.target_encoder.parameters(), strict=True))
```

- [ ] **Step 2: Run the engine tests and confirm the missing module failure**

Run: `pytest tests/test_jepa_engine.py -q`

Expected: collection fails because `event_jepa.engine` does not exist.

- [ ] **Step 3: Implement autocast selection, gradient accumulation, skip detection, and EMA schedule**

```python
# src/event_jepa/engine.py
import math
from contextlib import nullcontext

import torch


def ema_momentum(step: int, total_steps: int, start: float, end: float) -> float:
    progress = min(max(step / max(total_steps, 1), 0.0), 1.0)
    return end - (end - start) * (math.cos(math.pi * progress) + 1.0) / 2.0


def _autocast(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def train_micro_step(model, batch, optimizer, scaler, device, precision,
                     grad_clip_norm, ema_value, loss_divisor, should_step):
    context = batch["context"].to(device, non_blocking=True)
    target = batch["target"].to(device, non_blocking=True)
    delta_t = batch["delta_t"].to(device, non_blocking=True)
    with _autocast(device, precision):
        output = model(context, target, delta_t)
        scaled_loss = output["loss"] / loss_divisor
    if scaler is None:
        scaled_loss.backward()
    else:
        scaler.scale(scaled_loss).backward()
    optimizer_stepped = False
    grad_norm = 0.0
    if should_step:
        if scaler is None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                grad_clip_norm,
            ).item()
            optimizer.step()
            optimizer_stepped = True
        else:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                grad_clip_norm,
            ).item()
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer_stepped = scaler.get_scale() >= previous_scale
        optimizer.zero_grad(set_to_none=True)
        if optimizer_stepped:
            core_model = model.module if hasattr(model, "module") else model
            core_model.update_target(ema_value)
    return {
        "loss": float(output["loss"].detach()),
        "representation_std": float(output["representation_std"]),
        "mean_cosine": float(output["mean_cosine"]),
        "grad_norm": grad_norm,
        "optimizer_stepped": optimizer_stepped,
    }
```

- [ ] **Step 4: Run the engine tests**

Run: `pytest tests/test_jepa_engine.py -q`

Expected: `4 passed`.

- [ ] **Step 5: Commit the training engine**

```bash
git add src/event_jepa/engine.py tests/test_jepa_engine.py
git commit -m "feat: add Event-JEPA optimizer step engine"
```

### Task 8: Training CLI, DDP lifecycle, scheduler, validation, and synthetic resume smoke test

**Files:**
- Create: `src/train_jepa.py`
- Create: `tests/test_train_jepa_smoke.py`
- Modify: `src/event_jepa/engine.py`

**Interfaces:**
- Consumes: `--config`, optional `--resume`, and optional `--data-root` override.
- Produces: rank-aware training, TensorBoard logs, `latest.pt`, periodic checkpoints, validation metrics, and deterministic resume.

- [ ] **Step 1: Add a three-step save/resume integration test using real temporary token files**

```python
# tests/test_train_jepa_smoke.py
from pathlib import Path

import torch

from train_jepa import run_training


def write_split(root: Path, split: str):
    folder = root / f"{split}_images/seq/images/left/eventToken"
    folder.mkdir(parents=True)
    for index, timestamp in enumerate([100, 110, 120, 130, 140, 150]):
        torch.save(torch.randn(4, 32), folder / f"{timestamp}.pt")


def test_training_saves_and_resumes(tmp_path: Path):
    write_split(tmp_path, "train")
    write_split(tmp_path, "test")
    config = tmp_path / "smoke.yaml"
    output = tmp_path / "run"
    config.write_text(
        f"data_root: {tmp_path}\noutput_dir: {output}\n"
        "context_frames: 2\nhorizons: [1]\nn_tokens: 4\nembed_dim: 32\n"
        "num_heads: 4\nencoder_layers: 1\npredictor_layers: 1\n"
        "max_positions: 32\nmax_context_frames: 3\nbatch_size: 2\n"
        "num_workers: 0\ntotal_steps: 2\nwarmup_steps: 1\n"
        "precision: fp32\nlog_every: 1\nvalidate_every: 1\nsave_every: 1\n",
        encoding="utf-8",
    )
    first = run_training(config, max_steps=2)
    assert first == 2
    checkpoint = output / "latest.pt"
    assert checkpoint.exists()
    second = run_training(config, resume=checkpoint, max_steps=3)
    assert second == 3
```

- [ ] **Step 2: Run the smoke test and verify the missing entry-point failure**

Run: `pytest tests/test_train_jepa_smoke.py -q`

Expected: collection fails because `train_jepa` does not exist.

- [ ] **Step 3: Add learning-rate and validation helpers to the engine**

```python
# append to src/event_jepa/engine.py
def cosine_learning_rate(step, warmup_steps, total_steps, peak, minimum):
    if step < warmup_steps:
        return peak * step / max(warmup_steps, 1)
    progress = min((step - warmup_steps) / max(total_steps - warmup_steps, 1), 1.0)
    return minimum + 0.5 * (peak - minimum) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def validate(model, loader, device, precision, max_batches=None):
    model.eval()
    totals = {"loss": 0.0, "representation_std": 0.0, "mean_cosine": 0.0}
    count = 0
    for batch in loader:
        with _autocast(device, precision):
            output = model(
                batch["context"].to(device),
                batch["target"].to(device),
                batch["delta_t"].to(device),
            )
        for key in totals:
            totals[key] += float(output[key])
        count += 1
        if max_batches is not None and count >= max_batches:
            break
    model.train()
    return {key: value / max(count, 1) for key, value in totals.items()}
```

- [ ] **Step 4: Implement `run_training` and CLI parsing**

```python
# src/train_jepa.py
import argparse
import os
import random
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from event_jepa.checkpoint import load_checkpoint, save_checkpoint
from event_jepa.config import EventJEPAConfig, load_config
from event_jepa.dataset import EventJEPADataset
from event_jepa.engine import (
    cosine_learning_rate,
    ema_momentum,
    train_micro_step,
    validate,
)
from event_jepa.model import EventJEPA


def distributed_context() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("distributed Event-JEPA training requires CUDA")
        torch.distributed.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return rank, world_size, local_rank, device


def build_model(config: EventJEPAConfig) -> EventJEPA:
    return EventJEPA(
        config.embed_dim,
        config.num_heads,
        config.encoder_layers,
        config.predictor_layers,
        config.n_tokens,
        config.max_positions,
        config.max_context_frames,
        max_horizons=len(config.horizons),
    )


def _dataset(config: EventJEPAConfig, split: str) -> EventJEPADataset:
    return EventJEPADataset(
        config.data_root,
        split,
        config.context_frames,
        config.horizons,
        config.n_tokens,
        config.embed_dim,
        config.timestamp_scale,
    )


def build_loaders(config: EventJEPAConfig, rank: int, world_size: int):
    train_dataset = _dataset(config, "train")
    valid_dataset = _dataset(config, "test")
    if len(train_dataset) == 0 or len(valid_dataset) == 0:
        raise ValueError("no valid temporal samples found")
    train_sampler = None
    valid_sampler = None
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True,
            seed=config.seed, drop_last=False,
        )
    generator = torch.Generator().manual_seed(config.seed + rank)
    common = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
        "generator": generator,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        **common,
    )
    valid_loader = DataLoader(
        valid_dataset,
        shuffle=False,
        sampler=valid_sampler,
        **common,
    )
    return train_loader, valid_loader, train_sampler


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _core_model(model):
    return model.module if hasattr(model, "module") else model


def run_training(config_path, resume=None, data_root=None, max_steps=None) -> int:
    config = load_config(config_path)
    if data_root is not None:
        config = replace(config, data_root=Path(data_root))
    rank, world_size, local_rank, device = distributed_context()
    _seed_everything(config.seed + rank)
    writer = None
    try:
        train_loader, valid_loader, train_sampler = build_loaders(config, rank, world_size)
        model = build_model(config).to(device)
        if world_size > 1:
            model = DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
        core = _core_model(model)
        trainable = list(core.online_encoder.parameters()) + list(core.predictor.parameters())
        optimizer = torch.optim.AdamW(
            trainable,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scaler = None
        if device.type == "cuda" and config.precision == "fp16":
            scaler = torch.amp.GradScaler("cuda")
        step = 0
        if resume is not None:
            state = load_checkpoint(resume, core, optimizer, scaler)
            step = int(state["step"])
        target_steps = config.total_steps if max_steps is None else int(max_steps)
        if target_steps < step:
            raise ValueError("max_steps cannot be smaller than resumed step")
        if rank == 0:
            config.output_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=str(config.output_dir / "tensorboard"))
        optimizer.zero_grad(set_to_none=True)
        micro_step = 0
        epoch = 0
        last_validation_step = -1
        last_save_step = -1
        while step < target_steps:
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            for batch in train_loader:
                should_step = (micro_step + 1) % config.grad_accum_steps == 0
                lr = cosine_learning_rate(
                    step,
                    config.warmup_steps,
                    config.total_steps,
                    config.learning_rate,
                    config.min_learning_rate,
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr
                sync_context = nullcontext()
                if world_size > 1 and not should_step:
                    sync_context = model.no_sync()
                with sync_context:
                    metrics = train_micro_step(
                        model,
                        batch,
                        optimizer,
                        scaler,
                        device,
                        config.precision,
                        config.grad_clip_norm,
                        ema_momentum(
                            step,
                            config.total_steps,
                            config.ema_start,
                            config.ema_end,
                        ),
                        config.grad_accum_steps,
                        should_step,
                    )
                micro_step += 1
                if metrics["optimizer_stepped"]:
                    step += 1
                if rank == 0 and metrics["optimizer_stepped"] and step % config.log_every == 0:
                    for key, value in metrics.items():
                        if key != "optimizer_stepped":
                            writer.add_scalar(f"train/{key}", value, step)
                    writer.add_scalar("train/learning_rate", lr, step)
                    writer.add_scalar("train/mean_delta_t", float(batch["delta_t"].mean()), step)
                if rank == 0 and step > 0 and step % config.validate_every == 0 and step != last_validation_step:
                    validation = validate(core, valid_loader, device, config.precision)
                    for key, value in validation.items():
                        writer.add_scalar(f"valid/{key}", value, step)
                    last_validation_step = step
                if rank == 0 and step > 0 and step % config.save_every == 0 and step != last_save_step:
                    save_checkpoint(
                        config.output_dir / f"step_{step:08d}.pt",
                        core, optimizer, scaler, config, step,
                    )
                    save_checkpoint(
                        config.output_dir / "latest.pt",
                        core, optimizer, scaler, config, step,
                    )
                    last_save_step = step
                if step >= target_steps:
                    break
            epoch += 1
        if rank == 0 and step != last_save_step:
            save_checkpoint(config.output_dir / "latest.pt", core, optimizer, scaler, config, step)
        return step
    finally:
        if writer is not None:
            writer.close()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--data-root")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    run_training(args.config, args.resume, args.data_root, args.max_steps)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the synthetic resume integration test**

Run: `pytest tests/test_train_jepa_smoke.py -q`

Expected: `1 passed`; the test performs two optimizer steps, resumes from `latest.pt`, and reaches step 3.

- [ ] **Step 6: Run the complete CPU suite**

Run: `pytest -q`

Expected: all tests pass with no warnings or errors.

- [ ] **Step 7: Commit the complete training entry point**

```bash
git add src/train_jepa.py src/event_jepa/engine.py tests/test_train_jepa_smoke.py
git commit -m "feat: add resumable Event-JEPA training CLI"
```

### Task 9: Downstream hook, exact runbook, and hardware smoke commands

**Files:**
- Modify: `src/config.py`
- Modify: `readme.md`
- Create: `docs/event_jepa_runbook.md`
- Create: `tests/test_gep_downstream_export.py`

**Interfaces:**
- Consumes: GEP-compatible exported checkpoint path through `EVENT_JEPA_TRANSFORMER_CKPT`.
- Produces: existing `CLSConfig.transformer_weight`, exact local/DDP commands, and a documented GEP-vs-JEPA evaluation matrix.

- [ ] **Step 1: Add a pure export-loader test**

```python
# tests/test_gep_downstream_export.py
from pathlib import Path

import pytest
import torch

from event_jepa.checkpoint import load_gep_transformer_export


def test_load_gep_export_returns_transformer_and_metadata(tmp_path: Path):
    path = tmp_path / "export.pt"
    torch.save({
        "format_version": 1,
        "source": "event_jepa",
        "transformer": {"norm.weight": torch.ones(4)},
        "metadata": {"max_positions": 32},
    }, path)
    weights, metadata = load_gep_transformer_export(path)
    assert torch.equal(weights["norm.weight"], torch.ones(4))
    assert metadata["max_positions"] == 32


def test_load_gep_export_rejects_wrong_source(tmp_path: Path):
    path = tmp_path / "bad.pt"
    torch.save({"format_version": 1, "source": "other"}, path)
    with pytest.raises(ValueError, match="Event-JEPA export"):
        load_gep_transformer_export(path)
```

- [ ] **Step 2: Run the tests and verify the missing loader failure**

Run: `pytest tests/test_gep_downstream_export.py -q`

Expected: import fails because `load_gep_transformer_export` does not exist.

- [ ] **Step 3: Add the validated export loader and optional `CLSConfig` hook**

Add to `src/event_jepa/checkpoint.py`:

```python
def load_gep_transformer_export(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1 or payload.get("source") != "event_jepa":
        raise ValueError(f"not a supported Event-JEPA export: {path}")
    if not isinstance(payload.get("transformer"), dict):
        raise ValueError(f"Event-JEPA export has no transformer state: {path}")
    return payload["transformer"], payload.get("metadata", {})
```

In `CLSConfig.__init__`, immediately after the existing `self.n_layer = 12`, add:

```python
event_jepa_export = os.environ.get("EVENT_JEPA_TRANSFORMER_CKPT")
if event_jepa_export:
    from event_jepa.checkpoint import load_gep_transformer_export
    self.transformer_weight, export_metadata = load_gep_transformer_export(event_jepa_export)
    self.window_size = int(export_metadata.get("max_positions", self.window_size))
    exported_dim = int(export_metadata.get("embed_dim", self.n_embed))
    if exported_dim != self.n_embed:
        raise ValueError(
            f"Event-JEPA embed_dim {exported_dim} does not match classifier embed_dim {self.n_embed}"
        )
    self.n_layer = int(export_metadata.get("encoder_layers", self.n_layer))
```

The existing `CLSConfig` contains only commented examples of `self.transformer_weight`, so no second assignment is added. With the environment variable absent, `CLSConfig` retains its original behavior.

- [ ] **Step 4: Write the exact runbook**

`docs/event_jepa_runbook.md` must contain these executable sections and commands:

```bash
# Environment
conda env create -f environment.yml
conda activate gep
pytest -q

# RTX 5070 Ti 16 GB: edit only data_root/output_dir in the smoke YAML
PYTHONPATH=src python src/train_jepa.py \
  --config configs/event_jepa_smoke.yaml

# Resume local smoke run
PYTHONPATH=src python src/train_jepa.py \
  --config configs/event_jepa_smoke.yaml \
  --resume runs/event_jepa_smoke/latest.pt

# 4 x A100 80 GB
PYTHONPATH=src torchrun --standalone --nproc_per_node=4 src/train_jepa.py \
  --config configs/event_jepa_a100.yaml

# Export the online context Transformer
PYTHONPATH=src python src/export_jepa.py \
  --checkpoint runs/event_jepa_a100/latest.pt \
  --output runs/event_jepa_a100/gep_transformer.pt

# N-ImageNet linear probe through the existing GEP classifier
EVENT_JEPA_TRANSFORMER_CKPT=runs/event_jepa_a100/gep_transformer.pt \
PYTHONPATH=src python src/cls.py
```

The runbook must also state the exact classifier settings before the last command: `CLSConfig.dataset_name="nima"`, `modality="event"`, `transfer="linear"`, identical Stage-1 encoder checkpoint across Scratch/GEP/Event-JEPA, and three seeds. Record GPU count, effective batch size, token count, optimizer steps, and wall-clock time for fairness.

- [ ] **Step 5: Add a concise Event-JEPA section to the repository README**

Link the design, implementation plan, and runbook. State that V1 is token-level temporal pretraining over frozen Stage-1 features, list the 5070 Ti smoke and 4 x A100 entry commands, and avoid first-method or arbitrary-continuous-time claims.

- [ ] **Step 6: Run focused and full verification**

Run: `pytest tests/test_gep_downstream_export.py -q`

Expected: `2 passed`.

Run: `pytest -q`

Expected: all tests pass with zero warnings.

Run: `git diff --check`

Expected: no output.

- [ ] **Step 7: Run the 5070 Ti real-data smoke test**

Run after editing `data_root` and `output_dir` in the smoke YAML:

```bash
nvidia-smi
PYTHONPATH=src python src/train_jepa.py --config configs/event_jepa_smoke.yaml
```

Expected: 20 successful optimizer steps, finite loss, non-zero gradient norm, positive representation standard deviation, changing EMA weights, peak GPU memory below 16 GB, and `latest.pt` plus a step-20 checkpoint.

- [ ] **Step 8: Commit documentation and downstream integration**

```bash
git add src/config.py src/event_jepa/checkpoint.py readme.md docs/event_jepa_runbook.md tests/test_gep_downstream_export.py
git commit -m "docs: add Event-JEPA training and evaluation runbook"
```

## Final Verification Gate

- [ ] Run `pytest -q` and record the exact pass count.
- [ ] Run `git diff --check` and confirm empty output.
- [ ] Run `git status --short` and explain every remaining file.
- [ ] Inspect `git log --oneline -10` and confirm one focused commit per task.
- [ ] Run the synthetic save/resume smoke test independently with `pytest tests/test_train_jepa_smoke.py -q`.
- [ ] On the RTX 5070 Ti machine, record peak VRAM, step time, final loss, representation standard deviation, and checkpoint path.
- [ ] Export the online Transformer and strict-load it through the downstream hook.
- [ ] Before claiming completion, invoke `superpowers:verification-before-completion` and report any failed command by name.
