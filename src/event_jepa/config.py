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
    residual_weight: float = 0.0
    order_weight: float = 0.0
    order_margin: float = 0.005
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
        if self.residual_weight < 0:
            raise ValueError("residual_weight must be nonnegative")
        if self.order_weight < 0:
            raise ValueError("order_weight must be nonnegative")
        if self.order_margin < 0:
            raise ValueError("order_margin must be nonnegative")
        if self.order_weight > 0 and self.context_frames < 3:
            raise ValueError(
                "order objective requires context_frames >= 3"
            )
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
