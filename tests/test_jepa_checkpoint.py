from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from event_jepa.checkpoint import (
    export_gep_transformer,
    load_checkpoint,
    save_checkpoint,
)
from event_jepa.config import EventJEPAConfig
from event_jepa.model import EventJEPA
from model import Block


def make_model():
    return EventJEPA(32, 4, 2, 1, 4, 32, 3, 2)


def test_checkpoint_restores_model_optimizer_and_step(tmp_path: Path):
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    config = EventJEPAConfig(
        data_root=tmp_path,
        embed_dim=32,
        num_heads=4,
        n_tokens=4,
        encoder_layers=2,
        predictor_layers=1,
        max_positions=32,
        context_frames=2,
        max_context_frames=3,
    )
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, None, config, step=7)
    restored = make_model()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)

    state = load_checkpoint(path, restored, restored_optimizer, None)

    assert state["step"] == 7
    for left, right in zip(
        model.parameters(), restored.parameters(), strict=True
    ):
        assert torch.equal(left, right)


def test_gep_export_strict_loads_into_matching_transformer(tmp_path: Path):
    model = make_model()
    path = tmp_path / "gep_transformer.pt"
    export_gep_transformer(
        path,
        model,
        {"max_positions": 32, "embed_dim": 32},
    )
    exported = torch.load(path, map_location="cpu", weights_only=True)
    cfg = SimpleNamespace(n_embed=32, n_head=4)
    receiver = nn.ModuleDict(
        {
            "modality_embed": nn.Embedding(5, 32),
            "pos_embed": nn.Embedding(32, 32),
            "blocks": nn.ModuleList([Block(cfg) for _ in range(2)]),
            "norm": nn.LayerNorm(32),
        }
    )

    receiver.load_state_dict(exported["transformer"], strict=True)
