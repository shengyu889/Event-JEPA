from pathlib import Path

import pytest
import torch

from event_jepa.checkpoint import load_gep_transformer_export


def test_load_gep_export_returns_transformer_and_metadata(tmp_path: Path):
    path = tmp_path / "export.pt"
    torch.save(
        {
            "format_version": 1,
            "source": "event_jepa",
            "transformer": {"norm.weight": torch.ones(4)},
            "metadata": {"max_positions": 32},
        },
        path,
    )

    weights, metadata = load_gep_transformer_export(path)

    assert torch.equal(weights["norm.weight"], torch.ones(4))
    assert metadata["max_positions"] == 32


def test_load_gep_export_rejects_wrong_source(tmp_path: Path):
    path = tmp_path / "bad.pt"
    torch.save({"format_version": 1, "source": "other"}, path)

    with pytest.raises(ValueError, match="Event-JEPA export"):
        load_gep_transformer_export(path)
