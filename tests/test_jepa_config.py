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
