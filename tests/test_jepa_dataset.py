from pathlib import Path

import pytest
import torch

from event_jepa.dataset import EventJEPADataset


def write_sequence(
    root: Path,
    name: str,
    timestamps: list[int],
    value: float,
    shape: tuple[int, int] = (4, 8),
):
    folder = root / "train_images" / name / "images" / "left" / "eventToken"
    folder.mkdir(parents=True)
    for index, timestamp in enumerate(timestamps):
        torch.save(torch.full(shape, value + index), folder / f"{timestamp}.pt")


def test_dataset_keeps_samples_inside_sequence(tmp_path: Path):
    write_sequence(tmp_path, "seq_a", [100, 110, 120, 130], 10.0)
    write_sequence(tmp_path, "seq_b", [131, 141, 151, 161], 20.0)
    dataset = EventJEPADataset(
        tmp_path,
        "train",
        context_frames=2,
        horizons=(1,),
        n_tokens=4,
        embed_dim=8,
        timestamp_scale=0.1,
    )
    assert len(dataset) == 4
    for sample in dataset:
        context_family = sample["context"].div(10, rounding_mode="floor").unique().tolist()
        target_family = sample["target"].div(10, rounding_mode="floor").unique().tolist()
        assert context_family == target_family


def test_dataset_returns_measured_delta_t(tmp_path: Path):
    write_sequence(tmp_path, "seq", [100, 110, 145, 200], 0.0)
    dataset = EventJEPADataset(
        tmp_path,
        "train",
        context_frames=2,
        horizons=(1, 2),
        n_tokens=4,
        embed_dim=8,
        timestamp_scale=0.001,
    )
    sample = dataset[0]
    assert sample["context"].shape == (2, 4, 8)
    assert sample["target"].shape == (2, 4, 8)
    assert torch.allclose(sample["delta_t"], torch.tensor([0.035, 0.090]))
    assert sample["sequence"] == "seq"


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
