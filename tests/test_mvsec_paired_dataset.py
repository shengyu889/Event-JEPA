from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from dataset import MVSECPairedDataset


def _make_sample(
    root: Path,
    stem: str = "000000",
    channels: int = 20,
):
    seq = root / "outdoor_day2"

    rgb_dir = seq / "rgb"
    event_dir = seq / "eventTensor_ecddp"

    rgb_dir.mkdir(parents=True, exist_ok=True)
    event_dir.mkdir(parents=True, exist_ok=True)

    # MVSEC intensity image: grayscale in our preprocessing pipeline.
    image = np.full(
        (260, 346),
        127,
        dtype=np.uint8,
    )
    Image.fromarray(image).save(rgb_dir / f"{stem}.png")

    event = torch.randn(
        channels,
        224,
        224,
        dtype=torch.float32,
    )

    # Match src/pre_mvs_ecddp.py storage format.
    torch.save(
        {"tensor": event},
        event_dir / f"{stem}.pt",
    )

    return event


def test_mvsec_paired_dataset_loads_20ch_tensor(tmp_path):
    expected = _make_sample(tmp_path)

    dataset = MVSECPairedDataset(
        root_dir=str(tmp_path),
        sequences=("outdoor_day2",),
        event_channels=20,
    )

    assert len(dataset) == 1

    event, image = dataset[0]

    assert isinstance(event, torch.Tensor)
    assert event.shape == (20, 224, 224)
    assert event.dtype == torch.float32

    assert image.mode == "RGB"
    assert image.size == (346, 260)

    torch.testing.assert_close(event, expected)


def test_mvsec_paired_dataset_matches_by_stem(tmp_path):
    _make_sample(tmp_path, stem="000002")

    seq = tmp_path / "outdoor_day2"

    # Add unmatched files: they must not create bogus pairs.
    Image.fromarray(
        np.zeros((260, 346), dtype=np.uint8)
    ).save(seq / "rgb" / "999999.png")

    torch.save(
        {"tensor": torch.zeros(20, 224, 224)},
        seq / "eventTensor_ecddp" / "888888.pt",
    )

    dataset = MVSECPairedDataset(
        root_dir=str(tmp_path),
        sequences=("outdoor_day2",),
    )

    assert len(dataset) == 1
    assert Path(dataset.image_names[0]).stem == "000002"
    assert Path(dataset.event_names[0]).stem == "000002"


def test_mvsec_paired_dataset_rejects_wrong_channel_count(tmp_path):
    _make_sample(
        tmp_path,
        channels=3,
    )

    dataset = MVSECPairedDataset(
        root_dir=str(tmp_path),
        sequences=("outdoor_day2",),
        event_channels=20,
    )

    with pytest.raises(ValueError, match="Expected 20 event channels"):
        _ = dataset[0]


def test_mvsec_paired_dataset_only_uses_requested_sequence(tmp_path):
    _make_sample(tmp_path, stem="000001")

    other = tmp_path / "outdoor_day1"
    (other / "rgb").mkdir(parents=True)
    (other / "eventTensor_ecddp").mkdir(parents=True)

    Image.fromarray(
        np.zeros((260, 346), dtype=np.uint8)
    ).save(other / "rgb" / "000000.png")

    torch.save(
        {"tensor": torch.zeros(20, 224, 224)},
        other / "eventTensor_ecddp" / "000000.pt",
    )

    dataset = MVSECPairedDataset(
        root_dir=str(tmp_path),
        sequences=("outdoor_day2",),
    )

    assert len(dataset) == 1
    assert dataset.sample_sequences == ["outdoor_day2"]
