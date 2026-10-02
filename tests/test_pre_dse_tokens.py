from pathlib import Path

import pytest

from pre_dse_cli import prepare_token_jobs


def test_event_only_jobs_write_event_tokens_without_touching_image_tokens(
    tmp_path: Path,
):
    save_root = tmp_path / "images" / "left"
    event_images = save_root / "eventImage"
    event_images.mkdir(parents=True)
    (event_images / "100.png").write_bytes(b"event-png")
    (event_images / "200.png").write_bytes(b"event-png")

    old_event_tokens = save_root / "eventToken"
    old_event_tokens.mkdir()
    (old_event_tokens / "old.pt").write_bytes(b"stale")
    image_tokens = save_root / "imageToken"
    image_tokens.mkdir()
    sentinel = image_tokens / "keep.pt"
    sentinel.write_bytes(b"must-survive")

    jobs = prepare_token_jobs(save_root, event_only=True)
    for job in jobs:
        assert job.image_input is None
        assert job.image_output is None
        job.event_output.write_bytes(b"event-token")

    assert [job.name for job in jobs] == ["100.png", "200.png"]
    assert sorted(path.name for path in old_event_tokens.iterdir()) == [
        "100.pt",
        "200.pt",
    ]
    assert sentinel.read_bytes() == b"must-survive"


def test_paired_jobs_require_matching_warped_image(tmp_path: Path):
    save_root = tmp_path / "images" / "left"
    event_images = save_root / "eventImage"
    event_images.mkdir(parents=True)
    (event_images / "100.png").write_bytes(b"event-png")
    (save_root / "warpped").mkdir()

    with pytest.raises(FileNotFoundError, match="100.png"):
        prepare_token_jobs(save_root, event_only=False)
