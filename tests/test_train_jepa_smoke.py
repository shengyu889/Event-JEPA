from pathlib import Path

import torch

from train_jepa import run_training


def write_split(root: Path, split: str):
    folder = root / f"{split}_images/seq/images/left/eventToken"
    folder.mkdir(parents=True)
    for timestamp in [100, 110, 120, 130, 140, 150]:
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
