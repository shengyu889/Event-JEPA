from argparse import Namespace
from pathlib import Path

import pytest

from pre_dse_cli import (
    build_parser,
    extract_encoder_state_dict,
    run_from_args,
    select_sequences,
    validate_args,
)


def test_parser_accepts_documented_single_sequence_event_only_command():
    args = build_parser().parse_args(
        [
            "--root",
            "/home/tom/event-jepa/datasets/DSEC",
            "--split",
            "train",
            "--sequence",
            "zurich_city_00_a",
            "--event-encoder-ckpt",
            "/home/tom/event-jepa/checkpoints/gep_stage1_small.pt",
            "--prepare-pairs",
            "--event-only",
            "--device",
            "cuda:0",
            "--workers",
            "1",
        ]
    )

    assert args.root == Path("/home/tom/event-jepa/datasets/DSEC")
    assert args.split == "train"
    assert args.sequence == ["zurich_city_00_a"]
    assert args.event_encoder_ckpt.name == "gep_stage1_small.pt"
    assert args.prepare_pairs is True
    assert args.event_only is True


def test_validate_args_requires_image_checkpoint_only_for_paired_mode(tmp_path: Path):
    event_checkpoint = tmp_path / "event.pt"
    event_checkpoint.touch()
    args = Namespace(
        root=tmp_path,
        split="train",
        event_encoder_ckpt=event_checkpoint,
        image_encoder_ckpt=None,
        event_only=False,
        workers=1,
    )

    with pytest.raises(ValueError, match="--image-encoder-ckpt"):
        validate_args(args)

    args.event_only = True
    validate_args(args)


def test_extract_encoder_state_dict_accepts_wrapped_and_raw_mappings():
    wrapped_weights = {"blocks.0.weight": object()}
    assert extract_encoder_state_dict(
        {"event_encoder": wrapped_weights, "optimizer": {}, "epoch": 2},
        "event_encoder",
    ) is wrapped_weights

    raw_weights = {"blocks.0.weight": object(), "norm.weight": object()}
    assert extract_encoder_state_dict(raw_weights, "event_encoder") is raw_weights


def test_extract_encoder_state_dict_rejects_training_checkpoint_without_key():
    with pytest.raises(ValueError, match="event_encoder"):
        extract_encoder_state_dict({"optimizer": {}, "epoch": 2}, "event_encoder")


def test_select_sequences_rejects_missing_requested_sequence(tmp_path: Path):
    (tmp_path / "zurich_city_00_a").mkdir()

    with pytest.raises(FileNotFoundError, match="missing_sequence"):
        select_sequences(tmp_path, ["missing_sequence"])


def test_run_from_args_routes_one_sequence_to_pair_and_token_steps(tmp_path: Path):
    sequence = "zurich_city_00_a"
    (tmp_path / "train_images" / sequence).mkdir(parents=True)
    event_checkpoint = tmp_path / "event.pt"
    event_checkpoint.touch()
    args = build_parser().parse_args(
        [
            "--root",
            str(tmp_path),
            "--split",
            "train",
            "--sequence",
            sequence,
            "--event-encoder-ckpt",
            str(event_checkpoint),
            "--prepare-pairs",
            "--event-only",
        ]
    )

    calls = []

    class RecordingProcessor:
        def __init__(self, parsed_args):
            calls.append(("init", parsed_args.split))

        def run(self, subfolders, n_workers):
            calls.append(("run", subfolders, n_workers))

        def process_tokens(self, **kwargs):
            calls.append(("process_tokens", kwargs))

        def compute_rgb_stats(self, **kwargs):
            calls.append(("compute_rgb_stats", kwargs))

    run_from_args(args, RecordingProcessor)

    assert calls[0] == ("init", "train")
    assert calls[1] == ("run", [sequence], 1)
    assert calls[2] == (
        "process_tokens",
        {
            "subfolders": [sequence],
            "workers": 1,
            "event_encoder_ckpt": event_checkpoint,
            "image_encoder_ckpt": None,
            "event_only": True,
        },
    )
    assert len(calls) == 3
