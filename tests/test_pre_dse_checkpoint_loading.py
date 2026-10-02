import pytest
import torch

from pre_dse_cli import load_encoder_state_dict


def test_load_encoder_state_dict_accepts_complete_wrapped_weights():
    source = torch.nn.Linear(2, 3)
    target = torch.nn.Linear(2, 3)

    loaded = load_encoder_state_dict(
        target,
        {"event_encoder": source.state_dict(), "epoch": 4},
        "event_encoder",
    )

    assert loaded is target
    assert torch.equal(target.weight, source.weight)
    assert torch.equal(target.bias, source.bias)


def test_load_encoder_state_dict_rejects_missing_parameter():
    model = torch.nn.Linear(2, 3)
    incomplete = {"event_encoder": {"weight": model.weight.detach().clone()}}

    with pytest.raises(RuntimeError, match="Missing key"):
        load_encoder_state_dict(model, incomplete, "event_encoder")
