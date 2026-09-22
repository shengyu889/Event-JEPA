import pytest
import torch

from event_jepa.encoders import EventTokenEncoder
from event_jepa.predictor import EventPredictor


def make_encoder():
    return EventTokenEncoder(
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        n_tokens=4,
        max_positions=32,
        max_context_frames=3,
    )


def test_encoder_flattens_context_without_losing_embedding_dimension():
    output = make_encoder()(torch.randn(2, 3, 4, 32))
    assert output.shape == (2, 12, 32)


def test_encoder_rejects_context_beyond_configured_maximum():
    with pytest.raises(ValueError, match="max_context_frames"):
        make_encoder()(torch.randn(2, 4, 4, 32))


def test_predictor_supports_multiple_horizons():
    predictor = EventPredictor(32, 4, 2, n_tokens=4, max_horizons=3)
    prediction = predictor(
        torch.randn(2, 12, 32),
        torch.tensor([[0.01, 0.05], [0.02, 0.08]]),
    )
    assert prediction.shape == (2, 2, 4, 32)


def test_predictor_rejects_too_many_horizons():
    predictor = EventPredictor(32, 4, 1, n_tokens=4, max_horizons=2)
    with pytest.raises(ValueError, match="max_horizons"):
        predictor(
            torch.randn(1, 8, 32),
            torch.tensor([[0.01, 0.02, 0.03]]),
        )
