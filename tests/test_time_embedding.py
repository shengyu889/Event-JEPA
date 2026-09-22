import pytest
import torch

from event_jepa.time_embedding import ContinuousTimeEmbedding


def test_time_embedding_is_deterministic_and_shape_correct():
    module = ContinuousTimeEmbedding(embed_dim=32, fourier_dim=16)
    delta_t = torch.tensor([[0.01, 0.05], [0.02, 0.10]])

    first = module(delta_t)
    second = module(delta_t)

    assert first.shape == (2, 2, 32)
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()


def test_time_embedding_accepts_zero():
    output = ContinuousTimeEmbedding(16, 8)(torch.zeros(3))

    assert output.shape == (3, 16)
    assert torch.isfinite(output).all()


def test_time_embedding_rejects_negative_elapsed_time():
    with pytest.raises(ValueError, match="nonnegative"):
        ContinuousTimeEmbedding(16, 8)(torch.tensor([-0.01]))
