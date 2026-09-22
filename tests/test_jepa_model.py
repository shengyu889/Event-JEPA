import torch

from event_jepa.model import EventJEPA, cosine_jepa_loss


def make_model():
    return EventJEPA(
        embed_dim=32,
        num_heads=4,
        encoder_layers=2,
        predictor_layers=1,
        n_tokens=4,
        max_positions=32,
        max_context_frames=3,
        max_horizons=2,
    )


def test_cosine_loss_is_zero_for_identical_nonzero_vectors():
    target = torch.randn(2, 1, 4, 32)
    assert cosine_jepa_loss(target, target).item() < 1e-6


def test_forward_returns_patchwise_prediction_and_no_target_gradients():
    model = make_model()
    result = model(
        torch.randn(2, 2, 4, 32),
        torch.randn(2, 1, 4, 32),
        torch.tensor([[0.01], [0.02]]),
    )
    assert result["prediction"].shape == (2, 1, 4, 32)
    result["loss"].backward()
    assert any(
        parameter.grad is not None
        for parameter in model.online_encoder.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in model.target_encoder.parameters()
    )


def test_ema_update_matches_weighted_average():
    model = make_model()
    with torch.no_grad():
        for parameter in model.online_encoder.parameters():
            parameter.fill_(2.0)
        for parameter in model.target_encoder.parameters():
            parameter.fill_(0.0)
    model.update_target(momentum=0.75)
    for parameter in model.target_encoder.parameters():
        assert torch.allclose(parameter, torch.full_like(parameter, 0.5))


def test_train_keeps_target_encoder_in_eval_mode():
    model = make_model().train()
    assert model.online_encoder.training
    assert not model.target_encoder.training
