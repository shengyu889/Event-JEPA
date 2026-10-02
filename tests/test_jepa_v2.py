import torch

from event_jepa.model import EventJEPA
from event_jepa.model_v2 import (
    OrderAwareResidualEventJEPA,
    history_order_ranking_loss,
    permute_history_keep_current,
    residual_jepa_loss,
)


def make_base():
    return EventJEPA(
        embed_dim=32,
        num_heads=4,
        encoder_layers=2,
        predictor_layers=1,
        n_tokens=4,
        max_positions=32,
        max_context_frames=4,
        max_horizons=3,
    )


def make_v2(
    residual_weight=0.5,
    order_weight=0.5,
):
    return OrderAwareResidualEventJEPA(
        embed_dim=32,
        num_heads=4,
        encoder_layers=2,
        predictor_layers=1,
        n_tokens=4,
        max_positions=32,
        max_context_frames=4,
        max_horizons=3,
        residual_weight=residual_weight,
        order_weight=order_weight,
        order_margin=0.005,
    )


def test_permutation_changes_only_history_order():
    x = torch.arange(
        1 * 4 * 2 * 3
    ).reshape(
        1, 4, 2, 3
    )

    y = permute_history_keep_current(x)

    # current t stays exactly identical
    assert torch.equal(
        y[:, -1],
        x[:, -1],
    )

    # chronological history changes
    assert not torch.equal(
        y[:, :-1],
        x[:, :-1],
    )

    # expected Tc=4 permutation:
    # [0,1,2,3] -> [1,2,0,3]
    assert torch.equal(
        y[:, 0],
        x[:, 1],
    )

    assert torch.equal(
        y[:, 1],
        x[:, 2],
    )

    assert torch.equal(
        y[:, 2],
        x[:, 0],
    )


def test_order_ranking_prefers_correct_prediction():
    torch.manual_seed(0)

    target = torch.randn(
        2, 3, 4, 32
    )

    correct = target.clone()
    wrong = -target

    loss, gap = (
        history_order_ranking_loss(
            correct,
            wrong,
            target,
            margin=0.005,
        )
    )

    assert loss.item() < 1e-6
    assert gap.item() > 1.0


def test_residual_loss_is_zero_for_correct_delta():
    torch.manual_seed(0)

    current = torch.randn(
        2, 4, 32
    )

    delta = torch.randn(
        2, 3, 4, 32
    )

    target = (
        current[:, None]
        + delta
    )

    prediction = target.clone()

    loss = residual_jepa_loss(
        prediction,
        target,
        current,
    )

    assert loss.item() < 1e-6


def test_v2_forward_is_finite_and_has_expected_metrics():
    torch.manual_seed(0)

    model = make_v2()

    context = torch.randn(
        2, 4, 4, 32
    )

    target = torch.randn(
        2, 3, 4, 32
    )

    delta_t = torch.tensor(
        [
            [0.05, 0.10, 0.20],
            [0.05, 0.10, 0.20],
        ]
    )

    out = model(
        context,
        target,
        delta_t,
    )

    assert out[
        "prediction"
    ].shape == (
        2, 3, 4, 32
    )

    for key in [
        "loss",
        "future_loss",
        "residual_loss",
        "order_loss",
        "order_gap",
        "representation_std",
        "mean_cosine",
    ]:
        assert torch.isfinite(
            out[key]
        ).all(), key

    out["loss"].backward()

    assert any(
        p.grad is not None
        for p in
        model.online_encoder.parameters()
    )

    assert any(
        p.grad is not None
        for p in
        model.predictor.parameters()
    )

    assert all(
        p.grad is None
        for p in
        model.target_encoder.parameters()
    )


def test_v2_zero_aux_weights_matches_v1():
    torch.manual_seed(123)

    base = make_base()

    v2 = make_v2(
        residual_weight=0.0,
        order_weight=0.0,
    )

    v2.load_state_dict(
        base.state_dict(),
        strict=True,
    )

    context = torch.randn(
        2, 4, 4, 32
    )

    target = torch.randn(
        2, 3, 4, 32
    )

    delta_t = torch.tensor(
        [
            [0.05, 0.10, 0.20],
            [0.05, 0.10, 0.20],
        ]
    )

    base.eval()
    v2.eval()

    with torch.no_grad():
        a = base(
            context,
            target,
            delta_t,
        )

        b = v2(
            context,
            target,
            delta_t,
        )

    assert torch.allclose(
        a["prediction"],
        b["prediction"],
        atol=1e-6,
        rtol=1e-6,
    )

    assert torch.allclose(
        a["loss"],
        b["loss"],
        atol=1e-6,
        rtol=1e-6,
    )
