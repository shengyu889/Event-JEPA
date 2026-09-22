import torch

from event_jepa.engine import ema_momentum, train_micro_step
from event_jepa.model import EventJEPA


def make_batch():
    return {
        "context": torch.randn(2, 2, 4, 32),
        "target": torch.randn(2, 1, 4, 32),
        "delta_t": torch.tensor([[0.01], [0.02]]),
    }


def make_model():
    return EventJEPA(32, 4, 1, 1, 4, 32, 3, 2)


def test_successful_optimizer_step_updates_target():
    model = make_model()
    optimizer = torch.optim.AdamW(
        list(model.online_encoder.parameters())
        + list(model.predictor.parameters()),
        lr=1e-3,
    )
    before = [
        parameter.clone() for parameter in model.target_encoder.parameters()
    ]
    metrics = train_micro_step(
        model,
        make_batch(),
        optimizer,
        scaler=None,
        device=torch.device("cpu"),
        precision="fp32",
        grad_clip_norm=1.0,
        ema_value=0.9,
        loss_divisor=1,
        should_step=True,
    )
    assert metrics["optimizer_stepped"] is True
    assert any(
        not torch.equal(left, right)
        for left, right in zip(
            before, model.target_encoder.parameters(), strict=True
        )
    )


def test_accumulation_micro_step_does_not_update_target():
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = [
        parameter.clone() for parameter in model.target_encoder.parameters()
    ]
    metrics = train_micro_step(
        model,
        make_batch(),
        optimizer,
        None,
        torch.device("cpu"),
        "fp32",
        1.0,
        0.9,
        loss_divisor=2,
        should_step=False,
    )
    assert metrics["optimizer_stepped"] is False
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            before, model.target_encoder.parameters(), strict=True
        )
    )


def test_ema_schedule_reaches_endpoints():
    assert ema_momentum(0, 100, 0.996, 0.9999) == 0.996
    assert ema_momentum(100, 100, 0.996, 0.9999) == 0.9999


class SkippingScaler:
    def __init__(self):
        self.scale_value = 1024.0

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        return None

    def step(self, optimizer):
        return None

    def update(self):
        self.scale_value /= 2

    def get_scale(self):
        return self.scale_value


def test_skipped_scaled_step_does_not_update_ema():
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = [
        parameter.clone() for parameter in model.target_encoder.parameters()
    ]
    metrics = train_micro_step(
        model,
        make_batch(),
        optimizer,
        SkippingScaler(),
        torch.device("cpu"),
        "fp32",
        1.0,
        0.9,
        loss_divisor=1,
        should_step=True,
    )
    assert metrics["optimizer_stepped"] is False
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            before, model.target_encoder.parameters(), strict=True
        )
    )
