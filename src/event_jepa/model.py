import copy

import torch
import torch.nn.functional as F
from torch import nn

from event_jepa.encoders import EventTokenEncoder
from event_jepa.predictor import EventPredictor


def cosine_jepa_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Patchwise cosine distance with a stop-gradient target."""

    return (
        1.0 - F.cosine_similarity(prediction, target.detach(), dim=-1)
    ).mean()


class EventJEPA(nn.Module):
    """Joint-embedding predictive model for precomputed GEP event tokens."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        encoder_layers: int,
        predictor_layers: int,
        n_tokens: int,
        max_positions: int,
        max_context_frames: int,
        max_horizons: int,
    ) -> None:
        super().__init__()
        self.online_encoder = EventTokenEncoder(
            embed_dim,
            num_heads,
            encoder_layers,
            n_tokens,
            max_positions,
            max_context_frames,
        )
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_encoder.requires_grad_(False)
        self.target_encoder.eval()
        self.predictor = EventPredictor(
            embed_dim,
            num_heads,
            predictor_layers,
            n_tokens,
            max_horizons,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_encoder.eval()
        return self

    def forward(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        context_latent = self.online_encoder(context)
        prediction = self.predictor(context_latent, delta_t)
        batch, horizons, patches, dim = target.shape
        with torch.no_grad():
            encoded_target = self.target_encoder(
                target.reshape(batch * horizons, 1, patches, dim)
            )
            encoded_target = encoded_target.reshape(
                batch,
                horizons,
                patches,
                dim,
            )

        loss = cosine_jepa_loss(prediction, encoded_target)
        normalized_prediction = F.normalize(prediction.detach(), dim=-1)
        normalized_target = F.normalize(encoded_target.detach(), dim=-1)
        return {
            "loss": loss,
            "prediction": prediction,
            "target": encoded_target,
            "representation_std": encoded_target.float()
            .std(dim=(0, 1, 2))
            .mean(),
            "mean_cosine": (
                normalized_prediction * normalized_target
            ).sum(dim=-1).mean(),
        }

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("momentum must lie within [0, 1]")
        for online, target in zip(
            self.online_encoder.parameters(),
            self.target_encoder.parameters(),
            strict=True,
        ):
            target.mul_(momentum).add_(online, alpha=1.0 - momentum)
