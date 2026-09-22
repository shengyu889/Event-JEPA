import math

import torch
from torch import nn


class ContinuousTimeEmbedding(nn.Module):
    """Encode elapsed seconds using fixed Fourier features and a learnable MLP."""

    def __init__(
        self,
        embed_dim: int,
        fourier_dim: int = 64,
        max_frequency: float = 1000.0,
    ) -> None:
        super().__init__()
        if embed_dim < 1:
            raise ValueError("embed_dim must be positive")
        if fourier_dim < 2 or fourier_dim % 2:
            raise ValueError("fourier_dim must be a positive even integer")
        if max_frequency <= 0:
            raise ValueError("max_frequency must be positive")

        frequencies = torch.logspace(
            0.0,
            math.log10(max_frequency),
            fourier_dim // 2,
        )
        self.register_buffer("frequencies", frequencies, persistent=True)
        self.projection = nn.Sequential(
            nn.Linear(fourier_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        if torch.any(delta_t < 0):
            raise ValueError("delta_t must be nonnegative")
        angles = (
            2.0
            * torch.pi
            * delta_t.to(dtype=torch.float32).unsqueeze(-1)
            * self.frequencies
        )
        features = torch.cat((angles.sin(), angles.cos()), dim=-1)
        return self.projection(features)
