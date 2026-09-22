import torch
from torch import nn

from event_jepa.time_embedding import ContinuousTimeEmbedding


class EventPredictor(nn.Module):
    """Predict token-grid targets at one or more continuous-time horizons."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        n_tokens: int,
        max_horizons: int = 8,
    ) -> None:
        super().__init__()
        self.n_tokens = n_tokens
        self.max_horizons = max_horizons
        self.spatial_query = nn.Parameter(torch.zeros(n_tokens, embed_dim))
        nn.init.trunc_normal_(self.spatial_query, std=0.02)
        self.horizon_embed = nn.Embedding(max_horizons, embed_dim)
        self.time_embed = ContinuousTimeEmbedding(embed_dim)
        layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim),
        )

    def forward(
        self,
        memory: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> torch.Tensor:
        if memory.ndim != 3:
            raise ValueError("memory must have shape [B,S,D]")
        if delta_t.ndim != 2:
            raise ValueError("delta_t must have shape [B,K]")
        batch, horizons = delta_t.shape
        if horizons > self.max_horizons:
            raise ValueError("horizons exceeds max_horizons")
        if memory.shape[0] != batch:
            raise ValueError("memory and delta_t batch sizes differ")

        horizon_ids = torch.arange(horizons, device=memory.device)
        query = self.spatial_query[None, None, :, :]
        query = query + self.time_embed(delta_t)[:, :, None, :]
        query = query + self.horizon_embed(horizon_ids)[None, :, None, :]
        query = query.reshape(batch * horizons, self.n_tokens, memory.shape[-1])
        repeated_memory = memory[:, None].expand(-1, horizons, -1, -1)
        repeated_memory = repeated_memory.reshape(
            batch * horizons,
            memory.shape[1],
            memory.shape[2],
        )
        output = self.decoder(query, repeated_memory)
        return output.reshape(batch, horizons, self.n_tokens, memory.shape[-1])
