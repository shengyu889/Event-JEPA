from types import SimpleNamespace

import torch
from torch import nn

from model import Block


class EventTokenEncoder(nn.Module):
    """Encode a short temporal context while retaining GEP transformer keys."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        n_tokens: int,
        max_positions: int,
        max_context_frames: int,
    ) -> None:
        super().__init__()
        if n_tokens > max_positions:
            raise ValueError("n_tokens must not exceed max_positions")
        self.embed_dim = embed_dim
        self.n_tokens = n_tokens
        self.max_context_frames = max_context_frames
        block_config = SimpleNamespace(n_embed=embed_dim, n_head=num_heads)
        self.transformer = nn.ModuleDict(
            {
                "modality_embed": nn.Embedding(5, embed_dim),
                "pos_embed": nn.Embedding(max_positions, embed_dim),
                "blocks": nn.ModuleList(
                    [Block(block_config) for _ in range(num_layers)]
                ),
                "norm": nn.LayerNorm(embed_dim),
            }
        )
        self.temporal_embed = nn.Embedding(max_context_frames, embed_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 4:
            raise ValueError("tokens must have shape [B,T,N,D]")
        batch, frames, patches, dim = tokens.shape
        if frames > self.max_context_frames:
            raise ValueError("frames exceeds max_context_frames")
        if (patches, dim) != (self.n_tokens, self.embed_dim):
            raise ValueError(
                f"expected patch shape {(self.n_tokens, self.embed_dim)}"
            )

        spatial_ids = torch.arange(patches, device=tokens.device)
        temporal_ids = torch.arange(frames, device=tokens.device)
        x = tokens
        x = x + self.transformer.pos_embed(spatial_ids)[None, None, :, :]
        x = x + self.temporal_embed(temporal_ids)[None, :, None, :]
        modality_ids = torch.full(
            (batch, frames, patches),
            2,
            device=tokens.device,
            dtype=torch.long,
        )
        x = x + self.transformer.modality_embed(modality_ids)
        x = x.reshape(batch, frames * patches, dim)
        for block in self.transformer.blocks:
            x = block(x, is_causal=False)
        return self.transformer.norm(x)

    def gep_transformer_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu()
            for key, value in self.transformer.state_dict().items()
        }
