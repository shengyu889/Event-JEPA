from __future__ import annotations

import torch
from torch import nn


class PatchLinearFlowHead(nn.Module):
    """
    [B,256,384]
       ↓ per-patch Linear
    [B,256,2*14*14]
       ↓ rearrange
    [B,2,224,224]

    Mirrors the lightweight patch-wise decoder idea
    used by GEP optical-flow downstream evaluation.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        grid_size: int = 16,
        patch_size: int = 14,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.patch_size = patch_size

        self.n_tokens = (
            grid_size * grid_size
        )

        self.proj = nn.Linear(
            embed_dim,
            2 * patch_size * patch_size,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "expected [B,N,D]"
            )

        B, N, D = x.shape

        if N != self.n_tokens:
            raise ValueError(
                f"expected {self.n_tokens} tokens, "
                f"got {N}"
            )

        if D != self.embed_dim:
            raise ValueError(
                f"expected embed_dim={self.embed_dim}, "
                f"got {D}"
            )

        p = self.patch_size
        g = self.grid_size

        x = self.proj(x)

        x = x.reshape(
            B,
            g,
            g,
            2,
            p,
            p,
        )

        x = x.permute(
            0,
            3,
            1,
            4,
            2,
            5,
        ).contiguous()

        return x.reshape(
            B,
            2,
            g * p,
            g * p,
        )


def masked_flow_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    target: [B,3,H,W]
      target[:,:2] = flow
      target[:,2:] = validity mask
    """

    gt = target[:, :2]
    mask = target[:, 2:3]

    valid = mask.sum()

    if valid <= 0:
        return pred.sum() * 0.0

    error = (
        (pred - gt).abs()
        * mask
    )

    # divide by two flow channels as well
    return error.sum() / (
        valid * 2.0
    )
