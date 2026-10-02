from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn

from event_jepa.encoders import EventTokenEncoder
from train_gep_ar_matched import MatchedGEPAR
from utils import spatiotemporal_aggregate


def module_sha256(module: nn.Module) -> str:
    h = hashlib.sha256()

    for name, tensor in sorted(
        module.state_dict().items()
    ):
        x = (
            tensor.detach()
            .cpu()
            .contiguous()
        )

        h.update(name.encode())
        h.update(
            str(tuple(x.shape)).encode()
        )
        h.update(str(x.dtype).encode())
        h.update(
            x.view(torch.uint8)
            .numpy()
            .tobytes()
        )

    return h.hexdigest()


class FrozenFlowBackbone(nn.Module):
    """
    Frozen representation backbone for controlled DSEC-flow probing.

    mode="stage1":
        context [B,4,N,D]
        -> use raw last Stage-1 token frame
        -> [B,N,D]

    mode="jepa":
        load full Event-JEPA online encoder
        -> use exactly the context length that checkpoint was trained with
        -> select last spatial frame after temporal encoding
        -> [B,N,D]
    """

    def __init__(
        self,
        mode: str,
        checkpoint: str | Path | None = None,
    ):
        super().__init__()

        if mode not in {
            "stage1",
            "stage1-mean4",
            "jepa",
            "gep-ar",
            "gep-opt-style",
        }:
            raise ValueError(
                f"unsupported mode: {mode}"
            )

        self.mode = mode
        self.encoder = None
        self.config = None

        self.n_tokens = 256
        self.embed_dim = 384
        self.context_frames = 1

        if mode in {
            "stage1",
            "stage1-mean4",
        }:
            # No trainable representation module:
            # input files are already frozen Stage-1 tokens.
            self.context_frames = (
                1
                if mode == "stage1"
                else 4
            )
            return

        if mode in {
            "gep-ar",
            "gep-opt-style",
        }:
            if checkpoint is None:
                raise ValueError(
                    "GEP-AR mode requires checkpoint"
                )

            checkpoint = Path(checkpoint)

            state = torch.load(
                checkpoint,
                map_location="cpu",
                weights_only=False,
            )

            if "transformer" not in state:
                raise KeyError(
                    f"{checkpoint}: missing transformer"
                )

            if "config" not in state:
                raise KeyError(
                    f"{checkpoint}: missing config"
                )

            cfg = dict(state["config"])

            self.config = cfg
            self.context_frames = (
                1
                if mode == "gep-opt-style"
                else 4
            )

            tmp = MatchedGEPAR(
                embed_dim=int(
                    cfg["embed_dim"]
                ),
                num_heads=int(
                    cfg["num_heads"]
                ),
                num_layers=int(
                    cfg["num_layers"]
                ),
                n_tokens=int(
                    cfg["n_tokens"]
                ),
                window_size=4096,
                images_per_group=int(
                    cfg["images_per_group"]
                ),
            )

            tmp.transformer.load_state_dict(
                state["transformer"],
                strict=True,
            )

            self.encoder = tmp.transformer

            self.encoder.requires_grad_(False)
            self.encoder.eval()

            self.embed_dim = int(
                cfg["embed_dim"]
            )
            self.n_tokens = int(
                cfg["n_tokens"]
            )
            self.images_per_group = int(
                cfg["images_per_group"]
            )

            return

        if checkpoint is None:
            raise ValueError(
                "JEPA mode requires checkpoint"
            )

        checkpoint = Path(checkpoint)

        state = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False,
        )

        if "online_encoder" not in state:
            raise KeyError(
                f"{checkpoint}: missing online_encoder"
            )

        if "config" not in state:
            raise KeyError(
                f"{checkpoint}: missing config"
            )

        cfg = state["config"]

        required = [
            "embed_dim",
            "num_heads",
            "encoder_layers",
            "n_tokens",
            "max_positions",
            "max_context_frames",
            "context_frames",
            "horizons",
        ]

        missing = [
            k for k in required
            if k not in cfg
        ]

        if missing:
            raise KeyError(
                f"checkpoint config missing: {missing}"
            )

        self.config = dict(cfg)

        self.embed_dim = int(
            cfg["embed_dim"]
        )

        self.n_tokens = int(
            cfg["n_tokens"]
        )

        self.context_frames = int(
            cfg["context_frames"]
        )

        self.encoder = EventTokenEncoder(
            embed_dim=self.embed_dim,
            num_heads=int(
                cfg["num_heads"]
            ),
            num_layers=int(
                cfg["encoder_layers"]
            ),
            n_tokens=self.n_tokens,
            max_positions=int(
                cfg["max_positions"]
            ),
            max_context_frames=int(
                cfg["max_context_frames"]
            ),
        )

        self.encoder.load_state_dict(
            state["online_encoder"],
            strict=True,
        )

        self.encoder.requires_grad_(False)
        self.encoder.eval()

    def train(
        self,
        mode: bool = True,
    ):
        # The representation must ALWAYS stay in eval mode.
        super().train(False)

        if self.encoder is not None:
            self.encoder.eval()

        return self

    @torch.no_grad()
    def forward(
        self,
        context: torch.Tensor,
    ) -> torch.Tensor:
        if context.ndim != 4:
            raise ValueError(
                "context must be [B,T,N,D]"
            )

        B, T, N, D = context.shape

        if (
            N != self.n_tokens
            or D != self.embed_dim
        ):
            raise ValueError(
                f"expected N,D="
                f"{self.n_tokens},{self.embed_dim}, "
                f"got {N},{D}"
            )

        if self.mode == "stage1":
            # Raw latest Stage-1 frame.
            return context[:, -1].float()

        if self.mode == "stage1-mean4":
            if T < 4:
                raise ValueError(
                    "stage1-mean4 requires 4 frames"
                )

            # Exact pre-Transformer temporal patch
            # aggregation used by GEP for the
            # patch-token branch.
            return (
                context[:, -4:]
                .float()
                .mean(dim=1)
            )

        if self.mode == "gep-opt-style":
            if T < 1:
                raise ValueError(
                    "gep-opt-style requires >=1 frame"
                )

            # ------------------------------------------------
            # Match src/opt.py transfer behavior:
            #
            # current Stage-1 patch tokens only
            # + modality embedding
            # + position embedding
            # non-causal Transformer
            # NO final LayerNorm
            # residual connection to raw Stage-1 tokens
            # ------------------------------------------------

            x = context[:, -1].float()

            B, N, D = x.shape

            ids = torch.full(
                (B, N),
                2,
                dtype=torch.long,
                device=x.device,
            )

            pos = torch.arange(
                N,
                dtype=torch.long,
                device=x.device,
            )

            z = (
                x
                + self.encoder.modality_embed(ids)
                + self.encoder.pos_embed(pos)[None]
            )

            for block in self.encoder.blocks:
                # Intentionally NON-CAUSAL:
                # matches src/opt.py
                z = block(z)

            # Intentionally:
            #   no self.encoder.norm(z)
            #
            # src/opt.py returns x_ + x
            z = z + x

            return z.float()

        if self.mode == "gep-ar":
            if T < 4:
                raise ValueError(
                    "GEP-AR flow probe requires 4 frames"
                )

            x_in = context[:, -4:]

            B, T4, N, D = x_in.shape

            slot = x_in.reshape(
                B,
                T4 * N,
                D,
            )

            ids = torch.full(
                (B, T4 * N),
                2,
                dtype=torch.long,
                device=x_in.device,
            )

            slot, ids = (
                spatiotemporal_aggregate(
                    slot,
                    ids,
                    tokens_per_image=N,
                    images_per_group=(
                        self.images_per_group
                    ),
                )
            )

            # Four real frames + four padded frames:
            # 256 temporal patch tokens
            # + 4 valid frame-summary tokens.
            expected_len = N + T4

            if slot.shape != (
                B,
                expected_len,
                D,
            ):
                raise RuntimeError(
                    "unexpected GEP aggregation "
                    f"shape: {tuple(slot.shape)}, "
                    f"expected "
                    f"{(B, expected_len, D)}"
                )

            pos = torch.arange(
                expected_len,
                dtype=torch.long,
                device=x_in.device,
            )

            z = (
                slot
                + self.encoder
                    .modality_embed(ids)
                + self.encoder
                    .pos_embed(pos)[None]
            )

            for block in (
                self.encoder.blocks
            ):
                z = block(
                    z,
                    is_causal=True,
                )

            z = self.encoder.norm(z)

            # Dense spatially aligned representation.
            # Do NOT use the 4 global frame-summary tokens.
            z_patch = z[:, :N]

            return z_patch.float()

        required_t = self.context_frames

        if T < required_t:
            raise ValueError(
                f"checkpoint requires "
                f"{required_t} context frames, "
                f"batch only has {T}"
            )

        # Important:
        # Tc1 checkpoint receives only t0.
        # Tc4 checkpoint receives t-3:t0.
        x = context[
            :,
            -required_t:,
            :,
            :,
        ]

        z = self.encoder(x)

        expected = (
            B,
            required_t * self.n_tokens,
            self.embed_dim,
        )

        if tuple(z.shape) != expected:
            raise RuntimeError(
                f"unexpected encoded shape "
                f"{tuple(z.shape)}, "
                f"expected {expected}"
            )

        # Last 256 tokens correspond to the latest
        # context frame t0, but after non-causal
        # temporal/contextual encoding.
        z_last = z[
            :,
            -self.n_tokens:,
            :,
        ]

        return z_last.float()
