from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset


# ---------------------------------------------------------------------
# Neutralize the legacy/local transfer-protocol patch while SegConfig
# and SEG are constructed. This launcher applies the FINAL protocol
# itself and rebuilds the optimizer afterwards.
# ---------------------------------------------------------------------
os.environ["SEG_TRANSFER_PROTOCOL"] = "full"

from config import SegConfig
from seg import SEG
from utils import get_param_groups


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def module_sha256(module: torch.nn.Module) -> str:
    h = hashlib.sha256()

    for name, tensor in sorted(module.state_dict().items()):
        x = (
            tensor.detach()
            .cpu()
            .contiguous()
        )

        h.update(name.encode())
        h.update(str(tuple(x.shape)).encode())
        h.update(str(x.dtype).encode())
        h.update(
            x.view(torch.uint8)
            .numpy()
            .tobytes()
        )

    return h.hexdigest()


def deterministic_10pct_subset(
    dataset,
    seed: int,
):
    """
    Deterministic 10% label subset, sampled independently
    inside every DSEC semantic training sequence.

    This keeps all semantic training sequences represented.
    """
    if not hasattr(dataset, "sequence_names"):
        raise TypeError(
            "10% protocol requires DSECSegmentSequenceDataset "
            "with sequence_names"
        )

    if len(dataset.sequence_names) != len(dataset):
        raise RuntimeError(
            "sequence_names length mismatch"
        )

    groups = defaultdict(list)

    for i, seq in enumerate(dataset.sequence_names):
        groups[str(seq)].append(i)

    rng = np.random.default_rng(seed)

    selected = []
    per_sequence = {}

    for seq in sorted(groups):
        indices = np.asarray(
            groups[seq],
            dtype=np.int64,
        )

        # 10% labels per training sequence.
        k = max(
            1,
            int(round(len(indices) * 0.10)),
        )

        chosen = rng.choice(
            indices,
            size=k,
            replace=False,
        )

        chosen = sorted(
            int(x) for x in chosen
        )

        selected.extend(chosen)

        per_sequence[seq] = {
            "available": int(len(indices)),
            "selected": int(len(chosen)),
        }

    return sorted(selected), per_sequence


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--protocol",
        required=True,
        choices=[
            "ft10",
            "full",
            "stage2-only",
            "stage1-only",
        ],
    )

    p.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    args = p.parse_args()

    seed_everything(args.seed)

    output_dir = Path(
        os.environ.get(
            "SEG_OUTPUT_DIR",
            "runs/seg_transfer_formal",
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # Build exact same Semantic/Event-JEPA configuration as Protocol A.
    # ---------------------------------------------------------------
    config = SegConfig()

    if getattr(
        config,
        "seg_stage2_source",
        None,
    ) != "event_jepa":
        raise RuntimeError(
            "Formal B/C/D requires raw Event-JEPA "
            "Stage-2 checkpoint"
        )

    if int(
        getattr(
            config,
            "seg_temporal_context",
            -1,
        )
    ) != 4:
        raise RuntimeError(
            "Formal Event-JEPA Semantic requires Tc=4"
        )

    # The launcher owns freezing.
    config.encoder_frozen = False

    # Keep any older local protocol implementation in a neutral/full mode.
    if hasattr(config, "transfer_protocol"):
        config.transfer_protocol = "full"

    if hasattr(config, "seg_transfer_protocol"):
        config.seg_transfer_protocol = "full"

    base_train_dataset = (
        config.train_dataset
    )

    base_train_size = len(
        base_train_dataset
    )

    # ---------------------------------------------------------------
    # Label protocol.
    # ---------------------------------------------------------------
    subset_info = None
    selected_indices = None

    if args.protocol == "ft10":
        (
            selected_indices,
            subset_info,
        ) = deterministic_10pct_subset(
            base_train_dataset,
            seed=args.seed,
        )

        config.train_dataset = Subset(
            base_train_dataset,
            selected_indices,
        )

        label_fraction = 0.10

    else:
        label_fraction = 1.00

    print()
    print("=" * 78)
    print("SEMANTIC FORMAL DATA PROTOCOL")
    print("=" * 78)

    print(
        "protocol          :",
        args.protocol,
    )
    print(
        "base train size   :",
        base_train_size,
    )
    print(
        "selected samples  :",
        len(config.train_dataset),
    )
    print(
        "label fraction    :",
        label_fraction,
    )
    print(
        "valid/test size   :",
        len(config.valid_dataset),
    )

    manifest = {
        "protocol":
            args.protocol,
        "seed":
            args.seed,
        "label_fraction":
            label_fraction,
        "base_train_size":
            base_train_size,
        "selected_train_size":
            len(config.train_dataset),
        "per_sequence":
            subset_info,
    }

    if selected_indices is not None:
        manifest["selected_indices"] = (
            selected_indices
        )

        if hasattr(
            base_train_dataset,
            "current_keys",
        ):
            manifest[
                "selected_samples"
            ] = [
                {
                    "sequence":
                        str(
                            base_train_dataset
                            .sequence_names[i]
                        ),
                    "key":
                        str(
                            base_train_dataset
                            .current_keys[i]
                        ),
                }
                for i in selected_indices
            ]

    (
        output_dir
        / "protocol_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        )
    )

    # ---------------------------------------------------------------
    # Construct model.
    # ---------------------------------------------------------------
    model = SEG(config).to(
        config.device
    )

    if isinstance(
        model.transformer,
        torch.nn.Identity,
    ):
        raise RuntimeError(
            "Stage-2 transformer unexpectedly missing"
        )

    if model.decoder is None:
        raise RuntimeError(
            "Expected BaselineSegHead decoder"
        )

    # ---------------------------------------------------------------
    # FINAL protocol definition.
    #
    # B: Stage-1 T + Stage-2 T + Head T, 10% labels
    # C: Stage-1 T + Stage-2 T + Head T, 100% labels
    # D: Stage-1 F + Stage-2 T + Head T, 100% labels
    # ---------------------------------------------------------------

    # First remove any freezing done by an older local patch.
    model.requires_grad_(True)

    stage1_trainable = (
        args.protocol
        in {
            "ft10",
            "full",
            "stage1-only",
        }
    )

    stage2_trainable = (
        args.protocol
        != "stage1-only"
    )

    model.event_encoder.requires_grad_(
        stage1_trainable
    )

    model.transformer.requires_grad_(
        stage2_trainable
    )

    model.decoder.requires_grad_(
        True
    )

    # SEG.train_step() calls self.train() every step.
    # Restore eval mode for every frozen component.
    if (
        not stage1_trainable
        or not stage2_trainable
    ):
        def protocol_train(
            self,
            mode: bool = True,
        ):
            torch.nn.Module.train(
                self,
                mode,
            )

            if not stage1_trainable:
                self.event_encoder.eval()

            if not stage2_trainable:
                self.transformer.eval()

            return self

        model.train = types.MethodType(
            protocol_train,
            model,
        )

    # --------------------------------------------------------
    # E: Stage1-only control.
    #
    # Keep the same temporal input/sample pool Tc=4,
    # but use ONLY the current frame's Stage-1 tokens:
    #
    # [B,4,C,H,W]
    #   -> Stage1 on all frames
    #   -> [B,4,N,D]
    #   -> current frame [:,-1]
    #   -> SAME semantic decoder
    #
    # Event-JEPA Stage-2 remains loaded for hash auditing,
    # but is never executed.
    # --------------------------------------------------------

    if args.protocol == "stage1-only":

        original_forward_logits = (
            model._forward_logits
        )

        def stage1_only_forward_logits(
            self,
            x,
        ):
            if x.ndim == 5:

                sequence_tokens = (
                    self.forward_encoder_sequence(
                        x
                    )
                )

                if sequence_tokens.ndim != 4:
                    raise RuntimeError(
                        "bad Stage1 temporal output: "
                        f"{tuple(sequence_tokens.shape)}"
                    )

                tokens = (
                    sequence_tokens[
                        :,
                        -1,
                        :,
                        :,
                    ]
                )

                return self.forward_decoder(
                    tokens
                )

            # Safety fallback for any non-temporal path.
            return original_forward_logits(
                x
            )

        model._forward_logits = (
            types.MethodType(
                stage1_only_forward_logits,
                model,
            )
        )

    # ---------------------------------------------------------------
    # CRITICAL:
    # old optimizer may have been created while parameters had
    # the wrong requires_grad state. Rebuild it AFTER final freezing.
    # ---------------------------------------------------------------
    model.optimizer = torch.optim.AdamW(
        get_param_groups(
            model,
            config.wd,
            config.encoder_lr_mult,
            config.transformer_lr_mult,
        )
    )

    s1_trainable_params = sum(
        p.numel()
        for p in model.event_encoder.parameters()
        if p.requires_grad
    )

    s2_trainable_params = sum(
        p.numel()
        for p in model.transformer.parameters()
        if p.requires_grad
    )

    head_trainable_params = sum(
        p.numel()
        for p in model.decoder.parameters()
        if p.requires_grad
    )

    print()
    print("=" * 78)
    print("SEMANTIC FINAL TRANSFER PROTOCOL")
    print("=" * 78)

    print(
        "protocol          :",
        args.protocol,
    )
    print(
        "GEP Stage-1       :",
        (
            "TRAINABLE"
            if stage1_trainable
            else "FROZEN"
        ),
    )
    print(
        "Event-JEPA Stage-2:",
        (
            "TRAINABLE"
            if stage2_trainable
            else "FROZEN / BYPASSED"
        ),
    )
    print(
        "task head         :",
        "TRAINABLE",
    )
    print(
        "Stage-1 trainable :",
        s1_trainable_params,
    )
    print(
        "Stage-2 trainable :",
        s2_trainable_params,
    )
    print(
        "head trainable    :",
        head_trainable_params,
    )

    expected_trainability = {
        "ft10": {
            "stage1": True,
            "stage2": True,
        },
        "full": {
            "stage1": True,
            "stage2": True,
        },
        "stage2-only": {
            "stage1": False,
            "stage2": True,
        },
        "stage1-only": {
            "stage1": True,
            "stage2": False,
        },
    }[args.protocol]

    actual_trainability = {
        "stage1":
            s1_trainable_params > 0,
        "stage2":
            s2_trainable_params > 0,
    }

    if (
        actual_trainability
        != expected_trainability
    ):
        raise RuntimeError(
            "trainability audit failed: "
            f"actual={actual_trainability}, "
            f"expected={expected_trainability}"
        )

    if head_trainable_params == 0:
        raise RuntimeError(
            "task head must be trainable"
        )

    # State-integrity audit.
    s1_before = module_sha256(
        model.event_encoder
    )

    s2_before = module_sha256(
        model.transformer
    )

    head_before = module_sha256(
        model.decoder
    )

    # ---------------------------------------------------------------
    # Existing SEG.start() provides:
    #   12k optimizer-step loop
    #   chosen LR schedule
    #   validation
    #   checkpoints
    # ---------------------------------------------------------------
    model.start()

    s1_after = module_sha256(
        model.event_encoder
    )

    s2_after = module_sha256(
        model.transformer
    )

    head_after = module_sha256(
        model.decoder
    )

    s1_changed = (
        s1_before != s1_after
    )

    s2_changed = (
        s2_before != s2_after
    )

    head_changed = (
        head_before != head_after
    )

    expected_changes = {
        "ft10": {
            "stage1": True,
            "stage2": True,
            "head": True,
        },
        "full": {
            "stage1": True,
            "stage2": True,
            "head": True,
        },
        "stage2-only": {
            "stage1": False,
            "stage2": True,
            "head": True,
        },
        "stage1-only": {
            "stage1": True,
            "stage2": False,
            "head": True,
        },
    }[args.protocol]

    actual_changes = {
        "stage1": s1_changed,
        "stage2": s2_changed,
        "head": head_changed,
    }

    if actual_changes != expected_changes:
        raise RuntimeError(
            "parameter change audit failed: "
            f"actual={actual_changes}, "
            f"expected={expected_changes}"
        )

    audit = {
        "protocol":
            args.protocol,
        "stage1_changed":
            s1_changed,
        "stage2_changed":
            s2_changed,
        "head_changed":
            head_changed,
        "stage1_sha_before":
            s1_before,
        "stage1_sha_after":
            s1_after,
        "stage2_sha_before":
            s2_before,
        "stage2_sha_after":
            s2_after,
        "head_sha_before":
            head_before,
        "head_sha_after":
            head_after,
    }

    (
        output_dir
        / "parameter_audit.json"
    ).write_text(
        json.dumps(
            audit,
            indent=2,
        )
    )

    print()
    print("=" * 78)
    print("SEMANTIC FORMAL TRAINING: PASS")
    print("=" * 78)
    print(
        "protocol       :",
        args.protocol,
    )
    print(
        "Stage-1 changed:",
        s1_changed,
    )
    print(
        "Stage-2 changed:",
        s2_changed,
    )
    print(
        "head changed   :",
        head_changed,
    )


if __name__ == "__main__":
    main()
