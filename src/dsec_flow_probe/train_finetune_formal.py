from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset

from dsec_flow_probe.dataset import DSECSequenceFlowDataset
from dsec_flow_probe.head import PatchLinearFlowHead, masked_flow_l1
from dsec_flow_probe.backbone import module_sha256
from event_jepa.encoders import EventTokenEncoder
from utils import get_lr


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_names(path):
    path = Path(path)
    names = [
        x.strip()
        for x in path.read_text().splitlines()
        if x.strip()
    ]
    if not names:
        raise RuntimeError(f"empty sequence file: {path}")
    if len(names) != len(set(names)):
        raise RuntimeError(f"duplicate names: {path}")
    return sorted(names)


def build_encoder(checkpoint):
    state = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    if "online_encoder" not in state:
        raise KeyError("checkpoint missing online_encoder")

    cfg = dict(state["config"])

    encoder = EventTokenEncoder(
        embed_dim=int(cfg["embed_dim"]),
        num_heads=int(cfg["num_heads"]),
        num_layers=int(cfg["encoder_layers"]),
        n_tokens=int(cfg["n_tokens"]),
        max_positions=int(cfg["max_positions"]),
        max_context_frames=int(cfg["max_context_frames"]),
    )

    encoder.load_state_dict(
        state["online_encoder"],
        strict=True,
    )

    return encoder, cfg


def stratified_fraction_indices(dataset, fraction, seed):
    groups = defaultdict(list)

    for i, sample in enumerate(dataset.samples):
        groups[sample["sequence"]].append(i)

    rng = np.random.default_rng(seed)
    chosen = []

    per_sequence = {}

    for sequence in sorted(groups):
        indices = np.asarray(
            groups[sequence],
            dtype=np.int64,
        )

        k = max(
            1,
            int(round(len(indices) * fraction)),
        )

        selected = rng.choice(
            indices,
            size=k,
            replace=False,
        )

        selected = sorted(
            int(x) for x in selected
        )

        chosen.extend(selected)

        per_sequence[sequence] = {
            "available": len(indices),
            "selected": len(selected),
        }

    return sorted(chosen), per_sequence


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--train-sequences-file", required=True)
    p.add_argument("--final-sequences-file", required=True)
    p.add_argument("--output-dir", required=True)

    p.add_argument(
        "--protocol",
        required=True,
        choices=["ft10", "full"],
    )

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-steps", type=int, default=5000)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)

    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--stage2-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)

    p.add_argument(
        "--precision",
        choices=["fp32", "bf16"],
        default="bf16",
    )

    p.add_argument("--log-every", type=int, default=50)

    args = p.parse_args()

    seed_everything(args.seed)

    root = Path(args.root)

    train_names = read_names(
        args.train_sequences_file
    )
    final_names = read_names(
        args.final_sequences_file
    )

    # --------------------------------------------------
    # Formal split safety.
    # FINAL is declared but NEVER loaded.
    # --------------------------------------------------
    overlap = set(train_names) & set(final_names)
    if overlap:
        raise RuntimeError(
            f"FINAL leakage: {sorted(overlap)}"
        )

    flow_root = root / "train_optical_flow"

    all_names = sorted(
        p.name
        for p in flow_root.iterdir()
        if p.is_dir()
    )

    if (
        set(train_names) | set(final_names)
    ) != set(all_names):
        raise RuntimeError(
            "train + FINAL does not exactly cover "
            "all DSEC flow sequences"
        )

    base_dataset = DSECSequenceFlowDataset(
        root,
        train_names,
        context_frames=4,
    )

    if args.protocol == "ft10":
        indices, per_sequence = (
            stratified_fraction_indices(
                base_dataset,
                fraction=0.10,
                seed=args.seed,
            )
        )
        train_dataset = Subset(
            base_dataset,
            indices,
        )
        label_fraction = 0.10
    else:
        indices = list(range(len(base_dataset)))
        per_sequence = {
            s: {
                "available": sum(
                    x["sequence"] == s
                    for x in base_dataset.samples
                ),
                "selected": sum(
                    x["sequence"] == s
                    for x in base_dataset.samples
                ),
            }
            for s in train_names
        }
        train_dataset = base_dataset
        label_fraction = 1.0

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        generator=generator,
        persistent_workers=(args.num_workers > 0),
    )

    device = torch.device("cuda")

    encoder, cfg = build_encoder(
        args.checkpoint
    )
    encoder = encoder.to(device)

    if int(cfg["context_frames"]) != 4:
        raise RuntimeError(
            "formal downstream requires Tc=4"
        )

    if int(cfg["n_tokens"]) != 256:
        raise RuntimeError(
            "expected n_tokens=256"
        )

    if int(cfg["embed_dim"]) != 384:
        raise RuntimeError(
            "expected embed_dim=384"
        )

    # Stage-2 is trainable for B/C.
    encoder.requires_grad_(True)
    encoder.train()

    # Reset RNG so B/C get identical head initialization.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    head = PatchLinearFlowHead(
        embed_dim=384,
        grid_size=16,
        patch_size=14,
    ).to(device)

    encoder_hash_before = module_sha256(encoder)
    head_hash_before = module_sha256(head)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": encoder.parameters(),
                "lr": args.stage2_lr,
                "weight_decay": args.weight_decay,
                "name": "stage2",
            },
            {
                "params": head.parameters(),
                "lr": args.head_lr,
                "weight_decay": 0.0,
                "name": "head",
            },
        ]
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "protocol": args.protocol,
        "seed": args.seed,
        "label_fraction": label_fraction,
        "base_train_samples": len(base_dataset),
        "selected_train_samples": len(train_dataset),
        "per_sequence": per_sequence,
        "selected_indices": indices,
        "FINAL_sequences_declared_only": final_names,
    }

    (
        out / "subset_manifest.json"
    ).write_text(
        json.dumps(manifest, indent=2)
    )

    print("=" * 80)
    print("FORMAL EVENT-JEPA DSEC FLOW FINE-TUNING")
    print("=" * 80)
    print("protocol             :", args.protocol)
    print("Stage-1              : FROZEN / precomputed")
    print("Event-JEPA Stage-2   : TRAINABLE")
    print("flow head            : TRAINABLE")
    print("label fraction       :", label_fraction)
    print("all train samples    :", len(base_dataset))
    print("selected samples     :", len(train_dataset))
    print("FINAL sequences      :", len(final_names), "(NOT LOADED)")
    print("stage2 lr            :", args.stage2_lr)
    print("head lr              :", args.head_lr)
    print("precision            :", args.precision)

    train_iter = iter(loader)

    running_loss = 0.0
    running_count = 0
    start = time.time()

    params = (
        list(encoder.parameters())
        + list(head.parameters())
    )

    for step in range(args.total_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(loader)
            batch = next(train_iter)

        context = batch["context"].to(
            device,
            non_blocking=True,
        )

        target = batch["flow"].to(
            device,
            non_blocking=True,
        )

        stage2_lr = get_lr(
            step,
            args.warmup_steps,
            args.stage2_lr,
            args.total_steps,
            0.0,
        )

        head_lr = get_lr(
            step,
            args.warmup_steps,
            args.head_lr,
            args.total_steps,
            0.0,
        )

        optimizer.param_groups[0]["lr"] = stage2_lr
        optimizer.param_groups[1]["lr"] = head_lr

        optimizer.zero_grad(set_to_none=True)

        use_amp = args.precision == "bf16"

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            memory = encoder(context)

        expected = (
            context.shape[0],
            4 * 256,
            384,
        )

        if tuple(memory.shape) != expected:
            raise RuntimeError(
                f"bad Stage-2 shape: {tuple(memory.shape)}"
            )

        # Same representation used by frozen Flow-A:
        # latest frame after temporal contextualization.
        z = memory[:, -256:, :]

        pred = head(z.float())

        loss = masked_flow_l1(
            pred,
            target,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"non-finite loss at step {step + 1}"
            )

        loss.backward()

        grad = clip_grad_norm_(
            params,
            max_norm=1.0,
        )

        optimizer.step()

        running_loss += float(loss.item())
        running_count += 1

        if (
            step == 0
            or (step + 1) % args.log_every == 0
            or (step + 1) == args.total_steps
        ):
            print(
                f"step={step + 1:05d}/"
                f"{args.total_steps} "
                f"loss={loss.item():.5f} "
                f"grad={float(grad):.4f} "
                f"stage2_lr={stage2_lr:.8f} "
                f"head_lr={head_lr:.8f} "
                f"time={time.time() - start:.1f}s"
            )

    encoder_hash_after = module_sha256(encoder)
    head_hash_after = module_sha256(head)

    encoder_changed = (
        encoder_hash_before != encoder_hash_after
    )
    head_changed = (
        head_hash_before != head_hash_after
    )

    if not encoder_changed:
        raise RuntimeError(
            "Stage-2 encoder did not change"
        )

    if not head_changed:
        raise RuntimeError(
            "flow head did not change"
        )

    wallclock = time.time() - start

    summary = {
        **manifest,
        "source_stage2_checkpoint":
            str(Path(args.checkpoint).resolve()),
        "total_steps": args.total_steps,
        "warmup_steps": args.warmup_steps,
        "batch_size": args.batch_size,
        "stage2_lr": args.stage2_lr,
        "head_lr": args.head_lr,
        "weight_decay": args.weight_decay,
        "precision": args.precision,
        "train_mean_loss":
            running_loss / max(running_count, 1),
        "stage2_changed": encoder_changed,
        "head_changed": head_changed,
        "stage2_sha_before": encoder_hash_before,
        "stage2_sha_after": encoder_hash_after,
        "head_sha_before": head_hash_before,
        "head_sha_after": head_hash_after,
        "FINAL_evaluated": False,
        "wallclock_seconds": wallclock,
    }

    torch.save(
        {
            "online_encoder": encoder.state_dict(),
            "head": head.state_dict(),
            "source_config": cfg,
            "summary": summary,
        },
        out / "final.pt",
    )

    (
        out / "summary.json"
    ).write_text(
        json.dumps(summary, indent=2)
    )

    print()
    print("=" * 80)
    print("FORMAL FLOW FINE-TUNING TRAIN-ONLY: PASS")
    print("=" * 80)
    print("protocol           :", args.protocol)
    print("train samples      :", len(train_dataset))
    print("train mean loss    :", summary["train_mean_loss"])
    print("Stage-2 changed    :", encoder_changed)
    print("head changed       :", head_changed)
    print("FINAL evaluated    :", False)
    print("wallclock seconds  :", round(wallclock, 2))


if __name__ == "__main__":
    main()
