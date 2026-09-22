import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def save_checkpoint(path, model, optimizer, scaler, config, step) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "step": step,
            "online_encoder": model.online_encoder.state_dict(),
            "target_encoder": model.target_encoder.state_dict(),
            "predictor": model.predictor.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "config": config.to_dict(),
            "rng_state": _rng_state(),
        },
        path,
    )


def load_checkpoint(path, model, optimizer, scaler):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format_version") != 1:
        raise ValueError(
            f"unsupported checkpoint format: {state.get('format_version')}"
        )
    model.online_encoder.load_state_dict(state["online_encoder"], strict=True)
    model.target_encoder.load_state_dict(state["target_encoder"], strict=True)
    model.predictor.load_state_dict(state["predictor"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and state["scaler"] is not None:
        scaler.load_state_dict(state["scaler"])
    random.setstate(state["rng_state"]["python"])
    np.random.set_state(state["rng_state"]["numpy"])
    torch.set_rng_state(state["rng_state"]["torch"])
    if torch.cuda.is_available() and "cuda" in state["rng_state"]:
        torch.cuda.set_rng_state_all(state["rng_state"]["cuda"])
    return state


def export_gep_transformer(path, model, metadata) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "source": "event_jepa",
        "transformer": model.online_encoder.gep_transformer_state_dict(),
        "metadata": dict(metadata),
    }
    torch.save(payload, path)
