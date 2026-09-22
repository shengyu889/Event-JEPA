import argparse
from pathlib import Path

import torch

from event_jepa.checkpoint import export_gep_transformer
from event_jepa.config import EventJEPAConfig
from event_jepa.model import EventJEPA


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export an Event-JEPA online encoder in GEP transformer format."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = EventJEPAConfig(**state["config"])
    model = EventJEPA(
        cfg.embed_dim,
        cfg.num_heads,
        cfg.encoder_layers,
        cfg.predictor_layers,
        cfg.n_tokens,
        cfg.max_positions,
        cfg.max_context_frames,
        len(cfg.horizons),
    )
    model.online_encoder.load_state_dict(state["online_encoder"], strict=True)
    export_gep_transformer(Path(args.output), model, cfg.to_dict())


if __name__ == "__main__":
    main()
