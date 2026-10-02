# DSEC Event-Only Preprocessing Hotfix Design

## Goal

Make the documented single-sequence DSEC preprocessing command executable with
the downloaded GEP Stage-1 Event Encoder Small checkpoint, without requiring a
DINOv2 image checkpoint.

## Approved behavior

- The CLI accepts `--split`, repeatable `--sequence`,
  `--event-encoder-ckpt`, `--image-encoder-ckpt`, `--prepare-pairs`, and
  `--event-only`.
- `--root` and `--event-encoder-ckpt` are required and validated before heavy
  preprocessing begins.
- `--event-only` does not load an image encoder, does not require an image
  checkpoint, and does not create, delete, or write `imageToken`.
- Without `--event-only`, `--image-encoder-ckpt` is required and both event and
  image tokens are generated.
- A selected split and sequence restrict both pair preparation and tokenization.
- GEP checkpoints may be either a raw state dictionary or a mapping containing
  `event_encoder`; model loading remains strict.
- Existing `--stats_out` spelling remains accepted; `--stats-out` is added as
  the preferred spelling.
- `environment.yml` declares the previously missing OpenCV and Numba runtime
  dependencies.

## Non-goals

- Changing Stage-1 or Event-JEPA model architecture.
- Redesigning train/validation sequence splits.
- Adding disparity, optical-flow, LiDAR, or IMU preprocessing.
- Optimizing multi-GPU token extraction.
