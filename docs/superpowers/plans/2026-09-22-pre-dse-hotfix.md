# DSEC Event-Only Preprocessing Hotfix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the documented one-sequence DSEC event-token command work with the official wrapped GEP Stage-1 Event Encoder Small checkpoint.

**Architecture:** Put dependency-light argument parsing, validation, sequence selection, checkpoint unwrapping, and orchestration in `src/pre_dse_cli.py`. Keep DSEC image/event processing and ViT inference in `src/pre_dse.py`, with an explicit event-only branch that never constructs or touches the image encoder outputs.

**Tech Stack:** Python 3.12, PyTorch 2.7.1, torchvision 0.22.1, OpenCV, Numba, pytest.

**Spec:** `docs/superpowers/specs/2026-09-22-pre-dse-hotfix-design.md`

## Global Constraints

- Preserve strict Stage-1 encoder state loading.
- Preserve existing full paired-token behavior when `--event-only` is absent.
- Do not require semantic-segmentation labels for pair preparation.
- Keep the exact user command valid on Ubuntu with the `gep` Conda environment.
- Do not alter Event-JEPA training architecture or checkpoint format.

## Review Focus

- A missing requested sequence must fail before preprocessing rather than silently doing no work.
- Event-only mode must leave a pre-existing `imageToken` directory untouched.
- A wrapped checkpoint without `event_encoder` must fail clearly rather than being passed to `load_state_dict` as model weights.
- Paired mode without `--image-encoder-ckpt` must fail before model construction.
- An existing raw encoder state dictionary must remain accepted.

---

### Task 1: Dependency-light CLI and orchestration

**Files:**
- Create: `src/pre_dse_cli.py`
- Create: `tests/test_pre_dse_cli.py`

**Interfaces:**
- Consumes: CLI arguments and a processor factory implementing `run`, `process_tokens`, and `compute_rgb_stats`.
- Produces: `build_parser()`, `extract_encoder_state_dict()`, `select_sequences()`, `validate_args()`, and `run_from_args()`.

- [ ] Write tests for the exact approved command, paired-mode validation, wrapped/raw checkpoints, missing sequences, and routing one selected sequence.
- [ ] Run `pytest tests/test_pre_dse_cli.py -q` and confirm failure because `pre_dse_cli` does not exist.
- [ ] Implement the smallest dependency-light module satisfying the tests.
- [ ] Run `pytest tests/test_pre_dse_cli.py -q` and confirm all tests pass.
- [ ] Commit the CLI layer.

### Task 2: Event-only token generation

**Files:**
- Modify: `src/pre_dse.py`
- Create: `tests/test_pre_dse_tokens.py`

**Interfaces:**
- Consumes: the paths and booleans forwarded by `run_from_args()`.
- Produces: strict encoder loading and event-only `eventToken/*.pt` generation.

- [ ] Add a tokenization test using tiny fake encoders and real temporary PNG files; assert event tokens are written and a sentinel in `imageToken` survives.
- [ ] Run the focused test and confirm the old `process_tokens` signature/behavior fails it.
- [ ] Refactor encoder loading to use `extract_encoder_state_dict`, load on CPU, and call `load_state_dict(..., strict=True)`.
- [ ] Implement the event-only branch using `eventImage` inputs and `PairedProcessor(event, None)`.
- [ ] Replace the legacy `__main__` loop with the new parser and orchestration.
- [ ] Run the focused tests and confirm they pass.
- [ ] Commit token generation.

### Task 3: Dependencies, regression, and handoff

**Files:**
- Modify: `environment.yml`
- Modify: `readme.md`

**Interfaces:**
- Consumes: the executable hotfix CLI.
- Produces: reproducible environment declarations and the exact local command.

- [ ] Add `opencv-python-headless` and `numba` to the pip dependencies.
- [ ] Document environment update, checkpoint check, one-sequence preprocessing, output verification, and resume behavior.
- [ ] Run focused tests, the full pytest suite, `--help`, and Python bytecode compilation.
- [ ] Review the whole diff for destructive path handling and accidental image-checkpoint requirements.
- [ ] Generate a patch file and commit the documentation/dependency update.
