# Event-JEPA on GEP: Integration Design

**Date:** 2026-09-20  
**Target upstream:** `uzh-rpg/generative_event_pretraining`, branch `master`, commit `b07c78c`  
**Status:** Proposed for user review

## 1. Goal

Build a minimal, reproducible Event-JEPA pretraining path inside the official GEP repository so that the central research question can be tested fairly:

> Does non-generative latent prediction learn a more transferable event representation than GEP's autoregressive Stage-2 objective?

The first implementation must reuse GEP's Stage-1-aligned event tokens and downstream evaluation code. It must run a CPU unit-test suite, a small single-GPU smoke test on an RTX 5070 Ti 16 GB, and distributed full pretraining on 4 x A100 80 GB.

## 2. Verified Upstream Constraints

The upstream repository currently:

- extracts Stage-1 features to `eventToken/*.pt` and `imageToken/*.pt`;
- uses `src/dataset.py::SequenceDataset` to flatten saved token files into arbitrary windows;
- uses `src/gpt.py::GPT` as the Stage-2 causal/MAE model;
- saves a `transformer` state dict that `src/cls.py::Cls` can load for downstream transfer;
- hardcodes most data paths and hyperparameters inside `src/config.py`;
- has no test suite.

Consequently, V1 Event-JEPA is a **token-level temporal pretraining method on frozen Stage-1 event features**. It does not claim end-to-end raw-event learning. The online JEPA context Transformer is the transferable component; the EMA target encoder and predictor are pretraining-only.

## 3. Chosen Approach

### Recommended: frame-aware token JEPA

Keep each saved event-token file as one frame/interval, construct temporal context-target tuples without crossing sequence boundaries, and train an online context encoder to predict EMA target-encoder latents at a future timestamp.

This approach is selected because it preserves GEP's Stage-1 data and downstream contract, makes the GEP-vs-JEPA comparison controlled, and fits the available hardware.

### Rejected alternatives

1. **Patch the current flat `SequenceDataset` directly.** This is smaller in line count but can split frames and cross temporal boundaries, so its target time is not well defined.
2. **Train raw-event Event-JEPA end to end immediately.** This is scientifically attractive but changes the tokenizer, encoder, memory budget, and objective simultaneously. It belongs in V2 after the token-level hypothesis is validated.
3. **Reuse GEP's GPT class with only a cosine loss.** A causal next-token model remains autoregressive in structure and would not test a true joint-embedding predictor.

## 4. V1 Data Contract

Add `EventJEPADataset`, which scans each sequence independently:

```text
<root>/<split>_images/<sequence>/images/left/eventToken/*.pt
```

Each sample returns:

```python
{
    "context": Tensor[Tc, N, D],       # past Stage-1 event patch tokens
    "target": Tensor[K, N, D],         # future tokens for configured horizons
    "delta_t": Tensor[K],              # seconds from last context time
    "sequence": str,
    "context_timestamps": Tensor[Tc],
    "target_timestamps": Tensor[K],
}
```

Default dimensions for GEP ViT-S/14 at 224 x 224 are:

- `Tc = 4` context intervals;
- `K = 1` horizon in the minimal experiment;
- `N = 256` spatial patch tokens;
- `D = 384` embedding dimensions.

Timestamps are parsed from numeric filenames. If a filename is not numeric, dataset construction fails with a path-specific error instead of silently substituting frame indices. A sample is valid only when all context and target files belong to the same sequence. Tensor shape and dtype are validated on load.

The dataset accepts frame-offset horizons in V1 (`[1]` by default) but always returns measured `delta_t`; this allows the model and logs to use real elapsed time without inventing missing targets through nearest-neighbor matching. Millisecond target selection is deferred until the sampling policy is benchmarked.

## 5. Model Architecture

### 5.1 Online context encoder

Input `context [B, Tc, N, D]` is reshaped to `[B, Tc*N, D]`. It receives:

- learned spatial patch embeddings `[N, D]`;
- learned context-slot embeddings `[Tc, D]`;
- event modality embedding compatible with GEP;
- 4 non-causal Transformer blocks by default for the local smoke configuration, configurable to 12 for full training.

The output is context memory `z_context [B, Tc*N, D]`.

### 5.2 EMA target encoder

The target encoder is initialized as a deep copy of the online encoder, has `requires_grad=False`, remains in evaluation mode, and is updated only after a successful optimizer step:

\[
\bar\theta \leftarrow m\bar\theta + (1-m)\theta.
\]

Its forward pass is under `torch.no_grad()`. For a target interval, the same encoder consumes `[B, 1, N, D]` and produces `z_target [B, N, D]`.

### 5.3 Continuous-time predictor

For every future horizon, the predictor builds `N` target queries from:

- learned spatial query embeddings;
- a Fourier/sinusoidal embedding of measured `delta_t`, followed by an MLP;
- optional learned horizon-rank embedding.

Two Transformer decoder blocks cross-attend from these queries to `z_context`, producing `z_pred [B, K, N, D]`. No event image, raw event, or discrete token is reconstructed.

### 5.4 Loss

V1 uses normalized patchwise cosine distance only:

\[
\mathcal{L}_{JEPA}
= \frac{1}{BKN}\sum_{b,k,n}
\left(1-
\frac{\hat z_{bkn}\cdot \operatorname{sg}(z_{bkn})}
{\lVert\hat z_{bkn}\rVert_2\lVert z_{bkn}\rVert_2}
\right).
\]

No reconstruction, contrastive, rate-invariance, or motion auxiliary loss is included in the MVP. Those are ablations added only after the base hypothesis is validated.

## 6. Files and Responsibilities

```text
src/
├── event_jepa/
│   ├── __init__.py          # public exports
│   ├── dataset.py           # frame-aware temporal token dataset
│   ├── time_embedding.py    # continuous delta-t encoding
│   ├── encoders.py          # online/EMA token encoder
│   ├── predictor.py         # target-query cross-attention predictor
│   ├── model.py             # EventJEPA composition, loss, EMA update
│   ├── config.py            # side-effect-free dataclass config
│   └── checkpoint.py        # save/load/export contract
├── train_jepa.py            # CLI training and validation entry point
└── config.py                # only downstream checkpoint hook, if required
configs/
├── event_jepa_smoke.yaml    # 5070 Ti smoke configuration
└── event_jepa_a100.yaml     # 4 x A100 full configuration
tests/
├── test_jepa_dataset.py
├── test_time_embedding.py
├── test_jepa_model.py
├── test_ema.py
└── test_checkpoint.py
```

The new Event-JEPA modules must not import `src/config.py`, because importing it constructs real datasets and reads machine-specific paths. Configuration is parsed into a side-effect-free dataclass. Existing GEP files remain usable.

## 7. Training Behavior

One training step is:

1. Load context, target, and measured `delta_t`.
2. Compute online context memory under autocast.
3. Compute EMA target latents under `no_grad`.
4. Predict target latents and compute cosine loss.
5. Scale loss, backpropagate, unscale, clip online gradients, and step optimizer.
6. Update EMA only when the optimizer step is not skipped by gradient scaling.
7. Advance the cosine learning-rate schedule and log loss, learning rate, EMA momentum, throughput, and mean `delta_t`.

The checkpoint contains:

```python
{
    "format_version": 1,
    "step": int,
    "online_encoder": state_dict,
    "target_encoder": state_dict,
    "predictor": state_dict,
    "optimizer": state_dict,
    "scaler": state_dict,
    "config": dict,
    "rng_state": dict,
}
```

Resume must restore the complete training state. A separate export command writes a GEP-downstream-compatible `{"transformer": ...}` checkpoint from the online encoder. Compatibility is verified by a strict load test against the Transformer block structure used by `Cls`; if architecture keys diverge, export fails loudly.

## 8. Hardware Profiles

### RTX 5070 Ti 16 GB smoke/evaluation

- ViT-S token dimension 384;
- `Tc=2`, `K=1`;
- 2 encoder blocks and 1 predictor block;
- batch size 2, gradient accumulation 4;
- BF16 when supported, otherwise FP16;
- 20 training steps over a tiny manifest;
- no `torch.compile` in the first smoke run.

Success means finite loss, non-zero online gradients, zero target gradients, changing EMA weights, and a resumable checkpoint.

### 4 x A100 80 GB full training

- DDP, one process per GPU;
- `Tc=4`, configurable `K`;
- 12 encoder blocks and 2 predictor blocks;
- BF16;
- effective batch size recorded explicitly;
- distributed sampler with deterministic epoch seeding;
- validation and checkpoint writes on rank 0 only.

## 9. Testing Strategy

All production behavior is introduced test-first.

### Unit tests (CPU)

- dataset preserves sequence boundaries and returns exact context/target timestamps;
- malformed filenames and token shapes raise actionable errors;
- time embedding is deterministic, finite, and shape-correct;
- model output is `[B, K, N, D]` for variable `Tc` and `K`;
- loss is near zero when prediction equals target and increases when direction differs;
- target parameters never receive gradients;
- EMA update matches the analytical weighted average;
- checkpoint round-trip restores outputs and step;
- downstream export either strict-loads or raises a compatibility error.

### Integration smoke test

Generate temporary `.pt` token sequences, run two optimizer steps on CPU or one available GPU, save, resume, and run one further step. This test must not require DSEC or N-ImageNet.

### Real-data smoke test

Run 20 steps on a user-provided token root, then perform a short linear-probe sanity run. This is an execution check, not evidence for the paper.

## 10. Experimental Acceptance Criteria

Engineering acceptance:

- all CPU tests pass without warnings;
- the synthetic resume smoke test passes;
- 5070 Ti real-data smoke finishes within 16 GB VRAM;
- the exported online encoder loads through the downstream adapter;
- training can resume deterministically from a saved checkpoint.

Research go/no-go criterion for expanding to multi-horizon and rate-invariant losses:

- Event-JEPA exceeds the matched GEP Stage-2 baseline in at least one representation-sensitive setting (linear probe or 10% fine-tuning) without materially degrading full fine-tuning; and
- gains reproduce across at least two seeds before adding new losses.

## 11. Explicit Non-Goals for V1

- raw event-by-event asynchronous tokenization;
- end-to-end updating of the Stage-1 DINO event encoder;
- image-token or RGB conditioning;
- rate-invariant and motion-aware auxiliary losses;
- DSEC segmentation/depth or Ev-3DOD integration;
- claiming continuous arbitrary-time interpolation from frame-offset supervision;
- claiming Event-JEPA is the first event-camera pretraining method.

These are follow-up phases, not hidden requirements of the MVP.

## 12. Risks and Controls

- **Timestamp cadence varies.** Use measured `delta_t`, log its distribution, and avoid pretending frame offsets are fixed milliseconds.
- **Token files may encode incompatible shapes.** Validate every sequence once while building the manifest and fail with the offending path.
- **EMA collapse.** Monitor representation standard deviation and cosine similarity; do not add anti-collapse losses until collapse is observed.
- **Downstream incompatibility.** Keep export as an explicit, tested adapter rather than coupling the core model to `Cls` internals.
- **Unfair GEP comparison.** Match Stage-1 tokens, data split, optimizer budget, embedding size, and downstream protocol; report parameter and compute differences.

## 13. Deliverables After Approval

1. A detailed TDD implementation plan.
2. Tested Event-JEPA source modules and configurations.
3. Synthetic smoke-test fixtures and commands.
4. A patch/bundle ready to apply to the pinned GEP commit.
5. Separate commands for the 5070 Ti smoke run and 4 x A100 training.
