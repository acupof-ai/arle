# Qwen3.5 multi-turn RL checkpoints now resume exactly

## Context

`train_multi_turn` had already moved onto the active dense/full-attn
Qwen3.5-family path and could export merged inference weights plus PEFT-style
adapter files, but it still could not resume a run exactly. The remaining gap
was that the RL loop was hand-written, stateful RNGs were advanced in-process,
and no trainer-side AdamW state was persisted next to the model artifacts.

## What Worked

- Made the train-side parameter-name helper adapter-aware so LoRA-only runs can
  key optimizer state on adapter tensor names instead of silently exporting an
  empty map.
- Extended `train_multi_turn` checkpoints to write both:
  - merged inference weights in `model.safetensors`
  - exact train-resume weights in `train_model.safetensors`
  - PEFT-style `adapter_model.safetensors` + `adapter_config.json`
  - `trainer_state.json` + `optimizer.safetensors`
- Added `--resume-from` to `train_multi_turn` and wired exact resume through:
  - config validation
  - adapter-config validation
  - unmerged train-weight reload
  - adapter reload
  - AdamW moment restore
- Switched multi-turn rollout seeding to deterministic per-iter/per-episode
  derivation so resuming from step `N` reproduces the same prompt/sample stream
  for steps `N..`.
- Added tests that pin both the expanded checkpoint layout and exact
  resume-state restoration.

## Verification

- `cargo test -p train --release --bin train_multi_turn --test test_multi_turn -- --nocapture`
- `cargo test -p train --release`
- `cargo clippy -p train --all-targets -- -D warnings`

## Rule

For hand-written RL loops that stay outside the shared `Trainer`, checkpointing
still has to round-trip the same three layers as the supervised path: model
weights, adapter weights, and optimizer state. If RNG state is not persisted,
derive it statelessly from `(seed, step, lane)` so resume stays deterministic.
