# GRPO exact checkpoint/resume landed on the active Qwen-family RL path

## Context

`train_grpo` had already moved its SFT warm-up onto the shared trainer
surfaces, but the GRPO phase itself was still hand-written and had no durable
checkpoint story. That left the active RL path in an awkward split: LoRA-only
training was real, but an interrupted GRPO run could not be resumed exactly,
and recreating the run would silently change the frozen `ref_model` baseline.

## What Worked

- Added GRPO-side checkpoint artifacts for the active Qwen-family path:
  - merged inference weights in `model.safetensors`
  - current train weights in `train_model.safetensors`
  - frozen reference-model weights in `reference_model.safetensors`
  - current and reference adapter snapshots
  - `trainer_state.json` + `optimizer.safetensors`
- Added `--save-path`, `--save-every`, and `--resume-from` to `train_grpo`.
- Resume now restores:
  - live policy weights
  - frozen reference-model weights
  - adapter snapshots
  - AdamW moments and step counter
- Switched GRPO rollout seeding to a stateless `(seed, iter)` derivation so a
  resumed run replays the same prompt/sample stream from the saved iteration
  onward.
- Tightened the shared trainable-name helper to be adapter-aware, so LoRA-only
  optimizer state can be keyed on adapter tensor names instead of synthetic
  placeholders.

## Verification

- `cargo test -p train --release --bin train_grpo -- --nocapture`
- `cargo test -p train --release`
- `cargo clippy -p train --all-targets -- -D warnings`

## Rule

If an RL path keeps a frozen reference model outside the shared `Trainer`,
exact resume has to persist both sides of the KL contract: the current policy
state and the frozen reference baseline. Restoring only the live policy is not
an exact resume; it changes the algorithm.
