# Generic Qwen-family pretrain entrypoint — Qwen3.5 default, resume-aware

## Context

`crates/train/src/bin/pretrain_qwen3.rs` used to be a Qwen3-only training
binary with a separate resume path that rebuilt the model weights and then
silently reset AdamW state on restart. The train crate already had a shared
trainer/checkpoint surface, so the missing piece was to make the pretrain
entrypoint family-generic without giving up the current HF-style checkpoint
layout.

This change keeps the architecture generic and makes Qwen3.5 the optimized
default model family for the pretrain binary.

## What Worked

- Added model-family dispatch in the pretrain binary for `Qwen3` and
  `Qwen3.5`. `--model-family auto` now resolves to `Qwen3.5`, while explicit
  `qwen3` still routes to the legacy Qwen3 config/model path.
- Wired resume through the shared trainer checkpoint surface via
  `Trainer::resume_if_configured`, so optimizer state and step count restore
  from the checkpoint dir instead of being reset on resume.
- Preserved the existing HF-style step directory outputs:
  `config.json`, `generation_config.json`, `model.safetensors`, and
  `tokenizer.json`.
- Kept Qwen3.5 on the active dense/full-attn path only. The binary does not
  invent a hybrid training path; it uses the dense/full-attn family contract
  until the hybrid train path lands.
- Added tests that pin:
  - `ModelFamily::Auto` resolving to Qwen3.5
  - Qwen3.5 dense/full-attn config construction
  - checkpoint round-trip for model weights
  - trainer resume restoring AdamW state rather than reinitializing it

## Verification

- `cargo fmt --all`
- `cargo check -p train --release`
- `cargo test -p train --release --bin pretrain_qwen3 -- --nocapture`
- `cargo test -p train --release`
- `cargo clippy -p train --all-targets -- -D warnings`

## Rule

When a train entrypoint is family-specific only because of wiring, move the
dispatch and checkpoint logic into shared runtime surfaces first. Keep the
default on the optimized family, and only specialize the active path when a
real model-family implementation requires it.
