# Qwen-only runtime surface cleanup pending remote verification

## Goal

- Regression-check the runtime/model-surface cleanup that removes a retired
  model family from the CUDA runtime, CLI discovery, and user-facing docs
  while keeping the supported Qwen serving paths behaviorally unchanged.

## Hypothesis

- No measurable throughput or latency change on supported Qwen models, because
  the cleanup removes dead model wiring and discovery surfaces rather than
  changing the active Qwen hot path.

## Params

- Label: `qwen-runtime-surface-cleanup`
- Planned command: `scripts/bench_guidellm.sh qwen-runtime-surface-cleanup`
- Planned regression-check target: `Qwen/Qwen3-4B` on the canonical CUDA bench
  host
- Feature set: pending remote host selection

## Env

- Local change date: `2026-04-23`
- Remote benchmark pending because this workspace is not the canonical CUDA
  bench host

## Results

- Status: `pending-remote`
- Local verification covers:
  - `cargo check -p infer --no-default-features --features cuda,no-cuda`
  - `cargo test -p infer --release --no-default-features --features no-cuda`
  - `cargo test -p cli --release --no-default-features --features no-cuda`
  - `cargo clippy -p cli --release --no-default-features --features no-cuda -- -D warnings`

## Problems

- No remote `guidellm` run has been executed yet for this runtime cleanup, so
  throughput and latency regression-check data are still pending.
- `cargo clippy -p infer --no-default-features --features cuda,no-cuda -- -D warnings`
  is currently blocked by pre-existing repository lint failures in untouched
  files outside this cleanup slice.

## Learnings

- Model-family retirement in this codebase is not a single-directory delete:
  runtime dispatch, CLI discovery, generated site content, historical plan
  docs, and benchmark templates all carry references that must be removed
  together or dead paths and stale operator cues remain.
