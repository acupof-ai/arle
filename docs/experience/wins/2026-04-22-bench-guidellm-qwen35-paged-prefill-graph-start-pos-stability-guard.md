# Qwen3.5 paged-prefill graph start-pos stability guard

## Context

Commit `5c8aa81` removed the old `start_pos == 0` admission gate for Qwen3.5
paged-prefill graph capture. The canonical shipped path still routes paged
prefill through `forward_prefill_with_pool`, but the captured full-attention
prep kernel continues to take `start_pos` as a launch scalar. Replaying a
captured graph after the chunk offset changes is therefore not safe until that
scalar moves to device memory or the graph exec params are rewritten.

## What Worked

- `infer/src/model/qwen35/prefill_buffers.rs` now tracks the `start_pos` baked
  into the currently captured paged-prefill graph and clears that marker on
  graph invalidation.
- `infer/src/model/qwen35/prefill.rs` now invalidates and recaptures the
  full-forward paged-prefill graph when `start_pos` changes, while keeping the
  canonical paged-prefill path unchanged.
- Local validation passed:
  - `cargo check -p infer --release --no-default-features --features cuda,no-cuda`
  - `cargo test -p infer --release --no-default-features --features no-cuda scheduler -- --nocapture`

## Rule

Status: `pending-remote`

- Qwen3.5 paged prefill remains the canonical shipped path, but graph reuse is
  only claimed for matching captured `start_pos` values until the prep kernel
  becomes parameter-stable across chunk offsets.
- This runtime-facing correction still needs a remote CUDA guidellm regression
  run before any throughput claim is made.
