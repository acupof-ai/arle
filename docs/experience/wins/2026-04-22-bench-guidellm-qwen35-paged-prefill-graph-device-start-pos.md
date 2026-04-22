# Qwen3.5 paged-prefill graph uses device-backed start positions

## Context

Qwen3.5 paged-prefill already ran through the canonical full-forward graph
path, but replay still invalidated when `start_pos` changed. The remaining
cause was narrow: the HD256 paged prefill prep kernel still baked `start_pos`
into the launch parameters, so graph safety depended on a host scalar matching
the captured chunk offset.

## What Worked

- `infer/src/model/qwen35/prefill_buffers.rs` now uploads `start_pos` into
  stable paged-prefill metadata alongside the other per-launch GPU metadata.
- `crates/cuda-kernels/csrc/attention/prefill_attention_paged_prep.cu` and
  `crates/cuda-kernels/src/ffi/attention.rs` now route the HD256 paged-prefill
  prep kernel through a device-backed `start_pos_ptr`.
- `infer/src/model/qwen35/prefill.rs` now treats paged-prefill graph replay as
  shape-based on the canonical path: pointer-changing reallocations still
  invalidate the graph, but chunk-offset changes no longer force recapture.
- Local validation passed:
  - `cargo check -p infer --release --no-default-features --features cuda,no-cuda`
  - `cargo test -p infer --release --no-default-features --features no-cuda qwen35 -- --nocapture`

## Rule

Status: `pending-remote`

- CUDA Graph-safe kernel inputs must live in stable device-backed buffers when
  they vary across requests or chunk offsets.
- Qwen3.5 paged-prefill graph reuse now claims correctness across changing
  `start_pos` values, but any throughput claim still requires a remote CUDA
  `scripts/bench_guidellm.sh` sweep.
