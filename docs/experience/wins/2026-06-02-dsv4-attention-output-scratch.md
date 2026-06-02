# DSv4 attention decode writes into cache-owned output scratch

## Context

Goal: continue the DSv4 CUDA Graph readiness path without flipping the graph
capability bit. The previous scratch tranche stabilized batched-decode row
buffers and CSA/FlashMLA metadata, but incremental attention still returned a
fresh `HiddenStates` for the final attention output before `hc_post_to_stream`.
That owned allocation is on the per-token decode path and is not graph-replay
friendly.

## What Worked

- Added `DeepseekAttentionRuntimeCache::output_out` as a cache-owned attention
  output scratch.
- Changed `forward_attention_half_incremental_into` to take the output scratch,
  call the incremental attention `_into` path, then feed the same buffer into
  `hc_post_to_stream_into`.
- Added `_into` forms for the DSv4 incremental attention path:
  `forward_sliding_window_attention_incremental_into`,
  `forward_attention_gpu_into`, `forward_attention_gpu_cached_into`,
  `finish_attention_gpu_into`, and `forward_swa_attention_gpu_into`.
- Kept the old owned-return wrappers only where non-incremental/pre-cached
  callers still need them. The incremental owned-return wrapper was removed
  after `rg` showed no caller.
- Reused cache-owned decode scratch for SWA `q_prepared`, `k_prepared`,
  `local_attn`, and output latent buffers when `token_count == 1`.
- Routed `compress_ratio == 0` through `forward_swa_attention_gpu_into`, so SWA
  decode writes the final `wo_b` projection directly into the caller-provided
  output buffer.

## Verification

- `cargo fmt --check`
- `cargo check -p infer --no-default-features --features no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,no-cuda`
- `git diff --check -- infer/src/model/deepseek/state.rs infer/src/model/deepseek/weights.rs`
- Remote pod worktree `/tmp/arle-dsv4-attn-output-0ef119f1`, HEAD
  `0ef119f1900989cfa195aac6f83445b4c0e393cc`.
- Remote `git diff --check -- infer/src/model/deepseek/state.rs infer/src/model/deepseek/weights.rs docs/experience/wins/2026-06-02-dsv4-attention-output-scratch.md`
- Remote `cargo +stable check -p infer --no-default-features --features no-cuda --offline`
- Remote `CUDARC_CUDA_VERSION=12080 cargo +stable check -p infer --no-default-features --features cuda,no-cuda --offline`

The remote pod's direct `1.95.0` shim attempted a rustup channel sync and timed
out, so validation used the already-installed `stable` toolchain
(`cargo 1.92.0`, `rustc 1.92.0`) in offline mode. No runtime benchmark or TPOT
claim is made from this buildability tranche.

## Pending Graph Enablement

The DSv4 CUDA Graph capability gate remains closed. This change removes another
per-token allocation layer, but graph replay still needs device/update-safe
metadata for `start_pos`, compressed-row counters, FP8 pack high-water marks,
FlashMLA decode scheduling, and TP/EP collectives.

## Rule

CUDA Graph enablement must be built as replay-safe decode plumbing first. A
successful local typecheck does not imply graph support or a performance win.
