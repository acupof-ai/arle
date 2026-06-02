# DSv4 FlashMLA decode caches shape-only host meta

## Context

Goal: continue shrinking DSv4 FlashMLA decode metadata work toward a CUDA
Graph-capturable body. The sparse-FP8 decode path called
`arle_flashmla_sm90_sparse_decode_get_meta` every decode step even though, for
ARLE's MODEL1 decode branch, the result only depends on `local_heads`, `s_q=1`,
and `model_type`.

## What Worked

- Added `DeepseekFlashMlaDecodeMeta` to `DeepseekAttentionRuntimeCache`.
- Added `ensure_fm_decode_meta`, keyed by `(local_heads, model_type)`.
- Changed the FlashMLA decode branch to call the host shim only when the cached
  shape meta is missing or stale.
- Kept the DSv4 CUDA Graph capability gate closed; this only removes one
  per-token host metadata call.

## Verification

- `cargo fmt --check`
- `cargo check -p infer --no-default-features --features no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,no-cuda`
- Remote pod worktree `/tmp/arle-dsv4-flashmla-meta-df1102e1`, HEAD
  `df1102e1e350cf3c54b3a0a1d7d98fd45b94028e`.
- Remote `git diff --check -- infer/src/model/deepseek/state.rs infer/src/model/deepseek/weights.rs docs/experience/wins/2026-06-02-dsv4-flashmla-meta-cache.md`
- Remote `cargo +stable check -p infer --no-default-features --features no-cuda --offline`
- Remote `CUDARC_CUDA_VERSION=12080 cargo +stable check -p infer --no-default-features --features cuda,no-cuda --offline`

No runtime benchmark or TPOT claim is made from this buildability tranche.

## Pending Graph Enablement

FlashMLA decode still builds indices and sched metadata per token, and DSv4
still has host scalar `start_pos`, compressor counters, FP8 high-water marks,
and TP/EP collectives outside a proven graph-safe contract.

## Rule

Cache shape-only host metadata before attempting graph capture. Do not leave
per-token host shim calls in a path that is supposed to become a kernel-only
capture body.
