# DSv4 FlashMLA decode topk_length uses device fill

## Context

Goal: continue DSv4 CUDA Graph readiness by removing host-driven decode
metadata operations that live inside the attention path. FlashMLA sparse-FP8
decode reused a stable one-i32 `topk_length` arena, but still stamped its value
each decode step with `memcpy_htod(&[topk_unified])`. That H2D copy is not part
of a capture-safe kernel-only body.

## What Worked

- Replaced the FlashMLA decode `topk_length` H2D copy with the existing
  `dsv4_fill_i32_cuda` stream kernel.
- Kept the same stable `fm_decode_topk_length` arena and the same pointer passed
  to `arle_flashmla_sm90_sparse_decode_sched_meta` and
  `arle_flashmla_sm90_sparse_decode_fwd`.
- Added an explicit i32 bounds check before writing `topk_unified`.
- Did not add a new CUDA kernel or change the FlashMLA ABI.

## Verification

- `cargo fmt --check`
- `cargo check -p infer --no-default-features --features no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,no-cuda`
- Remote pod worktree `/tmp/arle-dsv4-topk-fill-5bd28386`, HEAD
  `5bd283863dd59ef36248961effbca3387beb233f`.
- Remote `git diff --check -- infer/src/model/deepseek/weights.rs docs/experience/wins/2026-06-02-dsv4-flashmla-topk-device-fill.md`
- Remote `cargo +stable check -p infer --no-default-features --features no-cuda --offline`
- Remote `CUDARC_CUDA_VERSION=12080 cargo +stable check -p infer --no-default-features --features cuda,no-cuda --offline`

No runtime benchmark or TPOT claim is made from this buildability tranche.

## Pending Graph Enablement

This removes one decode H2D site in the FlashMLA metadata path, but the DSv4
CUDA Graph gate remains closed. `start_pos`, compressor counters, FP8 pack
high-water marks, FlashMLA sched metadata, and TP/EP collectives still need
full replay-safe handling.

## Rule

Prefer replacing host metadata copies in the decode body with device-side
stream work, but do not claim graph support until the whole captured body is
allocation-free, H2D-free, and dynamic-launch-parameter safe.
