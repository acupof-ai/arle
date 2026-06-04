# DSv4 DeepGEMM production FP8-MoE backend — decode parity CLOSED (16/16, TP=8/EP=8)

**Status:** PASS — the native DeepGEMM expert backend now matches the bf16 oracle
exactly (16/16) on canonical DeepSeek-V4-Flash, all 8 ranks.
**Track:** R6 clean-CUDA DSv4 (`crates/infer-cuda`), branch `arch/ideal-inference-engine`.
**SKU:** H20 8×sm_90a, CUDA 12.9, DeepGEMM native (`ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE=1`).

## Context

DSv4 bf16 multi-GPU parity was already closed (3/3 16/16) on the **native-grouped
FP8 bypass** + bf16 KV. The vendored **DeepGEMM** expert backend (the production
FP8 grouped-GEMM path) loaded multi-rank and matched token-1, but its full-sequence
decode diverged: hash prompt 6/16, first flip at index 6. Full-*prefill* under
DeepGEMM was correct — the bug was specific to the **incremental m=1 decode** path.

## What Worked

Layer-bisect via controls (full-prefix-vs-decode, max_m-vs-scale_stride toggles)
isolated a **two-part** root cause, both in `crates/infer-cuda/src/moe.rs`:

1. **Routing/scatter invalid-row sentinel.** The DSv4 DeepGEMM branch initialized
   `packed_route_slot` with **0** (`alloc_zeros`); the scatter kernel treats only
   `route_slot < 0` as invalid. In m=1 decode the unfilled compact rows looked like
   valid slot-0 rows and overwrote route slot 0 with zero output. Fix: initialize to
   **−1** (`clone_htod(&vec![-1i32; ...])`), matching the native path. → index-6
   divergence fixed, matched through index 7.

2. **DeepGEMM small-m grouped-GEMM tile path.** After the sentinel fix the first
   divergence moved to index 8 (m=1 top-k flipped 469=30.75 vs 14=29.375 on
   DeepGEMM, native 14=30.375 vs 469=29.25). Controls: full-prefix correct;
   `scale_stride_m=128` alone did **not** fix it (rules out block-scale indexing);
   forcing DeepGEMM grouped **`max_m >= 128`** restores 16/16. So the small-m
   (m=1 decode) tile path is the culprit; flooring `max_m` at 128 matches the
   working production shape.

**Verified (H20, TP=8/EP=8, hash prompt, DeepGEMM backend):**
```
[260, 1499, 4456, 396, 20685, 411, 96958, 5554, 14, 260, 4456, 396, 588, 6403, 18222, 304]
```
= bf16 oracle exactly, all 8 ranks. FP8 MoE is now verified on **both** the
native-grouped bypass AND the production DeepGEMM backend.

## Rule

- **A compact/packed routing buffer needs the kernel's invalid sentinel, not
  `alloc_zeros`.** If the scatter kernel uses `slot < 0` as "invalid", zero-init
  silently makes unfilled rows valid slot-0 writes — invisible at large m (rows all
  filled), corrupting only the m=1 decode tail.
- **m=1 decode is a distinct numeric regime from prefill for grouped FP8 GEMM.** When
  full-prefill is correct but incremental decode flips tight margins, suspect the
  small-m tile/padding path (here `max_m`), and isolate it from block-scale stride
  with a control toggle before crediting either.
