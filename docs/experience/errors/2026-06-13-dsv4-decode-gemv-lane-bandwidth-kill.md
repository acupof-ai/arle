# DSv4 decode-band grouped-GEMV MoE lane: KILLED by scalar dequant bandwidth (34.9 < 36.5 baseline)

## Context

Regression-campaign endgame: after the 64-align fix recovered 33.5 → 36.5
tok/s B=1, the probe ladder pinned the remaining +2.8ms/tok vs era as the
residual MoE padding tax (576 pad rows × pack/swiglu/scatter/GEMM, visible
as 6.5ms GPU backlog at the step tail vs era's 1.2ms). The GEMV lane aimed
to zero the padding: compact pack → per-expert pointer-table pair-GEMV
(w8a16) → SwiGLU → w2 GEMV, on the existing `quantized_gemv.cu` grouped
kernels. Two boots, full gates.

## Root Cause (two findings)

1. **MoE expert scales are arbitrary f32, not UE8M0.** First boot: all 344
   (43 layers × 8 ranks) table builds refused — `scale[0] = 0.0002092634`,
   full mantissa. The raw-UE8M0-bytes path (`dsv4_scales`) exists for
   attention tensors only; the DeepGEMM expert caches carry true-f32 block
   scales. Fixed by `_f32s` kernel variants (scales read as f32 directly) —
   lane then activated (0 warns).
2. **Scalar dequant GEMV loses to DeepGEMM-on-pad-rows.** With the lane
   active: B=1 34.98 tok/s vs 36.5 contiguous baseline; tail backlog grew
   6.3 → 7.3ms. Per active expert the kernels stream w13+w2 ≈ 24MB FP8 with
   per-thread 1-byte loads ≈ 25% HBM bandwidth — slower than DeepGEMM's
   TMA/warpgroup pipeline grinding 9× the rows. The `DSV4_BATCH_TILE` tiled
   variants don't help at B=1 (`tile_M = 1` ⇒ no M-reuse, same scalar loads).
   Needle 512 also flipped exact→partial ×1 (within the MoE floor class,
   but a flip the contiguous lane didn't show this config).

## Fix

Lane flipped to opt-in (`ARLE_DSV4_MOE_DECODE_GEMV=1`); the contiguous
64-aligned lane stays the default at 36.5. Code retained deliberately: the
pointer-table/compact-pack scaffolding and `_f32s` kernels are the substrate
for the real fix — a decode-band expert pipe with DeepGEMM-grade streaming:
either vectorized (uint4 loads) / MMA grouped GEMV, or the vendored SGLang
`fused_moe` lane (already in-tree for Qwen3.6, #88 U3) extended to DSv4 FP8
— per the kernels-align-to-SGLang rule, the latter is the canonical move.

## Rule

- **A dequant-GEMV lane is licensed by achieved bandwidth, not by row
  count.** Zero pad rows at 25% BW loses to 9× pad rows at 80% BW; compute
  the bytes×BW budget for BOTH sides before wiring (the per-lane bytes were
  budgeted; the achievable-bandwidth asymmetry was not).
- **Check the actual scale encoding of the exact cache you read** — one
  checkpoint ships UE8M0 for attention and true-f32 for experts; "DSv4
  scales are UE8M0" was a tensor-class fact misapplied model-wide.
