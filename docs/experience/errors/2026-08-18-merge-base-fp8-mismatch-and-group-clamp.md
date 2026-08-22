# merge_base_fp8 matched FP4 matrices; FP4 decoders clamped the group index — 2026-08-18

> Status: Fixed in `11227da43`. Latent — never reachable in production
> (`promote_lora_target_to_bf16` rejects a non-FP8 format first), found by an
> adversarial review of the 4-bit training design.

## Context

`merge_base_fp8()` returned `self.qweight_u8.zip(self.scale_f32)` with no
format check. An `Fp4E2M1Group` matrix fills both accessors — `qweight_u8`
with packed nibbles (half the byte count of an FP8 weight) and `scale_f32`
with a 1-element global scale — so it matched, and the three callers in
`qwen35_lora.rs` would have read past the end of both buffers.

Separately, all five FP4 decoders clamped the group index
(`if (g > max_group) g = max_group`). NVFP4 requires
`scale_cols == K / group_size`, under which `col / group_size` can never reach
`scale_cols`, so the clamp was dead on every well-formed tensor while costing a
compare per element in the innermost loop — and on a malformed one it silently
decoded against the wrong scale instead of failing.

## Root Cause

The accessor trusted its type precondition to the caller, and the decoders
treated a malformed tensor as a clampable edge case instead of a contract
violation.

## Fix

- `merge_base_fp8` checks the format; the `pristine_fp8` branch needs no check
  (only `requant_merged_matrix` writes it, only on the FP8 path).
- The group relation is checked once in each of the four CUDA launchers and at
  `cuda_upload_fp4_e2m1_group` / `dequantize_fp4_e2m1_group_host`; the per-element
  clamps are gone.

## Rule

A public accessor validates its own type preconditions — "no caller passes the
wrong type today" is not a guard, because the accessor is the boundary that
makes the caller's assumption true. A clamp that only fires on malformed input
is worse than the failure it hides: it turns a contract violation into silent
wrong-scale decode.
