# FP8 probe numeric gate: 4 iterations to fix a 1-line problem

## Context

Operator evidence P1 — FP8 small-M GEMM probe needed to pass numeric gate
(21 cells: 3 shapes × 7 M values) before we could trust the dispatch policy.

## Root Cause

GEMV reference and DeepGEMM candidate used different activation precision
floors. 4 bugs in the host-side FP8 E4M3 encode/decode:

| # | Bug | Effect |
|---|-----|--------|
| 1 | `fp8_e4m3_to_f32`: exp=15 mant<7 → 448.0 | All [256, 448) values collapsed to 448 |
| 2 | `f32_to_fp8_e4m3`: `exp_rebiased >= 15` clip | Killed representable values in [256, 448) |
| 3 | `f32_to_fp8_e4m3`: `exp_final >= 15` clip | Same for rounded-to-exp=15 results |
| 4 | `(x + 0.5) as u8` rounding | Round-half-up ≠ device `__nv_fp8_e4m3` (banker's) |

Iterations wasted because each fix addressed one symptom without reading the
DeepGEMM `pack_quantize` kernel to confirm: per-row scale, per-128-col block,
real E4M3 grid, round-to-nearest-even.

## Fix

Proper `fp8_e4m3_to_f32` / `f32_to_fp8_e4m3` with `rint_ne` rounding.
21/21 component cells pass on H20. This proves numeric parity only. The prior
runner fabricated E2E and bundle identity, so no exact cell qualifies and the
M=1 GEMV / M>=2 PackDeepGemm policy remains an unqualified fallback.

## Rule

**Before writing a numeric reference, read the device kernel's exact
quantization behavior.** Not "FP8 means 8-bit floating point" — read the
specific encode: bias, max-finite, rounding mode, per-row vs per-tile scaling.
A reference that uses integer rounding is structurally guaranteed to fail
against real E4M3 hardware.

**A component probe is not an E2E artifact.** Exact policy qualification needs
the verified kernel manifest and an actual-model E2E result bound to the same
binary and bundle.

**80/20:** 4 iterations × ~30 min each = 2 hrs wasted on something that
reading one kernel function (5 min) would have prevented.
