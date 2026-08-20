# Quant-linear dispatch consolidation — remote gates pending

> Status: Pending-remote. Local gates pass; CUDA execution gates cannot run on
> the macOS dev host.

## Context

Tranche 1 of the Qwen CUDA quant-linear dispatch consolidation
(`docs/plans/2026-08-20-qwen-quant-linear-dispatch-consolidation.md`):
`crates/infer-cuda/src/ops/quant_linear.rs` (1,829 lines) was split into four
modules with one route owner per weight family —
`quant_linear.rs` (entry dispatch, shared Marlin scratch, profiling, stats,
DSv4 legacy arms, route test), `quant_linear_fp8.rs`, `quant_linear_fp4.rs`,
`quant_linear_int.rs`. `gemm_batch` and `gemv` are now three-line adapters over
one `run(ctx, weight, input, output, m)`; the duplicated full-format fallback
matches and the permanently disabled W4A16 dequant arm are deleted. The change
is structural: no kernel math, CUDA argument, counter, or route-order change is
intended.

## Root Cause

The defect class this refactor targets is parallel entry routing: `gemm_batch`
and `gemv` each carried their own quantized fallback match, which already
produced source-after-release route defects (FP4, FP8, W8A16 lm_head). The
structural fix is one dispatcher and one ordered route owner per family; the
fix cannot be validated for numerical and engagement equivalence without CUDA
hardware, which the dev host lacks.

## Fix

Landed in `762244190` (five files):

- `crates/infer-cuda/src/ops/quant_linear.rs` — 481 lines
- `crates/infer-cuda/src/ops/quant_linear_fp8.rs` — 1,021 lines (new)
- `crates/infer-cuda/src/ops/quant_linear_fp4.rs` — 224 lines (new)
- `crates/infer-cuda/src/ops/quant_linear_int.rs` — 349 lines (new)
- this report

Local gates, all green:

- `cargo fmt --check`
- `git diff --check`
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer-cuda --release --no-default-features --features cuda,no-cuda --tests --examples` — zero warnings; the table-driven host route test (FP8/FP4/INT matrices) compiles and is the one new test.

Forced deviations from the tranche brief, both behavior-neutral:

1. Sibling modules are declared `#[path = "quant_linear_fp8.rs"]` (no `ops/`
   prefix). `#[path]` resolves relative to the declaring file's directory, so
   the brief's `ops/quant_linear_fp8.rs` would resolve to
   `src/ops/ops/quant_linear_fp8.rs` and not compile.
2. The two lane-specific catch-all strings
   (`"gemm_batch unsupported resident quant weight format {f}"` /
   `"gemv unsupported ..."`) are unified to
   `"quant_linear unsupported resident quant weight format {f}"`, and FP4's
   no-route terminal state now bails with
   `"fp4_e2m1_group has no consumable representation"` instead of the old
   catch-all. Unification makes one string mandatory; no other error string
   changed.

Pending remote gates (H20, per the plan's test plan):

- Numerical harnesses: `marlin_fp8_parity`, `marlin_w8a16_parity`,
  `marlin_fp4_probe` at the plan's shapes and `M` values, three seeds,
  reference-error metrics within 5% of the accepted implementation.
- Engagement: same implementation-ID route counts as the archived baseline for
  one FP8, one NVFP4, and one W8A16 request; `M=1` parity between
  `gemm_batch` and `gemv`.
- Capture: eager and captured execution, no per-step allocation, no capture
  failure.
- Model gates: `scripts/needle_gate.py temp` at 512/4096/16384/32768 ×3 and
  `scripts/lever_gate.sh` against the same-config baseline envelope, on the
  three checkpoint families.
- Structural A/B: canonical 32K multi-turn workload at c={1,4,8,16}, ≥20
  requests per point, median of ≥3 trials; no unresolved negative sign.

Tranche 2A records the verdict against this entry and adds the CHANGELOG line.

## Rule

A structural dispatch change on a CUDA-only hot path ships with local type and
host-route gates green and a dated pending-remote entry; it is not accepted
until the numerical, engagement, capture, and A-B gates above pass on hardware.
