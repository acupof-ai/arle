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

Remote gate results (H20, 2026-08-20):

- Numerical harnesses — PASS. Baseline binary archived at
  `/host/baselines/arle-pre-tranche1-5853d306…` (pre-refactor `10df1d079`,
  kernel-build-id verified); new binary built from `b7432e52a` (2m08s, no
  warnings). `marlin_fp8_parity` ALL PASS (relL2 ≈ 1.65e-3, ratio 1.00 on
  every shape, declined n%64 cases ≈ 1.6e-3); `marlin_w8a16_parity` ALL PASS
  (relL2 ≈ 2.85e-3, ratio 1.00, declined cases under the 1.15e-2 cap);
  `marlin_fp4_probe` exit 0 (n%64 load bail OK, fp4/fp8 byte-rate 0.48–0.55,
  m=1…2048). Logs: `/host/harness-{fp8-parity,w8a16-parity,fp4-probe}.log`.
- Engagement — PASS. One request (prompt 2,612 tokens, max-tokens 16,
  temperature 0, GPU 0, sequential) against the archived baseline
  (`/host/baselines/arle-pre-tranche1-5853d306…`) and the real tranche-1
  binary `/host/arle-tranche1b` (sha256 `6a5cd23a…`): all five route counters
  match exactly — `cuda.fp4.widen_fp8_deepgemm` 224,
  `cuda.qwen.fp8_per_channel_deepgemm` 288, `cuda.fp4.marlin_tensorcore` 336,
  `cuda.qwen.fp8_marlin_tensorcore` 437, `cuda.qwen.fp8_gemv` ABSENT — and the
  completion text is identical. Note: `/host/arle-build/target/release/arle`
  was stale (byte-identical to the baseline; the receipt guard blocked the
  install after the 2m08s cargo build) — the tranche1b rebuild is the first
  real A/B. KV dtype was BF16 (the default); FP8 KV disables the decode graph,
  so the capture gate requires BF16. The counters are KV-dtype-independent and
  match the documented regime.
- Capture — PASS. `tranche1b` serve log: decode graph armed, paged slot 0
  captured, no eager fallback, no per-step allocation/readback warning across
  the 16 decode steps.
- Model gates / structural A-B — pending.

Tranche 2A records the verdict against this entry and adds the CHANGELOG line.

## Rule

A structural dispatch change on a CUDA-only hot path ships with local type and
host-route gates green and a dated pending-remote entry; it is not accepted
until the numerical, engagement, capture, and A-B gates above pass on hardware.
