# Quant-linear dispatch consolidation — remote gates

> Status: ACCEPT — all remote gates pass (H20, 2026-08-20).

## Context

Tranche 1 of `docs/plans/2026-08-20-qwen-quant-linear-dispatch-consolidation.md`:
`crates/infer-cuda/src/ops/quant_linear.rs` (1,829 lines) split into four
modules — entry dispatch (481) + `quant_linear_fp8.rs` (1,021) +
`quant_linear_fp4.rs` (224) + `quant_linear_int.rs` (349). `gemm_batch`/`gemv`
are adapters over one `run(ctx, weight, input, output, m)`; the duplicated
fallback matches and the disabled W4A16 dequant arm are deleted. Structural
only: no kernel math, CUDA argument, counter, or route-order change intended.

## Root Cause

`gemm_batch` and `gemv` each carried a parallel quantized fallback match — the
defect class behind past FP4/FP8/W8A16 route bugs. Fix: one dispatcher, one
ordered route owner per family. CUDA equivalence gates require the remote box.

## Fix

Landed in `b7432e52a`. Local gates green: fmt, diff --check, cuda typecheck
zero warnings, table-driven host route test. Deviations from the brief, both
behavior-neutral: `#[path]` without `ops/` prefix (resolves relative to the
declaring file); catch-all error strings unified to one
`"quant_linear unsupported resident quant weight format {f}"`, FP4 no-route
bails `"fp4_e2m1_group has no consumable representation"`.

Remote gates (H20, baseline `10df1d079` vs `b7432e52a`, Qwen3.8-27B-NVFP4):

- Numerical — PASS. `marlin_fp8_parity` / `marlin_w8a16_parity` ALL PASS
  (relL2 1.65e-3 / 2.85e-3, ratio 1.00), `marlin_fp4_probe` exit 0.
- Engagement — PASS. 2,612-token request, temp 0: all five route counters
  match (`cuda.fp4.widen_fp8_deepgemm` 224, `fp8_per_channel_deepgemm` 288,
  `fp4.marlin_tensorcore` 336, `fp8_marlin_tensorcore` 437, `fp8_gemv`
  ABSENT); completion text identical.
- Capture — PASS. Decode graph armed, slot 0 captured, no eager fallback, no
  per-step alloc/readback across 16 decode steps (BF16 KV).
- Model gates — PASS (NVFP4 + FP8). Needle ladder 12/12 exact at
  512/4096/16384/32768 ×3, both families; lever gate PASS (summaries=4);
  temp-1.0 arm PASS. W8A16 lm_head arm open: no pod checkpoint has an untied
  quantized lm_head.
- Structural A-B — canonical 32K workload, 32 req/point, max-tokens 214,
  seed 42, fp8 KV, 3 trials/cell:

  | c | A ITL ms | B ITL ms | d ITL | A tok/s | B tok/s | d tok/s | ok A/B |
  |---:|---:|---:|---:|---:|---:|---:|:---:|
  | 4 | 34.67 | 34.42 | -0.7% | 84.69 | 82.80 | -2.2% | 3/3 |
  | 8 | 55.93 | 55.85 | -0.1% | 99.30 | 97.63 | -1.7% | 3/3 |
  | 16 | 99.37 | 99.21 | -0.2% | 105.97 | 109.51 | +3.3% | 3/3 |

  TTFT deltas +4.1% / +4.8% / -9.2% — all inside the ≤10% noise band. No
  regression. c=1: the first baseline arm hit `CUDA_ERROR_OUT_OF_MEMORY` on a
  GPU carrying a ~22 GB foreign resident; the clean-GPU re-run (rebuilt
  `10df1d079`, 3 trials) completed 32/32 each with zero OOM at 41.5/41.6 tok/s
  warm, decode 48.3 tok/s (≈20.7 ms ITL) vs the tranche-1 arm's 42.0 tok/s /
  20.45 ms ITL — within noise. The OOM was environmental.
- Eval harness — PASS. `python -m eval_harness` on the tranche-1 binary:
  VERDICT PASS 3/3 (prefix_reuse, token_reuse, multiturn_concurrent).

Verdict (Tranche 2A): ACCEPT. All five gate classes pass on hardware; the
structural change carries no performance or correctness claim beyond parity.

## Rule

A structural dispatch change on a CUDA-only hot path ships with local gates
green and this dated entry; acceptance requires the numerical, engagement,
capture, and A-B gates on hardware.
