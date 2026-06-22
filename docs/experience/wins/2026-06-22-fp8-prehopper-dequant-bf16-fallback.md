# Qwen FP8 dense pre-Hopper dequant→BF16 GEMM fallback — pending-remote

## Context

DeepGEMM's dense FP8 GEMM is Hopper-only (`wgmma`, sm_90a); on pre-Hopper GPUs
the native kernel hard-checks `prop.major == 9` and returns
`CUDA_ERROR_NOT_SUPPORTED`. This change lets the Qwen `Fp8BlockScaled` dense
path run on sm < 9 without the slow scalar GEMV at large M:

- `dequantize_fp8_block_scaled_to_bf16_kernel` (`csrc/gemm/quantized_gemv.cu`):
  one thread per `(row, col)`, FP8 E4M3 + f32 block-scale → BF16 (no arch
  guard, sm_70+). FFI decl in `cuda-kernels/src/ffi/gemm.rs`.
- `qwen_fp8_dense_sm_supports_deepgemm` (`quant_linear.rs`): `OnceLock`-cached
  compute-cap major gate, so the DeepGEMM dispatch SM-gates without a per-step
  `cuDeviceGetAttribute` (the known per-step device-property perf bug).
- `try_fp8_dequant_bf16_gemm_batch`: on sm < 9 + `M >= 1024` (prefill) +
  canonical 128×128 block, dequant the weight once into a monotonically-growing
  resident BF16 scratch → reuse the BF16 cuBLAS GEMM (`gemm_cuda`). Decode
  (B = 1) keeps the scalar block-scaled GEMV (also sm_70+).

## Status: pending-remote

**Verified locally:** Mac CUDA typecheck clean
(`cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib`,
exit 0). Cross-SM **compile/codegen** of the surrounding vendored kernels is
green (sm_70 V100 `BUILD_EXIT=0` with this dequant kernel `.target sm_70`
confirmed; sm_100/120 Blackwell codegen+compile clean — PR #115).

**NOT yet verified (owed):**
- **Runtime correctness gate not run.** No `scripts/needle_gate.py` ×3 vs the
  scalar-GEMV envelope on any pre-Hopper box. The dequant→BF16 path is
  source-plausible (same FP8 decode + a standard cuBLAS GEMM) but unproven at
  runtime.
- **sm_70/75 bf16 risk (hypothesis, source-level).** `gemm_cuda` →
  `cublasGemmEx(CUDA_R_16BF, …, CUBLAS_GEMM_DEFAULT_TENSOR_OP)`. bf16 tensor-core
  GEMM is Ampere+ (sm_80); on sm_70 (Volta) / sm_75 (Turing) the explicit
  `TENSOR_OP` request may return `NOT_SUPPORTED`, regressing pre-Hopper FP8
  *prefill* from "slow but works (scalar)" to "errors". Verify on sm_80–89 (the
  intended target) before trusting; consider an `major >= 8` precondition on the
  fallback if the sm_70/75 failure reproduces.

**Production impact: none.** On Hopper (H20, sm_90) the SM-gate returns true →
DeepGEMM path; the fallback returns false. The decode (B = 1) path is unchanged
on all SMs. So the production serving path is byte-identical to before.

## Rule

A runtime quant/kernel change whose correctness gate (needle ×3 vs envelope)
cannot run locally lands with a `pending-remote` stub that states exactly what
is compile-verified vs runtime-unverified and names the hardware needed — never
a silent "done". Next: needle gate on an sm_80–89 box.
