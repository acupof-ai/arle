# Qwen FP8 dense pre-Hopper dequant→BF16 GEMM fallback — verified sm_80

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

## Status: verified on sm_80 (A100)

**Runtime parity verified** on an A100-SXM4 (sm_80, CUDA 12.8) via a kernel-level
A/B test (`fp8_dequant_bf16_gemm_matches_scalar`, `cuda-kernels/src/ffi/gemm_tests.rs`):
the fallback (`dequantize_fp8_block_scaled_to_bf16_cuda` + `gemm_cuda`) vs the
scalar `gemv_fp8_block_scaled_batch_cuda` on identical synthetic FP8 block-scaled
weights (B=1024, N=K=256, 128×128 block):

```
[fp8-dequant-bf16-parity] B=1024 N=256 K=256 block=128x128 cosine=1.000000 max_abs_err=0.0000
test result: ok. 1 passed
```

- **cosine 1.000000 / max_abs 0.0** — exact parity with the scalar reference
  (synthetic weights dequantize exactly; both paths f32-accumulate).
- **bf16 `gemm_cuda` ran without `NOT_SUPPORTED` on sm_80** — the test reached the
  parity print (it `.expect`s the gemm_cuda result, which panics on error). So the
  Ampere+ bf16 cuBLASLt path works; the sm_70/75 hypothesis below does NOT extend
  to sm_80. (Built with `ARLE_CUDA_KERNEL_SET=dsv4_flash` to skip TileLang AOT —
  the test uses only native gemm kernels, no TileLang dependency.)

Also compile/codegen-green cross-SM: Mac typecheck (cuda,no-cuda); sm_70 V100
`BUILD_EXIT=0` (dequant kernel `.target sm_70`); sm_100/120 Blackwell (PR #115).

**Still untested — sm_70/75 (Volta/Turing).** The fallback fires on all sm < 9,
but bf16 tensor-core cuBLAS is Ampere+ (sm_80). On sm_70/75 the explicit
`CUBLAS_GEMM_DEFAULT_TENSOR_OP` may still return `NOT_SUPPORTED`, regressing FP8
prefill there from "slow scalar" to "errors". No Volta/Turing box was available.
**Recommended follow-up:** add a `major >= 8` precondition to
`try_fp8_dequant_bf16_gemm_batch` so it engages only on the verified-safe sm_80–89
range (sm_70/75 keep the scalar GEMV). The intended target (sm_80–89, Ampere/Ada)
is now verified.

**Production impact: none.** On Hopper (H20, sm_90) the SM-gate returns true →
DeepGEMM path; the fallback returns false. The decode (B = 1) path is unchanged
on all SMs. So the production serving path is byte-identical to before.

## Rule

A runtime quant/kernel change whose correctness gate (needle ×3 vs envelope)
cannot run locally lands with a `pending-remote` stub that states exactly what
is compile-verified vs runtime-unverified and names the hardware needed — never
a silent "done". A kernel-level A/B (fallback vs scalar reference on synthetic
weights, cosine + max_abs) verifies a quant-GEMM path with no model download —
faster and tighter than a full needle run when the question is per-op parity.
Closed sm_80; sm_70/75 still open (add the `major >= 8` guard or test a Volta box).
