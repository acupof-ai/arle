# V100 BF16 GEMM — raw byte copy corrupts every operand

## Context

W4A16 (INT4) MoE grouped GEMV validation on V100 (sm_70). While running
BF16 end-to-end serve on Qwen3.5-0.8B to confirm the sm_70 build path,
greedy decode produced garbage text (乱码) — "2+2?" returned nonsensical
tokens instead of "Four".

## Root Cause

`gemm_fp16_cast_cuda` (the sm_70 BF16 GEMM fallback in
`crates/cuda-kernels/csrc/gemm/gemv.cu`) "converted" BF16↔FP16 with raw
`cudaMemcpyAsync`. That reinterprets bits instead of converting values:
BF16 and FP16 have different bit layouts (BF16 1+8+7, FP16 1+5+10;
exponent range differs by 3 bits). BF16 1.0 (0x3F80) reads as FP16
1.875 — every GEMM operand (W, X, Y) was corrupted, so the whole network
computed garbage.

The bug is invisible on sm_80+ (the hot path uses native BF16
`cublasGemmEx` and never touches the cast fallback), and invisible in a
build-only check — it only manifests at runtime on sm_70 with real BF16
data.

## Fix

Replace the raw `cudaMemcpyAsync` calls with value-conversion kernels
that promote through FP32 with round-to-nearest:

```cuda
__global__ void convert_bf16_to_fp16_kernel(const __nv_bfloat16 *in,
                                            __half *out, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __float2half_rn(__bfloat162float(in[i]));
}
__global__ void convert_fp16_to_bf16_kernel(const __half *in,
                                            __nv_bfloat16 *out, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __float2bfloat16_rn(__half2float(in[i]));
}
```

Both `cudaMemcpyAsync` calls (W and X) plus the Y-copy back now go
through these kernels with element counts (not byte counts). The sm_80+
hot path is untouched — gated on `device_compute_major() <= 7`.

## Verification

- Kernel-correctness test `w4a16_grouped_gemv_matches_dequantized_bf16`
  PASS on V100 sm_70 (the W4A16 path is BF16-storage/FP32-accumulate and
  never relied on the broken cast).
- BF16 end-to-end serve on V100: "2+2?"→"Four",
  "capital of France"→"Paris" — correct greedy output, not garbage.
- Commit `b94e2fc44`, pushed to `origin/main`.

## Rule

- A dtype "conversion" that copies raw bytes is a reinterpretation, not a
  conversion — BF16/FP16 share 16 bits but disagree on every value.
  Convert through FP32 with `__bfloat162float`/`__half2float` +
  round-to-nearest.
- sm_70 (no BF16 tensor cores) is the only platform that exercises the
  cast fallback — validate BF16 serve on real sm_70 hardware, not just
  sm_80+ where the fallback is dead code.
- Case-as-fact: garbage greedy output on a known-trivial prompt ("2+2?")
  is the signal to debug the GEMM path, not to blame the model or the
  sampling.
