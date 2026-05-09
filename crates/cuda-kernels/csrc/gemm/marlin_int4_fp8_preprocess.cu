// PF8.2 — INT4 weight preprocessing for W4+FP8 marlin GEMM (without_zp variant only)
//
// Verbatim port of vLLM's `csrc/quantization/marlin/marlin_int4_fp8_preprocess.cu`
// `_without_zp` kernel under Apache 2.0 license. Adapts torch FFI →
// extern "C" cudarc convention per ARLE's
// crates/cuda-kernels/src/ffi/gemm.rs pattern.
//
// Purpose: subtraction-merging zero-point=8 into INT4 weight tensor
// at offline weight-prep time, so the runtime W4+FP8 marlin GEMM
// kernel doesn't need per-element zero-point subtract. Saves ~1
// instruction per dequantized element.
//
// AWQ variant SKIPPED for PF8.2 (ARLE has no AWQ checkpoint loader);
// add later if AWQ format support lands.
//
// Used by NEW prefill-only FP8 directive (per
// docs/research/2026-05-10-prefill-only-fp8-directive-draft.md
// Substep PF8.2, ~120 LOC).
//
// License: Apache 2.0 (per vLLM upstream `marlin.cu` attribution)
// Adapted from: https://github.com/vllm-project/vllm/blob/main/csrc/quantization/marlin/marlin_int4_fp8_preprocess.cu

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

// for only non-zp format (like gptq)
__global__ void marlin_int4_fp8_preprocess_kernel_without_zp(
    // qweight: (size_k * size_n / 8,) packed INT4 in INT32
    const int32_t* __restrict__ qweight,
    // output: same shape as qweight, zero-point pre-merged
    int32_t* __restrict__ output) {
  int32_t val = qweight[blockIdx.x * 32 + threadIdx.x];
  int32_t new_val = 0;

#pragma unroll
  for (int32_t i = 0; i < 8; i++) {
    int32_t single_val = val & 0xF;
    // Bake zero-point subtract: single_val ∈ [0,15], target offset = 8.
    // For values ≥ 8: shift down by 8 (keep numeric distance to 0).
    // For values < 8: invert as 15 - single_val (mirror around boundary).
    // Net effect: matches what the upstream W4+FP8 marlin GEMM expects
    // for sign-extended INT4 with zero-point=8.
    single_val = single_val >= 8 ? single_val - 8 : 15 - single_val;
    new_val |= single_val << (i * 4);
    val >>= 4;
  }

  output[blockIdx.x * 32 + threadIdx.x] = new_val;
}

extern "C" cudaError_t marlin_int4_fp8_preprocess_without_zp_cuda(
    const int32_t* qweight,
    int32_t* output,
    int32_t numel,    // qweight.numel() (INT32 element count, packs 8 INT4 each)
    cudaStream_t stream) {
  if (numel <= 0) {
    return cudaSuccess;
  }
  // Grid: each block processes 32 INT32 elements (256 INT4 weights).
  // Per upstream check: numel * 8 % 256 == 0  ⇔  numel % 32 == 0.
  if (numel % 32 != 0) {
    // Caller should ensure alignment; bail to surface error.
    return cudaErrorInvalidValue;
  }
  int32_t blocks = numel / 32;
  marlin_int4_fp8_preprocess_kernel_without_zp<<<blocks, 32, 0, stream>>>(
      qweight, output);
  return cudaGetLastError();
}
