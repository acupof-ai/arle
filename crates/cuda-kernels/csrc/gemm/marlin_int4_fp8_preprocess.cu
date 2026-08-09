// INT4 weight preprocessing for W4+FP8 marlin GEMM (without_zp variant).
//
// Port of vLLM's marlin_int4_fp8_preprocess.cu `_without_zp` (Apache 2.0).
// Merges zero-point=8 into INT4 weights at prep time so the runtime GEMM
// skips the per-element zero-point subtract.
//
// AWQ variant omitted: ARLE has no AWQ checkpoint loader.
//
// Source: https://github.com/vllm-project/vllm/blob/main/csrc/quantization/marlin/marlin_int4_fp8_preprocess.cu

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

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
    // Merge zero-point=8: values >= 8 shift down, < 8 mirror (15 - v).
    // Matches upstream W4+FP8 sign-extended INT4 expectation.
    single_val = single_val >= 8 ? single_val - 8 : 15 - single_val;
    new_val |= single_val << (i * 4);
    val >>= 4;
  }

  output[blockIdx.x * 32 + threadIdx.x] = new_val;
}

extern "C" cudaError_t marlin_int4_fp8_preprocess_without_zp_cuda(
    const int32_t* qweight,
    int32_t* output,
    int32_t numel,    // INT32 element count (8 INT4 per INT32)
    cudaStream_t stream) {
  if (numel <= 0) {
    return cudaSuccess;
  }
  // Grid: each block processes 32 INT32 elements (256 INT4 weights).
  // Upstream requires numel * 8 % 256 == 0  <=>  numel % 32 == 0.
  if (numel % 32 != 0) {
    // Caller must ensure alignment; bail to surface the error.
    return cudaErrorInvalidValue;
  }
  int32_t blocks = numel / 32;
  marlin_int4_fp8_preprocess_kernel_without_zp<<<blocks, 32, 0, stream>>>(
      qweight, output);
  return cudaGetLastError();
}
