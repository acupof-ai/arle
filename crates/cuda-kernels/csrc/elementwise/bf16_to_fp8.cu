// Simple bf16 -> fp8 (e4m3) per-tensor cast with identity scale (scale=1.0).
// Used to feed the q8kv8 sparse prefill kernel, which expects fp8 Q/KV.
// SGLang uses identity per-tensor scale for DSv4 magnitudes; the fp8 e4m3
// dynamic range ([-448, 448]) covers DSv4's normalized Q/KV values.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

__global__ void arle_bf16_to_fp8_e4m3_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_fp8_e4m3* __restrict__ output,
    int64_t n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  float v = __bfloat162float(input[idx]);
  // Clamp to e4m3 range to avoid Inf/NaN saturation.
  v = fminf(fmaxf(v, -448.0f), 448.0f);
  output[idx].__x = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
}

extern "C" int32_t arle_bf16_to_fp8_e4m3_cuda(
    const void* input,
    void* output,
    int64_t n,
    cudaStream_t stream) {
  if (n <= 0) return 0;  // cudaSuccess = 0
  constexpr int BLOCK = 256;
  int64_t grid = (n + BLOCK - 1) / BLOCK;
  arle_bf16_to_fp8_e4m3_kernel<<<grid, BLOCK, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(input),
      reinterpret_cast<__nv_fp8_e4m3*>(output),
      n);
  return 0;
}
