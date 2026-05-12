// PF8.1 — BF16 → FP8 e4m3 per-row activation quantization
//
// Mirrors w4a8_activation_quant.cu (INT8 variant) with:
//   - Output type: __nv_fp8_e4m3 (sm_89 native conversion)
//   - Scale divisor: 448.0f (e4m3 finite max per IEEE FP8 spec)
//   - Clamp: ±448.0f
//
// Used by NEW prefill-only FP8 directive (per
// docs/research/2026-05-10-prefill-only-fp8-directive-draft.md
// Substep PF8.1, ~60 LOC). Per-row absmax scale stored in FP32
// sidecar tensor for downstream FP8 mma dequant.
//
// Why per-row absmax: e4m3 has limited dynamic range (~3 orders of
// magnitude); per-row scaling adapts to activation outliers without
// catastrophic clipping. Same strategy as INT8 variant.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

// e4m3 finite max (per IEEE FP8 spec). NVIDIA's __nv_fp8_e4m3 type
// uses bias=7, so max representable finite = (1 + 7/8) * 2^8 = 448.
constexpr float FP8_E4M3_MAX = 448.0f;

__device__ __forceinline__ float warp_reduce_max_abs(float val) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
  }
  return val;
}

__global__ void quantize_bf16_rows_to_fp8_e4m3_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_fp8_e4m3* __restrict__ output,
    float* __restrict__ scales,
    int rows,
    int cols) {
  extern __shared__ float smem[];
  int row = blockIdx.x;
  if (row >= rows) return;

  const __nv_bfloat16* in_row = input + (size_t)row * cols;
  __nv_fp8_e4m3* out_row = output + (size_t)row * cols;

  // Per-row absmax reduction
  float local_max = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    local_max = fmaxf(local_max, fabsf(__bfloat162float(in_row[col])));
  }
  local_max = warp_reduce_max_abs(local_max);

  int lane_id = threadIdx.x & 31;
  int warp_id = threadIdx.x >> 5;
  int num_warps = (blockDim.x + 31) >> 5;
  if (lane_id == 0) {
    smem[warp_id] = local_max;
  }
  __syncthreads();

  if (warp_id == 0) {
    float block_max = lane_id < num_warps ? smem[lane_id] : 0.0f;
    block_max = warp_reduce_max_abs(block_max);
    if (lane_id == 0) {
      smem[0] = block_max;
    }
  }
  __syncthreads();

  // Per-row scale = absmax / 448 (e4m3 finite max)
  float scale = smem[0] > 0.0f ? smem[0] / FP8_E4M3_MAX : 1.0f;
  if (threadIdx.x == 0) {
    scales[row] = scale;
  }

  // Quantize: clamp to e4m3 representable range, then convert
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    float qf = __bfloat162float(in_row[col]) / scale;
    qf = fminf(FP8_E4M3_MAX, fmaxf(-FP8_E4M3_MAX, qf));
    // __nv_fp8_e4m3 has constructor from float (sm_89 native conversion)
    out_row[col] = __nv_fp8_e4m3(qf);
  }
}

extern "C" cudaError_t quantize_bf16_rows_to_fp8_e4m3_cuda(
    const __nv_bfloat16* input,
    __nv_fp8_e4m3* output,
    float* scales,
    int rows,
    int cols,
    cudaStream_t stream) {
  if (rows <= 0 || cols <= 0) {
    return cudaSuccess;
  }
  constexpr int threads = 256;
  dim3 grid(rows);
  dim3 block(threads);
  size_t smem = ((threads + 31) / 32) * sizeof(float);
  quantize_bf16_rows_to_fp8_e4m3_kernel<<<grid, block, smem, stream>>>(
      input, output, scales, rows, cols);
  return cudaGetLastError();
}
