#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>

#define WARP_SIZE 32

// Warp-level sum reduction (fp32)
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Warp-level max reduction (fp32)
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// HD256 RMS norm over a warp-block (8 warps). Shared by the paged prefill prep
// and the 2D ring prefill dense prep — same math, one copy.
#define HD256 256
#define HD256_NUM_WARPS (HD256 / WARP_SIZE)

__device__ __forceinline__ float rms_norm_hd256(
    float val, float weight, float eps, int tid) {
  float sq_sum = warp_reduce_sum(val * val);
  __shared__ float scratch[HD256_NUM_WARPS];
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;
  if (lane_id == 0) {
    scratch[warp_id] = sq_sum;
  }
  __syncthreads();
  if (tid == 0) {
    float total = 0.0f;
    for (int i = 0; i < HD256_NUM_WARPS; ++i) {
      total += scratch[i];
    }
    scratch[0] = 1.0f / sqrtf(total / HD256 + eps);
  }
  __syncthreads();
  return val * scratch[0] * (1.0f + weight);  // #58: hd256 q/k_norm OFFSET
}

__device__ __forceinline__ float apply_rope_partial_hd256(
    float* smem, const __nv_bfloat16* cos_cache, const __nv_bfloat16* sin_cache,
    int pos, int tid, int rotary_dim) {
  int half_rotary = rotary_dim / 2;
  if (tid < half_rotary) {
    float cos_val = __bfloat162float(cos_cache[pos * rotary_dim + tid]);
    float sin_val = __bfloat162float(sin_cache[pos * rotary_dim + tid]);
    return smem[tid] * cos_val - smem[tid + half_rotary] * sin_val;
  }
  if (tid < rotary_dim) {
    int pair = tid - half_rotary;
    float cos_val = __bfloat162float(cos_cache[pos * rotary_dim + pair]);
    float sin_val = __bfloat162float(sin_cache[pos * rotary_dim + pair]);
    return smem[pair] * sin_val + smem[tid] * cos_val;
  }
  return smem[tid];
}
