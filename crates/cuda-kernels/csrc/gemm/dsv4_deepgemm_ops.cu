#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <limits.h>
#include <stdint.h>

namespace {

constexpr int DSV4_DEEPGEMM_GRAN_K = 128;
constexpr int DSV4_DEEPGEMM_BLOCK = 128;
constexpr float DSV4_FP8_E4M3_MAX = 448.0f;

bool dg_product_exceeds_i32(int a, int b, int c) {
  return static_cast<int64_t>(a) * static_cast<int64_t>(b) *
             static_cast<int64_t>(c) >
         INT_MAX;
}

__device__ __forceinline__ float dg_bf16_to_f32(uint16_t bits) {
  uint32_t raw = static_cast<uint32_t>(bits) << 16;
  return __uint_as_float(raw);
}

__device__ __forceinline__ uint16_t dg_f32_to_bf16(float value) {
  uint32_t raw = __float_as_uint(value);
  uint32_t lsb = (raw >> 16) & 1u;
  uint32_t rounding_bias = 0x7fffu + lsb;
  return static_cast<uint16_t>((raw + rounding_bias) >> 16);
}

__device__ __forceinline__ uint8_t dg_f32_to_fp8(float value) {
  __nv_fp8_e4m3 fp8 = __nv_fp8_e4m3(value);
  return fp8.__x;
}

__device__ __forceinline__ float dg_warp_reduce_max(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, offset));
  }
  return value;
}

__device__ __forceinline__ int dg_scale_offset(
    int expert,
    int row,
    int k_block,
    int scale_stride_m,
    int scale_k_blocks) {
  return expert * scale_stride_m * scale_k_blocks +
         k_block * scale_stride_m + row;
}

__device__ __forceinline__ float dg_swiglu(uint16_t gate_bits, uint16_t up_bits, float limit) {
  float gate = dg_bf16_to_f32(gate_bits);
  float up = dg_bf16_to_f32(up_bits);
  gate = fminf(gate, limit);
  up = fminf(fmaxf(up, -limit), limit);
  return (gate / (1.0f + expf(-gate))) * up;
}

__global__ void dsv4_deepgemm_pack_quantize_bf16_to_fp8_kernel(
    const uint16_t* __restrict__ input,
    uint8_t* __restrict__ output,
    float* __restrict__ scales,
    const int32_t* __restrict__ active_experts,
    const int32_t* __restrict__ active_offsets,
    const int32_t* __restrict__ active_counts,
    int active_count,
    int max_m,
    int cols,
    int scale_stride_m,
    int scale_k_blocks) {
  const int k_block = blockIdx.x;
  const int row = blockIdx.y;
  const int active = blockIdx.z;
  if (active >= active_count) return;
  const int count = active_counts[active];
  if (row >= count) return;
  const int expert = active_experts[active];
  const int src_row = active_offsets[active] + row;
  const int col_start = k_block * DSV4_DEEPGEMM_GRAN_K;
  const int col_end = min(col_start + DSV4_DEEPGEMM_GRAN_K, cols);

  float local_max = 0.0f;
  for (int col = col_start + threadIdx.x; col < col_end; col += blockDim.x) {
    local_max = fmaxf(local_max, fabsf(dg_bf16_to_f32(input[src_row * cols + col])));
  }
  local_max = dg_warp_reduce_max(local_max);
  __shared__ float warp_max[DSV4_DEEPGEMM_BLOCK / 32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) warp_max[warp] = local_max;
  __syncthreads();

  float block_max = 0.0f;
  if (warp == 0) {
    block_max = lane < (blockDim.x + 31) / 32 ? warp_max[lane] : 0.0f;
    block_max = dg_warp_reduce_max(block_max);
  }
  __shared__ float scale_shared;
  if (threadIdx.x == 0) {
    scale_shared = block_max > 0.0f ? block_max / DSV4_FP8_E4M3_MAX : 1.0f;
    scales[dg_scale_offset(expert, row, k_block, scale_stride_m, scale_k_blocks)] =
        scale_shared;
  }
  __syncthreads();

  const float scale = scale_shared;
  uint8_t* dst_row = output + (expert * max_m + row) * cols;
  for (int col = col_start + threadIdx.x; col < col_end; col += blockDim.x) {
    float value = dg_bf16_to_f32(input[src_row * cols + col]);
    dst_row[col] = dg_f32_to_fp8(scale > 0.0f ? value / scale : 0.0f);
  }
}

__global__ void dsv4_deepgemm_swiglu_quantize_w13_kernel(
    const uint16_t* __restrict__ w13,
    uint8_t* __restrict__ act,
    float* __restrict__ scales,
    const int32_t* __restrict__ active_experts,
    const int32_t* __restrict__ active_counts,
    int active_count,
    int max_m,
    int intermediate_dim,
    int scale_stride_m,
    int scale_k_blocks,
    float limit) {
  const int k_block = blockIdx.x;
  const int row = blockIdx.y;
  const int active = blockIdx.z;
  if (active >= active_count) return;
  const int count = active_counts[active];
  if (row >= count) return;
  const int expert = active_experts[active];
  const int col_start = k_block * DSV4_DEEPGEMM_GRAN_K;
  const int col_end = min(col_start + DSV4_DEEPGEMM_GRAN_K, intermediate_dim);
  const int w13_cols = intermediate_dim * 2;
  const uint16_t* w13_row = w13 + (expert * max_m + row) * w13_cols;

  float local_max = 0.0f;
  for (int col = col_start + threadIdx.x; col < col_end; col += blockDim.x) {
    local_max = fmaxf(local_max, fabsf(dg_swiglu(w13_row[col], w13_row[intermediate_dim + col], limit)));
  }
  local_max = dg_warp_reduce_max(local_max);
  __shared__ float warp_max[DSV4_DEEPGEMM_BLOCK / 32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) warp_max[warp] = local_max;
  __syncthreads();

  float block_max = 0.0f;
  if (warp == 0) {
    block_max = lane < (blockDim.x + 31) / 32 ? warp_max[lane] : 0.0f;
    block_max = dg_warp_reduce_max(block_max);
  }
  __shared__ float scale_shared;
  if (threadIdx.x == 0) {
    scale_shared = block_max > 0.0f ? block_max / DSV4_FP8_E4M3_MAX : 1.0f;
    scales[dg_scale_offset(expert, row, k_block, scale_stride_m, scale_k_blocks)] =
        scale_shared;
  }
  __syncthreads();

  const float scale = scale_shared;
  uint8_t* act_row = act + (expert * max_m + row) * intermediate_dim;
  for (int col = col_start + threadIdx.x; col < col_end; col += blockDim.x) {
    float value = dg_swiglu(w13_row[col], w13_row[intermediate_dim + col], limit);
    act_row[col] = dg_f32_to_fp8(scale > 0.0f ? value / scale : 0.0f);
  }
}

__global__ void dsv4_deepgemm_unpad_grouped_bf16_kernel(
    const uint16_t* __restrict__ grouped,
    uint16_t* __restrict__ compact,
    const int32_t* __restrict__ active_experts,
    const int32_t* __restrict__ active_offsets,
    const int32_t* __restrict__ active_counts,
    int active_count,
    int max_m,
    int hidden_dim) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = active_count * max_m * hidden_dim;
  if (idx >= total) return;
  const int col = idx % hidden_dim;
  const int row = (idx / hidden_dim) % max_m;
  const int active = idx / (hidden_dim * max_m);
  const int count = active_counts[active];
  if (row >= count) return;
  const int expert = active_experts[active];
  const int compact_row = active_offsets[active] + row;
  compact[compact_row * hidden_dim + col] =
      grouped[(expert * max_m + row) * hidden_dim + col];
}

}  // namespace

extern "C" CUresult dsv4_deepgemm_pack_quantize_bf16_to_fp8_cuda(
    const uint16_t* input,
    uint8_t* output,
    float* scales,
    const int32_t* active_experts,
    const int32_t* active_offsets,
    const int32_t* active_counts,
    int active_count,
    int max_m,
    int cols,
    int scale_stride_m,
    CUstream stream) {
  if (active_count < 0 || max_m <= 0 || cols <= 0 || scale_stride_m < max_m ||
      (cols % DSV4_DEEPGEMM_GRAN_K) != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (active_count == 0) return CUDA_SUCCESS;
  if (input == nullptr || output == nullptr || scales == nullptr ||
      active_experts == nullptr || active_offsets == nullptr ||
      active_counts == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int scale_k_blocks = (cols + DSV4_DEEPGEMM_GRAN_K - 1) / DSV4_DEEPGEMM_GRAN_K;
  dim3 grid(scale_k_blocks, max_m, active_count);
  dsv4_deepgemm_pack_quantize_bf16_to_fp8_kernel<<<grid, DSV4_DEEPGEMM_BLOCK, 0, (cudaStream_t)stream>>>(
      input, output, scales, active_experts, active_offsets, active_counts,
      active_count, max_m, cols, scale_stride_m, scale_k_blocks);
  return (CUresult)cudaGetLastError();
}

extern "C" CUresult dsv4_deepgemm_swiglu_quantize_w13_cuda(
    const uint16_t* w13,
    uint8_t* act,
    float* scales,
    const int32_t* active_experts,
    const int32_t* active_counts,
    int active_count,
    int max_m,
    int intermediate_dim,
    int scale_stride_m,
    float limit,
    CUstream stream) {
  if (active_count < 0 || max_m <= 0 || intermediate_dim <= 0 ||
      scale_stride_m < max_m || !(limit > 0.0f) ||
      (intermediate_dim % DSV4_DEEPGEMM_GRAN_K) != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (active_count == 0) return CUDA_SUCCESS;
  if (w13 == nullptr || act == nullptr || scales == nullptr ||
      active_experts == nullptr || active_counts == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int scale_k_blocks =
      (intermediate_dim + DSV4_DEEPGEMM_GRAN_K - 1) / DSV4_DEEPGEMM_GRAN_K;
  dim3 grid(scale_k_blocks, max_m, active_count);
  dsv4_deepgemm_swiglu_quantize_w13_kernel<<<grid, DSV4_DEEPGEMM_BLOCK, 0, (cudaStream_t)stream>>>(
      w13, act, scales, active_experts, active_counts, active_count, max_m,
      intermediate_dim, scale_stride_m, scale_k_blocks, limit);
  return (CUresult)cudaGetLastError();
}

extern "C" CUresult dsv4_deepgemm_unpad_grouped_bf16_cuda(
    const uint16_t* grouped,
    uint16_t* compact,
    const int32_t* active_experts,
    const int32_t* active_offsets,
    const int32_t* active_counts,
    int active_count,
    int max_m,
    int hidden_dim,
    CUstream stream) {
  if (active_count < 0 || max_m <= 0 || hidden_dim <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (active_count == 0) return CUDA_SUCCESS;
  if (grouped == nullptr || compact == nullptr || active_experts == nullptr ||
      active_offsets == nullptr || active_counts == nullptr ||
      dg_product_exceeds_i32(active_count, max_m, hidden_dim)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int total = active_count * max_m * hidden_dim;
  int grid = (total + DSV4_DEEPGEMM_BLOCK - 1) / DSV4_DEEPGEMM_BLOCK;
  dsv4_deepgemm_unpad_grouped_bf16_kernel<<<grid, DSV4_DEEPGEMM_BLOCK, 0, (cudaStream_t)stream>>>(
      grouped, compact, active_experts, active_offsets, active_counts,
      active_count, max_m, hidden_dim);
  return (CUresult)cudaGetLastError();
}
