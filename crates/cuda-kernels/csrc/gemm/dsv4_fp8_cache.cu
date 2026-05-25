// DeepSeek V4 block-scaled weight -> DeepGEMM SM90 FP8 cache.
//
// DSV4 Flash routed experts are shipped as row-major FP4 E2M1 with E8M0
// block scales. DeepGEMM's Hopper grouped GEMM path consumes FP8 E4M3 weights
// plus FP32 block scales. This file builds that resident cache once at load
// time so the runtime expert path can avoid per-route FP4 GEMV.

#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int DSV4_DEEPGEMM_SCALE_GRAN_M = 128;
constexpr int DSV4_DEEPGEMM_SCALE_GRAN_K = 128;
constexpr int DSV4_FP8_CACHE_THREADS = 256;
constexpr float DSV4_FP8_E4M3_MAX = 448.0f;

enum Dsv4SourceFormat : int {
  kDsv4SourceFp8 = 0,
  kDsv4SourceFp4 = 1,
};

__device__ __constant__ float DSV4_FP4_E2M1_CACHE_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

__device__ __forceinline__ float dsv4_cache_warp_reduce_max(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
  }
  return val;
}

__device__ __forceinline__ float dsv4_cache_decode_e8m0(uint8_t bits) {
  uint32_t raw = static_cast<uint32_t>(bits) << 23;
  return __uint_as_float(raw);
}

__device__ __forceinline__ float dsv4_cache_decode_fp8_e4m3(uint8_t bits) {
  if ((bits & 0x7f) == 0) return 0.0f;
  if ((bits & 0x7f) == 0x7f) {
    return (bits & 0x80) ? -DSV4_FP8_E4M3_MAX : DSV4_FP8_E4M3_MAX;
  }
  __nv_fp8_e4m3 value;
  value.__x = bits;
  return static_cast<float>(value);
}

__device__ __forceinline__ float dsv4_cache_decode_fp4_e2m1(uint8_t bits) {
  return DSV4_FP4_E2M1_CACHE_LUT[bits & 0x0f];
}

__device__ __forceinline__ float dsv4_cache_block_scale(
    const uint8_t* __restrict__ scales,
    int row,
    int col,
    int rows,
    int cols,
    int scale_rows,
    int scale_cols) {
  const int block_h = (rows + scale_rows - 1) / scale_rows;
  const int block_w = (cols + scale_cols - 1) / scale_cols;
  const int sr_raw = row / block_h;
  const int sc_raw = col / block_w;
  const int sr = sr_raw < scale_rows ? sr_raw : (scale_rows - 1);
  const int sc = sc_raw < scale_cols ? sc_raw : (scale_cols - 1);
  return dsv4_cache_decode_e8m0(scales[sr * scale_cols + sc]);
}

__device__ __forceinline__ float dsv4_cache_source_value(
    const uint8_t* __restrict__ weight,
    const uint8_t* __restrict__ scales,
    int row,
    int col,
    int rows,
    int cols,
    int scale_rows,
    int scale_cols,
    int source_format) {
  float encoded;
  if (source_format == kDsv4SourceFp8) {
    encoded = dsv4_cache_decode_fp8_e4m3(weight[row * cols + col]);
  } else {
    const int bytes_per_row = cols >> 1;
    const uint8_t packed = weight[row * bytes_per_row + (col >> 1)];
    const uint8_t nibble = (col & 1) ? ((packed >> 4) & 0x0f) : (packed & 0x0f);
    encoded = dsv4_cache_decode_fp4_e2m1(nibble);
  }
  return encoded *
         dsv4_cache_block_scale(scales, row, col, rows, cols, scale_rows, scale_cols);
}

__global__ void dsv4_block_scaled_to_fp8_cache_scales_kernel(
    const uint8_t* __restrict__ weight,
    const uint8_t* __restrict__ src_scales,
    float* __restrict__ dst_scales,
    int rows,
    int cols,
    int scale_rows,
    int scale_cols,
    int dst_scale_cols,
    int source_format) {
  extern __shared__ float smem[];
  const int scale_row = blockIdx.y;
  const int scale_col = blockIdx.x;
  const int row_start = scale_row * DSV4_DEEPGEMM_SCALE_GRAN_M;
  const int col_start = scale_col * DSV4_DEEPGEMM_SCALE_GRAN_K;
  const int row_end = min(row_start + DSV4_DEEPGEMM_SCALE_GRAN_M, rows);
  const int col_end = min(col_start + DSV4_DEEPGEMM_SCALE_GRAN_K, cols);

  float local_max = 0.0f;
  const int tile_cols = col_end - col_start;
  const int tile_elems = (row_end - row_start) * tile_cols;
  for (int idx = threadIdx.x; idx < tile_elems; idx += blockDim.x) {
    const int row = row_start + idx / tile_cols;
    const int col = col_start + idx % tile_cols;
    const float value = dsv4_cache_source_value(
        weight, src_scales, row, col, rows, cols, scale_rows, scale_cols,
        source_format);
    local_max = fmaxf(local_max, fabsf(value));
  }

  local_max = dsv4_cache_warp_reduce_max(local_max);
  const int lane_id = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int num_warps = (blockDim.x + 31) >> 5;
  if (lane_id == 0) smem[warp_id] = local_max;
  __syncthreads();

  if (warp_id == 0) {
    float block_max = lane_id < num_warps ? smem[lane_id] : 0.0f;
    block_max = dsv4_cache_warp_reduce_max(block_max);
    if (lane_id == 0) {
      const float scale = block_max > 0.0f ? block_max / DSV4_FP8_E4M3_MAX : 1.0f;
      dst_scales[scale_row * dst_scale_cols + scale_col] = scale;
    }
  }
}

__global__ void dsv4_block_scaled_to_fp8_cache_values_kernel(
    const uint8_t* __restrict__ weight,
    const uint8_t* __restrict__ src_scales,
    uint8_t* __restrict__ dst_weight,
    const float* __restrict__ dst_scales,
    int rows,
    int cols,
    int scale_rows,
    int scale_cols,
    int dst_scale_cols,
    int source_format) {
  const int row = blockIdx.x;
  if (row >= rows) return;

  const int scale_row = row / DSV4_DEEPGEMM_SCALE_GRAN_M;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int scale_col = col / DSV4_DEEPGEMM_SCALE_GRAN_K;
    const float scale = dst_scales[scale_row * dst_scale_cols + scale_col];
    const float value = dsv4_cache_source_value(
        weight, src_scales, row, col, rows, cols, scale_rows, scale_cols,
        source_format);
    const float q = scale > 0.0f ? value / scale : 0.0f;
    __nv_fp8_e4m3 fp8 = __nv_fp8_e4m3(q);
    dst_weight[row * cols + col] = fp8.__x;
  }
}

}  // namespace

extern "C" cudaError_t dsv4_block_scaled_to_fp8_deepgemm_cuda(
    const uint8_t* weight,
    const uint8_t* src_scales,
    uint8_t* dst_weight,
    float* dst_scales,
    int rows,
    int cols,
    int scale_rows,
    int scale_cols,
    int dst_scale_cols,
    int source_format,
    cudaStream_t stream) {
  if (rows <= 0 || cols <= 0 || scale_rows <= 0 || scale_cols <= 0 ||
      dst_scale_cols <= 0) {
    return cudaErrorInvalidValue;
  }
  if (weight == nullptr || src_scales == nullptr || dst_weight == nullptr ||
      dst_scales == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (source_format != kDsv4SourceFp8 && source_format != kDsv4SourceFp4) {
    return cudaErrorInvalidValue;
  }
  if (source_format == kDsv4SourceFp4 && (cols & 1) != 0) {
    return cudaErrorInvalidValue;
  }

  const int dst_scale_rows =
      (rows + DSV4_DEEPGEMM_SCALE_GRAN_M - 1) / DSV4_DEEPGEMM_SCALE_GRAN_M;
  const int expected_scale_cols =
      (cols + DSV4_DEEPGEMM_SCALE_GRAN_K - 1) / DSV4_DEEPGEMM_SCALE_GRAN_K;
  if (dst_scale_cols != expected_scale_cols) {
    return cudaErrorInvalidValue;
  }

  dim3 scale_grid(dst_scale_cols, dst_scale_rows);
  dim3 block(DSV4_FP8_CACHE_THREADS);
  size_t smem = ((DSV4_FP8_CACHE_THREADS + 31) / 32) * sizeof(float);
  dsv4_block_scaled_to_fp8_cache_scales_kernel<<<scale_grid, block, smem, stream>>>(
      weight, src_scales, dst_scales, rows, cols, scale_rows, scale_cols,
      dst_scale_cols, source_format);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  dsv4_block_scaled_to_fp8_cache_values_kernel<<<rows, block, 0, stream>>>(
      weight, src_scales, dst_weight, dst_scales, rows, cols, scale_rows,
      scale_cols, dst_scale_cols, source_format);
  return cudaGetLastError();
}
