#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
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
  // One warp per 128-element quantization block: 4 bf16 per lane held in
  // registers, so amax and the scaled write share a single read and the
  // reduction stays in shuffles. Caller guarantees cols % GRAN_K == 0.
  int64_t linear = (int64_t)blockIdx.x * (DSV4_DEEPGEMM_BLOCK / 32) + (threadIdx.x >> 5);
  const int k_block = static_cast<int>(linear % scale_k_blocks);
  linear /= scale_k_blocks;
  const int row = static_cast<int>(linear % max_m);
  const int active = static_cast<int>(linear / max_m);
  if (active >= active_count) return;
  const int count = active_counts[active];
  if (row >= count) return;
  const int expert = active_experts[active];
  const int src_row = active_offsets[active] + row;
  const int lane = threadIdx.x & 31;
  const int64_t col_base = (int64_t)k_block * DSV4_DEEPGEMM_GRAN_K;

  const ushort4 v = reinterpret_cast<const ushort4*>(
      input + (int64_t)src_row * cols + col_base)[lane];
  const float f0 = dg_bf16_to_f32(v.x), f1 = dg_bf16_to_f32(v.y);
  const float f2 = dg_bf16_to_f32(v.z), f3 = dg_bf16_to_f32(v.w);
  float local_max = fmaxf(fmaxf(fabsf(f0), fabsf(f1)), fmaxf(fabsf(f2), fabsf(f3)));
  local_max = dg_warp_reduce_max(local_max);

  const float scale = local_max > 0.0f ? local_max / DSV4_FP8_E4M3_MAX : 1.0f;
  if (lane == 0) {
    scales[dg_scale_offset(expert, row, k_block, scale_stride_m, scale_k_blocks)] = scale;
  }

  uchar4 out;
  out.x = dg_f32_to_fp8(f0 / scale);
  out.y = dg_f32_to_fp8(f1 / scale);
  out.z = dg_f32_to_fp8(f2 / scale);
  out.w = dg_f32_to_fp8(f3 / scale);
  reinterpret_cast<uchar4*>(
      output + ((int64_t)expert * max_m + row) * cols + col_base)[lane] = out;
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
  int64_t linear = blockIdx.x;
  const int k_block = static_cast<int>(linear % scale_k_blocks);
  linear /= scale_k_blocks;
  const int row = static_cast<int>(linear % max_m);
  const int active = static_cast<int>(linear / max_m);
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


// Masked 3-D variant of the dense swiglu+quantize kernel above. Operates
// directly on the deepep_ll grouped layout (input `[E, tok_padded, 2*hidden]`
// bf16, output `[E, tok_padded, hidden]` fp8) and skips invalid tokens per
// expert using `masked_m[expert]` (only the first `masked_m[e]` tokens of
// expert `e` carry valid GEMM1 output). Mirrors SGLang
// `silu_and_mul_masked_post_quant` math (clamped DSv4 SwiGLU + per-128-group
// FP8 e4m3 quant) but emits ARLE's column-major SFA scale layout
// (`dg_scale_offset`, stride `scale_stride_m`) so the masked grouped
// DeepGEMM (`make_tma_sfa_desc`) consumes it unchanged.
//
// `hidden_dim` is the gate||up width (= 2 * intermediate). Output width and
// quant axis is `intermediate = hidden_dim / 2`.
__global__ void dsv4_deepgemm_silu_mul_masked_quant_kernel(
    const uint16_t* __restrict__ input,
    uint8_t* __restrict__ output,
    float* __restrict__ output_scale,
    const int32_t* __restrict__ masked_m,
    int expert_num,
    int token_num_grid,
    int token_num_padded,
    int intermediate_dim,
    int scale_stride_m,
    int scale_k_blocks,
    float limit) {
  // Grid decomposes over `token_num_grid` rows (host-known per-expert valid-row
  // upper bound, e.g. the step's global token count) while all memory strides
  // keep the full `token_num_padded` band. DeepEP LL packs received rows from
  // row 0, so rows >= token_num_grid are guaranteed invalid when the bound
  // holds; the trap below makes a violated bound loud instead of silently
  // skipping valid rows.
  int64_t linear = blockIdx.x;
  const int k_block = static_cast<int>(linear % scale_k_blocks);
  linear /= scale_k_blocks;
  const int row = static_cast<int>(linear % token_num_grid);
  const int expert = static_cast<int>(linear / token_num_grid);
  if (expert >= expert_num) return;
  const int count = masked_m[expert];
  if (row == 0 && threadIdx.x == 0 && count > token_num_grid) __trap();
  if (row >= count) return;
  const int col_start = k_block * DSV4_DEEPGEMM_GRAN_K;
  const int col_end = min(col_start + DSV4_DEEPGEMM_GRAN_K, intermediate_dim);
  const int in_cols = intermediate_dim * 2;
  const uint16_t* in_row =
      input + (static_cast<int64_t>(expert) * token_num_padded + row) * in_cols;

  float local_max = 0.0f;
  for (int col = col_start + threadIdx.x; col < col_end; col += blockDim.x) {
    local_max = fmaxf(
        local_max,
        fabsf(dg_swiglu(in_row[col], in_row[intermediate_dim + col], limit)));
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
    output_scale[dg_scale_offset(
        expert, row, k_block, scale_stride_m, scale_k_blocks)] = scale_shared;
  }
  __syncthreads();

  const float scale = scale_shared;
  uint8_t* out_row =
      output +
      (static_cast<int64_t>(expert) * token_num_padded + row) * intermediate_dim;
  for (int col = col_start + threadIdx.x; col < col_end; col += blockDim.x) {
    float value = dg_swiglu(in_row[col], in_row[intermediate_dim + col], limit);
    out_row[col] = dg_f32_to_fp8(scale > 0.0f ? value / scale : 0.0f);
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
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(active_count) * max_m * hidden_dim;
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
  int64_t warps = static_cast<int64_t>(scale_k_blocks) * max_m * active_count;
  int64_t grid = (warps + (DSV4_DEEPGEMM_BLOCK / 32) - 1) / (DSV4_DEEPGEMM_BLOCK / 32);
  if (warps > INT_MAX) return CUDA_ERROR_INVALID_VALUE;
  dsv4_deepgemm_pack_quantize_bf16_to_fp8_kernel<<<static_cast<int>(grid), DSV4_DEEPGEMM_BLOCK, 0, (cudaStream_t)stream>>>(
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
  int64_t grid = static_cast<int64_t>(scale_k_blocks) * max_m * active_count;
  if (grid > INT_MAX) return CUDA_ERROR_INVALID_VALUE;
  dsv4_deepgemm_swiglu_quantize_w13_kernel<<<static_cast<int>(grid), DSV4_DEEPGEMM_BLOCK, 0, (cudaStream_t)stream>>>(
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
      active_offsets == nullptr || active_counts == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int64_t total = static_cast<int64_t>(active_count) * max_m * hidden_dim;
  int64_t grid = (total + DSV4_DEEPGEMM_BLOCK - 1) / DSV4_DEEPGEMM_BLOCK;
  if (grid > INT_MAX) return CUDA_ERROR_INVALID_VALUE;
  dsv4_deepgemm_unpad_grouped_bf16_kernel<<<static_cast<int>(grid), DSV4_DEEPGEMM_BLOCK, 0, (cudaStream_t)stream>>>(
      grouped, compact, active_experts, active_offsets, active_counts,
      active_count, max_m, hidden_dim);
  return (CUresult)cudaGetLastError();
}


extern "C" CUresult dsv4_deepgemm_silu_mul_masked_quant_cuda(
    const __nv_bfloat16* input,
    uint8_t* out_fp8,
    float* out_scale,
    const int* masked_m,
    int expert_num,
    int token_num_padded,
    int expected_m,
    int hidden_dim,
    float swiglu_limit,
    CUstream stream) {
  // `hidden_dim` is the gate||up width; the quant axis is intermediate = hidden_dim/2.
  // `expected_m` is the host-known upper bound on any expert's valid rows
  // (per-expert recv count <= the step's global token count); the launch grid
  // covers only `min(expected_m, token_num_padded)` rows per expert instead of
  // the full padded band — at decode B=1 this collapses the grid by ~2048x
  // (the empty-block drain over [E, m_padded, k_blocks] measured 631 us/call,
  // 52.9% of deepep_ll GPU time on 8xH20).
  if (expert_num < 0 || token_num_padded <= 0 || expected_m <= 0 ||
      hidden_dim <= 0 || (hidden_dim & 1) != 0 || !(swiglu_limit > 0.0f)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const int intermediate_dim = hidden_dim / 2;
  if ((intermediate_dim % DSV4_DEEPGEMM_GRAN_K) != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (expert_num == 0) return CUDA_SUCCESS;
  if (input == nullptr || out_fp8 == nullptr || out_scale == nullptr ||
      masked_m == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const int token_num_grid =
      expected_m < token_num_padded ? expected_m : token_num_padded;
  // Column-major SFA stride the masked grouped DeepGEMM expects: TMA-aligned to
  // a multiple of 4 over the padded token dim (`get_tma_aligned_size(m, 4)`).
  const int scale_stride_m = ((token_num_padded + 3) / 4) * 4;
  const int scale_k_blocks =
      (intermediate_dim + DSV4_DEEPGEMM_GRAN_K - 1) / DSV4_DEEPGEMM_GRAN_K;
  int64_t grid = static_cast<int64_t>(scale_k_blocks) * token_num_grid * expert_num;
  if (grid > INT_MAX) return CUDA_ERROR_INVALID_VALUE;
  dsv4_deepgemm_silu_mul_masked_quant_kernel<<<static_cast<int>(grid), DSV4_DEEPGEMM_BLOCK, 0, (cudaStream_t)stream>>>(
      reinterpret_cast<const uint16_t*>(input), out_fp8, out_scale, masked_m,
      expert_num, token_num_grid, token_num_padded, intermediate_dim,
      scale_stride_m, scale_k_blocks, swiglu_limit);
  return (CUresult)cudaGetLastError();
}
