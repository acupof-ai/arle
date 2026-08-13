#include "dsv4_attention_common.cuh"

__global__ void dsv4_oproj_group_gather_kernel(
    const uint16_t *__restrict__ src,
    uint16_t *__restrict__ dst,
    int num_tokens,
    int groups,
    int cols_per_group,
    int group) {
  int64_t total = (int64_t)num_tokens * cols_per_group;
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  int col = (int)(idx % cols_per_group);
  int row = (int)(idx / cols_per_group);
  int64_t src_idx = ((int64_t)row * groups + group) * cols_per_group + col;
  dst[idx] = src[src_idx];
}

__global__ void dsv4_oproj_group_scatter_kernel(
    const uint16_t *__restrict__ src,
    uint16_t *__restrict__ dst,
    int num_tokens,
    int groups,
    int rows_per_group,
    int group) {
  int64_t total = (int64_t)num_tokens * rows_per_group;
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  int col = (int)(idx % rows_per_group);
  int row = (int)(idx / rows_per_group);
  int64_t dst_idx = ((int64_t)row * groups + group) * rows_per_group + col;
  dst[dst_idx] = src[idx];
}

extern "C" CUresult dsv4_oproj_group_gather_cuda(
    const uint16_t *src,
    uint16_t *dst,
    int num_tokens,
    int groups,
    int cols_per_group,
    int group,
    cudaStream_t stream) {
  if (num_tokens < 0 || groups <= 0 || cols_per_group <= 0 || group < 0 ||
      group >= groups || src == nullptr || dst == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int64_t total = (int64_t)num_tokens * cols_per_group;
  if (total == 0) return CUDA_SUCCESS;
  int blocks = (int)((total + DSV4_ATTN_BLOCK - 1) / DSV4_ATTN_BLOCK);
  dsv4_oproj_group_gather_kernel<<<blocks, DSV4_ATTN_BLOCK, 0, stream>>>(
      src, dst, num_tokens, groups, cols_per_group, group);
  return (CUresult)cudaGetLastError();
}

extern "C" CUresult dsv4_oproj_group_scatter_cuda(
    const uint16_t *src,
    uint16_t *dst,
    int num_tokens,
    int groups,
    int rows_per_group,
    int group,
    cudaStream_t stream) {
  if (num_tokens < 0 || groups <= 0 || rows_per_group <= 0 || group < 0 ||
      group >= groups || src == nullptr || dst == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int64_t total = (int64_t)num_tokens * rows_per_group;
  if (total == 0) return CUDA_SUCCESS;
  int blocks = (int)((total + DSV4_ATTN_BLOCK - 1) / DSV4_ATTN_BLOCK);
  dsv4_oproj_group_scatter_kernel<<<blocks, DSV4_ATTN_BLOCK, 0, stream>>>(
      src, dst, num_tokens, groups, rows_per_group, group);
  return (CUresult)cudaGetLastError();
}

// FlashMLA sparse kernels don't apply output inverse-rope; callers must.
// One thread per RoPE pair reads both cols before writing to avoid RAW hazard.
__global__ void dsv4_output_inverse_rope_kernel(
    uint16_t *__restrict__ out,
    int token_count,
    int local_heads,
    int head_dim,
    int rope_dim,
    int start_pos,
    const int *__restrict__ start_pos_ptr,
    float rope_base,
    int original_seq_len,
    float factor,
    float beta_fast,
    float beta_slow) {
  int row = blockIdx.x;
  if (row >= token_count * local_heads) return;
  int token = row / local_heads;
  int head = row - token * local_heads;
  int local_width = local_heads * head_dim;
  int abs_pos = dsv4_graph_start_pos(start_pos, start_pos_ptr) + token;
  int rope_start = head_dim - rope_dim;
  int base = token * local_width + head * head_dim;

  int pair = threadIdx.x;
  if (pair >= rope_dim / 2) return;
  int pair_col = rope_start + pair * 2;
  float a = dsv4_attn_bf16_to_f32(out[base + pair_col]);
  float b = dsv4_attn_bf16_to_f32(out[base + pair_col + 1]);
  float out_a;
  float out_b;
  dsv4_apply_rope_pair(
      a, b, pair, abs_pos, rope_dim, rope_base, original_seq_len, factor,
      beta_fast, beta_slow, -1.0f, &out_a, &out_b);
  out[base + pair_col] = dsv4_attn_f32_to_bf16_bits(out_a);
  out[base + pair_col + 1] = dsv4_attn_f32_to_bf16_bits(out_b);
}

__global__ void dsv4_output_inverse_rope_batch_start_pos_kernel(
    uint16_t *__restrict__ out,
    int token_count,
    int local_heads,
    int head_dim,
    int rope_dim,
    const int *__restrict__ start_pos,
    float rope_base,
    int original_seq_len,
    float factor,
    float beta_fast,
    float beta_slow) {
  int row = blockIdx.x;
  if (row >= token_count * local_heads) return;
  int token = row / local_heads;
  int head = row - token * local_heads;
  int local_width = local_heads * head_dim;
  int abs_pos = start_pos[token];
  int rope_start = head_dim - rope_dim;
  int base = token * local_width + head * head_dim;

  int pair = threadIdx.x;
  if (pair >= rope_dim / 2) return;
  int pair_col = rope_start + pair * 2;
  float a = dsv4_attn_bf16_to_f32(out[base + pair_col]);
  float b = dsv4_attn_bf16_to_f32(out[base + pair_col + 1]);
  float out_a;
  float out_b;
  dsv4_apply_rope_pair(
      a, b, pair, abs_pos, rope_dim, rope_base, original_seq_len, factor,
      beta_fast, beta_slow, -1.0f, &out_a, &out_b);
  out[base + pair_col] = dsv4_attn_f32_to_bf16_bits(out_a);
  out[base + pair_col + 1] = dsv4_attn_f32_to_bf16_bits(out_b);
}

// Batched over non-contiguous per-row buffers; replaces n single-row launches.
__global__ void dsv4_output_inverse_rope_batched_ptr_kernel(
    uint16_t *const *__restrict__ out_arr,
    int n,
    int local_heads,
    int head_dim,
    int rope_dim,
    const int *__restrict__ start_pos,
    float rope_base,
    int original_seq_len,
    float factor,
    float beta_fast,
    float beta_slow) {
  int blk = blockIdx.x;
  if (blk >= n * local_heads) return;
  int rowi = blk / local_heads;
  int head = blk - rowi * local_heads;
  uint16_t *out = out_arr[rowi];
  int abs_pos = start_pos[rowi];
  int rope_start = head_dim - rope_dim;
  int base = head * head_dim;

  int pair = threadIdx.x;
  if (pair >= rope_dim / 2) return;
  int pair_col = rope_start + pair * 2;
  float a = dsv4_attn_bf16_to_f32(out[base + pair_col]);
  float b = dsv4_attn_bf16_to_f32(out[base + pair_col + 1]);
  float out_a;
  float out_b;
  dsv4_apply_rope_pair(
      a, b, pair, abs_pos, rope_dim, rope_base, original_seq_len, factor,
      beta_fast, beta_slow, -1.0f, &out_a, &out_b);
  out[base + pair_col] = dsv4_attn_f32_to_bf16_bits(out_a);
  out[base + pair_col + 1] = dsv4_attn_f32_to_bf16_bits(out_b);
}

extern "C" cudaError_t arle_dsv4_output_inverse_rope_cuda(
    uint16_t *out,
    int token_count,
    int local_heads,
    int head_dim,
    int rope_dim,
    int start_pos,
    float rope_base,
    int original_seq_len,
    float factor,
    float beta_fast,
    float beta_slow,
    cudaStream_t stream) {
  if (token_count < 0 || local_heads <= 0 || head_dim <= 0 ||
      head_dim > DSV4_ATTN_MAX_HEAD_DIM || rope_dim < 0 || rope_dim > head_dim ||
      start_pos < 0) {
    return cudaErrorInvalidValue;
  }
  if (token_count == 0 || rope_dim == 0) return cudaSuccess;
  if (out == nullptr) return cudaErrorInvalidValue;
  dsv4_output_inverse_rope_kernel<<<token_count * local_heads, rope_dim / 2, 0, stream>>>(
      out, token_count, local_heads, head_dim, rope_dim, start_pos, nullptr, rope_base,
      original_seq_len, factor, beta_fast, beta_slow);
  return cudaGetLastError();
}

extern "C" cudaError_t arle_dsv4_output_inverse_rope_start_pos_ptr_cuda(
    uint16_t *out,
    int token_count,
    int local_heads,
    int head_dim,
    int rope_dim,
    const int *start_pos_ptr,
    float rope_base,
    int original_seq_len,
    float factor,
    float beta_fast,
    float beta_slow,
    cudaStream_t stream) {
  if (token_count < 0 || local_heads <= 0 || head_dim <= 0 ||
      head_dim > DSV4_ATTN_MAX_HEAD_DIM || rope_dim < 0 || rope_dim > head_dim ||
      start_pos_ptr == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (token_count == 0 || rope_dim == 0) return cudaSuccess;
  if (out == nullptr) return cudaErrorInvalidValue;
  dsv4_output_inverse_rope_kernel<<<token_count * local_heads, rope_dim / 2, 0, stream>>>(
      out, token_count, local_heads, head_dim, rope_dim, 0, start_pos_ptr, rope_base,
      original_seq_len, factor, beta_fast, beta_slow);
  return cudaGetLastError();
}

extern "C" cudaError_t arle_dsv4_output_inverse_rope_batch_start_pos_cuda(
    uint16_t *out,
    int token_count,
    int local_heads,
    int head_dim,
    int rope_dim,
    const int *start_pos,
    float rope_base,
    int original_seq_len,
    float factor,
    float beta_fast,
    float beta_slow,
    cudaStream_t stream) {
  if (token_count < 0 || local_heads <= 0 || head_dim <= 0 ||
      head_dim > DSV4_ATTN_MAX_HEAD_DIM || rope_dim < 0 || rope_dim > head_dim ||
      start_pos == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (token_count == 0 || rope_dim == 0) return cudaSuccess;
  if (out == nullptr) return cudaErrorInvalidValue;
  dsv4_output_inverse_rope_batch_start_pos_kernel<<<token_count * local_heads, rope_dim / 2, 0, stream>>>(
      out, token_count, local_heads, head_dim, rope_dim, start_pos, rope_base,
      original_seq_len, factor, beta_fast, beta_slow);
  return cudaGetLastError();
}

extern "C" cudaError_t arle_dsv4_output_inverse_rope_batched_ptr_cuda(
    uint16_t *const *out_arr,
    int n,
    int local_heads,
    int head_dim,
    int rope_dim,
    const int *start_pos,
    float rope_base,
    int original_seq_len,
    float factor,
    float beta_fast,
    float beta_slow,
    cudaStream_t stream) {
  if (n < 0 || local_heads <= 0 || head_dim <= 0 ||
      head_dim > DSV4_ATTN_MAX_HEAD_DIM || rope_dim < 0 || rope_dim > head_dim ||
      start_pos == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (n == 0 || rope_dim == 0) return cudaSuccess;
  if (out_arr == nullptr) return cudaErrorInvalidValue;
  dsv4_output_inverse_rope_batched_ptr_kernel<<<n * local_heads, rope_dim / 2, 0, stream>>>(
      out_arr, n, local_heads, head_dim, rope_dim, start_pos, rope_base,
      original_seq_len, factor, beta_fast, beta_slow);
  return cudaGetLastError();
}
