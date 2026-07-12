#include "dsv4_attention_common.cuh"

#define DSV4_ATTN_MAX_WINDOW 1024

__global__ void dsv4_swa_attention_kernel(
    const uint16_t *__restrict__ q,
    const uint16_t *__restrict__ k_new,
    uint16_t *__restrict__ window_cache,
    const uint16_t *__restrict__ attn_sink,
    uint16_t *__restrict__ out,
    int num_tokens,
    int local_heads,
    int head_dim,
    int sliding_window,
    int start_pos,
    int sink_offset,
    float scale_value,
    int rope_dim,
    float rope_base,
    int original_seq_len,
    float factor,
    float beta_fast,
    float beta_slow,
    int write_window_cache,
    const int *__restrict__ start_pos_ptr) {
  int row = blockIdx.x;
  if (row >= num_tokens * local_heads) return;
  int token = row / local_heads;
  int head = row - token * local_heads;
  int local_width = local_heads * head_dim;
  int base_start_pos = dsv4_graph_start_pos(start_pos, start_pos_ptr);
  int abs_pos = base_start_pos + token;
  int sw_start = dsv4_imax(0, abs_pos + 1 - sliding_window);
  int key_count = abs_pos - sw_start + 1;

  __shared__ float logits[DSV4_ATTN_MAX_WINDOW];
  __shared__ float denom_shared;
  __shared__ float max_shared;
  __shared__ float out_vec[DSV4_ATTN_MAX_HEAD_DIM];

  int q_base = token * local_width + head * head_dim;
  for (int key_idx = threadIdx.x; key_idx < key_count; key_idx += blockDim.x) {
    int key_pos = sw_start + key_idx;
    float acc = 0.0f;
    for (int col = 0; col < head_dim; ++col) {
      float qv = dsv4_attn_bf16_to_f32(q[q_base + col]);
      float kv = dsv4_swa_key_value(k_new, window_cache, key_pos, base_start_pos, sliding_window, head_dim, col);
      acc += qv * kv;
    }
    logits[key_idx] = acc * scale_value;
  }
  __syncthreads();

  float local_max = -INFINITY;
  for (int key_idx = threadIdx.x; key_idx < key_count; key_idx += blockDim.x) {
    local_max = fmaxf(local_max, logits[key_idx]);
  }
  float sink = dsv4_attn_bf16_to_f32(attn_sink[sink_offset + head]);
  if (threadIdx.x == 0) local_max = fmaxf(local_max, sink);
  local_max = dsv4_attn_block_max(local_max);
  if (threadIdx.x == 0) max_shared = local_max;
  __syncthreads();

  float denom = 0.0f;
  for (int key_idx = threadIdx.x; key_idx < key_count; key_idx += blockDim.x) {
    float prob = expf(logits[key_idx] - max_shared);
    logits[key_idx] = prob;
    denom += prob;
  }
  if (threadIdx.x == 0) denom += expf(sink - max_shared);
  denom = dsv4_attn_block_sum(denom);
  if (threadIdx.x == 0) denom_shared = denom;
  __syncthreads();

  for (int col = threadIdx.x; col < head_dim; col += blockDim.x) {
    float acc = 0.0f;
    for (int key_idx = 0; key_idx < key_count; ++key_idx) {
      int key_pos = sw_start + key_idx;
      float kv = dsv4_swa_key_value(k_new, window_cache, key_pos, base_start_pos, sliding_window, head_dim, col);
      acc += (logits[key_idx] / denom_shared) * kv;
    }
    out_vec[col] = acc;
  }
  __syncthreads();

  int rope_start = head_dim - rope_dim;
  for (int col = threadIdx.x; col < head_dim; col += blockDim.x) {
    float value = out_vec[col];
    if (rope_dim > 0 && col >= rope_start) {
      int local = col - rope_start;
      int pair = local / 2;
      int pair_col = rope_start + pair * 2;
      float out_a;
      float out_b;
      dsv4_apply_rope_pair(
          out_vec[pair_col], out_vec[pair_col + 1], pair, abs_pos, rope_dim,
          rope_base, original_seq_len, factor, beta_fast, beta_slow, -1.0f,
          &out_a, &out_b);
      value = (local & 1) == 0 ? out_a : out_b;
    }
    out[token * local_width + head * head_dim + col] = dsv4_attn_f32_to_bf16_bits(value);
  }

  if (write_window_cache && head == 0) {
    int slot = abs_pos % sliding_window;
    for (int col = threadIdx.x; col < head_dim; col += blockDim.x) {
      window_cache[slot * head_dim + col] = k_new[token * head_dim + col];
    }
  }
}

extern "C" CUresult dsv4_swa_attention_cuda(
    const uint16_t *q,
    const uint16_t *k_new,
    uint16_t *window_cache,
    const uint16_t *attn_sink,
    uint16_t *out,
    int num_tokens,
    int local_heads,
    int head_dim,
    int sliding_window,
    int start_pos,
    int sink_offset,
    float scale_value,
    int rope_dim,
    float rope_base,
    int original_seq_len,
    float factor,
    float beta_fast,
    float beta_slow,
    int write_window_cache,
    CUstream stream) {
  if (num_tokens < 0 || local_heads <= 0 || head_dim <= 0 || sliding_window <= 0 ||
      sliding_window > DSV4_ATTN_MAX_WINDOW || head_dim > DSV4_ATTN_MAX_HEAD_DIM ||
      rope_dim < 0 || rope_dim > head_dim || start_pos < 0 || sink_offset < 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (num_tokens == 0) return CUDA_SUCCESS;
  dsv4_swa_attention_kernel<<<num_tokens * local_heads, DSV4_ATTN_BLOCK, 0, (cudaStream_t)stream>>>(
      q, k_new, window_cache, attn_sink, out, num_tokens, local_heads, head_dim,
      sliding_window, start_pos, sink_offset, scale_value, rope_dim, rope_base,
      original_seq_len, factor, beta_fast, beta_slow, write_window_cache, nullptr);
  return (CUresult)cudaGetLastError();
}

extern "C" CUresult dsv4_swa_attention_start_pos_ptr_cuda(
    const uint16_t *q,
    const uint16_t *k_new,
    uint16_t *window_cache,
    const uint16_t *attn_sink,
    uint16_t *out,
    int num_tokens,
    int local_heads,
    int head_dim,
    int sliding_window,
    const int *start_pos_ptr,
    int sink_offset,
    float scale_value,
    int rope_dim,
    float rope_base,
    int original_seq_len,
    float factor,
    float beta_fast,
    float beta_slow,
    int write_window_cache,
    CUstream stream) {
  if (num_tokens < 0 || local_heads <= 0 || head_dim <= 0 || sliding_window <= 0 ||
      sliding_window > DSV4_ATTN_MAX_WINDOW || head_dim > DSV4_ATTN_MAX_HEAD_DIM ||
      rope_dim < 0 || rope_dim > head_dim || start_pos_ptr == nullptr || sink_offset < 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (num_tokens == 0) return CUDA_SUCCESS;
  dsv4_swa_attention_kernel<<<num_tokens * local_heads, DSV4_ATTN_BLOCK, 0, (cudaStream_t)stream>>>(
      q, k_new, window_cache, attn_sink, out, num_tokens, local_heads, head_dim,
      sliding_window, 0, sink_offset, scale_value, rope_dim, rope_base,
      original_seq_len, factor, beta_fast, beta_slow, write_window_cache, start_pos_ptr);
  return (CUresult)cudaGetLastError();
}

__global__ void dsv4_update_window_cache_kernel(
    const uint16_t *__restrict__ k_new,
    uint16_t *__restrict__ window_cache,
    int num_tokens,
    int start_pos,
    int sliding_window,
    int head_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = num_tokens * head_dim;
  if (idx >= total) return;
  int token = idx / head_dim;
  int col = idx - token * head_dim;
  int slot = (start_pos + token) % sliding_window;
  window_cache[slot * head_dim + col] = k_new[token * head_dim + col];
}

__global__ void dsv4_update_window_cache_start_pos_ptr_kernel(
    const uint16_t *__restrict__ k_new,
    uint16_t *__restrict__ window_cache,
    int num_tokens,
    const int *__restrict__ start_pos_ptr,
    int sliding_window,
    int head_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = num_tokens * head_dim;
  if (idx >= total) return;
  int token = idx / head_dim;
  int col = idx - token * head_dim;
  int start_pos = *start_pos_ptr;
  int slot = (start_pos + token) % sliding_window;
  window_cache[slot * head_dim + col] = k_new[token * head_dim + col];
}

// Pointer-array batched SW-window write: ONE launch over N rows whose k_new
// (`k_prepared[head_dim,1]`) and window_cache (per-slot SW ring) buffers are NOT
// contiguous. `k_arr[row]`/`cache_arr[row]` are this row's buffers, this row
// writes its single new key into its own ring at slot `start_pos[row] %
// sliding_window`. Replaces n single-row
// `dsv4_update_window_cache_start_pos_ptr_cuda` calls (num_tokens=1 per row;
// byte-identical per-row write).
__global__ void dsv4_update_window_cache_batched_ptr_kernel(
    const uint16_t *const *__restrict__ k_arr,
    uint16_t *const *__restrict__ cache_arr,
    int n,
    const int *__restrict__ start_pos,
    int sliding_window,
    int head_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n * head_dim;
  if (idx >= total) return;
  int rowi = idx / head_dim;
  int col = idx - rowi * head_dim;
  int slot = start_pos[rowi] % sliding_window;
  cache_arr[rowi][slot * head_dim + col] = k_arr[rowi][col];
}

extern "C" CUresult dsv4_update_window_cache_batched_ptr_cuda(
    const uint16_t *const *k_arr,
    uint16_t *const *cache_arr,
    int n,
    const int *start_pos,
    int sliding_window,
    int head_dim,
    CUstream stream) {
  if (n < 0 || start_pos == nullptr || sliding_window <= 0 || head_dim <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int total = n * head_dim;
  if (total == 0) return CUDA_SUCCESS;
  if (k_arr == nullptr || cache_arr == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int grid = (total + DSV4_ATTN_BLOCK - 1) / DSV4_ATTN_BLOCK;
  dsv4_update_window_cache_batched_ptr_kernel<<<grid, DSV4_ATTN_BLOCK, 0, (cudaStream_t)stream>>>(
      k_arr, cache_arr, n, start_pos, sliding_window, head_dim);
  return (CUresult)cudaGetLastError();
}

extern "C" CUresult dsv4_update_window_cache_cuda(
    const uint16_t *k_new,
    uint16_t *window_cache,
    int num_tokens,
    int start_pos,
    int sliding_window,
    int head_dim,
    CUstream stream) {
  if (num_tokens < 0 || start_pos < 0 || sliding_window <= 0 || head_dim <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int total = num_tokens * head_dim;
  if (total == 0) return CUDA_SUCCESS;
  int grid = (total + DSV4_ATTN_BLOCK - 1) / DSV4_ATTN_BLOCK;
  dsv4_update_window_cache_kernel<<<grid, DSV4_ATTN_BLOCK, 0, (cudaStream_t)stream>>>(
      k_new, window_cache, num_tokens, start_pos, sliding_window, head_dim);
  return (CUresult)cudaGetLastError();
}

extern "C" CUresult dsv4_update_window_cache_start_pos_ptr_cuda(
    const uint16_t *k_new,
    uint16_t *window_cache,
    int num_tokens,
    const int *start_pos_ptr,
    int sliding_window,
    int head_dim,
    CUstream stream) {
  if (num_tokens < 0 || start_pos_ptr == nullptr || sliding_window <= 0 || head_dim <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int total = num_tokens * head_dim;
  if (total == 0) return CUDA_SUCCESS;
  int grid = (total + DSV4_ATTN_BLOCK - 1) / DSV4_ATTN_BLOCK;
  dsv4_update_window_cache_start_pos_ptr_kernel<<<grid, DSV4_ATTN_BLOCK, 0, (cudaStream_t)stream>>>(
      k_new, window_cache, num_tokens, start_pos_ptr, sliding_window, head_dim);
  return (CUresult)cudaGetLastError();
}
