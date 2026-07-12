#include "dsv4_attention_common.cuh"

#define DSV4_ATTN_MAX_KEYS 9216

__global__ void dsv4_hybrid_attention_kernel(
    const uint16_t *__restrict__ q,
    const uint16_t *__restrict__ k_new,
    uint16_t *__restrict__ window_cache,
    const uint16_t *__restrict__ compressed,
    const int32_t *__restrict__ selected,
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
    int mode,
    int compress_ratio,
    int compressed_count,
    int selected_topk,
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
  int sw_count = abs_pos - sw_start + 1;

  __shared__ float logits[DSV4_ATTN_MAX_KEYS];
  __shared__ float denom_shared;
  __shared__ float max_shared;
  __shared__ float out_vec[DSV4_ATTN_MAX_HEAD_DIM];
  __shared__ int total_keys_shared;
  __shared__ int comp_keys_shared;

  if (threadIdx.x == 0) {
    int comp_keys = 0;
    if (mode == 1) {
      comp_keys = selected_topk;
    } else if (mode == 2) {
      // HCA causal block count = floor(abs_pos / ratio), matching the CPU
      // reference gate `block_end = block*ratio + ratio-1 < t` (reference.rs:382,
      // which yields exactly floor(t/ratio) kept blocks). The prior `(abs_pos+1)`
      // admitted one extra (straddling) block whenever `abs_pos+1` was a multiple
      // of `ratio` — e.g. at abs_pos=127 with ratio=128, precisely the sliding-
      // window boundary where long-context attention first engages.
      comp_keys = dsv4_imin(compressed_count, abs_pos / compress_ratio);
    }
    comp_keys = dsv4_imin(comp_keys, DSV4_ATTN_MAX_KEYS);
    int total_keys = dsv4_imin(comp_keys + sw_count, DSV4_ATTN_MAX_KEYS);
    comp_keys_shared = comp_keys;
    total_keys_shared = total_keys;
  }
  __syncthreads();

  int q_base = token * local_width + head * head_dim;
  for (int key_idx = threadIdx.x; key_idx < total_keys_shared; key_idx += blockDim.x) {
    float acc = 0.0f;
    bool is_comp = key_idx < comp_keys_shared;
    int logical_idx;
    if (is_comp && mode == 1) {
      logical_idx = selected[token * selected_topk + key_idx];
      int block_end = logical_idx * compress_ratio + (compress_ratio - 1);
      if (logical_idx < 0 || logical_idx >= compressed_count || block_end > abs_pos) {
        logits[key_idx] = -INFINITY;
        continue;
      }
    } else if (is_comp) {
      logical_idx = key_idx;
    } else {
      logical_idx = sw_start + (key_idx - comp_keys_shared);
    }
    for (int col = 0; col < head_dim; ++col) {
      float qv = dsv4_attn_bf16_to_f32(q[q_base + col]);
      float kv;
      if (!is_comp) {
        kv = dsv4_swa_key_value(k_new, window_cache, logical_idx, base_start_pos, sliding_window, head_dim, col);
      } else {
        kv = dsv4_attn_bf16_to_f32(compressed[logical_idx * head_dim + col]);
      }
      acc += qv * kv;
    }
    logits[key_idx] = acc * scale_value;
  }
  __syncthreads();

  float local_max = -INFINITY;
  for (int key_idx = threadIdx.x; key_idx < total_keys_shared; key_idx += blockDim.x) {
    local_max = fmaxf(local_max, logits[key_idx]);
  }
  float sink = dsv4_attn_bf16_to_f32(attn_sink[sink_offset + head]);
  if (threadIdx.x == 0) local_max = fmaxf(local_max, sink);
  local_max = dsv4_attn_block_max(local_max);
  if (threadIdx.x == 0) max_shared = local_max;
  __syncthreads();

  float denom = 0.0f;
  for (int key_idx = threadIdx.x; key_idx < total_keys_shared; key_idx += blockDim.x) {
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
    for (int key_idx = 0; key_idx < total_keys_shared; ++key_idx) {
      if (!isfinite(logits[key_idx]) || logits[key_idx] == 0.0f) continue;
      bool is_comp = key_idx < comp_keys_shared;
      int logical_idx = is_comp && mode == 1
                            ? selected[token * selected_topk + key_idx]
                            : (is_comp ? key_idx : sw_start + (key_idx - comp_keys_shared));
      float kv = !is_comp
                     ? dsv4_swa_key_value(k_new, window_cache, logical_idx, base_start_pos, sliding_window, head_dim, col)
                     : dsv4_attn_bf16_to_f32(compressed[logical_idx * head_dim + col]);
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

extern "C" CUresult dsv4_hybrid_attention_cuda(
    const uint16_t *q,
    const uint16_t *k_new,
    uint16_t *window_cache,
    const uint16_t *compressed,
    const int32_t *selected,
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
    int mode,
    int compress_ratio,
    int compressed_count,
    int selected_topk,
    int write_window_cache,
    CUstream stream) {
  if (num_tokens < 0 || local_heads <= 0 || head_dim <= 0 ||
      head_dim > DSV4_ATTN_MAX_HEAD_DIM || sliding_window <= 0 ||
      rope_dim < 0 || rope_dim > head_dim || mode < 0 || mode > 2 ||
      compress_ratio < 0 || compressed_count < 0 || selected_topk < 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (num_tokens == 0) return CUDA_SUCCESS;
  dsv4_hybrid_attention_kernel<<<num_tokens * local_heads, DSV4_ATTN_BLOCK, 0, (cudaStream_t)stream>>>(
      q, k_new, window_cache, compressed, selected, attn_sink, out, num_tokens,
      local_heads, head_dim, sliding_window, start_pos, sink_offset, scale_value,
      rope_dim, rope_base, original_seq_len, factor, beta_fast, beta_slow, mode,
      compress_ratio, compressed_count, selected_topk, write_window_cache, nullptr);
  return (CUresult)cudaGetLastError();
}

extern "C" CUresult dsv4_hybrid_attention_start_pos_ptr_cuda(
    const uint16_t *q,
    const uint16_t *k_new,
    uint16_t *window_cache,
    const uint16_t *compressed,
    const int32_t *selected,
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
    int mode,
    int compress_ratio,
    int compressed_count,
    int selected_topk,
    int write_window_cache,
    CUstream stream) {
  if (num_tokens < 0 || local_heads <= 0 || head_dim <= 0 ||
      head_dim > DSV4_ATTN_MAX_HEAD_DIM || sliding_window <= 0 ||
      rope_dim < 0 || rope_dim > head_dim || mode < 0 || mode > 2 ||
      compress_ratio < 0 || compressed_count < 0 || selected_topk < 0 ||
      start_pos_ptr == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (num_tokens == 0) return CUDA_SUCCESS;
  dsv4_hybrid_attention_kernel<<<num_tokens * local_heads, DSV4_ATTN_BLOCK, 0, (cudaStream_t)stream>>>(
      q, k_new, window_cache, compressed, selected, attn_sink, out, num_tokens,
      local_heads, head_dim, sliding_window, 0, sink_offset, scale_value,
      rope_dim, rope_base, original_seq_len, factor, beta_fast, beta_slow, mode,
      compress_ratio, compressed_count, selected_topk, write_window_cache, start_pos_ptr);
  return (CUresult)cudaGetLastError();
}
