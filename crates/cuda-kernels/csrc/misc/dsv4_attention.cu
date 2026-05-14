#include "common.cuh"
#include <cuda.h>
#include <stdint.h>

#define DSV4_ATTN_BLOCK 256
#define DSV4_ATTN_MAX_HEAD_DIM 1024
#define DSV4_ATTN_MAX_WINDOW 1024
#define DSV4_CSA_MAX_TOPK 512
#define DSV4_PI 3.14159265358979323846f

__device__ __forceinline__ int dsv4_imax(int lhs, int rhs) {
  return lhs > rhs ? lhs : rhs;
}

__device__ __forceinline__ int dsv4_imin(int lhs, int rhs) {
  return lhs < rhs ? lhs : rhs;
}

__device__ __forceinline__ float dsv4_attn_bf16_to_f32(const uint16_t value) {
  return __bfloat162float(*reinterpret_cast<const __nv_bfloat16 *>(&value));
}

__device__ __forceinline__ uint16_t dsv4_attn_f32_to_bf16_bits(const float value) {
  __nv_bfloat16 out = __float2bfloat16(value);
  return *reinterpret_cast<uint16_t *>(&out);
}

__device__ float dsv4_attn_block_sum(float value) {
  __shared__ float warp_sums[DSV4_ATTN_BLOCK / WARP_SIZE];
  value = warp_reduce_sum(value);
  if ((threadIdx.x & (WARP_SIZE - 1)) == 0) {
    warp_sums[threadIdx.x / WARP_SIZE] = value;
  }
  __syncthreads();
  value = threadIdx.x < (DSV4_ATTN_BLOCK / WARP_SIZE) ? warp_sums[threadIdx.x] : 0.0f;
  if (threadIdx.x < WARP_SIZE) {
    value = warp_reduce_sum(value);
  }
  return value;
}

__device__ float dsv4_attn_block_max(float value) {
  __shared__ float warp_max[DSV4_ATTN_BLOCK / WARP_SIZE];
  value = warp_reduce_max(value);
  if ((threadIdx.x & (WARP_SIZE - 1)) == 0) {
    warp_max[threadIdx.x / WARP_SIZE] = value;
  }
  __syncthreads();
  value = threadIdx.x < (DSV4_ATTN_BLOCK / WARP_SIZE) ? warp_max[threadIdx.x] : -INFINITY;
  if (threadIdx.x < WARP_SIZE) {
    value = warp_reduce_max(value);
  }
  return value;
}

__device__ __forceinline__ float dsv4_rope_inv_freq(
    int pair_idx,
    int rope_dim,
    float rope_base,
    int original_seq_len,
    float factor,
    float beta_fast,
    float beta_slow) {
  float inv = powf(rope_base, -((float)(2 * pair_idx) / (float)rope_dim));
  if (original_seq_len <= 0) {
    return inv;
  }
  float low_f = floorf((float)rope_dim *
                      logf((float)original_seq_len / (beta_fast * 2.0f * DSV4_PI)) /
                      (2.0f * logf(rope_base)));
  float high_f = ceilf((float)rope_dim *
                       logf((float)original_seq_len / (beta_slow * 2.0f * DSV4_PI)) /
                       (2.0f * logf(rope_base)));
  int low = dsv4_imax(0, (int)low_f);
  int high = dsv4_imin(dsv4_imax(0, rope_dim - 1), (int)high_f);
  float denom = low == high ? 0.001f : (float)(high - low);
  float ramp = fminf(fmaxf(((float)pair_idx - (float)low) / denom, 0.0f), 1.0f);
  float smooth = 1.0f - ramp;
  return inv / factor * (1.0f - smooth) + inv * smooth;
}

__device__ __forceinline__ void dsv4_apply_rope_pair(
    float a,
    float b,
    int pair_idx,
    int abs_pos,
    int rope_dim,
    float rope_base,
    int original_seq_len,
    float factor,
    float beta_fast,
    float beta_slow,
    float sign,
    float *out_a,
    float *out_b) {
  float inv = dsv4_rope_inv_freq(
      pair_idx, rope_dim, rope_base, original_seq_len, factor, beta_fast, beta_slow);
  float angle = (float)abs_pos * inv;
  float c = cosf(angle);
  float s = sign * sinf(angle);
  *out_a = a * c - b * s;
  *out_b = b * c + a * s;
}

__global__ void dsv4_prepare_q_kernel(
    const uint16_t *__restrict__ q_raw,
    uint16_t *__restrict__ q_out,
    int num_tokens,
    int local_heads,
    int head_dim,
    int rope_dim,
    int start_pos,
    float rms_eps,
    float rope_base,
    int original_seq_len,
    float factor,
    float beta_fast,
    float beta_slow) {
  int row = blockIdx.x;
  if (row >= num_tokens * local_heads) return;
  int token = row / local_heads;
  int head = row - token * local_heads;
  int local_width = local_heads * head_dim;
  int base = token * local_width + head * head_dim;

  float sumsq = 0.0f;
  for (int col = threadIdx.x; col < head_dim; col += blockDim.x) {
    float value = dsv4_attn_bf16_to_f32(q_raw[base + col]);
    sumsq += value * value;
  }
  sumsq = dsv4_attn_block_sum(sumsq);
  __shared__ float scale;
  if (threadIdx.x == 0) {
    scale = rsqrtf(sumsq / fmaxf((float)head_dim, 1.0f) + rms_eps);
  }
  __syncthreads();

  int rope_start = head_dim - rope_dim;
  for (int col = threadIdx.x; col < head_dim; col += blockDim.x) {
    float value = dsv4_attn_bf16_to_f32(q_raw[base + col]) * scale;
    if (rope_dim > 0 && col >= rope_start) {
      int local = col - rope_start;
      int pair = local / 2;
      int pair_col = rope_start + pair * 2;
      float a = dsv4_attn_bf16_to_f32(q_raw[base + pair_col]) * scale;
      float b = dsv4_attn_bf16_to_f32(q_raw[base + pair_col + 1]) * scale;
      float out_a;
      float out_b;
      dsv4_apply_rope_pair(
          a, b, pair, start_pos + token, rope_dim, rope_base, original_seq_len,
          factor, beta_fast, beta_slow, 1.0f, &out_a, &out_b);
      value = (local & 1) == 0 ? out_a : out_b;
    }
    q_out[base + col] = dsv4_attn_f32_to_bf16_bits(value);
  }
}

__global__ void dsv4_prepare_k_kernel(
    const uint16_t *__restrict__ k_raw,
    uint16_t *__restrict__ k_out,
    int num_tokens,
    int head_dim,
    int rope_dim,
    int start_pos,
    float rope_base,
    int original_seq_len,
    float factor,
    float beta_fast,
    float beta_slow) {
  int token = blockIdx.x;
  if (token >= num_tokens) return;
  int base = token * head_dim;
  int rope_start = head_dim - rope_dim;
  for (int col = threadIdx.x; col < head_dim; col += blockDim.x) {
    float value = dsv4_attn_bf16_to_f32(k_raw[base + col]);
    if (rope_dim > 0 && col >= rope_start) {
      int local = col - rope_start;
      int pair = local / 2;
      int pair_col = rope_start + pair * 2;
      float a = dsv4_attn_bf16_to_f32(k_raw[base + pair_col]);
      float b = dsv4_attn_bf16_to_f32(k_raw[base + pair_col + 1]);
      float out_a;
      float out_b;
      dsv4_apply_rope_pair(
          a, b, pair, start_pos + token, rope_dim, rope_base, original_seq_len,
          factor, beta_fast, beta_slow, 1.0f, &out_a, &out_b);
      value = (local & 1) == 0 ? out_a : out_b;
    }
    k_out[base + col] = dsv4_attn_f32_to_bf16_bits(value);
  }
}

extern "C" CUresult dsv4_prepare_qk_cuda(
    const uint16_t *q_raw,
    const uint16_t *k_raw,
    uint16_t *q_out,
    uint16_t *k_out,
    int num_tokens,
    int local_heads,
    int head_dim,
    int rope_dim,
    int start_pos,
    float rms_eps,
    float rope_base,
    int original_seq_len,
    float factor,
    float beta_fast,
    float beta_slow,
    CUstream stream) {
  if (num_tokens < 0 || local_heads <= 0 || head_dim <= 0 || rope_dim < 0 ||
      rope_dim > head_dim || start_pos < 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (num_tokens == 0) return CUDA_SUCCESS;
  dsv4_prepare_q_kernel<<<num_tokens * local_heads, DSV4_ATTN_BLOCK, 0, (cudaStream_t)stream>>>(
      q_raw, q_out, num_tokens, local_heads, head_dim, rope_dim, start_pos,
      rms_eps, rope_base, original_seq_len, factor, beta_fast, beta_slow);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return (CUresult)err;
  dsv4_prepare_k_kernel<<<num_tokens, DSV4_ATTN_BLOCK, 0, (cudaStream_t)stream>>>(
      k_raw, k_out, num_tokens, head_dim, rope_dim, start_pos, rope_base,
      original_seq_len, factor, beta_fast, beta_slow);
  return (CUresult)cudaGetLastError();
}

__device__ __forceinline__ float dsv4_swa_key_value(
    const uint16_t *__restrict__ k_new,
    const uint16_t *__restrict__ window_cache,
    int key_pos,
    int start_pos,
    int sliding_window,
    int head_dim,
    int col) {
  if (key_pos >= start_pos) {
    int local = key_pos - start_pos;
    return dsv4_attn_bf16_to_f32(k_new[local * head_dim + col]);
  }
  int slot = key_pos % sliding_window;
  return dsv4_attn_bf16_to_f32(window_cache[slot * head_dim + col]);
}

__global__ void dsv4_swa_attention_kernel(
    const uint16_t *__restrict__ q,
    const uint16_t *__restrict__ k_new,
    const uint16_t *__restrict__ window_cache,
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
    float beta_slow) {
  int row = blockIdx.x;
  if (row >= num_tokens * local_heads) return;
  int token = row / local_heads;
  int head = row - token * local_heads;
  int local_width = local_heads * head_dim;
  int abs_pos = start_pos + token;
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
      float kv = dsv4_swa_key_value(k_new, window_cache, key_pos, start_pos, sliding_window, head_dim, col);
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
      float kv = dsv4_swa_key_value(k_new, window_cache, key_pos, start_pos, sliding_window, head_dim, col);
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
}

extern "C" CUresult dsv4_swa_attention_cuda(
    const uint16_t *q,
    const uint16_t *k_new,
    const uint16_t *window_cache,
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
      original_seq_len, factor, beta_fast, beta_slow);
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

__device__ __forceinline__ float dsv4_compressor_raw_value(
    const uint16_t *__restrict__ raw,
    const uint16_t *__restrict__ pending,
    int abs_pos,
    int start_pos,
    int block_start,
    int width,
    int col) {
  if (abs_pos < start_pos) {
    int pending_pos = abs_pos - block_start;
    return dsv4_attn_bf16_to_f32(pending[pending_pos * width + col]);
  }
  return dsv4_attn_bf16_to_f32(raw[(abs_pos - start_pos) * width + col]);
}

__device__ __forceinline__ float dsv4_compressor_score_value(
    const uint16_t *__restrict__ raw,
    const uint16_t *__restrict__ pending,
    const uint16_t *__restrict__ ape,
    int abs_pos,
    int start_pos,
    int block_start,
    int ratio,
    int width,
    int col) {
  if (abs_pos < start_pos) {
    int pending_pos = abs_pos - block_start;
    return dsv4_attn_bf16_to_f32(pending[pending_pos * width + col]);
  }
  return dsv4_attn_bf16_to_f32(raw[(abs_pos - start_pos) * width + col]) +
         dsv4_attn_bf16_to_f32(ape[(abs_pos % ratio) * width + col]);
}

__global__ void dsv4_compressor_update_kernel(
    const uint16_t *__restrict__ kv_raw,
    const uint16_t *__restrict__ score_raw,
    const uint16_t *__restrict__ ape,
    const uint16_t *__restrict__ norm,
    uint16_t *__restrict__ pending_kv,
    uint16_t *__restrict__ pending_score,
    uint16_t *__restrict__ prev_overlap_kv,
    uint16_t *__restrict__ prev_overlap_score,
    uint16_t *__restrict__ compressed,
    int num_tokens,
    int start_pos,
    int pending_len,
    int compressed_base,
    int head_dim,
    int ratio,
    int width,
    int overlap,
    int has_prev_overlap,
    float eps,
    int rope_dim,
    float rope_base,
    int original_seq_len,
    float factor,
    float beta_fast,
    float beta_slow) {
  __shared__ float row[DSV4_ATTN_MAX_HEAD_DIM];
  int total = pending_len + num_tokens;
  int completed = total / ratio;
  int block_start0 = start_pos - pending_len;

  for (int block = 0; block < completed; ++block) {
    int block_start = block_start0 + block * ratio;
    int block_end = block_start + ratio - 1;
    for (int col = threadIdx.x; col < head_dim; col += blockDim.x) {
      float max_logit = -INFINITY;
      float logits[256];
      int count = overlap ? 2 * ratio : ratio;
      for (int pos = 0; pos < count; ++pos) {
        float logit;
        if (overlap && pos < ratio) {
          logit = (has_prev_overlap || block > 0)
                      ? dsv4_attn_bf16_to_f32(prev_overlap_score[pos * head_dim + col])
                      : -INFINITY;
        } else {
          int local_pos = overlap ? (pos - ratio) : pos;
          int abs_pos = block_start + local_pos;
          int score_col = overlap ? (head_dim + col) : col;
          logit = dsv4_compressor_score_value(
              score_raw, pending_score, ape, abs_pos, start_pos, block_start,
              ratio, width, score_col);
        }
        logits[pos] = logit;
        max_logit = fmaxf(max_logit, logit);
      }
      float denom = 0.0f;
      for (int pos = 0; pos < count; ++pos) {
        float value = expf(logits[pos] - max_logit);
        logits[pos] = value;
        denom += value;
      }
      float acc = 0.0f;
      if (isfinite(max_logit) && denom > 0.0f) {
        for (int pos = 0; pos < count; ++pos) {
          float value;
          if (overlap && pos < ratio) {
            value = (has_prev_overlap || block > 0)
                        ? dsv4_attn_bf16_to_f32(prev_overlap_kv[pos * head_dim + col])
                        : 0.0f;
          } else {
            int local_pos = overlap ? (pos - ratio) : pos;
            int abs_pos = block_start + local_pos;
            int kv_col = overlap ? (head_dim + col) : col;
            value = dsv4_compressor_raw_value(
                kv_raw, pending_kv, abs_pos, start_pos, block_start, width, kv_col);
          }
          acc += (logits[pos] / denom) * value;
        }
      }
      row[col] = acc;
    }
    __syncthreads();

    float sumsq = 0.0f;
    for (int col = threadIdx.x; col < head_dim; col += blockDim.x) {
      sumsq += row[col] * row[col];
    }
    sumsq = dsv4_attn_block_sum(sumsq);
    __shared__ float norm_scale;
    if (threadIdx.x == 0) {
      norm_scale = rsqrtf(sumsq / fmaxf((float)head_dim, 1.0f) + eps);
    }
    __syncthreads();

    for (int col = threadIdx.x; col < head_dim; col += blockDim.x) {
      float value = row[col] * norm_scale * dsv4_attn_bf16_to_f32(norm[col]);
      if (rope_dim > 0 && col >= head_dim - rope_dim) {
        int local = col - (head_dim - rope_dim);
        int pair = local / 2;
        int pair_col = head_dim - rope_dim + pair * 2;
        float a = row[pair_col] * norm_scale * dsv4_attn_bf16_to_f32(norm[pair_col]);
        float b = row[pair_col + 1] * norm_scale * dsv4_attn_bf16_to_f32(norm[pair_col + 1]);
        float out_a;
        float out_b;
        dsv4_apply_rope_pair(
            a, b, pair, block_end, rope_dim, rope_base, original_seq_len,
            factor, beta_fast, beta_slow, 1.0f, &out_a, &out_b);
        value = (local & 1) == 0 ? out_a : out_b;
      }
      compressed[(compressed_base + block) * head_dim + col] =
          dsv4_attn_f32_to_bf16_bits(value);
    }
    __syncthreads();

    if (overlap) {
      for (int col = threadIdx.x; col < head_dim; col += blockDim.x) {
        for (int pos = 0; pos < ratio; ++pos) {
          int abs_pos = block_start + pos;
          float kv = dsv4_compressor_raw_value(
              kv_raw, pending_kv, abs_pos, start_pos, block_start, width, col);
          float score = dsv4_compressor_score_value(
              score_raw, pending_score, ape, abs_pos, start_pos, block_start,
              ratio, width, col);
          prev_overlap_kv[pos * head_dim + col] = dsv4_attn_f32_to_bf16_bits(kv);
          prev_overlap_score[pos * head_dim + col] = dsv4_attn_f32_to_bf16_bits(score);
        }
      }
    }
    __syncthreads();
  }

  int new_pending = total - completed * ratio;
  int tail_start = start_pos + num_tokens - new_pending;
  for (int idx = threadIdx.x; idx < new_pending * width; idx += blockDim.x) {
    int pos = idx / width;
    int col = idx - pos * width;
    int abs_pos = tail_start + pos;
    float kv = dsv4_compressor_raw_value(
        kv_raw, pending_kv, abs_pos, start_pos, block_start0, width, col);
    float score = dsv4_compressor_score_value(
        score_raw, pending_score, ape, abs_pos, start_pos, block_start0,
        ratio, width, col);
    pending_kv[idx] = dsv4_attn_f32_to_bf16_bits(kv);
    pending_score[idx] = dsv4_attn_f32_to_bf16_bits(score);
  }
}

extern "C" CUresult dsv4_compressor_update_cuda(
    const uint16_t *kv_raw,
    const uint16_t *score_raw,
    const uint16_t *ape,
    const uint16_t *norm,
    uint16_t *pending_kv,
    uint16_t *pending_score,
    uint16_t *prev_overlap_kv,
    uint16_t *prev_overlap_score,
    uint16_t *compressed,
    int num_tokens,
    int start_pos,
    int pending_len,
    int compressed_base,
    int head_dim,
    int ratio,
    int width,
    int overlap,
    int has_prev_overlap,
    float eps,
    int rope_dim,
    float rope_base,
    int original_seq_len,
    float factor,
    float beta_fast,
    float beta_slow,
    CUstream stream) {
  if (num_tokens < 0 || start_pos < 0 || pending_len < 0 || compressed_base < 0 ||
      head_dim <= 0 || head_dim > DSV4_ATTN_MAX_HEAD_DIM || ratio <= 0 ||
      ratio > 256 || width < head_dim || rope_dim < 0 || rope_dim > head_dim) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  dsv4_compressor_update_kernel<<<1, DSV4_ATTN_BLOCK, 0, (cudaStream_t)stream>>>(
      kv_raw, score_raw, ape, norm, pending_kv, pending_score, prev_overlap_kv,
      prev_overlap_score, compressed, num_tokens, start_pos, pending_len,
      compressed_base, head_dim, ratio, width, overlap, has_prev_overlap, eps,
      rope_dim, rope_base, original_seq_len, factor, beta_fast, beta_slow);
  return (CUresult)cudaGetLastError();
}

#define DSV4_ATTN_MAX_KEYS 9216
#define DSV4_CSA_SORT_MAX_KEYS 4096
#define DSV4_CSA_INVALID_INDEX 2147483647

__device__ __forceinline__ bool dsv4_csa_rhs_better(
    float lhs_score,
    int lhs_index,
    float rhs_score,
    int rhs_index) {
  return rhs_score > lhs_score || (rhs_score == lhs_score && rhs_index < lhs_index);
}

__device__ void dsv4_csa_bitonic_sort_desc(
    float *__restrict__ scores,
    int *__restrict__ indices,
    int sort_len) {
  for (int width = 2; width <= sort_len; width <<= 1) {
    for (int stride = width >> 1; stride > 0; stride >>= 1) {
      for (int idx = threadIdx.x; idx < sort_len; idx += blockDim.x) {
        int other = idx ^ stride;
        if (other <= idx) continue;
        bool descending = (idx & width) == 0;
        float lhs_score = scores[idx];
        int lhs_index = indices[idx];
        float rhs_score = scores[other];
        int rhs_index = indices[other];
        bool rhs_better = dsv4_csa_rhs_better(lhs_score, lhs_index, rhs_score, rhs_index);
        bool lhs_better = dsv4_csa_rhs_better(rhs_score, rhs_index, lhs_score, lhs_index);
        bool swap = descending ? rhs_better : lhs_better;
        if (swap) {
          scores[idx] = rhs_score;
          indices[idx] = rhs_index;
          scores[other] = lhs_score;
          indices[other] = lhs_index;
        }
      }
      __syncthreads();
    }
  }
}

__device__ __forceinline__ float dsv4_csa_score_block(
    const uint16_t *__restrict__ q,
    const uint16_t *__restrict__ weights,
    const uint16_t *__restrict__ keys,
    int token,
    int block_idx,
    int q_width,
    int local_heads,
    int index_dim,
    float score_scale) {
  float score = 0.0f;
  for (int head = 0; head < local_heads; ++head) {
    float dotv = 0.0f;
    int q_base = token * q_width + head * index_dim;
    int key_base = block_idx * index_dim;
    for (int col = 0; col < index_dim; ++col) {
      dotv += dsv4_attn_bf16_to_f32(q[q_base + col]) *
              dsv4_attn_bf16_to_f32(keys[key_base + col]);
    }
    float weight = dsv4_attn_bf16_to_f32(weights[token * local_heads + head]) * score_scale;
    score += weight * fmaxf(dotv, 0.0f);
  }
  return score;
}

__global__ void dsv4_hybrid_attention_kernel(
    const uint16_t *__restrict__ q,
    const uint16_t *__restrict__ k_new,
    const uint16_t *__restrict__ window_cache,
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
    int selected_topk) {
  int row = blockIdx.x;
  if (row >= num_tokens * local_heads) return;
  int token = row / local_heads;
  int head = row - token * local_heads;
  int local_width = local_heads * head_dim;
  int abs_pos = start_pos + token;
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
      comp_keys = dsv4_imin(compressed_count, (abs_pos + 1) / compress_ratio);
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
        kv = dsv4_swa_key_value(k_new, window_cache, logical_idx, start_pos, sliding_window, head_dim, col);
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
                     ? dsv4_swa_key_value(k_new, window_cache, logical_idx, start_pos, sliding_window, head_dim, col)
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
}

extern "C" CUresult dsv4_hybrid_attention_cuda(
    const uint16_t *q,
    const uint16_t *k_new,
    const uint16_t *window_cache,
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
      compress_ratio, compressed_count, selected_topk);
  return (CUresult)cudaGetLastError();
}

__global__ void dsv4_csa_select_kernel(
    const uint16_t *__restrict__ q,
    const uint16_t *__restrict__ weights,
    const uint16_t *__restrict__ keys,
    int32_t *__restrict__ selected,
    int num_tokens,
    int q_width,
    int local_heads,
    int index_dim,
    int key_count,
    int ratio,
    int topk,
    float score_scale,
    int start_pos) {
  int token = blockIdx.x;
  if (token >= num_tokens) return;
  int abs_pos = start_pos + token;
  int available = dsv4_imin(key_count, (abs_pos + 1) / ratio);

  __shared__ float sort_scores[DSV4_CSA_SORT_MAX_KEYS];
  __shared__ int sort_indices[DSV4_CSA_SORT_MAX_KEYS];
  __shared__ float global_scores[DSV4_CSA_MAX_TOPK];
  __shared__ int global_indices[DSV4_CSA_MAX_TOPK];
  __shared__ float tile_scores[DSV4_CSA_MAX_TOPK];
  __shared__ int tile_indices[DSV4_CSA_MAX_TOPK];

  if (available <= DSV4_CSA_SORT_MAX_KEYS) {
    int sort_len = 1;
    while (sort_len < available) sort_len <<= 1;
    for (int idx = threadIdx.x; idx < sort_len; idx += blockDim.x) {
      float score = -INFINITY;
      int out_idx = DSV4_CSA_INVALID_INDEX;
      if (idx < available) {
        score = dsv4_csa_score_block(
            q, weights, keys, token, idx, q_width, local_heads, index_dim, score_scale);
        if (isfinite(score)) {
          out_idx = idx;
        } else {
          score = -INFINITY;
        }
      }
      sort_scores[idx] = score;
      sort_indices[idx] = out_idx;
    }
    __syncthreads();

    dsv4_csa_bitonic_sort_desc(sort_scores, sort_indices, sort_len);

    for (int k = threadIdx.x; k < topk; k += blockDim.x) {
      int idx = k < available ? sort_indices[k] : -1;
      selected[token * topk + k] = idx == DSV4_CSA_INVALID_INDEX ? -1 : idx;
    }
    return;
  }

  for (int k = threadIdx.x; k < topk; k += blockDim.x) {
    global_scores[k] = -INFINITY;
    global_indices[k] = DSV4_CSA_INVALID_INDEX;
  }
  __syncthreads();

  for (int tile_start = 0; tile_start < available; tile_start += DSV4_CSA_SORT_MAX_KEYS) {
    int tile_len = dsv4_imin(DSV4_CSA_SORT_MAX_KEYS, available - tile_start);
    int sort_len = 1;
    while (sort_len < tile_len) sort_len <<= 1;

    for (int idx = threadIdx.x; idx < sort_len; idx += blockDim.x) {
      float score = -INFINITY;
      int out_idx = DSV4_CSA_INVALID_INDEX;
      if (idx < tile_len) {
        int block_idx = tile_start + idx;
        score = dsv4_csa_score_block(
            q, weights, keys, token, block_idx, q_width, local_heads, index_dim, score_scale);
        if (isfinite(score)) {
          out_idx = block_idx;
        } else {
          score = -INFINITY;
        }
      }
      sort_scores[idx] = score;
      sort_indices[idx] = out_idx;
    }
    __syncthreads();
    dsv4_csa_bitonic_sort_desc(sort_scores, sort_indices, sort_len);

    int tile_take = dsv4_imin(topk, tile_len);
    for (int k = threadIdx.x; k < tile_take; k += blockDim.x) {
      tile_scores[k] = sort_scores[k];
      tile_indices[k] = sort_indices[k];
    }
    __syncthreads();

    int merge_len = topk + tile_take;
    int merge_sort_len = 1;
    while (merge_sort_len < merge_len) merge_sort_len <<= 1;
    for (int idx = threadIdx.x; idx < merge_sort_len; idx += blockDim.x) {
      if (idx < topk) {
        sort_scores[idx] = global_scores[idx];
        sort_indices[idx] = global_indices[idx];
      } else if (idx < merge_len) {
        int tile_idx = idx - topk;
        sort_scores[idx] = tile_scores[tile_idx];
        sort_indices[idx] = tile_indices[tile_idx];
      } else {
        sort_scores[idx] = -INFINITY;
        sort_indices[idx] = DSV4_CSA_INVALID_INDEX;
      }
    }
    __syncthreads();
    dsv4_csa_bitonic_sort_desc(sort_scores, sort_indices, merge_sort_len);

    for (int k = threadIdx.x; k < topk; k += blockDim.x) {
      global_scores[k] = sort_scores[k];
      global_indices[k] = sort_indices[k];
    }
    __syncthreads();
  }

  for (int k = threadIdx.x; k < topk; k += blockDim.x) {
    int idx = global_indices[k];
    selected[token * topk + k] = idx == DSV4_CSA_INVALID_INDEX ? -1 : idx;
  }
}

extern "C" CUresult dsv4_csa_select_cuda(
    const uint16_t *q,
    const uint16_t *weights,
    const uint16_t *keys,
    int32_t *selected,
    int num_tokens,
    int q_width,
    int local_heads,
    int index_dim,
    int key_count,
    int ratio,
    int topk,
    float score_scale,
    int start_pos,
    CUstream stream) {
  if (num_tokens < 0 || q_width <= 0 || local_heads <= 0 || index_dim <= 0 ||
      key_count < 0 || ratio <= 0 || topk <= 0 || topk > DSV4_CSA_MAX_TOPK ||
      start_pos < 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (num_tokens == 0) return CUDA_SUCCESS;
  dsv4_csa_select_kernel<<<num_tokens, DSV4_ATTN_BLOCK, 0, (cudaStream_t)stream>>>(
      q, weights, keys, selected, num_tokens, q_width, local_heads, index_dim,
      key_count, ratio, topk, score_scale, start_pos);
  return (CUresult)cudaGetLastError();
}
