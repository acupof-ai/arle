#include "dsv4_attention_common.cuh"

__global__ void dsv4_prepare_q_kernel(
    const uint16_t *__restrict__ q_raw,
    uint16_t *__restrict__ q_out,
    int num_tokens,
    int local_heads,
    int head_dim,
    int rope_dim,
    int start_pos,
    const int *__restrict__ start_pos_ptr,
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
  int base_start_pos = dsv4_graph_start_pos(start_pos, start_pos_ptr);

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
          a, b, pair, base_start_pos + token, rope_dim, rope_base, original_seq_len,
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
    const int *__restrict__ start_pos_ptr,
    float rope_base,
    int original_seq_len,
    float factor,
    float beta_fast,
    float beta_slow) {
  int token = blockIdx.x;
  if (token >= num_tokens) return;
  int base = token * head_dim;
  int base_start_pos = dsv4_graph_start_pos(start_pos, start_pos_ptr);
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
          a, b, pair, base_start_pos + token, rope_dim, rope_base, original_seq_len,
          factor, beta_fast, beta_slow, 1.0f, &out_a, &out_b);
      value = (local & 1) == 0 ? out_a : out_b;
    }
    k_out[base + col] = dsv4_attn_f32_to_bf16_bits(value);
  }
}

__global__ void dsv4_prepare_qk_fused_kernel(
    const uint16_t *__restrict__ q_raw,
    const uint16_t *__restrict__ k_raw,
    uint16_t *__restrict__ q_out,
    uint16_t *__restrict__ k_out,
    int num_tokens,
    int local_heads,
    int head_dim,
    int rope_dim,
    int start_pos,
    const int *__restrict__ start_pos_ptr,
    float rms_eps,
    float rope_base,
    int original_seq_len,
    float factor,
    float beta_fast,
    float beta_slow) {
  int row = blockIdx.x;
  int q_rows = num_tokens * local_heads;
  int base_start_pos = dsv4_graph_start_pos(start_pos, start_pos_ptr);
  if (row < q_rows) {
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
            a, b, pair, base_start_pos + token, rope_dim, rope_base, original_seq_len,
            factor, beta_fast, beta_slow, 1.0f, &out_a, &out_b);
        value = (local & 1) == 0 ? out_a : out_b;
      }
      q_out[base + col] = dsv4_attn_f32_to_bf16_bits(value);
    }
    return;
  }

  int token = row - q_rows;
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
          a, b, pair, base_start_pos + token, rope_dim, rope_base, original_seq_len,
          factor, beta_fast, beta_slow, 1.0f, &out_a, &out_b);
      value = (local & 1) == 0 ? out_a : out_b;
    }
    k_out[base + col] = dsv4_attn_f32_to_bf16_bits(value);
  }
}

__global__ void dsv4_prepare_qk_fused_batch_start_pos_kernel(
    const uint16_t *__restrict__ q_raw,
    const uint16_t *__restrict__ k_raw,
    uint16_t *__restrict__ q_out,
    uint16_t *__restrict__ k_out,
    int num_tokens,
    int local_heads,
    int head_dim,
    int rope_dim,
    const int *__restrict__ start_pos,
    float rms_eps,
    float rope_base,
    int original_seq_len,
    float factor,
    float beta_fast,
    float beta_slow) {
  int row = blockIdx.x;
  int q_rows = num_tokens * local_heads;
  if (row < q_rows) {
    int token = row / local_heads;
    int head = row - token * local_heads;
    int local_width = local_heads * head_dim;
    int base = token * local_width + head * head_dim;
    int abs_pos = start_pos[token];

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
            a, b, pair, abs_pos, rope_dim, rope_base, original_seq_len,
            factor, beta_fast, beta_slow, 1.0f, &out_a, &out_b);
        value = (local & 1) == 0 ? out_a : out_b;
      }
      q_out[base + col] = dsv4_attn_f32_to_bf16_bits(value);
    }
    return;
  }

  int token = row - q_rows;
  if (token >= num_tokens) return;
  int base = token * head_dim;
  int abs_pos = start_pos[token];
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
          a, b, pair, abs_pos, rope_dim, rope_base, original_seq_len,
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
      nullptr, rms_eps, rope_base, original_seq_len, factor, beta_fast, beta_slow);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return (CUresult)err;
  dsv4_prepare_k_kernel<<<num_tokens, DSV4_ATTN_BLOCK, 0, (cudaStream_t)stream>>>(
      k_raw, k_out, num_tokens, head_dim, rope_dim, start_pos, nullptr, rope_base,
      original_seq_len, factor, beta_fast, beta_slow);
  return (CUresult)cudaGetLastError();
}

extern "C" CUresult dsv4_prepare_qk_start_pos_ptr_cuda(
    const uint16_t *q_raw,
    const uint16_t *k_raw,
    uint16_t *q_out,
    uint16_t *k_out,
    int num_tokens,
    int local_heads,
    int head_dim,
    int rope_dim,
    const int *start_pos_ptr,
    float rms_eps,
    float rope_base,
    int original_seq_len,
    float factor,
    float beta_fast,
    float beta_slow,
    CUstream stream) {
  if (num_tokens < 0 || local_heads <= 0 || head_dim <= 0 || rope_dim < 0 ||
      rope_dim > head_dim || start_pos_ptr == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (num_tokens == 0) return CUDA_SUCCESS;
  dsv4_prepare_q_kernel<<<num_tokens * local_heads, DSV4_ATTN_BLOCK, 0, (cudaStream_t)stream>>>(
      q_raw, q_out, num_tokens, local_heads, head_dim, rope_dim, 0,
      start_pos_ptr, rms_eps, rope_base, original_seq_len, factor, beta_fast, beta_slow);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return (CUresult)err;
  dsv4_prepare_k_kernel<<<num_tokens, DSV4_ATTN_BLOCK, 0, (cudaStream_t)stream>>>(
      k_raw, k_out, num_tokens, head_dim, rope_dim, 0, start_pos_ptr, rope_base,
      original_seq_len, factor, beta_fast, beta_slow);
  return (CUresult)cudaGetLastError();
}

extern "C" CUresult dsv4_prepare_qk_fused_start_pos_ptr_cuda(
    const uint16_t *q_raw,
    const uint16_t *k_raw,
    uint16_t *q_out,
    uint16_t *k_out,
    int num_tokens,
    int local_heads,
    int head_dim,
    int rope_dim,
    const int *start_pos_ptr,
    float rms_eps,
    float rope_base,
    int original_seq_len,
    float factor,
    float beta_fast,
    float beta_slow,
    CUstream stream) {
  if (num_tokens < 0 || local_heads <= 0 || head_dim <= 0 || rope_dim < 0 ||
      rope_dim > head_dim || start_pos_ptr == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (num_tokens == 0) return CUDA_SUCCESS;
  int rows = num_tokens * (local_heads + 1);
  dsv4_prepare_qk_fused_kernel<<<rows, DSV4_ATTN_BLOCK, 0, (cudaStream_t)stream>>>(
      q_raw, k_raw, q_out, k_out, num_tokens, local_heads, head_dim, rope_dim,
      0, start_pos_ptr, rms_eps, rope_base, original_seq_len, factor, beta_fast,
      beta_slow);
  return (CUresult)cudaGetLastError();
}

extern "C" CUresult dsv4_prepare_qk_fused_batch_start_pos_cuda(
    const uint16_t *q_raw,
    const uint16_t *k_raw,
    uint16_t *q_out,
    uint16_t *k_out,
    int num_tokens,
    int local_heads,
    int head_dim,
    int rope_dim,
    const int *start_pos,
    float rms_eps,
    float rope_base,
    int original_seq_len,
    float factor,
    float beta_fast,
    float beta_slow,
    CUstream stream) {
  if (num_tokens < 0 || local_heads <= 0 || head_dim <= 0 || rope_dim < 0 ||
      rope_dim > head_dim || start_pos == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (num_tokens == 0) return CUDA_SUCCESS;
  int rows = num_tokens * (local_heads + 1);
  dsv4_prepare_qk_fused_batch_start_pos_kernel<<<rows, DSV4_ATTN_BLOCK, 0, (cudaStream_t)stream>>>(
      q_raw, k_raw, q_out, k_out, num_tokens, local_heads, head_dim, rope_dim,
      start_pos, rms_eps, rope_base, original_seq_len, factor, beta_fast,
      beta_slow);
  return (CUresult)cudaGetLastError();
}

__global__ void dsv4_dsa_fill_context_lens_positions_start_pos_kernel(
    int32_t *context_lens,
    int32_t *positions,
    const int32_t *start_pos,
    int token_offset,
    int batch_size,
    int key_count,
    int ratio) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= batch_size) return;
  int abs_pos = *start_pos + token_offset + row;
  int context_len = 0;
  if (abs_pos > 0 && ratio > 0) {
    context_len = abs_pos / ratio;
    if (context_len > key_count) context_len = key_count;
  }
  context_lens[row] = context_len;
  positions[row] = abs_pos;
}

extern "C" CUresult dsv4_dsa_fill_context_lens_positions_start_pos_cuda(
    int32_t *context_lens,
    int32_t *positions,
    const int32_t *start_pos,
    int token_offset,
    int batch_size,
    int key_count,
    int ratio,
    CUstream stream) {
  if (context_lens == nullptr || positions == nullptr || start_pos == nullptr ||
      token_offset < 0 || batch_size < 0 || key_count < 0 || ratio <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (batch_size == 0) return CUDA_SUCCESS;
  constexpr int kBlock = 128;
  int grid = (batch_size + kBlock - 1) / kBlock;
  dsv4_dsa_fill_context_lens_positions_start_pos_kernel<<<grid, kBlock, 0, (cudaStream_t)stream>>>(
      context_lens, positions, start_pos, token_offset, batch_size, key_count, ratio);
  return (CUresult)cudaGetLastError();
}
