#include "common.cuh"
#include <cstdint>

#define NONPAGED_PREFILL_TILE 64
#define NONPAGED_PREFILL_MAX_HEAD_DIM 256
#define NONPAGED_PREFILL_MAX_WARPS 8

__global__ void nonpaged_prefill_attention_kernel(
    const __nv_bfloat16 *__restrict__ q,
    const __nv_bfloat16 *__restrict__ k_cache,
    const __nv_bfloat16 *__restrict__ v_cache,
    __nv_bfloat16 *__restrict__ out,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int seq_len,
    int start_pos,
    int max_seq_len,
    float sm_scale) {
  int q_head = blockIdx.x;
  int token = blockIdx.y;
  int dim = threadIdx.x;
  int lane = dim % WARP_SIZE;
  int warp = dim / WARP_SIZE;
  int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

  if (q_head >= num_q_heads || token >= seq_len || dim >= head_dim) {
    return;
  }

  int gqa_ratio = num_q_heads / num_kv_heads;
  int kv_head = q_head / gqa_ratio;
  int q_dim = num_q_heads * head_dim;
  int kv_len = start_pos + token + 1;

  __shared__ __nv_bfloat16 q_s[NONPAGED_PREFILL_MAX_HEAD_DIM];
  __shared__ float scores[NONPAGED_PREFILL_TILE];
  __shared__ float warp_partials[NONPAGED_PREFILL_MAX_WARPS *
                                 NONPAGED_PREFILL_TILE];
  __shared__ float warp_scratch[NONPAGED_PREFILL_MAX_WARPS];
  __shared__ float running_max_s;
  __shared__ float running_sum_s;
  __shared__ float rescale_s;

  q_s[dim] = q[token * q_dim + q_head * head_dim + dim];
  if (dim == 0) {
    running_max_s = -INFINITY;
    running_sum_s = 0.0f;
  }
  __syncthreads();

  float q_val = __bfloat162float(q_s[dim]);
  float o_acc = 0.0f;

  for (int tile_start = 0; tile_start < kv_len; tile_start += NONPAGED_PREFILL_TILE) {
    int tile_len = min(NONPAGED_PREFILL_TILE, kv_len - tile_start);

    for (int pos = 0; pos < tile_len; ++pos) {
      int abs_pos = tile_start + pos;
      int k_idx = (kv_head * max_seq_len + abs_pos) * head_dim + dim;
      float partial = q_val * __bfloat162float(k_cache[k_idx]);
      partial = warp_reduce_sum(partial);
      if (lane == 0) {
        warp_partials[warp * NONPAGED_PREFILL_TILE + pos] = partial;
      }
    }
    __syncthreads();

    if (dim < tile_len) {
      float score = 0.0f;
      for (int w = 0; w < num_warps; ++w) {
        score += warp_partials[w * NONPAGED_PREFILL_TILE + dim];
      }
      scores[dim] = score * sm_scale;
    }
    __syncthreads();

    float local_max = -INFINITY;
    if (dim < tile_len) {
      local_max = scores[dim];
    }
    local_max = warp_reduce_max(local_max);
    if (lane == 0) {
      warp_scratch[warp] = local_max;
    }
    __syncthreads();

    if (dim == 0) {
      float tile_max = warp_scratch[0];
      for (int w = 1; w < num_warps; ++w) {
        tile_max = fmaxf(tile_max, warp_scratch[w]);
      }
      float old_max = running_max_s;
      float new_max = fmaxf(old_max, tile_max);
      rescale_s = expf(old_max - new_max);
      running_sum_s *= rescale_s;
      running_max_s = new_max;
    }
    __syncthreads();

    o_acc *= rescale_s;
    float row_sum = 0.0f;
    float current_max = running_max_s;
    for (int pos = 0; pos < tile_len; ++pos) {
      float weight = expf(scores[pos] - current_max);
      int abs_pos = tile_start + pos;
      int v_idx = (kv_head * max_seq_len + abs_pos) * head_dim + dim;
      row_sum += weight;
      o_acc += weight * __bfloat162float(v_cache[v_idx]);
    }
    if (dim == 0) {
      running_sum_s += row_sum;
    }
    __syncthreads();
  }

  float denom = running_sum_s;
  float value = denom > 0.0f ? o_acc / denom : 0.0f;
  out[token * q_dim + q_head * head_dim + dim] = __float2bfloat16(value);
}

extern "C" cudaError_t nonpaged_prefill_attention_cuda(
    const uint16_t *q,
    const uint16_t *k_cache,
    const uint16_t *v_cache,
    uint16_t *out,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int seq_len,
    int kv_len,
    int max_seq_len,
    float sm_scale,
    cudaStream_t stream) {
  if (num_q_heads <= 0 || num_kv_heads <= 0 || seq_len < 0 || kv_len < seq_len ||
      max_seq_len < kv_len || (head_dim != 128 && head_dim != 256) ||
      num_q_heads % num_kv_heads != 0) {
    return cudaErrorInvalidValue;
  }
  if (seq_len == 0) {
    return cudaSuccess;
  }
  int start_pos = kv_len - seq_len;
  dim3 grid(num_q_heads, seq_len);
  nonpaged_prefill_attention_kernel<<<grid, head_dim, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16 *>(q),
      reinterpret_cast<const __nv_bfloat16 *>(k_cache),
      reinterpret_cast<const __nv_bfloat16 *>(v_cache),
      reinterpret_cast<__nv_bfloat16 *>(out),
      num_q_heads,
      num_kv_heads,
      head_dim,
      seq_len,
      start_pos,
      max_seq_len,
      sm_scale);
  return cudaGetLastError();
}
