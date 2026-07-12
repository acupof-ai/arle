// Shared DSv4 attention device helpers, extracted verbatim from the former
// dsv4_attention.cu god-file. Pure code motion, zero behavior change.
#ifndef DSV4_ATTENTION_COMMON_CUH
#define DSV4_ATTENTION_COMMON_CUH

#include "common.cuh"
#include <cuda.h>
#include <stdint.h>

#define DSV4_ATTN_BLOCK 256
#define DSV4_ATTN_MAX_HEAD_DIM 1024
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

__device__ __forceinline__ float dsv4_attn_block_sum(float value) {
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

__device__ __forceinline__ float dsv4_attn_block_max(float value) {
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

__device__ __forceinline__ int dsv4_graph_start_pos(
    int start_pos,
    const int *__restrict__ start_pos_ptr) {
  return start_pos_ptr == nullptr ? start_pos : *start_pos_ptr;
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

#endif  // DSV4_ATTENTION_COMMON_CUH
