// DSpark draft dense MLA-latent attention.
//
// Small non-causal dense attention over one head-SHARED compressed latent
// (K == V — MLA has no separate V). Every one of the `block_size` query rows
// attends the WHOLE `[kv_len]` latent range (draft context ++ noise block); the
// single `latent_kv` row is broadcast over all `local_heads` query heads.
//
// Frozen contract (mirrors ffi/attention.rs dsv4_dspark_draft_attention_cuda):
//   q         [block_size, local_heads, head_dim]  token-major, bf16
//   latent_kv [kv_len,     head_dim]               kv-major,   bf16, head-SHARED
//   out       [block_size, local_heads, head_dim]  token-major, bf16  (Flag #1)
//   head_dim == nope_dim + rope_dim.
//
//   for each block row r, head h:
//     score[j] = sm_scale * dot(q[r,h,0..head_dim], latent_kv[j,0..head_dim])
//                // full head_dim (NoPE + RoPE); RoPE is applied upstream to
//                // q/latent, the kernel does raw dot-products for scoring.
//     w[0..kv_len] = online_softmax(score[0..kv_len])   // non-causal, all keys
//     v[0..head_dim] = sum_j w[j] * latent_kv[j,0..head_dim]
//     out[r,h, rope tail] = inverse-RoPE(v tail, abs_pos = base_start_pos + r)
//                // K==V shares the forward-rotated latent tail, so the value's
//                // last `rope_dim` dims must be un-rotated at the QUERY position
//                // before o-proj — exactly as the main cr==0 path (dsv4_swa.cu).
//                // Skipping this leaks a position-correlated near-constant into
//                // block_hidden → tied-head argmax collapses to attractor tokens.

#include "dsv4_attention_common.cuh"

#define DSV4_DSPARK_MAX_KEYS 9216

__global__ void dsv4_dspark_draft_attention_kernel(
    const uint16_t *__restrict__ q,
    const uint16_t *__restrict__ latent_kv,
    uint16_t *__restrict__ out,
    int kv_len,
    int block_size,
    int local_heads,
    int head_dim,
    int rope_dim,
    int base_start_pos,
    float sm_scale,
    float rope_base,
    int original_seq_len,
    float factor,
    float beta_fast,
    float beta_slow) {
  int row = blockIdx.x;
  if (row >= block_size * local_heads) return;
  int token = row / local_heads;
  int head = row - token * local_heads;
  int local_width = local_heads * head_dim;
  int abs_pos = base_start_pos + token;

  __shared__ float logits[DSV4_DSPARK_MAX_KEYS];
  __shared__ float denom_shared;
  __shared__ float max_shared;
  __shared__ float out_vec[DSV4_ATTN_MAX_HEAD_DIM];

  int q_base = token * local_width + head * head_dim;

  for (int key_idx = threadIdx.x; key_idx < kv_len; key_idx += blockDim.x) {
    const uint16_t *k = latent_kv + (size_t)key_idx * head_dim;
    float acc = 0.0f;
    for (int col = 0; col < head_dim; ++col) {
      acc += dsv4_attn_bf16_to_f32(q[q_base + col]) * dsv4_attn_bf16_to_f32(k[col]);
    }
    logits[key_idx] = acc * sm_scale;
  }
  __syncthreads();

  float local_max = -INFINITY;
  for (int key_idx = threadIdx.x; key_idx < kv_len; key_idx += blockDim.x) {
    local_max = fmaxf(local_max, logits[key_idx]);
  }
  local_max = dsv4_attn_block_max(local_max);
  if (threadIdx.x == 0) max_shared = local_max;
  __syncthreads();

  float denom = 0.0f;
  for (int key_idx = threadIdx.x; key_idx < kv_len; key_idx += blockDim.x) {
    float prob = expf(logits[key_idx] - max_shared);
    logits[key_idx] = prob;
    denom += prob;
  }
  denom = dsv4_attn_block_sum(denom);
  if (threadIdx.x == 0) denom_shared = denom;
  __syncthreads();

  for (int col = threadIdx.x; col < head_dim; col += blockDim.x) {
    float acc = 0.0f;
    for (int key_idx = 0; key_idx < kv_len; ++key_idx) {
      float kv = dsv4_attn_bf16_to_f32(latent_kv[(size_t)key_idx * head_dim + col]);
      acc += (logits[key_idx] / denom_shared) * kv;
    }
    out_vec[col] = acc;
  }
  __syncthreads();

  // Inverse-RoPE the value's rope tail at the query's absolute position (sign
  // -1), mirroring dsv4_swa.cu:86-101. NoPE dims pass through unchanged.
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

extern "C" CUresult dsv4_dspark_draft_attention_cuda(
    const uint16_t *q,
    const uint16_t *latent_kv,
    uint16_t *out,
    int kv_len,
    int block_size,
    int local_heads,
    int head_dim,
    int nope_dim,
    int rope_dim,
    int base_start_pos,
    float sm_scale,
    float rope_base,
    int original_seq_len,
    float factor,
    float beta_fast,
    float beta_slow,
    cudaStream_t stream) {
  // nope_dim is a frozen-signature param, unused (Flag #1: full head_dim value;
  // the rope tail is the trailing `rope_dim` dims, handled by inverse-RoPE).
  (void)nope_dim;
  if (block_size < 0 || local_heads <= 0 || head_dim <= 0 ||
      head_dim > DSV4_ATTN_MAX_HEAD_DIM || kv_len <= 0 ||
      kv_len > DSV4_DSPARK_MAX_KEYS || rope_dim < 0 || rope_dim > head_dim) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (block_size == 0) return CUDA_SUCCESS;
  dsv4_dspark_draft_attention_kernel<<<block_size * local_heads, DSV4_ATTN_BLOCK, 0, stream>>>(
      q, latent_kv, out, kv_len, block_size, local_heads, head_dim, rope_dim,
      base_start_pos, sm_scale, rope_base, original_seq_len, factor, beta_fast,
      beta_slow);
  return (CUresult)cudaGetLastError();
}
