// DSpark draft dense MLA-latent attention (T4.1).
//
// Small non-causal dense attention over one head-SHARED compressed latent
// (K == V — MLA has no separate V). Every one of the `block_size` query rows
// attends the WHOLE `[kv_len]` latent range (draft context ++ noise block); the
// single `latent_kv` row is broadcast over all `local_heads` query heads.
//
// Contract (mirrors ffi/attention.rs dsv4_dspark_draft_attention_cuda):
//   q         [block_size, local_heads, head_dim]  token-major, bf16
//   latent_kv [kv_len,     head_dim]               kv-major,   bf16, head-SHARED
//   out       [block_size, local_heads, head_dim]  token-major, bf16  (Flag #1)
//   head_dim == nope_dim + rope_dim.
//
//   for each block row r, head h:
//     score[j] = sm_scale * dot(q[r,h,0..head_dim], latent_kv[j,0..head_dim])
//                // full head_dim (NoPE + RoPE); RoPE is applied upstream, the
//                // kernel does raw dot-products.
//     w[0..kv_len] = online_softmax(score[0..kv_len])   // non-causal, all keys
//     out[r,h,0..head_dim] = sum_j w[j] * latent_kv[j,0..head_dim]
//                // weighted sum of the FULL head_dim latent (Flag #1: the value
//                // is the whole latent, feeding mla_oproj as local_heads*head_dim).
//
// ponytail: naive one-block-per-(query row, head) online softmax — kv_len is
// tiny (draft context + block, a few dozen), so clarity beats micro-optimization.
// Mirrors the block/warp structure of dsv4_swa.cu / dsv4_hybrid.cu (shared
// logits, block-max / block-sum reductions, bf16<->f32 converters), minus the
// sliding-window/compressed key selection and the sink term.

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
    float sm_scale) {
  int row = blockIdx.x;
  if (row >= block_size * local_heads) return;
  int token = row / local_heads;
  int head = row - token * local_heads;
  int local_width = local_heads * head_dim;

  __shared__ float logits[DSV4_DSPARK_MAX_KEYS];
  __shared__ float denom_shared;
  __shared__ float max_shared;

  int q_base = token * local_width + head * head_dim;

  // score[j] = sm_scale * dot(q[row], latent_kv[j]) over the full head_dim.
  for (int key_idx = threadIdx.x; key_idx < kv_len; key_idx += blockDim.x) {
    const uint16_t *k = latent_kv + (size_t)key_idx * head_dim;
    float acc = 0.0f;
    for (int col = 0; col < head_dim; ++col) {
      acc += dsv4_attn_bf16_to_f32(q[q_base + col]) * dsv4_attn_bf16_to_f32(k[col]);
    }
    logits[key_idx] = acc * sm_scale;
  }
  __syncthreads();

  // Online softmax over all kv_len keys — non-causal, no sink.
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

  // out[col] = sum_j w[j] * latent_kv[j, col] over the full head_dim (Flag #1).
  for (int col = threadIdx.x; col < head_dim; col += blockDim.x) {
    float acc = 0.0f;
    for (int key_idx = 0; key_idx < kv_len; ++key_idx) {
      float kv = dsv4_attn_bf16_to_f32(latent_kv[(size_t)key_idx * head_dim + col]);
      acc += (logits[key_idx] / denom_shared) * kv;
    }
    out[token * local_width + head * head_dim + col] = dsv4_attn_f32_to_bf16_bits(acc);
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
    float sm_scale,
    cudaStream_t stream) {
  // nope_dim / rope_dim are frozen-signature params but unused here: Flag #1
  // makes the value the FULL head_dim latent, and RoPE is applied upstream.
  (void)nope_dim;
  (void)rope_dim;
  if (block_size < 0 || local_heads <= 0 || head_dim <= 0 ||
      head_dim > DSV4_ATTN_MAX_HEAD_DIM || kv_len <= 0 ||
      kv_len > DSV4_DSPARK_MAX_KEYS) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (block_size == 0) return CUDA_SUCCESS;
  dsv4_dspark_draft_attention_kernel<<<block_size * local_heads, DSV4_ATTN_BLOCK, 0, stream>>>(
      q, latent_kv, out, kv_len, block_size, local_heads, head_dim, sm_scale);
  return (CUresult)cudaGetLastError();
}
