// Native replacement for the TileLang chunk-prepare stage: the TileLang
// lowering replicated the full q/k row into every thread's registers
// (128 f32 x2 per thread -> local-memory spill, 128x redundant loads),
// costing 66 ms/layer vs the ~0.1 ms bandwidth roofline.
#include "common.cuh"
#include <cuda_runtime.h>

#define GDR_PREP_KEY_DIM 128
#define GDR_PREP_VALUE_DIM 128
#define GDR_PREP_WARPS 4

// One warp per (token, v_head); lane l covers dims {l, l+32, l+64, l+96}.
__global__ void gdr_prefill_chunk_prepare_kernel(
    const __nv_bfloat16 *__restrict__ qkv,
    const __nv_bfloat16 *__restrict__ b_proj,
    const __nv_bfloat16 *__restrict__ a_proj,
    const __nv_bfloat16 *__restrict__ dt_bias,
    const float *__restrict__ a_log,
    __nv_bfloat16 *__restrict__ q_out,
    __nv_bfloat16 *__restrict__ k_out,
    __nv_bfloat16 *__restrict__ v_out,
    float *__restrict__ g_out,
    float *__restrict__ beta_out,
    int num_key_heads,
    int num_value_heads,
    int qkv_dim,
    int seq_len) {
  int warp = threadIdx.x / 32;
  int lane = threadIdx.x % 32;
  int token = blockIdx.x * GDR_PREP_WARPS + warp;
  int v_head = blockIdx.y;
  if (token >= seq_len) {
    return;
  }

  int k_head = (v_head * num_key_heads) / num_value_heads;
  size_t row = static_cast<size_t>(token) * qkv_dim;
  const __nv_bfloat16 *q_in = qkv + row + static_cast<size_t>(k_head) * GDR_PREP_KEY_DIM;
  const __nv_bfloat16 *k_in = q_in + static_cast<size_t>(num_key_heads) * GDR_PREP_KEY_DIM;
  const __nv_bfloat16 *v_in =
      qkv + row + static_cast<size_t>(num_key_heads) * 2 * GDR_PREP_KEY_DIM +
      static_cast<size_t>(v_head) * GDR_PREP_VALUE_DIM;

  float q_val[GDR_PREP_KEY_DIM / 32];
  float k_val[GDR_PREP_KEY_DIM / 32];
  float qq = 0.0f;
  float kk = 0.0f;
#pragma unroll
  for (int i = 0; i < GDR_PREP_KEY_DIM / 32; ++i) {
    int d = lane + i * 32;
    q_val[i] = __bfloat162float(q_in[d]);
    k_val[i] = __bfloat162float(k_in[d]);
    qq += q_val[i] * q_val[i];
    kk += k_val[i] * k_val[i];
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    qq += __shfl_xor_sync(0xffffffffu, qq, off);
    kk += __shfl_xor_sync(0xffffffffu, kk, off);
  }
  float q_scale = rsqrtf(qq + 1e-12f);
  float k_scale = rsqrtf(kk + 1e-12f);

  size_t out_base =
      (static_cast<size_t>(token) * num_value_heads + v_head) * GDR_PREP_KEY_DIM;
#pragma unroll
  for (int i = 0; i < GDR_PREP_KEY_DIM / 32; ++i) {
    int d = lane + i * 32;
    q_out[out_base + d] = __float2bfloat16(q_val[i] * q_scale);
    k_out[out_base + d] = __float2bfloat16(k_val[i] * k_scale);
    v_out[out_base + d] = v_in[d];
  }

  if (lane == 0) {
    size_t head_idx = static_cast<size_t>(token) * num_value_heads + v_head;
    float a_val = __bfloat162float(a_proj[head_idx]);
    float b_val = __bfloat162float(b_proj[head_idx]);
    float dt = __bfloat162float(dt_bias[v_head]);
    float x = a_val + dt;
    float softplus_x = x > 20.0f ? x : __logf(1.0f + __expf(x));
    g_out[head_idx] = -__expf(a_log[v_head]) * softplus_x;
    beta_out[head_idx] = 1.0f / (1.0f + __expf(-b_val));
  }
}

extern "C" cudaError_t gated_delta_rule_prefill_chunk_prepare_cuda(
    const __nv_bfloat16 *qkv,
    const __nv_bfloat16 *b_proj,
    const __nv_bfloat16 *a_proj,
    const __nv_bfloat16 *dt_bias,
    const float *a_log,
    __nv_bfloat16 *q_out,
    __nv_bfloat16 *k_out,
    __nv_bfloat16 *v_out,
    float *g_out,
    float *beta_out,
    int num_key_heads,
    int num_value_heads,
    int qkv_dim,
    int seq_len,
    cudaStream_t stream) {
  if (seq_len < 0 || num_key_heads <= 0 || num_value_heads <= 0 || qkv_dim <= 0) {
    return cudaErrorInvalidValue;
  }
  if (seq_len == 0) {
    return cudaSuccess;
  }
  dim3 grid((seq_len + GDR_PREP_WARPS - 1) / GDR_PREP_WARPS, num_value_heads);
  gdr_prefill_chunk_prepare_kernel<<<grid, GDR_PREP_WARPS * 32, 0, stream>>>(
      qkv, b_proj, a_proj, dt_bias, a_log, q_out, k_out, v_out, g_out, beta_out,
      num_key_heads, num_value_heads, qkv_dim, seq_len);
  return cudaGetLastError();
}
