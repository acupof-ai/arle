#include "common.cuh"
#include <stdint.h>

#define NORM_BLOCK 256
#define NORM_NUM_WARPS (NORM_BLOCK / WARP_SIZE)

// RMSNorm: out[i] = x[i] * weight[i] / sqrt(mean(x^2) + eps)
// BF16×4 vectorized loads, warp shuffle reduction.
// Single block, 256 threads — suitable for decode (n=2560).
__global__ void rms_norm_kernel(const __nv_bfloat16 *__restrict__ x,
                                const __nv_bfloat16 *__restrict__ weight,
                                __nv_bfloat16 *__restrict__ out, int n, float eps) {
  int tid = threadIdx.x;
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;

  int n4 = n / 4;

  const uint2 *x_vec = reinterpret_cast<const uint2 *>(x);

  float local_sum = 0.0f;
  for (int i = tid; i < n4; i += NORM_BLOCK) {
    uint2 xv = x_vec[i];
    __nv_bfloat162 lo = *reinterpret_cast<__nv_bfloat162 *>(&xv.x);
    __nv_bfloat162 hi = *reinterpret_cast<__nv_bfloat162 *>(&xv.y);
    float v0 = __bfloat162float(lo.x);
    float v1 = __bfloat162float(lo.y);
    float v2 = __bfloat162float(hi.x);
    float v3 = __bfloat162float(hi.y);
    local_sum += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
  }
  for (int i = n4 * 4 + tid; i < n; i += NORM_BLOCK) {
    float val = __bfloat162float(x[i]);
    local_sum += val * val;
  }

  local_sum = warp_reduce_sum(local_sum);

  __shared__ float warp_sums[NORM_NUM_WARPS];
  if (lane_id == 0) warp_sums[warp_id] = local_sum;
  __syncthreads();

  float total = 0.0f;
  if (warp_id == 0) {
    float val = (lane_id < NORM_NUM_WARPS) ? warp_sums[lane_id] : 0.0f;
    total = warp_reduce_sum(val);
  }

  __shared__ float s_inv_rms;
  if (tid == 0) {
    s_inv_rms = 1.0f / sqrtf(total / n + eps);
  }
  __syncthreads();
  float inv_rms = s_inv_rms;

  const uint2 *w_vec = reinterpret_cast<const uint2 *>(weight);
  uint2 *out_vec = reinterpret_cast<uint2 *>(out);

  for (int i = tid; i < n4; i += NORM_BLOCK) {
    uint2 xv = x_vec[i];
    uint2 wv = w_vec[i];
    __nv_bfloat162 x_lo = *reinterpret_cast<__nv_bfloat162 *>(&xv.x);
    __nv_bfloat162 x_hi = *reinterpret_cast<__nv_bfloat162 *>(&xv.y);
    __nv_bfloat162 w_lo = *reinterpret_cast<__nv_bfloat162 *>(&wv.x);
    __nv_bfloat162 w_hi = *reinterpret_cast<__nv_bfloat162 *>(&wv.y);

    // Match HF: round normalized to bf16 before weight multiply
    __nv_bfloat16 n0 = __float2bfloat16(__bfloat162float(x_lo.x) * inv_rms);
    __nv_bfloat16 n1 = __float2bfloat16(__bfloat162float(x_lo.y) * inv_rms);
    __nv_bfloat16 n2 = __float2bfloat16(__bfloat162float(x_hi.x) * inv_rms);
    __nv_bfloat16 n3 = __float2bfloat16(__bfloat162float(x_hi.y) * inv_rms);

    uint2 result;
    __nv_bfloat162 r_lo, r_hi;
    r_lo.x = __float2bfloat16(__bfloat162float(n0) * __bfloat162float(w_lo.x));
    r_lo.y = __float2bfloat16(__bfloat162float(n1) * __bfloat162float(w_lo.y));
    r_hi.x = __float2bfloat16(__bfloat162float(n2) * __bfloat162float(w_hi.x));
    r_hi.y = __float2bfloat16(__bfloat162float(n3) * __bfloat162float(w_hi.y));
    result.x = *reinterpret_cast<unsigned int *>(&r_lo);
    result.y = *reinterpret_cast<unsigned int *>(&r_hi);
    out_vec[i] = result;
  }
  for (int i = n4 * 4 + tid; i < n; i += NORM_BLOCK) {
    __nv_bfloat16 normed = __float2bfloat16(__bfloat162float(x[i]) * inv_rms);
    out[i] = __float2bfloat16(__bfloat162float(normed) * __bfloat162float(weight[i]));
  }
}

// Batched RMSNorm: each block handles one vector (blockIdx.x = token index)
// BF16×4 vectorized, warp shuffle reduction.
__global__ void rms_norm_batched_kernel(const __nv_bfloat16 *__restrict__ x,
                                         const __nv_bfloat16 *__restrict__ weight,
                                         __nv_bfloat16 *__restrict__ out,
                                         int hidden_dim, float eps) {
  const __nv_bfloat16 *x_row = x + blockIdx.x * hidden_dim;
  __nv_bfloat16 *out_row = out + blockIdx.x * hidden_dim;

  int tid = threadIdx.x;
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;

  int n4 = hidden_dim / 4;
  const uint2 *x_vec = reinterpret_cast<const uint2 *>(x_row);

  float local_sum = 0.0f;
  for (int i = tid; i < n4; i += NORM_BLOCK) {
    uint2 xv = x_vec[i];
    __nv_bfloat162 lo = *reinterpret_cast<__nv_bfloat162 *>(&xv.x);
    __nv_bfloat162 hi = *reinterpret_cast<__nv_bfloat162 *>(&xv.y);
    float v0 = __bfloat162float(lo.x);
    float v1 = __bfloat162float(lo.y);
    float v2 = __bfloat162float(hi.x);
    float v3 = __bfloat162float(hi.y);
    local_sum += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
  }
  for (int i = n4 * 4 + tid; i < hidden_dim; i += NORM_BLOCK) {
    float val = __bfloat162float(x_row[i]);
    local_sum += val * val;
  }

  local_sum = warp_reduce_sum(local_sum);

  __shared__ float warp_sums[NORM_NUM_WARPS];
  if (lane_id == 0) warp_sums[warp_id] = local_sum;
  __syncthreads();

  float total = 0.0f;
  if (warp_id == 0) {
    float val = (lane_id < NORM_NUM_WARPS) ? warp_sums[lane_id] : 0.0f;
    total = warp_reduce_sum(val);
  }

  __shared__ float s_inv_rms;
  if (tid == 0) {
    s_inv_rms = 1.0f / sqrtf(total / hidden_dim + eps);
  }
  __syncthreads();
  float inv_rms = s_inv_rms;

  // The uint4 path requires every row start to be
  // 16-byte aligned; hidden_dim=260 is 8-byte aligned per row but not 16-byte.
  const bool use_uint4 =
      ((((uintptr_t)x_row | (uintptr_t)weight | (uintptr_t)out_row) & 0xF) == 0);
  if (use_uint4) {
    int n8 = hidden_dim / 8;
    const uint4 *x_vec8 = reinterpret_cast<const uint4 *>(x_row);
    const uint4 *w_vec8 = reinterpret_cast<const uint4 *>(weight);
    uint4 *out_vec8 = reinterpret_cast<uint4 *>(out_row);

    // Keep `x * inv_rms * weight` in fp32 throughout — round to bf16 once at
    // the final store. ggml/llama.cpp's `ggml_rms_norm` fuses the weight
    // multiply and stays in fp32 internally; our earlier "round to bf16 between
    // scale and weight" pattern loses ~0.4% per layer and compounds into
    // catastrophic drift by layer 5 when fed noisy (Q4_K) weights.
    for (int i = tid; i < n8; i += NORM_BLOCK) {
      uint4 xv = __ldg(x_vec8 + i);
      uint4 wv = __ldg(w_vec8 + i);

      uint2 xv0 = make_uint2(xv.x, xv.y);
      uint2 xv1 = make_uint2(xv.z, xv.w);
      uint2 wv0 = make_uint2(wv.x, wv.y);
      uint2 wv1 = make_uint2(wv.z, wv.w);

      __nv_bfloat162 x0_lo = *reinterpret_cast<__nv_bfloat162 *>(&xv0.x);
      __nv_bfloat162 x0_hi = *reinterpret_cast<__nv_bfloat162 *>(&xv0.y);
      __nv_bfloat162 x1_lo = *reinterpret_cast<__nv_bfloat162 *>(&xv1.x);
      __nv_bfloat162 x1_hi = *reinterpret_cast<__nv_bfloat162 *>(&xv1.y);
      __nv_bfloat162 w0_lo = *reinterpret_cast<__nv_bfloat162 *>(&wv0.x);
      __nv_bfloat162 w0_hi = *reinterpret_cast<__nv_bfloat162 *>(&wv0.y);
      __nv_bfloat162 w1_lo = *reinterpret_cast<__nv_bfloat162 *>(&wv1.x);
      __nv_bfloat162 w1_hi = *reinterpret_cast<__nv_bfloat162 *>(&wv1.y);

      float n0 = __bfloat162float(x0_lo.x) * inv_rms * __bfloat162float(w0_lo.x);
      float n1 = __bfloat162float(x0_lo.y) * inv_rms * __bfloat162float(w0_lo.y);
      float n2 = __bfloat162float(x0_hi.x) * inv_rms * __bfloat162float(w0_hi.x);
      float n3 = __bfloat162float(x0_hi.y) * inv_rms * __bfloat162float(w0_hi.y);
      float n4 = __bfloat162float(x1_lo.x) * inv_rms * __bfloat162float(w1_lo.x);
      float n5 = __bfloat162float(x1_lo.y) * inv_rms * __bfloat162float(w1_lo.y);
      float n6 = __bfloat162float(x1_hi.x) * inv_rms * __bfloat162float(w1_hi.x);
      float n7 = __bfloat162float(x1_hi.y) * inv_rms * __bfloat162float(w1_hi.y);

      uint4 result;
      __nv_bfloat162 r0_lo, r0_hi, r1_lo, r1_hi;
      r0_lo.x = __float2bfloat16(n0);
      r0_lo.y = __float2bfloat16(n1);
      r0_hi.x = __float2bfloat16(n2);
      r0_hi.y = __float2bfloat16(n3);
      r1_lo.x = __float2bfloat16(n4);
      r1_lo.y = __float2bfloat16(n5);
      r1_hi.x = __float2bfloat16(n6);
      r1_hi.y = __float2bfloat16(n7);
      result.x = *reinterpret_cast<unsigned int *>(&r0_lo);
      result.y = *reinterpret_cast<unsigned int *>(&r0_hi);
      result.z = *reinterpret_cast<unsigned int *>(&r1_lo);
      result.w = *reinterpret_cast<unsigned int *>(&r1_hi);
      out_vec8[i] = result;
    }
    for (int i = n8 * 8 + tid; i < hidden_dim; i += NORM_BLOCK) {
      float n = __bfloat162float(x_row[i]) * inv_rms * __bfloat162float(weight[i]);
      out_row[i] = __float2bfloat16(n);
    }
  } else {
    const uint2 *w_vec = reinterpret_cast<const uint2 *>(weight);
    uint2 *out_vec = reinterpret_cast<uint2 *>(out_row);
    for (int i = tid; i < n4; i += NORM_BLOCK) {
      uint2 xv = x_vec[i];
      uint2 wv = w_vec[i];
      __nv_bfloat162 x_lo = *reinterpret_cast<__nv_bfloat162 *>(&xv.x);
      __nv_bfloat162 x_hi = *reinterpret_cast<__nv_bfloat162 *>(&xv.y);
      __nv_bfloat162 w_lo = *reinterpret_cast<__nv_bfloat162 *>(&wv.x);
      __nv_bfloat162 w_hi = *reinterpret_cast<__nv_bfloat162 *>(&wv.y);

      uint2 result;
      __nv_bfloat162 r_lo, r_hi;
      r_lo.x = __float2bfloat16(__bfloat162float(x_lo.x) * inv_rms * __bfloat162float(w_lo.x));
      r_lo.y = __float2bfloat16(__bfloat162float(x_lo.y) * inv_rms * __bfloat162float(w_lo.y));
      r_hi.x = __float2bfloat16(__bfloat162float(x_hi.x) * inv_rms * __bfloat162float(w_hi.x));
      r_hi.y = __float2bfloat16(__bfloat162float(x_hi.y) * inv_rms * __bfloat162float(w_hi.y));
      result.x = *reinterpret_cast<unsigned int *>(&r_lo);
      result.y = *reinterpret_cast<unsigned int *>(&r_hi);
      out_vec[i] = result;
    }
    for (int i = n4 * 4 + tid; i < hidden_dim; i += NORM_BLOCK) {
      float n = __bfloat162float(x_row[i]) * inv_rms * __bfloat162float(weight[i]);
      out_row[i] = __float2bfloat16(n);
    }
  }
}

// bf16 → fp32 cast, used once per prefill to seed the fp32 residual shadow
// from the bf16 embedding output.
__global__ void cast_bf16_to_f32_kernel(
    const __nv_bfloat16 *__restrict__ in,
    float *__restrict__ out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __bfloat162float(in[i]);
}

// fp32 → bf16 cast, used at the end of prefill to hand back a bf16 hidden
// state for the final norm + LM head projection that still consume bf16.
__global__ void cast_f32_to_bf16_kernel(
    const float *__restrict__ in,
    __nv_bfloat16 *__restrict__ out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __float2bfloat16(in[i]);
}

extern "C" {
cudaError_t rms_norm_cuda(const __nv_bfloat16 *x, const __nv_bfloat16 *weight, __nv_bfloat16 *out, int n,
                   float eps, cudaStream_t stream) {
  rms_norm_kernel<<<1, NORM_BLOCK, 0, stream>>>(x, weight, out, n, eps);
    return cudaGetLastError();
}

cudaError_t rms_norm_batched_cuda(const __nv_bfloat16 *x, const __nv_bfloat16 *weight,
                            __nv_bfloat16 *out, int hidden_dim, int seq_len,
                            float eps, cudaStream_t stream) {
  rms_norm_batched_kernel<<<seq_len, NORM_BLOCK, 0, stream>>>(
      x, weight, out, hidden_dim, eps);
    return cudaGetLastError();
}

cudaError_t cast_bf16_to_f32_cuda(
    const __nv_bfloat16 *in, float *out, int n, cudaStream_t stream
) {
    int block = 256;
    int grid = (n + block - 1) / block;
    cast_bf16_to_f32_kernel<<<grid, block, 0, stream>>>(in, out, n);
    return cudaGetLastError();
}

cudaError_t cast_f32_to_bf16_cuda(
    const float *in, __nv_bfloat16 *out, int n, cudaStream_t stream
) {
    int block = 256;
    int grid = (n + block - 1) / block;
    cast_f32_to_bf16_kernel<<<grid, block, 0, stream>>>(in, out, n);
    return cudaGetLastError();
}

// RMSNorm with (1+weight) offset — Qwen3.5 / Gemma style
// out[i] = x[i] * (1 + weight[i]) / sqrt(mean(x^2) + eps)
cudaError_t rms_norm_offset_cuda(const __nv_bfloat16 *x, const __nv_bfloat16 *weight,
                           __nv_bfloat16 *out, int n, float eps, cudaStream_t stream);

cudaError_t rms_norm_batched_offset_cuda(const __nv_bfloat16 *x, const __nv_bfloat16 *weight,
                                    __nv_bfloat16 *out, int hidden_dim, int seq_len,
                                    float eps, cudaStream_t stream);

cudaError_t rms_norm_gated_cuda(const __nv_bfloat16 *x, const float *weight,
                          const __nv_bfloat16 *gate, __nv_bfloat16 *out,
                          int num_heads, int head_dim, float eps, cudaStream_t stream);
} // extern "C"

// (1+weight) RMSNorm kernel
__global__ void rms_norm_offset_kernel(const __nv_bfloat16 *__restrict__ x,
                                        const __nv_bfloat16 *__restrict__ weight,
                                        __nv_bfloat16 *__restrict__ out, int n, float eps) {
  int tid = threadIdx.x;
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;
  int n4 = n / 4;

  const uint2 *x_vec = reinterpret_cast<const uint2 *>(x);

  float local_sum = 0.0f;
  for (int i = tid; i < n4; i += NORM_BLOCK) {
    uint2 xv = x_vec[i];
    __nv_bfloat162 lo = *reinterpret_cast<__nv_bfloat162 *>(&xv.x);
    __nv_bfloat162 hi = *reinterpret_cast<__nv_bfloat162 *>(&xv.y);
    float v0 = __bfloat162float(lo.x), v1 = __bfloat162float(lo.y);
    float v2 = __bfloat162float(hi.x), v3 = __bfloat162float(hi.y);
    local_sum += v0*v0 + v1*v1 + v2*v2 + v3*v3;
  }
  for (int i = n4*4 + tid; i < n; i += NORM_BLOCK) {
    float val = __bfloat162float(x[i]);
    local_sum += val * val;
  }

  local_sum = warp_reduce_sum(local_sum);
  __shared__ float warp_sums[NORM_NUM_WARPS];
  if (lane_id == 0) warp_sums[warp_id] = local_sum;
  __syncthreads();

  float total = 0.0f;
  if (warp_id == 0) {
    float val = (lane_id < NORM_NUM_WARPS) ? warp_sums[lane_id] : 0.0f;
    total = warp_reduce_sum(val);
  }

  __shared__ float s_inv_rms;
  if (tid == 0) s_inv_rms = 1.0f / sqrtf(total / n + eps);
  __syncthreads();
  float inv_rms = s_inv_rms;

  // NOTE: GemmaRMSNorm does ALL computation in float32, only rounds to bf16 at the end.
  // No intermediate bf16 rounding (unlike Llama/Qwen3.5 RMSNorm).
  const uint2 *w_vec = reinterpret_cast<const uint2 *>(weight);
  uint2 *out_vec = reinterpret_cast<uint2 *>(out);

  for (int i = tid; i < n4; i += NORM_BLOCK) {
    uint2 xv = x_vec[i];
    uint2 wv = w_vec[i];
    __nv_bfloat162 x_lo = *reinterpret_cast<__nv_bfloat162 *>(&xv.x);
    __nv_bfloat162 x_hi = *reinterpret_cast<__nv_bfloat162 *>(&xv.y);
    __nv_bfloat162 w_lo = *reinterpret_cast<__nv_bfloat162 *>(&wv.x);
    __nv_bfloat162 w_hi = *reinterpret_cast<__nv_bfloat162 *>(&wv.y);

    uint2 result;
    __nv_bfloat162 r_lo, r_hi;
    r_lo.x = __float2bfloat16(__bfloat162float(x_lo.x) * inv_rms * (1.0f + __bfloat162float(w_lo.x)));
    r_lo.y = __float2bfloat16(__bfloat162float(x_lo.y) * inv_rms * (1.0f + __bfloat162float(w_lo.y)));
    r_hi.x = __float2bfloat16(__bfloat162float(x_hi.x) * inv_rms * (1.0f + __bfloat162float(w_hi.x)));
    r_hi.y = __float2bfloat16(__bfloat162float(x_hi.y) * inv_rms * (1.0f + __bfloat162float(w_hi.y)));
    result.x = *reinterpret_cast<unsigned int *>(&r_lo);
    result.y = *reinterpret_cast<unsigned int *>(&r_hi);
    out_vec[i] = result;
  }
  for (int i = n4*4 + tid; i < n; i += NORM_BLOCK) {
    out[i] = __float2bfloat16(__bfloat162float(x[i]) * inv_rms * (1.0f + __bfloat162float(weight[i])));
  }
}

// Batched (1+weight) RMSNorm: one block per token.
// Grid: <<<seq_len, NORM_BLOCK>>>
__global__ void rms_norm_batched_offset_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ weight,
    __nv_bfloat16 *__restrict__ out,
    int hidden_dim, float eps) {

  const __nv_bfloat16 *x_row = x + blockIdx.x * hidden_dim;
  __nv_bfloat16 *out_row = out + blockIdx.x * hidden_dim;

  int tid = threadIdx.x;
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;
  int n4 = hidden_dim / 4;

  const uint2 *x_vec = reinterpret_cast<const uint2 *>(x_row);

  float local_sum = 0.0f;
  for (int i = tid; i < n4; i += NORM_BLOCK) {
    uint2 xv = x_vec[i];
    __nv_bfloat162 lo = *reinterpret_cast<__nv_bfloat162 *>(&xv.x);
    __nv_bfloat162 hi = *reinterpret_cast<__nv_bfloat162 *>(&xv.y);
    float v0 = __bfloat162float(lo.x), v1 = __bfloat162float(lo.y);
    float v2 = __bfloat162float(hi.x), v3 = __bfloat162float(hi.y);
    local_sum += v0*v0 + v1*v1 + v2*v2 + v3*v3;
  }
  for (int i = n4*4 + tid; i < hidden_dim; i += NORM_BLOCK) {
    float val = __bfloat162float(x_row[i]);
    local_sum += val * val;
  }

  local_sum = warp_reduce_sum(local_sum);
  __shared__ float warp_sums[NORM_NUM_WARPS];
  if (lane_id == 0) warp_sums[warp_id] = local_sum;
  __syncthreads();

  float total = 0.0f;
  if (warp_id == 0) {
    float val = (lane_id < NORM_NUM_WARPS) ? warp_sums[lane_id] : 0.0f;
    total = warp_reduce_sum(val);
  }

  __shared__ float s_inv_rms;
  if (tid == 0) s_inv_rms = 1.0f / sqrtf(total / hidden_dim + eps);
  __syncthreads();
  float inv_rms = s_inv_rms;

  const uint2 *w_vec = reinterpret_cast<const uint2 *>(weight);
  uint2 *out_vec = reinterpret_cast<uint2 *>(out_row);

  for (int i = tid; i < n4; i += NORM_BLOCK) {
    uint2 xv = x_vec[i];
    uint2 wv = w_vec[i];
    __nv_bfloat162 x_lo = *reinterpret_cast<__nv_bfloat162 *>(&xv.x);
    __nv_bfloat162 x_hi = *reinterpret_cast<__nv_bfloat162 *>(&xv.y);
    __nv_bfloat162 w_lo = *reinterpret_cast<__nv_bfloat162 *>(&wv.x);
    __nv_bfloat162 w_hi = *reinterpret_cast<__nv_bfloat162 *>(&wv.y);

    uint2 result;
    __nv_bfloat162 r_lo, r_hi;
    r_lo.x = __float2bfloat16(__bfloat162float(x_lo.x) * inv_rms * (1.0f + __bfloat162float(w_lo.x)));
    r_lo.y = __float2bfloat16(__bfloat162float(x_lo.y) * inv_rms * (1.0f + __bfloat162float(w_lo.y)));
    r_hi.x = __float2bfloat16(__bfloat162float(x_hi.x) * inv_rms * (1.0f + __bfloat162float(w_hi.x)));
    r_hi.y = __float2bfloat16(__bfloat162float(x_hi.y) * inv_rms * (1.0f + __bfloat162float(w_hi.y)));
    result.x = *reinterpret_cast<unsigned int *>(&r_lo);
    result.y = *reinterpret_cast<unsigned int *>(&r_hi);
    out_vec[i] = result;
  }
  for (int i = n4*4 + tid; i < hidden_dim; i += NORM_BLOCK) {
    out_row[i] = __float2bfloat16(__bfloat162float(x_row[i]) * inv_rms * (1.0f + __bfloat162float(weight[i])));
  }
}

// Gated RMSNorm for linear attention output:
//   out = rms_norm(x, f32_weight) * silu(gate)
// Per-head normalization: x is [num_heads * head_dim], weight is [head_dim] (broadcast).
// Grid: num_heads blocks, head_dim threads.
__global__ void rms_norm_gated_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const float *__restrict__ weight,
    const __nv_bfloat16 *__restrict__ gate,
    __nv_bfloat16 *__restrict__ out,
    int head_dim,
    float eps
) {
  int head = blockIdx.x;
  int tid = threadIdx.x;
  if (tid >= head_dim) return;

  int offset = head * head_dim + tid;

  float x_val = __bfloat162float(x[offset]);
  float sq = x_val * x_val;
  sq = warp_reduce_sum(sq);

  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;
  int num_warps = (head_dim + WARP_SIZE - 1) / WARP_SIZE;

  __shared__ float warp_sums[8];  // max 8 warps for head_dim=256
  if (lane_id == 0) warp_sums[warp_id] = sq;
  __syncthreads();

  __shared__ float s_inv_rms;
  if (tid == 0) {
    float total = 0.0f;
    for (int i = 0; i < num_warps; i++) total += warp_sums[i];
    s_inv_rms = rsqrtf(total / head_dim + eps);
  }
  __syncthreads();

  float normed = x_val * s_inv_rms;
  // Weight is F32, per head_dim (broadcast across heads)
  float w = weight[tid];
  normed *= w;

  float g = __bfloat162float(gate[offset]);
  float silu_g = g / (1.0f + expf(-g));

  out[offset] = __float2bfloat16(normed * silu_g);
}

extern "C" {

cudaError_t rms_norm_offset_cuda(const __nv_bfloat16 *x, const __nv_bfloat16 *weight,
                           __nv_bfloat16 *out, int n, float eps, cudaStream_t stream) {
  rms_norm_offset_kernel<<<1, NORM_BLOCK, 0, stream>>>(x, weight, out, n, eps);
    return cudaGetLastError();
}

cudaError_t rms_norm_batched_offset_cuda(const __nv_bfloat16 *x, const __nv_bfloat16 *weight,
                                    __nv_bfloat16 *out, int hidden_dim, int seq_len,
                                    float eps, cudaStream_t stream) {
  rms_norm_batched_offset_kernel<<<seq_len, NORM_BLOCK, 0, stream>>>(
      x, weight, out, hidden_dim, eps);
    return cudaGetLastError();
}

cudaError_t rms_norm_gated_cuda(const __nv_bfloat16 *x, const float *weight,
                          const __nv_bfloat16 *gate, __nv_bfloat16 *out,
                          int num_heads, int head_dim, float eps, cudaStream_t stream) {
  rms_norm_gated_kernel<<<num_heads, head_dim, 0, stream>>>(x, weight, gate, out, head_dim, eps);
    return cudaGetLastError();
}

} // extern "C"
