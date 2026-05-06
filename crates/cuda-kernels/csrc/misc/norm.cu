#include "common.cuh"
#include <stdint.h>

#define NORM_BLOCK 256
#define NORM_NUM_WARPS (NORM_BLOCK / WARP_SIZE)

// ============================================================================
// RMSNorm: out[i] = x[i] * weight[i] / sqrt(mean(x^2) + eps)
// BF16×4 vectorized loads, warp shuffle reduction.
// Single block, 256 threads — suitable for decode (n=2560).
// ============================================================================
__global__ void rms_norm_kernel(const __nv_bfloat16 *__restrict__ x,
                                const __nv_bfloat16 *__restrict__ weight,
                                __nv_bfloat16 *__restrict__ out, int n, float eps) {
  int tid = threadIdx.x;
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;

  int n4 = n / 4;

  const uint2 *x_vec = reinterpret_cast<const uint2 *>(x);

  // Pass 1: Compute sum of squares (FP32 accumulator)
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
  // Scalar tail
  for (int i = n4 * 4 + tid; i < n; i += NORM_BLOCK) {
    float val = __bfloat162float(x[i]);
    local_sum += val * val;
  }

  // Warp shuffle reduction
  local_sum = warp_reduce_sum(local_sum);

  // Inter-warp reduction via shared memory
  __shared__ float warp_sums[NORM_NUM_WARPS];
  if (lane_id == 0) warp_sums[warp_id] = local_sum;
  __syncthreads();

  // First warp reduces
  float total = 0.0f;
  if (warp_id == 0) {
    float val = (lane_id < NORM_NUM_WARPS) ? warp_sums[lane_id] : 0.0f;
    total = warp_reduce_sum(val);
  }

  // Broadcast inv_rms to all threads
  __shared__ float s_inv_rms;
  if (tid == 0) {
    s_inv_rms = 1.0f / sqrtf(total / n + eps);
  }
  __syncthreads();
  float inv_rms = s_inv_rms;

  // Pass 2: Normalize and scale (BF16×4 vectorized)
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
  // Scalar tail
  for (int i = n4 * 4 + tid; i < n; i += NORM_BLOCK) {
    __nv_bfloat16 normed = __float2bfloat16(__bfloat162float(x[i]) * inv_rms);
    out[i] = __float2bfloat16(__bfloat162float(normed) * __bfloat162float(weight[i]));
  }
}

// ============================================================================
// Fused Add + RMSNorm: hidden += residual; out = rms_norm(hidden, weight)
// One kernel replaces two: saves one global read of hidden.
// BF16×4 vectorized, warp shuffle reduction.
// ============================================================================
__global__ void fused_add_rms_norm_kernel(
    __nv_bfloat16 *__restrict__ hidden,          // in/out: hidden state (updated in-place)
    const __nv_bfloat16 *__restrict__ residual,   // in: residual to add
    const __nv_bfloat16 *__restrict__ weight,     // in: rms_norm weight
    __nv_bfloat16 *__restrict__ out,              // out: normalized output
    int n, float eps) {

  int tid = threadIdx.x;
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;

  int n4 = n / 4;

  uint2 *hidden_vec = reinterpret_cast<uint2 *>(hidden);
  const uint2 *res_vec = reinterpret_cast<const uint2 *>(residual);

  // Pass 1: Add residual to hidden, compute sum of squares
  float local_sum = 0.0f;
  for (int i = tid; i < n4; i += NORM_BLOCK) {
    uint2 hv = hidden_vec[i];
    uint2 rv = res_vec[i];
    __nv_bfloat162 h_lo = *reinterpret_cast<__nv_bfloat162 *>(&hv.x);
    __nv_bfloat162 h_hi = *reinterpret_cast<__nv_bfloat162 *>(&hv.y);
    __nv_bfloat162 r_lo = *reinterpret_cast<__nv_bfloat162 *>(&rv.x);
    __nv_bfloat162 r_hi = *reinterpret_cast<__nv_bfloat162 *>(&rv.y);

    // Add in FP32 then store back as BF16
    float s0 = __bfloat162float(h_lo.x) + __bfloat162float(r_lo.x);
    float s1 = __bfloat162float(h_lo.y) + __bfloat162float(r_lo.y);
    float s2 = __bfloat162float(h_hi.x) + __bfloat162float(r_hi.x);
    float s3 = __bfloat162float(h_hi.y) + __bfloat162float(r_hi.y);

    // Write updated hidden
    __nv_bfloat162 s_lo, s_hi;
    s_lo.x = __float2bfloat16(s0);
    s_lo.y = __float2bfloat16(s1);
    s_hi.x = __float2bfloat16(s2);
    s_hi.y = __float2bfloat16(s3);
    uint2 sv;
    sv.x = *reinterpret_cast<unsigned int *>(&s_lo);
    sv.y = *reinterpret_cast<unsigned int *>(&s_hi);
    hidden_vec[i] = sv;

    // Accumulate sum of squares (use the bf16-rounded values for consistency)
    float v0 = __bfloat162float(s_lo.x);
    float v1 = __bfloat162float(s_lo.y);
    float v2 = __bfloat162float(s_hi.x);
    float v3 = __bfloat162float(s_hi.y);
    local_sum += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
  }
  // Scalar tail
  for (int i = n4 * 4 + tid; i < n; i += NORM_BLOCK) {
    float s = __bfloat162float(hidden[i]) + __bfloat162float(residual[i]);
    hidden[i] = __float2bfloat16(s);
    float v = __bfloat162float(hidden[i]);
    local_sum += v * v;
  }

  // Warp shuffle reduction
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

  // Pass 2: Normalize and scale (read updated hidden, write out)
  const uint2 *h_vec_r = reinterpret_cast<const uint2 *>(hidden);
  const uint2 *w_vec = reinterpret_cast<const uint2 *>(weight);
  uint2 *out_vec = reinterpret_cast<uint2 *>(out);

  for (int i = tid; i < n4; i += NORM_BLOCK) {
    uint2 hv = h_vec_r[i];
    uint2 wv = w_vec[i];
    __nv_bfloat162 h_lo = *reinterpret_cast<__nv_bfloat162 *>(&hv.x);
    __nv_bfloat162 h_hi = *reinterpret_cast<__nv_bfloat162 *>(&hv.y);
    __nv_bfloat162 w_lo = *reinterpret_cast<__nv_bfloat162 *>(&wv.x);
    __nv_bfloat162 w_hi = *reinterpret_cast<__nv_bfloat162 *>(&wv.y);

    __nv_bfloat16 n0 = __float2bfloat16(__bfloat162float(h_lo.x) * inv_rms);
    __nv_bfloat16 n1 = __float2bfloat16(__bfloat162float(h_lo.y) * inv_rms);
    __nv_bfloat16 n2 = __float2bfloat16(__bfloat162float(h_hi.x) * inv_rms);
    __nv_bfloat16 n3 = __float2bfloat16(__bfloat162float(h_hi.y) * inv_rms);

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
  // Scalar tail
  for (int i = n4 * 4 + tid; i < n; i += NORM_BLOCK) {
    __nv_bfloat16 normed = __float2bfloat16(__bfloat162float(hidden[i]) * inv_rms);
    out[i] = __float2bfloat16(__bfloat162float(normed) * __bfloat162float(weight[i]));
  }
}

// ============================================================================
// Batched RMSNorm: each block handles one vector (blockIdx.x = token index)
// BF16×4 vectorized, warp shuffle reduction.
// ============================================================================
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

  // Pass 1: sum of squares
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

  // Pass 2: normalize and scale. The uint4 path requires every row start to be
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

// ============================================================================
// Batched RMSNorm with fp32 input, bf16 output.
//
// Used to let the residual stream stay in fp32 across layers while still
// feeding bf16 downstream GEMMs. Reading fp32 here avoids the bf16 rounding
// that compounds Q4_K-level weight noise through 36 layers on Qwen3-4B.
// Precision chain: fp32 -> fp32 rsqrt -> fp32 * fp32 weight -> bf16 store.
// ============================================================================
__global__ void rms_norm_batched_f32_in_kernel(
    const float *__restrict__ x,                     // [seq_len, hidden_dim] fp32
    const __nv_bfloat16 *__restrict__ weight,        // [hidden_dim] bf16
    __nv_bfloat16 *__restrict__ out,                 // [seq_len, hidden_dim] bf16
    int hidden_dim, float eps) {
  const float *x_row = x + blockIdx.x * hidden_dim;
  __nv_bfloat16 *out_row = out + blockIdx.x * hidden_dim;

  int tid = threadIdx.x;
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;

  // Pass 1: sum of squares (fp32)
  float local_sum = 0.0f;
  for (int i = tid; i < hidden_dim; i += NORM_BLOCK) {
    float v = x_row[i];
    local_sum += v * v;
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

  // Pass 2: normalize × weight, store bf16
  for (int i = tid; i < hidden_dim; i += NORM_BLOCK) {
    float n = x_row[i] * inv_rms * __bfloat162float(weight[i]);
    out_row[i] = __float2bfloat16(n);
  }
}

// ============================================================================
// fp32 += bf16 accumulator. Used to maintain an fp32 residual shadow that
// absorbs bf16-produced layer outputs without losing precision across
// 36 layers of compounding.
// ============================================================================
__global__ void add_bf16_into_f32_kernel(
    float *__restrict__ out,                       // fp32 in/out
    const __nv_bfloat16 *__restrict__ in,          // bf16 read-only
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] += __bfloat162float(in[i]);
}

// ============================================================================
// bf16 → fp32 cast, used once per prefill to seed the fp32 residual shadow
// from the bf16 embedding output.
// ============================================================================
__global__ void cast_bf16_to_f32_kernel(
    const __nv_bfloat16 *__restrict__ in,
    float *__restrict__ out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __bfloat162float(in[i]);
}

// ============================================================================
// fp32 → bf16 cast, used at the end of prefill to hand back a bf16 hidden
// state for the final norm + LM head projection that still consume bf16.
// ============================================================================
__global__ void cast_f32_to_bf16_kernel(
    const float *__restrict__ in,
    __nv_bfloat16 *__restrict__ out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __float2bfloat16(in[i]);
}

// ============================================================================
// Batched Fused Add + RMSNorm: one block per token (blockIdx.x = token index)
// hidden[b] += residual[b]; out[b] = rms_norm(hidden[b], weight)
// Saves one global read of hidden compared to separate add + batched rms_norm.
// Grid: <<<seq_len, NORM_BLOCK>>>
// ============================================================================
__global__ void fused_add_rms_norm_batched_kernel(
    __nv_bfloat16 *__restrict__ hidden,          // in/out [seq_len, hidden_dim]
    const __nv_bfloat16 *__restrict__ residual,   // in [seq_len, hidden_dim]
    const __nv_bfloat16 *__restrict__ weight,     // in [hidden_dim]
    __nv_bfloat16 *__restrict__ out,              // out [seq_len, hidden_dim]
    int hidden_dim, float eps) {

  __nv_bfloat16 *hidden_row = hidden + blockIdx.x * hidden_dim;
  const __nv_bfloat16 *res_row = residual + blockIdx.x * hidden_dim;
  __nv_bfloat16 *out_row = out + blockIdx.x * hidden_dim;

  int tid = threadIdx.x;
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;

  int n4 = hidden_dim / 4;

  uint2 *hidden_vec = reinterpret_cast<uint2 *>(hidden_row);
  const uint2 *res_vec = reinterpret_cast<const uint2 *>(res_row);

  // Pass 1: Add residual to hidden, compute sum of squares
  float local_sum = 0.0f;
  for (int i = tid; i < n4; i += NORM_BLOCK) {
    uint2 hv = hidden_vec[i];
    uint2 rv = res_vec[i];
    __nv_bfloat162 h_lo = *reinterpret_cast<__nv_bfloat162 *>(&hv.x);
    __nv_bfloat162 h_hi = *reinterpret_cast<__nv_bfloat162 *>(&hv.y);
    __nv_bfloat162 r_lo = *reinterpret_cast<__nv_bfloat162 *>(&rv.x);
    __nv_bfloat162 r_hi = *reinterpret_cast<__nv_bfloat162 *>(&rv.y);

    // Add in FP32 then store back as BF16
    float s0 = __bfloat162float(h_lo.x) + __bfloat162float(r_lo.x);
    float s1 = __bfloat162float(h_lo.y) + __bfloat162float(r_lo.y);
    float s2 = __bfloat162float(h_hi.x) + __bfloat162float(r_hi.x);
    float s3 = __bfloat162float(h_hi.y) + __bfloat162float(r_hi.y);

    // Write updated hidden
    __nv_bfloat162 s_lo, s_hi;
    s_lo.x = __float2bfloat16(s0);
    s_lo.y = __float2bfloat16(s1);
    s_hi.x = __float2bfloat16(s2);
    s_hi.y = __float2bfloat16(s3);
    uint2 sv;
    sv.x = *reinterpret_cast<unsigned int *>(&s_lo);
    sv.y = *reinterpret_cast<unsigned int *>(&s_hi);
    hidden_vec[i] = sv;

    // Accumulate sum of squares (use the bf16-rounded values for consistency)
    float v0 = __bfloat162float(s_lo.x);
    float v1 = __bfloat162float(s_lo.y);
    float v2 = __bfloat162float(s_hi.x);
    float v3 = __bfloat162float(s_hi.y);
    local_sum += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
  }
  // Scalar tail
  for (int i = n4 * 4 + tid; i < hidden_dim; i += NORM_BLOCK) {
    float s = __bfloat162float(hidden_row[i]) + __bfloat162float(res_row[i]);
    hidden_row[i] = __float2bfloat16(s);
    float v = __bfloat162float(hidden_row[i]);
    local_sum += v * v;
  }

  // Warp shuffle reduction
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

  // Pass 2: Normalize and scale (read updated hidden, write out). Keep the
  // uint4 path only when row starts are 16-byte aligned.
  const bool use_uint4 =
      ((((uintptr_t)hidden_row | (uintptr_t)weight | (uintptr_t)out_row) & 0xF) == 0);
  if (use_uint4) {
    int n8 = hidden_dim / 8;
    const uint4 *h_vec_r8 = reinterpret_cast<const uint4 *>(hidden_row);
    const uint4 *w_vec8 = reinterpret_cast<const uint4 *>(weight);
    uint4 *out_vec8 = reinterpret_cast<uint4 *>(out_row);

    for (int i = tid; i < n8; i += NORM_BLOCK) {
      uint4 hv = __ldg(h_vec_r8 + i);
      uint4 wv = __ldg(w_vec8 + i);

      uint2 hv0 = make_uint2(hv.x, hv.y);
      uint2 hv1 = make_uint2(hv.z, hv.w);
      uint2 wv0 = make_uint2(wv.x, wv.y);
      uint2 wv1 = make_uint2(wv.z, wv.w);

      __nv_bfloat162 h0_lo = *reinterpret_cast<__nv_bfloat162 *>(&hv0.x);
      __nv_bfloat162 h0_hi = *reinterpret_cast<__nv_bfloat162 *>(&hv0.y);
      __nv_bfloat162 h1_lo = *reinterpret_cast<__nv_bfloat162 *>(&hv1.x);
      __nv_bfloat162 h1_hi = *reinterpret_cast<__nv_bfloat162 *>(&hv1.y);
      __nv_bfloat162 w0_lo = *reinterpret_cast<__nv_bfloat162 *>(&wv0.x);
      __nv_bfloat162 w0_hi = *reinterpret_cast<__nv_bfloat162 *>(&wv0.y);
      __nv_bfloat162 w1_lo = *reinterpret_cast<__nv_bfloat162 *>(&wv1.x);
      __nv_bfloat162 w1_hi = *reinterpret_cast<__nv_bfloat162 *>(&wv1.y);

      __nv_bfloat16 n0 = __float2bfloat16(__bfloat162float(h0_lo.x) * inv_rms);
      __nv_bfloat16 n1 = __float2bfloat16(__bfloat162float(h0_lo.y) * inv_rms);
      __nv_bfloat16 n2 = __float2bfloat16(__bfloat162float(h0_hi.x) * inv_rms);
      __nv_bfloat16 n3 = __float2bfloat16(__bfloat162float(h0_hi.y) * inv_rms);
      __nv_bfloat16 n4 = __float2bfloat16(__bfloat162float(h1_lo.x) * inv_rms);
      __nv_bfloat16 n5 = __float2bfloat16(__bfloat162float(h1_lo.y) * inv_rms);
      __nv_bfloat16 n6 = __float2bfloat16(__bfloat162float(h1_hi.x) * inv_rms);
      __nv_bfloat16 n7 = __float2bfloat16(__bfloat162float(h1_hi.y) * inv_rms);

      uint4 result;
      __nv_bfloat162 r0_lo, r0_hi, r1_lo, r1_hi;
      r0_lo.x = __float2bfloat16(__bfloat162float(n0) * __bfloat162float(w0_lo.x));
      r0_lo.y = __float2bfloat16(__bfloat162float(n1) * __bfloat162float(w0_lo.y));
      r0_hi.x = __float2bfloat16(__bfloat162float(n2) * __bfloat162float(w0_hi.x));
      r0_hi.y = __float2bfloat16(__bfloat162float(n3) * __bfloat162float(w0_hi.y));
      r1_lo.x = __float2bfloat16(__bfloat162float(n4) * __bfloat162float(w1_lo.x));
      r1_lo.y = __float2bfloat16(__bfloat162float(n5) * __bfloat162float(w1_lo.y));
      r1_hi.x = __float2bfloat16(__bfloat162float(n6) * __bfloat162float(w1_hi.x));
      r1_hi.y = __float2bfloat16(__bfloat162float(n7) * __bfloat162float(w1_hi.y));
      result.x = *reinterpret_cast<unsigned int *>(&r0_lo);
      result.y = *reinterpret_cast<unsigned int *>(&r0_hi);
      result.z = *reinterpret_cast<unsigned int *>(&r1_lo);
      result.w = *reinterpret_cast<unsigned int *>(&r1_hi);
      out_vec8[i] = result;
    }
    for (int i = n8 * 8 + tid; i < hidden_dim; i += NORM_BLOCK) {
      __nv_bfloat16 normed = __float2bfloat16(__bfloat162float(hidden_row[i]) * inv_rms);
      out_row[i] = __float2bfloat16(__bfloat162float(normed) * __bfloat162float(weight[i]));
    }
  } else {
    const uint2 *h_vec_r = reinterpret_cast<const uint2 *>(hidden_row);
    const uint2 *w_vec = reinterpret_cast<const uint2 *>(weight);
    uint2 *out_vec = reinterpret_cast<uint2 *>(out_row);
    for (int i = tid; i < n4; i += NORM_BLOCK) {
      uint2 hv = h_vec_r[i];
      uint2 wv = w_vec[i];
      __nv_bfloat162 h_lo = *reinterpret_cast<__nv_bfloat162 *>(&hv.x);
      __nv_bfloat162 h_hi = *reinterpret_cast<__nv_bfloat162 *>(&hv.y);
      __nv_bfloat162 w_lo = *reinterpret_cast<__nv_bfloat162 *>(&wv.x);
      __nv_bfloat162 w_hi = *reinterpret_cast<__nv_bfloat162 *>(&wv.y);

      __nv_bfloat16 n0 = __float2bfloat16(__bfloat162float(h_lo.x) * inv_rms);
      __nv_bfloat16 n1 = __float2bfloat16(__bfloat162float(h_lo.y) * inv_rms);
      __nv_bfloat16 n2 = __float2bfloat16(__bfloat162float(h_hi.x) * inv_rms);
      __nv_bfloat16 n3 = __float2bfloat16(__bfloat162float(h_hi.y) * inv_rms);

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
    for (int i = n4 * 4 + tid; i < hidden_dim; i += NORM_BLOCK) {
      __nv_bfloat16 normed = __float2bfloat16(__bfloat162float(hidden_row[i]) * inv_rms);
      out_row[i] = __float2bfloat16(__bfloat162float(normed) * __bfloat162float(weight[i]));
    }
  }
}

extern "C" {
cudaError_t rms_norm_cuda(const __nv_bfloat16 *x, const __nv_bfloat16 *weight, __nv_bfloat16 *out, int n,
                   float eps, cudaStream_t stream) {
  rms_norm_kernel<<<1, NORM_BLOCK, 0, stream>>>(x, weight, out, n, eps);
    return cudaGetLastError();
}

cudaError_t fused_add_rms_norm_cuda(__nv_bfloat16 *hidden, const __nv_bfloat16 *residual,
                              const __nv_bfloat16 *weight, __nv_bfloat16 *out, int n,
                              float eps, cudaStream_t stream) {
  fused_add_rms_norm_kernel<<<1, NORM_BLOCK, 0, stream>>>(hidden, residual, weight, out, n, eps);
    return cudaGetLastError();
}

cudaError_t fused_add_rms_norm_batched_cuda(__nv_bfloat16 *hidden, const __nv_bfloat16 *residual,
                                      const __nv_bfloat16 *weight, __nv_bfloat16 *out,
                                      int hidden_dim, int seq_len,
                                      float eps, cudaStream_t stream) {
  fused_add_rms_norm_batched_kernel<<<seq_len, NORM_BLOCK, 0, stream>>>(
      hidden, residual, weight, out, hidden_dim, eps);
    return cudaGetLastError();
}

cudaError_t rms_norm_batched_cuda(const __nv_bfloat16 *x, const __nv_bfloat16 *weight,
                            __nv_bfloat16 *out, int hidden_dim, int seq_len,
                            float eps, cudaStream_t stream) {
  rms_norm_batched_kernel<<<seq_len, NORM_BLOCK, 0, stream>>>(
      x, weight, out, hidden_dim, eps);
    return cudaGetLastError();
}

cudaError_t rms_norm_batched_f32_in_cuda(
    const float *x, const __nv_bfloat16 *weight,
    __nv_bfloat16 *out, int hidden_dim, int seq_len,
    float eps, cudaStream_t stream
) {
    rms_norm_batched_f32_in_kernel<<<seq_len, NORM_BLOCK, 0, stream>>>(
        x, weight, out, hidden_dim, eps);
    return cudaGetLastError();
}

cudaError_t add_bf16_into_f32_cuda(
    float *out, const __nv_bfloat16 *in, int n, cudaStream_t stream
) {
    int block = 256;
    int grid = (n + block - 1) / block;
    add_bf16_into_f32_kernel<<<grid, block, 0, stream>>>(out, in, n);
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

// ============================================================================
// RMSNorm with (1+weight) offset — Qwen3.5 / Gemma style
// out[i] = x[i] * (1 + weight[i]) / sqrt(mean(x^2) + eps)
// ============================================================================
cudaError_t rms_norm_offset_cuda(const __nv_bfloat16 *x, const __nv_bfloat16 *weight,
                           __nv_bfloat16 *out, int n, float eps, cudaStream_t stream);

cudaError_t fused_add_rms_norm_offset_cuda(__nv_bfloat16 *hidden, const __nv_bfloat16 *residual,
                                      const __nv_bfloat16 *weight, __nv_bfloat16 *out, int n,
                                      float eps, cudaStream_t stream);

cudaError_t rms_norm_batched_offset_cuda(const __nv_bfloat16 *x, const __nv_bfloat16 *weight,
                                    __nv_bfloat16 *out, int hidden_dim, int seq_len,
                                    float eps, cudaStream_t stream);

cudaError_t rms_norm_gated_cuda(const __nv_bfloat16 *x, const float *weight,
                          const __nv_bfloat16 *gate, __nv_bfloat16 *out,
                          int num_heads, int head_dim, float eps, cudaStream_t stream);
} // extern "C"

// ============================================================================
// (1+weight) RMSNorm kernel
// ============================================================================
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

  // Pass 2: out[i] = (x[i] * inv_rms * (1 + weight[i])) cast to bf16
  // NOTE: GemmaRMSNorm does ALL computation in float32, only rounds to bf16 at the end.
  // No intermediate bf16 rounding (unlike Llama/Qwen3 RMSNorm).
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

// ============================================================================
// Fused Add + (1+weight) RMSNorm
// ============================================================================
__global__ void fused_add_rms_norm_offset_kernel(
    __nv_bfloat16 *__restrict__ hidden,
    const __nv_bfloat16 *__restrict__ residual,
    const __nv_bfloat16 *__restrict__ weight,
    __nv_bfloat16 *__restrict__ out, int n, float eps) {

  int tid = threadIdx.x;
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;
  int n4 = n / 4;

  uint2 *hidden_vec = reinterpret_cast<uint2 *>(hidden);
  const uint2 *res_vec = reinterpret_cast<const uint2 *>(residual);

  float local_sum = 0.0f;
  for (int i = tid; i < n4; i += NORM_BLOCK) {
    uint2 hv = hidden_vec[i];
    uint2 rv = res_vec[i];
    __nv_bfloat162 h_lo = *reinterpret_cast<__nv_bfloat162 *>(&hv.x);
    __nv_bfloat162 h_hi = *reinterpret_cast<__nv_bfloat162 *>(&hv.y);
    __nv_bfloat162 r_lo = *reinterpret_cast<__nv_bfloat162 *>(&rv.x);
    __nv_bfloat162 r_hi = *reinterpret_cast<__nv_bfloat162 *>(&rv.y);

    float s0 = __bfloat162float(h_lo.x) + __bfloat162float(r_lo.x);
    float s1 = __bfloat162float(h_lo.y) + __bfloat162float(r_lo.y);
    float s2 = __bfloat162float(h_hi.x) + __bfloat162float(r_hi.x);
    float s3 = __bfloat162float(h_hi.y) + __bfloat162float(r_hi.y);

    __nv_bfloat162 s_lo, s_hi;
    s_lo.x = __float2bfloat16(s0); s_lo.y = __float2bfloat16(s1);
    s_hi.x = __float2bfloat16(s2); s_hi.y = __float2bfloat16(s3);
    uint2 sv;
    sv.x = *reinterpret_cast<unsigned int *>(&s_lo);
    sv.y = *reinterpret_cast<unsigned int *>(&s_hi);
    hidden_vec[i] = sv;

    float v0 = __bfloat162float(s_lo.x), v1 = __bfloat162float(s_lo.y);
    float v2 = __bfloat162float(s_hi.x), v3 = __bfloat162float(s_hi.y);
    local_sum += v0*v0 + v1*v1 + v2*v2 + v3*v3;
  }
  for (int i = n4*4 + tid; i < n; i += NORM_BLOCK) {
    float s = __bfloat162float(hidden[i]) + __bfloat162float(residual[i]);
    hidden[i] = __float2bfloat16(s);
    // Match vectorized path: sum-of-squares on bf16-rounded value
    float v = __bfloat162float(__float2bfloat16(s));
    local_sum += v * v;
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

  const uint2 *h_vec_r = reinterpret_cast<const uint2 *>(hidden);
  const uint2 *w_vec = reinterpret_cast<const uint2 *>(weight);
  uint2 *out_vec = reinterpret_cast<uint2 *>(out);

  for (int i = tid; i < n4; i += NORM_BLOCK) {
    uint2 hv = h_vec_r[i];
    uint2 wv = w_vec[i];
    __nv_bfloat162 h_lo = *reinterpret_cast<__nv_bfloat162 *>(&hv.x);
    __nv_bfloat162 h_hi = *reinterpret_cast<__nv_bfloat162 *>(&hv.y);
    __nv_bfloat162 w_lo = *reinterpret_cast<__nv_bfloat162 *>(&wv.x);
    __nv_bfloat162 w_hi = *reinterpret_cast<__nv_bfloat162 *>(&wv.y);

    // GemmaRMSNorm: all in float32, only round to bf16 at end
    uint2 result;
    __nv_bfloat162 r_lo, r_hi;
    r_lo.x = __float2bfloat16(__bfloat162float(h_lo.x) * inv_rms * (1.0f + __bfloat162float(w_lo.x)));
    r_lo.y = __float2bfloat16(__bfloat162float(h_lo.y) * inv_rms * (1.0f + __bfloat162float(w_lo.y)));
    r_hi.x = __float2bfloat16(__bfloat162float(h_hi.x) * inv_rms * (1.0f + __bfloat162float(w_hi.x)));
    r_hi.y = __float2bfloat16(__bfloat162float(h_hi.y) * inv_rms * (1.0f + __bfloat162float(w_hi.y)));
    result.x = *reinterpret_cast<unsigned int *>(&r_lo);
    result.y = *reinterpret_cast<unsigned int *>(&r_hi);
    out_vec[i] = result;
  }
  for (int i = n4*4 + tid; i < n; i += NORM_BLOCK) {
    out[i] = __float2bfloat16(__bfloat162float(hidden[i]) * inv_rms * (1.0f + __bfloat162float(weight[i])));
  }
}

// ============================================================================
// Batched (1+weight) RMSNorm: one block per token.
// Grid: <<<seq_len, NORM_BLOCK>>>
// ============================================================================
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

  // Pass 1: sum of squares
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

  // Pass 2: out = x * inv_rms * (1 + weight), all in f32
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

// ============================================================================
// Gated RMSNorm for linear attention output:
//   out = rms_norm(x, f32_weight) * silu(gate)
// Per-head normalization: x is [num_heads * head_dim], weight is [head_dim] (broadcast).
// Grid: num_heads blocks, head_dim threads.
// ============================================================================
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

  // RMSNorm over this head's slice
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

  // SiLU gate
  float g = __bfloat162float(gate[offset]);
  float silu_g = g / (1.0f + expf(-g));

  out[offset] = __float2bfloat16(normed * silu_g);
}

// C API implementations
extern "C" {

cudaError_t rms_norm_offset_cuda(const __nv_bfloat16 *x, const __nv_bfloat16 *weight,
                           __nv_bfloat16 *out, int n, float eps, cudaStream_t stream) {
  rms_norm_offset_kernel<<<1, NORM_BLOCK, 0, stream>>>(x, weight, out, n, eps);
    return cudaGetLastError();
}

cudaError_t fused_add_rms_norm_offset_cuda(__nv_bfloat16 *hidden, const __nv_bfloat16 *residual,
                                      const __nv_bfloat16 *weight, __nv_bfloat16 *out, int n,
                                      float eps, cudaStream_t stream) {
  fused_add_rms_norm_offset_kernel<<<1, NORM_BLOCK, 0, stream>>>(hidden, residual, weight, out, n, eps);
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
