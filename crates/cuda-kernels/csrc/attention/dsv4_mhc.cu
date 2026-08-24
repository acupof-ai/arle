#include "common.cuh"
#include <cuda.h>
#include <stdint.h>

#define DSV4_MHC_BLOCK 256
#define DSV4_MHC_MAX 8

__device__ __forceinline__ float bf16_to_f32(const uint16_t value) {
  return __bfloat162float(*reinterpret_cast<const __nv_bfloat16 *>(&value));
}

__device__ __forceinline__ uint16_t f32_to_bf16_bits(const float value) {
  __nv_bfloat16 out = __float2bfloat16(value);
  return *reinterpret_cast<uint16_t *>(&out);
}

__device__ __forceinline__ float dsv4_sigmoid(float value) {
  if (value >= 0.0f) {
    return 1.0f / (1.0f + __expf(-value));
  }
  float expv = __expf(value);
  return expv / (1.0f + expv);
}

__device__ float block_sum(float value) {
  // Sized for the max block (1024 = 32 warps) so launch block size is a free
  // parameter; smaller blocks use a prefix of the scratch — identical math.
  __shared__ float warp_sums[32];
  value = warp_reduce_sum(value);
  if ((threadIdx.x & (WARP_SIZE - 1)) == 0) {
    warp_sums[threadIdx.x / WARP_SIZE] = value;
  }
  __syncthreads();
  value = threadIdx.x < (blockDim.x / WARP_SIZE) ? warp_sums[threadIdx.x] : 0.0f;
  if (threadIdx.x < WARP_SIZE) {
    value = warp_reduce_sum(value);
  }
  return value;
}

// Sinkhorn phases, one row/column per calling lane. The n<=DSV4_MHC_MAX inner
// reductions stay serial in the scalar order, so per-element math is unchanged.
__device__ void row_softmax_plus_eps(float *raw, int n, float eps, int row) {
  float max_value = -INFINITY;
  for (int col = 0; col < n; ++col) {
    max_value = fmaxf(max_value, raw[row * n + col]);
  }
  float denom = 0.0f;
  for (int col = 0; col < n; ++col) {
    float value = expf(raw[row * n + col] - max_value);
    raw[row * n + col] = value;
    denom += value;
  }
  for (int col = 0; col < n; ++col) {
    raw[row * n + col] = raw[row * n + col] / denom + eps;
  }
}

__device__ void row_normalize(float *raw, int n, float eps, int row) {
  float sum = eps;
  for (int col = 0; col < n; ++col) {
    sum += raw[row * n + col];
  }
  for (int col = 0; col < n; ++col) {
    raw[row * n + col] /= sum;
  }
}

__device__ void column_normalize(float *raw, int n, float eps, int col) {
  float sum = eps;
  for (int row = 0; row < n; ++row) {
    sum += raw[row * n + col];
  }
  for (int row = 0; row < n; ++row) {
    raw[row * n + col] /= sum;
  }
}

// Params tail on the FIRST WARP: per-lane pre/post sigmoids, sinkhorn over
// `raw_shared` (hc_mult^2 <= 64 = 2*WARP_SIZE), comb writeback. `pre` is also
// staged into `pre_shared` so the fused kernel can consume it without a global
// round trip. Warp-synchronous only (__syncwarp, no block syncs inside):
// callers may retire warps > 0 before entry, or park them at a __syncthreads.
__device__ void dsv4_mhc_params_tail(
    const float *mixes_shared,
    const uint16_t *__restrict__ base,
    const uint16_t *__restrict__ scale,
    float *raw_shared,
    float *pre_shared,
    float *__restrict__ pre,
    float *__restrict__ post,
    float *__restrict__ comb,
    int token,
    int hc_mult,
    float eps,
    int sinkhorn_iters) {
  const int lane = threadIdx.x;
  const int token_hc = token * hc_mult;
  const int token_comb = token * hc_mult * hc_mult;
  if (lane < hc_mult) {
    float pre_value = dsv4_sigmoid(bf16_to_f32(scale[0]) * mixes_shared[lane] +
                                   bf16_to_f32(base[lane])) +
                      eps;
    pre_shared[lane] = pre_value;
    pre[token_hc + lane] = pre_value;
    post[token_hc + lane] =
        2.0f * dsv4_sigmoid(bf16_to_f32(scale[1]) * mixes_shared[hc_mult + lane] +
                            bf16_to_f32(base[hc_mult + lane]));
  }
  const float scale2 = bf16_to_f32(scale[2]);
  for (int idx = lane; idx < hc_mult * hc_mult; idx += WARP_SIZE) {
    raw_shared[idx] = scale2 * mixes_shared[2 * hc_mult + idx] +
                      bf16_to_f32(base[2 * hc_mult + idx]);
  }
  __syncwarp();
  if (lane < hc_mult) row_softmax_plus_eps(raw_shared, hc_mult, eps, lane);
  __syncwarp();
  if (lane < hc_mult) column_normalize(raw_shared, hc_mult, eps, lane);
  for (int iter = 1; iter < sinkhorn_iters; ++iter) {
    __syncwarp();
    if (lane < hc_mult) row_normalize(raw_shared, hc_mult, eps, lane);
    __syncwarp();
    if (lane < hc_mult) column_normalize(raw_shared, hc_mult, eps, lane);
  }
  __syncwarp();
  for (int idx = lane; idx < hc_mult * hc_mult; idx += WARP_SIZE) {
    comb[token_comb + idx] = raw_shared[idx];
  }
}

__global__ void dsv4_mhc_expand_kernel(
    const uint16_t *__restrict__ embeddings,
    uint16_t *__restrict__ out,
    int num_tokens,
    int hidden_size,
    int hc_mult) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = num_tokens * hidden_size * hc_mult;
  if (idx >= total) return;
  int col = idx % hidden_size;
  int token = idx / (hidden_size * hc_mult);
  out[idx] = embeddings[token * hidden_size + col];
}

extern "C" CUresult dsv4_mhc_expand_cuda(
    const uint16_t *embeddings,
    uint16_t *out,
    int num_tokens,
    int hidden_size,
    int hc_mult,
    CUstream stream) {
  if (num_tokens < 0 || hidden_size <= 0 || hc_mult <= 0) return CUDA_ERROR_INVALID_VALUE;
  int total = num_tokens * hidden_size * hc_mult;
  if (total == 0) return CUDA_SUCCESS;
  int grid = (total + DSV4_MHC_BLOCK - 1) / DSV4_MHC_BLOCK;
  dsv4_mhc_expand_kernel<<<grid, DSV4_MHC_BLOCK, 0, (cudaStream_t)stream>>>(
      embeddings, out, num_tokens, hidden_size, hc_mult);
  return (CUresult)cudaGetLastError();
}

__global__ void dsv4_mhc_params_kernel(
    const uint16_t *__restrict__ residual,
    const uint16_t *__restrict__ mixes,
    const uint16_t *__restrict__ base,
    const uint16_t *__restrict__ scale,
    float *__restrict__ pre,
    float *__restrict__ post,
    float *__restrict__ comb,
    int num_tokens,
    int residual_hidden_dim,
    int mix_dim,
    int hc_mult,
    float eps,
    int sinkhorn_iters) {
  int token = blockIdx.x;
  if (token >= num_tokens) return;

  float sumsq = 0.0f;
  int row_start = token * residual_hidden_dim;
  for (int idx = threadIdx.x; idx < residual_hidden_dim; idx += blockDim.x) {
    float value = bf16_to_f32(residual[row_start + idx]);
    sumsq += value * value;
  }
  sumsq = block_sum(sumsq);

  __shared__ float rsqrt_shared;
  __shared__ float mixes_shared[DSV4_MHC_MAX * (2 + DSV4_MHC_MAX)];
  if (threadIdx.x == 0) {
    rsqrt_shared = rsqrtf(sumsq / fmaxf((float)residual_hidden_dim, 1.0f) + eps);
  }
  __syncthreads();

  int need_mix = hc_mult * (2 + hc_mult);
  for (int idx = threadIdx.x; idx < need_mix; idx += blockDim.x) {
    mixes_shared[idx] = bf16_to_f32(mixes[token * mix_dim + idx]) * rsqrt_shared;
  }
  __syncthreads();

  // No block-level syncs remain: warps > 0 retire, the tail is warp 0 only.
  __shared__ float raw_shared[DSV4_MHC_MAX * DSV4_MHC_MAX];
  __shared__ float pre_shared[DSV4_MHC_MAX];
  if (threadIdx.x >= WARP_SIZE) return;
  dsv4_mhc_params_tail(mixes_shared, base, scale, raw_shared, pre_shared, pre,
                       post, comb, token, hc_mult, eps, sinkhorn_iters);
}

extern "C" CUresult dsv4_mhc_params_cuda(
    const uint16_t *residual,
    const uint16_t *mixes,
    const uint16_t *base,
    const uint16_t *scale,
    float *pre,
    float *post,
    float *comb,
    int num_tokens,
    int residual_hidden_dim,
    int mix_dim,
    int hc_mult,
    float eps,
    int sinkhorn_iters,
    CUstream stream) {
  if (num_tokens < 0 || residual_hidden_dim <= 0 || mix_dim <= 0 ||
      hc_mult <= 0 || hc_mult > DSV4_MHC_MAX ||
      mix_dim < hc_mult * (2 + hc_mult) || sinkhorn_iters <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (num_tokens == 0) return CUDA_SUCCESS;
  // One block reads the whole hc-stream row (32KB at H=4096, hc_mult=4) for
  // the rms factor alone: 1024 threads (idle-GPU CUDA timing: 16.22us @256
  // -> 8.51us @1024, -48%).
  dsv4_mhc_params_kernel<<<num_tokens, 1024, 0, (cudaStream_t)stream>>>(
      residual, mixes, base, scale, pre, post, comb, num_tokens,
      residual_hidden_dim, mix_dim, hc_mult, eps, sinkhorn_iters);
  return (CUresult)cudaGetLastError();
}

// Fused hc_pre + rms_norm for the decode/prefill critical path: mixes the
// hc stream into one lane (x[col] = sum_lane pre[lane]*stream[lane*H+col]),
// rounds to bf16 (matching the unfused chain's intermediate buffer), then
// rms-normalizes and scales in the same block — one kernel boundary and one
// 2*H-byte round trip instead of two kernels + an intermediate tensor.
// Grid: one block per token, DSV4_MHC_BLOCK threads; the bf16 mixed row is
// staged in shared memory (H bf16 <= 16KB at H=7168).
__global__ void dsv4_mhc_pre_rms_norm_kernel(
    const uint16_t *__restrict__ residual,
    const float *__restrict__ pre,
    const uint16_t *__restrict__ weight,
    uint16_t *__restrict__ out,
    int num_tokens,
    int hidden_size,
    int hc_mult,
    float eps) {
  extern __shared__ uint16_t x_row[];
  const int token = blockIdx.x;
  if (token >= num_tokens) return;

  float pre_lane[DSV4_MHC_MAX];
#pragma unroll
  for (int lane = 0; lane < DSV4_MHC_MAX; ++lane) {
    pre_lane[lane] = (lane < hc_mult) ? pre[token * hc_mult + lane] : 0.0f;
  }

  const int residual_base = token * hidden_size * hc_mult;
  float local_sum = 0.0f;
  for (int col = threadIdx.x; col < hidden_size; col += blockDim.x) {
    float value = 0.0f;
    for (int lane = 0; lane < hc_mult; ++lane) {
      value += pre_lane[lane] *
               bf16_to_f32(residual[residual_base + lane * hidden_size + col]);
    }
    const uint16_t bits = f32_to_bf16_bits(value);
    x_row[col] = bits;
    const float rounded = bf16_to_f32(bits);
    local_sum += rounded * rounded;
  }
  const float total = block_sum(local_sum);

  __shared__ float s_inv_rms;
  if (threadIdx.x == 0) {
    s_inv_rms = 1.0f / sqrtf(total / (float)hidden_size + eps);
  }
  __syncthreads();
  const float inv_rms = s_inv_rms;

  uint16_t *out_row = out + token * hidden_size;
  for (int col = threadIdx.x; col < hidden_size; col += blockDim.x) {
    const float value = bf16_to_f32(x_row[col]) * inv_rms *
                        bf16_to_f32(weight[col]);
    out_row[col] = f32_to_bf16_bits(value);
  }
}

extern "C" CUresult dsv4_mhc_pre_rms_norm_cuda(
    const uint16_t *residual,
    const float *pre,
    const uint16_t *weight,
    uint16_t *out,
    int num_tokens,
    int hidden_size,
    int hc_mult,
    float eps,
    CUstream stream) {
  if (num_tokens < 0 || hidden_size <= 0 || hc_mult <= 0 ||
      hc_mult > DSV4_MHC_MAX) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (num_tokens == 0) return CUDA_SUCCESS;
  const size_t shmem = (size_t)hidden_size * sizeof(uint16_t);
  if (shmem > 192 * 1024) return CUDA_ERROR_INVALID_VALUE;
  if (shmem > 48 * 1024) {
    cudaError_t attr = cudaFuncSetAttribute(
        dsv4_mhc_pre_rms_norm_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
    if (attr != cudaSuccess) return (CUresult)attr;
  }
  // Same single-block-bandwidth shape: 9.70us @256 -> 4.80us @1024 (-51%).
  dsv4_mhc_pre_rms_norm_kernel<<<num_tokens, 1024, shmem,
                                 (cudaStream_t)stream>>>(
      residual, pre, weight, out, num_tokens, hidden_size, hc_mult, eps);
  return (CUresult)cudaGetLastError();
}


__global__ void dsv4_mhc_post_kernel(
    const uint16_t *__restrict__ new_x,
    const uint16_t *__restrict__ residual,
    const float *__restrict__ post,
    const float *__restrict__ comb,
    uint16_t *__restrict__ out,
    int num_tokens,
    int hidden_size,
    int hc_mult) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = num_tokens * hidden_size * hc_mult;
  if (idx >= total) return;
  int col = idx % hidden_size;
  int dst_lane = (idx / hidden_size) % hc_mult;
  int token = idx / (hidden_size * hc_mult);
  int token_hc = token * hc_mult;
  int token_comb = token * hc_mult * hc_mult;
  int residual_base = token * hidden_size * hc_mult;
  float value = post[token_hc + dst_lane] *
                bf16_to_f32(new_x[token * hidden_size + col]);
  for (int src_lane = 0; src_lane < hc_mult; ++src_lane) {
    value += comb[token_comb + dst_lane * hc_mult + src_lane] *
             bf16_to_f32(residual[residual_base + src_lane * hidden_size + col]);
  }
  out[idx] = f32_to_bf16_bits(value);
}

extern "C" CUresult dsv4_mhc_post_cuda(
    const uint16_t *new_x,
    const uint16_t *residual,
    const float *post,
    const float *comb,
    uint16_t *out,
    int num_tokens,
    int hidden_size,
    int hc_mult,
    CUstream stream) {
  if (num_tokens < 0 || hidden_size <= 0 || hc_mult <= 0) return CUDA_ERROR_INVALID_VALUE;
  int total = num_tokens * hidden_size * hc_mult;
  if (total == 0) return CUDA_SUCCESS;
  int grid = (total + DSV4_MHC_BLOCK - 1) / DSV4_MHC_BLOCK;
  dsv4_mhc_post_kernel<<<grid, DSV4_MHC_BLOCK, 0, (cudaStream_t)stream>>>(
      new_x, residual, post, comb, out, num_tokens, hidden_size, hc_mult);
  return (CUresult)cudaGetLastError();
}

__global__ void dsv4_mhc_head_pre_kernel(
    const uint16_t *__restrict__ residual_row,
    const uint16_t *__restrict__ mixes,
    const uint16_t *__restrict__ base,
    const uint16_t *__restrict__ scale,
    uint16_t *__restrict__ out,
    int residual_hidden_dim,
    int hidden_size,
    int hc_mult,
    float eps) {
  float sumsq = 0.0f;
  for (int idx = threadIdx.x; idx < residual_hidden_dim; idx += blockDim.x) {
    float value = bf16_to_f32(residual_row[idx]);
    sumsq += value * value;
  }
  sumsq = block_sum(sumsq);
  __shared__ float pre[DSV4_MHC_MAX];
  if (threadIdx.x == 0) {
    float rsqrt_value = rsqrtf(sumsq / fmaxf((float)residual_hidden_dim, 1.0f) + eps);
    float scale0 = bf16_to_f32(scale[0]);
    for (int lane = 0; lane < hc_mult; ++lane) {
      pre[lane] = dsv4_sigmoid(scale0 * bf16_to_f32(mixes[lane]) * rsqrt_value +
                               bf16_to_f32(base[lane])) +
                  eps;
    }
  }
  __syncthreads();
  for (int col = threadIdx.x; col < hidden_size; col += blockDim.x) {
    float value = 0.0f;
    for (int lane = 0; lane < hc_mult; ++lane) {
      value += pre[lane] * bf16_to_f32(residual_row[lane * hidden_size + col]);
    }
    out[col] = f32_to_bf16_bits(value);
  }
}

extern "C" CUresult dsv4_mhc_head_pre_cuda(
    const uint16_t *residual_row,
    const uint16_t *mixes,
    const uint16_t *base,
    const uint16_t *scale,
    uint16_t *out,
    int residual_hidden_dim,
    int hidden_size,
    int hc_mult,
    float eps,
    CUstream stream) {
  if (residual_hidden_dim <= 0 || hidden_size <= 0 || hc_mult <= 0 ||
      hc_mult > DSV4_MHC_MAX || residual_hidden_dim != hidden_size * hc_mult) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  dsv4_mhc_head_pre_kernel<<<1, DSV4_MHC_BLOCK, 0, (cudaStream_t)stream>>>(
      residual_row, mixes, base, scale, out, residual_hidden_dim, hidden_size,
      hc_mult, eps);
  return (CUresult)cudaGetLastError();
}

__global__ void dsv4_mhc_lane_mean_kernel(
    const uint16_t *__restrict__ stream,
    uint16_t *__restrict__ out,
    int num_tokens,
    int hidden_size,
    int hc_mult,
    int out_stride,
    int tap_offset) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = num_tokens * hidden_size;
  if (idx >= total) return;
  int token = idx / hidden_size;
  int col = idx % hidden_size;
  int64_t stream_base =
      (int64_t)token * hc_mult * hidden_size + col;
  float sum = 0.0f;
  for (int lane = 0; lane < hc_mult; ++lane) {
    sum += bf16_to_f32(stream[stream_base + lane * hidden_size]);
  }
  int64_t out_idx = (int64_t)token * out_stride + tap_offset + col;
  out[out_idx] =
      f32_to_bf16_bits(sum / (float)hc_mult);
}

extern "C" CUresult dsv4_mhc_lane_mean_cuda(
    const uint16_t *stream,
    uint16_t *out,
    int num_tokens,
    int hidden_size,
    int hc_mult,
    int out_stride,
    int tap_offset,
    CUstream cuda_stream) {
  if (num_tokens < 0 || hidden_size <= 0 || hc_mult <= 0 ||
      tap_offset < 0 || out_stride < tap_offset + hidden_size) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (num_tokens == 0) return CUDA_SUCCESS;
  int total = num_tokens * hidden_size;
  int grid = (total + DSV4_MHC_BLOCK - 1) / DSV4_MHC_BLOCK;
  dsv4_mhc_lane_mean_kernel<<<grid, DSV4_MHC_BLOCK, 0,
                               (cudaStream_t)cuda_stream>>>(
      stream, out, num_tokens, hidden_size, hc_mult, out_stride, tap_offset);
  return (CUresult)cudaGetLastError();
}

__global__ void dsv4_mtp_add_eproj_hproj_kernel(
    const uint16_t *__restrict__ e_proj,
    const uint16_t *__restrict__ h_proj,
    uint16_t *__restrict__ out_stream,
    int hidden_size,
    int hc_mult) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = hidden_size * hc_mult;
  if (idx >= total) return;
  int col = idx % hidden_size;
  float value = bf16_to_f32(e_proj[col]) + bf16_to_f32(h_proj[idx]);
  out_stream[idx] = f32_to_bf16_bits(value);
}

extern "C" CUresult dsv4_mtp_add_eproj_hproj_cuda(
    const uint16_t *e_proj,
    const uint16_t *h_proj,
    uint16_t *out_stream,
    int hidden_size,
    int hc_mult,
    CUstream stream) {
  if (hidden_size <= 0 || hc_mult <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int total = hidden_size * hc_mult;
  int grid = (total + DSV4_MHC_BLOCK - 1) / DSV4_MHC_BLOCK;
  dsv4_mtp_add_eproj_hproj_kernel<<<grid, DSV4_MHC_BLOCK, 0, (cudaStream_t)stream>>>(
      e_proj, h_proj, out_stream, hidden_size, hc_mult);
  return (CUresult)cudaGetLastError();
}
