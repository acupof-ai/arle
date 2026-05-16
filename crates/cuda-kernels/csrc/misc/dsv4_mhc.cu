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
    return 1.0f / (1.0f + expf(-value));
  }
  float expv = expf(value);
  return expv / (1.0f + expv);
}

__device__ float block_sum(float value) {
  __shared__ float warp_sums[DSV4_MHC_BLOCK / WARP_SIZE];
  value = warp_reduce_sum(value);
  if ((threadIdx.x & (WARP_SIZE - 1)) == 0) {
    warp_sums[threadIdx.x / WARP_SIZE] = value;
  }
  __syncthreads();
  value = threadIdx.x < (DSV4_MHC_BLOCK / WARP_SIZE) ? warp_sums[threadIdx.x] : 0.0f;
  if (threadIdx.x < WARP_SIZE) {
    value = warp_reduce_sum(value);
  }
  return value;
}

__device__ void row_softmax_plus_eps(float *raw, int n, float eps) {
  for (int row = 0; row < n; ++row) {
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
}

__device__ void row_normalize(float *raw, int n, float eps) {
  for (int row = 0; row < n; ++row) {
    float sum = eps;
    for (int col = 0; col < n; ++col) {
      sum += raw[row * n + col];
    }
    for (int col = 0; col < n; ++col) {
      raw[row * n + col] /= sum;
    }
  }
}

__device__ void column_normalize(float *raw, int n, float eps) {
  for (int col = 0; col < n; ++col) {
    float sum = eps;
    for (int row = 0; row < n; ++row) {
      sum += raw[row * n + col];
    }
    for (int row = 0; row < n; ++row) {
      raw[row * n + col] /= sum;
    }
  }
}

__device__ __forceinline__ void row_softmax_plus_eps4(float *raw, float eps) {
#pragma unroll
  for (int row = 0; row < 4; ++row) {
    int base = row * 4;
    float max_value = raw[base];
    max_value = fmaxf(max_value, raw[base + 1]);
    max_value = fmaxf(max_value, raw[base + 2]);
    max_value = fmaxf(max_value, raw[base + 3]);

    float value0 = expf(raw[base] - max_value);
    float value1 = expf(raw[base + 1] - max_value);
    float value2 = expf(raw[base + 2] - max_value);
    float value3 = expf(raw[base + 3] - max_value);
    float denom = value0 + value1 + value2 + value3;

    raw[base] = value0 / denom + eps;
    raw[base + 1] = value1 / denom + eps;
    raw[base + 2] = value2 / denom + eps;
    raw[base + 3] = value3 / denom + eps;
  }
}

__device__ __forceinline__ void row_normalize4(float *raw, float eps) {
#pragma unroll
  for (int row = 0; row < 4; ++row) {
    int base = row * 4;
    float sum = eps + raw[base] + raw[base + 1] + raw[base + 2] + raw[base + 3];
    raw[base] /= sum;
    raw[base + 1] /= sum;
    raw[base + 2] /= sum;
    raw[base + 3] /= sum;
  }
}

__device__ __forceinline__ void column_normalize4(float *raw, float eps) {
#pragma unroll
  for (int col = 0; col < 4; ++col) {
    float sum = eps + raw[col] + raw[4 + col] + raw[8 + col] + raw[12 + col];
    raw[col] /= sum;
    raw[4 + col] /= sum;
    raw[8 + col] /= sum;
    raw[12 + col] /= sum;
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

  if (threadIdx.x != 0) return;

  float scale0 = bf16_to_f32(scale[0]);
  float scale1 = bf16_to_f32(scale[1]);
  float scale2 = bf16_to_f32(scale[2]);
  int token_hc = token * hc_mult;
  int token_comb = token * hc_mult * hc_mult;
  for (int lane = 0; lane < hc_mult; ++lane) {
    pre[token_hc + lane] =
        dsv4_sigmoid(scale0 * mixes_shared[lane] + bf16_to_f32(base[lane])) + eps;
    post[token_hc + lane] =
        2.0f * dsv4_sigmoid(scale1 * mixes_shared[hc_mult + lane] +
                            bf16_to_f32(base[hc_mult + lane]));
  }

  float raw[DSV4_MHC_MAX * DSV4_MHC_MAX];
  for (int row = 0; row < hc_mult; ++row) {
    for (int col = 0; col < hc_mult; ++col) {
      int idx = row * hc_mult + col;
      raw[idx] = scale2 * mixes_shared[2 * hc_mult + idx] +
                 bf16_to_f32(base[2 * hc_mult + idx]);
    }
  }
  if (hc_mult == 4) {
    row_softmax_plus_eps4(raw, eps);
    column_normalize4(raw, eps);
    for (int iter = 1; iter < sinkhorn_iters; ++iter) {
      row_normalize4(raw, eps);
      column_normalize4(raw, eps);
    }
  } else {
    row_softmax_plus_eps(raw, hc_mult, eps);
    column_normalize(raw, hc_mult, eps);
    for (int iter = 1; iter < sinkhorn_iters; ++iter) {
      row_normalize(raw, hc_mult, eps);
      column_normalize(raw, hc_mult, eps);
    }
  }
  for (int idx = 0; idx < hc_mult * hc_mult; ++idx) {
    comb[token_comb + idx] = raw[idx];
  }
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
  dsv4_mhc_params_kernel<<<num_tokens, DSV4_MHC_BLOCK, 0, (cudaStream_t)stream>>>(
      residual, mixes, base, scale, pre, post, comb, num_tokens,
      residual_hidden_dim, mix_dim, hc_mult, eps, sinkhorn_iters);
  return (CUresult)cudaGetLastError();
}

__global__ void dsv4_mhc_pre_kernel(
    const uint16_t *__restrict__ residual,
    const float *__restrict__ pre,
    uint16_t *__restrict__ out,
    int num_tokens,
    int hidden_size,
    int hc_mult) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = num_tokens * hidden_size;
  if (idx >= total) return;
  int col = idx % hidden_size;
  int token = idx / hidden_size;
  int residual_base = token * hidden_size * hc_mult;
  float value = 0.0f;
  for (int lane = 0; lane < hc_mult; ++lane) {
    value += pre[token * hc_mult + lane] *
             bf16_to_f32(residual[residual_base + lane * hidden_size + col]);
  }
  out[idx] = f32_to_bf16_bits(value);
}

extern "C" CUresult dsv4_mhc_pre_cuda(
    const uint16_t *residual,
    const float *pre,
    uint16_t *out,
    int num_tokens,
    int hidden_size,
    int hc_mult,
    CUstream stream) {
  if (num_tokens < 0 || hidden_size <= 0 || hc_mult <= 0) return CUDA_ERROR_INVALID_VALUE;
  int total = num_tokens * hidden_size;
  if (total == 0) return CUDA_SUCCESS;
  int grid = (total + DSV4_MHC_BLOCK - 1) / DSV4_MHC_BLOCK;
  dsv4_mhc_pre_kernel<<<grid, DSV4_MHC_BLOCK, 0, (cudaStream_t)stream>>>(
      residual, pre, out, num_tokens, hidden_size, hc_mult);
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
