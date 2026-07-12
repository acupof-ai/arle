// Native CUDA C element-wise kernels.
// These are pure element-wise / lookup ops — bandwidth-bound, no SM-specific tuning.

#include "common.cuh"
#include <cuda.h>
#include <stdint.h>

#define BASIC_BLOCK 256

__device__ __forceinline__ __nv_bfloat162 load_bf16x2(unsigned int packed) {
  return *reinterpret_cast<__nv_bfloat162 *>(&packed);
}

__device__ __forceinline__ unsigned int store_bf16x2(__nv_bfloat162 value) {
  return *reinterpret_cast<unsigned int *>(&value);
}

__device__ __forceinline__ __nv_bfloat16 silu_mul_one(__nv_bfloat16 gate,
                                                       __nv_bfloat16 up) {
  float g = __bfloat162float(gate);
  float u = __bfloat162float(up);
  float silu = g / (1.0f + expf(-g));
  return __float2bfloat16(silu * u);
}

__device__ __forceinline__ __nv_bfloat16 dsv4_swiglu_one(
    __nv_bfloat16 gate, __nv_bfloat16 up, float limit) {
  float g = fminf(__bfloat162float(gate), limit);
  float u = fminf(fmaxf(__bfloat162float(up), -limit), limit);
  float silu = g / (1.0f + expf(-g));
  return __float2bfloat16(silu * u);
}

__host__ __forceinline__ bool is_bf16x4_aligned(const void *ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & (sizeof(uint2) - 1)) == 0;
}

// ============================================================================
// SiLU(gate) * up — element-wise, BF16 in, FP32 compute, BF16 out
// Must compute sigmoid in FP32 to avoid precision loss.
// ============================================================================
__global__ void silu_mul_native_kernel(
    const __nv_bfloat16 *__restrict__ gate,
    const __nv_bfloat16 *__restrict__ up,
    __nv_bfloat16 *__restrict__ out,
    int n) {
  int idx4 = blockIdx.x * BASIC_BLOCK + threadIdx.x;
  int n4 = n / 4;

  const uint2 *gate_vec = reinterpret_cast<const uint2 *>(gate);
  const uint2 *up_vec = reinterpret_cast<const uint2 *>(up);
  uint2 *out_vec = reinterpret_cast<uint2 *>(out);

  if (idx4 < n4) {
    uint2 gv = gate_vec[idx4];
    uint2 uv = up_vec[idx4];
    __nv_bfloat162 g_lo = load_bf16x2(gv.x);
    __nv_bfloat162 g_hi = load_bf16x2(gv.y);
    __nv_bfloat162 u_lo = load_bf16x2(uv.x);
    __nv_bfloat162 u_hi = load_bf16x2(uv.y);

    __nv_bfloat162 r_lo, r_hi;
    r_lo.x = silu_mul_one(g_lo.x, u_lo.x);
    r_lo.y = silu_mul_one(g_lo.y, u_lo.y);
    r_hi.x = silu_mul_one(g_hi.x, u_hi.x);
    r_hi.y = silu_mul_one(g_hi.y, u_hi.y);
    out_vec[idx4] = make_uint2(store_bf16x2(r_lo), store_bf16x2(r_hi));
  }

  for (int idx = n4 * 4 + idx4; idx < n; idx += gridDim.x * BASIC_BLOCK) {
    out[idx] = silu_mul_one(gate[idx], up[idx]);
  }
}

__global__ void silu_mul_scalar_kernel(
    const __nv_bfloat16 *__restrict__ gate,
    const __nv_bfloat16 *__restrict__ up,
    __nv_bfloat16 *__restrict__ out,
    int n) {
  int idx = blockIdx.x * BASIC_BLOCK + threadIdx.x;
  if (idx < n) {
    out[idx] = silu_mul_one(gate[idx], up[idx]);
  }
}

extern "C" CUresult silu_mul_cuda(
    const uint16_t *gate, const uint16_t *up, uint16_t *out, int n,
    CUstream stream) {
  int grid = ((n + 3) / 4 + BASIC_BLOCK - 1) / BASIC_BLOCK;
  if (grid == 0) return CUDA_SUCCESS;
  if (is_bf16x4_aligned(gate) && is_bf16x4_aligned(up) && is_bf16x4_aligned(out)) {
    silu_mul_native_kernel<<<grid, BASIC_BLOCK, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16 *)gate, (const __nv_bfloat16 *)up,
        (__nv_bfloat16 *)out, n);
  } else {
    int scalar_grid = (n + BASIC_BLOCK - 1) / BASIC_BLOCK;
    silu_mul_scalar_kernel<<<scalar_grid, BASIC_BLOCK, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16 *)gate, (const __nv_bfloat16 *)up,
        (__nv_bfloat16 *)out, n);
  }
  return (CUresult)cudaGetLastError();
}

__global__ void dsv4_swiglu_clamped_kernel(
    const __nv_bfloat16 *__restrict__ gate,
    const __nv_bfloat16 *__restrict__ up,
    __nv_bfloat16 *__restrict__ out,
    int n,
    float limit) {
  int idx = blockIdx.x * BASIC_BLOCK + threadIdx.x;
  if (idx < n) {
    out[idx] = dsv4_swiglu_one(gate[idx], up[idx], limit);
  }
}

extern "C" CUresult dsv4_swiglu_clamped_cuda(
    const uint16_t *gate, const uint16_t *up, uint16_t *out, int n,
    float limit, CUstream stream) {
  if (n <= 0) return CUDA_SUCCESS;
  if (!(limit > 0.0f)) return CUDA_ERROR_INVALID_VALUE;
  int grid = (n + BASIC_BLOCK - 1) / BASIC_BLOCK;
  dsv4_swiglu_clamped_kernel<<<grid, BASIC_BLOCK, 0, (cudaStream_t)stream>>>(
      (const __nv_bfloat16 *)gate, (const __nv_bfloat16 *)up,
      (__nv_bfloat16 *)out, n, limit);
  return (CUresult)cudaGetLastError();
}

__global__ void add_scaled_row_kernel(
    const __nv_bfloat16 *__restrict__ row,
    __nv_bfloat16 *__restrict__ out,
    int hidden_dim,
    int token_idx,
    float scale) {
  int idx = blockIdx.x * BASIC_BLOCK + threadIdx.x;
  if (idx < hidden_dim) {
    int out_idx = token_idx * hidden_dim + idx;
    float prev = __bfloat162float(out[out_idx]);
    float value = __bfloat162float(row[idx]);
    out[out_idx] = __float2bfloat16(prev + scale * value);
  }
}

extern "C" CUresult add_scaled_row_cuda(
    const uint16_t *row, uint16_t *out, int hidden_dim, int token_idx,
    float scale, CUstream stream) {
  if (hidden_dim <= 0 || token_idx < 0) return CUDA_ERROR_INVALID_VALUE;
  int grid = (hidden_dim + BASIC_BLOCK - 1) / BASIC_BLOCK;
  add_scaled_row_kernel<<<grid, BASIC_BLOCK, 0, (cudaStream_t)stream>>>(
      (const __nv_bfloat16 *)row, (__nv_bfloat16 *)out, hidden_dim,
      token_idx, scale);
  return (CUresult)cudaGetLastError();
}

__global__ void add_scaled_row_segment_kernel(
    const __nv_bfloat16 *__restrict__ row,
    __nv_bfloat16 *__restrict__ out,
    int row_len,
    int out_hidden_dim,
    int token_idx,
    int segment_offset,
    float scale) {
  int idx = blockIdx.x * BASIC_BLOCK + threadIdx.x;
  if (idx < row_len) {
    int out_idx = token_idx * out_hidden_dim + segment_offset + idx;
    float prev = __bfloat162float(out[out_idx]);
    float value = __bfloat162float(row[idx]);
    out[out_idx] = __float2bfloat16(prev + scale * value);
  }
}

extern "C" CUresult add_scaled_row_segment_cuda(
    const uint16_t *row, uint16_t *out, int row_len, int out_hidden_dim,
    int token_idx, int segment_offset, float scale, CUstream stream) {
  if (row_len <= 0 || out_hidden_dim <= 0 || token_idx < 0 ||
      segment_offset < 0 || segment_offset + row_len > out_hidden_dim) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int grid = (row_len + BASIC_BLOCK - 1) / BASIC_BLOCK;
  add_scaled_row_segment_kernel<<<grid, BASIC_BLOCK, 0, (cudaStream_t)stream>>>(
      (const __nv_bfloat16 *)row, (__nv_bfloat16 *)out, row_len,
      out_hidden_dim, token_idx, segment_offset, scale);
  return (CUresult)cudaGetLastError();
}

// ============================================================================
// Element-wise BF16 add: out = a + b
// ============================================================================
__global__ void add_native_kernel(
    const __nv_bfloat16 *__restrict__ a,
    const __nv_bfloat16 *__restrict__ b,
    __nv_bfloat16 *__restrict__ out,
    int n) {
  int idx4 = blockIdx.x * BASIC_BLOCK + threadIdx.x;
  int n4 = n / 4;

  const uint2 *a_vec = reinterpret_cast<const uint2 *>(a);
  const uint2 *b_vec = reinterpret_cast<const uint2 *>(b);
  uint2 *out_vec = reinterpret_cast<uint2 *>(out);

  if (idx4 < n4) {
    uint2 av = a_vec[idx4];
    uint2 bv = b_vec[idx4];
    __nv_bfloat162 a_lo = load_bf16x2(av.x);
    __nv_bfloat162 a_hi = load_bf16x2(av.y);
    __nv_bfloat162 b_lo = load_bf16x2(bv.x);
    __nv_bfloat162 b_hi = load_bf16x2(bv.y);

    __nv_bfloat162 r_lo = __hadd2_rn(a_lo, b_lo);
    __nv_bfloat162 r_hi = __hadd2_rn(a_hi, b_hi);
    out_vec[idx4] = make_uint2(store_bf16x2(r_lo), store_bf16x2(r_hi));
  }

  for (int idx = n4 * 4 + idx4; idx < n; idx += gridDim.x * BASIC_BLOCK) {
    out[idx] = __hadd_rn(a[idx], b[idx]);
  }
}

__global__ void add_scalar_kernel(
    const __nv_bfloat16 *__restrict__ a,
    const __nv_bfloat16 *__restrict__ b,
    __nv_bfloat16 *__restrict__ out,
    int n) {
  int idx = blockIdx.x * BASIC_BLOCK + threadIdx.x;
  if (idx < n) {
    out[idx] = __hadd_rn(a[idx], b[idx]);
  }
}

extern "C" cudaError_t add_cuda(
    const __nv_bfloat16 *a, const __nv_bfloat16 *b, __nv_bfloat16 *out,
    int n, cudaStream_t stream) {
  int grid = ((n + 3) / 4 + BASIC_BLOCK - 1) / BASIC_BLOCK;
  if (grid == 0) return cudaSuccess;
  if (is_bf16x4_aligned(a) && is_bf16x4_aligned(b) && is_bf16x4_aligned(out)) {
    add_native_kernel<<<grid, BASIC_BLOCK, 0, stream>>>(a, b, out, n);
  } else {
    int scalar_grid = (n + BASIC_BLOCK - 1) / BASIC_BLOCK;
    add_scalar_kernel<<<scalar_grid, BASIC_BLOCK, 0, stream>>>(a, b, out, n);
  }
  return cudaGetLastError();
}

__global__ void add_assign_bf16_kernel(
    __nv_bfloat16 *a,
    const __nv_bfloat16 *b,
    int n) {
  int idx = blockIdx.x * BASIC_BLOCK + threadIdx.x;
  if (idx < n) {
    a[idx] = __hadd_rn(a[idx], b[idx]);
  }
}

extern "C" cudaError_t add_assign_cuda(
    __nv_bfloat16 *a, const __nv_bfloat16 *b,
    int n, cudaStream_t stream) {
  int grid = (n + BASIC_BLOCK - 1) / BASIC_BLOCK;
  if (grid == 0) return cudaSuccess;
  add_assign_bf16_kernel<<<grid, BASIC_BLOCK, 0, stream>>>(a, b, n);
  return cudaGetLastError();
}

// ============================================================================
// Embedding lookup — single token decode
// out[i] = table[token_id * hidden_dim + i] for i in 0..hidden_dim
// ============================================================================
__global__ void embedding_decode_native_kernel(
    const __nv_bfloat16 *__restrict__ table,
    const int *__restrict__ token_id,
    __nv_bfloat16 *__restrict__ out,
    int hidden_dim) {
  int tid = blockIdx.x * BASIC_BLOCK + threadIdx.x;
  if (tid < hidden_dim) {
    out[tid] = __ldg(&table[__ldg(&token_id[0]) * hidden_dim + tid]);
  }
}

extern "C" CUresult embedding_decode_cuda(
    const uint16_t *table, const int *token_id, uint16_t *out,
    int hidden_dim, CUstream stream) {
  int grid = (hidden_dim + BASIC_BLOCK - 1) / BASIC_BLOCK;
  embedding_decode_native_kernel<<<grid, BASIC_BLOCK, 0, (cudaStream_t)stream>>>(
      (const __nv_bfloat16 *)table, token_id, (__nv_bfloat16 *)out,
      hidden_dim);
  return (CUresult)cudaGetLastError();
}

// ============================================================================
// Embedding lookup — batched (B tokens)
// out[b * hidden_dim + i] = table[token_ids[b] * hidden_dim + i]
// ============================================================================
__global__ void embedding_batched_native_kernel(
    const __nv_bfloat16 *__restrict__ table,
    const int *__restrict__ token_ids,
    __nv_bfloat16 *__restrict__ out,
    int hidden_dim,
    int batch_size) {
  int tid = blockIdx.x * BASIC_BLOCK + threadIdx.x;
  int total = batch_size * hidden_dim;
  if (tid < total) {
    int b = tid / hidden_dim;
    int i = tid % hidden_dim;
    out[tid] = __ldg(&table[__ldg(&token_ids[b]) * hidden_dim + i]);
  }
}

extern "C" CUresult embedding_batched_cuda(
    const uint16_t *table, const int *token_ids, uint16_t *out,
    int hidden_dim, int batch_size, CUstream stream) {
  int total = batch_size * hidden_dim;
  int grid = (total + BASIC_BLOCK - 1) / BASIC_BLOCK;
  embedding_batched_native_kernel<<<grid, BASIC_BLOCK, 0, (cudaStream_t)stream>>>(
      (const __nv_bfloat16 *)table, token_ids, (__nv_bfloat16 *)out,
      hidden_dim, batch_size);
  return (CUresult)cudaGetLastError();
}
