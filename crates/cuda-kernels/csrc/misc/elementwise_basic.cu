// Native CUDA C replacements for trivial Triton AOT kernels.
// These are pure element-wise / lookup ops — bandwidth-bound, no SM-specific tuning.
// Replaces: silu_mul_kernel.py, basic_kernels.py (add, embedding_decode, embedding_batched)

#include "common.cuh"
#include <cuda.h>
#include <stdint.h>

#define BASIC_BLOCK 256

// ============================================================================
// SiLU(gate) * up — element-wise, BF16 in, FP32 compute, BF16 out
// Must compute sigmoid in FP32 to avoid precision loss (matches Triton version).
// ============================================================================
__global__ void silu_mul_native_kernel(
    const __nv_bfloat16 *__restrict__ gate,
    const __nv_bfloat16 *__restrict__ up,
    __nv_bfloat16 *__restrict__ out,
    int n) {
  int idx = blockIdx.x * BASIC_BLOCK + threadIdx.x;
  if (idx < n) {
    float g = __bfloat162float(gate[idx]);
    float u = __bfloat162float(up[idx]);
    float silu = g / (1.0f + expf(-g));
    out[idx] = __float2bfloat16(silu * u);
  }
}

// Keep the same symbol name so existing FFI declarations work without changes.
extern "C" CUresult silu_mul_triton_aot_cuda(
    const uint16_t *gate, const uint16_t *up, uint16_t *out, int n,
    CUstream stream) {
  int grid = (n + BASIC_BLOCK - 1) / BASIC_BLOCK;
  silu_mul_native_kernel<<<grid, BASIC_BLOCK, 0, (cudaStream_t)stream>>>(
      (const __nv_bfloat16 *)gate, (const __nv_bfloat16 *)up,
      (__nv_bfloat16 *)out, n);
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
  int idx = blockIdx.x * BASIC_BLOCK + threadIdx.x;
  if (idx < n) {
    out[idx] = __float2bfloat16(__bfloat162float(a[idx]) + __bfloat162float(b[idx]));
  }
}

extern "C" cudaError_t add_cuda(
    const __nv_bfloat16 *a, const __nv_bfloat16 *b, __nv_bfloat16 *out,
    int n, cudaStream_t stream) {
  int grid = (n + BASIC_BLOCK - 1) / BASIC_BLOCK;
  add_native_kernel<<<grid, BASIC_BLOCK, 0, stream>>>(a, b, out, n);
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
