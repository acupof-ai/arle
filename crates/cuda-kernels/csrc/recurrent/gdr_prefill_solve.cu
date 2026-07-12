#include "common.cuh"
#include <cuda_runtime.h>

#define GDR_CHUNK_TOKENS 64

__global__ void gdr_prefill_chunk_solve_kernel(
    const float *__restrict__ a_tril,
    __nv_bfloat16 *__restrict__ a_inv,
    int seq_len,
    int num_value_heads) {
  int chunk = blockIdx.x;
  int v_head = blockIdx.y;
  int base = chunk * GDR_CHUNK_TOKENS;

  if (threadIdx.x != 0) {
    return;
  }

  float inv[GDR_CHUNK_TOKENS][GDR_CHUNK_TOKENS];

  for (int i = 0; i < GDR_CHUNK_TOKENS; ++i) {
    for (int j = 0; j < GDR_CHUNK_TOKENS; ++j) {
      inv[i][j] = 0.0f;
    }
  }

  for (int i = 0; i < GDR_CHUNK_TOKENS; ++i) {
    int row_token = base + i;
    if (row_token >= seq_len) {
      break;
    }

    inv[i][i] = 1.0f;
    for (int j = 0; j < i; ++j) {
      size_t a_ij_idx =
          (static_cast<size_t>(row_token) * num_value_heads + v_head) *
              GDR_CHUNK_TOKENS +
          j;
      float value = -a_tril[a_ij_idx];
      for (int k = j + 1; k < i; ++k) {
        size_t a_ik_idx =
            (static_cast<size_t>(row_token) * num_value_heads + v_head) *
                GDR_CHUNK_TOKENS +
            k;
        value -= a_tril[a_ik_idx] * inv[k][j];
      }
      inv[i][j] = value;
    }
  }

  for (int i = 0; i < GDR_CHUNK_TOKENS; ++i) {
    int row_token = base + i;
    if (row_token >= seq_len) {
      break;
    }
    for (int j = 0; j < GDR_CHUNK_TOKENS; ++j) {
      size_t out_idx =
          (static_cast<size_t>(row_token) * num_value_heads + v_head) *
              GDR_CHUNK_TOKENS +
          j;
      a_inv[out_idx] = __float2bfloat16(inv[i][j]);
    }
  }
}

extern "C" cudaError_t gated_delta_rule_prefill_chunk_solve_cuda(
    const float *a_tril,
    __nv_bfloat16 *a_inv,
    int seq_len,
    int num_value_heads,
    cudaStream_t stream) {
  if (seq_len < 0 || num_value_heads <= 0) {
    return cudaErrorInvalidValue;
  }
  int num_chunks = (seq_len + GDR_CHUNK_TOKENS - 1) / GDR_CHUNK_TOKENS;
  if (num_chunks == 0) {
    return cudaSuccess;
  }
  dim3 grid(num_chunks, num_value_heads);
  gdr_prefill_chunk_solve_kernel<<<grid, 32, 0, stream>>>(
      a_tril, a_inv, seq_len, num_value_heads);
  return cudaGetLastError();
}
