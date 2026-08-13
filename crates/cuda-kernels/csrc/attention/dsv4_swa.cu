#include "dsv4_attention_common.cuh"

__global__ void dsv4_update_window_cache_kernel(
    const uint16_t *__restrict__ k_new,
    uint16_t *__restrict__ window_cache,
    int num_tokens,
    int start_pos,
    int sliding_window,
    int head_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = num_tokens * head_dim;
  if (idx >= total) return;
  int token = idx / head_dim;
  int col = idx - token * head_dim;
  int slot = (start_pos + token) % sliding_window;
  window_cache[slot * head_dim + col] = k_new[token * head_dim + col];
}

__global__ void dsv4_update_window_cache_start_pos_ptr_kernel(
    const uint16_t *__restrict__ k_new,
    uint16_t *__restrict__ window_cache,
    int num_tokens,
    const int *__restrict__ start_pos_ptr,
    int sliding_window,
    int head_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = num_tokens * head_dim;
  if (idx >= total) return;
  int token = idx / head_dim;
  int col = idx - token * head_dim;
  int start_pos = *start_pos_ptr;
  int slot = (start_pos + token) % sliding_window;
  window_cache[slot * head_dim + col] = k_new[token * head_dim + col];
}

// Batched over non-contiguous per-row buffers; replaces n single-row launches.
__global__ void dsv4_update_window_cache_batched_ptr_kernel(
    const uint16_t *const *__restrict__ k_arr,
    uint16_t *const *__restrict__ cache_arr,
    int n,
    const int *__restrict__ start_pos,
    int sliding_window,
    int head_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n * head_dim;
  if (idx >= total) return;
  int rowi = idx / head_dim;
  int col = idx - rowi * head_dim;
  int slot = start_pos[rowi] % sliding_window;
  cache_arr[rowi][slot * head_dim + col] = k_arr[rowi][col];
}

extern "C" CUresult dsv4_update_window_cache_batched_ptr_cuda(
    const uint16_t *const *k_arr,
    uint16_t *const *cache_arr,
    int n,
    const int *start_pos,
    int sliding_window,
    int head_dim,
    CUstream stream) {
  if (n < 0 || start_pos == nullptr || sliding_window <= 0 || head_dim <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int total = n * head_dim;
  if (total == 0) return CUDA_SUCCESS;
  if (k_arr == nullptr || cache_arr == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int grid = (total + DSV4_ATTN_BLOCK - 1) / DSV4_ATTN_BLOCK;
  dsv4_update_window_cache_batched_ptr_kernel<<<grid, DSV4_ATTN_BLOCK, 0, (cudaStream_t)stream>>>(
      k_arr, cache_arr, n, start_pos, sliding_window, head_dim);
  return (CUresult)cudaGetLastError();
}

extern "C" CUresult dsv4_update_window_cache_cuda(
    const uint16_t *k_new,
    uint16_t *window_cache,
    int num_tokens,
    int start_pos,
    int sliding_window,
    int head_dim,
    CUstream stream) {
  if (num_tokens < 0 || start_pos < 0 || sliding_window <= 0 || head_dim <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int total = num_tokens * head_dim;
  if (total == 0) return CUDA_SUCCESS;
  int grid = (total + DSV4_ATTN_BLOCK - 1) / DSV4_ATTN_BLOCK;
  dsv4_update_window_cache_kernel<<<grid, DSV4_ATTN_BLOCK, 0, (cudaStream_t)stream>>>(
      k_new, window_cache, num_tokens, start_pos, sliding_window, head_dim);
  return (CUresult)cudaGetLastError();
}

extern "C" CUresult dsv4_update_window_cache_start_pos_ptr_cuda(
    const uint16_t *k_new,
    uint16_t *window_cache,
    int num_tokens,
    const int *start_pos_ptr,
    int sliding_window,
    int head_dim,
    CUstream stream) {
  if (num_tokens < 0 || start_pos_ptr == nullptr || sliding_window <= 0 || head_dim <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int total = num_tokens * head_dim;
  if (total == 0) return CUDA_SUCCESS;
  int grid = (total + DSV4_ATTN_BLOCK - 1) / DSV4_ATTN_BLOCK;
  dsv4_update_window_cache_start_pos_ptr_kernel<<<grid, DSV4_ATTN_BLOCK, 0, (cudaStream_t)stream>>>(
      k_new, window_cache, num_tokens, start_pos_ptr, sliding_window, head_dim);
  return (CUresult)cudaGetLastError();
}
