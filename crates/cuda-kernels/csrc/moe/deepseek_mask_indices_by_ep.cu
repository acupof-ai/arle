#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

// DeepSeek TileKernels-compatible EP/TP expert id masking.
//
// Formula adapted from deepseek-ai/TileKernels
// tile_kernels/moe/mask_indices_by_tp_kernel.py:
// keep only experts assigned to tp_rank, then compact global expert ids by
// removing the gaps introduced by other TP ranks inside the MoE-DP group.

#define MASK_EP_BLOCK 256

template <typename T>
__global__ void dsv4_mask_indices_by_ep_kernel(
    const T *__restrict__ indices,
    T *__restrict__ masked_indices,
    int total_indices,
    int experts_per_ep_rank,
    int experts_per_moe_dp_group,
    int num_tp_ranks,
    int tp_rank) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_indices) {
    return;
  }

  T value = indices[idx];
  if (value < 0 ||
      ((value / experts_per_ep_rank) % num_tp_ranks) != tp_rank) {
    masked_indices[idx] = static_cast<T>(-1);
    return;
  }

  value -= static_cast<T>(tp_rank * experts_per_ep_rank);
  T dp_rank = value / experts_per_moe_dp_group;
  value -= dp_rank * static_cast<T>(experts_per_moe_dp_group - experts_per_ep_rank);
  masked_indices[idx] = value < 0 ? static_cast<T>(-1) : value;
}

template <typename T>
static CUresult launch_mask_indices_by_ep(
    const T *indices,
    T *masked_indices,
    int num_tokens,
    int num_topk,
    int experts_per_ep_rank,
    int experts_per_moe_dp_group,
    int num_tp_ranks,
    int tp_rank,
    CUstream stream) {
  if (num_tokens < 0 || num_topk < 0 || experts_per_ep_rank <= 0 ||
      experts_per_moe_dp_group <= 0 || num_tp_ranks <= 0 || tp_rank < 0 ||
      tp_rank >= num_tp_ranks) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int total_indices = num_tokens * num_topk;
  if (total_indices == 0) {
    return CUDA_SUCCESS;
  }
  int grid = (total_indices + MASK_EP_BLOCK - 1) / MASK_EP_BLOCK;
  dsv4_mask_indices_by_ep_kernel<T>
      <<<grid, MASK_EP_BLOCK, 0, (cudaStream_t)stream>>>(
          indices,
          masked_indices,
          total_indices,
          experts_per_ep_rank,
          experts_per_moe_dp_group,
          num_tp_ranks,
          tp_rank);
  return (CUresult)cudaGetLastError();
}

extern "C" {

CUresult dsv4_mask_indices_by_ep_i64_cuda(
    const int64_t *indices,
    int64_t *masked_indices,
    int num_tokens,
    int num_topk,
    int experts_per_ep_rank,
    int experts_per_moe_dp_group,
    int num_tp_ranks,
    int tp_rank,
    CUstream stream) {
  return launch_mask_indices_by_ep<int64_t>(
      indices,
      masked_indices,
      num_tokens,
      num_topk,
      experts_per_ep_rank,
      experts_per_moe_dp_group,
      num_tp_ranks,
      tp_rank,
      stream);
}

CUresult dsv4_mask_indices_by_ep_i32_cuda(
    const int32_t *indices,
    int32_t *masked_indices,
    int num_tokens,
    int num_topk,
    int experts_per_ep_rank,
    int experts_per_moe_dp_group,
    int num_tp_ranks,
    int tp_rank,
    CUstream stream) {
  return launch_mask_indices_by_ep<int32_t>(
      indices,
      masked_indices,
      num_tokens,
      num_topk,
      experts_per_ep_rank,
      experts_per_moe_dp_group,
      num_tp_ranks,
      tp_rank,
      stream);
}

} // extern "C"
