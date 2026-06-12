// Vendored from SGLang (Apache-2.0):
//   sgl-kernel/csrc/moe/moe_align_kernel.cu
//   `count_and_sort_expert_tokens_kernel` + `moe_align_block_size_kernel`
//   (the CUDA `#ifdef __CUDA_ARCH__` warp-scan variant) + the FULL launch path
//   of `moe_align_block_size`.
//
// Produces, from token-major `topk_ids [M*top_k]` (i32 expert ids in [0, E)):
//   * sorted_token_ids [max_num_tokens_padded] — token row ids grouped by
//     expert, each group padded up to a multiple of block_size; pad slots hold
//     `numel` (= M*top_k) so the GEMM masks them via `< num_valid_tokens`.
//   * expert_ids [num_blocks] — the expert id for each block_size-row block.
//   * total_tokens_post_pad [1] — Σ_e ceil(count_e / block_size) * block_size.
//   * cumsum_buffer [E+1] — scratch the kernel needs (padded exclusive prefix
//     sums over expert counts); caller preallocates it.
//
// DEVIATIONS (every intentional change vs. the upstream .cu file):
//   1. The torch::Tensor `moe_align_block_size` wrapper is replaced by a raw
//      `extern "C"` C shim taking device pointers + sizes + a stream. The 256-
//      expert Qwen3.6 config always hits the FULL path (small_batch_expert_mode
//      requires num_experts <= 64), so the small-batch kernel is NOT vendored.
//   2. The template `scalar_t` is fixed to `int32_t` (our topk_ids dtype); the
//      `DISPATCH_INTEGRAL_TYPES` macro is dropped.
//   3. `WARP_SIZE` (from the upstream `utils.h`) is defined locally as 32.
//   4. `pad_sorted_token_ids` is fixed to `true` (the GEMM relies on the pad
//      slots reading `numel`); the arg + the small-batch path are removed.
// The kernel device code (warp-scan prefix sum, expert-id binary search, the
// sort scatter) is byte-equal to upstream.

#include <cuda_runtime.h>
#include <stdint.h>

#define MOE_ALIGN_WARP_SIZE 32
#define MOE_ALIGN_VEC_SIZE 4
using MoeAlignVec = int4;

__global__ void arle_moe_count_and_sort_expert_tokens_kernel(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ cumsum_buffer,
    size_t numel) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;

  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = topk_ids[i] + 1;
    int32_t rank_post_pad = atomicAdd(&cumsum_buffer[expert_id], 1);
    sorted_token_ids[rank_post_pad] = i;
  }
}

__device__ __forceinline__ int arle_moe_warp_exclusive_scan(int v, unsigned mask = 0xffffffffu) {
  int original = v;
#pragma unroll
  for (int offset = 1; offset < MOE_ALIGN_WARP_SIZE; offset <<= 1) {
    int n = __shfl_up_sync(mask, v, offset);
    if ((threadIdx.x & (MOE_ALIGN_WARP_SIZE - 1)) >= offset) v += n;
  }
  return v - original;
}

__global__ void arle_moe_align_block_size_kernel(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t num_experts,
    int32_t block_size,
    size_t numel,
    int32_t* __restrict__ cumsum,
    const int32_t scan_size,
    int32_t max_num_tokens_padded) {
  // Use a separate thread block to populate sorted_token_ids.
  if (blockIdx.x == 1) {
    MoeAlignVec fill_vec;
    fill_vec.x = fill_vec.y = fill_vec.z = fill_vec.w = numel;
    int32_t total_vecs = (max_num_tokens_padded + MOE_ALIGN_VEC_SIZE - 1) / MOE_ALIGN_VEC_SIZE;
    MoeAlignVec* out_ptr = reinterpret_cast<MoeAlignVec*>(sorted_token_ids);
    for (int32_t i = threadIdx.x; i < total_vecs; i += blockDim.x) {
      out_ptr[i] = fill_vec;
    }
    return;
  }

  extern __shared__ int32_t smem[];
  int32_t* shared_counts = smem;                  // [num_experts]
  int32_t* prefix = shared_counts + num_experts;  // [num_experts + 1]
  int32_t* scan_buf = prefix + num_experts + 1;   // [scan_size]
  __shared__ int32_t s_total_tokens_post_pad;

  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;

  if (tid < num_experts) {
    shared_counts[tid] = 0;
  }

  __syncthreads();

  for (size_t i = tid; i < numel; i += stride) {
    int expert_id = topk_ids[i] + 1;
    atomicAdd(&shared_counts[expert_id], 1);
  }

  __syncthreads();

  int32_t padded_count = 0;
  if (tid < num_experts) {
    int32_t count = shared_counts[tid];
    padded_count = (count + block_size - 1) / block_size * block_size;
    scan_buf[tid] = padded_count;
  }

  // Intra warp prefix sum
  int32_t* warp_sums = scan_buf + scan_size;  // [<= 32]
  const int warp_id = tid / MOE_ALIGN_WARP_SIZE;
  const int lane_id = tid & (MOE_ALIGN_WARP_SIZE - 1);
  const int num_warps_for_scan = (scan_size + MOE_ALIGN_WARP_SIZE - 1) / MOE_ALIGN_WARP_SIZE;
  const int warp_sum = arle_moe_warp_exclusive_scan(padded_count) + padded_count;
  if (lane_id == MOE_ALIGN_WARP_SIZE - 1) warp_sums[warp_id] = warp_sum;
  __syncthreads();

  // warp0 accumulate all the block's prefix sum
  if (tid < MOE_ALIGN_WARP_SIZE) {
    int val = (tid < num_warps_for_scan) ? warp_sums[tid] : 0;
    int incl = arle_moe_warp_exclusive_scan(val) + val;
    warp_sums[tid] = incl;
  }
  __syncthreads();

  // Every thread obtains the whole block's sum
  if (tid == 0) {
    prefix[num_experts] = warp_sums[num_warps_for_scan - 1];
    s_total_tokens_post_pad = prefix[num_experts];
    *total_tokens_post_pad = s_total_tokens_post_pad;
  }
  __syncthreads();

  // Fill 0 to scan_buf extended area (tid >= num_expert)
  if (tid >= num_experts && tid < scan_size) scan_buf[tid] = 0;
  __syncthreads();

  // Perform 2 level exclusive-prefix-sum to scan_buf
  int v = (tid < scan_size) ? scan_buf[tid] : 0;
  int pre = arle_moe_warp_exclusive_scan(v);
  if (lane_id == MOE_ALIGN_WARP_SIZE - 1) warp_sums[warp_id] = pre + v;
  __syncthreads();

  if (warp_id == 0) {
    int val = (lane_id < num_warps_for_scan) ? warp_sums[lane_id] : 0;
    warp_sums[lane_id] = arle_moe_warp_exclusive_scan(val);
  }
  __syncthreads();

  int offset = warp_sums[warp_id];
  if (tid < scan_size) scan_buf[tid] = pre + offset;
  __syncthreads();

  // Write prefix[0..num_experts - 1] and cumsum
  if (tid < num_experts) prefix[tid] = scan_buf[tid];

  if (tid <= num_experts) {
    cumsum[tid] = prefix[tid];
  }
  // fill expert_ids
  const int32_t num_blocks = s_total_tokens_post_pad / block_size;
  for (int32_t i = tid; i < num_blocks; i += stride) {
    int32_t block_start = i * block_size;
    int left = 0, right = num_experts;
    while (left < right) {
      int mid = (left + right) >> 1;
      if (prefix[mid] <= block_start) {
        left = mid + 1;
      } else {
        right = mid;
      }
    }
    expert_ids[i] = left - 2;
  }
}

static int32_t arle_moe_next_pow2(int32_t v) {
  int32_t p = 1;
  while (p < v) p <<= 1;
  return p;
}

// FULL-path moe_align launch. `cumsum_buffer` must be a preallocated device
// buffer of [num_experts + 1] i32 (the count-scan scratch). Both `cumsum_buffer`
// and the `count_and_sort` accumulator MUST be zeroed by the caller before each
// call (atomicAdd accumulators); the count_and_sort kernel reuses the same
// cumsum_buffer the align kernel wrote (the exclusive-prefix base per expert).
extern "C" cudaError_t arle_moe_align_block_size_cuda(
    const int32_t* topk_ids,
    int32_t* sorted_token_ids,
    int32_t* expert_ids,
    int32_t* total_tokens_post_pad,
    int32_t* cumsum_buffer,
    int32_t num_experts,
    int32_t block_size,
    int32_t numel,
    int32_t max_num_tokens_padded,
    cudaStream_t stream) {
  if (num_experts <= 0 || block_size <= 0 || numel < 0 || max_num_tokens_padded < 0) {
    return cudaErrorInvalidValue;
  }

  int threads = 1024;
  threads = ((threads + MOE_ALIGN_WARP_SIZE - 1) / MOE_ALIGN_WARP_SIZE) * MOE_ALIGN_WARP_SIZE;

  const int32_t scan_size = arle_moe_next_pow2(num_experts);
  const size_t shared_mem_size =
      (num_experts + (num_experts + 1) + scan_size + MOE_ALIGN_WARP_SIZE) * sizeof(int32_t);

  arle_moe_align_block_size_kernel<<<2, threads, shared_mem_size, stream>>>(
      topk_ids,
      sorted_token_ids,
      expert_ids,
      total_tokens_post_pad,
      num_experts,
      block_size,
      (size_t)numel,
      cumsum_buffer,
      scan_size,
      max_num_tokens_padded);

  const int block_threads = (256 < threads) ? 256 : threads;
  const int n_blocks = (numel + block_threads - 1) / block_threads;
  const int max_blocks = 65535;
  const int actual_blocks = (n_blocks < max_blocks) ? n_blocks : max_blocks;
  if (actual_blocks > 0) {
    arle_moe_count_and_sort_expert_tokens_kernel<<<actual_blocks, block_threads, 0, stream>>>(
        topk_ids,
        sorted_token_ids,
        cumsum_buffer,
        (size_t)numel);
  }

  return cudaGetLastError();
}
