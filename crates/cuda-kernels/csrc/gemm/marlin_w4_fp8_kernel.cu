/*
 * PF8.3 — ARLE W4 + FP8-activation Marlin GEMM wrapper.
 *
 * The Marlin template under gemm/marlin_pf8/ is adapted from vLLM's
 * csrc/quantization/marlin sources (Apache-2.0, copyright contributors to
 * the vLLM project and Marlin.2024 Elias Frantar). This wrapper intentionally
 * instantiates only the sm_89 prefill shape ARLE needs:
 *   A: FP8 e4m3 activations
 *   B: GPTQ INT4 U4B8 weights, zero-point preprocessed by PF8.2
 *   C: FP16 output scratch (Rust converts to BF16)
 *   Scale: FP16 per-group W4A16 Marlin scales
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#define MARLIN_NAMESPACE_NAME arle_marlin_pf8
#include "marlin_pf8/kernel.h"
#include "marlin_pf8/marlin_template.h"

namespace {

constexpr int kStages = 4;
constexpr int kDefaultThreads = 256;
constexpr int kMinThreadN = 64;
constexpr int kMaxThreadN = 256;
constexpr int kMaxThreadMBlocks = 4;
constexpr int kMaxParDefault = 16;

constexpr int ceildiv_int(int a, int b) {
  return (a + b - 1) / b;
}

using Pf8Kernel = void (*)(MARLIN_KERNEL_PARAMS);

template <int THREADS, int THREAD_M_BLOCKS, int THREAD_N_BLOCKS,
          int THREAD_K_BLOCKS, int GROUP_BLOCKS>
void launch_pf8_kernel(const int4* A,
                       const int4* B,
                       int4* C,
                       int4* C_tmp,
                       const float* a_scales,
                       const int4* b_scales,
                       int num_groups,
                       int prob_m,
                       int prob_n,
                       int prob_k,
                       int* locks,
                       int max_shared_mem,
                       cudaStream_t stream,
                       int blocks) {
  auto kernel =
      arle_marlin_pf8::Marlin<vllm::kFE4M3fn.id(), vllm::kU4B8.id(),
                              vllm::kFloat16.id(), vllm::kFloat16.id(),
                              THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS,
                              THREAD_K_BLOCKS, false, kStages, GROUP_BLOCKS,
                              false>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       max_shared_mem);
  kernel<<<blocks, THREADS, max_shared_mem, stream>>>(
      A, B, C, C_tmp,
      nullptr,       // bias
      a_scales,
      b_scales,
      nullptr,       // global_scale (only used by fp4 paths)
      nullptr,       // zero-points (U4B8 has no runtime zp tensor)
      nullptr,       // act-order g_idx
      num_groups,
      prob_m,
      prob_n,
      prob_k,
      prob_k,        // lda
      locks,
      false,         // has_bias
      false,         // use_atomic_add
      true,          // use_fp32_reduce
      max_shared_mem);
}

template <int THREAD_M_BLOCKS, int THREAD_N_BLOCKS, int THREAD_K_BLOCKS,
          int GROUP_BLOCKS>
bool maybe_launch_pf8_kernel(int thread_m_blocks,
                             int thread_n_blocks,
                             int thread_k_blocks,
                             int group_blocks,
                             const int4* A,
                             const int4* B,
                             int4* C,
                             int4* C_tmp,
                             const float* a_scales,
                             const int4* b_scales,
                             int num_groups,
                             int prob_m,
                             int prob_n,
                             int prob_k,
                             int* locks,
                             int max_shared_mem,
                             cudaStream_t stream,
                             int blocks) {
  if (thread_m_blocks != THREAD_M_BLOCKS ||
      thread_n_blocks != THREAD_N_BLOCKS ||
      thread_k_blocks != THREAD_K_BLOCKS ||
      group_blocks != GROUP_BLOCKS) {
    return false;
  }
  launch_pf8_kernel<kDefaultThreads, THREAD_M_BLOCKS, THREAD_N_BLOCKS,
                    THREAD_K_BLOCKS, GROUP_BLOCKS>(
      A, B, C, C_tmp, a_scales, b_scales, num_groups, prob_m, prob_n, prob_k,
      locks, max_shared_mem, stream, blocks);
  return true;
}

}  // namespace

constexpr int ERR_PROB_SHAPE = 1;
constexpr int ERR_KERN_SHAPE = 2;
constexpr int ERR_ARCH = 3;

extern "C" int gemm_w4_fp8_marlin_cuda(
    const void* A,       // [M,K] row-major FP8 e4m3 activations
    const void* B,       // zero-point preprocessed Marlin INT4 weights
    void* C_tmp,         // FP32 global-reduce buffer
    void* D,             // [M,N] FP16 output scratch
    const void* s1,      // [M] FP32 activation scales
    const void* s2,      // [K/group_size,N] FP16 W4 group scales
    int prob_m,
    int prob_n,
    int prob_k,
    void* workspace,     // lock buffer
    int groupsize,
    int dev,
    cudaStream_t stream,
    int thread_k,
    int thread_n,
    int sms,
    int max_par) {
  // H8 diagnostic per docs/plans/M_pf83_h8_fix_patch.md: log + clear any
  // pre-existing sticky CUDA error so this kernel doesn't get blamed for
  // an unrelated prior-call error surfaced by cudaGetLastError() at end.
  {
    cudaError_t prev_err = cudaGetLastError();
    if (prev_err != cudaSuccess) {
      fprintf(stderr, "[gemm_w4_fp8_marlin_cuda] cleared pre-existing CUDA error: %d (%s)\n",
              static_cast<int>(prev_err), cudaGetErrorString(prev_err));
    }
  }
  if (prob_m == 0 || prob_n == 0 || prob_k == 0) {
    return 0;
  }
  if (groupsize != 128) {
    return ERR_PROB_SHAPE;
  }
  if (prob_k % groupsize != 0) {
    return ERR_PROB_SHAPE;
  }

  int major = 0;
  int minor = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
  int sm = major * 10 + minor;
  if (sm != 89 && sm != 120) {
    return ERR_ARCH;
  }

  if (sms <= 0) {
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  }

  int max_shared_mem = 0;
  cudaDeviceGetAttribute(&max_shared_mem,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  if (max_shared_mem <= 0) {
    return ERR_PROB_SHAPE;
  }

  if (thread_k == -1 || thread_n == -1) {
    if (prob_m <= 16) {
      thread_k = 128;
      thread_n = 128;
    } else {
      thread_k = 64;
      thread_n = 256;
    }
  }
  if (thread_n < kMinThreadN || thread_n > kMaxThreadN) {
    return ERR_KERN_SHAPE;
  }

  int thread_k_blocks = thread_k / 16;
  int thread_n_blocks = thread_n / 16;
  int group_blocks = groupsize / 16;
  if (prob_n % thread_n != 0 || prob_k % thread_k != 0 ||
      prob_k % group_blocks != 0) {
    return ERR_PROB_SHAPE;
  }

  if (max_par <= 0) {
    max_par = kMaxParDefault;
  }

  const int4* A_ptr = reinterpret_cast<const int4*>(A);
  const int4* B_ptr = reinterpret_cast<const int4*>(B);
  int4* C_ptr = reinterpret_cast<int4*>(D);
  int4* C_tmp_ptr = reinterpret_cast<int4*>(C_tmp);
  const float* s1_ptr = reinterpret_cast<const float*>(s1);
  const int4* s2_ptr = reinterpret_cast<const int4*>(s2);
  int* locks = reinterpret_cast<int*>(workspace);
  int blocks = sms;
  int num_groups = prob_k / groupsize;

  int total_m = prob_m;
  int total_m_blocks = ceildiv_int(total_m, 16);
  int ret = 0;
  for (int block_m = 0; block_m < total_m_blocks; block_m += kMaxThreadMBlocks) {
    int thread_m_blocks = total_m_blocks - block_m;
    prob_m = total_m - 16 * block_m;
    int par = 1;
    if (thread_m_blocks > kMaxThreadMBlocks) {
      par = prob_m / (kMaxThreadMBlocks * 16);
      if (par > max_par) {
        par = max_par;
      }
      prob_m = par * kMaxThreadMBlocks * 16;
      block_m += kMaxThreadMBlocks * (par - 1);
      thread_m_blocks = kMaxThreadMBlocks;
    }

    bool launched =
        maybe_launch_pf8_kernel<1, 8, 8, 8>(
            thread_m_blocks, thread_n_blocks, thread_k_blocks, group_blocks,
            A_ptr, B_ptr, C_ptr, C_tmp_ptr, s1_ptr, s2_ptr, num_groups,
            prob_m, prob_n, prob_k, locks, max_shared_mem, stream, blocks) ||
        maybe_launch_pf8_kernel<2, 16, 4, 8>(
            thread_m_blocks, thread_n_blocks, thread_k_blocks, group_blocks,
            A_ptr, B_ptr, C_ptr, C_tmp_ptr, s1_ptr, s2_ptr, num_groups,
            prob_m, prob_n, prob_k, locks, max_shared_mem, stream, blocks) ||
        maybe_launch_pf8_kernel<3, 16, 4, 8>(
            thread_m_blocks, thread_n_blocks, thread_k_blocks, group_blocks,
            A_ptr, B_ptr, C_ptr, C_tmp_ptr, s1_ptr, s2_ptr, num_groups,
            prob_m, prob_n, prob_k, locks, max_shared_mem, stream, blocks) ||
        maybe_launch_pf8_kernel<4, 16, 4, 8>(
            thread_m_blocks, thread_n_blocks, thread_k_blocks, group_blocks,
            A_ptr, B_ptr, C_ptr, C_tmp_ptr, s1_ptr, s2_ptr, num_groups,
            prob_m, prob_n, prob_k, locks, max_shared_mem, stream, blocks);

    if (!launched) {
      ret = ERR_KERN_SHAPE;
      break;
    }

    A_ptr += prob_m * (prob_k / 16);
    C_ptr += prob_m * (prob_n / 8);
    s1_ptr += prob_m;
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }
  return ret;
}
