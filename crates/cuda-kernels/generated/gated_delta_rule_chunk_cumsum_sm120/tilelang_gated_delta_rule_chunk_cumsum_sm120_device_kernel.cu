#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void kernel_kernel(const float* __restrict__ g_in, float* __restrict__ g_out, int hv, int hv_1, int num_value_heads, int seq_len, int seq_len_1, int seq_len_2);
extern "C" __global__ void __launch_bounds__(128, 1) kernel_kernel(const float* __restrict__ g_in, float* __restrict__ g_out, int hv, int hv_1, int num_value_heads, int seq_len, int seq_len_1, int seq_len_2) {
  float src_buffer[1];
  extern __shared__ __align__(1024) float scan_smem[];
  float condval;
  if ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) & 63)) < seq_len) && (((int)blockIdx.y) < hv)) && (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) & 63)) < seq_len_1))) {
    condval = g_in[((((((int64_t)((int)blockIdx.x)) * (int64_t)64) + (((int64_t)((int)threadIdx.x)) & (int64_t)63)) * ((int64_t)hv)) + ((int64_t)((int)blockIdx.y)))];
  } else {
    condval = 0x0p+0f/*0.000000e+00*/;
  }
  src_buffer[0] = condval;
  if ((((int)threadIdx.x) >> 6) == 0) {
    scan_smem[(((int)threadIdx.x) & 63)] = src_buffer[0];
  }
  __syncthreads();
  tl::CumSum1D<128, false>::run((&(scan_smem[0])), (&(scan_smem[0])), 64);
  __syncthreads();
  src_buffer[0] = scan_smem[(((int)threadIdx.x) & 63)];
  if ((((int)threadIdx.x) >> 6) == 0) {
    if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) & 63)) < seq_len) {
      if (((int)blockIdx.y) < hv_1) {
        if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) & 63)) < seq_len_2) {
          g_out[((((((int64_t)((int)blockIdx.x)) * (int64_t)64) + (((int64_t)((int)threadIdx.x)) & (int64_t)63)) * ((int64_t)hv_1)) + ((int64_t)((int)blockIdx.y)))] = src_buffer[0];
        }
      }
    }
  }
}

