#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <math_constants.h>
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

extern "C" __global__ void kernel_kernel(bfloat16_t* __restrict__ Output, const float* __restrict__ Partial_l, const float* __restrict__ Partial_m, const float* __restrict__ Partial_out, int num_splits, int num_splits_1, int num_splits_2, int num_splits_3, int total_q_tokens, int total_q_tokens_1, int total_q_tokens_2, int total_q_tokens_3, int total_q_tokens_4);
extern "C" __global__ void __launch_bounds__(128, 1) kernel_kernel(bfloat16_t* __restrict__ Output, const float* __restrict__ Partial_l, const float* __restrict__ Partial_m, const float* __restrict__ Partial_out, int num_splits, int num_splits_1, int num_splits_2, int num_splits_3, int total_q_tokens, int total_q_tokens_1, int total_q_tokens_2, int total_q_tokens_3, int total_q_tokens_4) {
  float final_o[1];
  float final_m[1];
  float final_l[1];
  final_o[0] = 0x0p+0f/*0.000000e+00*/;
  final_m[0] = -CUDART_INF_F;
  final_l[0] = 0x0p+0f/*0.000000e+00*/;
  for (int s = 0; s < num_splits; ++s) {
    float condval;
    if (((((int)blockIdx.x) < total_q_tokens_1) && (s < num_splits_1))) {
      condval = Partial_m[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + ((((int64_t)s) * ((int64_t)total_q_tokens_1)) * (int64_t)16)) + ((int64_t)((int)blockIdx.y)))];
    } else {
      condval = 0x0p+0f/*0.000000e+00*/;
    }
    float m_s = condval;
    float condval_1;
    if (((((int)blockIdx.x) < total_q_tokens_2) && (s < num_splits_2))) {
      condval_1 = Partial_l[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + ((((int64_t)s) * ((int64_t)total_q_tokens_2)) * (int64_t)16)) + ((int64_t)((int)blockIdx.y)))];
    } else {
      condval_1 = 0x0p+0f/*0.000000e+00*/;
    }
    float l_s = condval_1;
    float m_new = max(final_m[0], m_s);
    float s_prev = (final_l[0] * expf((final_m[0] - m_new)));
    float s_cur = (l_s * expf((m_s - m_new)));
    float l_new = (s_prev + (l_s * expf((m_s - m_new))));
    float condval_2;
    if (((((int)blockIdx.x) < total_q_tokens_3) && (s < num_splits_3))) {
      condval_2 = Partial_out[((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)s) * ((int64_t)total_q_tokens_3)) * (int64_t)2048)) + (((int64_t)((int)blockIdx.y)) * (int64_t)128)) + ((int64_t)((int)threadIdx.x)))];
    } else {
      condval_2 = 0x0p+0f/*0.000000e+00*/;
    }
    float o_s = condval_2;
    float condval_3;
    if ((0x0p+0f/*0.000000e+00*/ < (s_prev + (l_s * expf((m_s - m_new)))))) {
      condval_3 = (((final_o[0] * s_prev) + (o_s * (l_s * expf((m_s - m_new))))) / (s_prev + (l_s * expf((m_s - m_new)))));
    } else {
      condval_3 = 0x0p+0f/*0.000000e+00*/;
    }
    final_o[0] = condval_3;
    final_m[0] = m_new;
    final_l[0] = (s_prev + (l_s * expf((m_s - m_new))));
  }
  if (((int)blockIdx.x) < total_q_tokens_4) {
    Output[(((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + (((int64_t)((int)blockIdx.y)) * (int64_t)128)) + ((int64_t)((int)threadIdx.x)))] = ((bfloat16_t)final_o[0]);
  }
}

