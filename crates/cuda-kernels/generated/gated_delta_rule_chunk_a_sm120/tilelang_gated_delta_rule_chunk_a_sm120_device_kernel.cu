#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/mma.h>
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

extern "C" __global__ void kernel_kernel(float* __restrict__ a_tril, const float* __restrict__ beta, const float* __restrict__ g_cumsum, const bfloat16_t* __restrict__ k, int hv, int hv_1, int hv_2, int hv_3, int num_value_heads, int seq_len, int seq_len_1, int seq_len_2, int seq_len_3, int seq_len_4);
extern "C" __global__ void __launch_bounds__(128, 1) kernel_kernel(float* __restrict__ a_tril, const float* __restrict__ beta, const float* __restrict__ g_cumsum, const bfloat16_t* __restrict__ k, int hv, int hv_1, int hv_2, int hv_3, int num_value_heads, int seq_len, int seq_len_1, int seq_len_2, int seq_len_3, int seq_len_4) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* beta_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* k_tile = ((void*)((char*)buf_dyn_shmem + 0));
  void* g_shared = ((void*)((char*)buf_dyn_shmem + 256));
  float acc[32];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    bfloat16_t broadcast_var = bfloat16_t(0x0p+0f/*0.000000e+00*/);
    uint4 condval;
    if (((((((((int)blockIdx.x) * 64) + (i * 8)) + (((int)threadIdx.x) >> 4)) < seq_len) && (((int)blockIdx.y) < hv)) && ((((((int)blockIdx.x) * 64) + (i * 8)) + (((int)threadIdx.x) >> 4)) < seq_len_1))) {
      condval = *(uint4*)(k + (((((int64_t)((int)blockIdx.y)) * (int64_t)128) + (((((((int64_t)((int)blockIdx.x)) * (int64_t)64) + (((int64_t)i) * (int64_t)8)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4)) * ((int64_t)hv)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
    } else {
      condval = make_uint4(__pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var));
    }
    *(uint4*)(((bfloat16_t*)k_tile) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval;
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc + (i_1 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
  }
  {
    bfloat16_t A_local[8];
    bfloat16_t B_local[32];
    __syncthreads();
    for (int ki = 0; ki < 8; ++ki) {
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)k_tile)[(((((ki >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local[0])));
      for (int i_2 = 0; i_2 < 4; ++i_2) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)k_tile)[((((((((ki >> 2) * 4096) + (i_2 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[(i_2 * 8)])));
      }
      for (int j = 0; j < 4; ++j) {
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc + (j * 8)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
      }
    }
  }
  __syncthreads();
  if (((int)threadIdx.x) < 64) {
    float condval_1;
    if ((((((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) < seq_len) && (((int)blockIdx.y) < hv_1)) && (((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) < seq_len_2))) {
      condval_1 = g_cumsum[((((((int64_t)((int)blockIdx.x)) * (int64_t)64) + ((int64_t)((int)threadIdx.x))) * ((int64_t)hv_1)) + ((int64_t)((int)blockIdx.y)))];
    } else {
      condval_1 = 0x0p+0f/*0.000000e+00*/;
    }
    ((float*)g_shared)[((int)threadIdx.x)] = condval_1;
    float condval_2;
    if ((((((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) < seq_len) && (((int)blockIdx.y) < hv_2)) && (((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) < seq_len_3))) {
      condval_2 = beta[((((((int64_t)((int)blockIdx.x)) * (int64_t)64) + ((int64_t)((int)threadIdx.x))) * ((int64_t)hv_2)) + ((int64_t)((int)blockIdx.y)))];
    } else {
      condval_2 = 0x0p+0f/*0.000000e+00*/;
    }
    ((float*)beta_shared)[((int)threadIdx.x)] = condval_2;
  }
  __syncthreads();
  #pragma unroll
  for (int i_3 = 0; i_3 < 16; ++i_3) {
    for (int vec_s = 0; vec_s < 2; ++vec_s) {
      bool row_in = (((((((int)blockIdx.x) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_3 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < seq_len);
      bool col_in = (((((((int)blockIdx.x) * 64) + ((i_3 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + vec_s) < seq_len);
      bool masked = (((((((i_3 >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)) + vec_s) < ((((((int)threadIdx.x) >> 5) * 16) + ((i_3 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))) && (((((((int)blockIdx.x) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_3 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < seq_len)) && (((((((int)blockIdx.x) * 64) + ((i_3 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + vec_s) < seq_len));
      float condval_3;
      if ((((((((i_3 >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)) + vec_s) < ((((((int)threadIdx.x) >> 5) * 16) + ((i_3 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))) && (((((((int)blockIdx.x) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_3 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < seq_len)) && (((((((int)blockIdx.x) * 64) + ((i_3 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + vec_s) < seq_len))) {
        condval_3 = ((acc[((i_3 * 2) + vec_s)] * ((float*)beta_shared)[((((((int)threadIdx.x) >> 5) * 16) + ((i_3 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]) * expf((((float*)g_shared)[((((((int)threadIdx.x) >> 5) * 16) + ((i_3 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))] - ((float*)g_shared)[((((i_3 >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)) + vec_s)])));
      } else {
        condval_3 = 0x0p+0f/*0.000000e+00*/;
      }
      acc[((i_3 * 2) + vec_s)] = condval_3;
    }
  }
  if (((int)blockIdx.y) < hv_3) {
    #pragma unroll
    for (int i_4 = 0; i_4 < 16; ++i_4) {
      if (((((((int)blockIdx.x) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_4 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < seq_len) {
        if (((((((int)blockIdx.x) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_4 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < seq_len_4) {
          *(float2*)(a_tril + ((((((int64_t)((int)blockIdx.y)) * (int64_t)64) + ((((((((int64_t)((int)blockIdx.x)) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)16)) + ((((int64_t)i_4) & (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2)) * ((int64_t)hv_3)) * (int64_t)64)) + ((((int64_t)i_4) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2))) = *(float2*)(acc + (i_4 * 2));
        }
      }
    }
  }
}

