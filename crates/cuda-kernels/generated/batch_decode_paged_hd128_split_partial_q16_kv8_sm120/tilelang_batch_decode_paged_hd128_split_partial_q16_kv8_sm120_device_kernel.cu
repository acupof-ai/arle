#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/mma.h>
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

extern "C" __global__ void kernel_kernel(const int* __restrict__ KV_indices, const int* __restrict__ KV_indptr, const int* __restrict__ KV_last_page_len, const bfloat16_t* __restrict__ K_pool, float* __restrict__ Partial_l, float* __restrict__ Partial_m, float* __restrict__ Partial_out, const bfloat16_t* __restrict__ Q, const bfloat16_t* __restrict__ V_pool, int batch_size, int batch_size_1, int batch_size_plus_one, int num_pages, int num_pages_1, int num_splits, int num_splits_1, int num_splits_2, int num_splits_3, int total_pages, int total_q_tokens, int total_q_tokens_1, int total_q_tokens_2, int total_q_tokens_3);
extern "C" __global__ void __launch_bounds__(128, 1) kernel_kernel(const int* __restrict__ KV_indices, const int* __restrict__ KV_indptr, const int* __restrict__ KV_last_page_len, const bfloat16_t* __restrict__ K_pool, float* __restrict__ Partial_l, float* __restrict__ Partial_m, float* __restrict__ Partial_out, const bfloat16_t* __restrict__ Q, const bfloat16_t* __restrict__ V_pool, int batch_size, int batch_size_1, int batch_size_plus_one, int num_pages, int num_pages_1, int num_splits, int num_splits_1, int num_splits_2, int num_splits_3, int total_pages, int total_q_tokens, int total_q_tokens_1, int total_q_tokens_2, int total_q_tokens_3) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* q_tile = ((void*)((char*)buf_dyn_shmem + 0));
  void* k_tile = ((void*)((char*)buf_dyn_shmem + 16384));
  void* v_tile = ((void*)((char*)buf_dyn_shmem + 20480));
  float acc_o[64];
  float m_i[2];
  float l_i[2];
  float scores[8];
  float m_prev[2];
  float m_new[2];
  float p[8];
  float scale_i[2];
  float row_sum[2];
  bfloat16_t p_bf16[8];
  const dim3 blockIdx = tl::rasterization2DRow<8>();
  int condval;
  if ((((int)blockIdx.x) < batch_size_plus_one)) {
    condval = KV_indptr[((int)blockIdx.x)];
  } else {
    condval = 0;
  }
  int kv_page_start = condval;
  int condval_1;
  if (((((int)blockIdx.x) + 1) < batch_size_plus_one)) {
    condval_1 = KV_indptr[(((int64_t)((int)blockIdx.x)) + (int64_t)1)];
  } else {
    condval_1 = 0;
  }
  int kv_page_end = condval_1;
  int condval_2;
  if ((((int)blockIdx.x) < batch_size_1)) {
    condval_2 = KV_last_page_len[((int)blockIdx.x)];
  } else {
    condval_2 = 0;
  }
  int last_page_len = condval_2;
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_o + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  float broadcast_var_1 = -CUDART_INF_F;
  *(float2*)(m_i + 0) = make_float2(broadcast_var_1, broadcast_var_1);
  float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
  *(float2*)(l_i + 0) = make_float2(broadcast_var_2, broadcast_var_2);
  #pragma unroll
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    bfloat16_t broadcast_var_3 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
    uint4 condval_3;
    if (((((i_1 * 8) + (((int)threadIdx.x) >> 4)) == 0) && (((int)blockIdx.x) < total_q_tokens_3))) {
      condval_3 = *(uint4*)(Q + (((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + (((int64_t)((int)blockIdx.y)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
    } else {
      condval_3 = make_uint4(__pack_nv_bfloat162(broadcast_var_3, broadcast_var_3), __pack_nv_bfloat162(broadcast_var_3, broadcast_var_3), __pack_nv_bfloat162(broadcast_var_3, broadcast_var_3), __pack_nv_bfloat162(broadcast_var_3, broadcast_var_3));
    }
    *(uint4*)(((bfloat16_t*)q_tile) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_1 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_3;
  }
  int rmod = ((((((kv_page_end * 16) + last_page_len) + num_splits) - (kv_page_start * 16)) - 17) % num_splits);
  int rdiv = ((((((kv_page_end * 16) + last_page_len) + num_splits) - (kv_page_start * 16)) - 17) / num_splits);
  int rmod_1 = ((((((kv_page_end * 16) + last_page_len) + num_splits) - (kv_page_start * 16)) - 17) % num_splits);
  int rdiv_1 = ((((((kv_page_end * 16) + last_page_len) + num_splits) - (kv_page_start * 16)) - 17) / num_splits);
  for (int kn = 0; kn < ((max((min(((((int)blockIdx.z) + 1) * ((((0 <= num_splits) && (0 <= rmod)) || ((num_splits < 0) && (rmod <= 0))) ? rdiv : (rdiv - 1))), ((((kv_page_end * 16) + last_page_len) - (kv_page_start * 16)) - 16)) - (((int)blockIdx.z) * ((((0 <= num_splits) && (0 <= rmod_1)) || ((num_splits < 0) && (rmod_1 <= 0))) ? rdiv_1 : (rdiv_1 - 1)))), 0) + 15) >> 4); ++kn) {
    __syncthreads();
    #pragma unroll
    for (int i_2 = 0; i_2 < 2; ++i_2) {
      int rmod_2 = ((((((kv_page_end * 16) + last_page_len) + num_splits) - (kv_page_start * 16)) - 17) % num_splits);
      int rdiv_2 = ((((((kv_page_end * 16) + last_page_len) + num_splits) - (kv_page_start * 16)) - 17) / num_splits);
      int rmod_3 = ((((((kv_page_end * 16) + last_page_len) + num_splits) - (kv_page_start * 16)) - 17) % num_splits);
      int rdiv_3 = ((((((kv_page_end * 16) + last_page_len) + num_splits) - (kv_page_start * 16)) - 17) / num_splits);
      int rmod_4 = ((((((kv_page_end * 16) + last_page_len) + num_splits) - (kv_page_start * 16)) - 17) % num_splits);
      int rdiv_4 = ((((((kv_page_end * 16) + last_page_len) + num_splits) - (kv_page_start * 16)) - 17) / num_splits);
      int rmod_5 = ((((((kv_page_end * 16) + last_page_len) + num_splits) - (kv_page_start * 16)) - 17) % num_splits);
      int rdiv_5 = ((((((kv_page_end * 16) + last_page_len) + num_splits) - (kv_page_start * 16)) - 17) / num_splits);
      int rmod_6 = ((((((kv_page_end * 16) + last_page_len) + num_splits) - (kv_page_start * 16)) - 17) % num_splits);
      int rdiv_6 = ((((((kv_page_end * 16) + last_page_len) + num_splits) - (kv_page_start * 16)) - 17) / num_splits);
      int condval_4;
      if (((((((((kn * 16) + (i_2 * 8)) + (((int)threadIdx.x) >> 4)) + (((int)blockIdx.z) * ((((0 <= num_splits) && (0 <= rmod_2)) || ((num_splits < 0) && (rmod_2 <= 0))) ? rdiv_2 : (rdiv_2 - 1)))) < ((((int)blockIdx.z) + 1) * ((((0 <= num_splits) && (0 <= rmod_3)) || ((num_splits < 0) && (rmod_3 <= 0))) ? rdiv_3 : (rdiv_3 - 1)))) && ((((((kn * 16) + (i_2 * 8)) + (((int)threadIdx.x) >> 4)) + (((int)blockIdx.z) * ((((0 <= num_splits) && (0 <= rmod_4)) || ((num_splits < 0) && (rmod_4 <= 0))) ? rdiv_4 : (rdiv_4 - 1)))) + 16) < (((kv_page_end * 16) + last_page_len) - (kv_page_start * 16)))) && (0 <= (((((((((int)threadIdx.x) >> 4) + (((int)blockIdx.z) * ((((0 <= num_splits) && (0 <= rmod_5)) || ((num_splits < 0) && (rmod_5 <= 0))) ? rdiv_5 : (rdiv_5 - 1)))) >> 3) + i_2) >> 1) + kv_page_start) + kn))) && ((((((((((int)threadIdx.x) >> 4) + (((int)blockIdx.z) * ((((0 <= num_splits) && (0 <= rmod_6)) || ((num_splits < 0) && (rmod_6 <= 0))) ? rdiv_6 : (rdiv_6 - 1)))) >> 3) + i_2) >> 1) + kv_page_start) + kn) < total_pages))) {
        int64_t rmod_7 = ((((((((int64_t)kv_page_end) * (int64_t)16) + ((int64_t)last_page_len)) + ((int64_t)num_splits)) - (((int64_t)kv_page_start) * (int64_t)16)) - (int64_t)17) % ((int64_t)num_splits));
        int64_t rdiv_7 = ((((((((int64_t)kv_page_end) * (int64_t)16) + ((int64_t)last_page_len)) + ((int64_t)num_splits)) - (((int64_t)kv_page_start) * (int64_t)16)) - (int64_t)17) / ((int64_t)num_splits));
        int64_t rmod_8 = ((((((((int64_t)kv_page_end) * (int64_t)16) + ((int64_t)last_page_len)) + ((int64_t)num_splits)) - (((int64_t)kv_page_start) * (int64_t)16)) - (int64_t)17) % ((int64_t)num_splits));
        int64_t rdiv_8 = ((((((((int64_t)kv_page_end) * (int64_t)16) + ((int64_t)last_page_len)) + ((int64_t)num_splits)) - (((int64_t)kv_page_start) * (int64_t)16)) - (int64_t)17) / ((int64_t)num_splits));
        condval_4 = KV_indices[(((((((((int64_t)((int)threadIdx.x)) >> (int64_t)4) + (((int64_t)((int)blockIdx.z)) * (((((int64_t)0 <= ((int64_t)num_splits)) && ((int64_t)0 <= rmod_8)) || ((((int64_t)num_splits) < (int64_t)0) && (rmod_8 <= (int64_t)0))) ? rdiv_8 : (rdiv_8 - (int64_t)1)))) >> (int64_t)3) + ((int64_t)i_2)) >> (int64_t)1) + ((int64_t)kv_page_start)) + ((int64_t)kn))];
      } else {
        condval_4 = 0;
      }
      int page_idx = condval_4;
      bfloat16_t broadcast_var_4 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
      int rmod_9 = ((((((kv_page_end * 16) + last_page_len) + num_splits) - (kv_page_start * 16)) - 17) % num_splits);
      int rdiv_9 = ((((((kv_page_end * 16) + last_page_len) + num_splits) - (kv_page_start * 16)) - 17) / num_splits);
      int rmod_10 = ((((((kv_page_end * 16) + last_page_len) + num_splits) - (kv_page_start * 16)) - 17) % num_splits);
      int rdiv_10 = ((((((kv_page_end * 16) + last_page_len) + num_splits) - (kv_page_start * 16)) - 17) / num_splits);
      int rmod_11 = ((((((kv_page_end * 16) + last_page_len) + num_splits) - (kv_page_start * 16)) - 17) % num_splits);
      int rdiv_11 = ((((((kv_page_end * 16) + last_page_len) + num_splits) - (kv_page_start * 16)) - 17) / num_splits);
      uint4 condval_5;
      if (((((((((kn * 16) + (i_2 * 8)) + (((int)threadIdx.x) >> 4)) + (((int)blockIdx.z) * ((((0 <= num_splits) && (0 <= rmod_9)) || ((num_splits < 0) && (rmod_9 <= 0))) ? rdiv_9 : (rdiv_9 - 1)))) < ((((int)blockIdx.z) + 1) * ((((0 <= num_splits) && (0 <= rmod_10)) || ((num_splits < 0) && (rmod_10 <= 0))) ? rdiv_10 : (rdiv_10 - 1)))) && ((((((kn * 16) + (i_2 * 8)) + (((int)threadIdx.x) >> 4)) + (((int)blockIdx.z) * ((((0 <= num_splits) && (0 <= rmod_11)) || ((num_splits < 0) && (rmod_11 <= 0))) ? rdiv_11 : (rdiv_11 - 1)))) + 16) < (((kv_page_end * 16) + last_page_len) - (kv_page_start * 16)))) && (0 <= page_idx)) && (page_idx < num_pages))) {
        int64_t rmod_12 = ((((((((int64_t)kv_page_end) * (int64_t)16) + ((int64_t)last_page_len)) + ((int64_t)num_splits)) - (((int64_t)kv_page_start) * (int64_t)16)) - (int64_t)17) % ((int64_t)num_splits));
        int64_t rdiv_12 = ((((((((int64_t)kv_page_end) * (int64_t)16) + ((int64_t)last_page_len)) + ((int64_t)num_splits)) - (((int64_t)kv_page_start) * (int64_t)16)) - (int64_t)17) / ((int64_t)num_splits));
        int64_t rmod_13 = ((((((((int64_t)kv_page_end) * (int64_t)16) + ((int64_t)last_page_len)) + ((int64_t)num_splits)) - (((int64_t)kv_page_start) * (int64_t)16)) - (int64_t)17) % ((int64_t)num_splits));
        int64_t rdiv_13 = ((((((((int64_t)kv_page_end) * (int64_t)16) + ((int64_t)last_page_len)) + ((int64_t)num_splits)) - (((int64_t)kv_page_start) * (int64_t)16)) - (int64_t)17) / ((int64_t)num_splits));
        condval_5 = *(uint4*)(K_pool + ((((((int64_t)page_idx) * (int64_t)16384) + ((((int64_t)((int)blockIdx.y)) >> (int64_t)1) * (int64_t)2048)) + (((((((int64_t)i_2) * (int64_t)8) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4)) + (((int64_t)((int)blockIdx.z)) * (((((int64_t)0 <= ((int64_t)num_splits)) && ((int64_t)0 <= rmod_13)) || ((((int64_t)num_splits) < (int64_t)0) && (rmod_13 <= (int64_t)0))) ? rdiv_13 : (rdiv_13 - (int64_t)1)))) & (int64_t)15) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
      } else {
        condval_5 = make_uint4(__pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4));
      }
      *(uint4*)(((bfloat16_t*)k_tile) + ((((((((((int)threadIdx.x) & 15) >> 3) * 1024) + (i_2 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_5;
      bfloat16_t broadcast_var_5 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
      int rmod_14 = ((((((kv_page_end * 16) + last_page_len) + num_splits) - (kv_page_start * 16)) - 17) % num_splits);
      int rdiv_14 = ((((((kv_page_end * 16) + last_page_len) + num_splits) - (kv_page_start * 16)) - 17) / num_splits);
      int rmod_15 = ((((((kv_page_end * 16) + last_page_len) + num_splits) - (kv_page_start * 16)) - 17) % num_splits);
      int rdiv_15 = ((((((kv_page_end * 16) + last_page_len) + num_splits) - (kv_page_start * 16)) - 17) / num_splits);
      int rmod_16 = ((((((kv_page_end * 16) + last_page_len) + num_splits) - (kv_page_start * 16)) - 17) % num_splits);
      int rdiv_16 = ((((((kv_page_end * 16) + last_page_len) + num_splits) - (kv_page_start * 16)) - 17) / num_splits);
      uint4 condval_6;
      if (((((((((kn * 16) + (i_2 * 8)) + (((int)threadIdx.x) >> 4)) + (((int)blockIdx.z) * ((((0 <= num_splits) && (0 <= rmod_14)) || ((num_splits < 0) && (rmod_14 <= 0))) ? rdiv_14 : (rdiv_14 - 1)))) < ((((int)blockIdx.z) + 1) * ((((0 <= num_splits) && (0 <= rmod_15)) || ((num_splits < 0) && (rmod_15 <= 0))) ? rdiv_15 : (rdiv_15 - 1)))) && ((((((kn * 16) + (i_2 * 8)) + (((int)threadIdx.x) >> 4)) + (((int)blockIdx.z) * ((((0 <= num_splits) && (0 <= rmod_16)) || ((num_splits < 0) && (rmod_16 <= 0))) ? rdiv_16 : (rdiv_16 - 1)))) + 16) < (((kv_page_end * 16) + last_page_len) - (kv_page_start * 16)))) && (0 <= page_idx)) && (page_idx < num_pages_1))) {
        int64_t rmod_17 = ((((((((int64_t)kv_page_end) * (int64_t)16) + ((int64_t)last_page_len)) + ((int64_t)num_splits)) - (((int64_t)kv_page_start) * (int64_t)16)) - (int64_t)17) % ((int64_t)num_splits));
        int64_t rdiv_17 = ((((((((int64_t)kv_page_end) * (int64_t)16) + ((int64_t)last_page_len)) + ((int64_t)num_splits)) - (((int64_t)kv_page_start) * (int64_t)16)) - (int64_t)17) / ((int64_t)num_splits));
        int64_t rmod_18 = ((((((((int64_t)kv_page_end) * (int64_t)16) + ((int64_t)last_page_len)) + ((int64_t)num_splits)) - (((int64_t)kv_page_start) * (int64_t)16)) - (int64_t)17) % ((int64_t)num_splits));
        int64_t rdiv_18 = ((((((((int64_t)kv_page_end) * (int64_t)16) + ((int64_t)last_page_len)) + ((int64_t)num_splits)) - (((int64_t)kv_page_start) * (int64_t)16)) - (int64_t)17) / ((int64_t)num_splits));
        condval_6 = *(uint4*)(V_pool + ((((((int64_t)page_idx) * (int64_t)16384) + ((((int64_t)((int)blockIdx.y)) >> (int64_t)1) * (int64_t)2048)) + (((((((int64_t)i_2) * (int64_t)8) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4)) + (((int64_t)((int)blockIdx.z)) * (((((int64_t)0 <= ((int64_t)num_splits)) && ((int64_t)0 <= rmod_18)) || ((((int64_t)num_splits) < (int64_t)0) && (rmod_18 <= (int64_t)0))) ? rdiv_18 : (rdiv_18 - (int64_t)1)))) & (int64_t)15) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
      } else {
        condval_6 = make_uint4(__pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5));
      }
      *(uint4*)(((bfloat16_t*)v_tile) + ((((((((((int)threadIdx.x) & 15) >> 3) * 1024) + (i_2 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_6;
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 2; ++i_3) {
      float broadcast_var_6 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(scores + (i_3 * 4)) = make_float4(broadcast_var_6, broadcast_var_6, broadcast_var_6, broadcast_var_6);
    }
    {
      bfloat16_t A_local[8];
      bfloat16_t B_local[8];
      __syncthreads();
      for (int ki = 0; ki < 8; ++ki) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)q_tile)[(((((ki >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local[0])));
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)k_tile)[(((((((ki >> 2) * 1024) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[0])));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores + 0), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 0));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores + 4), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 4));
      }
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 8; ++i_4) {
      int rmod_19 = ((((((kv_page_end * 16) + last_page_len) + num_splits) - (kv_page_start * 16)) - 17) % num_splits);
      int rdiv_19 = ((((((kv_page_end * 16) + last_page_len) + num_splits) - (kv_page_start * 16)) - 17) / num_splits);
      int rmod_20 = ((((((kv_page_end * 16) + last_page_len) + num_splits) - (kv_page_start * 16)) - 17) % num_splits);
      int rdiv_20 = ((((((kv_page_end * 16) + last_page_len) + num_splits) - (kv_page_start * 16)) - 17) / num_splits);
      int rmod_21 = ((((((kv_page_end * 16) + last_page_len) + num_splits) - (kv_page_start * 16)) - 17) % num_splits);
      int rdiv_21 = ((((((kv_page_end * 16) + last_page_len) + num_splits) - (kv_page_start * 16)) - 17) / num_splits);
      bool in_bounds = (((((((((int)threadIdx.x) >> 5) * 16) + (((i_4 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) == 0) && ((((((kn * 16) + ((i_4 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (((int)blockIdx.z) * ((((0 <= num_splits) && (0 <= rmod_19)) || ((num_splits < 0) && (rmod_19 <= 0))) ? rdiv_19 : (rdiv_19 - 1)))) + (i_4 & 1)) < ((((int)blockIdx.z) + 1) * ((((0 <= num_splits) && (0 <= rmod_20)) || ((num_splits < 0) && (rmod_20 <= 0))) ? rdiv_20 : (rdiv_20 - 1))))) && (((((((kn * 16) + ((i_4 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (((int)blockIdx.z) * ((((0 <= num_splits) && (0 <= rmod_21)) || ((num_splits < 0) && (rmod_21 <= 0))) ? rdiv_21 : (rdiv_21 - 1)))) + (i_4 & 1)) + 16) < (((kv_page_end * 16) + last_page_len) - (kv_page_start * 16))));
      int rmod_22 = ((((((kv_page_end * 16) + last_page_len) + num_splits) - (kv_page_start * 16)) - 17) % num_splits);
      int rdiv_22 = ((((((kv_page_end * 16) + last_page_len) + num_splits) - (kv_page_start * 16)) - 17) / num_splits);
      int rmod_23 = ((((((kv_page_end * 16) + last_page_len) + num_splits) - (kv_page_start * 16)) - 17) % num_splits);
      int rdiv_23 = ((((((kv_page_end * 16) + last_page_len) + num_splits) - (kv_page_start * 16)) - 17) / num_splits);
      int rmod_24 = ((((((kv_page_end * 16) + last_page_len) + num_splits) - (kv_page_start * 16)) - 17) % num_splits);
      int rdiv_24 = ((((((kv_page_end * 16) + last_page_len) + num_splits) - (kv_page_start * 16)) - 17) / num_splits);
      float condval_7;
      if ((((((((((int)threadIdx.x) >> 5) * 16) + (((i_4 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) == 0) && ((((((kn * 16) + ((i_4 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (((int)blockIdx.z) * ((((0 <= num_splits) && (0 <= rmod_22)) || ((num_splits < 0) && (rmod_22 <= 0))) ? rdiv_22 : (rdiv_22 - 1)))) + (i_4 & 1)) < ((((int)blockIdx.z) + 1) * ((((0 <= num_splits) && (0 <= rmod_23)) || ((num_splits < 0) && (rmod_23 <= 0))) ? rdiv_23 : (rdiv_23 - 1))))) && (((((((kn * 16) + ((i_4 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (((int)blockIdx.z) * ((((0 <= num_splits) && (0 <= rmod_24)) || ((num_splits < 0) && (rmod_24 <= 0))) ? rdiv_24 : (rdiv_24 - 1)))) + (i_4 & 1)) + 16) < (((kv_page_end * 16) + last_page_len) - (kv_page_start * 16))))) {
        condval_7 = (scores[i_4] * 0x1.6a09e667f3bccp-4f/*8.838835e-02*/);
      } else {
        condval_7 = -CUDART_INF_F;
      }
      scores[i_4] = condval_7;
    }
    *(float2*)(m_prev + 0) = *(float2*)(m_i + 0);
    #pragma unroll
    for (int i_5 = 0; i_5 < 2; ++i_5) {
      m_new[i_5] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 4; ++rv) {
        m_new[i_5] = max(m_new[i_5], scores[((((rv & 1) * 4) + (i_5 * 2)) + (rv >> 1))]);
      }
      m_new[i_5] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(m_new[i_5]);
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 2; ++i_6) {
      m_new[i_6] = max(m_prev[i_6], m_new[i_6]);
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 8; ++i_7) {
      int rmod_25 = ((((((kv_page_end * 16) + last_page_len) + num_splits) - (kv_page_start * 16)) - 17) % num_splits);
      int rdiv_25 = ((((((kv_page_end * 16) + last_page_len) + num_splits) - (kv_page_start * 16)) - 17) / num_splits);
      int rmod_26 = ((((((kv_page_end * 16) + last_page_len) + num_splits) - (kv_page_start * 16)) - 17) % num_splits);
      int rdiv_26 = ((((((kv_page_end * 16) + last_page_len) + num_splits) - (kv_page_start * 16)) - 17) / num_splits);
      int rmod_27 = ((((((kv_page_end * 16) + last_page_len) + num_splits) - (kv_page_start * 16)) - 17) % num_splits);
      int rdiv_27 = ((((((kv_page_end * 16) + last_page_len) + num_splits) - (kv_page_start * 16)) - 17) / num_splits);
      float condval_8;
      if ((((((((((int)threadIdx.x) >> 5) * 16) + (((i_7 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) == 0) && ((((((kn * 16) + ((i_7 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (((int)blockIdx.z) * ((((0 <= num_splits) && (0 <= rmod_25)) || ((num_splits < 0) && (rmod_25 <= 0))) ? rdiv_25 : (rdiv_25 - 1)))) + (i_7 & 1)) < ((((int)blockIdx.z) + 1) * ((((0 <= num_splits) && (0 <= rmod_26)) || ((num_splits < 0) && (rmod_26 <= 0))) ? rdiv_26 : (rdiv_26 - 1))))) && (((((((kn * 16) + ((i_7 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (((int)blockIdx.z) * ((((0 <= num_splits) && (0 <= rmod_27)) || ((num_splits < 0) && (rmod_27 <= 0))) ? rdiv_27 : (rdiv_27 - 1)))) + (i_7 & 1)) + 16) < (((kv_page_end * 16) + last_page_len) - (kv_page_start * 16))))) {
        condval_8 = exp2f(((scores[i_7] - m_new[((i_7 & 3) >> 1)]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
      } else {
        condval_8 = 0x0p+0f/*0.000000e+00*/;
      }
      p[i_7] = condval_8;
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 2; ++i_8) {
      scale_i[i_8] = exp2f(((m_prev[i_8] - m_new[i_8]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
      l_i[i_8] = (l_i[i_8] * scale_i[i_8]);
    }
    #pragma unroll
    for (int i_9 = 0; i_9 < 64; ++i_9) {
      acc_o[i_9] = (acc_o[i_9] * scale_i[((i_9 & 3) >> 1)]);
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 2; ++i_10) {
      row_sum[i_10] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 4; ++rv_1) {
        row_sum[i_10] = (row_sum[i_10] + p[((((rv_1 & 1) * 4) + (i_10 * 2)) + (rv_1 >> 1))]);
      }
      row_sum[i_10] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(row_sum[i_10]);
    }
    #pragma unroll
    for (int i_11 = 0; i_11 < 2; ++i_11) {
      l_i[i_11] = (l_i[i_11] + row_sum[i_11]);
      m_i[i_11] = m_new[i_11];
    }
    #pragma unroll
    for (int i_12 = 0; i_12 < 2; ++i_12) {
      uint2 __1;
      float4 v_ = *(float4*)(p + (i_12 * 4));
      (reinterpret_cast<__nv_bfloat162*>(&__1))[0] = __float22bfloat162_rn(((float2*)(&v_))[0]);
      (reinterpret_cast<__nv_bfloat162*>(&__1))[1] = __float22bfloat162_rn(((float2*)(&v_))[1]);
      *(uint2*)(p_bf16 + (i_12 * 4)) = __1;
    }
    {
      bfloat16_t B_local_1[64];
      for (int i_13 = 0; i_13 < 8; ++i_13) {
        tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)v_tile)[((((i_13 >> 2) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_13 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_13 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local_1[(i_13 * 8)])));
      }
      for (int j = 0; j < 8; ++j) {
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j * 8)), reinterpret_cast<const unsigned*>(p_bf16 + 0), reinterpret_cast<const unsigned*>(B_local_1 + (j * 8)));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(p_bf16 + 0), reinterpret_cast<const unsigned*>(B_local_1 + ((j * 8) + 4)));
      }
    }
  }
  if (((int)blockIdx.x) < total_q_tokens) {
    #pragma unroll
    for (int i_14 = 0; i_14 < 32; ++i_14) {
      if (((((((int)threadIdx.x) >> 5) * 16) + ((i_14 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) == 0) {
        if (((int)blockIdx.z) < num_splits_1) {
          float broadcast_var_7 = 0x0p+0f/*0.000000e+00*/;
          float2 condval_9;
          if ((0x0p+0f/*0.000000e+00*/ < l_i[(i_14 & 1)])) {
            float2 __2;
              float2 v__1 = *(float2*)(acc_o + (i_14 * 2));
              float2 v__2 = make_float2(l_i[(i_14 & 1)], l_i[(i_14 & 1)]);
              __2.x = (v__1.x/v__2.x);
              __2.y = (v__1.y/v__2.y);
            condval_9 = __2;
          } else {
            condval_9 = make_float2(broadcast_var_7, broadcast_var_7);
          }
          *(float2*)(Partial_out + (((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)blockIdx.z)) * ((int64_t)total_q_tokens)) * (int64_t)2048)) + (((int64_t)((int)blockIdx.y)) * (int64_t)128)) + ((((int64_t)i_14) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2))) = condval_9;
        }
      }
    }
  }
  if ((((int)threadIdx.x) % 4) == 0) {
    #pragma unroll
    for (int i_15 = 0; i_15 < 2; ++i_15) {
      if (((((((int)threadIdx.x) >> 5) * 16) + (i_15 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) == 0) {
        if (((int)blockIdx.x) < total_q_tokens_2) {
          if (((int)blockIdx.z) < num_splits_3) {
            Partial_m[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + ((((int64_t)((int)blockIdx.z)) * ((int64_t)total_q_tokens_2)) * (int64_t)16)) + ((int64_t)((int)blockIdx.y)))] = m_i[i_15];
          }
        }
        if (((int)blockIdx.x) < total_q_tokens_1) {
          if (((int)blockIdx.z) < num_splits_2) {
            Partial_l[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + ((((int64_t)((int)blockIdx.z)) * ((int64_t)total_q_tokens_1)) * (int64_t)16)) + ((int64_t)((int)blockIdx.y)))] = l_i[i_15];
          }
        }
      }
    }
  }
}

