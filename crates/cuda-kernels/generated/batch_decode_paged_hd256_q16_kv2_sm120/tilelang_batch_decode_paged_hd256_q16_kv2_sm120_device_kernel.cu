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

extern "C" __global__ void kernel_kernel(const int* __restrict__ KV_indices, const int* __restrict__ KV_indptr, const int* __restrict__ KV_last_page_len, const bfloat16_t* __restrict__ K_pool, bfloat16_t* __restrict__ Output, const bfloat16_t* __restrict__ Q, const bfloat16_t* __restrict__ V_pool, int batch_size, int batch_size_1, int batch_size_plus_one, int num_pages, int num_pages_1, int total_pages, int total_q_tokens, int total_q_tokens_1);
extern "C" __global__ void __launch_bounds__(128, 1) kernel_kernel(const int* __restrict__ KV_indices, const int* __restrict__ KV_indptr, const int* __restrict__ KV_last_page_len, const bfloat16_t* __restrict__ K_pool, bfloat16_t* __restrict__ Output, const bfloat16_t* __restrict__ Q, const bfloat16_t* __restrict__ V_pool, int batch_size, int batch_size_1, int batch_size_plus_one, int num_pages, int num_pages_1, int total_pages, int total_q_tokens, int total_q_tokens_1) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* q_tile = ((void*)((char*)buf_dyn_shmem + 0));
  void* k_tile = ((void*)((char*)buf_dyn_shmem + 32768));
  void* v_tile = ((void*)((char*)buf_dyn_shmem + 40960));
  float acc_o[128];
  float m_i[2];
  float l_i[2];
  float scores[8];
  float m_prev[2];
  float m_new[2];
  float p[8];
  float scale_i[2];
  float row_sum[2];
  bfloat16_t p_bf16[8];
  bfloat16_t Output_local_cast[2];
  const dim3 blockIdx = tl::rasterization2DRow<8>();
  int condval;
  if ((((int)blockIdx.z) < batch_size_plus_one)) {
    condval = KV_indptr[((int)blockIdx.z)];
  } else {
    condval = 0;
  }
  int kv_page_start = condval;
  int condval_1;
  if (((((int)blockIdx.z) + 1) < batch_size_plus_one)) {
    condval_1 = KV_indptr[(((int64_t)((int)blockIdx.z)) + (int64_t)1)];
  } else {
    condval_1 = 0;
  }
  int kv_page_end = condval_1;
  int condval_2;
  if ((((int)blockIdx.z) < batch_size_1)) {
    condval_2 = KV_last_page_len[((int)blockIdx.z)];
  } else {
    condval_2 = 0;
  }
  int last_page_len = condval_2;
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_o + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  float broadcast_var_1 = -CUDART_INF_F;
  *(float2*)(m_i + 0) = make_float2(broadcast_var_1, broadcast_var_1);
  float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
  *(float2*)(l_i + 0) = make_float2(broadcast_var_2, broadcast_var_2);
  #pragma unroll
  for (int i_1 = 0; i_1 < 16; ++i_1) {
    bfloat16_t broadcast_var_3 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
    uint4 condval_3;
    if (((((i_1 * 4) + (((int)threadIdx.x) >> 5)) == 0) && (((int)blockIdx.z) < total_q_tokens))) {
      condval_3 = *(uint4*)(Q + (((((int64_t)((int)blockIdx.z)) * (int64_t)4096) + (((int64_t)((int)blockIdx.y)) * (int64_t)256)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) * (int64_t)8)));
    } else {
      condval_3 = make_uint4(__pack_nv_bfloat162(broadcast_var_3, broadcast_var_3), __pack_nv_bfloat162(broadcast_var_3, broadcast_var_3), __pack_nv_bfloat162(broadcast_var_3, broadcast_var_3), __pack_nv_bfloat162(broadcast_var_3, broadcast_var_3));
    }
    *(uint4*)(((bfloat16_t*)q_tile) + ((((((((((int)threadIdx.x) & 31) >> 3) * 4096) + (i_1 * 256)) + ((((int)threadIdx.x) >> 5) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_1 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_3;
  }
  for (int kn = 0; kn < ((((last_page_len - 1) >> 4) + kv_page_end) - kv_page_start); ++kn) {
    __syncthreads();
    #pragma unroll
    for (int i_2 = 0; i_2 < 4; ++i_2) {
      int condval_4;
      if ((((((((kn * 16) + (i_2 * 4)) + (((int)threadIdx.x) >> 5)) + 16) < (((kv_page_end * 16) + last_page_len) - (kv_page_start * 16))) && (0 <= (kv_page_start + kn))) && ((kv_page_start + kn) < total_pages))) {
        condval_4 = KV_indices[(((int64_t)kv_page_start) + ((int64_t)kn))];
      } else {
        condval_4 = 0;
      }
      int page_idx = condval_4;
      bfloat16_t broadcast_var_4 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
      uint4 condval_5;
      if ((((((((kn * 16) + (i_2 * 4)) + (((int)threadIdx.x) >> 5)) + 16) < (((kv_page_end * 16) + last_page_len) - (kv_page_start * 16))) && (0 <= page_idx)) && (page_idx < num_pages))) {
        condval_5 = *(uint4*)(K_pool + ((((((int64_t)page_idx) * (int64_t)8192) + ((((int64_t)((int)blockIdx.y)) >> (int64_t)3) * (int64_t)4096)) + (((int64_t)i_2) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)));
      } else {
        condval_5 = make_uint4(__pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4));
      }
      *(uint4*)(((bfloat16_t*)k_tile) + ((((((((((int)threadIdx.x) & 31) >> 3) * 1024) + (i_2 * 256)) + ((((int)threadIdx.x) >> 5) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_2 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_5;
      bfloat16_t broadcast_var_5 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
      uint4 condval_6;
      if ((((((((kn * 16) + (i_2 * 4)) + (((int)threadIdx.x) >> 5)) + 16) < (((kv_page_end * 16) + last_page_len) - (kv_page_start * 16))) && (0 <= page_idx)) && (page_idx < num_pages_1))) {
        condval_6 = *(uint4*)(V_pool + ((((((int64_t)page_idx) * (int64_t)8192) + ((((int64_t)((int)blockIdx.y)) >> (int64_t)3) * (int64_t)4096)) + (((int64_t)i_2) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)));
      } else {
        condval_6 = make_uint4(__pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5));
      }
      *(uint4*)(((bfloat16_t*)v_tile) + ((((((((((int)threadIdx.x) & 31) >> 3) * 1024) + (i_2 * 256)) + ((((int)threadIdx.x) >> 5) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_2 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_6;
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
      for (int ki = 0; ki < 16; ++ki) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)q_tile)[(((((ki >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local[0])));
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)k_tile)[(((((((ki >> 2) * 1024) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[0])));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores + 0), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 0));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores + 4), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 4));
      }
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 8; ++i_4) {
      bool in_bounds = ((((((((int)threadIdx.x) >> 5) * 16) + (((i_4 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) == 0) && ((((((kn * 16) + ((i_4 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_4 & 1)) + 16) < (((kv_page_end * 16) + last_page_len) - (kv_page_start * 16))));
      float condval_7;
      if (((((((((int)threadIdx.x) >> 5) * 16) + (((i_4 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) == 0) && ((((((kn * 16) + ((i_4 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_4 & 1)) + 16) < (((kv_page_end * 16) + last_page_len) - (kv_page_start * 16))))) {
        condval_7 = (scores[i_4] * 0x1p-4f/*6.250000e-02*/);
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
      float condval_8;
      if (((((((((int)threadIdx.x) >> 5) * 16) + (((i_7 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) == 0) && ((((((kn * 16) + ((i_7 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_7 & 1)) + 16) < (((kv_page_end * 16) + last_page_len) - (kv_page_start * 16))))) {
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
    for (int i_9 = 0; i_9 < 128; ++i_9) {
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
      bfloat16_t B_local_1[128];
      for (int i_13 = 0; i_13 < 16; ++i_13) {
        tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)v_tile)[((((i_13 >> 2) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_13 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_13 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local_1[(i_13 * 8)])));
      }
      for (int j = 0; j < 16; ++j) {
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j * 8)), reinterpret_cast<const unsigned*>(p_bf16 + 0), reinterpret_cast<const unsigned*>(B_local_1 + (j * 8)));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(p_bf16 + 0), reinterpret_cast<const unsigned*>(B_local_1 + ((j * 8) + 4)));
      }
    }
  }
  #pragma unroll
  for (int i_14 = 0; i_14 < 64; ++i_14) {
    if (((((((int)threadIdx.x) >> 5) * 16) + ((i_14 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) == 0) {
      uint1 __2;
      float2 __3;
        float2 v__1 = *(float2*)(acc_o + (i_14 * 2));
        float2 v__2 = make_float2(l_i[(i_14 & 1)], l_i[(i_14 & 1)]);
        __3.x = (v__1.x/v__2.x);
        __3.y = (v__1.y/v__2.y);
      (reinterpret_cast<__nv_bfloat162*>(&__2))[0] = __float22bfloat162_rn(((float2*)(&__3))[0]);
      *(uint1*)(Output_local_cast + 0) = __2;
      if (((int)blockIdx.z) < total_q_tokens_1) {
        *(uint1*)(Output + ((((((int64_t)((int)blockIdx.z)) * (int64_t)4096) + (((int64_t)((int)blockIdx.y)) * (int64_t)256)) + ((((int64_t)i_14) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2))) = *(uint1*)(Output_local_cast + 0);
      }
    }
  }
}

