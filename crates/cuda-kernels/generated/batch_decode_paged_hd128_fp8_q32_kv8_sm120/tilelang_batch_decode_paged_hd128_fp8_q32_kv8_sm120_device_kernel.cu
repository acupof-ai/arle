#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/mma.h>
#include <tl_templates/cuda/cuda_fp8.h>
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

extern "C" __global__ void kernel_kernel(const int* __restrict__ KV_indices, const int* __restrict__ KV_indptr, const int* __restrict__ KV_last_page_len, const fp8_e4_t* __restrict__ K_pool, const float* __restrict__ K_scales, bfloat16_t* __restrict__ Output, const bfloat16_t* __restrict__ Q, const fp8_e4_t* __restrict__ V_pool, const float* __restrict__ V_scales, int batch_size, int batch_size_1, int batch_size_plus_one, int num_pages, int num_pages_1, int num_pages_2, int num_pages_3, int total_pages, int total_q_tokens, int total_q_tokens_1);
extern "C" __global__ void __launch_bounds__(128, 1) kernel_kernel(const int* __restrict__ KV_indices, const int* __restrict__ KV_indptr, const int* __restrict__ KV_last_page_len, const fp8_e4_t* __restrict__ K_pool, const float* __restrict__ K_scales, bfloat16_t* __restrict__ Output, const bfloat16_t* __restrict__ Q, const fp8_e4_t* __restrict__ V_pool, const float* __restrict__ V_scales, int batch_size, int batch_size_1, int batch_size_plus_one, int num_pages, int num_pages_1, int num_pages_2, int num_pages_3, int total_pages, int total_q_tokens, int total_q_tokens_1) {
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
    if (((((i_1 * 8) + (((int)threadIdx.x) >> 4)) == 0) && (((int)blockIdx.z) < total_q_tokens_1))) {
      condval_3 = *(uint4*)(Q + (((((int64_t)((int)blockIdx.z)) * (int64_t)4096) + (((int64_t)((int)blockIdx.y)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
    } else {
      condval_3 = make_uint4(__pack_nv_bfloat162(broadcast_var_3, broadcast_var_3), __pack_nv_bfloat162(broadcast_var_3, broadcast_var_3), __pack_nv_bfloat162(broadcast_var_3, broadcast_var_3), __pack_nv_bfloat162(broadcast_var_3, broadcast_var_3));
    }
    *(uint4*)(((bfloat16_t*)q_tile) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_1 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_3;
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
      fp8_e4_t broadcast_var_4 = fp8_e4_t(0x0p+0f/*0.000000e+00*/);
      fp8_e4_4_t condval_5;
      if (((0 <= page_idx) && (page_idx < num_pages_2))) {
        condval_5 = *(fp8_e4_4_t*)(K_pool + ((((((int64_t)page_idx) * (int64_t)16384) + ((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * (int64_t)2048)) + (((int64_t)i_2) * (int64_t)512)) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)));
      } else {
        condval_5 = make_fp8_e4_4_t(broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4);
      }
      fp8_e4_4_t k_fp8 = condval_5;
      float condval_6;
      if (((0 <= page_idx) && (page_idx < num_pages))) {
        condval_6 = K_scales[((((((int64_t)page_idx) * (int64_t)128) + (((int64_t)i_2) * (int64_t)32)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)8)) + (((int64_t)((int)blockIdx.y)) >> (int64_t)2))];
      } else {
        condval_6 = 0x0p+0f/*0.000000e+00*/;
      }
      float k_scale = condval_6;
      uint2 __1;
      float4 __2;
        float4 __3;
        ((float2*)(&__3))[0] = __tl_cvt_fp8x2_to_float2((reinterpret_cast<__nv_fp8x2_storage_t*>(&k_fp8))[0], __NV_E4M3);
        ((float2*)(&__3))[1] = __tl_cvt_fp8x2_to_float2((reinterpret_cast<__nv_fp8x2_storage_t*>(&k_fp8))[1], __NV_E4M3);
        float4 v_ = make_float4(k_scale, k_scale, k_scale, k_scale);
        *(float2*)(&(__2.x)) = tl::mul2(*(float2*)(&(__3.x)), *(float2*)(&(v_.x)));
        *(float2*)(&(__2.z)) = tl::mul2(*(float2*)(&(__3.z)), *(float2*)(&(v_.z)));
      (reinterpret_cast<__nv_bfloat162*>(&__1))[0] = __float22bfloat162_rn(((float2*)(&__2))[0]);
      (reinterpret_cast<__nv_bfloat162*>(&__1))[1] = __float22bfloat162_rn(((float2*)(&__2))[1]);
      uint2 k_bf16 = __1;
      bfloat16_t broadcast_var_5 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
      uint2 condval_7;
      if ((((((kn * 16) + (i_2 * 4)) + (((int)threadIdx.x) >> 5)) + 16) < (((kv_page_end * 16) + last_page_len) - (kv_page_start * 16)))) {
        uint2 __4;
        float4 __5;
          float4 __6;
          ((float2*)(&__6))[0] = __tl_cvt_fp8x2_to_float2((reinterpret_cast<__nv_fp8x2_storage_t*>(&k_fp8))[0], __NV_E4M3);
          ((float2*)(&__6))[1] = __tl_cvt_fp8x2_to_float2((reinterpret_cast<__nv_fp8x2_storage_t*>(&k_fp8))[1], __NV_E4M3);
          float4 v__1 = make_float4(k_scale, k_scale, k_scale, k_scale);
          *(float2*)(&(__5.x)) = tl::mul2(*(float2*)(&(__6.x)), *(float2*)(&(v__1.x)));
          *(float2*)(&(__5.z)) = tl::mul2(*(float2*)(&(__6.z)), *(float2*)(&(v__1.z)));
        (reinterpret_cast<__nv_bfloat162*>(&__4))[0] = __float22bfloat162_rn(((float2*)(&__5))[0]);
        (reinterpret_cast<__nv_bfloat162*>(&__4))[1] = __float22bfloat162_rn(((float2*)(&__5))[1]);
        condval_7 = __4;
      } else {
        condval_7 = make_uint2(__pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5));
      }
      *(uint2*)(((bfloat16_t*)k_tile) + (((((((((((int)threadIdx.x) & 31) >> 4) * 1024) + (i_2 * 256)) + ((((int)threadIdx.x) >> 5) * 64)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_2 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 1) * 4))) = condval_7;
      fp8_e4_t broadcast_var_6 = fp8_e4_t(0x0p+0f/*0.000000e+00*/);
      fp8_e4_4_t condval_8;
      if (((0 <= page_idx) && (page_idx < num_pages_1))) {
        condval_8 = *(fp8_e4_4_t*)(V_pool + ((((((int64_t)page_idx) * (int64_t)16384) + ((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * (int64_t)2048)) + (((int64_t)i_2) * (int64_t)512)) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)));
      } else {
        condval_8 = make_fp8_e4_4_t(broadcast_var_6, broadcast_var_6, broadcast_var_6, broadcast_var_6);
      }
      fp8_e4_4_t v_fp8 = condval_8;
      float condval_9;
      if (((0 <= page_idx) && (page_idx < num_pages_3))) {
        condval_9 = V_scales[((((((int64_t)page_idx) * (int64_t)128) + (((int64_t)i_2) * (int64_t)32)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)8)) + (((int64_t)((int)blockIdx.y)) >> (int64_t)2))];
      } else {
        condval_9 = 0x0p+0f/*0.000000e+00*/;
      }
      float v_scale = condval_9;
      uint2 __7;
      float4 __8;
        float4 __9;
        ((float2*)(&__9))[0] = __tl_cvt_fp8x2_to_float2((reinterpret_cast<__nv_fp8x2_storage_t*>(&v_fp8))[0], __NV_E4M3);
        ((float2*)(&__9))[1] = __tl_cvt_fp8x2_to_float2((reinterpret_cast<__nv_fp8x2_storage_t*>(&v_fp8))[1], __NV_E4M3);
        float4 v__2 = make_float4(v_scale, v_scale, v_scale, v_scale);
        *(float2*)(&(__8.x)) = tl::mul2(*(float2*)(&(__9.x)), *(float2*)(&(v__2.x)));
        *(float2*)(&(__8.z)) = tl::mul2(*(float2*)(&(__9.z)), *(float2*)(&(v__2.z)));
      (reinterpret_cast<__nv_bfloat162*>(&__7))[0] = __float22bfloat162_rn(((float2*)(&__8))[0]);
      (reinterpret_cast<__nv_bfloat162*>(&__7))[1] = __float22bfloat162_rn(((float2*)(&__8))[1]);
      uint2 v_bf16 = __7;
      bfloat16_t broadcast_var_7 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
      uint2 condval_10;
      if ((((((kn * 16) + (i_2 * 4)) + (((int)threadIdx.x) >> 5)) + 16) < (((kv_page_end * 16) + last_page_len) - (kv_page_start * 16)))) {
        uint2 __10;
        float4 __11;
          float4 __12;
          ((float2*)(&__12))[0] = __tl_cvt_fp8x2_to_float2((reinterpret_cast<__nv_fp8x2_storage_t*>(&v_fp8))[0], __NV_E4M3);
          ((float2*)(&__12))[1] = __tl_cvt_fp8x2_to_float2((reinterpret_cast<__nv_fp8x2_storage_t*>(&v_fp8))[1], __NV_E4M3);
          float4 v__3 = make_float4(v_scale, v_scale, v_scale, v_scale);
          *(float2*)(&(__11.x)) = tl::mul2(*(float2*)(&(__12.x)), *(float2*)(&(v__3.x)));
          *(float2*)(&(__11.z)) = tl::mul2(*(float2*)(&(__12.z)), *(float2*)(&(v__3.z)));
        (reinterpret_cast<__nv_bfloat162*>(&__10))[0] = __float22bfloat162_rn(((float2*)(&__11))[0]);
        (reinterpret_cast<__nv_bfloat162*>(&__10))[1] = __float22bfloat162_rn(((float2*)(&__11))[1]);
        condval_10 = __10;
      } else {
        condval_10 = make_uint2(__pack_nv_bfloat162(broadcast_var_7, broadcast_var_7), __pack_nv_bfloat162(broadcast_var_7, broadcast_var_7));
      }
      *(uint2*)(((bfloat16_t*)v_tile) + (((((((((((int)threadIdx.x) & 31) >> 4) * 1024) + (i_2 * 256)) + ((((int)threadIdx.x) >> 5) * 64)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_2 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 1) * 4))) = condval_10;
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 2; ++i_3) {
      float broadcast_var_8 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(scores + (i_3 * 4)) = make_float4(broadcast_var_8, broadcast_var_8, broadcast_var_8, broadcast_var_8);
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
      bool in_bounds = ((((((((int)threadIdx.x) >> 5) * 16) + (((i_4 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) == 0) && ((((((kn * 16) + ((i_4 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_4 & 1)) + 16) < (((kv_page_end * 16) + last_page_len) - (kv_page_start * 16))));
      float condval_11;
      if (((((((((int)threadIdx.x) >> 5) * 16) + (((i_4 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) == 0) && ((((((kn * 16) + ((i_4 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_4 & 1)) + 16) < (((kv_page_end * 16) + last_page_len) - (kv_page_start * 16))))) {
        condval_11 = (scores[i_4] * 0x1.6a09e667f3bccp-4f/*8.838835e-02*/);
      } else {
        condval_11 = -CUDART_INF_F;
      }
      scores[i_4] = condval_11;
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
      float condval_12;
      if (((((((((int)threadIdx.x) >> 5) * 16) + (((i_7 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) == 0) && ((((((kn * 16) + ((i_7 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_7 & 1)) + 16) < (((kv_page_end * 16) + last_page_len) - (kv_page_start * 16))))) {
        condval_12 = exp2f(((scores[i_7] - m_new[((i_7 & 3) >> 1)]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
      } else {
        condval_12 = 0x0p+0f/*0.000000e+00*/;
      }
      p[i_7] = condval_12;
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
      uint2 __13;
      float4 v__4 = *(float4*)(p + (i_12 * 4));
      (reinterpret_cast<__nv_bfloat162*>(&__13))[0] = __float22bfloat162_rn(((float2*)(&v__4))[0]);
      (reinterpret_cast<__nv_bfloat162*>(&__13))[1] = __float22bfloat162_rn(((float2*)(&v__4))[1]);
      *(uint2*)(p_bf16 + (i_12 * 4)) = __13;
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
  #pragma unroll
  for (int i_14 = 0; i_14 < 32; ++i_14) {
    if (((((((int)threadIdx.x) >> 5) * 16) + ((i_14 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) == 0) {
      uint1 __14;
      float2 __15;
        float2 v__5 = *(float2*)(acc_o + (i_14 * 2));
        float2 v__6 = make_float2(l_i[(i_14 & 1)], l_i[(i_14 & 1)]);
        __15.x = (v__5.x/v__6.x);
        __15.y = (v__5.y/v__6.y);
      (reinterpret_cast<__nv_bfloat162*>(&__14))[0] = __float22bfloat162_rn(((float2*)(&__15))[0]);
      *(uint1*)(Output_local_cast + 0) = __14;
      if (((int)blockIdx.z) < total_q_tokens) {
        *(uint1*)(Output + ((((((int64_t)((int)blockIdx.z)) * (int64_t)4096) + (((int64_t)((int)blockIdx.y)) * (int64_t)128)) + ((((int64_t)i_14) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2))) = *(uint1*)(Output_local_cast + 0);
      }
    }
  }
}

