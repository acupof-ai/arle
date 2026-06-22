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

extern "C" __global__ void kernel_kernel(const bfloat16_t* __restrict__ a_inv, const float* __restrict__ beta, const float* __restrict__ g_cumsum, const bfloat16_t* __restrict__ k, bfloat16_t* __restrict__ u, const bfloat16_t* __restrict__ v, bfloat16_t* __restrict__ w, int hv, int hv_1, int hv_2, int hv_3, int hv_4, int hv_5, int hv_6, int num_value_heads, int seq_len, int seq_len_1, int seq_len_2, int seq_len_3, int seq_len_4, int seq_len_5, int seq_len_6, int seq_len_7);
extern "C" __global__ void __launch_bounds__(128, 1) kernel_kernel(const bfloat16_t* __restrict__ a_inv, const float* __restrict__ beta, const float* __restrict__ g_cumsum, const bfloat16_t* __restrict__ k, bfloat16_t* __restrict__ u, const bfloat16_t* __restrict__ v, bfloat16_t* __restrict__ w, int hv, int hv_1, int hv_2, int hv_3, int hv_4, int hv_5, int hv_6, int num_value_heads, int seq_len, int seq_len_1, int seq_len_2, int seq_len_3, int seq_len_4, int seq_len_5, int seq_len_6, int seq_len_7) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* ai_tile = ((void*)((char*)buf_dyn_shmem + 0));
  void* k_tile = ((void*)((char*)buf_dyn_shmem + 8192));
  void* v_tile = ((void*)((char*)buf_dyn_shmem + 8192));
  float beta_frag[1];
  float g_frag[1];
  bfloat16_t v_local_cast_1[8];
  bfloat16_t v_tile_local_cast[8];
  float u_acc[64];
  bfloat16_t u_local_cast_2[2];
  bfloat16_t k_local_cast_4[8];
  bfloat16_t k_tile_local_cast_3[8];
  float w_acc[64];
  bfloat16_t w_local_cast_5[2];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    bfloat16_t broadcast_var = bfloat16_t(0x0p+0f/*0.000000e+00*/);
    uint4 condval;
    if (((((((((int)blockIdx.x) * 64) + (i * 16)) + (((int)threadIdx.x) >> 3)) < seq_len) && (((int)blockIdx.y) < hv_6)) && ((((((int)blockIdx.x) * 64) + (i * 16)) + (((int)threadIdx.x) >> 3)) < seq_len_6))) {
      condval = *(uint4*)(a_inv + (((((int64_t)((int)blockIdx.y)) * (int64_t)64) + (((((((int64_t)((int)blockIdx.x)) * (int64_t)64) + (((int64_t)i) * (int64_t)16)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)3)) * ((int64_t)hv_6)) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)));
    } else {
      condval = make_uint4(__pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var));
    }
    *(uint4*)(((bfloat16_t*)ai_tile) + (((((i * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval;
  }
  bool in_range = (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) & 63)) < seq_len);
  float condval_1;
  if ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) & 63)) < seq_len) && (((int)blockIdx.y) < hv_5)) && (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) & 63)) < seq_len_4))) {
    condval_1 = beta[((((((int64_t)((int)blockIdx.x)) * (int64_t)64) + (((int64_t)((int)threadIdx.x)) & (int64_t)63)) * ((int64_t)hv_5)) + ((int64_t)((int)blockIdx.y)))];
  } else {
    condval_1 = 0x0p+0f/*0.000000e+00*/;
  }
  beta_frag[0] = condval_1;
  float condval_2;
  if ((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) & 63)) < seq_len)) {
    float condval_3;
    if (((((int)blockIdx.y) < hv_4) && (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) & 63)) < seq_len_2))) {
      condval_3 = g_cumsum[((((((int64_t)((int)blockIdx.x)) * (int64_t)64) + (((int64_t)((int)threadIdx.x)) & (int64_t)63)) * ((int64_t)hv_4)) + ((int64_t)((int)blockIdx.y)))];
    } else {
      condval_3 = 0x0p+0f/*0.000000e+00*/;
    }
    condval_2 = expf(condval_3);
  } else {
    condval_2 = 0x0p+0f/*0.000000e+00*/;
  }
  g_frag[0] = condval_2;
  if ((((int)threadIdx.x) >> 6) == 0) {
    __syncthreads();
    #pragma unroll
    for (int i_1 = 0; i_1 < 16; ++i_1) {
      bfloat16_t broadcast_var_1 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
      uint4 condval_4;
      if (((((int)blockIdx.y) < hv) && (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) & 63)) < seq_len_1))) {
        condval_4 = *(uint4*)(v + (((((int64_t)((int)blockIdx.y)) * (int64_t)128) + ((((((int64_t)((int)blockIdx.x)) * (int64_t)64) + (((int64_t)((int)threadIdx.x)) & (int64_t)63)) * ((int64_t)hv)) * (int64_t)128)) + (((int64_t)i_1) * (int64_t)8)));
      } else {
        condval_4 = make_uint4(__pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1));
      }
      *(uint4*)(v_local_cast_1 + 0) = condval_4;
      for (int vec = 0; vec < 2; ++vec) {
        bfloat16_t broadcast_var_2 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
        uint2 condval_5;
        if ((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) & 63)) < seq_len)) {
          uint2 __1;
          float4 __2;
            float4 __3;
            uint2 v_ = *(uint2*)(v_local_cast_1 + (vec * 4));
            ((float2*)(&__3))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v_))[0]);
            ((float2*)(&__3))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v_))[1]);
            float4 v__1 = make_float4(beta_frag[0], beta_frag[0], beta_frag[0], beta_frag[0]);
            *(float2*)(&(__2.x)) = tl::mul2(*(float2*)(&(__3.x)), *(float2*)(&(v__1.x)));
            *(float2*)(&(__2.z)) = tl::mul2(*(float2*)(&(__3.z)), *(float2*)(&(v__1.z)));
          (reinterpret_cast<__nv_bfloat162*>(&__1))[0] = __float22bfloat162_rn(((float2*)(&__2))[0]);
          (reinterpret_cast<__nv_bfloat162*>(&__1))[1] = __float22bfloat162_rn(((float2*)(&__2))[1]);
          condval_5 = __1;
        } else {
          condval_5 = make_uint2(__pack_nv_bfloat162(broadcast_var_2, broadcast_var_2), __pack_nv_bfloat162(broadcast_var_2, broadcast_var_2));
        }
        *(uint2*)(v_tile_local_cast + (vec * 4)) = condval_5;
      }
      *(uint4*)(((bfloat16_t*)v_tile) + ((((((i_1 >> 3) * 4096) + ((((int)threadIdx.x) & 63) * 64)) + (((((i_1 & 7) >> 2) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((i_1 & 3) >> 1) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + ((((i_1 & 1) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(v_tile_local_cast + 0);
    }
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 16; ++i_2) {
    float broadcast_var_3 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(u_acc + (i_2 * 4)) = make_float4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
  }
  {
    bfloat16_t A_local[8];
    bfloat16_t B_local[64];
    __syncthreads();
    for (int ki = 0; ki < 4; ++ki) {
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)ai_tile)[((((((int)threadIdx.x) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local[0])));
      for (int i_3 = 0; i_3 < 8; ++i_3) {
        tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)v_tile)[(((((i_3 >> 2) * 4096) + (ki * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_3 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_3 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local[(i_3 * 8)])));
      }
      for (int j = 0; j < 8; ++j) {
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(u_acc + (j * 8)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(u_acc + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
      }
    }
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 32; ++i_4) {
    if (((((((int)blockIdx.x) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_4 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < seq_len) {
      uint1 __4;
      float2 v__2 = *(float2*)(u_acc + (i_4 * 2));
      (reinterpret_cast<__nv_bfloat162*>(&__4))[0] = __float22bfloat162_rn(((float2*)(&v__2))[0]);
      *(uint1*)(u_local_cast_2 + 0) = __4;
      if (((int)blockIdx.y) < hv_3) {
        if (((((((int)blockIdx.x) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_4 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < seq_len_3) {
          *(uint1*)(u + ((((((int64_t)((int)blockIdx.y)) * (int64_t)128) + ((((((((int64_t)((int)blockIdx.x)) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)16)) + ((((int64_t)i_4) & (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2)) * ((int64_t)hv_3)) * (int64_t)128)) + ((((int64_t)i_4) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2))) = *(uint1*)(u_local_cast_2 + 0);
        }
      }
    }
  }
  if ((((int)threadIdx.x) >> 6) == 0) {
    __syncthreads();
    #pragma unroll
    for (int i_5 = 0; i_5 < 16; ++i_5) {
      bfloat16_t broadcast_var_4 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
      uint4 condval_6;
      if (((((int)blockIdx.y) < hv_2) && (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) & 63)) < seq_len_5))) {
        condval_6 = *(uint4*)(k + (((((int64_t)((int)blockIdx.y)) * (int64_t)128) + ((((((int64_t)((int)blockIdx.x)) * (int64_t)64) + (((int64_t)((int)threadIdx.x)) & (int64_t)63)) * ((int64_t)hv_2)) * (int64_t)128)) + (((int64_t)i_5) * (int64_t)8)));
      } else {
        condval_6 = make_uint4(__pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4));
      }
      *(uint4*)(k_local_cast_4 + 0) = condval_6;
      for (int vec_1 = 0; vec_1 < 2; ++vec_1) {
        bfloat16_t broadcast_var_5 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
        uint2 condval_7;
        if ((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) & 63)) < seq_len)) {
          uint2 __5;
          float4 __6;
            float4 __7;
              float4 __8;
              uint2 v__3 = *(uint2*)(k_local_cast_4 + (vec_1 * 4));
              ((float2*)(&__8))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__3))[0]);
              ((float2*)(&__8))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__3))[1]);
              float4 v__4 = make_float4(beta_frag[0], beta_frag[0], beta_frag[0], beta_frag[0]);
              *(float2*)(&(__7.x)) = tl::mul2(*(float2*)(&(__8.x)), *(float2*)(&(v__4.x)));
              *(float2*)(&(__7.z)) = tl::mul2(*(float2*)(&(__8.z)), *(float2*)(&(v__4.z)));
            float4 v__5 = make_float4(g_frag[0], g_frag[0], g_frag[0], g_frag[0]);
            *(float2*)(&(__6.x)) = tl::mul2(*(float2*)(&(__7.x)), *(float2*)(&(v__5.x)));
            *(float2*)(&(__6.z)) = tl::mul2(*(float2*)(&(__7.z)), *(float2*)(&(v__5.z)));
          (reinterpret_cast<__nv_bfloat162*>(&__5))[0] = __float22bfloat162_rn(((float2*)(&__6))[0]);
          (reinterpret_cast<__nv_bfloat162*>(&__5))[1] = __float22bfloat162_rn(((float2*)(&__6))[1]);
          condval_7 = __5;
        } else {
          condval_7 = make_uint2(__pack_nv_bfloat162(broadcast_var_5, broadcast_var_5), __pack_nv_bfloat162(broadcast_var_5, broadcast_var_5));
        }
        *(uint2*)(k_tile_local_cast_3 + (vec_1 * 4)) = condval_7;
      }
      *(uint4*)(((bfloat16_t*)k_tile) + ((((((i_5 >> 3) * 4096) + ((((int)threadIdx.x) & 63) * 64)) + (((((i_5 & 7) >> 2) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((i_5 & 3) >> 1) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + ((((i_5 & 1) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(k_tile_local_cast_3 + 0);
    }
  }
  #pragma unroll
  for (int i_6 = 0; i_6 < 16; ++i_6) {
    float broadcast_var_6 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(w_acc + (i_6 * 4)) = make_float4(broadcast_var_6, broadcast_var_6, broadcast_var_6, broadcast_var_6);
  }
  {
    bfloat16_t A_local_1[8];
    bfloat16_t B_local_1[64];
    __syncthreads();
    for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)ai_tile)[((((((int)threadIdx.x) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_1[0])));
      for (int i_7 = 0; i_7 < 8; ++i_7) {
        tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)k_tile)[(((((i_7 >> 2) * 4096) + (ki_1 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_7 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_7 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local_1[(i_7 * 8)])));
      }
      for (int j_1 = 0; j_1 < 8; ++j_1) {
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(w_acc + (j_1 * 8)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(w_acc + ((j_1 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
      }
    }
  }
  #pragma unroll
  for (int i_8 = 0; i_8 < 32; ++i_8) {
    if (((((((int)blockIdx.x) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_8 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < seq_len) {
      uint1 __9;
      float2 v__6 = *(float2*)(w_acc + (i_8 * 2));
      (reinterpret_cast<__nv_bfloat162*>(&__9))[0] = __float22bfloat162_rn(((float2*)(&v__6))[0]);
      *(uint1*)(w_local_cast_5 + 0) = __9;
      if (((int)blockIdx.y) < hv_1) {
        if (((((((int)blockIdx.x) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_8 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < seq_len_7) {
          *(uint1*)(w + ((((((int64_t)((int)blockIdx.y)) * (int64_t)128) + ((((((((int64_t)((int)blockIdx.x)) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)16)) + ((((int64_t)i_8) & (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2)) * ((int64_t)hv_1)) * (int64_t)128)) + ((((int64_t)i_8) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2))) = *(uint1*)(w_local_cast_5 + 0);
        }
      }
    }
  }
}

