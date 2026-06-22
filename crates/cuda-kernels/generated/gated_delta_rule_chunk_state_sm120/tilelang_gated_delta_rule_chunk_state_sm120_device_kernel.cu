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

extern "C" __global__ void kernel_kernel(float* __restrict__ chunk_state, float* __restrict__ final_state, const float* __restrict__ g_cumsum, const float* __restrict__ initial_state, const bfloat16_t* __restrict__ k, const bfloat16_t* __restrict__ u, bfloat16_t* __restrict__ v_new, const bfloat16_t* __restrict__ w, int hv, int hv_1, int hv_2, int hv_3, int hv_4, int hv_5, int hv_6, int hv_7, int num_chunks, int num_value_heads, int seq_len, int seq_len_1, int seq_len_2, int seq_len_3, int seq_len_4, int seq_len_5);
extern "C" __global__ void __launch_bounds__(128, 1) kernel_kernel(float* __restrict__ chunk_state, float* __restrict__ final_state, const float* __restrict__ g_cumsum, const float* __restrict__ initial_state, const bfloat16_t* __restrict__ k, const bfloat16_t* __restrict__ u, bfloat16_t* __restrict__ v_new, const bfloat16_t* __restrict__ w, int hv, int hv_1, int hv_2, int hv_3, int hv_4, int hv_5, int hv_6, int hv_7, int num_chunks, int num_value_heads, int seq_len, int seq_len_1, int seq_len_2, int seq_len_3, int seq_len_4, int seq_len_5) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* k_hi = ((void*)((char*)buf_dyn_shmem + 0));
  void* k_lo = ((void*)((char*)buf_dyn_shmem + 8192));
  void* w_hi = ((void*)((char*)buf_dyn_shmem + 16384));
  void* w_lo = ((void*)((char*)buf_dyn_shmem + 24576));
  void* h_hi_sh = ((void*)((char*)buf_dyn_shmem + 32768));
  void* h_lo_sh = ((void*)((char*)buf_dyn_shmem + 36864));
  void* u_tile = ((void*)((char*)buf_dyn_shmem + 40960));
  void* v_new_bf = ((void*)((char*)buf_dyn_shmem + 45056));
  float h_lo[16];
  float h_hi[16];
  float v_new_tile[16];
  bfloat16_t u_tile_local_cast[2];
  float wh_acc[16];
  bfloat16_t v_new_local_cast_1[2];
  float g_frag[2];
  float kh_lo_acc[16];
  float kh_hi_acc[16];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    float2 condval;
    if ((((int)blockIdx.y) < hv)) {
      condval = *(float2*)(initial_state + (((((((((int64_t)((int)blockIdx.y)) * (int64_t)16384) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)2048)) + ((((int64_t)i) & (int64_t)1) * (int64_t)1024)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)128)) + (((int64_t)((int)blockIdx.x)) * (int64_t)32)) + ((((int64_t)i) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)));
    } else {
      condval = make_float2(broadcast_var, broadcast_var);
    }
    *(float2*)(h_lo + (i * 2)) = condval;
    float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
    float2 condval_1;
    if ((((int)blockIdx.y) < hv)) {
      condval_1 = *(float2*)(initial_state + ((((((((((int64_t)((int)blockIdx.y)) * (int64_t)16384) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)2048)) + ((((int64_t)i) & (int64_t)1) * (int64_t)1024)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)128)) + (((int64_t)((int)blockIdx.x)) * (int64_t)32)) + ((((int64_t)i) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)) + (int64_t)8192));
    } else {
      condval_1 = make_float2(broadcast_var_1, broadcast_var_1);
    }
    *(float2*)(h_hi + (i * 2)) = condval_1;
  }
  for (int chunk_idx = 0; chunk_idx < ((seq_len_5 + 63) >> 6); ++chunk_idx) {
    if (((int)blockIdx.y) < hv_3) {
      #pragma unroll
      for (int i_1 = 0; i_1 < 8; ++i_1) {
        if (chunk_idx < num_chunks) {
          *(float2*)(chunk_state + ((((((((((int64_t)((int)blockIdx.y)) * (int64_t)16384) + ((((int64_t)chunk_idx) * ((int64_t)hv_3)) * (int64_t)16384)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)2048)) + ((((int64_t)i_1) & (int64_t)1) * (int64_t)1024)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)128)) + (((int64_t)((int)blockIdx.x)) * (int64_t)32)) + ((((int64_t)i_1) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2))) = *(float2*)(h_lo + (i_1 * 2));
          *(float2*)(chunk_state + (((((((((((int64_t)((int)blockIdx.y)) * (int64_t)16384) + ((((int64_t)chunk_idx) * ((int64_t)hv_3)) * (int64_t)16384)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)2048)) + ((((int64_t)i_1) & (int64_t)1) * (int64_t)1024)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)128)) + (((int64_t)((int)blockIdx.x)) * (int64_t)32)) + ((((int64_t)i_1) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)) + (int64_t)8192)) = *(float2*)(h_hi + (i_1 * 2));
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_2 = 0; i_2 < 4; ++i_2) {
      bfloat16_t broadcast_var_2 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
      uint4 condval_2;
      if (((((((chunk_idx * 64) + (i_2 * 16)) + (((int)threadIdx.x) >> 3)) < seq_len_5) && (((int)blockIdx.y) < hv_7)) && ((((chunk_idx * 64) + (i_2 * 16)) + (((int)threadIdx.x) >> 3)) < seq_len_3))) {
        condval_2 = *(uint4*)(w + (((((int64_t)((int)blockIdx.y)) * (int64_t)128) + (((((((int64_t)chunk_idx) * (int64_t)64) + (((int64_t)i_2) * (int64_t)16)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)3)) * ((int64_t)hv_7)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)));
      } else {
        condval_2 = make_uint4(__pack_nv_bfloat162(broadcast_var_2, broadcast_var_2), __pack_nv_bfloat162(broadcast_var_2, broadcast_var_2), __pack_nv_bfloat162(broadcast_var_2, broadcast_var_2), __pack_nv_bfloat162(broadcast_var_2, broadcast_var_2));
      }
      *(uint4*)(((bfloat16_t*)w_lo) + (((((i_2 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_2;
      bfloat16_t broadcast_var_3 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
      uint4 condval_3;
      if (((((((chunk_idx * 64) + (i_2 * 16)) + (((int)threadIdx.x) >> 3)) < seq_len_5) && (((int)blockIdx.y) < hv_7)) && ((((chunk_idx * 64) + (i_2 * 16)) + (((int)threadIdx.x) >> 3)) < seq_len_3))) {
        condval_3 = *(uint4*)(w + ((((((int64_t)((int)blockIdx.y)) * (int64_t)128) + (((((((int64_t)chunk_idx) * (int64_t)64) + (((int64_t)i_2) * (int64_t)16)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)3)) * ((int64_t)hv_7)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)) + (int64_t)64));
      } else {
        condval_3 = make_uint4(__pack_nv_bfloat162(broadcast_var_3, broadcast_var_3), __pack_nv_bfloat162(broadcast_var_3, broadcast_var_3), __pack_nv_bfloat162(broadcast_var_3, broadcast_var_3), __pack_nv_bfloat162(broadcast_var_3, broadcast_var_3));
      }
      *(uint4*)(((bfloat16_t*)w_hi) + (((((i_2 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_3;
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 2; ++i_3) {
      bfloat16_t broadcast_var_4 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
      uint4 condval_4;
      if (((((((chunk_idx * 64) + (i_3 * 32)) + (((int)threadIdx.x) >> 2)) < seq_len_5) && (((int)blockIdx.y) < hv_6)) && ((((chunk_idx * 64) + (i_3 * 32)) + (((int)threadIdx.x) >> 2)) < seq_len_1))) {
        condval_4 = *(uint4*)(u + ((((((int64_t)((int)blockIdx.y)) * (int64_t)128) + (((((((int64_t)chunk_idx) * (int64_t)64) + (((int64_t)i_3) * (int64_t)32)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)2)) * ((int64_t)hv_6)) * (int64_t)128)) + (((int64_t)((int)blockIdx.x)) * (int64_t)32)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)8)));
      } else {
        condval_4 = make_uint4(__pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4));
      }
      *(uint4*)(((bfloat16_t*)u_tile) + ((i_3 * 1024) + (((int)threadIdx.x) * 8))) = condval_4;
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 4; ++i_4) {
      float broadcast_var_5 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(v_new_tile + (i_4 * 4)) = make_float4(broadcast_var_5, broadcast_var_5, broadcast_var_5, broadcast_var_5);
    }
    __syncthreads();
    #pragma unroll
    for (int i_5 = 0; i_5 < 8; ++i_5) {
      *(uint1*)(u_tile_local_cast + 0) = *(uint1*)(((bfloat16_t*)u_tile) + ((((((((int)threadIdx.x) >> 5) * 512) + ((i_5 & 1) * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + ((i_5 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
      float2 __1;
      uint1 v_ = *(uint1*)(u_tile_local_cast + 0);
      ((float2*)(&__1))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v_))[0]);
      *(float2*)(v_new_tile + (i_5 * 2)) = __1;
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 8; ++i_6) {
      uint1 __2;
      float2 v__1 = *(float2*)(h_lo + (i_6 * 2));
      (reinterpret_cast<__nv_bfloat162*>(&__2))[0] = __float22bfloat162_rn(((float2*)(&v__1))[0]);
      *(uint1*)(((bfloat16_t*)h_lo_sh) + (((((((((int)threadIdx.x) >> 5) * 512) + ((i_6 & 1) * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_6 >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_6 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __2;
      uint1 __3;
      float2 v__2 = *(float2*)(h_hi + (i_6 * 2));
      (reinterpret_cast<__nv_bfloat162*>(&__3))[0] = __float22bfloat162_rn(((float2*)(&v__2))[0]);
      *(uint1*)(((bfloat16_t*)h_hi_sh) + (((((((((int)threadIdx.x) >> 5) * 512) + ((i_6 & 1) * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_6 >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_6 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __3;
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 4; ++i_7) {
      float broadcast_var_6 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(wh_acc + (i_7 * 4)) = make_float4(broadcast_var_6, broadcast_var_6, broadcast_var_6, broadcast_var_6);
    }
    {
      bfloat16_t A_local[8];
      bfloat16_t B_local[16];
      __syncthreads();
      for (int ki = 0; ki < 4; ++ki) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)w_lo)[((((((int)threadIdx.x) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local[0])));
        for (int i_8 = 0; i_8 < 2; ++i_8) {
          tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)h_lo_sh)[((((ki * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + i_8) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(B_local[(i_8 * 8)])));
        }
        for (int j = 0; j < 2; ++j) {
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(wh_acc + (j * 8)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(wh_acc + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
        }
      }
    }
    {
      bfloat16_t A_local_1[8];
      bfloat16_t B_local_1[16];
      for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)w_hi)[((((((int)threadIdx.x) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_1[0])));
        for (int i_9 = 0; i_9 < 2; ++i_9) {
          tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)h_hi_sh)[((((ki_1 * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + i_9) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(B_local_1[(i_9 * 8)])));
        }
        for (int j_1 = 0; j_1 < 2; ++j_1) {
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(wh_acc + (j_1 * 8)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(wh_acc + ((j_1 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
        }
      }
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 16; ++i_10) {
      v_new_tile[i_10] = (v_new_tile[i_10] - wh_acc[i_10]);
    }
    #pragma unroll
    for (int i_11 = 0; i_11 < 8; ++i_11) {
      if (((((chunk_idx * 64) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_11 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < seq_len_5) {
        uint1 __4;
        float2 v__3 = *(float2*)(v_new_tile + (i_11 * 2));
        (reinterpret_cast<__nv_bfloat162*>(&__4))[0] = __float22bfloat162_rn(((float2*)(&v__3))[0]);
        *(uint1*)(v_new_local_cast_1 + 0) = __4;
        if (((int)blockIdx.y) < hv_5) {
          if (((((chunk_idx * 64) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_11 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < seq_len) {
            *(uint1*)(v_new + (((((((int64_t)((int)blockIdx.y)) * (int64_t)128) + ((((((((int64_t)chunk_idx) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)16)) + ((((int64_t)i_11) & (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2)) * ((int64_t)hv_5)) * (int64_t)128)) + (((int64_t)((int)blockIdx.x)) * (int64_t)32)) + ((((int64_t)i_11) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2))) = *(uint1*)(v_new_local_cast_1 + 0);
          }
        }
      }
    }
    int condval_6;
    if ((((chunk_idx * 64) + 64) <= seq_len_5)) {
      condval_6 = ((chunk_idx * 64) + 63);
    } else {
      condval_6 = (seq_len_5 - 1);
    }
    int condval_7;
    if ((((chunk_idx * 64) + 64) <= seq_len_5)) {
      condval_7 = ((chunk_idx * 64) + 63);
    } else {
      condval_7 = (seq_len_5 - 1);
    }
    float condval_5;
    if ((((((int)blockIdx.y) < hv_4) && (0 <= condval_6)) && (condval_7 < seq_len_2))) {
      int64_t condval_8;
      if ((((((int64_t)chunk_idx) * (int64_t)64) + (int64_t)64) <= ((int64_t)seq_len_5))) {
        condval_8 = ((((int64_t)chunk_idx) * (int64_t)64) + (int64_t)63);
      } else {
        condval_8 = (((int64_t)seq_len_5) - (int64_t)1);
      }
      int64_t condval_9;
      if ((((((int64_t)chunk_idx) * (int64_t)64) + (int64_t)64) <= ((int64_t)seq_len_5))) {
        condval_9 = ((((int64_t)chunk_idx) * (int64_t)64) + (int64_t)63);
      } else {
        condval_9 = (((int64_t)seq_len_5) - (int64_t)1);
      }
      condval_5 = g_cumsum[((condval_9 * ((int64_t)hv_4)) + ((int64_t)((int)blockIdx.y)))];
    } else {
      condval_5 = 0x0p+0f/*0.000000e+00*/;
    }
    float g_last = condval_5;
    float decay = expf(g_last);
    #pragma unroll
    for (int i_12 = 0; i_12 < 16; ++i_12) {
      h_lo[i_12] = (h_lo[i_12] * expf(g_last));
      h_hi[i_12] = (h_hi[i_12] * expf(g_last));
    }
    #pragma unroll
    for (int i_13 = 0; i_13 < 2; ++i_13) {
      bool in_range = (((((chunk_idx * 64) + ((((int)threadIdx.x) >> 5) * 16)) + (i_13 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < seq_len_5);
      float condval_10;
      if ((((((chunk_idx * 64) + ((((int)threadIdx.x) >> 5) * 16)) + (i_13 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < seq_len_5)) {
        float condval_11;
        if (((((int)blockIdx.y) < hv_4) && (((((chunk_idx * 64) + ((((int)threadIdx.x) >> 5) * 16)) + (i_13 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < seq_len_2))) {
          condval_11 = g_cumsum[((((((((int64_t)chunk_idx) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)16)) + (((int64_t)i_13) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2)) * ((int64_t)hv_4)) + ((int64_t)((int)blockIdx.y)))];
        } else {
          condval_11 = 0x0p+0f/*0.000000e+00*/;
        }
        condval_10 = condval_11;
      } else {
        condval_10 = g_last;
      }
      g_frag[i_13] = condval_10;
    }
    __syncthreads();
    #pragma unroll
    for (int i_14 = 0; i_14 < 8; ++i_14) {
      bool in_range_1 = (((((chunk_idx * 64) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_14 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < seq_len_5);
      float condval_12;
      if ((((((chunk_idx * 64) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_14 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < seq_len_5)) {
        condval_12 = expf((g_last - g_frag[(i_14 & 1)]));
      } else {
        condval_12 = 0x0p+0f/*0.000000e+00*/;
      }
      float g_v = condval_12;
      float2 __5;
        float2 v__4 = *(float2*)(v_new_tile + (i_14 * 2));
        float2 v__5 = make_float2(g_v, g_v);
        *(float2*)(&(__5.x)) = tl::mul2(*(float2*)(&(v__4.x)), *(float2*)(&(v__5.x)));
      *(float2*)(v_new_tile + (i_14 * 2)) = __5;
      uint1 __6;
      float2 v__6 = *(float2*)(v_new_tile + (i_14 * 2));
      (reinterpret_cast<__nv_bfloat162*>(&__6))[0] = __float22bfloat162_rn(((float2*)(&v__6))[0]);
      *(uint1*)(((bfloat16_t*)v_new_bf) + (((((((((int)threadIdx.x) >> 5) * 512) + ((i_14 & 1) * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_14 >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_14 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __6;
    }
    #pragma unroll
    for (int i_15 = 0; i_15 < 32; ++i_15) {
      bfloat16_t condval_13;
      if ((((((chunk_idx * 64) + (((int)threadIdx.x) & 63)) < seq_len_5) && (((int)blockIdx.y) < hv_2)) && (((chunk_idx * 64) + (((int)threadIdx.x) & 63)) < seq_len_4))) {
        condval_13 = k[((((((int64_t)((int)blockIdx.y)) * (int64_t)128) + ((((((int64_t)chunk_idx) * (int64_t)64) + (((int64_t)((int)threadIdx.x)) & (int64_t)63)) * ((int64_t)hv_2)) * (int64_t)128)) + (((int64_t)i_15) * (int64_t)2)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)6))];
      } else {
        condval_13 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
      }
      ((bfloat16_t*)k_lo)[((((((i_15 * 128) + ((((int)threadIdx.x) >> 6) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((i_15 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_15 & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 8)) + (((int)threadIdx.x) & 7))] = condval_13;
      bfloat16_t condval_14;
      if ((((((chunk_idx * 64) + (((int)threadIdx.x) & 63)) < seq_len_5) && (((int)blockIdx.y) < hv_2)) && (((chunk_idx * 64) + (((int)threadIdx.x) & 63)) < seq_len_4))) {
        condval_14 = k[(((((((int64_t)((int)blockIdx.y)) * (int64_t)128) + ((((((int64_t)chunk_idx) * (int64_t)64) + (((int64_t)((int)threadIdx.x)) & (int64_t)63)) * ((int64_t)hv_2)) * (int64_t)128)) + (((int64_t)i_15) * (int64_t)2)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)6)) + (int64_t)64)];
      } else {
        condval_14 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
      }
      ((bfloat16_t*)k_hi)[((((((i_15 * 128) + ((((int)threadIdx.x) >> 6) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((i_15 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_15 & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 8)) + (((int)threadIdx.x) & 7))] = condval_14;
    }
    #pragma unroll
    for (int i_16 = 0; i_16 < 4; ++i_16) {
      float broadcast_var_7 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(kh_lo_acc + (i_16 * 4)) = make_float4(broadcast_var_7, broadcast_var_7, broadcast_var_7, broadcast_var_7);
    }
    #pragma unroll
    for (int i_17 = 0; i_17 < 4; ++i_17) {
      float broadcast_var_8 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(kh_hi_acc + (i_17 * 4)) = make_float4(broadcast_var_8, broadcast_var_8, broadcast_var_8, broadcast_var_8);
    }
    {
      bfloat16_t A_local_2[8];
      bfloat16_t B_local_2[16];
      __syncthreads();
      for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)k_lo)[((((((int)threadIdx.x) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_2 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_2[0])));
        for (int i_18 = 0; i_18 < 2; ++i_18) {
          tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)v_new_bf)[((((ki_2 * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + i_18) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(B_local_2[(i_18 * 8)])));
        }
        for (int j_2 = 0; j_2 < 2; ++j_2) {
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(kh_lo_acc + (j_2 * 8)), reinterpret_cast<const unsigned*>(A_local_2 + 0), reinterpret_cast<const unsigned*>(B_local_2 + (j_2 * 8)));
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(kh_lo_acc + ((j_2 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_2 + 0), reinterpret_cast<const unsigned*>(B_local_2 + ((j_2 * 8) + 4)));
        }
      }
    }
    {
      bfloat16_t A_local_3[8];
      bfloat16_t B_local_3[16];
      for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)k_hi)[((((((int)threadIdx.x) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_3 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_3 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_3[0])));
        for (int i_19 = 0; i_19 < 2; ++i_19) {
          tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)v_new_bf)[((((ki_3 * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + i_19) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(B_local_3[(i_19 * 8)])));
        }
        for (int j_3 = 0; j_3 < 2; ++j_3) {
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(kh_hi_acc + (j_3 * 8)), reinterpret_cast<const unsigned*>(A_local_3 + 0), reinterpret_cast<const unsigned*>(B_local_3 + (j_3 * 8)));
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(kh_hi_acc + ((j_3 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_3 + 0), reinterpret_cast<const unsigned*>(B_local_3 + ((j_3 * 8) + 4)));
        }
      }
    }
    #pragma unroll
    for (int i_20 = 0; i_20 < 16; ++i_20) {
      h_lo[i_20] = (h_lo[i_20] + kh_lo_acc[i_20]);
      h_hi[i_20] = (h_hi[i_20] + kh_hi_acc[i_20]);
    }
  }
  if (((int)blockIdx.y) < hv_1) {
    #pragma unroll
    for (int i_21 = 0; i_21 < 8; ++i_21) {
      *(float2*)(final_state + (((((((((int64_t)((int)blockIdx.y)) * (int64_t)16384) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)2048)) + ((((int64_t)i_21) & (int64_t)1) * (int64_t)1024)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)128)) + (((int64_t)((int)blockIdx.x)) * (int64_t)32)) + ((((int64_t)i_21) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2))) = *(float2*)(h_lo + (i_21 * 2));
      *(float2*)(final_state + ((((((((((int64_t)((int)blockIdx.y)) * (int64_t)16384) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)2048)) + ((((int64_t)i_21) & (int64_t)1) * (int64_t)1024)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)128)) + (((int64_t)((int)blockIdx.x)) * (int64_t)32)) + ((((int64_t)i_21) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)) + (int64_t)8192)) = *(float2*)(h_hi + (i_21 * 2));
    }
  }
}

