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

extern "C" __global__ void kernel_kernel(const float* __restrict__ a_log, const bfloat16_t* __restrict__ a_proj, const bfloat16_t* __restrict__ b_proj, float* __restrict__ beta_out, const bfloat16_t* __restrict__ dt_bias, float* __restrict__ g_out, bfloat16_t* __restrict__ k_out, bfloat16_t* __restrict__ q_out, const bfloat16_t* __restrict__ qkv, bfloat16_t* __restrict__ v_out, int hv, int hv_1, int hv_2, int hv_3, int hv_4, int hv_5, int hv_6, int hv_7, int hv_8, int num_key_heads, int num_value_heads, int qkv_dim, int seq_len, int seq_len_1, int seq_len_2, int seq_len_3, int seq_len_4, int seq_len_5, int seq_len_6, int seq_len_7, int seq_len_8);
extern "C" __global__ void __launch_bounds__(128, 1) kernel_kernel(const float* __restrict__ a_log, const bfloat16_t* __restrict__ a_proj, const bfloat16_t* __restrict__ b_proj, float* __restrict__ beta_out, const bfloat16_t* __restrict__ dt_bias, float* __restrict__ g_out, bfloat16_t* __restrict__ k_out, bfloat16_t* __restrict__ q_out, const bfloat16_t* __restrict__ qkv, bfloat16_t* __restrict__ v_out, int hv, int hv_1, int hv_2, int hv_3, int hv_4, int hv_5, int hv_6, int hv_7, int hv_8, int num_key_heads, int num_value_heads, int qkv_dim, int seq_len, int seq_len_1, int seq_len_2, int seq_len_3, int seq_len_4, int seq_len_5, int seq_len_6, int seq_len_7, int seq_len_8) {
  float q_frag[128];
  float k_frag[128];
  bfloat16_t v_frag[1];
  float qq_sum[1];
  float kk_sum[1];
  #pragma unroll
  for (int i = 0; i < 128; ++i) {
    int rmod = ((((int)blockIdx.y) * num_key_heads) % num_value_heads);
    int rdiv = ((((int)blockIdx.y) * num_key_heads) / num_value_heads);
    int rmod_1 = ((((int)blockIdx.y) * num_key_heads) % num_value_heads);
    int rdiv_1 = ((((int)blockIdx.y) * num_key_heads) / num_value_heads);
    bfloat16_t condval;
    if ((((0 <= ((((0 <= num_value_heads) && (0 <= rmod)) || ((num_value_heads < 0) && (rmod <= 0))) ? rdiv : (rdiv - 1))) && (((((((0 <= num_value_heads) && (0 <= rmod_1)) || ((num_value_heads < 0) && (rmod_1 <= 0))) ? rdiv_1 : (rdiv_1 - 1)) * 128) + i) < qkv_dim)) && (((int)blockIdx.x) < seq_len_6))) {
      int64_t rmod_2 = ((((int64_t)((int)blockIdx.y)) * ((int64_t)num_key_heads)) % ((int64_t)num_value_heads));
      int64_t rdiv_2 = ((((int64_t)((int)blockIdx.y)) * ((int64_t)num_key_heads)) / ((int64_t)num_value_heads));
      int64_t rmod_3 = ((((int64_t)((int)blockIdx.y)) * ((int64_t)num_key_heads)) % ((int64_t)num_value_heads));
      int64_t rdiv_3 = ((((int64_t)((int)blockIdx.y)) * ((int64_t)num_key_heads)) / ((int64_t)num_value_heads));
      condval = qkv[((((((((int64_t)0 <= ((int64_t)num_value_heads)) && ((int64_t)0 <= rmod_3)) || ((((int64_t)num_value_heads) < (int64_t)0) && (rmod_3 <= (int64_t)0))) ? rdiv_3 : (rdiv_3 - (int64_t)1)) * (int64_t)128) + (((int64_t)((int)blockIdx.x)) * ((int64_t)qkv_dim))) + ((int64_t)i))];
    } else {
      condval = bfloat16_t(0x0p+0f/*0.000000e+00*/);
    }
    q_frag[i] = ((float)condval);
    int rmod_4 = ((((int)blockIdx.y) * num_key_heads) % num_value_heads);
    int rdiv_4 = ((((int)blockIdx.y) * num_key_heads) / num_value_heads);
    int rmod_5 = ((((int)blockIdx.y) * num_key_heads) % num_value_heads);
    int rdiv_5 = ((((int)blockIdx.y) * num_key_heads) / num_value_heads);
    bfloat16_t condval_1;
    if ((((0 <= (((((0 <= num_value_heads) && (0 <= rmod_4)) || ((num_value_heads < 0) && (rmod_4 <= 0))) ? rdiv_4 : (rdiv_4 - 1)) + num_key_heads)) && ((((((((0 <= num_value_heads) && (0 <= rmod_5)) || ((num_value_heads < 0) && (rmod_5 <= 0))) ? rdiv_5 : (rdiv_5 - 1)) * 128) + (num_key_heads * 128)) + i) < qkv_dim)) && (((int)blockIdx.x) < seq_len_6))) {
      int64_t rmod_6 = ((((int64_t)((int)blockIdx.y)) * ((int64_t)num_key_heads)) % ((int64_t)num_value_heads));
      int64_t rdiv_6 = ((((int64_t)((int)blockIdx.y)) * ((int64_t)num_key_heads)) / ((int64_t)num_value_heads));
      int64_t rmod_7 = ((((int64_t)((int)blockIdx.y)) * ((int64_t)num_key_heads)) % ((int64_t)num_value_heads));
      int64_t rdiv_7 = ((((int64_t)((int)blockIdx.y)) * ((int64_t)num_key_heads)) / ((int64_t)num_value_heads));
      condval_1 = qkv[(((((((((int64_t)0 <= ((int64_t)num_value_heads)) && ((int64_t)0 <= rmod_7)) || ((((int64_t)num_value_heads) < (int64_t)0) && (rmod_7 <= (int64_t)0))) ? rdiv_7 : (rdiv_7 - (int64_t)1)) * (int64_t)128) + (((int64_t)num_key_heads) * (int64_t)128)) + (((int64_t)((int)blockIdx.x)) * ((int64_t)qkv_dim))) + ((int64_t)i))];
    } else {
      condval_1 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
    }
    k_frag[i] = ((float)condval_1);
  }
  bfloat16_t condval_2;
  if ((((0 <= ((num_key_heads * 2) + ((int)blockIdx.y))) && ((((num_key_heads * 256) + (((int)blockIdx.y) * 128)) + ((int)threadIdx.x)) < qkv_dim)) && (((int)blockIdx.x) < seq_len_6))) {
    condval_2 = qkv[((((((int64_t)num_key_heads) * (int64_t)256) + (((int64_t)((int)blockIdx.y)) * (int64_t)128)) + (((int64_t)((int)blockIdx.x)) * ((int64_t)qkv_dim))) + ((int64_t)((int)threadIdx.x)))];
  } else {
    condval_2 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
  }
  v_frag[0] = condval_2;
  qq_sum[0] = 0x0p+0f/*0.000000e+00*/;
  kk_sum[0] = 0x0p+0f/*0.000000e+00*/;
  for (int d = 0; d < 128; ++d) {
    qq_sum[0] = (qq_sum[0] + (q_frag[d] * q_frag[d]));
    kk_sum[0] = (kk_sum[0] + (k_frag[d] * k_frag[d]));
  }
  float q_scale = rsqrtf((qq_sum[0] + 0x1.19799812dea11p-40f/*1.000000e-12*/));
  float k_scale = rsqrtf((kk_sum[0] + 0x1.19799812dea11p-40f/*1.000000e-12*/));
  if (((int)blockIdx.y) < hv_8) {
    if (((int)blockIdx.x) < seq_len_5) {
      q_out[(((((int64_t)((int)blockIdx.y)) * (int64_t)128) + ((((int64_t)((int)blockIdx.x)) * ((int64_t)hv_8)) * (int64_t)128)) + ((int64_t)((int)threadIdx.x)))] = ((bfloat16_t)(q_frag[((int)threadIdx.x)] * q_scale));
    }
  }
  if (((int)blockIdx.y) < hv_7) {
    if (((int)blockIdx.x) < seq_len_4) {
      k_out[(((((int64_t)((int)blockIdx.y)) * (int64_t)128) + ((((int64_t)((int)blockIdx.x)) * ((int64_t)hv_7)) * (int64_t)128)) + ((int64_t)((int)threadIdx.x)))] = ((bfloat16_t)(k_frag[((int)threadIdx.x)] * k_scale));
    }
  }
  if (((int)blockIdx.y) < hv_6) {
    if (((int)blockIdx.x) < seq_len_2) {
      v_out[(((((int64_t)((int)blockIdx.y)) * (int64_t)128) + ((((int64_t)((int)blockIdx.x)) * ((int64_t)hv_6)) * (int64_t)128)) + ((int64_t)((int)threadIdx.x)))] = v_frag[0];
    }
  }
  bfloat16_t condval_3;
  if (((((int)blockIdx.y) < hv) && (((int)blockIdx.x) < seq_len))) {
    condval_3 = a_proj[((((int64_t)((int)blockIdx.x)) * ((int64_t)hv)) + ((int64_t)((int)blockIdx.y)))];
  } else {
    condval_3 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
  }
  float a_val = ((float)condval_3);
  bfloat16_t condval_4;
  if (((((int)blockIdx.y) < hv_5) && (((int)blockIdx.x) < seq_len_3))) {
    condval_4 = b_proj[((((int64_t)((int)blockIdx.x)) * ((int64_t)hv_5)) + ((int64_t)((int)blockIdx.y)))];
  } else {
    condval_4 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
  }
  float b_val = ((float)condval_4);
  bfloat16_t condval_5;
  if ((((int)blockIdx.y) < hv_4)) {
    condval_5 = dt_bias[((int)blockIdx.y)];
  } else {
    condval_5 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
  }
  float dt = ((float)condval_5);
  float condval_6;
  if ((((int)blockIdx.y) < hv_3)) {
    condval_6 = a_log[((int)blockIdx.y)];
  } else {
    condval_6 = 0x0p+0f/*0.000000e+00*/;
  }
  float al = condval_6;
  float x = (a_val + dt);
  float condval_7;
  if ((0x1.4p+4f/*2.000000e+01*/ < (a_val + dt))) {
    condval_7 = (a_val + dt);
  } else {
    condval_7 = logf((0x1p+0f/*1.000000e+00*/ + expf((a_val + dt))));
  }
  float softplus_x = condval_7;
  if (((int)blockIdx.y) < hv_2) {
    if (((int)blockIdx.x) < seq_len_7) {
      float condval_8;
      if ((0x1.4p+4f/*2.000000e+01*/ < (a_val + dt))) {
        condval_8 = (a_val + dt);
      } else {
        condval_8 = logf((0x1p+0f/*1.000000e+00*/ + expf((a_val + dt))));
      }
      g_out[((((int64_t)((int)blockIdx.x)) * ((int64_t)hv_2)) + ((int64_t)((int)blockIdx.y)))] = ((expf(al) * -0x1p+0f/*-1.000000e+00*/) * condval_8);
    }
  }
  if (((int)blockIdx.y) < hv_1) {
    if (((int)blockIdx.x) < seq_len_8) {
      beta_out[((((int64_t)((int)blockIdx.x)) * ((int64_t)hv_1)) + ((int64_t)((int)blockIdx.y)))] = (0x1p+0f/*1.000000e+00*/ / (0x1p+0f/*1.000000e+00*/ + expf((b_val * -0x1p+0f/*-1.000000e+00*/))));
    }
  }
}

