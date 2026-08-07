// Hand-written Flash-Attention-2 forward kernel for sm_70 (V100), hdim 128/256.
//
// sm_70 has no FA3 (CUTLASS-3.x needs sm_80+) and no BF16 compute — BF16 I/O is
// cast to FP16 on load, math runs in half2, accumulates in FP32. FA2 tiled
// online softmax (Br=16 Q tokens, Bc=64 KV tiles) shares K/V reads across the Q
// tile. Causal chunked-prefill: Q token at absolute position q attends KV 0..q.
// Interface matches nonpaged_prefill_attention_cuda (drop-in on sm_70).
//
// Shared memory layout (dynamic, 68 KB for hdim=256):
//   k_s [Bc * head_dim] half  = 32 KB
//   v_s [Bc * head_dim] half  = 32 KB
//   qk_s [Br * Bc]      float =  4 KB

#include "common.cuh"
#include <cuda_fp16.h>
#include <cstdint>

#define FA2_BR 16       // Q tokens per block
#define FA2_BC 64       // KV tokens per inner tile
#define FA2_BLOCK 512   // threads = Br * (head_dim / 8) for hdim=256
#define FA2_MAX_HD 256  // max head_dim

__global__ void arle_fa2_sm70_kernel(
    const __nv_bfloat16 *__restrict__ q,
    const __nv_bfloat16 *__restrict__ k_cache,
    const __nv_bfloat16 *__restrict__ v_cache,
    __nv_bfloat16 *__restrict__ out,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int seq_len,
    int kv_len,
    int max_seq_len,
    float sm_scale) {

  extern __shared__ half smem[];
  half *k_s = smem;
  half *v_s = smem + FA2_BC * head_dim;
  float *qk_s = reinterpret_cast<float *>(v_s + FA2_BC * head_dim);

  const int q_head = blockIdx.x;
  const int q_tile = blockIdx.y;
  const int warp_id = threadIdx.x / WARP_SIZE;   // 0..15 → Q token within tile
  const int lane = threadIdx.x % WARP_SIZE;      // 0..31 → dim group

  const int q_start = kv_len - seq_len;          // absolute pos of Q token 0
  const int q_pos = q_start + q_tile * FA2_BR + warp_id;

  if (q_head >= num_q_heads || q_pos >= kv_len) return;

  const int gqa_ratio = num_q_heads / num_kv_heads;
  const int kv_head = q_head / gqa_ratio;
  const int q_dim = num_q_heads * head_dim;

  // Each lane handles 8 dims (lane*8 .. lane*8+7); head_dim is a multiple of 8.
  const int d_start = lane * 8;
  const bool dim_active = d_start + 7 < head_dim;

  // Load Q (BF16 -> FP16, 8 dims = 4 half2).
  half2 q_reg[4];
  if (dim_active) {
    const __nv_bfloat16 *qp =
        q + (q_pos - q_start) * q_dim + q_head * head_dim + d_start;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      q_reg[i] = __floats2half2_rn(
          __bfloat162float(qp[i * 2]), __bfloat162float(qp[i * 2 + 1]));
    }
  }

  float o_reg[8] = {0};
  float running_max = -INFINITY;
  float running_sum = 0.0f;

  // Largest absolute KV position any Q token in this tile attends to (causal).
  const int tile_q_max = min(q_start + (q_tile + 1) * FA2_BR, kv_len) - 1;

  const int kv_row_bytes = head_dim;  // head-major cache: [h_k, max_seq, d]
  const __nv_bfloat16 *k_base = k_cache + kv_head * max_seq_len * head_dim;
  const __nv_bfloat16 *v_base = v_cache + kv_head * max_seq_len * head_dim;

  for (int k_start = 0; k_start <= tile_q_max; k_start += FA2_BC) {
    // Cooperative load K/V tile (BF16 -> FP16); vectorized via uint4 (8 BF16 = 16B).
    const int elems = FA2_BC * head_dim;
    const int vec_elems = elems / 8;  // 8 BF16 per uint4
    for (int i = threadIdx.x; i < vec_elems; i += FA2_BLOCK) {
      const int base = i * 8;
      const int row = base / head_dim;
      const int col = base % head_dim;
      const int k_abs = k_start + row;
      const bool valid = k_abs < kv_len;

      // Load 8 BF16 values (16 bytes) as uint4.
      uint4 kv = valid
          ? *reinterpret_cast<const uint4 *>(k_base + k_abs * kv_row_bytes + col)
          : make_uint4(0, 0, 0, 0);
      uint4 vv = valid
          ? *reinterpret_cast<const uint4 *>(v_base + k_abs * kv_row_bytes + col)
          : make_uint4(0, 0, 0, 0);

      // Convert 8 BF16 -> 8 FP16 and store to shared memory.
      __nv_bfloat16 kbf[8], vbf[8];
      *reinterpret_cast<uint4 *>(kbf) = kv;
      *reinterpret_cast<uint4 *>(vbf) = vv;
      half *k_dst = k_s + base;
      half *v_dst = v_s + base;
      #pragma unroll
      for (int j = 0; j < 8; j++) {
        k_dst[j] = __float2half(__bfloat162float(kbf[j]));
        v_dst[j] = __float2half(__bfloat162float(vbf[j]));
      }
    }
    __syncthreads();

    // QK^T: each warp computes its Q token vs the Bc KV positions.
    #pragma unroll
    for (int k_local = 0; k_local < FA2_BC; k_local++) {
      const int k_abs = k_start + k_local;
      float partial = 0.0f;
      if (dim_active && k_abs <= q_pos) {
        #pragma unroll
        for (int d = 0; d < 4; d++) {
          const half2 k2 = *reinterpret_cast<const half2 *>(
              &k_s[k_local * head_dim + d_start + d * 2]);
          const half2 prod = __hmul2(q_reg[d], k2);
          partial += __low2float(prod);
          partial += __high2float(prod);
        }
        partial *= sm_scale;
      } else if (dim_active) {
        partial = -INFINITY;  // causal mask: k > q_pos
      }
      partial = warp_reduce_sum(partial);
      if (dim_active && lane == 0) {
        qk_s[warp_id * FA2_BC + k_local] = partial;
      }
    }
    __syncthreads();

    // Online softmax + PV (each warp reads its own qk row).
    if (dim_active) {
      float tile_max = -INFINITY;
      #pragma unroll
      for (int k_local = 0; k_local < FA2_BC; k_local++) {
        tile_max = fmaxf(tile_max, qk_s[warp_id * FA2_BC + k_local]);
      }

      const float new_max = fmaxf(running_max, tile_max);
      const float rescale = expf(running_max - new_max);
      running_sum *= rescale;
      #pragma unroll
      for (int d = 0; d < 8; d++) {
        o_reg[d] *= rescale;
      }
      running_max = new_max;

      float row_sum = 0.0f;
      #pragma unroll
      for (int k_local = 0; k_local < FA2_BC; k_local++) {
        const float p = expf(qk_s[warp_id * FA2_BC + k_local] - running_max);
        row_sum += p;
        #pragma unroll
        for (int d = 0; d < 4; d++) {
          const half2 v2 = *reinterpret_cast<const half2 *>(
              &v_s[k_local * head_dim + d_start + d * 2]);
          o_reg[2 * d] += p * __low2float(v2);
          o_reg[2 * d + 1] += p * __high2float(v2);
        }
      }
      running_sum += row_sum;
    }
    __syncthreads();
  }

  // Store output (FP32 -> BF16), normalized by running_sum.
  if (dim_active) {
    const float denom = running_sum > 0.0f ? running_sum : 1.0f;
    __nv_bfloat16 *op =
        out + (q_pos - q_start) * q_dim + q_head * head_dim + d_start;
    #pragma unroll
    for (int d = 0; d < 4; d++) {
      *reinterpret_cast<__nv_bfloat162 *>(op + d * 2) = __floats2bfloat162_rn(
          o_reg[2 * d] / denom, o_reg[2 * d + 1] / denom);
    }
  }
}

extern "C" cudaError_t arle_fa2_sm70_attention_cuda(
    const uint16_t *q,
    const uint16_t *k_cache,
    const uint16_t *v_cache,
    uint16_t *out,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int seq_len,
    int kv_len,
    int max_seq_len,
    float sm_scale,
    cudaStream_t stream) {
  if (num_q_heads <= 0 || num_kv_heads <= 0 || seq_len < 0 || kv_len < seq_len ||
      max_seq_len < kv_len || (head_dim != 128 && head_dim != 256) ||
      num_q_heads % num_kv_heads != 0) {
    return cudaErrorInvalidValue;
  }
  if (seq_len == 0) {
    return cudaSuccess;
  }

  // Dynamic shared memory: K (Bc*hd*2) + V (Bc*hd*2) + QK (Br*Bc*4).
  const size_t shm_bytes =
      (size_t)FA2_BC * head_dim * 2 * 2 + (size_t)FA2_BR * FA2_BC * 4;

  // V100 supports up to 96 KB shared memory per SM; allow the kernel to use it.
  cudaFuncSetAttribute(arle_fa2_sm70_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, shm_bytes);

  const int num_q_tiles = (seq_len + FA2_BR - 1) / FA2_BR;
  dim3 grid(num_q_heads, num_q_tiles);
  dim3 block(FA2_BLOCK);
  arle_fa2_sm70_kernel<<<grid, block, shm_bytes, stream>>>(
      reinterpret_cast<const __nv_bfloat16 *>(q),
      reinterpret_cast<const __nv_bfloat16 *>(k_cache),
      reinterpret_cast<const __nv_bfloat16 *>(v_cache),
      reinterpret_cast<__nv_bfloat16 *>(out),
      num_q_heads, num_kv_heads, head_dim, seq_len, kv_len, max_seq_len,
      sm_scale);
  return cudaGetLastError();
}
