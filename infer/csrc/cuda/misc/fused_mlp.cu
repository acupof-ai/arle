#include "common.cuh"

// ============================================================================
// Phase 1: Gate + Up projection (interleaved) → SiLU activation
// Computes act[i] = silu(gate_proj[i] @ x) * (up_proj[i] @ x)
// BF16×4 vectorized loads, warp shuffle reduction.
// Gate and up share the same x vector — read x once per pass.
// ============================================================================
#define FUSED_MLP_TILE 256
#define FUSED_MLP_INTER_PER_BLOCK 4
#define MLP_NUM_WARPS (FUSED_MLP_TILE / WARP_SIZE)

__global__ void fused_mlp_intermediate_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ gate_proj,
    const __nv_bfloat16 *__restrict__ up_proj,
    __nv_bfloat16 *__restrict__ act,
    int intermediate_size,
    int hidden_size) {

  int inter_base = blockIdx.x * FUSED_MLP_INTER_PER_BLOCK;
  int tid = threadIdx.x;
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;

  int K4 = hidden_size / 4;
  int K_tail = hidden_size - K4 * 4;

  const uint2 *x_vec = reinterpret_cast<const uint2 *>(x);

  // Interleaved gate + up: compute both dot products in one pass over x
  float gate_sums[FUSED_MLP_INTER_PER_BLOCK];
  float up_sums[FUSED_MLP_INTER_PER_BLOCK];

  #pragma unroll
  for (int r = 0; r < FUSED_MLP_INTER_PER_BLOCK; r++) {
    gate_sums[r] = 0.0f;
    up_sums[r] = 0.0f;
  }

  #pragma unroll
  for (int r = 0; r < FUSED_MLP_INTER_PER_BLOCK; r++) {
    int inter_idx = inter_base + r;
    if (inter_idx >= intermediate_size) break;

    const uint2 *gate_row_vec = reinterpret_cast<const uint2 *>(gate_proj + inter_idx * hidden_size);
    const uint2 *up_row_vec = reinterpret_cast<const uint2 *>(up_proj + inter_idx * hidden_size);
    float g_sum = 0.0f;
    float u_sum = 0.0f;

    for (int k4 = tid; k4 < K4; k4 += FUSED_MLP_TILE) {
      uint2 x_val = x_vec[k4];
      uint2 g_val = gate_row_vec[k4];
      uint2 u_val = up_row_vec[k4];

      __nv_bfloat162 x_lo = *reinterpret_cast<__nv_bfloat162 *>(&x_val.x);
      __nv_bfloat162 x_hi = *reinterpret_cast<__nv_bfloat162 *>(&x_val.y);
      __nv_bfloat162 g_lo = *reinterpret_cast<__nv_bfloat162 *>(&g_val.x);
      __nv_bfloat162 g_hi = *reinterpret_cast<__nv_bfloat162 *>(&g_val.y);
      __nv_bfloat162 u_lo = *reinterpret_cast<__nv_bfloat162 *>(&u_val.x);
      __nv_bfloat162 u_hi = *reinterpret_cast<__nv_bfloat162 *>(&u_val.y);

      g_sum += __bfloat162float(g_lo.x) * __bfloat162float(x_lo.x);
      g_sum += __bfloat162float(g_lo.y) * __bfloat162float(x_lo.y);
      g_sum += __bfloat162float(g_hi.x) * __bfloat162float(x_hi.x);
      g_sum += __bfloat162float(g_hi.y) * __bfloat162float(x_hi.y);

      u_sum += __bfloat162float(u_lo.x) * __bfloat162float(x_lo.x);
      u_sum += __bfloat162float(u_lo.y) * __bfloat162float(x_lo.y);
      u_sum += __bfloat162float(u_hi.x) * __bfloat162float(x_hi.x);
      u_sum += __bfloat162float(u_hi.y) * __bfloat162float(x_hi.y);
    }

    // Scalar tail
    if (K_tail > 0) {
      const __nv_bfloat16 *gate_row = gate_proj + inter_idx * hidden_size;
      const __nv_bfloat16 *up_row = up_proj + inter_idx * hidden_size;
      int k_start = K4 * 4;
      for (int k = k_start + tid; k < hidden_size; k += FUSED_MLP_TILE) {
        float xv = __bfloat162float(x[k]);
        g_sum += __bfloat162float(gate_row[k]) * xv;
        u_sum += __bfloat162float(up_row[k]) * xv;
      }
    }

    gate_sums[r] = g_sum;
    up_sums[r] = u_sum;
  }

  // Warp-level reduction
  #pragma unroll
  for (int r = 0; r < FUSED_MLP_INTER_PER_BLOCK; r++) {
    gate_sums[r] = warp_reduce_sum(gate_sums[r]);
    up_sums[r] = warp_reduce_sum(up_sums[r]);
  }

  // Inter-warp reduction via shared memory
  __shared__ float warp_gate[FUSED_MLP_INTER_PER_BLOCK][MLP_NUM_WARPS];
  __shared__ float warp_up[FUSED_MLP_INTER_PER_BLOCK][MLP_NUM_WARPS];

  if (lane_id == 0) {
    #pragma unroll
    for (int r = 0; r < FUSED_MLP_INTER_PER_BLOCK; r++) {
      warp_gate[r][warp_id] = gate_sums[r];
      warp_up[r][warp_id] = up_sums[r];
    }
  }
  __syncthreads();

  // First warp reduces across all warps and writes activation
  if (warp_id == 0) {
    #pragma unroll
    for (int r = 0; r < FUSED_MLP_INTER_PER_BLOCK; r++) {
      float g = (lane_id < MLP_NUM_WARPS) ? warp_gate[r][lane_id] : 0.0f;
      float u = (lane_id < MLP_NUM_WARPS) ? warp_up[r][lane_id] : 0.0f;
      g = warp_reduce_sum(g);
      u = warp_reduce_sum(u);

      if (lane_id == 0) {
        int inter_idx = inter_base + r;
        if (inter_idx < intermediate_size) {
          // Match HF: GEMV outputs are bf16, silu output is bf16, then bf16 × bf16
          __nv_bfloat16 gate_bf16 = __float2bfloat16(g);
          float gf = __bfloat162float(gate_bf16);
          float silu_g = gf / (1.0f + expf(-gf));
          __nv_bfloat16 silu_bf16 = __float2bfloat16(silu_g);
          __nv_bfloat16 up_bf16 = __float2bfloat16(u);
          float result = __bfloat162float(silu_bf16) * __bfloat162float(up_bf16);
          act[inter_idx] = __float2bfloat16(result);
        }
      }
    }
  }
}

// ============================================================================
// Phase 2: Down projection — out = down_proj @ act
// Register accumulation across all K (single final reduction).
// BF16×4 vectorized loads, warp shuffle reduction.
// OUT_PER_BLOCK=8: each block processes 8 output rows.
// ============================================================================
#define FUSED_MLP_OUT_PER_BLOCK 8

__global__ void fused_mlp_output_kernel(
    const __nv_bfloat16 *__restrict__ act,
    const __nv_bfloat16 *__restrict__ down_proj,
    __nv_bfloat16 *__restrict__ out,
    int hidden_size,
    int intermediate_size) {

  int out_base = blockIdx.x * FUSED_MLP_OUT_PER_BLOCK;
  int tid = threadIdx.x;
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;

  int K4 = intermediate_size / 4;
  int K_tail = intermediate_size - K4 * 4;

  const uint2 *act_vec = reinterpret_cast<const uint2 *>(act);

  // Register accumulation: each thread accumulates across all K
  float acc[FUSED_MLP_OUT_PER_BLOCK];
  #pragma unroll
  for (int r = 0; r < FUSED_MLP_OUT_PER_BLOCK; r++) acc[r] = 0.0f;

  #pragma unroll
  for (int r = 0; r < FUSED_MLP_OUT_PER_BLOCK; r++) {
    int row = out_base + r;
    if (row >= hidden_size) break;

    const uint2 *dp_row_vec = reinterpret_cast<const uint2 *>(down_proj + row * intermediate_size);
    float sum = 0.0f;

    for (int k4 = tid; k4 < K4; k4 += FUSED_MLP_TILE) {
      uint2 a_val = act_vec[k4];
      uint2 d_val = dp_row_vec[k4];

      __nv_bfloat162 a_lo = *reinterpret_cast<__nv_bfloat162 *>(&a_val.x);
      __nv_bfloat162 a_hi = *reinterpret_cast<__nv_bfloat162 *>(&a_val.y);
      __nv_bfloat162 d_lo = *reinterpret_cast<__nv_bfloat162 *>(&d_val.x);
      __nv_bfloat162 d_hi = *reinterpret_cast<__nv_bfloat162 *>(&d_val.y);

      sum += __bfloat162float(d_lo.x) * __bfloat162float(a_lo.x);
      sum += __bfloat162float(d_lo.y) * __bfloat162float(a_lo.y);
      sum += __bfloat162float(d_hi.x) * __bfloat162float(a_hi.x);
      sum += __bfloat162float(d_hi.y) * __bfloat162float(a_hi.y);
    }

    // Scalar tail
    if (K_tail > 0) {
      const __nv_bfloat16 *dp_row = down_proj + row * intermediate_size;
      int k_start = K4 * 4;
      for (int k = k_start + tid; k < intermediate_size; k += FUSED_MLP_TILE) {
        sum += __bfloat162float(dp_row[k]) * __bfloat162float(act[k]);
      }
    }

    acc[r] = sum;
  }

  // Warp-level reduction
  #pragma unroll
  for (int r = 0; r < FUSED_MLP_OUT_PER_BLOCK; r++) {
    acc[r] = warp_reduce_sum(acc[r]);
  }

  // Inter-warp reduction via shared memory
  __shared__ float warp_sums[FUSED_MLP_OUT_PER_BLOCK][MLP_NUM_WARPS];

  if (lane_id == 0) {
    #pragma unroll
    for (int r = 0; r < FUSED_MLP_OUT_PER_BLOCK; r++) {
      warp_sums[r][warp_id] = acc[r];
    }
  }
  __syncthreads();

  // First warp reduces across all warps and writes output
  if (warp_id == 0) {
    #pragma unroll
    for (int r = 0; r < FUSED_MLP_OUT_PER_BLOCK; r++) {
      float val = (lane_id < MLP_NUM_WARPS) ? warp_sums[r][lane_id] : 0.0f;
      val = warp_reduce_sum(val);
      if (lane_id == 0) {
        int row = out_base + r;
        if (row < hidden_size) {
          out[row] = __float2bfloat16(val);
        }
      }
    }
  }
}

extern "C" {
cudaError_t fused_mlp_cuda(const __nv_bfloat16 *x, const __nv_bfloat16 *gate_proj, const __nv_bfloat16 *up_proj,
                    const __nv_bfloat16 *down_proj, __nv_bfloat16 *act, __nv_bfloat16 *out,
                    int hidden_size, int intermediate_size, cudaStream_t stream) {
  // Phase 1: Compute gate, up, and activation
  int inter_blocks = (intermediate_size + FUSED_MLP_INTER_PER_BLOCK - 1) / FUSED_MLP_INTER_PER_BLOCK;
  fused_mlp_intermediate_kernel<<<inter_blocks, FUSED_MLP_TILE, 0, stream>>>(
      x, gate_proj, up_proj, act, intermediate_size, hidden_size);

  // Phase 2: Down projection
  int out_blocks = (hidden_size + FUSED_MLP_OUT_PER_BLOCK - 1) / FUSED_MLP_OUT_PER_BLOCK;
  fused_mlp_output_kernel<<<out_blocks, FUSED_MLP_TILE, 0, stream>>>(
      act, down_proj, out, hidden_size, intermediate_size);
    return cudaGetLastError();
}
}
