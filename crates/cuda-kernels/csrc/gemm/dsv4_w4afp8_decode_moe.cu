// DSv4 W4AFP8 decode-band grouped MoE kernels (w4a16, BF16 per-128-group scales).
//
// The INT4 analog of dsv4_fp8_decode_moe.cu: same warp-per-output-row structure,
// same fused gate+up+SwiGLU, same compact (no padding) decode-band contract.
// The only difference is the weight path: packed INT4 two's complement (2/byte)
// with BF16 per-1×128-group scales, dequantized to FP32 on the fly.
//
// Layout (token-major, identical to the FP8 decode kernels):
//   input    : [num_routes, K]   BF16, packed grouped-by-expert
//   weight_e : [N, K//2]         packed INT4, one matrix per expert
//   scale_e  : [N, K//128]       BF16 per-128-group, one matrix per expert
//   act/out  : [num_routes, N]   BF16
//   offsets/counts/expert_indices : [num_experts]  (expert_indices null = id)

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

#define W4D_WARP_SIZE 32
#define W4D_WARPS 8
#define W4D_THREADS (W4D_WARPS * W4D_WARP_SIZE)
#define W4D_ACT_TILE 8    // activation rows per chunk (grid.y)
#define W4D_VEC 32        // INT4 elements per uint4 load (16 bytes)
#define W4D_SWIGLU_ROW_TILE 1 // gate/up rows per warp
#define W4D_ROW_TILE 4    // weight rows per warp (down kernel only)
#define W4D_SCALE_BLK 128
#define W4D_MIN_BLOCKS_PER_SM 4

static __device__ __forceinline__ float w4d_warp_reduce_sum(float val) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    val += __shfl_xor_sync(0xffffffff, val, off);
  return val;
}

static __device__ __forceinline__ float w4d_swiglu_clamped(float gate, float up,
                                                            float limit) {
  gate = fminf(gate, limit);
  up = fminf(fmaxf(up, -limit), limit);
  return (gate / (1.0f + __expf(-gate))) * up;
}

// Two's complement INT4 → float: 0x0-0x7 → 0..7, 0x8-0xF → -8..-1.
static __device__ __forceinline__ float w4d_s4(uint32_t nibble) {
  return (float)((int)nibble - 16 * (int)(nibble >> 3));
}

// Dequantize 32 packed INT4 values (one uint4 = 16 bytes) and dot with 32
// BF16 activations. All 32 elements share one per-128-group scale (32 | 128).
static __device__ __forceinline__ float
w4d_dot32(float acc, const uint4& packed, float scale,
          const __nv_bfloat16* x) {
  uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
  float sum = 0.0f;
#pragma unroll
  for (int w = 0; w < 4; w++) {
    uint32_t p = words[w];
    sum += w4d_s4(p & 0x0f) * __bfloat162float(x[w * 8 + 0])
         + w4d_s4((p >> 4) & 0x0f) * __bfloat162float(x[w * 8 + 1])
         + w4d_s4((p >> 8) & 0x0f) * __bfloat162float(x[w * 8 + 2])
         + w4d_s4((p >> 12) & 0x0f) * __bfloat162float(x[w * 8 + 3])
         + w4d_s4((p >> 16) & 0x0f) * __bfloat162float(x[w * 8 + 4])
         + w4d_s4((p >> 20) & 0x0f) * __bfloat162float(x[w * 8 + 5])
         + w4d_s4((p >> 24) & 0x0f) * __bfloat162float(x[w * 8 + 6])
         + w4d_s4((p >> 28) & 0x0f) * __bfloat162float(x[w * 8 + 7]);
  }
  return acc + scale * sum;
}

// Fused gate+up+SwiGLU decode grouped GEMV. Each warp owns one row of N
// (= moe intermediate), reads gate/up INT4 rows once, writes act =
// silu(gate·x) * (up·x) for the ≤8 routed rows.
__global__ __launch_bounds__(W4D_THREADS, W4D_MIN_BLOCKS_PER_SM) void
dsv4_w4afp8_grouped_swiglu_decode_kernel(
    const uint64_t* __restrict__ weight_gate_ptrs,
    const uint64_t* __restrict__ scale_gate_ptrs,
    const uint64_t* __restrict__ weight_up_ptrs,
    const uint64_t* __restrict__ scale_up_ptrs,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ act,
    const int* __restrict__ offsets,
    const int* __restrict__ counts,
    const int* __restrict__ expert_indices,
    int N, int K, int scale_cols, float limit) {
  const int compact_expert_idx = blockIdx.z;
  const int expert_M = counts[compact_expert_idx];
  const int chunk_base = blockIdx.y * W4D_ACT_TILE;
  if (chunk_base >= expert_M) return;
  const int warp = threadIdx.x / W4D_WARP_SIZE;
  const int lane = threadIdx.x % W4D_WARP_SIZE;
  const int row_base =
      (blockIdx.x * W4D_WARPS + warp) * W4D_SWIGLU_ROW_TILE;
  if (row_base >= N) return;
  const int tile_raw = expert_M - chunk_base;
  const int tile = tile_raw < W4D_ACT_TILE ? tile_raw : W4D_ACT_TILE;
  const int expert_idx =
      expert_indices ? expert_indices[compact_expert_idx] : compact_expert_idx;
  const int route_base = offsets[compact_expert_idx] + chunk_base;

  const auto* wg = reinterpret_cast<const uint8_t*>(weight_gate_ptrs[expert_idx]);
  const auto* wu = reinterpret_cast<const uint8_t*>(weight_up_ptrs[expert_idx]);
  const auto* sg = reinterpret_cast<const __nv_bfloat16*>(scale_gate_ptrs[expert_idx]);
  const auto* su = reinterpret_cast<const __nv_bfloat16*>(scale_up_ptrs[expert_idx]);
  const int row = row_base;  // SWIGLU_ROW_TILE = 1

  float acc_g[W4D_ACT_TILE];
  float acc_u[W4D_ACT_TILE];
#pragma unroll
  for (int b = 0; b < W4D_ACT_TILE; ++b) {
    acc_g[b] = 0.0f;
    acc_u[b] = 0.0f;
  }

  const int kv = K / W4D_VEC;
  const int bytes_per_row = K / 2;
  for (int v = lane; v < kv; v += W4D_WARP_SIZE) {
    const int k = v * W4D_VEC;
    const int sc = k / W4D_SCALE_BLK;
    const float scale_g = __bfloat162float(sg[row * scale_cols + sc]);
    const float scale_u = __bfloat162float(su[row * scale_cols + sc]);

    uint4 wg4 = *reinterpret_cast<const uint4*>(wg + (int64_t)row * bytes_per_row + k / 2);
    uint4 wu4 = *reinterpret_cast<const uint4*>(wu + (int64_t)row * bytes_per_row + k / 2);

#pragma unroll
    for (int b = 0; b < W4D_ACT_TILE; ++b) {
      if (b < tile) {
        const __nv_bfloat16* xp =
            input + (int64_t)(route_base + b) * K + k;
        acc_g[b] = w4d_dot32(acc_g[b], wg4, scale_g, xp);
        acc_u[b] = w4d_dot32(acc_u[b], wu4, scale_u, xp);
      }
    }
  }

#pragma unroll
  for (int b = 0; b < W4D_ACT_TILE; ++b) {
    if (b < tile) {
      acc_g[b] = w4d_warp_reduce_sum(acc_g[b]);
      acc_u[b] = w4d_warp_reduce_sum(acc_u[b]);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int b = 0; b < W4D_ACT_TILE; ++b) {
      if (b < tile) {
        act[(int64_t)(route_base + b) * N + row] =
            __float2bfloat16(w4d_swiglu_clamped(acc_g[b], acc_u[b], limit));
      }
    }
  }
}

// Single-output decode grouped GEMV (down/w2 projection). A warp owns
// W4D_ROW_TILE consecutive output rows. N = hidden, K = intermediate.
__global__ __launch_bounds__(W4D_THREADS, W4D_MIN_BLOCKS_PER_SM) void
dsv4_w4afp8_grouped_down_decode_kernel(
    const uint64_t* __restrict__ weight_ptrs,
    const uint64_t* __restrict__ scale_ptrs,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    const int* __restrict__ offsets,
    const int* __restrict__ counts,
    const int* __restrict__ expert_indices,
    int N, int K, int scale_cols) {
  const int compact_expert_idx = blockIdx.z;
  const int expert_M = counts[compact_expert_idx];
  const int chunk_base = blockIdx.y * W4D_ACT_TILE;
  if (chunk_base >= expert_M) return;
  const int warp = threadIdx.x / W4D_WARP_SIZE;
  const int lane = threadIdx.x % W4D_WARP_SIZE;
  const int row_base = (blockIdx.x * W4D_WARPS + warp) * W4D_ROW_TILE;
  if (row_base >= N) return;
  const int tile_raw = expert_M - chunk_base;
  const int tile = tile_raw < W4D_ACT_TILE ? tile_raw : W4D_ACT_TILE;
  const int expert_idx =
      expert_indices ? expert_indices[compact_expert_idx] : compact_expert_idx;
  const int route_base = offsets[compact_expert_idx] + chunk_base;

  const auto* weight = reinterpret_cast<const uint8_t*>(weight_ptrs[expert_idx]);
  const auto* scales = reinterpret_cast<const __nv_bfloat16*>(scale_ptrs[expert_idx]);
  const int rows = N - row_base < W4D_ROW_TILE ? N - row_base : W4D_ROW_TILE;
  const int bytes_per_row = K / 2;

  float acc[W4D_ACT_TILE][W4D_ROW_TILE];
#pragma unroll
  for (int b = 0; b < W4D_ACT_TILE; ++b)
#pragma unroll
    for (int r = 0; r < W4D_ROW_TILE; ++r) acc[b][r] = 0.0f;

  const int kv = K / W4D_VEC;
  for (int v = lane; v < kv; v += W4D_WARP_SIZE) {
    const int k = v * W4D_VEC;
    const int sc = k / W4D_SCALE_BLK;
    uint4 w4[W4D_ROW_TILE];
    float scale[W4D_ROW_TILE];
#pragma unroll
    for (int r = 0; r < W4D_ROW_TILE; ++r) {
      if (r < rows) {
        const int row = row_base + r;
        w4[r] = *reinterpret_cast<const uint4*>(
            weight + (int64_t)row * bytes_per_row + k / 2);
        scale[r] = __bfloat162float(scales[row * scale_cols + sc]);
      }
    }
#pragma unroll
    for (int b = 0; b < W4D_ACT_TILE; ++b) {
      if (b < tile) {
        const __nv_bfloat16* xp =
            input + (int64_t)(route_base + b) * K + k;
#pragma unroll
        for (int r = 0; r < W4D_ROW_TILE; ++r) {
          if (r < rows) {
            acc[b][r] = w4d_dot32(acc[b][r], w4[r], scale[r], xp);
          }
        }
      }
    }
  }

#pragma unroll
  for (int b = 0; b < W4D_ACT_TILE; ++b) {
    if (b < tile) {
#pragma unroll
      for (int r = 0; r < W4D_ROW_TILE; ++r) {
        if (r < rows) {
          acc[b][r] = w4d_warp_reduce_sum(acc[b][r]);
        }
      }
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int b = 0; b < W4D_ACT_TILE; ++b) {
      if (b < tile) {
#pragma unroll
        for (int r = 0; r < W4D_ROW_TILE; ++r) {
          if (r < rows) {
            output[(int64_t)(route_base + b) * N + row_base + r] =
                __float2bfloat16(acc[b][r]);
          }
        }
      }
    }
  }
}

extern "C" cudaError_t dsv4_w4afp8_grouped_swiglu_decode_cuda(
    const uint64_t* weight_gate_ptrs, const uint64_t* scale_gate_ptrs,
    const uint64_t* weight_up_ptrs, const uint64_t* scale_up_ptrs,
    const __nv_bfloat16* input, __nv_bfloat16* act, const int* offsets,
    const int* counts, const int* expert_indices, int num_experts,
    int total_routes, int N, int K, int scale_cols, float limit,
    cudaStream_t stream) {
  if (num_experts <= 0 || N <= 0 || K <= 0 || (K % W4D_VEC) != 0)
    return cudaErrorInvalidValue;
  dim3 block(W4D_THREADS);
  dim3 grid((N + W4D_WARPS * W4D_SWIGLU_ROW_TILE - 1) /
                (W4D_WARPS * W4D_SWIGLU_ROW_TILE),
            (total_routes + W4D_ACT_TILE - 1) / W4D_ACT_TILE, num_experts);
  dsv4_w4afp8_grouped_swiglu_decode_kernel<<<grid, block, 0, stream>>>(
      weight_gate_ptrs, scale_gate_ptrs, weight_up_ptrs, scale_up_ptrs, input,
      act, offsets, counts, expert_indices, N, K, scale_cols, limit);
  return cudaGetLastError();
}

extern "C" cudaError_t dsv4_w4afp8_grouped_down_decode_cuda(
    const uint64_t* weight_ptrs, const uint64_t* scale_ptrs,
    const __nv_bfloat16* input, __nv_bfloat16* output, const int* offsets,
    const int* counts, const int* expert_indices, int num_experts,
    int total_routes, int N, int K, int scale_cols, cudaStream_t stream) {
  if (num_experts <= 0 || N <= 0 || K <= 0 || (K % W4D_VEC) != 0)
    return cudaErrorInvalidValue;
  dim3 block(W4D_THREADS);
  dim3 grid((N + W4D_WARPS * W4D_ROW_TILE - 1) / (W4D_WARPS * W4D_ROW_TILE),
            (total_routes + W4D_ACT_TILE - 1) / W4D_ACT_TILE, num_experts);
  dsv4_w4afp8_grouped_down_decode_kernel<<<grid, block, 0, stream>>>(
      weight_ptrs, scale_ptrs, input, output, offsets, counts, expert_indices,
      N, K, scale_cols);
  return cudaGetLastError();
}
