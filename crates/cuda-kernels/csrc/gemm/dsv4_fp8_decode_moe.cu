// DSv4 FP8 decode-band grouped MoE kernels (w8a16, f32 block scales).
//
// The FP8 analog of moe_grouped_gemm.cu's BF16 `*_decode` kernels: compact
// (work scales with REAL routed rows, never padded to a 64/128 tile), 16-byte
// vectorized weight loads (uint4 = 16 FP8 e4m3), each expert's weights read
// exactly once per ≤8-row activation chunk, and per-route correct (a warp owns
// one output row — there is no per-tile single-group contract to violate, so
// no alignment padding). This is the regression endgame: it removes the
// contiguous DeepGEMM lane's pad-row tax at B=1 decode without the scalar
// grouped-GEMV bandwidth penalty (uint4 weight loads ≈ 80% HBM vs the 1-byte
// GEMV's 25% — see errors/2026-06-13-dsv4-decode-gemv-lane-bandwidth-kill).
//
// w8a16: activations stay BF16, weights are FP8 e4m3 with **f32** 128×128
// block scales (DSv4 MoE expert caches store f32 block scales, NOT UE8M0 —
// that encoding is attention-side only). A 16-element vector load lies wholly
// inside one 128-block (16 | 128), so one scale multiply per load.
//
// Layout (token-major, identical to the DSv4 packed grouped buffers and the
// BF16 decode kernels):
//   input    : [num_routes, K]   BF16, packed grouped-by-expert
//   weight_e : [N, K]            row-major FP8 e4m3, one matrix per expert
//   scale_e  : [N/128, K/128]    row-major f32 block scales, per expert
//   act/out  : [num_routes, N]   BF16
//   offsets/counts/expert_indices : [num_experts]  (expert_indices null = id)
//
// Refs: crates/cuda-kernels/csrc/gemm/moe_grouped_gemm.cu (BF16 template),
//       crates/cuda-kernels/csrc/gemm/quantized_gemv.cu (FP8 decode helper).

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

#define FP8D_WARP_SIZE 32
#define FP8D_WARPS 8
#define FP8D_THREADS (FP8D_WARPS * FP8D_WARP_SIZE)
#define FP8D_ACT_TILE 8   // activation rows per chunk (grid.y)
#define FP8D_VEC 16       // FP8 e4m3 elements per 16-byte (uint4) load
#define FP8D_ROW_TILE 4   // weight rows per warp (down kernel only)
#define FP8D_SCALE_BLK 128

static __device__ __forceinline__ float fp8d_e4m3(uint8_t bits) {
    if ((bits & 0x7f) == 0) return 0.0f;
    __nv_fp8_e4m3 v;
    v.__x = bits;
    return static_cast<float>(v);
}

static __device__ __forceinline__ float fp8d_warp_reduce_sum(float val) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, off);
    return val;
}

// DSv4 clamped SwiGLU (matches dsv4_swiglu_clamped_one / dg_swiglu exactly):
//   gate = min(gate, limit); up = clamp(up, -limit, limit);
//   out  = (gate / (1 + exp(-gate))) * up
static __device__ __forceinline__ float fp8d_swiglu_clamped(float gate, float up,
                                                            float limit) {
    gate = fminf(gate, limit);
    up = fminf(fmaxf(up, -limit), limit);
    return (gate / (1.0f + expf(-gate))) * up;
}

// Accumulate 16 FP8 weights (one block scale) against 16 BF16 activations in
// ascending k. `xp` points at the 16 contiguous BF16 activation elements.
static __device__ __forceinline__ float
fp8d_dot16(float acc, const uint8_t* __restrict__ w, float scale,
           const __nv_bfloat16* __restrict__ xp) {
    const uint4 x0 = *reinterpret_cast<const uint4*>(xp);       // bf16[0..8)
    const uint4 x1 = *reinterpret_cast<const uint4*>(xp + 8);   // bf16[8..16)
    const __nv_bfloat16* xb0 = reinterpret_cast<const __nv_bfloat16*>(&x0);
    const __nv_bfloat16* xb1 = reinterpret_cast<const __nv_bfloat16*>(&x1);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const float xv = __bfloat162float(i < 8 ? xb0[i] : xb1[i - 8]);
        acc += (fp8d_e4m3(w[i]) * scale) * xv;
    }
    return acc;
}

// Fused gate+up+SwiGLU decode grouped GEMM: each warp owns one output row of
// N (= moe intermediate), reads that row of the gate and up FP8 matrices once,
// and writes act = silu(gate·x) * (up·x) directly for the ≤8 routed rows.
// N = intermediate, K = hidden. scale geometry: [N/128, K/128] for both
// gate and up (scale_cols = K/128).
__global__ void dsv4_fp8_grouped_swiglu_decode_kernel(
    const uint64_t* __restrict__ weight_gate_ptrs,
    const uint64_t* __restrict__ scale_gate_ptrs,
    const uint64_t* __restrict__ weight_up_ptrs,
    const uint64_t* __restrict__ scale_up_ptrs,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ act,
    const int* __restrict__ offsets,
    const int* __restrict__ counts,
    const int* __restrict__ expert_indices,
    int N,
    int K,
    int scale_cols,
    float limit)
{
    const int compact_expert_idx = blockIdx.z;
    const int expert_M = counts[compact_expert_idx];
    const int chunk_base = blockIdx.y * FP8D_ACT_TILE;
    if (chunk_base >= expert_M) return;
    const int warp = threadIdx.x / FP8D_WARP_SIZE;
    const int lane = threadIdx.x % FP8D_WARP_SIZE;
    const int row = blockIdx.x * FP8D_WARPS + warp;
    if (row >= N) return;
    const int tile_raw = expert_M - chunk_base;
    const int tile = tile_raw < FP8D_ACT_TILE ? tile_raw : FP8D_ACT_TILE;
    const int expert_idx =
        expert_indices ? expert_indices[compact_expert_idx] : compact_expert_idx;
    const int route_base = offsets[compact_expert_idx] + chunk_base;

    const auto* wg = reinterpret_cast<const uint8_t*>(weight_gate_ptrs[expert_idx]);
    const auto* wu = reinterpret_cast<const uint8_t*>(weight_up_ptrs[expert_idx]);
    const auto* sg = reinterpret_cast<const float*>(scale_gate_ptrs[expert_idx]);
    const auto* su = reinterpret_cast<const float*>(scale_up_ptrs[expert_idx]);
    const uint8_t* gate_row = wg + (int64_t)row * K;
    const uint8_t* up_row = wu + (int64_t)row * K;
    const int scale_row_off = (row / FP8D_SCALE_BLK) * scale_cols;

    float acc_g[FP8D_ACT_TILE];
    float acc_u[FP8D_ACT_TILE];
#pragma unroll
    for (int b = 0; b < FP8D_ACT_TILE; ++b) { acc_g[b] = 0.0f; acc_u[b] = 0.0f; }

    const int kv = K / FP8D_VEC; // launcher enforces K % 16 == 0
    for (int v = lane; v < kv; v += FP8D_WARP_SIZE) {
        const int k = v * FP8D_VEC;
        const int sc = k / FP8D_SCALE_BLK;
        const float scale_g = sg[scale_row_off + sc];
        const float scale_u = su[scale_row_off + sc];
        const uint4 wg4 = *reinterpret_cast<const uint4*>(gate_row + k);
        const uint4 wu4 = *reinterpret_cast<const uint4*>(up_row + k);
        const uint8_t* wgb = reinterpret_cast<const uint8_t*>(&wg4);
        const uint8_t* wub = reinterpret_cast<const uint8_t*>(&wu4);
#pragma unroll
        for (int b = 0; b < FP8D_ACT_TILE; ++b) {
            if (b < tile) {
                const __nv_bfloat16* xp = input + (int64_t)(route_base + b) * K + k;
                acc_g[b] = fp8d_dot16(acc_g[b], wgb, scale_g, xp);
                acc_u[b] = fp8d_dot16(acc_u[b], wub, scale_u, xp);
            }
        }
    }

#pragma unroll
    for (int b = 0; b < FP8D_ACT_TILE; ++b) {
        if (b < tile) {
            acc_g[b] = fp8d_warp_reduce_sum(acc_g[b]);
            acc_u[b] = fp8d_warp_reduce_sum(acc_u[b]);
        }
    }
    if (lane == 0) {
        for (int b = 0; b < tile; ++b) {
            act[(int64_t)(route_base + b) * N + row] =
                __float2bfloat16(fp8d_swiglu_clamped(acc_g[b], acc_u[b], limit));
        }
    }
}

// Single-output decode grouped GEMM (down/w2 projection). A warp owns
// FP8D_ROW_TILE consecutive output rows (K is short for w2 — keeps enough
// loads in flight). N = hidden, K = intermediate. scale geometry [N/128, K/128].
__global__ void dsv4_fp8_grouped_down_decode_kernel(
    const uint64_t* __restrict__ weight_ptrs,
    const uint64_t* __restrict__ scale_ptrs,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    const int* __restrict__ offsets,
    const int* __restrict__ counts,
    const int* __restrict__ expert_indices,
    int N,
    int K,
    int scale_cols)
{
    const int compact_expert_idx = blockIdx.z;
    const int expert_M = counts[compact_expert_idx];
    const int chunk_base = blockIdx.y * FP8D_ACT_TILE;
    if (chunk_base >= expert_M) return;
    const int warp = threadIdx.x / FP8D_WARP_SIZE;
    const int lane = threadIdx.x % FP8D_WARP_SIZE;
    const int row_base = (blockIdx.x * FP8D_WARPS + warp) * FP8D_ROW_TILE;
    if (row_base >= N) return;
    const int tile_raw = expert_M - chunk_base;
    const int tile = tile_raw < FP8D_ACT_TILE ? tile_raw : FP8D_ACT_TILE;
    const int expert_idx =
        expert_indices ? expert_indices[compact_expert_idx] : compact_expert_idx;
    const int route_base = offsets[compact_expert_idx] + chunk_base;

    const auto* weight = reinterpret_cast<const uint8_t*>(weight_ptrs[expert_idx]);
    const auto* scales = reinterpret_cast<const float*>(scale_ptrs[expert_idx]);
    const int rows = N - row_base < FP8D_ROW_TILE ? N - row_base : FP8D_ROW_TILE;

    float acc[FP8D_ACT_TILE][FP8D_ROW_TILE];
#pragma unroll
    for (int b = 0; b < FP8D_ACT_TILE; ++b)
#pragma unroll
        for (int r = 0; r < FP8D_ROW_TILE; ++r) acc[b][r] = 0.0f;

    const int kv = K / FP8D_VEC; // launcher enforces K % 16 == 0
    for (int v = lane; v < kv; v += FP8D_WARP_SIZE) {
        const int k = v * FP8D_VEC;
        const int sc = k / FP8D_SCALE_BLK;
        uint4 w4[FP8D_ROW_TILE];
        float scale[FP8D_ROW_TILE];
#pragma unroll
        for (int r = 0; r < FP8D_ROW_TILE; ++r) {
            if (r < rows) {
                w4[r] = *reinterpret_cast<const uint4*>(weight + (int64_t)(row_base + r) * K + k);
                scale[r] = scales[((row_base + r) / FP8D_SCALE_BLK) * scale_cols + sc];
            }
        }
#pragma unroll
        for (int b = 0; b < FP8D_ACT_TILE; ++b) {
            if (b < tile) {
                const __nv_bfloat16* xp = input + (int64_t)(route_base + b) * K + k;
#pragma unroll
                for (int r = 0; r < FP8D_ROW_TILE; ++r) {
                    if (r < rows) {
                        acc[b][r] = fp8d_dot16(
                            acc[b][r], reinterpret_cast<const uint8_t*>(&w4[r]),
                            scale[r], xp);
                    }
                }
            }
        }
    }

#pragma unroll
    for (int b = 0; b < FP8D_ACT_TILE; ++b) {
        if (b < tile) {
#pragma unroll
            for (int r = 0; r < FP8D_ROW_TILE; ++r) {
                if (r < rows) acc[b][r] = fp8d_warp_reduce_sum(acc[b][r]);
            }
        }
    }
    if (lane == 0) {
        for (int b = 0; b < tile; ++b) {
            for (int r = 0; r < rows; ++r) {
                output[(int64_t)(route_base + b) * N + row_base + r] =
                    __float2bfloat16(acc[b][r]);
            }
        }
    }
}

extern "C" {

cudaError_t dsv4_fp8_grouped_swiglu_decode_cuda(
    const uint64_t* weight_gate_ptrs,
    const uint64_t* scale_gate_ptrs,
    const uint64_t* weight_up_ptrs,
    const uint64_t* scale_up_ptrs,
    const __nv_bfloat16* input,
    __nv_bfloat16* act,
    const int* offsets,
    const int* counts,
    const int* expert_indices,
    int num_experts,
    int max_count,
    int N,
    int K,
    int scale_cols,
    float limit,
    cudaStream_t stream) {
    if (num_experts <= 0 || max_count <= 0 || N <= 0 || K <= 0) return cudaSuccess;
    if (K % FP8D_VEC != 0) return cudaErrorInvalidValue;
    dim3 block(FP8D_THREADS);
    dim3 grid((N + FP8D_WARPS - 1) / FP8D_WARPS,
              (max_count + FP8D_ACT_TILE - 1) / FP8D_ACT_TILE,
              num_experts);
    dsv4_fp8_grouped_swiglu_decode_kernel<<<grid, block, 0, stream>>>(
        weight_gate_ptrs, scale_gate_ptrs, weight_up_ptrs, scale_up_ptrs,
        input, act, offsets, counts, expert_indices, N, K, scale_cols, limit);
    return cudaGetLastError();
}

cudaError_t dsv4_fp8_grouped_down_decode_cuda(
    const uint64_t* weight_ptrs,
    const uint64_t* scale_ptrs,
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    const int* offsets,
    const int* counts,
    const int* expert_indices,
    int num_experts,
    int max_count,
    int N,
    int K,
    int scale_cols,
    cudaStream_t stream) {
    if (num_experts <= 0 || max_count <= 0 || N <= 0 || K <= 0) return cudaSuccess;
    if (K % FP8D_VEC != 0) return cudaErrorInvalidValue;
    dim3 block(FP8D_THREADS);
    const int row_warps = (N + FP8D_ROW_TILE - 1) / FP8D_ROW_TILE;
    dim3 grid((row_warps + FP8D_WARPS - 1) / FP8D_WARPS,
              (max_count + FP8D_ACT_TILE - 1) / FP8D_ACT_TILE,
              num_experts);
    dsv4_fp8_grouped_down_decode_kernel<<<grid, block, 0, stream>>>(
        weight_ptrs, scale_ptrs, input, output, offsets, counts, expert_indices,
        N, K, scale_cols);
    return cudaGetLastError();
}

}  // extern "C"
