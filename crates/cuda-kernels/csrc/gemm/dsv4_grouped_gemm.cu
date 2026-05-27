// DSv4 grouped block-scaled FP8 GEMM kernels with M-tile=DSV4_BATCH_TILE
// weight reuse.
//
// Companion to the grouped GEMV kernels in quantized_gemv.cu (which index
// grid Y by max_count and have 1× weight reuse — fine for decode M=1,
// catastrophic at prefill where 29795-token batches hit 325s wall-clock,
// 67× off the SLO target). These kernels mirror the M-tile pattern from
// dsv4_fp8_gemv_batch_tiled_kernel (single-expert path) but adapt it to
// the grouped (multi-expert in one launch) call shape: grid Y tiles
// tokens into DSV4_BATCH_TILE=32 groups, each block loads weight once
// and reuses it across all 32 tokens in the tile.
//
// Pair variant computes gate/up outputs from the same input row in one
// pass for ~2× input bandwidth saving.
//
// Drop-in compatible ABI with the grouped GEMV cudaError_t wrappers so
// dispatch is a pure choice in mlp.rs::dsv4_run_grouped_block_scaled_gemv*.
//
// Refs:
//   docs/experience/errors/2026-05-27-dsv4-tp-allreduce-slo-prefill-kill.md
//   docs/bench-and-trace-spec.md §7.6 roofline gate
//   docs/bench-and-trace-spec.md §7.7 SLO-shape probe rule

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

#define WARP_SIZE 32
#define GEMV_THREADS 256
#define GEMV_ROWS 4
#define DSV4_BATCH_TILE 32

// Device helpers — static-linkage copies of the FP8 decode + warp reduce
// inlines defined identically in quantized_gemv.cu. Kept self-contained
// here (rather than extracting a shared header) to keep this change
// scoped — see memory/feedback_file_naming_semantic_alignment.md.
static __device__ __forceinline__ float dsv4_grouped_gemm_warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

static __device__ __forceinline__ float dsv4_grouped_gemm_decode_e8m0(uint8_t bits) {
    uint32_t raw = static_cast<uint32_t>(bits) << 23;
    return __uint_as_float(raw);
}

static __device__ __forceinline__ float dsv4_grouped_gemm_decode_fp8_e4m3(uint8_t bits) {
    if ((bits & 0x7f) == 0) return 0.0f;
    if ((bits & 0x7f) == 0x7f) {
        return (bits & 0x80) ? -448.0f : 448.0f;
    }
    __nv_fp8_e4m3 value;
    value.__x = bits;
    return static_cast<float>(value);
}

__global__ void dsv4_fp8_grouped_gemm_batch_kernel(
    const uint64_t* __restrict__ weight_ptrs,
    const uint64_t* __restrict__ scale_ptrs,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    const int* __restrict__ offsets,
    const int* __restrict__ counts,
    const int* __restrict__ expert_indices,
    int max_count,
    int N,
    int K,
    int scale_rows,
    int scale_cols)
{
    int row = blockIdx.x * GEMV_ROWS + threadIdx.x / (GEMV_THREADS / GEMV_ROWS);
    int batch_base_in_expert = blockIdx.y * DSV4_BATCH_TILE;
    int compact_expert_idx = blockIdx.z;
    int expert_idx = expert_indices ? expert_indices[compact_expert_idx] : compact_expert_idx;
    int tid_in_row = threadIdx.x % (GEMV_THREADS / GEMV_ROWS);
    int threads_per_row = GEMV_THREADS / GEMV_ROWS;
    int lane_id = threadIdx.x % WARP_SIZE;
    int row_in_block = threadIdx.x / threads_per_row;
    if (row >= N) return;
    const int expert_M = counts[compact_expert_idx];
    if (batch_base_in_expert >= expert_M) return;
    const int tile_M_raw = expert_M - batch_base_in_expert;
    const int tile_M = tile_M_raw < DSV4_BATCH_TILE ? tile_M_raw : DSV4_BATCH_TILE;
    const int route_base = offsets[compact_expert_idx] + batch_base_in_expert;

    const auto* weight = reinterpret_cast<const uint8_t*>(weight_ptrs[expert_idx]);
    const auto* scales = reinterpret_cast<const uint8_t*>(scale_ptrs[expert_idx]);
    const int block_h = (N + scale_rows - 1) / scale_rows;
    const int block_w = (K + scale_cols - 1) / scale_cols;
    const int sr_raw = row / block_h;
    const int sr = sr_raw < scale_rows ? sr_raw : (scale_rows - 1);
    const int scale_row_offset = sr * scale_cols;
    const uint8_t* weight_row = weight + row * K;
    const uint8_t* scale_row = scales + scale_row_offset;

    // Fast path: tile_M <= 4 — matches single-expert tiled kernel pattern,
    // lower register pressure when many experts only get a handful of tokens.
    if (tile_M <= 4) {
        float sums4[4];
#pragma unroll
        for (int b = 0; b < 4; ++b) sums4[b] = 0.0f;

        for (int sc = 0; sc < scale_cols; ++sc) {
            const int k_start = sc * block_w;
            if (k_start >= K) break;
            int k_end = k_start + block_w;
            if (k_end > K) k_end = K;
            const float scale = dsv4_grouped_gemm_decode_e8m0(scale_row[sc]);
            for (int k = k_start + tid_in_row; k < k_end; k += threads_per_row) {
                const float w = dsv4_grouped_gemm_decode_fp8_e4m3(weight_row[k]) * scale;
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    if (b < tile_M) {
                        sums4[b] += w * __bfloat162float(input[(route_base + b) * K + k]);
                    }
                }
            }
        }

        __shared__ float smem4[GEMV_ROWS * 8 * 4];
        int warps_per_row = threads_per_row / WARP_SIZE;
        int warp_in_row = (threadIdx.x % threads_per_row) / WARP_SIZE;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            sums4[b] = dsv4_grouped_gemm_warp_reduce_sum(sums4[b]);
            if (lane_id == 0) {
                smem4[(row_in_block * warps_per_row + warp_in_row) * 4 + b] = sums4[b];
            }
        }
        __syncthreads();
        if (tid_in_row == 0) {
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                if (b >= tile_M) continue;
                float total = 0.0f;
                for (int w = 0; w < warps_per_row; ++w) {
                    total += smem4[(row_in_block * warps_per_row + w) * 4 + b];
                }
                output[(route_base + b) * N + row] = __float2bfloat16(total);
            }
        }
        return;
    }

    // Full tile: 32-way M reuse. Per-thread accum array sits in registers.
    float sums[DSV4_BATCH_TILE];
#pragma unroll
    for (int b = 0; b < DSV4_BATCH_TILE; ++b) sums[b] = 0.0f;

    for (int sc = 0; sc < scale_cols; ++sc) {
        const int k_start = sc * block_w;
        if (k_start >= K) break;
        int k_end = k_start + block_w;
        if (k_end > K) k_end = K;
        const float scale = dsv4_grouped_gemm_decode_e8m0(scale_row[sc]);
        for (int k = k_start + tid_in_row; k < k_end; k += threads_per_row) {
            const float w = dsv4_grouped_gemm_decode_fp8_e4m3(weight_row[k]) * scale;
#pragma unroll
            for (int b = 0; b < DSV4_BATCH_TILE; ++b) {
                if (b < tile_M) {
                    sums[b] += w * __bfloat162float(input[(route_base + b) * K + k]);
                }
            }
        }
    }

    __shared__ float smem[GEMV_ROWS * 8 * DSV4_BATCH_TILE];
    int warps_per_row = threads_per_row / WARP_SIZE;
    int warp_in_row = (threadIdx.x % threads_per_row) / WARP_SIZE;
#pragma unroll
    for (int b = 0; b < DSV4_BATCH_TILE; ++b) {
        sums[b] = dsv4_grouped_gemm_warp_reduce_sum(sums[b]);
        if (lane_id == 0) {
            smem[(row_in_block * warps_per_row + warp_in_row) * DSV4_BATCH_TILE + b] = sums[b];
        }
    }
    __syncthreads();
    if (tid_in_row == 0) {
        for (int b = 0; b < tile_M; ++b) {
            float total = 0.0f;
            for (int w = 0; w < warps_per_row; ++w) {
                total += smem[(row_in_block * warps_per_row + w) * DSV4_BATCH_TILE + b];
            }
            output[(route_base + b) * N + row] = __float2bfloat16(total);
        }
    }
}

__global__ void dsv4_fp8_grouped_gemm_pair_batch_kernel(
    const uint64_t* __restrict__ weight_a_ptrs,
    const uint64_t* __restrict__ scale_a_ptrs,
    const uint64_t* __restrict__ weight_b_ptrs,
    const uint64_t* __restrict__ scale_b_ptrs,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output_a,
    __nv_bfloat16* __restrict__ output_b,
    const int* __restrict__ offsets,
    const int* __restrict__ counts,
    const int* __restrict__ expert_indices,
    int max_count,
    int N,
    int K,
    int scale_rows,
    int scale_cols)
{
    int row = blockIdx.x * GEMV_ROWS + threadIdx.x / (GEMV_THREADS / GEMV_ROWS);
    int batch_base_in_expert = blockIdx.y * DSV4_BATCH_TILE;
    int compact_expert_idx = blockIdx.z;
    int expert_idx = expert_indices ? expert_indices[compact_expert_idx] : compact_expert_idx;
    int tid_in_row = threadIdx.x % (GEMV_THREADS / GEMV_ROWS);
    int threads_per_row = GEMV_THREADS / GEMV_ROWS;
    int lane_id = threadIdx.x % WARP_SIZE;
    int row_in_block = threadIdx.x / threads_per_row;
    if (row >= N) return;
    const int expert_M = counts[compact_expert_idx];
    if (batch_base_in_expert >= expert_M) return;
    const int tile_M_raw = expert_M - batch_base_in_expert;
    const int tile_M = tile_M_raw < DSV4_BATCH_TILE ? tile_M_raw : DSV4_BATCH_TILE;
    const int route_base = offsets[compact_expert_idx] + batch_base_in_expert;

    const auto* weight_a = reinterpret_cast<const uint8_t*>(weight_a_ptrs[expert_idx]);
    const auto* scales_a = reinterpret_cast<const uint8_t*>(scale_a_ptrs[expert_idx]);
    const auto* weight_b = reinterpret_cast<const uint8_t*>(weight_b_ptrs[expert_idx]);
    const auto* scales_b = reinterpret_cast<const uint8_t*>(scale_b_ptrs[expert_idx]);
    const int block_h = (N + scale_rows - 1) / scale_rows;
    const int block_w = (K + scale_cols - 1) / scale_cols;
    const int sr_raw = row / block_h;
    const int sr = sr_raw < scale_rows ? sr_raw : (scale_rows - 1);
    const int scale_row_offset = sr * scale_cols;
    const uint8_t* weight_a_row = weight_a + row * K;
    const uint8_t* weight_b_row = weight_b + row * K;
    const uint8_t* scale_a_row = scales_a + scale_row_offset;
    const uint8_t* scale_b_row = scales_b + scale_row_offset;

    // Fast path: tile_M <= 4
    if (tile_M <= 4) {
        float sums_a4[4];
        float sums_b4[4];
#pragma unroll
        for (int b = 0; b < 4; ++b) { sums_a4[b] = 0.0f; sums_b4[b] = 0.0f; }

        for (int sc = 0; sc < scale_cols; ++sc) {
            const int k_start = sc * block_w;
            if (k_start >= K) break;
            int k_end = k_start + block_w;
            if (k_end > K) k_end = K;
            const float sa = dsv4_grouped_gemm_decode_e8m0(scale_a_row[sc]);
            const float sb = dsv4_grouped_gemm_decode_e8m0(scale_b_row[sc]);
            for (int k = k_start + tid_in_row; k < k_end; k += threads_per_row) {
                const float wa = dsv4_grouped_gemm_decode_fp8_e4m3(weight_a_row[k]) * sa;
                const float wb = dsv4_grouped_gemm_decode_fp8_e4m3(weight_b_row[k]) * sb;
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    if (b < tile_M) {
                        const float xv = __bfloat162float(input[(route_base + b) * K + k]);
                        sums_a4[b] += wa * xv;
                        sums_b4[b] += wb * xv;
                    }
                }
            }
        }

        __shared__ float smem_a4[GEMV_ROWS * 8 * 4];
        __shared__ float smem_b4[GEMV_ROWS * 8 * 4];
        int warps_per_row = threads_per_row / WARP_SIZE;
        int warp_in_row = (threadIdx.x % threads_per_row) / WARP_SIZE;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            sums_a4[b] = dsv4_grouped_gemm_warp_reduce_sum(sums_a4[b]);
            sums_b4[b] = dsv4_grouped_gemm_warp_reduce_sum(sums_b4[b]);
            if (lane_id == 0) {
                smem_a4[(row_in_block * warps_per_row + warp_in_row) * 4 + b] = sums_a4[b];
                smem_b4[(row_in_block * warps_per_row + warp_in_row) * 4 + b] = sums_b4[b];
            }
        }
        __syncthreads();
        if (tid_in_row == 0) {
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                if (b >= tile_M) continue;
                float ta = 0.0f, tb = 0.0f;
                for (int w = 0; w < warps_per_row; ++w) {
                    ta += smem_a4[(row_in_block * warps_per_row + w) * 4 + b];
                    tb += smem_b4[(row_in_block * warps_per_row + w) * 4 + b];
                }
                output_a[(route_base + b) * N + row] = __float2bfloat16(ta);
                output_b[(route_base + b) * N + row] = __float2bfloat16(tb);
            }
        }
        return;
    }

    // Full tile: 32-way M reuse for both outputs. Per-thread = 64 float accum.
    float sums_a[DSV4_BATCH_TILE];
    float sums_b[DSV4_BATCH_TILE];
#pragma unroll
    for (int b = 0; b < DSV4_BATCH_TILE; ++b) { sums_a[b] = 0.0f; sums_b[b] = 0.0f; }

    for (int sc = 0; sc < scale_cols; ++sc) {
        const int k_start = sc * block_w;
        if (k_start >= K) break;
        int k_end = k_start + block_w;
        if (k_end > K) k_end = K;
        const float sa = dsv4_grouped_gemm_decode_e8m0(scale_a_row[sc]);
        const float sb = dsv4_grouped_gemm_decode_e8m0(scale_b_row[sc]);
        for (int k = k_start + tid_in_row; k < k_end; k += threads_per_row) {
            const float wa = dsv4_grouped_gemm_decode_fp8_e4m3(weight_a_row[k]) * sa;
            const float wb = dsv4_grouped_gemm_decode_fp8_e4m3(weight_b_row[k]) * sb;
#pragma unroll
            for (int b = 0; b < DSV4_BATCH_TILE; ++b) {
                if (b < tile_M) {
                    const float xv = __bfloat162float(input[(route_base + b) * K + k]);
                    sums_a[b] += wa * xv;
                    sums_b[b] += wb * xv;
                }
            }
        }
    }

    __shared__ float smem_a[GEMV_ROWS * 8 * DSV4_BATCH_TILE];
    __shared__ float smem_b[GEMV_ROWS * 8 * DSV4_BATCH_TILE];
    int warps_per_row = threads_per_row / WARP_SIZE;
    int warp_in_row = (threadIdx.x % threads_per_row) / WARP_SIZE;
#pragma unroll
    for (int b = 0; b < DSV4_BATCH_TILE; ++b) {
        sums_a[b] = dsv4_grouped_gemm_warp_reduce_sum(sums_a[b]);
        sums_b[b] = dsv4_grouped_gemm_warp_reduce_sum(sums_b[b]);
        if (lane_id == 0) {
            smem_a[(row_in_block * warps_per_row + warp_in_row) * DSV4_BATCH_TILE + b] = sums_a[b];
            smem_b[(row_in_block * warps_per_row + warp_in_row) * DSV4_BATCH_TILE + b] = sums_b[b];
        }
    }
    __syncthreads();
    if (tid_in_row == 0) {
        for (int b = 0; b < tile_M; ++b) {
            float ta = 0.0f, tb = 0.0f;
            for (int w = 0; w < warps_per_row; ++w) {
                ta += smem_a[(row_in_block * warps_per_row + w) * DSV4_BATCH_TILE + b];
                tb += smem_b[(row_in_block * warps_per_row + w) * DSV4_BATCH_TILE + b];
            }
            output_a[(route_base + b) * N + row] = __float2bfloat16(ta);
            output_b[(route_base + b) * N + row] = __float2bfloat16(tb);
        }
    }
}

extern "C" {

cudaError_t dsv4_fp8_grouped_gemm_batch_cuda(
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
    int scale_rows,
    int scale_cols,
    cudaStream_t stream) {
    if (num_experts <= 0 || max_count <= 0 || N <= 0 || K <= 0) return cudaSuccess;
    dim3 block(GEMV_THREADS);
    dim3 grid((N + GEMV_ROWS - 1) / GEMV_ROWS,
              (max_count + DSV4_BATCH_TILE - 1) / DSV4_BATCH_TILE,
              num_experts);
    dsv4_fp8_grouped_gemm_batch_kernel<<<grid, block, 0, stream>>>(
        weight_ptrs, scale_ptrs, input, output, offsets, counts, expert_indices,
        max_count, N, K, scale_rows, scale_cols);
    return cudaGetLastError();
}

cudaError_t dsv4_fp8_grouped_gemm_pair_batch_cuda(
    const uint64_t* weight_a_ptrs,
    const uint64_t* scale_a_ptrs,
    const uint64_t* weight_b_ptrs,
    const uint64_t* scale_b_ptrs,
    const __nv_bfloat16* input,
    __nv_bfloat16* output_a,
    __nv_bfloat16* output_b,
    const int* offsets,
    const int* counts,
    const int* expert_indices,
    int num_experts,
    int max_count,
    int N,
    int K,
    int scale_rows,
    int scale_cols,
    cudaStream_t stream) {
    if (num_experts <= 0 || max_count <= 0 || N <= 0 || K <= 0) return cudaSuccess;
    dim3 block(GEMV_THREADS);
    dim3 grid((N + GEMV_ROWS - 1) / GEMV_ROWS,
              (max_count + DSV4_BATCH_TILE - 1) / DSV4_BATCH_TILE,
              num_experts);
    dsv4_fp8_grouped_gemm_pair_batch_kernel<<<grid, block, 0, stream>>>(
        weight_a_ptrs, scale_a_ptrs, weight_b_ptrs, scale_b_ptrs, input,
        output_a, output_b, offsets, counts, expert_indices, max_count, N, K,
        scale_rows, scale_cols);
    return cudaGetLastError();
}

}  // extern "C"
