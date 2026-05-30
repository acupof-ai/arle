// BF16 grouped expert GEMM kernels for the Qwen3.5-MoE / Qwen3.6 single-GPU
// SOTA-grouped MoE path (permute -> grouped GEMM -> combine).
//
// These mirror the *structure* of the DSv4 FP8 grouped GEMM kernels in
// dsv4_grouped_gemm.cu (M-axis grouping by expert via per-expert
// offsets/counts, CUDA-core warp-reduce, DSV4_BATCH_TILE=32-way M reuse,
// no tensor-core / mma => sm_70-safe). The only difference is the weight
// element type: BF16 (`__nv_bfloat16`) instead of FP8 E4M3 + E8M0 block
// scales, so there is no quant decode — just `__bfloat162float` MAC.
//
// Layout (token-major, identical to the DSv4 packed grouped buffers):
//   input    : [num_routes, K]   (each route = one (token, expert) pair's
//                                  full hidden vector, packed grouped-by-expert)
//   weight_e : [N, K]            row-major BF16, one matrix per expert
//   output   : [num_routes, N]
//   offsets  : [num_experts]     route_base of each (compact) expert group
//   counts   : [num_experts]     #routes assigned to each expert
//   expert_indices : optional [num_experts] compact->global expert remap;
//                    null => identity (expert e uses weight_ptrs[e]).
//
// The pair variant computes gate (a) and up (b) outputs from the same input
// row in one pass for ~2x input-bandwidth saving — matches the DSv4 pair
// kernel and the way the Rust orchestration pairs gate_proj + up_proj.
//
// W4 nibble-decode variant is an explicit follow-up (the Qwen3.6 production
// checkpoint ships 4-bit experts); this BF16 path is correctness-first and
// runs on tiny-random/qwen3.5-moe (BF16 HF safetensors).
//
// Refs: crates/cuda-kernels/csrc/gemm/dsv4_grouped_gemm.cu (FP8 template)

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

#ifndef MOE_GROUPED_WARP_SIZE
#define MOE_GROUPED_WARP_SIZE 32
#endif
#define MOE_GROUPED_THREADS 256
#define MOE_GROUPED_ROWS 4
#define MOE_GROUPED_BATCH_TILE 32

static __device__ __forceinline__ float moe_grouped_warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

// Single-output grouped GEMM (used for the down projection).
__global__ void moe_bf16_grouped_gemm_batch_kernel(
    const uint64_t* __restrict__ weight_ptrs,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    const int* __restrict__ offsets,
    const int* __restrict__ counts,
    const int* __restrict__ expert_indices,
    int N,
    int K)
{
    int row = blockIdx.x * MOE_GROUPED_ROWS + threadIdx.x / (MOE_GROUPED_THREADS / MOE_GROUPED_ROWS);
    int batch_base_in_expert = blockIdx.y * MOE_GROUPED_BATCH_TILE;
    int compact_expert_idx = blockIdx.z;
    int expert_idx = expert_indices ? expert_indices[compact_expert_idx] : compact_expert_idx;
    int tid_in_row = threadIdx.x % (MOE_GROUPED_THREADS / MOE_GROUPED_ROWS);
    int threads_per_row = MOE_GROUPED_THREADS / MOE_GROUPED_ROWS;
    int lane_id = threadIdx.x % MOE_GROUPED_WARP_SIZE;
    int row_in_block = threadIdx.x / threads_per_row;
    if (row >= N) return;
    const int expert_M = counts[compact_expert_idx];
    if (batch_base_in_expert >= expert_M) return;
    const int tile_M_raw = expert_M - batch_base_in_expert;
    const int tile_M = tile_M_raw < MOE_GROUPED_BATCH_TILE ? tile_M_raw : MOE_GROUPED_BATCH_TILE;
    const int route_base = offsets[compact_expert_idx] + batch_base_in_expert;

    const auto* weight = reinterpret_cast<const __nv_bfloat16*>(weight_ptrs[expert_idx]);
    const __nv_bfloat16* weight_row = weight + (int64_t)row * K;

    // Fast path: tile_M <= 4.
    if (tile_M <= 4) {
        float sums4[4];
#pragma unroll
        for (int b = 0; b < 4; ++b) sums4[b] = 0.0f;

        for (int k = tid_in_row; k < K; k += threads_per_row) {
            const float w = __bfloat162float(weight_row[k]);
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                if (b < tile_M) {
                    sums4[b] += w * __bfloat162float(input[(int64_t)(route_base + b) * K + k]);
                }
            }
        }

        __shared__ float smem4[MOE_GROUPED_ROWS * 8 * 4];
        int warps_per_row = threads_per_row / MOE_GROUPED_WARP_SIZE;
        int warp_in_row = (threadIdx.x % threads_per_row) / MOE_GROUPED_WARP_SIZE;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            sums4[b] = moe_grouped_warp_reduce_sum(sums4[b]);
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
                output[(int64_t)(route_base + b) * N + row] = __float2bfloat16(total);
            }
        }
        return;
    }

    // Full tile: 32-way M reuse.
    float sums[MOE_GROUPED_BATCH_TILE];
#pragma unroll
    for (int b = 0; b < MOE_GROUPED_BATCH_TILE; ++b) sums[b] = 0.0f;

    for (int k = tid_in_row; k < K; k += threads_per_row) {
        const float w = __bfloat162float(weight_row[k]);
#pragma unroll
        for (int b = 0; b < MOE_GROUPED_BATCH_TILE; ++b) {
            if (b < tile_M) {
                sums[b] += w * __bfloat162float(input[(int64_t)(route_base + b) * K + k]);
            }
        }
    }

    __shared__ float smem[MOE_GROUPED_ROWS * 8 * MOE_GROUPED_BATCH_TILE];
    int warps_per_row = threads_per_row / MOE_GROUPED_WARP_SIZE;
    int warp_in_row = (threadIdx.x % threads_per_row) / MOE_GROUPED_WARP_SIZE;
#pragma unroll
    for (int b = 0; b < MOE_GROUPED_BATCH_TILE; ++b) {
        sums[b] = moe_grouped_warp_reduce_sum(sums[b]);
        if (lane_id == 0) {
            smem[(row_in_block * warps_per_row + warp_in_row) * MOE_GROUPED_BATCH_TILE + b] = sums[b];
        }
    }
    __syncthreads();
    if (tid_in_row == 0) {
        for (int b = 0; b < tile_M; ++b) {
            float total = 0.0f;
            for (int w = 0; w < warps_per_row; ++w) {
                total += smem[(row_in_block * warps_per_row + w) * MOE_GROUPED_BATCH_TILE + b];
            }
            output[(int64_t)(route_base + b) * N + row] = __float2bfloat16(total);
        }
    }
}

// Paired grouped GEMM (gate + up from the same input row in one pass).
__global__ void moe_bf16_grouped_gemm_pair_batch_kernel(
    const uint64_t* __restrict__ weight_a_ptrs,
    const uint64_t* __restrict__ weight_b_ptrs,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output_a,
    __nv_bfloat16* __restrict__ output_b,
    const int* __restrict__ offsets,
    const int* __restrict__ counts,
    const int* __restrict__ expert_indices,
    int N,
    int K)
{
    int row = blockIdx.x * MOE_GROUPED_ROWS + threadIdx.x / (MOE_GROUPED_THREADS / MOE_GROUPED_ROWS);
    int batch_base_in_expert = blockIdx.y * MOE_GROUPED_BATCH_TILE;
    int compact_expert_idx = blockIdx.z;
    int expert_idx = expert_indices ? expert_indices[compact_expert_idx] : compact_expert_idx;
    int tid_in_row = threadIdx.x % (MOE_GROUPED_THREADS / MOE_GROUPED_ROWS);
    int threads_per_row = MOE_GROUPED_THREADS / MOE_GROUPED_ROWS;
    int lane_id = threadIdx.x % MOE_GROUPED_WARP_SIZE;
    int row_in_block = threadIdx.x / threads_per_row;
    if (row >= N) return;
    const int expert_M = counts[compact_expert_idx];
    if (batch_base_in_expert >= expert_M) return;
    const int tile_M_raw = expert_M - batch_base_in_expert;
    const int tile_M = tile_M_raw < MOE_GROUPED_BATCH_TILE ? tile_M_raw : MOE_GROUPED_BATCH_TILE;
    const int route_base = offsets[compact_expert_idx] + batch_base_in_expert;

    const auto* weight_a = reinterpret_cast<const __nv_bfloat16*>(weight_a_ptrs[expert_idx]);
    const auto* weight_b = reinterpret_cast<const __nv_bfloat16*>(weight_b_ptrs[expert_idx]);
    const __nv_bfloat16* weight_a_row = weight_a + (int64_t)row * K;
    const __nv_bfloat16* weight_b_row = weight_b + (int64_t)row * K;

    // Fast path: tile_M <= 4.
    if (tile_M <= 4) {
        float sums_a4[4];
        float sums_b4[4];
#pragma unroll
        for (int b = 0; b < 4; ++b) { sums_a4[b] = 0.0f; sums_b4[b] = 0.0f; }

        for (int k = tid_in_row; k < K; k += threads_per_row) {
            const float wa = __bfloat162float(weight_a_row[k]);
            const float wb = __bfloat162float(weight_b_row[k]);
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                if (b < tile_M) {
                    const float xv = __bfloat162float(input[(int64_t)(route_base + b) * K + k]);
                    sums_a4[b] += wa * xv;
                    sums_b4[b] += wb * xv;
                }
            }
        }

        __shared__ float smem_a4[MOE_GROUPED_ROWS * 8 * 4];
        __shared__ float smem_b4[MOE_GROUPED_ROWS * 8 * 4];
        int warps_per_row = threads_per_row / MOE_GROUPED_WARP_SIZE;
        int warp_in_row = (threadIdx.x % threads_per_row) / MOE_GROUPED_WARP_SIZE;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            sums_a4[b] = moe_grouped_warp_reduce_sum(sums_a4[b]);
            sums_b4[b] = moe_grouped_warp_reduce_sum(sums_b4[b]);
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
                output_a[(int64_t)(route_base + b) * N + row] = __float2bfloat16(ta);
                output_b[(int64_t)(route_base + b) * N + row] = __float2bfloat16(tb);
            }
        }
        return;
    }

    // Full tile: 32-way M reuse for both outputs.
    float sums_a[MOE_GROUPED_BATCH_TILE];
    float sums_b[MOE_GROUPED_BATCH_TILE];
#pragma unroll
    for (int b = 0; b < MOE_GROUPED_BATCH_TILE; ++b) { sums_a[b] = 0.0f; sums_b[b] = 0.0f; }

    for (int k = tid_in_row; k < K; k += threads_per_row) {
        const float wa = __bfloat162float(weight_a_row[k]);
        const float wb = __bfloat162float(weight_b_row[k]);
#pragma unroll
        for (int b = 0; b < MOE_GROUPED_BATCH_TILE; ++b) {
            if (b < tile_M) {
                const float xv = __bfloat162float(input[(int64_t)(route_base + b) * K + k]);
                sums_a[b] += wa * xv;
                sums_b[b] += wb * xv;
            }
        }
    }

    __shared__ float smem_a[MOE_GROUPED_ROWS * 8 * MOE_GROUPED_BATCH_TILE];
    __shared__ float smem_b[MOE_GROUPED_ROWS * 8 * MOE_GROUPED_BATCH_TILE];
    int warps_per_row = threads_per_row / MOE_GROUPED_WARP_SIZE;
    int warp_in_row = (threadIdx.x % threads_per_row) / MOE_GROUPED_WARP_SIZE;
#pragma unroll
    for (int b = 0; b < MOE_GROUPED_BATCH_TILE; ++b) {
        sums_a[b] = moe_grouped_warp_reduce_sum(sums_a[b]);
        sums_b[b] = moe_grouped_warp_reduce_sum(sums_b[b]);
        if (lane_id == 0) {
            smem_a[(row_in_block * warps_per_row + warp_in_row) * MOE_GROUPED_BATCH_TILE + b] = sums_a[b];
            smem_b[(row_in_block * warps_per_row + warp_in_row) * MOE_GROUPED_BATCH_TILE + b] = sums_b[b];
        }
    }
    __syncthreads();
    if (tid_in_row == 0) {
        for (int b = 0; b < tile_M; ++b) {
            float ta = 0.0f, tb = 0.0f;
            for (int w = 0; w < warps_per_row; ++w) {
                ta += smem_a[(row_in_block * warps_per_row + w) * MOE_GROUPED_BATCH_TILE + b];
                tb += smem_b[(row_in_block * warps_per_row + w) * MOE_GROUPED_BATCH_TILE + b];
            }
            output_a[(int64_t)(route_base + b) * N + row] = __float2bfloat16(ta);
            output_b[(int64_t)(route_base + b) * N + row] = __float2bfloat16(tb);
        }
    }
}

extern "C" {

cudaError_t moe_bf16_grouped_gemm_batch_cuda(
    const uint64_t* weight_ptrs,
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    const int* offsets,
    const int* counts,
    const int* expert_indices,
    int num_experts,
    int max_count,
    int N,
    int K,
    cudaStream_t stream) {
    if (num_experts <= 0 || max_count <= 0 || N <= 0 || K <= 0) return cudaSuccess;
    dim3 block(MOE_GROUPED_THREADS);
    dim3 grid((N + MOE_GROUPED_ROWS - 1) / MOE_GROUPED_ROWS,
              (max_count + MOE_GROUPED_BATCH_TILE - 1) / MOE_GROUPED_BATCH_TILE,
              num_experts);
    moe_bf16_grouped_gemm_batch_kernel<<<grid, block, 0, stream>>>(
        weight_ptrs, input, output, offsets, counts, expert_indices, N, K);
    return cudaGetLastError();
}

cudaError_t moe_bf16_grouped_gemm_pair_batch_cuda(
    const uint64_t* weight_a_ptrs,
    const uint64_t* weight_b_ptrs,
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
    cudaStream_t stream) {
    if (num_experts <= 0 || max_count <= 0 || N <= 0 || K <= 0) return cudaSuccess;
    dim3 block(MOE_GROUPED_THREADS);
    dim3 grid((N + MOE_GROUPED_ROWS - 1) / MOE_GROUPED_ROWS,
              (max_count + MOE_GROUPED_BATCH_TILE - 1) / MOE_GROUPED_BATCH_TILE,
              num_experts);
    moe_bf16_grouped_gemm_pair_batch_kernel<<<grid, block, 0, stream>>>(
        weight_a_ptrs, weight_b_ptrs, input, output_a, output_b, offsets, counts,
        expert_indices, N, K);
    return cudaGetLastError();
}

}  // extern "C"
