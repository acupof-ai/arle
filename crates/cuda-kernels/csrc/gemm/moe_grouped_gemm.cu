// BF16 grouped expert GEMM kernels for the Qwen3.5-MoE / Qwen3.6 single-GPU
// SOTA-grouped MoE path (permute -> grouped GEMM -> combine).
//
// Structure: M-axis grouping by expert via per-expert offsets/counts,
// CUDA-core warp-reduce, 32-way M-tile weight reuse, no tensor-core / mma
// => sm_70-safe. Weight element type is BF16 (`__nv_bfloat16`), so there
// is no quant decode — just `__bfloat162float` MAC.
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
// row in one pass for ~2x input-bandwidth saving — matches the way the Rust
// orchestration pairs gate_proj + up_proj.
//
// W4 nibble-decode variant is an explicit follow-up (the Qwen3.6 production
// checkpoint ships 4-bit experts); this BF16 path is correctness-first and
// runs on tiny-random/qwen3.5-moe (BF16 HF safetensors).

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

// ── Decode-band (small-R) weight-read-bound grouped GEMM ───────────────────
//
// At decode R = num_routes is tiny (R = topk = 8 at B=1; R = 8B on the
// stage-1 batched path) and per-expert counts are 1..B, so the batch kernels
// above run their tile_M<=4 fast path with zero M reuse: 64-thread row
// groups issue SCALAR 2-byte weight loads (<= 64B per warp request, ~2
// outstanding loads per thread -> latency-bound), the down grid launches
// N/4 x E blocks of which ~94% early-exit on counts==0, and a separate
// silu_mul pass re-reads gate/up. Measured (nsys 2026-06-11, B=1 decode,
// Qwen3.6-35B-A3B, H20): 250 us/layer pair + 237 us/layer down = 76.6% of
// decode GPU time at ~3% of HBM bandwidth.
//
// These decode kernels are weight-read-bound by construction: the grid
// covers (weight-row tiles x activation chunks x experts) so every weight
// row of every TOUCHED expert is read exactly once per <=8-row activation
// chunk, coalesced. One WARP owns one weight row [row, 0..K) and streams it
// with 16-byte (8 x bf16) vector loads, dotting against the <=
// MOE_DECODE_ACT_TILE activation rows routed to the expert (read from the
// packed activations + per-expert offsets/counts the pack path already
// produces; activation rows are L1/L2-hot, R*K*2B total). Experts with
// count 0 exit after a single counts read. count_e > ACT_TILE (B > 8 batched
// decode) pays ceil(count_e/8) weight passes — still >= 8-way activation
// reuse per pass.
//
// Ideal bytes (Qwen3.6-35B shapes I=512, H=2048, E_active experts):
//   gate[512,2048] + up[512,2048] + down[2048,512] = 3*512*2048*2B
//   = 6.29 MB/expert -> E_active=8: 50.3 MB -> 12.6 us at 4 TB/s;
//   realistic 25-60 us/layer target vs the measured 487 us/layer.
//
// Numerics contract: fp32 accumulate, bf16 in/out. Accumulation order per
// output element: lane l sums its k-segments {k : (k/8) % 32 == l} in
// ascending k (8 sequential fp32 FMAs per 16B vector), then a 5-step
// shfl_xor butterfly (offsets 16,8,4,2,1) combines lanes. This REORDERS the
// reduction vs the batch kernels' stride-64 lanes + smem two-stage sum —
// ulp-level only, covered by the correct-inference gate (needle +
// same-config-twice floor), NOT byte-identity.
//
// Requires K % 8 == 0 (16-byte row alignment: weight slab bases are
// >=256B-aligned allocations and row offsets are row*K*2B; same for the
// packed activation rows). The Rust dispatch guards and falls back to the
// batch kernels otherwise.

#define MOE_DECODE_WARPS 8    // warps per block
#define MOE_DECODE_THREADS (MOE_DECODE_WARPS * MOE_GROUPED_WARP_SIZE)
#define MOE_DECODE_ACT_TILE 8 // activation rows per chunk (grid.y)
#define MOE_DECODE_VEC 8      // bf16 elements per 16-byte vector load
// Down-projection K is short (Qwen3.6: K=I=512 -> 2 v-iterations/lane), so a
// warp owning ONE row keeps only ~2 weight loads in flight and runs
// latency-bound at 6% of HBM peak (nsys 2026-06-11: 69 us/layer vs the
// swiglu kernel's 25% at K=2048). A warp therefore owns MOE_DECODE_ROW_TILE
// consecutive rows: 4x the loads in flight per lane, and the activation
// vector is loaded once per v-iteration and reused across the row tile.
#define MOE_DECODE_ROW_TILE 4 // weight rows per warp (down kernel only)

struct alignas(16) moe_bf16x8 {
    __nv_bfloat162 h[4];
};

static __device__ __forceinline__ moe_bf16x8
moe_load_bf16x8(const __nv_bfloat16* __restrict__ p) {
    moe_bf16x8 v;
    *reinterpret_cast<uint4*>(&v) = *reinterpret_cast<const uint4*>(p);
    return v;
}

// 8 sequential fp32 FMAs in ascending k — the per-lane accumulation order.
static __device__ __forceinline__ float moe_decode_dot8(float acc,
                                                        const moe_bf16x8& w,
                                                        const moe_bf16x8& x) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 wf = __bfloat1622float2(w.h[i]);
        const float2 xf = __bfloat1622float2(x.h[i]);
        acc += wf.x * xf.x;
        acc += wf.y * xf.y;
    }
    return acc;
}

// Matches silu_mul_one (csrc/misc/elementwise_basic.cu): fp32 sigmoid form.
static __device__ __forceinline__ float moe_decode_silu(float g) {
    return g / (1.0f + __expf(-g));
}

// Single-output decode grouped GEMM (down projection).
__global__ void moe_bf16_grouped_gemm_decode_kernel(
    const uint64_t* __restrict__ weight_ptrs,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    const int* __restrict__ offsets,
    const int* __restrict__ counts,
    const int* __restrict__ expert_indices,
    int N,
    int K)
{
    const int compact_expert_idx = blockIdx.z;
    const int expert_M = counts[compact_expert_idx];
    const int chunk_base = blockIdx.y * MOE_DECODE_ACT_TILE;
    if (chunk_base >= expert_M) return;
    const int warp = threadIdx.x / MOE_GROUPED_WARP_SIZE;
    const int lane = threadIdx.x % MOE_GROUPED_WARP_SIZE;
    const int row_base =
        (blockIdx.x * MOE_DECODE_WARPS + warp) * MOE_DECODE_ROW_TILE;
    if (row_base >= N) return;
    const int tile_raw = expert_M - chunk_base;
    const int tile = tile_raw < MOE_DECODE_ACT_TILE ? tile_raw : MOE_DECODE_ACT_TILE;
    const int expert_idx = expert_indices ? expert_indices[compact_expert_idx] : compact_expert_idx;
    const int route_base = offsets[compact_expert_idx] + chunk_base;

    const auto* weight = reinterpret_cast<const __nv_bfloat16*>(weight_ptrs[expert_idx]);
    const int rows = N - row_base < MOE_DECODE_ROW_TILE ? N - row_base
                                                        : MOE_DECODE_ROW_TILE;

    float acc[MOE_DECODE_ACT_TILE][MOE_DECODE_ROW_TILE];
#pragma unroll
    for (int b = 0; b < MOE_DECODE_ACT_TILE; ++b)
#pragma unroll
        for (int r = 0; r < MOE_DECODE_ROW_TILE; ++r) acc[b][r] = 0.0f;

    const int kv = K / MOE_DECODE_VEC; // launcher enforces K % 8 == 0
    for (int v = lane; v < kv; v += MOE_GROUPED_WARP_SIZE) {
        moe_bf16x8 w[MOE_DECODE_ROW_TILE];
#pragma unroll
        for (int r = 0; r < MOE_DECODE_ROW_TILE; ++r) {
            if (r < rows) {
                w[r] = moe_load_bf16x8(weight + (int64_t)(row_base + r) * K +
                                       v * MOE_DECODE_VEC);
            }
        }
#pragma unroll
        for (int b = 0; b < MOE_DECODE_ACT_TILE; ++b) {
            if (b < tile) {
                const moe_bf16x8 x = moe_load_bf16x8(
                    input + (int64_t)(route_base + b) * K + v * MOE_DECODE_VEC);
#pragma unroll
                for (int r = 0; r < MOE_DECODE_ROW_TILE; ++r) {
                    if (r < rows) acc[b][r] = moe_decode_dot8(acc[b][r], w[r], x);
                }
            }
        }
    }

    // tile/rows are warp-uniform, so the predicated reduce keeps all lanes
    // converged. Per-element reduction order is unchanged from the one-row
    // form: same lane->k-segment mapping, same butterfly.
#pragma unroll
    for (int b = 0; b < MOE_DECODE_ACT_TILE; ++b) {
        if (b < tile) {
#pragma unroll
            for (int r = 0; r < MOE_DECODE_ROW_TILE; ++r) {
                if (r < rows) acc[b][r] = moe_grouped_warp_reduce_sum(acc[b][r]);
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

// Fused gate+up+SwiGLU decode grouped GEMM: reads each expert's gate and up
// matrices once and writes act = silu(gate_dot) * up_dot directly — the
// separate silu_mul pass (read 2x + write 1x over [num_routes, N]) folds into
// the epilogue. silu applies to the fp32 accumulators BEFORE the single bf16
// round; the batch path rounds gate/up to bf16 first — a rounding-class
// deviation in the same ulp class as the reduction reorder (gate-covered).
__global__ void moe_bf16_grouped_gemm_swiglu_decode_kernel(
    const uint64_t* __restrict__ weight_gate_ptrs,
    const uint64_t* __restrict__ weight_up_ptrs,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ act,
    const int* __restrict__ offsets,
    const int* __restrict__ counts,
    const int* __restrict__ expert_indices,
    int N,
    int K)
{
    const int compact_expert_idx = blockIdx.z;
    const int expert_M = counts[compact_expert_idx];
    const int chunk_base = blockIdx.y * MOE_DECODE_ACT_TILE;
    if (chunk_base >= expert_M) return;
    const int warp = threadIdx.x / MOE_GROUPED_WARP_SIZE;
    const int lane = threadIdx.x % MOE_GROUPED_WARP_SIZE;
    const int row = blockIdx.x * MOE_DECODE_WARPS + warp;
    if (row >= N) return;
    const int tile_raw = expert_M - chunk_base;
    const int tile = tile_raw < MOE_DECODE_ACT_TILE ? tile_raw : MOE_DECODE_ACT_TILE;
    const int expert_idx = expert_indices ? expert_indices[compact_expert_idx] : compact_expert_idx;
    const int route_base = offsets[compact_expert_idx] + chunk_base;

    const auto* weight_gate = reinterpret_cast<const __nv_bfloat16*>(weight_gate_ptrs[expert_idx]);
    const auto* weight_up = reinterpret_cast<const __nv_bfloat16*>(weight_up_ptrs[expert_idx]);
    const __nv_bfloat16* gate_row = weight_gate + (int64_t)row * K;
    const __nv_bfloat16* up_row = weight_up + (int64_t)row * K;

    float acc_g[MOE_DECODE_ACT_TILE];
    float acc_u[MOE_DECODE_ACT_TILE];
#pragma unroll
    for (int b = 0; b < MOE_DECODE_ACT_TILE; ++b) { acc_g[b] = 0.0f; acc_u[b] = 0.0f; }

    const int kv = K / MOE_DECODE_VEC; // launcher enforces K % 8 == 0
    for (int v = lane; v < kv; v += MOE_GROUPED_WARP_SIZE) {
        const moe_bf16x8 wg = moe_load_bf16x8(gate_row + v * MOE_DECODE_VEC);
        const moe_bf16x8 wu = moe_load_bf16x8(up_row + v * MOE_DECODE_VEC);
#pragma unroll
        for (int b = 0; b < MOE_DECODE_ACT_TILE; ++b) {
            if (b < tile) {
                const moe_bf16x8 x = moe_load_bf16x8(
                    input + (int64_t)(route_base + b) * K + v * MOE_DECODE_VEC);
                acc_g[b] = moe_decode_dot8(acc_g[b], wg, x);
                acc_u[b] = moe_decode_dot8(acc_u[b], wu, x);
            }
        }
    }

#pragma unroll
    for (int b = 0; b < MOE_DECODE_ACT_TILE; ++b) {
        if (b < tile) {
            acc_g[b] = moe_grouped_warp_reduce_sum(acc_g[b]);
            acc_u[b] = moe_grouped_warp_reduce_sum(acc_u[b]);
        }
    }
    if (lane == 0) {
        for (int b = 0; b < tile; ++b) {
            act[(int64_t)(route_base + b) * N + row] =
                __float2bfloat16(moe_decode_silu(acc_g[b]) * acc_u[b]);
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

cudaError_t moe_bf16_grouped_gemm_decode_cuda(
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
    if (K % MOE_DECODE_VEC != 0) return cudaErrorInvalidValue;
    dim3 block(MOE_DECODE_THREADS);
    const int rows_per_block = MOE_DECODE_WARPS * MOE_DECODE_ROW_TILE;
    dim3 grid((N + rows_per_block - 1) / rows_per_block,
              (max_count + MOE_DECODE_ACT_TILE - 1) / MOE_DECODE_ACT_TILE,
              num_experts);
    moe_bf16_grouped_gemm_decode_kernel<<<grid, block, 0, stream>>>(
        weight_ptrs, input, output, offsets, counts, expert_indices, N, K);
    return cudaGetLastError();
}

cudaError_t moe_bf16_grouped_gemm_swiglu_decode_cuda(
    const uint64_t* weight_gate_ptrs,
    const uint64_t* weight_up_ptrs,
    const __nv_bfloat16* input,
    __nv_bfloat16* act,
    const int* offsets,
    const int* counts,
    const int* expert_indices,
    int num_experts,
    int max_count,
    int N,
    int K,
    cudaStream_t stream) {
    if (num_experts <= 0 || max_count <= 0 || N <= 0 || K <= 0) return cudaSuccess;
    if (K % MOE_DECODE_VEC != 0) return cudaErrorInvalidValue;
    dim3 block(MOE_DECODE_THREADS);
    dim3 grid((N + MOE_DECODE_WARPS - 1) / MOE_DECODE_WARPS,
              (max_count + MOE_DECODE_ACT_TILE - 1) / MOE_DECODE_ACT_TILE,
              num_experts);
    moe_bf16_grouped_gemm_swiglu_decode_kernel<<<grid, block, 0, stream>>>(
        weight_gate_ptrs, weight_up_ptrs, input, act, offsets, counts,
        expert_indices, N, K);
    return cudaGetLastError();
}

}  // extern "C"
