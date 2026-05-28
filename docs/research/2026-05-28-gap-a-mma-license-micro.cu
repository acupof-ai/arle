// GAP-A Phase 2 license-or-kill micro-experiment.
// Standalone test: bench the existing scalar
// `dsv4_fp8_gemv_batch_tiled_kernel` on the canonical DSv4 decode shape
// (B=1,4,16; N=2048; K=7168) on H20 and compute achieved HBM3 bandwidth.
//
// Decision rule:
//   achieved_bw / 4 TB/s >= 0.75 ==> BW-bound ==> MMA cannot deliver 1.5x ==> KILL
//   achieved_bw / 4 TB/s <  0.50 ==> headroom ==> proceed to write MMA kernel
//   0.50 <= ratio < 0.75 ==> ambiguous; report and decide
//
// We do NOT build an MMA kernel in this pass — just measure the floor.
// If the floor is high, the ceiling is low. License-or-kill at the
// cheapest possible point per CLAUDE.md §0.
//
// Build: nvcc -O3 -arch=sm_90 -std=c++17 gap_a_micro.cu -o gap_a_micro
// Run:   ./gap_a_micro

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA error %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); std::exit(1); } } while (0)

#define WARP_SIZE 32
#define GEMV_THREADS 256
#define GEMV_ROWS 4
#define DSV4_BATCH_TILE 32

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float dsv4_decode_e8m0(uint8_t bits) {
    uint32_t raw = static_cast<uint32_t>(bits) << 23;
    return __uint_as_float(raw);
}

__device__ __forceinline__ float dsv4_decode_fp8_e4m3(uint8_t bits) {
    if ((bits & 0x7f) == 0) return 0.0f;
    if ((bits & 0x7f) == 0x7f) {
        return (bits & 0x80) ? -448.0f : 448.0f;
    }
    __nv_fp8_e4m3 value;
    value.__x = bits;
    return static_cast<float>(value);
}

// Verbatim copy of dsv4_fp8_gemv_batch_tiled_kernel from
// crates/cuda-kernels/csrc/gemm/quantized_gemv.cu:392-515 — we want to
// measure exactly the production kernel, not a re-derived variant.
__global__ void dsv4_fp8_gemv_batch_tiled_kernel(
    const uint8_t* __restrict__ weight,
    const uint8_t* __restrict__ scales,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int B,
    int N,
    int K,
    int scale_rows,
    int scale_cols)
{
    int row = blockIdx.x * GEMV_ROWS + threadIdx.x / (GEMV_THREADS / GEMV_ROWS);
    int batch_base = blockIdx.y * DSV4_BATCH_TILE;
    int tid_in_row = threadIdx.x % (GEMV_THREADS / GEMV_ROWS);
    int threads_per_row = GEMV_THREADS / GEMV_ROWS;
    int lane_id = threadIdx.x % WARP_SIZE;
    int row_in_block = threadIdx.x / threads_per_row;
    if (row >= N) return;

    const int block_h = (N + scale_rows - 1) / scale_rows;
    const int block_w = (K + scale_cols - 1) / scale_cols;
    const int sr_raw = row / block_h;
    const int sr = sr_raw < scale_rows ? sr_raw : (scale_rows - 1);
    const int scale_row_offset = sr * scale_cols;
    const uint8_t* weight_row = weight + row * K;
    const uint8_t* scale_row = scales + scale_row_offset;
    const int tile_batches_raw = B - batch_base;
    const int tile_batches = tile_batches_raw < DSV4_BATCH_TILE ? tile_batches_raw : DSV4_BATCH_TILE;

    if (tile_batches <= 4) {
        float sums4[4];
#pragma unroll
        for (int b = 0; b < 4; ++b) sums4[b] = 0.0f;

        for (int sc = 0; sc < scale_cols; ++sc) {
            const int k_start = sc * block_w;
            if (k_start >= K) break;
            int k_end = k_start + block_w;
            if (k_end > K) k_end = K;
            const float scale = dsv4_decode_e8m0(scale_row[sc]);
            for (int k = k_start + tid_in_row; k < k_end; k += threads_per_row) {
                const float w = dsv4_decode_fp8_e4m3(weight_row[k]) * scale;
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    if (b < tile_batches) {
                        const int batch_idx = batch_base + b;
                        sums4[b] += w * __bfloat162float(input[batch_idx * K + k]);
                    }
                }
            }
        }

        __shared__ float smem4[GEMV_ROWS * 8 * 4];
        int warps_per_row = threads_per_row / WARP_SIZE;
        int warp_in_row = (threadIdx.x % threads_per_row) / WARP_SIZE;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            sums4[b] = warp_reduce_sum(sums4[b]);
            if (lane_id == 0) {
                smem4[(row_in_block * warps_per_row + warp_in_row) * 4 + b] = sums4[b];
            }
        }
        __syncthreads();
        if (tid_in_row == 0) {
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                if (b >= tile_batches) continue;
                const int batch_idx = batch_base + b;
                float total = 0.0f;
                for (int w = 0; w < warps_per_row; ++w) {
                    total += smem4[(row_in_block * warps_per_row + w) * 4 + b];
                }
                output[batch_idx * N + row] = __float2bfloat16(total);
            }
        }
        return;
    }

    float sums[DSV4_BATCH_TILE];
#pragma unroll
    for (int b = 0; b < DSV4_BATCH_TILE; ++b) sums[b] = 0.0f;

    for (int sc = 0; sc < scale_cols; ++sc) {
        const int k_start = sc * block_w;
        if (k_start >= K) break;
        int k_end = k_start + block_w;
        if (k_end > K) k_end = K;
        const float scale = dsv4_decode_e8m0(scale_row[sc]);
        for (int k = k_start + tid_in_row; k < k_end; k += threads_per_row) {
            const float w = dsv4_decode_fp8_e4m3(weight_row[k]) * scale;
#pragma unroll
            for (int b = 0; b < DSV4_BATCH_TILE; ++b) {
                int batch_idx = batch_base + b;
                if (batch_idx < B) {
                    sums[b] += w * __bfloat162float(input[batch_idx * K + k]);
                }
            }
        }
    }

    __shared__ float smem[GEMV_ROWS * 8 * DSV4_BATCH_TILE];
    int warps_per_row = threads_per_row / WARP_SIZE;
    int warp_in_row = (threadIdx.x % threads_per_row) / WARP_SIZE;
#pragma unroll
    for (int b = 0; b < DSV4_BATCH_TILE; ++b) {
        sums[b] = warp_reduce_sum(sums[b]);
        if (lane_id == 0) {
            smem[(row_in_block * warps_per_row + warp_in_row) * DSV4_BATCH_TILE + b] = sums[b];
        }
    }
    __syncthreads();
    if (tid_in_row == 0) {
#pragma unroll
        for (int b = 0; b < DSV4_BATCH_TILE; ++b) {
            int batch_idx = batch_base + b;
            if (batch_idx >= B) continue;
            float total = 0.0f;
            for (int w = 0; w < warps_per_row; ++w) {
                total += smem[(row_in_block * warps_per_row + w) * DSV4_BATCH_TILE + b];
            }
            output[batch_idx * N + row] = __float2bfloat16(total);
        }
    }
}

static double bench_one(int B, int N, int K, int scale_rows, int scale_cols,
                        const uint8_t* d_weight, const uint8_t* d_scales,
                        const __nv_bfloat16* d_input, __nv_bfloat16* d_output) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    dim3 grid((N + GEMV_ROWS - 1) / GEMV_ROWS, (B + DSV4_BATCH_TILE - 1) / DSV4_BATCH_TILE);
    dim3 block(GEMV_THREADS);

    // Warmup
    for (int i = 0; i < 50; ++i) {
        dsv4_fp8_gemv_batch_tiled_kernel<<<grid, block>>>(d_weight, d_scales, d_input, d_output,
                                                          B, N, K, scale_rows, scale_cols);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    const int iters = 200;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        dsv4_fp8_gemv_batch_tiled_kernel<<<grid, block>>>(d_weight, d_scales, d_input, d_output,
                                                          B, N, K, scale_rows, scale_cols);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / iters / 1000.0;  // seconds per call
}

int main() {
    // DSv4 canonical decode shape: K=7168 (hidden), N=2048 (one TP shard of
    // q_b_proj-like).
    const int N = 2048;
    const int K = 7168;
    // DSv4 block-scaled FP8 layout: block_h = block_w = 128 (per
    // dsv4_deepgemm_ops.cu kScaleGranK = 128).
    const int scale_rows = (N + 127) / 128;
    const int scale_cols = (K + 127) / 128;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    fprintf(stderr, "Device: %s  SM count=%d  MEM clock=%d MHz  bus=%d-bit\n",
            prop.name, prop.multiProcessorCount, prop.memoryClockRate / 1000, prop.memoryBusWidth);
    // Peak BW = 2 * memClock (kHz) * busWidth (bits) / 8 / 1e9 GB/s
    double peak_bw_gbs = 2.0 * (prop.memoryClockRate * 1.0e3) * (prop.memoryBusWidth / 8.0) / 1.0e9;
    fprintf(stderr, "Reported peak HBM BW: %.0f GB/s\n", peak_bw_gbs);

    // Allocate buffers sized for B=16 (max we test).
    const int B_max = 16;
    std::vector<uint8_t> h_weight(static_cast<size_t>(N) * K, 0x38u);  // 0x38 = +1.0 in FP8E4M3
    std::vector<uint8_t> h_scales(static_cast<size_t>(scale_rows) * scale_cols, 0x7fu);  // E8M0 = 1.0
    std::vector<__nv_bfloat16> h_input(static_cast<size_t>(B_max) * K, __float2bfloat16(1.0f));

    uint8_t* d_weight = nullptr;
    uint8_t* d_scales = nullptr;
    __nv_bfloat16* d_input = nullptr;
    __nv_bfloat16* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_weight, h_weight.size()));
    CUDA_CHECK(cudaMalloc(&d_scales, h_scales.size()));
    CUDA_CHECK(cudaMalloc(&d_input, h_input.size() * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_output, static_cast<size_t>(B_max) * N * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight.data(), h_weight.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scales, h_scales.data(), h_scales.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    printf("# DSv4 FP8 batched GEMV scalar kernel — BW achievement on %s\n", prop.name);
    printf("# shape N=%d K=%d  scale[rows=%d cols=%d]  peak HBM BW=%.0f GB/s\n",
           N, K, scale_rows, scale_cols, peak_bw_gbs);
    printf("# %-4s %-12s %-12s %-12s %-12s %-12s\n",
           "B", "kernel_us", "weight_GB", "act_GB", "achieved_GB/s", "frac_peak");
    for (int B : {1, 4, 16}) {
        double sec = bench_one(B, N, K, scale_rows, scale_cols, d_weight, d_scales, d_input, d_output);
        // Bytes moved per call:
        //   weight: N*K bytes (FP8 = 1 B/elem), loaded once per call (no reuse across B in scalar)
        //   activation: B*K * 2 bytes (BF16)
        //   scales: scale_rows*scale_cols (tiny, ignored)
        //   output: B*N * 2 bytes (BF16; tiny vs weight)
        double weight_gb = (static_cast<double>(N) * K) / 1.0e9;
        double act_gb = (static_cast<double>(B) * K * 2.0) / 1.0e9;
        double out_gb = (static_cast<double>(B) * N * 2.0) / 1.0e9;
        double total_gb = weight_gb + act_gb + out_gb;
        double achieved_bw = total_gb / sec;
        double frac_peak = achieved_bw / peak_bw_gbs;
        printf("  %-4d %-12.2f %-12.4f %-12.4f %-12.0f %-12.3f\n",
               B, sec * 1.0e6, weight_gb, act_gb, achieved_bw, frac_peak);
    }
    printf("# Decision rule: frac_peak >= 0.75 ==> BW-bound ==> KILL GAP-A on H20\n");
    printf("#                frac_peak <  0.50 ==> compute-bound ==> proceed to MMA kernel\n");
    printf("#                0.50..0.75 ==> ambiguous, report and gate manually\n");

    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_scales));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    return 0;
}
