// Marlin-style W4A16 GEMV/GEMM for sm_70 (V100) — bandwidth-optimized.
//
// For B=1: standard GEMV (1 row × 1 input).
// For B>=2: batched GEMM where BTILE inputs share one weight read.
//
// Key optimizations:
//   - uint4 (128-bit) weight loads via __ldg (read-only cache).
//   - 1 warp (32 threads) per row; each thread owns K/32 columns.
//   - Marlin FP16 dequant: nibble → FP16 mantissa trick.
//   - Software pipelining: prefetch next weight while processing current.
//   - Input loads hoisted ahead of FMAs for memory-level parallelism.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

#define WARP_SIZE 32
#define GEMV_THREADS 512
#define GEMV_ROWS 16
#define W4A16_MARLIN_BTILE 16

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

__global__ void w4a16_gemv_batch_kernel_marlin(
    const uint8_t* __restrict__ weight,
    const __nv_bfloat16* __restrict__ scales,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int B, int N, int K, int group_size)
{
    int batch_base = blockIdx.y * W4A16_MARLIN_BTILE;
    int threads_per_row = GEMV_THREADS / GEMV_ROWS;
    int row = blockIdx.x * GEMV_ROWS + threadIdx.x / threads_per_row;
    int tid_in_row = threadIdx.x % threads_per_row;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (row >= N) return;

    int valid_b = min(W4A16_MARLIN_BTILE, B - batch_base);

    half2 sum_h2[W4A16_MARLIN_BTILE];
    #pragma unroll
    for (int b = 0; b < W4A16_MARLIN_BTILE; b++)
        sum_h2[b] = __float2half2_rn(0.0f);

    int num_groups = K / group_size;
    int bytes_per_row = K / 2;

    const uint32_t MASK4 = 0x0f0f0f0fu;
    const uint32_t SUB   = 0x64086408u;
    const half2 SUB_H2   = *reinterpret_cast<const half2*>(&SUB);

    const uint8_t* weight_row = weight + (int64_t)row * bytes_per_row;

    // Software pipeline: prefetch first weight.
    int k = tid_in_row * 32;
    uint4 packed = (k < K) ? __ldg(reinterpret_cast<const uint4*>(weight_row + k / 2)) : make_uint4(0,0,0,0);

    for (; k < K; k += threads_per_row * 32) {
        float scale_f = __bfloat162float(scales[row * num_groups + k / group_size]);
        half2 scale_h2 = __half2half2(__float2half(scale_f));

        // Prefetch next weight while processing current.
        int next_k = k + threads_per_row * 32;
        uint4 next_packed = (next_k < K) ? __ldg(reinterpret_cast<const uint4*>(weight_row + next_k / 2)) : make_uint4(0,0,0,0);

        uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};

        #pragma unroll
        for (int w = 0; w < 4; w++) {
            uint32_t p = words[w];
            int kk = k + w * 8;

            uint32_t lo_all = p & MASK4;
            uint32_t hi_all = (p >> 4) & MASK4;

            uint32_t lo01 = (0x6400u | (lo_all & 0xffu)) |
                            ((0x6400u | ((lo_all >> 8) & 0xffu)) << 16);
            uint32_t lo23 = (0x6400u | ((lo_all >> 16) & 0xffu)) |
                            ((0x6400u | ((lo_all >> 24) & 0xffu)) << 16);
            uint32_t hi01 = (0x6400u | (hi_all & 0xffu)) |
                            ((0x6400u | ((hi_all >> 8) & 0xffu)) << 16);
            uint32_t hi23 = (0x6400u | ((hi_all >> 16) & 0xffu)) |
                            ((0x6400u | ((hi_all >> 24) & 0xffu)) << 16);

            half2 w0 = __hsub2(*reinterpret_cast<half2*>(&lo01), SUB_H2);
            half2 w1 = __hsub2(*reinterpret_cast<half2*>(&hi01), SUB_H2);
            half2 w2 = __hsub2(*reinterpret_cast<half2*>(&lo23), SUB_H2);
            half2 w3 = __hsub2(*reinterpret_cast<half2*>(&hi23), SUB_H2);

            half2 wx  = __halves2half2(w0.x, w1.x);
            half2 wy  = __halves2half2(w0.y, w1.y);
            half2 wx2 = __halves2half2(w2.x, w3.x);
            half2 wy2 = __halves2half2(w2.y, w3.y);

            half2 wsx  = __hmul2(wx, scale_h2);
            half2 wsy  = __hmul2(wy, scale_h2);
            half2 wsx2 = __hmul2(wx2, scale_h2);
            half2 wsy2 = __hmul2(wy2, scale_h2);

            // Hoist input loads: load all batch inputs first, then FMAs.
            half2 x01[W4A16_MARLIN_BTILE], x23[W4A16_MARLIN_BTILE];
            half2 x45[W4A16_MARLIN_BTILE], x67[W4A16_MARLIN_BTILE];

            #pragma unroll
            for (int b = 0; b < W4A16_MARLIN_BTILE; b++) {
                if (b >= valid_b) break;
                const __nv_bfloat16* xb = input + (batch_base + b) * K;
                uint4 xpacked = __ldg(reinterpret_cast<const uint4*>(&xb[kk]));
                const __nv_bfloat16* xbv = reinterpret_cast<const __nv_bfloat16*>(&xpacked);
                x01[b] = __halves2half2(__float2half(__bfloat162float(xbv[0])),
                                        __float2half(__bfloat162float(xbv[1])));
                x23[b] = __halves2half2(__float2half(__bfloat162float(xbv[2])),
                                        __float2half(__bfloat162float(xbv[3])));
                x45[b] = __halves2half2(__float2half(__bfloat162float(xbv[4])),
                                        __float2half(__bfloat162float(xbv[5])));
                x67[b] = __halves2half2(__float2half(__bfloat162float(xbv[6])),
                                        __float2half(__bfloat162float(xbv[7])));
            }

            #pragma unroll
            for (int b = 0; b < W4A16_MARLIN_BTILE; b++) {
                if (b >= valid_b) break;
                sum_h2[b] = __hfma2(wsx,  x01[b], sum_h2[b]);
                sum_h2[b] = __hfma2(wsy,  x23[b], sum_h2[b]);
                sum_h2[b] = __hfma2(wsx2, x45[b], sum_h2[b]);
                sum_h2[b] = __hfma2(wsy2, x67[b], sum_h2[b]);
            }
        }

        packed = next_packed;
    }

    #pragma unroll
    for (int b = 0; b < W4A16_MARLIN_BTILE; b++) {
        if (b >= valid_b) break;
        half sum_h = __hadd(sum_h2[b].x, sum_h2[b].y);
        float sum = __half2float(sum_h);
        sum = warp_reduce_sum(sum);
        if (lane_id == 0)
            output[(batch_base + b) * N + row] = __float2bfloat16(sum);
    }
}

extern "C" cudaError_t w4a16_gemv_batch_cuda_marlin(
    const uint8_t* weight, const __nv_bfloat16* scales,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    int B, int N, int K, int group_size, cudaStream_t stream)
{
    dim3 grid((N + GEMV_ROWS - 1) / GEMV_ROWS,
              (B + W4A16_MARLIN_BTILE - 1) / W4A16_MARLIN_BTILE);
    dim3 block(GEMV_THREADS);
    static int printed = 0;
    if (!printed) {
        fprintf(stderr, "MARLIN_GEMV: grid=(%d,%d) block=%d GEMV_ROWS=%d GEMV_THREADS=%d BTILE=%d B=%d N=%d K=%d\n",
                grid.x, grid.y, block.x, GEMV_ROWS, GEMV_THREADS, W4A16_MARLIN_BTILE, B, N, K);
        printed = 1;
    }
    w4a16_gemv_batch_kernel_marlin<<<grid, block, 0, stream>>>(
        weight, scales, input, output, B, N, K, group_size);
    return cudaGetLastError();
}
