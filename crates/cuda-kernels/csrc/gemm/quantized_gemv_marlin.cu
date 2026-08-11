// Marlin-style W4A16 GEMV for sm_70 (V100) — bandwidth-optimized.
//
// Key optimizations vs the original w4a16_gemv_batch_kernel:
//   - uint4 (128-bit) loads: 32 int4 per memory transaction, maximizing
//     coalesced bandwidth (V100 HBM is the bottleneck, not compute).
//   - 1 warp (32 threads) per row: each thread owns K/32 columns.
//   - Marlin FP16 dequant: nibble → FP16 mantissa (0x6400|nibble = 1024+nibble),
//     subtract 0x6408 (1032) → nibble-8.
//   - FP16 multiply (w*scale*x), FP16 accumulate to minimize compute.
//
// Theoretical decode ceiling for a 27B W4 model on V100 (900 GB/s):
//   13.5 GB / 900 GB/s ≈ 15 ms/tok ≈ 67 tok/s.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

#define WARP_SIZE 32
#define GEMV_THREADS 512
#define GEMV_ROWS 16

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
    extern __shared__ half smem_input_fp16[];

    int batch_idx = blockIdx.y;
    int threads_per_row = GEMV_THREADS / GEMV_ROWS;
    int row = blockIdx.x * GEMV_ROWS + threadIdx.x / threads_per_row;
    int tid_in_row = threadIdx.x % threads_per_row;
    int lane_id = threadIdx.x % WARP_SIZE;

    const __nv_bfloat16* x = input + batch_idx * K;
    for (int i = threadIdx.x; i < K; i += GEMV_THREADS)
        smem_input_fp16[i] = __float2half(__bfloat162float(x[i]));
    __syncthreads();

    if (row >= N) return;

    half sum_h = __float2half(0.0f);
    int num_groups = K / group_size;
    int bytes_per_row = K / 2;

    const uint32_t MASK4 = 0x0f0f0f0fu;
    const uint32_t SUB   = 0x64086408u;
    const half2 SUB_H2   = *reinterpret_cast<const half2*>(&SUB);

    const uint8_t* weight_row = weight + (int64_t)row * bytes_per_row;

    for (int k = tid_in_row * 32; k < K; k += threads_per_row * 32) {
        float scale_f = __bfloat162float(scales[row * num_groups + k / group_size]);
        half scale_h = __float2half(scale_f);
        half2 scale_h2 = __half2half2(scale_h);

        uint4 packed = *reinterpret_cast<const uint4*>(weight_row + k / 2);

        uint32_t words[4];
        words[0] = packed.x;
        words[1] = packed.y;
        words[2] = packed.z;
        words[3] = packed.w;

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

            half2 x01 = *reinterpret_cast<const half2*>(&smem_input_fp16[kk]);
            half2 x23 = *reinterpret_cast<const half2*>(&smem_input_fp16[kk + 2]);
            half2 x45 = *reinterpret_cast<const half2*>(&smem_input_fp16[kk + 4]);
            half2 x67 = *reinterpret_cast<const half2*>(&smem_input_fp16[kk + 6]);

            half2 wx  = __halves2half2(w0.x, w1.x);
            half2 wy  = __halves2half2(w0.y, w1.y);
            half2 wx2 = __halves2half2(w2.x, w3.x);
            half2 wy2 = __halves2half2(w2.y, w3.y);

            half2 p01 = __hmul2(__hmul2(wx, scale_h2), x01);
            half2 p23 = __hmul2(__hmul2(wy, scale_h2), x23);
            half2 p45 = __hmul2(__hmul2(wx2, scale_h2), x45);
            half2 p67 = __hmul2(__hmul2(wy2, scale_h2), x67);

            // FP16 accumulate: sum 8 half products using half2 adds
            half2 s01 = __hadd2(p01, p23);
            half2 s45 = __hadd2(p45, p67);
            half2 s_all = __hadd2(s01, s45);
            sum_h = __hadd(sum_h, __hadd(s_all.x, s_all.y));
        }
    }

    float sum = __half2float(sum_h);
    sum = warp_reduce_sum(sum);
    if (lane_id == 0)
        output[batch_idx * N + row] = __float2bfloat16(sum);
}

extern "C" cudaError_t w4a16_gemv_marlin_cuda(
    const uint8_t* weight, const __nv_bfloat16* scales,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    int B, int N, int K, int group_size, cudaStream_t stream)
{
    dim3 grid((N + GEMV_ROWS - 1) / GEMV_ROWS, B);
    dim3 block(GEMV_THREADS);
    size_t smem = (size_t)K * sizeof(half);
    w4a16_gemv_batch_kernel_marlin<<<grid, block, smem, stream>>>(
        weight, scales, input, output, B, N, K, group_size);
    return cudaGetLastError();
}
