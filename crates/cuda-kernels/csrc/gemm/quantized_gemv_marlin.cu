// Marlin-style W4A16 GEMV for sm_70 (V100) — bandwidth-optimized.
//
// Key optimizations:
//   - uint4 (128-bit) weight loads via __ldg (read-only cache).
//   - 1 warp (32 threads) per row; each thread owns K/32 columns.
//   - Marlin FP16 dequant: nibble → FP16 mantissa (0x6400|nibble = 1024+nibble),
//     subtract 0x6408 (1032) → nibble-8.
//   - FP16 FMA: sum += (w*scale) * x in one instruction.
//   - half2 accumulator.
//   - Outer loop unrolled by 2: two weight loads issued before processing to
//     keep more memory transactions in flight.
//   - Input loaded directly from global memory via __ldg (no shared mem).
//
// Theoretical decode ceiling for a 27B W4 model on V100 (900 GB/s):
//   13.5 GB / 900 GB/s ≈ 15 ms/tok ≈ 67 tok/s.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

#define WARP_SIZE 32
#define GEMV_THREADS 256
#define GEMV_ROWS 8

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ void process_word(
    uint32_t p, int kk, const __nv_bfloat16* x,
    half2 scale_h2, half2& sum_h2,
    uint32_t MASK4, half2 SUB_H2)
{
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

    uint4 xpacked = __ldg(reinterpret_cast<const uint4*>(&x[kk]));
    const __nv_bfloat16* xb = reinterpret_cast<const __nv_bfloat16*>(&xpacked);
    half2 x01 = __halves2half2(__float2half(__bfloat162float(xb[0])),
                               __float2half(__bfloat162float(xb[1])));
    half2 x23 = __halves2half2(__float2half(__bfloat162float(xb[2])),
                               __float2half(__bfloat162float(xb[3])));
    half2 x45 = __halves2half2(__float2half(__bfloat162float(xb[4])),
                               __float2half(__bfloat162float(xb[5])));
    half2 x67 = __halves2half2(__float2half(__bfloat162float(xb[6])),
                               __float2half(__bfloat162float(xb[7])));

    half2 wx  = __halves2half2(w0.x, w1.x);
    half2 wy  = __halves2half2(w0.y, w1.y);
    half2 wx2 = __halves2half2(w2.x, w3.x);
    half2 wy2 = __halves2half2(w2.y, w3.y);

    half2 wsx  = __hmul2(wx, scale_h2);
    half2 wsy  = __hmul2(wy, scale_h2);
    half2 wsx2 = __hmul2(wx2, scale_h2);
    half2 wsy2 = __hmul2(wy2, scale_h2);

    sum_h2 = __hfma2(wsx,  x01, sum_h2);
    sum_h2 = __hfma2(wsy,  x23, sum_h2);
    sum_h2 = __hfma2(wsx2, x45, sum_h2);
    sum_h2 = __hfma2(wsy2, x67, sum_h2);
}

__global__ void w4a16_gemv_batch_kernel_marlin(
    const uint8_t* __restrict__ weight,
    const __nv_bfloat16* __restrict__ scales,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int B, int N, int K, int group_size)
{
    int batch_idx = blockIdx.y;
    int threads_per_row = GEMV_THREADS / GEMV_ROWS;
    int row = blockIdx.x * GEMV_ROWS + threadIdx.x / threads_per_row;
    int tid_in_row = threadIdx.x % threads_per_row;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (row >= N) return;

    half2 sum_h2 = __float2half2_rn(0.0f);
    int num_groups = K / group_size;
    int bytes_per_row = K / 2;

    const uint32_t MASK4 = 0x0f0f0f0fu;
    const uint32_t SUB   = 0x64086408u;
    const half2 SUB_H2   = *reinterpret_cast<const half2*>(&SUB);

    const uint8_t* weight_row = weight + (int64_t)row * bytes_per_row;
    const __nv_bfloat16* x = input + batch_idx * K;

    int k = tid_in_row * 32;
    const int k_step = threads_per_row * 32;

    // Unroll by 2: issue two weight loads before processing.
    for (; k + k_step < K; k += 2 * k_step) {
        float scale_f0 = __bfloat162float(scales[row * num_groups + k / group_size]);
        half2 scale_h2_0 = __half2half2(__float2half(scale_f0));

        float scale_f1 = __bfloat162float(scales[row * num_groups + (k + k_step) / group_size]);
        half2 scale_h2_1 = __half2half2(__float2half(scale_f1));

        uint4 packed0 = __ldg(reinterpret_cast<const uint4*>(weight_row + k / 2));
        uint4 packed1 = __ldg(reinterpret_cast<const uint4*>(weight_row + (k + k_step) / 2));

        uint32_t words0[4] = {packed0.x, packed0.y, packed0.z, packed0.w};
        uint32_t words1[4] = {packed1.x, packed1.y, packed1.z, packed1.w};

        #pragma unroll
        for (int w = 0; w < 4; w++) {
            process_word(words0[w], k + w * 8, x, scale_h2_0, sum_h2, MASK4, SUB_H2);
        }
        #pragma unroll
        for (int w = 0; w < 4; w++) {
            process_word(words1[w], k + k_step + w * 8, x, scale_h2_1, sum_h2, MASK4, SUB_H2);
        }
    }

    // Remainder.
    for (; k < K; k += k_step) {
        float scale_f = __bfloat162float(scales[row * num_groups + k / group_size]);
        half2 scale_h2 = __half2half2(__float2half(scale_f));

        uint4 packed = __ldg(reinterpret_cast<const uint4*>(weight_row + k / 2));
        uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};

        #pragma unroll
        for (int w = 0; w < 4; w++) {
            process_word(words[w], k + w * 8, x, scale_h2, sum_h2, MASK4, SUB_H2);
        }
    }

    half sum_h = __hadd(sum_h2.x, sum_h2.y);
    float sum = __half2float(sum_h);
    sum = warp_reduce_sum(sum);
    if (lane_id == 0)
        output[batch_idx * N + row] = __float2bfloat16(sum);
}

extern "C" cudaError_t w4a16_gemv_batch_cuda_marlin(
    const uint8_t* weight, const __nv_bfloat16* scales,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    int B, int N, int K, int group_size, cudaStream_t stream)
{
    dim3 grid((N + GEMV_ROWS - 1) / GEMV_ROWS, B);
    dim3 block(GEMV_THREADS);
    w4a16_gemv_batch_kernel_marlin<<<grid, block, 0, stream>>>(
        weight, scales, input, output, B, N, K, group_size);
    return cudaGetLastError();
}
