// W4A16 batched GEMM using V100 (sm_70) FP16 WMMA tensor cores.
//
// output[N, M] = weight[N, K] (int4) * input[M, K] (bf16)^T
//
// The weight is dequantized int4->fp16 in shared memory (no global memory write),
// then mma.sync.m16n16k16 does the matmul on tensor cores.
//
// Uses K-splitting to increase occupancy: each block processes a chunk of K,
// and partial results are accumulated via atomicAdd to an fp32 buffer.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>

using namespace nvcuda;

#define WMMA_TILE_N 16
#define WMMA_TILE_M 16
#define WMMA_TILE_K 16
#define WMMA_WARPS 1
#define WMMA_K_SPLITS 16

__global__ void w4a16_gemm_wmma_kernel(
    const uint8_t* __restrict__ weight,
    const __nv_bfloat16* __restrict__ scales,
    const __nv_bfloat16* __restrict__ input,
    float* __restrict__ output,
    int M, int N, int K, int group_size, int k_chunk_size)
{
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int n0 = (blockIdx.x * WMMA_WARPS + warp_id) * WMMA_TILE_N;
    const int m0 = blockIdx.y * WMMA_TILE_M;
    const int k_start = blockIdx.z * k_chunk_size;
    const int k_end = min(k_start + k_chunk_size, K);
    if (n0 >= N || m0 >= M) return;

    const int groups_per_row = (K + group_size - 1) / group_size;

    wmma::fragment<wmma::accumulator, WMMA_TILE_N, WMMA_TILE_M, WMMA_TILE_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    __shared__ half w_smem[WMMA_WARPS * WMMA_TILE_N * WMMA_TILE_K];
    __shared__ half x_smem[WMMA_TILE_M * WMMA_TILE_K];

    // Marlin dequant constants
    const uint32_t MASK4 = 0x0f0f0f0fu;
    const uint32_t SUB   = 0x64086408u;
    const half2 SUB_H2   = *reinterpret_cast<const half2*>(&SUB);

    for (int k0 = k_start; k0 < k_end; k0 += WMMA_TILE_K) {
        half* w_smem_warp = w_smem + warp_id * WMMA_TILE_N * WMMA_TILE_K;
        {
            const int row = lane;
            if (row < WMMA_TILE_N) {
                const int n = n0 + row;
                const int g = k0 / group_size;
                const float scale_f = __bfloat162float(scales[n * groups_per_row + g]);
                half2 scale_h2 = __half2half2(__float2half(scale_f));
                const uint8_t* wrow = weight + (size_t)n * (K / 2) + k0 / 2;
                half* wout = w_smem_warp + row * WMMA_TILE_K;

                uint32_t w0 = *reinterpret_cast<const uint32_t*>(wrow);
                uint32_t w1 = *reinterpret_cast<const uint32_t*>(wrow + 4);

                uint32_t lo0 = w0 & MASK4;
                uint32_t hi0 = (w0 >> 4) & MASK4;

                uint32_t lo01 = (0x6400u | (lo0 & 0xffu)) |
                                ((0x6400u | ((lo0 >> 8) & 0xffu)) << 16);
                uint32_t lo23 = (0x6400u | ((lo0 >> 16) & 0xffu)) |
                                ((0x6400u | ((lo0 >> 24) & 0xffu)) << 16);
                uint32_t hi01 = (0x6400u | (hi0 & 0xffu)) |
                                ((0x6400u | ((hi0 >> 8) & 0xffu)) << 16);
                uint32_t hi23 = (0x6400u | ((hi0 >> 16) & 0xffu)) |
                                ((0x6400u | ((hi0 >> 24) & 0xffu)) << 16);

                half2 w0h = __hsub2(*reinterpret_cast<half2*>(&lo01), SUB_H2);
                half2 w1h = __hsub2(*reinterpret_cast<half2*>(&hi01), SUB_H2);
                half2 w2h = __hsub2(*reinterpret_cast<half2*>(&lo23), SUB_H2);
                half2 w3h = __hsub2(*reinterpret_cast<half2*>(&hi23), SUB_H2);

                half2 wx = __halves2half2(w0h.x, w1h.x);
                half2 wy = __halves2half2(w0h.y, w1h.y);
                half2 wx2 = __halves2half2(w2h.x, w3h.x);
                half2 wy2 = __halves2half2(w2h.y, w3h.y);

                wx = __hmul2(wx, scale_h2);
                wy = __hmul2(wy, scale_h2);
                wx2 = __hmul2(wx2, scale_h2);
                wy2 = __hmul2(wy2, scale_h2);

                *reinterpret_cast<half2*>(&wout[0]) = wx;
                *reinterpret_cast<half2*>(&wout[2]) = wy;
                *reinterpret_cast<half2*>(&wout[4]) = wx2;
                *reinterpret_cast<half2*>(&wout[6]) = wy2;

                uint32_t lo1 = w1 & MASK4;
                uint32_t hi1 = (w1 >> 4) & MASK4;

                lo01 = (0x6400u | (lo1 & 0xffu)) |
                       ((0x6400u | ((lo1 >> 8) & 0xffu)) << 16);
                lo23 = (0x6400u | ((lo1 >> 16) & 0xffu)) |
                       ((0x6400u | ((lo1 >> 24) & 0xffu)) << 16);
                hi01 = (0x6400u | (hi1 & 0xffu)) |
                       ((0x6400u | ((hi1 >> 8) & 0xffu)) << 16);
                hi23 = (0x6400u | ((hi1 >> 16) & 0xffu)) |
                       ((0x6400u | ((hi1 >> 24) & 0xffu)) << 16);

                w0h = __hsub2(*reinterpret_cast<half2*>(&lo01), SUB_H2);
                w1h = __hsub2(*reinterpret_cast<half2*>(&hi01), SUB_H2);
                w2h = __hsub2(*reinterpret_cast<half2*>(&lo23), SUB_H2);
                w3h = __hsub2(*reinterpret_cast<half2*>(&hi23), SUB_H2);

                wx = __halves2half2(w0h.x, w1h.x);
                wy = __halves2half2(w0h.y, w1h.y);
                wx2 = __halves2half2(w2h.x, w3h.x);
                wy2 = __halves2half2(w2h.y, w3h.y);

                wx = __hmul2(wx, scale_h2);
                wy = __hmul2(wy, scale_h2);
                wx2 = __hmul2(wx2, scale_h2);
                wy2 = __hmul2(wy2, scale_h2);

                *reinterpret_cast<half2*>(&wout[8]) = wx;
                *reinterpret_cast<half2*>(&wout[10]) = wy;
                *reinterpret_cast<half2*>(&wout[12]) = wx2;
                *reinterpret_cast<half2*>(&wout[14]) = wy2;
            }
        }

        if (warp_id == 0) {
            for (int i = lane; i < WMMA_TILE_M * WMMA_TILE_K; i += 32) {
                int m = i / WMMA_TILE_K;
                int k = i % WMMA_TILE_K;
                half val = __float2half(0.0f);
                if (m0 + m < M) {
                    val = __float2half(__bfloat162float(input[(size_t)(m0 + m) * K + k0 + k]));
                }
                x_smem[i] = val;
            }
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, WMMA_TILE_N, WMMA_TILE_M, WMMA_TILE_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_TILE_N, WMMA_TILE_M, WMMA_TILE_K, half, wmma::col_major> b_frag;
        wmma::load_matrix_sync(a_frag, w_smem_warp, WMMA_TILE_K);
        wmma::load_matrix_sync(b_frag, x_smem, WMMA_TILE_K);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }

    // Accumulate partial results via atomicAdd to fp32 output.
    __shared__ float c_smem[WMMA_WARPS * WMMA_TILE_N * WMMA_TILE_M];
    float* c_smem_warp = c_smem + warp_id * WMMA_TILE_N * WMMA_TILE_M;
    wmma::store_matrix_sync(c_smem_warp, c_frag, WMMA_TILE_M, wmma::mem_row_major);
    __syncthreads();

    for (int i = lane; i < WMMA_TILE_N * WMMA_TILE_M; i += 32) {
        const int n_idx = i / WMMA_TILE_M;
        const int m_idx = i % WMMA_TILE_M;
        const int n = n0 + n_idx;
        const int m = m0 + m_idx;
        if (n < N && m < M) {
            atomicAdd(&output[(size_t)m * N + n], c_smem_warp[n_idx * WMMA_TILE_M + m_idx]);
        }
    }
}

__global__ void convert_fp32_to_bf16_kernel(const float* __restrict__ in,
                                             __nv_bfloat16* __restrict__ out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __float2bfloat16(in[i]);
    }
}

static float* g_out_fp32 = nullptr;
static size_t g_out_fp32_elems = 0;

extern "C" cudaError_t w4a16_gemm_wmma_cuda(
    const uint8_t* weight,
    const __nv_bfloat16* scales,
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    int M, int N, int K, int group_size,
    cudaStream_t stream)
{
    const int k_chunk_size = (K + WMMA_K_SPLITS - 1) / WMMA_K_SPLITS;
    const size_t out_elems = (size_t)M * N;

    if (out_elems > g_out_fp32_elems) {
        if (g_out_fp32) cudaFree(g_out_fp32);
        cudaError_t err = cudaMalloc(&g_out_fp32, out_elems * sizeof(float));
        if (err != cudaSuccess) return err;
        g_out_fp32_elems = out_elems;
    }
    cudaError_t err = cudaMemsetAsync(g_out_fp32, 0, out_elems * sizeof(float), stream);
    if (err != cudaSuccess) return err;

    dim3 grid((N + WMMA_WARPS * WMMA_TILE_N - 1) / (WMMA_WARPS * WMMA_TILE_N),
              (M + WMMA_TILE_M - 1) / WMMA_TILE_M,
              WMMA_K_SPLITS);
    dim3 block(WMMA_WARPS * 32);
    w4a16_gemm_wmma_kernel<<<grid, block, 0, stream>>>(
        weight, scales, input, g_out_fp32, M, N, K, group_size, k_chunk_size);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    constexpr int CONV_BLOCK = 256;
    convert_fp32_to_bf16_kernel<<<(out_elems + CONV_BLOCK - 1) / CONV_BLOCK, CONV_BLOCK, 0, stream>>>(
        g_out_fp32, output, out_elems);
    return cudaGetLastError();
}
