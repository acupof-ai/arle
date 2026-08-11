// W4A16 batched GEMM using V100 (sm_70) FP16 WMMA tensor cores.
//
// output[N, M] = weight[N, K] (int4) * input[M, K] (bf16)^T
//
// The weight is dequantized int4→fp16 in shared memory (no global memory write),
// then mma.sync.m16n16k16 does the matmul on tensor cores. This avoids the
// 2× memory blowup of the dequant-to-global + cuBLAS path while getting 8× the
// compute throughput of the int-bit-bashing CUDA-core GEMV.
//
// Only used for M >= 2 (the DSpark verify batch); M=1 decode keeps the GEMV.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>

using namespace nvcuda;

#define WMMA_TILE_N 16
#define WMMA_TILE_M 16
#define WMMA_TILE_K 16

__global__ void w4a16_gemm_wmma_kernel(
    const uint8_t* __restrict__ weight,   // [N, K/2] int4
    const __nv_bfloat16* __restrict__ scales, // [N, ceil(K/group_size)]
    const __nv_bfloat16* __restrict__ input,  // [M, K] bf16
    __nv_bfloat16* __restrict__ output,       // [N, M] bf16
    int M, int N, int K, int group_size)
{
    const int n0 = blockIdx.x * WMMA_TILE_N;
    const int m0 = blockIdx.y * WMMA_TILE_M;
    if (n0 >= N || m0 >= M) return;

    const int tid = threadIdx.x;
    const int groups_per_row = (K + group_size - 1) / group_size;

    wmma::fragment<wmma::accumulator, WMMA_TILE_N, WMMA_TILE_M, WMMA_TILE_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    // Shared memory for the dequantized weight tile [16, 16] fp16 and the
    // input tile [16, 16] fp16.
    __shared__ half w_smem[WMMA_TILE_N * WMMA_TILE_K];
    __shared__ half x_smem[WMMA_TILE_M * WMMA_TILE_K];

    for (int k0 = 0; k0 < K; k0 += WMMA_TILE_K) {
        // --- Dequant weight tile [n0..n0+16, k0..k0+16] into w_smem ---
        // Each thread handles one row of the tile.
        if (tid < WMMA_TILE_N) {
            const int n = n0 + tid;
            const int g = k0 / group_size;
            const float scale_f = __bfloat162float(scales[n * groups_per_row + g]);
            const uint8_t* wrow = weight + (size_t)n * (K / 2) + k0 / 2;
            half* wout = w_smem + tid * WMMA_TILE_K;
            // 16 int4 values = 8 bytes. Load as 2 uint32.
            const uint32_t w0 = *reinterpret_cast<const uint32_t*>(wrow);
            const uint32_t w1 = *reinterpret_cast<const uint32_t*>(wrow + 4);
            // Unpack 8 nibbles from w0.
            const uint32_t lo0 = w0 & 0x0F0F0F0Fu;
            const uint32_t hi0 = (w0 >> 4) & 0x0F0F0F0Fu;
            const uint32_t lo1 = w1 & 0x0F0F0F0Fu;
            const uint32_t hi1 = (w1 >> 4) & 0x0F0F0F0Fu;
            // Dequant: extract nibbles, subtract 8, multiply by scale.
            // Use the same int->float path as the scalar GEMV for bit-exact
            // dequant values; the Marlin OR-with-0x6400 trick is equivalent but
            // we keep the obvious form here while debugging the WMMA path.
            const uint8_t* lo0b = reinterpret_cast<const uint8_t*>(&lo0);
            const uint8_t* hi0b = reinterpret_cast<const uint8_t*>(&hi0);
            const uint8_t* lo1b = reinterpret_cast<const uint8_t*>(&lo1);
            const uint8_t* hi1b = reinterpret_cast<const uint8_t*>(&hi1);
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                float w0 = (float)((int)lo0b[i] - 8) * scale_f;
                float w1 = (float)((int)hi0b[i] - 8) * scale_f;
                wout[i * 2]     = __float2half(w0);
                wout[i * 2 + 1] = __float2half(w1);
            }
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                float w0 = (float)((int)lo1b[i] - 8) * scale_f;
                float w1 = (float)((int)hi1b[i] - 8) * scale_f;
                wout[8 + i * 2]     = __float2half(w0);
                wout[8 + i * 2 + 1] = __float2half(w1);
            }
        }
        __syncthreads();

        // --- Load input tile [m0..m0+16, k0..k0+16], transpose to [K, M] ---
        // Zero the whole tile first so padding rows (m >= M) are 0.
        for (int i = tid; i < WMMA_TILE_M * WMMA_TILE_K; i += blockDim.x) {
            x_smem[i] = __float2half(0.0f);
        }
        __syncthreads();
        if (tid < WMMA_TILE_M) {
            const int m = m0 + tid;
            if (m < M) {
                const __nv_bfloat16* xrow = input + (size_t)m * K + k0;
                // Store transposed: x_smem[k * WMMA_TILE_M + m_idx] = input[m, k].
                for (int i = 0; i < WMMA_TILE_K; i++) {
                    x_smem[i * WMMA_TILE_M + tid] = __float2half(__bfloat162float(xrow[i]));
                }
            }
        }
        __syncthreads();

        // --- WMMA matmul ---
        wmma::fragment<wmma::matrix_a, WMMA_TILE_N, WMMA_TILE_M, WMMA_TILE_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_TILE_N, WMMA_TILE_M, WMMA_TILE_K, half, wmma::row_major> b_frag;
        wmma::load_matrix_sync(a_frag, w_smem, WMMA_TILE_K);
        wmma::load_matrix_sync(b_frag, x_smem, WMMA_TILE_M);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }

    // --- Store output tile ---
    // output[batch, output_dim] = C[output_dim, batch]
    __shared__ float c_smem[WMMA_TILE_N * WMMA_TILE_M];
    wmma::store_matrix_sync(c_smem, c_frag, WMMA_TILE_M, wmma::mem_row_major);
    __syncthreads();

    // Each thread stores multiple elements to cover the full 16x16 tile.
    for (int i = tid; i < WMMA_TILE_N * WMMA_TILE_M; i += blockDim.x) {
        const int n_idx = i / WMMA_TILE_M;  // output_dim
        const int m_idx = i % WMMA_TILE_M;  // batch
        const int n = n0 + n_idx;
        const int m = m0 + m_idx;
        if (n < N && m < M) {
            output[(size_t)m * N + n] = __float2bfloat16(c_smem[n_idx * WMMA_TILE_M + m_idx]);
        }
    }
}

extern "C" cudaError_t w4a16_gemm_wmma_cuda(
    const uint8_t* weight,
    const __nv_bfloat16* scales,
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    int M, int N, int K, int group_size,
    cudaStream_t stream)
{
    dim3 grid((N + WMMA_TILE_N - 1) / WMMA_TILE_N,
              (M + WMMA_TILE_M - 1) / WMMA_TILE_M);
    dim3 block(32);  // one warp for WMMA; 16 threads dequant weight rows
    w4a16_gemm_wmma_kernel<<<grid, block, 0, stream>>>(
        weight, scales, input, output, M, N, K, group_size);
    return cudaGetLastError();
}
