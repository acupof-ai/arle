// BF16 batched GEMM using V100 (sm_70) WMMA tensor cores.
//
// output[N, M] = input[N, K] (bf16) * weight[M, K] (fp16)^T
//
// N = batch, M = output_dim, K = input_dim.
//
// Used after W4A16→BF16→FP16 dequant to avoid the 2× memory blowup of cuBLAS
// for small M (DSpark verify batch). The weight is already in FP16, so this
// kernel only does the matmul on tensor cores.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>

using namespace nvcuda;

#define FP16_WMMA_TILE_N 16
#define FP16_WMMA_TILE_M 16
#define FP16_WMMA_TILE_K 16
#define FP16_WMMA_WARPS 4

__global__ void fp16_gemm_wmma_kernel(
    const half* __restrict__ weight,      // [M, K] fp16 (output_dim, input_dim)
    const __nv_bfloat16* __restrict__ input, // [N, K] bf16 (batch, input_dim)
    __nv_bfloat16* __restrict__ output,      // [N, M] bf16 (batch, output_dim)
    int M, int N, int K)
{
    const int warp_id = threadIdx.x / 32;
    const int n0 = (blockIdx.x * FP16_WMMA_WARPS + warp_id) * FP16_WMMA_TILE_N;
    const int m0 = blockIdx.y * FP16_WMMA_TILE_M;
    if (n0 >= M || m0 >= N) return;

    wmma::fragment<wmma::accumulator, FP16_WMMA_TILE_N, FP16_WMMA_TILE_M, FP16_WMMA_TILE_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    __shared__ half w_smem[FP16_WMMA_WARPS * FP16_WMMA_TILE_N * FP16_WMMA_TILE_K];
    __shared__ half x_smem[FP16_WMMA_TILE_M * FP16_WMMA_TILE_K];

    for (int k0 = 0; k0 < K; k0 += FP16_WMMA_TILE_K) {
        const int tid = threadIdx.x;
        const int lane = tid % 32;

        // Load weight tile for this warp (256 elements, 32 threads).
        half* w_smem_warp = w_smem + warp_id * FP16_WMMA_TILE_N * FP16_WMMA_TILE_K;
        for (int i = lane; i < FP16_WMMA_TILE_N * FP16_WMMA_TILE_K; i += 32) {
            int n = i / FP16_WMMA_TILE_K;
            int k = i % FP16_WMMA_TILE_K;
            half val = __float2half(0.0f);
            if (n0 + n < M) {
                val = weight[(size_t)(n0 + n) * K + k0 + k];
            }
            w_smem_warp[i] = val;
        }

        // Load input tile (shared by all warps in the block).
        if (warp_id == 0) {
            for (int i = lane; i < FP16_WMMA_TILE_M * FP16_WMMA_TILE_K; i += 32) {
                int m = i / FP16_WMMA_TILE_K;
                int k = i % FP16_WMMA_TILE_K;
                half val = __float2half(0.0f);
                if (m0 + m < N) {
                    val = __float2half(__bfloat162float(input[(size_t)(m0 + m) * K + k0 + k]));
                }
                x_smem[i] = val;
            }
        }
        __syncthreads();

        // A = weight tile, row_major.
        wmma::fragment<wmma::matrix_a, FP16_WMMA_TILE_N, FP16_WMMA_TILE_M, FP16_WMMA_TILE_K, half, wmma::row_major> a_frag;
        wmma::load_matrix_sync(a_frag, w_smem_warp, FP16_WMMA_TILE_K);

        // B = input^T, col_major.
        wmma::fragment<wmma::matrix_b, FP16_WMMA_TILE_N, FP16_WMMA_TILE_M, FP16_WMMA_TILE_K, half, wmma::col_major> b_frag;
        wmma::load_matrix_sync(b_frag, x_smem, FP16_WMMA_TILE_K);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }

    // Store output: output[m_batch, n_out] = C[n_out, m_batch]
    __shared__ float c_smem[FP16_WMMA_WARPS * FP16_WMMA_TILE_N * FP16_WMMA_TILE_M];
    float* c_smem_warp = c_smem + warp_id * FP16_WMMA_TILE_N * FP16_WMMA_TILE_M;
    wmma::store_matrix_sync(c_smem_warp, c_frag, FP16_WMMA_TILE_M, wmma::mem_row_major);
    __syncthreads();

    // Each thread stores multiple elements to cover the full 16x16 tile.
    const int lane = threadIdx.x % 32;
    for (int i = lane; i < FP16_WMMA_TILE_N * FP16_WMMA_TILE_M; i += 32) {
        const int n_idx = i / FP16_WMMA_TILE_M;
        const int m_idx = i % FP16_WMMA_TILE_M;
        const int n = n0 + n_idx;
        const int m = m0 + m_idx;
        if (n < M && m < N) {
            output[(size_t)m * M + n] = __float2bfloat16(c_smem_warp[n_idx * FP16_WMMA_TILE_M + m_idx]);
        }
    }
}

extern "C" cudaError_t fp16_gemm_wmma_cuda(
    const half* weight,
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    int M, int N, int K,
    cudaStream_t stream)
{
    dim3 grid((M + FP16_WMMA_WARPS * FP16_WMMA_TILE_N - 1) / (FP16_WMMA_WARPS * FP16_WMMA_TILE_N),
              (N + FP16_WMMA_TILE_M - 1) / FP16_WMMA_TILE_M);
    dim3 block(FP16_WMMA_WARPS * 32);
    fp16_gemm_wmma_kernel<<<grid, block, 0, stream>>>(weight, input, output, M, N, K);
    return cudaGetLastError();
}
