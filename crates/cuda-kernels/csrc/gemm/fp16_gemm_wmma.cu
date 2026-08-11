// BF16 batched GEMM using V100 (sm_70) WMMA tensor cores.
//
// output[N, M] = input[N, K] (bf16) * weight[M, K] (bf16)^T
//
// N = batch, M = output_dim, K = input_dim.
//
// Used after W4A16→BF16 dequant to avoid the 2× memory blowup of cuBLAS for
// small M (DSpark verify batch). Both inputs are BF16; the kernel converts
// to FP16 on the fly for the WMMA tensor cores.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>

using namespace nvcuda;

#define FP16_WMMA_TILE_N 16
#define FP16_WMMA_TILE_M 16
#define FP16_WMMA_TILE_K 16

__global__ void fp16_gemm_wmma_kernel(
    const __nv_bfloat16* __restrict__ weight, // [M, K] bf16 (output_dim, input_dim)
    const __nv_bfloat16* __restrict__ input,  // [N, K] bf16 (batch, input_dim)
    __nv_bfloat16* __restrict__ output,       // [N, M] bf16 (batch, output_dim)
    int M, int N, int K)
{
    const int n0 = blockIdx.x * FP16_WMMA_TILE_N;  // output dimension (M)
    const int m0 = blockIdx.y * FP16_WMMA_TILE_M;  // batch (N)
    if (n0 >= M || m0 >= N) return;

    wmma::fragment<wmma::accumulator, FP16_WMMA_TILE_N, FP16_WMMA_TILE_M, FP16_WMMA_TILE_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    __shared__ half w_smem[FP16_WMMA_TILE_N * FP16_WMMA_TILE_K];
    __shared__ half x_smem[FP16_WMMA_TILE_M * FP16_WMMA_TILE_K];

    for (int k0 = 0; k0 < K; k0 += FP16_WMMA_TILE_K) {
        const int tid = threadIdx.x;

        // Load weight tile and convert bf16->fp16.
        for (int i = tid; i < FP16_WMMA_TILE_N * FP16_WMMA_TILE_K; i += blockDim.x) {
            int n = i / FP16_WMMA_TILE_K;
            int k = i % FP16_WMMA_TILE_K;
            half val = __float2half(0.0f);
            if (n0 + n < M) {
                val = __float2half(__bfloat162float(weight[(size_t)(n0 + n) * K + k0 + k]));
            }
            w_smem[i] = val;
        }

        // Load input tile and convert bf16->fp16.
        for (int i = tid; i < FP16_WMMA_TILE_M * FP16_WMMA_TILE_K; i += blockDim.x) {
            int m = i / FP16_WMMA_TILE_K;
            int k = i % FP16_WMMA_TILE_K;
            half val = __float2half(0.0f);
            if (m0 + m < N) {
                val = __float2half(__bfloat162float(input[(size_t)(m0 + m) * K + k0 + k]));
            }
            x_smem[i] = val;
        }
        __syncthreads();

        // A = weight tile, row_major: A[n][k] = w_smem[n * K + k]
        wmma::fragment<wmma::matrix_a, FP16_WMMA_TILE_N, FP16_WMMA_TILE_M, FP16_WMMA_TILE_K, half, wmma::row_major> a_frag;
        wmma::load_matrix_sync(a_frag, w_smem, FP16_WMMA_TILE_K);

        // B = input^T, col_major: B[k][m] = x_smem[m * K + k] = input[m][k]
        wmma::fragment<wmma::matrix_b, FP16_WMMA_TILE_N, FP16_WMMA_TILE_M, FP16_WMMA_TILE_K, half, wmma::col_major> b_frag;
        wmma::load_matrix_sync(b_frag, x_smem, FP16_WMMA_TILE_K);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }

    // Store output: output[m_batch, n_out] = C[n_out, m_batch]
    __shared__ float c_smem[FP16_WMMA_TILE_N * FP16_WMMA_TILE_M];
    wmma::store_matrix_sync(c_smem, c_frag, FP16_WMMA_TILE_M, wmma::mem_row_major);
    __syncthreads();

    const int tid = threadIdx.x;
    const int n_idx = tid / FP16_WMMA_TILE_M;
    const int m_idx = tid % FP16_WMMA_TILE_M;
    if (n_idx < FP16_WMMA_TILE_N && m_idx < FP16_WMMA_TILE_M) {
        const int n = n0 + n_idx;  // output dimension
        const int m = m0 + m_idx;  // batch
        if (n < M && m < N) {
            output[(size_t)m * M + n] = __float2bfloat16(c_smem[n_idx * FP16_WMMA_TILE_M + m_idx]);
        }
    }
}

extern "C" cudaError_t fp16_gemm_wmma_cuda(
    const __nv_bfloat16* weight,
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    int M, int N, int K,
    cudaStream_t stream)
{
    dim3 grid((M + FP16_WMMA_TILE_N - 1) / FP16_WMMA_TILE_N,
              (N + FP16_WMMA_TILE_M - 1) / FP16_WMMA_TILE_M);
    dim3 block(32);
    fp16_gemm_wmma_kernel<<<grid, block, 0, stream>>>(weight, input, output, M, N, K);
    return cudaGetLastError();
}
