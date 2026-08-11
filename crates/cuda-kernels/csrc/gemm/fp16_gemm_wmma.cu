// FP16 batched GEMM using V100 (sm_70) WMMA tensor cores.
//
// output[N, M] = input[N, K] (bf16) * weight[M, K] (fp16)^T
//
// N = batch, M = output_dim, K = input_dim.
//
// Used after W4A16→FP16 dequant to avoid the 2× memory blowup of cuBLAS for
// small M (DSpark verify batch). The weight is already in FP16, so this kernel
// only does the matmul on tensor cores.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>

using namespace nvcuda;

#define FP16_WMMA_TILE_N 16
#define FP16_WMMA_TILE_M 16
#define FP16_WMMA_TILE_K 16

// Simple FP32 GEMM for debugging: output[m][n] = sum_k weight[n][k] * input[m][k]
__global__ void fp16_gemm_debug_kernel(
    const half* __restrict__ weight,      // [M, K] fp16
    const __nv_bfloat16* __restrict__ input, // [N, K] bf16
    __nv_bfloat16* __restrict__ output,      // [N, M] bf16
    int M, int N, int K)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;  // output dim
    int m = blockIdx.y;  // batch
    if (n >= M || m >= N) return;

    float sum = 0.0f;
    const half* wrow = weight + (size_t)n * K;
    const __nv_bfloat16* xrow = input + (size_t)m * K;
    for (int k = 0; k < K; k++) {
        sum += __half2float(wrow[k]) * __bfloat162float(xrow[k]);
    }
    output[(size_t)m * M + n] = __float2bfloat16(sum);
}

__global__ void fp16_gemm_wmma_kernel(
    const half* __restrict__ weight,      // [M, K] fp16 (output_dim, input_dim)
    const __nv_bfloat16* __restrict__ input, // [N, K] bf16 (batch, input_dim)
    __nv_bfloat16* __restrict__ output,      // [N, M] bf16 (batch, output_dim)
    int M, int N, int K)
{
    const int n0 = blockIdx.x * FP16_WMMA_TILE_N;  // output dimension (M)
    const int m0 = blockIdx.y * FP16_WMMA_TILE_M;  // batch (N)
    if (n0 >= M || m0 >= N) return;

    const int tid = threadIdx.x;

    wmma::fragment<wmma::accumulator, FP16_WMMA_TILE_N, FP16_WMMA_TILE_M, FP16_WMMA_TILE_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    __shared__ half w_smem[FP16_WMMA_TILE_N * FP16_WMMA_TILE_K];
    __shared__ half x_smem[FP16_WMMA_TILE_K * FP16_WMMA_TILE_M];

    for (int k0 = 0; k0 < K; k0 += FP16_WMMA_TILE_K) {
        // Load weight tile [n0..n0+16, k0..k0+16] fp16.
        if (tid < FP16_WMMA_TILE_N) {
            const int n = n0 + tid;
            half* wout = w_smem + tid * FP16_WMMA_TILE_K;
            if (n < M) {
                const half* wrow = weight + (size_t)n * K + k0;
                #pragma unroll
                for (int i = 0; i < FP16_WMMA_TILE_K; i++) {
                    wout[i] = wrow[i];
                }
            } else {
                #pragma unroll
                for (int i = 0; i < FP16_WMMA_TILE_K; i++) {
                    wout[i] = __float2half(0.0f);
                }
            }
        }
        __syncthreads();

        // Load input tile [m0..m0+16, k0..k0+16] directly (no transpose).
        // Zero the whole tile first so padding rows (m >= N) are 0.
        for (int i = tid; i < FP16_WMMA_TILE_M * FP16_WMMA_TILE_K; i += blockDim.x) {
            x_smem[i] = __float2half(0.0f);
        }
        __syncthreads();
        if (tid < FP16_WMMA_TILE_M) {
            const int m = m0 + tid;
            if (m < N) {
                const __nv_bfloat16* xrow = input + (size_t)m * K + k0;
                for (int i = 0; i < FP16_WMMA_TILE_K; i++) {
                    x_smem[tid * FP16_WMMA_TILE_K + i] = __float2half(__bfloat162float(xrow[i]));
                }
            }
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, FP16_WMMA_TILE_N, FP16_WMMA_TILE_M, FP16_WMMA_TILE_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, FP16_WMMA_TILE_N, FP16_WMMA_TILE_M, FP16_WMMA_TILE_K, half, wmma::col_major> b_frag;
        wmma::load_matrix_sync(a_frag, w_smem, FP16_WMMA_TILE_K);
        wmma::load_matrix_sync(b_frag, x_smem, FP16_WMMA_TILE_K);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }

    // Store output as [N, M] row-major (N=batch, M=output_dim).
    // Element [m_batch, n_out] is at m_batch * M + n_out.
    __shared__ float c_smem[FP16_WMMA_TILE_N * FP16_WMMA_TILE_M];
    wmma::store_matrix_sync(c_smem, c_frag, FP16_WMMA_TILE_M, wmma::mem_row_major);
    __syncthreads();

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
    const half* weight,
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    int M, int N, int K,
    cudaStream_t stream)
{
    // Debug: use FP32 GEMM to verify dequantized weights.
    dim3 grid((M + 255) / 256, N);
    dim3 block(256);
    fp16_gemm_debug_kernel<<<grid, block, 0, stream>>>(weight, input, output, M, N, K);
    return cudaGetLastError();
}
