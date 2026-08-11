// BF16 batched GEMM using V100 (sm_70) WMMA tensor cores.
//
// output[N, M] = input[N, K] (bf16) * weight[M, K] (fp16)^T
//
// N = batch, M = output_dim, K = input_dim.
//
// Uses K-splitting to increase occupancy: each block processes a K chunk and
// atomically accumulates its partial sum into a float output buffer, which is
// then converted to bf16.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>

using namespace nvcuda;

#define FP16_WMMA_TILE_N 16
#define FP16_WMMA_TILE_M 16
#define FP16_WMMA_TILE_K 16
#define FP16_WMMA_WARPS 4
#define FP16_WMMA_K_SPLITS 8

__global__ void fp16_gemm_wmma_kernel(
    const half* __restrict__ weight,      // [M, K] fp16 (output_dim, input_dim)
    const __nv_bfloat16* __restrict__ input, // [N, K] bf16 (batch, input_dim)
    float* __restrict__ output,              // [N, M] float accumulator
    int M, int N, int K, int k_chunk_size)
{
    const int warp_id = threadIdx.x / 32;
    const int n0 = (blockIdx.x * FP16_WMMA_WARPS + warp_id) * FP16_WMMA_TILE_N;
    const int m0 = blockIdx.y * FP16_WMMA_TILE_M;
    const int k_start = blockIdx.z * k_chunk_size;
    const int k_end = min(k_start + k_chunk_size, K);
    if (n0 >= M || m0 >= N) return;

    wmma::fragment<wmma::accumulator, FP16_WMMA_TILE_N, FP16_WMMA_TILE_M, FP16_WMMA_TILE_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    __shared__ half w_smem[FP16_WMMA_WARPS * FP16_WMMA_TILE_N * FP16_WMMA_TILE_K];
    __shared__ half x_smem[FP16_WMMA_TILE_M * FP16_WMMA_TILE_K];

    for (int k0 = k_start; k0 < k_end; k0 += FP16_WMMA_TILE_K) {
        const int tid = threadIdx.x;
        const int lane = tid % 32;

        // Load weight tile for this warp.
        half* w_smem_warp = w_smem + warp_id * FP16_WMMA_TILE_N * FP16_WMMA_TILE_K;
        for (int i = lane; i < FP16_WMMA_TILE_N * FP16_WMMA_TILE_K; i += 32) {
            int n = i / FP16_WMMA_TILE_K;
            int k = i % FP16_WMMA_TILE_K;
            half val = __float2half(0.0f);
            if (n0 + n < M && k0 + k < K) {
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
                if (m0 + m < N && k0 + k < K) {
                    val = __float2half(__bfloat162float(input[(size_t)(m0 + m) * K + k0 + k]));
                }
                x_smem[i] = val;
            }
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, FP16_WMMA_TILE_N, FP16_WMMA_TILE_M, FP16_WMMA_TILE_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, FP16_WMMA_TILE_N, FP16_WMMA_TILE_M, FP16_WMMA_TILE_K, half, wmma::col_major> b_frag;
        wmma::load_matrix_sync(a_frag, w_smem_warp, FP16_WMMA_TILE_K);
        wmma::load_matrix_sync(b_frag, x_smem, FP16_WMMA_TILE_K);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }

    // Atomically accumulate partial sum into the float output buffer.
    __shared__ float c_smem[FP16_WMMA_WARPS * FP16_WMMA_TILE_N * FP16_WMMA_TILE_M];
    float* c_smem_warp = c_smem + warp_id * FP16_WMMA_TILE_N * FP16_WMMA_TILE_M;
    wmma::store_matrix_sync(c_smem_warp, c_frag, FP16_WMMA_TILE_M, wmma::mem_row_major);
    __syncthreads();

    const int lane = threadIdx.x % 32;
    for (int i = lane; i < FP16_WMMA_TILE_N * FP16_WMMA_TILE_M; i += 32) {
        const int n_idx = i / FP16_WMMA_TILE_M;
        const int m_idx = i % FP16_WMMA_TILE_M;
        const int n = n0 + n_idx;
        const int m = m0 + m_idx;
        if (n < M && m < N) {
            atomicAdd(&output[(size_t)m * M + n], c_smem_warp[n_idx * FP16_WMMA_TILE_M + m_idx]);
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

// Pre-allocated float accumulator buffer for K-split reduction. Sized to the
// largest [N, M] output seen; grows monotonically so the decode loop (which is
// CUDA-graph captured) never triggers a cudaMalloc mid-step.
static float* g_out_fp32 = nullptr;
static size_t g_out_fp32_elems = 0;

static cudaError_t ensure_out_fp32(size_t elems) {
    if (elems <= g_out_fp32_elems) return cudaSuccess;
    if (g_out_fp32) cudaFree(g_out_fp32);
    cudaError_t err = cudaMalloc(&g_out_fp32, elems * sizeof(float));
    if (err == cudaSuccess) g_out_fp32_elems = elems;
    return err;
}

extern "C" cudaError_t fp16_gemm_wmma_cuda(
    const half* weight,
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    int M, int N, int K,
    cudaStream_t stream)
{
    const int k_chunk_size = (K + FP16_WMMA_K_SPLITS - 1) / FP16_WMMA_K_SPLITS;
    const size_t out_elems = (size_t)N * M;

    cudaError_t err = ensure_out_fp32(out_elems);
    if (err != cudaSuccess) return err;
    err = cudaMemsetAsync(g_out_fp32, 0, out_elems * sizeof(float), stream);
    if (err != cudaSuccess) return err;

    dim3 grid((M + FP16_WMMA_WARPS * FP16_WMMA_TILE_N - 1) / (FP16_WMMA_WARPS * FP16_WMMA_TILE_N),
              (N + FP16_WMMA_TILE_M - 1) / FP16_WMMA_TILE_M,
              FP16_WMMA_K_SPLITS);
    dim3 block(FP16_WMMA_WARPS * 32);
    fp16_gemm_wmma_kernel<<<grid, block, 0, stream>>>(
        weight, input, g_out_fp32, M, N, K, k_chunk_size);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    // Convert float accumulator to bf16.
    constexpr int CONV_BLOCK = 256;
    convert_fp32_to_bf16_kernel<<<(out_elems + CONV_BLOCK - 1) / CONV_BLOCK, CONV_BLOCK, 0, stream>>>(
        g_out_fp32, output, out_elems);
    return cudaGetLastError();
}
