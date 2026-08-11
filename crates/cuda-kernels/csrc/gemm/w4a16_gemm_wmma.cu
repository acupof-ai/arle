// W4A16 batched GEMM using V100 (sm_70) FP16 WMMA tensor cores.
//
// output[N, M] = weight[N, K] (int4) * input[M, K] (bf16)^T
//
// The weight is dequantized int4→fp16 in shared memory (no global memory write),
// then mma.sync.m16n16k16 does the matmul on tensor cores.
//
// Uses 4 warps per block (4 n-tiles) to share the input tile, and K-splitting
// to increase occupancy.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>

using namespace nvcuda;

#define WMMA_TILE_N 16
#define WMMA_TILE_M 16
#define WMMA_TILE_K 16
#define WMMA_WARPS 4
#define WMMA_K_SPLITS 4

__global__ void w4a16_gemm_wmma_kernel(
    const uint8_t* __restrict__ weight,   // [N, K/2] int4
    const __nv_bfloat16* __restrict__ scales, // [N, ceil(K/group_size)]
    const __nv_bfloat16* __restrict__ input,  // [M, K] bf16
    float* __restrict__ output,                // [M, N] float accumulator
    int M, int N, int K, int group_size, int k_chunk_size)
{
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int n0 = (blockIdx.x * WMMA_WARPS + warp_id) * WMMA_TILE_N;
    const int m0 = blockIdx.y * WMMA_TILE_M;
    const int k_start = blockIdx.z * k_chunk_size;
    const int k_end = min(k_start + k_chunk_size, K);
    if (n0 >= N || m0 >= M) return;

    const int tid = threadIdx.x;
    const int groups_per_row = (K + group_size - 1) / group_size;

    wmma::fragment<wmma::accumulator, WMMA_TILE_N, WMMA_TILE_M, WMMA_TILE_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    __shared__ half w_smem[WMMA_WARPS * WMMA_TILE_N * WMMA_TILE_K];
    __shared__ half x_smem[WMMA_TILE_M * WMMA_TILE_K];

    for (int k0 = k_start; k0 < k_end; k0 += WMMA_TILE_K) {
        // --- Dequant weight tile for this warp ---
        half* w_smem_warp = w_smem + warp_id * WMMA_TILE_N * WMMA_TILE_K;
        {
            const int row = lane / 2;
            const int hf = lane % 2;
            if (row < WMMA_TILE_N) {
                const int n = n0 + row;
                const int g = k0 / group_size;
                const float scale_f = __bfloat162float(scales[n * groups_per_row + g]);
                const uint8_t* wrow = weight + (size_t)n * (K / 2) + k0 / 2;
                half* wout = w_smem_warp + row * WMMA_TILE_K;
                const uint32_t w = *reinterpret_cast<const uint32_t*>(wrow + hf * 4);
                const uint32_t lo = w & 0x0F0F0F0Fu;
                const uint32_t hi = (w >> 4) & 0x0F0F0F0Fu;
                const uint8_t* lob = reinterpret_cast<const uint8_t*>(&lo);
                const uint8_t* hib = reinterpret_cast<const uint8_t*>(&hi);
                const int base = hf * 8;
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    wout[base + i * 2]     = __float2half((float)((int)lob[i] - 8) * scale_f);
                    wout[base + i * 2 + 1] = __float2half((float)((int)hib[i] - 8) * scale_f);
                }
            }
        }

        // --- Load input tile (shared by all warps) ---
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

        // --- WMMA matmul ---
        wmma::fragment<wmma::matrix_a, WMMA_TILE_N, WMMA_TILE_M, WMMA_TILE_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_TILE_N, WMMA_TILE_M, WMMA_TILE_K, half, wmma::col_major> b_frag;
        wmma::load_matrix_sync(a_frag, w_smem_warp, WMMA_TILE_K);
        wmma::load_matrix_sync(b_frag, x_smem, WMMA_TILE_K);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }

    // --- Atomically accumulate partial sum ---
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

static cudaError_t ensure_out_fp32(size_t elems) {
    if (elems <= g_out_fp32_elems) return cudaSuccess;
    if (g_out_fp32) cudaFree(g_out_fp32);
    cudaError_t err = cudaMalloc(&g_out_fp32, elems * sizeof(float));
    if (err == cudaSuccess) g_out_fp32_elems = elems;
    return err;
}

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

    cudaError_t err = ensure_out_fp32(out_elems);
    if (err != cudaSuccess) return err;
    err = cudaMemsetAsync(g_out_fp32, 0, out_elems * sizeof(float), stream);
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
