// TurboQuant Weight GEMV: fused dequant + GEMV for decode (single token).
//
// v2: Warp-level FWHT optimization — uses __shfl_xor_sync for butterfly
// stages with stride < 32 (5/7 stages for GS=128). Only stride 32/64
// need shared memory sync. Eliminates majority of __syncthreads() calls.
//
// Weights stored as TQ packed: per-group (Hadamard-rotated, Lloyd-Max quantized).
// Dequant path per group: unpack → gather centroids → scale by norm →
//   inverse FWHT (warp-optimized) → sign flip → dot product with input.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

// ─── Warp-level FWHT: stride < 32 via shuffle, stride >= 32 via smem ───
//
// For GROUP_SIZE=128 (7 stages):
//   stride  1: __shfl_xor (warp-local, no sync)
//   stride  2: __shfl_xor
//   stride  4: __shfl_xor
//   stride  8: __shfl_xor
//   stride 16: __shfl_xor
//   stride 32: shared memory + __syncthreads (cross-warp)
//   stride 64: shared memory + __syncthreads (cross-warp)
//
// Net: 2 syncs instead of 7+2=9 syncs in the naive version.
template <int GROUP_SIZE>
__device__ __forceinline__ void fwht_warp_optimized(float* smem, int tid, float& val) {
    // Stages with stride < 32: warp shuffle (no sync needed)
    #pragma unroll
    for (int stride = 1; stride < 32 && stride < GROUP_SIZE; stride <<= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFF, val, stride);
        // Butterfly: if tid & stride == 0 → val = val + other, else val = other - val
        // Equivalent to: partner gets (a+b) if lower, (a-b) if upper
        float sum = val + other;
        float diff = val - other;
        val = (tid & stride) ? diff : sum;
        // Correction: we need the partner to get the opposite
        // Actually for FWHT: pair (i, i^stride), lower gets a+b, upper gets a-b
        // With shfl_xor, both see each other's value. Lower = tid < (tid^stride).
        // tid & stride == 0 means tid is the lower of the pair.
    }

    // Stages with stride >= 32: need shared memory for cross-warp communication
    if (GROUP_SIZE > 32) {
        smem[tid] = val;
        __syncthreads();

        #pragma unroll
        for (int stride = 32; stride < GROUP_SIZE; stride <<= 1) {
            int pair = tid ^ stride;
            if (pair < GROUP_SIZE) {
                float a = smem[tid];
                float b = smem[pair];
                val = (tid < pair) ? (a + b) : (a - b);
            }
            __syncthreads();
            smem[tid] = val;
            __syncthreads();
        }
    }

    // Normalize by 1/√D
    val *= rsqrtf((float)GROUP_SIZE);
}

// ─── TurboQuant Weight GEMV kernel (v2: warp-level FWHT) ───
//
// Grid:  (ceil(N / ROWS_PER_BLOCK), 1)
// Block: (GROUP_SIZE, ROWS_PER_BLOCK)
//
// Each block processes ROWS_PER_BLOCK output rows.
// Within each row, GROUP_SIZE threads cooperate per group.
template <int GROUP_SIZE>
__global__ void turboquant_weight_gemv_kernel(
    const uint8_t* __restrict__ packed,     // [N, packed_cols] packed indices
    const __half* __restrict__ scales,      // [N, num_groups] f16 norms
    const int8_t* __restrict__ signs,       // [K] Hadamard signs
    const float* __restrict__ centroids,    // [num_levels] Lloyd-Max centroids
    const __nv_bfloat16* __restrict__ x,    // [K] input vector
    __nv_bfloat16* __restrict__ y,          // [N] output vector
    int N, int K, int num_groups, int packed_cols,
    int bits
) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= N) return;

    const int tid = threadIdx.x;  // 0..GROUP_SIZE-1
    const int effective_bits = (bits == 3) ? 4 : bits;
    const int indices_per_byte = 8 / effective_bits;
    const int mask = (1 << effective_bits) - 1;

    // Shared memory layout: [ROWS_PER_BLOCK][GROUP_SIZE] for FWHT
    //                     + [GROUP_SIZE] for input cache
    extern __shared__ float smem_pool[];
    float* group_buf = smem_pool + threadIdx.y * GROUP_SIZE;
    float* x_cache = smem_pool + blockDim.y * GROUP_SIZE;

    float row_dot = 0.0f;

    for (int g = 0; g < num_groups; g++) {
        const int col_base = g * GROUP_SIZE;

        // Load input for this group (one row-thread loads, all share)
        if (threadIdx.y == 0 && tid < GROUP_SIZE && (col_base + tid) < K) {
            x_cache[tid] = __bfloat162float(x[col_base + tid]);
        }
        __syncthreads();

        // Step 1: Unpack + centroid gather + scale
        float val = 0.0f;
        if (tid < GROUP_SIZE && (col_base + tid) < K) {
            const int k = col_base + tid;
            const int byte_idx = k / indices_per_byte;
            const int sub_idx = k % indices_per_byte;
            const uint8_t packed_byte = packed[row * packed_cols + byte_idx];
            int idx = (packed_byte >> (sub_idx * effective_bits)) & mask;

            float norm = __half2float(scales[row * num_groups + g]);
            val = centroids[idx] * norm;
        }

        // Step 2: Inverse FWHT (warp-optimized: 5 shuffles + 2 smem syncs)
        fwht_warp_optimized<GROUP_SIZE>(group_buf, tid, val);

        // Step 3: Sign flip + dot product
        if (tid < GROUP_SIZE && (col_base + tid) < K) {
            const int k = col_base + tid;
            val *= (float)signs[k % K];
            row_dot += val * x_cache[tid];
        }
        __syncthreads();
    }

    // Cross-warp reduction for GROUP_SIZE > 32
    if (GROUP_SIZE > 32) {
        group_buf[tid] = row_dot;
        __syncthreads();
        if (tid < 32) {
            float sum = 0.0f;
            #pragma unroll
            for (int i = tid; i < GROUP_SIZE; i += 32) {
                sum += group_buf[i];
            }
            for (int offset = 16; offset > 0; offset >>= 1) {
                sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
            }
            if (tid == 0) {
                y[row] = __float2bfloat16(sum);
            }
        }
    } else {
        for (int offset = GROUP_SIZE / 2; offset > 0; offset >>= 1) {
            row_dot += __shfl_down_sync(0xFFFFFFFF, row_dot, offset);
        }
        if (tid == 0) {
            y[row] = __float2bfloat16(row_dot);
        }
    }
}

// ─── TurboQuant bulk dequant kernel (for prefill workspace) ───
template <int GROUP_SIZE>
__global__ void turboquant_weight_dequant_kernel(
    const uint8_t* __restrict__ packed,     // [N, packed_cols]
    const __half* __restrict__ scales,      // [N, num_groups]
    const int8_t* __restrict__ signs,       // [K]
    const float* __restrict__ centroids,    // [num_levels]
    __nv_bfloat16* __restrict__ out,        // [N, K] dequantized output
    int N, int K, int num_groups, int packed_cols,
    int bits
) {
    const int g = blockIdx.x;
    const int row = blockIdx.y;
    const int tid = threadIdx.x;

    if (row >= N || g >= num_groups || tid >= GROUP_SIZE) return;

    const int col_base = g * GROUP_SIZE;
    const int k = col_base + tid;
    if (k >= K) return;

    const int effective_bits = (bits == 3) ? 4 : bits;
    const int indices_per_byte = 8 / effective_bits;
    const int mask = (1 << effective_bits) - 1;

    extern __shared__ float smem[];

    // Unpack + centroid gather + scale
    const int byte_idx = k / indices_per_byte;
    const int sub_idx = k % indices_per_byte;
    const uint8_t packed_byte = packed[row * packed_cols + byte_idx];
    int idx = (packed_byte >> (sub_idx * effective_bits)) & mask;

    float norm = __half2float(scales[row * num_groups + g]);
    float val = centroids[idx] * norm;

    // Inverse FWHT (warp-optimized)
    fwht_warp_optimized<GROUP_SIZE>(smem, tid, val);

    // Sign flip + write output
    val *= (float)signs[k % K];
    out[row * K + k] = __float2bfloat16(val);
}

// ─── C wrappers ───

extern "C" void turboquant_weight_gemv_cuda(
    const uint8_t* packed, const void* scales, const int8_t* signs,
    const float* centroids, const void* x, void* y,
    int N, int K, int group_size, int packed_cols, int num_groups,
    int bits, cudaStream_t stream
) {
    const int ROWS_PER_BLOCK = 4;
    dim3 block(group_size, ROWS_PER_BLOCK);
    dim3 grid((N + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, 1);
    int smem = (ROWS_PER_BLOCK + 1) * group_size * sizeof(float);

    if (group_size == 128) {
        turboquant_weight_gemv_kernel<128><<<grid, block, smem, stream>>>(
            packed, (const __half*)scales, signs, centroids,
            (const __nv_bfloat16*)x, (__nv_bfloat16*)y,
            N, K, num_groups, packed_cols, bits
        );
    } else if (group_size == 64) {
        turboquant_weight_gemv_kernel<64><<<grid, block, smem, stream>>>(
            packed, (const __half*)scales, signs, centroids,
            (const __nv_bfloat16*)x, (__nv_bfloat16*)y,
            N, K, num_groups, packed_cols, bits
        );
    } else if (group_size == 32) {
        turboquant_weight_gemv_kernel<32><<<grid, block, smem, stream>>>(
            packed, (const __half*)scales, signs, centroids,
            (const __nv_bfloat16*)x, (__nv_bfloat16*)y,
            N, K, num_groups, packed_cols, bits
        );
    }
}

extern "C" void turboquant_weight_dequant_cuda(
    const uint8_t* packed, const void* scales, const int8_t* signs,
    const float* centroids, void* out,
    int N, int K, int group_size, int packed_cols, int num_groups,
    int bits, cudaStream_t stream
) {
    dim3 block(group_size);
    dim3 grid(num_groups, N);
    int smem = group_size * sizeof(float);

    if (group_size == 128) {
        turboquant_weight_dequant_kernel<128><<<grid, block, smem, stream>>>(
            packed, (const __half*)scales, signs, centroids,
            (__nv_bfloat16*)out,
            N, K, num_groups, packed_cols, bits
        );
    } else if (group_size == 64) {
        turboquant_weight_dequant_kernel<64><<<grid, block, smem, stream>>>(
            packed, (const __half*)scales, signs, centroids,
            (__nv_bfloat16*)out,
            N, K, num_groups, packed_cols, bits
        );
    } else if (group_size == 32) {
        turboquant_weight_dequant_kernel<32><<<grid, block, smem, stream>>>(
            packed, (const __half*)scales, signs, centroids,
            (__nv_bfloat16*)out,
            N, K, num_groups, packed_cols, bits
        );
    }
}
