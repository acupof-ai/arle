// TurboQuant Weight GEMV: fused dequant + GEMV for decode (single token).
//
// Weights stored as TQ packed: per-group (Hadamard-rotated, Lloyd-Max quantized).
// Dequant path per group: unpack → gather centroids → scale by norm →
//   inverse FWHT → sign flip → dot product with input.
//
// Memory-bandwidth advantage: reads 3-bit packed (0.5 bytes/elem) instead of
// 2 bytes/elem (BF16) → ~4x bandwidth reduction. Decode is memory-bound,
// so this directly translates to throughput.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

// ─── In-place FWHT in shared memory (reused from turboquant_fast.cu) ───
__device__ __forceinline__ void tq_fwht_smem(float* smem, int D, int tid) {
    for (int stride = 1; stride < D; stride <<= 1) {
        __syncthreads();
        int pair = tid ^ stride;
        if (pair > tid && pair < D) {
            float a = smem[tid];
            float b = smem[pair];
            smem[tid] = a + b;
            smem[pair] = a - b;
        }
    }
    __syncthreads();
    smem[tid] *= rsqrtf((float)D);
    __syncthreads();
}

// ─── TurboQuant Weight GEMV kernel ───
//
// Grid:  (ceil(N / ROWS_PER_BLOCK), 1)
// Block: (GROUP_SIZE, ROWS_PER_BLOCK)
//
// Each block processes ROWS_PER_BLOCK output rows.
// Within each row, threads cooperate on groups of GROUP_SIZE elements.
// For each group: unpack → centroid gather → scale → iFWHT → sign flip → dot.
//
// Template on GROUP_SIZE (must be power of 2, typically 128).
// 3-bit uses 4-bit nibble packing (2 indices per byte).
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

    // Shared memory: per-row group buffer for FWHT + input cache
    extern __shared__ float smem_pool[];
    // Layout: [ROWS_PER_BLOCK][GROUP_SIZE] for group values
    //       + [GROUP_SIZE] for input cache (shared across rows in block)
    float* group_buf = smem_pool + threadIdx.y * GROUP_SIZE;
    float* x_cache = smem_pool + blockDim.y * GROUP_SIZE;

    float row_dot = 0.0f;

    for (int g = 0; g < num_groups; g++) {
        const int col_base = g * GROUP_SIZE;

        // Load input for this group into shared memory (only one row-thread does it)
        if (threadIdx.y == 0 && tid < GROUP_SIZE && (col_base + tid) < K) {
            x_cache[tid] = __bfloat162float(x[col_base + tid]);
        }
        __syncthreads();

        // Step 1: Unpack index and gather centroid value
        if (tid < GROUP_SIZE && (col_base + tid) < K) {
            const int k = col_base + tid;
            const int byte_idx = k / indices_per_byte;
            const int sub_idx = k % indices_per_byte;
            const uint8_t packed_byte = packed[row * packed_cols + byte_idx];
            int idx = (packed_byte >> (sub_idx * effective_bits)) & mask;

            // Gather centroid and scale by per-group norm
            float norm = __half2float(scales[row * num_groups + g]);
            group_buf[tid] = centroids[idx] * norm;
        } else {
            group_buf[tid] = 0.0f;
        }
        __syncthreads();

        // Step 2: Inverse FWHT (self-inverse, in shared memory)
        tq_fwht_smem(group_buf, GROUP_SIZE, tid);

        // Step 3: Sign flip + dot product with input
        if (tid < GROUP_SIZE && (col_base + tid) < K) {
            const int k = col_base + tid;
            float val = group_buf[tid] * (float)signs[k % K];
            row_dot += val * x_cache[tid];
        }
        __syncthreads();
    }

    // Reduce row_dot across threads in this row (warp shuffle)
    // GROUP_SIZE threads per row; reduce within the row's warp(s)
    for (int offset = GROUP_SIZE / 2; offset > 0; offset >>= 1) {
        row_dot += __shfl_down_sync(0xFFFFFFFF, row_dot, offset);
    }
    // For GROUP_SIZE > 32, need cross-warp reduction via shared memory
    if (GROUP_SIZE > 32) {
        // Use group_buf for cross-warp reduce
        group_buf[tid] = row_dot;
        __syncthreads();
        if (tid < 32) {
            float sum = 0.0f;
            for (int i = tid; i < GROUP_SIZE; i += 32) {
                sum += group_buf[i];
            }
            // Warp reduce the partial sums
            for (int offset = 16; offset > 0; offset >>= 1) {
                sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
            }
            if (tid == 0) {
                y[row] = __float2bfloat16(sum);
            }
        }
    } else {
        if (tid == 0) {
            y[row] = __float2bfloat16(row_dot);
        }
    }
}

// ─── TurboQuant bulk dequant kernel (for prefill workspace) ───
//
// Dequantizes packed weights to BF16 workspace buffer.
// Grid: (num_groups, N)  Block: (GROUP_SIZE)
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
    const int g = blockIdx.x;       // group index
    const int row = blockIdx.y;     // output row
    const int tid = threadIdx.x;    // element within group

    if (row >= N || g >= num_groups || tid >= GROUP_SIZE) return;

    const int col_base = g * GROUP_SIZE;
    const int k = col_base + tid;
    if (k >= K) return;

    const int effective_bits = (bits == 3) ? 4 : bits;
    const int indices_per_byte = 8 / effective_bits;
    const int mask = (1 << effective_bits) - 1;

    // Shared memory for FWHT
    extern __shared__ float smem[];

    // Unpack + centroid gather + scale
    const int byte_idx = k / indices_per_byte;
    const int sub_idx = k % indices_per_byte;
    const uint8_t packed_byte = packed[row * packed_cols + byte_idx];
    int idx = (packed_byte >> (sub_idx * effective_bits)) & mask;

    float norm = __half2float(scales[row * num_groups + g]);
    smem[tid] = centroids[idx] * norm;
    __syncthreads();

    // Inverse FWHT
    tq_fwht_smem(smem, GROUP_SIZE, tid);

    // Sign flip + write output
    float val = smem[tid] * (float)signs[k % K];
    out[row * K + k] = __float2bfloat16(val);
}

// ─── C wrappers ───

extern "C" void turboquant_weight_gemv_cuda(
    const uint8_t* packed, const void* scales, const int8_t* signs,
    const float* centroids, const void* x, void* y,
    int N, int K, int group_size, int packed_cols, int num_groups,
    int bits, cudaStream_t stream
) {
    // ROWS_PER_BLOCK: how many output rows per block
    // Limit by shared memory: (ROWS_PER_BLOCK + 1) * group_size * sizeof(float)
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
