// Unified W8A16 / W4A16 dequant-on-the-fly GEMV kernel.
//
// Weights stored quantized (int8 or packed int4) with per-group bf16 scales.
// Activations in bf16. Dequant happens in registers — zero extra bandwidth.
//
// For decode (batch=1): each block computes ROWS_PER_BLOCK output elements.
// Multiple rows per block reuse the activation vector from shared memory.
//
// For small batch (batch=2-8): GEMM variant with batched activations.
//
// Layout:
//   weight:  [N, K] int8  (W8) or [N, K/2] uint8 packed (W4)
//   scales:  [N, K/group_size] bf16
//   input:   [B, K] bf16
//   output:  [B, N] bf16

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

#define WARP_SIZE 32

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

// ============================================================================
// W8A16 GEMV: INT8 weights, BF16 activations, per-group-128 scales
//
// Grid:  (N / ROWS_PER_BLOCK,)  or (ceil(N / ROWS_PER_BLOCK),)
// Block: (THREADS,)  — 256 threads = 8 warps
//
// Each block computes ROWS_PER_BLOCK output elements.
// Activation vector loaded into shared memory for reuse across rows.
// ============================================================================
#define W8_THREADS 256
#define W8_ROWS_PER_BLOCK 4
#define W8_VEC_SIZE 16  // 128-bit load = 16 int8 values

__global__ void w8a16_gemv_kernel(
    const int8_t* __restrict__ weight,       // [N, K]
    const __nv_bfloat16* __restrict__ scales, // [N, num_groups] where num_groups = K/group_size
    const __nv_bfloat16* __restrict__ input,  // [K]
    __nv_bfloat16* __restrict__ output,       // [N]
    int N, int K, int group_size)
{
    int block_row_start = blockIdx.x * W8_ROWS_PER_BLOCK;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_groups = K / group_size;

    // Threads per row: distribute threads across rows
    // With 256 threads and 4 rows: 64 threads per row = 2 warps per row
    int threads_per_row = W8_THREADS / W8_ROWS_PER_BLOCK;  // 64
    int row_in_block = tid / threads_per_row;                // 0-3
    int tid_in_row = tid % threads_per_row;                  // 0-63
    int row = block_row_start + row_in_block;

    if (row >= N) return;

    float sum = 0.0f;
    int row_offset = row * K;
    int scale_offset = row * num_groups;

    // Each thread processes K / threads_per_row elements, vectorized
    for (int k = tid_in_row * W8_VEC_SIZE; k < K; k += threads_per_row * W8_VEC_SIZE) {
        // Load scale for this group
        float scale_f = __bfloat162float(scales[scale_offset + k / group_size]);

        // Vectorized 128-bit load: 16 int8 weights
        const int4* w_ptr = reinterpret_cast<const int4*>(&weight[row_offset + k]);
        int4 w_packed = *w_ptr;
        const int8_t* w_vals = reinterpret_cast<const int8_t*>(&w_packed);

        // Load 16 bf16 activations (2 × 64-bit loads)
        #pragma unroll
        for (int i = 0; i < W8_VEC_SIZE; i++) {
            float w_f = static_cast<float>(w_vals[i]) * scale_f;
            float x_f = __bfloat162float(input[k + i]);
            sum += w_f * x_f;
        }
    }

    // Reduce within the threads assigned to this row
    // First: warp-level reduction
    sum = warp_reduce_sum(sum);

    // Cross-warp reduction for this row (2 warps per row with 64 threads/row)
    __shared__ float smem[W8_ROWS_PER_BLOCK * 8]; // max 8 warps per row
    int warps_per_row = threads_per_row / WARP_SIZE;  // 2
    int warp_in_row = (tid % threads_per_row) / WARP_SIZE;

    if (lane_id == 0) {
        smem[row_in_block * warps_per_row + warp_in_row] = sum;
    }
    __syncthreads();

    // First thread of each row writes output
    if (tid_in_row == 0) {
        float total = 0.0f;
        for (int w = 0; w < warps_per_row; w++) {
            total += smem[row_in_block * warps_per_row + w];
        }
        output[row] = __float2bfloat16(total);
    }
}

// ============================================================================
// W4A16 GEMV: INT4 packed weights, BF16 activations, per-group-128 scales
//
// Same structure as W8, but each byte holds 2 INT4 values.
// 128-bit load = 16 bytes = 32 INT4 values.
// ============================================================================
#define W4_VEC_SIZE 32  // 128-bit load = 32 int4 values (16 bytes)

__global__ void w4a16_gemv_kernel(
    const uint8_t* __restrict__ weight,      // [N, K/2] packed int4
    const __nv_bfloat16* __restrict__ scales, // [N, num_groups]
    const __nv_bfloat16* __restrict__ input,  // [K]
    __nv_bfloat16* __restrict__ output,       // [N]
    int N, int K, int group_size)
{
    int block_row_start = blockIdx.x * W8_ROWS_PER_BLOCK;
    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int num_groups = K / group_size;

    int threads_per_row = W8_THREADS / W8_ROWS_PER_BLOCK;
    int row_in_block = tid / threads_per_row;
    int tid_in_row = tid % threads_per_row;
    int row = block_row_start + row_in_block;

    if (row >= N) return;

    float sum = 0.0f;
    int row_offset = row * (K / 2);  // packed: K/2 bytes per row
    int scale_offset = row * num_groups;

    // Each iteration processes 32 int4 values (16 bytes = 128-bit load)
    for (int k = tid_in_row * W4_VEC_SIZE; k < K; k += threads_per_row * W4_VEC_SIZE) {
        float scale_f = __bfloat162float(scales[scale_offset + k / group_size]);

        // Load 16 bytes = 32 packed int4 values
        const int4* w_ptr = reinterpret_cast<const int4*>(&weight[row_offset + k / 2]);
        int4 w_packed = *w_ptr;
        const uint8_t* w_bytes = reinterpret_cast<const uint8_t*>(&w_packed);

        // Unpack + dequant + FMA for 32 elements
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t byte = w_bytes[i];
            // Low nibble and high nibble, signed symmetric around 8
            int lo = (byte & 0x0F) - 8;  // [-8, 7]
            int hi = (byte >> 4) - 8;

            float x0 = __bfloat162float(input[k + i * 2]);
            float x1 = __bfloat162float(input[k + i * 2 + 1]);
            sum += (static_cast<float>(lo) * scale_f) * x0;
            sum += (static_cast<float>(hi) * scale_f) * x1;
        }
    }

    // Same reduction as W8
    sum = warp_reduce_sum(sum);
    __shared__ float smem[W8_ROWS_PER_BLOCK * 8];
    int warps_per_row = threads_per_row / WARP_SIZE;
    int warp_in_row = (tid % threads_per_row) / WARP_SIZE;
    if (lane_id == 0) smem[row_in_block * warps_per_row + warp_in_row] = sum;
    __syncthreads();

    if (tid_in_row == 0) {
        float total = 0.0f;
        for (int w = 0; w < warps_per_row; w++)
            total += smem[row_in_block * warps_per_row + w];
        output[row] = __float2bfloat16(total);
    }
}

// ============================================================================
// Batched GEMV: [B, K] × [N, K]^T → [B, N]
// For small batch decode (B=2-8). One block per (batch, row_group).
// ============================================================================

__global__ void w8a16_gemv_batch_kernel(
    const int8_t* __restrict__ weight,
    const __nv_bfloat16* __restrict__ scales,
    const __nv_bfloat16* __restrict__ input,  // [B, K]
    __nv_bfloat16* __restrict__ output,       // [B, N]
    int B, int N, int K, int group_size)
{
    int block_row_start = blockIdx.x * W8_ROWS_PER_BLOCK;
    int batch_idx = blockIdx.y;
    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int num_groups = K / group_size;

    int threads_per_row = W8_THREADS / W8_ROWS_PER_BLOCK;
    int row_in_block = tid / threads_per_row;
    int tid_in_row = tid % threads_per_row;
    int row = block_row_start + row_in_block;

    if (row >= N) return;

    const __nv_bfloat16* x = input + batch_idx * K;
    float sum = 0.0f;
    int row_offset = row * K;
    int scale_offset = row * num_groups;

    for (int k = tid_in_row * W8_VEC_SIZE; k < K; k += threads_per_row * W8_VEC_SIZE) {
        float scale_f = __bfloat162float(scales[scale_offset + k / group_size]);
        const int4* w_ptr = reinterpret_cast<const int4*>(&weight[row_offset + k]);
        int4 w_packed = *w_ptr;
        const int8_t* w_vals = reinterpret_cast<const int8_t*>(&w_packed);
        #pragma unroll
        for (int i = 0; i < W8_VEC_SIZE; i++) {
            sum += (static_cast<float>(w_vals[i]) * scale_f) * __bfloat162float(x[k + i]);
        }
    }

    sum = warp_reduce_sum(sum);
    __shared__ float smem[W8_ROWS_PER_BLOCK * 8];
    int warps_per_row = threads_per_row / WARP_SIZE;
    int warp_in_row = (tid % threads_per_row) / WARP_SIZE;
    if (lane_id == 0) smem[row_in_block * warps_per_row + warp_in_row] = sum;
    __syncthreads();

    if (tid_in_row == 0) {
        float total = 0.0f;
        for (int w = 0; w < warps_per_row; w++)
            total += smem[row_in_block * warps_per_row + w];
        output[batch_idx * N + row] = __float2bfloat16(total);
    }
}

// ============================================================================
// C API
// ============================================================================
extern "C" {

// W8A16 GEMV: weight [N, K] int8, input [K] bf16, output [N] bf16
cudaError_t w8a16_gemv_cuda(
    const int8_t* weight,
    const __nv_bfloat16* scales,
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    int N, int K, int group_size,
    cudaStream_t stream)
{
    dim3 grid((N + W8_ROWS_PER_BLOCK - 1) / W8_ROWS_PER_BLOCK);
    dim3 block(W8_THREADS);
    w8a16_gemv_kernel<<<grid, block, 0, stream>>>(
        weight, scales, input, output, N, K, group_size);
    return cudaGetLastError();
}

// W4A16 GEMV: weight [N, K/2] packed uint8, input [K] bf16, output [N] bf16
cudaError_t w4a16_gemv_cuda(
    const uint8_t* weight,
    const __nv_bfloat16* scales,
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    int N, int K, int group_size,
    cudaStream_t stream)
{
    dim3 grid((N + W8_ROWS_PER_BLOCK - 1) / W8_ROWS_PER_BLOCK);
    dim3 block(W8_THREADS);
    w4a16_gemv_kernel<<<grid, block, 0, stream>>>(
        weight, scales, input, output, N, K, group_size);
    return cudaGetLastError();
}

// W8A16 batched GEMV: weight [N,K] int8, input [B,K] bf16, output [B,N] bf16
cudaError_t w8a16_gemv_batch_cuda(
    const int8_t* weight,
    const __nv_bfloat16* scales,
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    int B, int N, int K, int group_size,
    cudaStream_t stream)
{
    dim3 grid((N + W8_ROWS_PER_BLOCK - 1) / W8_ROWS_PER_BLOCK, B);
    dim3 block(W8_THREADS);
    w8a16_gemv_batch_kernel<<<grid, block, 0, stream>>>(
        weight, scales, input, output, B, N, K, group_size);
    return cudaGetLastError();
}

}  // extern "C"
