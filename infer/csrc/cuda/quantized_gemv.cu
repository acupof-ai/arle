// Unified W2/W4/W8 A16 dequant-on-the-fly GEMV kernel.
//
// Nibble extraction uses parallel bitmask on uint32 (like llama.cpp/vLLM),
// NOT per-element shift/mask or pointer aliasing on register variables.
//
// W8: signed int8, no zero-point. Direct cast to float.
// W4: unsigned nibbles, zero-point=8. Parallel extract via 0x0F0F0F0F mask.
// W2: unsigned 2-bit, zero-point=2. Extract via 0x03030303 mask.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

#define WARP_SIZE 32
#define GEMV_THREADS 256
#define GEMV_ROWS 4

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

// ============================================================================
// W8A16 GEMV: signed INT8 weights, BF16 activations.
// Each uint32 = 4 signed int8 values. No zero-point.
// ============================================================================
__global__ void w8a16_gemv_kernel(
    const uint8_t* __restrict__ weight,  // [N, K] int8
    const __nv_bfloat16* __restrict__ scales,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int N, int K, int group_size)
{
    int row = blockIdx.x * GEMV_ROWS + threadIdx.x / (GEMV_THREADS / GEMV_ROWS);
    int tid_in_row = threadIdx.x % (GEMV_THREADS / GEMV_ROWS);
    int threads_per_row = GEMV_THREADS / GEMV_ROWS;
    int lane_id = threadIdx.x % WARP_SIZE;
    int row_in_block = threadIdx.x / threads_per_row;

    if (row >= N) return;

    float sum = 0.0f;
    int num_groups = K / group_size;

    // Process 4 int8 elements per iteration (one uint32)
    for (int k = tid_in_row * 4; k < K; k += threads_per_row * 4) {
        float scale_f = __bfloat162float(scales[row * num_groups + k / group_size]);

        // Load 4 bytes as uint32
        uint32_t packed = *reinterpret_cast<const uint32_t*>(&weight[row * K + k]);

        // Extract 4 signed int8 values via byte shifts
        int8_t v0 = static_cast<int8_t>(packed);
        int8_t v1 = static_cast<int8_t>(packed >> 8);
        int8_t v2 = static_cast<int8_t>(packed >> 16);
        int8_t v3 = static_cast<int8_t>(packed >> 24);

        sum += static_cast<float>(v0) * scale_f * __bfloat162float(input[k]);
        sum += static_cast<float>(v1) * scale_f * __bfloat162float(input[k + 1]);
        sum += static_cast<float>(v2) * scale_f * __bfloat162float(input[k + 2]);
        sum += static_cast<float>(v3) * scale_f * __bfloat162float(input[k + 3]);
    }

    // Warp + cross-warp reduction
    sum = warp_reduce_sum(sum);
    __shared__ float smem[GEMV_ROWS * 8];
    int warps_per_row = threads_per_row / WARP_SIZE;
    int warp_in_row = (threadIdx.x % threads_per_row) / WARP_SIZE;
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
// W4A16 GEMV: packed INT4 weights, BF16 activations.
// Each uint32 = 8 unsigned nibbles. Zero-point = 8.
// Parallel nibble extract via 0x0F0F0F0F bitmask (llama.cpp pattern).
// ============================================================================
__global__ void w4a16_gemv_kernel(
    const uint8_t* __restrict__ weight,  // [N, K/2] packed
    const __nv_bfloat16* __restrict__ scales,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int N, int K, int group_size)
{
    int row = blockIdx.x * GEMV_ROWS + threadIdx.x / (GEMV_THREADS / GEMV_ROWS);
    int tid_in_row = threadIdx.x % (GEMV_THREADS / GEMV_ROWS);
    int threads_per_row = GEMV_THREADS / GEMV_ROWS;
    int lane_id = threadIdx.x % WARP_SIZE;
    int row_in_block = threadIdx.x / threads_per_row;

    if (row >= N) return;

    float sum = 0.0f;
    int num_groups = K / group_size;
    int bytes_per_row = K / 2;

    // Process 8 INT4 elements per iteration (one uint32 = 4 packed bytes)
    for (int k = tid_in_row * 8; k < K; k += threads_per_row * 8) {
        float scale_f = __bfloat162float(scales[row * num_groups + k / group_size]);

        // Load 4 packed bytes as uint32
        uint32_t packed = *reinterpret_cast<const uint32_t*>(&weight[row * bytes_per_row + k / 2]);

        // Parallel nibble extract (llama.cpp pattern):
        // Low nibbles: bytes[0]&0xF, bytes[1]&0xF, bytes[2]&0xF, bytes[3]&0xF
        // High nibbles: bytes[0]>>4, bytes[1]>>4, bytes[2]>>4, bytes[3]>>4
        uint32_t lo4 = packed & 0x0F0F0F0Fu;        // 4 low nibbles as separate bytes
        uint32_t hi4 = (packed >> 4) & 0x0F0F0F0Fu;  // 4 high nibbles as separate bytes

        // Extract individual nibble values from lo4 and hi4
        // lo4 byte 0 = element k+0, hi4 byte 0 = element k+1
        // lo4 byte 1 = element k+2, hi4 byte 1 = element k+3
        // lo4 byte 2 = element k+4, hi4 byte 2 = element k+5
        // lo4 byte 3 = element k+6, hi4 byte 3 = element k+7

        int lo0 = static_cast<int>(lo4 & 0xFF) - 8;
        int hi0 = static_cast<int>(hi4 & 0xFF) - 8;
        int lo1 = static_cast<int>((lo4 >> 8) & 0xFF) - 8;
        int hi1 = static_cast<int>((hi4 >> 8) & 0xFF) - 8;
        int lo2 = static_cast<int>((lo4 >> 16) & 0xFF) - 8;
        int hi2 = static_cast<int>((hi4 >> 16) & 0xFF) - 8;
        int lo3 = static_cast<int>((lo4 >> 24) & 0xFF) - 8;
        int hi3 = static_cast<int>((hi4 >> 24) & 0xFF) - 8;

        sum += static_cast<float>(lo0) * scale_f * __bfloat162float(input[k]);
        sum += static_cast<float>(hi0) * scale_f * __bfloat162float(input[k + 1]);
        sum += static_cast<float>(lo1) * scale_f * __bfloat162float(input[k + 2]);
        sum += static_cast<float>(hi1) * scale_f * __bfloat162float(input[k + 3]);
        sum += static_cast<float>(lo2) * scale_f * __bfloat162float(input[k + 4]);
        sum += static_cast<float>(hi2) * scale_f * __bfloat162float(input[k + 5]);
        sum += static_cast<float>(lo3) * scale_f * __bfloat162float(input[k + 6]);
        sum += static_cast<float>(hi3) * scale_f * __bfloat162float(input[k + 7]);
    }

    sum = warp_reduce_sum(sum);
    __shared__ float smem[GEMV_ROWS * 8];
    int warps_per_row = threads_per_row / WARP_SIZE;
    int warp_in_row = (threadIdx.x % threads_per_row) / WARP_SIZE;
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
// W2A16 GEMV: packed INT2 weights, BF16 activations.
// Each uint32 = 16 unsigned 2-bit values. Zero-point = 2.
// ============================================================================
__global__ void w2a16_gemv_kernel(
    const uint8_t* __restrict__ weight,  // [N, K/4] packed
    const __nv_bfloat16* __restrict__ scales,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int N, int K, int group_size)
{
    int row = blockIdx.x * GEMV_ROWS + threadIdx.x / (GEMV_THREADS / GEMV_ROWS);
    int tid_in_row = threadIdx.x % (GEMV_THREADS / GEMV_ROWS);
    int threads_per_row = GEMV_THREADS / GEMV_ROWS;
    int lane_id = threadIdx.x % WARP_SIZE;
    int row_in_block = threadIdx.x / threads_per_row;

    if (row >= N) return;

    float sum = 0.0f;
    int num_groups = K / group_size;
    int bytes_per_row = K / 4;

    // Process 16 INT2 elements per iteration (one uint32)
    for (int k = tid_in_row * 16; k < K; k += threads_per_row * 16) {
        float scale_f = __bfloat162float(scales[row * num_groups + k / group_size]);
        uint32_t packed = *reinterpret_cast<const uint32_t*>(&weight[row * bytes_per_row + k / 4]);

        // Extract 16 x 2-bit values via shift + mask
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int val = static_cast<int>((packed >> (i * 2)) & 0x3) - 2;
            sum += static_cast<float>(val) * scale_f * __bfloat162float(input[k + i]);
        }
    }

    sum = warp_reduce_sum(sum);
    __shared__ float smem[GEMV_ROWS * 8];
    int warps_per_row = threads_per_row / WARP_SIZE;
    int warp_in_row = (threadIdx.x % threads_per_row) / WARP_SIZE;
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
// Batched W8A16 GEMV: [B, K] × [N, K]^T → [B, N]
// ============================================================================
__global__ void w8a16_gemv_batch_kernel(
    const uint8_t* __restrict__ weight,
    const __nv_bfloat16* __restrict__ scales,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int B, int N, int K, int group_size)
{
    int row = blockIdx.x * GEMV_ROWS + threadIdx.x / (GEMV_THREADS / GEMV_ROWS);
    int batch_idx = blockIdx.y;
    int tid_in_row = threadIdx.x % (GEMV_THREADS / GEMV_ROWS);
    int threads_per_row = GEMV_THREADS / GEMV_ROWS;
    int lane_id = threadIdx.x % WARP_SIZE;
    int row_in_block = threadIdx.x / threads_per_row;

    if (row >= N) return;
    const __nv_bfloat16* x = input + batch_idx * K;
    float sum = 0.0f;
    int num_groups = K / group_size;

    for (int k = tid_in_row * 4; k < K; k += threads_per_row * 4) {
        float scale_f = __bfloat162float(scales[row * num_groups + k / group_size]);
        uint32_t packed = *reinterpret_cast<const uint32_t*>(&weight[row * K + k]);
        int8_t v0 = static_cast<int8_t>(packed);
        int8_t v1 = static_cast<int8_t>(packed >> 8);
        int8_t v2 = static_cast<int8_t>(packed >> 16);
        int8_t v3 = static_cast<int8_t>(packed >> 24);
        sum += static_cast<float>(v0) * scale_f * __bfloat162float(x[k]);
        sum += static_cast<float>(v1) * scale_f * __bfloat162float(x[k + 1]);
        sum += static_cast<float>(v2) * scale_f * __bfloat162float(x[k + 2]);
        sum += static_cast<float>(v3) * scale_f * __bfloat162float(x[k + 3]);
    }

    sum = warp_reduce_sum(sum);
    __shared__ float smem[GEMV_ROWS * 8];
    int warps_per_row = threads_per_row / WARP_SIZE;
    int warp_in_row = (threadIdx.x % threads_per_row) / WARP_SIZE;
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

cudaError_t w8a16_gemv_cuda(
    const int8_t* weight, const __nv_bfloat16* scales,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    int N, int K, int group_size, cudaStream_t stream)
{
    dim3 grid((N + GEMV_ROWS - 1) / GEMV_ROWS);
    dim3 block(GEMV_THREADS);
    w8a16_gemv_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(weight), scales, input, output, N, K, group_size);
    return cudaGetLastError();
}

cudaError_t w4a16_gemv_cuda(
    const uint8_t* weight, const __nv_bfloat16* scales,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    int N, int K, int group_size, cudaStream_t stream)
{
    dim3 grid((N + GEMV_ROWS - 1) / GEMV_ROWS);
    dim3 block(GEMV_THREADS);
    w4a16_gemv_kernel<<<grid, block, 0, stream>>>(
        weight, scales, input, output, N, K, group_size);
    return cudaGetLastError();
}

cudaError_t w2a16_gemv_cuda(
    const uint8_t* weight, const __nv_bfloat16* scales,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    int N, int K, int group_size, cudaStream_t stream)
{
    dim3 grid((N + GEMV_ROWS - 1) / GEMV_ROWS);
    dim3 block(GEMV_THREADS);
    w2a16_gemv_kernel<<<grid, block, 0, stream>>>(
        weight, scales, input, output, N, K, group_size);
    return cudaGetLastError();
}

cudaError_t w8a16_gemv_batch_cuda(
    const int8_t* weight, const __nv_bfloat16* scales,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    int B, int N, int K, int group_size, cudaStream_t stream)
{
    dim3 grid((N + GEMV_ROWS - 1) / GEMV_ROWS, B);
    dim3 block(GEMV_THREADS);
    w8a16_gemv_batch_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(weight), scales, input, output, B, N, K, group_size);
    return cudaGetLastError();
}

}  // extern "C"
