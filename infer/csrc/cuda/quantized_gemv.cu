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
// Batched W4A16 GEMV: [B, K] × [N, K/2]^T → [B, N]
// Same nibble extraction as single W4A16, with batch dimension in grid.y.
// ============================================================================
__global__ void w4a16_gemv_batch_kernel(
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
    int bytes_per_row = K / 2;

    for (int k = tid_in_row * 8; k < K; k += threads_per_row * 8) {
        float scale_f = __bfloat162float(scales[row * num_groups + k / group_size]);
        uint32_t packed = *reinterpret_cast<const uint32_t*>(&weight[row * bytes_per_row + k / 2]);

        uint32_t lo4 = packed & 0x0F0F0F0Fu;
        uint32_t hi4 = (packed >> 4) & 0x0F0F0F0Fu;

        int lo0 = static_cast<int>(lo4 & 0xFF) - 8;
        int hi0 = static_cast<int>(hi4 & 0xFF) - 8;
        int lo1 = static_cast<int>((lo4 >> 8) & 0xFF) - 8;
        int hi1 = static_cast<int>((hi4 >> 8) & 0xFF) - 8;
        int lo2 = static_cast<int>((lo4 >> 16) & 0xFF) - 8;
        int hi2 = static_cast<int>((hi4 >> 16) & 0xFF) - 8;
        int lo3 = static_cast<int>((lo4 >> 24) & 0xFF) - 8;
        int hi3 = static_cast<int>((hi4 >> 24) & 0xFF) - 8;

        sum += static_cast<float>(lo0) * scale_f * __bfloat162float(x[k]);
        sum += static_cast<float>(hi0) * scale_f * __bfloat162float(x[k + 1]);
        sum += static_cast<float>(lo1) * scale_f * __bfloat162float(x[k + 2]);
        sum += static_cast<float>(hi1) * scale_f * __bfloat162float(x[k + 3]);
        sum += static_cast<float>(lo2) * scale_f * __bfloat162float(x[k + 4]);
        sum += static_cast<float>(hi2) * scale_f * __bfloat162float(x[k + 5]);
        sum += static_cast<float>(lo3) * scale_f * __bfloat162float(x[k + 6]);
        sum += static_cast<float>(hi3) * scale_f * __bfloat162float(x[k + 7]);
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
// Batched W2A16 GEMV: [B, K] × [N, K/4]^T → [B, N]
// Same 2-bit extraction as single W2A16, with batch dimension in grid.y.
// ============================================================================
__global__ void w2a16_gemv_batch_kernel(
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
    int bytes_per_row = K / 4;

    for (int k = tid_in_row * 16; k < K; k += threads_per_row * 16) {
        float scale_f = __bfloat162float(scales[row * num_groups + k / group_size]);
        uint32_t packed = *reinterpret_cast<const uint32_t*>(&weight[row * bytes_per_row + k / 4]);

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int val = static_cast<int>((packed >> (i * 2)) & 0x3) - 2;
            sum += static_cast<float>(val) * scale_f * __bfloat162float(x[k + i]);
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
        output[batch_idx * N + row] = __float2bfloat16(total);
    }
}

// ============================================================================
// Q4_K (GGUF Q4_K_M / Q4_K_S) native packed GEMV + dequant.
//
// One superblock = 256 K-dim elements = 144 bytes:
//   d:f16(2) | dmin:f16(2) | scales_packed(12) | qs(128)
//
// scales_packed encodes 8 sub-block scales and 8 sub-block mins as 6-bit values:
//   first 4:  lower 6 bits of bytes[0..4]
//   last  4:  upper 2 bits of bytes[0..4] ORed with low 4 bits of bytes[8..12]
// mins follow the same pattern over bytes[4..8] / bytes[8..12] high nibbles.
//
// Dequant:  w = d * sub_scale[j] * nibble - dmin * sub_min[j]    (llama.cpp)
//
// Packed row stride = (K / 256) * 144 bytes.
//
// Block layout: 256 threads, 8 rows per block, 32 threads (1 warp) per row.
// Each warp processes one row's superblocks sequentially. Within a superblock,
// the 32 lanes cover 1 sub-block (32 elements) per iteration for 8 iterations,
// yielding 256 elements/superblock with every lane active.
// ============================================================================
#define Q4K_GEMV_ROWS 8
#define Q4K_GEMV_THREADS 256  // = Q4K_GEMV_ROWS * 32
#define Q4K_SB_SIZE 256
#define Q4K_SB_BYTES 144

// Decode 8 6-bit scales + 8 6-bit mins from the 12 scale bytes.
// Matches dequant_q4_k in gguf.rs and llama.cpp's get_scale_min_k4 layout.
__device__ __forceinline__ void q4k_decode_scales(
    const uint8_t* __restrict__ scales_raw,
    uint8_t sc[8],
    uint8_t mn[8])
{
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        sc[i] = scales_raw[i] & 0x3F;
        mn[i] = scales_raw[i + 4] & 0x3F;
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        sc[4 + i] = (scales_raw[i]     >> 6) | ((scales_raw[8 + i] & 0x0F) << 2);
        mn[4 + i] = (scales_raw[i + 4] >> 6) | ((scales_raw[8 + i] >> 4)   << 2);
    }
}

__global__ void q4k_gemv_kernel(
    const uint8_t* __restrict__ weight,        // [N, (K/256) * 144]
    const __nv_bfloat16* __restrict__ input,   // [K]
    __nv_bfloat16* __restrict__ output,        // [N]
    int N, int K)
{
    const int warp_id   = threadIdx.x / WARP_SIZE;    // 0..7  → row_in_block
    const int lane      = threadIdx.x % WARP_SIZE;    // 0..31
    const int row       = blockIdx.x * Q4K_GEMV_ROWS + warp_id;
    if (row >= N) return;

    const int num_sb      = K / Q4K_SB_SIZE;
    const int row_bytes   = num_sb * Q4K_SB_BYTES;
    const uint8_t* row_p  = weight + row * row_bytes;

    float sum = 0.0f;

    for (int sb = 0; sb < num_sb; ++sb) {
        const uint8_t* sb_p = row_p + sb * Q4K_SB_BYTES;

        // Decode (d, dmin) once per superblock (fp16 LE bytes).
        const unsigned short d_u16    = ((const unsigned short*)sb_p)[0];
        const unsigned short dmin_u16 = ((const unsigned short*)sb_p)[1];
        const float d     = __half2float(*reinterpret_cast<const __half*>(&d_u16));
        const float dmin  = __half2float(*reinterpret_cast<const __half*>(&dmin_u16));

        // Decode 8 sub-scales and 8 sub-mins into registers.
        uint8_t sc[8], mn[8];
        q4k_decode_scales(sb_p + 4, sc, mn);

        const uint8_t* qs = sb_p + 16;  // 128 bytes of packed nibbles
        const int k_base  = sb * Q4K_SB_SIZE;

        // 8 sub-blocks × 32 elements, one per lane per iteration.
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float sub_scale = d * (float)sc[j];
            const float sub_min   = dmin * (float)mn[j];

            // Lane l reads byte (j*16 + l/2), low nibble if l even else high.
            const uint8_t byte = qs[j * 16 + (lane >> 1)];
            const int q = (lane & 1) ? (byte >> 4) : (byte & 0x0F);
            const float w = (float)q * sub_scale - sub_min;

            const int k_idx = k_base + j * 32 + lane;
            sum += w * __bfloat162float(input[k_idx]);
        }
    }

    // Warp reduce (1 warp per row → direct write, no cross-warp smem).
    sum = warp_reduce_sum(sum);
    if (lane == 0) output[row] = __float2bfloat16(sum);
}

// Batched variant: [B, K] × [N, packed]^T → [B, N]. Batch in grid.y.
__global__ void q4k_gemv_batch_kernel(
    const uint8_t* __restrict__ weight,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int B, int N, int K)
{
    const int warp_id  = threadIdx.x / WARP_SIZE;
    const int lane     = threadIdx.x % WARP_SIZE;
    const int row      = blockIdx.x * Q4K_GEMV_ROWS + warp_id;
    const int batch    = blockIdx.y;
    if (row >= N || batch >= B) return;

    const int num_sb     = K / Q4K_SB_SIZE;
    const int row_bytes  = num_sb * Q4K_SB_BYTES;
    const uint8_t* row_p = weight + row * row_bytes;
    const __nv_bfloat16* x = input + batch * K;

    float sum = 0.0f;

    for (int sb = 0; sb < num_sb; ++sb) {
        const uint8_t* sb_p = row_p + sb * Q4K_SB_BYTES;

        const unsigned short d_u16    = ((const unsigned short*)sb_p)[0];
        const unsigned short dmin_u16 = ((const unsigned short*)sb_p)[1];
        const float d    = __half2float(*reinterpret_cast<const __half*>(&d_u16));
        const float dmin = __half2float(*reinterpret_cast<const __half*>(&dmin_u16));

        uint8_t sc[8], mn[8];
        q4k_decode_scales(sb_p + 4, sc, mn);

        const uint8_t* qs = sb_p + 16;
        const int k_base  = sb * Q4K_SB_SIZE;

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float sub_scale = d * (float)sc[j];
            const float sub_min   = dmin * (float)mn[j];
            const uint8_t byte = qs[j * 16 + (lane >> 1)];
            const int q = (lane & 1) ? (byte >> 4) : (byte & 0x0F);
            const float w = (float)q * sub_scale - sub_min;
            const int k_idx = k_base + j * 32 + lane;
            sum += w * __bfloat162float(x[k_idx]);
        }
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) output[batch * N + row] = __float2bfloat16(sum);
}

// Dequantize a K-dim chunk [k_start..k_start+k_len) of a Q4_K weight matrix into BF16.
// k_start and k_len must be multiples of 256.
// Output layout: [N, k_len] row-major BF16.
//
// Grid:  (N, k_len / 256)       — one block per (row, superblock in chunk)
// Block: 256 threads            — one thread per element in the superblock
__global__ void q4k_dequant_chunk_kernel(
    const uint8_t* __restrict__ weight,
    __nv_bfloat16* __restrict__ out,
    int N, int K, int k_start, int k_len)
{
    const int row = blockIdx.x;
    const int sb_in_chunk = blockIdx.y;
    const int tid = threadIdx.x;
    if (row >= N) return;

    const int num_sb_total = K / Q4K_SB_SIZE;
    const int sb_global    = (k_start / Q4K_SB_SIZE) + sb_in_chunk;
    const int row_bytes    = num_sb_total * Q4K_SB_BYTES;
    const uint8_t* sb_p    = weight + row * row_bytes + sb_global * Q4K_SB_BYTES;

    // Shared scratch: decode (d, dmin, sc[8], mn[8]) once per block.
    __shared__ float s_d;
    __shared__ float s_dmin;
    __shared__ uint8_t s_sc[8];
    __shared__ uint8_t s_mn[8];

    if (tid == 0) {
        const unsigned short d_u16    = ((const unsigned short*)sb_p)[0];
        const unsigned short dmin_u16 = ((const unsigned short*)sb_p)[1];
        s_d    = __half2float(*reinterpret_cast<const __half*>(&d_u16));
        s_dmin = __half2float(*reinterpret_cast<const __half*>(&dmin_u16));
        q4k_decode_scales(sb_p + 4, s_sc, s_mn);
    }
    __syncthreads();

    const uint8_t* qs = sb_p + 16;
    // tid 0..255 → sub-block j = tid/32, in-sub offset i = tid%32.
    const int j = tid >> 5;
    const int i = tid & 31;
    const uint8_t byte = qs[j * 16 + (i >> 1)];
    const int q = (i & 1) ? (byte >> 4) : (byte & 0x0F);
    const float sub_scale = s_d    * (float)s_sc[j];
    const float sub_min   = s_dmin * (float)s_mn[j];
    const float w = (float)q * sub_scale - sub_min;

    const int out_k = sb_in_chunk * Q4K_SB_SIZE + j * 32 + i;
    out[row * k_len + out_k] = __float2bfloat16(w);
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

cudaError_t w4a16_gemv_batch_cuda(
    const uint8_t* weight, const __nv_bfloat16* scales,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    int B, int N, int K, int group_size, cudaStream_t stream)
{
    dim3 grid((N + GEMV_ROWS - 1) / GEMV_ROWS, B);
    dim3 block(GEMV_THREADS);
    w4a16_gemv_batch_kernel<<<grid, block, 0, stream>>>(
        weight, scales, input, output, B, N, K, group_size);
    return cudaGetLastError();
}

cudaError_t w2a16_gemv_batch_cuda(
    const uint8_t* weight, const __nv_bfloat16* scales,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    int B, int N, int K, int group_size, cudaStream_t stream)
{
    dim3 grid((N + GEMV_ROWS - 1) / GEMV_ROWS, B);
    dim3 block(GEMV_THREADS);
    w2a16_gemv_batch_kernel<<<grid, block, 0, stream>>>(
        weight, scales, input, output, B, N, K, group_size);
    return cudaGetLastError();
}

// ── Q4_K (GGUF) native packed ──

cudaError_t q4k_gemv_cuda(
    const uint8_t* weight,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    int N, int K, cudaStream_t stream)
{
    dim3 grid((N + Q4K_GEMV_ROWS - 1) / Q4K_GEMV_ROWS);
    dim3 block(Q4K_GEMV_THREADS);
    q4k_gemv_kernel<<<grid, block, 0, stream>>>(weight, input, output, N, K);
    return cudaGetLastError();
}

cudaError_t q4k_gemv_batch_cuda(
    const uint8_t* weight,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    int B, int N, int K, cudaStream_t stream)
{
    dim3 grid((N + Q4K_GEMV_ROWS - 1) / Q4K_GEMV_ROWS, B);
    dim3 block(Q4K_GEMV_THREADS);
    q4k_gemv_batch_kernel<<<grid, block, 0, stream>>>(weight, input, output, B, N, K);
    return cudaGetLastError();
}

cudaError_t q4k_dequant_chunk_cuda(
    const uint8_t* weight, __nv_bfloat16* out,
    int N, int K, int k_start, int k_len, cudaStream_t stream)
{
    // Safety: chunk must align to superblock boundaries.
    if ((k_start % Q4K_SB_SIZE) != 0 || (k_len % Q4K_SB_SIZE) != 0) {
        return cudaErrorInvalidValue;
    }
    dim3 grid(N, k_len / Q4K_SB_SIZE);
    dim3 block(Q4K_SB_SIZE);
    q4k_dequant_chunk_kernel<<<grid, block, 0, stream>>>(
        weight, out, N, K, k_start, k_len);
    return cudaGetLastError();
}

}  // extern "C"
