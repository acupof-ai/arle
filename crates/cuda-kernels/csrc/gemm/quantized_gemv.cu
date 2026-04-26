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

__global__ void q8_embedding_batched_kernel(
    const int8_t* __restrict__ weight,
    const __nv_bfloat16* __restrict__ scales,
    const int* __restrict__ token_ids,
    __nv_bfloat16* __restrict__ out,
    int hidden_dim,
    int batch_size,
    int group_size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = hidden_dim * batch_size;
    if (idx >= total) return;

    const int batch = idx / hidden_dim;
    const int col = idx - batch * hidden_dim;
    const int row = token_ids[batch];
    const int num_groups = hidden_dim / group_size;
    const float scale = __bfloat162float(scales[row * num_groups + col / group_size]);
    const int8_t q = weight[row * hidden_dim + col];
    out[idx] = __float2bfloat16(static_cast<float>(q) * scale);
}

__global__ void q8_embedding_decode_kernel(
    const int8_t* __restrict__ weight,
    const __nv_bfloat16* __restrict__ scales,
    const int* __restrict__ token_id,
    __nv_bfloat16* __restrict__ out,
    int hidden_dim,
    int group_size)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= hidden_dim) return;

    const int row = token_id[0];
    const int num_groups = hidden_dim / group_size;
    const float scale = __bfloat162float(scales[row * num_groups + col / group_size]);
    const int8_t q = weight[row * hidden_dim + col];
    out[col] = __float2bfloat16(static_cast<float>(q) * scale);
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
// Q6_K (GGUF) native packed GEMV + dequant.
//
// One superblock = 256 K-dim elements = 210 bytes:
//   ql:[128]  | qh:[64]  | scales:[16 × i8]  | d:f16(2)
//
// Element layout mirrors llama.cpp `dequantize_row_q6_K`. Each half of 128
// elements interleaves four 32-element quadrants drawn from:
//   q0 at y[l+  0] = (ql[l+ 0] & 0xF) | ((qh[l]>>0 & 3)<<4)
//   q1 at y[l+ 32] = (ql[l+32] & 0xF) | ((qh[l]>>2 & 3)<<4)
//   q2 at y[l+ 64] = (ql[l+ 0] >> 4)  | ((qh[l]>>4 & 3)<<4)
//   q3 at y[l+ 96] = (ql[l+32] >> 4)  | ((qh[l]>>6 & 3)<<4)
// Signed weight = (6bit - 32). Scale: scales[is + quadrant*2], is = l/16.
// Second half uses ql+=64, qh+=32, sc+=8.
// ============================================================================
#define Q6K_SB_SIZE 256
#define Q6K_SB_BYTES 210
#define Q6K_GEMV_ROWS 8
#define Q6K_GEMV_THREADS 256  // = Q6K_GEMV_ROWS * 32

__global__ void q6k_gemv_kernel(
    const uint8_t* __restrict__ weight,       // [N, (K/256) * 210]
    const __nv_bfloat16* __restrict__ input,  // [K]
    __nv_bfloat16* __restrict__ output,       // [N]
    int N, int K)
{
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane    = threadIdx.x % WARP_SIZE;
    const int row     = blockIdx.x * Q6K_GEMV_ROWS + warp_id;
    if (row >= N) return;

    const int num_sb    = K / Q6K_SB_SIZE;
    const int row_bytes = num_sb * Q6K_SB_BYTES;
    const uint8_t* row_p = weight + row * row_bytes;

    float sum = 0.0f;

    for (int sb = 0; sb < num_sb; ++sb) {
        const uint8_t* sb_p = row_p + sb * Q6K_SB_BYTES;
        const uint8_t* ql_all = sb_p + 0;    // 128 bytes
        const uint8_t* qh_all = sb_p + 128;  // 64 bytes
        const int8_t*  sc_all = (const int8_t*)(sb_p + 192); // 16 bytes signed
        const unsigned short d_u16 = ((const unsigned short*)(sb_p + 208))[0];
        const float d = __half2float(*reinterpret_cast<const __half*>(&d_u16));

        const int k_base = sb * Q6K_SB_SIZE;
        const int l = lane;           // 0..32 — position within a 32-element quadrant
        const int is = l / 16;        // 0 or 1

        // Process both halves × four quadrants per lane = 8 elements/superblock.
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const uint8_t* ql = ql_all + half * 64;
            const uint8_t* qh = qh_all + half * 32;
            const int8_t*  sc = sc_all + half * 8;
            const int k_half_base = k_base + half * 128;
            const uint8_t qh_l = qh[l];
            const uint8_t ql_0 = ql[l];
            const uint8_t ql_1 = ql[l + 32];

            // Quadrant 0: y[l+0]
            {
                const int low4 = ql_0 & 0x0F;
                const int high2 = (qh_l >> 0) & 0x03;
                const int q = (low4 | (high2 << 4)) - 32;
                const float w = d * (float)sc[is + 0] * (float)q;
                sum += w * __bfloat162float(input[k_half_base + l + 0]);
            }
            // Quadrant 1: y[l+32]
            {
                const int low4 = ql_1 & 0x0F;
                const int high2 = (qh_l >> 2) & 0x03;
                const int q = (low4 | (high2 << 4)) - 32;
                const float w = d * (float)sc[is + 2] * (float)q;
                sum += w * __bfloat162float(input[k_half_base + l + 32]);
            }
            // Quadrant 2: y[l+64]
            {
                const int low4 = ql_0 >> 4;
                const int high2 = (qh_l >> 4) & 0x03;
                const int q = (low4 | (high2 << 4)) - 32;
                const float w = d * (float)sc[is + 4] * (float)q;
                sum += w * __bfloat162float(input[k_half_base + l + 64]);
            }
            // Quadrant 3: y[l+96]
            {
                const int low4 = ql_1 >> 4;
                const int high2 = (qh_l >> 6) & 0x03;
                const int q = (low4 | (high2 << 4)) - 32;
                const float w = d * (float)sc[is + 6] * (float)q;
                sum += w * __bfloat162float(input[k_half_base + l + 96]);
            }
        }
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) output[row] = __float2bfloat16(sum);
}

__global__ void q6k_gemv_batch_kernel(
    const uint8_t* __restrict__ weight,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int B, int N, int K)
{
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane    = threadIdx.x % WARP_SIZE;
    const int row     = blockIdx.x * Q6K_GEMV_ROWS + warp_id;
    const int batch   = blockIdx.y;
    if (row >= N || batch >= B) return;

    const int num_sb    = K / Q6K_SB_SIZE;
    const int row_bytes = num_sb * Q6K_SB_BYTES;
    const uint8_t* row_p = weight + row * row_bytes;
    const __nv_bfloat16* x = input + batch * K;

    float sum = 0.0f;

    for (int sb = 0; sb < num_sb; ++sb) {
        const uint8_t* sb_p = row_p + sb * Q6K_SB_BYTES;
        const uint8_t* ql_all = sb_p + 0;
        const uint8_t* qh_all = sb_p + 128;
        const int8_t*  sc_all = (const int8_t*)(sb_p + 192);
        const unsigned short d_u16 = ((const unsigned short*)(sb_p + 208))[0];
        const float d = __half2float(*reinterpret_cast<const __half*>(&d_u16));

        const int k_base = sb * Q6K_SB_SIZE;
        const int l = lane;
        const int is = l / 16;

        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const uint8_t* ql = ql_all + half * 64;
            const uint8_t* qh = qh_all + half * 32;
            const int8_t*  sc = sc_all + half * 8;
            const int k_half_base = k_base + half * 128;
            const uint8_t qh_l = qh[l];
            const uint8_t ql_0 = ql[l];
            const uint8_t ql_1 = ql[l + 32];

            {
                const int q = ((ql_0 & 0x0F) | (((qh_l >> 0) & 0x03) << 4)) - 32;
                sum += d * (float)sc[is + 0] * (float)q
                       * __bfloat162float(x[k_half_base + l + 0]);
            }
            {
                const int q = ((ql_1 & 0x0F) | (((qh_l >> 2) & 0x03) << 4)) - 32;
                sum += d * (float)sc[is + 2] * (float)q
                       * __bfloat162float(x[k_half_base + l + 32]);
            }
            {
                const int q = ((ql_0 >> 4) | (((qh_l >> 4) & 0x03) << 4)) - 32;
                sum += d * (float)sc[is + 4] * (float)q
                       * __bfloat162float(x[k_half_base + l + 64]);
            }
            {
                const int q = ((ql_1 >> 4) | (((qh_l >> 6) & 0x03) << 4)) - 32;
                sum += d * (float)sc[is + 6] * (float)q
                       * __bfloat162float(x[k_half_base + l + 96]);
            }
        }
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) output[batch * N + row] = __float2bfloat16(sum);
}

// Dequantize chunk kernel: each block handles ONE (row, superblock) and 256
// threads write the 256 dequanted elements of that superblock to the BF16 tile.
__global__ void q6k_dequant_chunk_kernel(
    const uint8_t* __restrict__ weight,
    __nv_bfloat16* __restrict__ out,
    int N, int K, int k_start, int k_len)
{
    const int row = blockIdx.x;
    const int sb_in_chunk = blockIdx.y;
    const int tid = threadIdx.x;
    if (row >= N) return;

    const int num_sb_total = K / Q6K_SB_SIZE;
    const int sb_global    = (k_start / Q6K_SB_SIZE) + sb_in_chunk;
    const int row_bytes    = num_sb_total * Q6K_SB_BYTES;
    const uint8_t* sb_p    = weight + row * row_bytes + sb_global * Q6K_SB_BYTES;

    __shared__ float s_d;
    __shared__ int8_t s_scales[16];

    if (tid == 0) {
        const unsigned short d_u16 = ((const unsigned short*)(sb_p + 208))[0];
        s_d = __half2float(*reinterpret_cast<const __half*>(&d_u16));
    }
    if (tid < 16) {
        s_scales[tid] = ((const int8_t*)(sb_p + 192))[tid];
    }
    __syncthreads();

    // tid 0..255 → half, quadrant, l
    const int half = tid / 128;          // 0,1
    const int j_local = tid % 128;
    const int quad = j_local / 32;       // 0..4
    const int l = j_local % 32;
    const int is = l / 16;

    const uint8_t* ql = sb_p + half * 64;                  // ql[half*64..(half+1)*64]
    const uint8_t* qh = sb_p + 128 + half * 32;
    const int sc_base = half * 8;

    uint8_t low4, high2;
    switch (quad) {
        case 0: low4 = ql[l] & 0x0F;        high2 = (qh[l] >> 0) & 0x03; break;
        case 1: low4 = ql[l + 32] & 0x0F;   high2 = (qh[l] >> 2) & 0x03; break;
        case 2: low4 = ql[l] >> 4;          high2 = (qh[l] >> 4) & 0x03; break;
        default: low4 = ql[l + 32] >> 4;    high2 = (qh[l] >> 6) & 0x03; break;
    }
    const int q = (int)(low4 | (high2 << 4)) - 32;
    const int8_t sc = s_scales[sc_base + is + quad * 2];
    const float w = s_d * (float)sc * (float)q;

    const int out_k = sb_in_chunk * Q6K_SB_SIZE + half * 128 + quad * 32 + l;
    out[row * k_len + out_k] = __float2bfloat16(w);
}

// ============================================================================
// Q3_K (GGUF) native packed GEMV + dequant.
//
// One superblock = 256 K-dim elements = 110 bytes:
//   hmask:[32]  | qs:[64, 2-bit]  | scales:[12, 6-bit signed]  | d:f16(2)
//
// Element dequant:
//   q2  = (qs[k/4]    >> ((k%4)*2)) & 0x3
//   hbit= (hmask[k/8] >> (k%8))     & 0x1
//   q3  = q2 | (hbit << 2)
//   scale[i=k/16] = (scales_lo[i] | scales_hi[i] << 4) - 8 (signed, -8..55)
//   w   = d * scale * (q3 - 4)
//
// Scales decode (12 bytes → 16 sub-block scales, signed i8, one per 16 elements).
//
// Each scale is a 6-bit UNSIGNED value in 0..63. Low 4 bits come from the
// low/high nibble of scales_raw[0..8] (i<8 → low nibble of raw[i], i≥8 →
// high nibble of raw[i-8]). High 2 bits come from scales_raw[8+(i&3)] shifted
// right 2*(i/4) then masked with 0x3.
//
// Signed scale = unsigned6 - 32. Range: -32..31.
//
// NOTE: must combine the 6 bits BEFORE subtracting 32. Subtracting first and
// then OR'ing bit 4 into a negative i8 loses the bit to sign extension.
// (matches dequant_q3_k in gguf.rs after fix for the same bug.)
// ============================================================================
#define Q3K_SB_SIZE 256
#define Q3K_SB_BYTES 110
#define Q3K_GEMV_ROWS 8
#define Q3K_GEMV_THREADS 256  // = Q3K_GEMV_ROWS * 32

__device__ __forceinline__ void q3k_decode_scales(
    const uint8_t* __restrict__ scales_raw,  // 12 bytes
    int8_t scales[16])
{
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const uint8_t low4 = (i < 8)
            ? (scales_raw[i] & 0x0F)
            : ((scales_raw[i - 8] >> 4) & 0x0F);
        const uint8_t high2 = (scales_raw[8 + (i & 3)] >> (2 * (i / 4))) & 0x03;
        const uint8_t u6 = low4 | (high2 << 4);
        scales[i] = (int8_t)((int)u6 - 32);
    }
}

__global__ void q3k_gemv_kernel(
    const uint8_t* __restrict__ weight,       // [N, (K/256) * 110]
    const __nv_bfloat16* __restrict__ input,  // [K]
    __nv_bfloat16* __restrict__ output,       // [N]
    int N, int K)
{
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane    = threadIdx.x % WARP_SIZE;
    const int row     = blockIdx.x * Q3K_GEMV_ROWS + warp_id;
    if (row >= N) return;

    const int num_sb    = K / Q3K_SB_SIZE;
    const int row_bytes = num_sb * Q3K_SB_BYTES;
    const uint8_t* row_p = weight + row * row_bytes;

    float sum = 0.0f;

    for (int sb = 0; sb < num_sb; ++sb) {
        const uint8_t* sb_p = row_p + sb * Q3K_SB_BYTES;
        const uint8_t* hmask = sb_p + 0;
        const uint8_t* qs    = sb_p + 32;
        const uint8_t* sc_raw = sb_p + 96;

        const unsigned short d_u16 = ((const unsigned short*)(sb_p + 108))[0];
        const float d = __half2float(*reinterpret_cast<const __half*>(&d_u16));

        int8_t scales[16];
        q3k_decode_scales(sc_raw, scales);

        const int k_base = sb * Q3K_SB_SIZE;

        // Each lane handles 8 elements per superblock, stride 32 → adjacent lanes
        // touch adjacent K indices → coalesced input loads.
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int k_local = i * 32 + lane;  // 0..255
            const int q2 = (qs[k_local >> 2] >> ((k_local & 3) << 1)) & 0x3;
            const int hbit = (hmask[k_local >> 3] >> (k_local & 7)) & 0x1;
            const int q3 = q2 | (hbit << 2);
            const int sub_idx = k_local >> 4;  // /16
            const float scale = d * (float)scales[sub_idx];
            const float w = scale * ((float)q3 - 4.0f);
            sum += w * __bfloat162float(input[k_base + k_local]);
        }
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) output[row] = __float2bfloat16(sum);
}

__global__ void q3k_gemv_batch_kernel(
    const uint8_t* __restrict__ weight,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int B, int N, int K)
{
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane    = threadIdx.x % WARP_SIZE;
    const int row     = blockIdx.x * Q3K_GEMV_ROWS + warp_id;
    const int batch   = blockIdx.y;
    if (row >= N || batch >= B) return;

    const int num_sb    = K / Q3K_SB_SIZE;
    const int row_bytes = num_sb * Q3K_SB_BYTES;
    const uint8_t* row_p = weight + row * row_bytes;
    const __nv_bfloat16* x = input + batch * K;

    float sum = 0.0f;

    for (int sb = 0; sb < num_sb; ++sb) {
        const uint8_t* sb_p = row_p + sb * Q3K_SB_BYTES;
        const uint8_t* hmask = sb_p + 0;
        const uint8_t* qs    = sb_p + 32;
        const uint8_t* sc_raw = sb_p + 96;
        const unsigned short d_u16 = ((const unsigned short*)(sb_p + 108))[0];
        const float d = __half2float(*reinterpret_cast<const __half*>(&d_u16));

        int8_t scales[16];
        q3k_decode_scales(sc_raw, scales);
        const int k_base = sb * Q3K_SB_SIZE;

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int k_local = i * 32 + lane;
            const int q2 = (qs[k_local >> 2] >> ((k_local & 3) << 1)) & 0x3;
            const int hbit = (hmask[k_local >> 3] >> (k_local & 7)) & 0x1;
            const int q3 = q2 | (hbit << 2);
            const int sub_idx = k_local >> 4;
            const float scale = d * (float)scales[sub_idx];
            const float w = scale * ((float)q3 - 4.0f);
            sum += w * __bfloat162float(x[k_base + k_local]);
        }
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) output[batch * N + row] = __float2bfloat16(sum);
}

// Dequant chunk kernel: writes a BF16 tile [N, k_len] starting at k_start.
// Grid: (N, k_len / 256).  Block: 256 threads — one per element in the superblock.
__global__ void q3k_dequant_chunk_kernel(
    const uint8_t* __restrict__ weight,
    __nv_bfloat16* __restrict__ out,
    int N, int K, int k_start, int k_len)
{
    const int row = blockIdx.x;
    const int sb_in_chunk = blockIdx.y;
    const int tid = threadIdx.x;
    if (row >= N) return;

    const int num_sb_total = K / Q3K_SB_SIZE;
    const int sb_global    = (k_start / Q3K_SB_SIZE) + sb_in_chunk;
    const int row_bytes    = num_sb_total * Q3K_SB_BYTES;
    const uint8_t* sb_p    = weight + row * row_bytes + sb_global * Q3K_SB_BYTES;

    __shared__ float s_d;
    __shared__ int8_t s_scales[16];

    if (tid == 0) {
        const unsigned short d_u16 = ((const unsigned short*)(sb_p + 108))[0];
        s_d = __half2float(*reinterpret_cast<const __half*>(&d_u16));
        q3k_decode_scales(sb_p + 96, s_scales);
    }
    __syncthreads();

    const uint8_t* hmask = sb_p + 0;
    const uint8_t* qs    = sb_p + 32;

    const int k_local = tid;
    const int q2 = (qs[k_local >> 2] >> ((k_local & 3) << 1)) & 0x3;
    const int hbit = (hmask[k_local >> 3] >> (k_local & 7)) & 0x1;
    const int q3 = q2 | (hbit << 2);
    const int sub_idx = k_local >> 4;
    const float scale = s_d * (float)s_scales[sub_idx];
    const float w = scale * ((float)q3 - 4.0f);

    const int out_k = sb_in_chunk * Q3K_SB_SIZE + k_local;
    out[row * k_len + out_k] = __float2bfloat16(w);
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
        sc[4 + i] = (scales_raw[8 + i] & 0x0F) | ((scales_raw[i]     >> 6) << 4);
        mn[4 + i] = (scales_raw[8 + i] >> 4)   | ((scales_raw[i + 4] >> 6) << 4);
    }
}

// Element layout for Q4_K — MUST match llama.cpp `dequantize_row_q4_K`:
//   for iter in 0..4:
//     for l in 0..32:  y[iter*64 + l    ] = sc[2*iter+0] * (qs[iter*32+l] & 0x0F) - mn[2*iter+0]
//     for l in 0..32:  y[iter*64 + l+32] = sc[2*iter+1] * (qs[iter*32+l] >>  4) - mn[2*iter+1]
// NOT the naive "2 elements per ql byte" interpretation!
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

        const unsigned short d_u16    = ((const unsigned short*)sb_p)[0];
        const unsigned short dmin_u16 = ((const unsigned short*)sb_p)[1];
        const float d     = __half2float(*reinterpret_cast<const __half*>(&d_u16));
        const float dmin  = __half2float(*reinterpret_cast<const __half*>(&dmin_u16));

        uint8_t sc[8], mn[8];
        q4k_decode_scales(sb_p + 4, sc, mn);

        const uint8_t* qs = sb_p + 16;  // 128 bytes
        const int k_base  = sb * Q4K_SB_SIZE;

        // 4 outer iterations of 64 elements, 2 sub-blocks each.
        // Each lane processes 2 elements per iter (one lo nibble + one hi nibble
        // of the SAME ql byte) — so 8 elements/superblock/lane, 256/superblock total.
        #pragma unroll
        for (int iter = 0; iter < 4; ++iter) {
            const int j_lo = iter * 2;
            const int j_hi = j_lo + 1;
            const float d1 = d * (float)sc[j_lo];
            const float m1 = dmin * (float)mn[j_lo];
            const float d2 = d * (float)sc[j_hi];
            const float m2 = dmin * (float)mn[j_hi];
            const uint8_t byte = qs[iter * 32 + lane];
            const float q_lo = (float)(byte & 0x0F);
            const float q_hi = (float)(byte >> 4);
            const float w_lo = q_lo * d1 - m1;
            const float w_hi = q_hi * d2 - m2;
            const int k_lo = k_base + j_lo * 32 + lane;
            const int k_hi = k_base + j_hi * 32 + lane;
            sum += w_lo * __bfloat162float(input[k_lo]);
            sum += w_hi * __bfloat162float(input[k_hi]);
        }
    }

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
        for (int iter = 0; iter < 4; ++iter) {
            const int j_lo = iter * 2;
            const int j_hi = j_lo + 1;
            const float d1 = d * (float)sc[j_lo];
            const float m1 = dmin * (float)mn[j_lo];
            const float d2 = d * (float)sc[j_hi];
            const float m2 = dmin * (float)mn[j_hi];
            const uint8_t byte = qs[iter * 32 + lane];
            const float q_lo = (float)(byte & 0x0F);
            const float q_hi = (float)(byte >> 4);
            sum += (q_lo * d1 - m1) * __bfloat162float(x[k_base + j_lo * 32 + lane]);
            sum += (q_hi * d2 - m2) * __bfloat162float(x[k_base + j_hi * 32 + lane]);
        }
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) output[batch * N + row] = __float2bfloat16(sum);
}

// Dequantize a K-dim chunk [k_start..k_start+k_len) of a Q4_K weight matrix into BF16.
// Grid:  (N, k_len / 256), Block: 256 threads — but element-to-thread mapping follows
// llama.cpp's canonical iter/half/l layout, NOT the naive "tid is element index"
// interpretation. 256 threads cover one superblock:
//   thread t → iter = (t >> 6) & 3, half = (t >> 5) & 1, l = t & 31
//   writes y[iter*64 + half*32 + l]
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
    const int iter = (tid >> 6) & 3;  // 0..4
    const int half = (tid >> 5) & 1;  // 0..2
    const int l    = tid & 31;
    const int sub  = iter * 2 + half;  // 0..8
    const uint8_t byte = qs[iter * 32 + l];
    const int q = half ? (byte >> 4) : (byte & 0x0F);
    const float w = (float)q * (s_d * (float)s_sc[sub]) - (s_dmin * (float)s_mn[sub]);

    const int out_k = sb_in_chunk * Q4K_SB_SIZE + sub * 32 + l;
    out[row * k_len + out_k] = __float2bfloat16(w);
}

// ============================================================================
// Q5_K (GGUF Q5_K_M / Q5_K_S) native packed GEMV + dequant.
//
// One superblock = 256 K-dim elements = 176 bytes:
//   d:f16(2) | dmin:f16(2) | scales_packed(12) | qh(32) | qs(128)
//
// Q5_K shares Q4_K's scale/min packing and element order. `qs` stores low
// nibbles, while `qh[l]` contributes one high bit for each of the 8 sub-blocks.
// ============================================================================
#define Q5K_GEMV_ROWS 8
#define Q5K_GEMV_THREADS 256
#define Q5K_SB_SIZE 256
#define Q5K_SB_BYTES 176

__global__ void q5k_gemv_kernel(
    const uint8_t* __restrict__ weight,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int N, int K)
{
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane    = threadIdx.x % WARP_SIZE;
    const int row     = blockIdx.x * Q5K_GEMV_ROWS + warp_id;
    if (row >= N) return;

    const int num_sb = K / Q5K_SB_SIZE;
    const int row_bytes = num_sb * Q5K_SB_BYTES;
    const uint8_t* row_p = weight + row * row_bytes;

    float sum = 0.0f;

    for (int sb = 0; sb < num_sb; ++sb) {
        const uint8_t* sb_p = row_p + sb * Q5K_SB_BYTES;
        const unsigned short d_u16    = ((const unsigned short*)sb_p)[0];
        const unsigned short dmin_u16 = ((const unsigned short*)sb_p)[1];
        const float d    = __half2float(*reinterpret_cast<const __half*>(&d_u16));
        const float dmin = __half2float(*reinterpret_cast<const __half*>(&dmin_u16));

        uint8_t sc[8], mn[8];
        q4k_decode_scales(sb_p + 4, sc, mn);

        const uint8_t* qh = sb_p + 16;
        const uint8_t* qs = sb_p + 48;
        const int k_base = sb * Q5K_SB_SIZE;

        #pragma unroll
        for (int iter = 0; iter < 4; ++iter) {
            const int j_lo = iter * 2;
            const int j_hi = j_lo + 1;
            const float d1 = d * (float)sc[j_lo];
            const float m1 = dmin * (float)mn[j_lo];
            const float d2 = d * (float)sc[j_hi];
            const float m2 = dmin * (float)mn[j_hi];
            const uint8_t byte = qs[iter * 32 + lane];
            const int q_lo = (int)(byte & 0x0F) | (((int)(qh[lane] >> j_lo) & 1) << 4);
            const int q_hi = (int)(byte >> 4) | (((int)(qh[lane] >> j_hi) & 1) << 4);
            sum += ((float)q_lo * d1 - m1) * __bfloat162float(input[k_base + j_lo * 32 + lane]);
            sum += ((float)q_hi * d2 - m2) * __bfloat162float(input[k_base + j_hi * 32 + lane]);
        }
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) output[row] = __float2bfloat16(sum);
}

__global__ void q5k_gemv_batch_kernel(
    const uint8_t* __restrict__ weight,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int B, int N, int K)
{
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane    = threadIdx.x % WARP_SIZE;
    const int row     = blockIdx.x * Q5K_GEMV_ROWS + warp_id;
    const int batch   = blockIdx.y;
    if (row >= N || batch >= B) return;

    const int num_sb = K / Q5K_SB_SIZE;
    const int row_bytes = num_sb * Q5K_SB_BYTES;
    const uint8_t* row_p = weight + row * row_bytes;
    const __nv_bfloat16* x = input + batch * K;

    float sum = 0.0f;

    for (int sb = 0; sb < num_sb; ++sb) {
        const uint8_t* sb_p = row_p + sb * Q5K_SB_BYTES;
        const unsigned short d_u16    = ((const unsigned short*)sb_p)[0];
        const unsigned short dmin_u16 = ((const unsigned short*)sb_p)[1];
        const float d    = __half2float(*reinterpret_cast<const __half*>(&d_u16));
        const float dmin = __half2float(*reinterpret_cast<const __half*>(&dmin_u16));

        uint8_t sc[8], mn[8];
        q4k_decode_scales(sb_p + 4, sc, mn);

        const uint8_t* qh = sb_p + 16;
        const uint8_t* qs = sb_p + 48;
        const int k_base = sb * Q5K_SB_SIZE;

        #pragma unroll
        for (int iter = 0; iter < 4; ++iter) {
            const int j_lo = iter * 2;
            const int j_hi = j_lo + 1;
            const float d1 = d * (float)sc[j_lo];
            const float m1 = dmin * (float)mn[j_lo];
            const float d2 = d * (float)sc[j_hi];
            const float m2 = dmin * (float)mn[j_hi];
            const uint8_t byte = qs[iter * 32 + lane];
            const int q_lo = (int)(byte & 0x0F) | (((int)(qh[lane] >> j_lo) & 1) << 4);
            const int q_hi = (int)(byte >> 4) | (((int)(qh[lane] >> j_hi) & 1) << 4);
            sum += ((float)q_lo * d1 - m1) * __bfloat162float(x[k_base + j_lo * 32 + lane]);
            sum += ((float)q_hi * d2 - m2) * __bfloat162float(x[k_base + j_hi * 32 + lane]);
        }
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) output[batch * N + row] = __float2bfloat16(sum);
}

__global__ void q5k_dequant_chunk_kernel(
    const uint8_t* __restrict__ weight,
    __nv_bfloat16* __restrict__ out,
    int N, int K, int k_start, int k_len)
{
    const int row = blockIdx.x;
    const int sb_in_chunk = blockIdx.y;
    const int tid = threadIdx.x;
    if (row >= N) return;

    const int num_sb_total = K / Q5K_SB_SIZE;
    const int sb_global = (k_start / Q5K_SB_SIZE) + sb_in_chunk;
    const int row_bytes = num_sb_total * Q5K_SB_BYTES;
    const uint8_t* sb_p = weight + row * row_bytes + sb_global * Q5K_SB_BYTES;

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

    const uint8_t* qh = sb_p + 16;
    const uint8_t* qs = sb_p + 48;
    const int iter = (tid >> 6) & 3;
    const int half = (tid >> 5) & 1;
    const int l = tid & 31;
    const int sub = iter * 2 + half;
    const uint8_t byte = qs[iter * 32 + l];
    const int low = half ? (byte >> 4) : (byte & 0x0F);
    const int q = low | ((((int)qh[l] >> sub) & 1) << 4);
    const float w = (float)q * (s_d * (float)s_sc[sub]) - (s_dmin * (float)s_mn[sub]);

    const int out_k = sb_in_chunk * Q5K_SB_SIZE + sub * 32 + l;
    out[row * k_len + out_k] = __float2bfloat16(w);
}

__device__ __forceinline__ float q3k_value(const uint8_t* __restrict__ sb_p, int k_local)
{
    int8_t scales[16];
    q3k_decode_scales(sb_p + 96, scales);
    const unsigned short d_u16 = ((const unsigned short*)(sb_p + 108))[0];
    const float d = __half2float(*reinterpret_cast<const __half*>(&d_u16));
    const uint8_t* hmask = sb_p;
    const uint8_t* qs = sb_p + 32;
    const int q2 = (qs[k_local >> 2] >> ((k_local & 3) << 1)) & 0x3;
    const int hbit = (hmask[k_local >> 3] >> (k_local & 7)) & 0x1;
    const int q3 = q2 | (hbit << 2);
    return d * (float)scales[k_local >> 4] * ((float)q3 - 4.0f);
}

__device__ __forceinline__ float q4k_value(const uint8_t* __restrict__ sb_p, int k_local)
{
    uint8_t sc[8], mn[8];
    q4k_decode_scales(sb_p + 4, sc, mn);
    const unsigned short d_u16 = ((const unsigned short*)sb_p)[0];
    const unsigned short dmin_u16 = ((const unsigned short*)sb_p)[1];
    const float d = __half2float(*reinterpret_cast<const __half*>(&d_u16));
    const float dmin = __half2float(*reinterpret_cast<const __half*>(&dmin_u16));
    const int iter = k_local >> 6;
    const int half = (k_local >> 5) & 1;
    const int l = k_local & 31;
    const int sub = iter * 2 + half;
    const uint8_t byte = sb_p[16 + iter * 32 + l];
    const int q = half ? (byte >> 4) : (byte & 0x0F);
    return (float)q * (d * (float)sc[sub]) - (dmin * (float)mn[sub]);
}

__device__ __forceinline__ float q5k_value(const uint8_t* __restrict__ sb_p, int k_local)
{
    uint8_t sc[8], mn[8];
    q4k_decode_scales(sb_p + 4, sc, mn);
    const unsigned short d_u16 = ((const unsigned short*)sb_p)[0];
    const unsigned short dmin_u16 = ((const unsigned short*)sb_p)[1];
    const float d = __half2float(*reinterpret_cast<const __half*>(&d_u16));
    const float dmin = __half2float(*reinterpret_cast<const __half*>(&dmin_u16));
    const int iter = k_local >> 6;
    const int half = (k_local >> 5) & 1;
    const int l = k_local & 31;
    const int sub = iter * 2 + half;
    const uint8_t byte = sb_p[48 + iter * 32 + l];
    const int low = half ? (byte >> 4) : (byte & 0x0F);
    const int q = low | ((((int)sb_p[16 + l] >> sub) & 1) << 4);
    return (float)q * (d * (float)sc[sub]) - (dmin * (float)mn[sub]);
}

__device__ __forceinline__ float q6k_value(const uint8_t* __restrict__ sb_p, int k_local)
{
    const int half = k_local >> 7;
    const int j_local = k_local & 127;
    const int quad = j_local >> 5;
    const int l = j_local & 31;
    const int is = l >> 4;
    const uint8_t* ql = sb_p + half * 64;
    const uint8_t* qh = sb_p + 128 + half * 32;
    uint8_t low4, high2;
    switch (quad) {
        case 0: low4 = ql[l] & 0x0F;      high2 = (qh[l] >> 0) & 0x03; break;
        case 1: low4 = ql[l + 32] & 0x0F; high2 = (qh[l] >> 2) & 0x03; break;
        case 2: low4 = ql[l] >> 4;        high2 = (qh[l] >> 4) & 0x03; break;
        default: low4 = ql[l + 32] >> 4;  high2 = (qh[l] >> 6) & 0x03; break;
    }
    const int q = (int)(low4 | (high2 << 4)) - 32;
    const int8_t sc = ((const int8_t*)(sb_p + 192))[half * 8 + is + quad * 2];
    const unsigned short d_u16 = ((const unsigned short*)(sb_p + 208))[0];
    const float d = __half2float(*reinterpret_cast<const __half*>(&d_u16));
    return d * (float)sc * (float)q;
}

__global__ void qxk_embedding_batched_kernel(
    const uint8_t* __restrict__ weight,
    const int* __restrict__ token_ids,
    __nv_bfloat16* __restrict__ out,
    int hidden_dim,
    int batch_size,
    int format,
    int block_bytes)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = hidden_dim * batch_size;
    if (tid >= total) return;
    const int b = tid / hidden_dim;
    const int k = tid - b * hidden_dim;
    const int row = __ldg(&token_ids[b]);
    const int num_sb = hidden_dim / 256;
    const uint8_t* row_p = weight + row * num_sb * block_bytes;
    const uint8_t* sb_p = row_p + (k >> 8) * block_bytes;
    float value;
    switch (format) {
        case 3: value = q3k_value(sb_p, k & 255); break;
        case 4: value = q4k_value(sb_p, k & 255); break;
        case 5: value = q5k_value(sb_p, k & 255); break;
        default: value = q6k_value(sb_p, k & 255); break;
    }
    out[tid] = __float2bfloat16(value);
}

__global__ void qxk_embedding_decode_kernel(
    const uint8_t* __restrict__ weight,
    const int* __restrict__ token_id,
    __nv_bfloat16* __restrict__ out,
    int hidden_dim,
    int format,
    int block_bytes)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= hidden_dim) return;
    const int row = __ldg(&token_id[0]);
    const int num_sb = hidden_dim / 256;
    const uint8_t* row_p = weight + row * num_sb * block_bytes;
    const uint8_t* sb_p = row_p + (tid >> 8) * block_bytes;
    float value;
    switch (format) {
        case 3: value = q3k_value(sb_p, tid & 255); break;
        case 4: value = q4k_value(sb_p, tid & 255); break;
        case 5: value = q5k_value(sb_p, tid & 255); break;
        default: value = q6k_value(sb_p, tid & 255); break;
    }
    out[tid] = __float2bfloat16(value);
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

// ── Q6_K (GGUF) native packed ──

cudaError_t q6k_gemv_cuda(
    const uint8_t* weight,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    int N, int K, cudaStream_t stream)
{
    dim3 grid((N + Q6K_GEMV_ROWS - 1) / Q6K_GEMV_ROWS);
    dim3 block(Q6K_GEMV_THREADS);
    q6k_gemv_kernel<<<grid, block, 0, stream>>>(weight, input, output, N, K);
    return cudaGetLastError();
}

cudaError_t q6k_gemv_batch_cuda(
    const uint8_t* weight,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    int B, int N, int K, cudaStream_t stream)
{
    dim3 grid((N + Q6K_GEMV_ROWS - 1) / Q6K_GEMV_ROWS, B);
    dim3 block(Q6K_GEMV_THREADS);
    q6k_gemv_batch_kernel<<<grid, block, 0, stream>>>(weight, input, output, B, N, K);
    return cudaGetLastError();
}

cudaError_t q6k_dequant_chunk_cuda(
    const uint8_t* weight, __nv_bfloat16* out,
    int N, int K, int k_start, int k_len, cudaStream_t stream)
{
    if ((k_start % Q6K_SB_SIZE) != 0 || (k_len % Q6K_SB_SIZE) != 0) {
        return cudaErrorInvalidValue;
    }
    dim3 grid(N, k_len / Q6K_SB_SIZE);
    dim3 block(Q6K_SB_SIZE);
    q6k_dequant_chunk_kernel<<<grid, block, 0, stream>>>(
        weight, out, N, K, k_start, k_len);
    return cudaGetLastError();
}

// ── Q3_K (GGUF) native packed ──

cudaError_t q3k_gemv_cuda(
    const uint8_t* weight,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    int N, int K, cudaStream_t stream)
{
    dim3 grid((N + Q3K_GEMV_ROWS - 1) / Q3K_GEMV_ROWS);
    dim3 block(Q3K_GEMV_THREADS);
    q3k_gemv_kernel<<<grid, block, 0, stream>>>(weight, input, output, N, K);
    return cudaGetLastError();
}

cudaError_t q3k_gemv_batch_cuda(
    const uint8_t* weight,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    int B, int N, int K, cudaStream_t stream)
{
    dim3 grid((N + Q3K_GEMV_ROWS - 1) / Q3K_GEMV_ROWS, B);
    dim3 block(Q3K_GEMV_THREADS);
    q3k_gemv_batch_kernel<<<grid, block, 0, stream>>>(weight, input, output, B, N, K);
    return cudaGetLastError();
}

cudaError_t q3k_dequant_chunk_cuda(
    const uint8_t* weight, __nv_bfloat16* out,
    int N, int K, int k_start, int k_len, cudaStream_t stream)
{
    if ((k_start % Q3K_SB_SIZE) != 0 || (k_len % Q3K_SB_SIZE) != 0) {
        return cudaErrorInvalidValue;
    }
    dim3 grid(N, k_len / Q3K_SB_SIZE);
    dim3 block(Q3K_SB_SIZE);
    q3k_dequant_chunk_kernel<<<grid, block, 0, stream>>>(
        weight, out, N, K, k_start, k_len);
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

// ── Q5_K (GGUF) native packed ──

cudaError_t q5k_gemv_cuda(
    const uint8_t* weight,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    int N, int K, cudaStream_t stream)
{
    dim3 grid((N + Q5K_GEMV_ROWS - 1) / Q5K_GEMV_ROWS);
    dim3 block(Q5K_GEMV_THREADS);
    q5k_gemv_kernel<<<grid, block, 0, stream>>>(weight, input, output, N, K);
    return cudaGetLastError();
}

cudaError_t q5k_gemv_batch_cuda(
    const uint8_t* weight,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    int B, int N, int K, cudaStream_t stream)
{
    dim3 grid((N + Q5K_GEMV_ROWS - 1) / Q5K_GEMV_ROWS, B);
    dim3 block(Q5K_GEMV_THREADS);
    q5k_gemv_batch_kernel<<<grid, block, 0, stream>>>(weight, input, output, B, N, K);
    return cudaGetLastError();
}

cudaError_t q5k_dequant_chunk_cuda(
    const uint8_t* weight, __nv_bfloat16* out,
    int N, int K, int k_start, int k_len, cudaStream_t stream)
{
    if ((k_start % Q5K_SB_SIZE) != 0 || (k_len % Q5K_SB_SIZE) != 0) {
        return cudaErrorInvalidValue;
    }
    dim3 grid(N, k_len / Q5K_SB_SIZE);
    dim3 block(Q5K_SB_SIZE);
    q5k_dequant_chunk_kernel<<<grid, block, 0, stream>>>(
        weight, out, N, K, k_start, k_len);
    return cudaGetLastError();
}

cudaError_t q8_embedding_batched_cuda(
    const int8_t* weight, const __nv_bfloat16* scales, const int* token_ids,
    __nv_bfloat16* out, int hidden_dim, int batch_size, int group_size,
    cudaStream_t stream)
{
    if (hidden_dim <= 0 || group_size <= 0 || (hidden_dim % group_size) != 0) {
        return cudaErrorInvalidValue;
    }
    const int total = hidden_dim * batch_size;
    const int block = 256;
    const int grid = (total + block - 1) / block;
    q8_embedding_batched_kernel<<<grid, block, 0, stream>>>(
        weight, scales, token_ids, out, hidden_dim, batch_size, group_size);
    return cudaGetLastError();
}

cudaError_t q8_embedding_decode_cuda(
    const int8_t* weight, const __nv_bfloat16* scales, const int* token_id,
    __nv_bfloat16* out, int hidden_dim, int group_size, cudaStream_t stream)
{
    if (hidden_dim <= 0 || group_size <= 0 || (hidden_dim % group_size) != 0) {
        return cudaErrorInvalidValue;
    }
    const int block = 256;
    const int grid = (hidden_dim + block - 1) / block;
    q8_embedding_decode_kernel<<<grid, block, 0, stream>>>(
        weight, scales, token_id, out, hidden_dim, group_size);
    return cudaGetLastError();
}

static cudaError_t qxk_embedding_batched_cuda(
    const uint8_t* weight,
    const int* token_ids,
    __nv_bfloat16* out,
    int hidden_dim,
    int batch_size,
    int format,
    int block_bytes,
    cudaStream_t stream)
{
    if ((hidden_dim % 256) != 0) {
        return cudaErrorInvalidValue;
    }
    const int total = hidden_dim * batch_size;
    const int block = 256;
    const int grid = (total + block - 1) / block;
    qxk_embedding_batched_kernel<<<grid, block, 0, stream>>>(
        weight, token_ids, out, hidden_dim, batch_size, format, block_bytes);
    return cudaGetLastError();
}

static cudaError_t qxk_embedding_decode_cuda(
    const uint8_t* weight,
    const int* token_id,
    __nv_bfloat16* out,
    int hidden_dim,
    int format,
    int block_bytes,
    cudaStream_t stream)
{
    if ((hidden_dim % 256) != 0) {
        return cudaErrorInvalidValue;
    }
    const int block = 256;
    const int grid = (hidden_dim + block - 1) / block;
    qxk_embedding_decode_kernel<<<grid, block, 0, stream>>>(
        weight, token_id, out, hidden_dim, format, block_bytes);
    return cudaGetLastError();
}

cudaError_t q3k_embedding_batched_cuda(
    const uint8_t* weight, const int* token_ids, __nv_bfloat16* out,
    int hidden_dim, int batch_size, cudaStream_t stream)
{
    return qxk_embedding_batched_cuda(
        weight, token_ids, out, hidden_dim, batch_size, 3, Q3K_SB_BYTES, stream);
}

cudaError_t q4k_embedding_batched_cuda(
    const uint8_t* weight, const int* token_ids, __nv_bfloat16* out,
    int hidden_dim, int batch_size, cudaStream_t stream)
{
    return qxk_embedding_batched_cuda(
        weight, token_ids, out, hidden_dim, batch_size, 4, Q4K_SB_BYTES, stream);
}

cudaError_t q5k_embedding_batched_cuda(
    const uint8_t* weight, const int* token_ids, __nv_bfloat16* out,
    int hidden_dim, int batch_size, cudaStream_t stream)
{
    return qxk_embedding_batched_cuda(
        weight, token_ids, out, hidden_dim, batch_size, 5, Q5K_SB_BYTES, stream);
}

cudaError_t q6k_embedding_batched_cuda(
    const uint8_t* weight, const int* token_ids, __nv_bfloat16* out,
    int hidden_dim, int batch_size, cudaStream_t stream)
{
    return qxk_embedding_batched_cuda(
        weight, token_ids, out, hidden_dim, batch_size, 6, Q6K_SB_BYTES, stream);
}

cudaError_t q3k_embedding_decode_cuda(
    const uint8_t* weight, const int* token_id, __nv_bfloat16* out,
    int hidden_dim, cudaStream_t stream)
{
    return qxk_embedding_decode_cuda(weight, token_id, out, hidden_dim, 3, Q3K_SB_BYTES, stream);
}

cudaError_t q4k_embedding_decode_cuda(
    const uint8_t* weight, const int* token_id, __nv_bfloat16* out,
    int hidden_dim, cudaStream_t stream)
{
    return qxk_embedding_decode_cuda(weight, token_id, out, hidden_dim, 4, Q4K_SB_BYTES, stream);
}

cudaError_t q5k_embedding_decode_cuda(
    const uint8_t* weight, const int* token_id, __nv_bfloat16* out,
    int hidden_dim, cudaStream_t stream)
{
    return qxk_embedding_decode_cuda(weight, token_id, out, hidden_dim, 5, Q5K_SB_BYTES, stream);
}

cudaError_t q6k_embedding_decode_cuda(
    const uint8_t* weight, const int* token_id, __nv_bfloat16* out,
    int hidden_dim, cudaStream_t stream)
{
    return qxk_embedding_decode_cuda(weight, token_id, out, hidden_dim, 6, Q6K_SB_BYTES, stream);
}

}  // extern "C"
