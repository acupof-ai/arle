// KV cache quantization: per-head per-token symmetric INT8.
//
// Quantize: bf16 → int8 + f32 scale
//   scale = max(|x|) / 127.0,  x_q = round(x / scale), clamped to [-127, 127]
//
// Dequantize: int8 + f32 scale → bf16
//   x = x_q * scale
//
// Cache layout (HND): [num_kv_heads, max_seq_len, head_dim]
// Scale layout:       [num_kv_heads, max_seq_len]
//
// Grid: (num_kv_heads, token_count)   Block: (head_dim)

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>

// ─── warp reduction helpers ───

__device__ __forceinline__ float warp_reduce_max_abs(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}

// ============================================================================
// Quantize: bf16 → int8 + f32 scale
//
// Processes tokens [start_pos .. start_pos + token_count).
// Grid: (num_kv_heads, token_count)   Block: (head_dim)
// head_dim must be <= 1024 and a multiple of 32 (warp size).
// ============================================================================
__global__ void quantize_kv_kernel(
    const __nv_bfloat16* __restrict__ kv_bf16,   // [num_kv_heads, max_seq_len, head_dim]
    int8_t* __restrict__ kv_int8,                 // [num_kv_heads, max_seq_len, head_dim]
    float* __restrict__ scales,                   // [num_kv_heads, max_seq_len]
    int head_dim,
    int max_seq_len,
    int start_pos)
{
    int kv_head = blockIdx.x;
    int token   = blockIdx.y;  // relative to start_pos
    int d       = threadIdx.x;
    int pos     = start_pos + token;

    if (d >= head_dim) return;

    // HND layout offset
    int offset = kv_head * max_seq_len * head_dim + pos * head_dim + d;
    float val = __bfloat162float(kv_bf16[offset]);

    // ─── compute per-head per-token absmax via warp + shared mem reduction ───
    float abs_val = fabsf(val);
    abs_val = warp_reduce_max_abs(abs_val);

    // Cross-warp reduction via shared memory
    int warp_id = d / 32;
    int lane_id = d % 32;
    int num_warps = (head_dim + 31) / 32;

    extern __shared__ float smem[];  // [num_warps]
    if (lane_id == 0) smem[warp_id] = abs_val;
    __syncthreads();

    // Final reduction in warp 0
    __shared__ float s_scale;
    if (warp_id == 0) {
        float v = (lane_id < num_warps) ? smem[lane_id] : 0.0f;
        v = warp_reduce_max_abs(v);
        if (lane_id == 0) {
            float absmax = v;
            s_scale = (absmax > 0.0f) ? (absmax / 127.0f) : 1.0f;
            // Store scale
            scales[kv_head * max_seq_len + pos] = s_scale;
        }
    }
    __syncthreads();

    // Quantize
    float scale = s_scale;
    int q = __float2int_rn(val / scale);
    q = max(-127, min(127, q));
    kv_int8[offset] = static_cast<int8_t>(q);
}

// ============================================================================
// Dequantize: int8 + f32 scale → bf16
//
// Processes tokens [0 .. token_count).
// Grid: (num_kv_heads, token_count)   Block: (head_dim)
// ============================================================================
__global__ void dequantize_kv_kernel(
    const int8_t* __restrict__ kv_int8,          // [num_kv_heads, max_seq_len, head_dim]
    const float* __restrict__ scales,            // [num_kv_heads, max_seq_len]
    __nv_bfloat16* __restrict__ kv_bf16,         // [num_kv_heads, max_seq_len, head_dim]
    int head_dim,
    int max_seq_len)
{
    int kv_head = blockIdx.x;
    int pos     = blockIdx.y;
    int d       = threadIdx.x;

    if (d >= head_dim) return;

    int offset = kv_head * max_seq_len * head_dim + pos * head_dim + d;
    float scale = scales[kv_head * max_seq_len + pos];
    float val = static_cast<float>(kv_int8[offset]) * scale;
    kv_bf16[offset] = __float2bfloat16(val);
}

// ============================================================================
// C API
// ============================================================================
extern "C" {

// Quantize bf16 KV data to INT8 for tokens [start_pos .. start_pos + token_count).
// kv_bf16 and kv_int8 share the same HND layout: [num_kv_heads, max_seq_len, head_dim].
// scales layout: [num_kv_heads, max_seq_len].
cudaError_t quantize_kv_bf16_to_int8_cuda(
    const __nv_bfloat16* kv_bf16,
    int8_t* kv_int8,
    float* scales,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    int start_pos,
    int token_count,
    cudaStream_t stream)
{
    if (token_count <= 0) return cudaSuccess;
    dim3 grid(num_kv_heads, token_count);
    dim3 block(head_dim);
    int smem_bytes = ((head_dim + 31) / 32) * sizeof(float);
    quantize_kv_kernel<<<grid, block, smem_bytes, stream>>>(
        kv_bf16, kv_int8, scales, head_dim, max_seq_len, start_pos);
    return cudaGetLastError();
}

// Dequantize INT8 KV data to bf16 for tokens [0 .. token_count).
// Same layout conventions as quantize.
cudaError_t dequantize_kv_int8_to_bf16_cuda(
    const int8_t* kv_int8,
    const float* scales,
    __nv_bfloat16* kv_bf16,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    int token_count,
    cudaStream_t stream)
{
    if (token_count <= 0) return cudaSuccess;
    dim3 grid(num_kv_heads, token_count);
    dim3 block(head_dim);
    dequantize_kv_kernel<<<grid, block, 0, stream>>>(
        kv_int8, scales, kv_bf16, head_dim, max_seq_len);
    return cudaGetLastError();
}

// ============================================================================
// BF16 → FP8 E4M3 quantize for paged KV pool (NHD layout).
//
// Converts token rows from the bf16 HND working buffer to scaled FP8 E4M3
// durable storage in NHD row layout.
//
// Grid: (num_kv_heads, batch_size)   Block: (head_dim)
// ============================================================================
__global__ void quantize_paged_kv_fp8_kernel(
    const __nv_bfloat16* __restrict__ kv_bf16,    // working buffer [page, head, token, dim]
    __nv_fp8_e4m3* __restrict__ kv_fp8,           // FP8 pool [max_total_tokens * kv_dim]
    float* __restrict__ scales,                   // [max_total_tokens * num_kv_heads]
    const int32_t* __restrict__ new_token_indices, // [batch_size] token row of newest token
    int num_kv_heads,
    int head_dim,
    int kv_dim)
{
    int kv_head = blockIdx.x;
    int batch_idx = blockIdx.y;
    int d = threadIdx.x;

    if (d >= head_dim) return;

    constexpr int kPageSize = 16;
    int token_row = new_token_indices[batch_idx];
    int page_idx = token_row / kPageSize;
    int offset_in_page = token_row % kPageSize;
    int row_idx = page_idx * kPageSize + offset_in_page;
    int src_offset = page_idx * kPageSize * kv_dim
                   + kv_head * kPageSize * head_dim
                   + offset_in_page * head_dim
                   + d;
    int dst_offset = row_idx * kv_dim + kv_head * head_dim + d;
    float val = __bfloat162float(kv_bf16[src_offset]);

    float abs_val = warp_reduce_max_abs(fabsf(val));
    int warp_id = d / 32;
    int lane_id = d % 32;
    int num_warps = (head_dim + 31) / 32;
    extern __shared__ float smem[];
    if (lane_id == 0) smem[warp_id] = abs_val;
    __syncthreads();

    __shared__ float s_scale;
    if (warp_id == 0) {
        float v = (lane_id < num_warps) ? smem[lane_id] : 0.0f;
        v = warp_reduce_max_abs(v);
        if (lane_id == 0) {
            // Match the INT8 guard: only protect against divide-by-zero, no
            // numerical floor. The earlier 1e-6 floor activated whenever the
            // row absmax fell below 4.48e-4 (typical at deep Qwen3 layers
            // where K activations shrink), forcing val/s_scale to underflow
            // FP8 E4M3's subnormal threshold and round to ±0 — the root
            // cause of the 2026-05-26 mean_match=0.0156 catastrophic
            // divergence (layer 17/35 K bytes mostly 0x00/0x80 in the
            // INFER_FP8_DEBUG dump).
            s_scale = (v > 0.0f) ? (v / 448.0f) : 1.0f;
            scales[row_idx * num_kv_heads + kv_head] = s_scale;
        }
    }
    __syncthreads();
    kv_fp8[dst_offset] = __nv_fp8_e4m3(val / s_scale);
}

// BF16 → FP8 E4M3 quantize for contiguous → paged migration.
// Reads from HND contiguous layout, writes to NHD paged layout.
// Grid: (num_kv_heads, seq_len)   Block: ceil(head_dim / 4), rounded to a warp
__global__ void quantize_scatter_kv_fp8_kernel(
    const __nv_bfloat16* __restrict__ kv_cont,    // [num_kv_heads, max_seq_len, head_dim] HND
    __nv_fp8_e4m3* __restrict__ kv_fp8,           // [max_total_tokens, kv_dim] NHD
    float* __restrict__ scales,                   // [max_total_tokens, num_kv_heads]
    const int32_t* __restrict__ page_indices,     // [token_count] token rows
    int start_pos,
    int max_seq_len,
    int num_kv_heads,
    int head_dim,
    int kv_dim)
{
    int kv_head = blockIdx.x;
    int rel_pos = blockIdx.y;
    int d = threadIdx.x * 4;

    int pos = start_pos + rel_pos;
    constexpr int kPageSize = 16;
    int token_row = page_indices[rel_pos];
    int page_idx = token_row / kPageSize;
    int offset_in_page = token_row % kPageSize;
    int row_idx = page_idx * kPageSize + offset_in_page;
    // Source: HND
    int src = kv_head * max_seq_len * head_dim + pos * head_dim + d;
    // Dest: NHD
    int dst = row_idx * kv_dim + kv_head * head_dim + d;
    float val0 = 0.0f;
    float val1 = 0.0f;
    float val2 = 0.0f;
    float val3 = 0.0f;
    if (d + 3 < head_dim && (src & 1) == 0) {
        __nv_bfloat162 packed01 =
            *reinterpret_cast<const __nv_bfloat162*>(kv_cont + src);
        __nv_bfloat162 packed23 =
            *reinterpret_cast<const __nv_bfloat162*>(kv_cont + src + 2);
        float2 vals01 = __bfloat1622float2(packed01);
        float2 vals23 = __bfloat1622float2(packed23);
        val0 = vals01.x;
        val1 = vals01.y;
        val2 = vals23.x;
        val3 = vals23.y;
    } else {
        val0 = (d < head_dim) ? __bfloat162float(kv_cont[src]) : 0.0f;
        val1 = (d + 1 < head_dim) ? __bfloat162float(kv_cont[src + 1]) : 0.0f;
        val2 = (d + 2 < head_dim) ? __bfloat162float(kv_cont[src + 2]) : 0.0f;
        val3 = (d + 3 < head_dim) ? __bfloat162float(kv_cont[src + 3]) : 0.0f;
    }

    float abs_val = fmaxf(fabsf(val0), fabsf(val1));
    abs_val = fmaxf(abs_val, fabsf(val2));
    abs_val = warp_reduce_max_abs(fmaxf(abs_val, fabsf(val3)));
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int num_warps = (blockDim.x + 31) / 32;
    extern __shared__ float smem[];
    if (lane_id == 0) smem[warp_id] = abs_val;
    __syncthreads();

    __shared__ float s_scale;
    if (warp_id == 0) {
        float v = (lane_id < num_warps) ? smem[lane_id] : 0.0f;
        v = warp_reduce_max_abs(v);
        if (lane_id == 0) {
            // Same guard as the per-token decode quant kernel: no
            // 1e-6 numerical floor, only divide-by-zero protection.
            s_scale = (v > 0.0f) ? (v / 448.0f) : 1.0f;
            scales[row_idx * num_kv_heads + kv_head] = s_scale;
        }
    }
    __syncthreads();
    if (d + 3 < head_dim && (dst & 3) == 0) {
        float4 scaled = make_float4(
            val0 / s_scale,
            val1 / s_scale,
            val2 / s_scale,
            val3 / s_scale);
        *reinterpret_cast<__nv_fp8x4_e4m3*>(kv_fp8 + dst) = __nv_fp8x4_e4m3(scaled);
    } else {
        if (d < head_dim) {
            kv_fp8[dst] = __nv_fp8_e4m3(val0 / s_scale);
        }
        if (d + 1 < head_dim) {
            kv_fp8[dst + 1] = __nv_fp8_e4m3(val1 / s_scale);
        }
        if (d + 2 < head_dim) {
            kv_fp8[dst + 2] = __nv_fp8_e4m3(val2 / s_scale);
        }
        if (d + 3 < head_dim) {
            kv_fp8[dst + 3] = __nv_fp8_e4m3(val3 / s_scale);
        }
    }
}

// Quantize 1 new token per request: bf16 working → FP8 paged pool.
cudaError_t quantize_paged_kv_fp8_cuda(
    const __nv_bfloat16* kv_bf16,
    __nv_fp8_e4m3* kv_fp8,
    float* scales,
    const int32_t* new_token_indices,
    int num_kv_heads, int head_dim, int kv_dim,
    int batch_size,
    cudaStream_t stream)
{
    if (batch_size <= 0) return cudaSuccess;
    dim3 grid(num_kv_heads, batch_size);
    dim3 block(head_dim);
    int smem_bytes = ((head_dim + 31) / 32) * sizeof(float);
    quantize_paged_kv_fp8_kernel<<<grid, block, smem_bytes, stream>>>(
        kv_bf16, kv_fp8, scales, new_token_indices,
        num_kv_heads, head_dim, kv_dim);
    return cudaGetLastError();
}

// Quantize + scatter contiguous bf16 KV → FP8 paged pool (for migration).
cudaError_t quantize_scatter_kv_fp8_cuda(
    const __nv_bfloat16* kv_cont,
    __nv_fp8_e4m3* kv_fp8,
    float* scales,
    const int32_t* page_indices,
    int max_seq_len, int seq_len,
    int num_kv_heads, int head_dim, int kv_dim,
    cudaStream_t stream)
{
    if (seq_len <= 0) return cudaSuccess;
    dim3 grid(num_kv_heads, seq_len);
    int group_threads = (head_dim + 3) / 4;
    int block_threads = ((group_threads + 31) / 32) * 32;
    dim3 block(block_threads);
    int smem_bytes = ((block_threads + 31) / 32) * sizeof(float);
    quantize_scatter_kv_fp8_kernel<<<grid, block, smem_bytes, stream>>>(
        kv_cont, kv_fp8, scales, page_indices,
        0, max_seq_len, num_kv_heads, head_dim, kv_dim);
    return cudaGetLastError();
}

cudaError_t quantize_scatter_kv_fp8_range_cuda(
    const __nv_bfloat16* kv_cont,
    __nv_fp8_e4m3* kv_fp8,
    float* scales,
    const int32_t* page_indices,
    int start_pos, int max_seq_len, int token_count,
    int num_kv_heads, int head_dim, int kv_dim,
    cudaStream_t stream)
{
    if (token_count <= 0) return cudaSuccess;
    dim3 grid(num_kv_heads, token_count);
    int group_threads = (head_dim + 3) / 4;
    int block_threads = ((group_threads + 31) / 32) * 32;
    dim3 block(block_threads);
    int smem_bytes = ((block_threads + 31) / 32) * sizeof(float);
    quantize_scatter_kv_fp8_kernel<<<grid, block, smem_bytes, stream>>>(
        kv_cont, kv_fp8, scales, page_indices,
        start_pos, max_seq_len, num_kv_heads, head_dim, kv_dim);
    return cudaGetLastError();
}

// Durable FP8 NHD → BF16 HND work-buffer refill for paged prefill.
//
// The quantized decode kernels read durable FP8 as NHD token rows:
//   [page, token, head, dim].
// TileLang paged prefill reads BF16 work as HND pages:
//   [page, head, token, dim].
// Refill only the historical prefix rows before the prefill prep kernel
// overwrites the current chunk rows in the same BF16 work buffer.
__global__ void dequantize_paged_kv_fp8_to_hnd_kernel(
    const __nv_fp8_e4m3* __restrict__ kv_fp8,
    const float* __restrict__ scales,
    __nv_bfloat16* __restrict__ kv_bf16_hnd,
    const int32_t* __restrict__ token_rows,
    int num_kv_heads,
    int head_dim,
    int kv_dim)
{
    int kv_head = blockIdx.x;
    int tok_flat = blockIdx.y;
    int d = threadIdx.x * 4;
    if (d >= head_dim) return;

    int token_row = token_rows[tok_flat];
    constexpr int kPageSize = 16;
    int page_idx = token_row / kPageSize;
    int offset_in_page = token_row % kPageSize;
    int src_offset = token_row * kv_dim + kv_head * head_dim + d;
    int scale_offset = token_row * num_kv_heads + kv_head;
    int dst_offset = page_idx * kPageSize * kv_dim
                   + kv_head * kPageSize * head_dim
                   + offset_in_page * head_dim
                   + d;
    float scale = scales[scale_offset];
    if (d + 3 < head_dim && ((src_offset | dst_offset) & 3) == 0) {
        __nv_fp8x4_e4m3 packed =
            *reinterpret_cast<const __nv_fp8x4_e4m3*>(kv_fp8 + src_offset);
        float4 vals = static_cast<float4>(packed);
        __nv_bfloat162 out01 = __floats2bfloat162_rn(vals.x * scale, vals.y * scale);
        __nv_bfloat162 out23 = __floats2bfloat162_rn(vals.z * scale, vals.w * scale);
        *reinterpret_cast<__nv_bfloat162*>(kv_bf16_hnd + dst_offset) = out01;
        *reinterpret_cast<__nv_bfloat162*>(kv_bf16_hnd + dst_offset + 2) = out23;
    } else {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (d + i < head_dim) {
                float val = static_cast<float>(kv_fp8[src_offset + i]) * scale;
                kv_bf16_hnd[dst_offset + i] = __float2bfloat16(val);
            }
        }
    }
}

cudaError_t dequantize_paged_kv_fp8_to_hnd_cuda(
    const __nv_fp8_e4m3* kv_fp8,
    const float* scales,
    __nv_bfloat16* kv_bf16_hnd,
    const int32_t* token_rows,
    int num_kv_heads,
    int head_dim,
    int kv_dim,
    int total_tokens,
    cudaStream_t stream)
{
    if (total_tokens <= 0) return cudaSuccess;
    dim3 grid(num_kv_heads, total_tokens);
    dim3 block((head_dim + 3) / 4);
    dequantize_paged_kv_fp8_to_hnd_kernel<<<grid, block, 0, stream>>>(
        kv_fp8, scales, kv_bf16_hnd, token_rows, num_kv_heads, head_dim, kv_dim);
    return cudaGetLastError();
}

// Durable INT8 NHD → BF16 HND work-buffer refill for paged prefill.
__global__ void dequantize_paged_kv_int8_to_hnd_kernel(
    const int8_t* __restrict__ kv_int8,
    const float* __restrict__ scales,
    __nv_bfloat16* __restrict__ kv_bf16_hnd,
    const int32_t* __restrict__ token_rows,
    int num_kv_heads,
    int head_dim,
    int kv_dim)
{
    int kv_head = blockIdx.x;
    int tok_flat = blockIdx.y;
    int d = threadIdx.x * 2;
    if (d >= head_dim) return;

    int token_row = token_rows[tok_flat];
    constexpr int kPageSize = 16;
    int page_idx = token_row / kPageSize;
    int offset_in_page = token_row % kPageSize;
    int src_offset = token_row * kv_dim + kv_head * head_dim + d;
    int scale_offset = token_row * num_kv_heads + kv_head;
    int dst_offset = page_idx * kPageSize * kv_dim
                   + kv_head * kPageSize * head_dim
                   + offset_in_page * head_dim
                   + d;
    float scale = scales[scale_offset];
    if (d + 1 < head_dim) {
        if (((src_offset | dst_offset) & 1) == 0) {
            uint16_t packed = *reinterpret_cast<const uint16_t*>(kv_int8 + src_offset);
            int8_t lo = static_cast<int8_t>(packed & 0xffu);
            int8_t hi = static_cast<int8_t>((packed >> 8) & 0xffu);
            __nv_bfloat162 out = __floats2bfloat162_rn(
                static_cast<float>(lo) * scale,
                static_cast<float>(hi) * scale);
            *reinterpret_cast<__nv_bfloat162*>(kv_bf16_hnd + dst_offset) = out;
        } else {
            float val0 = static_cast<float>(kv_int8[src_offset]) * scale;
            float val1 = static_cast<float>(kv_int8[src_offset + 1]) * scale;
            kv_bf16_hnd[dst_offset] = __float2bfloat16(val0);
            kv_bf16_hnd[dst_offset + 1] = __float2bfloat16(val1);
        }
    } else {
        float val = static_cast<float>(kv_int8[src_offset]) * scale;
        kv_bf16_hnd[dst_offset] = __float2bfloat16(val);
    }
}

cudaError_t dequantize_paged_kv_int8_to_hnd_cuda(
    const int8_t* kv_int8,
    const float* scales,
    __nv_bfloat16* kv_bf16_hnd,
    const int32_t* token_rows,
    int num_kv_heads,
    int head_dim,
    int kv_dim,
    int total_tokens,
    cudaStream_t stream)
{
    if (total_tokens <= 0) return cudaSuccess;
    dim3 grid(num_kv_heads, total_tokens);
    dim3 block((head_dim + 1) / 2);
    dequantize_paged_kv_int8_to_hnd_kernel<<<grid, block, 0, stream>>>(
        kv_int8, scales, kv_bf16_hnd, token_rows, num_kv_heads, head_dim, kv_dim);
    return cudaGetLastError();
}

// ============================================================================
// Dequantize paged INT8 KV → bf16 working buffer (NHD paged layout).
//
// Reads INT8 data + f32 scales at scattered pool indices and writes bf16
// to the same pool indices in the working buffer.
//
// NHD data layout:  pool_idx * kv_dim + kv_head * head_dim + d
// NHD scale layout: pool_idx * num_kv_heads + kv_head
//
// Grid: (num_kv_heads, total_tokens)   Block: (head_dim)
// ============================================================================
__global__ void dequantize_paged_kv_kernel(
    const int8_t* __restrict__ kv_int8,          // [max_total_tokens * kv_dim]
    const float* __restrict__ scales,            // [max_total_tokens * num_kv_heads]
    __nv_bfloat16* __restrict__ kv_bf16,         // [max_total_tokens * kv_dim]
    const int32_t* __restrict__ token_indices,   // [total_tokens] pool indices
    int num_kv_heads,
    int head_dim,
    int kv_dim)
{
    int kv_head = blockIdx.x;
    int tok_flat = blockIdx.y;
    int d = threadIdx.x;

    if (d >= head_dim) return;

    int pool_idx = token_indices[tok_flat];
    int data_offset = pool_idx * kv_dim + kv_head * head_dim + d;
    int scale_offset = pool_idx * num_kv_heads + kv_head;

    float scale = scales[scale_offset];
    float val = static_cast<float>(kv_int8[data_offset]) * scale;
    kv_bf16[data_offset] = __float2bfloat16(val);
}

// ============================================================================
// Quantize new tokens (1 per request) from bf16 working → INT8 paged pool.
//
// Grid: (num_kv_heads, batch_size)   Block: (head_dim)
// head_dim must be <= 1024 and a multiple of 32.
// ============================================================================
__global__ void quantize_paged_kv_single_kernel(
    const __nv_bfloat16* __restrict__ kv_bf16,   // HND work buffer [page, head, token, dim]
    int8_t* __restrict__ kv_int8,                 // INT8 pool [max_total_tokens * kv_dim]
    float* __restrict__ scales,                   // [max_total_tokens * num_kv_heads]
    const int32_t* __restrict__ new_token_indices, // [batch_size] pool index of newest token
    int num_kv_heads,
    int head_dim,
    int kv_dim)
{
    int kv_head = blockIdx.x;
    int batch_idx = blockIdx.y;
    int d = threadIdx.x;

    if (d >= head_dim) return;

    constexpr int kPageSize = 16;
    int pool_idx = new_token_indices[batch_idx];
    int page_idx = pool_idx / kPageSize;
    int offset_in_page = pool_idx % kPageSize;
    int src_offset = page_idx * kPageSize * kv_dim
                   + kv_head * kPageSize * head_dim
                   + offset_in_page * head_dim
                   + d;
    int data_offset = pool_idx * kv_dim + kv_head * head_dim + d;
    float val = __bfloat162float(kv_bf16[src_offset]);

    // ─── per-head per-token absmax via warp + shared mem reduction ───
    float abs_val = fabsf(val);
    abs_val = warp_reduce_max_abs(abs_val);

    int warp_id = d / 32;
    int lane_id = d % 32;
    int num_warps = (head_dim + 31) / 32;

    extern __shared__ float smem[];
    if (lane_id == 0) smem[warp_id] = abs_val;
    __syncthreads();

    __shared__ float s_scale;
    if (warp_id == 0) {
        float v = (lane_id < num_warps) ? smem[lane_id] : 0.0f;
        v = warp_reduce_max_abs(v);
        if (lane_id == 0) {
            float absmax = v;
            s_scale = (absmax > 0.0f) ? (absmax / 127.0f) : 1.0f;
            int scale_offset = pool_idx * num_kv_heads + kv_head;
            scales[scale_offset] = s_scale;
        }
    }
    __syncthreads();

    float scale = s_scale;
    int q = __float2int_rn(val / scale);
    q = max(-127, min(127, q));
    kv_int8[data_offset] = static_cast<int8_t>(q);
}

// Dequantize paged INT8 KV to bf16 working buffer for all tokens in the batch.
cudaError_t dequantize_paged_kv_cuda(
    const int8_t* kv_int8,
    const float* scales,
    __nv_bfloat16* kv_bf16,
    const int32_t* token_indices,
    int num_kv_heads,
    int head_dim,
    int kv_dim,
    int total_tokens,
    cudaStream_t stream)
{
    if (total_tokens <= 0) return cudaSuccess;
    dim3 grid(num_kv_heads, total_tokens);
    dim3 block(head_dim);
    dequantize_paged_kv_kernel<<<grid, block, 0, stream>>>(
        kv_int8, scales, kv_bf16, token_indices,
        num_kv_heads, head_dim, kv_dim);
    return cudaGetLastError();
}

// Quantize 1 new token per request from bf16 working to INT8 paged pool.
cudaError_t quantize_paged_kv_single_cuda(
    const __nv_bfloat16* kv_bf16,
    int8_t* kv_int8,
    float* scales,
    const int32_t* new_token_indices,
    int num_kv_heads,
    int head_dim,
    int kv_dim,
    int batch_size,
    cudaStream_t stream)
{
    if (batch_size <= 0) return cudaSuccess;
    dim3 grid(num_kv_heads, batch_size);
    dim3 block(head_dim);
    int smem_bytes = ((head_dim + 31) / 32) * sizeof(float);
    quantize_paged_kv_single_kernel<<<grid, block, smem_bytes, stream>>>(
        kv_bf16, kv_int8, scales, new_token_indices,
        num_kv_heads, head_dim, kv_dim);
    return cudaGetLastError();
}

// ============================================================================
// KIVI per-channel K scale quantization (FP8 E4M3 only).
//
// Reads K from the HND-paged bf16 working buffer and writes FP8 E4M3 to the
// NHD durable pool, using a **pre-computed per-(kv_head, head_dim) scale
// table** rather than computing per-(token, head) absmax on the fly. The
// scale table layout is `[num_kv_heads, head_dim]` f32 (one scale per
// channel of each KV head), populated once per layer at calibration time
// (via `compute_k_per_channel_absmax_cuda` below on the first prefill batch).
//
// This is the KIVI ICML 2024 fix for K's outlier-channel-dominated
// quantization error that catastrophically compounds through deep dense
// transformers. V keeps its per-(token, head) scales (KIVI's asymmetric
// choice — V doesn't show the same outlier-channel structure).
// ============================================================================

__global__ void quantize_paged_kv_fp8_per_channel_kernel(
    const __nv_bfloat16* __restrict__ kv_bf16,      // HND-paged work buffer [page, head, token, dim]
    __nv_fp8_e4m3* __restrict__ kv_fp8,             // FP8 pool [max_total_tokens * kv_dim]
    const float* __restrict__ k_static_scales,      // [num_kv_heads, head_dim] per-channel scale
    const int32_t* __restrict__ new_token_indices,  // [batch_size] token row of newest token
    int num_kv_heads,
    int head_dim,
    int kv_dim)
{
    int kv_head = blockIdx.x;
    int batch_idx = blockIdx.y;
    int d = threadIdx.x;

    if (d >= head_dim) return;

    constexpr int kPageSize = 16;
    int token_row = new_token_indices[batch_idx];
    int page_idx = token_row / kPageSize;
    int offset_in_page = token_row % kPageSize;
    int row_idx = page_idx * kPageSize + offset_in_page;
    int src_offset = page_idx * kPageSize * kv_dim
                   + kv_head * kPageSize * head_dim
                   + offset_in_page * head_dim
                   + d;
    int dst_offset = row_idx * kv_dim + kv_head * head_dim + d;

    float val = __bfloat162float(kv_bf16[src_offset]);
    float scale = k_static_scales[kv_head * head_dim + d];
    // Threshold matches the finalize floor (1e-30). Above that we trust the
    // calibration; at-or-below we treat as untouched channel (val should be
    // ~0 too, so producing 0 is correct).
    float inv_scale = (scale > 1.0e-29f) ? (1.0f / scale) : 0.0f;
    kv_fp8[dst_offset] = __nv_fp8_e4m3(val * inv_scale);
}

// ============================================================================
// Per-channel K absmax: scan a batch of K rows and update the per-channel
// scale table. Uses atomicMax-on-bits to maintain the running absmax
// across multiple invocations (so multiple prefill batches contribute).
// Final scale = running_absmax / 448.0 (FP8 E4M3 max representable).
// ============================================================================

static __device__ __forceinline__ void atomic_max_float(float* addr, float val) {
    int* iaddr = reinterpret_cast<int*>(addr);
    int old = __float_as_int(*addr);
    int assumed;
    do {
        if (__int_as_float(old) >= val) return;
        assumed = old;
        old = atomicCAS(iaddr, assumed, __float_as_int(val));
    } while (assumed != old);
}

__global__ void compute_k_per_channel_absmax_kernel(
    const __nv_bfloat16* __restrict__ kv_bf16,      // HND-paged work [page, head, token, dim]
    float* __restrict__ k_static_scales,            // [num_kv_heads, head_dim] running absmax
    const int32_t* __restrict__ token_rows,         // [batch_size] token rows to scan
    int num_kv_heads,
    int head_dim,
    int kv_dim)
{
    int kv_head = blockIdx.x;
    int batch_idx = blockIdx.y;
    int d = threadIdx.x;

    if (d >= head_dim) return;

    constexpr int kPageSize = 16;
    int token_row = token_rows[batch_idx];
    int page_idx = token_row / kPageSize;
    int offset_in_page = token_row % kPageSize;
    int src_offset = page_idx * kPageSize * kv_dim
                   + kv_head * kPageSize * head_dim
                   + offset_in_page * head_dim
                   + d;

    float val = fabsf(__bfloat162float(kv_bf16[src_offset]));
    int channel_idx = kv_head * head_dim + d;
    atomic_max_float(&k_static_scales[channel_idx], val);
}

// (Already inside the outer `extern "C" {` block — no extra block needed.)

// Quantize new K tokens using a pre-computed per-channel scale table.
// Mirrors `quantize_paged_kv_fp8_cuda` API but consumes static per-channel
// scales instead of computing per-(token, head) absmax. Caller is
// responsible for populating `k_static_scales` (via
// `compute_k_per_channel_absmax_cuda` during calibration / first prefill).
cudaError_t quantize_paged_kv_fp8_per_channel_cuda(
    const __nv_bfloat16* kv_bf16,
    __nv_fp8_e4m3* kv_fp8,
    const float* k_static_scales,
    const int32_t* new_token_indices,
    int num_kv_heads, int head_dim, int kv_dim,
    int batch_size,
    cudaStream_t stream)
{
    if (batch_size <= 0) return cudaSuccess;
    dim3 grid(num_kv_heads, batch_size);
    dim3 block(head_dim);
    quantize_paged_kv_fp8_per_channel_kernel<<<grid, block, 0, stream>>>(
        kv_bf16, kv_fp8, k_static_scales, new_token_indices,
        num_kv_heads, head_dim, kv_dim);
    return cudaGetLastError();
}

// Compute / accumulate per-channel K absmax from a batch of bf16 K rows
// in the HND-paged work buffer. Stores **raw absmax** into k_static_scales
// (caller divides by 448.0 to get FP8 E4M3 scale after all calibration
// batches are consumed). Called via `finalize_paged_prefill_kv_layer`
// before per-channel quantization on the first FP8 prefill batch.
cudaError_t compute_k_per_channel_absmax_cuda(
    const __nv_bfloat16* kv_bf16,
    float* k_static_scales,
    const int32_t* token_rows,
    int num_kv_heads, int head_dim, int kv_dim,
    int batch_size,
    cudaStream_t stream)
{
    if (batch_size <= 0) return cudaSuccess;
    dim3 grid(num_kv_heads, batch_size);
    dim3 block(head_dim);
    compute_k_per_channel_absmax_kernel<<<grid, block, 0, stream>>>(
        kv_bf16, k_static_scales, token_rows,
        num_kv_heads, head_dim, kv_dim);
    return cudaGetLastError();
}

// Finalize the per-channel scale table by dividing accumulated absmax by
// FP8 E4M3 max (448.0). Idempotent if all values are already <= 448 (the
// first call divides by 448, subsequent calls just re-divide further —
// caller must only invoke once after all calibration batches).
__global__ void finalize_k_per_channel_scales_kernel(
    float* k_static_scales,
    int num_channels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_channels) return;
    float v = k_static_scales[idx];
    // Divide accumulated absmax by FP8 E4M3 max (448) to yield the scale.
    // Floor at 1e-30 (essentially only catches truly-zero absmax — untouched
    // channels) so real small-magnitude channels keep their natural scale
    // instead of being artificially inflated, which would saturate the
    // quantized representation. Earlier 1e-6 floor caused FP8 KV to look
    // broken even with KIVI scales — see commit history for context.
    k_static_scales[idx] = fmaxf(v / 448.0f, 1.0e-30f);
}

cudaError_t finalize_k_per_channel_scales_cuda(
    float* k_static_scales,
    int num_channels,
    cudaStream_t stream)
{
    if (num_channels <= 0) return cudaSuccess;
    int block = 256;
    int grid = (num_channels + block - 1) / block;
    finalize_k_per_channel_scales_kernel<<<grid, block, 0, stream>>>(
        k_static_scales, num_channels);
    return cudaGetLastError();
}

}  // extern "C"
