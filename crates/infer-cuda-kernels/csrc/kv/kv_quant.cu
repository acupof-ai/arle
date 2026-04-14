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
// Converts 1 new token per request from bf16 working buffer to FP8 E4M3
// in the paged pool. No separate scale — FP8 E4M3 is self-contained.
//
// Grid: (num_kv_heads, batch_size)   Block: (head_dim)
// ============================================================================
__global__ void quantize_paged_kv_fp8_kernel(
    const __nv_bfloat16* __restrict__ kv_bf16,    // working buffer [max_total_tokens * kv_dim]
    __nv_fp8_e4m3* __restrict__ kv_fp8,           // FP8 pool [max_total_tokens * kv_dim]
    const int32_t* __restrict__ new_token_indices, // [batch_size] pool index of newest token
    int num_kv_heads,
    int head_dim,
    int kv_dim)
{
    int kv_head = blockIdx.x;
    int batch_idx = blockIdx.y;
    int d = threadIdx.x;

    if (d >= head_dim) return;

    int pool_idx = new_token_indices[batch_idx];
    int offset = pool_idx * kv_dim + kv_head * head_dim + d;
    float val = __bfloat162float(kv_bf16[offset]);
    kv_fp8[offset] = __nv_fp8_e4m3(val);
}

// BF16 → FP8 E4M3 quantize for contiguous → paged migration.
// Reads from HND contiguous layout, writes to NHD paged layout.
// Grid: (num_kv_heads, seq_len)   Block: (head_dim)
__global__ void quantize_scatter_kv_fp8_kernel(
    const __nv_bfloat16* __restrict__ kv_cont,    // [num_kv_heads, max_seq_len, head_dim] HND
    __nv_fp8_e4m3* __restrict__ kv_fp8,           // [max_total_tokens, kv_dim] NHD
    const int32_t* __restrict__ page_indices,     // [seq_len] pool indices
    int start_pos,
    int max_seq_len,
    int num_kv_heads,
    int head_dim,
    int kv_dim)
{
    int kv_head = blockIdx.x;
    int rel_pos = blockIdx.y;
    int d = threadIdx.x;
    if (d >= head_dim) return;

    int pos = start_pos + rel_pos;
    int pool_idx = page_indices[rel_pos];
    // Source: HND
    int src = kv_head * max_seq_len * head_dim + pos * head_dim + d;
    // Dest: NHD
    int dst = pool_idx * kv_dim + kv_head * head_dim + d;
    float val = __bfloat162float(kv_cont[src]);
    kv_fp8[dst] = __nv_fp8_e4m3(val);
}

// Quantize 1 new token per request: bf16 working → FP8 paged pool.
cudaError_t quantize_paged_kv_fp8_cuda(
    const __nv_bfloat16* kv_bf16,
    __nv_fp8_e4m3* kv_fp8,
    const int32_t* new_token_indices,
    int num_kv_heads, int head_dim, int kv_dim,
    int batch_size,
    cudaStream_t stream)
{
    if (batch_size <= 0) return cudaSuccess;
    dim3 grid(num_kv_heads, batch_size);
    dim3 block(head_dim);
    quantize_paged_kv_fp8_kernel<<<grid, block, 0, stream>>>(
        kv_bf16, kv_fp8, new_token_indices,
        num_kv_heads, head_dim, kv_dim);
    return cudaGetLastError();
}

// Quantize + scatter contiguous bf16 KV → FP8 paged pool (for migration).
cudaError_t quantize_scatter_kv_fp8_cuda(
    const __nv_bfloat16* kv_cont,
    __nv_fp8_e4m3* kv_fp8,
    const int32_t* page_indices,
    int max_seq_len, int seq_len,
    int num_kv_heads, int head_dim, int kv_dim,
    cudaStream_t stream)
{
    if (seq_len <= 0) return cudaSuccess;
    dim3 grid(num_kv_heads, seq_len);
    dim3 block(head_dim);
    quantize_scatter_kv_fp8_kernel<<<grid, block, 0, stream>>>(
        kv_cont, kv_fp8, page_indices,
        0, max_seq_len, num_kv_heads, head_dim, kv_dim);
    return cudaGetLastError();
}

cudaError_t quantize_scatter_kv_fp8_range_cuda(
    const __nv_bfloat16* kv_cont,
    __nv_fp8_e4m3* kv_fp8,
    const int32_t* page_indices,
    int start_pos, int max_seq_len, int token_count,
    int num_kv_heads, int head_dim, int kv_dim,
    cudaStream_t stream)
{
    if (token_count <= 0) return cudaSuccess;
    dim3 grid(num_kv_heads, token_count);
    dim3 block(head_dim);
    quantize_scatter_kv_fp8_kernel<<<grid, block, 0, stream>>>(
        kv_cont, kv_fp8, page_indices,
        start_pos, max_seq_len, num_kv_heads, head_dim, kv_dim);
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
    const __nv_bfloat16* __restrict__ kv_bf16,   // working buffer [max_total_tokens * kv_dim]
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

    int pool_idx = new_token_indices[batch_idx];
    int data_offset = pool_idx * kv_dim + kv_head * head_dim + d;
    float val = __bfloat162float(kv_bf16[data_offset]);

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

}  // extern "C"
