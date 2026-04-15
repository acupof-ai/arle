// Copy KV cache from contiguous layout to FlashInfer paged layout.
//
// Contiguous layout: [num_kv_heads, max_seq_len, head_dim] (per layer)
//   offset = kv_head * max_seq_len * head_dim + pos * head_dim + dim
//
// Paged layout (HND): [max_pages, num_kv_heads, page_size, head_dim]
//   logical_page = pos / page_size
//   offset_in_page = pos % page_size
//   physical_page = page_indices[indptr[batch_idx] + logical_page]
//   offset = physical_page * stride_page + kv_head * page_size * head_dim + offset_in_page * head_dim + dim
//
// Grid: (num_kv_heads, seq_len)  Threads: head_dim

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

__global__ void kv_cache_to_paged_kernel(
    const __nv_bfloat16* __restrict__ k_contiguous,  // [num_kv_heads, max_seq_len, head_dim]
    const __nv_bfloat16* __restrict__ v_contiguous,  // [num_kv_heads, max_seq_len, head_dim]
    __nv_bfloat16* __restrict__ k_paged,             // paged pool
    __nv_bfloat16* __restrict__ v_paged,             // paged pool
    const int32_t* __restrict__ page_indices,        // page table for this slot
    int max_seq_len,         // contiguous max_seq_len
    int seq_len,             // actual tokens to copy
    int num_kv_heads,
    int page_size,
    int head_dim,
    int stride_page)         // num_kv_heads * page_size * head_dim
{
    int kv_head = blockIdx.x;
    int pos = blockIdx.y;
    int dim = threadIdx.x;

    if (pos >= seq_len || dim >= head_dim) return;

    // Source: contiguous offset
    int src_offset = kv_head * max_seq_len * head_dim + pos * head_dim + dim;

    // Destination: paged offset
    int logical_page = pos / page_size;
    int offset_in_page = pos % page_size;
    int physical_page = page_indices[logical_page];
    int dst_offset = physical_page * stride_page
                   + kv_head * page_size * head_dim
                   + offset_in_page * head_dim
                   + dim;

    k_paged[dst_offset] = k_contiguous[src_offset];
    v_paged[dst_offset] = v_contiguous[src_offset];
}

// Range variant for token-level (page_size=1) paged pools.
// Copies contiguous positions [start_pos, start_pos + token_count) into the
// provided physical token slots.
__global__ void kv_cache_to_paged_range_kernel(
    const __nv_bfloat16* __restrict__ k_contiguous,  // [num_kv_heads, max_seq_len, head_dim]
    const __nv_bfloat16* __restrict__ v_contiguous,  // [num_kv_heads, max_seq_len, head_dim]
    __nv_bfloat16* __restrict__ k_paged,             // [max_tokens, kv_dim]
    __nv_bfloat16* __restrict__ v_paged,             // [max_tokens, kv_dim]
    const int32_t* __restrict__ token_indices,       // [token_count] physical pool indices
    int start_pos,
    int max_seq_len,
    int token_count,
    int num_kv_heads,
    int head_dim,
    int kv_dim)
{
    int kv_head = blockIdx.x;
    int rel_pos = blockIdx.y;
    int dim = threadIdx.x;

    if (rel_pos >= token_count || dim >= head_dim) return;

    int pos = start_pos + rel_pos;
    int src_offset = kv_head * max_seq_len * head_dim + pos * head_dim + dim;
    int pool_idx = token_indices[rel_pos];
    int dst_offset = pool_idx * kv_dim + kv_head * head_dim + dim;

    k_paged[dst_offset] = k_contiguous[src_offset];
    v_paged[dst_offset] = v_contiguous[src_offset];
}

// Range variant for HND paged pools.
// Copies contiguous positions [start_pos, start_pos + token_count) using the
// full page table for the slot.
__global__ void kv_cache_to_paged_range_hnd_kernel(
    const __nv_bfloat16* __restrict__ k_contiguous,  // [num_kv_heads, max_seq_len, head_dim]
    const __nv_bfloat16* __restrict__ v_contiguous,  // [num_kv_heads, max_seq_len, head_dim]
    __nv_bfloat16* __restrict__ k_paged,             // [max_pages, num_kv_heads, page_size, head_dim]
    __nv_bfloat16* __restrict__ v_paged,             // [max_pages, num_kv_heads, page_size, head_dim]
    const int32_t* __restrict__ page_indices,        // full page table for this slot
    int start_pos,
    int max_seq_len,
    int token_count,
    int num_kv_heads,
    int page_size,
    int head_dim,
    int stride_page)
{
    int kv_head = blockIdx.x;
    int rel_pos = blockIdx.y;
    int dim = threadIdx.x;

    if (rel_pos >= token_count || dim >= head_dim) return;

    int pos = start_pos + rel_pos;
    int src_offset = kv_head * max_seq_len * head_dim + pos * head_dim + dim;

    int logical_page = pos / page_size;
    int offset_in_page = pos % page_size;
    int physical_page = page_indices[logical_page];
    int dst_offset = physical_page * stride_page
                   + kv_head * page_size * head_dim
                   + offset_in_page * head_dim
                   + dim;

    k_paged[dst_offset] = k_contiguous[src_offset];
    v_paged[dst_offset] = v_contiguous[src_offset];
}

// ============================================================================
// INT8 variant: copy quantized INT8 data + scales from contiguous (HND)
// to paged (NHD) layout with scale transposition.
//
// Contiguous INT8 data: [num_kv_heads, max_seq_len, head_dim] (HND)
// Contiguous scales:    [num_kv_heads, max_seq_len]
// Paged INT8 data:      pool_idx * kv_dim + kv_head * head_dim + d (NHD)
// Paged scales:         pool_idx * num_kv_heads + kv_head
//
// Grid: (num_kv_heads, seq_len)  Threads: head_dim
// ============================================================================
__global__ void kv_cache_to_paged_int8_kernel(
    const int8_t* __restrict__ k_cont,           // [num_kv_heads, max_seq_len, head_dim]
    const int8_t* __restrict__ v_cont,
    const float* __restrict__ k_scales_cont,     // [num_kv_heads, max_seq_len]
    const float* __restrict__ v_scales_cont,
    int8_t* __restrict__ k_paged,                // paged pool [max_tokens, kv_dim]
    int8_t* __restrict__ v_paged,
    float* __restrict__ k_scales_paged,          // [max_tokens, num_kv_heads]
    float* __restrict__ v_scales_paged,
    const int32_t* __restrict__ page_indices,    // [seq_len] token pool indices
    int max_seq_len,
    int seq_len,
    int num_kv_heads,
    int head_dim,
    int kv_dim)                                  // num_kv_heads * head_dim
{
    int kv_head = blockIdx.x;
    int pos = blockIdx.y;
    int dim = threadIdx.x;

    if (pos >= seq_len || dim >= head_dim) return;

    int pool_idx = page_indices[pos];

    // Source: HND contiguous
    int src_data = kv_head * max_seq_len * head_dim + pos * head_dim + dim;
    // Destination: NHD paged
    int dst_data = pool_idx * kv_dim + kv_head * head_dim + dim;

    k_paged[dst_data] = k_cont[src_data];
    v_paged[dst_data] = v_cont[src_data];

    // Copy scales (one thread per (kv_head, pos) pair)
    if (dim == 0) {
        int src_scale = kv_head * max_seq_len + pos;
        int dst_scale = pool_idx * num_kv_heads + kv_head;
        k_scales_paged[dst_scale] = k_scales_cont[src_scale];
        v_scales_paged[dst_scale] = v_scales_cont[src_scale];
    }
}

__global__ void kv_cache_to_paged_int8_range_kernel(
    const int8_t* __restrict__ k_cont,           // [num_kv_heads, max_seq_len, head_dim]
    const int8_t* __restrict__ v_cont,
    const float* __restrict__ k_scales_cont,     // [num_kv_heads, max_seq_len]
    const float* __restrict__ v_scales_cont,
    int8_t* __restrict__ k_paged,                // [max_tokens, kv_dim]
    int8_t* __restrict__ v_paged,
    float* __restrict__ k_scales_paged,          // [max_tokens, num_kv_heads]
    float* __restrict__ v_scales_paged,
    const int32_t* __restrict__ token_indices,   // [token_count] token pool indices
    int start_pos,
    int max_seq_len,
    int token_count,
    int num_kv_heads,
    int head_dim,
    int kv_dim)
{
    int kv_head = blockIdx.x;
    int rel_pos = blockIdx.y;
    int dim = threadIdx.x;

    if (rel_pos >= token_count || dim >= head_dim) return;

    int pos = start_pos + rel_pos;
    int pool_idx = token_indices[rel_pos];

    int src_data = kv_head * max_seq_len * head_dim + pos * head_dim + dim;
    int dst_data = pool_idx * kv_dim + kv_head * head_dim + dim;

    k_paged[dst_data] = k_cont[src_data];
    v_paged[dst_data] = v_cont[src_data];

    if (dim == 0) {
        int src_scale = kv_head * max_seq_len + pos;
        int dst_scale = pool_idx * num_kv_heads + kv_head;
        k_scales_paged[dst_scale] = k_scales_cont[src_scale];
        v_scales_paged[dst_scale] = v_scales_cont[src_scale];
    }
}

extern "C" cudaError_t kv_cache_to_paged_int8_cuda(
    const int8_t* k_cont, const int8_t* v_cont,
    const float* k_scales_cont, const float* v_scales_cont,
    int8_t* k_paged, int8_t* v_paged,
    float* k_scales_paged, float* v_scales_paged,
    const int32_t* page_indices,
    int max_seq_len, int seq_len, int num_kv_heads,
    int head_dim, int kv_dim,
    cudaStream_t stream)
{
    if (seq_len <= 0) return cudaSuccess;
    dim3 grid(num_kv_heads, seq_len);
    dim3 block(head_dim);
    kv_cache_to_paged_int8_kernel<<<grid, block, 0, stream>>>(
        k_cont, v_cont, k_scales_cont, v_scales_cont,
        k_paged, v_paged, k_scales_paged, v_scales_paged,
        page_indices, max_seq_len, seq_len, num_kv_heads,
        head_dim, kv_dim);
    return cudaGetLastError();
}

extern "C" cudaError_t kv_cache_to_paged_int8_range_cuda(
    const int8_t* k_cont, const int8_t* v_cont,
    const float* k_scales_cont, const float* v_scales_cont,
    int8_t* k_paged, int8_t* v_paged,
    float* k_scales_paged, float* v_scales_paged,
    const int32_t* token_indices,
    int start_pos, int max_seq_len, int token_count, int num_kv_heads,
    int head_dim, int kv_dim,
    cudaStream_t stream)
{
    if (token_count <= 0) return cudaSuccess;
    dim3 grid(num_kv_heads, token_count);
    dim3 block(head_dim);
    kv_cache_to_paged_int8_range_kernel<<<grid, block, 0, stream>>>(
        k_cont, v_cont, k_scales_cont, v_scales_cont,
        k_paged, v_paged, k_scales_paged, v_scales_paged,
        token_indices, start_pos, max_seq_len, token_count, num_kv_heads,
        head_dim, kv_dim);
    return cudaGetLastError();
}

extern "C" cudaError_t kv_cache_to_paged_cuda(
    const __nv_bfloat16* k_contiguous,
    const __nv_bfloat16* v_contiguous,
    __nv_bfloat16* k_paged,
    __nv_bfloat16* v_paged,
    const int32_t* page_indices,
    int max_seq_len,
    int seq_len,
    int num_kv_heads,
    int page_size,
    int head_dim,
    int stride_page,
    cudaStream_t stream)
{
    dim3 grid(num_kv_heads, seq_len);
    dim3 block(head_dim);
    kv_cache_to_paged_kernel<<<grid, block, 0, stream>>>(
        k_contiguous, v_contiguous, k_paged, v_paged, page_indices,
        max_seq_len, seq_len, num_kv_heads, page_size, head_dim, stride_page);
    return cudaGetLastError();
}

extern "C" cudaError_t kv_cache_to_paged_range_cuda(
    const __nv_bfloat16* k_contiguous,
    const __nv_bfloat16* v_contiguous,
    __nv_bfloat16* k_paged,
    __nv_bfloat16* v_paged,
    const int32_t* token_indices,
    int start_pos,
    int max_seq_len,
    int token_count,
    int num_kv_heads,
    int head_dim,
    int kv_dim,
    cudaStream_t stream)
{
    if (token_count <= 0) return cudaSuccess;
    dim3 grid(num_kv_heads, token_count);
    dim3 block(head_dim);
    kv_cache_to_paged_range_kernel<<<grid, block, 0, stream>>>(
        k_contiguous, v_contiguous, k_paged, v_paged, token_indices, start_pos,
        max_seq_len, token_count, num_kv_heads, head_dim, kv_dim);
    return cudaGetLastError();
}

extern "C" cudaError_t kv_cache_to_paged_range_hnd_cuda(
    const __nv_bfloat16* k_contiguous,
    const __nv_bfloat16* v_contiguous,
    __nv_bfloat16* k_paged,
    __nv_bfloat16* v_paged,
    const int32_t* page_indices,
    int start_pos,
    int max_seq_len,
    int token_count,
    int num_kv_heads,
    int page_size,
    int head_dim,
    int stride_page,
    cudaStream_t stream)
{
    if (token_count <= 0) return cudaSuccess;
    dim3 grid(num_kv_heads, token_count);
    dim3 block(head_dim);
    kv_cache_to_paged_range_hnd_kernel<<<grid, block, 0, stream>>>(
        k_contiguous, v_contiguous, k_paged, v_paged, page_indices, start_pos,
        max_seq_len, token_count, num_kv_heads, page_size, head_dim, stride_page);
    return cudaGetLastError();
}
