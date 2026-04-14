// Append K/V tokens to a paged KV cache (FlashInfer HND layout).
//
// For each request in a batch, writes one new token's K and V values
// to the correct position in the paged cache.
//
// Paged cache layout (HND): [max_pages, num_kv_heads, page_size, head_dim]
// stride_page = num_kv_heads * page_size * head_dim
// stride_h = page_size * head_dim
// stride_n = head_dim
//
// For a token at position `pos` in request `b`:
//   logical_page = pos / page_size
//   offset_in_page = pos % page_size
//   physical_page = page_indices[indptr[b] + logical_page]
//   element_offset = physical_page * stride_page + kv_head * stride_h + offset_in_page * stride_n + dim

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

// Append one token of K and V per request into paged cache.
// Grid: (num_kv_heads, batch_size)
// Block: (head_dim)
__global__ void paged_kv_append_kernel(
    const __nv_bfloat16* __restrict__ k_batch,    // [B, kv_dim] — new K values (after norm+RoPE)
    const __nv_bfloat16* __restrict__ v_batch,    // [B, kv_dim] — new V values
    __nv_bfloat16* __restrict__ k_data,           // All K pages
    __nv_bfloat16* __restrict__ v_data,           // All V pages
    const int32_t* __restrict__ page_indices,     // Flattened page table
    const int32_t* __restrict__ indptr,           // [B+1] page boundaries
    const int32_t* __restrict__ positions,        // [B] token position for each request
    int num_kv_heads,
    int page_size,
    int head_dim)
{
    const int kv_head = blockIdx.x;
    const int b = blockIdx.y;
    const int dim = threadIdx.x;

    if (dim >= head_dim) return;

    const int pos = positions[b];
    const int logical_page = pos / page_size;
    const int offset_in_page = pos % page_size;

    // Look up physical page
    const int physical_page = page_indices[indptr[b] + logical_page];

    // Compute offsets
    const int stride_page = num_kv_heads * page_size * head_dim;
    const int stride_h = page_size * head_dim;
    const int stride_n = head_dim;

    const int cache_offset = physical_page * stride_page
                           + kv_head * stride_h
                           + offset_in_page * stride_n
                           + dim;

    // Source offset in batch buffer: [B, num_kv_heads * head_dim]
    const int kv_dim = num_kv_heads * head_dim;
    const int src_offset = b * kv_dim + kv_head * head_dim + dim;

    k_data[cache_offset] = k_batch[src_offset];
    v_data[cache_offset] = v_batch[src_offset];
}

// C entry point
extern "C" cudaError_t paged_kv_append_cuda(
    const __nv_bfloat16* k_batch,
    const __nv_bfloat16* v_batch,
    __nv_bfloat16* k_data,
    __nv_bfloat16* v_data,
    const int32_t* page_indices,
    const int32_t* indptr,
    const int32_t* positions,
    int batch_size,
    int num_kv_heads,
    int page_size,
    int head_dim,
    cudaStream_t stream)
{
    dim3 grid(num_kv_heads, batch_size);
    dim3 block(head_dim);
    paged_kv_append_kernel<<<grid, block, 0, stream>>>(
        k_batch, v_batch, k_data, v_data,
        page_indices, indptr, positions,
        num_kv_heads, page_size, head_dim);
    return cudaGetLastError();
}
