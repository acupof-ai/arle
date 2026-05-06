// Scatter-write prefill K/V from contiguous GEMM output to token-level KV pool.
//
// After QKV projection produces contiguous K [seq_len, kv_dim] and V [seq_len, kv_dim],
// this kernel copies each token's K and V to the pool at a specified token index.
// No norm or RoPE is applied; downstream attention handles those internally.
//
// Input layout  (row-major): K/V_batch[pos * kv_dim + kv_head * head_dim + dim]
// Output layout (row-major): K/V_pool [token_idx * kv_dim + kv_head * head_dim + dim]
//
// Grid: (num_kv_heads, seq_len)   Threads: head_dim
// Each block copies one head for one token.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

__global__ void scatter_write_kv_kernel(
    const __nv_bfloat16* __restrict__ k_batch,      // [seq_len, kv_dim]
    const __nv_bfloat16* __restrict__ v_batch,      // [seq_len, kv_dim]
    __nv_bfloat16* __restrict__ k_pool,             // [max_tokens, kv_dim]
    __nv_bfloat16* __restrict__ v_pool,             // [max_tokens, kv_dim]
    const int32_t* __restrict__ token_indices,      // [seq_len] — pool index per token
    int seq_len,
    int num_kv_heads,
    int head_dim)
{
    int kv_head = blockIdx.x;
    int pos     = blockIdx.y;
    int dim     = threadIdx.x;

    if (pos >= seq_len || dim >= head_dim) return;

    int kv_dim = num_kv_heads * head_dim;

    // Source offset: row-major [seq_len, kv_dim]
    int src_offset = pos * kv_dim + kv_head * head_dim + dim;

    // Destination offset: row-major [max_tokens, kv_dim]
    int token_idx  = token_indices[pos];
    int dst_offset = token_idx * kv_dim + kv_head * head_dim + dim;

    k_pool[dst_offset] = k_batch[src_offset];
    v_pool[dst_offset] = v_batch[src_offset];
}

extern "C" cudaError_t scatter_write_kv_cuda(
    const __nv_bfloat16* k_batch,
    const __nv_bfloat16* v_batch,
    __nv_bfloat16* k_pool,
    __nv_bfloat16* v_pool,
    const int32_t* token_indices,
    int seq_len,
    int num_kv_heads,
    int head_dim,
    cudaStream_t stream)
{
    dim3 grid(num_kv_heads, seq_len);
    dim3 block(head_dim);
    scatter_write_kv_kernel<<<grid, block, 0, stream>>>(
        k_batch, v_batch, k_pool, v_pool, token_indices,
        seq_len, num_kv_heads, head_dim);
    return cudaGetLastError();
}
