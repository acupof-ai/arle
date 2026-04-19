// Batched decode prep for FlashInfer paged attention:
//   1. Per-head RMSNorm on Q and K
//   2. Half-split RoPE on Q and K (Qwen3/Llama style)
//   3. Write K (normed+roped) and V (raw) to paged KV cache
//
// Q input:  [B, num_qo_heads * head_dim] row-major (column-major HiddenStates)
// K input:  [B, num_kv_heads * head_dim]
// V input:  [B, num_kv_heads * head_dim]
// Q output: [B, num_qo_heads, head_dim] — same buffer, in-place
// K/V written to paged cache at correct page + offset
//
// Grid: (num_kv_heads, B)   Threads: head_dim (128)
// Each block processes one KV head for one batch element.
// Q heads are processed in a loop (gqa_ratio iterations).

#include "common.cuh"

#define HEAD_DIM 128
#define NUM_WARPS (HEAD_DIM / WARP_SIZE)  // 4

// Per-head RMSNorm: compute 1/sqrt(mean(x^2) + eps), apply weight, return normalized value
__device__ __forceinline__ float rms_norm_head(
    float val,
    float weight,
    float eps,
    int tid
) {
    float sq = val * val;
    float sq_sum = warp_reduce_sum(sq);

    __shared__ float scratch[NUM_WARPS];
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    if (lane_id == 0) scratch[warp_id] = sq_sum;
    __syncthreads();

    if (tid == 0) {
        float total = 0.0f;
        for (int i = 0; i < NUM_WARPS; i++) total += scratch[i];
        scratch[0] = 1.0f / sqrtf(total / HEAD_DIM + eps);
    }
    __syncthreads();

    return val * scratch[0] * weight;
}

// Half-split RoPE: first half and second half paired
// Requires shared memory buffer of HEAD_DIM floats, pre-filled with normed values
__device__ __forceinline__ float apply_rope_half_split(
    float* smem,
    const __nv_bfloat16* __restrict__ cos_cache,
    const __nv_bfloat16* __restrict__ sin_cache,
    int pos,
    int tid
) {
    int half_dim = HEAD_DIM / 2;
    float result;
    if (tid < half_dim) {
        float cos_val = __bfloat162float(__ldg(cos_cache + pos * HEAD_DIM + tid));
        float sin_val = __bfloat162float(__ldg(sin_cache + pos * HEAD_DIM + tid));
        result = smem[tid] * cos_val - smem[tid + half_dim] * sin_val;
    } else {
        int pair = tid - half_dim;
        float cos_val = __bfloat162float(__ldg(cos_cache + pos * HEAD_DIM + pair));
        float sin_val = __bfloat162float(__ldg(sin_cache + pos * HEAD_DIM + pair));
        result = smem[pair] * sin_val + smem[tid] * cos_val;
    }
    return result;
}

// Main kernel: one block per (kv_head, batch_element)
__global__ void decode_prep_paged_kernel(
    __nv_bfloat16* __restrict__ q_batch,         // [B, num_qo_heads * head_dim], in-place
    const __nv_bfloat16* __restrict__ k_batch,   // [B, num_kv_heads * head_dim]
    const __nv_bfloat16* __restrict__ v_batch,   // [B, num_kv_heads * head_dim]
    const __nv_bfloat16* __restrict__ q_norm_weight, // [head_dim]
    const __nv_bfloat16* __restrict__ k_norm_weight, // [head_dim]
    const __nv_bfloat16* __restrict__ cos_cache, // [max_pos * head_dim]
    const __nv_bfloat16* __restrict__ sin_cache, // [max_pos * head_dim]
    const int* __restrict__ positions,           // [B] current position per request
    __nv_bfloat16* __restrict__ k_pool,          // paged K pool [max_pages * num_kv_heads * page_size * head_dim]
    __nv_bfloat16* __restrict__ v_pool,          // paged V pool [same]
    const int* __restrict__ page_table,          // flattened page indices (via indptr)
    const int* __restrict__ page_indptr,         // [B+1] cumulative page counts
    const int* __restrict__ last_page_len,       // [B] tokens in last page
    int num_qo_heads,
    int num_kv_heads,
    int page_size,
    int stride_page,                             // num_kv_heads * page_size * head_dim
    float rms_eps
) {
    int kv_head_idx = blockIdx.x;
    int batch_idx = blockIdx.y;
    int tid = threadIdx.x;
    int gqa_ratio = num_qo_heads / num_kv_heads;

    int pos = positions[batch_idx];

    // ---- Process Q heads (gqa_ratio per KV head) ----
    __shared__ float smem_rope[HEAD_DIM];

    float q_norm_w = __bfloat162float(__ldg(q_norm_weight + tid));
    float k_norm_w = __bfloat162float(__ldg(k_norm_weight + tid));

    #pragma unroll
    for (int g = 0; g < gqa_ratio; g++) {
        int q_head = kv_head_idx * gqa_ratio + g;
        int q_offset = batch_idx * num_qo_heads * HEAD_DIM + q_head * HEAD_DIM + tid;

        float q_val = __bfloat162float(__ldg(q_batch + q_offset));
        float q_normed = rms_norm_head(q_val, q_norm_w, rms_eps, tid);

        smem_rope[tid] = q_normed;
        __syncthreads();

        float q_roped = apply_rope_half_split(smem_rope, cos_cache, sin_cache, pos, tid);
        __syncthreads();

        q_batch[q_offset] = __float2bfloat16(q_roped);
    }

    // ---- Process K: norm + rope ----
    int kv_offset = batch_idx * num_kv_heads * HEAD_DIM + kv_head_idx * HEAD_DIM + tid;
    float k_val = __bfloat162float(__ldg(k_batch + kv_offset));
    float k_normed = rms_norm_head(k_val, k_norm_w, rms_eps, tid);

    smem_rope[tid] = k_normed;
    __syncthreads();

    float k_roped = apply_rope_half_split(smem_rope, cos_cache, sin_cache, pos, tid);

    // ---- Load V (raw, no norm/rope) ----
    float v_val = __bfloat162float(__ldg(v_batch + kv_offset));

    // ---- Write K and V to paged cache ----
    // Find the correct page and offset within it
    // page_table layout: page_table[page_indptr[b]..page_indptr[b+1]] are page IDs for batch b
    int page_start = page_indptr[batch_idx];
    int num_pages = page_indptr[batch_idx + 1] - page_start;
    int last_len = last_page_len[batch_idx]; // tokens already in last page BEFORE this write

    // The current token goes at position last_len within the last page
    // (last_page_len accounts for the token we're about to write, since grow_slot was called)
    // Actually: last_page_len = how many tokens are in the last page INCLUDING this new one
    // So the write offset within the last page is (last_len - 1)
    int last_page_idx = num_pages - 1;
    int page_id = page_table[page_start + last_page_idx];
    int token_in_page = last_len - 1;

    // Paged cache layout (HND): [max_pages, num_kv_heads, page_size, head_dim]
    // Offset = page_id * stride_page + kv_head_idx * page_size * HEAD_DIM + token_in_page * HEAD_DIM + tid
    int cache_offset = page_id * stride_page
                     + kv_head_idx * page_size * HEAD_DIM
                     + token_in_page * HEAD_DIM
                     + tid;

    k_pool[cache_offset] = __float2bfloat16(k_roped);
    v_pool[cache_offset] = __float2bfloat16(v_val);
}

extern "C" {

cudaError_t decode_prep_paged_cuda(
    __nv_bfloat16* q_batch,
    const __nv_bfloat16* __restrict__ k_batch,
    const __nv_bfloat16* __restrict__ v_batch,
    const __nv_bfloat16* __restrict__ q_norm_weight,
    const __nv_bfloat16* __restrict__ k_norm_weight,
    const __nv_bfloat16* __restrict__ cos_cache,
    const __nv_bfloat16* __restrict__ sin_cache,
    const int* __restrict__ positions,
    __nv_bfloat16* k_pool,
    __nv_bfloat16* v_pool,
    const int* __restrict__ page_table,
    const int* __restrict__ page_indptr,
    const int* __restrict__ last_page_len,
    int num_qo_heads,
    int num_kv_heads,
    int page_size,
    int stride_page,
    int batch_size,
    float rms_eps,
    cudaStream_t stream
) {
    dim3 grid(num_kv_heads, batch_size);
    int threads = HEAD_DIM;

    decode_prep_paged_kernel<<<grid, threads, 0, stream>>>(
        q_batch, k_batch, v_batch,
        q_norm_weight, k_norm_weight,
        cos_cache, sin_cache,
        positions,
        k_pool, v_pool,
        page_table, page_indptr, last_page_len,
        num_qo_heads, num_kv_heads,
        page_size, stride_page,
        rms_eps
    );
    return cudaGetLastError();
}

// Fused QKV variant: reads Q/K/V from a single merged buffer [B, qkv_stride],
// writes RoPE'd Q to a separate output buffer. Eliminates the split_qkv kernel.
__global__ void decode_prep_paged_fused_qkv_kernel(
    const __nv_bfloat16* __restrict__ qkv_batch,    // [B, qkv_stride] merged QKV
    __nv_bfloat16* __restrict__ q_out,               // [B, num_qo_heads * head_dim] write
    const __nv_bfloat16* __restrict__ q_norm_weight,
    const __nv_bfloat16* __restrict__ k_norm_weight,
    const __nv_bfloat16* __restrict__ cos_cache,
    const __nv_bfloat16* __restrict__ sin_cache,
    const int* __restrict__ positions,
    __nv_bfloat16* __restrict__ k_pool,
    __nv_bfloat16* __restrict__ v_pool,
    const int* __restrict__ page_table,
    const int* __restrict__ page_indptr,
    const int* __restrict__ last_page_len,
    int num_qo_heads,
    int num_kv_heads,
    int page_size,
    int stride_page,
    float rms_eps,
    int qkv_stride,
    int q_dim
) {
    int kv_head_idx = blockIdx.x;
    int batch_idx = blockIdx.y;
    int tid = threadIdx.x;
    int gqa_ratio = num_qo_heads / num_kv_heads;
    int kv_dim = num_kv_heads * HEAD_DIM;

    int pos = positions[batch_idx];
    int qkv_row = batch_idx * qkv_stride;

    __shared__ float smem_rope[HEAD_DIM];

    float q_norm_w = __bfloat162float(__ldg(q_norm_weight + tid));
    float k_norm_w = __bfloat162float(__ldg(k_norm_weight + tid));

    #pragma unroll
    for (int g = 0; g < gqa_ratio; g++) {
        int q_head = kv_head_idx * gqa_ratio + g;
        float q_val = __bfloat162float(__ldg(qkv_batch + qkv_row + q_head * HEAD_DIM + tid));
        float q_normed = rms_norm_head(q_val, q_norm_w, rms_eps, tid);

        smem_rope[tid] = q_normed;
        __syncthreads();

        float q_roped = apply_rope_half_split(smem_rope, cos_cache, sin_cache, pos, tid);
        __syncthreads();

        int q_out_offset = batch_idx * num_qo_heads * HEAD_DIM + q_head * HEAD_DIM + tid;
        q_out[q_out_offset] = __float2bfloat16(q_roped);
    }

    float k_val = __bfloat162float(__ldg(qkv_batch + qkv_row + q_dim + kv_head_idx * HEAD_DIM + tid));
    float k_normed = rms_norm_head(k_val, k_norm_w, rms_eps, tid);

    smem_rope[tid] = k_normed;
    __syncthreads();

    float k_roped = apply_rope_half_split(smem_rope, cos_cache, sin_cache, pos, tid);

    float v_val = __bfloat162float(__ldg(qkv_batch + qkv_row + q_dim + kv_dim + kv_head_idx * HEAD_DIM + tid));

    int page_start = page_indptr[batch_idx];
    int num_pages = page_indptr[batch_idx + 1] - page_start;
    int last_len = last_page_len[batch_idx];
    int last_page_idx = num_pages - 1;
    int page_id = page_table[page_start + last_page_idx];
    int token_in_page = last_len - 1;

    int cache_offset = page_id * stride_page
                     + kv_head_idx * page_size * HEAD_DIM
                     + token_in_page * HEAD_DIM
                     + tid;

    k_pool[cache_offset] = __float2bfloat16(k_roped);
    v_pool[cache_offset] = __float2bfloat16(v_val);
}

cudaError_t decode_prep_paged_fused_qkv_cuda(
    const __nv_bfloat16* qkv_batch,
    __nv_bfloat16* q_out,
    const __nv_bfloat16* __restrict__ q_norm_weight,
    const __nv_bfloat16* __restrict__ k_norm_weight,
    const __nv_bfloat16* __restrict__ cos_cache,
    const __nv_bfloat16* __restrict__ sin_cache,
    const int* __restrict__ positions,
    __nv_bfloat16* k_pool,
    __nv_bfloat16* v_pool,
    const int* __restrict__ page_table,
    const int* __restrict__ page_indptr,
    const int* __restrict__ last_page_len,
    int num_qo_heads,
    int num_kv_heads,
    int page_size,
    int stride_page,
    int batch_size,
    float rms_eps,
    int qkv_stride,
    int q_dim,
    cudaStream_t stream
) {
    dim3 grid(num_kv_heads, batch_size);
    int threads = HEAD_DIM;

    decode_prep_paged_fused_qkv_kernel<<<grid, threads, 0, stream>>>(
        qkv_batch, q_out,
        q_norm_weight, k_norm_weight,
        cos_cache, sin_cache,
        positions,
        k_pool, v_pool,
        page_table, page_indptr, last_page_len,
        num_qo_heads, num_kv_heads,
        page_size, stride_page,
        rms_eps,
        qkv_stride, q_dim
    );
    return cudaGetLastError();
}

} // extern "C"
