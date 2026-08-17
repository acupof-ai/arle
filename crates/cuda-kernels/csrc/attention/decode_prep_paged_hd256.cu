// Batched decode prep for TileLang paged attention - HD256 variant (Qwen3.5).
//
// Differences from decode_prep_paged.cu (HD128):
//   1. HEAD_DIM = 256 (256 threads per block)
//   2. RMSNorm applies weight directly (standard convention, matches hd128)
//   3. Partial RoPE: only first `rotary_dim` dims (typically 64) get rotation
//   4. Q has gate: q_full layout is [B, num_q_heads * head_dim * 2]
//      where each head has [head_dim Q values, head_dim gate values]
//      The prep kernel normalizes + ropes only the Q portion;
//      gate is applied AFTER attention via a separate kernel.
//
// Grid: (num_kv_heads, B)   Threads: 256 (= HEAD_DIM)

#include "common.cuh"

#define NUM_WARPS_HD256 (HD256 / WARP_SIZE)  // 8

// Per-head RMSNorm, OFFSET convention: output = x * rms_inv * (1 + weight).
// #58: hd256 q/k_norm ships OFFSET (mean|w| ~0.49 < 0.75); dropping the +1
// (b4b293f0c) shrank q/k ~3x, collapsing attention scores at length.
__device__ __forceinline__ float rms_norm_head_hd256(
    float val,
    float weight,
    float eps,
    int tid
) {
    float sq = val * val;
    float sq_sum = warp_reduce_sum(sq);

    __shared__ float scratch[NUM_WARPS_HD256];
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    if (lane_id == 0) scratch[warp_id] = sq_sum;
    __syncthreads();

    // Cross-warp reduce in warp 0 via shuffle (NUM_WARPS_HD256=8<=32), not a
    // tid==0 serial loop — matches prefill_attention_paged_prep's idiom.
    if (warp_id == 0) {
        float warp_sum = (lane_id < NUM_WARPS_HD256) ? scratch[lane_id] : 0.0f;
        float total = warp_reduce_sum(warp_sum);
        if (lane_id == 0) scratch[0] = 1.0f / sqrtf(total / HD256 + eps);
    }
    __syncthreads();

    return val * scratch[0] * (1.0f + weight);
}

// Main kernel: one block per (kv_head, batch_element)
__global__ void decode_prep_paged_hd256_kernel(
    const __nv_bfloat16* __restrict__ q_full_batch,  // [B, num_q_heads * HD256 * 2] (Q + gate)
    __nv_bfloat16* __restrict__ q_out_batch,         // [B, num_q_heads * HD256] (Q only, normed + roped)
    const __nv_bfloat16* __restrict__ k_batch,       // [B, num_kv_heads * HD256]
    const __nv_bfloat16* __restrict__ v_batch,       // [B, num_kv_heads * HD256]
    const __nv_bfloat16* __restrict__ q_norm_weight,  // [HD256]
    const __nv_bfloat16* __restrict__ k_norm_weight,  // [HD256]
    const __nv_bfloat16* __restrict__ cos_cache,      // [max_pos * rotary_dim]
    const __nv_bfloat16* __restrict__ sin_cache,
    const int* __restrict__ positions,                // [B] current position per request
    __nv_bfloat16* __restrict__ k_pool,               // paged K pool
    __nv_bfloat16* __restrict__ v_pool,               // paged V pool
    const int* __restrict__ page_table,
    const int* __restrict__ page_indptr,
    const int* __restrict__ last_page_len,
    int num_qo_heads,
    int num_kv_heads,
    int page_size,
    int stride_page,                                  // num_kv_heads * page_size * HD256
    int rotary_dim,
    float rms_eps,
    int write_kv                                      // 0 = skip the K/V pool write (2D non-owner shard)
) {
    int kv_head_idx = blockIdx.x;
    int batch_idx = blockIdx.y;
    int tid = threadIdx.x;  // 0..255
    int gqa_ratio = num_qo_heads / num_kv_heads;

    int pos = positions[batch_idx];

    // ---- Process Q heads (gqa_ratio per KV head) ----
    __shared__ float smem_rope[HD256];

    float q_norm_w = __bfloat162float(q_norm_weight[tid]);
    float k_norm_w = __bfloat162float(k_norm_weight[tid]);

    int q_full_dim = num_qo_heads * HD256 * 2;  // total Q+gate dim per token
    int q_dim = num_qo_heads * HD256;

    for (int g = 0; g < gqa_ratio; g++) {
        int q_head = kv_head_idx * gqa_ratio + g;
        // Q values are at head offset [head * 2 * HD256 + 0..HD256)
        int q_src = batch_idx * q_full_dim + q_head * 2 * HD256 + tid;

        float q_val = __bfloat162float(q_full_batch[q_src]);
        float q_normed = rms_norm_head_hd256(q_val, q_norm_w, rms_eps, tid);

        smem_rope[tid] = q_normed;
        __syncthreads();

        float q_roped = apply_rope_partial_hd256(smem_rope, cos_cache, sin_cache, pos, tid, rotary_dim);
        __syncthreads();

        // Write to Q output buffer (without gate)
        int q_dst = batch_idx * q_dim + q_head * HD256 + tid;
        q_out_batch[q_dst] = __float2bfloat16(q_roped);
    }

    // ---- Process K: norm + partial rope ----
    int kv_dim = num_kv_heads * HD256;
    int kv_offset = batch_idx * kv_dim + kv_head_idx * HD256 + tid;
    float k_val = __bfloat162float(k_batch[kv_offset]);
    float k_normed = rms_norm_head_hd256(k_val, k_norm_w, rms_eps, tid);

    smem_rope[tid] = k_normed;
    __syncthreads();

    float k_roped = apply_rope_partial_hd256(smem_rope, cos_cache, sin_cache, pos, tid, rotary_dim);

    // ---- Load V (raw, no norm/rope) ----
    float v_val = __bfloat162float(v_batch[kv_offset]);

    // ---- Write K and V to paged cache ----
    // Under 2D sequence-sharding only the shard owning the new token's page
    // (logical page pos/page_size, i % cp == c) writes; the other shards skip.
    if (write_kv) {
        int page_start = page_indptr[batch_idx];
        int num_pages = page_indptr[batch_idx + 1] - page_start;
        int last_len = last_page_len[batch_idx];

        int last_page_idx = num_pages - 1;
        int page_id = page_table[page_start + last_page_idx];
        int token_in_page = last_len - 1;

        // HND layout: [max_pages, num_kv_heads, page_size, HD256]
        int cache_offset = page_id * stride_page
                         + kv_head_idx * page_size * HD256
                         + token_in_page * HD256
                         + tid;

        k_pool[cache_offset] = __float2bfloat16(k_roped);
        v_pool[cache_offset] = __float2bfloat16(v_val);
    }
}

// Gate kernel: apply sigmoid gate from q_full to attention output.
// After TileLang attention writes to attn_out, this kernel reads the gate
// portion of q_full and multiplies: attn_out[i] *= sigmoid(gate[i])
__global__ void attention_gate_paged_hd256_kernel(
    const __nv_bfloat16* __restrict__ q_full_batch,  // [B, num_q_heads * HD256 * 2]
    __nv_bfloat16* __restrict__ attn_out,            // [B, num_q_heads * HD256]
    int num_q_heads,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int q_dim = num_q_heads * HD256;
    int total = q_dim * batch_size;
    if (idx >= total) return;

    int b = idx / q_dim;
    int q_offset = idx - b * q_dim;
    int q_head = q_offset / HD256;
    int dim = q_offset % HD256;
    int q_full_dim = q_dim * 2;
    // Gate values are at [head * 2 * HD256 + HD256 + dim]
    int gate_idx = b * q_full_dim + q_head * 2 * HD256 + HD256 + dim;

    float gate = __bfloat162float(q_full_batch[gate_idx]);
    float sig_gate = 1.0f / (1.0f + __expf(-gate));
    float out = __bfloat162float(attn_out[idx]);
    attn_out[idx] = __float2bfloat16(out * sig_gate);
}

extern "C" {

cudaError_t decode_prep_paged_hd256_cuda(
    const __nv_bfloat16* q_full_batch,
    __nv_bfloat16* q_out_batch,
    const __nv_bfloat16* k_batch,
    const __nv_bfloat16* v_batch,
    const __nv_bfloat16* q_norm_weight,
    const __nv_bfloat16* k_norm_weight,
    const __nv_bfloat16* cos_cache,
    const __nv_bfloat16* sin_cache,
    const int* positions,
    __nv_bfloat16* k_pool,
    __nv_bfloat16* v_pool,
    const int* page_table,
    const int* page_indptr,
    const int* last_page_len,
    int num_qo_heads,
    int num_kv_heads,
    int page_size,
    int stride_page,
    int batch_size,
    int rotary_dim,
    float rms_eps,
    int write_kv,
    cudaStream_t stream
) {
    if (q_full_batch == nullptr || q_out_batch == nullptr || k_batch == nullptr ||
        v_batch == nullptr || q_norm_weight == nullptr || k_norm_weight == nullptr ||
        cos_cache == nullptr || sin_cache == nullptr || positions == nullptr ||
        k_pool == nullptr || v_pool == nullptr || page_table == nullptr ||
        page_indptr == nullptr || last_page_len == nullptr || num_qo_heads <= 0 ||
        num_kv_heads <= 0 || num_qo_heads % num_kv_heads != 0 || page_size <= 0 ||
        stride_page <= 0 ||
        static_cast<long long>(stride_page) <
            static_cast<long long>(num_kv_heads) * page_size * HD256 ||
        batch_size <= 0 || rotary_dim <= 0 || rotary_dim > HD256 ||
        rotary_dim % 2 != 0) {
        return cudaErrorInvalidValue;
    }
    dim3 grid(num_kv_heads, batch_size);
    int threads = HD256;

    decode_prep_paged_hd256_kernel<<<grid, threads, 0, stream>>>(
        q_full_batch, q_out_batch, k_batch, v_batch,
        q_norm_weight, k_norm_weight,
        cos_cache, sin_cache,
        positions,
        k_pool, v_pool,
        page_table, page_indptr, last_page_len,
        num_qo_heads, num_kv_heads,
        page_size, stride_page,
        rotary_dim, rms_eps, write_kv
    );
    return cudaGetLastError();
}

cudaError_t attention_gate_paged_hd256_cuda(
    const __nv_bfloat16* q_full_batch,
    __nv_bfloat16* attn_out,
    int num_q_heads,
    int batch_size,
    cudaStream_t stream
) {
    if (q_full_batch == nullptr || attn_out == nullptr || num_q_heads <= 0 || batch_size <= 0) {
        return cudaErrorInvalidValue;
    }
    int q_dim = num_q_heads * HD256;
    int total = q_dim * batch_size;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    attention_gate_paged_hd256_kernel<<<blocks, threads, 0, stream>>>(
        q_full_batch, attn_out, num_q_heads, batch_size
    );
    return cudaGetLastError();
}

} // extern "C"
