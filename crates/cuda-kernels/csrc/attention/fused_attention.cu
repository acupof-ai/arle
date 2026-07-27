#include "common.cuh"
#include <cstdio>

// Fused GQA Attention Kernel (bf16 version) — Tiled Online Softmax
//
// Processes KV cache in tiles of TILE_SIZE from global memory using the
// online softmax algorithm. No MAX_SEQ_LEN cap — supports full causal
// attention up to max_seq_len (4096).
//
// Architecture:
// - Each block processes 1 KV head + gqa_ratio Q heads (passed as param)
// - Tiles of K/V loaded from global cache into shared memory
// - Online softmax merges partial results across tiles
// - bf16 storage, fp32 accumulators

#define TILE_SIZE 64
#define HEAD_DIM 128
#define THREADS_PER_BLOCK 128
#define NUM_WARPS (THREADS_PER_BLOCK / WARP_SIZE)  // 4


__device__ __forceinline__ __nv_bfloat16 rms_norm_elem(
    __nv_bfloat16 x,
    float rms_inv,
    __nv_bfloat16 weight
) {
    // Match HF: round normalized value to bf16 before weight multiply
    __nv_bfloat16 normed = __float2bfloat16(__bfloat162float(x) * rms_inv);
    float val = __bfloat162float(normed) * __bfloat162float(weight);
    return __float2bfloat16(val);
}

__device__ __forceinline__ void apply_rope_pair(
    __nv_bfloat16& x0,
    __nv_bfloat16& x1,
    __nv_bfloat16 cos_val,
    __nv_bfloat16 sin_val
) {
    float fx0 = __bfloat162float(x0);
    float fx1 = __bfloat162float(x1);
    float fc = __bfloat162float(cos_val);
    float fs = __bfloat162float(sin_val);

    float temp = fx0;
    x0 = __float2bfloat16(fx0 * fc - fx1 * fs);
    x1 = __float2bfloat16(temp * fs + fx1 * fc);
}

// Tiled attention for a single Q head using online softmax.
//
// All shared memory buffers are allocated by the caller (kernel) and passed in.
// No __shared__ declarations inside this function.
__device__ void tiled_attention(
    const __nv_bfloat16* __restrict__ smem_q,
    const __nv_bfloat16* __restrict__ k_cache_base,
    const __nv_bfloat16* __restrict__ v_cache_base,
    __nv_bfloat16* __restrict__ smem_k,       // [TILE_SIZE * HEAD_DIM]
    __nv_bfloat16* __restrict__ smem_v,       // [TILE_SIZE * HEAD_DIM]
    float* __restrict__ smem_scores,           // [TILE_SIZE]
    float* __restrict__ warp_partial,          // [NUM_WARPS * (TILE_SIZE + 1)]
    float* __restrict__ smem_scratch,          // [NUM_WARPS] scratch for reductions
    float& smem_running_max,
    float& smem_running_sum,
    __nv_bfloat16* __restrict__ output_buf,
    int q_head_idx,
    int seq_len,
    int max_seq_len,
    int head_dim,
    float scale,
    int tid,
    int warp_id,
    int lane_id
) {
    float o_acc = 0.0f;  // output accumulator for dimension tid (register)

    if (tid == 0) {
        smem_running_max = -INFINITY;
        smem_running_sum = 0.0f;
    }
    __syncthreads();

    for (int tile_start = 0; tile_start < seq_len; tile_start += TILE_SIZE) {
        int tile_len = min(TILE_SIZE, seq_len - tile_start);

        for (int i = tid; i < tile_len * HEAD_DIM; i += THREADS_PER_BLOCK) {
            int pos_in_tile = i / HEAD_DIM;
            int dim = i % HEAD_DIM;
            int abs_pos = tile_start + pos_in_tile;
            smem_k[pos_in_tile * HEAD_DIM + dim] = k_cache_base[abs_pos * head_dim + dim];
            smem_v[pos_in_tile * HEAD_DIM + dim] = v_cache_base[abs_pos * head_dim + dim];
        }
        __syncthreads();

        // Thread-per-dimension dot product, warp reduce, cross-warp combine
        float q_val = __bfloat162float(smem_q[tid]);

        for (int pos = 0; pos < tile_len; pos++) {
            float partial = q_val * __bfloat162float(smem_k[pos * HEAD_DIM + tid]);
            partial = warp_reduce_sum(partial);
            if (lane_id == 0) {
                warp_partial[warp_id * (TILE_SIZE + 1) + pos] = partial;
            }
        }
        __syncthreads();

        // Combine warp partials into final scores
        if (tid < tile_len) {
            float score = 0.0f;
            for (int w = 0; w < NUM_WARPS; w++) {
                score += warp_partial[w * (TILE_SIZE + 1) + tid];
            }
            smem_scores[tid] = score * scale;
        }
        __syncthreads();

        float tile_max_local = -INFINITY;
        if (tid < tile_len) {
            tile_max_local = smem_scores[tid];
        }
        tile_max_local = warp_reduce_max(tile_max_local);

        if (lane_id == 0) {
            smem_scratch[warp_id] = tile_max_local;
        }
        __syncthreads();

        // Thread 0: compute tile_max, online softmax merge, broadcast scale_old
        if (tid == 0) {
            float tile_max = smem_scratch[0];
            for (int i = 1; i < NUM_WARPS; i++) {
                tile_max = fmaxf(tile_max, smem_scratch[i]);
            }
            float old_max = smem_running_max;
            float new_max = fmaxf(old_max, tile_max);
            float scale_old = expf(old_max - new_max);
            smem_running_sum *= scale_old;
            smem_running_max = new_max;
            // Broadcast scale_old via smem_scratch[0]
            smem_scratch[0] = scale_old;
        }
        __syncthreads();

        // ALL threads rescale their output accumulator
        float scale_old = smem_scratch[0];
        o_acc *= scale_old;

        float local_sum = 0.0f;
        float current_max = smem_running_max;
        for (int pos = 0; pos < tile_len; pos++) {
            float w = expf(smem_scores[pos] - current_max);
            local_sum += w;
            o_acc += w * __bfloat162float(smem_v[pos * HEAD_DIM + tid]);
        }

        // local_sum is identical across all threads (smem_scores is shared)
        if (tid == 0) {
            smem_running_sum += local_sum;
        }
        __syncthreads();
    }

    float final_sum = smem_running_sum;
    float result = (final_sum > 0.0f) ? (o_acc / final_sum) : 0.0f;
    output_buf[q_head_idx * head_dim + tid] = __float2bfloat16(result);
}

// Main kernel — supports arbitrary GQA ratio via gqa_ratio parameter
__global__ void fused_gqa_attention_single_token_kernel(
    const __nv_bfloat16* __restrict__ q_full,
    const __nv_bfloat16* __restrict__ k_full,
    const __nv_bfloat16* __restrict__ v_full,
    const __nv_bfloat16* __restrict__ q_norm_weight,
    const __nv_bfloat16* __restrict__ k_norm_weight,
    const __nv_bfloat16* __restrict__ cos_cache,
    const __nv_bfloat16* __restrict__ sin_cache,
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    __nv_bfloat16* __restrict__ output,
    int num_qheads,
    int num_kvheads,
    int gqa_ratio,
    int head_dim,
    int current_pos,
    int seq_len,
    int max_seq_len,
    float scale,
    float rms_eps
) {
    int kv_head_idx = blockIdx.x;

    int tid = threadIdx.x;  // 0..127
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // Shared memory layout — all buffers declared here, passed to device functions
    __shared__ __nv_bfloat16 smem_k[TILE_SIZE * HEAD_DIM];       // 16,384 B
    __shared__ __nv_bfloat16 smem_v[TILE_SIZE * HEAD_DIM];       // 16,384 B
    __shared__ __nv_bfloat16 smem_q[HEAD_DIM];                   // 256 B (reused per Q head)
    __shared__ float smem_scores[TILE_SIZE];                      // 256 B
    __shared__ float warp_partial[NUM_WARPS * (TILE_SIZE + 1)];   // 1,040 B
    __shared__ float smem_scratch[NUM_WARPS];                     // 16 B
    __shared__ float smem_rms[2];                                 // 8 B
    __shared__ float smem_running_max;                            // 4 B
    __shared__ float smem_running_sum;                            // 4 B
    // Total: ~34.0 KB (fits 48 KB limit)

    int cache_base_offset = kv_head_idx * max_seq_len * head_dim;

    // Phase 1: K head — slice → norm → rope → write to global cache
    __nv_bfloat16 k_elem = k_full[kv_head_idx * head_dim + tid];

    float k_sq = __bfloat162float(k_elem);
    k_sq = k_sq * k_sq;
    float k_sq_sum = warp_reduce_sum(k_sq);

    if (lane_id == 0) {
        smem_scratch[warp_id] = k_sq_sum;
    }
    __syncthreads();

    if (tid == 0) {
        float total = 0.0f;
        for (int i = 0; i < NUM_WARPS; i++) {
            total += smem_scratch[i];
        }
        smem_rms[1] = 1.0f / sqrtf(total / head_dim + rms_eps);
    }
    __syncthreads();

    __nv_bfloat16 k_normed = rms_norm_elem(k_elem, smem_rms[1], k_norm_weight[tid]);

    // Half-split RoPE: pair (tid, tid + half_dim), only threads 0..half_dim-1 rotate
    int half_dim = head_dim / 2;
    // Store normed K in shared memory so paired thread can read it
    smem_k[tid] = k_normed;
    __syncthreads();

    if (tid < half_dim) {
        __nv_bfloat16 k_lo = smem_k[tid];
        __nv_bfloat16 k_hi = smem_k[tid + half_dim];

        apply_rope_pair(k_lo, k_hi, cos_cache[tid], sin_cache[tid]);

        int cache_offset = cache_base_offset + current_pos * head_dim;
        k_cache[cache_offset + tid] = k_lo;
        k_cache[cache_offset + tid + half_dim] = k_hi;
    }
    __syncthreads();

    // Phase 2: V head — slice → write to global cache
    __nv_bfloat16 v_elem = v_full[kv_head_idx * head_dim + tid];
    v_cache[cache_base_offset + current_pos * head_dim + tid] = v_elem;
    __syncthreads();

    // Phase 3: Loop over all Q heads for this KV head
    for (int q = 0; q < gqa_ratio; q++) {
        int q_head_idx = kv_head_idx * gqa_ratio + q;
        if (q_head_idx >= num_qheads) break;

        // Q head — slice → norm → rope → smem_q
        __nv_bfloat16 q_elem = q_full[q_head_idx * head_dim + tid];

        float q_sq = __bfloat162float(q_elem);
        q_sq = q_sq * q_sq;
        float q_sq_sum = warp_reduce_sum(q_sq);

        if (lane_id == 0) {
            smem_scratch[warp_id] = q_sq_sum;
        }
        __syncthreads();

        if (tid == 0) {
            float total = 0.0f;
            for (int i = 0; i < NUM_WARPS; i++) {
                total += smem_scratch[i];
            }
            smem_rms[0] = 1.0f / sqrtf(total / head_dim + rms_eps);
        }
        __syncthreads();

        __nv_bfloat16 q_normed = rms_norm_elem(q_elem, smem_rms[0], q_norm_weight[tid]);

        // Half-split RoPE: pair (tid, tid + half_dim)
        smem_q[tid] = q_normed;
        __syncthreads();

        if (tid < half_dim) {
            __nv_bfloat16 q_lo = smem_q[tid];
            __nv_bfloat16 q_hi = smem_q[tid + half_dim];

            apply_rope_pair(q_lo, q_hi, cos_cache[tid], sin_cache[tid]);

            smem_q[tid] = q_lo;
            smem_q[tid + half_dim] = q_hi;
        }
        __syncthreads();

        // Tiled attention for this Q head
        tiled_attention(
            smem_q,
            k_cache + cache_base_offset,
            v_cache + cache_base_offset,
            smem_k, smem_v,
            smem_scores, warp_partial, smem_scratch,
            smem_running_max, smem_running_sum,
            output, q_head_idx,
            seq_len, max_seq_len, head_dim, scale,
            tid, warp_id, lane_id
        );
        __syncthreads();
    }
}

// Batched decode attention — split-KV variant
//
// Processes B requests in a single launch. Each request has its own KV cache
// (accessed via pointer arrays). Uses split-KV + online softmax.
//
// Grid: (num_kvheads, num_kv_splits, batch_size)
//   blockIdx.x = kv_head_idx (serves gqa_ratio query heads)
//   blockIdx.y = split_id (KV chunk index)
//   blockIdx.z = batch_idx
// Threads: HEAD_DIM (256)

// Default split count. The launcher raises it to fill the GPU: the grid is
// (kv_heads, splits, batch), so at batch 1 four kv heads leave 78 SMs idle
// unless the KV axis supplies the blocks.
#define NUM_KV_SPLITS 4
#define BATCHED_DECODE_MAX_KV_SPLITS 64
#define BATCHED_DECODE_MAX_GQA 8
#define BATCHED_BLOCK_N 64
#define BATCHED_DECODE_HEAD_DIM 256
#define BATCHED_DECODE_THREADS 256
#define BATCHED_DECODE_NUM_WARPS (BATCHED_DECODE_THREADS / WARP_SIZE)

__global__ void fused_gqa_attention_decode_batched_kernel(
    const __nv_bfloat16* __restrict__ q_batch,    // [B, num_qheads * 2 * head_dim] (Q + gate)
    const __nv_bfloat16* __restrict__ k_batch,    // [B, kv_dim]
    const __nv_bfloat16* __restrict__ v_batch,    // [B, kv_dim]
    const __nv_bfloat16* __restrict__ q_norm_weight,
    const __nv_bfloat16* __restrict__ k_norm_weight,
    const __nv_bfloat16* __restrict__ cos_cache,  // [max_seq_len, head_dim]
    const __nv_bfloat16* __restrict__ sin_cache,  // [max_seq_len, head_dim]
    const int* __restrict__ positions,             // [B] current_pos per request
    const int* __restrict__ seq_lens,              // [B] seq_len (= pos + 1)
    const __nv_bfloat16* const* __restrict__ k_cache_ptrs, // [B] device ptrs to per-request K cache
    const __nv_bfloat16* const* __restrict__ v_cache_ptrs, // [B] device ptrs to per-request V cache
    float* __restrict__ partial_out,               // [B, num_qheads, num_kv_splits, HEAD_DIM]
    float* __restrict__ partial_m,                 // [B, num_qheads, num_kv_splits]
    float* __restrict__ partial_l,                 // [B, num_qheads, num_kv_splits]
    int num_qheads,
    int num_kvheads,
    int gqa_ratio,
    int head_dim,
    int rotary_dim,
    int max_seq_len,
    int num_kv_splits,
    float rms_eps
) {
    // One CTA per (KV head, split, request) — NOT per query head. The KV cache
    // is stored per KV head, so a query-head grid made all `gqa_ratio` blocks
    // walk the same cache independently: gqa_ratio x the KV traffic and
    // gqa_ratio x the current-token cache write. Here the tile is loaded once
    // and reused across the group from registers.
    int kv_head_idx = blockIdx.x;
    int split_id = blockIdx.y;
    int batch_idx = blockIdx.z;
    int q_head_0 = kv_head_idx * gqa_ratio;

    int tid = threadIdx.x;  // 0..HEAD_DIM-1
    int half_rotary = rotary_dim / 2;

    int current_pos = positions[batch_idx];
    float scale = 1.0f / sqrtf((float)head_dim);
    float qk_scale = scale * 1.44269504f;  // scale * log2(e) for exp2 trick

    __shared__ float smem_scratch[BATCHED_DECODE_NUM_WARPS];
    __shared__ float smem_q[BATCHED_DECODE_MAX_GQA][BATCHED_DECODE_HEAD_DIM];
    __shared__ float smem_qk[BATCHED_DECODE_MAX_GQA][BATCHED_BLOCK_N];

    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    int q_full_dim = num_qheads * head_dim * 2;

    // RMSNorm + RoPE every query head of the group into shared. Q is 256 floats
    // per head, so the whole group is 6 KB and the tile loop reads it from
    // there instead of re-deriving it per block.
    for (int h = 0; h < gqa_ratio; h++) {
        int q_base = batch_idx * q_full_dim + (q_head_0 + h) * 2 * head_dim;
        float q_val = __bfloat162float(q_batch[q_base + tid]);
        float q_sq_sum = warp_reduce_sum(q_val * q_val);
        if (lane_id == 0) smem_scratch[warp_id] = q_sq_sum;
        __syncthreads();
        if (tid == 0) {
            float total = 0.0f;
            for (int i = 0; i < BATCHED_DECODE_NUM_WARPS; i++) total += smem_scratch[i];
            smem_scratch[0] = 1.0f / sqrtf(total / head_dim + rms_eps);
        }
        __syncthreads();
        // #58: hd256 q_norm is an OFFSET
        float q_normed = q_val * smem_scratch[0] * (1.0f + __bfloat162float(q_norm_weight[tid]));
        smem_q[h][tid] = q_normed;
        __syncthreads();
        // RoPE — partial half-split: lo = 0..rotary_dim/2-1,
        // hi = rotary_dim/2..rotary_dim-1; dims >= rotary_dim pass through.
        float q_rot = q_normed;
        if (tid < half_rotary) {
            float cos_val = __bfloat162float(cos_cache[current_pos * rotary_dim + tid]);
            float sin_val = __bfloat162float(sin_cache[current_pos * rotary_dim + tid]);
            q_rot = smem_q[h][tid] * cos_val - smem_q[h][tid + half_rotary] * sin_val;
        } else if (tid < rotary_dim) {
            int pair = tid - half_rotary;
            float cos_val = __bfloat162float(cos_cache[current_pos * rotary_dim + pair]);
            float sin_val = __bfloat162float(sin_cache[current_pos * rotary_dim + pair]);
            q_rot = smem_q[h][pair] * sin_val + smem_q[h][tid] * cos_val;
        }
        __syncthreads();
        smem_q[h][tid] = q_rot;
    }
    __syncthreads();

    int kv_base = batch_idx * num_kvheads * head_dim + kv_head_idx * head_dim;
    float k_val = __bfloat162float(k_batch[kv_base + tid]);

    float k_sq_sum = warp_reduce_sum(k_val * k_val);
    if (lane_id == 0) smem_scratch[warp_id] = k_sq_sum;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (int i = 0; i < BATCHED_DECODE_NUM_WARPS; i++) total += smem_scratch[i];
        smem_scratch[0] = 1.0f / sqrtf(total / head_dim + rms_eps);
    }
    __syncthreads();
    float k_normed = k_val * smem_scratch[0] * (1.0f + __bfloat162float(k_norm_weight[tid]));  // #58 OFFSET

    __shared__ float smem_k_rope[BATCHED_DECODE_HEAD_DIM];
    smem_k_rope[tid] = k_normed;
    __syncthreads();

    float k_rot = k_normed;
    if (tid < half_rotary) {
        float cos_val = __bfloat162float(cos_cache[current_pos * rotary_dim + tid]);
        float sin_val = __bfloat162float(sin_cache[current_pos * rotary_dim + tid]);
        k_rot = smem_k_rope[tid] * cos_val - smem_k_rope[tid + half_rotary] * sin_val;
    } else if (tid < rotary_dim) {
        int pair = tid - half_rotary;
        float cos_val = __bfloat162float(cos_cache[current_pos * rotary_dim + pair]);
        float sin_val = __bfloat162float(sin_cache[current_pos * rotary_dim + pair]);
        k_rot = smem_k_rope[pair] * sin_val + smem_k_rope[tid] * cos_val;
    }

    float v_val = __bfloat162float(v_batch[kv_base + tid]);

    // Cast away const for cache write — k_cache_ptrs/v_cache_ptrs point to mutable cache buffers
    // but the pointer array itself is const.
    __nv_bfloat16* k_cache = const_cast<__nv_bfloat16*>(k_cache_ptrs[batch_idx]);
    __nv_bfloat16* v_cache = const_cast<__nv_bfloat16*>(v_cache_ptrs[batch_idx]);
    int cache_head_offset = kv_head_idx * max_seq_len * head_dim;

    if (split_id == 0) {
        int cur_off = cache_head_offset + current_pos * head_dim + tid;
        k_cache[cur_off] = __float2bfloat16(k_rot);
        v_cache[cur_off] = __float2bfloat16(v_val);
    }

    // seq_lens includes the current decode token; split-KV scans only the prefix
    // because split 0 handles the current token from registers below.
    int past_seq_len = max(0, seq_lens[batch_idx] - 1);
    int tiles_total = (past_seq_len + BATCHED_BLOCK_N - 1) / BATCHED_BLOCK_N;
    int tiles_per_split = (tiles_total + num_kv_splits - 1) / num_kv_splits;
    int split_start = split_id * tiles_per_split * BATCHED_BLOCK_N;
    int split_end = min((split_id + 1) * tiles_per_split * BATCHED_BLOCK_N, past_seq_len);

    // Unrolled over the compile-time cap and predicated on `gqa_ratio`: a
    // runtime bound makes the index non-constant and spills all three arrays to
    // local memory, which is the hot loop's innermost access.
    float acc[BATCHED_DECODE_MAX_GQA];
    float m_i[BATCHED_DECODE_MAX_GQA];
    float l_i[BATCHED_DECODE_MAX_GQA];
    #pragma unroll
    for (int h = 0; h < BATCHED_DECODE_MAX_GQA; h++) {
        acc[h] = 0.0f;
        m_i[h] = -1e38f;  // finite instead of -inf to avoid NaN
        l_i[h] = 0.0f;
    }

    const __nv_bfloat16* k_cache_head = k_cache_ptrs[batch_idx] + cache_head_offset;
    const __nv_bfloat16* v_cache_head = v_cache_ptrs[batch_idx] + cache_head_offset;

    // Per-key: one warp owns one key and reduces its own dot product with a
    // shuffle. The previous shape gave every key two __syncthreads() and a
    // single-threaded 8-add — 65536 barriers per CTA at 32k, with 255 threads
    // idle inside each one. Barriers now cost one per tile, not per key.
    for (int tile_start = split_start; tile_start < split_end; tile_start += BATCHED_BLOCK_N) {
        int tile_len = min(BATCHED_BLOCK_N, split_end - tile_start);

        for (int pos = warp_id; pos < tile_len; pos += BATCHED_DECODE_NUM_WARPS) {
            const __nv_bfloat16* k_row = k_cache_head + (size_t)(tile_start + pos) * head_dim;
            float k_reg[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                k_reg[i] = __bfloat162float(k_row[lane_id + i * WARP_SIZE]);
            }
            // One K row, `gqa_ratio` dot products — this is the read the
            // query-head grid used to repeat.
            #pragma unroll
            for (int h = 0; h < BATCHED_DECODE_MAX_GQA; h++) {
                if (h >= gqa_ratio) continue;
                float dot = 0.0f;
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    dot += smem_q[h][lane_id + i * WARP_SIZE] * k_reg[i];
                }
                dot = warp_reduce_sum(dot);
                if (lane_id == 0) smem_qk[h][pos] = dot * qk_scale;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int h = 0; h < BATCHED_DECODE_MAX_GQA; h++) {
            if (h >= gqa_ratio) continue;
            float tile_max = -INFINITY;
            for (int pos = 0; pos < tile_len; pos++) {
                tile_max = fmaxf(tile_max, smem_qk[h][pos]);
            }
            float m_new = fmaxf(m_i[h], tile_max);
            float alpha = exp2f(m_i[h] - m_new);
            acc[h] *= alpha;
            l_i[h] *= alpha;
            m_i[h] = m_new;
        }

        for (int pos = 0; pos < tile_len; pos++) {
            // One V element per thread per key, shared by the whole group.
            float v_elem = __bfloat162float(v_cache_head[(size_t)(tile_start + pos) * head_dim + tid]);
            #pragma unroll
            for (int h = 0; h < BATCHED_DECODE_MAX_GQA; h++) {
                if (h >= gqa_ratio) continue;
                float w = exp2f(smem_qk[h][pos] - m_i[h]);
                acc[h] += w * v_elem;
                l_i[h] += w;
            }
        }
        __syncthreads();
    }

    if (split_id == 0) {
        // Current token, whole group in one barrier pair: each warp reduces its
        // slice of every head's dot into smem_qk (free at this point), then one
        // sync and each head folds its own partials.
        #pragma unroll
        for (int h = 0; h < BATCHED_DECODE_MAX_GQA; h++) {
            if (h >= gqa_ratio) continue;
            float dot = warp_reduce_sum(smem_q[h][tid] * k_rot);
            if (lane_id == 0) smem_qk[h][warp_id] = dot;
        }
        __syncthreads();
        #pragma unroll
        for (int h = 0; h < BATCHED_DECODE_MAX_GQA; h++) {
            if (h >= gqa_ratio) continue;
            float qk_cur = 0.0f;
            for (int w = 0; w < BATCHED_DECODE_NUM_WARPS; w++) qk_cur += smem_qk[h][w];
            qk_cur *= qk_scale;

            float m_new = fmaxf(m_i[h], qk_cur);
            float alpha = exp2f(m_i[h] - m_new);
            float p_cur = exp2f(qk_cur - m_new);

            acc[h] = acc[h] * alpha + v_val * p_cur;
            l_i[h] = l_i[h] * alpha + p_cur;
            m_i[h] = m_new;
        }
    }

    #pragma unroll
    for (int h = 0; h < BATCHED_DECODE_MAX_GQA; h++) {
        if (h >= gqa_ratio) continue;
        int partial_base_head = (batch_idx * num_qheads + q_head_0 + h) * num_kv_splits;
        partial_out[(partial_base_head + split_id) * head_dim + tid] = acc[h];
        if (tid == 0) {
            partial_m[partial_base_head + split_id] = m_i[h];
            partial_l[partial_base_head + split_id] = l_i[h];
        }
    }
}

// Batched attention reduce kernel
//
// Merges `num_kv_splits` partial results per Q head per batch item.
// Grid: (num_qheads, batch_size)
// Threads: HEAD_DIM (128)
__global__ void attention_decode_reduce_batched_kernel(
    const float* __restrict__ partial_out, // [B, num_qheads, num_kv_splits, HEAD_DIM]
    const float* __restrict__ partial_m,   // [B, num_qheads, num_kv_splits]
    const float* __restrict__ partial_l,   // [B, num_qheads, num_kv_splits]
    __nv_bfloat16* __restrict__ output,    // [B, q_dim]
    int num_qheads,
    int head_dim,
    int num_kv_splits
) {
    int q_head_idx = blockIdx.x;
    int batch_idx = blockIdx.y;
    int tid = threadIdx.x;  // 0..HEAD_DIM-1

    int base = (batch_idx * num_qheads + q_head_idx) * num_kv_splits;

    float acc = 0.0f;
    float m_global = -INFINITY;
    float l_global = 0.0f;

    for (int s = 0; s < num_kv_splits; s++) {
        float m_s = partial_m[base + s];
        float l_s = partial_l[base + s];
        float p = partial_out[(base + s) * head_dim + tid];

        float m_new = fmaxf(m_global, m_s);
        float alpha_old = exp2f(m_global - m_new);
        float alpha_new = exp2f(m_s - m_new);

        acc = acc * alpha_old + p * alpha_new;
        l_global = l_global * alpha_old + l_s * alpha_new;
        m_global = m_new;
    }

    float result = (l_global > 0.0f) ? (acc / l_global) : 0.0f;
    int out_offset = batch_idx * num_qheads * head_dim + q_head_idx * head_dim + tid;
    output[out_offset] = __float2bfloat16(result);
}

extern "C" {

cudaError_t fused_gqa_attention_decode_batched(
    const __nv_bfloat16* q_batch,
    const __nv_bfloat16* k_batch,
    const __nv_bfloat16* v_batch,
    const __nv_bfloat16* q_norm_weight,
    const __nv_bfloat16* k_norm_weight,
    const __nv_bfloat16* cos_cache,
    const __nv_bfloat16* sin_cache,
    const int* positions,
    const int* seq_lens,
    const __nv_bfloat16* const* k_cache_ptrs,
    const __nv_bfloat16* const* v_cache_ptrs,
    float* partial_out,
    float* partial_m,
    float* partial_l,
    int num_qheads,
    int num_kvheads,
    int gqa_ratio,
    int head_dim,
    int rotary_dim,
    int max_seq_len,
    int batch_size,
    int num_kv_splits,
    float rms_eps,
    cudaStream_t stream
) {
    if (q_batch == nullptr || k_batch == nullptr || v_batch == nullptr ||
        q_norm_weight == nullptr || k_norm_weight == nullptr || cos_cache == nullptr ||
        sin_cache == nullptr || positions == nullptr || seq_lens == nullptr ||
        k_cache_ptrs == nullptr || v_cache_ptrs == nullptr || partial_out == nullptr ||
        partial_m == nullptr || partial_l == nullptr || num_qheads <= 0 ||
        num_kvheads <= 0 || gqa_ratio <= 0 ||
        num_qheads != num_kvheads * gqa_ratio || head_dim != BATCHED_DECODE_HEAD_DIM ||
        rotary_dim <= 0 || rotary_dim > head_dim || rotary_dim % 2 != 0 ||
        max_seq_len <= 0 || batch_size <= 0 || gqa_ratio > BATCHED_DECODE_MAX_GQA ||
        num_kv_splits <= 0 || num_kv_splits > BATCHED_DECODE_MAX_KV_SPLITS) {
        return cudaErrorInvalidValue;
    }
    dim3 grid(num_kvheads, num_kv_splits, batch_size);
    int threads = BATCHED_DECODE_THREADS;

    fused_gqa_attention_decode_batched_kernel<<<grid, threads, 0, stream>>>(
        q_batch, k_batch, v_batch,
        q_norm_weight, k_norm_weight,
        cos_cache, sin_cache,
        positions, seq_lens,
        k_cache_ptrs, v_cache_ptrs,
        partial_out, partial_m, partial_l,
        num_qheads, num_kvheads, gqa_ratio, head_dim,
        rotary_dim, max_seq_len, num_kv_splits, rms_eps
    );
    return cudaGetLastError();
}

cudaError_t attention_decode_reduce_batched(
    const float* partial_out,
    const float* partial_m,
    const float* partial_l,
    __nv_bfloat16* output,
    int num_qheads,
    int head_dim,
    int batch_size,
    int num_kv_splits,
    cudaStream_t stream
) {
    if (partial_out == nullptr || partial_m == nullptr || partial_l == nullptr ||
        output == nullptr || num_qheads <= 0 || head_dim != BATCHED_DECODE_HEAD_DIM ||
        batch_size <= 0 || num_kv_splits <= 0 ||
        num_kv_splits > BATCHED_DECODE_MAX_KV_SPLITS) {
        return cudaErrorInvalidValue;
    }
    dim3 grid(num_qheads, batch_size);
    int threads = head_dim;

    attention_decode_reduce_batched_kernel<<<grid, threads, 0, stream>>>(
        partial_out, partial_m, partial_l,
        output, num_qheads, head_dim, num_kv_splits
    );
    return cudaGetLastError();
}

void fused_gqa_attention_single_token(
    const __nv_bfloat16* q_full,
    const __nv_bfloat16* k_full,
    const __nv_bfloat16* v_full,
    const __nv_bfloat16* q_norm_weight,
    const __nv_bfloat16* k_norm_weight,
    const __nv_bfloat16* cos_cache,
    const __nv_bfloat16* sin_cache,
    __nv_bfloat16* k_cache,
    __nv_bfloat16* v_cache,
    __nv_bfloat16* output,
    int num_qheads,
    int num_kvheads,
    int gqa_ratio,
    int head_dim,
    int current_pos,
    int seq_len,
    float scale,
    float rms_eps,
    cudaStream_t stream
) {
    int num_blocks = num_kvheads;
    int threads_per_block = head_dim;  // 128
    int max_seq_len = 4096;

    fused_gqa_attention_single_token_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        q_full, k_full, v_full,
        q_norm_weight, k_norm_weight,
        cos_cache, sin_cache,
        k_cache, v_cache,
        output,
        num_qheads, num_kvheads, gqa_ratio, head_dim,
        current_pos, seq_len, max_seq_len,
        scale, rms_eps
    );
}

} // extern "C"
