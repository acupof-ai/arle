#include "common.cuh"
#include <cstdio>

// ============================================================================
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
// ============================================================================

#define TILE_SIZE 64
#define HEAD_DIM 128
#define THREADS_PER_BLOCK 128
#define NUM_WARPS (THREADS_PER_BLOCK / WARP_SIZE)  // 4

// ============================================================================
// Device helpers
// ============================================================================

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

// ============================================================================
// Tiled attention for a single Q head using online softmax.
//
// All shared memory buffers are allocated by the caller (kernel) and passed in.
// No __shared__ declarations inside this function.
// ============================================================================
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
    // Initialize online softmax state
    float o_acc = 0.0f;  // output accumulator for dimension tid (register)

    if (tid == 0) {
        smem_running_max = -INFINITY;
        smem_running_sum = 0.0f;
    }
    __syncthreads();

    // Tile loop over KV cache
    for (int tile_start = 0; tile_start < seq_len; tile_start += TILE_SIZE) {
        int tile_len = min(TILE_SIZE, seq_len - tile_start);

        // --- Load K/V tile from global cache into shared memory ---
        for (int i = tid; i < tile_len * HEAD_DIM; i += THREADS_PER_BLOCK) {
            int pos_in_tile = i / HEAD_DIM;
            int dim = i % HEAD_DIM;
            int abs_pos = tile_start + pos_in_tile;
            smem_k[pos_in_tile * HEAD_DIM + dim] = k_cache_base[abs_pos * head_dim + dim];
            smem_v[pos_in_tile * HEAD_DIM + dim] = v_cache_base[abs_pos * head_dim + dim];
        }
        __syncthreads();

        // --- Compute scores: Q · K^T * scale ---
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

        // --- Find tile max ---
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

        // --- Exp weights + V accumulation ---
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

    // --- Final normalize ---
    float final_sum = smem_running_sum;
    float result = (final_sum > 0.0f) ? (o_acc / final_sum) : 0.0f;
    output_buf[q_head_idx * head_dim + tid] = __float2bfloat16(result);
}

// ============================================================================
// Main kernel — supports arbitrary GQA ratio via gqa_ratio parameter
// ============================================================================
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

    // ========================================================================
    // Phase 1: K head — slice → norm → rope → write to global cache
    // ========================================================================
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

    // ========================================================================
    // Phase 2: V head — slice → write to global cache
    // ========================================================================
    __nv_bfloat16 v_elem = v_full[kv_head_idx * head_dim + tid];
    v_cache[cache_base_offset + current_pos * head_dim + tid] = v_elem;
    __syncthreads();

    // ========================================================================
    // Phase 3: Loop over all Q heads for this KV head
    // ========================================================================
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

// ============================================================================
// Batched decode attention — split-KV variant
//
// Processes B requests in a single launch. Each request has its own KV cache
// (accessed via pointer arrays). Uses split-KV + online softmax.
//
// Grid: (num_qheads, NUM_KV_SPLITS, batch_size)
//   blockIdx.x = q_head_idx
//   blockIdx.y = split_id (KV chunk index)
//   blockIdx.z = batch_idx
// Threads: HEAD_DIM (128)
// ============================================================================

#define NUM_KV_SPLITS 4
#define BATCHED_BLOCK_N 64

__global__ void fused_gqa_attention_decode_batched_kernel(
    const __nv_bfloat16* __restrict__ q_batch,    // [B, q_dim]
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
    float* __restrict__ partial_out,               // [B, num_qheads, NUM_KV_SPLITS, HEAD_DIM]
    float* __restrict__ partial_m,                 // [B, num_qheads, NUM_KV_SPLITS]
    float* __restrict__ partial_l,                 // [B, num_qheads, NUM_KV_SPLITS]
    int num_qheads,
    int num_kvheads,
    int gqa_ratio,
    int head_dim,
    int max_seq_len,
    float rms_eps
) {
    int q_head_idx = blockIdx.x;
    int split_id = blockIdx.y;
    int batch_idx = blockIdx.z;
    int kv_head_idx = q_head_idx / gqa_ratio;

    int tid = threadIdx.x;  // 0..HEAD_DIM-1
    int half_dim = head_dim / 2;

    int current_pos = positions[batch_idx];
    float scale = 1.0f / sqrtf((float)head_dim);
    float qk_scale = scale * 1.44269504f;  // scale * log2(e) for exp2 trick

    // ---- Shared memory for Q/K norm computation ----
    __shared__ float smem_scratch[NUM_WARPS];

    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // ---- Load Q, apply RMSNorm + RoPE ----
    int q_base = batch_idx * num_qheads * head_dim + q_head_idx * head_dim;
    float q_val = __bfloat162float(q_batch[q_base + tid]);

    // RMSNorm for Q
    float q_sq = q_val * q_val;
    float q_sq_sum = warp_reduce_sum(q_sq);
    if (lane_id == 0) smem_scratch[warp_id] = q_sq_sum;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (int i = 0; i < NUM_WARPS; i++) total += smem_scratch[i];
        smem_scratch[0] = 1.0f / sqrtf(total / head_dim + rms_eps);
    }
    __syncthreads();
    float q_rms = smem_scratch[0];
    float q_normed = q_val * q_rms * __bfloat162float(q_norm_weight[tid]);

    // RoPE for Q — half-split: lo = 0..half_dim-1, hi = half_dim..head_dim-1
    // Store in shared memory for paired access
    __shared__ float smem_q_rope[HEAD_DIM];
    smem_q_rope[tid] = q_normed;
    __syncthreads();

    float q_rot;
    if (tid < half_dim) {
        float cos_val = __bfloat162float(cos_cache[current_pos * head_dim + tid]);
        float sin_val = __bfloat162float(sin_cache[current_pos * head_dim + tid]);
        q_rot = smem_q_rope[tid] * cos_val - smem_q_rope[tid + half_dim] * sin_val;
    } else {
        int pair = tid - half_dim;
        float cos_val = __bfloat162float(cos_cache[current_pos * head_dim + pair]);
        float sin_val = __bfloat162float(sin_cache[current_pos * head_dim + pair]);
        q_rot = smem_q_rope[pair] * sin_val + smem_q_rope[tid] * cos_val;
    }

    // ---- Load K, apply RMSNorm + RoPE ----
    int kv_base = batch_idx * num_kvheads * head_dim + kv_head_idx * head_dim;
    float k_val = __bfloat162float(k_batch[kv_base + tid]);

    float k_sq = k_val * k_val;
    float k_sq_sum = warp_reduce_sum(k_sq);
    if (lane_id == 0) smem_scratch[warp_id] = k_sq_sum;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (int i = 0; i < NUM_WARPS; i++) total += smem_scratch[i];
        smem_scratch[0] = 1.0f / sqrtf(total / head_dim + rms_eps);
    }
    __syncthreads();
    float k_rms = smem_scratch[0];
    float k_normed = k_val * k_rms * __bfloat162float(k_norm_weight[tid]);

    __shared__ float smem_k_rope[HEAD_DIM];
    smem_k_rope[tid] = k_normed;
    __syncthreads();

    float k_rot;
    if (tid < half_dim) {
        float cos_val = __bfloat162float(cos_cache[current_pos * head_dim + tid]);
        float sin_val = __bfloat162float(sin_cache[current_pos * head_dim + tid]);
        k_rot = smem_k_rope[tid] * cos_val - smem_k_rope[tid + half_dim] * sin_val;
    } else {
        int pair = tid - half_dim;
        float cos_val = __bfloat162float(cos_cache[current_pos * head_dim + pair]);
        float sin_val = __bfloat162float(sin_cache[current_pos * head_dim + pair]);
        k_rot = smem_k_rope[pair] * sin_val + smem_k_rope[tid] * cos_val;
    }

    // ---- Load V ----
    float v_val = __bfloat162float(v_batch[kv_base + tid]);

    // ---- Split 0 only: write current K/V to KV cache ----
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

    // ---- Compute this split's KV range ----
    // seq_len here means number of *past* tokens (before the current one)
    int past_seq_len = current_pos;
    int tiles_total = (past_seq_len + BATCHED_BLOCK_N - 1) / BATCHED_BLOCK_N;
    int tiles_per_split = (tiles_total + NUM_KV_SPLITS - 1) / NUM_KV_SPLITS;
    int split_start = split_id * tiles_per_split * BATCHED_BLOCK_N;
    int split_end = min((split_id + 1) * tiles_per_split * BATCHED_BLOCK_N, past_seq_len);

    // ---- Online softmax attention over this split's KV chunk ----
    float acc = 0.0f;  // output accumulator for dimension tid
    float m_i = -1e38f;  // running max (finite instead of -inf to avoid NaN)
    float l_i = 0.0f;    // running sum

    // K/V cache base for this KV head
    const __nv_bfloat16* k_cache_head = k_cache_ptrs[batch_idx] + cache_head_offset;
    const __nv_bfloat16* v_cache_head = v_cache_ptrs[batch_idx] + cache_head_offset;

    // Shared memory for scores within a tile
    __shared__ float smem_qk[BATCHED_BLOCK_N];

    for (int tile_start = split_start; tile_start < split_end; tile_start += BATCHED_BLOCK_N) {
        int tile_len = min(BATCHED_BLOCK_N, split_end - tile_start);

        // Compute QK dot products for all positions in this tile
        // Each thread holds one dimension of Q; we iterate over K positions
        for (int pos = 0; pos < tile_len; pos++) {
            int abs_pos = tile_start + pos;
            float k_elem = __bfloat162float(k_cache_head[abs_pos * head_dim + tid]);
            float dot = q_rot * k_elem;
            dot = warp_reduce_sum(dot);
            if (lane_id == 0) {
                smem_scratch[warp_id] = dot;
            }
            __syncthreads();
            if (tid == 0) {
                float score = 0.0f;
                for (int w = 0; w < NUM_WARPS; w++) score += smem_scratch[w];
                smem_qk[pos] = score * qk_scale;
            }
            __syncthreads();
        }

        // Find tile max
        float tile_max = -INFINITY;
        for (int pos = 0; pos < tile_len; pos++) {
            tile_max = fmaxf(tile_max, smem_qk[pos]);
        }

        // Online softmax update
        float m_new = fmaxf(m_i, tile_max);
        float alpha = exp2f(m_i - m_new);
        acc *= alpha;
        l_i *= alpha;

        // Accumulate weighted V
        for (int pos = 0; pos < tile_len; pos++) {
            float w = exp2f(smem_qk[pos] - m_new);
            float v_elem = __bfloat162float(v_cache_head[(tile_start + pos) * head_dim + tid]);
            acc += w * v_elem;
            l_i += w;
        }
        m_i = m_new;
    }

    // ---- Split 0: handle current token from registers ----
    if (split_id == 0) {
        // Dot product of q_rot and k_rot
        float dot = q_rot * k_rot;
        dot = warp_reduce_sum(dot);
        if (lane_id == 0) smem_scratch[warp_id] = dot;
        __syncthreads();
        if (tid == 0) {
            float score = 0.0f;
            for (int w = 0; w < NUM_WARPS; w++) score += smem_scratch[w];
            smem_scratch[0] = score * qk_scale;
        }
        __syncthreads();
        float qk_cur = smem_scratch[0];

        float m_new = fmaxf(m_i, qk_cur);
        float alpha = exp2f(m_i - m_new);
        float p_cur = exp2f(qk_cur - m_new);

        acc = acc * alpha + v_val * p_cur;
        l_i = l_i * alpha + p_cur;
        m_i = m_new;
    }

    // ---- Write partial results (FP32, unnormalized) ----
    int partial_base_head = (batch_idx * num_qheads + q_head_idx) * NUM_KV_SPLITS;
    int partial_out_offset = (partial_base_head + split_id) * head_dim + tid;
    partial_out[partial_out_offset] = acc;

    int scalar_offset = partial_base_head + split_id;
    if (tid == 0) {
        partial_m[scalar_offset] = m_i;
        partial_l[scalar_offset] = l_i;
    }
}

// ============================================================================
// Batched attention reduce kernel
//
// Merges NUM_KV_SPLITS partial results per Q head per batch item.
// Grid: (num_qheads, batch_size)
// Threads: HEAD_DIM (128)
// ============================================================================
__global__ void attention_decode_reduce_batched_kernel(
    const float* __restrict__ partial_out, // [B, num_qheads, NUM_KV_SPLITS, HEAD_DIM]
    const float* __restrict__ partial_m,   // [B, num_qheads, NUM_KV_SPLITS]
    const float* __restrict__ partial_l,   // [B, num_qheads, NUM_KV_SPLITS]
    __nv_bfloat16* __restrict__ output,    // [B, q_dim]
    int num_qheads,
    int head_dim
) {
    int q_head_idx = blockIdx.x;
    int batch_idx = blockIdx.y;
    int tid = threadIdx.x;  // 0..HEAD_DIM-1

    int base = (batch_idx * num_qheads + q_head_idx) * NUM_KV_SPLITS;

    float acc = 0.0f;
    float m_global = -INFINITY;
    float l_global = 0.0f;

    #pragma unroll
    for (int s = 0; s < NUM_KV_SPLITS; s++) {
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

// ============================================================================
// Single-request (batch=1) split-KV decode attention
//
// Specialization of `fused_gqa_attention_decode_batched_kernel` with the
// batch dim dropped and `current_pos` read from the on-device `decode_meta`
// buffer (`[token_id, current_pos, seq_len]`), matching the CUDA-Graph-safe
// contract used by CUDA Graph capture. Online-softmax numerics and tile layout
// are copied verbatim from the batched kernel so there is no numerical drift.
// HEAD_DIM / rms_eps / NUM_KV_SPLITS are hardcoded to match the Rust FFI
// signature that does not plumb them through.
//
// Grid: (num_qheads, NUM_KV_SPLITS)
//   blockIdx.x = q_head_idx
//   blockIdx.y = split_id (KV chunk index)
// Threads: HEAD_DIM (128)
// ============================================================================
__global__ void fused_gqa_attention_decode_single_kernel(
    const __nv_bfloat16* __restrict__ q_full,
    const __nv_bfloat16* __restrict__ k_full,
    const __nv_bfloat16* __restrict__ v_full,
    const __nv_bfloat16* __restrict__ q_norm_weight,
    const __nv_bfloat16* __restrict__ k_norm_weight,
    const __nv_bfloat16* __restrict__ cos_cache,
    const __nv_bfloat16* __restrict__ sin_cache,
    const int* __restrict__ decode_meta,       // [token_id, current_pos, seq_len]
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    float* __restrict__ partial_out,           // [num_qheads, NUM_KV_SPLITS, HEAD_DIM]
    float* __restrict__ partial_m,             // [num_qheads, NUM_KV_SPLITS]
    float* __restrict__ partial_l,             // [num_qheads, NUM_KV_SPLITS]
    int num_qheads,
    int num_kvheads,
    int gqa_ratio,
    int max_seq_len
) {
    constexpr int head_dim = HEAD_DIM;
    constexpr float rms_eps = 1e-6f;

    int q_head_idx = blockIdx.x;
    int split_id = blockIdx.y;
    int kv_head_idx = q_head_idx / gqa_ratio;

    int tid = threadIdx.x;  // 0..HEAD_DIM-1
    int half_dim = head_dim / 2;

    int current_pos = decode_meta[1];
    float scale = 1.0f / sqrtf((float)head_dim);
    float qk_scale = scale * 1.44269504f;  // scale * log2(e) for exp2 trick

    __shared__ float smem_scratch[NUM_WARPS];

    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // ---- Load Q, apply RMSNorm + RoPE ----
    int q_base = q_head_idx * head_dim;
    float q_val = __bfloat162float(q_full[q_base + tid]);

    float q_sq = q_val * q_val;
    float q_sq_sum = warp_reduce_sum(q_sq);
    if (lane_id == 0) smem_scratch[warp_id] = q_sq_sum;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (int i = 0; i < NUM_WARPS; i++) total += smem_scratch[i];
        smem_scratch[0] = 1.0f / sqrtf(total / head_dim + rms_eps);
    }
    __syncthreads();
    float q_rms = smem_scratch[0];
    float q_normed = q_val * q_rms * __bfloat162float(q_norm_weight[tid]);

    __shared__ float smem_q_rope[HEAD_DIM];
    smem_q_rope[tid] = q_normed;
    __syncthreads();

    float q_rot;
    if (tid < half_dim) {
        float cos_val = __bfloat162float(cos_cache[current_pos * head_dim + tid]);
        float sin_val = __bfloat162float(sin_cache[current_pos * head_dim + tid]);
        q_rot = smem_q_rope[tid] * cos_val - smem_q_rope[tid + half_dim] * sin_val;
    } else {
        int pair = tid - half_dim;
        float cos_val = __bfloat162float(cos_cache[current_pos * head_dim + pair]);
        float sin_val = __bfloat162float(sin_cache[current_pos * head_dim + pair]);
        q_rot = smem_q_rope[pair] * sin_val + smem_q_rope[tid] * cos_val;
    }

    // ---- Load K, apply RMSNorm + RoPE ----
    int kv_base = kv_head_idx * head_dim;
    float k_val = __bfloat162float(k_full[kv_base + tid]);

    float k_sq = k_val * k_val;
    float k_sq_sum = warp_reduce_sum(k_sq);
    if (lane_id == 0) smem_scratch[warp_id] = k_sq_sum;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (int i = 0; i < NUM_WARPS; i++) total += smem_scratch[i];
        smem_scratch[0] = 1.0f / sqrtf(total / head_dim + rms_eps);
    }
    __syncthreads();
    float k_rms = smem_scratch[0];
    float k_normed = k_val * k_rms * __bfloat162float(k_norm_weight[tid]);

    __shared__ float smem_k_rope[HEAD_DIM];
    smem_k_rope[tid] = k_normed;
    __syncthreads();

    float k_rot;
    if (tid < half_dim) {
        float cos_val = __bfloat162float(cos_cache[current_pos * head_dim + tid]);
        float sin_val = __bfloat162float(sin_cache[current_pos * head_dim + tid]);
        k_rot = smem_k_rope[tid] * cos_val - smem_k_rope[tid + half_dim] * sin_val;
    } else {
        int pair = tid - half_dim;
        float cos_val = __bfloat162float(cos_cache[current_pos * head_dim + pair]);
        float sin_val = __bfloat162float(sin_cache[current_pos * head_dim + pair]);
        k_rot = smem_k_rope[pair] * sin_val + smem_k_rope[tid] * cos_val;
    }

    // ---- Load V ----
    float v_val = __bfloat162float(v_full[kv_base + tid]);

    // ---- Split 0 only: write current K/V to KV cache ----
    int cache_head_offset = kv_head_idx * max_seq_len * head_dim;
    if (split_id == 0) {
        int cur_off = cache_head_offset + current_pos * head_dim + tid;
        k_cache[cur_off] = __float2bfloat16(k_rot);
        v_cache[cur_off] = __float2bfloat16(v_val);
    }

    // ---- Compute this split's KV range (past tokens only) ----
    int past_seq_len = current_pos;
    int tiles_total = (past_seq_len + BATCHED_BLOCK_N - 1) / BATCHED_BLOCK_N;
    int tiles_per_split = (tiles_total + NUM_KV_SPLITS - 1) / NUM_KV_SPLITS;
    int split_start = split_id * tiles_per_split * BATCHED_BLOCK_N;
    int split_end = min((split_id + 1) * tiles_per_split * BATCHED_BLOCK_N, past_seq_len);

    // ---- Online softmax attention over this split's KV chunk ----
    float acc = 0.0f;
    float m_i = -1e38f;
    float l_i = 0.0f;

    const __nv_bfloat16* k_cache_head = k_cache + cache_head_offset;
    const __nv_bfloat16* v_cache_head = v_cache + cache_head_offset;

    __shared__ float smem_qk[BATCHED_BLOCK_N];

    for (int tile_start = split_start; tile_start < split_end; tile_start += BATCHED_BLOCK_N) {
        int tile_len = min(BATCHED_BLOCK_N, split_end - tile_start);

        for (int pos = 0; pos < tile_len; pos++) {
            int abs_pos = tile_start + pos;
            float k_elem = __bfloat162float(k_cache_head[abs_pos * head_dim + tid]);
            float dot = q_rot * k_elem;
            dot = warp_reduce_sum(dot);
            if (lane_id == 0) {
                smem_scratch[warp_id] = dot;
            }
            __syncthreads();
            if (tid == 0) {
                float score = 0.0f;
                for (int w = 0; w < NUM_WARPS; w++) score += smem_scratch[w];
                smem_qk[pos] = score * qk_scale;
            }
            __syncthreads();
        }

        float tile_max = -INFINITY;
        for (int pos = 0; pos < tile_len; pos++) {
            tile_max = fmaxf(tile_max, smem_qk[pos]);
        }

        float m_new = fmaxf(m_i, tile_max);
        float alpha = exp2f(m_i - m_new);
        acc *= alpha;
        l_i *= alpha;

        for (int pos = 0; pos < tile_len; pos++) {
            float w = exp2f(smem_qk[pos] - m_new);
            float v_elem = __bfloat162float(v_cache_head[(tile_start + pos) * head_dim + tid]);
            acc += w * v_elem;
            l_i += w;
        }
        m_i = m_new;
    }

    // ---- Split 0: handle current token from registers ----
    if (split_id == 0) {
        float dot = q_rot * k_rot;
        dot = warp_reduce_sum(dot);
        if (lane_id == 0) smem_scratch[warp_id] = dot;
        __syncthreads();
        if (tid == 0) {
            float score = 0.0f;
            for (int w = 0; w < NUM_WARPS; w++) score += smem_scratch[w];
            smem_scratch[0] = score * qk_scale;
        }
        __syncthreads();
        float qk_cur = smem_scratch[0];

        float m_new = fmaxf(m_i, qk_cur);
        float alpha = exp2f(m_i - m_new);
        float p_cur = exp2f(qk_cur - m_new);

        acc = acc * alpha + v_val * p_cur;
        l_i = l_i * alpha + p_cur;
        m_i = m_new;
    }

    // ---- Write partial results (FP32, unnormalized) ----
    int partial_base_head = q_head_idx * NUM_KV_SPLITS;
    int partial_out_offset = (partial_base_head + split_id) * head_dim + tid;
    partial_out[partial_out_offset] = acc;

    int scalar_offset = partial_base_head + split_id;
    if (tid == 0) {
        partial_m[scalar_offset] = m_i;
        partial_l[scalar_offset] = l_i;
    }
}

// ============================================================================
// Single-request reduce kernel — merges NUM_KV_SPLITS partials per Q head.
// Grid: (num_qheads); Threads: HEAD_DIM (128).
// ============================================================================
__global__ void attention_decode_reduce_single_kernel(
    const float* __restrict__ partial_out,
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    __nv_bfloat16* __restrict__ output,
    int num_qheads
) {
    constexpr int head_dim = HEAD_DIM;

    int q_head_idx = blockIdx.x;
    int tid = threadIdx.x;

    int base = q_head_idx * NUM_KV_SPLITS;

    float acc = 0.0f;
    float m_global = -INFINITY;
    float l_global = 0.0f;

    #pragma unroll
    for (int s = 0; s < NUM_KV_SPLITS; s++) {
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
    int out_offset = q_head_idx * head_dim + tid;
    output[out_offset] = __float2bfloat16(result);
}

// ============================================================================
// C API
// ============================================================================
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
    int max_seq_len,
    int batch_size,
    float rms_eps,
    cudaStream_t stream
) {
    dim3 grid(num_qheads, NUM_KV_SPLITS, batch_size);
    int threads = head_dim;

    fused_gqa_attention_decode_batched_kernel<<<grid, threads, 0, stream>>>(
        q_batch, k_batch, v_batch,
        q_norm_weight, k_norm_weight,
        cos_cache, sin_cache,
        positions, seq_lens,
        k_cache_ptrs, v_cache_ptrs,
        partial_out, partial_m, partial_l,
        num_qheads, num_kvheads, gqa_ratio, head_dim,
        max_seq_len, rms_eps
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
    cudaStream_t stream
) {
    dim3 grid(num_qheads, batch_size);
    int threads = head_dim;

    attention_decode_reduce_batched_kernel<<<grid, threads, 0, stream>>>(
        partial_out, partial_m, partial_l,
        output, num_qheads, head_dim
    );
    return cudaGetLastError();
}

// Single-request (batch=1) decode attention - CUDA-Graph-safe because
// `decode_meta` lives on device.
cudaError_t fused_gqa_attention_decode(
    const __nv_bfloat16* q_full,
    const __nv_bfloat16* k_full,
    const __nv_bfloat16* v_full,
    const __nv_bfloat16* q_norm_weight,
    const __nv_bfloat16* k_norm_weight,
    const __nv_bfloat16* cos_cache_base,
    const __nv_bfloat16* sin_cache_base,
    const int* decode_meta,
    __nv_bfloat16* k_cache,
    __nv_bfloat16* v_cache,
    float* partial_out,
    float* partial_m,
    float* partial_l,
    int num_qheads,
    int num_kvheads,
    int gqa_ratio,
    int max_seq_len,
    cudaStream_t stream
) {
    dim3 grid(num_qheads, NUM_KV_SPLITS);
    int threads = HEAD_DIM;

    fused_gqa_attention_decode_single_kernel<<<grid, threads, 0, stream>>>(
        q_full, k_full, v_full,
        q_norm_weight, k_norm_weight,
        cos_cache_base, sin_cache_base,
        decode_meta,
        k_cache, v_cache,
        partial_out, partial_m, partial_l,
        num_qheads, num_kvheads, gqa_ratio, max_seq_len
    );
    return cudaGetLastError();
}

cudaError_t attention_decode_reduce(
    const float* partial_out,
    const float* partial_m,
    const float* partial_l,
    __nv_bfloat16* output,
    int num_qheads,
    cudaStream_t stream
) {
    dim3 grid(num_qheads);
    int threads = HEAD_DIM;

    attention_decode_reduce_single_kernel<<<grid, threads, 0, stream>>>(
        partial_out, partial_m, partial_l,
        output, num_qheads
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
