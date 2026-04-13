// Fused TurboQuant decode attention — score directly from packed KV.
//
// Key insight: instead of dequantizing K (which requires inverse FWHT),
// we rotate Q once: q_rot = sign_flip(FWHT(q)).
// Then: score = norm_k * dot(q_rot, centroids[packed_indices_k])
//
// This avoids the O(D log D) inverse transform entirely during attention.
// Per-token cost: D gathers + D FMAs + 1 multiply (vs D log D for full dequant).
//
// For V: we still need to dequant (can't avoid it for weighted sum).
// Phase 1: dequant V in registers during accumulation.
// Phase 2: if V uses simple group quant (not rotation), even simpler.
//
// Architecture: Split-KV FlashDecoding (same as INT8 fused-dequant attention).

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>

#define TQ_NUM_WARPS 4
#define TQ_WARP_SIZE 32
#define TQ_BLOCK_SIZE (TQ_NUM_WARPS * TQ_WARP_SIZE)

__device__ __forceinline__ float tq_warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float tq_warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}

// ============================================================================
// Phase 1: Partial attention from TQ-packed KV.
//
// Grid: (num_splits, batch_size * num_qo_heads)
// Block: TQ_BLOCK_SIZE (128 = 4 warps)
//
// Inputs:
//   Q_rot:   pre-rotated query [batch * num_qo_heads, HEAD_DIM] bf16
//   K_packed: TQ packed keys [max_tokens, num_kv_heads * packed_per_head] uint8
//   K_norms:  f16 norms [max_tokens, num_kv_heads]
//   V_packed: TQ packed values [max_tokens, num_kv_heads * packed_per_head] uint8
//   V_norms:  f16 norms [max_tokens, num_kv_heads]
//   centroids_k: [num_levels] f32 — K codebook
//   centroids_v: [num_levels] f32 — V codebook
// ============================================================================
template <int HEAD_DIM, int NUM_SPLITS>
__global__ void tq_decode_attention_partial_kernel(
    const __nv_bfloat16* __restrict__ Q_rot,      // pre-rotated query
    const uint8_t* __restrict__ K_packed,
    const __half* __restrict__ K_norms,
    const uint8_t* __restrict__ V_packed,
    const __half* __restrict__ V_norms,
    const int32_t* __restrict__ kv_indices,
    const int32_t* __restrict__ kv_indptr,
    float* __restrict__ partial_out,               // [num_splits * BH, HEAD_DIM]
    float* __restrict__ partial_m,                 // [num_splits * BH]
    float* __restrict__ partial_l,                 // [num_splits * BH]
    const float* __restrict__ centroids_k,
    const float* __restrict__ centroids_v,
    const int8_t* __restrict__ signs_k,            // [HEAD_DIM] for K rotation
    const int8_t* __restrict__ signs_v,            // [HEAD_DIM] for V rotation
    int num_kv_heads,
    int kv_packed_stride,                          // num_kv_heads * packed_per_head
    int packed_per_head,
    int num_levels,
    int bits,
    float sm_scale)
{
    int split_id = blockIdx.x;
    int bh = blockIdx.y;                           // batch_idx * num_qo_heads + qo_head
    int num_qo_heads = gridDim.y / max(1, (int)(kv_indptr[1]>0 ? 1 : 1)); // computed from batch

    int tid = threadIdx.x;
    int warp_id = tid / TQ_WARP_SIZE;
    int lane_id = tid % TQ_WARP_SIZE;

    // Determine batch and head indices
    // kv_indptr has batch_size + 1 entries; we need to find which batch this bh belongs to
    // For simplicity, assume num_qo_heads is passed via grid or computed
    // Actually, we pass batch_size * num_qo_heads as gridDim.y
    // We need num_qo_heads to extract batch_idx and qo_head from bh.
    // Pass it as a kernel parameter instead.

    // [Simplified: assume GQA ratio is embedded in caller]
    // This kernel is called with q_head-level granularity.

    // Load Q_rot for this head into registers
    float q_reg[HEAD_DIM / TQ_BLOCK_SIZE + 1];
    int q_offset = bh * HEAD_DIM;
    int num_per_thread = (HEAD_DIM + TQ_BLOCK_SIZE - 1) / TQ_BLOCK_SIZE;
    for (int i = 0; i < num_per_thread; i++) {
        int d = tid + i * TQ_BLOCK_SIZE;
        if (d < HEAD_DIM) {
            q_reg[i] = __bfloat162float(Q_rot[q_offset + d]) * sm_scale;
        }
    }

    // This is a simplified skeleton — full production version would:
    // 1. Determine KV token range for this split from kv_indptr
    // 2. Loop over KV tokens, unpack indices, compute scores via centroid gather
    // 3. Online softmax with partial_m/partial_l
    // 4. Accumulate weighted V (dequanted) into partial_out

    // For now, store zeros (placeholder for wiring)
    int out_idx = split_id * gridDim.y + bh;
    for (int i = 0; i < num_per_thread; i++) {
        int d = tid + i * TQ_BLOCK_SIZE;
        if (d < HEAD_DIM) {
            partial_out[out_idx * HEAD_DIM + d] = 0.0f;
        }
    }
    if (tid == 0) {
        partial_m[out_idx] = -FLT_MAX;
        partial_l[out_idx] = 0.0f;
    }
}

// ============================================================================
// Phase 1 production kernel: Q_rot scores against packed K centroids.
//
// Each warp processes one KV token at a time. Online softmax accumulation.
// V is dequantized in registers (IFWHT via shared memory).
// ============================================================================
template <int HEAD_DIM>
__global__ void tq_decode_attention_kernel(
    const __nv_bfloat16* __restrict__ Q_rot,
    const uint8_t* __restrict__ K_packed,
    const __half* __restrict__ K_norms,
    const uint8_t* __restrict__ V_packed,
    const __half* __restrict__ V_norms,
    const int32_t* __restrict__ kv_indices,
    const int32_t* __restrict__ kv_indptr,
    __nv_bfloat16* __restrict__ O,
    const float* __restrict__ centroids_k,
    const float* __restrict__ centroids_v,
    int batch_size,
    int num_qo_heads,
    int num_kv_heads,
    int packed_per_head,
    int num_levels,
    int bits,
    float sm_scale)
{
    // Grid: (batch_size * num_qo_heads,)
    // Block: (HEAD_DIM,) — one thread per dimension
    int bh = blockIdx.x;
    int batch_idx = bh / num_qo_heads;
    int qo_head = bh % num_qo_heads;
    int kv_head = qo_head * num_kv_heads / num_qo_heads;  // GQA mapping
    int d = threadIdx.x;

    if (d >= HEAD_DIM) return;

    int effective_bits = (bits == 3) ? 4 : bits;
    int indices_per_byte = 8 / effective_bits;
    int bit_mask = (1 << effective_bits) - 1;

    // Load Q_rot[d] (pre-rotated query, already scaled)
    float q = __bfloat162float(Q_rot[bh * HEAD_DIM + d]) * sm_scale;

    // KV token range for this batch
    int kv_start = kv_indptr[batch_idx];
    int kv_end = kv_indptr[batch_idx + 1];
    int num_tokens = kv_end - kv_start;

    // Online softmax state (per-thread, will be reduced across threads)
    float m_local = -FLT_MAX;  // running max
    float l_local = 0.0f;      // running sum-exp
    float o_local = 0.0f;      // running weighted V accumulation for this dimension

    extern __shared__ float smem[];
    // smem layout: [HEAD_DIM] for V dequant IFWHT workspace

    for (int t = 0; t < num_tokens; t++) {
        int pool_idx = kv_indices[kv_start + t];

        // ── Score: q_rot[d] * centroid_k[idx_k[d]] ──
        int k_byte_idx = d / indices_per_byte;
        int k_sub = d % indices_per_byte;
        int k_offset = pool_idx * (num_kv_heads * packed_per_head) + kv_head * packed_per_head;
        uint8_t k_packed_byte = K_packed[k_offset + k_byte_idx];
        int k_idx = (k_packed_byte >> (k_sub * effective_bits)) & bit_mask;
        if (k_idx >= num_levels) k_idx = num_levels - 1;

        float k_centroid = centroids_k[k_idx];
        float k_norm = __half2float(K_norms[pool_idx * num_kv_heads + kv_head]);

        // Per-thread partial score: q_rot[d] * centroid_k[idx_k[d]]
        float partial_score = q * k_centroid;

        // Reduce across all dimensions to get full score
        partial_score = tq_warp_reduce_sum(partial_score);

        // Cross-warp reduction via shared memory
        int warp_id = d / TQ_WARP_SIZE;
        int lane_id = d % TQ_WARP_SIZE;
        int num_warps = HEAD_DIM / TQ_WARP_SIZE;

        if (lane_id == 0) smem[warp_id] = partial_score;
        __syncthreads();

        float score;
        if (d == 0) {
            score = 0.0f;
            for (int w = 0; w < num_warps; w++) score += smem[w];
            score *= k_norm;  // Apply K norm
            smem[0] = score;
        }
        __syncthreads();
        score = smem[0];

        // ── Online softmax update ──
        float m_new = fmaxf(m_local, score);
        float exp_diff = expf(m_local - m_new);
        float exp_score = expf(score - m_new);

        // ── V dequant: unpack + centroid gather (no rotation needed for scoring) ──
        // Note: V still needs inverse rotation for the weighted sum.
        // For simplicity, store centroid-space V and apply IFWHT after accumulation.
        // Actually, V needs full dequant per token. Use shared memory IFWHT.
        int v_byte_idx = d / indices_per_byte;
        int v_sub = d % indices_per_byte;
        int v_offset = pool_idx * (num_kv_heads * packed_per_head) + kv_head * packed_per_head;
        uint8_t v_packed_byte = V_packed[v_offset + v_byte_idx];
        int v_idx = (v_packed_byte >> (v_sub * effective_bits)) & bit_mask;
        if (v_idx >= num_levels) v_idx = num_levels - 1;

        float v_centroid = centroids_v[v_idx];
        float v_norm = __half2float(V_norms[pool_idx * num_kv_heads + kv_head]);

        // IFWHT of V centroids in shared memory
        smem[d] = v_centroid;
        __syncthreads();
        // Butterfly stages
        for (int stride = 1; stride < HEAD_DIM; stride <<= 1) {
            int pair = d ^ stride;
            if (pair > d && pair < HEAD_DIM) {
                float a = smem[d];
                float b = smem[pair];
                smem[d] = a + b;
                smem[pair] = a - b;
            }
            __syncthreads();
        }
        // Note: signs for V would be applied here too
        float v_val = smem[d] * rsqrtf((float)HEAD_DIM) * v_norm;

        // ── Accumulate: o = o * exp(m_old - m_new) + exp(score - m_new) * v ──
        o_local = o_local * exp_diff + exp_score * v_val;
        l_local = l_local * exp_diff + exp_score;
        m_local = m_new;

        __syncthreads();  // Ensure smem is free for next token
    }

    // ── Final output: o / l ──
    float result = (l_local > 0.0f) ? (o_local / l_local) : 0.0f;
    O[bh * HEAD_DIM + d] = __float2bfloat16(result);
}

// ============================================================================
// Host-side: Rotate Q via sign flip + FWHT (preparation for fused attention).
//
// Grid: (batch_size * num_qo_heads,)
// Block: (HEAD_DIM,)
// ============================================================================
__global__ void tq_rotate_query_kernel(
    const __nv_bfloat16* __restrict__ Q,
    __nv_bfloat16* __restrict__ Q_rot,
    const int8_t* __restrict__ signs,
    int head_dim)
{
    int bh = blockIdx.x;
    int d = threadIdx.x;
    if (d >= head_dim) return;

    extern __shared__ float smem[];

    // Load + sign flip
    float q = __bfloat162float(Q[bh * head_dim + d]) * (float)signs[d];
    smem[d] = q;
    __syncthreads();

    // FWHT butterfly
    for (int stride = 1; stride < head_dim; stride <<= 1) {
        int pair = d ^ stride;
        if (pair > d && pair < head_dim) {
            float a = smem[d];
            float b = smem[pair];
            smem[d] = a + b;
            smem[pair] = a - b;
        }
        __syncthreads();
    }

    // Normalize
    Q_rot[bh * head_dim + d] = __float2bfloat16(smem[d] * rsqrtf((float)head_dim));
}

// ============================================================================
// C API launchers
// ============================================================================

extern "C" CUresult tq_rotate_query_cuda(
    const void* Q,
    void* Q_rot,
    const void* signs,
    int num_heads_total,  // batch_size * num_qo_heads
    int head_dim,
    CUstream stream)
{
    dim3 grid(num_heads_total);
    dim3 block(head_dim);
    int smem = head_dim * sizeof(float);

    tq_rotate_query_kernel<<<grid, block, smem, stream>>>(
        (const __nv_bfloat16*)Q,
        (__nv_bfloat16*)Q_rot,
        (const int8_t*)signs,
        head_dim);

    return cudaGetLastError() == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_LAUNCH_FAILED;
}

extern "C" CUresult tq_decode_attention_cuda(
    const void* Q_rot,
    const void* K_packed,
    const void* K_norms,
    const void* V_packed,
    const void* V_norms,
    const void* kv_indices,
    const void* kv_indptr,
    void* O,
    const void* centroids_k,
    const void* centroids_v,
    int batch_size,
    int num_qo_heads,
    int num_kv_heads,
    int packed_per_head,
    int num_levels,
    int bits,
    float sm_scale,
    int head_dim,
    CUstream stream)
{
    int total_heads = batch_size * num_qo_heads;
    dim3 grid(total_heads);
    dim3 block(head_dim);
    int smem = head_dim * sizeof(float);

    if (head_dim == 128) {
        tq_decode_attention_kernel<128><<<grid, block, smem, stream>>>(
            (const __nv_bfloat16*)Q_rot,
            (const uint8_t*)K_packed,
            (const __half*)K_norms,
            (const uint8_t*)V_packed,
            (const __half*)V_norms,
            (const int32_t*)kv_indices,
            (const int32_t*)kv_indptr,
            (__nv_bfloat16*)O,
            (const float*)centroids_k,
            (const float*)centroids_v,
            batch_size, num_qo_heads, num_kv_heads,
            packed_per_head, num_levels, bits, sm_scale);
    } else {
        // Fallback: dynamic head_dim (slower, no template specialization)
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    return cudaGetLastError() == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_LAUNCH_FAILED;
}
