// Fused-dequant decode attention — Split-KV + vectorized loads.
//
// FlashDecoding-style: multiple blocks per query head, each processing a chunk
// of KV tokens. Phase 1 computes partials, Phase 2 merges via log-sum-exp.
//
// Optimizations:
// 1. Split-KV: N blocks per head → saturate GPU at low batch sizes
// 2. Vectorized packed-value loads for K/V bytes
// 3. Warp-level QK reduction via shuffle (no __syncthreads per token)
// 4. Cross-warp merge only once at end of block

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <cstdint>
#include <cfloat>

#define NUM_WARPS 4
#define WARP_SIZE 32
#define BLOCK_SIZE (NUM_WARPS * WARP_SIZE)

// Tokens per shared memory tile (loaded via cp.async pipeline)
#define TILE_TOKENS 16

namespace {
constexpr int kQuantPageSize = TILE_TOKENS;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}


// ============================================================================
// Phase 2: Merge partial results across splits.
//
// Grid: (total_q_heads,)
// Block: (HEAD_DIM,) — each thread handles 1 dimension
// ============================================================================
template <int HEAD_DIM>
__global__ void decode_attention_merge_kernel(
    const float* __restrict__ partial_out,
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    __nv_bfloat16* __restrict__ O,
    int total_q_heads,
    int num_splits)
{
    int q_idx = blockIdx.x;
    int d = threadIdx.x;
    if (q_idx >= total_q_heads || d >= HEAD_DIM) return;

    float final_m = -FLT_MAX;
    float final_l = 0.0f;
    float final_o = 0.0f;

    for (int s = 0; s < num_splits; s++) {
        int idx = s * total_q_heads + q_idx;
        float m_s = partial_m[idx];
        float l_s = partial_l[idx];
        float o_s = partial_out[idx * HEAD_DIM + d];

        if (l_s == 0.0f) continue;

        float m_new = fmaxf(final_m, m_s);
        float s_prev = final_l * __expf(final_m - m_new);
        float s_cur  = l_s * __expf(m_s - m_new);
        float l_new  = s_prev + s_cur;

        final_o = (l_new > 0.0f) ? (final_o * s_prev + o_s * s_cur) / l_new : 0.0f;
        final_m = m_new;
        final_l = l_new;
    }

    O[q_idx * HEAD_DIM + d] = __float2bfloat16(final_o);
}

// ============================================================================

// ============================================================================
// C API
// ============================================================================
extern "C" {

// Workspace size for partial results.
// Returns bytes needed for partial_out + partial_m + partial_l.
size_t decode_attention_int8_workspace_bytes(
    int batch_size, int num_qo_heads, int head_dim, int num_splits)
{
    size_t total_q = (size_t)batch_size * num_qo_heads;
    size_t out_bytes = (size_t)num_splits * total_q * head_dim * sizeof(float);
    size_t m_bytes   = (size_t)num_splits * total_q * sizeof(float);
    size_t l_bytes   = (size_t)num_splits * total_q * sizeof(float);
    return out_bytes + m_bytes + l_bytes;
}

static int choose_decode_num_splits(
    int batch_size,
    int num_qo_heads,
    int head_dim,
    int total_q_heads,
    size_t workspace_bytes)
{
    if (total_q_heads <= 0 || workspace_bytes == 0) return 1;

    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) return 1;

    cudaDeviceProp props;
    err = cudaGetDeviceProperties(&props, device);
    if (err != cudaSuccess || props.multiProcessorCount <= 0) return 1;

    // 32 blocks/SM saturates the kMaxSplits=32 cap on L4 (SM89, 58 SMs) for
    // Qwen3.5-4B. num_splits=8 (old kTargetBlocksPerSm=4) left us at ~14% warp
    // occupancy and compute-bound on softmax/reduce; splits=16 recovered
    // ~11% ITL at 25k; splits=32 aims for the workspace ceiling. Both INT8
    // and FP8 variants pre-allocate their workspace at num_splits=32 in
    // paged_kv.rs, so 32 is the hard ceiling we must not exceed.
    constexpr int kTargetBlocksPerSm = 32;
    constexpr int kMaxSplits = 32;

    int target_blocks = props.multiProcessorCount * kTargetBlocksPerSm;
    int desired_splits = (target_blocks + total_q_heads - 1) / total_q_heads;
    if (desired_splits < 1) desired_splits = 1;
    if (desired_splits > kMaxSplits) desired_splits = kMaxSplits;

    size_t bytes_per_split = decode_attention_int8_workspace_bytes(
        batch_size, num_qo_heads, head_dim, 1);
    if (bytes_per_split == 0) return 1;

    int max_splits_by_workspace = (int)(workspace_bytes / bytes_per_split);
    if (max_splits_by_workspace < 1) return 1;
    if (max_splits_by_workspace > kMaxSplits) max_splits_by_workspace = kMaxSplits;

    return (desired_splits < max_splits_by_workspace) ? desired_splits : max_splits_by_workspace;
}


// FP8 E4M3 fused-dequant decode attention (same split-KV, no scales).
// ============================================================================
// KIVI per-channel K partial kernel: same as decode_attention_fp8_partial_kernel
// but reads K scale from a `[num_kv_heads, head_dim]` static table instead of
// per-(token, head) `K_scales`. V keeps per-(token, head) scales (KIVI's
// asymmetric scheme). The per-channel K scale lookup uses each thread's
// own dim block — pre-loaded into registers once at kernel entry, then
// reused across all tokens (no per-token scale load, faster than the
// per-token variant).
// ============================================================================
extern "C++" {
template <int HEAD_DIM>
__global__ void decode_attention_fp8_per_channel_k_partial_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_fp8_e4m3* __restrict__ K_data,
    const __nv_fp8_e4m3* __restrict__ V_data,
    const float* __restrict__ K_static_scales,   // [num_kv_heads, HEAD_DIM]
    const float* __restrict__ V_scales,          // per-(row, head)
    const int32_t* __restrict__ kv_indices,
    const int32_t* __restrict__ kv_meta,
    float* __restrict__ partial_out,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    int batch_size,
    int num_qo_heads,
    int num_kv_heads,
    int kv_dim,
    float sm_scale,
    int num_splits)
{
    constexpr int EPT = HEAD_DIM / WARP_SIZE;

    int split_idx = blockIdx.x;
    int total_q_idx = blockIdx.y;
    int req_idx = total_q_idx / num_qo_heads;
    int q_head = total_q_idx % num_qo_heads;

    if (req_idx >= batch_size) return;

    int gqa_ratio = num_qo_heads / num_kv_heads;
    int kv_head = q_head / gqa_ratio;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    const int32_t* page_indptr = kv_meta;
    const int32_t* last_page_len = kv_meta + (batch_size + 1);
    int page_start_global = page_indptr[req_idx];
    int page_end_global = page_indptr[req_idx + 1];
    int total_pages = page_end_global - page_start_global;
    int total_tokens = total_pages == 0
        ? 0
        : (total_pages - 1) * kQuantPageSize + last_page_len[req_idx];
    (void)total_tokens;
    int page_chunk_size = (total_pages + num_splits - 1) / num_splits;
    int my_page_start = split_idx * page_chunk_size;
    int my_page_end = min(my_page_start + page_chunk_size, total_pages);

    if (my_page_start >= total_pages) {
        int out_idx = split_idx * (batch_size * num_qo_heads) + total_q_idx;
        if (threadIdx.x == 0) {
            partial_m[out_idx] = -FLT_MAX;
            partial_l[out_idx] = 0.0f;
        }
        if (threadIdx.x < HEAD_DIM)
            partial_out[out_idx * HEAD_DIM + threadIdx.x] = 0.0f;
        return;
    }

    // Pre-load Q values + per-channel K scales for this warp's dims into
    // registers. K scales are loop-invariant across tokens (per-channel,
    // not per-token), so this is read once per kernel invocation per thread.
    float q_reg[EPT];
    float k_scale_reg[EPT];
    int k_scale_base = kv_head * HEAD_DIM;
    #pragma unroll
    for (int i = 0; i < EPT; i++) {
        int d = lane_id * EPT + i;
        q_reg[i] = __bfloat162float(Q[total_q_idx * HEAD_DIM + d]) * sm_scale;
        k_scale_reg[i] = K_static_scales[k_scale_base + d];
    }

    float o_reg[EPT];
    #pragma unroll
    for (int i = 0; i < EPT; i++) o_reg[i] = 0.0f;
    float m_local = -FLT_MAX;
    float l_local = 0.0f;

    for (int page_local_idx = 0; page_local_idx < my_page_end - my_page_start; page_local_idx++) {
        int global_page = my_page_start + page_local_idx;
        int page_idx = kv_indices[page_start_global + global_page];
        int row_base = page_idx * kQuantPageSize;
        int page_tokens = (global_page == total_pages - 1) ? last_page_len[req_idx] : kQuantPageSize;

        for (int t = warp_id; t < page_tokens; t += NUM_WARPS) {
            int row_idx = row_base + t;
            int base = row_idx * kv_dim + kv_head * HEAD_DIM;
            int scale_offset = row_idx * num_kv_heads + kv_head;
            float v_scale = V_scales[scale_offset];

            float qk = 0.0f;
            #pragma unroll
            for (int i = 0; i < EPT; i += 4) {
                int d = lane_id * EPT + i;
                __nv_fp8x4_e4m3 packed =
                    *reinterpret_cast<const __nv_fp8x4_e4m3*>(K_data + base + d);
                float4 k_vals = static_cast<float4>(packed);
                qk += q_reg[i + 0] * k_vals.x * k_scale_reg[i + 0];
                qk += q_reg[i + 1] * k_vals.y * k_scale_reg[i + 1];
                qk += q_reg[i + 2] * k_vals.z * k_scale_reg[i + 2];
                qk += q_reg[i + 3] * k_vals.w * k_scale_reg[i + 3];
            }
            qk = warp_reduce_sum(qk);

            float m_new = fmaxf(m_local, qk);
            float exp_diff = __expf(m_local - m_new);
            float exp_qk = __expf(qk - m_new);
            float l_new = l_local * exp_diff + exp_qk;

            #pragma unroll
            for (int i = 0; i < EPT; i += 4) {
                int d = lane_id * EPT + i;
                __nv_fp8x4_e4m3 packed =
                    *reinterpret_cast<const __nv_fp8x4_e4m3*>(V_data + base + d);
                float4 v_vals = static_cast<float4>(packed);
                o_reg[i + 0] = o_reg[i + 0] * exp_diff + exp_qk * v_vals.x * v_scale;
                o_reg[i + 1] = o_reg[i + 1] * exp_diff + exp_qk * v_vals.y * v_scale;
                o_reg[i + 2] = o_reg[i + 2] * exp_diff + exp_qk * v_vals.z * v_scale;
                o_reg[i + 3] = o_reg[i + 3] * exp_diff + exp_qk * v_vals.w * v_scale;
            }
            m_local = m_new;
            l_local = l_new;
        }
    }

    // Cross-warp merge (identical to per-token-scale variant)
    __shared__ float smem_m[NUM_WARPS];
    __shared__ float smem_l[NUM_WARPS];
    __shared__ float smem_o[NUM_WARPS * HEAD_DIM];

    if (lane_id == 0) {
        smem_m[warp_id] = m_local;
        smem_l[warp_id] = l_local;
    }
    #pragma unroll
    for (int i = 0; i < EPT; i++)
        smem_o[warp_id * HEAD_DIM + lane_id * EPT + i] = o_reg[i];
    __syncthreads();

    if (warp_id == 0) {
        float final_m = smem_m[0], final_l = smem_l[0];
        float final_o[EPT];
        #pragma unroll
        for (int i = 0; i < EPT; i++) final_o[i] = smem_o[lane_id * EPT + i];

        for (int w = 1; w < NUM_WARPS; w++) {
            float m_w = smem_m[w];
            float l_w = smem_l[w];
            float new_m = fmaxf(final_m, m_w);
            float a = __expf(final_m - new_m);
            float b = __expf(m_w - new_m);
            #pragma unroll
            for (int i = 0; i < EPT; i++)
                final_o[i] = final_o[i] * a + smem_o[w * HEAD_DIM + lane_id * EPT + i] * b;
            final_l = final_l * a + l_w * b;
            final_m = new_m;
        }

        int out_idx = split_idx * (batch_size * num_qo_heads) + total_q_idx;
        if (lane_id == 0) {
            partial_m[out_idx] = final_m;
            partial_l[out_idx] = final_l;
        }
        // CRITICAL: write the *normalized* per-split average (final_o /
        // final_l), matching `decode_attention_fp8_partial_kernel` and the
        // merge-kernel contract at line ~295: `o_s * s_cur / l_new` with
        // `s_cur = l_s * exp(...)` only balances if o_s is pre-normalized.
        // Unnormalized writes produce O(l_s)-scale-off attention output and
        // were the actual root cause of the 2026-05-26 KIVI bit-identical
        // failure pattern (`fp8 mean_match=0.0156` unchanged regardless of
        // K calibration quality).
        float inv_final_l = (final_l > 0.0f) ? (1.0f / final_l) : 0.0f;
        #pragma unroll
        for (int i = 0; i < EPT; i++)
            partial_out[out_idx * HEAD_DIM + lane_id * EPT + i] = final_o[i] * inv_final_l;
    }
}
}  // extern "C++" — KIVI per-channel K template kernel


// KIVI per-channel K decode attention: same shape as decode_attention_fp8_cuda
// but consumes a `[num_kv_heads, head_dim]` static K scale table instead of
// per-(row, head) K scales. V keeps per-(row, head) scales.
cudaError_t decode_attention_fp8_per_channel_k_cuda(
    const __nv_bfloat16* Q,
    const __nv_fp8_e4m3* K_data,
    const __nv_fp8_e4m3* V_data,
    const float* K_static_scales,
    const float* V_scales,
    const int32_t* kv_indices,
    const int32_t* kv_indptr,
    __nv_bfloat16* O,
    int batch_size,
    int num_qo_heads,
    int num_kv_heads,
    int head_dim,
    int kv_dim,
    float sm_scale,
    cudaStream_t stream,
    void* workspace,
    size_t workspace_bytes)
{
    if (batch_size <= 0) return cudaSuccess;

    int total_q_heads = batch_size * num_qo_heads;
    int num_splits = choose_decode_num_splits(
        batch_size, num_qo_heads, head_dim, total_q_heads, workspace_bytes);
    size_t needed = decode_attention_int8_workspace_bytes(
        batch_size, num_qo_heads, head_dim, num_splits);
    if (workspace == nullptr || workspace_bytes < needed) {
        return cudaErrorInvalidValue;
    }

    float* ws_float = reinterpret_cast<float*>(workspace);
    size_t total_q = (size_t)total_q_heads;
    float* p_out = ws_float;
    float* p_m   = ws_float + num_splits * total_q * head_dim;
    float* p_l   = p_m + num_splits * total_q;

    // Phase 1
    {
        dim3 grid(num_splits, total_q_heads);
        dim3 block(BLOCK_SIZE);
        if (head_dim == 128) {
            decode_attention_fp8_per_channel_k_partial_kernel<128><<<grid, block, 0, stream>>>(
                Q, K_data, V_data, K_static_scales, V_scales, kv_indices, kv_indptr,
                p_out, p_m, p_l,
                batch_size, num_qo_heads, num_kv_heads, kv_dim, sm_scale, num_splits);
        } else if (head_dim == 256) {
            decode_attention_fp8_per_channel_k_partial_kernel<256><<<grid, block, 0, stream>>>(
                Q, K_data, V_data, K_static_scales, V_scales, kv_indices, kv_indptr,
                p_out, p_m, p_l,
                batch_size, num_qo_heads, num_kv_heads, kv_dim, sm_scale, num_splits);
        } else {
            return cudaErrorInvalidValue;
        }
    }

    // Phase 2: merge (shared with INT8 / per-token FP8)
    {
        dim3 grid(total_q_heads);
        dim3 block(head_dim);
        if (head_dim == 128) {
            decode_attention_merge_kernel<128><<<grid, block, 0, stream>>>(
                p_out, p_m, p_l, O, total_q_heads, num_splits);
        } else if (head_dim == 256) {
            decode_attention_merge_kernel<256><<<grid, block, 0, stream>>>(
                p_out, p_m, p_l, O, total_q_heads, num_splits);
        }
    }

    return cudaGetLastError();
}

// ============================================================================
// INT8 KIVI per-channel K decode attention.
//
// Mirrors `decode_attention_int8_partial_kernel` (cp.async pipelined) but
// reads K scale from a `[num_kv_heads, head_dim]` static table preloaded
// into registers, instead of per-(token, head) `K_scales`. V keeps its
// per-(token, head) scales (KIVI's asymmetric design). See FP8 sibling
// `decode_attention_fp8_per_channel_k_partial_kernel` (line ~606) for the
// algorithm; the only differences here are int8/float dequant and the
// cp.async pipelining (which the INT8 sibling uses but the FP8 sibling
// does not).
// ============================================================================
extern "C++" {
template <int HEAD_DIM>
__global__ void decode_attention_int8_per_channel_k_partial_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const int8_t* __restrict__ K_data,
    const int8_t* __restrict__ V_data,
    const float* __restrict__ K_static_scales,  // [num_kv_heads, HEAD_DIM]
    const float* __restrict__ V_scales,          // per-(row, head)
    const int32_t* __restrict__ kv_indices,
    const int32_t* __restrict__ kv_meta,
    float* __restrict__ partial_out,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    int batch_size,
    int num_qo_heads,
    int num_kv_heads,
    int kv_dim,
    float sm_scale,
    int num_splits)
{
    constexpr int EPT = HEAD_DIM / WARP_SIZE;

    int split_idx = blockIdx.x;
    int total_q_idx = blockIdx.y;
    int req_idx = total_q_idx / num_qo_heads;
    int q_head  = total_q_idx % num_qo_heads;

    if (req_idx >= batch_size) return;

    int gqa_ratio = num_qo_heads / num_kv_heads;
    int kv_head = q_head / gqa_ratio;

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    const int32_t* page_indptr = kv_meta;
    const int32_t* last_page_len = kv_meta + (batch_size + 1);
    int page_start_global = page_indptr[req_idx];
    int page_end_global = page_indptr[req_idx + 1];
    int total_pages = page_end_global - page_start_global;
    int page_chunk_size = (total_pages + num_splits - 1) / num_splits;
    int my_page_start = split_idx * page_chunk_size;
    int my_page_end = min(my_page_start + page_chunk_size, total_pages);

    if (my_page_start >= total_pages) {
        int out_idx = split_idx * (batch_size * num_qo_heads) + total_q_idx;
        if (threadIdx.x == 0) {
            partial_m[out_idx] = -FLT_MAX;
            partial_l[out_idx] = 0.0f;
        }
        if (threadIdx.x < HEAD_DIM) {
            partial_out[out_idx * HEAD_DIM + threadIdx.x] = 0.0f;
        }
        return;
    }

    // Pre-load Q and per-channel K scales into registers (KIVI: K scales are
    // loop-invariant across tokens once per-channel).
    float q_reg[EPT];
    float k_scale_reg[EPT];
    int k_scale_base = kv_head * HEAD_DIM;
    #pragma unroll
    for (int i = 0; i < EPT; i++) {
        int d = lane_id * EPT + i;
        q_reg[i] = __bfloat162float(Q[total_q_idx * HEAD_DIM + d]) * sm_scale;
        k_scale_reg[i] = K_static_scales[k_scale_base + d];
    }

    float o_reg[EPT];
    #pragma unroll
    for (int i = 0; i < EPT; i++) o_reg[i] = 0.0f;
    float m_local = -FLT_MAX;
    float l_local = 0.0f;

    __shared__ int8_t smem_k[2][TILE_TOKENS][HEAD_DIM];
    __shared__ int8_t smem_v[2][TILE_TOKENS][HEAD_DIM];
    __shared__ float smem_v_scales[2][TILE_TOKENS];

    __shared__ float smem_m[NUM_WARPS];
    __shared__ float smem_l[NUM_WARPS];
    __shared__ float smem_o[NUM_WARPS * HEAD_DIM];

    const int d_base = lane_id * EPT;
    auto preload_page = [&](int stage, int page_local_idx) {
        int global_page = my_page_start + page_local_idx;
        int page_idx = kv_indices[page_start_global + global_page];
        int row_base = page_idx * kQuantPageSize;
        int page_tokens = (global_page == total_pages - 1) ? last_page_len[req_idx] : kQuantPageSize;
        for (int t = warp_id; t < page_tokens; t += NUM_WARPS) {
            int row_idx = row_base + t;
            int base = row_idx * kv_dim + kv_head * HEAD_DIM;
            int scale_off = row_idx * num_kv_heads + kv_head;

            __pipeline_memcpy_async(
                &smem_k[stage][t][d_base],
                &K_data[base + d_base],
                sizeof(int8_t) * EPT);
            __pipeline_memcpy_async(
                &smem_v[stage][t][d_base],
                &V_data[base + d_base],
                sizeof(int8_t) * EPT);
            // K_static_scales is per-channel (already in registers), so we
            // only need to async-load V scale here.
            if (lane_id == 0) {
                __pipeline_memcpy_async(&smem_v_scales[stage][t], &V_scales[scale_off], sizeof(float));
            }
        }
        __pipeline_commit();
    };

    preload_page(0, 0);

    for (int page_local_idx = 0; page_local_idx < my_page_end - my_page_start; page_local_idx++) {
        int stage = page_local_idx & 1;
        int global_page = my_page_start + page_local_idx;
        int page_tokens = (global_page == total_pages - 1) ? last_page_len[req_idx] : kQuantPageSize;

        __pipeline_wait_prior(0);
        __syncthreads();

        int next_page_local_idx = page_local_idx + 1;
        if (next_page_local_idx < my_page_end - my_page_start) {
            preload_page(next_page_local_idx & 1, next_page_local_idx);
        }

        for (int t = warp_id; t < page_tokens; t += NUM_WARPS) {
            float qk = 0.0f;
            #pragma unroll
            for (int i = 0; i < EPT; i++) {
                // KIVI: K dequant uses the per-channel scale (loop-invariant).
                float k_val = static_cast<float>(smem_k[stage][t][d_base + i]) * k_scale_reg[i];
                qk += q_reg[i] * k_val;
            }
            qk = warp_reduce_sum(qk);

            float m_new = fmaxf(m_local, qk);
            float exp_diff = __expf(m_local - m_new);
            float exp_qk = __expf(qk - m_new);
            float l_new = l_local * exp_diff + exp_qk;

            float v_scale = smem_v_scales[stage][t];
            #pragma unroll
            for (int i = 0; i < EPT; i++) {
                float v_val = static_cast<float>(smem_v[stage][t][d_base + i]) * v_scale;
                o_reg[i] = o_reg[i] * exp_diff + exp_qk * v_val;
            }

            m_local = m_new;
            l_local = l_new;
        }

        __syncthreads();
    }

    if (lane_id == 0) {
        smem_m[warp_id] = m_local;
        smem_l[warp_id] = l_local;
    }
    #pragma unroll
    for (int i = 0; i < EPT; i++) {
        smem_o[warp_id * HEAD_DIM + lane_id * EPT + i] = o_reg[i];
    }
    __syncthreads();

    if (warp_id == 0) {
        float final_m = smem_m[0];
        float final_l = smem_l[0];
        float final_o[EPT];
        #pragma unroll
        for (int i = 0; i < EPT; i++) {
            final_o[i] = smem_o[lane_id * EPT + i];
        }

        #pragma unroll
        for (int w = 1; w < NUM_WARPS; w++) {
            float m_w = smem_m[w];
            float l_w = smem_l[w];
            if (l_w == 0.0f) continue;

            float m_new = fmaxf(final_m, m_w);
            float scale_prev = __expf(final_m - m_new);
            float scale_w    = __expf(m_w - m_new);

            #pragma unroll
            for (int i = 0; i < EPT; i++) {
                float o_w = smem_o[w * HEAD_DIM + lane_id * EPT + i];
                final_o[i] = final_o[i] * scale_prev + o_w * scale_w;
            }
            final_l = final_l * scale_prev + l_w * scale_w;
            final_m = m_new;
        }

        int out_idx = split_idx * (batch_size * num_qo_heads) + total_q_idx;
        if (lane_id == 0) {
            partial_m[out_idx] = final_m;
            partial_l[out_idx] = final_l;
        }
        // Write normalized partial (matches Phase-2 merge contract — see FP8
        // sibling's comment at line ~766 for the unnormalized-write incident).
        float inv_final_l = (final_l > 0.0f) ? (1.0f / final_l) : 0.0f;
        #pragma unroll
        for (int i = 0; i < EPT; i++) {
            int d = lane_id * EPT + i;
            partial_out[out_idx * HEAD_DIM + d] = final_o[i] * inv_final_l;
        }
    }
}
}  // extern "C++" — INT8 KIVI per-channel K template kernel

cudaError_t decode_attention_int8_per_channel_k_cuda(
    const __nv_bfloat16* Q,
    const int8_t* K_data,
    const int8_t* V_data,
    const float* K_static_scales,
    const float* V_scales,
    const int32_t* kv_indices,
    const int32_t* kv_indptr,
    __nv_bfloat16* O,
    int batch_size,
    int num_qo_heads,
    int num_kv_heads,
    int head_dim,
    int kv_dim,
    float sm_scale,
    cudaStream_t stream,
    void* workspace,
    size_t workspace_bytes)
{
    if (batch_size <= 0) return cudaSuccess;

    int total_q_heads = batch_size * num_qo_heads;
    int num_splits = choose_decode_num_splits(
        batch_size, num_qo_heads, head_dim, total_q_heads, workspace_bytes);
    size_t needed = decode_attention_int8_workspace_bytes(
        batch_size, num_qo_heads, head_dim, num_splits);
    if (workspace == nullptr || workspace_bytes < needed) {
        return cudaErrorInvalidValue;
    }

    float* ws_float = reinterpret_cast<float*>(workspace);
    size_t total_q = (size_t)total_q_heads;
    float* p_out = ws_float;
    float* p_m   = ws_float + num_splits * total_q * head_dim;
    float* p_l   = p_m + num_splits * total_q;

    // Phase 1
    {
        dim3 grid(num_splits, total_q_heads);
        dim3 block(BLOCK_SIZE);
        if (head_dim == 128) {
            decode_attention_int8_per_channel_k_partial_kernel<128><<<grid, block, 0, stream>>>(
                Q, K_data, V_data, K_static_scales, V_scales, kv_indices, kv_indptr,
                p_out, p_m, p_l,
                batch_size, num_qo_heads, num_kv_heads, kv_dim, sm_scale, num_splits);
        } else if (head_dim == 256) {
            decode_attention_int8_per_channel_k_partial_kernel<256><<<grid, block, 0, stream>>>(
                Q, K_data, V_data, K_static_scales, V_scales, kv_indices, kv_indptr,
                p_out, p_m, p_l,
                batch_size, num_qo_heads, num_kv_heads, kv_dim, sm_scale, num_splits);
        } else {
            return cudaErrorInvalidValue;
        }
    }

    // Phase 2: merge (shared with FP8 / per-token INT8)
    {
        dim3 grid(total_q_heads);
        dim3 block(head_dim);
        if (head_dim == 128) {
            decode_attention_merge_kernel<128><<<grid, block, 0, stream>>>(
                p_out, p_m, p_l, O, total_q_heads, num_splits);
        } else if (head_dim == 256) {
            decode_attention_merge_kernel<256><<<grid, block, 0, stream>>>(
                p_out, p_m, p_l, O, total_q_heads, num_splits);
        }
    }

    return cudaGetLastError();
}

}  // extern "C"
