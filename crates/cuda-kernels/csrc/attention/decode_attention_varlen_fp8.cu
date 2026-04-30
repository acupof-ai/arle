// Variable-length Q + paged quantized KV attention for Qwen3 HD128.
//
// This is the mixed decode+prefill attention path for FP8 E4M3 and INT8 KV
// pools. It uses FlashDecoding-style split-KV: phase 1 computes one partial
// softmax accumulator per (q_token, q_head, split), and phase 2 merges those
// partials over the split axis.
//
// Pool layout is NHD durable storage:
//   data:   [max_pages, page_size, num_kv_heads, HEAD_DIM]
//   scales: [max_pages * page_size, num_kv_heads] when present
//
// The FP8 pool currently stores scaled E4M3 values. Therefore FP8 may pass
// K/V scale pointers. Null scale pointers are still accepted for scale-free
// fixtures; INT8 always requires scales and is selected by the `int8_kv` C API
// flag.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>

#define VLF8_NUM_WARPS 4
#define VLF8_WARP_SIZE 32
#define VLF8_BLOCK_SIZE (VLF8_NUM_WARPS * VLF8_WARP_SIZE)

namespace {
constexpr int kPageSize = 16;
constexpr int kMaxSplits = 16;
constexpr int kSplitTokens = 4096;
}

__device__ __forceinline__ float vlf8_warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

static int choose_varlen_num_splits(int max_kv_len) {
    int splits = (max_kv_len + kSplitTokens - 1) / kSplitTokens;
    if (splits < 1) splits = 1;
    if (splits > kMaxSplits) splits = kMaxSplits;
    return splits;
}

extern "C" size_t decode_attention_varlen_fp8_workspace_bytes(
    int total_q_tokens,
    int num_q_heads,
    int head_dim,
    int num_splits)
{
    if (total_q_tokens <= 0 || num_q_heads <= 0 || head_dim <= 0 || num_splits <= 0) {
        return 0;
    }
    size_t total_q_heads = (size_t)total_q_tokens * (size_t)num_q_heads;
    size_t out_bytes = (size_t)num_splits * total_q_heads * (size_t)head_dim * sizeof(float);
    size_t m_bytes = (size_t)num_splits * total_q_heads * sizeof(float);
    size_t l_bytes = (size_t)num_splits * total_q_heads * sizeof(float);
    return out_bytes + m_bytes + l_bytes;
}

template <bool INT8_KV>
__device__ __forceinline__ float load_quantized_value(
    const void* __restrict__ data,
    size_t offset,
    const float* __restrict__ scales,
    int scale_offset)
{
    if (INT8_KV) {
        float scale = scales ? scales[scale_offset] : 1.0f;
        return static_cast<float>(reinterpret_cast<const int8_t*>(data)[offset]) * scale;
    }
    float value = static_cast<float>(reinterpret_cast<const __nv_fp8_e4m3*>(data)[offset]);
    return scales ? value * scales[scale_offset] : value;
}

template <int HEAD_DIM, bool CAUSAL, bool INT8_KV>
__global__ void decode_attention_varlen_quantized_partial_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const int* __restrict__ qo_indptr,
    const void* __restrict__ K_pool,
    const void* __restrict__ V_pool,
    const float* __restrict__ K_scales,
    const float* __restrict__ V_scales,
    const int* __restrict__ kv_indptr,
    const int* __restrict__ kv_indices,
    const int* __restrict__ last_page_len,
    float* __restrict__ partial_out,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    int num_q_heads,
    int num_kv_heads,
    int batch_size,
    float sm_scale,
    int num_splits)
{
    constexpr int EPT = HEAD_DIM / VLF8_WARP_SIZE;

    int q_token = blockIdx.x;
    int q_head = blockIdx.y;
    int split_idx = blockIdx.z;
    if (q_head >= num_q_heads || split_idx >= num_splits) return;

    int warp_id = threadIdx.x / VLF8_WARP_SIZE;
    int lane_id = threadIdx.x % VLF8_WARP_SIZE;

    __shared__ int sm_bz;
    __shared__ int sm_qlen;
    __shared__ int sm_qrow_local;
    __shared__ int sm_kv_pages_start;
    __shared__ int sm_kv_pages_end;
    __shared__ int sm_kv_total_len;
    __shared__ int sm_kv_last_page_len;

    if (threadIdx.x == 0) {
        int bz = 0;
        for (int i = 0; i < batch_size; i++) {
            if (q_token >= qo_indptr[i] && q_token < qo_indptr[i + 1]) {
                bz = i;
                break;
            }
        }
        sm_bz = bz;
        int q_start = qo_indptr[bz];
        int q_end = qo_indptr[bz + 1];
        sm_qlen = q_end - q_start;
        sm_qrow_local = q_token - q_start;

        int kv_p_start = kv_indptr[bz];
        int kv_p_end = kv_indptr[bz + 1];
        sm_kv_pages_start = kv_p_start;
        sm_kv_pages_end = kv_p_end;
        int num_kv_pages = kv_p_end - kv_p_start;
        int lpl = (num_kv_pages > 0) ? last_page_len[bz] : 0;
        sm_kv_last_page_len = lpl;
        sm_kv_total_len = (num_kv_pages == 0) ? 0 : (num_kv_pages - 1) * kPageSize + lpl;
    }
    __syncthreads();

    int qlen = sm_qlen;
    int qrow_local = sm_qrow_local;
    int kv_pages_start = sm_kv_pages_start;
    int kv_pages_end = sm_kv_pages_end;
    int kv_total_len = sm_kv_total_len;
    int kv_last_page_len_v = sm_kv_last_page_len;
    int num_kv_pages = kv_pages_end - kv_pages_start;

    int total_q_heads = gridDim.x * num_q_heads;
    int q_idx = q_token * num_q_heads + q_head;
    int out_idx = split_idx * total_q_heads + q_idx;

    auto write_empty_partial = [&]() {
        if (threadIdx.x == 0) {
            partial_m[out_idx] = -FLT_MAX;
            partial_l[out_idx] = 0.0f;
        }
        if (threadIdx.x < HEAD_DIM) {
            partial_out[(size_t)out_idx * HEAD_DIM + threadIdx.x] = 0.0f;
        }
    };

    if (num_kv_pages == 0 || kv_total_len == 0) {
        write_empty_partial();
        return;
    }

    int kv_chunk_size = (kv_total_len + num_splits - 1) / num_splits;
    int kv_chunk_start = split_idx * kv_chunk_size;
    int kv_chunk_end = min(kv_chunk_start + kv_chunk_size, kv_total_len);
    if (kv_chunk_start >= kv_total_len || kv_chunk_start >= kv_chunk_end) {
        write_empty_partial();
        return;
    }

    int causal_limit = (kv_total_len - qlen) + qrow_local;
    if (CAUSAL && qlen > 1 && kv_chunk_start > causal_limit) {
        write_empty_partial();
        return;
    }
    bool chunk_fully_visible = !(CAUSAL && qlen > 1) || (kv_chunk_end - 1 <= causal_limit);

    int gqa_ratio = num_q_heads / num_kv_heads;
    int kv_head = q_head / gqa_ratio;
    int kv_dim = num_kv_heads * HEAD_DIM;
    int stride_page = kPageSize * kv_dim;

    float q_reg[EPT];
    {
        int q_base = q_token * num_q_heads * HEAD_DIM + q_head * HEAD_DIM;
        #pragma unroll
        for (int i = 0; i < EPT; i++) {
            int d = lane_id * EPT + i;
            q_reg[i] = __bfloat162float(Q[q_base + d]) * sm_scale;
        }
    }

    float o_reg[EPT];
    #pragma unroll
    for (int i = 0; i < EPT; i++) o_reg[i] = 0.0f;
    float m_local = -FLT_MAX;
    float l_local = 0.0f;

    int first_page = kv_chunk_start / kPageSize;
    int end_page = (kv_chunk_end + kPageSize - 1) / kPageSize;
    int head_off = kv_head * HEAD_DIM;

    for (int p = first_page; p < end_page; p++) {
        int phys_page = kv_indices[kv_pages_start + p];
        int page_tokens = (p == num_kv_pages - 1) ? kv_last_page_len_v : kPageSize;
        int token_start = max(0, kv_chunk_start - p * kPageSize);
        int token_end = min(page_tokens, kv_chunk_end - p * kPageSize);
        if (token_start >= token_end) continue;

        size_t page_base = (size_t)phys_page * stride_page;
        for (int t = token_start + warp_id; t < token_end; t += VLF8_NUM_WARPS) {
            int kv_pos = p * kPageSize + t;
            if (!chunk_fully_visible && kv_pos > causal_limit) {
                continue;
            }

            size_t row_off = page_base + (size_t)t * kv_dim + head_off;
            int scale_offset = (phys_page * kPageSize + t) * num_kv_heads + kv_head;

            float qk = 0.0f;
            #pragma unroll
            for (int i = 0; i < EPT; i++) {
                int d = lane_id * EPT + i;
                float k_val = load_quantized_value<INT8_KV>(K_pool, row_off + d, K_scales, scale_offset);
                qk += q_reg[i] * k_val;
            }
            qk = vlf8_warp_reduce_sum(qk);

            float m_new = fmaxf(m_local, qk);
            float exp_diff = __expf(m_local - m_new);
            float exp_qk = __expf(qk - m_new);
            float l_new = l_local * exp_diff + exp_qk;

            #pragma unroll
            for (int i = 0; i < EPT; i++) {
                int d = lane_id * EPT + i;
                float v_val = load_quantized_value<INT8_KV>(V_pool, row_off + d, V_scales, scale_offset);
                o_reg[i] = o_reg[i] * exp_diff + exp_qk * v_val;
            }
            m_local = m_new;
            l_local = l_new;
        }
    }

    __shared__ float smem_m[VLF8_NUM_WARPS];
    __shared__ float smem_l[VLF8_NUM_WARPS];
    __shared__ float smem_o[VLF8_NUM_WARPS * HEAD_DIM];

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
        for (int w = 1; w < VLF8_NUM_WARPS; w++) {
            float m_w = smem_m[w];
            float l_w = smem_l[w];
            if (l_w == 0.0f) continue;

            float m_new = fmaxf(final_m, m_w);
            float scale_prev = __expf(final_m - m_new);
            float scale_w = __expf(m_w - m_new);

            #pragma unroll
            for (int i = 0; i < EPT; i++) {
                float o_w = smem_o[w * HEAD_DIM + lane_id * EPT + i];
                final_o[i] = final_o[i] * scale_prev + o_w * scale_w;
            }
            final_l = final_l * scale_prev + l_w * scale_w;
            final_m = m_new;
        }

        if (lane_id == 0) {
            partial_m[out_idx] = final_m;
            partial_l[out_idx] = final_l;
        }
        float inv_l = (final_l > 0.0f) ? (1.0f / final_l) : 0.0f;
        #pragma unroll
        for (int i = 0; i < EPT; i++) {
            int d = lane_id * EPT + i;
            partial_out[(size_t)out_idx * HEAD_DIM + d] = final_o[i] * inv_l;
        }
    }
}

template <int HEAD_DIM>
__global__ void decode_attention_varlen_quantized_merge_kernel(
    const float* __restrict__ partial_out,
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    __nv_bfloat16* __restrict__ O,
    int total_q_tokens,
    int num_q_heads,
    int num_splits)
{
    int q_token = blockIdx.x;
    int q_head = blockIdx.y;
    int d = threadIdx.x;
    if (q_token >= total_q_tokens || q_head >= num_q_heads || d >= HEAD_DIM) return;

    int total_q_heads = total_q_tokens * num_q_heads;
    int q_idx = q_token * num_q_heads + q_head;

    float final_m = -FLT_MAX;
    float final_l = 0.0f;
    float final_o = 0.0f;

    for (int s = 0; s < num_splits; s++) {
        int idx = s * total_q_heads + q_idx;
        float m_s = partial_m[idx];
        float l_s = partial_l[idx];
        float o_s = partial_out[(size_t)idx * HEAD_DIM + d];
        if (l_s == 0.0f) continue;

        float m_new = fmaxf(final_m, m_s);
        float s_prev = final_l * __expf(final_m - m_new);
        float s_cur = l_s * __expf(m_s - m_new);
        float l_new = s_prev + s_cur;

        final_o = (l_new > 0.0f) ? (final_o * s_prev + o_s * s_cur) / l_new : 0.0f;
        final_m = m_new;
        final_l = l_new;
    }

    int o_base = q_token * num_q_heads * HEAD_DIM + q_head * HEAD_DIM;
    O[o_base + d] = __float2bfloat16(final_o);
}

extern "C" cudaError_t decode_attention_varlen_fp8_cuda(
    const __nv_bfloat16* q_packed,
    const int* qo_indptr,
    const void* k_pool,
    const void* v_pool,
    const float* k_scales,
    const float* v_scales,
    const int* kv_indptr,
    const int* kv_indices,
    const int* last_page_len,
    __nv_bfloat16* output,
    int num_q_heads,
    int num_kv_heads,
    int page_size,
    int batch_size,
    int total_q_tokens,
    int max_kv_len,
    bool int8_kv,
    bool causal,
    float sm_scale,
    cudaStream_t stream,
    void* workspace,
    size_t workspace_bytes)
{
    if (batch_size <= 0 || total_q_tokens <= 0) return cudaSuccess;

    constexpr int HEAD_DIM = 128;
    if (page_size != kPageSize || num_q_heads <= 0 || num_kv_heads <= 0) {
        return cudaErrorInvalidValue;
    }
    if (int8_kv && (k_scales == nullptr || v_scales == nullptr)) {
        return cudaErrorInvalidValue;
    }

    int num_splits = choose_varlen_num_splits(max_kv_len);
    size_t needed = decode_attention_varlen_fp8_workspace_bytes(
        total_q_tokens, num_q_heads, HEAD_DIM, num_splits);
    if (workspace == nullptr || workspace_bytes < needed) {
        return cudaErrorInvalidValue;
    }

    float* ws_float = reinterpret_cast<float*>(workspace);
    size_t total_q_heads = (size_t)total_q_tokens * (size_t)num_q_heads;
    float* partial_out = ws_float;
    float* partial_m = partial_out + (size_t)num_splits * total_q_heads * HEAD_DIM;
    float* partial_l = partial_m + (size_t)num_splits * total_q_heads;

    dim3 partial_grid(total_q_tokens, num_q_heads, num_splits);
    dim3 partial_block(VLF8_BLOCK_SIZE);

#define LAUNCH_PARTIAL(INT8_FLAG, CAUSAL_FLAG) \
    decode_attention_varlen_quantized_partial_kernel<HEAD_DIM, CAUSAL_FLAG, INT8_FLAG> \
        <<<partial_grid, partial_block, 0, stream>>>( \
            q_packed, qo_indptr, k_pool, v_pool, k_scales, v_scales, \
            kv_indptr, kv_indices, last_page_len, partial_out, partial_m, partial_l, \
            num_q_heads, num_kv_heads, batch_size, sm_scale, num_splits)

    if (int8_kv) {
        if (causal) {
            LAUNCH_PARTIAL(true, true);
        } else {
            LAUNCH_PARTIAL(true, false);
        }
    } else {
        if (causal) {
            LAUNCH_PARTIAL(false, true);
        } else {
            LAUNCH_PARTIAL(false, false);
        }
    }

#undef LAUNCH_PARTIAL

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    dim3 merge_grid(total_q_tokens, num_q_heads);
    dim3 merge_block(HEAD_DIM);
    decode_attention_varlen_quantized_merge_kernel<HEAD_DIM><<<merge_grid, merge_block, 0, stream>>>(
        partial_out, partial_m, partial_l, output, total_q_tokens, num_q_heads, num_splits);

    return cudaGetLastError();
}
