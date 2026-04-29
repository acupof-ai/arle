// Variable-length Q + paged FP8 E4M3 KV decode/prefill attention.
//
// Unblocks `StepPlan::Mixed` for FP8 KV pools. Mirrors the qlen=1 split-KV
// kernel in `decode_attention_quantized.cu` but generalizes it to:
//
//   1. Variable Q length per request (qo_indptr-driven).
//      qlen=1 rows are pure decode; qlen>1 rows are prefill chunks.
//   2. Optional FlashAttention-2 style causal masking on prefill rows.
//   3. Single-CTA-per-(q_token, q_head) — one block tile streams the
//      entire KV history for that row, accumulates online softmax, and
//      writes the output. No split-KV reduction; varlen makes split-KV
//      hard because per-row qlen varies and the partial-merge math
//      assumes uniform qlen=1.
//
// FP8 E4M3 has no scale tensor (self-describing 8-bit float). We dequant
// inline as we read the pool. The pool layout is the same NHD page layout
// the rest of the FP8 path uses:
//
//     k_pool / v_pool: __nv_fp8_e4m3 [max_pages, num_kv_heads, page_size, HEAD_DIM]
//     stride_page = num_kv_heads * page_size * HEAD_DIM
//
// Indexing helpers:
//   - qo_indptr[bz..bz+1]   → q_token range for sequence `bz`
//   - kv_indptr[bz..bz+1]   → page-index range for sequence `bz`
//   - kv_indices[...]       → physical page ids
//   - last_page_len[bz]     → tokens used in the final page (others full)
//
// Reads `qlen` from `qo_indptr` per row; reads `kv_total_len` from
// `(num_kv_pages - 1) * page_size + last_page_len[bz]` (matches the
// existing FP8 decode kernel and prefill_attention_paged_prep).
//
// HD128 only for now — Qwen3-4B/8B/14B all have head_dim=128. HD256
// (Qwen3.5) reuses the FlashInfer mixed path; quantized HD256 is a
// future extension.
//
// Block size: BLOCK_SIZE = 128 threads = 4 warps. EPT = HEAD_DIM/WARP_SIZE
// = 4. Each warp owns one Q feature stripe (dims [lane*4 .. lane*4+3])
// and reduces the QK partial across lanes via __shfl_xor.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>

#define VLF8_NUM_WARPS 4
#define VLF8_WARP_SIZE 32
#define VLF8_BLOCK_SIZE (VLF8_NUM_WARPS * VLF8_WARP_SIZE)

namespace {
// Pool page size baked into the existing FP8 path. Aligned with
// `kQuantPageSize` in decode_attention_quantized.cu.
constexpr int kPageSize = 16;
}

__device__ __forceinline__ float vlf8_warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

// ============================================================================
// Single-CTA-per-(q_token, q_head) varlen FP8 attention.
//
// Grid: (total_q_tokens, num_q_heads)
// Block: VLF8_BLOCK_SIZE = 128 threads
//
// Each block:
//   1. Resolves its sequence index `bz` via a binary search over qo_indptr.
//   2. Loads its Q row (HEAD_DIM bf16 elements) into registers.
//   3. Streams KV pages for `bz`, dequantizing FP8 → float in registers.
//   4. Applies causal mask if `causal && qlen > 1`.
//   5. Online softmax across all KV tokens.
//   6. Cross-warp merge via shared mem, writes bf16 output.
// ============================================================================
template <int HEAD_DIM, bool CAUSAL>
__global__ void decode_attention_varlen_fp8_kernel(
    const __nv_bfloat16* __restrict__ Q,            // [total_q_tokens, num_q_heads * HEAD_DIM]
    const int* __restrict__ qo_indptr,              // [batch_size + 1]
    const __nv_fp8_e4m3* __restrict__ K_pool,       // [max_pages, num_kv_heads, page_size, HEAD_DIM]
    const __nv_fp8_e4m3* __restrict__ V_pool,
    const int* __restrict__ kv_indptr,              // [batch_size + 1]
    const int* __restrict__ kv_indices,             // [total_pages]
    const int* __restrict__ last_page_len,          // [batch_size]
    __nv_bfloat16* __restrict__ O,                  // [total_q_tokens, num_q_heads * HEAD_DIM]
    int num_q_heads,
    int num_kv_heads,
    int batch_size,
    float sm_scale)
{
    constexpr int EPT = HEAD_DIM / VLF8_WARP_SIZE;  // 4 for HD128

    int q_token = blockIdx.x;
    int q_head  = blockIdx.y;
    if (q_head >= num_q_heads) return;

    int warp_id = threadIdx.x / VLF8_WARP_SIZE;
    int lane_id = threadIdx.x % VLF8_WARP_SIZE;

    // ── Resolve sequence index for this q_token via linear scan over qo_indptr.
    // Done in shared memory so all threads agree without launch-time fixup.
    // batch_size <= num_slots (typically <= 32), so linear scan is cheap.
    __shared__ int sm_bz;
    __shared__ int sm_qlen;
    __shared__ int sm_qrow_local;     // row within sequence
    __shared__ int sm_kv_pages_start;
    __shared__ int sm_kv_pages_end;
    __shared__ int sm_kv_total_len;
    __shared__ int sm_kv_last_page_len;

    if (threadIdx.x == 0) {
        int bz = 0;
        // Find bz such that qo_indptr[bz] <= q_token < qo_indptr[bz+1].
        for (int i = 0; i < batch_size; i++) {
            if (q_token >= qo_indptr[i] && q_token < qo_indptr[i + 1]) {
                bz = i;
                break;
            }
        }
        sm_bz = bz;
        int q_start = qo_indptr[bz];
        int q_end   = qo_indptr[bz + 1];
        sm_qlen = q_end - q_start;
        sm_qrow_local = q_token - q_start;

        int kv_p_start = kv_indptr[bz];
        int kv_p_end   = kv_indptr[bz + 1];
        sm_kv_pages_start = kv_p_start;
        sm_kv_pages_end   = kv_p_end;
        int num_kv_pages  = kv_p_end - kv_p_start;
        int lpl = (num_kv_pages > 0) ? last_page_len[bz] : 0;
        sm_kv_last_page_len = lpl;
        sm_kv_total_len = (num_kv_pages == 0) ? 0
                                              : (num_kv_pages - 1) * kPageSize + lpl;
    }
    __syncthreads();

    int qlen           = sm_qlen;
    int qrow_local     = sm_qrow_local;
    int kv_pages_start = sm_kv_pages_start;
    int kv_pages_end   = sm_kv_pages_end;
    int kv_total_len   = sm_kv_total_len;
    int kv_last_page_len_v = sm_kv_last_page_len;
    int num_kv_pages   = kv_pages_end - kv_pages_start;

    // FlashAttention-2 causal: q_token at logical position
    //   pos_q = (kv_total_len - qlen) + qrow_local
    // attends to kv positions [0, pos_q].
    int kv_offset_for_causal = kv_total_len - qlen;
    int causal_limit = kv_offset_for_causal + qrow_local;  // inclusive

    // Empty KV → write zeros and return.
    if (num_kv_pages == 0 || kv_total_len == 0) {
        if (threadIdx.x < HEAD_DIM) {
            O[(int64_t)q_token * num_q_heads * HEAD_DIM + q_head * HEAD_DIM + threadIdx.x]
                = __float2bfloat16(0.0f);
        }
        return;
    }

    int gqa_ratio = num_q_heads / num_kv_heads;
    int kv_head = q_head / gqa_ratio;

    // ── Load Q row into registers (each lane owns HEAD_DIM/WARP_SIZE = 4 dims).
    float q_reg[EPT];
    {
        int q_base = q_token * num_q_heads * HEAD_DIM + q_head * HEAD_DIM;
        #pragma unroll
        for (int i = 0; i < EPT; i++) {
            int d = lane_id * EPT + i;
            q_reg[i] = __bfloat162float(Q[q_base + d]) * sm_scale;
        }
    }

    // ── Per-warp online softmax state ──
    float o_reg[EPT];
    #pragma unroll
    for (int i = 0; i < EPT; i++) o_reg[i] = 0.0f;
    float m_local = -FLT_MAX;
    float l_local = 0.0f;

    // Stream every KV page assigned to this sequence; warps split tokens within
    // a page (warp-stride loop, same shape as the qlen=1 FP8 kernel).
    int stride_page = num_kv_heads * kPageSize * HEAD_DIM;
    int kv_pos_running = 0;  // global kv position cursor across pages
    for (int p = 0; p < num_kv_pages; p++) {
        int phys_page = kv_indices[kv_pages_start + p];
        int page_tokens = (p == num_kv_pages - 1) ? kv_last_page_len_v : kPageSize;
        // Pointer to this page's KV-head slice: NHD layout
        //   pool_base = phys_page * stride_page
        //             + kv_head  * page_size * HEAD_DIM
        const __nv_fp8_e4m3* k_page_base =
            K_pool + (size_t)phys_page * stride_page
                   + (size_t)kv_head   * kPageSize * HEAD_DIM;
        const __nv_fp8_e4m3* v_page_base =
            V_pool + (size_t)phys_page * stride_page
                   + (size_t)kv_head   * kPageSize * HEAD_DIM;

        for (int t = warp_id; t < page_tokens; t += VLF8_NUM_WARPS) {
            int kv_pos = kv_pos_running + t;

            // Causal mask: skip kv tokens beyond causal_limit when CAUSAL && qlen>1.
            // For qlen==1 (decode rows) we never mask: causal_limit = kv_total_len-1
            // and decode rows always see the full history.
            bool keep = true;
            if (CAUSAL && qlen > 1) {
                keep = (kv_pos <= causal_limit);
            }

            float qk = 0.0f;
            #pragma unroll
            for (int i = 0; i < EPT; i++) {
                int d = lane_id * EPT + i;
                float k_val = static_cast<float>(k_page_base[t * HEAD_DIM + d]);
                qk += q_reg[i] * k_val;
            }
            qk = vlf8_warp_reduce_sum(qk);
            // Broadcast qk: all lanes already have the reduced sum after xor-reduce.

            if (!keep) {
                // Skip this kv position (don't update softmax state, don't accumulate V).
                continue;
            }

            float m_new   = fmaxf(m_local, qk);
            float exp_diff = __expf(m_local - m_new);
            float exp_qk   = __expf(qk - m_new);
            float l_new   = l_local * exp_diff + exp_qk;

            #pragma unroll
            for (int i = 0; i < EPT; i++) {
                int d = lane_id * EPT + i;
                float v_val = static_cast<float>(v_page_base[t * HEAD_DIM + d]);
                o_reg[i] = o_reg[i] * exp_diff + exp_qk * v_val;
            }
            m_local = m_new;
            l_local = l_new;
        }

        kv_pos_running += page_tokens;
    }

    // ── Cross-warp merge via shared memory ──
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
        // Unnormalized accumulator: max, sum-of-exp, sum-of-(exp * v).
        float final_m = smem_m[0];
        float final_l = smem_l[0];
        float final_o[EPT];
        #pragma unroll
        for (int i = 0; i < EPT; i++) {
            final_o[i] = smem_o[lane_id * EPT + i];
        }

        // FlashAttention merge across warps: keep accumulator unnormalized
        // (sum-of-exp · V) so the normalization is one divide at the end.
        #pragma unroll
        for (int w = 1; w < VLF8_NUM_WARPS; w++) {
            float m_w = smem_m[w];
            float l_w = smem_l[w];
            if (l_w == 0.0f) continue;

            float m_new   = fmaxf(final_m, m_w);
            float scale_a = __expf(final_m - m_new);
            float scale_b = __expf(m_w     - m_new);

            #pragma unroll
            for (int i = 0; i < EPT; i++) {
                float o_w = smem_o[w * HEAD_DIM + lane_id * EPT + i];
                final_o[i] = final_o[i] * scale_a + o_w * scale_b;
            }
            final_l = final_l * scale_a + l_w * scale_b;
            final_m = m_new;
        }

        // Final normalize (single divide). Guard against empty KV path.
        float inv_l = (final_l > 0.0f) ? (1.0f / final_l) : 0.0f;

        int o_base = q_token * num_q_heads * HEAD_DIM + q_head * HEAD_DIM;
        #pragma unroll
        for (int i = 0; i < EPT; i++) {
            int d = lane_id * EPT + i;
            O[o_base + d] = __float2bfloat16(final_o[i] * inv_l);
        }
    }
}

// ============================================================================
// C API
// ============================================================================
extern "C" {

cudaError_t decode_attention_varlen_fp8_cuda(
    const __nv_bfloat16* q_packed,
    const int* qo_indptr,
    const __nv_fp8_e4m3* k_pool,
    const __nv_fp8_e4m3* v_pool,
    const int* kv_indptr,
    const int* kv_indices,
    const int* last_page_len,
    __nv_bfloat16* output,
    int num_q_heads,
    int num_kv_heads,
    int page_size,
    int batch_size,
    int total_q_tokens,
    bool causal,
    float sm_scale,
    cudaStream_t stream)
{
    if (batch_size <= 0 || total_q_tokens <= 0) return cudaSuccess;

    // Today: HEAD_DIM=128 + page_size=16 only. The pool's NHD layout and the
    // online-softmax math is otherwise dimension-agnostic; HD256 / page_size
    // variants can specialize the template later.
    constexpr int HEAD_DIM = 128;
    if (page_size != kPageSize) {
        return cudaErrorInvalidValue;
    }

    dim3 grid(total_q_tokens, num_q_heads);
    dim3 block(VLF8_BLOCK_SIZE);

    if (causal) {
        decode_attention_varlen_fp8_kernel<HEAD_DIM, true><<<grid, block, 0, stream>>>(
            q_packed, qo_indptr,
            k_pool, v_pool,
            kv_indptr, kv_indices, last_page_len,
            output,
            num_q_heads, num_kv_heads, batch_size,
            sm_scale);
    } else {
        decode_attention_varlen_fp8_kernel<HEAD_DIM, false><<<grid, block, 0, stream>>>(
            q_packed, qo_indptr,
            k_pool, v_pool,
            kv_indptr, kv_indices, last_page_len,
            output,
            num_q_heads, num_kv_heads, batch_size,
            sm_scale);
    }

    return cudaGetLastError();
}

}  // extern "C"
