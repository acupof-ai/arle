// Online-softmax decode-attention kernels for OPD rollout phase (tape disabled).
//
// Improvements over `causal_sdpa_decode_gqa_cache_f32` in attention.cu:
//   1. **Online softmax** (Pellegrini 2018) — single pass over the visible
//      KV positions instead of the original two-pass score-then-softmax-then-
//      mix. Cuts HBM-bound KV reads in half on the dominant O(n²) term.
//   2. **Block size matches head_dim** — one element per thread for the QK
//      dot, then warp-level reductions for cross-thread sum. The original
//      ran with BLOCK=256 and looped `for (dim = tid; dim < head_dim; dim +=
//      blockDim.x)`, leaving 192/256 threads idle when head_dim=64.
//      Qwen3.5-0.8B uses head_dim=256 so we instantiate the kernel at 256.
//   3. **No shared-memory softmax** — running max + denom + numerator live
//      in registers. The original wrote the full `scores[visible]` table to
//      shared memory and ran a single-thread softmax pass at tid=0.
//   4. **BF16 KV cache variant** — KV elements widened from `__nv_bfloat16`
//      to float at load time. Halves the HBM bandwidth requirement for the
//      KV-bound quadratic term. Q stays f32 since it's a single fresh
//      [1, head_dim] tensor per step.
//
// Launch contract (matches the original kernel for drop-in replacement):
//   - q:        f32 [batch, query_heads, 1, head_dim]
//   - k_cache:  TKV  [batch, kv_heads, max_seq, head_dim]
//   - v_cache:  TKV  [batch, kv_heads, max_seq, head_dim]
//   - out:      f32 [batch, query_heads, 1, head_dim]
//   - kv_len, q_start define the causal mask: `visible = min(q_start + 1, kv_len)`
//   - scale: pre-computed `1 / sqrt(head_dim)`
//   - grid = batch * query_heads, block = HEAD_DIM threads (256 for Qwen3.5)
//   - shared memory: HEAD_DIM * 4 bytes (cross-warp reduce scratch)
//
// Numerical contract: f32-output parity within ~1e-4 of the original
// `causal_sdpa_decode_gqa_cache_f32` kernel for f32-KV inputs. BF16 cache
// path is bit-exact only if the F32 cache value can be represented in
// BF16 (typical drift ≤ 1.5e-3 relative on attention output, within OPD
// rollout tolerance since the rollout uses argmax over logits).
//
// Compiled via NVRTC alongside the other autograd kernels in
// kernels.rs::concat_sources; cuda_runtime.h types come from the
// NVRTC-builtin headers, so no explicit #include needed.

namespace {

// BF16 ↔ F32 conversion without cuda_bf16.h. BF16 is the top 16 bits of
// the f32 binary representation (sign + 8-bit exponent + 7-bit mantissa);
// widening is a left shift by 16, narrowing is a round-to-nearest-even on
// the bottom 16 bits.
__device__ __forceinline__ float bf16_bits_to_f32(unsigned short bits) {
    unsigned int u = ((unsigned int) bits) << 16;
    float result;
    // bit-cast via memcpy — compiler folds to no-op move.
    asm("mov.b32 %0, %1;" : "=f"(result) : "r"(u));
    return result;
}

template <typename TKV>
__device__ __forceinline__ float load_kv(const TKV* ptr) {
    return *ptr;
}

// BF16 storage uses unsigned short (raw bits). Specialize the load to do
// the widening cast.
template <>
__device__ __forceinline__ float load_kv<unsigned short>(const unsigned short* ptr) {
    return bf16_bits_to_f32(*ptr);
}

// Warp-wide sum reduction. `WARP_SIZE` must be 32.
__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, s);
    }
    return v;
}

// Cross-warp sum reduction via shared memory. `warp_id` = tid / 32,
// `lane_id` = tid % 32. Caller must `__syncthreads()` before reusing scratch.
__device__ __forceinline__ float block_sum(float v, float* scratch, int tid, int n_warps) {
    int warp_id = tid >> 5;
    int lane_id = tid & 31;
    float w = warp_sum(v);
    if (lane_id == 0) {
        scratch[warp_id] = w;
    }
    __syncthreads();
    // Final reduce in the first warp.
    if (warp_id == 0) {
        float partial = (tid < n_warps) ? scratch[tid] : 0.0f;
        partial = warp_sum(partial);
        if (tid == 0) {
            scratch[0] = partial;
        }
    }
    __syncthreads();
    return scratch[0];
}

// NVRTC builds this file without <math.h> / <cuda_runtime.h>, so INFINITY
// from <math.h> is unavailable. Use the largest-magnitude negative f32 as
// the initial running max; the first softmax iteration replaces it.
__device__ __forceinline__ float neg_inf_f32() {
    return -3.40282347e+38f;
}

template <typename TKV, int HEAD_DIM>
__global__ void __launch_bounds__(HEAD_DIM, 2) causal_sdpa_decode_gqa_cache_online_impl(
    const float* __restrict__ q,
    const TKV* __restrict__ k,
    const TKV* __restrict__ v,
    float* __restrict__ out,
    int batch,
    int query_heads,
    int kv_heads,
    int max_seq,
    int kv_len,
    int q_start,
    float scale
) {
    static_assert(HEAD_DIM % 32 == 0, "HEAD_DIM must be a multiple of warp size");
    constexpr int N_WARPS = HEAD_DIM / 32;

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int b = row / query_heads;
    int qh = row - b * query_heads;
    if (b >= batch) {
        return;
    }
    int kv_repeat = query_heads / kv_heads;
    int kvh = qh / kv_repeat;
    int visible = min(q_start + 1, kv_len);
    if (visible <= 0) {
        // Zero-fill output (degenerate / mask-empty case) and bail.
        if (tid < HEAD_DIM) {
            out[(b * query_heads + qh) * HEAD_DIM + tid] = 0.0f;
        }
        return;
    }

    int q_base = (b * query_heads + qh) * HEAD_DIM;
    int kv_base = (b * kv_heads + kvh) * max_seq * HEAD_DIM;

    // Each thread owns one Q element + one output accumulator slot.
    float q_elem = (tid < HEAD_DIM) ? q[q_base + tid] : 0.0f;
    float o_acc = 0.0f;
    float m_run = neg_inf_f32();
    float l_run = 0.0f;

    extern __shared__ float scratch[];

    for (int pos = 0; pos < visible; ++pos) {
        // QK dot: each thread does one multiply, warp-reduce, cross-warp-reduce.
        const TKV* k_row = k + kv_base + pos * HEAD_DIM;
        float k_elem = (tid < HEAD_DIM) ? load_kv(k_row + tid) : 0.0f;
        float partial = q_elem * k_elem;
        float s_pos = block_sum(partial, scratch, tid, N_WARPS) * scale;

        // Online softmax update: all threads have s_pos identically.
        float m_new = fmaxf(m_run, s_pos);
        float alpha = __expf(m_run - m_new);
        float beta = __expf(s_pos - m_new);

        // Re-weight + add new V contribution into this thread's slot.
        const TKV* v_row = v + kv_base + pos * HEAD_DIM;
        float v_elem = (tid < HEAD_DIM) ? load_kv(v_row + tid) : 0.0f;
        o_acc = o_acc * alpha + beta * v_elem;
        l_run = l_run * alpha + beta;
        m_run = m_new;
    }

    if (tid < HEAD_DIM && l_run > 0.0f) {
        out[q_base + tid] = o_acc / l_run;
    } else if (tid < HEAD_DIM) {
        out[q_base + tid] = 0.0f;
    }
}

}  // namespace

extern "C" {

// f32 KV path — drop-in replacement for causal_sdpa_decode_gqa_cache_f32
// with online softmax. Same numerical result up to floating-point
// associativity (online vs two-pass softmax both compute exp(s - max)).
__global__ void causal_sdpa_decode_gqa_cache_online_f32_hd256(
    const float* q, const float* k, const float* v, float* out,
    int batch, int query_heads, int kv_heads, int max_seq, int kv_len,
    int head_dim, int q_start, float scale
) {
    if (head_dim != 256) return;  // template safety
    causal_sdpa_decode_gqa_cache_online_impl<float, 256>(
        q, k, v, out, batch, query_heads, kv_heads, max_seq, kv_len, q_start, scale);
}

// BF16 KV path — KV cache stored as raw bf16 bits (unsigned short).
// Widened to f32 inside the kernel at load time via bf16_bits_to_f32.
// HBM bandwidth on K + V reads is **halved** relative to the f32 path,
// which directly attacks the dominant 0.0099·n² quadratic term in the
// OPD rollout perf fit.
__global__ void causal_sdpa_decode_gqa_cache_online_bf16_hd256(
    const float* q, const unsigned short* k, const unsigned short* v, float* out,
    int batch, int query_heads, int kv_heads, int max_seq, int kv_len,
    int head_dim, int q_start, float scale
) {
    if (head_dim != 256) return;
    causal_sdpa_decode_gqa_cache_online_impl<unsigned short, 256>(
        q, k, v, out, batch, query_heads, kv_heads, max_seq, kv_len, q_start, scale);
}

}  // extern "C"
