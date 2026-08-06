#include "common.cuh"

#define MAX_HEAD_DIM 256
#define THREADS_HEAD_DIM 256
#define NUM_WARPS_HEAD_DIM (THREADS_HEAD_DIM / WARP_SIZE)

__device__ __forceinline__ __nv_bfloat16 rms_norm_elem(
    __nv_bfloat16 x, float rms_inv, __nv_bfloat16 weight) {
    return __float2bfloat16(__bfloat162float(x) * rms_inv * (1.0f + __bfloat162float(weight)));
}

__device__ __forceinline__ void apply_rope_pair(
    __nv_bfloat16& x0, __nv_bfloat16& x1,
    __nv_bfloat16 cos_val, __nv_bfloat16 sin_val) {
    float fx0 = __bfloat162float(x0);
    float fx1 = __bfloat162float(x1);
    float fc = __bfloat162float(cos_val);
    float fs = __bfloat162float(sin_val);
    x0 = __float2bfloat16(fx0 * fc - fx1 * fs);
    x1 = __float2bfloat16(fx0 * fs + fx1 * fc);
}

__global__ void prefill_qk_norm_rope_kernel(
    const __nv_bfloat16* __restrict__ q_full_batch,  // [q_full_dim, seq_len]
    const __nv_bfloat16* __restrict__ k_batch,       // [kv_dim, seq_len]
    const __nv_bfloat16* __restrict__ q_norm_weight, // [head_dim]
    const __nv_bfloat16* __restrict__ k_norm_weight, // [head_dim]
    const __nv_bfloat16* __restrict__ cos_cache,     // [max_seq * rotary_dim]
    const __nv_bfloat16* __restrict__ sin_cache,
    __nv_bfloat16* __restrict__ q_batch_out,         // [q_dim, seq_len]
    __nv_bfloat16* __restrict__ k_cache,             // [num_kvheads * max_seq_len * head_dim]
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int seq_len,
    const int* __restrict__ start_pos_ptr,           // GPU-resident for CUDA Graph safety
    int rotary_dim,
    float rms_eps,
    int max_seq_len,
    int ring_modulus                                 // >0: cache row = pos % modulus (sliding-window ring)
) {
    int start_pos = *start_pos_ptr;
    int head_global = blockIdx.x;
    int token = blockIdx.y;
    int d = threadIdx.x;

    bool is_q = head_global < num_q_heads;
    int head_local = is_q ? head_global : (head_global - num_q_heads);
    int q_full_dim = num_q_heads * head_dim * 2;
    int q_dim = num_q_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;
    bool active = d < head_dim;

    int src_offset = is_q
        ? token * q_full_dim + head_local * 2 * head_dim + d
        : token * kv_dim + head_local * head_dim + d;
    __nv_bfloat16 x = active
        ? (is_q ? q_full_batch[src_offset] : k_batch[src_offset])
        : __float2bfloat16(0.0f);
    const __nv_bfloat16* norm_w = is_q ? q_norm_weight : k_norm_weight;

    float sq = __bfloat162float(x);
    sq *= sq;
    float sq_sum = warp_reduce_sum(sq);

    int warp_id = d / WARP_SIZE;
    int lane_id = d % WARP_SIZE;
    __shared__ float warp_sums[NUM_WARPS_HEAD_DIM];
    __shared__ float inv_rms;
    __shared__ __nv_bfloat16 smem[MAX_HEAD_DIM];

    if (lane_id == 0) warp_sums[warp_id] = sq_sum;
    __syncthreads();

    if (d == 0) {
        float total = 0.0f;
        for (int i = 0; i < NUM_WARPS_HEAD_DIM; i++) total += warp_sums[i];
        inv_rms = 1.0f / sqrtf(total / head_dim + rms_eps);
    }
    __syncthreads();

    if (active) {
        smem[d] = rms_norm_elem(x, inv_rms, norm_w[d]);
    }
    __syncthreads();

    // RoPE always indexes the ABSOLUTE position `pos`; the cache write row wraps
    // through the ring modulus for sliding-window layers (else linear `pos`).
    int pos = start_pos + token;
    int cache_row = ring_modulus > 0 ? (pos % ring_modulus) : pos;
    int half_rotary = rotary_dim / 2;

    if (d < half_rotary) {
        __nv_bfloat16 lo = smem[d];
        __nv_bfloat16 hi = smem[d + half_rotary];
        apply_rope_pair(
            lo,
            hi,
            cos_cache[pos * rotary_dim + d],
            sin_cache[pos * rotary_dim + d]
        );

        if (is_q) {
            int dst = token * q_dim + head_local * head_dim;
            q_batch_out[dst + d] = lo;
            q_batch_out[dst + d + half_rotary] = hi;
        } else {
            int dst = head_local * max_seq_len * head_dim + cache_row * head_dim;
            k_cache[dst + d] = lo;
            k_cache[dst + d + half_rotary] = hi;
        }
    }

    if (active && d >= rotary_dim) {
        if (is_q) {
            int dst = token * q_dim + head_local * head_dim;
            q_batch_out[dst + d] = smem[d];
        } else {
            int dst = head_local * max_seq_len * head_dim + cache_row * head_dim;
            k_cache[dst + d] = smem[d];
        }
    }
}

__global__ void prefill_v_cache_write_kernel(
    const __nv_bfloat16* __restrict__ v_batch,  // [kv_dim, seq_len]
    __nv_bfloat16* __restrict__ v_cache,        // [num_kvheads * max_seq_len * head_dim]
    int num_kv_heads,
    int head_dim,
    int seq_len,
    const int* __restrict__ start_pos_ptr,      // GPU-resident
    int max_seq_len,
    int ring_modulus                            // >0: cache row = pos % modulus
) {
    int start_pos = *start_pos_ptr;
    int kv_head = blockIdx.x;
    int token = blockIdx.y;
    int d = threadIdx.x;
    if (d >= head_dim) return;

    int kv_dim = num_kv_heads * head_dim;
    int pos = start_pos + token;
    int cache_row = ring_modulus > 0 ? (pos % ring_modulus) : pos;
    int src = token * kv_dim + kv_head * head_dim + d;
    int dst = kv_head * max_seq_len * head_dim + cache_row * head_dim + d;
    v_cache[dst] = v_batch[src];
}

__global__ void attention_gate_batch_kernel(
    const __nv_bfloat16* __restrict__ q_full_batch,  // [q_full_dim, seq_len]
    __nv_bfloat16* __restrict__ attn_out,            // [q_dim, seq_len]
    int num_q_heads,
    int head_dim,
    int seq_len,
    int use_swish
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int q_dim = num_q_heads * head_dim;
    int total = q_dim * seq_len;
    if (idx >= total) return;

    int token = idx / q_dim;
    int q_offset = idx - token * q_dim;
    int q_head = q_offset / head_dim;
    int dim = q_offset % head_dim;
    int q_full_dim = q_dim * 2;
    int gate_idx = token * q_full_dim + q_head * 2 * head_dim + head_dim + dim;

    float gate = __bfloat162float(q_full_batch[gate_idx]);
    float sig_gate = 1.0f / (1.0f + expf(-gate));
    float gate_val = use_swish ? (gate * sig_gate) : sig_gate;
    float out = __bfloat162float(attn_out[idx]);
    attn_out[idx] = __float2bfloat16(out * gate_val);
}

extern "C" {

static cudaError_t prefill_attention_hd256_prep_impl(
    const __nv_bfloat16* q_full_batch,
    const __nv_bfloat16* k_batch,
    const __nv_bfloat16* v_batch,
    const __nv_bfloat16* q_norm_weight,
    const __nv_bfloat16* k_norm_weight,
    const __nv_bfloat16* cos_cache,
    const __nv_bfloat16* sin_cache,
    __nv_bfloat16* q_batch_out,
    __nv_bfloat16* k_cache,
    __nv_bfloat16* v_cache,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int seq_len,
    const int* start_pos_ptr,
    int rotary_dim,
    float rms_eps,
    int max_seq_len,
    int ring_modulus,
    cudaStream_t stream
) {
    if (num_q_heads <= 0 || num_kv_heads <= 0 || (head_dim != 128 && head_dim != 256) ||
        seq_len < 0 || start_pos_ptr == nullptr || rotary_dim <= 0 ||
        rotary_dim > head_dim || (rotary_dim % 2) != 0 || max_seq_len < seq_len ||
        (num_q_heads % num_kv_heads) != 0 || ring_modulus < 0 ||
        (ring_modulus > 0 && max_seq_len < ring_modulus)) {
        return cudaErrorInvalidValue;
    }
    if (seq_len == 0) {
        return cudaSuccess;
    }
    dim3 prep_grid(num_q_heads + num_kv_heads, seq_len);
    prefill_qk_norm_rope_kernel<<<prep_grid, THREADS_HEAD_DIM, 0, stream>>>(
        q_full_batch,
        k_batch,
        q_norm_weight,
        k_norm_weight,
        cos_cache,
        sin_cache,
        q_batch_out,
        k_cache,
        num_q_heads,
        num_kv_heads,
        head_dim,
        seq_len,
        start_pos_ptr,
        rotary_dim,
        rms_eps,
        max_seq_len,
        ring_modulus
    );

    dim3 v_grid(num_kv_heads, seq_len);
    prefill_v_cache_write_kernel<<<v_grid, THREADS_HEAD_DIM, 0, stream>>>(
        v_batch,
        v_cache,
        num_kv_heads,
        head_dim,
        seq_len,
        start_pos_ptr,
        max_seq_len,
        ring_modulus
    );
    return cudaGetLastError();
}

cudaError_t prefill_attention_hd256_prep_cuda(
    const __nv_bfloat16* q_full_batch,
    const __nv_bfloat16* k_batch,
    const __nv_bfloat16* v_batch,
    const __nv_bfloat16* q_norm_weight,
    const __nv_bfloat16* k_norm_weight,
    const __nv_bfloat16* cos_cache,
    const __nv_bfloat16* sin_cache,
    __nv_bfloat16* q_batch_out,
    __nv_bfloat16* k_cache,
    __nv_bfloat16* v_cache,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int seq_len,
    const int* start_pos_ptr,
    int rotary_dim,
    float rms_eps,
    int max_seq_len,
    cudaStream_t stream
) {
    return prefill_attention_hd256_prep_impl(
        q_full_batch, k_batch, v_batch, q_norm_weight, k_norm_weight,
        cos_cache, sin_cache, q_batch_out, k_cache, v_cache,
        num_q_heads, num_kv_heads, head_dim, seq_len, start_pos_ptr,
        rotary_dim, rms_eps, max_seq_len, /*ring_modulus=*/0, stream);
}

// Sliding-window ring variant: the K/V cache write row wraps as `pos %
// ring_modulus` (ring_modulus == the per-head stride == window+block). Callers
// pass an ABSOLUTE `start_pos` (RoPE indexes it directly, unshifted). A single
// launch must write <= ring_modulus rows, else two tokens alias one ring row.
cudaError_t prefill_attention_hd256_prep_ring_cuda(
    const __nv_bfloat16* q_full_batch,
    const __nv_bfloat16* k_batch,
    const __nv_bfloat16* v_batch,
    const __nv_bfloat16* q_norm_weight,
    const __nv_bfloat16* k_norm_weight,
    const __nv_bfloat16* cos_cache,
    const __nv_bfloat16* sin_cache,
    __nv_bfloat16* q_batch_out,
    __nv_bfloat16* k_cache,
    __nv_bfloat16* v_cache,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int seq_len,
    const int* start_pos_ptr,
    int rotary_dim,
    float rms_eps,
    int ring_modulus,
    cudaStream_t stream
) {
    if (ring_modulus <= 0 || seq_len > ring_modulus) {
        return cudaErrorInvalidValue;
    }
    return prefill_attention_hd256_prep_impl(
        q_full_batch, k_batch, v_batch, q_norm_weight, k_norm_weight,
        cos_cache, sin_cache, q_batch_out, k_cache, v_cache,
        num_q_heads, num_kv_heads, head_dim, seq_len, start_pos_ptr,
        rotary_dim, rms_eps, /*max_seq_len=*/ring_modulus, ring_modulus, stream);
}

cudaError_t attention_gate_batch_hd256_cuda(
    const __nv_bfloat16* q_full_batch,
    __nv_bfloat16* attn_out,
    int num_q_heads,
    int head_dim,
    int seq_len,
    int use_swish,
    cudaStream_t stream
) {
    if (num_q_heads <= 0 || (head_dim != 128 && head_dim != 256) || seq_len < 0) {
        return cudaErrorInvalidValue;
    }
    if (seq_len == 0) {
        return cudaSuccess;
    }
    int total = num_q_heads * head_dim * seq_len;
    int block = 256;
    int grid = (total + block - 1) / block;
    attention_gate_batch_kernel<<<grid, block, 0, stream>>>(
        q_full_batch,
        attn_out,
        num_q_heads,
        head_dim,
        seq_len,
        use_swish
    );
    return cudaGetLastError();
}

} // extern "C"
