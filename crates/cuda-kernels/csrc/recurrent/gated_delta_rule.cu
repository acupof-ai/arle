#include "common.cuh"
#include <cmath>

// Gated Delta Rule — Recurrent decode step for linear attention
//
// Each block handles one value head (32 blocks total for Qwen3.5-4B).
// State layout: [key_dim, val_dim] f32 per head — val_dim is contiguous.
//   Matches FLA convention [H, K, V] where V stride = 1.
//
// Parallelism strategy: 512 threads per block, split into J_SLICES=4 groups.
//   Each group handles key_dim/4 = 32 rows of the j-loop.
//   val_idx = threadIdx.x % 128, j_slice = threadIdx.x / 128.
//   This gives 16 warps/block → better latency hiding on the 32-block grid.
//   Partial kv_mem and output reductions across j_slices via shared memory.
//
// Thread mapping: val_idx ∈ [0, val_dim), j_slice ∈ [0, J_SLICES).
// GQA: num_key_heads < num_value_heads, so multiple value heads share one key head.

#define GDR_KEY_DIM 128
#define GDR_VAL_DIM 128
#define GDR_J_SLICES 4
#define GDR_BLOCK_DIM (GDR_VAL_DIM * GDR_J_SLICES)  // 512
#define GDR_J_PER_SLICE (GDR_KEY_DIM / GDR_J_SLICES) // 32
// Decode: split each head's val columns across 2 blocks (96 blocks fill the
// 78-SM H20; 48 left 30 SMs idle) and keep the state tile in registers
// between the decay and update passes (one DRAM read + one write instead of
// two of each — state traffic is the whole kernel).
#define GDR_DEC_VAL_SPLIT 2
#define GDR_DEC_VAL_COLS (GDR_VAL_DIM / GDR_DEC_VAL_SPLIT)         // 64
#define GDR_DEC_BLOCK_DIM (GDR_DEC_VAL_COLS * GDR_J_SLICES)        // 256

__global__ void gated_delta_rule_decode_kernel(
    const __nv_bfloat16* __restrict__ qkv,   // [q_dim + k_dim + v_dim] after conv1d+SiLU
    const __nv_bfloat16* __restrict__ b_proj, // [num_value_heads]
    const __nv_bfloat16* __restrict__ a_proj, // [num_value_heads]
    const __nv_bfloat16* __restrict__ dt_bias,// [num_value_heads] bf16
    const float* __restrict__ A_log,          // [num_value_heads] f32
    float* __restrict__ state,                // [num_value_heads, key_dim, val_dim] f32 (V contiguous)
    __nv_bfloat16* __restrict__ output,       // [num_value_heads * val_dim] bf16
    int num_key_heads,
    int num_value_heads,
    int key_dim,    // 128
    int val_dim     // 128
) {
    int v_head = blockIdx.x / GDR_DEC_VAL_SPLIT;
    int val_base = (blockIdx.x % GDR_DEC_VAL_SPLIT) * GDR_DEC_VAL_COLS;
    int val_col = threadIdx.x & (GDR_DEC_VAL_COLS - 1);
    int val_idx = val_base + val_col;
    int j_slice = threadIdx.x / GDR_DEC_VAL_COLS;  // 0..3
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 0x1F;

    int k_head = v_head * num_key_heads / num_value_heads;
    int q_dim_total = key_dim * num_key_heads;
    int k_dim_total = q_dim_total;

    __shared__ float smem_q[GDR_KEY_DIM];
    __shared__ float smem_k[GDR_KEY_DIM];
    __shared__ float smem_norm[2];
    __shared__ float warp_norms[GDR_DEC_BLOCK_DIM / WARP_SIZE];  // 8
    __shared__ float s_exp_g;
    __shared__ float s_beta;
    __shared__ float smem_kv_partial[GDR_J_SLICES][GDR_DEC_VAL_COLS];
    __shared__ float smem_out_partial[GDR_J_SLICES][GDR_DEC_VAL_COLS];
    __shared__ float smem_v[GDR_DEC_VAL_COLS];

    // Threads 0..key_dim load the full q/k rows (independent of the val split).
    float q_val = 0.0f;
    float k_val = 0.0f;
    if (threadIdx.x < GDR_KEY_DIM) {
        q_val = __bfloat162float(qkv[k_head * key_dim + threadIdx.x]);
        k_val = __bfloat162float(qkv[q_dim_total + k_head * key_dim + threadIdx.x]);
    }
    if (threadIdx.x < GDR_DEC_VAL_COLS) {
        smem_v[threadIdx.x] =
            __bfloat162float(qkv[q_dim_total + k_dim_total + v_head * val_dim + val_base + threadIdx.x]);
    }

    // L2 normalize q and k (sum over the first key_dim threads' squares).
    float q_sq = q_val * q_val;
    q_sq = warp_reduce_sum(q_sq);
    if (lane_id == 0) warp_norms[warp_id] = q_sq;
    __syncthreads();

    if (threadIdx.x == 0) {
        float total = warp_norms[0] + warp_norms[1] + warp_norms[2] + warp_norms[3];
        smem_norm[0] = rsqrtf(total + 1e-12f);
    }
    // Thread 0 above reads warp_norms[0..3] (q partials) while warps 1-3,
    // already past the previous barrier, would otherwise overwrite them with
    // k partials below — order the consumption before the overwrite.
    __syncthreads();

    float k_sq = k_val * k_val;
    k_sq = warp_reduce_sum(k_sq);
    if (lane_id == 0) warp_norms[warp_id] = k_sq;
    __syncthreads();

    if (threadIdx.x == 0) {
        float total = warp_norms[0] + warp_norms[1] + warp_norms[2] + warp_norms[3];
        smem_norm[1] = rsqrtf(total + 1e-12f);
    }
    __syncthreads();

    if (threadIdx.x < GDR_KEY_DIM) {
        smem_q[threadIdx.x] = q_val * smem_norm[0] * rsqrtf((float)key_dim);
        smem_k[threadIdx.x] = k_val * smem_norm[1];
    }

    // Compute g and beta for this value head
    if (threadIdx.x == 0) {
        float a_val = __bfloat162float(a_proj[v_head]);
        float b_val = __bfloat162float(b_proj[v_head]);
        float bias = __bfloat162float(dt_bias[v_head]);
        float a_log = A_log[v_head];

        float x = a_val + bias;
        float softplus_x = (x > 20.0f) ? x : logf(1.0f + expf(x));
        float g = -expf(a_log) * softplus_x;
        s_exp_g = expf(g);
        s_beta = 1.0f / (1.0f + expf(-b_val));
    }
    __syncthreads();

    float exp_g = s_exp_g;
    float beta = s_beta;

    // State pointer — layout [key_dim, val_dim], val_dim contiguous
    float* my_state = state + v_head * key_dim * val_dim;

    int j_start = j_slice * GDR_J_PER_SLICE;

    // Pass 1: decay + partial kv_mem, keeping the decayed tile in registers so
    // pass 2 re-reads no state from DRAM.
    float s_reg[GDR_J_PER_SLICE];
    float partial_kv = 0.0f;
    #pragma unroll
    for (int jj = 0; jj < GDR_J_PER_SLICE; jj++) {
        int j = j_start + jj;
        float s = my_state[j * val_dim + val_idx];
        s *= exp_g;
        s_reg[jj] = s;
        partial_kv += s * smem_k[j];
    }

    // Reduce partial kv_mem across j_slices
    smem_kv_partial[j_slice][val_col] = partial_kv;
    __syncthreads();

    float kv_mem = smem_kv_partial[0][val_col] + smem_kv_partial[1][val_col]
                 + smem_kv_partial[2][val_col] + smem_kv_partial[3][val_col];

    float my_delta = (smem_v[val_col] - kv_mem) * beta;

    // Pass 2: rank-1 update from registers + partial output; single state write.
    float partial_out = 0.0f;
    #pragma unroll
    for (int jj = 0; jj < GDR_J_PER_SLICE; jj++) {
        int j = j_start + jj;
        float s = s_reg[jj] + my_delta * smem_k[j];
        my_state[j * val_dim + val_idx] = s;
        partial_out += s * smem_q[j];
    }

    // Reduce partial output across j_slices, j_slice=0 writes result
    smem_out_partial[j_slice][val_col] = partial_out;
    __syncthreads();

    if (j_slice == 0) {
        float out = smem_out_partial[0][val_col] + smem_out_partial[1][val_col]
                   + smem_out_partial[2][val_col] + smem_out_partial[3][val_col];
        output[v_head * val_dim + val_idx] = __float2bfloat16(out);
    }
}

extern "C" {

cudaError_t gated_delta_rule_decode_cuda(
    const __nv_bfloat16* qkv,
    const __nv_bfloat16* b_proj,
    const __nv_bfloat16* a_proj,
    const __nv_bfloat16* dt_bias,
    const float* A_log,
    float* state,
    __nv_bfloat16* output,
    int num_key_heads,
    int num_value_heads,
    int key_dim,
    int val_dim,
    cudaStream_t stream
) {
    // Fixed-size shared/thread tiling assumes the GDR_* dims (same guard as
    // gated_delta_rule_prefill_recurrent_cuda).
    if (key_dim != GDR_KEY_DIM || val_dim != GDR_VAL_DIM) {
        return cudaErrorInvalidValue;
    }
    // Two blocks per value head (64 val cols × 4 j_slices = 256 threads each).
    gated_delta_rule_decode_kernel<<<num_value_heads * GDR_DEC_VAL_SPLIT,
                                     GDR_DEC_BLOCK_DIM, 0, stream>>>(
        qkv, b_proj, a_proj, dt_bias, A_log,
        state, output,
        num_key_heads, num_value_heads, key_dim, val_dim
    );
    return cudaGetLastError();
}

// Non-null `state_ptrs` selects the varlen form: `blockIdx.y` is one slot,
// scanning `qkv + s*seq_len` and its own b/a capture into its own state.
__global__ void gated_delta_rule_prefill_recurrent_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ b_proj,
    const __nv_bfloat16* __restrict__ a_proj,
    const __nv_bfloat16* const* __restrict__ b_ptrs,
    const __nv_bfloat16* const* __restrict__ a_ptrs,
    const __nv_bfloat16* __restrict__ dt_bias,
    const float* __restrict__ A_log,
    float* __restrict__ state,
    float* const* __restrict__ state_ptrs,
    const int* __restrict__ row_len,
    __nv_bfloat16* __restrict__ output,
    int num_key_heads,
    int num_value_heads,
    int key_dim,
    int val_dim,
    int seq_len
) {
    const int slot = blockIdx.y;
    int row_base = 0;
    if (state_ptrs) {
        state = state_ptrs[slot];
        b_proj = b_ptrs[slot];
        a_proj = a_ptrs[slot];
        row_base = slot * seq_len;
        seq_len = row_len[slot];
    }
    int v_head = blockIdx.x;
    int val_idx = threadIdx.x & 0x7F;
    int j_slice = threadIdx.x >> 7;
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 0x1F;

    int k_head = v_head * num_key_heads / num_value_heads;
    int q_dim_total = key_dim * num_key_heads;
    int k_dim_total = q_dim_total;
    int qkv_stride = q_dim_total + k_dim_total + val_dim * num_value_heads;
    int out_stride = num_value_heads * val_dim;
    // b/a already point at this slot's capture, whose rows start at 0.
    qkv += static_cast<size_t>(row_base) * qkv_stride;
    output += static_cast<size_t>(row_base) * out_stride;

    __shared__ float smem_q[GDR_KEY_DIM];
    __shared__ float smem_k[GDR_KEY_DIM];
    __shared__ float smem_v[GDR_VAL_DIM];
    __shared__ float smem_norm[2];
    __shared__ float warp_norms[GDR_BLOCK_DIM / WARP_SIZE];
    __shared__ float s_exp_g;
    __shared__ float s_beta;
    __shared__ float smem_kv_partial[GDR_J_SLICES][GDR_VAL_DIM];
    __shared__ float smem_out_partial[GDR_J_SLICES][GDR_VAL_DIM];

    float* my_state = state + v_head * key_dim * val_dim;
    int j_start = j_slice * GDR_J_PER_SLICE;
    int j_end = j_start + GDR_J_PER_SLICE;

    for (int token_idx = 0; token_idx < seq_len; ++token_idx) {
        const __nv_bfloat16* token_qkv = qkv + static_cast<size_t>(token_idx) * qkv_stride;

        float q_val = 0.0f;
        float k_val = 0.0f;
        if (j_slice == 0) {
            q_val = __bfloat162float(token_qkv[k_head * key_dim + val_idx]);
            k_val = __bfloat162float(token_qkv[q_dim_total + k_head * key_dim + val_idx]);
            smem_v[val_idx] = __bfloat162float(
                token_qkv[q_dim_total + k_dim_total + v_head * val_dim + val_idx]);
        }

        float q_sq = (j_slice == 0) ? q_val * q_val : 0.0f;
        q_sq = warp_reduce_sum(q_sq);
        if (lane_id == 0) warp_norms[warp_id] = q_sq;
        __syncthreads();

        if (threadIdx.x == 0) {
            float total = warp_norms[0] + warp_norms[1] + warp_norms[2] + warp_norms[3];
            smem_norm[0] = rsqrtf(total + 1e-12f);
        }
        // Same q-vs-k warp_norms ordering hazard as the decode kernel: the
        // q-partial read by thread 0 must complete before warps 1-3 overwrite
        // warp_norms with k partials.
        __syncthreads();

        float k_sq = (j_slice == 0) ? k_val * k_val : 0.0f;
        k_sq = warp_reduce_sum(k_sq);
        if (lane_id == 0) warp_norms[warp_id] = k_sq;
        __syncthreads();

        if (threadIdx.x == 0) {
            float total = warp_norms[0] + warp_norms[1] + warp_norms[2] + warp_norms[3];
            smem_norm[1] = rsqrtf(total + 1e-12f);
        }
        __syncthreads();

        q_val *= smem_norm[0];
        k_val *= smem_norm[1];
        q_val *= rsqrtf((float)key_dim);

        if (j_slice == 0) {
            smem_q[val_idx] = q_val;
            smem_k[val_idx] = k_val;
        }

        if (threadIdx.x == 0) {
            float a_val = __bfloat162float(a_proj[token_idx * num_value_heads + v_head]);
            float b_val = __bfloat162float(b_proj[token_idx * num_value_heads + v_head]);
            float bias = __bfloat162float(dt_bias[v_head]);
            float a_log = A_log[v_head];

            float x = a_val + bias;
            float softplus_x = (x > 20.0f) ? x : logf(1.0f + expf(x));
            float g = -expf(a_log) * softplus_x;
            s_exp_g = expf(g);
            s_beta = 1.0f / (1.0f + expf(-b_val));
        }
        __syncthreads();

        float exp_g = s_exp_g;
        float beta = s_beta;

        float partial_kv = 0.0f;
        for (int j = j_start; j < j_end; j++) {
            float s = my_state[j * val_dim + val_idx];
            s *= exp_g;
            my_state[j * val_dim + val_idx] = s;
            partial_kv += s * smem_k[j];
        }

        smem_kv_partial[j_slice][val_idx] = partial_kv;
        __syncthreads();

        float kv_mem = smem_kv_partial[0][val_idx] + smem_kv_partial[1][val_idx]
                     + smem_kv_partial[2][val_idx] + smem_kv_partial[3][val_idx];
        float my_delta = (smem_v[val_idx] - kv_mem) * beta;

        float partial_out = 0.0f;
        for (int j = j_start; j < j_end; j++) {
            float s = my_state[j * val_dim + val_idx];
            s += my_delta * smem_k[j];
            my_state[j * val_dim + val_idx] = s;
            partial_out += s * smem_q[j];
        }

        smem_out_partial[j_slice][val_idx] = partial_out;
        __syncthreads();

        if (j_slice == 0) {
            float out = smem_out_partial[0][val_idx] + smem_out_partial[1][val_idx]
                      + smem_out_partial[2][val_idx] + smem_out_partial[3][val_idx];
            output[static_cast<size_t>(token_idx) * out_stride + v_head * val_dim + val_idx] =
                __float2bfloat16(out);
        }
        __syncthreads();
    }
}

cudaError_t gated_delta_rule_prefill_recurrent_cuda(
    const __nv_bfloat16* qkv,
    const __nv_bfloat16* b_proj,
    const __nv_bfloat16* a_proj,
    const __nv_bfloat16* dt_bias,
    const float* A_log,
    float* state,
    __nv_bfloat16* output,
    int num_key_heads,
    int num_value_heads,
    int key_dim,
    int val_dim,
    int seq_len,
    cudaStream_t stream
) {
    if (key_dim != GDR_KEY_DIM || val_dim != GDR_VAL_DIM || seq_len < 0) {
        return cudaErrorInvalidValue;
    }
    gated_delta_rule_prefill_recurrent_kernel<<<num_value_heads, GDR_BLOCK_DIM, 0, stream>>>(
        qkv, b_proj, a_proj, nullptr, nullptr, dt_bias, A_log, state, nullptr, nullptr, output,
        num_key_heads, num_value_heads, key_dim, val_dim, seq_len
    );
    return cudaGetLastError();
}

// Varlen twin: slot `s` reads qkv / writes output at its `s*max_len` block.
cudaError_t gated_delta_rule_prefill_recurrent_varlen_cuda(
    const __nv_bfloat16* qkv,
    const __nv_bfloat16* const* b_ptrs,
    const __nv_bfloat16* const* a_ptrs,
    const __nv_bfloat16* dt_bias,
    const float* A_log,
    float* const* state_ptrs,
    const int* row_len,
    __nv_bfloat16* output,
    int num_key_heads,
    int num_value_heads,
    int key_dim,
    int val_dim,
    int max_len,
    int batch,
    cudaStream_t stream
) {
    if (key_dim != GDR_KEY_DIM || val_dim != GDR_VAL_DIM || batch <= 0 || max_len <= 0) {
        return cudaErrorInvalidValue;
    }
    dim3 grid(num_value_heads, batch);
    gated_delta_rule_prefill_recurrent_kernel<<<grid, GDR_BLOCK_DIM, 0, stream>>>(
        qkv, nullptr, nullptr, b_ptrs, a_ptrs, dt_bias, A_log, nullptr, state_ptrs, row_len,
        output, num_key_heads, num_value_heads, key_dim, val_dim, max_len
    );
    return cudaGetLastError();
}

} // extern "C"

// FlashQLA chunked-prefill prep — unpack the fused [q|k|v] projection row
// into the FlashQLA tensor layouts and derive g/beta.
//
// Mirrors the recurrent kernels' math LINE-FOR-LINE (same l2norm eps 1e-12,
// same softplus/sigmoid forms) so the chunked path inherits the licensed
// semantics; the ONLY deliberate difference is that q is NOT pre-scaled by
// rsqrt(key_dim) — FlashQLA bakes scale=K^-0.5 into its fused kernel (applied
// to P and O, mathematically the same place).
//
// Outputs (token-major, dense — TileLang AOT tensors assume contiguous):
//   q_out  [S, Hg, 128] bf16  l2-normalized
//   k_out  [S, Hg, 128] bf16  l2-normalized
//   v_out  [S, H, 128]  bf16  copy (qkv rows are strided by qkv_dim)
//   g_out  [S, H]       f32   -exp(A_log)*softplus(a+dt_bias)  (pre-cumsum)
//   beta_out [S, H]     f32   sigmoid(b)
//
// Grid (S, H): block y handles v-head y (v copy + g/beta); blocks y < Hg
// additionally l2norm key-head y's q/k. 128 threads = one element per
// thread; block reduce via 4-warp partials.

__global__ void gdr_fq_prep_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ b_proj,
    const __nv_bfloat16* __restrict__ a_proj,
    const __nv_bfloat16* __restrict__ dt_bias,
    const float* __restrict__ A_log,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v_out,
    float* __restrict__ g_out,
    float* __restrict__ beta_out,
    int num_key_heads,
    int num_value_heads,
    int seq_len
) {
    const int token = blockIdx.x;
    const int head = blockIdx.y;  // v-head index; also key-head when < Hg
    const int d = threadIdx.x;    // 0..127
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1F;

    const int q_dim_total = num_key_heads * GDR_KEY_DIM;
    const int qkv_stride = 2 * q_dim_total + num_value_heads * GDR_VAL_DIM;
    const __nv_bfloat16* token_qkv = qkv + (size_t)token * qkv_stride;

    // v copy + g/beta (every block: head is a v-head).
    const float v_val =
        __bfloat162float(token_qkv[2 * q_dim_total + head * GDR_VAL_DIM + d]);
    v_out[((size_t)token * num_value_heads + head) * GDR_VAL_DIM + d] =
        __float2bfloat16(v_val);

    if (threadIdx.x == 0) {
        const float a_val =
            __bfloat162float(a_proj[(size_t)token * num_value_heads + head]);
        const float b_val =
            __bfloat162float(b_proj[(size_t)token * num_value_heads + head]);
        const float bias = __bfloat162float(dt_bias[head]);
        const float a_log = A_log[head];
        const float x = a_val + bias;
        const float softplus_x = (x > 20.0f) ? x : logf(1.0f + expf(x));
        g_out[(size_t)token * num_value_heads + head] = -expf(a_log) * softplus_x;
        beta_out[(size_t)token * num_value_heads + head] =
            1.0f / (1.0f + expf(-b_val));
    }

    // q/k l2norm (key-head blocks only).
    if (head >= num_key_heads) return;

    __shared__ float warp_partials[2][GDR_VAL_DIM / WARP_SIZE];  // [q|k][4]
    __shared__ float norms[2];

    const float q_val = __bfloat162float(token_qkv[head * GDR_KEY_DIM + d]);
    const float k_val =
        __bfloat162float(token_qkv[q_dim_total + head * GDR_KEY_DIM + d]);

    float q_sq = warp_reduce_sum(q_val * q_val);
    float k_sq = warp_reduce_sum(k_val * k_val);
    if (lane_id == 0) {
        warp_partials[0][warp_id] = q_sq;
        warp_partials[1][warp_id] = k_sq;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        norms[0] = rsqrtf(warp_partials[0][0] + warp_partials[0][1] +
                          warp_partials[0][2] + warp_partials[0][3] + 1e-12f);
        norms[1] = rsqrtf(warp_partials[1][0] + warp_partials[1][1] +
                          warp_partials[1][2] + warp_partials[1][3] + 1e-12f);
    }
    __syncthreads();

    const size_t qk_base = ((size_t)token * num_key_heads + head) * GDR_KEY_DIM + d;
    q_out[qk_base] = __float2bfloat16(q_val * norms[0]);
    k_out[qk_base] = __float2bfloat16(k_val * norms[1]);
}

extern "C" {

cudaError_t gdr_fq_prep_cuda(
    const __nv_bfloat16* qkv,
    const __nv_bfloat16* b_proj,
    const __nv_bfloat16* a_proj,
    const __nv_bfloat16* dt_bias,
    const float* A_log,
    __nv_bfloat16* q_out,
    __nv_bfloat16* k_out,
    __nv_bfloat16* v_out,
    float* g_out,
    float* beta_out,
    int num_key_heads,
    int num_value_heads,
    int key_dim,
    int val_dim,
    int seq_len,
    cudaStream_t stream
) {
    if (key_dim != GDR_KEY_DIM || val_dim != GDR_VAL_DIM || seq_len <= 0 ||
        num_key_heads <= 0 || num_value_heads < num_key_heads) {
        return cudaErrorInvalidValue;
    }
    dim3 grid(seq_len, num_value_heads);
    gdr_fq_prep_kernel<<<grid, GDR_VAL_DIM, 0, stream>>>(
        qkv, b_proj, a_proj, dt_bias, A_log,
        q_out, k_out, v_out, g_out, beta_out,
        num_key_heads, num_value_heads, seq_len
    );
    return cudaGetLastError();
}

} // extern "C"
