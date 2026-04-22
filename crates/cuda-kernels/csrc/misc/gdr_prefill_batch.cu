#include "common.cuh"
#include <cmath>
#include <cstdint>

extern "C" {

cudaError_t gated_delta_rule_prefill_chunk_prepare_cuda(
    const __nv_bfloat16* qkv,
    const __nv_bfloat16* b_proj,
    const __nv_bfloat16* a_proj,
    const __nv_bfloat16* dt_bias,
    const float* a_log,
    __nv_bfloat16* q_out,
    __nv_bfloat16* k_out,
    __nv_bfloat16* v_out,
    float* g_out,
    float* beta_out,
    int num_key_heads,
    int num_value_heads,
    int qkv_dim,
    int seq_len,
    cudaStream_t stream
);

cudaError_t gated_delta_rule_prefill_chunk_cumsum_cuda(
    const float* g_in,
    float* g_out,
    int seq_len,
    int num_value_heads,
    cudaStream_t stream
);

cudaError_t gated_delta_rule_prefill_chunk_a_cuda(
    const __nv_bfloat16* k,
    const float* g_cumsum,
    const float* beta,
    float* a_tril,
    int seq_len,
    int num_value_heads,
    cudaStream_t stream
);

cudaError_t gated_delta_rule_prefill_chunk_solve_cuda(
    const float* a_tril,
    __nv_bfloat16* a_inv,
    int seq_len,
    int num_value_heads,
    cudaStream_t stream
);

cudaError_t gated_delta_rule_prefill_chunk_recompute_cuda(
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    const float* beta,
    __nv_bfloat16* w,
    __nv_bfloat16* u,
    const __nv_bfloat16* a_inv,
    const float* g_cumsum,
    int seq_len,
    int num_value_heads,
    cudaStream_t stream
);

cudaError_t gated_delta_rule_prefill_chunk_state_cuda(
    const __nv_bfloat16* k,
    const __nv_bfloat16* w,
    const __nv_bfloat16* u,
    const float* g_cumsum,
    const float* initial_state,
    float* chunk_state,
    __nv_bfloat16* v_new,
    float* final_state,
    int seq_len,
    int num_value_heads,
    cudaStream_t stream
);

cudaError_t gated_delta_rule_prefill_chunk_o_cuda(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v_new,
    const float* chunk_state,
    const float* g_cumsum,
    __nv_bfloat16* output,
    int seq_len,
    int num_value_heads,
    float scale,
    cudaStream_t stream
);

} // extern "C"

namespace {

inline cudaError_t launch_chunkwise_for_sequence(
    const __nv_bfloat16* qkv,
    const __nv_bfloat16* b_proj,
    const __nv_bfloat16* a_proj,
    const __nv_bfloat16* dt_bias,
    const float* a_log,
    float* state,
    __nv_bfloat16* q_out,
    __nv_bfloat16* k_out,
    __nv_bfloat16* v_out,
    float* g_cumsum,
    float* beta,
    float* a_tril,
    __nv_bfloat16* a_inv,
    __nv_bfloat16* w,
    __nv_bfloat16* u,
    float* chunk_state,
    __nv_bfloat16* v_new,
    __nv_bfloat16* output,
    int num_key_heads,
    int num_value_heads,
    int key_dim,
    int val_dim,
    int seq_len,
    cudaStream_t stream
) {
    const int qkv_dim = 2 * num_key_heads * key_dim + num_value_heads * val_dim;
    const float scale = rsqrtf(static_cast<float>(key_dim));

    cudaError_t err = gated_delta_rule_prefill_chunk_prepare_cuda(
        qkv,
        b_proj,
        a_proj,
        dt_bias,
        a_log,
        q_out,
        k_out,
        v_out,
        g_cumsum,
        beta,
        num_key_heads,
        num_value_heads,
        qkv_dim,
        seq_len,
        stream
    );
    if (err != cudaSuccess) {
        return err;
    }

    err = gated_delta_rule_prefill_chunk_cumsum_cuda(
        g_cumsum,
        g_cumsum,
        seq_len,
        num_value_heads,
        stream
    );
    if (err != cudaSuccess) {
        return err;
    }

    err = gated_delta_rule_prefill_chunk_a_cuda(
        k_out,
        g_cumsum,
        beta,
        a_tril,
        seq_len,
        num_value_heads,
        stream
    );
    if (err != cudaSuccess) {
        return err;
    }

    err = gated_delta_rule_prefill_chunk_solve_cuda(
        a_tril,
        a_inv,
        seq_len,
        num_value_heads,
        stream
    );
    if (err != cudaSuccess) {
        return err;
    }

    err = gated_delta_rule_prefill_chunk_recompute_cuda(
        k_out,
        v_out,
        beta,
        w,
        u,
        a_inv,
        g_cumsum,
        seq_len,
        num_value_heads,
        stream
    );
    if (err != cudaSuccess) {
        return err;
    }

    err = gated_delta_rule_prefill_chunk_state_cuda(
        k_out,
        w,
        u,
        g_cumsum,
        state,
        chunk_state,
        v_new,
        state,
        seq_len,
        num_value_heads,
        stream
    );
    if (err != cudaSuccess) {
        return err;
    }

    return gated_delta_rule_prefill_chunk_o_cuda(
        q_out,
        k_out,
        v_new,
        chunk_state,
        g_cumsum,
        output,
        seq_len,
        num_value_heads,
        scale,
        stream
    );
}

} // namespace

extern "C" {

cudaError_t gated_delta_rule_prefill_chunkwise_batch_cuda(
    const __nv_bfloat16* qkv_batch,
    const __nv_bfloat16* b_proj_batch,
    const __nv_bfloat16* a_proj_batch,
    const __nv_bfloat16* dt_bias,
    const float* a_log,
    const uint64_t* state_ptrs,
    const uint64_t* q_ptrs,
    const uint64_t* k_ptrs,
    const uint64_t* v_ptrs,
    const uint64_t* g_cumsum_ptrs,
    const uint64_t* beta_ptrs,
    const uint64_t* a_tril_ptrs,
    const uint64_t* a_inv_ptrs,
    const uint64_t* w_ptrs,
    const uint64_t* u_ptrs,
    const uint64_t* chunk_state_ptrs,
    const uint64_t* v_new_ptrs,
    const int32_t* seq_indptr,
    __nv_bfloat16* output_batch,
    int num_key_heads,
    int num_value_heads,
    int key_dim,
    int val_dim,
    int batch_size,
    cudaStream_t stream
) {
    if (batch_size < 0 || seq_indptr == nullptr) {
        return cudaErrorInvalidValue;
    }

    const int qkv_dim = 2 * num_key_heads * key_dim + num_value_heads * val_dim;
    const int head_dim = num_value_heads;
    const int output_dim = num_value_heads * val_dim;

    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        const int token_start = seq_indptr[batch_idx];
        const int token_end = seq_indptr[batch_idx + 1];
        const int seq_len = token_end - token_start;
        if (seq_len <= 0) {
            return cudaErrorInvalidValue;
        }

        const __nv_bfloat16* qkv = qkv_batch + static_cast<size_t>(token_start) * qkv_dim;
        const __nv_bfloat16* b_proj =
            b_proj_batch + static_cast<size_t>(token_start) * head_dim;
        const __nv_bfloat16* a_proj =
            a_proj_batch + static_cast<size_t>(token_start) * head_dim;
        __nv_bfloat16* output =
            output_batch + static_cast<size_t>(token_start) * output_dim;

        cudaError_t err = launch_chunkwise_for_sequence(
            qkv,
            b_proj,
            a_proj,
            dt_bias,
            a_log,
            reinterpret_cast<float*>(state_ptrs[batch_idx]),
            reinterpret_cast<__nv_bfloat16*>(q_ptrs[batch_idx]),
            reinterpret_cast<__nv_bfloat16*>(k_ptrs[batch_idx]),
            reinterpret_cast<__nv_bfloat16*>(v_ptrs[batch_idx]),
            reinterpret_cast<float*>(g_cumsum_ptrs[batch_idx]),
            reinterpret_cast<float*>(beta_ptrs[batch_idx]),
            reinterpret_cast<float*>(a_tril_ptrs[batch_idx]),
            reinterpret_cast<__nv_bfloat16*>(a_inv_ptrs[batch_idx]),
            reinterpret_cast<__nv_bfloat16*>(w_ptrs[batch_idx]),
            reinterpret_cast<__nv_bfloat16*>(u_ptrs[batch_idx]),
            reinterpret_cast<float*>(chunk_state_ptrs[batch_idx]),
            reinterpret_cast<__nv_bfloat16*>(v_new_ptrs[batch_idx]),
            output,
            num_key_heads,
            num_value_heads,
            key_dim,
            val_dim,
            seq_len,
            stream
        );
        if (err != cudaSuccess) {
            return err;
        }
    }

    return cudaSuccess;
}

} // extern "C"
