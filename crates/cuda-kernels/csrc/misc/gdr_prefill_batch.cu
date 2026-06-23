#include "common.cuh"
#include <cmath>
#include <cstdint>
#include <cstdlib>

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
);

} // extern "C"
