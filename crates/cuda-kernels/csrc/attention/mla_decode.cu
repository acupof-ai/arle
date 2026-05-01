// DeepSeek-family MLA paged decode skeleton.
//
// This file reserves the CUDA ABI for DS3 without implementing the kernel.
// MLA uses compressed KV (`ckv_pool`) plus RoPE key payload (`kpe_pool`), so it
// must remain separate from the expanded-GQA kernels in
// `decode_attention_varlen_fp8.cu`.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

extern "C" int mla_decode_paged_bf16_cuda(
    const __nv_bfloat16* q_nope_or_absorbed,
    const __nv_bfloat16* q_rope,
    const __nv_bfloat16* ckv_pool,
    const __nv_bfloat16* kpe_pool,
    const __nv_bfloat16* w_uv_or_absorbed,
    const int32_t* qo_indptr,
    const int32_t* kv_indptr,
    const int32_t* kv_indices,
    const int32_t* last_page_len,
    __nv_bfloat16* out,
    int32_t total_q_tokens,
    int32_t batch_size,
    int32_t num_q_heads,
    int32_t kv_lora_rank,
    int32_t qk_rope_head_dim,
    int32_t v_head_dim,
    int32_t page_size,
    float sm_scale,
    int32_t num_splits,
    cudaStream_t stream)
{
    (void)q_nope_or_absorbed;
    (void)q_rope;
    (void)ckv_pool;
    (void)kpe_pool;
    (void)w_uv_or_absorbed;
    (void)qo_indptr;
    (void)kv_indptr;
    (void)kv_indices;
    (void)last_page_len;
    (void)out;
    (void)total_q_tokens;
    (void)batch_size;
    (void)num_q_heads;
    (void)kv_lora_rank;
    (void)qk_rope_head_dim;
    (void)v_head_dim;
    (void)page_size;
    (void)sm_scale;
    (void)num_splits;
    (void)stream;

    // TODO(DS3): implement BF16 paged MLA decode over compressed c_kv + k_rope.
    // The first real implementation should follow
    // docs/plans/2026-05-01-mla-kernel-design.md:
    // - split-KV partial + merge scheduling from decode_attention_varlen_fp8.cu;
    // - latent cache layout [page, token, kv_lora_rank] for ckv_pool;
    // - RoPE key payload [page, token, qk_rope_head_dim] for kpe_pool;
    // - output shape [total_q, num_q_heads, v_head_dim].
    return static_cast<int>(cudaErrorNotSupported);
}
