// FlashMLA SM90 decode FFI stubs — compiled only when
// ARLE_CUDA_DISABLE_FLASHMLA=1 (or vendor/flashmla missing). Provides the
// `arle_flashmla_sm90_sparse_decode_*` symbol set so the Rust crate can
// link on SM89-only boxes (4070 Ti SUPER, etc.) where the real FlashMLA
// kernels cannot compile (SM90 launch_bounds + WGMMA-specific code).
//
// All entry points return cudaErrorNotSupported / -1, matching the
// runtime gate in `infer/src/model/deepseek/weights.rs::dsv4_flashmla_decode_enabled`
// (default OFF). Callers that opt in via `ARLE_DSV4_FLASHMLA_DECODE=1`
// on an SM89 box will hit the runtime error at the FFI boundary instead
// of a static link failure.
//
// The actual FlashMLA-built shims are at
// `csrc/attention/arle_flashmla_decode_shim.cu` + the vendored kernels;
// build.rs picks exactly one of (real shim, this stub) based on
// `enable_flashmla`.

#include <cuda_runtime.h>
#include <cstdint>

// half is `__half` from cuda_fp16, but the FFI types are pointer-only,
// so we don't need to include cuda_fp16.h here — opaque pointers suffice.
struct __half;

extern "C" {

cudaError_t arle_flashmla_sm90_sparse_decode_fwd(
    const __half* /*q*/,
    const __half* /*kv*/,
    const int32_t* /*indices*/,
    const int32_t* /*topk_length*/,
    const float* /*attn_sink*/,
    __half* /*out*/,
    float* /*lse*/,
    float* /*lse_accum*/,
    float* /*o_accum*/,
    const int32_t* /*tile_scheduler_metadata*/,
    const int32_t* /*num_splits*/,
    int32_t /*b*/, int32_t /*s_q*/, int32_t /*h_q*/, int32_t /*h_kv*/,
    int32_t /*d_qk*/, int32_t /*d_v*/, int32_t /*num_blocks*/,
    int32_t /*page_block_size*/, int32_t /*topk*/, int32_t /*num_sm_parts*/,
    int32_t /*model_type_int*/, float /*sm_scale*/,
    int32_t /*stride_q_b*/, int32_t /*stride_q_s_q*/, int32_t /*stride_q_h_q*/,
    int32_t /*stride_kv_block_bytes*/, int32_t /*stride_kv_row_bytes*/,
    int32_t /*stride_indices_b*/, int32_t /*stride_indices_s_q*/,
    int32_t /*stride_lse_b*/, int32_t /*stride_lse_s_q*/,
    int32_t /*stride_o_b*/, int32_t /*stride_o_s_q*/, int32_t /*stride_o_h_q*/,
    int32_t /*stride_lse_accum_split*/, int32_t /*stride_lse_accum_s_q*/,
    int32_t /*stride_o_accum_split*/, int32_t /*stride_o_accum_s_q*/,
    int32_t /*stride_o_accum_h_q*/,
    CUstream_st* /*stream*/
) {
    return cudaErrorNotSupported;
}

int32_t arle_flashmla_sm90_sparse_decode_bytes_per_token(int32_t /*d_qk*/, int32_t /*model_type_int*/) {
    return -1;
}

cudaError_t arle_flashmla_sm90_sparse_decode_get_meta(
    int32_t /*h_q*/, int32_t /*s_q*/, int32_t /*model_type_int*/,
    int32_t* out_num_sm_parts,
    int32_t* out_fixed_overhead_num_blocks,
    int32_t* out_block_size_topk
) {
    if (out_num_sm_parts) *out_num_sm_parts = 0;
    if (out_fixed_overhead_num_blocks) *out_fixed_overhead_num_blocks = 0;
    if (out_block_size_topk) *out_block_size_topk = 0;
    return cudaErrorNotSupported;
}

cudaError_t arle_flashmla_sm90_sparse_decode_sched_meta(
    int32_t /*b*/, int32_t /*s_q*/, int32_t /*block_size_topk*/,
    int32_t /*fixed_overhead_num_blocks*/, int32_t /*topk*/, int32_t /*extra_topk*/,
    const int32_t* /*topk_length*/, const int32_t* /*extra_topk_length*/,
    int32_t* /*tile_scheduler_metadata*/, int32_t* /*num_splits*/,
    int32_t /*num_sm_parts*/, CUstream_st* /*stream*/
) {
    return cudaErrorNotSupported;
}

}  // extern "C"
