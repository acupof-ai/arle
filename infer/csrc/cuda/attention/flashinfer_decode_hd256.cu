// FlashInfer batch-decode wrapper for Rust FFI — HEAD_DIM=256 variant
// Wraps BatchDecodeWithPagedKVCacheDispatched for bf16, HEAD_DIM=256, PosEncodingMode::kNone
// Used by Qwen3.5 full attention layers (head_dim=256, partial RoPE applied externally).

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>

#include <flashinfer/pos_enc.cuh>
#include <flashinfer/page.cuh>
#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/decode.cuh>

// ---------------------------------------------------------------------------
// Fixed template parameters
// ---------------------------------------------------------------------------
using DType   = __nv_bfloat16;
using IdType  = int32_t;

constexpr uint32_t HEAD_DIM = 256;
constexpr flashinfer::PosEncodingMode POS_MODE = flashinfer::PosEncodingMode::kNone;

using Variant = flashinfer::DefaultAttention<false, false, false, false>;
using Params  = flashinfer::BatchDecodeParams<DType, DType, DType, IdType>;

// ---------------------------------------------------------------------------
// flashinfer_batch_decode_hd256_plan  –  CPU-side scheduling
// ---------------------------------------------------------------------------
extern "C" int flashinfer_batch_decode_hd256_plan(
    void*         float_workspace,
    size_t        float_workspace_bytes,
    void*         int_workspace,
    void*         page_locked_workspace,
    size_t        int_workspace_bytes,
    int32_t*      indptr_h,
    int           batch_size,
    int           num_qo_heads,
    int           num_kv_heads,
    int           page_size,
    int           head_dim,
    void*         plan_info_out,
    cudaStream_t  stream)
{
    (void)head_dim;  // fixed at 256

    flashinfer::DecodePlanInfo plan_info;
    uint32_t gqa_group = uint32_t(num_qo_heads) / uint32_t(num_kv_heads);

    cudaError_t err;

#define DISPATCH_PLAN(GS)                                                                      \
    do {                                                                                       \
        auto work_est_gs = [](bool& split_kv, uint32_t& max_grid_size,                        \
                              uint32_t& max_num_pages_per_batch, uint32_t& new_batch_size,     \
                              uint32_t& gdy, uint32_t bs, IdType* indptr,                      \
                              uint32_t nqo, uint32_t ps, bool enable_cg,                       \
                              cudaStream_t s) -> cudaError_t {                                 \
            return flashinfer::BatchDecodeWithPagedKVCacheWorkEstimationDispatched<             \
                GS, HEAD_DIM, POS_MODE, Variant, Params>(                                      \
                split_kv, max_grid_size, max_num_pages_per_batch, new_batch_size,              \
                gdy, bs, indptr, nqo, ps, enable_cg, s);                                       \
        };                                                                                     \
        err = flashinfer::DecodePlan<HEAD_DIM, POS_MODE, Variant, Params>(                     \
            float_workspace, float_workspace_bytes,                                            \
            int_workspace, page_locked_workspace, int_workspace_bytes,                         \
            plan_info, indptr_h, (uint32_t)batch_size,                                         \
            (uint32_t)num_qo_heads, (uint32_t)page_size,                                       \
            /*enable_cuda_graph=*/false, stream, work_est_gs);                                 \
    } while (0)

    switch (gqa_group) {
        case 1:  DISPATCH_PLAN(1);  break;
        case 2:  DISPATCH_PLAN(2);  break;
        case 4:  DISPATCH_PLAN(4);  break;
        case 8:  DISPATCH_PLAN(8);  break;
        default: DISPATCH_PLAN(1);  break;
    }
#undef DISPATCH_PLAN

    if (err != cudaSuccess) return (int)err;

    std::memcpy(plan_info_out, &plan_info, sizeof(flashinfer::DecodePlanInfo));
    return 0;
}

// ---------------------------------------------------------------------------
// flashinfer_batch_decode_hd256_run  –  launch the GPU kernel
// ---------------------------------------------------------------------------
extern "C" int flashinfer_batch_decode_hd256_run(
    void*          float_workspace,
    void*          int_workspace,
    void*          plan_info_ptr,
    __nv_bfloat16* q,
    __nv_bfloat16* k_data,
    __nv_bfloat16* v_data,
    int32_t*       kv_indptr,
    int32_t*       kv_indices,
    int32_t*       kv_last_page_len,
    __nv_bfloat16* o,
    float*         lse,
    int            batch_size,
    int            num_qo_heads,
    int            num_kv_heads,
    int            page_size,
    int            head_dim,
    float          sm_scale,
    cudaStream_t   stream)
{
    (void)head_dim;  // fixed at 256

    flashinfer::DecodePlanInfo plan_info;
    std::memcpy(&plan_info, plan_info_ptr, sizeof(flashinfer::DecodePlanInfo));

    flashinfer::paged_kv_t<DType, IdType> paged_kv(
        (uint32_t)num_kv_heads,
        (uint32_t)page_size,
        HEAD_DIM,
        (uint32_t)batch_size,
        flashinfer::QKVLayout::kHND,
        k_data, v_data,
        kv_indices, kv_indptr, kv_last_page_len,
        /*rope_pos_offset=*/nullptr);

    Params params(
        q, nullptr, paged_kv, o, lse, nullptr,
        (uint32_t)num_qo_heads,
        (IdType)(num_qo_heads * HEAD_DIM),
        (IdType)HEAD_DIM,
        -1, 0.f, sm_scale, 1.f, 1e4f);

    params.padded_batch_size = (uint32_t)plan_info.padded_batch_size;

    params.request_indices = flashinfer::GetPtrFromBaseOffset<IdType>(
        int_workspace, plan_info.request_indices_offset);
    params.kv_tile_indices = flashinfer::GetPtrFromBaseOffset<IdType>(
        int_workspace, plan_info.kv_tile_indices_offset);
    params.o_indptr = flashinfer::GetPtrFromBaseOffset<IdType>(
        int_workspace, plan_info.o_indptr_offset);
    params.kv_chunk_size_ptr = flashinfer::GetPtrFromBaseOffset<IdType>(
        int_workspace, plan_info.kv_chunk_size_ptr_offset);

    if (plan_info.split_kv) {
        params.block_valid_mask = flashinfer::GetPtrFromBaseOffset<bool>(
            int_workspace, plan_info.block_valid_mask_offset);
    } else {
        params.block_valid_mask = nullptr;
    }

    DType* tmp_v = nullptr;
    float* tmp_s = nullptr;
    if (plan_info.split_kv) {
        tmp_v = flashinfer::GetPtrFromBaseOffset<DType>(
            float_workspace, plan_info.v_offset);
        tmp_s = flashinfer::GetPtrFromBaseOffset<float>(
            float_workspace, plan_info.s_offset);
    }

    cudaError_t err = flashinfer::BatchDecodeWithPagedKVCacheDispatched<
        HEAD_DIM, POS_MODE, Variant, Params>(
        params, tmp_v, tmp_s, /*enable_pdl=*/false, stream);

    return (int)err;
}
