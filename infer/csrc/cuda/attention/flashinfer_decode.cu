// FlashInfer batch-decode wrapper for Rust FFI
// Wraps BatchDecodeWithPagedKVCacheDispatched for bf16, HEAD_DIM=128, PosEncodingMode::kNone
//
// Build with:
//   nvcc -std=c++17 -c -O2
//     -I/usr/local/lib/python3.12/dist-packages/flashinfer/data/include
//     -gencode arch=compute_80,code=sm_80
//     flashinfer_decode.cu -o flashinfer_decode.o

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

constexpr uint32_t HEAD_DIM = 128;
constexpr flashinfer::PosEncodingMode POS_MODE = flashinfer::PosEncodingMode::kNone;

// DefaultAttention<use_custom_mask=false, use_sliding_window=false,
//                  use_logits_soft_cap=false, use_alibi=false>
using Variant = flashinfer::DefaultAttention<false, false, false, false>;
using Params  = flashinfer::BatchDecodeParams<DType, DType, DType, IdType>;

// ---------------------------------------------------------------------------
// flashinfer_batch_decode_plan  –  CPU-side scheduling
// ---------------------------------------------------------------------------
extern "C" int flashinfer_batch_decode_plan(
    void*         float_workspace,        // GPU scratch
    size_t        float_workspace_bytes,
    void*         int_workspace,           // GPU scratch
    void*         page_locked_workspace,   // CPU pinned scratch
    size_t        int_workspace_bytes,
    int32_t*      indptr_h,                // [batch_size+1] host
    int           batch_size,
    int           num_qo_heads,
    int           num_kv_heads,
    int           page_size,
    int           head_dim,
    void*         plan_info_out,           // opaque buffer >= sizeof(DecodePlanInfo)
    cudaStream_t  stream)
{
    (void)num_kv_heads; // used implicitly inside WorkEstimation via num_qo_heads / GROUP_SIZE
    (void)head_dim;     // fixed at 128

    flashinfer::DecodePlanInfo plan_info;

    // GROUP_SIZE in WorkEstimation must match actual GQA ratio.
    uint32_t gqa_group = uint32_t(num_qo_heads) / uint32_t(num_kv_heads);

    // We need to create a work estimation lambda that uses the correct GROUP_SIZE.
    // Since DecodePlan is templated we use a macro-style dispatch.
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

    // Copy plan_info to caller's opaque buffer
    std::memcpy(plan_info_out, &plan_info, sizeof(flashinfer::DecodePlanInfo));
    return 0;
}

// ---------------------------------------------------------------------------
// flashinfer_batch_decode_run  –  launch the GPU kernel
// ---------------------------------------------------------------------------
extern "C" int flashinfer_batch_decode_run(
    void*          float_workspace,
    void*          int_workspace,
    void*          plan_info_ptr,         // opaque, from plan
    __nv_bfloat16* q,
    __nv_bfloat16* k_data,
    __nv_bfloat16* v_data,
    int32_t*       kv_indptr,             // GPU
    int32_t*       kv_indices,            // GPU
    int32_t*       kv_last_page_len,      // GPU
    __nv_bfloat16* o,
    float*         lse,                   // nullable
    int            batch_size,
    int            num_qo_heads,
    int            num_kv_heads,
    int            page_size,
    int            head_dim,
    float          sm_scale,
    cudaStream_t   stream)
{
    (void)head_dim; // fixed at 128

    // Recover plan info
    flashinfer::DecodePlanInfo plan_info;
    std::memcpy(&plan_info, plan_info_ptr, sizeof(flashinfer::DecodePlanInfo));

    // Build paged_kv_t (HND layout)
    flashinfer::paged_kv_t<DType, IdType> paged_kv(
        /*num_heads=*/    (uint32_t)num_kv_heads,
        /*page_size=*/    (uint32_t)page_size,
        /*head_dim=*/     HEAD_DIM,
        /*batch_size=*/   (uint32_t)batch_size,
        /*layout=*/       flashinfer::QKVLayout::kHND,
        /*k_data=*/       k_data,
        /*v_data=*/       v_data,
        /*indices=*/      kv_indices,
        /*indptr=*/       kv_indptr,
        /*last_page_len=*/kv_last_page_len,
        /*rope_pos_offset=*/nullptr);

    // Build BatchDecodeParams via the parameterised constructor
    Params params(
        /*q=*/                q,
        /*q_rope_offset=*/    nullptr,
        /*paged_kv=*/         paged_kv,
        /*o=*/                o,
        /*lse=*/              lse,
        /*maybe_alibi_slopes=*/nullptr,
        /*num_qo_heads=*/     (uint32_t)num_qo_heads,
        /*q_stride_n=*/       (IdType)(num_qo_heads * HEAD_DIM),
        /*q_stride_h=*/       (IdType)HEAD_DIM,
        /*window_left=*/      -1,
        /*logits_soft_cap=*/  0.f,
        /*sm_scale=*/         sm_scale,
        /*rope_scale=*/       1.f,
        /*rope_theta=*/       1e4f);

    // Fill in plan fields from int_workspace
    params.padded_batch_size = (uint32_t)plan_info.padded_batch_size;

    params.request_indices  = flashinfer::GetPtrFromBaseOffset<IdType>(
        int_workspace, plan_info.request_indices_offset);
    params.kv_tile_indices  = flashinfer::GetPtrFromBaseOffset<IdType>(
        int_workspace, plan_info.kv_tile_indices_offset);
    params.o_indptr         = flashinfer::GetPtrFromBaseOffset<IdType>(
        int_workspace, plan_info.o_indptr_offset);
    params.kv_chunk_size_ptr = flashinfer::GetPtrFromBaseOffset<IdType>(
        int_workspace, plan_info.kv_chunk_size_ptr_offset);

    if (plan_info.split_kv) {
        params.block_valid_mask = flashinfer::GetPtrFromBaseOffset<bool>(
            int_workspace, plan_info.block_valid_mask_offset);
    } else {
        params.block_valid_mask = nullptr;
    }

    // Determine tmp buffers for split-kv
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
