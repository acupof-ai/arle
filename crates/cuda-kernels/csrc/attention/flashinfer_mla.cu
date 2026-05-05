// FlashInfer MLA paged-attention wrapper for Rust FFI.
//
// Wraps `flashinfer::MLAPlan` (CPU-side scheduling) + `flashinfer::mla::
// BatchMLAPagedAttention` (GPU launch) for BF16 q_nope / q_pe / ckv / kpe
// over paged KV. Mirrors the structure of `flashinfer_decode.cu` and
// `flashinfer_prefill_paged.cu`: one `_plan` entry that fills an opaque
// 256-byte plan buffer, one `_run` entry that consumes the plan and
// launches the kernel.
//
// Build: picked up automatically by `crates/cuda-kernels/build.rs`'s
// recursive csrc walk. `flashinfer_*` filename prefix triggers the
// FlashInfer `-I<flashinfer/data/include>` flag block.
//
// MLA cache layout (matches FlashInfer's `BatchMLAPagedAttentionWrapper`):
//   q_nope:   [total_q_tokens, num_q_heads, head_dim_ckv]   — bf16
//   q_pe:     [total_q_tokens, num_q_heads, head_dim_kpe]   — bf16
//   ckv:      [num_pages, page_size, head_dim_ckv]          — bf16
//   kpe:      [num_pages, page_size, head_dim_kpe]          — bf16
//   o:        [total_q_tokens, num_q_heads, head_dim_ckv]   — bf16
//
// HEAD_DIM_CKV / HEAD_DIM_KPE pair support — current FA2 MLA kernel
// (`flashinfer/attention/mla.cuh`) requires (verified against the kernel
// loop bounds at the FlashInfer 0.6.9 headers shipped here):
//   - HEAD_DIM_CKV >= 128  (output store loop uses NUM_MMA_D_CKV / 8 with
//                          NUM_MMA_D_CKV = HEAD_DIM_CKV / 16; values < 128
//                          truncate the loop to zero iterations and silently
//                          drop output writes — see mla.cuh:730 / mla.cuh:769.)
//   - HEAD_DIM_KPE >= 64   (Q-PE / K-PE load loops use NUM_MMA_D_KPE / 4
//                          with NUM_MMA_D_KPE = HEAD_DIM_KPE / 16; smaller
//                          KPE truncates to zero and silently drops PE — see
//                          mla.cuh:167 / mla.cuh:221 / mla.cuh:255.)
//
// The DeepSeek V2 / V3 reference shape `(512, 64)` is the only AOT-supported
// pair upstream and the only pair this wrapper currently dispatches. The
// DSV4 small-substrate SKUs in
// `docs/plans/2026-05-05-deepseek-v4-small-substrate.md` §2 use
// `(64, 16) / (128, 32) / (192, 32)` — those need a different MLA kernel
// (the cute_dsl SM80 path or a custom variant), tracked as future work in
// the same plan.
//
// Add new (CKV, KPE) tuples to DISPATCH_MLA_DIMS only after verifying the
// dim pair satisfies the divisibility constraints above (or a different
// kernel is wired through this entry point).

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>

#include <flashinfer/attention/mla.cuh>
#include <flashinfer/attention/mla_params.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/fastdiv.cuh>

using DTypeQ = __nv_bfloat16;
using DTypeKV = __nv_bfloat16;
using DTypeO = __nv_bfloat16;
using IdType = int32_t;

using MLAParamsT = flashinfer::MLAParams<DTypeQ, DTypeKV, DTypeO, IdType>;

// Dispatch over compile-time (HEAD_DIM_CKV, HEAD_DIM_KPE) pairs.
//
// Currently the only safe pair is `(512, 64)` — see the constraint comment
// above. The macro emits the kernel call only for that pair; everything
// else falls through to a `return cudaErrorInvalidConfiguration` at the
// call site so the model layer fails loudly instead of silently producing
// wrong attention output.
//
// Each arm ends in `return` (the kernel launch returns its `cudaError_t`
// cast to int).
#define DISPATCH_MLA_DIMS(HEAD_DIM_CKV, HEAD_DIM_KPE, ...)                               \
    if ((HEAD_DIM_CKV) == 512 && (HEAD_DIM_KPE) == 64) {                                 \
        constexpr uint32_t HEAD_DIM_CKV_C = 512;                                         \
        constexpr uint32_t HEAD_DIM_KPE_C = 64;                                          \
        __VA_ARGS__                                                                       \
    }

// ---------------------------------------------------------------------------
// flashinfer_mla_paged_attention_plan – CPU-side scheduling
// ---------------------------------------------------------------------------
//
// Produces an `MLAPlanInfo` (18×i64 = 144 bytes) into the caller-provided
// 256-byte `plan_info_out` buffer. Same workspace conventions as the
// FlashInfer prefill/decode wrappers: float_workspace + int_workspace are
// device buffers, page_locked_workspace is host-pinned.
//
// `qo_indptr_h`, `kv_indptr_h`, `kv_len_h` must be host arrays of length
// batch_size+1, batch_size+1, batch_size respectively.
extern "C" int flashinfer_mla_paged_attention_plan(
    void*        float_workspace,
    size_t       float_workspace_bytes,
    void*        int_workspace,
    void*        page_locked_workspace,
    size_t       int_workspace_bytes,
    int32_t*     qo_indptr_h,
    int32_t*     kv_indptr_h,
    int32_t*     kv_len_h,
    int          batch_size,
    int          num_heads,
    int          head_dim_ckv,
    int          head_dim_kpe,
    int          causal,
    void*        plan_info_out,
    cudaStream_t stream)
{
    (void)head_dim_kpe; // plan only needs head_dim_o == head_dim_ckv

    if (batch_size <= 0 || num_heads <= 0 || head_dim_ckv <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    flashinfer::MLAPlanInfo plan_info;
    cudaError_t err = flashinfer::MLAPlan<IdType>(
        float_workspace, float_workspace_bytes,
        int_workspace, page_locked_workspace, int_workspace_bytes,
        plan_info,
        qo_indptr_h, kv_indptr_h, kv_len_h,
        static_cast<uint32_t>(batch_size),
        static_cast<uint32_t>(num_heads),
        static_cast<uint32_t>(head_dim_ckv),
        causal != 0,
        stream);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    // The plan_info struct itself is layout-compatible with the
    // 18×int64 ToVector representation; copy directly into the caller
    // buffer (matches how flashinfer_decode.cu serializes DecodePlanInfo).
    static_assert(sizeof(flashinfer::MLAPlanInfo) <= 256,
                  "MLAPlanInfo must fit in the 256-byte plan_info buffer");
    std::memcpy(plan_info_out, &plan_info, sizeof(flashinfer::MLAPlanInfo));
    return 0;
}

// ---------------------------------------------------------------------------
// flashinfer_mla_paged_attention_run – launch the GPU kernel
// ---------------------------------------------------------------------------
//
// Strides match the contiguous defaults for the canonical MLA shapes (the
// same convention `flashinfer/data/csrc/batch_mla_run.cu` reads off
// `tensor.stride(0/1)`).
extern "C" int flashinfer_mla_paged_attention_run(
    void*          float_workspace,
    void*          int_workspace,
    const void*    plan_info_ptr,
    __nv_bfloat16* q_nope,
    __nv_bfloat16* q_pe,
    __nv_bfloat16* ckv,
    __nv_bfloat16* kpe,
    int32_t*       kv_indices,        // GPU
    __nv_bfloat16* o,
    float*         lse,                // nullable
    int            num_heads,
    int            page_size,
    int            head_dim_ckv,
    int            head_dim_kpe,
    int            causal,
    float          sm_scale,
    cudaStream_t   stream)
{
    if (num_heads <= 0 || page_size <= 0 || head_dim_ckv <= 0 || head_dim_kpe <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    flashinfer::MLAPlanInfo plan_info;
    std::memcpy(&plan_info, plan_info_ptr, sizeof(flashinfer::MLAPlanInfo));

    DISPATCH_MLA_DIMS(head_dim_ckv, head_dim_kpe, {
        MLAParamsT params;

        params.q_nope = q_nope;
        params.q_pe = q_pe;
        params.ckv = ckv;
        params.kpe = kpe;

        params.q_indptr =
            flashinfer::GetPtrFromBaseOffset<IdType>(int_workspace, plan_info.q_indptr_offset);
        params.kv_indptr =
            flashinfer::GetPtrFromBaseOffset<IdType>(int_workspace, plan_info.kv_indptr_offset);
        params.partial_indptr = flashinfer::GetPtrFromBaseOffset<IdType>(
            int_workspace, plan_info.partial_indptr_offset);
        params.kv_indices = kv_indices;
        params.q_len =
            flashinfer::GetPtrFromBaseOffset<IdType>(int_workspace, plan_info.q_len_offset);
        params.kv_len =
            flashinfer::GetPtrFromBaseOffset<IdType>(int_workspace, plan_info.kv_len_offset);
        params.q_start =
            flashinfer::GetPtrFromBaseOffset<IdType>(int_workspace, plan_info.q_start_offset);
        params.kv_start =
            flashinfer::GetPtrFromBaseOffset<IdType>(int_workspace, plan_info.kv_start_offset);
        params.kv_end =
            flashinfer::GetPtrFromBaseOffset<IdType>(int_workspace, plan_info.kv_end_offset);
        params.work_indptr = flashinfer::GetPtrFromBaseOffset<IdType>(
            int_workspace, plan_info.work_indptr_offset);
        params.merge_packed_offset_start = flashinfer::GetPtrFromBaseOffset<IdType>(
            int_workspace, plan_info.merge_packed_offset_start_offset);
        params.merge_packed_offset_end = flashinfer::GetPtrFromBaseOffset<IdType>(
            int_workspace, plan_info.merge_packed_offset_end_offset);
        params.merge_partial_packed_offset_start = flashinfer::GetPtrFromBaseOffset<IdType>(
            int_workspace, plan_info.merge_partial_packed_offset_start_offset);
        params.merge_partial_packed_offset_end = flashinfer::GetPtrFromBaseOffset<IdType>(
            int_workspace, plan_info.merge_partial_packed_offset_end_offset);
        params.merge_partial_stride = flashinfer::GetPtrFromBaseOffset<IdType>(
            int_workspace, plan_info.merge_partial_stride_offset);

        params.final_o = o;
        params.final_lse = lse;
        params.partial_o = flashinfer::GetPtrFromBaseOffset<DTypeO>(
            float_workspace, plan_info.partial_o_offset);
        params.partial_lse = flashinfer::GetPtrFromBaseOffset<float>(
            float_workspace, plan_info.partial_lse_offset);

        params.num_heads = flashinfer::uint_fastdiv(static_cast<uint32_t>(num_heads));
        params.block_size = flashinfer::uint_fastdiv(static_cast<uint32_t>(page_size));

        // Contiguous strides for the canonical [n, num_heads, head_dim_*]
        // q tensors and [num_pages, page_size, head_dim_*] caches.
        params.q_nope_stride_n = static_cast<uint32_t>(num_heads) * HEAD_DIM_CKV_C;
        params.q_nope_stride_h = HEAD_DIM_CKV_C;
        params.q_pe_stride_n = static_cast<uint32_t>(num_heads) * HEAD_DIM_KPE_C;
        params.q_pe_stride_h = HEAD_DIM_KPE_C;
        params.ckv_stride_page = static_cast<uint32_t>(page_size) * HEAD_DIM_CKV_C;
        params.ckv_stride_n = HEAD_DIM_CKV_C;
        params.kpe_stride_page = static_cast<uint32_t>(page_size) * HEAD_DIM_KPE_C;
        params.kpe_stride_n = HEAD_DIM_KPE_C;
        params.o_stride_n = static_cast<uint32_t>(num_heads) * HEAD_DIM_CKV_C;
        params.o_stride_h = HEAD_DIM_CKV_C;

        params.sm_scale = sm_scale;
        params.return_lse_base_on_e = false;

        const flashinfer::MaskMode mask_mode =
            (causal != 0) ? flashinfer::MaskMode::kCausal : flashinfer::MaskMode::kNone;

        cudaError_t err;
        if (mask_mode == flashinfer::MaskMode::kCausal) {
            err = flashinfer::mla::BatchMLAPagedAttention<
                flashinfer::MaskMode::kCausal, HEAD_DIM_CKV_C, HEAD_DIM_KPE_C>(
                params,
                static_cast<uint32_t>(plan_info.num_blks_x),
                static_cast<uint32_t>(plan_info.num_blks_y),
                stream);
        } else {
            err = flashinfer::mla::BatchMLAPagedAttention<
                flashinfer::MaskMode::kNone, HEAD_DIM_CKV_C, HEAD_DIM_KPE_C>(
                params,
                static_cast<uint32_t>(plan_info.num_blks_x),
                static_cast<uint32_t>(plan_info.num_blks_y),
                stream);
        }
        return static_cast<int>(err);
    })

    return static_cast<int>(cudaErrorInvalidConfiguration);
}
