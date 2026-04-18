#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <exception>

#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/page.cuh>
#include <flashinfer/pos_enc.cuh>

using DType = __nv_bfloat16;
using IdType = int32_t;

constexpr uint32_t HEAD_DIM = 256;
constexpr flashinfer::PosEncodingMode POS_MODE = flashinfer::PosEncodingMode::kNone;
constexpr flashinfer::MaskMode MASK_MODE = flashinfer::MaskMode::kCausal;
constexpr bool USE_FP16_QK_REDUCTION = false;

using Variant = flashinfer::DefaultAttention<false, false, false, false>;
using Params = flashinfer::BatchPrefillPagedParams<DType, DType, DType, IdType>;

extern "C" int flashinfer_batch_prefill_paged_hd256_plan(
    void* float_workspace,
    size_t float_workspace_bytes,
    void* int_workspace,
    void* page_locked_workspace,
    size_t int_workspace_bytes,
    int32_t* qo_indptr_h,
    int32_t* kv_indptr_h,
    int batch_size,
    int num_qo_heads,
    int num_kv_heads,
    int page_size,
    int head_dim,
    void* plan_info_out,
    cudaStream_t stream) {
  (void)head_dim;

  try {
    flashinfer::PrefillPlanInfo plan_info;
    uint32_t total_num_rows = static_cast<uint32_t>(qo_indptr_h[batch_size]);

    cudaError_t err = flashinfer::PrefillPlan<IdType>(
        float_workspace,
        float_workspace_bytes,
        int_workspace,
        page_locked_workspace,
        int_workspace_bytes,
        plan_info,
        qo_indptr_h,
        kv_indptr_h,
        total_num_rows,
        static_cast<uint32_t>(batch_size),
        static_cast<uint32_t>(num_qo_heads),
        static_cast<uint32_t>(num_kv_heads),
        HEAD_DIM,
        HEAD_DIM,
        static_cast<uint32_t>(page_size),
        // See `flashinfer_prefill_paged.cu` for the detailed rationale.
        // We do not CUDA-graph-capture prefill; keeping this true makes
        // FlashInfer reserve workspace for an inflated padded_batch_size.
        /*enable_cuda_graph=*/false,
        /*sizeof_dtype_o=*/static_cast<uint32_t>(sizeof(DType)),
        /*window_left=*/-1,
        /*fixed_split_size=*/-1,
        /*disable_split_kv=*/false,
        /*num_colocated_ctas=*/0,
        stream);
    if (err != cudaSuccess) {
      return static_cast<int>(err);
    }

    std::memcpy(plan_info_out, &plan_info, sizeof(flashinfer::PrefillPlanInfo));
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr,
                 "flashinfer_batch_prefill_paged_hd256_plan exception: %s\n",
                 e.what());
    return -1;
  } catch (...) {
    std::fprintf(stderr, "flashinfer_batch_prefill_paged_hd256_plan exception: unknown\n");
    return -2;
  }
}

extern "C" int flashinfer_batch_prefill_paged_hd256_run(
    void* float_workspace,
    void* int_workspace,
    void* plan_info_ptr,
    __nv_bfloat16* q,
    int32_t* q_indptr,
    __nv_bfloat16* k_data,
    __nv_bfloat16* v_data,
    int32_t* kv_indptr,
    int32_t* kv_indices,
    int32_t* kv_last_page_len,
    __nv_bfloat16* o,
    float* lse,
    int batch_size,
    int num_qo_heads,
    int num_kv_heads,
    int page_size,
    float sm_scale,
    cudaStream_t stream) {
  try {
    flashinfer::PrefillPlanInfo plan_info;
    std::memcpy(&plan_info, plan_info_ptr, sizeof(flashinfer::PrefillPlanInfo));

    flashinfer::paged_kv_t<DType, IdType> paged_kv(
        static_cast<uint32_t>(num_kv_heads),
        static_cast<uint32_t>(page_size),
        HEAD_DIM,
        static_cast<uint32_t>(batch_size),
        flashinfer::QKVLayout::kHND,
        k_data,
        v_data,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
        /*rope_pos_offset=*/nullptr);

    Params params(
        /*q=*/q,
        /*paged_kv=*/paged_kv,
        /*custom_mask=*/nullptr,
        /*q_indptr=*/q_indptr,
        /*mask_indptr=*/nullptr,
        /*q_rope_offset=*/nullptr,
        /*o=*/o,
        /*lse=*/lse,
        /*alibi_slopes=*/nullptr,
        /*num_qo_heads=*/static_cast<uint32_t>(num_qo_heads),
        /*q_stride_n=*/static_cast<IdType>(num_qo_heads * HEAD_DIM),
        /*q_stride_h=*/static_cast<IdType>(HEAD_DIM),
        /*window_left=*/-1,
        /*logits_soft_cap=*/0.0f,
        /*sm_scale=*/sm_scale,
        /*rope_scale=*/1.0f,
        /*rope_theta=*/1e4f);

    params.padded_batch_size = plan_info.padded_batch_size;
    params.max_total_num_rows = plan_info.total_num_rows;
    if (plan_info.enable_cuda_graph) {
      params.total_num_rows =
          flashinfer::GetPtrFromBaseOffset<uint32_t>(int_workspace, plan_info.total_num_rows_offset);
    }
    params.partition_kv = plan_info.split_kv;
    params.request_indices =
        flashinfer::GetPtrFromBaseOffset<IdType>(int_workspace, plan_info.request_indices_offset);
    params.qo_tile_indices =
        flashinfer::GetPtrFromBaseOffset<IdType>(int_workspace, plan_info.qo_tile_indices_offset);
    params.kv_tile_indices =
        flashinfer::GetPtrFromBaseOffset<IdType>(int_workspace, plan_info.kv_tile_indices_offset);
    params.o_indptr =
        flashinfer::GetPtrFromBaseOffset<IdType>(int_workspace, plan_info.o_indptr_offset);
    params.kv_chunk_size_ptr =
        flashinfer::GetPtrFromBaseOffset<IdType>(int_workspace, plan_info.kv_chunk_size_ptr_offset);

    if (plan_info.split_kv) {
      params.merge_indptr =
          flashinfer::GetPtrFromBaseOffset<IdType>(int_workspace, plan_info.merge_indptr_offset);
      params.block_valid_mask =
          flashinfer::GetPtrFromBaseOffset<bool>(int_workspace, plan_info.block_valid_mask_offset);
    } else {
      params.merge_indptr = nullptr;
      params.block_valid_mask = nullptr;
    }

    DType* tmp_v = nullptr;
    float* tmp_s = nullptr;
    if (plan_info.split_kv) {
      tmp_v = flashinfer::GetPtrFromBaseOffset<DType>(float_workspace, plan_info.v_offset);
      tmp_s = flashinfer::GetPtrFromBaseOffset<float>(float_workspace, plan_info.s_offset);
    }

    cudaError_t err;
    uint32_t cta_tile_q = plan_info.cta_tile_q;

#define DISPATCH_BATCH_PREFILL_PAGED_HD256(CTQ)                                      \
  err = flashinfer::BatchPrefillWithPagedKVCacheDispatched<CTQ,                      \
                                                           HEAD_DIM,                 \
                                                           HEAD_DIM,                 \
                                                           POS_MODE,                 \
                                                           USE_FP16_QK_REDUCTION,    \
                                                           MASK_MODE,                \
                                                           Variant,                  \
                                                           Params>(                  \
      params, tmp_v, tmp_s, /*enable_pdl=*/false, stream)

  switch (cta_tile_q) {
    case 16:
      DISPATCH_BATCH_PREFILL_PAGED_HD256(16);
      break;
    case 32:
      DISPATCH_BATCH_PREFILL_PAGED_HD256(32);
      break;
    case 64:
      DISPATCH_BATCH_PREFILL_PAGED_HD256(64);
      break;
    case 128:
      DISPATCH_BATCH_PREFILL_PAGED_HD256(128);
      break;
    default:
      DISPATCH_BATCH_PREFILL_PAGED_HD256(16);
      break;
  }

#undef DISPATCH_BATCH_PREFILL_PAGED_HD256

    return static_cast<int>(err);
  } catch (const std::exception& e) {
    std::fprintf(stderr,
                 "flashinfer_batch_prefill_paged_hd256_run exception: %s\n",
                 e.what());
    return -1;
  } catch (...) {
    std::fprintf(stderr, "flashinfer_batch_prefill_paged_hd256_run exception: unknown\n");
    return -2;
  }
}
