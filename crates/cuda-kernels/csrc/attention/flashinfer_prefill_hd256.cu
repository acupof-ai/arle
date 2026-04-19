// FlashInfer single-prefill wrapper for Rust FFI — HEAD_DIM=256 variant
// Wraps SinglePrefillWithKVCacheDispatched for bf16, HEAD_DIM=256, causal mask.
//
// Matches the existing HD128 single-prefill wrapper but instantiates the
// HEAD_DIM=256 template used by Qwen3.5 full-attention prefill.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

#include <flashinfer/pos_enc.cuh>
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/attention/prefill.cuh>

using DType = __nv_bfloat16;

constexpr uint32_t HEAD_DIM = 256;
constexpr flashinfer::PosEncodingMode POS_MODE = flashinfer::PosEncodingMode::kNone;
constexpr flashinfer::MaskMode MASK_MODE = flashinfer::MaskMode::kCausal;

using Variant = flashinfer::DefaultAttention<false, false, false, false>;
using Params = flashinfer::SinglePrefillParams<DType, DType, DType>;

extern "C" int flashinfer_single_prefill_hd256(
    __nv_bfloat16* q,
    __nv_bfloat16* k_cache,
    __nv_bfloat16* v_cache,
    __nv_bfloat16* output,
    int num_q_heads,
    int num_kv_heads,
    int seq_len,
    int kv_len,
    int max_seq_len,
    float sm_scale,
    __nv_bfloat16* tmp_buf,
    cudaStream_t stream) {
  const uint32_t kv_stride_h = static_cast<uint32_t>(max_seq_len) * HEAD_DIM;
  const uint32_t kv_stride_n = HEAD_DIM;
  const uint32_t q_stride_n = static_cast<uint32_t>(num_q_heads) * HEAD_DIM;
  const uint32_t q_stride_h = HEAD_DIM;

  Params params(
      /*q=*/q,
      /*k=*/k_cache,
      /*v=*/v_cache,
      /*custom_mask=*/nullptr,
      /*o=*/output,
      /*lse=*/nullptr,
      /*alibi_slopes=*/nullptr,
      /*num_qo_heads=*/static_cast<uint32_t>(num_q_heads),
      /*num_kv_heads=*/static_cast<uint32_t>(num_kv_heads),
      /*qo_len=*/static_cast<uint32_t>(seq_len),
      /*kv_len=*/static_cast<uint32_t>(kv_len),
      /*q_stride_n=*/q_stride_n,
      /*q_stride_h=*/q_stride_h,
      /*kv_stride_n=*/kv_stride_n,
      /*kv_stride_h=*/kv_stride_h,
      /*head_dim=*/HEAD_DIM,
      /*window_left=*/-1,
      /*logits_soft_cap=*/0.0f,
      /*sm_scale=*/sm_scale,
      /*rope_scale=*/1.0f,
      /*rope_theta=*/1e4f);

  cudaError_t err = flashinfer::SinglePrefillWithKVCacheDispatched<
      HEAD_DIM,
      HEAD_DIM,
      POS_MODE,
      /*USE_FP16_QK_REDUCTION=*/false,
      MASK_MODE,
      Variant,
      Params>(params, tmp_buf, stream);

  return static_cast<int>(err);
}
