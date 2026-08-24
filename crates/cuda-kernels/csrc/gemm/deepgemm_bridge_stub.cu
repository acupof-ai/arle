#include <cuda.h>

#include <cstddef>
#include <cstdint>

struct Sm90MegaMoeWorkspaceLayout {
  uint64_t num_bytes;
  uint64_t x;
  uint64_t x_sf;
  uint64_t topk_idx;
  uint64_t topk_weights;
  uint64_t l1_acts;
  uint64_t l1_acts_sf;
  uint64_t l1_topk_weights;
  uint64_t l2_acts;
  uint64_t l2_acts_sf;
  uint64_t combine;
  int num_max_pool_tokens;
  int num_padded_sf_pool_tokens;
  int num_max_tokens_per_rank;
};

#ifndef ARLE_ENABLE_DEEPGEMM_NATIVE
extern "C" CUresult dsv4_deepgemm_native_preflight_cuda(char* out, size_t out_len) {
  static constexpr const char* kMessage =
      "status=failed native_bridge=not_compiled "
      "reason=deepgemm_not_buildable "
      "hint=build_with_sm90_vendored_deepgemm_or_disable_with_ARLE_CUDA_DISABLE_DEEPGEMM_NATIVE=1";
  if (out != nullptr && out_len > 0) {
    size_t n = 0;
    while (n + 1 < out_len && kMessage[n] != '\0') {
      out[n] = kMessage[n];
      ++n;
    }
    out[n] = '\0';
  }
  return CUDA_ERROR_NOT_SUPPORTED;
}

extern "C" CUresult dsv4_sm90_mega_moe_workspace_layout_cuda(
    int,
    int,
    int,
    int,
    int,
    int,
    Sm90MegaMoeWorkspaceLayout*) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

extern "C" CUresult dsv4_sm90_mega_moe_pre_dispatch_cuda(
    const unsigned short*,
    const int*,
    const float*,
    unsigned char*,
    float*,
    int64_t*,
    float*,
    int,
    int,
    int,
    int,
    int,
    CUstream) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

extern "C" CUresult dsv4_sm90_mega_moe_launch_cuda(
    void*,
    int*,
    const uint64_t*,
    void*,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    float,
    int,
    int,
    const unsigned char*,
    int,
    const float*,
    const unsigned char*,
    int,
    const float*,
    CUstream) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

extern "C" CUresult dsv4_deepgemm_m_grouped_fp8_gemm_nt_masked_cuda(
    const unsigned char*,
    const float*,
    const unsigned char*,
    const float*,
    unsigned short*,
    const int*,
    int,
    int,
    int,
    int,
    int,
    CUstream) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

extern "C" CUresult dsv4_deepgemm_m_grouped_fp8_gemm_nt_contiguous_cuda(
    const unsigned char*,
    const float*,
    const unsigned char*,
    const float*,
    unsigned short*,
    const int*,
    int,
    int,
    int,
    int,
    int,
    int,
    CUstream) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

extern "C" CUresult deepgemm_m_grouped_bf16_gemm_nt_masked_cuda(
    const unsigned short*,
    const unsigned short*,
    unsigned short*,
    const int*,
    int,
    int,
    int,
    int,
    int,
    CUstream) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

extern "C" CUresult deepgemm_m_grouped_bf16_gemm_nt_contiguous_cuda(
    const unsigned short*,
    const unsigned short*,
    unsigned short*,
    const int*,
    int,
    int,
    int,
    int,
    CUstream) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

extern "C" CUresult dsv4_deepgemm_paged_mqa_logits_metadata_cuda(
    const int*,
    int*,
    int,
    int,
    int,
    int,
    CUstream) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

// Dense FP8 GEMM (a, sfa, b, sfb, d, m, n, k, sfa_aligned_m, stream).
extern "C" CUresult dsv4_deepgemm_fp8_gemm_nt_cuda(
    const unsigned char*,
    const float*,
    const unsigned char*,
    const float*,
    unsigned short*,
    int,
    int,
    int,
    int,
    CUstream) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

// FP8 paged-MQA logits with fused KV-cache scale layout
// (q, kv_cache_with_scale, weights, context_lens, block_table, schedule_meta,
//  logits, 11×int dims, stream).
extern "C" CUresult dsv4_deepgemm_fp8_paged_mqa_logits_fused_cache_cuda(
    const unsigned char*,
    const unsigned char*,
    const float*,
    const int*,
    const int*,
    const int*,
    float*,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    CUstream) {
  return CUDA_ERROR_NOT_SUPPORTED;
}
#endif
