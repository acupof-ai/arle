// ARLE → SGLang sparse_mla_q8kv8_prefill_sm90 (FP8 QK + FP8 KV sparse MLA
// prefill) shim.
//
// Bypasses SGLang's TVM-FFI JIT entry (entry.cuh) and calls the raw
// `sm90::fwd::run_sparse_mla_q8kv8_prefill_kernel<D_QK, HAVE_TOPK_LENGTH,
// HAVE_ATTN_SINK>(SparseMlaQ8Kv8PrefillParams&)` directly so ARLE can drive
// it via cudarc + cuda-kernels FFI without linking libtorch / TVM.
//
// Kernel source: vendor/q8kv8_prefill/ (clean-room FP8 re-implementation of
// FlashMLA sparse prefill, Apache-2.0, SGLang Team). Uses native FP8 GMMA
// (E4M3 x E4M3 -> F32) for the QK GEMM at ~2x the bf16 throughput.
//
// Contract (all device pointers, caller-owned):
//   q:        fp8  [s_q, h_q, d_qk]
//   kv:       fp8  [s_kv, h_kv=1, d_qk]
//   indices:  i32  [s_q, h_kv=1, topk]
//   q_scale:  f32  [1]   (per-tensor; SGLang uses identity 1.0)
//   kv_scale: f32  [1]
//   out:      bf16 [s_q, h_q, d_v=512]
//   lse:      f32  [s_q, h_q]
//
// h_kv must be 1; h_q % 64 == 0; topk % 128 == 0; d_qk in {512, 576}; d_v=512.

#include <cuda_runtime.h>
#include <cstdint>

#include "../../vendor/q8kv8_prefill/kernel.cuh"

extern "C" {

cudaError_t arle_q8kv8_sparse_prefill_fwd(
    const void* q,            // fp8 [s_q, h_q, d_qk]
    const void* kv,           // fp8 [s_kv, 1, d_qk]
    const int32_t* indices,   // i32 [s_q, 1, topk]
    const float* q_scale,     // f32 [1]
    const float* kv_scale,    // f32 [1]
    const float* attn_sink,   // f32 [h_q] or nullptr
    const int32_t* topk_length, // i32 [s_q] or nullptr
    void* out,                // bf16 [s_q, h_q, 512]
    float* max_logits,        // f32 [s_q, h_q] (kernel writes unconditionally)
    float* lse,               // f32 [s_q, h_q]
    int s_q, int s_kv,
    int h_q,
    int d_qk,
    int topk,
    float sm_scale,
    cudaStream_t stream) {
  // Mirror the kernel's KU_ASSERT preconditions so a bad config returns
  // cudaErrorInvalidValue instead of aborting inside the launch.
  if (h_q <= 0 || (h_q % 64) != 0) return cudaErrorInvalidValue;
  if (topk <= 0 || (topk % 128) != 0) return cudaErrorInvalidValue;
  if (d_qk != 512 && d_qk != 576) return cudaErrorInvalidValue;
  if (s_q <= 0 || s_kv <= 0) return cudaErrorInvalidValue;

  SparseMlaQ8Kv8PrefillParams p{};
  p.s_q = s_q;
  p.s_kv = s_kv;
  p.h_q = h_q;
  p.h_kv = 1;
  p.d_qk = d_qk;
  p.d_v = 512;
  p.topk = topk;
  p.sm_scale_div_log2 = sm_scale * 1.44269504f;  // * log2(e)
  p.q = reinterpret_cast<const uint8_t*>(q);
  p.kv = reinterpret_cast<const uint8_t*>(kv);
  p.indices = const_cast<int32_t*>(indices);
  p.attn_sink = const_cast<float*>(attn_sink);
  p.topk_length = const_cast<int32_t*>(topk_length);
  p.q_scale_ptr = q_scale;
  p.kv_scale_ptr = kv_scale;
  p.stride_q_s_q = h_q * d_qk;
  p.stride_q_h_q = d_qk;
  p.stride_kv_s_kv = 1LL * d_qk;
  p.stride_kv_h_kv = d_qk;
  p.stride_indices_s_q = topk;
  p.stride_indices_h_kv = topk;
  p.out = reinterpret_cast<cutlass::bfloat16_t*>(out);
  p.max_logits = max_logits;
  p.lse = lse;
  p.stream = stream;

  try {
    if (topk_length != nullptr) {
      if (d_qk == 512)
        sm90::fwd::run_sparse_mla_q8kv8_prefill_kernel<512, true, true>(p);
      else
        sm90::fwd::run_sparse_mla_q8kv8_prefill_kernel<576, true, true>(p);
    } else {
      if (d_qk == 512)
        sm90::fwd::run_sparse_mla_q8kv8_prefill_kernel<512, false, false>(p);
      else
        sm90::fwd::run_sparse_mla_q8kv8_prefill_kernel<576, false, false>(p);
    }
  } catch (const std::exception& e) {
    return cudaErrorUnknown;
  }
  return cudaSuccess;
}

}
