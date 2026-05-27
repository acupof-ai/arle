// ARLE → FlashMLA (sgl-project/FlashMLA @ df022eb) sparse prefill shim.
//
// Bypasses FlashMLA's PyTorch-bound `sparse_attn_prefill_interface` and calls
// the raw-pointer SM90 entry `sm90::run_fwd_kernel(SparseAttnFwdParams&)`
// directly so ARLE can drive it via cudarc + cuda-kernels FFI without
// linking libtorch.
//
// FlashMLA supports head_dim_qk ∈ {512, 576} and head_dim_v = 512 for the
// SM90 sparse prefill path — exactly DSv4-Flash's MLA shape (NoPE 512 +
// optional 64-dim RoPE tail). topk ∈ {512} (DSv4 CSA = 512, HCA = full
// compressed-page count via `topk_length`).
//
// See params.h::SparseAttnFwdParams for the input contract. All tensors
// are caller-owned device buffers; this shim does no allocations.
//
// Refs:
//   docs/experience/errors/2026-05-27-dsv4-grouped-gemm-marginal-prefill-kernel-not-blocker.md
//   vendor/flashmla/csrc/sm90/prefill/sparse/fwd.cu

#include <cuda_runtime.h>
#include <cstdint>
#include <exception>
#include <stdexcept>

// FlashMLA internals (vendored): SparseAttnFwdParams + the SM90 entry.
// FlashMLA's `params.h` only pulls in `cutlass/bfloat16.h` — no torch.
#include "../../vendor/flashmla/csrc/params.h"
#include "../../vendor/flashmla/csrc/sm90/prefill/sparse/fwd.h"

extern "C" {

// Run FlashMLA SM90 sparse prefill attention. All pointer args are CUDA
// device pointers. Strides are in element count (not bytes).
//
// Returns cudaSuccess on completion (or the first failing cudaError_t if
// the underlying launch fails — FlashMLA's launcher throws on the
// unsupported-d_qk path, caught here).
cudaError_t arle_flashmla_sm90_sparse_prefill_fwd(
    const void* q,            // bf16 [s_q, h_q, d_qk]
    const void* kv,           // bf16 [s_kv, h_kv, d_qk]
    const int32_t* indices,   // int32 [s_q, h_kv, topk]
    const float* attn_sink,   // float [h_q] or nullptr
    const int32_t* topk_length, // int32 [s_q] or nullptr (HCA: pass non-null + per-row variable lengths)
    void* out,                // bf16 [s_q, h_q, d_v]
    float* max_logits,        // float [s_q, h_q] or nullptr (not required, but FlashMLA writes if provided)
    float* lse,               // float [s_q, h_q] or nullptr
    int s_q, int s_kv,
    int h_q, int h_kv,
    int d_qk, int d_v,
    int topk,
    float sm_scale,
    int stride_q_s_q, int stride_q_h_q,
    int stride_kv_s_kv, int stride_kv_h_kv,
    int stride_indices_s_q, int stride_indices_h_kv,
    int num_sm,
    cudaStream_t stream
) {
    // Pre-flight: validate FlashMLA SM90 sparse-prefill invariants. Failing
    // here returns cudaErrorInvalidValue cleanly; failing inside the kernel
    // throws kerutils::KUException (extends std::exception, NOT
    // std::runtime_error — V1 abort root cause), and that throw across the
    // extern "C" boundary aborts the host process. Mirror the asserts at
    // vendor/flashmla/csrc/sm90/prefill/sparse/phase1.cuh:576-579 and
    // sm90/prefill/sparse/config.h:26-27 (B_H = 64, B_TOPK = 64).
    constexpr int B_H = 64;
    constexpr int B_TOPK = 64;
    if (h_kv != 1) return cudaErrorInvalidValue;
    if (h_q <= 0 || (h_q % B_H) != 0) return cudaErrorInvalidValue;
    if (topk <= 0 || (topk % (2 * B_TOPK)) != 0) return cudaErrorInvalidValue;
    if (d_qk != 512 && d_qk != 576) return cudaErrorInvalidValue;
    if (d_v != 512) return cudaErrorInvalidValue;
    if (s_q <= 0 || s_kv <= 0) return cudaErrorInvalidValue;

    SparseAttnFwdParams p{};
    p.s_q = s_q;
    p.s_kv = s_kv;
    p.h_q = h_q;
    p.h_kv = h_kv;
    p.d_qk = d_qk;
    p.d_v = d_v;
    p.topk = topk;
    p.sm_scale = sm_scale;
    // FlashMLA expects log2-scaled softmax scale to skip the per-iter log
    // conversion. LOG_2_E = 1.44269504f (api/common.h).
    p.sm_scale_div_log2 = sm_scale * 1.44269504f;

    p.q = reinterpret_cast<cutlass::bfloat16_t*>(const_cast<void*>(q));
    p.kv = reinterpret_cast<cutlass::bfloat16_t*>(const_cast<void*>(kv));
    p.indices = const_cast<int*>(indices);
    p.attn_sink = const_cast<float*>(attn_sink);
    p.topk_length = const_cast<int*>(topk_length);

    p.out = reinterpret_cast<cutlass::bfloat16_t*>(out);
    p.max_logits = max_logits;
    p.lse = lse;

    p.stride_q_s_q = stride_q_s_q;
    p.stride_q_h_q = stride_q_h_q;
    p.stride_kv_s_kv = stride_kv_s_kv;
    p.stride_kv_h_kv = stride_kv_h_kv;
    p.stride_indices_s_q = stride_indices_s_q;
    p.stride_indices_h_kv = stride_indices_h_kv;

    p.num_sm = num_sm;
    p.stream = stream;

    // Belt-and-braces: catch anything the launcher might throw, including
    // kerutils::KUException (extends std::exception, not std::runtime_error
    // — V1 caught only runtime_error, KUException escaped → process abort).
    // Pre-flight above should already prevent the documented assert paths.
    try {
        sm90::run_fwd_kernel(p);
    } catch (const std::exception&) {
        return cudaErrorInvalidValue;
    } catch (...) {
        return cudaErrorUnknown;
    }
    return cudaGetLastError();
}

}  // extern "C"
