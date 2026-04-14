// FlashInfer single-prefill wrapper for Rust FFI
// Wraps SinglePrefillWithKVCacheDispatched for bf16, HEAD_DIM=128, causal mask
//
// Replaces the Triton FlashAttention-2 prefill kernel for better performance
// on both Ampere (SM80) and Hopper (SM90) GPUs.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

#include <flashinfer/pos_enc.cuh>
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/attention/prefill.cuh>

// ---------------------------------------------------------------------------
// Fixed template parameters
// ---------------------------------------------------------------------------
using DType  = __nv_bfloat16;
using IdType = int32_t;

constexpr uint32_t HEAD_DIM = 128;
constexpr flashinfer::PosEncodingMode POS_MODE = flashinfer::PosEncodingMode::kNone;
constexpr flashinfer::MaskMode MASK_MODE = flashinfer::MaskMode::kCausal;

// DefaultAttention<use_custom_mask=false, use_sliding_window=false,
//                  use_logits_soft_cap=false, use_alibi=false>
using Variant = flashinfer::DefaultAttention<false, false, false, false>;
using Params  = flashinfer::SinglePrefillParams<DType, DType, DType>;

// ---------------------------------------------------------------------------
// flashinfer_single_prefill  –  replace Triton FA2 with FlashInfer
// ---------------------------------------------------------------------------
//
// Q layout:       [seq_len, num_q_heads * head_dim] row-major (NHD interleaved)
//   q_stride_n = num_q_heads * head_dim
//   q_stride_h = head_dim
//
// K/V cache layout: [num_kv_heads, max_seq_len, head_dim] (HND)
//   kv_stride_h = max_seq_len * head_dim
//   kv_stride_n = head_dim
//
// Output layout: same as Q [seq_len, num_q_heads * head_dim]
//
extern "C" int flashinfer_single_prefill(
    __nv_bfloat16* q,         // [seq_len, q_dim] normed+RoPE'd
    __nv_bfloat16* k_cache,   // [num_kv_heads, max_seq_len, head_dim] with KV already written
    __nv_bfloat16* v_cache,   // same layout
    __nv_bfloat16* output,    // [seq_len, q_dim]
    int            num_q_heads,
    int            num_kv_heads,
    int            seq_len,      // query length (new tokens being prefilled)
    int            kv_len,       // total KV length = start_pos + seq_len
    int            max_seq_len,  // max capacity of KV cache
    float          sm_scale,     // 1/sqrt(head_dim)
    void*          tmp_buffer,   // GPU scratch for split-KV (nullable)
    cudaStream_t   stream)
{
    // KV cache strides (HND layout)
    uint32_t kv_stride_h = (uint32_t)max_seq_len * HEAD_DIM;
    uint32_t kv_stride_n = HEAD_DIM;

    // Q/O strides (NHD interleaved layout)
    uint32_t q_stride_n = (uint32_t)num_q_heads * HEAD_DIM;
    uint32_t q_stride_h = HEAD_DIM;

    Params params(
        /*q=*/            q,
        /*k=*/            k_cache,
        /*v=*/            v_cache,
        /*custom_mask=*/  nullptr,
        /*o=*/            output,
        /*lse=*/          nullptr,
        /*alibi_slopes=*/ nullptr,
        /*num_qo_heads=*/ (uint32_t)num_q_heads,
        /*num_kv_heads=*/ (uint32_t)num_kv_heads,
        /*qo_len=*/       (uint32_t)seq_len,
        /*kv_len=*/       (uint32_t)kv_len,
        /*q_stride_n=*/   q_stride_n,
        /*q_stride_h=*/   q_stride_h,
        /*kv_stride_n=*/  kv_stride_n,
        /*kv_stride_h=*/  kv_stride_h,
        /*head_dim=*/     HEAD_DIM,
        /*window_left=*/  -1,
        /*logits_soft_cap=*/ 0.0f,
        /*sm_scale=*/     sm_scale,
        /*rope_scale=*/   1.0f,
        /*rope_theta=*/   1e4f
    );

    cudaError_t err = flashinfer::SinglePrefillWithKVCacheDispatched<
        HEAD_DIM, HEAD_DIM,
        POS_MODE,
        /*USE_FP16_QK_REDUCTION=*/false,
        MASK_MODE,
        Variant,
        Params
    >(params, (DType*)tmp_buffer, stream);

    return (int)err;
}
