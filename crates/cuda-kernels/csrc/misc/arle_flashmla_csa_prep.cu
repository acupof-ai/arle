// ARLE → FlashMLA CSA prep: build the unified KV pool's per-token index
// array and topk_length so FlashMLA's single-pool sparse-prefill kernel can
// attend to ARLE's sliding-window + compressed pools jointly.
//
// Pool layout (Variant B1 — dense live range, -1 tail padding):
//   kv_unified[s_kv_total, 1, d_qk]
//     [0,             sw_window)         ← window_cache rebased to sw_base
//     [sw_window,     sw_window + N)     ← k_prepared (current chunk K, RoPE'd)
//     [sw_window + N, sw_window + N + C) ← compressed pool
//   s_kv_total = sw_window + N + C
//
// Per-row indices [s_q, 1, topk_unified] where topk_unified = sw_window + index_topk
// (multiple of FlashMLA's 2*B_TOPK = 128). Row layout:
//   [0,             sw_count_i)               ← SW pool offsets, oldest→newest
//   [sw_count_i,    sw_count_i + index_topk)  ← compressed pool offsets (with -1 for
//                                                CSA selector padding or invalid)
//   [sw_count_i + index_topk, topk_unified)   ← -1 padding
// topk_length[i] = sw_count_i + index_topk.
//
// `selected` is the existing CSA top-k tensor; entries are compressed-block IDs
// in [0, compressed_count). Negative entries (incomplete CSA selections in
// early prefill) are propagated as -1 directly so FlashMLA masks them via
// the `is_token_valid = t >= 0 && t < params.s_kv` check at phase1.cuh:485.
//
// Refs:
//   crates/cuda-kernels/csrc/misc/arle_flashmla_shim.cu
//   crates/cuda-kernels/vendor/flashmla/csrc/params.h::SparseAttnFwdParams
//   crates/cuda-kernels/vendor/flashmla/csrc/sm90/prefill/sparse/phase1.cuh
//   crates/cuda-kernels/csrc/misc/dsv4_attention.cu::dsv4_hybrid_attention_kernel

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <algorithm>

namespace {

// One block per Q token. Threads parallelise over the topk_unified row.
// Grid:  (s_q, 1, 1)
// Block: (kBlock=128, 1, 1)
__global__ void arle_csa_build_indices_kernel(
        int32_t* __restrict__ indices,       // [s_q, topk_unified] int32
        int32_t* __restrict__ topk_length,   // [s_q] int32
        const int32_t* __restrict__ selected,// [s_q, index_topk] int32
        int s_q,
        int start_pos,
        int sw_window,        // 128
        int index_topk,       // 512
        int topk_unified,     // sw_window + index_topk = 640
        int n_tokens,         // = s_q
        int compressed_count, // C
        int compress_ratio,   // for compress-block causality gate (4 for CSA)
        int sw_base) {        // max(0, start_pos - sw_window)
    int token = blockIdx.x;
    if (token >= s_q) return;

    const int abs_pos = start_pos + token;
    const int sw_start = max(0, abs_pos + 1 - sw_window);
    const int sw_count = abs_pos - sw_start + 1;          // ∈ [1, sw_window]

    const int comp_base_in_pool = sw_window + n_tokens;   // start of compressed region

    int32_t* row = indices + (size_t)token * topk_unified;

    // [0, sw_count): SW pool offsets.
    // p ∈ [sw_start, abs_pos]. Pool offset:
    //   p <  start_pos  →  p - sw_base                  (rolling region [0, sw_window))
    //   p >= start_pos  →  sw_window + (p - start_pos)  (linear k_prepared region)
    for (int j = threadIdx.x; j < sw_count; j += blockDim.x) {
        int p = sw_start + j;
        int slot = (p < start_pos)
                 ? (p - sw_base)
                 : (sw_window + (p - start_pos));
        row[j] = slot;
    }

    // [sw_count, sw_count + index_topk): compressed pool offsets.
    // Apply defensive compress-block causality gate (mirrors
    // dsv4_hybrid_attention_kernel:898-901): selected block c covers
    // tokens [c*compress_ratio .. (c+1)*compress_ratio - 1]; if
    // block_end > abs_pos the block summarises future tokens — mask as -1.
    //
    // selected can be nullptr when the CSA selector hasn't run yet (very early
    // prefill, or modes that hit this path without a selector populated). Treat
    // as "no compressed entries selected" — fill -1 padding so FlashMLA masks
    // the entire compressed range.
    if (selected == nullptr) {
        for (int k = threadIdx.x; k < index_topk; k += blockDim.x) {
            row[sw_count + k] = -1;
        }
    } else {
        const int32_t* sel = selected + (size_t)token * index_topk;
        for (int k = threadIdx.x; k < index_topk; k += blockDim.x) {
            int32_t c = sel[k];
            bool valid = (c >= 0) && (c < compressed_count);
            if (valid && compress_ratio > 0) {
                int block_end = c * compress_ratio + (compress_ratio - 1);
                if (block_end > abs_pos) valid = false;
            }
            row[sw_count + k] = valid ? (comp_base_in_pool + c) : -1;
        }
    }

    // [sw_count + index_topk, topk_unified): -1 padding.
    int pad_start = sw_count + index_topk;
    for (int k = pad_start + threadIdx.x; k < topk_unified; k += blockDim.x) {
        row[k] = -1;
    }

    if (threadIdx.x == 0) {
        topk_length[token] = sw_count + index_topk;
    }
}

// Pack the rolling window_cache into linear absolute-position order.
// dst[i] = key_at_absolute_position(sw_base + i) for i in [0, sw_window).
// `slot = (sw_base + i) % sw_window` in the rolling buffer.
//
// Grid: (sw_window, 1, 1)
// Block: (kBlock=256, 1, 1)
__global__ void arle_csa_pack_sw_region_kernel(
        __nv_bfloat16* __restrict__ dst,          // [sw_window, d_qk]
        const __nv_bfloat16* __restrict__ window_cache, // [sw_window, d_qk] rolling
        int sw_window,
        int sw_base,
        int d_qk) {
    int row = blockIdx.x;
    if (row >= sw_window) return;
    int slot = (sw_base + row) % sw_window;
    const __nv_bfloat16* src = window_cache + (size_t)slot * d_qk;
    __nv_bfloat16* dst_row = dst + (size_t)row * d_qk;
    for (int c = threadIdx.x; c < d_qk; c += blockDim.x) {
        dst_row[c] = src[c];
    }
}

}  // namespace

namespace {

// HCA build indices variant.
//
// HCA (HybridCompressed) attention has NO top-k selector — it attends to
// all compressed pages whose block_end <= abs_pos (causal). For each Q-token,
// the unified pool indices are:
//   [0,           sw_count_t)            ← SW pool offsets (same as CSA)
//   [sw_count_t,  sw_count_t + comp_t)   ← identity 0..comp_t-1 into compressed
//   [sw_count_t + comp_t, topk_unified)  ← -1 padding
//
// where comp_t = min(compressed_count, (start_pos + t + 1) / compress_ratio)
// mirrors `comp_keys = dsv4_imin(compressed_count, (abs_pos+1)/compress_ratio)`
// at dsv4_hybrid_attention_kernel:882.
//
// topk_length[t] = sw_count_t + comp_t.
__global__ void arle_hca_build_indices_kernel(
        int32_t* __restrict__ indices,       // [s_q, topk_unified] int32
        int32_t* __restrict__ topk_length,   // [s_q] int32
        int s_q,
        int start_pos,
        int sw_window,
        int topk_unified,                    // sw_window + max_compressed (padded to 128)
        int n_tokens,
        int compressed_count,
        int compress_ratio,
        int sw_base) {
    int token = blockIdx.x;
    if (token >= s_q) return;

    const int abs_pos = start_pos + token;
    const int sw_start = max(0, abs_pos + 1 - sw_window);
    const int sw_count = abs_pos - sw_start + 1;

    // HCA per-token compressed key count (mirrors dsv4_hybrid_attention_kernel:882).
    int comp_keys = (compress_ratio > 0) ? ((abs_pos + 1) / compress_ratio) : 0;
    if (comp_keys > compressed_count) comp_keys = compressed_count;
    if (comp_keys < 0) comp_keys = 0;

    const int comp_base_in_pool = sw_window + n_tokens;
    int32_t* row = indices + (size_t)token * topk_unified;

    // [0, sw_count): SW pool offsets (same arithmetic as CSA path).
    for (int j = threadIdx.x; j < sw_count; j += blockDim.x) {
        int p = sw_start + j;
        int slot = (p < start_pos)
                 ? (p - sw_base)
                 : (sw_window + (p - start_pos));
        row[j] = slot;
    }

    // [sw_count, sw_count + comp_keys): identity 0..comp_keys-1 into compressed.
    for (int k = threadIdx.x; k < comp_keys; k += blockDim.x) {
        row[sw_count + k] = comp_base_in_pool + k;
    }

    // [sw_count + comp_keys, topk_unified): -1 padding.
    int pad_start = sw_count + comp_keys;
    for (int k = pad_start + threadIdx.x; k < topk_unified; k += blockDim.x) {
        row[k] = -1;
    }

    if (threadIdx.x == 0) {
        topk_length[token] = sw_count + comp_keys;
    }
}

}  // namespace (HCA helpers)

extern "C" {

// Build kv_unified = concat(window_cache_rebased, k_prepared, compressed).
// All pointers are device pointers. Stream-ordered.
cudaError_t arle_flashmla_csa_pack_kv(
        __nv_bfloat16* kv_unified,                // [s_kv_total, d_qk]
        const __nv_bfloat16* window_cache,        // [sw_window, d_qk] rolling
        const __nv_bfloat16* k_prepared,          // [n_tokens, d_qk] linear
        const __nv_bfloat16* compressed,          // [compressed_count, d_qk], nullable iff C==0
        int start_pos,
        int sw_window,
        int n_tokens,
        int compressed_count,
        int d_qk,
        cudaStream_t stream) {
    const size_t row_bytes = (size_t)d_qk * sizeof(__nv_bfloat16);

    // [0, sw_window): SW region. Skip when start_pos == 0 — the SW window for
    // every token in this chunk lies inside k_prepared; this slab is unreferenced.
    if (start_pos > 0) {
        const int sw_base = std::max(0, start_pos - sw_window);
        // sw_base == 0 still requires the copy when 0 < start_pos < sw_window
        // because indices for early tokens reach into [0, sw_window).
        (void)sw_base;
        constexpr int kBlock = 256;
        arle_csa_pack_sw_region_kernel<<<sw_window, kBlock, 0, stream>>>(
            kv_unified, window_cache, sw_window,
            std::max(0, start_pos - sw_window), d_qk);
        auto err = cudaGetLastError();
        if (err != cudaSuccess) return err;
    }

    // [sw_window, sw_window + n_tokens): linear k_prepared.
    if (n_tokens > 0) {
        auto err = cudaMemcpyAsync(
            kv_unified + (size_t)sw_window * d_qk,
            k_prepared,
            (size_t)n_tokens * row_bytes,
            cudaMemcpyDeviceToDevice, stream);
        if (err != cudaSuccess) return err;
    }

    // [sw_window + n_tokens, s_kv_total): compressed.
    if (compressed_count > 0 && compressed != nullptr) {
        auto err = cudaMemcpyAsync(
            kv_unified + (size_t)(sw_window + n_tokens) * d_qk,
            compressed,
            (size_t)compressed_count * row_bytes,
            cudaMemcpyDeviceToDevice, stream);
        if (err != cudaSuccess) return err;
    }
    return cudaSuccess;
}

// Build the matching indices + topk_length for the unified pool.
cudaError_t arle_flashmla_csa_build_indices(
        int32_t* indices,
        int32_t* topk_length,
        const int32_t* selected,
        int s_q,
        int start_pos,
        int sw_window,
        int index_topk,
        int compressed_count,
        int compress_ratio,
        cudaStream_t stream) {
    if (s_q <= 0) return cudaSuccess;
    if (sw_window <= 0 || index_topk < 0 || compressed_count < 0 || start_pos < 0) {
        return cudaErrorInvalidValue;
    }
    const int topk_unified = sw_window + index_topk;
    // FlashMLA's params.topk must satisfy topk % (2*B_TOPK) == 0 with B_TOPK=64.
    if ((topk_unified & 127) != 0) return cudaErrorInvalidValue;

    const int sw_base = std::max(0, start_pos - sw_window);
    constexpr int kBlock = 128;
    arle_csa_build_indices_kernel<<<s_q, kBlock, 0, stream>>>(
        indices, topk_length, selected,
        s_q, start_pos, sw_window, index_topk, topk_unified,
        /*n_tokens=*/s_q, compressed_count, compress_ratio, sw_base);
    return cudaGetLastError();
}

// HCA (HybridCompressed) indices launcher. No selector — attend to all
// compressed pages causally. topk_unified must be a multiple of 128.
//
// max_compressed_keys is the cap padded into the indices buffer; pass the
// total compressed_count for this chunk (or the next-multiple-of-128 padded
// length the caller allocated).
cudaError_t arle_flashmla_hca_build_indices(
        int32_t* indices,
        int32_t* topk_length,
        int s_q,
        int start_pos,
        int sw_window,
        int max_compressed_keys,    // pool capacity for compressed slots in indices row
        int compressed_count,
        int compress_ratio,
        cudaStream_t stream) {
    if (s_q <= 0) return cudaSuccess;
    if (sw_window <= 0 || max_compressed_keys < 0 || compressed_count < 0 || start_pos < 0) {
        return cudaErrorInvalidValue;
    }
    const int topk_unified = sw_window + max_compressed_keys;
    if ((topk_unified & 127) != 0) return cudaErrorInvalidValue;

    const int sw_base = std::max(0, start_pos - sw_window);
    constexpr int kBlock = 128;
    arle_hca_build_indices_kernel<<<s_q, kBlock, 0, stream>>>(
        indices, topk_length,
        s_q, start_pos, sw_window, topk_unified,
        /*n_tokens=*/s_q, compressed_count, compress_ratio, sw_base);
    return cudaGetLastError();
}

}  // extern "C"
