// Build per-token index array + topk_length so FlashMLA's single-pool
// sparse-prefill kernel attends to ARLE's SW + compressed pools jointly.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <algorithm>

namespace {

__global__ void arle_csa_build_indices_kernel(
        int32_t* __restrict__ indices,
        int32_t* __restrict__ topk_length,
        const int32_t* __restrict__ selected,
        int s_q,
        int start_pos,
        int sw_window,
        int index_topk,
        int topk_unified,
        int n_tokens,
        int compressed_count,
        int compress_ratio,
        int sw_base) {
    int token = blockIdx.x;
    if (token >= s_q) return;

    const int abs_pos = start_pos + token;
    const int sw_start = max(0, abs_pos + 1 - sw_window);
    const int sw_count = abs_pos - sw_start + 1;

    const int comp_base_in_pool = sw_window + n_tokens;
    int32_t* row = indices + (size_t)token * topk_unified;

    for (int j = threadIdx.x; j < sw_count; j += blockDim.x) {
        int p = sw_start + j;
        int slot = (p < start_pos)
                 ? (p - sw_base)
                 : (sw_window + (p - start_pos));
        row[j] = slot;
    }

    // selected == nullptr: no selector yet — fill -1 so FlashMLA masks the range.
    if (selected == nullptr) {
        for (int k = threadIdx.x; k < index_topk; k += blockDim.x) {
            row[sw_count + k] = -1;
        }
    } else {
        const int32_t* sel = selected + (size_t)token * index_topk;
        for (int k = threadIdx.x; k < index_topk; k += blockDim.x) {
            int32_t c = sel[k];
            bool valid = (c >= 0) && (c < compressed_count);
            // Causality: block c covers [c*ratio, (c+1)*ratio-1]; mask future blocks.
            if (valid && compress_ratio > 0) {
                int block_end = c * compress_ratio + (compress_ratio - 1);
                if (block_end > abs_pos) valid = false;
            }
            row[sw_count + k] = valid ? (comp_base_in_pool + c) : -1;
        }
    }

    int pad_start = sw_count + index_topk;
    for (int k = pad_start + threadIdx.x; k < topk_unified; k += blockDim.x) {
        row[k] = -1;
    }

    if (threadIdx.x == 0) {
        topk_length[token] = sw_count + index_topk;
    }
}

__global__ void arle_csa_pack_sw_region_kernel(
        __nv_bfloat16* __restrict__ dst,
        const __nv_bfloat16* __restrict__ window_cache,
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

__global__ void arle_hca_build_indices_kernel(
        int32_t* __restrict__ indices,
        int32_t* __restrict__ topk_length,
        int s_q,
        int start_pos,
        int sw_window,
        int topk_unified,
        int n_tokens,
        int compressed_count,
        int compress_ratio,
        int sw_base) {
    int token = blockIdx.x;
    if (token >= s_q) return;

    const int abs_pos = start_pos + token;
    const int sw_start = max(0, abs_pos + 1 - sw_window);
    const int sw_count = abs_pos - sw_start + 1;

    // Causal floor: only compressed blocks fully past are visible.
    int comp_keys = (compress_ratio > 0) ? (abs_pos / compress_ratio) : 0;
    if (comp_keys > compressed_count) comp_keys = compressed_count;
    if (comp_keys < 0) comp_keys = 0;

    const int comp_base_in_pool = sw_window + n_tokens;
    int32_t* row = indices + (size_t)token * topk_unified;

    for (int j = threadIdx.x; j < sw_count; j += blockDim.x) {
        int p = sw_start + j;
        int slot = (p < start_pos)
                 ? (p - sw_base)
                 : (sw_window + (p - start_pos));
        row[j] = slot;
    }

    for (int k = threadIdx.x; k < comp_keys; k += blockDim.x) {
        row[sw_count + k] = comp_base_in_pool + k;
    }

    int pad_start = sw_count + comp_keys;
    for (int k = pad_start + threadIdx.x; k < topk_unified; k += blockDim.x) {
        row[k] = -1;
    }

    if (threadIdx.x == 0) {
        topk_length[token] = sw_count + comp_keys;
    }
}

__global__ void arle_chain_verify_build_indices_kernel(
        int32_t* __restrict__ indices,
        int32_t* __restrict__ topk_length,
        const int32_t* __restrict__ positions,
        const int32_t* __restrict__ ancestors,
        int max_anc,
        const int32_t* __restrict__ selected,
        int s_q,
        int start_pos,
        int sw_window,
        int index_topk,
        int max_compressed,
        int topk_unified,
        int n_tokens,
        int compressed_count,
        int compress_ratio) {
    int token = blockIdx.x;
    if (token >= s_q) return;

    const int abs_pos = positions[token];
    const int sw_start = max(0, abs_pos + 1 - sw_window);
    const int committed = max(0, start_pos - sw_start);
    const int sw_base = max(0, start_pos - sw_window);

    int32_t* row = indices + (size_t)token * topk_unified;

    for (int j = threadIdx.x; j < committed; j += blockDim.x) {
        int p = sw_start + j;
        row[j] = p - sw_base;
    }

    int anc = 0;
    if (threadIdx.x == 0) {
        const int32_t* anc_row = ancestors + (size_t)token * max_anc;
        for (int j = 0; j < max_anc; ++j) {
            int32_t a = anc_row[j];
            if (a < 0) break;
            row[committed + anc] = sw_window + a;
            ++anc;
        }
        row[committed + anc] = sw_window + token;
    }
    __syncthreads();

    {
        int count = 0;
        const int32_t* anc_row = ancestors + (size_t)token * max_anc;
        for (int j = 0; j < max_anc; ++j) {
            if (anc_row[j] < 0) break;
            ++count;
        }
        anc = count;
    }
    const int chunk_part = anc + 1;
    const int comp_base_in_pool = sw_window + n_tokens;
    const int window_part = committed + chunk_part;

    int comp_part = 0;
    if (selected != nullptr) {
        comp_part = index_topk;
        const int32_t* sel = selected + (size_t)token * index_topk;
        for (int k = threadIdx.x; k < index_topk; k += blockDim.x) {
            int32_t c = sel[k];
            bool valid = (c >= 0) && (c < compressed_count);
            if (valid && compress_ratio > 0) {
                int block_end = c * compress_ratio + (compress_ratio - 1);
                if (block_end > abs_pos) valid = false;
            }
            row[window_part + k] = valid ? (comp_base_in_pool + c) : -1;
        }
    } else if (max_compressed > 0) {
        int comp_keys = (compress_ratio > 0) ? (abs_pos / compress_ratio) : 0;
        if (comp_keys > compressed_count) comp_keys = compressed_count;
        if (comp_keys < 0) comp_keys = 0;
        comp_part = comp_keys;
        for (int k = threadIdx.x; k < comp_keys; k += blockDim.x) {
            row[window_part + k] = comp_base_in_pool + k;
        }
    }

    int pad_start = window_part + comp_part;
    for (int k = pad_start + threadIdx.x; k < topk_unified; k += blockDim.x) {
        row[k] = -1;
    }

    if (threadIdx.x == 0) {
        topk_length[token] = pad_start;
    }
}

}  // namespace

extern "C" {

cudaError_t arle_flashmla_csa_pack_kv(
        __nv_bfloat16* kv_unified,
        const __nv_bfloat16* window_cache,
        const __nv_bfloat16* k_prepared,
        const __nv_bfloat16* compressed,
        int start_pos,
        int sw_window,
        int n_tokens,
        int compressed_count,
        int d_qk,
        cudaStream_t stream) {
    const size_t row_bytes = (size_t)d_qk * sizeof(__nv_bfloat16);

    if (start_pos > 0) {
        constexpr int kBlock = 256;
        arle_csa_pack_sw_region_kernel<<<sw_window, kBlock, 0, stream>>>(
            kv_unified, window_cache, sw_window,
            std::max(0, start_pos - sw_window), d_qk);
        auto err = cudaGetLastError();
        if (err != cudaSuccess) return err;
    }

    if (n_tokens > 0) {
        auto err = cudaMemcpyAsync(
            kv_unified + (size_t)sw_window * d_qk,
            k_prepared,
            (size_t)n_tokens * row_bytes,
            cudaMemcpyDeviceToDevice, stream);
        if (err != cudaSuccess) return err;
    }

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
    // FlashMLA requires topk % 128 == 0.
    if ((topk_unified & 127) != 0) return cudaErrorInvalidValue;

    const int sw_base = std::max(0, start_pos - sw_window);
    constexpr int kBlock = 128;
    arle_csa_build_indices_kernel<<<s_q, kBlock, 0, stream>>>(
        indices, topk_length, selected,
        s_q, start_pos, sw_window, index_topk, topk_unified,
        s_q, compressed_count, compress_ratio, sw_base);
    return cudaGetLastError();
}

cudaError_t arle_flashmla_hca_build_indices(
        int32_t* indices,
        int32_t* topk_length,
        int s_q,
        int start_pos,
        int sw_window,
        int max_compressed_keys,
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
        s_q, compressed_count, compress_ratio, sw_base);
    return cudaGetLastError();
}

cudaError_t arle_flashmla_chain_verify_build_indices(
        int32_t* indices,
        int32_t* topk_length,
        const int32_t* positions,
        const int32_t* ancestors,
        int max_anc,
        const int32_t* selected,
        int s_q,
        int start_pos,
        int sw_window,
        int index_topk,
        int max_compressed,
        int topk_unified,
        int compressed_count,
        int compress_ratio,
        cudaStream_t stream) {
    if (s_q <= 0) return cudaSuccess;
    if (sw_window <= 0 || index_topk < 0 || max_compressed < 0 ||
        compressed_count < 0 || start_pos < 0 || max_anc < 0 ||
        positions == nullptr || (ancestors == nullptr && max_anc > 0)) {
        return cudaErrorInvalidValue;
    }
    if ((topk_unified & 127) != 0) return cudaErrorInvalidValue;
    const int comp_cap = (selected != nullptr) ? index_topk : max_compressed;
    if (sw_window + max_anc + 1 + comp_cap > topk_unified) {
        return cudaErrorInvalidValue;
    }
    constexpr int kBlock = 128;
    arle_chain_verify_build_indices_kernel<<<s_q, kBlock, 0, stream>>>(
        indices, topk_length, positions, ancestors, max_anc, selected,
        s_q, start_pos, sw_window, index_topk, max_compressed, topk_unified,
        s_q, compressed_count, compress_ratio);
    return cudaGetLastError();
}

}  // extern "C"
