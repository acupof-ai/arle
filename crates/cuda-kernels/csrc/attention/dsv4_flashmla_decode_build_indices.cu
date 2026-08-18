// DSv4 FlashMLA sparse-decode indices builder (block-paged FP8 KV pool coords).

#include <cuda_runtime.h>
#include <cstdint>

namespace {

// Stage-B page-table lookup: when page_table != nullptr, slot-relative `out`
// carries a logical page routed to physical via page_table[logical], emitting
// pool-absolute indices (batched kernel skips its block_offset add).
// page_table == nullptr (Stage-A) keeps `out` slot-relative verbatim.
__device__ __forceinline__ int32_t arle_dsv4_decode_route_index(
        int32_t out, const int32_t* __restrict__ page_table,
        int num_logical_pages, int page_block_size, int total_blocks) {
    if (out < 0 || page_table == nullptr) return out;
    const int logical = out / page_block_size;
    if (logical < 0 || logical >= num_logical_pages) return -1;
    const int physical = page_table[logical];
    if (physical < 0 || (total_blocks > 0 && physical >= total_blocks)) return -1;
    return physical * page_block_size + (out % page_block_size);
}

__device__ __forceinline__ int32_t arle_dsv4_flashmla_decode_index_at(
        int tid,
        const int32_t* __restrict__ selected,
        int sw_blocks,
        int sliding_window,
        int start_pos,
        int max_compressed_keys,
        int compress_ratio,
        int mode_int,
        int page_block_size,
        const int32_t* __restrict__ page_table,
        int num_logical_pages,
        int total_blocks) {
    if (start_pos < 0) return -1;
    int sw_start = start_pos - sliding_window + 1;
    if (sw_start < 0) sw_start = 0;
    const int sw_count = start_pos - sw_start + 1;

    if (tid < sw_count) {
        const int p = sw_start + tid;
        const int ring_idx = p % sliding_window;
        const int block_id = ring_idx / page_block_size;
        const int row_in_block = ring_idx % page_block_size;
        int32_t out = block_id * page_block_size + row_in_block;
        // Defensive: mask if position maps past sw_blocks (config drift).
        if (block_id >= sw_blocks) out = -1;
        return arle_dsv4_decode_route_index(out, page_table, num_logical_pages, page_block_size, total_blocks);
    } else if (tid < sw_count + max_compressed_keys) {
        const int k = tid - sw_count;
        if (mode_int == 1) {
            int32_t c = (selected != nullptr) ? selected[k] : -1;
            bool valid = (c >= 0);
            // Causality: block c covers [c*ratio, (c+1)*ratio-1]; mask future blocks.
            if (valid && compress_ratio > 0) {
                const int block_end = c * compress_ratio + (compress_ratio - 1);
                if (block_end > start_pos) valid = false;
            }
            if (valid) {
                const int abs_block = sw_blocks + c / page_block_size;
                const int row_in_block = c % page_block_size;
                return arle_dsv4_decode_route_index(
                    abs_block * page_block_size + row_in_block, page_table,
                    num_logical_pages, page_block_size, total_blocks);
            } else {
                return -1;
            }
        } else {
            // Causal floor: only compressed blocks fully past are visible.
            int comp_keys = (compress_ratio > 0) ? (start_pos / compress_ratio) : 0;
            if (k < comp_keys) {
                const int r = k;
                const int abs_block = sw_blocks + r / page_block_size;
                const int row_in_block = r % page_block_size;
                return arle_dsv4_decode_route_index(
                    abs_block * page_block_size + row_in_block, page_table,
                    num_logical_pages, page_block_size, total_blocks);
            } else {
                return -1;
            }
        }
    }
    return -1;
}

__global__ void arle_dsv4_flashmla_decode_build_indices_kernel(
        int32_t* __restrict__ indices,
        const int32_t* __restrict__ selected,
        int sw_blocks,
        int sliding_window,
        int start_pos,
        int max_compressed_keys,
        int compress_ratio,
        int mode_int,
        int page_block_size,
        const int32_t* __restrict__ page_table,
        int num_logical_pages,
        int total_blocks,
        int topk_unified) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= topk_unified) return;
    indices[tid] = arle_dsv4_flashmla_decode_index_at(
        tid, selected, sw_blocks, sliding_window, start_pos,
        max_compressed_keys, compress_ratio, mode_int, page_block_size,
        page_table, num_logical_pages, total_blocks);
}

__global__ void arle_dsv4_flashmla_decode_build_indices_start_pos_ptr_kernel(
        int32_t* __restrict__ indices,
        const int32_t* __restrict__ selected,
        int sw_blocks,
        int sliding_window,
        const int32_t* __restrict__ start_pos_ptr,
        int max_compressed_keys,
        int compress_ratio,
        int mode_int,
        int page_block_size,
        const int32_t* __restrict__ page_table,
        int num_logical_pages,
        int total_blocks,
        int topk_unified) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= topk_unified) return;
    const int start_pos = *start_pos_ptr;
    indices[tid] = arle_dsv4_flashmla_decode_index_at(
        tid, selected, sw_blocks, sliding_window, start_pos,
        max_compressed_keys, compress_ratio, mode_int, page_block_size,
        page_table, num_logical_pages, total_blocks);
}

__global__ void arle_dsv4_flashmla_decode_build_indices_batched_kernel(
        int32_t* __restrict__ indices,
        const int32_t* __restrict__ start_pos,
        const int32_t* __restrict__ slot_layer_block_offsets,
        const int32_t* __restrict__ selected,
        int32_t* __restrict__ topk_length,
        int b,
        int sw_blocks,
        int sliding_window,
        int max_compressed_keys,
        int compress_ratio,
        int mode_int,
        int page_block_size,
        int total_blocks,
        const int32_t* __restrict__ page_table,
        int num_logical_pages,
        int topk_unified) {
    const int row = blockIdx.y;
    if (row >= b) return;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= topk_unified) return;

    if (tid == 0) {
        topk_length[row] = topk_unified;
    }

    const int32_t* selected_row =
        selected != nullptr ? selected + static_cast<int64_t>(row) * max_compressed_keys : nullptr;
    // Stage-B: page table yields pool-absolute indices; skip block_offset add.
    // Stage-A (null): keep the band shift.
    const int32_t* page_table_row =
        page_table != nullptr ? page_table + static_cast<int64_t>(row) * num_logical_pages : nullptr;
    int32_t out = arle_dsv4_flashmla_decode_index_at(
        tid, selected_row, sw_blocks, sliding_window, start_pos[row],
        max_compressed_keys, compress_ratio, mode_int, page_block_size,
        page_table_row, num_logical_pages, total_blocks);
    if (out >= 0 && page_table == nullptr) {
        const int block_offset = slot_layer_block_offsets[row];
        if (block_offset < 0 || block_offset >= total_blocks) {
            out = -1;
        } else {
            out += block_offset * page_block_size;
        }
    }
    indices[static_cast<int64_t>(row) * topk_unified + tid] = out;
}

}  // namespace

extern "C" {

cudaError_t arle_dsv4_flashmla_decode_build_indices_start_pos_ptr_cuda(
        int32_t* indices,
        const int32_t* selected,
        int sw_blocks,
        int sliding_window,
        const int32_t* start_pos_ptr,
        int max_compressed_keys,
        int compress_ratio,
        int mode_int,
        int page_block_size,
        const int32_t* page_table,
        int num_logical_pages,
        int total_blocks,
        cudaStream_t stream) {
    if (indices == nullptr || start_pos_ptr == nullptr) return cudaErrorInvalidValue;
    if (sliding_window <= 0) return cudaErrorInvalidValue;
    if (max_compressed_keys < 0 || page_block_size <= 0) return cudaErrorInvalidValue;
    if (mode_int != 1 && mode_int != 2) return cudaErrorInvalidValue;
    if (mode_int == 1 && selected == nullptr) return cudaErrorInvalidValue;
    if (sw_blocks < 0) return cudaErrorInvalidValue;
    if (page_table != nullptr && num_logical_pages <= 0) return cudaErrorInvalidValue;

    const int topk_unified = sliding_window + max_compressed_keys;
    if ((topk_unified & 127) != 0) return cudaErrorInvalidValue;

    constexpr int kBlock = 128;
    const int grid = (topk_unified + kBlock - 1) / kBlock;
    arle_dsv4_flashmla_decode_build_indices_start_pos_ptr_kernel<<<grid, kBlock, 0, stream>>>(
        indices, selected, sw_blocks, sliding_window, start_pos_ptr,
        max_compressed_keys, compress_ratio, mode_int, page_block_size,
        page_table, num_logical_pages, total_blocks, topk_unified);
    return cudaGetLastError();
}

cudaError_t arle_dsv4_flashmla_decode_build_indices_batched_cuda(
        int32_t* indices,
        const int32_t* start_pos,
        const int32_t* slot_layer_block_offsets,
        const int32_t* selected,
        int32_t* topk_length,
        int b,
        int sw_blocks,
        int sliding_window,
        int max_compressed_keys,
        int compress_ratio,
        int mode_int,
        int page_block_size,
        int total_blocks,
        const int32_t* page_table,
        int num_logical_pages,
        cudaStream_t stream) {
    if (indices == nullptr || start_pos == nullptr || slot_layer_block_offsets == nullptr ||
        topk_length == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (b <= 0 || sliding_window <= 0) return cudaErrorInvalidValue;
    if (max_compressed_keys < 0 || page_block_size <= 0) return cudaErrorInvalidValue;
    if (mode_int != 1 && mode_int != 2) return cudaErrorInvalidValue;
    if (mode_int == 1 && selected == nullptr) return cudaErrorInvalidValue;
    if (sw_blocks < 0 || total_blocks <= 0 || sw_blocks > total_blocks) {
        return cudaErrorInvalidValue;
    }
    if (page_table != nullptr && num_logical_pages <= 0) return cudaErrorInvalidValue;

    const int topk_unified = sliding_window + max_compressed_keys;
    if ((topk_unified & 127) != 0) return cudaErrorInvalidValue;

    constexpr int kBlock = 128;
    dim3 grid((topk_unified + kBlock - 1) / kBlock, static_cast<unsigned>(b), 1);
    arle_dsv4_flashmla_decode_build_indices_batched_kernel<<<grid, kBlock, 0, stream>>>(
        indices, start_pos, slot_layer_block_offsets, selected, topk_length, b, sw_blocks,
        sliding_window, max_compressed_keys, compress_ratio, mode_int, page_block_size,
        total_blocks, page_table, num_logical_pages, topk_unified);
    return cudaGetLastError();
}

}  // extern "C"
