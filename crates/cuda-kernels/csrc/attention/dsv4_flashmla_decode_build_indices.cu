// Phase D-4 step 1 — DSv4 FlashMLA sparse-decode indices builder (GPU-side).
//
// Builds the unified per-token indices buffer that FlashMLA's
// `sm90::decode::sparse_fp8::run_flash_splitkv_mla_fp8_sparse_kernel`
// consumes, in the BLOCK-PAGED coord space of the FP8 KV pool described in
// `docs/plans/2026-05-28-dsv4-flashmla-decode-integration.md` Phase D-3'.
//
// Pool layout (one contiguous u8 buffer per layer, page_block_size=64):
//   blocks [0,            sw_blocks)                 ← SW sub-pool   (per-token K stream)
//   blocks [sw_blocks,    sw_blocks + comp_blocks)   ← compressed sub-pool (compressor output)
//
// The decode kernel ingests an `int32[s_q=1, topk_unified]` row of pool
// slot indices (slot = block_id * page_block_size + in_block_row). Negative
// entries are masked via the kernel's `is_token_valid` check.
//
// Row layout (matches the prefill-side `arle_flashmla_csa_build_indices` /
// `arle_flashmla_hca_build_indices` shape, modulo the coord scheme: prefill
// keeps a flat `[sw_window | k_prepared | compressed]` pool indexed by
// element offset; decode uses block-paged indices in [0, total_slots)):
//
//   [0,                    sw_count_dec)               ← SW slots
//   [sw_count_dec,         sw_count_dec + max_comp)    ← compressed selections
//                                                       (CSA: from `selected`;
//                                                        HCA: identity 0..comp_keys)
//   [sw_count_dec + max_comp, topk_unified)            ← -1 padding
//
// `topk_unified = sliding_window + max_compressed_keys` must be a multiple
// of 128 (FlashMLA invariant `topk % (2*B_TOPK) == 0` per
// `vendor/flashmla/csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh`).
//
// SW slot mapping (ring-indexed, mirrors the bf16 SW ring populated by
// `dsv4_update_window_cache_kernel`):
//   absolute position p ∈ [max(0, N - sliding_window + 1), N]  (N = start_pos)
//   ring_idx = p % sliding_window
//   pool_slot = (ring_idx / 64) * page_block_size + ring_idx % 64
//             = ring_idx  (since page_block_size == 64)
//   ⇒ pool_slot ∈ [0, sw_blocks * 64)
//
// Compressed coord mapping:
//   compressed row r causally valid iff (r + 1) * compress_ratio - 1 <= start_pos
//   pool_slot = sw_blocks * page_block_size + r
//             ∈ [sw_blocks * 64, (sw_blocks + comp_blocks) * 64)
//
// Refs:
//   crates/cuda-kernels/csrc/misc/arle_flashmla_csa_prep.cu (prefill twin)
//   crates/cuda-kernels/csrc/misc/arle_flashmla_decode_shim.cu (consumer)
//   docs/experience/wins/2026-05-28-dsv4-flashmla-decode-d4-plumbing.md  Finding 3

#include <cuda_runtime.h>
#include <cstdint>

namespace {

// One block, 128 threads. Each thread handles one position of the
// `topk_unified` row. `mode_int`: 1 = CSA (selected != nullptr),
//                                 2 = HCA (selected == nullptr, identity range).
__global__ void arle_dsv4_flashmla_decode_build_indices_kernel(
        int32_t* __restrict__ indices,          // [topk_unified] int32 (s_q=1)
        const int32_t* __restrict__ selected,   // [max_compressed_keys] int32 (CSA) or nullptr (HCA)
        int sw_blocks,                          // SW sub-pool block count (= ceil(sliding_window / page_block_size))
        int sliding_window,
        int start_pos,                          // absolute position of the (single) decode token
        int max_compressed_keys,                // index_topk (CSA) or padded compressed_count (HCA)
        int compress_ratio,
        int mode_int,
        int page_block_size,
        int topk_unified) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= topk_unified) return;

    // SW slot count for this decode token. Decode is single-token (s_q=1)
    // so the SW window covers positions [max(0, start_pos - sw + 1), start_pos].
    int sw_start = start_pos - sliding_window + 1;
    if (sw_start < 0) sw_start = 0;
    const int sw_count = start_pos - sw_start + 1;  // ∈ [1, sliding_window]

    // Segment dispatch.
    int32_t out;
    if (tid < sw_count) {
        // SW slot region.
        const int p = sw_start + tid;
        const int ring_idx = p % sliding_window;
        const int block_id = ring_idx / page_block_size;
        const int row_in_block = ring_idx % page_block_size;
        out = block_id * page_block_size + row_in_block;
        // Defensive: if a position would map past sw_blocks (config drift), mask.
        if (block_id >= sw_blocks) out = -1;
    } else if (tid < sw_count + max_compressed_keys) {
        const int k = tid - sw_count;
        if (mode_int == 1) {
            // CSA — index into `selected`, apply causality gate matching
            // dsv4_hybrid_attention_kernel:898-901: block c covers tokens
            // [c*ratio, (c+1)*ratio - 1]; mask when block_end > start_pos.
            int32_t c = (selected != nullptr) ? selected[k] : -1;
            bool valid = (c >= 0);
            if (valid && compress_ratio > 0) {
                const int block_end = c * compress_ratio + (compress_ratio - 1);
                if (block_end > start_pos) valid = false;
            }
            if (valid) {
                const int abs_block = sw_blocks + c / page_block_size;
                const int row_in_block = c % page_block_size;
                out = abs_block * page_block_size + row_in_block;
            } else {
                out = -1;
            }
        } else {
            // HCA — identity 0..comp_keys-1 into compressed; causality cap
            // mirrors dsv4_hybrid_attention_kernel:882.
            // floor(start_pos / ratio) — matches the reference HCA causal gate
            // (block_end < t ⟹ floor(t/ratio) kept blocks) and the legacy
            // hybrid kernel. Was `(start_pos+1)/ratio`, off by one at the SW
            // boundary (start_pos+1 ≡ 0 mod ratio).
            int comp_keys = (compress_ratio > 0) ? (start_pos / compress_ratio) : 0;
            // Caller's `max_compressed_keys` is the padded capacity; clamp
            // `comp_keys` to its lower bound (kept-keys) below.
            // Note: caller is responsible for ensuring max_compressed_keys
            // is the correct upper bound (typically the compressed buffer's
            // monotonic high-water mark rounded up); we just gate on it here.
            if (k < comp_keys) {
                const int r = k;
                const int abs_block = sw_blocks + r / page_block_size;
                const int row_in_block = r % page_block_size;
                out = abs_block * page_block_size + row_in_block;
            } else {
                out = -1;
            }
        }
    } else {
        // Padding tail.
        out = -1;
    }
    indices[tid] = out;
}

}  // namespace

extern "C" {

// Build the unified decode indices row in block-paged pool coords.
//
// Parameters:
//   indices                — out, int32 [topk_unified] (s_q = 1 implied)
//   selected               — in,  int32 [max_compressed_keys] (CSA mode) or nullptr (HCA)
//   sw_blocks              — SW sub-pool block count (caller computes as
//                            ceil(sliding_window / page_block_size))
//   sliding_window         — bf16 SW ring length (matches the FlashMLA SW
//                            sub-pool's sliding_window * 1 = sw_blocks*64 capacity)
//   start_pos              — absolute position of the (single) decode token
//   max_compressed_keys    — index_topk (CSA) or padded compressed_count (HCA)
//   compress_ratio         — ratio for compressor causality gate
//   mode_int               — 1 = CSA, 2 = HCA
//   page_block_size        — 64 for DSv4-Flash MODEL1 (matches upstream
//                            `vendor/flashmla/csrc/sm90/decode/sparse_fp8/config.h`)
//   stream                 — CUDA stream
//
// `topk_unified = sliding_window + max_compressed_keys` is derived inside
// (must be a multiple of 128 per FlashMLA's hard-assert).
cudaError_t arle_dsv4_flashmla_decode_build_indices_cuda(
        int32_t* indices,
        const int32_t* selected,
        int sw_blocks,
        int sliding_window,
        int start_pos,
        int max_compressed_keys,
        int compress_ratio,
        int mode_int,
        int page_block_size,
        cudaStream_t stream) {
    if (indices == nullptr) return cudaErrorInvalidValue;
    if (sliding_window <= 0 || start_pos < 0) return cudaErrorInvalidValue;
    if (max_compressed_keys < 0 || page_block_size <= 0) return cudaErrorInvalidValue;
    if (mode_int != 1 && mode_int != 2) return cudaErrorInvalidValue;
    if (mode_int == 1 && selected == nullptr) return cudaErrorInvalidValue;
    if (sw_blocks < 0) return cudaErrorInvalidValue;

    const int topk_unified = sliding_window + max_compressed_keys;
    if ((topk_unified & 127) != 0) return cudaErrorInvalidValue;  // 2 * B_TOPK = 128

    constexpr int kBlock = 128;
    const int grid = (topk_unified + kBlock - 1) / kBlock;
    arle_dsv4_flashmla_decode_build_indices_kernel<<<grid, kBlock, 0, stream>>>(
        indices, selected, sw_blocks, sliding_window, start_pos,
        max_compressed_keys, compress_ratio, mode_int, page_block_size,
        topk_unified);
    return cudaGetLastError();
}

}  // extern "C"
