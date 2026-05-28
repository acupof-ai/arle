// ARLE → FlashMLA (sgl-project/FlashMLA @ df022eb) sparse decode shim.
//
// Wraps the SM90 sparse-FP8 split-KV decode kernel + the CPU-side
// `get_decoding_sched_meta` scheduler + the `combine` kernel that merges
// per-split partial outputs into the final response.
//
// !!! IMPORTANT — KV layout reality check !!!
//
// `SparseAttnDecodeParams::kv` is typed `cutlass::bfloat16_t*` in
// vendor/flashmla/csrc/params.h but the kernel reinterprets it as `fp8*`
// and asserts `stride_kv_row == BYTES_PER_TOKEN` (a model-specific packed
// byte layout):
//
//   MODEL1 (d_qk == 512):  448 fp8 NoPE + 128 bf16 RoPE +
//                           8 fp8_e8m0 scales = 584 bytes/token
//   V32    (d_qk == 576):  512 fp8 NoPE + 128 bf16 RoPE +
//                          16 bytes  scales = 656 bytes/token
//
// See vendor/flashmla/csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh:
// the kernel calls `cvt_fp8x8_bf16x8` per-tile to dequantize FP8 NoPE
// + the fp8_e8m0 scales, and reads the bf16 RoPE tail. This shim does
// **NOT** quantize ARLE's existing bf16 KV pool — callers must provide a
// pre-packed FP8 KV buffer. The bf16-typed `q` argument matches upstream
// (Q stays bf16).
//
// The KV-quantization kernel that converts ARLE's bf16 sliding-window
// cache + bf16 compressed pool into this FP8-packed contract is a separate
// project (~weeks of kernel work) and is the gating dependency for
// `dsv4_flashmla_decode_enabled()` to default ON. Until then this shim
// returns cudaErrorInvalidValue at the pre-flight if invariants fail and
// is only intended for the future wire-up + isolated FlashMLA unit tests.
//
// V2.4 lesson: phase1.cuh:457-458 unconditionally SM90_BULK_COPY_S2G writes
// to params.max_logits and params.lse. The decode kernel has the same
// shape for lse_accum / o_accum (each split writes its partial). This
// shim allocates `lse`, `lse_accum`, `o_accum` real buffers via the
// caller (we never accept nullptr for the split-KV scratch).
//
// Refs:
//   docs/plans/2026-05-28-dsv4-flashmla-decode-integration.md
//   docs/experience/wins/2026-05-28-dsv4-v2-4-flashmla-root-cause-fix.md
//   crates/cuda-kernels/vendor/flashmla/csrc/sm90/decode/sparse_fp8/
//   crates/cuda-kernels/vendor/flashmla/csrc/smxx/decode/

#include <cuda_runtime.h>
#include <cstdint>
#include <exception>
#include <stdexcept>

// FlashMLA internals (vendored): SparseAttnDecodeParams + the SM90 entry
// + combine + sched meta. FlashMLA's `params.h` only pulls in
// `cutlass/bfloat16.h` — no torch.
#include "../../vendor/flashmla/csrc/params.h"
#include "../../vendor/flashmla/csrc/sm90/decode/sparse_fp8/splitkv_mla.h"
#include "../../vendor/flashmla/csrc/smxx/decode/combine/combine.h"
#include "../../vendor/flashmla/csrc/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.h"

// MODEL1 = d_qk 512, V32 = d_qk 576.
// Encoded as `model_type_int`: 0 = V32, 1 = MODEL1 — matches the
// `ModelType` enum order in vendor/flashmla/csrc/params.h.
namespace {
constexpr int MODEL1_BYTES_PER_TOKEN = 584; // 448 + 128 + 8
constexpr int V32_BYTES_PER_TOKEN = 656;    // 512 + 128 + 16
}

extern "C" {

// Reports the FP8 KV bytes/token packed layout for a (d_qk, model_type_int)
// pair so ARLE-side allocation code can size buffers consistently with the
// kernel's stride_kv_row hard-assert. Returns -1 for unsupported pairs.
//
// model_type_int: 0 = V32 (d_qk=576), 1 = MODEL1 (d_qk=512).
int32_t arle_flashmla_sm90_sparse_decode_bytes_per_token(
    int32_t d_qk,
    int32_t model_type_int
) {
    if (model_type_int == 0 && d_qk == 576) return V32_BYTES_PER_TOKEN;
    if (model_type_int == 1 && d_qk == 512) return MODEL1_BYTES_PER_TOKEN;
    return -1;
}

// Compute the decode scheduler tuning meta (`num_sm_parts`,
// `fixed_overhead_num_blocks`, `block_size_topk`) for a (h_q, s_q,
// model_type_int) tuple by invoking
// `sm90::decode::sparse_fp8::Decode_Sm90_Impl::get_meta(h_q, s_q)` on the
// host. The values are CUDA-device-property dependent; arch detection is
// done by `kerutils::Arch` (which reads `cudaGetDeviceProperties` from the
// current device).
//
// Caller writes the three meta ints into out_meta in this order so they
// can be used to size the GPU-side `tile_scheduler_metadata` / `num_splits`
// tensors before the sched_meta kernel launch.
cudaError_t arle_flashmla_sm90_sparse_decode_get_meta(
    int32_t h_q,
    int32_t s_q,
    int32_t model_type_int,
    int32_t* out_num_sm_parts,
    int32_t* out_fixed_overhead_num_blocks,
    int32_t* out_block_size_topk
) {
    if (h_q != 64 && h_q != 128) return cudaErrorInvalidValue;
    if (s_q <= 0) return cudaErrorInvalidValue;
    if (model_type_int != 0 && model_type_int != 1) return cudaErrorInvalidValue;
    if (!out_num_sm_parts || !out_fixed_overhead_num_blocks || !out_block_size_topk) {
        return cudaErrorInvalidValue;
    }
    try {
        sm90::decode::sparse_fp8::Decode_Sm90_Impl impl;
        // Decode_Sm90_Impl is declared in vendor/flashmla/csrc/api/sparse_decode.h
        // — but that header pulls in <ATen/...> via TORCH_CHECK macros. To
        // avoid a libtorch dependency we re-implement the meta computation
        // locally from the upstream formula (api/sparse_decode.h:60-66):
        //   {std::max(arch.num_sms / s_q / (h_q/64), 1), 5, 64}
        cudaDeviceProp prop;
        int dev = 0;
        cudaError_t st = cudaGetDevice(&dev);
        if (st != cudaSuccess) return st;
        st = cudaGetDeviceProperties(&prop, dev);
        if (st != cudaSuccess) return st;
        int num_sms = prop.multiProcessorCount;
        int denom = s_q * (h_q / 64);
        int n = (denom > 0) ? (num_sms / denom) : num_sms;
        if (n < 1) n = 1;
        *out_num_sm_parts = n;
        *out_fixed_overhead_num_blocks = 5;
        *out_block_size_topk = 64;
        (void)impl; // silence unused-warning if NDEBUG strips ctor side-effects
    } catch (const std::exception&) {
        return cudaErrorInvalidValue;
    } catch (...) {
        return cudaErrorUnknown;
    }
    return cudaSuccess;
}

// Run the CPU-side decode scheduler kernel. Populates
// `tile_scheduler_metadata` (`num_sm_parts * DecodingSchedMetaSize/4`
// int32s) and `num_splits` (`b+1` int32s) from `topk_length` (per-batch
// effective topk for sparse decode; pass nullptr for `topk` = -1 dense
// path). `extra_topk_length` is for V32's two-tier (sliding-window +
// compressed) split — pass nullptr if no extra topk.
cudaError_t arle_flashmla_sm90_sparse_decode_sched_meta(
    int32_t b,
    int32_t s_q,
    int32_t block_size_topk,
    int32_t fixed_overhead_num_blocks,
    int32_t topk,
    int32_t extra_topk,
    const int32_t* topk_length,
    const int32_t* extra_topk_length,
    int32_t* tile_scheduler_metadata,
    int32_t* num_splits,
    int32_t num_sm_parts,
    cudaStream_t stream
) {
    if (b <= 0 || s_q <= 0 || num_sm_parts <= 0) return cudaErrorInvalidValue;
    if (!tile_scheduler_metadata || !num_splits) return cudaErrorInvalidValue;
    GetDecodeSchedMetaParams p{};
    p.b = b;
    p.s_q = s_q;
    p.block_size_n = block_size_topk;
    p.fixed_overhead_num_blocks = fixed_overhead_num_blocks;
    p.topk = topk;
    p.extra_topk = extra_topk;
    p.topk_length = const_cast<int*>(topk_length);
    p.extra_topk_length = const_cast<int*>(extra_topk_length);
    p.seqlens_k_ptr = nullptr;
    p.tile_scheduler_metadata_ptr =
        reinterpret_cast<DecodingSchedMeta*>(tile_scheduler_metadata);
    p.num_splits_ptr = num_splits;
    p.num_sm_parts = num_sm_parts;
    p.stream = stream;
    try {
        smxx::decode::run_get_decoding_sched_meta_kernel(p);
    } catch (const std::exception&) {
        return cudaErrorInvalidValue;
    } catch (...) {
        return cudaErrorUnknown;
    }
    return cudaGetLastError();
}

// Run FlashMLA SM90 sparse decode + combine into a single shim call. All
// pointer args are CUDA device pointers. Strides are in element count
// for non-KV tensors and **in bytes for the FP8-packed `kv` block stride**
// (kernel asserts `stride_kv_row == BYTES_PER_TOKEN`).
//
// q: bf16 [b, s_q, h_q, d_qk] (contiguous along d_qk)
// kv: FP8-packed bytes [num_blocks, page_block_size=64, bytes_per_token]
//     with the exact byte layout the kernel requires (see top-of-file
//     and `arle_flashmla_sm90_sparse_decode_bytes_per_token`)
// indices: int32 [b, s_q, topk]
// topk_length: int32 [b] or nullptr
// attn_sink: float [h_q] or nullptr
// out: bf16 [b, s_q, h_q, d_v]
// lse: float [b, h_q, s_q] (split-KV → combine writes into here)
// lse_accum: float [num_sm_parts + b, s_q, h_q] split-KV scratch
// o_accum: float  [num_sm_parts + b, s_q, h_q, d_v] split-KV scratch
// tile_scheduler_metadata: int32 [num_sm_parts * DecodingSchedMetaSize/4]
//   pre-populated by `arle_flashmla_sm90_sparse_decode_sched_meta` above
// num_splits: int32 [b+1] pre-populated by sched_meta
//
// model_type_int: 0 = V32 (d_qk=576), 1 = MODEL1 (d_qk=512)
cudaError_t arle_flashmla_sm90_sparse_decode_fwd(
    const void* q,
    const void* kv,
    const int32_t* indices,
    const int32_t* topk_length,
    const float* attn_sink,
    void* out,
    float* lse,
    float* lse_accum,
    float* o_accum,
    const int32_t* tile_scheduler_metadata,
    const int32_t* num_splits,
    int32_t b,
    int32_t s_q,
    int32_t h_q,
    int32_t h_kv,
    int32_t d_qk,
    int32_t d_v,
    int32_t num_blocks,
    int32_t page_block_size,
    int32_t topk,
    int32_t num_sm_parts,
    int32_t model_type_int,
    float sm_scale,
    // strides (elements unless suffixed _bytes)
    int32_t stride_q_b,
    int32_t stride_q_s_q,
    int32_t stride_q_h_q,
    int32_t stride_kv_block_bytes,
    int32_t stride_kv_row_bytes,
    int32_t stride_indices_b,
    int32_t stride_indices_s_q,
    int32_t stride_lse_b,
    int32_t stride_lse_s_q,
    int32_t stride_o_b,
    int32_t stride_o_s_q,
    int32_t stride_o_h_q,
    int32_t stride_lse_accum_split,
    int32_t stride_lse_accum_s_q,
    int32_t stride_o_accum_split,
    int32_t stride_o_accum_s_q,
    int32_t stride_o_accum_h_q,
    cudaStream_t stream
) {
    // Hard pre-flight matching the upstream torch wrapper's TORCH_CHECKs
    // — fail cleanly with cudaErrorInvalidValue rather than crashing inside
    // the kernel where errors surface as illegal-memory-access at the next
    // sync. Mirrors `sparse_attn_decode_interface` in
    // vendor/flashmla/csrc/api/sparse_decode.h.
    if (b <= 0 || s_q <= 0 || h_q <= 0) return cudaErrorInvalidValue;
    if (h_kv != 1) return cudaErrorInvalidValue;
    if (h_q != 64 && h_q != 128) return cudaErrorInvalidValue;
    if (d_qk != 576 && d_qk != 512) return cudaErrorInvalidValue;
    if (d_v != 512) return cudaErrorInvalidValue;
    if (topk <= 0) return cudaErrorInvalidValue;
    if (num_sm_parts <= 0) return cudaErrorInvalidValue;
    if (page_block_size <= 0) return cudaErrorInvalidValue;
    if (model_type_int != 0 && model_type_int != 1) return cudaErrorInvalidValue;
    if (!q || !kv || !indices || !out || !lse || !lse_accum || !o_accum) {
        return cudaErrorInvalidValue;
    }
    if (!tile_scheduler_metadata || !num_splits) return cudaErrorInvalidValue;

    // FP8 KV byte-layout contract — the kernel's `KU_ASSERT(stride_kv_row ==
    // BYTES_PER_TOKEN)` will fire inside the launch if this doesn't match
    // upstream's per-model packing. Catch it here cleanly.
    int32_t expected_bpt = arle_flashmla_sm90_sparse_decode_bytes_per_token(
        d_qk, model_type_int);
    if (expected_bpt < 0 || stride_kv_row_bytes != expected_bpt) {
        return cudaErrorInvalidValue;
    }

    ModelType model_type =
        (model_type_int == 0) ? ModelType::V32 : ModelType::MODEL1;

    SparseAttnDecodeParams p{};
    p.b = b;
    p.s_q = s_q;
    p.h_q = h_q;
    p.h_kv = h_kv;
    p.d_qk = d_qk;
    p.d_v = d_v;
    p.sm_scale = sm_scale;
    // Log2-scaled softmax scale: kernel uses exp2f, so feeding the
    // pre-multiplied factor skips a multiply per iteration. LOG_2_E
    // (1.44269504f) per api/common.h.
    p.sm_scale_div_log2 = sm_scale * 1.44269504f;
    p.num_blocks = num_blocks;
    p.page_block_size = page_block_size;
    p.topk = topk;
    p.model_type = model_type;

    p.q = reinterpret_cast<cutlass::bfloat16_t*>(const_cast<void*>(q));
    // FP8-packed bytes; the cutlass::bfloat16_t typing is the upstream
    // declared type but the kernel reinterprets via `(fp8*)params.kv`.
    p.kv = reinterpret_cast<cutlass::bfloat16_t*>(const_cast<void*>(kv));
    p.indices = const_cast<int*>(indices);
    p.topk_length = const_cast<int*>(topk_length);
    p.attn_sink = const_cast<float*>(attn_sink);

    p.lse = lse;
    p.out = reinterpret_cast<cutlass::bfloat16_t*>(out);

    // No extra KV path — ARLE doesn't use the two-tier compressed/extra
    // split today (HCA is via the prefill kernel's topk_length, not via
    // the decode kernel's extra_kv mechanism).
    p.extra_num_blocks = 0;
    p.extra_page_block_size = 0;
    p.extra_topk = 0;
    p.extra_kv = nullptr;
    p.extra_indices = nullptr;
    p.extra_topk_length = nullptr;

    p.stride_q_b = stride_q_b;
    p.stride_q_s_q = stride_q_s_q;
    p.stride_q_h_q = stride_q_h_q;
    p.stride_kv_block = stride_kv_block_bytes;
    p.stride_kv_row = stride_kv_row_bytes;
    p.stride_indices_b = stride_indices_b;
    p.stride_indices_s_q = stride_indices_s_q;
    p.stride_lse_b = stride_lse_b;
    p.stride_lse_s_q = stride_lse_s_q;
    p.stride_o_b = stride_o_b;
    p.stride_o_s_q = stride_o_s_q;
    p.stride_o_h_q = stride_o_h_q;
    p.stride_extra_kv_block = 0;
    p.stride_extra_kv_row = 0;
    p.stride_extra_indices_b = 0;
    p.stride_extra_indices_s_q = 0;

    p.stream = stream;

    p.lse_accum = lse_accum;
    p.o_accum = o_accum;
    p.stride_lse_accum_split = stride_lse_accum_split;
    p.stride_lse_accum_s_q = stride_lse_accum_s_q;
    p.stride_o_accum_split = stride_o_accum_split;
    p.stride_o_accum_s_q = stride_o_accum_s_q;
    p.stride_o_accum_h_q = stride_o_accum_h_q;
    p.tile_scheduler_metadata_ptr = reinterpret_cast<DecodingSchedMeta*>(
        const_cast<int32_t*>(tile_scheduler_metadata));
    p.num_splits_ptr = const_cast<int*>(num_splits);
    p.num_sm_parts = num_sm_parts;

    // Dispatch via h_q × model_type. Catch KUException (extends
    // std::exception, not runtime_error — V1 lesson from prefill) plus
    // any other escapes.
    try {
        if (h_q == 64 && model_type == ModelType::MODEL1) {
            sm90::decode::sparse_fp8::run_flash_splitkv_mla_fp8_sparse_kernel<
                ModelType::MODEL1, 64>(p);
        } else if (h_q == 128 && model_type == ModelType::MODEL1) {
            sm90::decode::sparse_fp8::run_flash_splitkv_mla_fp8_sparse_kernel<
                ModelType::MODEL1, 128>(p);
        } else if (h_q == 64 && model_type == ModelType::V32) {
            sm90::decode::sparse_fp8::run_flash_splitkv_mla_fp8_sparse_kernel<
                ModelType::V32, 64>(p);
        } else if (h_q == 128 && model_type == ModelType::V32) {
            sm90::decode::sparse_fp8::run_flash_splitkv_mla_fp8_sparse_kernel<
                ModelType::V32, 128>(p);
        } else {
            return cudaErrorInvalidValue;
        }

        // Combine step: merges split-KV partial outputs into the final
        // output. Reads lse_accum / o_accum, writes lse + out. Skipped
        // implicitly when num_splits[b] = 1 (CTA early-returns in
        // combine.cu line 39-41).
        CombineParams cp{};
        cp.b = b;
        cp.s_q = s_q;
        cp.h_q = h_q;
        cp.d_v = d_v;
        cp.lse = lse;
        cp.out = out;
        cp.stride_lse_b = stride_lse_b;
        cp.stride_lse_s_q = stride_lse_s_q;
        cp.stride_o_b = stride_o_b;
        cp.stride_o_s_q = stride_o_s_q;
        cp.stride_o_h_q = stride_o_h_q;
        cp.lse_accum = lse_accum;
        cp.o_accum = o_accum;
        cp.stride_lse_accum_split = stride_lse_accum_split;
        cp.stride_lse_accum_s_q = stride_lse_accum_s_q;
        cp.stride_o_accum_split = stride_o_accum_split;
        cp.stride_o_accum_s_q = stride_o_accum_s_q;
        cp.stride_o_accum_h_q = stride_o_accum_h_q;
        cp.tile_scheduler_metadata_ptr = reinterpret_cast<DecodingSchedMeta*>(
            const_cast<int32_t*>(tile_scheduler_metadata));
        cp.num_splits_ptr = const_cast<int*>(num_splits);
        cp.num_sm_parts = num_sm_parts;
        cp.attn_sink = const_cast<float*>(attn_sink);
        cp.stream = stream;
        smxx::decode::run_flash_mla_combine_kernel<cutlass::bfloat16_t>(cp);
    } catch (const std::exception&) {
        return cudaErrorInvalidValue;
    } catch (...) {
        return cudaErrorUnknown;
    }
    return cudaGetLastError();
}

}  // extern "C"
