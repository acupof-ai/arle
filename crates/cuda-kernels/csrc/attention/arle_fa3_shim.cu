// ARLE torch-free shim over vendored FA3 hopper fwd (hdim256, bf16, sm_90a).
//
// Replaces flash_api.cpp's torch surface with a C ABI; the param-fill mirrors
// mha_fwd's non-varlen batch=1 flow exactly (vendor/flash-attention/hopper/
// flash_api.cpp:849-1198 — set_params_fprop + the scheduler-semaphore block).
// Step-1 scope:
//   - b=1, contiguous q/o, contiguous slot KV viewed at exact seqlen_k
//     (no seqused_k => is_varlen=false => no prepare_varlen machinery),
//   - num_splits=1 for prefill; opt-in decode may pass num_splits>1
//     (out_accum + softmax_lse_accum + combine), which forces PackGQA,
//   - causal bottom-right alignment (chunked-prefill semantics), with the
//     upstream seqlen_q==1 causal->non-causal demotion (hdim 256 branch).
// The split + packgqa + paged instantiations are vendored and compiled so the
// decode tranche can extend this shim without touching build.rs again.
//
// Compile contract (build.rs): FA3 flag set mirrors hopper/setup.py —
// -DNDEBUG and -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED are mandatory
// (upstream marks NDEBUG "otherwise performance is severely impacted").

#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>

#include "flash.h"

template <typename T, typename Tpartial, int kBlockK>
void run_mha_fwd_combine_(Flash_fwd_params &params, cudaStream_t stream,
                          bool enable_pdl);

namespace {

inline int round_multiple(int x, int m) { return (x + m - 1) / m * m; }

inline int device_num_sm() {
    static int cached = -1;
    if (cached < 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&cached, cudaDevAttrMultiProcessorCount, dev);
    }
    return cached;
}

}  // namespace

extern "C" {

// All strides in ELEMENTS; the last dim (head_dim) must be contiguous. The
// separate row/head strides express both token-major q/o ([S, h, d]:
// row=h*d, head=d) and the qwen35 head-major slot caches ([h_k, max_seq, d]:
// row=d, head=max_seq*d) without any relayout — they feed FA3's TMA
// descriptors directly (16-byte alignment required, satisfied for d=256).
typedef struct {
    const void* q;              // bf16, seqlen_q x h x d view
    const void* k;              // bf16, seqlen_k x h_k x d view of the cache
    const void* v;
    void* o;                    // bf16, seqlen_q x h x d view
    float* softmax_lse;         // fp32 [h * seqlen_q] scratch
    float* out_accum;           // fp32 [splits, b=1, h, seqlen_q, d], split only
    float* softmax_lse_accum;   // fp32 [splits, b=1, h, seqlen_q], split only
    int* tile_count_semaphore;  // device i32 scratch (>= 1 element)
    int seqlen_q;
    int seqlen_k;
    int num_heads;    // h   (Qwen3.6: 16)
    int num_heads_k;  // h_k (Qwen3.6: 2)
    int head_dim;     // must be 256
    long long q_row_stride;
    long long k_row_stride;
    long long v_row_stride;
    long long o_row_stride;
    long long q_head_stride;
    long long k_head_stride;
    long long v_head_stride;
    long long o_head_stride;
    float softmax_scale;
    int is_causal;
    int num_splits;  // 1 = direct fwd; >1 = split-KV + combine (<=256)
} ArleFa3FwdHd256Args;

cudaError_t arle_fa3_fwd_hd256_bf16_cuda(const ArleFa3FwdHd256Args* a,
                                         cudaStream_t stream) {
    if (a == nullptr || a->head_dim != 256 || a->seqlen_q <= 0 ||
        a->seqlen_k <= 0 || a->num_heads <= 0 || a->num_heads_k <= 0 ||
        a->num_heads % a->num_heads_k != 0) {
        return cudaErrorInvalidValue;
    }

    Flash_fwd_params params{};
    params.is_bf16 = true;

    params.q_ptr = const_cast<void*>(a->q);
    params.k_ptr = const_cast<void*>(a->k);
    params.v_ptr = const_cast<void*>(a->v);
    params.o_ptr = a->o;
    params.q_row_stride = a->q_row_stride;
    params.k_row_stride = a->k_row_stride;
    params.v_row_stride = a->v_row_stride;
    params.o_row_stride = a->o_row_stride;
    params.q_head_stride = a->q_head_stride;
    params.k_head_stride = a->k_head_stride;
    params.v_head_stride = a->v_head_stride;
    params.o_head_stride = a->o_head_stride;
    params.v_dim_stride = 1;
    // b=1: batch strides are never walked past index 0, but mirror the
    // non-varlen fill (flash_api.cpp:101-108) so the TMA descriptors see
    // self-consistent extents. Take the larger of the row/head walks so both
    // token-major ([S, h, d]) and head-major cache ([h_k, max_seq, d]) views
    // yield a plausible batch extent.
    auto batch_extent = [](int64_t rows, int64_t row_stride, int64_t heads,
                           int64_t head_stride) {
        int64_t by_row = rows * row_stride;
        int64_t by_head = heads * head_stride;
        return by_row > by_head ? by_row : by_head;
    };
    params.q_batch_stride =
        batch_extent(a->seqlen_q, a->q_row_stride, a->num_heads, a->q_head_stride);
    params.k_batch_stride =
        batch_extent(a->seqlen_k, a->k_row_stride, a->num_heads_k, a->k_head_stride);
    params.v_batch_stride =
        batch_extent(a->seqlen_k, a->v_row_stride, a->num_heads_k, a->v_head_stride);
    params.o_batch_stride =
        batch_extent(a->seqlen_q, a->o_row_stride, a->num_heads, a->o_head_stride);

    params.softmax_lse_ptr = a->softmax_lse;

    params.b = 1;
    params.b_k = 1;
    params.h = a->num_heads;
    params.h_k = a->num_heads_k;
    params.seqlen_q = a->seqlen_q;
    params.seqlen_k = a->seqlen_k;
    params.total_q = a->seqlen_q;
    params.total_k = a->seqlen_k;
    params.seqlen_q_rounded = round_multiple(a->seqlen_q, 128);
    params.seqlen_k_rounded = round_multiple(a->seqlen_k, 128);
    params.d = 256;
    params.d_rounded = 256;
    params.dv = 256;
    params.dv_rounded = 256;

    params.scale_softmax = a->softmax_scale;
    params.softcap = 0.0f;

    // Dropout disabled: keep-probability form (flash_api.cpp:134-140).
    params.p_dropout = 1.0f;
    params.p_dropout_in_uint8_t = 255;
    params.rp_dropout = 1.0f;

    // Causal/window fill mirrors flash_api.cpp:576-597. seqlen_q==1 demotes
    // causal to full attention (identical math, better tile scheduling; the
    // hdim>128 branch applies for d=256).
    bool is_causal = a->is_causal != 0 && a->seqlen_q > 1;
    params.is_causal = is_causal;
    params.is_local = false;
    params.window_size_left = a->seqlen_k - 1;
    params.window_size_right = is_causal ? 0 : a->seqlen_q - 1;
    params.attention_chunk = 0;

    params.arch = 90;
    params.num_sm = device_num_sm();

    params.page_table = nullptr;
    params.page_size = 1;
    params.num_pages = 0;
    params.pagedkv_tma = false;

    params.num_splits = a->num_splits <= 1 ? 1 : a->num_splits;
    if (params.num_splits > 256) return cudaErrorInvalidValue;
    // Upstream always enables PackGQA for split to avoid rereading the same KV
    // head for each GQA query head.
    params.pack_gqa = params.num_splits > 1;

    // Scheduler template selectors (flash_api.cpp:993-994).
    params.varlen_sort_batches = true;  // !is_local
    params.head_swizzle = params.is_causal;

    // Non-varlen, sm90: upstream needs a zeroed semaphore for causal
    // non-split. The split decode path also passes it through combine, so keep
    // one reusable zeroed device i32 for both cases.
    if (params.is_causal || params.num_splits > 1) {
        if (a->tile_count_semaphore == nullptr) return cudaErrorInvalidValue;
        cudaError_t st =
            cudaMemsetAsync(a->tile_count_semaphore, 0, sizeof(int), stream);
        if (st != cudaSuccess) return st;
        params.tile_count_semaphore = a->tile_count_semaphore;
    } else {
        params.tile_count_semaphore = nullptr;
    }
    params.tile_count_semaphore_offset = 0;
    params.skip_scheduler_metadata_computation = false;
    params.num_splits_dynamic_ptr = nullptr;
    params.num_m_blocks_ptr = nullptr;
    params.varlen_batch_idx_ptr = nullptr;
    params.num_nheads_in_l2_ptr = nullptr;

    if (params.num_splits > 1) {
        if (a->out_accum == nullptr || a->softmax_lse_accum == nullptr) {
            return cudaErrorInvalidValue;
        }
        // Match flash_api_stable.cpp's non-varlen split allocation:
        // out_accum [splits, batch=1, h, seqlen_q, dv] contiguous,
        // lse_accum [splits, batch=1, h, seqlen_q] contiguous.
        params.is_fp32 = false;
        params.oaccum_ptr = a->out_accum;
        params.softmax_lseaccum_ptr = a->softmax_lse_accum;
        params.oaccum_split_stride =
            static_cast<int64_t>(a->num_heads) * a->seqlen_q * a->head_dim;
        params.oaccum_batch_stride = params.oaccum_split_stride;
        params.oaccum_head_stride =
            static_cast<int64_t>(a->seqlen_q) * a->head_dim;
        params.oaccum_row_stride = a->head_dim;
        params.lseaccum_split_stride =
            static_cast<int64_t>(a->num_heads) * a->seqlen_q;
        params.lseaccum_batch_stride = params.lseaccum_split_stride;
        params.lseaccum_head_stride = a->seqlen_q;

        run_mha_fwd_<90, cutlass::bfloat16_t, 256, 256, /*Split=*/true,
                     /*PagedKVNonTMA=*/false, /*Has_softcap=*/false,
                     /*PackGQA=*/true>(params, stream);
        cudaError_t st = cudaGetLastError();
        if (st != cudaSuccess) return st;
        params.is_bf16 = true;
        run_mha_fwd_combine_<cutlass::bfloat16_t, float, 128>(
            params, stream, true /*enable_pdl*/);
        return cudaGetLastError();
    }

    run_mha_fwd_<90, cutlass::bfloat16_t, 256, 256, /*Split=*/false,
                 /*PagedKVNonTMA=*/false, /*Has_softcap=*/false,
                 /*PackGQA=*/false>(params, stream);
    return cudaGetLastError();
}

// Stub-detection marker (build.rs validates the archive exports this and the
// Rust side can assert the real shim linked — flashmla pattern).
int arle_fa3_real_kernel_marker_cuda(void) { return 1; }

}  // extern "C"
