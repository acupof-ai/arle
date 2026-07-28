// ARLE torch-free shim over vendored FA3 hopper fwd (hdim256, bf16, sm_90a).
//
// Replaces flash_api.cpp's torch surface with a C ABI; the param-fill mirrors
// mha_fwd (vendor/flash-attention/hopper/flash_api.cpp:849-1198 —
// set_params_fprop + the scheduler-semaphore block).
//
// ONE call per layer, whatever the batch: q/o are packed [total_q, h, d] with
// `cu_seqlens_q` (so rows may have different query lengths — decode is qlen 1
// everywhere, spec verify is one chain per row), and each row's KV extent comes
// from `seqused_k` against a rectangular page table strided by
// `page_table_batch_stride`. `seqused_k` does NOT drop the K/V batch strides —
// only `cu_seqlens_k` does (flash_api.cpp:105-108) — which is what lets a paged
// batch share one launch.
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
    const int* cu_seqlens_q;    // i32 [batch+1] prefix sum over query rows
    const int* seqused_k;       // i32 [batch] per-row KV extent in tokens
    int batch;
    int total_q;                // cu_seqlens_q[batch]
    int seqlen_q;               // longest row's query length
    int seqlen_k;               // longest row's KV length
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
    // Paged KV (null = contiguous): k/v are the pool base and the strides
    // describe one page, matching the HND pool [page, h_k, page_size, d].
    const int* page_table;  // i32 [batch, page_table_batch_stride]
    long long page_table_batch_stride;
    int page_size;
    int num_pages;          // pages in the POOL (the 4th dim's extent)
    long long k_page_stride;  // elements between pages in the pool
    long long v_page_stride;
} ArleFa3FwdHd256Args;

cudaError_t arle_fa3_fwd_hd256_bf16_cuda(const ArleFa3FwdHd256Args* a,
                                         cudaStream_t stream) {
    if (a == nullptr || a->head_dim != 256 || a->seqlen_q <= 0 ||
        a->seqlen_k <= 0 || a->num_heads <= 0 || a->num_heads_k <= 0 ||
        a->num_heads % a->num_heads_k != 0 || a->batch <= 0 ||
        a->total_q <= 0) {
        return cudaErrorInvalidValue;
    }
    // Both null = one uniform row (the contiguous slot-cache lane); both set =
    // a ragged batch sharing one launch. Upstream's own nullability contract.
    const bool varlen = a->cu_seqlens_q != nullptr;
    if (varlen != (a->seqused_k != nullptr) || (!varlen && a->batch != 1)) {
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
    // Varlen: q/o are packed and addressed through cu_seqlens_q, so their batch
    // strides are never walked (flash_api.cpp:101-104 skips them). Otherwise
    // mirror the non-varlen fill so the TMA descriptors see self-consistent
    // extents; take the larger of the row/head walks so both token-major
    // ([S, h, d]) and head-major cache ([h_k, max_seq, d]) views work.
    auto batch_extent = [](int64_t rows, int64_t row_stride, int64_t heads,
                           int64_t head_stride) {
        int64_t by_row = rows * row_stride;
        int64_t by_head = heads * head_stride;
        return by_row > by_head ? by_row : by_head;
    };
    params.q_batch_stride =
        varlen ? 0
               : batch_extent(a->seqlen_q, a->q_row_stride, a->num_heads,
                              a->q_head_stride);
    // Paged: FA3's 4th K/V dim is the page dim (launch template :98-100).
    params.k_batch_stride =
        a->page_table != nullptr
            ? a->k_page_stride
            : batch_extent(a->seqlen_k, a->k_row_stride, a->num_heads_k, a->k_head_stride);
    params.v_batch_stride =
        a->page_table != nullptr
            ? a->v_page_stride
            : batch_extent(a->seqlen_k, a->v_row_stride, a->num_heads_k, a->v_head_stride);
    params.o_batch_stride =
        varlen ? 0
               : batch_extent(a->seqlen_q, a->o_row_stride, a->num_heads,
                              a->o_head_stride);

    params.softmax_lse_ptr = a->softmax_lse;
    params.cu_seqlens_q = const_cast<int*>(a->cu_seqlens_q);
    params.seqused_k = const_cast<int*>(a->seqused_k);

    params.b = a->batch;
    params.b_k = a->batch;
    params.h = a->num_heads;
    params.h_k = a->num_heads_k;
    params.seqlen_q = a->seqlen_q;
    params.seqlen_k = a->seqlen_k;
    params.total_q = a->total_q;
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

    // Non-TMA gather: qwen35's page_size (16) is below the hdim256 kBlockN and
    // the TMA paged path asserts page_size % kBlockN == 0.
    const bool paged = a->page_table != nullptr;
    if (paged && (a->page_size <= 0 || a->num_pages <= 0 ||
                  a->k_page_stride <= 0 || a->v_page_stride <= 0)) {
        return cudaErrorInvalidValue;
    }
    params.page_table = paged ? const_cast<int*>(a->page_table) : nullptr;
    params.page_table_batch_stride = paged ? a->page_table_batch_stride : 0;
    params.page_size = paged ? a->page_size : 1;
    params.num_pages = paged ? a->num_pages : 0;
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
        // Varlen split allocation (flash_api.cpp:1102-1112):
        // out_accum [splits, h, total_q, dv], lse_accum [splits, h, total_q].
        // No batch dim — cu_seqlens_q already flattens the rows.
        params.is_fp32 = false;
        params.oaccum_ptr = a->out_accum;
        params.softmax_lseaccum_ptr = a->softmax_lse_accum;
        params.oaccum_split_stride =
            static_cast<int64_t>(a->num_heads) * a->total_q * a->head_dim;
        params.oaccum_batch_stride = 0;
        params.oaccum_head_stride =
            static_cast<int64_t>(a->total_q) * a->head_dim;
        params.oaccum_row_stride = a->head_dim;
        params.lseaccum_split_stride =
            static_cast<int64_t>(a->num_heads) * a->total_q;
        params.lseaccum_batch_stride = 0;
        params.lseaccum_head_stride = a->total_q;

        if (paged) {
            run_mha_fwd_<90, cutlass::bfloat16_t, 256, 256, /*Split=*/true,
                         /*PagedKVNonTMA=*/true, /*Has_softcap=*/false,
                         /*PackGQA=*/true>(params, stream);
        } else {
            run_mha_fwd_<90, cutlass::bfloat16_t, 256, 256, /*Split=*/true,
                         /*PagedKVNonTMA=*/false, /*Has_softcap=*/false,
                         /*PackGQA=*/true>(params, stream);
        }
        cudaError_t st = cudaGetLastError();
        if (st != cudaSuccess) return st;
        params.is_bf16 = true;
        run_mha_fwd_combine_<cutlass::bfloat16_t, float, 128>(
            params, stream, true /*enable_pdl*/);
        return cudaGetLastError();
    }

    if (paged) {
        params.pack_gqa = true;  // the vendored paged units are PackGQA-only
        run_mha_fwd_<90, cutlass::bfloat16_t, 256, 256, /*Split=*/false,
                     /*PagedKVNonTMA=*/true, /*Has_softcap=*/false,
                     /*PackGQA=*/true>(params, stream);
    } else {
        run_mha_fwd_<90, cutlass::bfloat16_t, 256, 256, /*Split=*/false,
                     /*PagedKVNonTMA=*/false, /*Has_softcap=*/false,
                     /*PackGQA=*/false>(params, stream);
    }
    return cudaGetLastError();
}

// Stub-detection marker (build.rs validates the archive exports this and the
// Rust side can assert the real shim linked — flashmla pattern).
int arle_fa3_real_kernel_marker_cuda(void) { return 1; }

}  // extern "C"
