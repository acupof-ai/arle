// ARLE torch-free shim over vendored FA3 hopper fwd + bwd (hdim256, bf16,
// sm_90a).
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
    // Device i32 scratch for the scheduler metadata + semaphore. Size it
    // `round_up(batch, 4) * 4 + 1` and the shim's own check can never trip.
    int* tile_count_semaphore;
    int metadata_capacity;  // elements available at tile_count_semaphore
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

    params.varlen_sort_batches = !params.is_local;

    // Scheduler metadata (flash_api.cpp:995-1027). Varlen makes the launch
    // template run `prepare_varlen_num_blocks` for us
    // (flash_fwd_launch_template.h:163), but that kernel WRITES through
    // num_splits_dynamic / num_m_blocks / varlen_batch_idx / num_nheads_in_l2 —
    // leaving them null is an illegal access, not a fallback. One caller-owned
    // i32 buffer carves all of them plus the semaphore, exactly as upstream
    // slices its `tile_count_semaphore` tensor.
    const bool needs_semaphore =
        ((params.is_causal || params.is_local) && params.num_splits == 1) ||
        varlen;
    params.head_swizzle = params.is_causal || params.is_local;
    const int b_rounded = round_multiple(params.b, 4);  // 16B pointer alignment
    int num_prepare_vectors = varlen ? 2 : 0;
    if (params.varlen_sort_batches) num_prepare_vectors += 1;
    if (params.head_swizzle) num_prepare_vectors += 1;
    const int head_swizzle_offset =
        b_rounded * (params.varlen_sort_batches ? 3 : 2);
    const int sem_offset = b_rounded * num_prepare_vectors;
    const int metadata_size = int(needs_semaphore) + sem_offset;
    int* meta = a->tile_count_semaphore;
    if (metadata_size > 0 && meta == nullptr) return cudaErrorInvalidValue;
    if (a->metadata_capacity < metadata_size) return cudaErrorInvalidValue;
    // Varlen zeroes it inside prepare_varlen_num_blocks; otherwise do it here.
    if (needs_semaphore && !varlen) {
        cudaError_t st = cudaMemsetAsync(meta, 0, sizeof(int), stream);
        if (st != cudaSuccess) return st;
    }
    params.prepare_varlen_pdl = false;
    params.num_splits_dynamic_ptr = varlen ? meta : nullptr;
    params.num_m_blocks_ptr = varlen ? meta + b_rounded : nullptr;
    params.varlen_batch_idx_ptr =
        varlen && params.varlen_sort_batches ? meta + b_rounded * 2 : nullptr;
    params.num_nheads_in_l2_ptr =
        varlen && params.head_swizzle ? meta + head_swizzle_offset : nullptr;
    params.tile_count_semaphore = needs_semaphore ? meta + sem_offset : nullptr;
    params.tile_count_semaphore_offset = sem_offset;
    params.skip_scheduler_metadata_computation = false;

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

// Backward (hdim256/bf16/sm_90a). Param-fill mirrors mha_bwd
// (flash_api.cpp:1267-1568 — set_params_dgrad + the scratch-tensor block).
// Scope: NON-varlen, NON-paged, batch=1 contiguous [S, h, d] views; no
// dropout, no softcap, deterministic=false (upstream forbids deterministic
// for hdim 256). All scratch is CALLER-provided; the shim allocates nothing.
//
// Tile shape for hdim256 sm90 is kBlockM=64, kBlockN=80
// (flash_api.cpp:1373-1390 / run_mha_bwd_hdim256), so the rounded extents
// below use 64/80, NOT the fwd path's 128. With sq_r = round_up(seqlen_q, 64)
// and sk_r = round_up(seqlen_k, 80), required element counts are:
//   softmax_d, softmax_lse_log2:  fp32 h * sq_r
//   dq_accum:                     fp32 h * sq_r * 256
//   dk_accum, dv_accum:           fp32 h_k * sk_r * 256  (GQA only, h != h_k)
//   dq_semaphore:                 i32  ceil(seqlen_q / 64) * h
// dq_accum and dq_semaphore are zeroed by the bwd preprocess kernel
// (Clear_dQaccum=true); dk/dv_accum must start zeroed (upstream at::zeros) —
// the shim cudaMemsetAsync's them. dk/dv_semaphore are deterministic-only.
typedef struct {
    const void* q;        // bf16, seqlen_q x h x d view
    const void* k;        // bf16, seqlen_k x h_k x d view
    const void* v;
    const void* o;        // bf16, forward output, seqlen_q x h x d view
    const void* dout;     // bf16, seqlen_q x h x d view
    const float* softmax_lse;  // fp32 [h * seqlen_q] from the forward
    void* dq;             // bf16, seqlen_q x h x d view
    void* dk;             // bf16, seqlen_k x h_k x d view
    void* dv;
    float* softmax_d;          // fp32 [h * sq_r]
    float* softmax_lse_log2;   // fp32 [h * sq_r]
    float* dq_accum;           // fp32 [h * sq_r * 256]
    float* dk_accum;           // fp32 [h_k * sk_r * 256], GQA only else null
    float* dv_accum;           // fp32 [h_k * sk_r * 256], GQA only else null
    int* dq_semaphore;         // i32 [ceil(seqlen_q / 64) * h]
    long long softmax_d_capacity;  // elements available at each scratch ptr
    long long softmax_lse_log2_capacity;
    long long dq_accum_capacity;
    long long dk_accum_capacity;
    long long dv_accum_capacity;
    long long dq_semaphore_capacity;
    int seqlen_q;
    int seqlen_k;
    int num_heads;    // h
    int num_heads_k;  // h_k
    int head_dim;     // must be 256
    long long q_row_stride;
    long long k_row_stride;
    long long v_row_stride;
    long long o_row_stride;
    long long do_row_stride;
    long long dq_row_stride;
    long long dk_row_stride;
    long long dv_row_stride;
    long long q_head_stride;
    long long k_head_stride;
    long long v_head_stride;
    long long o_head_stride;
    long long do_head_stride;
    long long dq_head_stride;
    long long dk_head_stride;
    long long dv_head_stride;
    float softmax_scale;
    int is_causal;
} ArleFa3BwdHd256Args;

cudaError_t arle_fa3_bwd_hd256_bf16_cuda(const ArleFa3BwdHd256Args* a,
                                         cudaStream_t stream) {
    if (a == nullptr || a->head_dim != 256 || a->seqlen_q <= 0 ||
        a->seqlen_k <= 0 || a->num_heads <= 0 || a->num_heads_k <= 0 ||
        a->num_heads % a->num_heads_k != 0) {
        return cudaErrorInvalidValue;
    }
    if (a->q == nullptr || a->k == nullptr || a->v == nullptr ||
        a->o == nullptr || a->dout == nullptr || a->softmax_lse == nullptr ||
        a->dq == nullptr || a->dk == nullptr || a->dv == nullptr ||
        a->softmax_d == nullptr || a->softmax_lse_log2 == nullptr ||
        a->dq_accum == nullptr || a->dq_semaphore == nullptr) {
        return cudaErrorInvalidValue;
    }
    // hdim256 sm90 bwd tiles (flash_api.cpp:1373-1390).
    constexpr int kBlockM = 64;
    constexpr int kBlockN = 80;
    const int64_t sq_r = round_multiple(a->seqlen_q, kBlockM);
    const int64_t sk_r = round_multiple(a->seqlen_k, kBlockN);
    const int64_t h = a->num_heads;
    const int64_t h_k = a->num_heads_k;
    const bool gqa = a->num_heads != a->num_heads_k;
    if (a->softmax_d_capacity < h * sq_r ||
        a->softmax_lse_log2_capacity < h * sq_r ||
        a->dq_accum_capacity < h * sq_r * 256 ||
        a->dq_semaphore_capacity < ((a->seqlen_q + kBlockM - 1) / kBlockM) * h) {
        return cudaErrorInvalidValue;
    }
    if (gqa && (a->dk_accum == nullptr || a->dv_accum == nullptr ||
                a->dk_accum_capacity < h_k * sk_r * 256 ||
                a->dv_accum_capacity < h_k * sk_r * 256)) {
        return cudaErrorInvalidValue;
    }

    Flash_bwd_params params{};
    params.is_bf16 = true;

    params.q_ptr = const_cast<void*>(a->q);
    params.k_ptr = const_cast<void*>(a->k);
    params.v_ptr = const_cast<void*>(a->v);
    params.o_ptr = const_cast<void*>(a->o);
    params.do_ptr = const_cast<void*>(a->dout);
    params.dq_ptr = a->dq;
    params.dk_ptr = a->dk;
    params.dv_ptr = a->dv;
    params.q_row_stride = a->q_row_stride;
    params.k_row_stride = a->k_row_stride;
    params.v_row_stride = a->v_row_stride;
    params.o_row_stride = a->o_row_stride;
    params.do_row_stride = a->do_row_stride;
    params.dq_row_stride = a->dq_row_stride;
    params.dk_row_stride = a->dk_row_stride;
    params.dv_row_stride = a->dv_row_stride;
    params.q_head_stride = a->q_head_stride;
    params.k_head_stride = a->k_head_stride;
    params.v_head_stride = a->v_head_stride;
    params.o_head_stride = a->o_head_stride;
    params.do_head_stride = a->do_head_stride;
    params.dq_head_stride = a->dq_head_stride;
    params.dk_head_stride = a->dk_head_stride;
    params.dv_head_stride = a->dv_head_stride;
    params.v_dim_stride = 1;
    // batch=1: the batch strides are never walked, but fill them
    // self-consistently the way the fwd shim does.
    auto batch_extent = [](int64_t rows, int64_t row_stride, int64_t heads,
                           int64_t head_stride) {
        int64_t by_row = rows * row_stride;
        int64_t by_head = heads * head_stride;
        return by_row > by_head ? by_row : by_head;
    };
    params.q_batch_stride =
        batch_extent(a->seqlen_q, a->q_row_stride, h, a->q_head_stride);
    params.k_batch_stride =
        batch_extent(a->seqlen_k, a->k_row_stride, h_k, a->k_head_stride);
    params.v_batch_stride =
        batch_extent(a->seqlen_k, a->v_row_stride, h_k, a->v_head_stride);
    params.o_batch_stride =
        batch_extent(a->seqlen_q, a->o_row_stride, h, a->o_head_stride);
    params.do_batch_stride =
        batch_extent(a->seqlen_q, a->do_row_stride, h, a->do_head_stride);
    params.dq_batch_stride =
        batch_extent(a->seqlen_q, a->dq_row_stride, h, a->dq_head_stride);
    params.dk_batch_stride =
        batch_extent(a->seqlen_k, a->dk_row_stride, h_k, a->dk_head_stride);
    params.dv_batch_stride =
        batch_extent(a->seqlen_k, a->dv_row_stride, h_k, a->dv_head_stride);

    params.softmax_lse_ptr = const_cast<float*>(a->softmax_lse);
    params.dsoftmax_sum = a->softmax_d;
    params.softmax_lse_log2_ptr = a->softmax_lse_log2;
    params.dq_accum_ptr = a->dq_accum;
    params.dk_accum_ptr = gqa ? a->dk_accum : nullptr;
    params.dv_accum_ptr = gqa ? a->dv_accum : nullptr;
    params.dq_semaphore = a->dq_semaphore;

    params.b = 1;
    params.h = a->num_heads;
    params.h_k = a->num_heads_k;
    params.seqlen_q = a->seqlen_q;
    params.seqlen_k = a->seqlen_k;
    params.total_q = a->seqlen_q;
    params.total_k = a->seqlen_k;
    params.seqlen_q_rounded = int(sq_r);
    params.seqlen_k_rounded = int(sk_r);
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

    // Causal fill mirrors mha_bwd:1360-1365 + set_params_fprop:146-159. No
    // seqlen_q==1 causal demotion here — that is a fwd-only scheduling trick.
    bool is_causal = a->is_causal != 0;
    params.is_causal = is_causal;
    params.is_local = false;
    params.window_size_left = a->seqlen_k - 1;
    params.window_size_right = is_causal ? 0 : a->seqlen_q - 1;
    params.attention_chunk = 0;

    params.arch = 90;
    params.num_sm = device_num_sm();

    params.deterministic = false;

    // GQA epilogue accumulates dK/dV into the fp32 accum buffers, which must
    // start zeroed (upstream allocates them with at::zeros).
    if (gqa) {
        cudaError_t st = cudaMemsetAsync(a->dk_accum, 0,
                                         size_t(h_k * sk_r * 256) * sizeof(float), stream);
        if (st != cudaSuccess) return st;
        st = cudaMemsetAsync(a->dv_accum, 0,
                             size_t(h_k * sk_r * 256) * sizeof(float), stream);
        if (st != cudaSuccess) return st;
    }

    run_mha_bwd_<90, cutlass::bfloat16_t, 256, /*Has_softcap=*/false>(params,
                                                                      stream);
    return cudaGetLastError();
}

// Stub-detection marker (build.rs validates the archive exports this and the
// Rust side can assert the real shim linked — flashmla pattern).
int arle_fa3_real_kernel_marker_cuda(void) { return 1; }

}  // extern "C"
