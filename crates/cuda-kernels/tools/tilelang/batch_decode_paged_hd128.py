"""TileLang batch decode HD128 paged attention (BF16, kv_heads=8 GQA family).

Decode = one Q row per request (qo_len == 1): read paged K/V, run that row
against the full kv_total_len, write one output row. Sister to the HD256 decode
template (deltas: HEAD_DIM=128, SM_SCALE=1/sqrt(128), the HD128 SUPPORTED_HEADS).
Tile tunables and the int32 scalar args mirror HD256 (so gen_tilelang_aot.py's
WRAPPER_FILL_RULES works unchanged).

sm_70 (Volta) variant: identical strategy to the prefill kernels. I/O stays
bf16 (runtime ABI); on sm<80 the GEMM operands are fed fp16 so the gemms route
to TileLang's stock Volta tensor-core path (GemmMMASm70 / mma.sync) instead of
the scalar cuda.fma fallback whose per-output fragment layout triggers the
LayoutInference m_prev/scale_i conflict. The PV operand A is fed from shared on
sm<80. sm>=80 keeps the byte-identical bf16 fragment path. The split_merge
kernel has no GEMM (pure f32 partial merge) and needs no gate.

Tile tunables: BLOCK_M=64, BLOCK_N=16 (=PAGE_SIZE), NUM_STAGES=2, NUM_THREADS=128,
no causal mask. Shared-mem ~32 KB double-buffered (Q 16 KB + 2*(K 4 KB + V 4 KB)),
well under every SM target's cap.
"""

import math
import os

import tilelang
import tilelang.language as T

HEAD_DIM = 128
PAGE_SIZE = 16
BLOCK_M = 64
# BLOCK_N = one page keeps dynamic shared memory SM-cap-safe.
BLOCK_N = 16
NUM_STAGES = 2
NUM_THREADS = 128
MAX_SPLITS = 16

# (num_q_heads, num_kv_heads). Extend here + build.rs + the FFI/Rust dispatch
# arms in lockstep.
SUPPORTED_HEADS = (
    (16, 8),
    (32, 8),
    (40, 8),
    (64, 8),
)


def _make_kernel(num_q_heads: int, num_kv_heads: int):
    assert num_q_heads % num_kv_heads == 0, (
        f"num_q_heads ({num_q_heads}) must be a multiple of num_kv_heads ({num_kv_heads})"
    )
    gqa_group = num_q_heads // num_kv_heads
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    log2e = 1.4426950408889634

    dtype = "bfloat16"
    accum_dtype = "float32"
    index_dtype = "int32"

    # sm<80 feeds the GEMMs fp16 so dispatch picks kCudaMMA (GemmMMASm70), not
    # the scalar kCudaFMA fallback; accum stays f32. Trace-time no-op on sm>=80
    # → the bf16 AST stays byte-identical to the repo kernel. The bf16->fp16
    # cast routes through f32 (direct bf16->fp16 is an ambiguous CUTLASS
    # conversion nvcc rejects). See batch_prefill_paged_hd128.py for details.
    sm_arch = int(os.environ.get("ARLE_TILELANG_CUDA_ARCH", "90"))
    gemm_dtype = "float16" if sm_arch < 80 else "bfloat16"

    def to_gemm(x):
        if gemm_dtype == dtype:
            return x
        return T.cast(T.cast(x, accum_dtype), gemm_dtype)

    @T.prim_func
    def kernel(
        # Q layout for decode: one row per request, no Q_indptr needed.
        Q: T.Tensor((T.symbolic("total_q_tokens"), num_q_heads * HEAD_DIM), dtype),
        K_pool: T.Tensor((T.symbolic("num_pages"), num_kv_heads, PAGE_SIZE, HEAD_DIM), dtype),
        V_pool: T.Tensor((T.symbolic("num_pages"), num_kv_heads, PAGE_SIZE, HEAD_DIM), dtype),
        KV_indptr: T.Tensor((T.symbolic("batch_size_plus_one"),), index_dtype),
        KV_indices: T.Tensor((T.symbolic("total_pages"),), index_dtype),
        KV_last_page_len: T.Tensor((T.symbolic("batch_size"),), index_dtype),
        Output: T.Tensor((T.symbolic("total_q_tokens"), num_q_heads * HEAD_DIM), dtype),
        # TileLang 0.1.9 can't use T.symbolic in grid extents; keep the HD256
        # scalar-arg shape (max_qlen unused for decode) so WRAPPER_FILL_RULES
        # fills them unchanged.
        batch_size: T.int32,
        max_qlen: T.int32,
    ):
        # Grid: (1, num_q_heads, batch_size). One Q row per (request, head);
        # outer-x is 1 (qlen == 1 for decode).
        with T.Kernel(
            1,
            num_q_heads,
            batch_size,
            threads=NUM_THREADS,
        ) as (bx, by, bz):
            q_tile = T.alloc_shared((BLOCK_M, HEAD_DIM), gemm_dtype)
            k_tile = T.alloc_shared((BLOCK_N, HEAD_DIM), gemm_dtype)
            v_tile = T.alloc_shared((BLOCK_N, HEAD_DIM), gemm_dtype)
            acc_o = T.alloc_fragment((BLOCK_M, HEAD_DIM), accum_dtype)
            scores = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
            m_i = T.alloc_fragment((BLOCK_M,), accum_dtype)
            l_i = T.alloc_fragment((BLOCK_M,), accum_dtype)

            T.use_swizzle(panel_size=8)

            # bz indexes the request; bx == 0 always. Single Q row, no qlen math.
            kv_page_start = KV_indptr[bz]
            kv_page_end = KV_indptr[bz + 1]
            num_kv_pages = kv_page_end - kv_page_start
            last_page_len = KV_last_page_len[bz]
            kv_total_len = (num_kv_pages - 1) * PAGE_SIZE + last_page_len

            kv_head = by // gqa_group

            T.fill(acc_o, 0)
            T.fill(m_i, -T.infinity(accum_dtype))
            T.fill(l_i, 0)

            # Real Q row → q_tile[0, :]; rows 1..63 are M-divisibility padding,
            # masked out below and never written to Output.
            for i, d in T.Parallel(BLOCK_M, HEAD_DIM):
                q_tile[i, d] = T.if_then_else(
                    i == 0,
                    to_gemm(Q[bz, by * HEAD_DIM + d]),
                    T.cast(0, gemm_dtype),
                )

            for kn in T.Pipelined(T.ceildiv(kv_total_len, BLOCK_N), num_stages=NUM_STAGES):
                col0 = kn * BLOCK_N
                for j, d in T.Parallel(BLOCK_N, HEAD_DIM):
                    abs_col = col0 + j
                    page_local = abs_col // PAGE_SIZE
                    in_page = abs_col % PAGE_SIZE
                    page_idx = T.if_then_else(
                        abs_col < kv_total_len,
                        KV_indices[kv_page_start + page_local],
                        0,
                    )
                    k_tile[j, d] = T.if_then_else(
                        abs_col < kv_total_len,
                        to_gemm(K_pool[page_idx, kv_head, in_page, d]),
                        T.cast(0, gemm_dtype),
                    )
                    v_tile[j, d] = T.if_then_else(
                        abs_col < kv_total_len,
                        to_gemm(V_pool[page_idx, kv_head, in_page, d]),
                        T.cast(0, gemm_dtype),
                    )

                T.clear(scores)
                T.gemm(q_tile, k_tile, scores, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                # No causal mask (qlen == 1 attends to all of [0, kv_total_len)).
                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    col = col0 + j
                    in_bounds = (i == 0) and (col < kv_total_len)
                    scores[i, j] = T.if_then_else(
                        in_bounds,
                        scores[i, j] * sm_scale,
                        -T.infinity(accum_dtype),
                    )

                m_prev = T.alloc_fragment((BLOCK_M,), accum_dtype)
                m_new = T.alloc_fragment((BLOCK_M,), accum_dtype)
                p = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
                T.copy(m_i, m_prev)
                T.reduce_max(scores, m_new, dim=1, clear=True)
                for i in T.Parallel(BLOCK_M):
                    m_new[i] = T.max(m_prev[i], m_new[i])
                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    col = col0 + j
                    p[i, j] = T.if_then_else(
                        (i == 0) and (col < kv_total_len),
                        T.exp2((scores[i, j] - m_new[i]) * log2e),
                        T.cast(0, accum_dtype),
                    )
                # Hoist per-row alpha + rescale acc_o as a 2D T.Parallel.
                scale_i = T.alloc_fragment((BLOCK_M,), accum_dtype)
                for i in T.Parallel(BLOCK_M):
                    scale_i[i] = T.exp2((m_prev[i] - m_new[i]) * log2e)
                    l_i[i] = l_i[i] * scale_i[i]
                for i, d in T.Parallel(BLOCK_M, HEAD_DIM):
                    acc_o[i, d] = acc_o[i, d] * scale_i[i]
                row_sum = T.alloc_fragment((BLOCK_M,), accum_dtype)
                T.reduce_sum(p, row_sum, dim=1)
                for i in T.Parallel(BLOCK_M):
                    l_i[i] = l_i[i] + row_sum[i]
                    m_i[i] = m_new[i]
                # PV operand A. sm_80+: bf16 fragment (byte-identical to repo).
                # sm<80 Volta MMA: feed A from shared (mirrors the QK gemm).
                if gemm_dtype != dtype:
                    p_shared = T.alloc_shared((BLOCK_M, BLOCK_N), gemm_dtype)
                    T.copy(p, p_shared)
                    T.gemm(p_shared, v_tile, acc_o, policy=T.GemmWarpPolicy.FullRow)
                else:
                    p_bf16 = T.alloc_fragment((BLOCK_M, BLOCK_N), gemm_dtype)
                    T.copy(p, p_bf16)
                    T.gemm(p_bf16, v_tile, acc_o, policy=T.GemmWarpPolicy.FullRow)

            # One output row per (request, head); padded rows dropped.
            for i, d in T.Parallel(BLOCK_M, HEAD_DIM):
                if i == 0:
                    Output[bz, by * HEAD_DIM + d] = T.cast(
                        acc_o[i, d] / l_i[i], dtype
                    )

    # Pin the symbol name the AOT build.rs / FFI side looks up.
    kernel.__name__ = f"batch_decode_paged_hd128_q{num_q_heads}_kv{num_kv_heads}_run"
    return kernel


def _make_split_partial_kernel(num_q_heads: int, num_kv_heads: int):
    assert num_q_heads % num_kv_heads == 0, (
        f"num_q_heads ({num_q_heads}) must be a multiple of num_kv_heads ({num_kv_heads})"
    )
    gqa_group = num_q_heads // num_kv_heads
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    log2e = 1.4426950408889634

    dtype = "bfloat16"
    accum_dtype = "float32"
    index_dtype = "int32"

    # See _make_kernel / batch_prefill_paged_hd128.py for the sm_70 rationale.
    sm_arch = int(os.environ.get("ARLE_TILELANG_CUDA_ARCH", "90"))
    gemm_dtype = "float16" if sm_arch < 80 else "bfloat16"

    def to_gemm(x):
        if gemm_dtype == dtype:
            return x
        return T.cast(T.cast(x, accum_dtype), gemm_dtype)

    @T.prim_func
    def kernel(
        Q: T.Tensor((T.symbolic("total_q_tokens"), num_q_heads * HEAD_DIM), dtype),
        K_pool: T.Tensor((T.symbolic("num_pages"), num_kv_heads, PAGE_SIZE, HEAD_DIM), dtype),
        V_pool: T.Tensor((T.symbolic("num_pages"), num_kv_heads, PAGE_SIZE, HEAD_DIM), dtype),
        KV_indptr: T.Tensor((T.symbolic("batch_size_plus_one"),), index_dtype),
        KV_indices: T.Tensor((T.symbolic("total_pages"),), index_dtype),
        KV_last_page_len: T.Tensor((T.symbolic("batch_size"),), index_dtype),
        Partial_out: T.Tensor(
            (T.symbolic("num_splits"), T.symbolic("total_q_tokens"), num_q_heads, HEAD_DIM),
            accum_dtype,
        ),
        Partial_m: T.Tensor(
            (T.symbolic("num_splits"), T.symbolic("total_q_tokens"), num_q_heads),
            accum_dtype,
        ),
        Partial_l: T.Tensor(
            (T.symbolic("num_splits"), T.symbolic("total_q_tokens"), num_q_heads),
            accum_dtype,
        ),
        batch_size: T.int32,
        max_qlen: T.int32,
        num_splits: T.int32,
    ):
        # Grid: (batch_size, num_q_heads, num_splits). Each CTA handles one
        # FlashDecoding split for one decode row/head pair.
        with T.Kernel(
            batch_size,
            num_q_heads,
            num_splits,
            threads=NUM_THREADS,
        ) as (bx, by, bz):
            q_tile = T.alloc_shared((BLOCK_M, HEAD_DIM), gemm_dtype)
            k_tile = T.alloc_shared((BLOCK_N, HEAD_DIM), gemm_dtype)
            v_tile = T.alloc_shared((BLOCK_N, HEAD_DIM), gemm_dtype)
            acc_o = T.alloc_fragment((BLOCK_M, HEAD_DIM), accum_dtype)
            scores = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
            m_i = T.alloc_fragment((BLOCK_M,), accum_dtype)
            l_i = T.alloc_fragment((BLOCK_M,), accum_dtype)

            T.use_swizzle(panel_size=8)

            kv_page_start = KV_indptr[bx]
            kv_page_end = KV_indptr[bx + 1]
            num_kv_pages = kv_page_end - kv_page_start
            last_page_len = KV_last_page_len[bx]
            kv_total_len = (num_kv_pages - 1) * PAGE_SIZE + last_page_len

            kv_chunk_size = T.ceildiv(kv_total_len, num_splits)
            kv_chunk_start = bz * kv_chunk_size
            kv_chunk_end = T.min(kv_chunk_start + kv_chunk_size, kv_total_len)
            kv_chunk_len = T.max(kv_chunk_end - kv_chunk_start, 0)

            kv_head = by // gqa_group

            T.fill(acc_o, 0)
            T.fill(m_i, -T.infinity(accum_dtype))
            T.fill(l_i, 0)

            for i, d in T.Parallel(BLOCK_M, HEAD_DIM):
                q_tile[i, d] = T.if_then_else(
                    i == 0,
                    to_gemm(Q[bx, by * HEAD_DIM + d]),
                    T.cast(0, gemm_dtype),
                )

            for kn in T.Pipelined(T.ceildiv(kv_chunk_len, BLOCK_N), num_stages=NUM_STAGES):
                col0 = kv_chunk_start + kn * BLOCK_N
                for j, d in T.Parallel(BLOCK_N, HEAD_DIM):
                    abs_col = col0 + j
                    page_local = abs_col // PAGE_SIZE
                    in_page = abs_col % PAGE_SIZE
                    page_idx = T.if_then_else(
                        abs_col < kv_chunk_end,
                        KV_indices[kv_page_start + page_local],
                        0,
                    )
                    k_tile[j, d] = T.if_then_else(
                        abs_col < kv_chunk_end,
                        to_gemm(K_pool[page_idx, kv_head, in_page, d]),
                        T.cast(0, gemm_dtype),
                    )
                    v_tile[j, d] = T.if_then_else(
                        abs_col < kv_chunk_end,
                        to_gemm(V_pool[page_idx, kv_head, in_page, d]),
                        T.cast(0, gemm_dtype),
                    )

                T.clear(scores)
                T.gemm(q_tile, k_tile, scores, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    col = col0 + j
                    in_bounds = (i == 0) and (col < kv_chunk_end)
                    scores[i, j] = T.if_then_else(
                        in_bounds,
                        scores[i, j] * sm_scale,
                        -T.infinity(accum_dtype),
                    )

                m_prev = T.alloc_fragment((BLOCK_M,), accum_dtype)
                m_new = T.alloc_fragment((BLOCK_M,), accum_dtype)
                p = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
                T.copy(m_i, m_prev)
                T.reduce_max(scores, m_new, dim=1, clear=True)
                for i in T.Parallel(BLOCK_M):
                    m_new[i] = T.max(m_prev[i], m_new[i])
                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    col = col0 + j
                    p[i, j] = T.if_then_else(
                        (i == 0) and (col < kv_chunk_end),
                        T.exp2((scores[i, j] - m_new[i]) * log2e),
                        T.cast(0, accum_dtype),
                    )
                scale_i = T.alloc_fragment((BLOCK_M,), accum_dtype)
                for i in T.Parallel(BLOCK_M):
                    scale_i[i] = T.exp2((m_prev[i] - m_new[i]) * log2e)
                    l_i[i] = l_i[i] * scale_i[i]
                for i, d in T.Parallel(BLOCK_M, HEAD_DIM):
                    acc_o[i, d] = acc_o[i, d] * scale_i[i]
                row_sum = T.alloc_fragment((BLOCK_M,), accum_dtype)
                T.reduce_sum(p, row_sum, dim=1)
                for i in T.Parallel(BLOCK_M):
                    l_i[i] = l_i[i] + row_sum[i]
                    m_i[i] = m_new[i]
                # PV operand A. sm_80+: bf16 fragment (byte-identical to repo).
                # sm<80 Volta MMA: feed A from shared (mirrors the QK gemm).
                if gemm_dtype != dtype:
                    p_shared = T.alloc_shared((BLOCK_M, BLOCK_N), gemm_dtype)
                    T.copy(p, p_shared)
                    T.gemm(p_shared, v_tile, acc_o, policy=T.GemmWarpPolicy.FullRow)
                else:
                    p_bf16 = T.alloc_fragment((BLOCK_M, BLOCK_N), gemm_dtype)
                    T.copy(p, p_bf16)
                    T.gemm(p_bf16, v_tile, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i, d in T.Parallel(BLOCK_M, HEAD_DIM):
                if i == 0:
                    Partial_out[bz, bx, by, d] = T.if_then_else(
                        l_i[i] > 0,
                        acc_o[i, d] / l_i[i],
                        T.cast(0, accum_dtype),
                    )
            for i in T.Parallel(BLOCK_M):
                if i == 0:
                    Partial_m[bz, bx, by] = m_i[i]
                    Partial_l[bz, bx, by] = l_i[i]

    kernel.__name__ = (
        f"batch_decode_paged_hd128_split_partial_q{num_q_heads}_kv{num_kv_heads}_run"
    )
    return kernel


def _make_split_merge_kernel(num_q_heads: int, num_kv_heads: int):
    assert num_q_heads % num_kv_heads == 0, (
        f"num_q_heads ({num_q_heads}) must be a multiple of num_kv_heads ({num_kv_heads})"
    )

    dtype = "bfloat16"
    accum_dtype = "float32"

    @T.prim_func
    def kernel(
        Partial_out: T.Tensor(
            (T.symbolic("num_splits"), T.symbolic("total_q_tokens"), num_q_heads, HEAD_DIM),
            accum_dtype,
        ),
        Partial_m: T.Tensor(
            (T.symbolic("num_splits"), T.symbolic("total_q_tokens"), num_q_heads),
            accum_dtype,
        ),
        Partial_l: T.Tensor(
            (T.symbolic("num_splits"), T.symbolic("total_q_tokens"), num_q_heads),
            accum_dtype,
        ),
        Output: T.Tensor((T.symbolic("total_q_tokens"), num_q_heads * HEAD_DIM), dtype),
        total_q_tokens: T.int32,
        num_splits: T.int32,
    ):
        # Grid: (total_q_tokens, num_q_heads, 1). Each CTA merges all split
        # partials for one output row/head.
        with T.Kernel(
            total_q_tokens,
            num_q_heads,
            1,
            threads=NUM_THREADS,
        ) as (bx, by, bz):
            final_o = T.alloc_fragment((HEAD_DIM,), accum_dtype)
            final_m = T.alloc_fragment((1,), accum_dtype)
            final_l = T.alloc_fragment((1,), accum_dtype)

            for d in T.Parallel(HEAD_DIM):
                final_o[d] = T.cast(0, accum_dtype)
            final_m[0] = -T.infinity(accum_dtype)
            final_l[0] = T.cast(0, accum_dtype)

            for s in T.serial(num_splits):
                m_s = Partial_m[s, bx, by]
                l_s = Partial_l[s, bx, by]
                m_new = T.max(final_m[0], m_s)
                s_prev = final_l[0] * T.exp(final_m[0] - m_new)
                s_cur = l_s * T.exp(m_s - m_new)
                l_new = s_prev + s_cur

                for d in T.Parallel(HEAD_DIM):
                    o_s = Partial_out[s, bx, by, d]
                    final_o[d] = T.if_then_else(
                        l_new > 0,
                        (final_o[d] * s_prev + o_s * s_cur) / l_new,
                        T.cast(0, accum_dtype),
                    )
                final_m[0] = m_new
                final_l[0] = l_new

            for d in T.Parallel(HEAD_DIM):
                Output[bx, by * HEAD_DIM + d] = T.cast(final_o[d], dtype)

    kernel.__name__ = f"batch_decode_paged_hd128_split_merge_q{num_q_heads}_kv{num_kv_heads}_run"
    return kernel


def get_kernel(num_q_heads: int, num_kv_heads: int, kernel_key: str | None = None):
    """Entry point for gen_tilelang_aot.py. One specialization per call."""
    if kernel_key is None or kernel_key == "default":
        return _make_kernel(num_q_heads, num_kv_heads)
    if kernel_key == "split_partial":
        return _make_split_partial_kernel(num_q_heads, num_kv_heads)
    if kernel_key == "split_merge":
        return _make_split_merge_kernel(num_q_heads, num_kv_heads)
    raise ValueError(f"unknown HD128 decode kernel_key: {kernel_key!r}")
