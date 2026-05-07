"""TileLang batch decode HD128 paged attention.

TileLang HD128 paged decode path used by Qwen3 full-attention layers
(Qwen3-1.5B / 4B / 8B / 14B+ — all share head_dim=128 with kv_heads=8).
Decode = single-token Q per request (qo_len == 1), so the kernel reads
paged K/V, runs one row of Q against the full kv_total_len for that
request, and writes one output row.

Sister to ``batch_decode_paged_hd256.py``. Deltas vs the HD256 template:

  * ``HEAD_DIM = 128`` (vs 256).
  * ``SM_SCALE = 1.0 / sqrt(128)``.
  * ``SUPPORTED_HEADS`` covers the Qwen3 family rather than Qwen3.5.

Everything else (BLOCK_M=64, BLOCK_N=16=PAGE_SIZE, NUM_STAGES=2,
NUM_THREADS=128, no causal mask, padded BLOCK_M layout) is identical to
the HD256 decode template — TileLang's tile/pipeline tunables are
HEAD_DIM-independent at this scale.

Symbolic runtime int32 args (``batch_size``, ``total_q_tokens``,
``max_qlen``, ``num_pages``, ``total_pages``) are kept identical to the
HD256 decode kernel so ``gen_tilelang_aot.py::WRAPPER_FILL_RULES`` works
without modification.

Supported (num_q_heads, num_kv_heads) configurations (Qwen3 full-attn
families, mirrors ``TILELANG_PREFILL_HD128_HEAD_CONFIGS``):
  (16, 8)  — Qwen3-1.5B
  (32, 8)  — Qwen3-4B
  (40, 8)  — Qwen3-8B
  (64, 8)  — Qwen3-14B / 32B-class

Tile / pipeline tunables (mirror HD256 decode):

  BLOCK_M = 64
  BLOCK_N = 16        (= PAGE_SIZE)
  NUM_STAGES = 2
  NUM_THREADS = 128   (4 warps)

Shared-memory budget (HD128 is half of HD256, so the budget is well
below every SM target's cap):

  Q tile : BLOCK_M * HEAD_DIM * 2 B   = 64 * 128 * 2 =  16_384 B
  K tile : BLOCK_N * HEAD_DIM * 2 B   = 16 * 128 * 2 =   4_096 B
  V tile : BLOCK_N * HEAD_DIM * 2 B   = 16 * 128 * 2 =   4_096 B
  Single-buffered total                              =  24_576 B (~24 KB)
  With NUM_STAGES=2 double-buffering K/V:
    Q(16 KB) + 2 * (K 4 KB + V 4 KB) = 32 KB.
  Fits H100 (228 KB), L4 / sm_89 (99 KB), and every other SM target the
  generator builds for.
"""

import math

import tilelang
import tilelang.language as T

HEAD_DIM = 128
PAGE_SIZE = 16
BLOCK_M = 64
# Decode keeps the prefill-compatible BLOCK_M=64 fragment layout, then lowers
# BLOCK_N to one page so dynamic shared memory remains SM-cap-safe.
BLOCK_N = 16
NUM_STAGES = 2
NUM_THREADS = 128
MAX_SPLITS = 16

# (num_q_heads, num_kv_heads) configurations. Mirrors the Qwen3 HD128
# full-attn family. Extend here + the build.rs list + the matching
# FFI/Rust dispatch arms in lockstep when adding a new size.
SUPPORTED_HEADS = (
    (16, 8),  # Qwen3-1.5B
    (32, 8),  # Qwen3-4B
    (40, 8),  # Qwen3-8B
    (64, 8),  # Qwen3-14B / 32B-class
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
        # TileLang 0.1.9 cannot use T.symbolic in grid extents — symbols
        # there must come from a tensor shape or a kernel scalar arg.
        # Keep the same scalar-arg shape as the HD256 decode kernel even
        # though `max_qlen` is always 1 for decode; this lets
        # gen_tilelang_aot.py's WRAPPER_FILL_RULES fill them without any
        # changes.
        batch_size: T.int32,
        max_qlen: T.int32,
    ):
        # Grid: (1, num_q_heads, batch_size). One Q row per (request,
        # head). Outer-x extent is fixed at 1 because qlen == 1 always
        # for decode (no varlen Q-tile sweep).
        with T.Kernel(
            1,
            num_q_heads,
            batch_size,
            threads=NUM_THREADS,
        ) as (bx, by, bz):
            q_tile = T.alloc_shared((BLOCK_M, HEAD_DIM), dtype)
            k_tile = T.alloc_shared((BLOCK_N, HEAD_DIM), dtype)
            v_tile = T.alloc_shared((BLOCK_N, HEAD_DIM), dtype)
            acc_o = T.alloc_fragment((BLOCK_M, HEAD_DIM), accum_dtype)
            scores = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
            m_i = T.alloc_fragment((BLOCK_M,), accum_dtype)
            l_i = T.alloc_fragment((BLOCK_M,), accum_dtype)

            T.use_swizzle(panel_size=8)

            # Decode: bz indexes the request; bx == 0 always (grid x == 1).
            # Single Q row per request, no q_start/qlen arithmetic.
            kv_page_start = KV_indptr[bz]
            kv_page_end = KV_indptr[bz + 1]
            num_kv_pages = kv_page_end - kv_page_start
            last_page_len = KV_last_page_len[bz]
            kv_total_len = (num_kv_pages - 1) * PAGE_SIZE + last_page_len

            kv_head = by // gqa_group

            T.fill(acc_o, 0)
            T.fill(m_i, -T.infinity(accum_dtype))
            T.fill(l_i, 0)

            # Load the single real Q row for (request bz, head by) into
            # q_tile[0, :]. Rows 1..63 are padding to satisfy TileLang's
            # tensor-core M-divisibility constraint; they are masked out below
            # and never written to Output.
            for i, d in T.Parallel(BLOCK_M, HEAD_DIM):
                q_tile[i, d] = T.if_then_else(
                    i == 0,
                    Q[bz, by * HEAD_DIM + d],
                    T.cast(0, dtype),
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
                        K_pool[page_idx, kv_head, in_page, d],
                        T.cast(0, dtype),
                    )
                    v_tile[j, d] = T.if_then_else(
                        abs_col < kv_total_len,
                        V_pool[page_idx, kv_head, in_page, d],
                        T.cast(0, dtype),
                    )

                T.clear(scores)
                T.gemm(q_tile, k_tile, scores, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                # No causal mask: qlen == 1 means the single Q row
                # legally attends to every KV position in
                # [0, kv_total_len). Keep only the bounds clause.
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
                # Hoist the per-row alpha into its own fragment then drive
                # the acc_o rescale as a 2D T.Parallel — same layout pattern
                # as the HD256 decode kernel.
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
                # Narrow the f32 softmax output to bf16 to match v_tile
                # before the P @ V matmul. TileLang 0.1.9's gemm asserts
                # A.dtype == B.dtype.
                p_bf16 = T.alloc_fragment((BLOCK_M, BLOCK_N), dtype)
                T.copy(p, p_bf16)
                T.gemm(p_bf16, v_tile, acc_o, policy=T.GemmWarpPolicy.FullRow)

            # Single output row per (request, head). bz indexes the request
            # directly; padded rows are intentionally dropped.
            for i, d in T.Parallel(BLOCK_M, HEAD_DIM):
                if i == 0:
                    Output[bz, by * HEAD_DIM + d] = T.cast(
                        acc_o[i, d] / l_i[i], dtype
                    )

    # Pin the kernel name so the generated symbol matches what the AOT
    # build.rs / FFI side will look up: batch_decode_paged_hd128_q{Q}_kv{K}_run.
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
            q_tile = T.alloc_shared((BLOCK_M, HEAD_DIM), dtype)
            k_tile = T.alloc_shared((BLOCK_N, HEAD_DIM), dtype)
            v_tile = T.alloc_shared((BLOCK_N, HEAD_DIM), dtype)
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
                    Q[bx, by * HEAD_DIM + d],
                    T.cast(0, dtype),
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
                        K_pool[page_idx, kv_head, in_page, d],
                        T.cast(0, dtype),
                    )
                    v_tile[j, d] = T.if_then_else(
                        abs_col < kv_chunk_end,
                        V_pool[page_idx, kv_head, in_page, d],
                        T.cast(0, dtype),
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
                p_bf16 = T.alloc_fragment((BLOCK_M, BLOCK_N), dtype)
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
