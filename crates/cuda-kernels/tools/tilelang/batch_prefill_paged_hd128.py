"""TileLang batch prefill HD128 paged attention (BF16, causal, page_size=16).

One kernel is AOT-specialized per (num_q_heads, num_kv_heads) in SUPPORTED_HEADS
(compile-time constants → per-shape codegen). Add a head config by extending the
lockstep lists here + build.rs + ffi/attention.rs + infer/src/ops/attention.rs.

Tile tunables (Hopper defaults, docs/plans/tilelang-integration.md §6):
BLOCK_M=64 q-tile rows, BLOCK_N=64 kv-tile cols, NUM_STAGES=2, NUM_THREADS=128.
"""

import math

import tilelang
import tilelang.language as T

HEAD_DIM = 128
PAGE_SIZE = 16
BLOCK_M = 64
BLOCK_N = 64
NUM_STAGES = 2
NUM_THREADS = 128

# (num_q_heads, num_kv_heads) the build emits. Extend here + build.rs + the
# FFI/Rust dispatch arms in lockstep.
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

    @T.prim_func
    def kernel(
        Q: T.Tensor((T.symbolic("total_q_tokens"), num_q_heads * HEAD_DIM), dtype),
        Q_indptr: T.Tensor((T.symbolic("batch_size_plus_one"),), index_dtype),
        K_pool: T.Tensor((T.symbolic("num_pages"), num_kv_heads, PAGE_SIZE, HEAD_DIM), dtype),
        V_pool: T.Tensor((T.symbolic("num_pages"), num_kv_heads, PAGE_SIZE, HEAD_DIM), dtype),
        KV_indptr: T.Tensor((T.symbolic("batch_size_plus_one"),), index_dtype),
        KV_indices: T.Tensor((T.symbolic("total_pages"),), index_dtype),
        KV_last_page_len: T.Tensor((T.symbolic("batch_size"),), index_dtype),
        Output: T.Tensor((T.symbolic("total_q_tokens"), num_q_heads * HEAD_DIM), dtype),
        # TileLang 0.1.9 can't use T.symbolic in grid extents (must come from a
        # tensor shape or scalar arg), so batch/max_qlen are int32 scalars.
        batch_size: T.int32,
        max_qlen: T.int32,
    ):
        # Grid: (q_tile_blocks, num_q_heads, batch_size). Each block = BLOCK_M q
        # rows of one request/head; KV pages walked via KV_indices.
        with T.Kernel(
            T.ceildiv(max_qlen, BLOCK_M),
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

            q_start = Q_indptr[bz]
            q_end = Q_indptr[bz + 1]
            qlen = q_end - q_start
            kv_page_start = KV_indptr[bz]
            kv_page_end = KV_indptr[bz + 1]
            num_kv_pages = kv_page_end - kv_page_start
            last_page_len = KV_last_page_len[bz]
            kv_total_len = (num_kv_pages - 1) * PAGE_SIZE + last_page_len
            kv_offset = kv_total_len - qlen  # hoisted: also feeds the causal bound

            row0 = bx * BLOCK_M
            kv_head = by // gqa_group

            # Causal-bound KV loop: a Q-tile attends at most to KV col
            # `kv_offset + min(row0+BLOCK_M, qlen)`, clamped to kv_total_len. Drops
            # ~35-50% of per-tile work vs an unbounded loop on cold 4096-in prefill.
            q_rows_in_tile = T.if_then_else(
                row0 + BLOCK_M < qlen, row0 + BLOCK_M, qlen
            )
            kv_diag_limit = kv_offset + q_rows_in_tile
            kv_visible_end = T.if_then_else(
                kv_diag_limit < kv_total_len, kv_diag_limit, kv_total_len
            )

            T.fill(acc_o, 0)
            T.fill(m_i, -T.infinity(accum_dtype))
            T.fill(l_i, 0)

            for i, d in T.Parallel(BLOCK_M, HEAD_DIM):
                row = row0 + i
                src = q_start + row
                q_tile[i, d] = T.if_then_else(
                    row < qlen,
                    Q[src, by * HEAD_DIM + d],
                    T.cast(0, dtype),
                )

            # Per-tile page-index precomputes (depend on j only, not d): hoisting
            # to a 1D fragment kills the ~128x duplicate divmod + KV_indices gather
            # a (j, d) loop would incur.
            page_idx_j = T.alloc_fragment((BLOCK_N,), index_dtype)
            in_page_j = T.alloc_fragment((BLOCK_N,), index_dtype)
            valid_j = T.alloc_fragment((BLOCK_N,), index_dtype)

            for kn in T.Pipelined(T.ceildiv(kv_visible_end, BLOCK_N), num_stages=NUM_STAGES):
                col0 = kn * BLOCK_N
                for j in T.Parallel(BLOCK_N):
                    abs_col = col0 + j
                    page_local = abs_col // PAGE_SIZE
                    in_page_j[j] = abs_col % PAGE_SIZE
                    valid_j[j] = T.if_then_else(abs_col < kv_total_len, 1, 0)
                    page_idx_j[j] = T.if_then_else(
                        abs_col < kv_total_len,
                        KV_indices[kv_page_start + page_local],
                        0,
                    )
                for j, d in T.Parallel(BLOCK_N, HEAD_DIM):
                    is_valid = valid_j[j] != 0
                    k_tile[j, d] = T.if_then_else(
                        is_valid,
                        K_pool[page_idx_j[j], kv_head, in_page_j[j], d],
                        T.cast(0, dtype),
                    )
                    v_tile[j, d] = T.if_then_else(
                        is_valid,
                        V_pool[page_idx_j[j], kv_head, in_page_j[j], d],
                        T.cast(0, dtype),
                    )

                T.clear(scores)
                T.gemm(q_tile, k_tile, scores, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                # Causal mask: q's absolute pos = kv_offset + row.
                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    row = row0 + i
                    col = col0 + j
                    in_bounds = (row < qlen) and (col < kv_total_len)
                    causal = col <= kv_offset + row
                    scores[i, j] = T.if_then_else(
                        in_bounds and causal,
                        scores[i, j] * sm_scale,
                        -T.infinity(accum_dtype),
                    )

                m_prev = T.alloc_fragment((BLOCK_M,), accum_dtype)
                m_new = T.alloc_fragment((BLOCK_M,), accum_dtype)
                p = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
                T.copy(m_i, m_prev)
                # clear=True is load-bearing: clear=False leaves m_new uninit, and
                # codegen's `max(m_new[i], m_new_clear[i])` then leaks stack garbage
                # (incl. NaN) → short-qlen NaN regression. See
                # docs/experience/errors/2026-04-28-tilelang-prefill-short-qlen-nan.md.
                T.reduce_max(scores, m_new, dim=1, clear=True)
                for i in T.Parallel(BLOCK_M):
                    m_new[i] = T.max(m_prev[i], m_new[i])
                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    p[i, j] = T.exp2((scores[i, j] - m_new[i]) * log2e)
                # Hoist per-row alpha to a fragment + rescale acc_o as a 2D
                # T.Parallel: the nested T.serial(HEAD_DIM) form trips TileLang
                # 0.1.9's LayoutInferencer (`loop_var_to_thread ... inner var d`).
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
                # Narrow f32 softmax to bf16 to match v_tile before P @ V (gemm
                # asserts A.dtype == B.dtype in TileLang 0.1.9).
                p_bf16 = T.alloc_fragment((BLOCK_M, BLOCK_N), dtype)
                T.copy(p, p_bf16)
                T.gemm(p_bf16, v_tile, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i, d in T.Parallel(BLOCK_M, HEAD_DIM):
                row = row0 + i
                if row < qlen:
                    Output[q_start + row, by * HEAD_DIM + d] = T.cast(
                        acc_o[i, d] / l_i[i], dtype
                    )

    return kernel


def get_kernel(num_q_heads: int, num_kv_heads: int):
    """Entry point for gen_tilelang_aot.py. One specialization per call."""
    return _make_kernel(num_q_heads, num_kv_heads)
