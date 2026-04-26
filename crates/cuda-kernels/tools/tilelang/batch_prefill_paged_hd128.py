"""TileLang kernel for paged batch prefill attention, HD128, BF16, causal.

Contract (mirrors docs/plans/tilelang-integration.md §2):
  q                : [total_q_tokens, NUM_Q_HEADS * HEAD_DIM]  bf16
  q_indptr         : [batch_size + 1]                          int32
  k_pool           : paged K storage, HND layout               bf16
  v_pool           : paged V storage, HND layout               bf16
  kv_indptr        : [batch_size + 1]                          int32
  kv_indices       : flattened page indices                    int32
  kv_last_page_len : [batch_size]                              int32
  o                : [total_q_tokens, NUM_Q_HEADS * HEAD_DIM]  bf16

Compile-time constants (Qwen3-4B / Qwen3-8B share these):
  NUM_Q_HEADS  = 32
  NUM_KV_HEADS = 8
  HEAD_DIM     = 128
  PAGE_SIZE    = 16
  sm_scale     = 1 / sqrt(HEAD_DIM)

Tunables chosen as defaults for HD128 on Hopper:
  BLOCK_M = 64   q-tile rows; balances register pressure vs. occupancy at HD=128.
  BLOCK_N = 64   kv-tile cols; matches PAGE_SIZE * 4 so each step covers 4 pages.
  NUM_STAGES = 2 software pipeline depth for shared-memory loads.
  NUM_THREADS = 128 (4 warps) — standard FlashAttention-2 launch shape.

The H100 spike will tune these; see docs/plans/tilelang-integration.md §6.
"""

import math

import tilelang
import tilelang.language as T

NUM_Q_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
PAGE_SIZE = 16
GQA_GROUP = NUM_Q_HEADS // NUM_KV_HEADS

BLOCK_M = 64
BLOCK_N = 64
NUM_STAGES = 2
NUM_THREADS = 128


def _make_kernel(block_m: int = BLOCK_M, block_n: int = BLOCK_N,
                 num_stages: int = NUM_STAGES, num_threads: int = NUM_THREADS):
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    log2e = 1.4426950408889634

    dtype = "bfloat16"
    accum_dtype = "float32"
    index_dtype = "int32"

    @T.prim_func
    def batch_prefill_paged_hd128_kernel(
        Q: T.Tensor((T.symbolic("total_q_tokens"), NUM_Q_HEADS * HEAD_DIM), dtype),
        Q_indptr: T.Tensor((T.symbolic("batch_size_plus_one"),), index_dtype),
        K_pool: T.Tensor((T.symbolic("num_pages"), NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM), dtype),
        V_pool: T.Tensor((T.symbolic("num_pages"), NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM), dtype),
        KV_indptr: T.Tensor((T.symbolic("batch_size_plus_one"),), index_dtype),
        KV_indices: T.Tensor((T.symbolic("total_pages"),), index_dtype),
        KV_last_page_len: T.Tensor((T.symbolic("batch_size"),), index_dtype),
        Output: T.Tensor((T.symbolic("total_q_tokens"), NUM_Q_HEADS * HEAD_DIM), dtype),
    ):
        # Grid: (q_tile_blocks_per_request, num_q_heads, batch_size).
        # Each block handles BLOCK_M consecutive query rows of one request,
        # for one query head. The request's KV pages are walked sequentially.
        with T.Kernel(
            T.ceildiv(T.symbolic("max_qlen"), block_m),
            NUM_Q_HEADS,
            T.symbolic("batch_size"),
            threads=num_threads,
        ) as (bx, by, bz):
            q_tile = T.alloc_shared((block_m, HEAD_DIM), dtype)
            k_tile = T.alloc_shared((block_n, HEAD_DIM), dtype)
            v_tile = T.alloc_shared((block_n, HEAD_DIM), dtype)
            acc_o = T.alloc_fragment((block_m, HEAD_DIM), accum_dtype)
            scores = T.alloc_fragment((block_m, block_n), accum_dtype)
            m_i = T.alloc_fragment((block_m,), accum_dtype)
            l_i = T.alloc_fragment((block_m,), accum_dtype)

            T.use_swizzle(panel_size=8)

            q_start = Q_indptr[bz]
            q_end = Q_indptr[bz + 1]
            qlen = q_end - q_start
            kv_page_start = KV_indptr[bz]
            kv_page_end = KV_indptr[bz + 1]
            num_kv_pages = kv_page_end - kv_page_start
            last_page_len = KV_last_page_len[bz]
            kv_total_len = (num_kv_pages - 1) * PAGE_SIZE + last_page_len

            row0 = bx * block_m
            kv_head = by // GQA_GROUP

            T.fill(acc_o, 0)
            T.fill(m_i, -T.infinity(accum_dtype))
            T.fill(l_i, 0)

            # Load Q tile for this (request, q-head, row-tile).
            for i, d in T.Parallel(block_m, HEAD_DIM):
                row = row0 + i
                src = q_start + row
                q_tile[i, d] = T.if_then_else(
                    row < qlen,
                    Q[src, by * HEAD_DIM + d],
                    T.cast(0, dtype),
                )

            # Iterate paged KV in BLOCK_N-sized windows.
            for kn in T.Pipelined(
                T.ceildiv(kv_total_len, block_n), num_stages=num_stages
            ):
                col0 = kn * block_n
                # Materialize K/V tile from the page table.
                for j, d in T.Parallel(block_n, HEAD_DIM):
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
                T.gemm(q_tile, k_tile, scores, transpose_B=True)

                # Causal mask: q's absolute pos = (kv_total_len - qlen) + row.
                # Out-of-range rows / cols -> -inf so softmax ignores.
                kv_offset = kv_total_len - qlen
                for i, j in T.Parallel(block_m, block_n):
                    row = row0 + i
                    col = col0 + j
                    in_bounds = (row < qlen) and (col < kv_total_len)
                    causal = col <= kv_offset + row
                    scores[i, j] = T.if_then_else(
                        in_bounds and causal,
                        scores[i, j] * sm_scale,
                        -T.infinity(accum_dtype),
                    )

                # Online softmax + V accumulation.
                m_prev = T.alloc_fragment((block_m,), accum_dtype)
                m_new = T.alloc_fragment((block_m,), accum_dtype)
                p = T.alloc_fragment((block_m, block_n), accum_dtype)
                T.copy(m_i, m_prev)
                T.reduce_max(scores, m_new, dim=1, clear=False)
                for i in T.Parallel(block_m):
                    m_new[i] = T.max(m_prev[i], m_new[i])
                for i, j in T.Parallel(block_m, block_n):
                    p[i, j] = T.exp2((scores[i, j] - m_new[i]) * log2e)
                for i in T.Parallel(block_m):
                    alpha = T.exp2((m_prev[i] - m_new[i]) * log2e)
                    l_i[i] = l_i[i] * alpha
                    for d in T.serial(HEAD_DIM):
                        acc_o[i, d] = acc_o[i, d] * alpha
                row_sum = T.alloc_fragment((block_m,), accum_dtype)
                T.reduce_sum(p, row_sum, dim=1)
                for i in T.Parallel(block_m):
                    l_i[i] = l_i[i] + row_sum[i]
                    m_i[i] = m_new[i]
                T.gemm(p, v_tile, acc_o)

            # Normalize + write back, gated on row < qlen.
            for i, d in T.Parallel(block_m, HEAD_DIM):
                row = row0 + i
                if row < qlen:
                    Output[q_start + row, by * HEAD_DIM + d] = T.cast(
                        acc_o[i, d] / l_i[i], dtype
                    )

    return batch_prefill_paged_hd128_kernel


def get_kernel():
    """Entry point for gen_tilelang_aot.py."""
    return _make_kernel()
