"""TileLang batch prefill HD128 paged attention.

HD128, BF16, causal, page_size=16. One kernel is AOT-specialized per
(num_q_heads, num_kv_heads) pair in SUPPORTED_HEADS — keeping these as
compile-time constants gives TileLang the freedom to specialize codegen
per shape instead of paying for runtime parameterization. Add a new Qwen3
size by extending the lockstep lists in this module, cuda-kernels/build.rs,
cuda-kernels/src/ffi/attention.rs, and infer/src/ops/attention.rs.

Tile / pipeline tunables (chosen as Hopper defaults; tuned during the
H100 spike per docs/plans/tilelang-integration.md §6):
  BLOCK_M = 64   q-tile rows
  BLOCK_N = 64   kv-tile cols (= PAGE_SIZE * 4)
  NUM_STAGES = 2
  NUM_THREADS = 128 (4 warps)
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

# (num_q_heads, num_kv_heads) configurations the Phase 0 build emits.
# Mirrors the Qwen3 HD128 family at the time of writing. Extend here +
# the build.rs list + the matching FFI/Rust dispatch arms in lockstep.
SUPPORTED_HEADS = (
    (16, 8),   # Qwen3-0.6B / 1.7B
    (32, 8),   # Qwen3-4B / 8B
    (40, 8),   # Qwen3-14B
    (64, 8),   # Qwen3-32B
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
        # TileLang 0.1.9 cannot use T.symbolic in grid extents — symbols
        # there must come from a tensor shape or a kernel scalar arg.
        # Pass batch / max_qlen as int32 runtime scalars instead, mirroring
        # tile-ai/tilelang's example_gqa_fwd_varlen pattern.
        batch_size: T.int32,
        max_qlen: T.int32,
    ):
        # Grid: (q_tile_blocks_for_longest_request, num_q_heads, batch_size).
        # Each block handles BLOCK_M consecutive q rows of one request, for
        # one query head. KV pages walked sequentially via KV_indices.
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
            # Hoisted out of the KV loop — same per-request constant the mask
            # below references; precomputing here also feeds the causal-bound
            # loop count.
            kv_offset = kv_total_len - qlen

            row0 = bx * BLOCK_M
            kv_head = by // gqa_group

            # Causal-bound KV loop: rows row0..min(row0+BLOCK_M, qlen)-1
            # attend at most to KV col `kv_offset + last_row` (the last
            # real row's diagonal). For full Q-tiles (row0+BLOCK_M ≤ qlen)
            # the bound is `kv_offset + row0 + BLOCK_M`; for the last
            # partial Q-tile (row0+BLOCK_M > qlen) the actual rows stop at
            # qlen-1, so the tighter bound is `kv_offset + qlen = kv_total_len`.
            # Pick the tighter of the two — `min(row0+BLOCK_M, qlen)`. The
            # outer `min(_, kv_total_len)` then clamps to the cache extent.
            # Mirrors FlashInfer's `mask_iteration` pattern in `prefill.cuh`.
            # For 4096-in cold prefill this drops ~35-50% of the per-tile
            # QK + softmax + PV work the unbounded loop wasted.
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

            # Per-tile page-index precomputes — only depend on j, not on
            # the head-dim d. Hoisting these into a 1D fragment kills the
            # ~128x duplicate divmod + KV_indices gather the original
            # (j, d) loop incurred. Mirrors FlashInfer's per-thread
            # `thr_local_kv_offset[]` cache in `prefill.cuh:2192-2287`.
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

                # Causal mask: q's absolute pos = (kv_total_len - qlen) + row.
                # `kv_offset` was hoisted above the loop.
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
                # `clear=True` initializes m_new to -inf before the reduction.
                # The previous `clear=False` left m_new uninitialized — TileLang
                # codegen emits `m_new[i] = max(m_new[i], m_new_clear[i])`
                # reading uninit stack memory, which is the actual root cause
                # of the short-qlen NaN regression: any nonzero garbage (incl.
                # NaN) leaks into m_new even on valid rows. See
                # docs/experience/errors/2026-04-28-tilelang-prefill-short-qlen-nan.md.
                T.reduce_max(scores, m_new, dim=1, clear=True)
                for i in T.Parallel(BLOCK_M):
                    m_new[i] = T.max(m_prev[i], m_new[i])
                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    p[i, j] = T.exp2((scores[i, j] - m_new[i]) * log2e)
                # Hoist the per-row alpha into its own fragment then drive
                # the acc_o rescale as a 2D T.Parallel — the nested
                # T.serial(HEAD_DIM) inside T.Parallel(BLOCK_M) version
                # produced a layout TileLang 0.1.9's LayoutInferencer can't
                # map to threads (`loop_var_to_thread ... contains inner
                # var d`).
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
                # before the P @ V matmul (standard FlashAttention-2
                # pattern). TileLang 0.1.9's gemm asserts A.dtype ==
                # B.dtype; older versions auto-cast silently.
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
