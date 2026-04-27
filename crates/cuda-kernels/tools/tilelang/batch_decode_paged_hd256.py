"""TileLang batch decode HD256 paged attention.

Replaces FlashInfer's `flashinfer_batch_decode_hd256_run` for the HD256
paged decode path used by Qwen3.5 full-attention layers (the families
listed in SUPPORTED_HEADS below). Decode = single-token Q per request
(qo_len == 1), so the kernel reads paged K/V, runs one row of Q against
the full kv_total_len for that request, and writes one output row.

Adapted from `batch_prefill_paged_hd128.py`. Deltas vs the HD128 prefill
template (decode shape):

  * `HEAD_DIM = 256` (vs 128).
  * `SM_SCALE = 1.0 / sqrt(256)`.
  * `qo_len == 1` per request — drop `Q_indptr`. Q layout is
    `[batch_size, num_q_heads * 256]` (one row per request); the wrapper
    still passes `q_indptr` because the existing wrapper FFI is shared
    across kernels, but the kernel itself does not consume it (we drop
    the parameter).
  * No causal mask within Q. With qlen == 1, the single Q row legally
    attends to every KV position in `[0, kv_total_len_for_request_i)`,
    so we keep only the `col < kv_total_len` bounds clause.
  * `BLOCK_M = 1` (one Q row per tile).
  * `BLOCK_N = 64` (= 4 × PAGE_SIZE, same as HD128 prefill — decode
    tiles are KV-bound, not Q-bound).
  * Grid is `(1, num_q_heads, batch_size)` — drop the
    `T.ceildiv(max_qlen, BLOCK_M)` outer extent since there is no
    varlen Q.
  * Output layout `[batch_size, num_q_heads * 256]`.

Symbolic runtime int32 args (`batch_size`, `total_q_tokens`, `max_qlen`,
`num_pages`, `total_pages`) are kept identical to the HD128 kernel even
though `total_q_tokens == batch_size` and `max_qlen == 1` for decode.
This lets `gen_tilelang_aot.py::WRAPPER_FILL_RULES` work without
modification — no new symbolic shape variables are introduced.

Supported (num_q_heads, num_kv_heads) configurations (Qwen3.5 full-attn
families):
  (8, 2)   — Qwen3.5-0.8B
  (16, 2)  — Qwen3.6 MoE 30B-A3B
  (16, 4)  — Qwen3.5 medium / 14B / 32B-class

Tile / pipeline tunables (mirrors the HD128 prefill choices and the
upstream `tile-ai/tilelang/examples/flash_attention/example_gqa_decode*`
pattern):

  BLOCK_M = 1
  BLOCK_N = 64        (= PAGE_SIZE * 4)
  NUM_STAGES = 2
  NUM_THREADS = 128   (4 warps)

Shared-memory budget (static; TileLang's launcher will lift
`CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` to whatever total
the host source records):

  Q tile : BLOCK_M * HEAD_DIM * 2 B   = 1  * 256 * 2 =     512 B
  K tile : BLOCK_N * HEAD_DIM * 2 B   = 64 * 256 * 2 =  32_768 B
  V tile : BLOCK_N * HEAD_DIM * 2 B   = 64 * 256 * 2 =  32_768 B
  Sum (single-buffered)                              =  66_048 B (~64 KB)
  K/V are reused inside the `T.Pipelined(..., num_stages=2)` loop, so
  TileLang will double-buffer them: total dyn shmem ≈
    Q(512) + 2 * (32_768 + 32_768) = 131_584 B (~128 KB).
  Fits on H100 (per-block cap 228 KB after lift). On sm_89 / L4 the
  per-block cap is 99 KB; if this kernel must run there, drop
  NUM_STAGES to 1 or shrink BLOCK_N to 32. H100 is the target; sm_89
  is not part of HD256 decode's primary deployment.
"""

import math

import tilelang
import tilelang.language as T

HEAD_DIM = 256
PAGE_SIZE = 16
BLOCK_M = 1
BLOCK_N = 64
NUM_STAGES = 2
NUM_THREADS = 128

# (num_q_heads, num_kv_heads) configurations. Mirrors the Qwen3.5 HD256
# full-attn family. Extend here + the build.rs list + the matching
# FFI/Rust dispatch arms in lockstep when adding a new size.
SUPPORTED_HEADS = (
    (8, 2),    # Qwen3.5-0.8B
    (16, 2),   # Qwen3.6 MoE 30B-A3B
    (16, 4),   # Qwen3.5 medium / 14B / 32B-class
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
        # Keep the same scalar-arg shape as the HD128 prefill kernel even
        # though `max_qlen` is always 1 for decode; this lets
        # gen_tilelang_aot.py's WRAPPER_FILL_RULES fill them without any
        # changes (decode wrapper still passes them, HD128 wrapper still
        # passes them — same call shape).
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

            # Load the single Q row for (request bz, head by) into q_tile[0, :].
            for i, d in T.Parallel(BLOCK_M, HEAD_DIM):
                q_tile[i, d] = Q[bz + i, by * HEAD_DIM + d]

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
                    in_bounds = col < kv_total_len
                    scores[i, j] = T.if_then_else(
                        in_bounds,
                        scores[i, j] * sm_scale,
                        -T.infinity(accum_dtype),
                    )

                m_prev = T.alloc_fragment((BLOCK_M,), accum_dtype)
                m_new = T.alloc_fragment((BLOCK_M,), accum_dtype)
                p = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
                T.copy(m_i, m_prev)
                T.reduce_max(scores, m_new, dim=1, clear=False)
                for i in T.Parallel(BLOCK_M):
                    m_new[i] = T.max(m_prev[i], m_new[i])
                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    p[i, j] = T.exp2((scores[i, j] - m_new[i]) * log2e)
                # Hoist the per-row alpha into its own fragment then drive
                # the acc_o rescale as a 2D T.Parallel — same pattern as
                # the HD128 prefill kernel, required by TileLang 0.1.9's
                # LayoutInferencer.
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
                # before the P @ V matmul (FlashAttention-2 pattern).
                # TileLang 0.1.9's gemm asserts A.dtype == B.dtype.
                p_bf16 = T.alloc_fragment((BLOCK_M, BLOCK_N), dtype)
                T.copy(p, p_bf16)
                T.gemm(p_bf16, v_tile, acc_o, policy=T.GemmWarpPolicy.FullRow)

            # Single output row per (request, head). bz indexes the
            # request directly.
            for i, d in T.Parallel(BLOCK_M, HEAD_DIM):
                Output[bz + i, by * HEAD_DIM + d] = T.cast(
                    acc_o[i, d] / l_i[i], dtype
                )

    # Pin the kernel name so the generated symbol matches what the AOT
    # build.rs / FFI side will look up: batch_decode_paged_hd256_q{Q}_kv{K}_run.
    kernel.__name__ = f"batch_decode_paged_hd256_q{num_q_heads}_kv{num_kv_heads}_run"
    return kernel


def get_kernel(num_q_heads: int, num_kv_heads: int):
    """Entry point for gen_tilelang_aot.py. One specialization per call."""
    return _make_kernel(num_q_heads, num_kv_heads)
