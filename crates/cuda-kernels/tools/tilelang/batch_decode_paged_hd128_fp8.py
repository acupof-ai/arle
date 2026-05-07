"""TileLang batch decode HD128 paged attention with FP8 E4M3 KV.

M_b.2 Phase A0 — single-config (q32_kv8 = Qwen3-4B) FP8 codegen smoke
test. Predicates the rest of M_b.2 phasing per
[`docs/plans/M_b.2-tilelang-hd128-fp8-decode.md`](../../docs/plans/M_b.2-tilelang-hd128-fp8-decode.md).

Sister to ``batch_decode_paged_hd128.py`` (BF16 KV path landed in
M_b.1 Phase A — `b42da5d`). Deltas:

  * ``K_pool`` / ``V_pool`` dtype:
    ``"bfloat16"`` → ``"float8_e4m3fn"``.
  * Adds ``K_scales`` / ``V_scales`` tensors of shape
    ``(num_pages, PAGE_SIZE, num_kv_heads)`` in fp32 — per-token
    per-kv-head scale factors. Layout matches the hand-CUDA
    ``[max_pages * page_size, num_kv_heads]`` indexing
    (`(phys_page * PAGE_SIZE + t) * num_kv_heads + kv_head`).
  * Inside the parallel load loop, dequant FP8 → BF16 inline:
    ``T.cast(T.cast(fp8, "float32") * scale, "bfloat16")``.
    The dequant happens *before* the value lands in `k_tile` /
    `v_tile` (shared mem), so `T.gemm(q_tile, k_tile, ...)` sees
    matching BF16 dtypes — required by TileLang 0.1.9
    (mixed-dtype GEMM is unsupported).
  * Q (input) and Output stay BF16.

Tile / pipeline tunables unchanged from the BF16 HD128 decode kernel.

Why A0 single-config: this validates that TileLang 0.1.9 can lower
``T.cast(fp8_e4m3fn, "float32")`` end-to-end (codegen → nvcc cubin →
ARLE link). If A0 cubin builds and links, A1 extends to all four
Qwen3 head shapes and adds a numerical-diff test against hand-CUDA
``decode_attention_varlen_fp8``. If A0 fails to codegen, M_b.2
pivots to fallback (B) — see plan §Risks.
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

# A0 scope: single config only (Qwen3-4B). Extend to (16,8) / (40,8) / (64,8)
# in A1 once codegen + numerical correctness are proven on (32, 8).
SUPPORTED_HEADS = (
    (32, 8),  # Qwen3-4B
)


def _make_kernel(num_q_heads: int, num_kv_heads: int):
    assert num_q_heads % num_kv_heads == 0, (
        f"num_q_heads ({num_q_heads}) must be a multiple of num_kv_heads ({num_kv_heads})"
    )
    gqa_group = num_q_heads // num_kv_heads
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    log2e = 1.4426950408889634

    out_dtype = "bfloat16"
    kv_dtype = "float8_e4m3fn"
    accum_dtype = "float32"
    index_dtype = "int32"

    @T.prim_func
    def kernel(
        # Q layout for decode: one row per request, no Q_indptr needed.
        Q: T.Tensor((T.symbolic("total_q_tokens"), num_q_heads * HEAD_DIM), out_dtype),
        K_pool: T.Tensor((T.symbolic("num_pages"), num_kv_heads, PAGE_SIZE, HEAD_DIM), kv_dtype),
        V_pool: T.Tensor((T.symbolic("num_pages"), num_kv_heads, PAGE_SIZE, HEAD_DIM), kv_dtype),
        # Per-token per-kv-head scales. Layout matches hand-CUDA
        # `[max_pages * page_size, num_kv_heads]` viewed as 3D
        # `[max_pages, page_size, num_kv_heads]` (row-major, last-dim contiguous).
        K_scales: T.Tensor((T.symbolic("num_pages"), PAGE_SIZE, num_kv_heads), accum_dtype),
        V_scales: T.Tensor((T.symbolic("num_pages"), PAGE_SIZE, num_kv_heads), accum_dtype),
        KV_indptr: T.Tensor((T.symbolic("batch_size_plus_one"),), index_dtype),
        KV_indices: T.Tensor((T.symbolic("total_pages"),), index_dtype),
        KV_last_page_len: T.Tensor((T.symbolic("batch_size"),), index_dtype),
        Output: T.Tensor((T.symbolic("total_q_tokens"), num_q_heads * HEAD_DIM), out_dtype),
        batch_size: T.int32,
        max_qlen: T.int32,
    ):
        # Grid: (1, num_q_heads, batch_size). One Q row per (request, head).
        with T.Kernel(
            1,
            num_q_heads,
            batch_size,
            threads=NUM_THREADS,
        ) as (bx, by, bz):
            q_tile = T.alloc_shared((BLOCK_M, HEAD_DIM), out_dtype)
            # k_tile / v_tile are BF16 *after dequant* — GEMM uses these.
            k_tile = T.alloc_shared((BLOCK_N, HEAD_DIM), out_dtype)
            v_tile = T.alloc_shared((BLOCK_N, HEAD_DIM), out_dtype)
            acc_o = T.alloc_fragment((BLOCK_M, HEAD_DIM), accum_dtype)
            scores = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
            m_i = T.alloc_fragment((BLOCK_M,), accum_dtype)
            l_i = T.alloc_fragment((BLOCK_M,), accum_dtype)

            T.use_swizzle(panel_size=8)

            kv_page_start = KV_indptr[bz]
            kv_page_end = KV_indptr[bz + 1]
            num_kv_pages = kv_page_end - kv_page_start
            last_page_len = KV_last_page_len[bz]
            kv_total_len = (num_kv_pages - 1) * PAGE_SIZE + last_page_len

            kv_head = by // gqa_group

            T.fill(acc_o, 0)
            T.fill(m_i, -T.infinity(accum_dtype))
            T.fill(l_i, 0)

            # Load Q (BF16 — no quantization on Q).
            for i, d in T.Parallel(BLOCK_M, HEAD_DIM):
                q_tile[i, d] = T.if_then_else(
                    i == 0,
                    Q[bz, by * HEAD_DIM + d],
                    T.cast(0, out_dtype),
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
                    # FP8 → f32 → * scale → BF16. Out-of-bounds path zeroes
                    # the BF16 store; the dequant work on the OOB lane is
                    # cheap and avoids branchy reads.
                    k_fp8 = K_pool[page_idx, kv_head, in_page, d]
                    k_scale = K_scales[page_idx, in_page, kv_head]
                    k_bf16 = T.cast(T.cast(k_fp8, accum_dtype) * k_scale, out_dtype)
                    k_tile[j, d] = T.if_then_else(
                        abs_col < kv_total_len,
                        k_bf16,
                        T.cast(0, out_dtype),
                    )
                    v_fp8 = V_pool[page_idx, kv_head, in_page, d]
                    v_scale = V_scales[page_idx, in_page, kv_head]
                    v_bf16 = T.cast(T.cast(v_fp8, accum_dtype) * v_scale, out_dtype)
                    v_tile[j, d] = T.if_then_else(
                        abs_col < kv_total_len,
                        v_bf16,
                        T.cast(0, out_dtype),
                    )

                T.clear(scores)
                T.gemm(q_tile, k_tile, scores, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                # No causal mask: qlen == 1 → single Q row attends to every
                # KV position in [0, kv_total_len).
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
                # Narrow softmax output to BF16 to match v_tile dtype before
                # the P @ V matmul.
                p_bf16 = T.alloc_fragment((BLOCK_M, BLOCK_N), out_dtype)
                T.copy(p, p_bf16)
                T.gemm(p_bf16, v_tile, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i, d in T.Parallel(BLOCK_M, HEAD_DIM):
                if i == 0:
                    Output[bz, by * HEAD_DIM + d] = T.cast(
                        acc_o[i, d] / l_i[i], out_dtype
                    )

    kernel.__name__ = f"batch_decode_paged_hd128_fp8_q{num_q_heads}_kv{num_kv_heads}_run"
    return kernel


def get_kernel(num_q_heads: int, num_kv_heads: int):
    """Entry point for gen_tilelang_aot.py. One specialization per call."""
    return _make_kernel(num_q_heads, num_kv_heads)
