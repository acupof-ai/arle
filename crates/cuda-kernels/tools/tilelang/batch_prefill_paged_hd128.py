"""TileLang batch prefill HD128 paged attention (BF16 I/O, causal, page_size=16).

sm_70 (Volta) variant: I/O tensors stay bf16 (runtime ABI), but the GEMM
operands are fed in fp16 on sm<80 so the gemms route to TileLang's stock Volta
tensor-core path (GemmMMASm70 / mma.sync) instead of the scalar cuda.fma
fallback. The fma fallback's per-output _linear_fragment layout diverges between
scores(64x64) and acc_o(64x128) → the LayoutInference m_prev/scale_i conflict.
The MMA path reconciles both layouts exactly like sm_80+, so the conflict
vanishes and we get real tensor cores. sm>=80 keeps the byte-identical bf16 path.

One kernel is AOT-specialized per (num_q_heads, num_kv_heads) in SUPPORTED_HEADS.
Tile tunables: BLOCK_M=64, BLOCK_N=64, NUM_STAGES=2, NUM_THREADS=128.
"""

import math
import os

import tilelang
import tilelang.language as T

HEAD_DIM = 128
PAGE_SIZE = 16
BLOCK_M = 64
BLOCK_N = 64
NUM_STAGES = 2
NUM_THREADS = 128

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

    dtype = "bfloat16"  # I/O ABI: model weights + KV pool are bf16
    accum_dtype = "float32"
    index_dtype = "int32"

    # bf16 tensor cores require sm_80+. On Volta/Turing (sm<80) feed the GEMMs
    # fp16 so AllowVoltaMma passes and dispatch picks kCudaMMA (GemmMMASm70),
    # not the scalar kCudaFMA fallback. fp16 has 10-bit mantissa (vs bf16's 7);
    # range ±65504 is ample for RMSNorm-bounded q/k/v activations; accum stays f32.
    sm_arch = int(os.environ.get("ARLE_TILELANG_CUDA_ARCH", "90"))
    gemm_dtype = "float16" if sm_arch < 80 else "bfloat16"

    # Trace-time no-op when gemm_dtype == dtype → sm_80+ AST stays byte-identical
    # to the repo kernel; only sm<80 inserts the bf16->fp16 operand cast. The
    # cast routes through f32: a direct bf16->fp16 cast lowers to an ambiguous
    # user-defined conversion (__nv_bfloat16 -> cutlass::half_t has >1 path) that
    # nvcc rejects; bf16->f32->fp16 has a unique conversion at each step.
    def to_gemm(x):
        if gemm_dtype == dtype:
            return x
        return T.cast(T.cast(x, accum_dtype), gemm_dtype)

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
        batch_size: T.int32,
        max_qlen: T.int32,
    ):
        with T.Kernel(
            T.ceildiv(max_qlen, BLOCK_M),
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

            q_start = Q_indptr[bz]
            q_end = Q_indptr[bz + 1]
            qlen = q_end - q_start
            kv_page_start = KV_indptr[bz]
            kv_page_end = KV_indptr[bz + 1]
            num_kv_pages = kv_page_end - kv_page_start
            last_page_len = KV_last_page_len[bz]
            kv_total_len = (num_kv_pages - 1) * PAGE_SIZE + last_page_len
            kv_offset = kv_total_len - qlen

            row0 = bx * BLOCK_M
            kv_head = by // gqa_group

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
                    to_gemm(Q[src, by * HEAD_DIM + d]),
                    T.cast(0, gemm_dtype),
                )

            for kn in T.Pipelined(T.ceildiv(kv_visible_end, BLOCK_N), num_stages=NUM_STAGES):
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
                T.reduce_max(scores, m_new, dim=1, clear=True)
                for i in T.Parallel(BLOCK_M):
                    m_new[i] = T.max(m_prev[i], m_new[i])
                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    p[i, j] = T.exp2((scores[i, j] - m_new[i]) * log2e)
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
                # sm<80 Volta MMA (GemmMMASm70): feed A from SHARED — the f32
                # accumulator->fp16 MMA-operand-A fragment relayout isn't inferred
                # on the Volta path (p/p_bf16 conflict), but a fragment->shared
                # store + shared-operand gemm is, and mirrors the QK gemm (both
                # operands shared) which already lowers cleanly.
                if gemm_dtype != dtype:
                    p_shared = T.alloc_shared((BLOCK_M, BLOCK_N), gemm_dtype)
                    T.copy(p, p_shared)
                    T.gemm(p_shared, v_tile, acc_o, policy=T.GemmWarpPolicy.FullRow)
                else:
                    p_bf16 = T.alloc_fragment((BLOCK_M, BLOCK_N), gemm_dtype)
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
    return _make_kernel(num_q_heads, num_kv_heads)
