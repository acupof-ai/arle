"""TileLang batch prefill HD256 paged attention.

HD256, BF16 I/O, causal, page_size=16. One kernel is AOT-specialized per
(num_q_heads, num_kv_heads) pair in SUPPORTED_HEADS — keeping these as
compile-time constants gives TileLang the freedom to specialize codegen
per shape instead of paying for runtime parameterization. Add a new
Qwen3.5 size by extending the lockstep lists in this module,
cuda-kernels/build.rs, cuda-kernels/src/ffi/attention.rs, and
infer/src/ops/attention.rs.

Mirror twin of `batch_prefill_paged_hd128.py`; the deltas vs HD128 are:

  1. `HEAD_DIM = 256` (was 128).
  2. `SM_SCALE = 1.0 / sqrt(256)` (folded in via `1.0 / math.sqrt(HEAD_DIM)`).
  3. `SUPPORTED_HEADS` covers the Qwen3.5 full-attention head configs:
       (8, 2)   — Qwen3.5-0.8B
       (16, 2)  — Qwen3.6 MoE 30B-A3B
       (16, 4)  — Qwen3.5 medium / 14B / 32B-class
  4. `BLOCK_N = 32` (was 64). Halving the KV tile keeps shared memory
     under sm_89's 100 KB per-block cap. Bf16 budget at NUM_STAGES=2:
       Q tile : BLOCK_M * HEAD_DIM * 2          = 64 * 256 * 2  = 32 KB
       K tile : BLOCK_N * HEAD_DIM * 2 * stages = 32 * 256 * 2 * 2 = 32 KB
       V tile : BLOCK_N * HEAD_DIM * 2 * stages = 32 * 256 * 2 * 2 = 32 KB
       total  ≈ 96 KB, fits the sm_89 100 KB cap with the same
       `MAX_DYNAMIC_SHARED_SIZE_BYTES` lift the HD128 wrapper already
       performs (see gen_tilelang_aot.py's wrapper template). fp16 GEMM
       operands on sm<80 are also 2 bytes, so the budget is unchanged.

sm_70 (Volta) variant: identical strategy to `batch_prefill_paged_hd128.py`.
I/O stays bf16 (runtime ABI); on sm<80 the GEMM operands are fed fp16 so the
gemms route to TileLang's stock Volta tensor-core path (GemmMMASm70 /
mma.sync) instead of the scalar cuda.fma fallback whose per-output fragment
layout triggers the LayoutInference m_prev/scale_i conflict. The PV operand A
is fed from shared on sm<80. sm>=80 keeps the byte-identical bf16 fragment
path. This replaces the earlier `SM70_FORCE_TWO_KV_TILES` band-aid, which only
worked around the broken fma-fallback lowering and is no longer needed once the
gemms take the MMA path.

Tile / pipeline tunables:
  BLOCK_M = 64    q-tile rows
  BLOCK_N = 32    kv-tile cols (= PAGE_SIZE * 2)
  NUM_STAGES = 2
  NUM_THREADS = 128 (4 warps)
"""

import math
import os

import tilelang
import tilelang.language as T

HEAD_DIM = 256
PAGE_SIZE = 16
BLOCK_M = 64
BLOCK_N = 32
NUM_STAGES = 2
NUM_THREADS = 128

# (num_q_heads, num_kv_heads) configurations the Phase 0 build emits.
# Mirrors the Qwen3.5 HD256 family at the time of writing. Extend here +
# the build.rs list + the matching FFI/Rust dispatch arms in lockstep.
SUPPORTED_HEADS = (
    (8, 1),    # Qwen3.5-122B-A10B at TP=4 (KV replication, 2 kv_heads / 2 → 1 per rank)
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

    # See batch_prefill_paged_hd128.py for the full rationale. sm<80 feeds the
    # GEMMs fp16 so dispatch picks kCudaMMA (GemmMMASm70), not the scalar
    # kCudaFMA fallback; accum stays f32. Trace-time no-op on sm>=80 → the
    # bf16 AST stays byte-identical to the repo kernel.
    sm_arch = int(os.environ.get("ARLE_TILELANG_CUDA_ARCH", "90"))
    gemm_dtype = "float16" if sm_arch < 80 else "bfloat16"

    # bf16->fp16 routes through f32: a direct cast lowers to an ambiguous
    # CUTLASS user-defined conversion nvcc rejects; bf16->f32->fp16 is unique.
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

            row0 = bx * BLOCK_M
            kv_head = by // gqa_group

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

                # Causal mask: q's absolute pos = (kv_total_len - qlen) + row.
                kv_offset = kv_total_len - qlen
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
                # PV operand A. sm_80+: bf16 fragment (byte-identical to repo).
                # sm<80 Volta MMA (GemmMMASm70): feed A from SHARED — mirrors
                # the QK gemm (both operands shared) which already lowers
                # cleanly; the f32->fp16 MMA-operand-A fragment relayout the
                # Volta path can't infer is sidestepped.
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
    """Entry point for gen_tilelang_aot.py. One specialization per call."""
    return _make_kernel(num_q_heads, num_kv_heads)
