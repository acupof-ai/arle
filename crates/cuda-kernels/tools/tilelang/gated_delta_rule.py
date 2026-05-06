"""TileLang chunk-wise Gated Delta Rule (GDR) kernels for Qwen3.5 hybrid.

Phase 2 of `docs/plans/2026-05-05-cuda-kernel-tilelang-unification.md`:
the canonical chunk-wise GDR pipeline with TileLang AOT stages. The
strict-lower triangular solve is kept as native CUDA C because TileLang 0.1.9
cannot lower that stage's mixed-index fragment layout on sm_89.

Upstream references this file is adapted from (per user direction
2026-05-05 "可以直接抄过来用"):

  1. Flash Linear Attention (FLA) — Apache-2.0
     https://github.com/fla-org/flash-linear-attention
     Specifically:
       - fla/ops/gated_delta_rule/chunk.py
       - fla/ops/common/chunk_delta_h.py
       - fla/ops/common/chunk_o.py
       - fla/ops/gated_delta_rule/wy_fast.py
     ARLE's implementation keeps the FLA stage structure while using the
     TileLang kernel substrate.

  2. FlashQLA (Qwen team / Alibaba) — MIT, commit e88d71a1
     https://github.com/QwenLM/FlashQLA
     Specifically:
       - flash_qla/ops/utils/cumsum.py             (T.cumsum reference)
       - flash_qla/ops/gated_delta_rule/chunk/hopper/kkt_solve.py
                                                   (4-block solve_tril
                                                    inversion pattern)
       - flash_qla/ops/gated_delta_rule/chunk/hopper/fused_fwd.py
                                                   (chunk-state + output
                                                    fusion pattern)
     FlashQLA is sm_90/Hopper-only (uses TMA + warp-specialization +
     `T.alloc_barrier` / `T.barrier_arrive`). ARLE supports the
     sm_75/80/86/89/90 fat-build, so we cannot drop FlashQLA's Hopper
     kernels in directly. We use FlashQLA for *algorithmic structure*
     (block decomposition of solve_tril, fused chunk-state recurrence)
     while keeping the SM-portable TileLang primitives (T.gemm,
     T.alloc_shared, T.alloc_fragment, T.Pipelined) that already work
     across the ARLE SM tier.

ARLE-specific deltas vs both upstream sources:

  - Fixed Qwen3.5 shape: KEY_DIM=128, VALUE_DIM=128, BLOCK_T=64,
    KEY_BLOCK=64, num_value_heads=32, batch=1.
  - No backward, no varlen surface (single-sequence prefill only at
    the operator level — multi-sequence batches are scheduled by the
    surrounding scheduler, not the kernel).
  - Decode-compatible final-state layout `[H, K, V]` with V contiguous
    (matches the FLA decode convention ARLE standardized on).
  - Per-token v_new staging tensor for the chunk-state -> chunk-o handoff.

Phase 2b status — AOT swap wired
---------------------------------

`tools/tilelang/gen_tilelang_aot.py` now has a `gdr` kernel family beside the
paged-attention family. `build.rs` emits the TileLang-backed public C symbols
(`gated_delta_rule_prefill_chunk_*_cuda`) for the AOT-compatible stages, while
`csrc/misc/gdr_prefill_solve.cu` owns the solve symbol. Rust FFI and
`infer/src/ops/recurrent.rs` call sites remain stable. The old external AOT
directory is gone. GPU numerical validation compares this path via the
existing Qwen3.5 e2e tests and JSON baselines.
"""

import tilelang  # noqa: F401  (imported for side-effect-free version probe)
import tilelang.language as T

# Fixed Qwen3.5 GDR runtime shape. This is the only combination ARLE ships today.
QWEN35_GDR_HEADS = 32
QWEN35_GDR_CHUNK_SIZE = 64
QWEN35_GDR_KEY_DIM = 128
QWEN35_GDR_VALUE_DIM = 128
QWEN35_GDR_KEY_BLOCK = 64

# Tile / pipeline tunables.
# (num_warps=4 → ~128 threads, num_stages=2). The AOT generator's
# nvcc -O3 + cuFuncSetAttribute already lifts the dyn-shmem cap, so
# the per-kernel choices below stay portable across sm_75..sm_90.
NUM_THREADS = 128
NUM_STAGES = 2

# Internal block sizes for the chunk-wise decomposition.
BLOCK_T = 64       # chunk size in tokens
BLOCK_K = 64       # key tile width (= KEY_DIM // 2 partitioning for state)
BLOCK_V = 32       # value tile width for chunk-state / chunk-o sweeps
BLOCK_K_TILE = 64  # full-KEY_DIM tile width for chunk-o GEMM
KEY_BLOCK = QWEN35_GDR_KEY_BLOCK


def _gdr_chunk_prepare_kernel():
    """Stage 1 — prepare normalized q/k, raw v, raw g/beta from packed QKV.

    One thread block per (token, value_head); inside
    the block we read one q row, one k row, one v row, normalize q/k
    (RMSNorm-style L2), and emit the per-token gate / beta scalars.
    """
    dtype = "bfloat16"
    accum_dtype = "float32"
    index_dtype = "int32"  # noqa: F841  (reserved for future T.dynamic shape)
    KEY_DIM = QWEN35_GDR_KEY_DIM
    VALUE_DIM = QWEN35_GDR_VALUE_DIM

    @T.prim_func
    def kernel(
        qkv: T.Tensor((T.symbolic("seq_len"), T.symbolic("qkv_dim")), dtype),
        b_proj: T.Tensor((T.symbolic("seq_len"), T.symbolic("hv")), dtype),
        a_proj: T.Tensor((T.symbolic("seq_len"), T.symbolic("hv")), dtype),
        dt_bias: T.Tensor((T.symbolic("hv"),), dtype),
        a_log: T.Tensor((T.symbolic("hv"),), accum_dtype),
        q_out: T.Tensor((T.symbolic("seq_len"), T.symbolic("hv"), KEY_DIM), dtype),
        k_out: T.Tensor((T.symbolic("seq_len"), T.symbolic("hv"), KEY_DIM), dtype),
        v_out: T.Tensor((T.symbolic("seq_len"), T.symbolic("hv"), VALUE_DIM), dtype),
        g_out: T.Tensor((T.symbolic("seq_len"), T.symbolic("hv")), accum_dtype),
        beta_out: T.Tensor((T.symbolic("seq_len"), T.symbolic("hv")), accum_dtype),
        num_key_heads: T.int32,
        num_value_heads: T.int32,
        qkv_dim: T.int32,
        seq_len: T.int32,
    ):
        # Grid: (seq_len, num_value_heads). One block reads one
        # (token, head) row's q/k/v slice and a/b/dt_bias/a_log scalars.
        with T.Kernel(seq_len, num_value_heads, threads=NUM_THREADS) as (token_idx, v_head):
            q_frag = T.alloc_fragment((KEY_DIM,), accum_dtype)
            k_frag = T.alloc_fragment((KEY_DIM,), accum_dtype)
            v_frag = T.alloc_fragment((VALUE_DIM,), dtype)
            qq_sum = T.alloc_fragment((1,), accum_dtype)
            kk_sum = T.alloc_fragment((1,), accum_dtype)

            k_head = (v_head * num_key_heads) // num_value_heads
            qk_dim_total = num_key_heads * KEY_DIM
            v_offset = qk_dim_total * 2 + v_head * VALUE_DIM

            # Load q, k (group-shared between v_heads in the same group)
            # and v (per-v_head). Cast to fp32 for the rsqrt normalization.
            for d in T.Parallel(KEY_DIM):
                q_frag[d] = T.cast(qkv[token_idx, k_head * KEY_DIM + d], accum_dtype)
                k_frag[d] = T.cast(qkv[token_idx, qk_dim_total + k_head * KEY_DIM + d], accum_dtype)
            for d in T.Parallel(VALUE_DIM):
                v_frag[d] = qkv[token_idx, v_offset + d]

            # L2 normalize q and k: q *= rsqrt(sum(q*q) + 1e-12).
            T.clear(qq_sum)
            T.clear(kk_sum)
            for d in T.serial(KEY_DIM):
                qq_sum[0] += q_frag[d] * q_frag[d]
                kk_sum[0] += k_frag[d] * k_frag[d]
            q_scale = T.rsqrt(qq_sum[0] + T.cast(1e-12, accum_dtype))
            k_scale = T.rsqrt(kk_sum[0] + T.cast(1e-12, accum_dtype))
            for d in T.Parallel(KEY_DIM):
                q_out[token_idx, v_head, d] = T.cast(q_frag[d] * q_scale, dtype)
                k_out[token_idx, v_head, d] = T.cast(k_frag[d] * k_scale, dtype)
            for d in T.Parallel(VALUE_DIM):
                v_out[token_idx, v_head, d] = v_frag[d]

            # g = -exp(a_log) * softplus(a + dt_bias); beta = sigmoid(b).
            a_val = T.cast(a_proj[token_idx, v_head], accum_dtype)
            b_val = T.cast(b_proj[token_idx, v_head], accum_dtype)
            dt = T.cast(dt_bias[v_head], accum_dtype)
            al = a_log[v_head]
            x = a_val + dt
            softplus_x = T.if_then_else(
                x > T.cast(20.0, accum_dtype),
                x,
                T.log(T.cast(1.0, accum_dtype) + T.exp(x)),
            )
            g_out[token_idx, v_head] = -T.exp(al) * softplus_x
            beta_out[token_idx, v_head] = T.cast(1.0, accum_dtype) / (
                T.cast(1.0, accum_dtype) + T.exp(-b_val)
            )

    return kernel


def _gdr_chunk_local_cumsum_kernel():
    """Stage 2 — chunk-local prefix sum over the gate tensor.

    Translation of `gdr_chunk_local_cumsum_qwen35_kernel`. Reuses the
    FlashQLA pattern (`tilelang_chunk_local_cumsum`) of T.cumsum on a
    [BLOCK_T] fragment, but specializes for the fixed BLOCK_T=64 and
    drops the varlen + reverse paths we don't need.
    """
    dtype = "float32"
    BT = BLOCK_T

    @T.prim_func
    def kernel(
        g_in: T.Tensor((T.symbolic("seq_len"), T.symbolic("hv")), dtype),
        g_out: T.Tensor((T.symbolic("seq_len"), T.symbolic("hv")), dtype),
        seq_len: T.int32,
        num_value_heads: T.int32,
    ):
        # Grid: (num_chunks = ceildiv(seq_len, BT), num_value_heads).
        with T.Kernel(
            T.ceildiv(seq_len, BT), num_value_heads, threads=NUM_THREADS
        ) as (chunk_idx, v_head):
            g_frag = T.alloc_fragment((BT,), dtype)
            base = chunk_idx * BT
            for t in T.Parallel(BT):
                g_frag[t] = T.if_then_else(
                    base + t < seq_len,
                    g_in[base + t, v_head],
                    T.cast(0.0, dtype),
                )
            T.cumsum(g_frag, dim=0)
            for t in T.Parallel(BT):
                if base + t < seq_len:
                    g_out[base + t, v_head] = g_frag[t]

    return kernel


def _gdr_chunk_scaled_dot_kkt_kernel():
    """Stage 3 — build the chunk-local strict lower-triangular A block.

    Translation of `gdr_chunk_scaled_dot_kkt_qwen35_kernel`. The
    kernel walks `KEY_DIM // BLOCK_K` k-slabs and accumulates
    `K @ K^T` into a `[BT, BT]` accumulator, then applies the gate
    decay and beta scaling and zeroes the upper-triangular + diagonal
    via a causal mask.

    The TileLang translation uses one `T.gemm(transpose_B=True)` call
    on the full `[BT, KEY_DIM]` tile (KEY_DIM=128 fits comfortably in
    shared memory) and skips the explicit slab loop.
    """
    dtype = "bfloat16"
    accum_dtype = "float32"
    g_dtype = "float32"
    KEY_DIM = QWEN35_GDR_KEY_DIM
    BT = BLOCK_T

    @T.prim_func
    def kernel(
        k: T.Tensor((T.symbolic("seq_len"), T.symbolic("hv"), KEY_DIM), dtype),
        g_cumsum: T.Tensor((T.symbolic("seq_len"), T.symbolic("hv")), g_dtype),
        beta: T.Tensor((T.symbolic("seq_len"), T.symbolic("hv")), g_dtype),
        a_tril: T.Tensor((T.symbolic("seq_len"), T.symbolic("hv"), BT), g_dtype),
        seq_len: T.int32,
        num_value_heads: T.int32,
    ):
        # Grid: (num_chunks, num_value_heads).
        with T.Kernel(
            T.ceildiv(seq_len, BT), num_value_heads, threads=NUM_THREADS
        ) as (chunk_idx, v_head):
            k_tile = T.alloc_shared((BT, KEY_DIM), dtype)
            g_shared = T.alloc_shared((BT,), accum_dtype)
            beta_shared = T.alloc_shared((BT,), accum_dtype)
            acc = T.alloc_fragment((BT, BT), accum_dtype)

            base = chunk_idx * BT
            # Load K[base:base+BT, v_head, :] — boundary-safe.
            for t, d in T.Parallel(BT, KEY_DIM):
                k_tile[t, d] = T.if_then_else(
                    base + t < seq_len,
                    k[base + t, v_head, d],
                    T.cast(0, dtype),
                )

            T.clear(acc)
            T.gemm(k_tile, k_tile, acc, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

            # Load g and beta for this chunk's tokens.
            for t in T.Parallel(BT):
                g_shared[t] = T.if_then_else(
                    base + t < seq_len,
                    g_cumsum[base + t, v_head],
                    T.cast(0.0, accum_dtype),
                )
                beta_shared[t] = T.if_then_else(
                    base + t < seq_len,
                    beta[base + t, v_head],
                    T.cast(0.0, accum_dtype),
                )

            # acc[i,j] *= beta[i] * exp(g[i] - g[j]); zero upper+diag.
            for i, j in T.Parallel(BT, BT):
                row_in = base + i < seq_len
                col_in = base + j < seq_len
                masked = (i > j) and row_in and col_in
                acc[i, j] = T.if_then_else(
                    masked,
                    acc[i, j] * beta_shared[i] * T.exp(g_shared[i] - g_shared[j]),
                    T.cast(0.0, accum_dtype),
                )

            for i, j in T.Parallel(BT, BT):
                if base + i < seq_len:
                    a_tril[base + i, v_head, j] = acc[i, j]

    return kernel


def _gdr_solve_tril_64_kernel():
    """Stage 4 — fixed-size BT=64 strict-lower-triangular solve.

    Translation of `gdr_solve_tril_64_qwen35_kernel`. This is the
    HARDEST stage: the implementation decomposes the
    64x64 strict-lower-triangular inverse into 4 diagonal 16x16 blocks
    + 6 off-diagonal 16x16 blocks, then composes them via 9 GEMMs
    (one per 16x16 piece of the lower triangle).

    The TileLang translation mirrors the FlashQLA `kkt_solve.py`
    inversion pattern (4 levels: 16x16 diagonal forward-substitution,
    then 1×, 2×, 1× off-diagonal GEMMs to extend) but uses generic
    `T.gemm` + `T.Parallel` + `T.alloc_shared` instead of FlashQLA's
    Hopper-specific `T.gemm_v1` + `T.alloc_barrier`. This keeps the
    cubin loadable on sm_75 through sm_90.

    Phase 2b note: this stage needs the most GPU validation. The
    4-level block decomposition is identical
    in structure to both upstream sources, but the exact T.Parallel
    layout choices interact with TileLang's LayoutInferencer and may
    need tuning per docs/experience/errors/2026-04-28-tilelang-prefill-short-qlen-nan.md
    style adjustments. Author's intent: keep the *algorithm* faithful
    to the FLA reference and use GPU validation to pin down layout details.
    """
    dtype = "bfloat16"
    accum_dtype = "float32"
    BT = BLOCK_T

    @T.prim_func
    def kernel(
        a_tril: T.Tensor((T.symbolic("seq_len"), T.symbolic("hv"), BT), accum_dtype),
        a_inv: T.Tensor((T.symbolic("seq_len"), T.symbolic("hv"), BT), dtype),
        seq_len: T.int32,
        num_value_heads: T.int32,
    ):
        with T.Kernel(
            T.ceildiv(seq_len, BT), num_value_heads, threads=NUM_THREADS
        ) as (chunk_idx, v_head):
            # Load the full 64x64 a_tril block for this chunk.
            a_full = T.alloc_shared((BT, BT), accum_dtype)
            base = chunk_idx * BT
            for i, j in T.Parallel(BT, BT):
                a_full[i, j] = T.if_then_else(
                    base + i < seq_len,
                    a_tril[base + i, v_head, j],
                    T.cast(0.0, accum_dtype),
                )

            # 4 diagonal 16x16 blocks initialized as -StrictLower(A) + I.
            ai_diag = T.alloc_shared((4, 16, 16), accum_dtype)
            for blk, i, j in T.Parallel(4, 16, 16):
                row = blk * 16 + i
                col = blk * 16 + j
                lower = i > j
                eye = i == j
                ai_diag[blk, i, j] = T.if_then_else(
                    lower,
                    -a_full[row, col],
                    T.if_then_else(eye, T.cast(1.0, accum_dtype), T.cast(0.0, accum_dtype)),
                )

            # Forward-substitution on each diagonal block: rows 2..15.
            # We unroll over `blk` to keep the inner reduction shape regular.
            row_buf = T.alloc_shared((4, 16), accum_dtype)
            for i in T.serial(2, 16):
                for blk, k_t in T.Parallel(4, 16):
                    base_i = chunk_idx * BT + blk * 16 + i
                    in_range = base_i < seq_len
                    a_row_val = T.if_then_else(
                        in_range and (k_t < i),
                        -a_full[blk * 16 + i, blk * 16 + k_t],
                        T.cast(0.0, accum_dtype),
                    )
                    row_buf[blk, k_t] = a_row_val
                # Accumulate row_buf @ ai_diag[blk] into row i of ai_diag[blk].
                for blk, k_t in T.Parallel(4, 16):
                    accum = T.alloc_fragment((1,), accum_dtype)
                    accum[0] = T.cast(0.0, accum_dtype)
                    for r in T.serial(16):
                        accum[0] += row_buf[blk, r] * ai_diag[blk, r, k_t]
                    if k_t < i:
                        ai_diag[blk, i, k_t] = row_buf[blk, k_t] + accum[0]

            # Identity on the diagonal is already captured by the eye
            # initialization above, so this is a no-op.

            # Six off-diagonal 16x16 blocks of A: A21, A31, A32, A41, A42, A43.
            # Loaded into a single (6, 16, 16) fragment for the composition.
            a_off = T.alloc_shared((6, 16, 16), accum_dtype)
            # Block index → (row_block, col_block) of A:
            #   0: (1, 0)  A21
            #   1: (2, 0)  A31
            #   2: (2, 1)  A32
            #   3: (3, 0)  A41
            #   4: (3, 1)  A42
            #   5: (3, 2)  A43
            for slot, i, j in T.Parallel(6, 16, 16):
                # Branchless lookup for (row_block, col_block).
                row_block = T.if_then_else(
                    slot == 0, 1,
                    T.if_then_else(slot == 1, 2,
                    T.if_then_else(slot == 2, 2,
                    T.if_then_else(slot == 3, 3,
                    T.if_then_else(slot == 4, 3, 3)))))
                col_block = T.if_then_else(
                    slot == 0, 0,
                    T.if_then_else(slot == 1, 0,
                    T.if_then_else(slot == 2, 1,
                    T.if_then_else(slot == 3, 0,
                    T.if_then_else(slot == 4, 1, 2)))))
                a_off[slot, i, j] = a_full[row_block * 16 + i, col_block * 16 + j]

            # Compose the off-diagonal pieces of the inverse:
            #   b_ai_21 = -ai22 @ A21 @ ai11
            #   b_ai_32 = -ai33 @ A32 @ ai22
            #   b_ai_43 = -ai44 @ A43 @ ai33
            #   b_ai_31 = -ai33 @ (A31 @ ai11 + A32 @ b_ai_21)
            #   b_ai_42 = -ai44 @ (A42 @ ai22 + A43 @ b_ai_32)
            #   b_ai_41 = -ai44 @ (A41 @ ai11 + A42 @ b_ai_21 + A43 @ b_ai_31)
            #
            # Implemented as a sequence of 16x16 matmuls expressed as
            # T.Parallel reductions. Each output is stored back into a
            # dedicated (6, 16, 16) `ai_off` fragment.
            ai_off = T.alloc_shared((6, 16, 16), accum_dtype)

            # ai_21 = -ai_22 @ A21 @ ai_11
            tmp_a = T.alloc_shared((16, 16), accum_dtype)
            tmp_b = T.alloc_shared((16, 16), accum_dtype)
            for i, j in T.Parallel(16, 16):
                acc = T.alloc_fragment((1,), accum_dtype)
                acc[0] = T.cast(0.0, accum_dtype)
                for r in T.serial(16):
                    acc[0] += ai_diag[1, i, r] * a_off[0, r, j]
                tmp_a[i, j] = acc[0]
            for i, j in T.Parallel(16, 16):
                acc = T.alloc_fragment((1,), accum_dtype)
                acc[0] = T.cast(0.0, accum_dtype)
                for r in T.serial(16):
                    acc[0] += tmp_a[i, r] * ai_diag[0, r, j]
                ai_off[0, i, j] = -acc[0]

            # ai_32 = -ai_33 @ A32 @ ai_22
            for i, j in T.Parallel(16, 16):
                acc = T.alloc_fragment((1,), accum_dtype)
                acc[0] = T.cast(0.0, accum_dtype)
                for r in T.serial(16):
                    acc[0] += ai_diag[2, i, r] * a_off[2, r, j]
                tmp_a[i, j] = acc[0]
            for i, j in T.Parallel(16, 16):
                acc = T.alloc_fragment((1,), accum_dtype)
                acc[0] = T.cast(0.0, accum_dtype)
                for r in T.serial(16):
                    acc[0] += tmp_a[i, r] * ai_diag[1, r, j]
                ai_off[2, i, j] = -acc[0]

            # ai_43 = -ai_44 @ A43 @ ai_33
            for i, j in T.Parallel(16, 16):
                acc = T.alloc_fragment((1,), accum_dtype)
                acc[0] = T.cast(0.0, accum_dtype)
                for r in T.serial(16):
                    acc[0] += ai_diag[3, i, r] * a_off[5, r, j]
                tmp_a[i, j] = acc[0]
            for i, j in T.Parallel(16, 16):
                acc = T.alloc_fragment((1,), accum_dtype)
                acc[0] = T.cast(0.0, accum_dtype)
                for r in T.serial(16):
                    acc[0] += tmp_a[i, r] * ai_diag[2, r, j]
                ai_off[5, i, j] = -acc[0]

            # ai_31 = -ai_33 @ (A31 @ ai_11 + A32 @ ai_21)
            for i, j in T.Parallel(16, 16):
                acc = T.alloc_fragment((1,), accum_dtype)
                acc[0] = T.cast(0.0, accum_dtype)
                for r in T.serial(16):
                    acc[0] += a_off[1, i, r] * ai_diag[0, r, j]  # A31 @ ai_11
                tmp_a[i, j] = acc[0]
            for i, j in T.Parallel(16, 16):
                acc = T.alloc_fragment((1,), accum_dtype)
                acc[0] = T.cast(0.0, accum_dtype)
                for r in T.serial(16):
                    acc[0] += a_off[2, i, r] * ai_off[0, r, j]   # A32 @ ai_21
                tmp_b[i, j] = acc[0]
            for i, j in T.Parallel(16, 16):
                tmp_a[i, j] = tmp_a[i, j] + tmp_b[i, j]
            for i, j in T.Parallel(16, 16):
                acc = T.alloc_fragment((1,), accum_dtype)
                acc[0] = T.cast(0.0, accum_dtype)
                for r in T.serial(16):
                    acc[0] += ai_diag[2, i, r] * tmp_a[r, j]
                ai_off[1, i, j] = -acc[0]

            # ai_42 = -ai_44 @ (A42 @ ai_22 + A43 @ ai_32)
            for i, j in T.Parallel(16, 16):
                acc = T.alloc_fragment((1,), accum_dtype)
                acc[0] = T.cast(0.0, accum_dtype)
                for r in T.serial(16):
                    acc[0] += a_off[4, i, r] * ai_diag[1, r, j]  # A42 @ ai_22
                tmp_a[i, j] = acc[0]
            for i, j in T.Parallel(16, 16):
                acc = T.alloc_fragment((1,), accum_dtype)
                acc[0] = T.cast(0.0, accum_dtype)
                for r in T.serial(16):
                    acc[0] += a_off[5, i, r] * ai_off[2, r, j]   # A43 @ ai_32
                tmp_b[i, j] = acc[0]
            for i, j in T.Parallel(16, 16):
                tmp_a[i, j] = tmp_a[i, j] + tmp_b[i, j]
            for i, j in T.Parallel(16, 16):
                acc = T.alloc_fragment((1,), accum_dtype)
                acc[0] = T.cast(0.0, accum_dtype)
                for r in T.serial(16):
                    acc[0] += ai_diag[3, i, r] * tmp_a[r, j]
                ai_off[4, i, j] = -acc[0]

            # ai_41 = -ai_44 @ (A41 @ ai_11 + A42 @ ai_21 + A43 @ ai_31)
            for i, j in T.Parallel(16, 16):
                acc = T.alloc_fragment((1,), accum_dtype)
                acc[0] = T.cast(0.0, accum_dtype)
                for r in T.serial(16):
                    acc[0] += a_off[3, i, r] * ai_diag[0, r, j]  # A41 @ ai_11
                tmp_a[i, j] = acc[0]
            for i, j in T.Parallel(16, 16):
                acc = T.alloc_fragment((1,), accum_dtype)
                acc[0] = T.cast(0.0, accum_dtype)
                for r in T.serial(16):
                    acc[0] += a_off[4, i, r] * ai_off[0, r, j]   # A42 @ ai_21
                tmp_b[i, j] = acc[0]
            for i, j in T.Parallel(16, 16):
                tmp_a[i, j] = tmp_a[i, j] + tmp_b[i, j]
            for i, j in T.Parallel(16, 16):
                acc = T.alloc_fragment((1,), accum_dtype)
                acc[0] = T.cast(0.0, accum_dtype)
                for r in T.serial(16):
                    acc[0] += a_off[5, i, r] * ai_off[1, r, j]   # A43 @ ai_31
                tmp_b[i, j] = acc[0]
            for i, j in T.Parallel(16, 16):
                tmp_a[i, j] = tmp_a[i, j] + tmp_b[i, j]
            for i, j in T.Parallel(16, 16):
                acc = T.alloc_fragment((1,), accum_dtype)
                acc[0] = T.cast(0.0, accum_dtype)
                for r in T.serial(16):
                    acc[0] += ai_diag[3, i, r] * tmp_a[r, j]
                ai_off[3, i, j] = -acc[0]

            # Write all 10 16x16 blocks back to a_inv (4 diagonal + 6
            # off-diagonal). Upper triangle stays zero because the chunk-A
            # contract stores only the lower-triangular layout.
            for blk, i, j in T.Parallel(4, 16, 16):
                row = base + blk * 16 + i
                col = blk * 16 + j
                if row < seq_len:
                    a_inv[row, v_head, col] = T.cast(ai_diag[blk, i, j], dtype)
            for slot, i, j in T.Parallel(6, 16, 16):
                row_block = T.if_then_else(
                    slot == 0, 1,
                    T.if_then_else(slot == 1, 2,
                    T.if_then_else(slot == 2, 2,
                    T.if_then_else(slot == 3, 3,
                    T.if_then_else(slot == 4, 3, 3)))))
                col_block = T.if_then_else(
                    slot == 0, 0,
                    T.if_then_else(slot == 1, 0,
                    T.if_then_else(slot == 2, 1,
                    T.if_then_else(slot == 3, 0,
                    T.if_then_else(slot == 4, 1, 2)))))
                row = base + row_block * 16 + i
                col = col_block * 16 + j
                if row < seq_len:
                    a_inv[row, v_head, col] = T.cast(ai_off[slot, i, j], dtype)

    return kernel


def _gdr_recompute_w_u_kernel():
    """Stage 5 — recompute chunk-wise w/u from k, v, beta, a_inv.

    Translation of `gdr_recompute_w_u_qwen35_kernel`. Two GEMMs per
    chunk: u = a_inv @ (v * beta), w = a_inv @ (k * beta * exp(g)).
    The TileLang kernel issues one full-shape GEMM per output since
    KEY_DIM=128 / VALUE_DIM=128 fit comfortably in registers.
    """
    dtype = "bfloat16"
    accum_dtype = "float32"
    g_dtype = "float32"
    KEY_DIM = QWEN35_GDR_KEY_DIM
    VALUE_DIM = QWEN35_GDR_VALUE_DIM
    BT = BLOCK_T

    @T.prim_func
    def kernel(
        k: T.Tensor((T.symbolic("seq_len"), T.symbolic("hv"), KEY_DIM), dtype),
        v: T.Tensor((T.symbolic("seq_len"), T.symbolic("hv"), VALUE_DIM), dtype),
        beta: T.Tensor((T.symbolic("seq_len"), T.symbolic("hv")), g_dtype),
        w: T.Tensor((T.symbolic("seq_len"), T.symbolic("hv"), KEY_DIM), dtype),
        u: T.Tensor((T.symbolic("seq_len"), T.symbolic("hv"), VALUE_DIM), dtype),
        a_inv: T.Tensor((T.symbolic("seq_len"), T.symbolic("hv"), BT), dtype),
        g_cumsum: T.Tensor((T.symbolic("seq_len"), T.symbolic("hv")), g_dtype),
        seq_len: T.int32,
        num_value_heads: T.int32,
    ):
        with T.Kernel(
            T.ceildiv(seq_len, BT), num_value_heads, threads=NUM_THREADS
        ) as (chunk_idx, v_head):
            ai_tile = T.alloc_shared((BT, BT), dtype)
            v_tile = T.alloc_shared((BT, VALUE_DIM), dtype)
            k_tile = T.alloc_shared((BT, KEY_DIM), dtype)
            beta_frag = T.alloc_fragment((BT,), accum_dtype)
            g_frag = T.alloc_fragment((BT,), accum_dtype)
            u_acc = T.alloc_fragment((BT, VALUE_DIM), accum_dtype)
            w_acc = T.alloc_fragment((BT, KEY_DIM), accum_dtype)

            base = chunk_idx * BT
            for t, j in T.Parallel(BT, BT):
                ai_tile[t, j] = T.if_then_else(
                    base + t < seq_len,
                    a_inv[base + t, v_head, j],
                    T.cast(0, dtype),
                )
            for t in T.Parallel(BT):
                in_range = base + t < seq_len
                beta_frag[t] = T.if_then_else(
                    in_range, beta[base + t, v_head], T.cast(0.0, accum_dtype)
                )
                g_frag[t] = T.if_then_else(
                    in_range,
                    T.exp(g_cumsum[base + t, v_head]),
                    T.cast(0.0, accum_dtype),
                )

            # u_block = a_inv @ (v * beta).
            for t, d in T.Parallel(BT, VALUE_DIM):
                v_tile[t, d] = T.if_then_else(
                    base + t < seq_len,
                    T.cast(T.cast(v[base + t, v_head, d], accum_dtype) * beta_frag[t], dtype),
                    T.cast(0, dtype),
                )
            T.clear(u_acc)
            T.gemm(ai_tile, v_tile, u_acc, policy=T.GemmWarpPolicy.FullRow)
            for t, d in T.Parallel(BT, VALUE_DIM):
                if base + t < seq_len:
                    u[base + t, v_head, d] = T.cast(u_acc[t, d], dtype)

            # w_block = a_inv @ (k * beta * exp(g_cumsum)).
            for t, d in T.Parallel(BT, KEY_DIM):
                k_tile[t, d] = T.if_then_else(
                    base + t < seq_len,
                    T.cast(
                        T.cast(k[base + t, v_head, d], accum_dtype)
                        * beta_frag[t]
                        * g_frag[t],
                        dtype,
                    ),
                    T.cast(0, dtype),
                )
            T.clear(w_acc)
            T.gemm(ai_tile, k_tile, w_acc, policy=T.GemmWarpPolicy.FullRow)
            for t, d in T.Parallel(BT, KEY_DIM):
                if base + t < seq_len:
                    w[base + t, v_head, d] = T.cast(w_acc[t, d], dtype)

    return kernel


def _gdr_chunk_state_kernel():
    """Stage 6 — chunk-wise recurrent state update.

    Translation of `gdr_chunk_state_qwen35_kernel`. Walks the chunks
    serially within a single thread block per (v_tile, v_head),
    maintains the running state h ∈ [KEY_DIM, BLOCK_V] in registers
    (split into h_lo / h_hi over the two KEY_BLOCK halves), writes
    per-chunk snapshots into chunk_state, accumulates into the
    decode-compatible final_state at [H, K, V].
    """
    dtype = "bfloat16"
    accum_dtype = "float32"
    g_dtype = "float32"
    KEY_DIM = QWEN35_GDR_KEY_DIM
    VALUE_DIM = QWEN35_GDR_VALUE_DIM
    BT = BLOCK_T
    BV = BLOCK_V
    KB = KEY_BLOCK

    @T.prim_func
    def kernel(
        k: T.Tensor((T.symbolic("seq_len"), T.symbolic("hv"), KEY_DIM), dtype),
        w: T.Tensor((T.symbolic("seq_len"), T.symbolic("hv"), KEY_DIM), dtype),
        u: T.Tensor((T.symbolic("seq_len"), T.symbolic("hv"), VALUE_DIM), dtype),
        g_cumsum: T.Tensor((T.symbolic("seq_len"), T.symbolic("hv")), g_dtype),
        initial_state: T.Tensor((T.symbolic("hv"), KEY_DIM, VALUE_DIM), accum_dtype),
        chunk_state: T.Tensor(
            (T.symbolic("num_chunks"), T.symbolic("hv"), KEY_DIM, VALUE_DIM),
            accum_dtype,
        ),
        v_new: T.Tensor((T.symbolic("seq_len"), T.symbolic("hv"), VALUE_DIM), dtype),
        final_state: T.Tensor((T.symbolic("hv"), KEY_DIM, VALUE_DIM), accum_dtype),
        seq_len: T.int32,
        num_value_heads: T.int32,
    ):
        # Grid: (KEY_DIM/BV value-tiles, num_value_heads).
        with T.Kernel(
            T.ceildiv(VALUE_DIM, BV), num_value_heads, threads=NUM_THREADS
        ) as (v_tile, v_head):
            h_lo = T.alloc_fragment((KB, BV), accum_dtype)
            h_hi = T.alloc_fragment((KB, BV), accum_dtype)
            w_lo = T.alloc_shared((BT, KB), dtype)
            w_hi = T.alloc_shared((BT, KB), dtype)
            k_lo = T.alloc_shared((KB, BT), dtype)
            k_hi = T.alloc_shared((KB, BT), dtype)
            u_tile = T.alloc_shared((BT, BV), dtype)
            v_new_tile = T.alloc_fragment((BT, BV), accum_dtype)
            v_new_bf = T.alloc_shared((BT, BV), dtype)
            g_frag = T.alloc_fragment((BT,), accum_dtype)

            v_off = v_tile * BV

            # Load initial_state into h_lo (rows 0..KB) and h_hi (rows KB..2KB).
            for r, c in T.Parallel(KB, BV):
                h_lo[r, c] = initial_state[v_head, r, v_off + c]
                h_hi[r, c] = initial_state[v_head, KB + r, v_off + c]

            num_chunks = T.ceildiv(seq_len, BT)
            for chunk_idx in T.serial(num_chunks):
                base = chunk_idx * BT

                # Snapshot h into chunk_state[chunk_idx, v_head, :, v_off..].
                for r, c in T.Parallel(KB, BV):
                    chunk_state[chunk_idx, v_head, r, v_off + c] = h_lo[r, c]
                    chunk_state[chunk_idx, v_head, KB + r, v_off + c] = h_hi[r, c]

                # Load w_lo, w_hi for this chunk.
                for t, c in T.Parallel(BT, KB):
                    w_lo[t, c] = T.if_then_else(
                        base + t < seq_len, w[base + t, v_head, c], T.cast(0, dtype)
                    )
                    w_hi[t, c] = T.if_then_else(
                        base + t < seq_len, w[base + t, v_head, KB + c], T.cast(0, dtype)
                    )
                # Load u (BLOCK_V column) and compute v_new = u - w @ h.
                for t, c in T.Parallel(BT, BV):
                    u_tile[t, c] = T.if_then_else(
                        base + t < seq_len,
                        u[base + t, v_head, v_off + c],
                        T.cast(0, dtype),
                    )

                T.clear(v_new_tile)
                # v_new += u
                for t, c in T.Parallel(BT, BV):
                    v_new_tile[t, c] = T.cast(u_tile[t, c], accum_dtype)

                # v_new -= w_lo @ h_lo + w_hi @ h_hi. Materialize h_lo/h_hi
                # into shared for the GEMM.
                h_lo_sh = T.alloc_shared((KB, BV), dtype)
                h_hi_sh = T.alloc_shared((KB, BV), dtype)
                for r, c in T.Parallel(KB, BV):
                    h_lo_sh[r, c] = T.cast(h_lo[r, c], dtype)
                    h_hi_sh[r, c] = T.cast(h_hi[r, c], dtype)
                # NOTE: TileLang's T.gemm can subtract via a manual
                # negation pass; here we emit two add-GEMMs into a tmp
                # buffer then subtract in T.Parallel.
                wh_acc = T.alloc_fragment((BT, BV), accum_dtype)
                T.clear(wh_acc)
                T.gemm(w_lo, h_lo_sh, wh_acc, policy=T.GemmWarpPolicy.FullRow)
                T.gemm(w_hi, h_hi_sh, wh_acc, policy=T.GemmWarpPolicy.FullRow)
                for t, c in T.Parallel(BT, BV):
                    v_new_tile[t, c] = v_new_tile[t, c] - wh_acc[t, c]

                # Write per-token v_new (pre-gating).
                for t, c in T.Parallel(BT, BV):
                    if base + t < seq_len:
                        v_new[base + t, v_head, v_off + c] = T.cast(v_new_tile[t, c], dtype)

                # Decay h by exp(g_last) and apply per-token gate.
                # g_last = g_cumsum[min((chunk_idx+1)*BT, seq_len) - 1, v_head]
                g_last_idx = T.if_then_else(
                    base + BT <= seq_len, base + BT - 1, seq_len - 1
                )
                g_last = g_cumsum[g_last_idx, v_head]
                decay = T.exp(g_last)
                for r, c in T.Parallel(KB, BV):
                    h_lo[r, c] = h_lo[r, c] * decay
                    h_hi[r, c] = h_hi[r, c] * decay

                # gate[t] = exp(g_last - g_cumsum[base+t, v_head]) * mask_t
                for t in T.Parallel(BT):
                    in_range = base + t < seq_len
                    g_frag[t] = T.if_then_else(in_range, g_cumsum[base + t, v_head], g_last)
                # v_new *= gate (in-place).
                for t, c in T.Parallel(BT, BV):
                    in_range = base + t < seq_len
                    g_v = T.if_then_else(in_range, T.exp(g_last - g_frag[t]), T.cast(0.0, accum_dtype))
                    v_new_tile[t, c] = v_new_tile[t, c] * g_v
                    v_new_bf[t, c] = T.cast(v_new_tile[t, c], dtype)

                # Load K^T (key-major layout) for h += k @ v_new.
                for r, t in T.Parallel(KB, BT):
                    k_lo[r, t] = T.if_then_else(
                        base + t < seq_len, k[base + t, v_head, r], T.cast(0, dtype)
                    )
                    k_hi[r, t] = T.if_then_else(
                        base + t < seq_len, k[base + t, v_head, KB + r], T.cast(0, dtype)
                    )

                kh_lo_acc = T.alloc_fragment((KB, BV), accum_dtype)
                kh_hi_acc = T.alloc_fragment((KB, BV), accum_dtype)
                T.clear(kh_lo_acc)
                T.clear(kh_hi_acc)
                T.gemm(k_lo, v_new_bf, kh_lo_acc, policy=T.GemmWarpPolicy.FullRow)
                T.gemm(k_hi, v_new_bf, kh_hi_acc, policy=T.GemmWarpPolicy.FullRow)
                for r, c in T.Parallel(KB, BV):
                    h_lo[r, c] = h_lo[r, c] + kh_lo_acc[r, c]
                    h_hi[r, c] = h_hi[r, c] + kh_hi_acc[r, c]

            # Write final state.
            for r, c in T.Parallel(KB, BV):
                final_state[v_head, r, v_off + c] = h_lo[r, c]
                final_state[v_head, KB + r, v_off + c] = h_hi[r, c]

    return kernel


def _gdr_chunk_o_kernel():
    """Stage 7 — chunk-wise output stage.

    Translation of `gdr_chunk_o_qwen35_kernel`. Three pieces per
    chunk: acc_o = q @ h, acc_a = q @ k^T (causal-masked), then
    out = (acc_o + acc_a @ v_new) * scale, all gated by exp(g_cumsum).
    """
    dtype = "bfloat16"
    accum_dtype = "float32"
    g_dtype = "float32"
    KEY_DIM = QWEN35_GDR_KEY_DIM
    VALUE_DIM = QWEN35_GDR_VALUE_DIM
    BT = BLOCK_T
    BV = BLOCK_V

    @T.prim_func
    def kernel(
        q: T.Tensor((T.symbolic("seq_len"), T.symbolic("hv"), KEY_DIM), dtype),
        k: T.Tensor((T.symbolic("seq_len"), T.symbolic("hv"), KEY_DIM), dtype),
        v_new: T.Tensor((T.symbolic("seq_len"), T.symbolic("hv"), VALUE_DIM), dtype),
        chunk_state: T.Tensor(
            (T.symbolic("num_chunks"), T.symbolic("hv"), KEY_DIM, VALUE_DIM),
            accum_dtype,
        ),
        g_cumsum: T.Tensor((T.symbolic("seq_len"), T.symbolic("hv")), g_dtype),
        output: T.Tensor((T.symbolic("seq_len"), T.symbolic("hv"), VALUE_DIM), dtype),
        seq_len: T.int32,
        num_value_heads: T.int32,
        scale: T.float32,
    ):
        # Grid: (VALUE_DIM/BV, num_chunks, num_value_heads).
        with T.Kernel(
            T.ceildiv(VALUE_DIM, BV),
            T.ceildiv(seq_len, BT),
            num_value_heads,
            threads=NUM_THREADS,
        ) as (v_tile, chunk_idx, v_head):
            q_tile = T.alloc_shared((BT, KEY_DIM), dtype)
            k_tile = T.alloc_shared((KEY_DIM, BT), dtype)
            h_tile = T.alloc_shared((KEY_DIM, BV), dtype)
            v_new_tile = T.alloc_shared((BT, BV), dtype)
            acc_o = T.alloc_fragment((BT, BV), accum_dtype)
            acc_a = T.alloc_fragment((BT, BT), accum_dtype)
            g_shared = T.alloc_shared((BT,), accum_dtype)

            base = chunk_idx * BT
            v_off = v_tile * BV

            for t, d in T.Parallel(BT, KEY_DIM):
                q_tile[t, d] = T.if_then_else(
                    base + t < seq_len, q[base + t, v_head, d], T.cast(0, dtype)
                )
            for d, t in T.Parallel(KEY_DIM, BT):
                k_tile[d, t] = T.if_then_else(
                    base + t < seq_len, k[base + t, v_head, d], T.cast(0, dtype)
                )
            for d, c in T.Parallel(KEY_DIM, BV):
                h_tile[d, c] = T.cast(chunk_state[chunk_idx, v_head, d, v_off + c], dtype)
            for t, c in T.Parallel(BT, BV):
                v_new_tile[t, c] = T.if_then_else(
                    base + t < seq_len,
                    v_new[base + t, v_head, v_off + c],
                    T.cast(0, dtype),
                )

            T.clear(acc_o)
            T.clear(acc_a)
            T.gemm(q_tile, h_tile, acc_o, policy=T.GemmWarpPolicy.FullRow)
            T.gemm(q_tile, k_tile, acc_a, policy=T.GemmWarpPolicy.FullRow)

            for t in T.Parallel(BT):
                g_shared[t] = T.if_then_else(
                    base + t < seq_len,
                    g_cumsum[base + t, v_head],
                    T.cast(0.0, accum_dtype),
                )

            # acc_o *= exp(g[t]) ; acc_a[i,j] *= exp(g[i] - g[j]) (causal).
            for t, c in T.Parallel(BT, BV):
                acc_o[t, c] = acc_o[t, c] * T.exp(g_shared[t])
            for i, j in T.Parallel(BT, BT):
                row_in = base + i < seq_len
                col_in = base + j < seq_len
                causal = (i >= j) and row_in and col_in
                acc_a[i, j] = T.if_then_else(
                    causal,
                    acc_a[i, j] * T.exp(g_shared[i] - g_shared[j]),
                    T.cast(0.0, accum_dtype),
                )

            # out = (acc_o + acc_a @ v_new) * scale
            acc_a_bf = T.alloc_shared((BT, BT), dtype)
            for i, j in T.Parallel(BT, BT):
                acc_a_bf[i, j] = T.cast(acc_a[i, j], dtype)
            T.gemm(acc_a_bf, v_new_tile, acc_o, policy=T.GemmWarpPolicy.FullRow)
            for t, c in T.Parallel(BT, BV):
                if base + t < seq_len:
                    output[base + t, v_head, v_off + c] = T.cast(acc_o[t, c] * scale, dtype)

    return kernel


# Public registry consumed by `gen_tilelang_aot.py --kernel-family gdr`.
KERNELS = {
    "gdr_chunk_prepare":   _gdr_chunk_prepare_kernel,
    "gdr_chunk_cumsum":    _gdr_chunk_local_cumsum_kernel,
    "gdr_chunk_a":         _gdr_chunk_scaled_dot_kkt_kernel,
    "gdr_chunk_solve":     _gdr_solve_tril_64_kernel,
    "gdr_chunk_recompute": _gdr_recompute_w_u_kernel,
    "gdr_chunk_state":     _gdr_chunk_state_kernel,
    "gdr_chunk_o":         _gdr_chunk_o_kernel,
}


def get_kernel(name: str):
    """Return the TileLang stage selected by `--kernel-key`.

    The GDR family does not parameterize on head config: Qwen3.5 fixes
    (num_value_heads, num_key_heads, KEY_DIM, VALUE_DIM, BLOCK_T) at the
    constants above and ships one specialization per stage.
    """
    factory = KERNELS.get(name)
    if factory is None:
        raise KeyError(
            f"unknown TileLang GDR kernel {name!r}; valid names: {sorted(KERNELS)}"
        )
    return factory()
