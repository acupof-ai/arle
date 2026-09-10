"""TileLang chunk-wise Gated Delta Rule (GDR) kernels for Qwen3.5 hybrid.

The canonical chunk-wise GDR pipeline with TileLang AOT stages. The
strict-lower triangular solve is kept as native CUDA C because TileLang 0.1.9
cannot lower that stage's mixed-index fragment layout on sm_89. The prepare
stage is also native CUDA C (csrc/recurrent/gdr_prefill_prepare.cu): the
TileLang lowering replicated the full q/k row into every thread's registers
(local-memory spill, 128x redundant loads, 66 ms/layer vs ~0.1 ms roofline).
Both stay in KERNELS below as lowering references only.

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
`csrc/misc/gdr_prefill_solve.cu` owns the solve symbol. The Rust FFI is
`cuda-kernels/src/ffi/recurrent.rs` (shim in `cuda-kernels/src/recurrent.rs`),
called from `infer-cuda/src/qwen35_attention.rs`. The old external AOT
directory is gone. GPU numerical validation compares this path via the
existing Qwen3.5 e2e tests and JSON baselines.
"""

import os

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


def _sm70_gemm_gate(dtype, accum_dtype):
    """SM-portable T.gemm operand dtype + a trace-time cast helper.

    bf16 tensor cores need sm_80+. On Volta/Turing (sm<80) Volta MMA
    (GemmMMASm70) only supports m16n16k4 with FP16 inputs, so feed every
    T.gemm operand fp16 (accum stays f32); on sm_80+ keep bf16 so the AST is
    byte-identical to the repo kernel. The cast routes bf16->f32->fp16: a
    direct bf16->fp16 cast lowers to an ambiguous user-defined conversion that
    nvcc rejects. When gemm_dtype == dtype the helper is a no-op passthrough.
    """
    sm_arch = int(os.environ.get("ARLE_TILELANG_CUDA_ARCH", "90"))
    gemm_dtype = "float16" if sm_arch < 80 else dtype

    def to_gemm(x):
        if gemm_dtype == dtype:
            return x
        return T.cast(T.cast(x, accum_dtype), gemm_dtype)

    return gemm_dtype, to_gemm


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



# Public registry consumed by `gen_tilelang_aot.py --kernel-family gdr`.
KERNELS = {
    "gdr_chunk_prepare":   _gdr_chunk_prepare_kernel,
    "gdr_chunk_solve":     _gdr_solve_tril_64_kernel,
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
