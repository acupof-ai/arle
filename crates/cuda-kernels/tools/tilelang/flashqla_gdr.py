"""FlashQLA chunked Gated Delta Rule forward kernels (Hopper, sm_90a only).

Adapted from QwenLM/FlashQLA @ 6ef4858b5446e05bd461d9658d877e548182dbcb (MIT):

  - flash_qla/ops/utils/cumsum.py                      -> fq_cumsum
  - flash_qla/ops/gated_delta_rule/chunk/hopper/kkt_solve.py -> fq_kkt
  - flash_qla/ops/gated_delta_rule/chunk/hopper/fused_fwd.py -> fq_fwd

ARLE deltas vs upstream (deletion-only where possible; the kernel bodies are
kept verbatim):

  - torch host wrappers replaced by the ARLE TileLang AOT surface
    (`get_kernel(key)` returning a raw `T.prim_func`; the `@tilelang.jit`
    decorators are dropped — `gen_tilelang_aot.py` compiles the prim_func).
  - varlen and intra-card-CP variants deleted: ARLE calls this per engine
    prefill chunk with batch=1 and chains the recurrent state via
    h0/ht, so `is_varlen=False`, `is_cp=False` always.
  - Shapes fixed to the Qwen3.6-35B GDN single-GPU shard: H=32 value heads,
    Hg=16 key heads, DK=DV=128, chunk_size=64, scale=128^-0.5 (baked, same
    place FLA applies it: P and O), block_DV=64 (grid = 2*H = 64 CTAs).
  - dtypes fixed: q/k/v/a bf16, g/beta fp32, h0/ht fp32 (the FLA-convention
    `[H, K, V]` V-contiguous state ARLE already standardizes on — the slot
    state pointer is passed as BOTH h0 and ht; the kernel reads h0 fully
    before writing ht for the same (bh, bv) slice, so in-place is safe).
  - fused fwd: use_initial_state=True (a zero state equals "no initial
    state"), store_final_state=True, store_h=False, store_o=True.

Upstream pins tilelang==0.1.8; ARLE builds with the pod-proven 0.1.11
(`feedback_pin_from_current_proven_env`). The Hopper feature surface used
here (T.gemm_v1, T.alloc_barrier, T.set_max_nreg, T.use_swizzle,
make_linear_layout) exists in both; the AOT build is the compatibility gate.
"""

import tilelang
import tilelang.language as T

# Fixed Qwen3.6-35B GDN shard (single GPU). TP shards would need a second
# (H, Hg) instantiation — gated off in Rust until baked.
FQ_H = 32          # value heads (g/beta/A/v/o are per-H)
FQ_HG = 16         # key heads (q/k are per-Hg; bhg = bh // (H // Hg))
FQ_DK = 128
FQ_DV = 128
FQ_CHUNK = 64
FQ_BLOCK_DV = 64   # grid = ceildiv(DV, block_DV) * H = 64 CTAs on 78 SMs
FQ_SCALE = FQ_DK ** -0.5

ACCUM_DTYPE = "float32"
QKVA_DTYPE = "bfloat16"
G_DTYPE = "float32"
B_DTYPE = "float32"
STATE_DTYPE = "float32"

PASS_CONFIGS = {
    # Upstream sets TL_ENABLE_FAST_MATH on kkt + fused fwd (exp2-heavy).
    "fq_cumsum": {},
    "fq_kkt": {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
    "fq_fwd": {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
}


def _fq_cumsum_kernel():
    """Chunk-local cumsum over the per-token log-decay g (pre-decay input).

    Upstream `tilelang_chunk_local_cumsum` non-varlen, reverse=False.
    """
    H = FQ_H
    chunk_size = FQ_CHUNK
    accum_dtype = ACCUM_DTYPE
    g_dtype = G_DTYPE

    data_batch_size = T.dynamic("data_batch_size")
    num_tokens = T.dynamic("num_tokens")
    block_S = chunk_size

    g_shape = (data_batch_size, num_tokens, H)

    @T.prim_func
    def fq_cumsum_kernel(
        g_raw: T.Tensor(g_shape, dtype=g_dtype),
        g_cumsum: T.Tensor(g_shape, dtype=g_dtype),
        num_chunks: T.int32,
    ):
        with T.Kernel(num_chunks, threads=128) as (bc,):
            bb = bc % data_batch_size
            chunk_idx = bc // data_batch_size

            left = chunk_idx * block_S
            right = left + block_S

            g_fragment = T.alloc_fragment((H, block_S), dtype=accum_dtype)
            gT_fragment = T.alloc_fragment((block_S, H), dtype=g_dtype)
            gT_shared = T.alloc_shared((block_S, H + 1), dtype=g_dtype)

            if right <= num_tokens:
                T.copy(g_raw[bb, left:right, 0:H], gT_fragment)
            else:
                for j, i in T.Parallel(block_S, H):
                    if left + j < num_tokens:
                        gT_fragment[j, i] = g_raw[bb, left + j, i]
                    else:
                        gT_fragment[j, i] = 0
            T.copy(gT_fragment, gT_shared[:, :H])

            for i, j in T.Parallel(H, block_S):
                g_fragment[i, j] = gT_shared[j, i]

            T.cumsum(g_fragment, dim=1, reverse=False)

            for i, j in T.Parallel(H, block_S):
                gT_shared[j, i] = g_fragment[i, j]

            T.copy(gT_shared[:, :H], gT_fragment)
            if right <= num_tokens:
                T.copy(gT_fragment, g_cumsum[bb, left:right, 0:H])
            else:
                for j, i in T.Parallel(block_S, H):
                    if left + j < num_tokens:
                        g_cumsum[bb, left + j, i] = gT_fragment[j, i]

    return fq_cumsum_kernel


def _fq_kkt_kernel():
    """A = (I + StrictLower(diag(beta) K K^T))^{-1}, per 64-chunk per H head.

    Upstream `tilelang_kkt_solve` non-varlen. 256 threads: 128 consumer
    (4-block inversion), 32 K-loader, 96 A-store, warp-specialized.
    """
    H = FQ_H
    Hg = FQ_HG
    DK = FQ_DK
    chunk_size = FQ_CHUNK
    accum_dtype = ACCUM_DTYPE
    qkva_dtype = QKVA_DTYPE
    b_dtype = B_DTYPE

    data_batch_size = T.dynamic("data_batch_size")
    num_tokens = T.dynamic("num_tokens")
    block_S = chunk_size

    k_shape = (data_batch_size, num_tokens, Hg, DK)
    a_shape = (data_batch_size, num_tokens, H, chunk_size)
    b_shape = (data_batch_size, num_tokens, H)

    @T.prim_func
    def fq_kkt_kernel(
        k: T.Tensor(k_shape, dtype=qkva_dtype),
        b: T.Tensor(b_shape, dtype=b_dtype),
        a: T.Tensor(a_shape, dtype=qkva_dtype),
        num_chunks: T.int32,
    ):
        with T.Kernel(num_chunks * H, threads=256) as (bch,):
            bc, bh = bch // H, bch % H
            bhg = bh // (H // Hg)

            bb = bc % data_batch_size
            chunk_idx = bc // data_batch_size

            left = chunk_idx * block_S
            right = left + block_S

            k_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            b_shared = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")
            a64_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)

            a16i_row = T.alloc_fragment((4, 16), dtype=accum_dtype)
            a16i_sum = T.alloc_fragment((4, 16), dtype=accum_dtype)

            a16i_shared = T.alloc_shared((4, 17, 16), dtype=accum_dtype)
            a16o_shared = T.alloc_shared((2, 17, 16), dtype=accum_dtype)
            a16o_fragment = T.alloc_fragment((2, 16, 16), dtype=accum_dtype)

            a32i_fragment = T.alloc_fragment((2, 32, 32), dtype=accum_dtype)
            a32i0_shared = T.alloc_shared((32, 32), dtype=accum_dtype)
            a32i1_shared = T.alloc_shared((32, 32), dtype=accum_dtype)
            a32o_shared = T.alloc_shared((32, 32), dtype=accum_dtype)
            a32o_fragment = T.alloc_fragment((32, 32), dtype=accum_dtype)

            a64_shared = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)

            T.annotate_layout(
                {
                    a16i_shared: tilelang.layout.make_linear_layout(a16i_shared),
                    a16o_shared: tilelang.layout.make_linear_layout(a16o_shared),
                }
            )

            k_is_ready = T.alloc_barrier(arrive_count=32)
            a_is_ready = T.alloc_barrier(arrive_count=128)

            tx = T.get_thread_binding()

            PRODUCER_NREG = 24
            CONSUMER_NREG = 64

            if tx < 128:
                T.set_max_nreg(CONSUMER_NREG, 1)

                # Load b
                if right <= num_tokens:
                    for j_s in T.Parallel(block_S):
                        b_shared[j_s] = b[bb, left + j_s, bh]
                else:
                    for j_s in T.Parallel(block_S):
                        if left + j_s < num_tokens:
                            b_shared[j_s] = b[bb, left + j_s, bh]
                        else:
                            b_shared[j_s] = 0

                T.barrier_wait(k_is_ready, 0)

                # A = K @ K^T
                T.gemm(
                    k_shared, k_shared, a64_fragment, transpose_B=True, clear_accum=True
                )

                # A = b * A
                for j_s, j_t in T.Parallel(block_S, block_S):
                    a64_fragment[j_s, j_t] *= b_shared[j_s]

                # A = I + StrictLower(A)
                for j_s, j_t in T.Parallel(block_S, block_S):
                    if j_s < j_t:
                        a64_fragment[j_s, j_t] = 0
                    elif j_s == j_t:
                        a64_fragment[j_s, j_t] = 1

                # Prepare inversion input
                for j_s, j_t in T.Parallel(block_S, block_S):
                    if j_s >= 32 and j_t < 32:
                        a32o_shared[j_s - 32, j_t] = -a64_fragment[j_s, j_t]
                    elif (j_s // 16) == (j_t // 16) + 1:
                        a16o_shared[j_s // 32, j_s % 16, j_t % 16] = -a64_fragment[
                            j_s, j_t
                        ]
                    elif (j_s // 16) == (j_t // 16):
                        a16i_shared[j_s // 16, j_s % 16, j_t % 16] = a64_fragment[
                            j_s, j_t
                        ]

                # Diagonal 4x16x16
                T.clear(a16i_row)
                for k_s in T.unroll(1, 16):
                    for j_s, k_t in T.Parallel(4, 16):
                        if k_t < k_s:
                            a16i_row[j_s, k_t] = a16i_shared[j_s, k_s, k_t]
                    T.clear(a16i_sum)
                    for k_r in T.unroll(k_s):
                        for j_s, k_t in T.Parallel(4, 16):
                            a16i_sum[j_s, k_t] -= (
                                a16i_shared[j_s, k_r, k_t] * a16i_row[j_s, k_r]
                            )
                    for j_s, k_t in T.Parallel(4, 16):
                        if k_t < k_s:
                            a16i_shared[j_s, k_s, k_t] = a16i_sum[j_s, k_t]

                # First level 2x16x16
                T.clear(a16o_fragment)
                for k_r in T.unroll(16):
                    for j_s, k_s, k_t in T.Parallel(2, 16, 16):
                        a16o_fragment[j_s, k_s, k_t] += (
                            a16i_shared[j_s * 2 + 1, k_s, k_r]
                            * a16o_shared[j_s, k_r, k_t]
                        )
                for j_s, k_s, k_t in T.Parallel(2, 16, 16):
                    a16o_shared[j_s, k_t, k_s] = a16o_fragment[j_s, k_s, k_t]
                T.clear(a16o_fragment)
                for k_r in T.unroll(16):
                    for j_s, k_s, k_t in T.Parallel(2, 16, 16):
                        a16o_fragment[j_s, k_s, k_t] += (
                            a16o_shared[j_s, k_r, k_s] * a16i_shared[j_s * 2, k_r, k_t]
                        )
                T.copy(a16o_fragment, a16o_shared[:, 0:16, 0:16])

                # Second level 1x32x32
                for j_s, k_s, k_t in T.Parallel(2, 32, 32):
                    if k_s < 16 and k_t >= 16:
                        a32i_fragment[j_s, k_s, k_t] = 0
                for j_s, k_s, k_t in T.Parallel(2, 32, 32):
                    if k_s >= 16 and k_t < 16:
                        a32i_fragment[j_s, k_s, k_t] = a16o_shared[j_s, k_s - 16, k_t]
                for j_s, k_s, k_t in T.Parallel(2, 32, 32):
                    if k_s // 16 == k_t // 16:
                        a32i_fragment[j_s, k_s, k_t] = a16i_shared[
                            j_s * 2 + k_s // 16, k_s % 16, k_t % 16
                        ]
                for j_s, k_s, k_t in T.Parallel(2, 32, 32):
                    if j_s == 0:
                        a32i0_shared[k_s, k_t] = a32i_fragment[j_s, k_s, k_t]
                    else:
                        a32i1_shared[k_s, k_t] = a32i_fragment[j_s, k_s, k_t]
                T.gemm(a32i1_shared, a32o_shared, a32o_fragment, clear_accum=True)
                T.copy(a32o_fragment, a32o_shared)
                T.gemm(a32o_shared, a32i0_shared, a32o_fragment, clear_accum=True)

                # Combine inversion output
                for j_s, k_s, k_t in T.Parallel(2, 32, 32):
                    a64_shared[j_s * 32 + k_s, j_s * 32 + k_t] = a32i_fragment[
                        j_s, k_s, k_t
                    ]
                for k_s, k_t in T.Parallel(32, 32):
                    a64_shared[32 + k_s, k_t] = a32o_fragment[k_s, k_t]
                for k_s, k_t in T.Parallel(32, 32):
                    a64_shared[k_s, 32 + k_t] = 0

                T.barrier_arrive(a_is_ready)

            else:
                T.set_max_nreg(PRODUCER_NREG, 0)

                if tx < 128 + 32:
                    # Load K
                    T.copy(k[bb, left:right, bhg, 0:DK], k_shared)

                    T.barrier_arrive(k_is_ready)

                elif tx < 128 + 64:
                    T.barrier_wait(a_is_ready, 0)

                    # Save A (unmasked)
                    if right <= num_tokens:
                        T.copy(a64_shared, a[bb, left:right, bh, 0:block_S])

                else:
                    T.barrier_wait(a_is_ready, 0)

                    # Save A (masked)
                    if right > num_tokens:
                        for j_s, j_t in T.Parallel(block_S, block_S):
                            if left + j_s < num_tokens:
                                a[bb, left + j_s, bh, j_t] = a64_shared[j_s, j_t]

    return fq_kkt_kernel


def _fq_fwd_kernel():
    """Fused chunk-state recurrence + output, warp-specialized (512 threads).

    Upstream `tilelang_fused_chunk_gdr_fwd` non-varlen non-CP with
    use_initial_state=True, store_final_state=True, store_h=False,
    store_o=True baked. h0 and ht may alias (in-place slot state): each CTA
    reads its h0 slice fully (initial T.copy) before writing the same ht
    slice after the loop.
    """
    H = FQ_H
    Hg = FQ_HG
    DK = FQ_DK
    DV = FQ_DV
    chunk_size = FQ_CHUNK
    scale = FQ_SCALE
    block_DV = FQ_BLOCK_DV
    accum_dtype = ACCUM_DTYPE
    qkva_dtype = QKVA_DTYPE
    g_dtype = G_DTYPE
    b_dtype = B_DTYPE
    h0_dtype = STATE_DTYPE
    ht_dtype = STATE_DTYPE
    o_dtype = QKVA_DTYPE

    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    raw_batch_size = T.dynamic("raw_batch_size")
    block_S = chunk_size

    q_shape = (batch_size, num_tokens, Hg, DK)
    k_shape = (batch_size, num_tokens, Hg, DK)
    v_shape = (batch_size, num_tokens, H, DV)
    o_shape = (batch_size, num_tokens, H, DV)
    a_shape = (batch_size, num_tokens, H, chunk_size)
    g_shape = (batch_size, num_tokens, H)
    b_shape = (batch_size, num_tokens, H)
    h0_shape = (batch_size, H, DK, DV)
    ht_shape = (raw_batch_size, H, DK, DV)

    @T.prim_func
    def fq_fwd_kernel(
        q: T.Tensor(q_shape, dtype=qkva_dtype),
        k: T.Tensor(k_shape, dtype=qkva_dtype),
        v: T.Tensor(v_shape, dtype=qkva_dtype),
        a: T.Tensor(a_shape, dtype=qkva_dtype),
        g: T.Tensor(g_shape, dtype=g_dtype),
        b: T.Tensor(b_shape, dtype=b_dtype),
        h0: T.Tensor(h0_shape, dtype=h0_dtype),
        o: T.Tensor(o_shape, dtype=o_dtype),
        ht: T.Tensor(ht_shape, dtype=ht_dtype),
    ):
        with T.Kernel(T.ceildiv(DV, block_DV) * batch_size * H, threads=512) as (
            bbhv,
        ):
            bbh, bv = bbhv // T.ceildiv(DV, block_DV), bbhv % T.ceildiv(DV, block_DV)
            bb, bh = bbh // H, bbh % H

            batch_idx = bb

            raw_batch_idx = bb

            num_iters = T.alloc_var("int32")
            num_unmasked_iters = T.alloc_var("int32")
            num_iters = T.ceildiv(num_tokens, block_S)
            num_unmasked_iters = num_tokens // block_S

            q_shared = T.alloc_shared((2, block_S, DK), dtype=qkva_dtype)
            k_shared = T.alloc_shared((2, block_S, DK), dtype=qkva_dtype)
            v_shared = T.alloc_shared((2, block_S, block_DV), dtype=qkva_dtype)
            a_shared = T.alloc_shared((2, block_S, block_S), dtype=qkva_dtype)
            g_shared = T.alloc_shared((2, block_S), dtype=accum_dtype, scope="shared")
            b_shared = T.alloc_shared((2, block_S), dtype=accum_dtype, scope="shared")

            o_shared = T.alloc_shared((block_S, block_DV), dtype=o_dtype)
            h_shared = T.alloc_shared((DK, block_DV), dtype=qkva_dtype)
            vd_shared = T.alloc_shared((block_S, block_DV), dtype=qkva_dtype)
            vn_shared = T.alloc_shared((block_S, block_DV), dtype=qkva_dtype)
            p_shared = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)
            g_exp_shared = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")
            g_rev_exp_shared = T.alloc_shared(
                (block_S), dtype=accum_dtype, scope="shared"
            )

            h_fragment = T.alloc_fragment((DK, block_DV), dtype=accum_dtype)
            o_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            v_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            u_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            p_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            a_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            g_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            g_last_local = T.alloc_local((1), dtype=accum_dtype)

            data_is_ready = T.alloc_barrier(arrive_count=[96] * 2)
            data_is_free = T.alloc_barrier(arrive_count=[384] * 2)

            bar_o = T.alloc_barrier(arrive_count=128)
            bar_0 = T.alloc_barrier(arrive_count=416)
            bar_1 = T.alloc_barrier(arrive_count=256)
            _bar_2 = T.alloc_barrier(arrive_count=128)
            bar_3 = T.alloc_barrier(arrive_count=128)
            bar_4 = T.alloc_barrier(arrive_count=128)
            bar_5 = T.alloc_barrier(arrive_count=416)

            T.use_swizzle(10)

            tx = T.get_thread_binding()

            PRODUCER_NREG = 32
            CONSUMER_V_NREG = 128
            CONSUMER_S_NREG = 160
            CONSUMER_O_NREG = 128

            if tx < 128:
                T.set_max_nreg(CONSUMER_S_NREG, 1)

                # Initialize S (zero slot state == "no initial state")
                T.copy(
                    h0[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV],
                    h_fragment,
                )

                # Main Loop
                for i_s in T.serial(num_iters):
                    # [STAGE 0]
                    T.barrier_wait(data_is_ready[i_s % 2], (i_s // 2 + 0) % 2)
                    T.barrier_arrive(bar_0)

                    # [STAGE 0] 0
                    T.barrier_wait(bar_0, i_s % 2)
                    # S4[S] S
                    T.copy(h_fragment, h_shared)
                    T.barrier_arrive(bar_1)

                    # [STAGE 0] 2, 3, 4
                    T.barrier_wait(bar_1, i_s % 2)
                    # S = g_last * S
                    g_last_local[0] = g_exp_shared[block_S - 1]
                    for j_k, j_v in T.Parallel(DK, block_DV):
                        h_fragment[j_k, j_v] *= g_last_local[0]
                    T.barrier_arrive(bar_5)

                    # [STAGE 0] 5
                    T.barrier_wait(bar_5, i_s % 2)
                    # S += K^T @ V'
                    T.gemm(
                        k_shared[i_s % 2, :, :],
                        vn_shared,
                        h_fragment,
                        transpose_A=True,
                        clear_accum=False,
                    )

                    T.barrier_arrive(data_is_free[i_s % 2])

                # Store final S
                T.copy(
                    h_fragment,
                    ht[raw_batch_idx, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV],
                )

            elif tx < 256:
                T.set_max_nreg(CONSUMER_V_NREG, 1)

                # Main Loop
                for i_s in T.serial(num_iters):
                    # [STAGE 0]
                    T.barrier_wait(data_is_ready[i_s % 2], (i_s // 2 + 0) % 2)
                    T.barrier_arrive(bar_0)

                    # [STAGE 0] 0
                    T.barrier_wait(bar_0, i_s % 2)
                    # Precompute g, g_last/g
                    for j_s in T.Parallel(block_S):
                        g_exp_shared[j_s] = T.exp2(g_shared[i_s % 2, j_s] * 1.442695)
                    for j_s in T.Parallel(block_S):
                        g_rev_exp_shared[j_s] = T.if_then_else(
                            i_s * block_S + j_s < num_tokens,
                            T.exp2(
                                (
                                    g_shared[i_s % 2, block_S - 1]
                                    - g_shared[i_s % 2, j_s]
                                )
                                * 1.442695
                            ),
                            0.0,
                        )
                    T.barrier_arrive(bar_1)

                    # [STAGE 0] 1
                    T.barrier_wait(bar_1, i_s % 2)
                    # U = K @ S
                    T.gemm(
                        k_shared[i_s % 2, :, :], h_shared, u_fragment, clear_accum=True
                    )

                    # [STAGE 0] 2
                    # W = V - g * U
                    for j_s, j_v in T.Parallel(block_S, block_DV):
                        u_fragment[j_s, j_v] *= -g_exp_shared[j_s]
                    for j_s, j_v in T.Parallel(block_S, block_DV):
                        u_fragment[j_s, j_v] += v_shared[i_s % 2, j_s, j_v]
                    # S2[V] W
                    for j_s, j_v in T.Parallel(block_S, block_DV):
                        v_shared[i_s % 2, j_s, j_v] = u_fragment[j_s, j_v]

                    # [STAGE 0] 3
                    T.barrier_wait(bar_3, i_s % 2)
                    # Vd = Ag @ W
                    T.gemm(
                        a_shared[i_s % 2, :, :],
                        v_shared[i_s % 2, :, :],
                        v_fragment,
                        clear_accum=True,
                    )
                    # S2[2] Vd
                    T.copy(v_fragment, vd_shared)
                    T.barrier_arrive(bar_4)

                    # [STAGE 0] 4
                    # V' = g_last/g Vd
                    for j_s, j_v in T.Parallel(block_S, block_DV):
                        v_fragment[j_s, j_v] *= g_rev_exp_shared[j_s]
                    # S2[1] V'
                    T.copy(v_fragment, vn_shared)
                    T.barrier_arrive(bar_5)

                    T.barrier_wait(bar_5, i_s % 2)

                    T.barrier_arrive(data_is_free[i_s % 2])

            elif tx < 384:
                T.set_max_nreg(CONSUMER_O_NREG, 1)

                # Main Loop
                for i_s in T.serial(num_iters):
                    # [STAGE 0]
                    T.barrier_wait(data_is_ready[i_s % 2], (i_s // 2 + 0) % 2)
                    T.barrier_arrive(bar_0)

                    # [STAGE 0] 0
                    T.barrier_wait(bar_0, i_s % 2)
                    # P = Q K^T
                    T.gemm(
                        q_shared[i_s % 2, :, :],
                        k_shared[i_s % 2, :, :],
                        p_fragment,
                        transpose_B=True,
                        clear_accum=True,
                    )

                    # [STAGE 0] 1
                    # G = Lower(diag(g) @ I @ diag(1/g))
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        g_fragment[j_s, j_t] = (
                            g_shared[i_s % 2, j_s] - g_shared[i_s % 2, j_t]
                        )
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        if j_s >= j_t:
                            g_fragment[j_s, j_t] = T.exp2(
                                g_fragment[j_s, j_t] * 1.442695
                            )
                        else:
                            g_fragment[j_s, j_t] = 0
                    # Ag = G * Ar * b
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        a_fragment[j_s, j_t] = a_shared[i_s % 2, j_s, j_t]
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        a_fragment[j_s, j_t] *= g_fragment[j_s, j_t]
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        a_fragment[j_s, j_t] *= b_shared[i_s % 2, j_t]
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        a_shared[i_s % 2, j_s, j_t] = a_fragment[j_s, j_t]

                    # [STAGE 0] 2
                    T.barrier_wait(bar_1, i_s % 2)
                    # O = Q @ S
                    T.gemm(
                        q_shared[i_s % 2, :, :], h_shared, o_fragment, clear_accum=True
                    )

                    # [STAGE 0] 3
                    # Pg = s * G * P
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        p_fragment[j_s, j_t] *= scale * g_fragment[j_s, j_t]
                    # S1[1] Pg
                    T.copy(p_fragment, p_shared)
                    T.barrier_arrive(bar_3)
                    # O = s * g * O
                    for j_s, j_k in T.Parallel(block_S, DK):
                        o_fragment[j_s, j_k] *= scale * g_exp_shared[j_s]

                    # [STAGE 0] 4
                    T.barrier_wait(bar_4, i_s % 2)
                    # O += Pg @ Vd
                    T.gemm(p_shared, vd_shared, o_fragment, clear_accum=False)
                    T.barrier_arrive(bar_5)

                    # [STAGE 0] 5
                    T.barrier_wait(bar_5, i_s % 2)
                    # S2[S] O
                    T.copy(o_fragment, o_shared)

                    T.barrier_arrive(data_is_free[i_s % 2])

                T.barrier_arrive(bar_o)

            else:
                T.set_max_nreg(PRODUCER_NREG, 0)

                if tx < 384 + 32:
                    for i_s in T.serial(num_iters):
                        T.barrier_wait(data_is_free[i_s % 2], (i_s // 2 + 1) % 2)
                        left = i_s * block_S
                        right = left + block_S

                        # Load Q
                        T.copy(
                            q[batch_idx, left:right, bhg, 0:DK],
                            q_shared[i_s % 2, :, :],
                        )
                        # Load K
                        T.copy(
                            k[batch_idx, left:right, bhg, 0:DK],
                            k_shared[i_s % 2, :, :],
                        )

                        T.barrier_arrive(data_is_ready[i_s % 2])

                elif tx < 384 + 64:
                    for i_s in T.serial(num_iters):
                        T.barrier_wait(data_is_free[i_s % 2], (i_s // 2 + 1) % 2)
                        left = i_s * block_S
                        right = left + block_S

                        # Load V
                        T.copy(
                            v[
                                batch_idx,
                                left:right,
                                bh,
                                bv * block_DV : (bv + 1) * block_DV,
                            ],
                            v_shared[i_s % 2, :, :],
                        )
                        # Load beta
                        if right <= num_tokens:
                            for j_s in T.Parallel(block_S):
                                b_shared[i_s % 2, j_s] = b[batch_idx, left + j_s, bh]
                        else:
                            for j_s in T.Parallel(block_S):
                                if left + j_s < num_tokens:
                                    b_shared[i_s % 2, j_s] = b[
                                        batch_idx, left + j_s, bh
                                    ]
                                else:
                                    b_shared[i_s % 2, j_s] = 0

                        T.barrier_arrive(data_is_ready[i_s % 2])

                elif tx < 384 + 96:
                    for i_s in T.serial(num_iters):
                        T.barrier_wait(data_is_free[i_s % 2], (i_s // 2 + 1) % 2)
                        left = i_s * block_S
                        right = left + block_S

                        # Load A
                        T.copy(
                            a[batch_idx, left:right, bh, 0:block_S],
                            a_shared[i_s % 2, :, :],
                        )
                        # Load gamma
                        if right <= num_tokens:
                            for j_s in T.Parallel(block_S):
                                g_shared[i_s % 2, j_s] = g[batch_idx, left + j_s, bh]
                        else:
                            for j_s in T.Parallel(block_S):
                                if left + j_s < num_tokens:
                                    g_shared[i_s % 2, j_s] = g[
                                        batch_idx, left + j_s, bh
                                    ]
                                else:
                                    g_shared[i_s % 2, j_s] = g[
                                        batch_idx, num_tokens - 1, bh
                                    ]

                        T.barrier_arrive(data_is_ready[i_s % 2])

                else:
                    for i_s in T.serial(num_unmasked_iters):
                        right = i_s * block_S
                        left = right - block_S

                        T.barrier_arrive(bar_0)

                        T.barrier_wait(bar_0, i_s % 2)
                        # Store O
                        if i_s > 0:
                            T.copy(
                                o_shared,
                                o[
                                    batch_idx,
                                    left:right,
                                    bh,
                                    bv * block_DV : (bv + 1) * block_DV,
                                ],
                            )
                        T.barrier_arrive(bar_5)

                        T.barrier_wait(bar_1, i_s % 2)

                    if num_unmasked_iters < num_iters:
                        seq_split_idx = num_unmasked_iters * block_S

                        T.barrier_arrive(bar_0)

                        T.barrier_wait(bar_0, num_unmasked_iters % 2)
                        # Store O
                        if num_unmasked_iters > 0:
                            T.copy(
                                o_shared,
                                o[
                                    batch_idx,
                                    seq_split_idx - block_S : seq_split_idx,
                                    bh,
                                    bv * block_DV : (bv + 1) * block_DV,
                                ],
                            )
                        T.barrier_arrive(bar_5)

                        T.barrier_wait(bar_1, num_unmasked_iters % 2)

                    seq_split_idx = (num_iters - 1) * block_S

                    # Store O
                    T.barrier_wait(bar_o, 0)
                    for j_s, j_v in T.Parallel(block_S, block_DV):
                        with T.If(seq_split_idx + j_s < num_tokens):
                            with T.Then():
                                o[
                                    batch_idx,
                                    seq_split_idx + j_s,
                                    bh,
                                    bv * block_DV + j_v,
                                ] = o_shared[j_s, j_v]

    return fq_fwd_kernel


_KERNELS = {
    "fq_cumsum": _fq_cumsum_kernel,
    "fq_kkt": _fq_kkt_kernel,
    "fq_fwd": _fq_fwd_kernel,
}


def get_kernel(name: str):
    if name not in _KERNELS:
        raise KeyError(f"unknown flashqla_gdr kernel key {name!r}")
    return _KERNELS[name]()


def get_pass_configs(name: str):
    return PASS_CONFIGS.get(name, {})
