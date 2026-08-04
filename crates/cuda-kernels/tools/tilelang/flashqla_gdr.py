"""FlashQLA chunked Gated Delta Rule kernels (Hopper, sm_90a only).

Adapted from QwenLM/FlashQLA @ 6ef4858b5446e05bd461d9658d877e548182dbcb (MIT):

  - flash_qla/ops/utils/cumsum.py                      -> fq_cumsum
  - flash_qla/ops/gated_delta_rule/chunk/hopper/kkt_solve.py -> fq_kkt
  - flash_qla/ops/gated_delta_rule/chunk/hopper/fused_fwd.py -> fq_fwd
  - flash_qla/ops/gated_delta_rule/chunk/hopper/prepare_h.py -> fq_prepare_h
  - flash_qla/ops/gated_delta_rule/chunk/hopper/fused_bwd.py -> fq_bwd

ARLE deltas vs upstream (deletion-only where possible; the kernel bodies are
kept verbatim):

  - torch host wrappers replaced by the ARLE TileLang AOT surface
    (`get_kernel(key)` returning a raw `T.prim_func`; the `@tilelang.jit`
    decorators are dropped — `gen_tilelang_aot.py` compiles the prim_func).
  - varlen and intra-card-CP variants deleted: ARLE calls this per engine
    prefill chunk with batch=1 and chains the recurrent state via
    h0/ht, so `is_varlen=False`, `is_cp=False` always.
  - (H, Hg) parameterized per AOT instantiation (kernels.toml rows);
    DK=DV=128, chunk_size=64, scale=128^-0.5 (baked, same place FLA applies
    it: P and O), block_DV=64 (fq_fwd grid = 2*H CTAs).
  - dtypes fixed: q/k/v/a bf16, g/beta fp32, h0/ht fp32 (the FLA-convention
    `[H, K, V]` V-contiguous state ARLE already standardizes on — the slot
    state pointer is passed as BOTH h0 and ht; the kernel reads h0 fully
    before writing ht for the same (bh, bv) slice, so in-place is safe).
  - fused fwd: use_initial_state=True (a zero state equals "no initial
    state"), store_final_state=True, store_h=False, store_o=True.
  - prepare_h: use_initial_state=True, store_h=True, store_final_state=False
    (the fwd already produced ht). is_cp=False makes upstream's `calc_mt`
    statically false, so the M/Z correction buffers and the `mt` output —
    which would only ever be written as zeros — are deleted with it.
  - fused bwd: use_dht=True (a chunked/CP carry does feed a final-state
    gradient back in).
  - `T.tma_copy(src, dst, barrier=B)` -> `T.copy(src, dst)`: TMA lowering is
    off for the AOT wrapper, and the ordering the barrier= form buys is
    already implied — the issuing warp only arrives at B after a plain copy
    returns.

Upstream pins tilelang==0.1.8; ARLE builds with the pod-proven 0.1.11
(`feedback_pin_from_current_proven_env`). The Hopper feature surface used
here (T.gemm_v1, T.alloc_barrier, T.set_max_nreg, T.use_swizzle,
make_linear_layout) exists in both; the AOT build is the compatibility gate.
"""

import tilelang
import tilelang.language as T

# (H, Hg) is an AOT instantiation parameter (one kernels.toml row triple per
# geometry); DK/DV/chunk stay fixed.
FQ_DK = 128
FQ_DV = 128
FQ_CHUNK = 64
FQ_BLOCK_DV = 64   # fq_fwd grid = ceildiv(DV, block_DV) * H
FQ_SCALE = FQ_DK ** -0.5

ACCUM_DTYPE = "float32"
QKVA_DTYPE = "bfloat16"
G_DTYPE = "float32"
B_DTYPE = "float32"
STATE_DTYPE = "float32"

PASS_CONFIGS = {
    # Upstream sets TL_ENABLE_FAST_MATH on kkt + fused fwd (exp2-heavy).
    # TMA lowering stays off: it adds host-built descriptor params the AOT C
    # wrapper cannot construct (tilelang 0.1.11 surfaces them as `*_desc`).
    "fq_cumsum": {},
    "fq_kkt": {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
    "fq_fwd": {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
    "fq_prepare_h": {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
    "fq_bwd": {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        # Upstream: the warp-specialized shared-buffer reuse trips the checker.
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
}


def _fq_cumsum_kernel(h, hg):
    """Chunk-local cumsum over the per-token log-decay g (pre-decay input).

    Upstream `tilelang_chunk_local_cumsum` non-varlen, reverse=False.
    """
    H = h
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


def _fq_kkt_kernel(h, hg):
    """A = (I + StrictLower(diag(beta) K K^T))^{-1}, per 64-chunk per H head.

    Upstream `tilelang_kkt_solve` non-varlen. 256 threads: 128 consumer
    (4-block inversion), 32 K-loader, 96 A-store, warp-specialized.
    """
    H = h
    Hg = hg
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


def _fq_fwd_kernel(h, hg):
    """Fused chunk-state recurrence + output, warp-specialized (512 threads).

    Upstream `tilelang_fused_chunk_gdr_fwd` non-varlen non-CP with
    use_initial_state=True, store_final_state=True, store_h=False,
    store_o=True baked. h0 and ht may alias (in-place slot state): each CTA
    reads its h0 slice fully (initial T.copy) before writing the same ht
    slice after the loop.
    """
    H = h
    Hg = hg
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
            bhg = bh // (H // Hg)

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


def _fq_prepare_h_kernel(h, hg):
    """Per-chunk incoming states h[b, c, H, DK, DV], warp-specialized (512 threads).

    Upstream `tilelang_prepare_h` non-varlen non-CP with use_initial_state=True,
    store_h=True, store_final_state=False. Recomputed in the backward so h stays
    transient per layer instead of resident on the tape.
    """
    H = h
    Hg = hg
    DK = FQ_DK
    DV = FQ_DV
    chunk_size = FQ_CHUNK
    num_stages = 2
    accum_dtype = ACCUM_DTYPE
    qkva_dtype = QKVA_DTYPE
    g_dtype = G_DTYPE
    b_dtype = B_DTYPE
    h0_dtype = STATE_DTYPE
    h_dtype = QKVA_DTYPE

    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    num_chunks = T.dynamic("num_chunks")
    block_S = chunk_size

    k_shape = (batch_size, num_tokens, Hg, DK)
    v_shape = (batch_size, num_tokens, H, DV)
    a_shape = (batch_size, num_tokens, H, chunk_size)
    g_shape = (batch_size, num_tokens, H)
    b_shape = (batch_size, num_tokens, H)
    h_shape = (batch_size, num_chunks, H, DK, DV)
    h0_shape = (batch_size, H, DK, DV)

    @T.prim_func
    def fq_prepare_h_kernel(
        k: T.Tensor(k_shape, dtype=qkva_dtype),
        v: T.Tensor(v_shape, dtype=qkva_dtype),
        a: T.Tensor(a_shape, dtype=qkva_dtype),
        g: T.Tensor(g_shape, dtype=g_dtype),
        b: T.Tensor(b_shape, dtype=b_dtype),
        h0: T.Tensor(h0_shape, dtype=h0_dtype),
        h: T.Tensor(h_shape, dtype=h_dtype),
    ):
        with T.Kernel(batch_size * H, threads=512) as (bbh,):
            bb, bh = bbh // H, bbh % H
            bhg = bh // (H // Hg)

            batch_idx = bb
            seq_start_idx = 0
            seq_end_idx = num_tokens
            chunk_start_idx = 0

            num_iters = T.alloc_var("int32")
            num_iters = T.ceildiv(seq_end_idx - seq_start_idx, block_S)

            k_shared = T.alloc_shared((num_stages, block_S, DK), dtype=qkva_dtype)
            v_shared = T.alloc_shared((num_stages, block_S, DV), dtype=qkva_dtype)
            a_shared = T.alloc_shared((num_stages, block_S, block_S), dtype=qkva_dtype)
            g_shared = T.alloc_shared(
                (num_stages, block_S), dtype=accum_dtype, scope="shared"
            )
            b_shared = T.alloc_shared(
                (num_stages, block_S), dtype=accum_dtype, scope="shared"
            )
            h_shared = T.alloc_shared((DK, DV), dtype=qkva_dtype)
            x_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            y_shared = T.alloc_shared((block_S, DV), dtype=qkva_dtype)
            g_rev_exp_shared = T.alloc_shared(
                (block_S), dtype=accum_dtype, scope="shared"
            )

            h_fragment = T.alloc_fragment((DK, DV), dtype=accum_dtype)
            x_fragment = T.alloc_fragment((block_S, DK), dtype=accum_dtype)
            y_fragment = T.alloc_fragment((block_S, DV), dtype=accum_dtype)
            g_last_local_S = T.alloc_local((1), dtype=accum_dtype)
            g_last_local_Y = T.alloc_local((1), dtype=accum_dtype)

            data_is_ready = T.alloc_barrier(arrive_count=[96] * num_stages)
            data_is_free = T.alloc_barrier(arrive_count=[384] * num_stages)

            bar_0 = T.alloc_barrier(arrive_count=416)
            bar_1 = T.alloc_barrier(arrive_count=256)
            bar_2 = T.alloc_barrier(arrive_count=384)

            T.use_swizzle(10)

            tx = T.get_thread_binding()

            PRODUCER_NREG = 24
            CONSUMER_S_NREG = 168
            CONSUMER_X_NREG = 160
            CONSUMER_Y_NREG = 160

            if tx < 128:
                T.set_max_nreg(CONSUMER_S_NREG, 1)

                # Initialize S
                T.copy(h0[bb, bh, 0:DK, 0:DV], h_fragment)

                # Main Loop
                for i_s in T.serial(num_iters):
                    # [STAGE = i_s % num_stages]
                    T.barrier_wait(
                        data_is_ready[i_s % num_stages], (i_s // num_stages + 0) % 2
                    )
                    T.barrier_arrive(bar_0)

                    # [STAGE = i_s % num_stages] 0
                    T.barrier_wait(bar_0, i_s % 2)
                    # S4[1] S
                    T.copy(h_fragment, h_shared)
                    T.barrier_arrive(bar_1)

                    # [STAGE = i_s % num_stages] 1
                    T.barrier_wait(bar_1, i_s % 2)
                    # S = g_last * S
                    g_last_local_S[0] = T.exp2(
                        g_shared[i_s % num_stages, block_S - 1] * 1.442695
                    )
                    for j_k, j_v in T.Parallel(DK, DV):
                        h_fragment[j_k, j_v] *= g_last_local_S[0]
                    T.barrier_arrive(bar_2)

                    # [STAGE = i_s % num_stages] 2
                    T.barrier_wait(bar_2, i_s % 2)
                    # S += X^T @ Y
                    T.gemm(
                        x_shared,
                        y_shared,
                        h_fragment,
                        transpose_A=True,
                        clear_accum=False,
                    )

                    T.barrier_arrive(data_is_free[i_s % num_stages])

            elif tx < 256:
                T.set_max_nreg(CONSUMER_X_NREG, 1)

                # Main Loop
                for i_s in T.serial(num_iters):
                    # [STAGE = i_s % num_stages]
                    T.barrier_wait(
                        data_is_ready[i_s % num_stages], (i_s // num_stages + 0) % 2
                    )
                    T.barrier_arrive(bar_0)

                    # [STAGE = i_s % num_stages] 0
                    T.barrier_wait(bar_0, i_s % 2)
                    # X = A^T @ K
                    T.gemm(
                        a_shared[i_s % num_stages, :, :],
                        k_shared[i_s % num_stages, :, :],
                        x_fragment,
                        transpose_A=True,
                        clear_accum=True,
                    )

                    # [STAGE = i_s % num_stages] 1
                    # X = - b * X
                    for j_s, j_k in T.Parallel(block_S, DK):
                        x_fragment[j_s, j_k] *= -b_shared[i_s % num_stages, j_s]
                    # S2[1] X
                    T.copy(x_fragment, x_shared)
                    T.barrier_arrive(bar_2)

                    T.barrier_arrive(data_is_free[i_s % num_stages])

            elif tx < 384:
                T.set_max_nreg(CONSUMER_Y_NREG, 1)

                # Main Loop
                for i_s in T.serial(num_iters):
                    # [STAGE = i_s % num_stages]
                    T.barrier_wait(
                        data_is_ready[i_s % num_stages], (i_s // num_stages + 0) % 2
                    )
                    T.barrier_arrive(bar_0)

                    # [STAGE = i_s % num_stages] 0
                    T.barrier_wait(bar_0, i_s % 2)
                    # Precompute g_last/g
                    g_last_local_Y[0] = g_shared[i_s % num_stages, block_S - 1]
                    for j_s in T.Parallel(block_S):
                        g_rev_exp_shared[j_s] = T.exp2(
                            (g_last_local_Y[0] - g_shared[i_s % num_stages, j_s])
                            * 1.442695
                        )
                    g_last_local_Y[0] = T.exp2(g_last_local_Y[0] * 1.442695)
                    T.barrier_arrive(bar_1)

                    # [STAGE = i_s % num_stages] 1
                    T.barrier_wait(bar_1, i_s % 2)
                    # U = K @ S
                    T.gemm(
                        k_shared[i_s % num_stages, :, :],
                        h_shared,
                        y_fragment,
                        clear_accum=True,
                    )
                    # Y = g_last * U - g_last/g * V
                    for j_s, j_v in T.Parallel(block_S, DV):
                        y_fragment[j_s, j_v] *= g_last_local_Y[0]
                    for j_s, j_v in T.Parallel(block_S, DV):
                        y_fragment[j_s, j_v] -= (
                            v_shared[i_s % num_stages, j_s, j_v] * g_rev_exp_shared[j_s]
                        )
                    # S2[2] Y
                    T.copy(y_fragment, y_shared)
                    T.barrier_arrive(bar_2)

                    T.barrier_arrive(data_is_free[i_s % num_stages])

            else:
                T.set_max_nreg(PRODUCER_NREG, 0)

                if tx < 384 + 32:
                    for i_s in T.serial(num_iters):
                        T.barrier_wait(
                            data_is_free[i_s % num_stages], (i_s // num_stages + 1) % 2
                        )
                        left = seq_start_idx + i_s * block_S
                        right = left + block_S

                        # Load K
                        if right <= seq_end_idx:
                            T.copy(
                                k[batch_idx, left:right, bhg, 0:DK],
                                k_shared[i_s % num_stages, :, :],
                            )
                        else:
                            for j_s, j_k in T.Parallel(block_S, DK):
                                if left + j_s < seq_end_idx:
                                    k_shared[i_s % num_stages, j_s, j_k] = k[batch_idx, left + j_s, bhg, j_k]
                                else:
                                    k_shared[i_s % num_stages, j_s, j_k] = 0

                        T.barrier_arrive(data_is_ready[i_s % num_stages])

                elif tx < 384 + 64:
                    for i_s in T.serial(num_iters):
                        T.barrier_wait(
                            data_is_free[i_s % num_stages], (i_s // num_stages + 1) % 2
                        )
                        left = seq_start_idx + i_s * block_S
                        right = left + block_S

                        # Load V
                        if right <= seq_end_idx:
                            T.copy(
                                v[batch_idx, left:right, bh, 0:DV],
                                v_shared[i_s % num_stages, :, :],
                            )
                        else:
                            for j_s, j_v in T.Parallel(block_S, DV):
                                if left + j_s < seq_end_idx:
                                    v_shared[i_s % num_stages, j_s, j_v] = v[batch_idx, left + j_s, bh, j_v]
                                else:
                                    v_shared[i_s % num_stages, j_s, j_v] = 0
                        # Load A
                        if right <= seq_end_idx:
                            T.copy(
                                a[batch_idx, left:right, bh, 0:block_S],
                                a_shared[i_s % num_stages, :, :],
                            )
                        else:
                            for j_s, j_t in T.Parallel(block_S, block_S):
                                if left + j_s < seq_end_idx:
                                    a_shared[i_s % num_stages, j_s, j_t] = a[batch_idx, left + j_s, bh, j_t]
                                else:
                                    a_shared[i_s % num_stages, j_s, j_t] = 0

                        T.barrier_arrive(data_is_ready[i_s % num_stages])

                elif tx < 384 + 96:
                    for i_s in T.serial(num_iters):
                        T.barrier_wait(
                            data_is_free[i_s % num_stages], (i_s // num_stages + 1) % 2
                        )
                        left = seq_start_idx + i_s * block_S
                        right = left + block_S

                        # Load gamma
                        if right <= seq_end_idx:
                            for j_s in T.Parallel(block_S):
                                g_shared[i_s % num_stages, j_s] = g[
                                    batch_idx, left + j_s, bh
                                ]
                        else:
                            for j_s in T.Parallel(block_S):
                                if left + j_s < seq_end_idx:
                                    g_shared[i_s % num_stages, j_s] = g[
                                        batch_idx, left + j_s, bh
                                    ]
                                else:
                                    g_shared[i_s % num_stages, j_s] = g[
                                        batch_idx, seq_end_idx - 1, bh
                                    ]
                        # Load beta
                        if right <= seq_end_idx:
                            for j_s in T.Parallel(block_S):
                                b_shared[i_s % num_stages, j_s] = b[
                                    batch_idx, left + j_s, bh
                                ]
                        else:
                            for j_s in T.Parallel(block_S):
                                if left + j_s < seq_end_idx:
                                    b_shared[i_s % num_stages, j_s] = b[
                                        batch_idx, left + j_s, bh
                                    ]
                                else:
                                    b_shared[i_s % num_stages, j_s] = 0

                        T.barrier_arrive(data_is_ready[i_s % num_stages])

                else:
                    for i_s in T.serial(num_iters):
                        T.barrier_arrive(bar_0)

                        T.barrier_wait(bar_0, i_s % 2)
                        T.barrier_wait(bar_1, i_s % 2)
                        # Store S
                        T.copy(
                            h_shared,
                            h[batch_idx, chunk_start_idx + i_s, bh, 0:DK, 0:DV],
                        )

    return fq_prepare_h_kernel


def _fq_bwd_kernel(h, hg):
    """Fused chunked GDR backward, warp-specialized (512 threads), grid batch*H.

    Upstream `tilelang_fused_chunk_gdr_bwd` non-varlen with use_dht=True.
    dq/dk/dv all carry `v_shape` and are indexed by the value head `bh`, so the
    Hg<H key-head grads stay unreduced here — the caller sums the group.
    """
    H = h
    Hg = hg
    DK = FQ_DK
    DV = FQ_DV
    chunk_size = FQ_CHUNK
    scale = FQ_SCALE
    accum_dtype = ACCUM_DTYPE
    qkva_dtype = QKVA_DTYPE
    g_dtype = G_DTYPE
    b_dtype = B_DTYPE
    h_dtype = QKVA_DTYPE
    o_dtype = QKVA_DTYPE

    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    num_chunks = T.dynamic("num_chunks")
    block_S = chunk_size

    q_shape = (batch_size, num_tokens, Hg, DK)
    k_shape = (batch_size, num_tokens, Hg, DK)
    v_shape = (batch_size, num_tokens, H, DV)
    o_shape = (batch_size, num_tokens, H, DV)
    a_shape = (batch_size, num_tokens, H, chunk_size)
    g_shape = (batch_size, num_tokens, H)
    b_shape = (batch_size, num_tokens, H)
    h_shape = (batch_size, num_chunks, H, DK, DV)
    h0_shape = (batch_size, H, DK, DV)
    ht_shape = (batch_size, H, DK, DV)

    @T.prim_func
    def fq_bwd_kernel(
        do: T.Tensor(o_shape, dtype=o_dtype),
        dht: T.Tensor(ht_shape, dtype=accum_dtype),
        q: T.Tensor(q_shape, dtype=qkva_dtype),
        k: T.Tensor(k_shape, dtype=qkva_dtype),
        v: T.Tensor(v_shape, dtype=qkva_dtype),
        a: T.Tensor(a_shape, dtype=qkva_dtype),
        g: T.Tensor(g_shape, dtype=g_dtype),
        b: T.Tensor(b_shape, dtype=b_dtype),
        h: T.Tensor(h_shape, dtype=h_dtype),
        dq: T.Tensor(v_shape, dtype=qkva_dtype),
        dk: T.Tensor(v_shape, dtype=qkva_dtype),
        dv: T.Tensor(v_shape, dtype=qkva_dtype),
        dg: T.Tensor(g_shape, dtype=g_dtype),
        db: T.Tensor(b_shape, dtype=b_dtype),
        dh0: T.Tensor(h0_shape, dtype=accum_dtype),
    ):
        with T.Kernel(batch_size * H, threads=512) as (bbh,):
            bb, bh = bbh // H, bbh % H
            bhg = bh // (H // Hg)

            batch_idx = bb
            seq_start_idx = 0
            seq_end_idx = num_tokens
            chunk_start_idx = 0

            num_iters = T.alloc_var("int32")
            num_iters = T.ceildiv(seq_end_idx - seq_start_idx, block_S)

            # 2+2+2+2 + 1 + 4 = 13 units
            do_shared = T.alloc_shared((block_S, DV), dtype=o_dtype)
            q_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            k_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            v_shared = T.alloc_shared((block_S, DV), dtype=qkva_dtype)
            a_shared = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)
            h_shared = T.alloc_shared((DK, DV), dtype=h_dtype)
            g_shared = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")
            g_exp_shared = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")
            g_rev_exp_shared = T.alloc_shared(
                (block_S), dtype=accum_dtype, scope="shared"
            )
            b_shared = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")

            # 2 units
            dqkv_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            dg_shared = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")
            db_shared = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")

            # 1+1 + 2+2+2 + 4 = 12 units
            tmp_shared_1_1 = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)
            tmp_shared_1_2 = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)
            tmp_shared_1_3 = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)
            tmp_shared_2_1 = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            tmp_shared_2_2 = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            tmp_shared_2_3 = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            tmp_shared_4_1 = T.alloc_shared((DK, DV), dtype=qkva_dtype)

            # CONSUMER_K
            dk_fragment = T.alloc_fragment((block_S, DK), dtype=accum_dtype)
            dv_fragment = T.alloc_fragment((block_S, DK), dtype=accum_dtype)
            odot_fragment_1 = T.alloc_fragment((block_S, DK), dtype=accum_dtype)
            dg_fragment_1 = T.alloc_fragment((block_S), dtype=accum_dtype)
            dg_last_local_1 = T.alloc_fragment((1), dtype=accum_dtype)

            # CONSUMER_A
            mask_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            p_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            a_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            dp_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            da_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            hi_fragment = T.alloc_fragment((block_S, block_S), dtype="uint16")
            lo_fragment = T.alloc_fragment((block_S, block_S), dtype="uint16")
            uint32_fragment = T.alloc_fragment((block_S, block_S), dtype="uint32")
            u_fragment = T.alloc_fragment((block_S, DK), dtype=accum_dtype)
            dq_fragment = T.alloc_fragment((block_S, DK), dtype=accum_dtype)
            db_fragment = T.alloc_fragment((block_S), dtype=accum_dtype)
            odot_fragment_2 = T.alloc_fragment((block_S, DK), dtype=accum_dtype)
            dg_fragment_2 = T.alloc_fragment((block_S), dtype=accum_dtype)

            # CONSUMER_S
            dh_fragment = T.alloc_fragment((DK, DV), dtype=accum_dtype)
            reduce_fragment = T.alloc_fragment((128, 2), dtype=accum_dtype)
            dg_last_local_3 = T.alloc_fragment((1), dtype=accum_dtype)
            g_last_local_3 = T.alloc_local((1), dtype=accum_dtype)

            # 16 stages
            bar_00 = T.alloc_barrier(arrive_count=448)
            bar_01 = T.alloc_barrier(arrive_count=384)
            bar_02 = T.alloc_barrier(arrive_count=288)
            bar_03 = T.alloc_barrier(arrive_count=256)
            bar_04 = T.alloc_barrier(arrive_count=416)
            bar_05 = T.alloc_barrier(arrive_count=288)
            bar_06 = T.alloc_barrier(arrive_count=256)
            bar_07 = T.alloc_barrier(arrive_count=256)
            bar_08 = T.alloc_barrier(arrive_count=384)
            bar_09 = T.alloc_barrier(arrive_count=256)
            bar_10 = T.alloc_barrier(arrive_count=288)
            bar_11 = T.alloc_barrier(arrive_count=256)
            bar_12 = T.alloc_barrier(arrive_count=128)
            bar_13 = T.alloc_barrier(arrive_count=256)
            bar_14 = T.alloc_barrier(arrive_count=256)
            bar_15 = T.alloc_barrier(arrive_count=256)

            T.annotate_layout(
                {
                    do_shared: tilelang.layout.make_swizzled_layout(do_shared),
                    q_shared: tilelang.layout.make_swizzled_layout(q_shared),
                    k_shared: tilelang.layout.make_swizzled_layout(k_shared),
                    v_shared: tilelang.layout.make_swizzled_layout(v_shared),
                    a_shared: tilelang.layout.make_swizzled_layout(a_shared),
                    h_shared: tilelang.layout.make_swizzled_layout(h_shared),
                    dqkv_shared: tilelang.layout.make_swizzled_layout(dqkv_shared),
                    tmp_shared_1_1: tilelang.layout.make_swizzled_layout(
                        tmp_shared_1_1
                    ),
                    tmp_shared_1_2: tilelang.layout.make_swizzled_layout(
                        tmp_shared_1_2
                    ),
                    tmp_shared_1_3: tilelang.layout.make_swizzled_layout(
                        tmp_shared_1_3
                    ),
                    tmp_shared_2_1: tilelang.layout.make_swizzled_layout(
                        tmp_shared_2_1
                    ),
                    tmp_shared_2_2: tilelang.layout.make_swizzled_layout(
                        tmp_shared_2_2
                    ),
                    tmp_shared_2_3: tilelang.layout.make_swizzled_layout(
                        tmp_shared_2_3
                    ),
                    tmp_shared_4_1: tilelang.layout.make_swizzled_layout(
                        tmp_shared_4_1
                    ),
                }
            )

            tx = T.get_thread_binding()

            PRODUCER_NREG = 24
            CONSUMER_K_NREG = 144
            CONSUMER_A_NREG = 184
            CONSUMER_S_NREG = 160

            # Prefetch the last chunk of data
            T.copy(
                h[batch_idx, chunk_start_idx + num_iters - 1, bh, 0:DK, 0:DV],
                h_shared,
            )
            for j_s, j_k in T.Parallel(block_S, DK):
                if seq_start_idx + (num_iters - 1) * block_S + j_s < seq_end_idx:
                    q_shared[j_s, j_k] = q[
                        batch_idx,
                        seq_start_idx + (num_iters - 1) * block_S + j_s,
                        bhg,
                        j_k,
                    ]
                else:
                    q_shared[j_s, j_k] = 0
            for j_s, j_k in T.Parallel(block_S, DK):
                if seq_start_idx + (num_iters - 1) * block_S + j_s < seq_end_idx:
                    k_shared[j_s, j_k] = k[
                        batch_idx,
                        seq_start_idx + (num_iters - 1) * block_S + j_s,
                        bhg,
                        j_k,
                    ]
                else:
                    k_shared[j_s, j_k] = 0
            for j_s, j_v in T.Parallel(block_S, DV):
                if seq_start_idx + (num_iters - 1) * block_S + j_s < seq_end_idx:
                    v_shared[j_s, j_v] = v[
                        batch_idx,
                        seq_start_idx + (num_iters - 1) * block_S + j_s,
                        bh,
                        j_v,
                    ]
                else:
                    v_shared[j_s, j_v] = 0
            for j_s, j_t in T.Parallel(block_S, block_S):
                if seq_start_idx + (num_iters - 1) * block_S + j_s < seq_end_idx:
                    a_shared[j_s, j_t] = a[
                        batch_idx,
                        seq_start_idx + (num_iters - 1) * block_S + j_s,
                        bh,
                        j_t,
                    ]
                else:
                    a_shared[j_s, j_t] = 0
            for j_s, j_v in T.Parallel(block_S, DV):
                if seq_start_idx + (num_iters - 1) * block_S + j_s < seq_end_idx:
                    do_shared[j_s, j_v] = do[
                        batch_idx,
                        seq_start_idx + (num_iters - 1) * block_S + j_s,
                        bh,
                        j_v,
                    ]
                else:
                    do_shared[j_s, j_v] = 0
            for j_s in T.Parallel(block_S):
                if seq_start_idx + (num_iters - 1) * block_S + j_s < seq_end_idx:
                    g_shared[j_s] = g[
                        batch_idx, seq_start_idx + (num_iters - 1) * block_S + j_s, bh
                    ]
                else:
                    g_shared[j_s] = g[batch_idx, seq_end_idx - 1, bh]
            for j_s in T.Parallel(block_S):
                if seq_start_idx + (num_iters - 1) * block_S + j_s < seq_end_idx:
                    b_shared[j_s] = b[
                        batch_idx, seq_start_idx + (num_iters - 1) * block_S + j_s, bh
                    ]
                else:
                    b_shared[j_s] = 0

            if tx < 128:
                T.set_max_nreg(CONSUMER_S_NREG, 1)

                T.copy(dht[bb, bh, 0:DK, 0:DV], dh_fragment)
                T.copy(dh_fragment, tmp_shared_4_1)

                for i_s in T.serial(num_iters):
                    T.barrier_arrive(bar_00)

                    # 00
                    T.barrier_wait(bar_00, (i_s + 0) % 2)
                    for j_s in T.Parallel(block_S):
                        g_exp_shared[j_s] = T.exp2(g_shared[j_s] * 1.442695)
                        g_rev_exp_shared[j_s] = T.exp2(
                            (g_shared[block_S - 1] - g_shared[j_s]) * 1.442695
                        )
                    T.barrier_arrive(bar_01)

                    # 01, 02, 03
                    T.barrier_wait(bar_01, (i_s + 0) % 2)
                    g_last_local_3[0] = g_exp_shared[block_S - 1]
                    # dS0 = g_last * dSt
                    for j_k, j_v in T.Parallel(DK, DV):
                        dh_fragment[j_k, j_v] *= g_last_local_3[0]
                    T.barrier_arrive(bar_04)

                    # 04, 05, 06, 07
                    T.barrier_wait(bar_04, (i_s + 0) % 2)
                    # dg_last += sum(dS0 * S0)
                    T.clear(reduce_fragment)
                    for j_k, j_v in T.Parallel(DK, DV):
                        reduce_fragment[
                            j_k % 64 // 16 * 32
                            + j_k % 8 * 4
                            + j_v % 8 // 2,
                            j_v % 2,
                        ] += dh_fragment[j_k, j_v] * h_shared[j_k, j_v]
                    T.barrier_arrive(bar_08)
                    T.barrier_wait(bar_08, (i_s + 0) % 2)
                    T.barrier_wait(bar_09, (i_s + 0) % 2)

                    # 10
                    T.barrier_wait(bar_10, (i_s + 0) % 2)
                    T.reduce_sum(
                        T.reshape(reduce_fragment, (128 * 2,)),
                        dg_last_local_3,
                        dim=0,
                        clear=True,
                    )
                    dg_shared[block_S - 1] += dg_last_local_3[0]
                    T.barrier_arrive(bar_11)

                    # 11
                    T.barrier_wait(bar_11, (i_s + 0) % 2)
                    # dS0 += K^T @ dVg
                    T.gemm(
                        tmp_shared_2_2,
                        tmp_shared_2_3,
                        dh_fragment,
                        transpose_A=True,
                        clear_accum=False,
                    )
                    T.barrier_arrive(bar_12)
                    T.barrier_wait(bar_12, (i_s + 0) % 2)

                    # 13
                    T.barrier_wait(bar_13, (i_s + 0) % 2)
                    # dOg = s * g * dO
                    for j_s, j_v in T.Parallel(block_S, DV):
                        tmp_shared_2_3[j_s, j_v] = (
                            scale * do_shared[j_s, j_v] * g_exp_shared[j_s]
                        )
                    T.barrier_arrive(bar_14)

                    # 14
                    T.barrier_wait(bar_14, (i_s + 0) % 2)
                    # dS0 += Q^T @ dOg
                    T.gemm(
                        tmp_shared_2_1,
                        tmp_shared_2_3,
                        dh_fragment,
                        transpose_A=True,
                        clear_accum=False,
                    )
                    T.barrier_arrive(bar_15)

                    # 15
                    T.barrier_wait(bar_15, (i_s + 0) % 2)
                    # S4[1] = dS0
                    T.copy(dh_fragment, tmp_shared_4_1)

                T.copy(dh_fragment, dh0[bb, bh, 0:DK, 0:DV])

            elif tx < 256:
                T.set_max_nreg(CONSUMER_K_NREG, 1)

                for i_s in T.serial(num_iters):
                    T.barrier_arrive(bar_00)

                    # 16 == 00
                    T.barrier_wait(bar_00, (i_s + 0) % 2)
                    # S2[S] dK
                    if i_s > 0:
                        T.copy(dk_fragment, dqkv_shared)
                    T.barrier_arrive(bar_01)

                    # 01
                    T.barrier_wait(bar_01, (i_s + 0) % 2)
                    # dV' = K @ dSt
                    T.gemm(k_shared, tmp_shared_4_1, dv_fragment, clear_accum=True)
                    # dV' = g_last/g * dV'
                    for j_s, j_v in T.Parallel(block_S, DV):
                        dv_fragment[j_s, j_v] *= g_rev_exp_shared[j_s]
                    T.barrier_arrive(bar_02)

                    # 02
                    T.barrier_wait(bar_02, (i_s + 0) % 2)
                    # dV' += Pg^T @ dO
                    T.gemm(
                        tmp_shared_1_1,
                        do_shared,
                        dv_fragment,
                        transpose_A=True,
                        clear_accum=False,
                    )
                    T.barrier_arrive(bar_03)

                    # 03
                    T.barrier_wait(bar_03, (i_s + 0) % 2)
                    # S2[1] dV'
                    T.copy(dv_fragment, tmp_shared_2_1)
                    T.barrier_arrive(bar_04)

                    # 04
                    T.barrier_wait(bar_04, (i_s + 0) % 2)
                    # dV = Ag^T @ dV'
                    T.gemm(
                        tmp_shared_1_2,
                        tmp_shared_2_1,
                        dv_fragment,
                        transpose_A=True,
                        clear_accum=True,
                    )
                    # S2[S] dV
                    T.copy(dv_fragment, dqkv_shared)
                    T.barrier_arrive(bar_05)

                    # 05
                    T.barrier_wait(bar_05, (i_s + 0) % 2)
                    # dVg = -g * dV
                    for j_s, j_v in T.Parallel(block_S, DV):
                        dv_fragment[j_s, j_v] = (
                            -dv_fragment[j_s, j_v] * g_exp_shared[j_s]
                        )
                    # dg += sum(dVg * U)
                    T.copy(tmp_shared_2_3, odot_fragment_1)
                    for j_s, j_v in T.Parallel(block_S, DV):
                        odot_fragment_1[j_s, j_v] *= dv_fragment[j_s, j_v]
                    T.reduce_sum(odot_fragment_1, dg_fragment_1, dim=1, clear=True)
                    T.copy(dg_fragment_1, dg_shared)
                    # S2[3] dVg
                    T.copy(dv_fragment, tmp_shared_2_3)
                    T.barrier_arrive(bar_06)

                    # 06
                    T.barrier_wait(bar_06, (i_s + 0) % 2)
                    # S2[2] K
                    T.copy(k_shared, odot_fragment_1)
                    T.copy(odot_fragment_1, tmp_shared_2_2)
                    T.barrier_arrive(bar_07)

                    # 07
                    T.barrier_wait(bar_07, (i_s + 0) % 2)
                    # dK = V' @ dSt^T
                    T.gemm(
                        tmp_shared_2_1,
                        tmp_shared_4_1,
                        dk_fragment,
                        transpose_B=True,
                        clear_accum=True,
                    )
                    T.barrier_arrive(bar_08)

                    # 08
                    T.barrier_wait(bar_08, (i_s + 0) % 2)
                    # dK = g_last/g * dK
                    for j_s, j_k in T.Parallel(block_S, DK):
                        dk_fragment[j_s, j_k] *= g_rev_exp_shared[j_s]
                    # dg -= sum(K * dK)
                    for j_s, j_k in T.Parallel(block_S, DK):
                        odot_fragment_1[j_s, j_k] *= -dk_fragment[j_s, j_k]
                    T.reduce_sum(odot_fragment_1, dg_fragment_1, dim=1, clear=True)
                    for j_s in T.Parallel(block_S):
                        dg_shared[j_s] += dg_fragment_1[j_s]
                    # dg_last += sum(K * dK)
                    T.reduce_sum(dg_fragment_1, dg_last_local_1, dim=0, clear=True)
                    # Sg[S] dg
                    dg_shared[block_S - 1] -= dg_last_local_1[0]
                    T.barrier_arrive(bar_09)

                    # 09
                    T.barrier_wait(bar_09, (i_s + 0) % 2)
                    # dK += dVg @ S0^T
                    T.gemm(
                        tmp_shared_2_3,
                        h_shared,
                        dk_fragment,
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.barrier_arrive(bar_10)
                    T.barrier_wait(bar_10, (i_s + 0) % 2)

                    # 12
                    T.barrier_wait(bar_12, (i_s + 0) % 2)
                    # dK += dP^T @ Q
                    T.gemm(
                        tmp_shared_1_1,
                        tmp_shared_2_1,
                        dk_fragment,
                        transpose_A=True,
                        clear_accum=False,
                    )
                    T.barrier_arrive(bar_13)
                    T.barrier_wait(bar_13, (i_s + 0) % 2)

                    # 15
                    T.barrier_wait(bar_15, (i_s + 0) % 2)
                    # dK += dAs @ K
                    T.gemm(
                        tmp_shared_1_2, tmp_shared_2_2, dk_fragment, clear_accum=False
                    )

                for j_s, j_k in T.Parallel(block_S, DK):
                    if seq_start_idx + j_s < seq_end_idx:
                        dk[batch_idx, seq_start_idx + j_s, bh, j_k] = dk_fragment[
                            j_s, j_k
                        ]

            elif tx < 384:
                T.set_max_nreg(CONSUMER_A_NREG, 1)

                for i_s in T.serial(num_iters):
                    T.barrier_arrive(bar_00)

                    # 00
                    T.barrier_wait(bar_00, (i_s + 0) % 2)
                    # P = Q @ K^T
                    T.gemm(
                        q_shared,
                        k_shared,
                        p_fragment,
                        transpose_B=True,
                        clear_accum=True,
                    )
                    T.barrier_arrive(bar_01)

                    # 01
                    T.barrier_wait(bar_01, (i_s + 0) % 2)
                    # G = Lower(diag(g) @ I @ diag(1/g))
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        mask_fragment[j_s, j_t] = g_shared[j_s] - g_shared[j_t]
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        if j_s >= j_t:
                            mask_fragment[j_s, j_t] = T.exp2(
                                mask_fragment[j_s, j_t] * 1.442695
                            )
                        else:
                            mask_fragment[j_s, j_t] = 0
                    # Pg = s * P * G
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        p_fragment[j_s, j_t] *= mask_fragment[j_s, j_t]
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        p_fragment[j_s, j_t] *= scale
                    # S1[1] Pg
                    T.copy(p_fragment, tmp_shared_1_1)
                    T.barrier_arrive(bar_02)

                    # 02
                    T.barrier_wait(bar_02, (i_s + 0) % 2)
                    # Ab = Ar * b
                    T.copy(a_shared, a_fragment)
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        a_fragment[j_s, j_t] *= b_shared[j_t]
                    # Ag = G * Ab
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        a_fragment[j_s, j_t] *= mask_fragment[j_s, j_t]
                    # S1[2] Ag
                    T.copy(a_fragment, tmp_shared_1_2)
                    T.barrier_arrive(bar_03)

                    # 03
                    T.barrier_wait(bar_03, (i_s + 0) % 2)
                    # U = K @ S0
                    T.gemm(k_shared, h_shared, u_fragment, clear_accum=True)
                    T.barrier_arrive(bar_04)

                    # 04
                    T.barrier_wait(bar_04, (i_s + 0) % 2)
                    # S2[3] U
                    T.copy(u_fragment, tmp_shared_2_3)
                    # W = V - g * U
                    for j_s, j_v in T.Parallel(block_S, DV):
                        u_fragment[j_s, j_v] *= -g_exp_shared[j_s]
                    for j_s, j_v in T.Parallel(block_S, DV):
                        u_fragment[j_s, j_v] += v_shared[j_s, j_v]
                    # S2[2] W
                    T.copy(u_fragment, tmp_shared_2_2)
                    T.barrier_arrive(bar_05)

                    # 05
                    T.barrier_wait(bar_05, (i_s + 0) % 2)
                    # dAg = dV' @ W^T
                    T.gemm(
                        tmp_shared_2_1,
                        tmp_shared_2_2,
                        da_fragment,
                        transpose_B=True,
                        clear_accum=True,
                    )
                    # V' = Ag @ W
                    T.gemm(
                        tmp_shared_1_2, tmp_shared_2_2, u_fragment, clear_accum=True
                    )
                    # S2[1] V'
                    T.copy(u_fragment, tmp_shared_2_1)
                    T.barrier_arrive(bar_06)

                    # 06
                    T.barrier_wait(bar_06, (i_s + 0) % 2)
                    # dPg = dO @ V'^T
                    T.gemm(
                        do_shared,
                        tmp_shared_2_1,
                        dp_fragment,
                        transpose_B=True,
                        clear_accum=True,
                    )
                    T.barrier_arrive(bar_07)

                    # 07
                    T.barrier_wait(bar_07, (i_s + 0) % 2)
                    # dAb = G * dAg
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        da_fragment[j_s, j_t] *= mask_fragment[j_s, j_t]
                    # dg += sum((dPg * P) - (dPg * P)^T)
                    T.copy(tmp_shared_1_1, p_fragment)
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        p_fragment[j_s, j_t] *= dp_fragment[j_s, j_t]
                    # dP = s * G * dPg
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        dp_fragment[j_s, j_t] *= mask_fragment[j_s, j_t]
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        dp_fragment[j_s, j_t] *= scale
                    # S1[1] dP
                    T.copy(dp_fragment, tmp_shared_1_1)
                    T.barrier_arrive(bar_08)

                    # 08
                    T.barrier_wait(bar_08, (i_s + 0) % 2)
                    # dQ = dO @ S0^T
                    T.gemm(
                        do_shared,
                        h_shared,
                        dq_fragment,
                        transpose_B=True,
                        clear_accum=True,
                    )
                    T.barrier_arrive(bar_09)

                    # 09
                    T.barrier_wait(bar_09, (i_s + 0) % 2)
                    # dQ = s * g * dQ
                    for j_s, j_k in T.Parallel(block_S, DK):
                        dq_fragment[j_s, j_k] *= g_exp_shared[j_s]
                    for j_s, j_k in T.Parallel(block_S, DK):
                        dq_fragment[j_s, j_k] *= scale
                    # S2[1] Q
                    T.copy(q_shared, odot_fragment_2)
                    # dg += sum(Q * dQ)
                    T.copy(odot_fragment_2, tmp_shared_2_1)
                    for j_s, j_k in T.Parallel(block_S, DK):
                        odot_fragment_2[j_s, j_k] *= dq_fragment[j_s, j_k]
                    T.reduce_sum(odot_fragment_2, dg_fragment_2, dim=1, clear=True)
                    T.barrier_arrive(bar_10)

                    # 10
                    T.barrier_wait(bar_10, (i_s + 0) % 2)
                    # dQ += dP @ K
                    T.gemm(
                        tmp_shared_1_1, tmp_shared_2_2, dq_fragment, clear_accum=False
                    )
                    # S2[S] dQ
                    T.copy(dq_fragment, dqkv_shared)
                    T.barrier_arrive(bar_11)

                    # 11, 12
                    T.barrier_wait(bar_11, (i_s + 0) % 2)
                    # dAb * Ar
                    T.copy(a_shared, a_fragment)
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        a_fragment[j_s, j_t] *= da_fragment[j_s, j_t]
                    T.copy(a_fragment, tmp_shared_1_3)
                    # dAb * Ab [ = G * dAg * Ab ]
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        a_fragment[j_s, j_t] *= b_shared[j_t]
                    # dg += sum((dAb * Ab) - (dAb * Ab)^T)
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        a_fragment[j_s, j_t] += p_fragment[j_s, j_t]
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        x = T.reinterpret(a_fragment[j_s, j_t], dtype="uint32")
                        lo_fragment[j_s, j_t] = x & 0xffff
                        hi_fragment[j_s, j_t] = x >> 16
                    for j_s, j_t in T.Parallel(block_S, block_S // 2):
                        for j_t_vec in T.vectorized(2):
                            tmp_shared_1_2[j_s, j_t * 2 + j_t_vec] = T.reinterpret(
                                hi_fragment[j_s, j_t * 2 + j_t_vec],
                                dtype=qkva_dtype,
                            )
                    for j_s, j_t in T.Parallel(block_S, block_S // 2):
                        for j_t_vec in T.vectorized(2):
                            hi_fragment[j_s, j_t * 2 + j_t_vec] = T.reinterpret(
                                tmp_shared_1_2[j_t * 2 + j_t_vec, j_s],
                                dtype="uint16",
                            )
                    for j_s, j_t in T.Parallel(block_S, block_S // 2):
                        for j_t_vec in T.vectorized(2):
                            tmp_shared_1_2[j_s, j_t * 2 + j_t_vec] = T.reinterpret(
                                lo_fragment[j_s, j_t * 2 + j_t_vec],
                                dtype=qkva_dtype,
                            )
                    for j_s, j_t in T.Parallel(block_S, block_S // 2):
                        for j_t_vec in T.vectorized(2):
                            lo_fragment[j_s, j_t * 2 + j_t_vec] = T.reinterpret(
                                tmp_shared_1_2[j_t * 2 + j_t_vec, j_s],
                                dtype="uint16",
                            )
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        uint32_fragment[j_s, j_t] = (hi_fragment[j_s, j_t] << 16) + \
                            lo_fragment[j_s, j_t]
                        p_fragment[j_s, j_t] = T.reinterpret(
                            uint32_fragment[j_s, j_t],
                            dtype=accum_dtype,
                        )
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        a_fragment[j_s, j_t] -= p_fragment[j_s, j_t]
                    T.reduce_sum(a_fragment, dg_fragment_2, dim=1, clear=False)
                    # Sg[S] dg
                    for j_s in T.Parallel(block_S):
                        dg_shared[j_s] += dg_fragment_2[j_s]
                    # db = sum((dAb * Ar)^T)
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        a_fragment[j_s, j_t] = tmp_shared_1_3[j_t, j_s]
                    T.reduce_sum(a_fragment, db_fragment, dim=1, clear=True)
                    # dAr = dAb * b
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        da_fragment[j_s, j_t] *= b_shared[j_t]
                    # S1[2] dAr
                    T.copy(da_fragment, tmp_shared_1_2)
                    T.barrier_arrive(bar_13)

                    # 13
                    T.barrier_wait(bar_13, (i_s + 0) % 2)
                    # dA = -Ar^T @ dAr @ Ar^T
                    T.gemm(
                        a_shared,
                        tmp_shared_1_2,
                        da_fragment,
                        transpose_A=True,
                        clear_accum=True,
                    )
                    T.copy(da_fragment, tmp_shared_1_2)
                    T.gemm(
                        tmp_shared_1_2,
                        a_shared,
                        da_fragment,
                        transpose_B=True,
                        clear_accum=True,
                    )
                    # At = K @ K^T
                    T.gemm(
                        tmp_shared_2_2,
                        tmp_shared_2_2,
                        a_fragment,
                        transpose_B=True,
                        clear_accum=True,
                    )
                    T.barrier_arrive(bar_14)

                    # 14
                    T.barrier_wait(bar_14, (i_s + 0) % 2)
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        if j_s <= j_t:
                            da_fragment[j_s, j_t] = 0
                        else:
                            da_fragment[j_s, j_t] = -da_fragment[j_s, j_t]
                    # db += sum(dA * At)
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        a_fragment[j_s, j_t] *= da_fragment[j_s, j_t]
                    T.reduce_sum(a_fragment, db_fragment, dim=1, clear=False)
                    T.copy(db_fragment, db_shared)
                    # dAt = b * dA
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        da_fragment[j_s, j_t] *= b_shared[j_s]
                    # dAs = dAt + dAt^T
                    T.copy(da_fragment, tmp_shared_1_2)
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        da_fragment[j_s, j_t] += tmp_shared_1_2[j_t, j_s]
                    # S1[1] dAs
                    T.copy(da_fragment, tmp_shared_1_2)
                    T.barrier_arrive(bar_15)
                    T.barrier_wait(bar_15, (i_s + 0) % 2)

            else:
                T.set_max_nreg(PRODUCER_NREG, 0)

                if tx < 384 + 32:
                    for i_s in T.serial(num_iters - 1):
                        chunk_idx = num_iters - i_s - 2
                        left = seq_start_idx + chunk_idx * block_S
                        right = left + block_S

                        T.barrier_arrive(bar_00)
                        T.barrier_wait(bar_00, (i_s + 0) % 2)

                        T.barrier_wait(bar_03, (i_s + 0) % 2)
                        for j_s in T.Parallel(block_S):
                            g_shared[j_s] = g[batch_idx, left + j_s, bh]

                        T.barrier_wait(bar_05, (i_s + 0) % 2)
                        T.copy(v[batch_idx, left:right, bh, 0:DV], v_shared)

                        T.barrier_wait(bar_07, (i_s + 0) % 2)
                        T.copy(k[batch_idx, left:right, bhg, 0:DK], k_shared)

                        T.barrier_wait(bar_10, (i_s + 0) % 2)
                        T.copy(q[batch_idx, left:right, bhg, 0:DK], q_shared)

                    if num_iters > 0:
                        T.barrier_arrive(bar_00)

                elif tx < 384 + 64:
                    if bb == batch_size - 1:
                        for j_s, j_v in T.Parallel(block_S, DV):
                            if seq_end_idx + j_s < num_tokens:
                                dv[batch_idx, seq_end_idx + j_s, bh, j_v] = 0
                        for j_s, j_k in T.Parallel(block_S, DK):
                            if seq_end_idx + j_s < num_tokens:
                                dq[batch_idx, seq_end_idx + j_s, bh, j_k] = 0
                        for j_s, j_k in T.Parallel(block_S, DK):
                            if seq_end_idx + j_s < num_tokens:
                                dk[batch_idx, seq_end_idx + j_s, bh, j_k] = 0

                    for i_s in T.serial(num_iters):
                        left = seq_start_idx + (num_iters - i_s - 1) * block_S
                        right = left + block_S

                        T.barrier_arrive(bar_00)
                        T.barrier_wait(bar_00, (i_s + 0) % 2)

                        T.barrier_wait(bar_01, (i_s + 0) % 2)
                        if i_s == 1:
                            for j_s, j_k in T.Parallel(block_S, DK):
                                if left + block_S + j_s < seq_end_idx:
                                    dk[batch_idx, left + block_S + j_s, bh, j_k] = (
                                        dqkv_shared[j_s, j_k]
                                    )
                        elif i_s > 1:
                            T.copy(
                                dqkv_shared,
                                dk[
                                    batch_idx,
                                    left + block_S : right + block_S,
                                    bh,
                                    0:DK,
                                ],
                            )
                        T.barrier_arrive(bar_04)
                        T.barrier_wait(bar_04, (i_s + 0) % 2)

                        T.barrier_wait(bar_05, (i_s + 0) % 2)
                        if i_s == 0:
                            for j_s, j_v in T.Parallel(block_S, DV):
                                if left + j_s < seq_end_idx:
                                    dv[batch_idx, left + j_s, bh, j_v] = dqkv_shared[
                                        j_s, j_v
                                    ]
                        else:
                            T.copy(dqkv_shared, dv[batch_idx, left:right, bh, 0:DV])
                        T.barrier_arrive(bar_10)
                        T.barrier_wait(bar_10, (i_s + 0) % 2)

                        T.barrier_wait(bar_11, (i_s + 0) % 2)
                        if i_s == 0:
                            for j_s, j_k in T.Parallel(block_S, DK):
                                if left + j_s < seq_end_idx:
                                    dq[batch_idx, left + j_s, bh, j_k] = dqkv_shared[
                                        j_s, j_k
                                    ]
                        else:
                            T.copy(dqkv_shared, dq[batch_idx, left:right, bh, 0:DK])

                elif tx < 384 + 96:
                    for i_s in T.serial(num_iters - 1):
                        chunk_idx = num_iters - i_s - 2
                        left = seq_start_idx + chunk_idx * block_S
                        right = left + block_S

                        T.barrier_arrive(bar_02)
                        T.barrier_wait(bar_02, (i_s + 0) % 2)

                        T.barrier_wait(bar_10, (i_s + 0) % 2)
                        T.copy(
                            h[batch_idx, chunk_start_idx + chunk_idx, bh, 0:DK, 0:DV],
                            h_shared,
                        )

                        T.barrier_wait(bar_14, (i_s + 0) % 2)
                        T.copy(a[batch_idx, left:right, bh, 0:block_S], a_shared)

                        T.copy(do[batch_idx, left:right, bh, 0:DV], do_shared)

                        T.barrier_wait(bar_15, (i_s + 0) % 2)
                        for j_s in T.Parallel(block_S):
                            b_shared[j_s] = b[batch_idx, left + j_s, bh]

                    if num_iters > 0:
                        T.barrier_wait(bar_00, (num_iters - 1) % 2)
                        T.barrier_arrive(bar_02)

                else:
                    if bb == batch_size - 1:
                        for j_s, j_v in T.Parallel(block_S, DV):
                            if seq_end_idx + j_s < num_tokens:
                                dv[batch_idx, seq_end_idx + j_s, bh, j_v] = 0
                        for j_s, j_k in T.Parallel(block_S, DK):
                            if seq_end_idx + j_s < num_tokens:
                                dq[batch_idx, seq_end_idx + j_s, bh, j_k] = 0
                        for j_s, j_k in T.Parallel(block_S, DK):
                            if seq_end_idx + j_s < num_tokens:
                                dk[batch_idx, seq_end_idx + j_s, bh, j_k] = 0

                    for i_s in T.serial(num_iters):
                        left = seq_start_idx + (num_iters - i_s - 1) * block_S

                        T.barrier_arrive(bar_05)
                        T.barrier_wait(bar_05, (i_s + 0) % 2)

                        T.barrier_wait(bar_15, (i_s + 0) % 2)

                        if i_s == 0:
                            for j_s in T.Parallel(block_S):
                                if left + j_s < seq_end_idx:
                                    dg[batch_idx, left + j_s, bh] = dg_shared[j_s]
                            if (seq_end_idx - seq_start_idx) % block_S > 0:
                                dg[batch_idx, seq_end_idx - 1, bh] += dg_shared[
                                    block_S - 1
                                ]
                        else:
                            for j_s in T.Parallel(block_S):
                                dg[batch_idx, left + j_s, bh] = dg_shared[j_s]

                        if i_s == 0:
                            for j_s in T.Parallel(block_S):
                                if left + j_s < seq_end_idx:
                                    db[batch_idx, left + j_s, bh] = db_shared[j_s]
                        else:
                            for j_s in T.Parallel(block_S):
                                db[batch_idx, left + j_s, bh] = db_shared[j_s]

    return fq_bwd_kernel


_KERNELS = {
    "fq_cumsum": _fq_cumsum_kernel,
    "fq_kkt": _fq_kkt_kernel,
    "fq_fwd": _fq_fwd_kernel,
    "fq_prepare_h": _fq_prepare_h_kernel,
    "fq_bwd": _fq_bwd_kernel,
}


def get_kernel(name: str, h: int, hg: int):
    if name not in _KERNELS:
        raise KeyError(f"unknown flashqla_gdr kernel key {name!r}")
    if h <= 0 or hg <= 0 or h % hg != 0:
        raise ValueError(f"invalid GDN head geometry H={h}, Hg={hg}")
    return _KERNELS[name](h, hg)


def get_pass_configs(name: str):
    return PASS_CONFIGS.get(name, {})
