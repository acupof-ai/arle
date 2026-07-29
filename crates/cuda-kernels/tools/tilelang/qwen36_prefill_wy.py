"""S3 Kernel A — chunkwise-prefill "wy" for Qwen3.6-27B GDR (stages 1-5).

Chunk-parallel: grid (num_chunks, B*num_value_heads). Per (chunk, value_head),
fuses the intra-chunk work of the gated delta rule:

  1 prepare : L2-norm q/k (q also *key_dim**-0.5), g = -exp(A_log)*softplus(a+dt_bias), beta = sigmoid(b)
  2 cumsum  : chunk-local prefix sum of g  -> gcs[chunk]
  3 kkt     : A[i,j] = beta_i * exp(gcs_i - gcs_j) * (k_i . k_j),  strict-lower (i>j) masked
  4 solve   : M = (I + StrictLower(A))^{-1}   (exact forward-substitution, fp32)
  5 wy      : u = M @ (v * beta) ,  w = M @ (k * beta * exp(gcs))

Writes u, w, gcs to HBM for the serial state scan (Kernel B). A and M never
leave shared/fragment.

Gate convention: T.exp everywhere (matches example_chunk_scaled_dot_kkt.py:117
and the validated S1a decode kernel), NOT exp2 — kept consistent across the
GDR kernels so numerics line up.

solve_tril here is the single-block serial forward-substitution proven bit-exact
in qwen36_solve_tril.py (realistic GDN scale). The KDA 4x(16x16) blocked variant
is a perf optimization for later, not needed for correctness.
"""

import tilelang
import tilelang.language as T

QWEN36_27B_NUM_KEY_HEADS = 16
QWEN36_27B_NUM_VALUE_HEADS = 48
QWEN36_27B_KEY_DIM = 128
QWEN36_27B_VAL_DIM = 128
QWEN36_27B_CHUNK = 64


def prefill_wy(
    S: int,
    B: int = 1,
    num_key_heads: int = QWEN36_27B_NUM_KEY_HEADS,
    num_value_heads: int = QWEN36_27B_NUM_VALUE_HEADS,
    key_dim: int = QWEN36_27B_KEY_DIM,
    val_dim: int = QWEN36_27B_VAL_DIM,
    chunk: int = QWEN36_27B_CHUNK,
    in_dtype: str = "bfloat16",
    accum_dtype: str = "float32",
):
    """Build the chunkwise-prefill wy kernel (stages 1-5).

    Inputs are already-projected, post-conv q/k/v (bf16), per-token gate scalars.
    grid = (num_chunks, B * num_value_heads); threads = chunk (64), one thread per
    row of the chunk.  Contractions over key_dim are serial per-thread loops (like
    the validated decode kernel) to keep this first version simple and correct;
    WGMMA tiling is a later optimization.
    """
    NT = S // chunk
    scale = float(key_dim) ** -0.5

    @T.prim_func
    def kernel(
        q: T.Tensor((B, S, num_key_heads, key_dim), in_dtype),
        k: T.Tensor((B, S, num_key_heads, key_dim), in_dtype),
        v: T.Tensor((B, S, num_value_heads, val_dim), in_dtype),
        g: T.Tensor((B, S, num_value_heads), accum_dtype),       # raw log-decay per token
        beta: T.Tensor((B, S, num_value_heads), accum_dtype),    # already sigmoid(b)
        u: T.Tensor((B, S, num_value_heads, val_dim), accum_dtype),
        w: T.Tensor((B, S, num_value_heads, key_dim), accum_dtype),
        gcs: T.Tensor((B, S, num_value_heads), accum_dtype),     # cumsum(g) per chunk
    ):
        with T.Kernel(NT, B * num_value_heads, threads=chunk) as (ct, bvh):
            bb = bvh // num_value_heads
            vh = bvh % num_value_heads
            kh = vh * num_key_heads // num_value_heads
            ti = T.get_thread_binding(0)  # row within the chunk, 0..chunk-1
            s0 = ct * chunk               # first token of this chunk

            qn = T.alloc_shared((chunk, key_dim), accum_dtype)   # L2-normed q rows
            kn = T.alloc_shared((chunk, key_dim), accum_dtype)   # L2-normed k rows
            vb = T.alloc_shared((chunk, val_dim), accum_dtype)   # v * beta
            gcs_s = T.alloc_shared((chunk,), accum_dtype)        # cumsum(g)
            beta_s = T.alloc_shared((chunk,), accum_dtype)
            A = T.alloc_shared((chunk, chunk), accum_dtype)      # (I+L) strict-lower part
            M = T.alloc_shared((chunk, chunk), accum_dtype)      # inverse

            # ---- stage 1: load + L2-norm q/k (thread ti owns row ti) ----
            nq = T.alloc_local((1,), accum_dtype)
            nk = T.alloc_local((1,), accum_dtype)
            T.clear(nq)
            T.clear(nk)
            for d in T.serial(key_dim):
                qv = q[bb, s0 + ti, kh, d].astype(accum_dtype)
                kv = k[bb, s0 + ti, kh, d].astype(accum_dtype)
                qn[ti, d] = qv
                kn[ti, d] = kv
                nq[0] += qv * qv
                nk[0] += kv * kv
            rq = T.rsqrt(nq[0] + 1e-12) * scale
            rk = T.rsqrt(nk[0] + 1e-12)
            for d in T.serial(key_dim):
                qn[ti, d] = qn[ti, d] * rq
                kn[ti, d] = kn[ti, d] * rk
            beta_s[ti] = beta[bb, s0 + ti, vh]
            for d in T.serial(val_dim):
                vb[ti, d] = v[bb, s0 + ti, vh, d].astype(accum_dtype) * beta_s[ti]
            T.tvm_storage_sync("shared")

            # ---- stage 2: chunk-local cumsum of g (single thread; chunk=64 small) ----
            if ti == 0:
                run = T.alloc_local((1,), accum_dtype)
                T.clear(run)
                for r in T.serial(chunk):
                    run[0] += g[bb, s0 + r, vh]
                    gcs_s[r] = run[0]
            T.tvm_storage_sync("shared")
            gcs[bb, s0 + ti, vh] = gcs_s[ti]

            # ---- stage 3: A[i,j] = beta_i*exp(gcs_i-gcs_j)*(k_i.k_j), strict-lower ----
            # thread ti computes row i=ti
            for j in T.serial(chunk):
                dot = T.alloc_local((1,), accum_dtype)
                T.clear(dot)
                for d in T.serial(key_dim):
                    dot[0] += kn[ti, d] * kn[j, d]
                gd = gcs_s[ti] - gcs_s[j]
                A[ti, j] = T.if_then_else(
                    (j < ti),  # strict lower
                    beta_s[ti] * T.exp(gd) * dot[0],
                    0.0,
                )
            T.tvm_storage_sync("shared")

            # ---- stage 4: M = (I + StrictLower(A))^{-1} via forward substitution ----
            # M[i,:] = e_i - sum_{j<i} A[i,j]*M[j,:] ; rows resolved in order.
            for c in T.serial(chunk):
                M[ti, c] = T.if_then_else(c == ti, 1.0, 0.0)
            T.tvm_storage_sync("shared")
            for r in T.serial(chunk):
                if ti == r:
                    for c in T.serial(chunk):
                        acc = T.alloc_local((1,), accum_dtype)
                        T.clear(acc)
                        for j in T.serial(chunk):
                            acc[0] += A[r, j] * M[j, c]
                        M[r, c] = M[r, c] - acc[0]
                T.tvm_storage_sync("shared")

            # ---- stage 5: u = M @ (v*beta) , w = M @ (k*beta*exp(gcs)) ----
            # thread ti computes output row i=ti
            for d in T.serial(val_dim):
                accu = T.alloc_local((1,), accum_dtype)
                T.clear(accu)
                for j in T.serial(chunk):
                    accu[0] += M[ti, j] * vb[j, d]
                u[bb, s0 + ti, vh, d] = accu[0]
            for d in T.serial(key_dim):
                accw = T.alloc_local((1,), accum_dtype)
                T.clear(accw)
                for j in T.serial(chunk):
                    # w uses k * beta * exp(gcs) as the RHS row
                    accw[0] += M[ti, j] * (kn[j, d] * beta_s[j] * T.exp(gcs_s[j]))
                w[bb, s0 + ti, vh, d] = accw[0]

    return kernel


def _self_check():
    import torch

    dev = "cuda"
    torch.manual_seed(0)
    B, NKH, NVH, KD, VD, C = 1, QWEN36_27B_NUM_KEY_HEADS, QWEN36_27B_NUM_VALUE_HEADS, \
        QWEN36_27B_KEY_DIM, QWEN36_27B_VAL_DIM, QWEN36_27B_CHUNK
    S = C * 2  # 2 chunks

    q = torch.randn(B, S, NKH, KD, dtype=torch.bfloat16, device=dev)
    k = torch.randn(B, S, NKH, KD, dtype=torch.bfloat16, device=dev)
    v = torch.randn(B, S, NVH, VD, dtype=torch.bfloat16, device=dev)
    g = -torch.rand(B, S, NVH, dtype=torch.float32, device=dev) * 0.1  # small negative log-decay
    beta = torch.sigmoid(torch.randn(B, S, NVH, dtype=torch.float32, device=dev))

    # reference (fp32), per chunk / value head
    scale = KD ** -0.5
    u_ref = torch.zeros(B, S, NVH, VD, dtype=torch.float32, device=dev)
    w_ref = torch.zeros(B, S, NVH, KD, dtype=torch.float32, device=dev)
    for b in range(B):
        for vh in range(NVH):
            kh = vh * NKH // NVH
            for ct in range(S // C):
                sl = slice(ct * C, (ct + 1) * C)
                qc = q[b, sl, kh].float()
                kc = k[b, sl, kh].float()
                vc = v[b, sl, vh].float()
                qc = qc * torch.rsqrt((qc * qc).sum(-1, keepdim=True) + 1e-12) * scale
                kc = kc * torch.rsqrt((kc * kc).sum(-1, keepdim=True) + 1e-12)
                bt = beta[b, sl, vh]
                gcs = torch.cumsum(g[b, sl, vh], 0)
                A = torch.zeros(C, C, device=dev)
                for i in range(C):
                    for j in range(i):
                        A[i, j] = bt[i] * torch.exp(gcs[i] - gcs[j]) * (kc[i] @ kc[j])
                M = torch.linalg.inv(torch.eye(C, device=dev) + A)
                u_ref[b, sl, vh] = M @ (vc * bt[:, None])
                w_ref[b, sl, vh] = M @ (kc * bt[:, None] * torch.exp(gcs)[:, None])

    kernel = tilelang.compile(prefill_wy(S=S, B=B), target="cuda")
    u_o = torch.zeros(B, S, NVH, VD, dtype=torch.float32, device=dev)
    w_o = torch.zeros(B, S, NVH, KD, dtype=torch.float32, device=dev)
    gcs_o = torch.zeros(B, S, NVH, dtype=torch.float32, device=dev)
    kernel(q, k, v, g, beta, u_o, w_o, gcs_o)
    torch.cuda.synchronize()

    for name, got, ref in (("u", u_o, u_ref), ("w", w_o, w_ref)):
        gf, rf = got.float(), ref.float()
        within = ((gf - rf).abs() <= 3e-2 + 3e-2 * rf.abs()).float().mean().item()
        print(f"{name}: {within*100:.2f}% within tol, abs_max={(gf-rf).abs().max().item():.4e}")
        assert within > 0.99, f"{name} mismatch"
    print("prefill_wy self-check PASSED")


if __name__ == "__main__":
    _self_check()
