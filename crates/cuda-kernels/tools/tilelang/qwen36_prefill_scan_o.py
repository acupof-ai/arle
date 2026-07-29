"""S3 Kernel B — chunkwise-prefill state-scan + output for Qwen3.6-27B GDR (stages 6+7).

Chunk-SERIAL: grid (B * num_value_heads,), one block per (batch, value-head),
carrying a running state h[key_dim, val_dim] across chunks in shared memory —
the h->HBM epilogue write of the unfused path is dropped (the fusion win).

Per chunk (from Kernel A's u, w, gcs, plus q, k):
  v_new     = u - w @ h                        (delta correction against carried state)
  o_inter   = (q @ h) * exp(gcs)               (contribution from prior chunks)
  o_intra   = tril( (q @ k^T) * exp(gcs_i - gcs_j), i>j ) @ v_new   (within chunk, causal)
  out       = (o_inter + o_intra) * scale
  h         = h * exp(gcs_last) + k^T @ v_new  (gated decay + rank-chunk update)

Contractions are per-thread serial loops (thread tv owns state column h[:, tv] and
output column, val_dim threads), mirroring the validated decode/wy kernels —
correctness first; WGMMA tiling is a later optimization.

Gate: T.exp throughout (consistent with Kernel A and the decode kernel).
"""

import tilelang
import tilelang.language as T

QWEN36_27B_NUM_KEY_HEADS = 16
QWEN36_27B_NUM_VALUE_HEADS = 48
QWEN36_27B_KEY_DIM = 128
QWEN36_27B_VAL_DIM = 128
QWEN36_27B_CHUNK = 64


def prefill_scan_o(
    S: int,
    B: int = 1,
    num_key_heads: int = QWEN36_27B_NUM_KEY_HEADS,
    num_value_heads: int = QWEN36_27B_NUM_VALUE_HEADS,
    key_dim: int = QWEN36_27B_KEY_DIM,
    val_dim: int = QWEN36_27B_VAL_DIM,
    chunk: int = QWEN36_27B_CHUNK,
    in_dtype: str = "bfloat16",
    accum_dtype: str = "float32",
    out_dtype: str = "bfloat16",
):
    """Build the serial state-scan + output kernel (stages 6+7).

    grid = (B * num_value_heads,); threads = val_dim (128); thread tv owns state
    column h[:, tv] and output column tv. Serial loop over chunks carries h.
    """
    NT = S // chunk
    scale = float(key_dim) ** -0.5

    @T.prim_func
    def kernel(
        q: T.Tensor((B, S, num_key_heads, key_dim), in_dtype),
        k: T.Tensor((B, S, num_key_heads, key_dim), in_dtype),
        u: T.Tensor((B, S, num_value_heads, val_dim), accum_dtype),
        w: T.Tensor((B, S, num_value_heads, key_dim), accum_dtype),
        gcs: T.Tensor((B, S, num_value_heads), accum_dtype),
        h0: T.Tensor((B, num_value_heads, key_dim, val_dim), accum_dtype),  # initial state
        out: T.Tensor((B, S, num_value_heads, val_dim), out_dtype),
        hT: T.Tensor((B, num_value_heads, key_dim, val_dim), accum_dtype),  # final state
    ):
        with T.Kernel(B * num_value_heads, threads=val_dim) as bvh:
            bb = bvh // num_value_heads
            vh = bvh % num_value_heads
            kh = vh * num_key_heads // num_value_heads
            tv = T.get_thread_binding(0)  # value-dim column this thread owns

            h = T.alloc_shared((key_dim, val_dim), accum_dtype)   # running state
            qn = T.alloc_shared((chunk, key_dim), accum_dtype)    # q normed (per chunk)
            kn = T.alloc_shared((chunk, key_dim), accum_dtype)
            vnew = T.alloc_shared((chunk, val_dim), accum_dtype)
            gcs_s = T.alloc_shared((chunk,), accum_dtype)

            # load initial state column tv
            for j in T.serial(key_dim):
                h[j, tv] = h0[bb, vh, j, tv]
            T.tvm_storage_sync("shared")

            for ct in T.serial(NT):
                s0 = ct * chunk
                # load q,k for this chunk (L2-norm to match Kernel A), gcs, v_new base.
                # thread tv loads column tv across all rows.
                for r in T.serial(chunk):
                    qn[r, tv] = q[bb, s0 + r, kh, tv].astype(accum_dtype)
                    kn[r, tv] = k[bb, s0 + r, kh, tv].astype(accum_dtype)
                if tv == 0:
                    for r in T.serial(chunk):
                        gcs_s[r] = gcs[bb, s0 + r, vh]
                T.tvm_storage_sync("shared")
                # L2-norm q/k per row (row r, over key_dim) — single-thread per row via tv==r style.
                # thread tv normalizes row tv if tv<chunk (chunk==64<=val_dim==128).
                if tv < chunk:
                    nq = T.alloc_local((1,), accum_dtype)
                    nk = T.alloc_local((1,), accum_dtype)
                    T.clear(nq)
                    T.clear(nk)
                    for d in T.serial(key_dim):
                        nq[0] += qn[tv, d] * qn[tv, d]
                        nk[0] += kn[tv, d] * kn[tv, d]
                    rq = T.rsqrt(nq[0] + 1e-12) * scale
                    rk = T.rsqrt(nk[0] + 1e-12)
                    for d in T.serial(key_dim):
                        qn[tv, d] = qn[tv, d] * rq
                        kn[tv, d] = kn[tv, d] * rk
                T.tvm_storage_sync("shared")

                # v_new[r, tv] = u[r,tv] - sum_j w[r,j]*h[j,tv]
                for r in T.serial(chunk):
                    acc = T.alloc_local((1,), accum_dtype)
                    T.clear(acc)
                    for j in T.serial(key_dim):
                        acc[0] += w[bb, s0 + r, vh, j] * h[j, tv]
                    vnew[r, tv] = u[bb, s0 + r, vh, tv] - acc[0]
                T.tvm_storage_sync("shared")

                # output row r, column tv:
                #   o_inter = (sum_j q[r,j]*h[j,tv]) * exp(gcs_r)
                #   o_intra = sum_{p<r} ( (q_r . k_p) * exp(gcs_r-gcs_p) ) * v_new[p,tv]
                #   out = (o_inter + o_intra) * scale
                for r in T.serial(chunk):
                    o_inter = T.alloc_local((1,), accum_dtype)
                    T.clear(o_inter)
                    for j in T.serial(key_dim):
                        o_inter[0] += qn[r, j] * h[j, tv]
                    o_inter[0] = o_inter[0] * T.exp(gcs_s[r])
                    o_intra = T.alloc_local((1,), accum_dtype)
                    T.clear(o_intra)
                    for p in T.serial(chunk):
                        qk = T.alloc_local((1,), accum_dtype)
                        T.clear(qk)
                        for d in T.serial(key_dim):
                            qk[0] += qn[r, d] * kn[p, d]
                        contrib = T.if_then_else(
                            p < r,
                            qk[0] * T.exp(gcs_s[r] - gcs_s[p]) * vnew[p, tv],
                            0.0,
                        )
                        o_intra[0] += contrib
                    out[bb, s0 + r, vh, tv] = (o_inter[0] + o_intra[0]) * scale
                T.tvm_storage_sync("shared")

                # state update: h[j,tv] = h[j,tv]*exp(gcs_last) + sum_r k[r,j]*v_new[r,tv]
                g_last = gcs_s[chunk - 1]
                for j in T.serial(key_dim):
                    upd = T.alloc_local((1,), accum_dtype)
                    T.clear(upd)
                    for r in T.serial(chunk):
                        upd[0] += kn[r, j] * vnew[r, tv]
                    h[j, tv] = h[j, tv] * T.exp(g_last) + upd[0]
                T.tvm_storage_sync("shared")

            # write final state column tv
            for j in T.serial(key_dim):
                hT[bb, vh, j, tv] = h[j, tv]

    return kernel


def _self_check():
    import torch

    dev = "cuda"
    torch.manual_seed(0)
    B, NKH, NVH, KD, VD, C = 1, QWEN36_27B_NUM_KEY_HEADS, QWEN36_27B_NUM_VALUE_HEADS, \
        QWEN36_27B_KEY_DIM, QWEN36_27B_VAL_DIM, QWEN36_27B_CHUNK
    S = C * 2
    scale = KD ** -0.5

    q = torch.randn(B, S, NKH, KD, dtype=torch.bfloat16, device=dev)
    k = torch.randn(B, S, NKH, KD, dtype=torch.bfloat16, device=dev)
    u = torch.randn(B, S, NVH, VD, dtype=torch.float32, device=dev) * 0.1
    w = torch.randn(B, S, NVH, KD, dtype=torch.float32, device=dev) * 0.1
    g = -torch.rand(B, S, NVH, dtype=torch.float32, device=dev) * 0.1
    # gcs is the chunk-local cumsum of g (what Kernel A emits); Kernel B consumes it.
    gcs_in = torch.zeros_like(g)
    for ct in range(S // C):
        sl = slice(ct * C, (ct + 1) * C)
        gcs_in[:, sl] = torch.cumsum(g[:, sl], dim=1)
    h0 = torch.randn(B, NVH, KD, VD, dtype=torch.float32, device=dev) * 0.1

    # reference
    out_ref = torch.zeros(B, S, NVH, VD, dtype=torch.float32, device=dev)
    hT_ref = torch.zeros(B, NVH, KD, VD, dtype=torch.float32, device=dev)
    for b in range(B):
        for vh in range(NVH):
            kh = vh * NKH // NVH
            h = h0[b, vh].clone()
            for ct in range(S // C):
                sl = slice(ct * C, (ct + 1) * C)
                qc = q[b, sl, kh].float()
                kc = k[b, sl, kh].float()
                qc = qc * torch.rsqrt((qc * qc).sum(-1, keepdim=True) + 1e-12) * scale
                kc = kc * torch.rsqrt((kc * kc).sum(-1, keepdim=True) + 1e-12)
                uc = u[b, sl, vh]
                wc = w[b, sl, vh]
                gcs = gcs_in[b, sl, vh]
                vnew = uc - wc @ h                       # [C,VD]
                o_inter = (qc @ h) * torch.exp(gcs)[:, None]
                qk = qc @ kc.T                           # [C,C]
                mask = torch.tril(torch.ones(C, C, device=dev), diagonal=-1)
                gdiff = gcs[:, None] - gcs[None, :]
                A = qk * torch.exp(gdiff) * mask
                o_intra = A @ vnew
                out_ref[b, sl, vh] = (o_inter + o_intra) * scale
                h = h * torch.exp(gcs[-1]) + kc.T @ vnew
            hT_ref[b, vh] = h

    kernel = tilelang.compile(prefill_scan_o(S=S, B=B), target="cuda")
    out_o = torch.zeros(B, S, NVH, VD, dtype=torch.bfloat16, device=dev)
    hT_o = torch.zeros(B, NVH, KD, VD, dtype=torch.float32, device=dev)
    kernel(q, k, u, w, gcs_in, h0, out_o, hT_o)
    torch.cuda.synchronize()

    for name, got, ref in (("out", out_o, out_ref), ("hT", hT_o, hT_ref)):
        gf, rf = got.float(), ref.float()
        within = ((gf - rf).abs() <= 3e-2 + 3e-2 * rf.abs()).float().mean().item()
        print(f"{name}: {within*100:.2f}% within tol, abs_max={(gf-rf).abs().max().item():.4e}")
        assert within > 0.99, f"{name} mismatch"
    print("prefill_scan_o self-check PASSED")


if __name__ == "__main__":
    _self_check()
