"""Stage S3.0 — standalone strict-lower triangular inverse (solve_tril).

The hardest, most independent piece of the S3 chunkwise-prefill GDN kernel:
given a 64x64 strictly-lower-triangular matrix L, compute (I + L)^{-1}.

Because L is strictly lower triangular it is nilpotent (L^64 = 0), so the inverse
is exact:  (I+L)^{-1} = I - L + L^2 - ... , equivalently row-by-row forward
substitution:  M = (I+L)^{-1}  satisfies  M[i] = e_i - sum_{j<i} L[i,j] * M[j].

This first version does a single-block serial forward substitution (one warp,
MMA path — 64x64 is below the WGMMA M>=64 floor and this is control-heavy, not
GEMM-bound). Correctness first; the KDA-style 4x(16x16) blocked variant
(examples/kda/chunk_inter_solve_fused.py) is an optimization for later.

Validation: M @ (I + L) ≈ I.
"""

import tilelang
import tilelang.language as T

CHUNK = 64


def solve_tril(chunk: int = CHUNK, accum_dtype: str = "float32"):
    """Build a kernel that inverts (I + StrictLower(L)) for a `chunk`x`chunk` block.

    Grid = (B,), one block per matrix, `chunk` threads (thread i owns row i of M).
    """

    @T.prim_func
    def kernel(
        L: T.Tensor((1, chunk, chunk), accum_dtype),   # strict-lower (diag/upper ignored)
        M: T.Tensor((1, chunk, chunk), accum_dtype),   # output inverse
    ):
        with T.Kernel(1, threads=chunk) as _:
            ti = T.get_thread_binding(0)  # row this thread owns
            Ls = T.alloc_shared((chunk, chunk), accum_dtype)
            Ms = T.alloc_shared((chunk, chunk), accum_dtype)

            # load L (strict-lower only), init M = I
            for j in T.serial(chunk):
                Ls[ti, j] = T.if_then_else(j < ti, L[0, ti, j], 0.0)
                Ms[ti, j] = T.if_then_else(j == ti, 1.0, 0.0)
            T.tvm_storage_sync("shared")

            # forward substitution: rows resolved in order. Row i depends only on
            # rows j<i, so after row (r-1) is final, thread r can compute its row.
            # M[i,:] = e_i - sum_{j<i} L[i,j] * M[j,:]
            for r in T.serial(chunk):
                if ti == r:
                    acc = T.alloc_local((1,), accum_dtype)
                    for c in T.serial(chunk):
                        T.clear(acc)
                        for j in T.serial(chunk):
                            # only j<r contribute (Ls is strict-lower, zero elsewhere)
                            acc[0] += Ls[r, j] * Ms[j, c]
                        Ms[r, c] = T.if_then_else(c == r, 1.0, 0.0) - acc[0]
                T.tvm_storage_sync("shared")

            for j in T.serial(chunk):
                M[0, ti, j] = Ms[ti, j]

    return kernel


def _self_check():
    import torch

    dev = "cuda"
    torch.manual_seed(0)
    # Realistic GDN scale: L = beta*exp(g_i-g_j)*(K K^T) has small entries. A
    # randn-scale 64x64 strict-lower matrix is pathologically ill-conditioned
    # (its inverse grows exponentially down the rows), which is not the operating
    # regime — use small entries so the residual reflects the algorithm, not
    # input conditioning.
    L = torch.tril(torch.randn(CHUNK, CHUNK, dtype=torch.float32, device=dev) * 0.05, diagonal=-1)
    A = torch.eye(CHUNK, device=dev) + L
    M_ref = torch.linalg.inv(A)

    kernel = tilelang.compile(solve_tril(CHUNK), target="cuda")
    M = torch.zeros(1, CHUNK, CHUNK, dtype=torch.float32, device=dev)
    kernel(L.view(1, CHUNK, CHUNK).contiguous(), M)
    torch.cuda.synchronize()
    Mk = M[0]

    # direct: compare to torch inverse
    err_inv = (Mk - M_ref).abs().max().item()
    # residual: M @ A ≈ I
    resid = (Mk @ A - torch.eye(CHUNK, device=dev)).abs().max().item()
    print(f"solve_tril: max|M-inv|={err_inv:.3e}  max|MA-I|={resid:.3e}")
    assert resid < 1e-3, f"residual too large: {resid}"
    print("solve_tril self-check PASSED")


if __name__ == "__main__":
    _self_check()
