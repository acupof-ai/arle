#!/usr/bin/env python3
"""ARLE local TileLang Metal dev backend smoke.

This script deliberately imports TileLang from a local checkout, then checks
two things ARLE needs before a real Metal hot-path integration is useful:

1. ARLE's in-tree TileLang attention kernel lowers to Metal from this checkout.
2. A TileLang T.gemm kernel executes on MPS and matches torch matmul.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use a local TileLang checkout as ARLE's Metal dev backend smoke."
    )
    parser.add_argument(
        "--arle-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="ARLE repository root.",
    )
    parser.add_argument(
        "--tilelang-repo",
        type=Path,
        default=Path("/tmp/tilelang-metal-pr"),
        help="Local TileLang checkout to import.",
    )
    parser.add_argument(
        "--heads",
        default="32,8",
        help="ARLE HD128 attention head config as '<num_q_heads>,<num_kv_heads>'.",
    )
    parser.add_argument(
        "--gemm-size",
        type=int,
        default=64,
        help="Square T.gemm smoke size.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=10,
        help="Timed T.gemm iterations.",
    )
    return parser.parse_args()


def import_local_tilelang(tilelang_repo: Path):
    repo = tilelang_repo.resolve()
    if not repo.exists():
        raise SystemExit(f"TileLang checkout does not exist: {repo}")
    sys.path.insert(0, str(repo))

    import tilelang  # noqa: PLC0415
    from tilelang import tvm  # noqa: PLC0415

    actual = Path(tilelang.__file__).resolve()
    try:
        actual.relative_to(repo)
    except ValueError as exc:
        raise SystemExit(
            "Imported TileLang is not from the requested checkout: "
            f"requested={repo}, actual={actual}"
        ) from exc
    return tilelang, tvm, actual


def load_arle_kernel_module(arle_root: Path):
    kernel_path = (
        arle_root
        / "crates"
        / "cuda-kernels"
        / "tools"
        / "tilelang"
        / "batch_prefill_paged_hd128.py"
    )
    if not kernel_path.exists():
        raise SystemExit(f"Missing ARLE TileLang kernel: {kernel_path}")

    spec = importlib.util.spec_from_file_location("arle_tilelang_hd128", kernel_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load ARLE TileLang kernel: {kernel_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, kernel_path


def lower_arle_attention(tilelang, tvm, arle_root: Path, heads: str) -> int:
    try:
        num_q_heads_raw, num_kv_heads_raw = heads.split(",", 1)
        num_q_heads = int(num_q_heads_raw)
        num_kv_heads = int(num_kv_heads_raw)
    except ValueError as exc:
        raise SystemExit("--heads must be '<num_q_heads>,<num_kv_heads>'") from exc

    module, kernel_path = load_arle_kernel_module(arle_root)
    prim_func = module.get_kernel(num_q_heads, num_kv_heads)
    with tvm.transform.PassContext(), tvm.target.Target("metal"):
        artifact = tilelang.lower(prim_func, target="metal")

    source = artifact.kernel_source or ""
    if "kernel void" not in source:
        raise SystemExit("ARLE attention kernel lowered, but no Metal kernel source was emitted.")
    if "threadgroup" not in source:
        raise SystemExit("ARLE attention Metal source is missing threadgroup memory.")

    print(
        "lower_arle_attention "
        f"kernel={kernel_path.relative_to(arle_root)} "
        f"heads=q{num_q_heads}_kv{num_kv_heads} "
        f"kernel_source_len={len(source)}"
    )
    return len(source)


def build_t_gemm_kernel(tilelang, T, size: int):
    block = 16

    @tilelang.jit(target="metal")
    def matmul_with_t_gemm():
        @T.prim_func
        def gemm(
            A: T.Tensor((size, size), T.float32),
            B: T.Tensor((size, size), T.float32),
            C: T.Tensor((size, size), T.float32),
        ):
            with T.Kernel(T.ceildiv(size, block), T.ceildiv(size, block), threads=128) as (
                bx,
                by,
            ):
                A_shared = T.alloc_shared((block, block), T.float32, scope="shared")
                B_shared = T.alloc_shared((block, block), T.float32, scope="shared")
                C_local = T.alloc_fragment((block, block), T.float32)

                T.clear(C_local)
                for ko in T.Pipelined(T.ceildiv(size, block), num_stages=0):
                    T.copy(A[by * block, ko * block], A_shared, coalesced_width=2)
                    T.copy(B[ko * block, bx * block], B_shared, coalesced_width=2)
                    T.gemm(A_shared, B_shared, C_local)
                T.copy(C_local, C[by * block, bx * block], coalesced_width=2)

        return gemm

    return matmul_with_t_gemm()


def run_t_gemm(tilelang, gemm_size: int, iters: int) -> tuple[float, float]:
    if gemm_size <= 0 or gemm_size % 16 != 0:
        raise SystemExit("--gemm-size must be a positive multiple of 16.")
    if iters <= 0:
        raise SystemExit("--iters must be positive.")

    import torch  # noqa: PLC0415
    import tilelang.language as T  # noqa: PLC0415

    if not torch.backends.mps.is_available():
        raise SystemExit("Torch MPS is unavailable; cannot execute TileLang Metal kernel.")

    kernel = build_t_gemm_kernel(tilelang, T, gemm_size)
    a = torch.randn(gemm_size, gemm_size, dtype=torch.float32, device="mps")
    b = torch.randn(gemm_size, gemm_size, dtype=torch.float32, device="mps")
    c = torch.zeros(gemm_size, gemm_size, dtype=torch.float32, device="mps")

    kernel(a, b, c)
    torch.mps.synchronize()
    max_abs = float((a @ b - c).abs().max().item())
    if max_abs > 1e-3:
        raise SystemExit(f"TileLang Metal T.gemm mismatch: max_abs={max_abs:.6g}")

    for _ in range(2):
        kernel(a, b, c)
    torch.mps.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        kernel(a, b, c)
    torch.mps.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / iters
    gflops = (2.0 * gemm_size * gemm_size * gemm_size) / (elapsed_ms * 1.0e6)

    source = kernel.kernel_source or ""
    if "kernel void" not in source:
        raise SystemExit("TileLang T.gemm did not expose Metal kernel source.")
    if "threadIdx.x) == 0" in source:
        raise SystemExit("TileLang T.gemm lowered to a single-thread scalar stub.")

    print(
        "run_t_gemm "
        f"n={gemm_size} "
        f"max_abs={max_abs:.6g} "
        f"tilelang_ms={elapsed_ms:.3f} "
        f"tilelang_gflops={gflops:.2f} "
        f"kernel_source_len={len(source)}"
    )
    return max_abs, elapsed_ms


def main() -> int:
    args = parse_args()
    arle_root = args.arle_root.resolve()
    tilelang, tvm, tilelang_file = import_local_tilelang(args.tilelang_repo)

    print("ARLE TileLang Metal dev backend")
    print(f"arle_root={arle_root}")
    print(f"tilelang_repo={args.tilelang_repo.resolve()}")
    print(f"tilelang_file={tilelang_file}")

    lower_arle_attention(tilelang, tvm, arle_root, args.heads)
    run_t_gemm(tilelang, args.gemm_size, args.iters)
    print("status=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
