"""TileLang AOT generator. Mirrors tools/triton/gen_triton_aot.py contract.

Inputs (CLI flags):
  --kernel-path : path to a .py module exposing get_kernel() -> tilelang prim_func
  --kernel-name : symbolic name; used to prefix the emitted C function
  --out-dir     : directory to write CUBIN + C wrapper into
  --target      : "cuda:<sm>" (e.g. "cuda:90"). The :32 warp-size suffix
                  used by Triton is not honored here — TileLang infers it.
  --out-name    : basename for the generated artifacts

Outputs (stdout, one per line, parsed by build.rs):
  FUNC_NAME=<exported C function>
  C_PATH=<absolute path to generated .c wrapper>

The generated C wrapper exposes a single
    extern "C" CUresult <kernel-name>_cuda(<params>, CUstream stream)
that loads the AOT CUBIN once and dispatches to the kernel. The exact
parameter list is hard-coded for the prefill HD128 contract because Phase 0
ships exactly one TileLang kernel; widening this to a generic dispatcher is
a Phase 1 concern.

If TileLang's installed version cannot AOT-export for the requested SM, this
script exits non-zero with a clear error per docs/plans/tilelang-integration.md
§5 risk #1.
"""

import argparse
import importlib.util
import sys
from pathlib import Path


def load_kernel(kernel_path: str):
    spec = importlib.util.spec_from_file_location("tilelang_kernel_module", kernel_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load kernel module from {kernel_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "get_kernel"):
        raise RuntimeError(
            f"{kernel_path} must expose get_kernel() returning a tilelang prim_func"
        )
    return module.get_kernel()


def parse_target(target: str) -> str:
    if not target.startswith("cuda:"):
        raise ValueError(f"unsupported TileLang AOT target: {target}")
    return target


def compile_kernel(prim_func, target: str, out_dir: Path, out_name: str):
    try:
        import tilelang
    except ImportError as exc:
        raise RuntimeError(
            "TileLang is not installed in the active Python interpreter. "
            "Bootstrap with `pip install -e .[tilelang]` or set "
            "INFER_TILELANG_PYTHON to an interpreter that has tilelang."
        ) from exc

    out_dir.mkdir(parents=True, exist_ok=True)

    # tilelang.compile signature varies across versions; try the common shape
    # first and fall back if it differs. Phase 0 H100 spike will pin the
    # version per docs/plans/tilelang-integration.md §6.
    try:
        compiled = tilelang.compile(prim_func, target=target)
    except TypeError:
        compiled = tilelang.compile(prim_func, out_idx=-1, target=target)

    # Locate the AOT cubin and host wrapper symbol. TileLang exposes either
    # `.lib_path` / `.kernel_name` (newer) or `.adapter` (older).
    cubin_path = None
    func_symbol = None
    for attr in ("kernel_lib_path", "lib_path", "cubin_path"):
        cand = getattr(compiled, attr, None)
        if cand:
            cubin_path = Path(cand)
            break
    for attr in ("kernel_name", "func_name", "name"):
        cand = getattr(compiled, attr, None)
        if cand:
            func_symbol = str(cand)
            break

    if cubin_path is None or func_symbol is None:
        raise RuntimeError(
            "TileLang compiled kernel did not expose a CUBIN path / kernel name "
            "(probed: kernel_lib_path/lib_path/cubin_path and "
            "kernel_name/func_name/name). Phase 0 risk gate hit — see "
            "docs/plans/tilelang-integration.md §5 risk #1."
        )

    # Stage the cubin into out_dir so build.rs `rerun-if-changed` is stable.
    staged_cubin = out_dir / f"{out_name}.cubin"
    if cubin_path.suffix in {".cubin", ".bin"}:
        staged_cubin.write_bytes(Path(cubin_path).read_bytes())
    else:
        # Some TileLang versions ship a shared object, not a raw cubin.
        # Phase 0 only supports raw cubin; fail loudly otherwise.
        raise RuntimeError(
            f"TileLang produced {cubin_path.suffix} ({cubin_path}); Phase 0 "
            "expects a raw .cubin. Pin a TileLang version that emits CUBIN, "
            "or extend this generator."
        )

    return func_symbol, staged_cubin


def write_c_wrapper(c_path: Path, kernel_name: str, cubin_path: Path, kernel_symbol: str) -> None:
    # The wrapper loads the cubin lazily on first call, then launches via
    # cuLaunchKernel. The grid dims are derived from runtime scalars, mirroring
    # how Triton AOT wrappers handle dynamic shapes. The contract matches the
    # Rust FFI declaration in crates/cuda-kernels/src/ffi/attention.rs.
    src = f"""#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static CUmodule g_module = NULL;
static CUfunction g_function = NULL;
static const char *kCubinPath = "{cubin_path.as_posix()}";
static const char *kFuncSymbol = "{kernel_symbol}";

static CUresult ensure_loaded(void) {{
    if (g_function != NULL) return CUDA_SUCCESS;
    CUresult r = cuModuleLoad(&g_module, kCubinPath);
    if (r != CUDA_SUCCESS) return r;
    return cuModuleGetFunction(&g_function, g_module, kFuncSymbol);
}}

CUresult {kernel_name}_cuda(
    uint16_t *q,
    const int32_t *q_indptr,
    uint16_t *k_pool,
    uint16_t *v_pool,
    const int32_t *kv_indptr,
    const int32_t *kv_indices,
    const int32_t *kv_last_page_len,
    uint16_t *o,
    int32_t batch_size,
    int32_t total_q_tokens,
    int32_t num_q_heads,
    int32_t num_kv_heads,
    int32_t page_size,
    float sm_scale,
    CUstream stream
) {{
    (void)num_q_heads;
    (void)num_kv_heads;
    (void)page_size;
    (void)sm_scale;
    (void)total_q_tokens;
    CUresult r = ensure_loaded();
    if (r != CUDA_SUCCESS) return r;

    void *args[] = {{
        &q, &q_indptr, &k_pool, &v_pool,
        &kv_indptr, &kv_indices, &kv_last_page_len, &o,
    }};

    // Grid dims: (q_tile_blocks, num_q_heads, batch_size). The Python kernel
    // uses block_m=64 for query rows; must match BLOCK_M in batch_prefill_paged_hd128.py.
    const int block_m = 64;
    int max_qlen = total_q_tokens; // upper bound; per-request rows are masked.
    int grid_x = (max_qlen + block_m - 1) / block_m;
    int grid_y = num_q_heads;
    int grid_z = batch_size;

    return cuLaunchKernel(
        g_function,
        grid_x, grid_y, grid_z,
        128, 1, 1,
        0, stream, args, NULL
    );
}}
"""
    c_path.write_text(src)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-path", required=True)
    parser.add_argument("--kernel-name", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--out-name", required=True)
    args = parser.parse_args()

    target = parse_target(args.target)
    prim_func = load_kernel(args.kernel_path)
    out_dir = Path(args.out_dir).resolve()

    func_symbol, cubin_path = compile_kernel(prim_func, target, out_dir, args.out_name)

    c_path = (out_dir / f"{args.out_name}.c").resolve()
    write_c_wrapper(c_path, args.kernel_name, cubin_path, func_symbol)

    print(f"FUNC_NAME={args.kernel_name}_cuda")
    print(f"C_PATH={c_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
