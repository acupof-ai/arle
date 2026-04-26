"""TileLang AOT generator (TileLang 0.1.9-compatible).

TileLang 0.1.9 emits a TVM-FFI shared object instead of a raw cubin, so
the previous "lift `compiled.cubin_path` then embed" recipe no longer
applies. The new pipeline:

  1. Run `tilelang.compile()` — TileLang writes the codegen'd
     `device_kernel.cu` into its `~/.tilelang/cache/...` directory.
  2. Locate that .cu, parse the `kernel_kernel(...)` signature so we
     know the exact arg list TileLang baked in (8 tensors + N
     auto-promoted int32 shape params).
  3. nvcc-compile the .cu into a raw cubin against TileLang's own
     `tl_templates/cuda` and the bundled cutlass headers.
  4. Generate a C wrapper that exposes a stable signature to Rust
     (`<name>_cuda(q, q_indptr, k_pool, v_pool, kv_indptr, kv_indices,
     kv_last_page_len, o, batch_size, total_q_tokens, max_qlen,
     num_pages, total_pages, num_q_heads, num_kv_heads, page_size,
     sm_scale, stream)`) and drives `cuLaunchKernel` with the parsed
     argument order — duplicating user inputs into TileLang's
     `<name>_1` / `<name>_plus_one` slots as needed.

Inputs (CLI flags):
  --kernel-path    : .py module exposing get_kernel(num_q_heads, num_kv_heads)
  --kernel-name    : C function name (the wrapper exports `<name>_cuda`)
  --out-dir        : directory to write CUBIN + C wrapper into
  --target         : "cuda -arch=sm_<sm>" (e.g. "cuda -arch=sm_90")
  --out-name       : basename for the generated artifacts
  --num-q-heads    : Q-head count this AOT specialization is for
  --num-kv-heads   : KV-head count this AOT specialization is for
  --cuda-arch      : SM arch number (e.g. 89, 90) for the nvcc -gencode flag
  --tilelang-src   : tilelang/src directory (for tl_templates includes)
  --cutlass-include: cutlass/include directory bundled with TileLang
  --cuda-include   : CUDA toolkit include dir

Outputs (stdout, one per line, parsed by build.rs):
  FUNC_NAME=<exported C function>
  C_PATH=<absolute path to generated .c wrapper>
"""

import argparse
import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path

# TileLang 0.1.9 emits one int32 arg per (symbolic shape var × use site).
# The user-facing wrapper takes a small fixed signature; this map carries
# the user input → kernel arg name → expression so we can fill the args[]
# array in whatever order TileLang chose. Names that don't appear in the
# kernel signature are silently skipped — the kernel may have specialized
# them away if they were not load-bearing.
WRAPPER_FILL_RULES = {
    # name in kernel sig          C expression in the generated wrapper
    "batch_size": "batch_size",
    "batch_size_1": "batch_size",
    "batch_size_plus_one": "(batch_size + 1)",
    "batch_size_plus_one_1": "(batch_size + 1)",
    "max_qlen": "max_qlen",
    "max_qlen_1": "max_qlen",
    "num_pages": "num_pages",
    "num_pages_1": "num_pages",
    "total_pages": "total_pages",
    "total_pages_1": "total_pages",
    "total_q_tokens": "total_q_tokens",
    "total_q_tokens_1": "total_q_tokens",
}

# The 8 tensor-pointer parameter names TileLang generates for
# batch_prefill_paged_hd128 (alphabetical, matches the .cu order).
TENSOR_NAME_TO_USER_INPUT = {
    "KV_indices": "kv_indices",
    "KV_indptr": "kv_indptr",
    "KV_last_page_len": "kv_last_page_len",
    "K_pool": "k_pool",
    "Output": "o",
    "Q": "q",
    "Q_indptr": "q_indptr",
    "V_pool": "v_pool",
}


def load_kernel(kernel_path: str, num_q_heads: int, num_kv_heads: int):
    spec = importlib.util.spec_from_file_location("tilelang_kernel_module", kernel_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load kernel module from {kernel_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "get_kernel"):
        raise RuntimeError(
            f"{kernel_path} must expose get_kernel(num_q_heads, num_kv_heads)"
        )
    return module.get_kernel(num_q_heads, num_kv_heads)


def parse_target(target: str) -> str:
    if not target.startswith("cuda"):
        raise ValueError(f"unsupported TileLang AOT target: {target}")
    return target


def compile_kernel(prim_func, target: str):
    """Run TileLang compile() and return (device_source, kernel_func_name, parsed_args).

    `device_source` is the codegen'd CUDA source (the body of
    `device_kernel.cu` in TileLang's cache layout). We pull it from
    `adapter.device_kernel_source` because TileLang 0.1.9 only writes
    the cache file once the kernel has been *executed*; right after
    `compile()` the source lives in memory only and `adapter.libpath`
    is `None`.

    `parsed_args` is the ordered list of (kind, name) tuples extracted
    from the kernel signature, where kind is "tensor" or "scalar".
    """
    try:
        import tilelang
    except ImportError as exc:
        raise RuntimeError(
            "TileLang is not installed in the active Python interpreter. "
            "Bootstrap with `pip install -e .[tilelang]` or set "
            "INFER_TILELANG_PYTHON to an interpreter that has tilelang."
        ) from exc

    compiled = tilelang.compile(prim_func, target=target)
    adapter = getattr(compiled, "adapter", None)
    device_source = getattr(adapter, "device_kernel_source", None) if adapter is not None else None
    host_source = getattr(adapter, "host_kernel_source", None) if adapter is not None else None
    if not device_source:
        adapter_attrs = sorted(a for a in dir(adapter) if not a.startswith("_")) if adapter is not None else []
        raise RuntimeError(
            "TileLang JITKernel did not expose adapter.device_kernel_source. "
            f"compiled type={type(compiled).__name__!r}, "
            f"adapter type={type(adapter).__name__!r}, "
            f"adapter attrs: {adapter_attrs}. "
            "TileLang ABI changed — update gen_tilelang_aot.py."
        )

    # Extract the dynamic-shared-memory size TileLang baked into the kernel.
    # The host source's TVM FFI call sequence stores it as the trailing
    # int64 argument before the stream slot (host_kernel.cu line of the
    # form `[N].v_int64) = ((int64_t)<bytes>);` immediately followed by
    # the stream `[N+1].v_int64) = (int64_t)0` slot).
    dyn_shmem_match = re.search(
        r'v_int64\)\s*=\s*\(\(int64_t\)(\d+)\)\s*;\s*'
        r'\(\(\(TVMFFIAny\*\)stack_ffi_any\)\[\d+\]\.type_index\)\s*=\s*0\s*;',
        host_source or "",
    )
    if dyn_shmem_match is None:
        raise RuntimeError(
            "Could not extract dynamic shared-memory size from host_kernel_source. "
            "TileLang ABI changed — update gen_tilelang_aot.py."
        )
    dyn_shmem_bytes = int(dyn_shmem_match.group(1))

    match = re.search(
        r'extern "C" __global__ void __launch_bounds__\([^)]+\) (\w+)\((.*?)\)\s*\{',
        device_source,
        re.DOTALL,
    )
    if match is None:
        match = re.search(
            r'extern "C" __global__ void (\w+)\((.*?)\)\s*\{',
            device_source,
            re.DOTALL,
        )
    if match is None:
        raise RuntimeError(
            "Could not find __global__ kernel declaration in device source"
        )

    kernel_func_name = match.group(1)
    raw_args = match.group(2)
    parsed = []
    for chunk in raw_args.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        # Strip __restrict__, qualifiers, etc. The interesting bits are
        # the trailing identifier and whether it has a `*`.
        is_pointer = "*" in chunk
        ident = chunk.rsplit(maxsplit=1)[-1].lstrip("*")
        if is_pointer:
            parsed.append(("tensor", ident))
        else:
            parsed.append(("scalar", ident))

    return device_source, kernel_func_name, parsed, dyn_shmem_bytes


def nvcc_compile_cubin(
    device_cu: Path,
    cubin_path: Path,
    cuda_arch: int,
    tilelang_src: Path,
    cutlass_include: Path,
    cuda_include: Path,
) -> None:
    """Compile a device_kernel.cu source file to a raw cubin via nvcc."""
    nvcc_bin = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    cmd = [
        nvcc_bin,
        "-cubin",
        "-O3",
        f"-gencode=arch=compute_{cuda_arch},code=sm_{cuda_arch}",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "-Xcompiler=-fPIC",
        f"-I{tilelang_src}",
        f"-I{cutlass_include}",
        f"-I{cuda_include}",
        "-DENABLE_BF16",
        f"-DCUDA_ARCH={cuda_arch}0",
        str(device_cu),
        "-o",
        str(cubin_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "nvcc failed to compile TileLang device_kernel.cu to cubin.\n"
            f"command: {' '.join(cmd)}\n"
            f"stderr:\n{result.stderr}"
        )


def _format_cubin_bytes(data: bytes) -> str:
    lines = []
    for i in range(0, len(data), 16):
        chunk = data[i:i + 16]
        lines.append("    " + ", ".join(f"0x{b:02x}" for b in chunk) + ",")
    return "\n".join(lines)


def _build_args_array(parsed_args):
    """Map TileLang's parsed arg list to user-input expressions.

    For tensor pointers, look up the user-facing input name. For scalars,
    use the WRAPPER_FILL_RULES table. Anything missing means the kernel
    needs a value the wrapper doesn't currently take — fail loudly.
    """
    lines = []
    for kind, name in parsed_args:
        if kind == "tensor":
            user = TENSOR_NAME_TO_USER_INPUT.get(name)
            if user is None:
                raise RuntimeError(
                    f"unknown tensor parameter {name!r} in TileLang kernel — "
                    "extend TENSOR_NAME_TO_USER_INPUT in gen_tilelang_aot.py."
                )
            lines.append(f"        &{user},")
        else:  # scalar
            expr = WRAPPER_FILL_RULES.get(name)
            if expr is None:
                raise RuntimeError(
                    f"unknown scalar parameter {name!r} in TileLang kernel — "
                    "extend WRAPPER_FILL_RULES in gen_tilelang_aot.py."
                )
            lines.append(f"        &args_{name},  // {expr}")
    return "\n".join(lines)


def _build_scalar_locals(parsed_args):
    """Emit `int args_<name> = <expr>;` for each scalar arg the kernel needs.

    Scalars must be addressable for `cuLaunchKernel`; storing each in a
    local variable gives a stable address regardless of whether the
    expression came from an arithmetic op (e.g. batch_size + 1).
    """
    lines = []
    for kind, name in parsed_args:
        if kind != "scalar":
            continue
        expr = WRAPPER_FILL_RULES[name]
        lines.append(f"    int args_{name} = {expr};")
    return "\n".join(lines)


def write_c_wrapper(
    c_path: Path,
    kernel_name: str,
    cubin_path: Path,
    kernel_symbol: str,
    parsed_args,
    dyn_shmem_bytes: int,
) -> None:
    """Emit the C wrapper that exposes <kernel_name>_cuda to Rust."""
    cubin_bytes = Path(cubin_path).read_bytes()
    cubin_array = _format_cubin_bytes(cubin_bytes)
    args_lines = _build_args_array(parsed_args)
    scalar_locals = _build_scalar_locals(parsed_args)
    src = f"""#include <cuda.h>
#include <stdint.h>

static CUmodule g_module = NULL;
static CUfunction g_function = NULL;
static const char *kFuncSymbol = "{kernel_symbol}";

static const unsigned char kCubinData[] = {{
{cubin_array}
}};
static const unsigned int kCubinSize = (unsigned int)sizeof(kCubinData);

static CUresult ensure_loaded(void) {{
    if (g_function != NULL) return CUDA_SUCCESS;
    (void)kCubinSize;
    CUresult r = cuModuleLoadData(&g_module, kCubinData);
    if (r != CUDA_SUCCESS) return r;
    r = cuModuleGetFunction(&g_function, g_module, kFuncSymbol);
    if (r != CUDA_SUCCESS) return r;
    // Lift the dynamic-shared-memory cap so >48KB launches are accepted on
    // sm_75+ — TileLang's prefill HD128 needs ~48KB, exactly at the
    // default per-block limit on Ada/Hopper.
    return cuFuncSetAttribute(
        g_function,
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        {dyn_shmem_bytes}
    );
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
    int32_t max_qlen,
    int32_t num_pages,
    int32_t total_pages,
    int32_t num_q_heads,
    int32_t num_kv_heads,
    int32_t page_size,
    float sm_scale,
    CUstream stream
) {{
    (void)num_kv_heads;
    (void)page_size;
    (void)sm_scale;
    CUresult r = ensure_loaded();
    if (r != CUDA_SUCCESS) return r;

    // Per-arg locals so cuLaunchKernel has stable addresses for the
    // scalar parameters TileLang baked into the kernel signature.
{scalar_locals}

    // Argument order is parsed from TileLang-codegen'd device_kernel.cu
    // — TileLang emits tensors then auto-promoted symbolic scalars in a
    // deterministic order; the parser table keeps us in lockstep.
    void *args[] = {{
{args_lines}
    }};

    // Grid dims: (per-request q-tile blocks, num_q_heads, batch_size).
    const int block_m = 64;
    int qlen = max_qlen > 0 ? max_qlen : 1;
    int grid_x = (qlen + block_m - 1) / block_m;
    int grid_y = num_q_heads;
    int grid_z = batch_size;

    return cuLaunchKernel(
        g_function,
        grid_x, grid_y, grid_z,
        128, 1, 1,
        {dyn_shmem_bytes}, stream, args, NULL
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
    parser.add_argument("--num-q-heads", type=int, required=True)
    parser.add_argument("--num-kv-heads", type=int, required=True)
    parser.add_argument("--cuda-arch", type=int, required=True,
                        help="SM arch number (e.g. 89 for L4, 90 for H100).")
    parser.add_argument("--tilelang-src", required=True,
                        help="tilelang/src dir (parent of tl_templates/).")
    parser.add_argument("--cutlass-include", required=True,
                        help="cutlass/include dir bundled inside the tilelang package.")
    parser.add_argument("--cuda-include", required=True,
                        help="CUDA toolkit include dir (e.g. /usr/local/cuda/include).")
    args = parser.parse_args()

    target = parse_target(args.target)
    prim_func = load_kernel(args.kernel_path, args.num_q_heads, args.num_kv_heads)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device_source, kernel_func_name, parsed_args, dyn_shmem_bytes = compile_kernel(prim_func, target)

    # Stage the in-memory device source into out_dir so build.rs's
    # rerun-if-changed sees a stable file path. Then nvcc-compile it.
    device_cu_staged = out_dir / f"{args.out_name}_device_kernel.cu"
    device_cu_staged.write_text(device_source)
    cubin_path = out_dir / f"{args.out_name}.cubin"
    nvcc_compile_cubin(
        device_cu_staged,
        cubin_path,
        args.cuda_arch,
        Path(args.tilelang_src),
        Path(args.cutlass_include),
        Path(args.cuda_include),
    )

    c_path = (out_dir / f"{args.out_name}.c").resolve()
    write_c_wrapper(
        c_path,
        args.kernel_name,
        cubin_path,
        kernel_func_name,
        parsed_args,
        dyn_shmem_bytes,
    )

    print(f"FUNC_NAME={args.kernel_name}_cuda")
    print(f"C_PATH={c_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
