"""TileLang AOT generator (TileLang 0.1.9-compatible).

The generator compiles a TileLang ``T.prim_func`` into a raw cubin and emits a
small C wrapper with ARLE's stable CUDA FFI ABI.  It supports two kernel
families:

* ``attention``: paged prefill/decode kernels specialized by
  ``(num_q_heads, num_kv_heads)``.
* ``gdr``: the seven Qwen3.5 chunk-wise Gated Delta Rule stages, selected by
  ``--kernel-key``.

TileLang 0.1.9 emits TVM-FFI host/device source instead of a directly reusable
cubin.  We intentionally keep the pipeline explicit: compile with TileLang,
extract the codegen'd device source and launch metadata, nvcc it to cubin, then
embed that cubin in a stable C wrapper.  The wrapper maps TileLang's generated
argument order back to ARLE's C ABI by parsing the generated device signature.
"""

import argparse
import importlib.util
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class WrapperSpec:
    public_params: str
    tensor_inputs: dict[str, str]
    scalar_inputs: dict[str, tuple[str, str]]
    prelude: str
    grid: str
    block: str = "128, 1, 1"


ATTENTION_PUBLIC_PARAMS = """    uint16_t *q,
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
    CUstream stream"""

ATTENTION_SPEC = WrapperSpec(
    public_params=ATTENTION_PUBLIC_PARAMS,
    tensor_inputs={
        "KV_indices": "kv_indices",
        "KV_indptr": "kv_indptr",
        "KV_last_page_len": "kv_last_page_len",
        "K_pool": "k_pool",
        "Output": "o",
        "Q": "q",
        "Q_indptr": "q_indptr",
        "V_pool": "v_pool",
    },
    scalar_inputs={
        "batch_size": ("int32_t", "batch_size"),
        "max_qlen": ("int32_t", "max_qlen"),
        "num_pages": ("int32_t", "num_pages"),
        "total_pages": ("int32_t", "total_pages"),
        "total_q_tokens": ("int32_t", "total_q_tokens"),
    },
    prelude="""    (void)num_kv_heads;
    (void)page_size;
    (void)sm_scale;""",
    grid="""    const int block_m = 64;
    int qlen = max_qlen > 0 ? max_qlen : 1;
    int grid_x = (qlen + block_m - 1) / block_m;
    int grid_y = num_q_heads;
    int grid_z = batch_size;""",
)

# M_b.2 — FP8 E4M3 KV variant. K_pool / V_pool come in as `uint8_t*` (FP8
# bytes), and per-token / per-kv-head scales come in as `float*`. The kernel
# dequantizes inline before the GEMM (TileLang 0.1.9 disallows mixed-dtype
# GEMM). All other shape symbols + grid extents mirror ATTENTION_SPEC.
ATTENTION_FP8_PUBLIC_PARAMS = """    uint16_t *q,
    const int32_t *q_indptr,
    const uint8_t *k_pool,
    const uint8_t *v_pool,
    const float *k_scales,
    const float *v_scales,
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
    CUstream stream"""

ATTENTION_FP8_SPEC = WrapperSpec(
    public_params=ATTENTION_FP8_PUBLIC_PARAMS,
    tensor_inputs={
        "KV_indices": "kv_indices",
        "KV_indptr": "kv_indptr",
        "KV_last_page_len": "kv_last_page_len",
        "K_pool": "k_pool",
        "K_scales": "k_scales",
        "Output": "o",
        "Q": "q",
        "Q_indptr": "q_indptr",
        "V_pool": "v_pool",
        "V_scales": "v_scales",
    },
    scalar_inputs={
        "batch_size": ("int32_t", "batch_size"),
        "max_qlen": ("int32_t", "max_qlen"),
        "num_pages": ("int32_t", "num_pages"),
        "total_pages": ("int32_t", "total_pages"),
        "total_q_tokens": ("int32_t", "total_q_tokens"),
    },
    prelude="""    (void)num_kv_heads;
    (void)page_size;
    (void)sm_scale;""",
    grid="""    const int block_m = 64;
    int qlen = max_qlen > 0 ? max_qlen : 1;
    int grid_x = (qlen + block_m - 1) / block_m;
    int grid_y = num_q_heads;
    int grid_z = batch_size;""",
)

GDR_SCALAR_INPUTS = {
    "hv": ("int32_t", "num_value_heads"),
    "num_chunks": ("int32_t", "ceildiv_i32(seq_len, 64)"),
    "num_key_heads": ("int32_t", "num_key_heads"),
    "num_value_heads": ("int32_t", "num_value_heads"),
    "qkv_dim": ("int32_t", "qkv_dim"),
    "seq_len": ("int32_t", "seq_len"),
    "scale": ("float", "scale"),
}

GDR_SPECS = {
    "gdr_chunk_prepare": WrapperSpec(
        public_params="""    const uint16_t *qkv,
    const uint16_t *b_proj,
    const uint16_t *a_proj,
    const uint16_t *dt_bias,
    const float *a_log,
    uint16_t *q_out,
    uint16_t *k_out,
    uint16_t *v_out,
    float *g_out,
    float *beta_out,
    int32_t num_key_heads,
    int32_t num_value_heads,
    int32_t qkv_dim,
    int32_t seq_len,
    CUstream stream""",
        tensor_inputs={
            "qkv": "qkv",
            "b_proj": "b_proj",
            "a_proj": "a_proj",
            "dt_bias": "dt_bias",
            "a_log": "a_log",
            "q_out": "q_out",
            "k_out": "k_out",
            "v_out": "v_out",
            "g_out": "g_out",
            "beta_out": "beta_out",
        },
        scalar_inputs=GDR_SCALAR_INPUTS,
        prelude="",
        grid="""    int grid_x = seq_len;
    int grid_y = num_value_heads;
    int grid_z = 1;""",
    ),
    "gdr_chunk_cumsum": WrapperSpec(
        public_params="""    const float *g_in,
    float *g_out,
    int32_t seq_len,
    int32_t num_value_heads,
    CUstream stream""",
        tensor_inputs={"g_in": "g_in", "g_out": "g_out"},
        scalar_inputs=GDR_SCALAR_INPUTS,
        prelude="",
        grid="""    int grid_x = ceildiv_i32(seq_len, 64);
    int grid_y = num_value_heads;
    int grid_z = 1;""",
    ),
    "gdr_chunk_a": WrapperSpec(
        public_params="""    const uint16_t *k,
    const float *g_cumsum,
    const float *beta,
    float *a_tril,
    int32_t seq_len,
    int32_t num_value_heads,
    CUstream stream""",
        tensor_inputs={"k": "k", "g_cumsum": "g_cumsum", "beta": "beta", "a_tril": "a_tril"},
        scalar_inputs=GDR_SCALAR_INPUTS,
        prelude="",
        grid="""    int grid_x = ceildiv_i32(seq_len, 64);
    int grid_y = num_value_heads;
    int grid_z = 1;""",
    ),
    "gdr_chunk_solve": WrapperSpec(
        public_params="""    const float *a_tril,
    uint16_t *a_inv,
    int32_t seq_len,
    int32_t num_value_heads,
    CUstream stream""",
        tensor_inputs={"a_tril": "a_tril", "a_inv": "a_inv"},
        scalar_inputs=GDR_SCALAR_INPUTS,
        prelude="",
        grid="""    int grid_x = ceildiv_i32(seq_len, 64);
    int grid_y = num_value_heads;
    int grid_z = 1;""",
    ),
    "gdr_chunk_recompute": WrapperSpec(
        public_params="""    const uint16_t *k,
    const uint16_t *v,
    const float *beta,
    uint16_t *w,
    uint16_t *u,
    const uint16_t *a_inv,
    const float *g_cumsum,
    int32_t seq_len,
    int32_t num_value_heads,
    CUstream stream""",
        tensor_inputs={"k": "k", "v": "v", "beta": "beta", "w": "w", "u": "u", "a_inv": "a_inv", "g_cumsum": "g_cumsum"},
        scalar_inputs=GDR_SCALAR_INPUTS,
        prelude="",
        grid="""    int grid_x = ceildiv_i32(seq_len, 64);
    int grid_y = num_value_heads;
    int grid_z = 1;""",
    ),
    "gdr_chunk_state": WrapperSpec(
        public_params="""    const uint16_t *k,
    const uint16_t *w,
    const uint16_t *u,
    const float *g_cumsum,
    const float *initial_state,
    float *chunk_state,
    uint16_t *v_new,
    float *final_state,
    int32_t seq_len,
    int32_t num_value_heads,
    CUstream stream""",
        tensor_inputs={
            "k": "k",
            "w": "w",
            "u": "u",
            "g_cumsum": "g_cumsum",
            "initial_state": "initial_state",
            "chunk_state": "chunk_state",
            "v_new": "v_new",
            "final_state": "final_state",
        },
        scalar_inputs=GDR_SCALAR_INPUTS,
        prelude="",
        grid="""    int grid_x = 4;
    int grid_y = num_value_heads;
    int grid_z = 1;""",
    ),
    "gdr_chunk_o": WrapperSpec(
        public_params="""    const uint16_t *q,
    const uint16_t *k,
    const uint16_t *v_new,
    const float *chunk_state,
    const float *g_cumsum,
    uint16_t *output,
    int32_t seq_len,
    int32_t num_value_heads,
    float scale,
    CUstream stream""",
        tensor_inputs={"q": "q", "k": "k", "v_new": "v_new", "chunk_state": "chunk_state", "g_cumsum": "g_cumsum", "output": "output"},
        scalar_inputs=GDR_SCALAR_INPUTS,
        prelude="",
        grid="""    int grid_x = 4;
    int grid_y = ceildiv_i32(seq_len, 64);
    int grid_z = num_value_heads;""",
    ),
}


def load_module(kernel_path: str):
    spec = importlib.util.spec_from_file_location("tilelang_kernel_module", kernel_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load kernel module from {kernel_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_attention_kernel(kernel_path: str, num_q_heads: int, num_kv_heads: int):
    module = load_module(kernel_path)
    if not hasattr(module, "get_kernel"):
        raise RuntimeError(
            f"{kernel_path} must expose get_kernel(num_q_heads, num_kv_heads)"
        )
    return module.get_kernel(num_q_heads, num_kv_heads)


def load_gdr_kernel(kernel_path: str, kernel_key: str):
    module = load_module(kernel_path)
    if not hasattr(module, "get_kernel"):
        raise RuntimeError(f"{kernel_path} must expose get_kernel(name)")
    return module.get_kernel(kernel_key)


def parse_target(target: str) -> str:
    if not target.startswith("cuda"):
        raise ValueError(f"unsupported TileLang AOT target: {target}")
    return target


def compile_kernel(prim_func, target: str):
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
    if adapter is not None and hasattr(adapter, "get_device_source"):
        device_source = adapter.get_device_source()
    else:
        device_source = getattr(adapter, "device_kernel_source", None) if adapter is not None else None
    if adapter is not None and hasattr(adapter, "get_host_source"):
        host_source = adapter.get_host_source()
    else:
        host_source = getattr(adapter, "host_kernel_source", None) if adapter is not None else None
    if not device_source:
        adapter_attrs = sorted(dir(adapter)) if adapter is not None else []
        raise RuntimeError(
            "TileLang JITKernel did not expose adapter.device_kernel_source. "
            f"compiled type={type(compiled).__name__!r}, "
            f"adapter type={type(adapter).__name__!r}, "
            f"adapter attrs: {adapter_attrs}. "
            "TileLang ABI changed — update gen_tilelang_aot.py."
        )

    src = host_source or ""
    int_assign_re = re.compile(
        r'\(\(\(TVMFFIAny\*\)stack_ffi_any\)\[(\d+)\]\.v_int64\)\s*=\s*'
        r'\(\(int64_t\)(\d+)\)\s*;'
    )
    zero_assign_re = re.compile(
        r'\(\(\(TVMFFIAny\*\)stack_ffi_any\)\[(\d+)\]\.v_int64\)\s*=\s*'
        r'\(int64_t\)0\s*;'
    )
    int_by_slot = {int(m.group(1)): (int(m.group(2)), m.start()) for m in int_assign_re.finditer(src)}
    zero_slots = {int(m.group(1)) for m in zero_assign_re.finditer(src)}
    launch_call_pos = src.rfind("kernel_kernel_packed")
    candidates = [
        (slot, val)
        for slot, (val, pos) in int_by_slot.items()
        if (slot + 1) in zero_slots and pos < launch_call_pos
    ]
    if not candidates:
        raise RuntimeError(
            "Could not extract dynamic shared-memory size from host_kernel_source. "
            "TileLang ABI changed — update gen_tilelang_aot.py. "
            f"Diagnostics: host_source_len={len(src)}, "
            f"int_slots={sorted(int_by_slot)}, zero_slots={sorted(zero_slots)}, "
            f"launch_call_pos={launch_call_pos}, int_int_pairs_total={len(int_by_slot)}"
        )
    candidates.sort()
    dyn_shmem_bytes = candidates[-1][1]

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
        raise RuntimeError("Could not find __global__ kernel declaration in device source")

    parsed = []
    for chunk in match.group(2).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        is_pointer = "*" in chunk
        ident = chunk.rsplit(maxsplit=1)[-1].lstrip("*")
        parsed.append(("tensor" if is_pointer else "scalar", ident))

    return device_source, match.group(1), parsed, dyn_shmem_bytes


def nvcc_compile_cubin(
    device_cu: Path,
    cubin_path: Path,
    cuda_arch: int,
    tilelang_src: Path,
    cutlass_include: Path,
    cuda_include: Path,
) -> None:
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
    return "\n".join(
        "    " + ", ".join(f"0x{b:02x}" for b in data[i:i + 16]) + ","
        for i in range(0, len(data), 16)
    )


def _canonical_scalar_name(name: str) -> tuple[str, Callable[[str], str]]:
    stem = re.sub(r"_\d+$", "", name)
    if stem.endswith("_plus_one"):
        base = stem[: -len("_plus_one")]
        return base, lambda expr: f"({expr} + 1)"
    return stem, lambda expr: expr


def _scalar_expr(spec: WrapperSpec, name: str) -> tuple[str, str]:
    canonical, transform = _canonical_scalar_name(name)
    if canonical not in spec.scalar_inputs:
        raise RuntimeError(
            f"unknown scalar parameter {name!r} in TileLang kernel — "
            "extend the family WrapperSpec in gen_tilelang_aot.py."
        )
    ctype, expr = spec.scalar_inputs[canonical]
    return ctype, transform(expr)


def _build_args_array(parsed_args, spec: WrapperSpec) -> str:
    lines = []
    for kind, name in parsed_args:
        if kind == "tensor":
            user = spec.tensor_inputs.get(name)
            if user is None:
                raise RuntimeError(
                    f"unknown tensor parameter {name!r} in TileLang kernel — "
                    "extend the family WrapperSpec in gen_tilelang_aot.py."
                )
            lines.append(f"        &{user},")
        else:
            _scalar_expr(spec, name)
            lines.append(f"        &args_{name},")
    return "\n".join(lines)


def _build_scalar_locals(parsed_args, spec: WrapperSpec) -> str:
    lines = []
    seen = set()
    for kind, name in parsed_args:
        if kind != "scalar" or name in seen:
            continue
        seen.add(name)
        ctype, expr = _scalar_expr(spec, name)
        lines.append(f"    {ctype} args_{name} = {expr};")
    return "\n".join(lines)


def write_c_wrapper(
    c_path: Path,
    kernel_name: str,
    cubin_path: Path,
    kernel_symbol: str,
    parsed_args,
    dyn_shmem_bytes: int,
    spec: WrapperSpec,
) -> None:
    cubin_array = _format_cubin_bytes(Path(cubin_path).read_bytes())
    args_lines = _build_args_array(parsed_args, spec)
    scalar_locals = _build_scalar_locals(parsed_args, spec)
    src = f"""#include <cuda.h>
#include <stdint.h>

static CUmodule g_module = NULL;
static CUfunction g_function = NULL;
static const char *kFuncSymbol = "{kernel_symbol}";

static const unsigned char kCubinData[] = {{
{cubin_array}
}};
static const unsigned int kCubinSize = (unsigned int)sizeof(kCubinData);

static int32_t ceildiv_i32(int32_t n, int32_t d) {{
    return (n + d - 1) / d;
}}

static CUresult ensure_loaded(void) {{
    if (g_function != NULL) return CUDA_SUCCESS;
    (void)kCubinSize;
    CUresult r = cuModuleLoadData(&g_module, kCubinData);
    if (r != CUDA_SUCCESS) return r;
    r = cuModuleGetFunction(&g_function, g_module, kFuncSymbol);
    if (r != CUDA_SUCCESS) return r;
    return cuFuncSetAttribute(
        g_function,
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        {dyn_shmem_bytes}
    );
}}

CUresult {kernel_name}_cuda(
{spec.public_params}
) {{
{spec.prelude}
    CUresult r = ensure_loaded();
    if (r != CUDA_SUCCESS) return r;

{scalar_locals}

    void *args[] = {{
{args_lines}
    }};

{spec.grid}

    return cuLaunchKernel(
        g_function,
        grid_x, grid_y, grid_z,
        {spec.block},
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
    parser.add_argument(
        "--kernel-family",
        choices=["attention", "attention_fp8", "gdr"],
        default="attention",
    )
    parser.add_argument("--kernel-key")
    parser.add_argument("--num-q-heads", type=int)
    parser.add_argument("--num-kv-heads", type=int)
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
    if args.kernel_family == "attention":
        if args.num_q_heads is None or args.num_kv_heads is None:
            raise RuntimeError("attention kernels require --num-q-heads and --num-kv-heads")
        prim_func = load_attention_kernel(args.kernel_path, args.num_q_heads, args.num_kv_heads)
        wrapper_spec = ATTENTION_SPEC
    elif args.kernel_family == "attention_fp8":
        if args.num_q_heads is None or args.num_kv_heads is None:
            raise RuntimeError("attention_fp8 kernels require --num-q-heads and --num-kv-heads")
        prim_func = load_attention_kernel(args.kernel_path, args.num_q_heads, args.num_kv_heads)
        wrapper_spec = ATTENTION_FP8_SPEC
    else:
        if not args.kernel_key:
            raise RuntimeError("gdr kernels require --kernel-key")
        if args.kernel_key not in GDR_SPECS:
            raise RuntimeError(f"unknown GDR kernel key {args.kernel_key!r}; valid keys: {sorted(GDR_SPECS)}")
        prim_func = load_gdr_kernel(args.kernel_path, args.kernel_key)
        wrapper_spec = GDR_SPECS[args.kernel_key]

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device_source, kernel_func_name, parsed_args, dyn_shmem_bytes = compile_kernel(prim_func, target)
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
        wrapper_spec,
    )

    print(f"FUNC_NAME={args.kernel_name}_cuda")
    print(f"C_PATH={c_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
