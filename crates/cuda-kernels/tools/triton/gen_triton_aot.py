#!/usr/bin/env python3
"""Triton AOT generator for ARLE's CUDA kernel lane.

Programmatically AOT-compiles a `@triton.jit` kernel to a cubin and emits ONE
self-contained C file that embeds the cubin bytes and exposes a launch wrapper
returning a `CUresult` (no `exit()` on error — unlike triton's
`tools/extra/cuda/compile.c` template, which we deliberately do not copy).

Modeled on triton's upstream `tools/compile.py` (signature grammar) and on this
repo's `tools/tilelang/gen_tilelang_aot.py` (FUNC_NAME=/C_PATH= stdout contract,
cubin-byte emission, idempotent loader). The pod's working programmatic compile
recipe (triton 3.5.1) is reproduced exactly:

    target  = GPUTarget("cuda", int(arch), 32)   # int arch; CLI --target is
                                                  # broken upstream (str-vs-int)
    backend = make_backend(target)
    opts    = backend.parse_options({...})
    src     = ASTSource(fn, signature, constexprs, attrs)
    cc      = triton.compile(src, target=target, options=opts.__dict__)
    cubin   = cc.asm[backend.binary_ext]
    name    = cc.metadata.name        # unmangled entry symbol
    shared  = cc.metadata.shared

We assert `global_scratch_size == 0` and `profile_scratch_size == 0` (upstream
refuses an AOT launch with non-zero scratch). The launch contract appends two
trailing `CUdeviceptr 0` args (global_scratch, profile_scratch) after the
kernel args, and the C wrapper takes `gX, gY, gZ` FIRST and `stream` LAST
(repo FFI convention).

Signature grammar (subset of triton's, sufficient for ARLE's GDN kernels):
  * `*<dtype>[:N]`   pointer; `:N` = tt.divisibility hint (typically 16).
                     Emitted as a runtime `CUdeviceptr` arg.
  * `<dtype>`        runtime scalar (e.g. `fp32`, `i32`). Emitted as a runtime
                     `float`/`int32_t` arg.
  * `<int|float>`    a literal = baked constexpr; removed from the C prototype
                     and passed to the kernel as a compile-time constant.
  * `"<string>"`     a baked string constexpr (e.g. activation "swish").
  * `none`           a baked `None` constexpr; removed from the C prototype.
                     Triton treats a `None` constant specially: every
                     `if <param> is not None` branch DCEs at compile time. Used
                     to specialize away the optional pointer args of SGLang's
                     fused_moe_kernel (a_desc/b_desc/bias_ptr/add_mask_ptr and
                     the unused-on-bf16 scale ptrs) so the bf16 cubin carries no
                     TMA/quant code.

Argument ORDER in the signature string MUST match the kernel's parameter order;
constexpr parameters appear in that same positional order as literals.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import triton  # noqa: E402  (import after argparse docstring is fine)

# triton 3.5.x public compiler surface.
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource, make_backend


# ---- dtype maps -----------------------------------------------------------

# triton signature dtype -> (C scalar type for runtime args, is_pointer).
# Pointers are always passed as CUdeviceptr regardless of pointee dtype.
_SCALAR_CTYPE = {
    "i1": "int32_t",
    "i8": "int32_t",
    "i16": "int32_t",
    "i32": "int32_t",
    "i64": "int64_t",
    "u32": "uint32_t",
    "u64": "uint64_t",
    "fp16": "float",
    "bf16": "float",
    "fp32": "float",
    "fp64": "double",
}


def _is_int_literal(tok: str) -> bool:
    try:
        int(tok, 0)
        return True
    except ValueError:
        return False


def _is_float_literal(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return False


def _parse_signature(sig: str):
    """Parse a comma-separated signature into ordered argument descriptors.

    Returns a list of dicts, one per kernel parameter (in order):
      {"kind": "pointer", "dtype": <str>, "divisibility": <int|None>}
      {"kind": "scalar",  "dtype": <str>}
      {"kind": "constexpr_int",    "value": <int>}
      {"kind": "constexpr_float",  "value": <float>}
      {"kind": "constexpr_str",    "value": <str>}
    """
    descriptors = []
    for raw in sig.split(","):
        tok = raw.strip()
        if not tok:
            continue
        if tok.startswith("*"):
            body = tok[1:]
            div = None
            if ":" in body:
                dtype, div_s = body.split(":", 1)
                div = int(div_s)
            else:
                dtype = body
            descriptors.append(
                {"kind": "pointer", "dtype": dtype.strip(), "divisibility": div}
            )
        elif tok == "none" or tok == "None":
            descriptors.append({"kind": "constexpr_none"})
        elif tok.startswith('"') and tok.endswith('"'):
            descriptors.append({"kind": "constexpr_str", "value": tok[1:-1]})
        elif tok in _SCALAR_CTYPE:
            descriptors.append({"kind": "scalar", "dtype": tok})
        elif _is_int_literal(tok):
            descriptors.append({"kind": "constexpr_int", "value": int(tok, 0)})
        elif _is_float_literal(tok):
            descriptors.append({"kind": "constexpr_float", "value": float(tok)})
        else:
            raise RuntimeError(
                f"unrecognized signature token {tok!r} in {sig!r}. "
                "Expected *<dtype>[:N], a scalar dtype, an int/float literal, "
                'or a "string" constexpr.'
            )
    return descriptors


def _triton_signature_and_constants(fn, descriptors):
    """Build the ASTSource `signature` dict and `constexprs` (constants) dict.

    Maps the kernel's parameter names (in declaration order) to the parsed
    descriptors. Pointer/scalar params go into `signature`; literal/string
    params go into `constexprs`. `tt.divisibility` hints are returned for the
    AttrsDescriptor.
    """
    params = fn.arg_names
    if len(params) != len(descriptors):
        raise RuntimeError(
            f"signature has {len(descriptors)} args but kernel "
            f"{fn.__name__!r} declares {len(params)}: {params}"
        )

    signature = {}
    constants = {}
    divisibilities = {}
    for name, desc in zip(params, descriptors):
        kind = desc["kind"]
        if kind == "pointer":
            signature[name] = f"*{desc['dtype']}"
            if desc["divisibility"] is not None:
                divisibilities[name] = desc["divisibility"]
        elif kind == "scalar":
            signature[name] = desc["dtype"]
        elif kind == "constexpr_int":
            signature[name] = "constexpr"
            constants[name] = desc["value"]
        elif kind == "constexpr_float":
            signature[name] = "constexpr"
            constants[name] = desc["value"]
        elif kind == "constexpr_str":
            signature[name] = "constexpr"
            constants[name] = desc["value"]
        elif kind == "constexpr_none":
            signature[name] = "constexpr"
            constants[name] = None
        else:  # pragma: no cover - defensive
            raise RuntimeError(f"unhandled descriptor kind {kind!r}")
    return signature, constants, divisibilities


def _load_kernel(kernel_path: str, kernel_name: str):
    path = Path(kernel_path).resolve()
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, kernel_name, None)
    if fn is None:
        raise RuntimeError(f"kernel {kernel_name!r} not found in {kernel_path}")
    return fn


def _compile(fn, descriptors, arch: int, num_warps: int, num_stages: int):
    target = GPUTarget("cuda", int(arch), 32)
    backend = make_backend(target)
    signature, constants, divisibilities = _triton_signature_and_constants(
        fn, descriptors
    )

    opts = backend.parse_options(
        {
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
    )

    # ASTSource API varies slightly across triton 3.x; pass constexprs by the
    # name the installed version accepts. 3.5.x takes `constexprs`.
    try:
        src = ASTSource(
            fn=fn,
            signature=signature,
            constexprs=constants,
            attrs=divisibilities or None,
        )
    except TypeError:
        # Older 3.x kwarg name.
        src = ASTSource(
            fn=fn,
            signature=signature,
            constants=constants,
            attrs=divisibilities or None,
        )

    cc = triton.compile(src, target=target, options=opts.__dict__)

    global_scratch = getattr(cc.metadata, "global_scratch_size", 0)
    profile_scratch = getattr(cc.metadata, "profile_scratch_size", 0)
    if global_scratch != 0:
        raise RuntimeError(
            f"{fn.__name__}: global_scratch_size={global_scratch} != 0; "
            "AOT launch unsupported (triton refuses it)."
        )
    if profile_scratch != 0:
        raise RuntimeError(
            f"{fn.__name__}: profile_scratch_size={profile_scratch} != 0; "
            "AOT launch unsupported (triton refuses it)."
        )

    cubin = cc.asm[backend.binary_ext]
    return cubin, cc.metadata.name, int(cc.metadata.shared)


def _runtime_params(descriptors):
    """Return the ordered list of (ctype, argname) for the C prototype.

    Only pointer + scalar descriptors are runtime args; constexprs are baked.
    """
    out = []
    idx = 0
    for desc in descriptors:
        if desc["kind"] == "pointer":
            out.append(("CUdeviceptr", f"a{idx}"))
            idx += 1
        elif desc["kind"] == "scalar":
            out.append((_SCALAR_CTYPE[desc["dtype"]], f"a{idx}"))
            idx += 1
    return out


def _format_cubin_bytes(data: bytes) -> str:
    return "\n".join(
        "    " + ", ".join(f"0x{b:02x}" for b in data[i : i + 16]) + ","
        for i in range(0, len(data), 16)
    )


def _write_c(
    c_path: Path,
    out_name: str,
    cubin: bytes,
    kernel_symbol: str,
    shared: int,
    num_warps: int,
    runtime_params,
) -> None:
    cubin_array = _format_cubin_bytes(cubin)
    block_x = num_warps * 32

    # Public launch prototype: gX/gY/gZ FIRST, then the kernel runtime args,
    # then stream LAST (repo FFI convention).
    proto_args = ["uint32_t gX", "uint32_t gY", "uint32_t gZ"]
    proto_args += [f"{ctype} {name}" for ctype, name in runtime_params]
    proto_args.append("CUstream stream")
    proto = ", ".join(proto_args)

    # void* p[] = { &a0, ..., &gs, &ps }; the two trailing entries are the
    # global_scratch / profile_scratch null pointers triton's launch ABI
    # expects after the kernel args.
    arg_addr_lines = "\n".join(f"        &{name}," for _, name in runtime_params)

    # Shared-memory opt-in: if the kernel needs > 48 KB dynamic shared memory,
    # set the max-dynamic-shared attribute (mirrors triton's launcher template).
    src = f"""#include <cuda.h>
#include <stdint.h>

static CUmodule g_module = NULL;
static CUfunction g_function = NULL;
static const char *kFuncSymbol = "{kernel_symbol}";

static const unsigned char kCubinData[] = {{
{cubin_array}
}};
static const unsigned int kCubinSize = (unsigned int)sizeof(kCubinData);
static const int kSharedBytes = {shared};

/* Idempotent module load. Returns CUDA_SUCCESS if already loaded. */
CUresult {out_name}_load(void) {{
    if (g_function != NULL) return CUDA_SUCCESS;
    (void)kCubinSize;
    CUresult r = cuModuleLoadData(&g_module, kCubinData);
    if (r != CUDA_SUCCESS) return r;
    r = cuModuleGetFunction(&g_function, g_module, kFuncSymbol);
    if (r != CUDA_SUCCESS) return r;
    if (kSharedBytes > 49152) {{
        r = cuFuncSetAttribute(
            g_function,
            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            kSharedBytes
        );
        if (r != CUDA_SUCCESS) return r;
    }}
    return CUDA_SUCCESS;
}}

CUresult {out_name}({proto}) {{
    if (g_function == NULL) {{
        CUresult lr = {out_name}_load();
        if (lr != CUDA_SUCCESS) return lr;
    }}
    if (gX == 0 || gY == 0 || gZ == 0) return CUDA_SUCCESS;

    CUdeviceptr gs = 0, ps = 0;
    void *args[] = {{
{arg_addr_lines}
        &gs,
        &ps,
    }};

    return cuLaunchKernel(
        g_function,
        gX, gY, gZ,
        {block_x}, 1, 1,
        kSharedBytes, stream, args, NULL
    );
}}
"""
    c_path.write_text(src)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-path", required=True, help="path to the .py kernel file")
    parser.add_argument("--kernel-name", required=True, help="@triton.jit fn name in the file")
    parser.add_argument("--out-name", required=True, help="base name for the emitted C symbol")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--signature",
        required=True,
        help="comma-separated triton signature (kernel-parameter order)",
    )
    parser.add_argument("--arch", type=int, required=True, help="SM arch int (e.g. 90)")
    parser.add_argument("--num-warps", type=int, required=True)
    parser.add_argument("--num-stages", type=int, required=True)
    args = parser.parse_args()

    fn = _load_kernel(args.kernel_path, args.kernel_name)
    descriptors = _parse_signature(args.signature)
    cubin, kernel_symbol, shared = _compile(
        fn, descriptors, args.arch, args.num_warps, args.num_stages
    )

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    c_path = (out_dir / f"{args.out_name}.c").resolve()
    _write_c(
        c_path,
        args.out_name,
        cubin,
        kernel_symbol,
        shared,
        args.num_warps,
        _runtime_params(descriptors),
    )

    print(f"FUNC_NAME={args.out_name}")
    print(f"C_PATH={c_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
