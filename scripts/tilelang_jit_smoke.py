#!/usr/bin/env python3
"""TileLang kernel definition smoke test (no GPU required).

Imports every TileLang kernel module under
``crates/cuda-kernels/tools/tilelang/`` and constructs a ``T.prim_func``
for every entry in the module's ``SUPPORTED_HEADS`` tuple. Catches:

  * Python syntax errors
  * Import errors (missing tilelang APIs, e.g. after upgrade)
  * ``get_kernel`` exceptions during prim_func construction
  * SUPPORTED_HEADS misconfigurations (mismatched arity etc.)

What this does NOT verify: cubin emit, kernel correctness, or runtime
behavior. That's ``cargo build --features cuda`` (build.rs invokes nvcc
+ TileLang AOT) and ``cargo test``. This script's job is to fail fast
on Python-level issues *before* triggering the full nvcc compile path —
the cargo path takes minutes; this takes seconds.

Use during E1/E2 sweep iteration (``BLOCK_M`` / ``BLOCK_N`` /
``NUM_STAGES`` edits) and after any TileLang upgrade.

Auto-skips kernel files that don't exist yet — the manifest can list
in-flight kernels (e.g. HD64 added by E4 subagent) without the script
breaking before they land.

Run via the workspace TileLang Python env:

    .venv/bin/python scripts/tilelang_jit_smoke.py
    .venv/bin/python scripts/tilelang_jit_smoke.py --verbose

Exits 0 on success; 1 on first kernel that fails; 2 on environment error.
"""

import argparse
import importlib.util
import sys
import warnings
from pathlib import Path

KERNELS_DIR = (
    Path(__file__).resolve().parent.parent
    / "crates"
    / "cuda-kernels"
    / "tools"
    / "tilelang"
)

# Module -> required public attributes (must include SUPPORTED_HEADS + get_kernel)
KERNEL_MANIFEST = (
    "batch_prefill_paged_hd128",
    "batch_decode_paged_hd128",
    "batch_decode_paged_hd128_fp8",
    "batch_prefill_paged_hd256",
    "batch_decode_paged_hd256",
    # E4 substrate (subagent in flight; auto-skip if not yet committed):
    "batch_prefill_paged_hd64",
    "batch_decode_paged_hd64",
)


def load_module(module_name):
    path = KERNELS_DIR / f"{module_name}.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def smoke_one(module_name, verbose):
    mod = load_module(module_name)
    if mod is None:
        if verbose:
            print(f"SKIP {module_name} (file not present)")
        return True, 0

    for attr in ("SUPPORTED_HEADS", "get_kernel"):
        if not hasattr(mod, attr):
            print(f"FAIL {module_name}: missing attribute '{attr}'")
            return False, 0

    supported = mod.SUPPORTED_HEADS
    get_kernel = mod.get_kernel

    if not isinstance(supported, tuple) or not supported:
        print(
            f"FAIL {module_name}: SUPPORTED_HEADS must be a non-empty tuple, "
            f"got {type(supported).__name__}"
        )
        return False, 0

    constructed = 0
    for cfg in supported:
        try:
            kernel = get_kernel(*cfg)
        except Exception as e:
            print(f"FAIL {module_name}({cfg}): {type(e).__name__}: {e}")
            return False, constructed
        if kernel is None:
            print(f"FAIL {module_name}({cfg}): get_kernel returned None")
            return False, constructed
        constructed += 1
        if verbose:
            print(f"  OK  {module_name}{cfg} -> {type(kernel).__name__}")

    if not verbose:
        print(f"OK   {module_name}: {constructed} configs")
    return True, constructed


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--quiet-tvm-warnings",
        action="store_true",
        default=True,
        help="suppress noisy tvm_ffi registry duplicate-field warnings (default on)",
    )
    args = parser.parse_args()

    if args.quiet_tvm_warnings:
        warnings.filterwarnings("ignore", category=UserWarning, module=r"tvm_ffi.*")

    try:
        import tilelang  # noqa: F401
        import tilelang.language  # noqa: F401
    except ImportError as e:
        print(
            f"FAIL: tilelang import error: {e}\n"
            "Run via .venv/bin/python or set INFER_TILELANG_PYTHON.",
            file=sys.stderr,
        )
        sys.exit(2)

    if not KERNELS_DIR.is_dir():
        print(f"FAIL: KERNELS_DIR not found: {KERNELS_DIR}", file=sys.stderr)
        sys.exit(2)

    failures = 0
    total_configs = 0
    total_modules = 0
    for module_name in KERNEL_MANIFEST:
        ok, n = smoke_one(module_name, args.verbose)
        if ok:
            total_configs += n
            if n > 0:
                total_modules += 1
        else:
            failures += 1

    if failures:
        print(
            f"\n{failures} kernel module(s) failed smoke test", file=sys.stderr
        )
        sys.exit(1)
    print(
        f"\nAll kernel modules passed: {total_modules} modules, "
        f"{total_configs} (q,kv) configs."
    )


if __name__ == "__main__":
    main()
