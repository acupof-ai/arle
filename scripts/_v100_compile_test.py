#!/usr/bin/env python3
"""sm_70 LayoutInference compile-test harness (V100 box scratch helper).

Reproduces / validates the TileLang AOT layout-inference pass for a single
attention kernel WITHOUT nvcc or a GPU — `tilelang.compile()` runs the TVM IR
LayoutInferencer, which is where the Volta bf16 conflicts surface. Seconds per
run → fast edit-compile loop on the kernel .py.

Usage:
  python _v100_compile_test.py <kernel.py> <sm_arch> [q_heads kv_heads ...]
  python _v100_compile_test.py batch_prefill_paged_hd128.py 70 32 8

With no head pairs it sweeps the module's SUPPORTED_HEADS. Exit 0 = all green.
"""
import importlib.util
import os
import sys
import traceback


def load_module(path):
    spec = importlib.util.spec_from_file_location("tl_kernel_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    kernel_path = sys.argv[1]
    arch = int(sys.argv[2])
    # Gen sets this before importing the kernel; mirror it so the kernel's
    # SM-gate (os.environ ARLE_TILELANG_CUDA_ARCH) sees the right arch.
    os.environ["ARLE_TILELANG_CUDA_ARCH"] = str(arch)

    import tilelang  # noqa: E402  (after env set, like gen_tilelang_aot.py)

    mod = load_module(kernel_path)
    if len(sys.argv) > 3:
        rest = list(map(int, sys.argv[3:]))
        heads = list(zip(rest[0::2], rest[1::2]))
    else:
        heads = list(getattr(mod, "SUPPORTED_HEADS"))

    target = tilelang.tvm.target.Target({"kind": "cuda", "arch": f"sm_{arch}"})
    print(f"=== compile-test {kernel_path} @ sm_{arch} heads={heads} ===", flush=True)
    failures = []
    for (qh, kvh) in heads:
        tag = f"q{qh}_kv{kvh}"
        try:
            pf = mod.get_kernel(qh, kvh)
            tilelang.compile(pf, target=target)
            print(f"[PASS] {tag}", flush=True)
        except Exception as exc:  # noqa: BLE001 — want the message verbatim
            msg = str(exc).strip().splitlines()
            head = msg[0] if msg else repr(exc)
            print(f"[FAIL] {tag}: {head}", flush=True)
            failures.append((tag, exc))

    if failures:
        print(f"\n=== {len(failures)}/{len(heads)} FAILED ===", flush=True)
        # Full traceback for the first failure (the diagnostic).
        tag, exc = failures[0]
        print(f"--- first failure ({tag}) traceback ---", flush=True)
        traceback.print_exception(type(exc), exc, exc.__traceback__)
        sys.exit(1)
    print(f"\n=== ALL {len(heads)} GREEN @ sm_{arch} ===", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
