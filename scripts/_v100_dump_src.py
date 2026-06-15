#!/usr/bin/env python3
"""Dump the generated CUDA device source for a kernel @ arch (no nvcc needed for
the codegen text). Used to prove the sm_80+ AST is byte-identical between the
repo kernel and the sm_70-gated variant. Usage: dump_src.py <kernel.py> <arch> <qh> <kvh>"""
import importlib.util
import os
import sys

arch = int(sys.argv[2])
os.environ["ARLE_TILELANG_CUDA_ARCH"] = str(arch)
import tilelang  # noqa: E402

spec = importlib.util.spec_from_file_location("m", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
pf = m.get_kernel(int(sys.argv[3]), int(sys.argv[4]))
target = tilelang.tvm.target.Target({"kind": "cuda", "arch": f"sm_{arch}"})
compiled = tilelang.compile(pf, target=target)
adapter = compiled.adapter
src = (adapter.get_device_source() if hasattr(adapter, "get_device_source")
       else adapter.device_kernel_source)
sys.stdout.write(src)
