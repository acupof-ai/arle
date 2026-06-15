#!/usr/bin/env bash
# One-shot: build STOCK TileLang 0.1.11 from source on the V100 box (sm_70).
# Mirrors the CI build-linux-v100 recipe. The sm_70 attention AOT kernels reach
# Volta tensor cores via the in-kernel fp16-MMA gate (batch_*_paged_hd*.py) — no
# source patch: the old cuda.fma patch was dropped 2026-06-15, confirmed by a
# no-patch build on real V100 (all 4 kernels cosine 0.999999). Source build (not
# the wheel) gives build.rs the src/tl_templates + 3rdparty/cutlass layout it
# probes. Uses CPU torch (the source build + nvcc codegen via CUDA_HOME don't need
# torch CUDA; the needle gate is the Rust binary). gcc-11 host compiler.
# NOTE: stock 0.1.11 JIT emits -std=c++20 → CUDA_HOME must be CUDA >=12 (the box
# /usr/local/cuda is 12.4); the default /usr/bin/nvcc 11.8 lacks c++20.
set -euo pipefail

DIG=$HOME/sm70-dig
VENV=$DIG/venv
TL=$DIG/tilelang-sm70

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
# Force gcc-11 for every C/C++/CUDA TU (conservative host compiler for nvcc).
export CC=gcc-11 CXX=g++-11
export CUDAHOSTCXX=g++-11
export CMAKE_CUDA_HOST_COMPILER=g++-11

echo "=== [1/5] python venv (reuse if present) ==="
[ -d "$VENV" ] || python3 -m venv "$VENV"
. "$VENV/bin/activate"
python -m pip install -U pip wheel setuptools >/dev/null

echo "=== [2/5] build deps + CPU torch + apache-tvm-ffi 0.1.11 ==="
pip install --no-cache-dir "cython>=3.1.0" scikit-build-core "z3-solver>=4.13.0,<4.15.5" ninja cmake
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
pip install --no-cache-dir apache-tvm-ffi==0.1.11

echo "=== [3/5] clone STOCK tilelang v0.1.11 (recursive, no patch) ==="
rm -rf "$TL"
git clone --recursive --shallow-submodules --depth 1 --branch v0.1.11 \
  https://github.com/tile-ai/tilelang.git "$TL"
cd "$TL"
# Assert this is the unpatched tree — the cuda.fma fallback must NOT be present.
if grep -q 'kCudaFMA = "cuda.fma"' src/cuda/op/gemm.cc 2>/dev/null; then
  echo "ERROR: cuda.fma patch present — expected STOCK tilelang" >&2
  exit 1
fi
echo "  confirmed stock (no kCudaFMA)"

echo "=== [4/5] build + install (non-editable, no build isolation) ==="
pip install --no-cache-dir --no-build-isolation .

echo "=== [5/5] verify import ==="
cd "$DIG"
python -c "import tilelang; print('tilelang', tilelang.__version__, 'from', tilelang.__file__)"
echo "=== DONE_TILELANG_BUILD ==="
