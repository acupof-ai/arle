#!/usr/bin/env bash
# One-shot: build patched TileLang 0.1.11 from source on the V100 box (sm_70).
# Mirrors the CI build-linux-v100 recipe. Uses CPU torch (the layout-inference
# compile-test + the tilelang source build only need torch's CPU DLPack adapter;
# nvcc owns the CUDA codegen via CUDA_HOME, and the needle gate is the Rust
# binary — Python torch CUDA is never on this path). gcc-11 host compiler
# because CUDA 11.8 nvcc rejects gcc 12. The cuda.fma C++ patch lives in
# libtilelang.so → source build is mandatory.
set -euo pipefail

DIG=$HOME/sm70-dig
VENV=$DIG/venv
TL=$DIG/tilelang-sm70
PATCH=$DIG/sm70_tilelang.patch

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
# CUDA 11.8 nvcc caps host gcc at 11; force gcc-11 for every C/C++/CUDA TU.
export CC=gcc-11 CXX=g++-11
export CUDAHOSTCXX=g++-11
export CMAKE_CUDA_HOST_COMPILER=g++-11

echo "=== [1/6] python venv (reuse if present) ==="
[ -d "$VENV" ] || python3 -m venv "$VENV"
. "$VENV/bin/activate"
python -m pip install -U pip wheel setuptools >/dev/null

echo "=== [2/6] build deps + CPU torch + apache-tvm-ffi 0.1.11 ==="
pip install --no-cache-dir "cython>=3.1.0" scikit-build-core "z3-solver>=4.13.0,<4.15.5" ninja cmake
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
pip install --no-cache-dir apache-tvm-ffi==0.1.11

echo "=== [3/6] clone tilelang v0.1.11 (recursive) ==="
rm -rf "$TL"
git clone --recursive --shallow-submodules --depth 1 --branch v0.1.11 \
  https://github.com/tile-ai/tilelang.git "$TL"

echo "=== [4/6] apply sm_70 cuda.fma patch ==="
cd "$TL"
git apply --check "$PATCH"
git apply "$PATCH"
echo "patch applied; verifying markers:"
grep -q 'kCudaFMA = "cuda.fma"' src/cuda/op/gemm.cc && echo "  gemm.cc kCudaFMA OK"
grep -q "GEMM_INST_FMA" tilelang/cuda/op/gemm/gemm_fma.py && echo "  gemm_fma.py OK"

echo "=== [5/6] build + install (non-editable, no build isolation) ==="
pip install --no-cache-dir --no-build-isolation .

echo "=== [6/6] verify import ==="
cd "$DIG"
python -c "import tilelang; print('tilelang', tilelang.__version__, 'from', tilelang.__file__)"
echo "=== DONE_TILELANG_BUILD ==="
