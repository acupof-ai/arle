#!/usr/bin/env bash
# Apply the TileLang sm_70 fallback patch to a local TileLang checkout so ARLE
# V100 builds work today. The patch carries Volta fragment-copy staging plus a
# cuda.fma T.gemm fallback for dtype/layout combinations unsupported by SM70 MMA.
#
# REMOVE WHEN: PR https://github.com/tile-ai/tilelang/pull/2279 merges and a
# TileLang release containing it is published. At that point pin the version
# in crates/cuda-kernels/tools/tilelang/ instead.
#
# Usage:
#   scripts/patch_tilelang_sm70.sh <tilelang-checkout>
#
# Example:
#   git clone https://github.com/tile-ai/tilelang.git ~/tilelang
#   scripts/patch_tilelang_sm70.sh ~/tilelang
#   cd ~/tilelang && pip install --no-build-isolation -e .

set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <tilelang-checkout>" >&2
  exit 2
fi

TILELANG_DIR=$1
PATCH=$(cd "$(dirname "$0")" && pwd)/sm70_tilelang.patch

if [ ! -d "$TILELANG_DIR/src/backend/cuda/op" ]; then
  echo "error: $TILELANG_DIR does not look like a TileLang checkout" >&2
  echo "  (expected src/backend/cuda/op/copy.cc)" >&2
  exit 1
fi

if [ ! -f "$PATCH" ]; then
  echo "error: patch file not found at $PATCH" >&2
  exit 1
fi

cd "$TILELANG_DIR"

PATCH_PATHS=(
  src/backend/cuda/op/copy.cc
  src/backend/cuda/op/gemm.cc
  src/tl_templates/cuda/instruction/mma_sm70.h
  testing/python/cuda/test_cuda_mma_sm75_dispatch.py
  testing/python/kernel/test_tilelang_kernel_sm70_fragment_copy.py
  testing/python/kernel/test_tilelang_kernel_sm70_gemm_fma.py
  tilelang/cuda/op/gemm/__init__.py
  tilelang/cuda/op/gemm/gemm_fma.py
)

if grep -q "NeedsVoltaFragmentStaging" src/backend/cuda/op/copy.cc 2>/dev/null &&
  grep -q "GEMM_INST_FMA" tilelang/cuda/op/gemm/gemm_fma.py 2>/dev/null; then
  echo "patch already applied to $TILELANG_DIR (fragment staging + GemmFMA found); skipping"
  exit 0
fi

if ! git diff --quiet -- "${PATCH_PATHS[@]}" 2>/dev/null; then
  echo "error: target files already have unstaged changes; abort" >&2
  git diff --stat -- "${PATCH_PATHS[@]}" >&2
  exit 1
fi

git apply --check "$PATCH"
git apply "$PATCH"
echo "applied $PATCH to $TILELANG_DIR"
echo "next: cd $TILELANG_DIR && pip install --no-build-isolation -e ."
