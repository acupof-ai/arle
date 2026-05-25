#!/usr/bin/env bash
# Apply the TileLang sm_70 BF16 fallback patch (upstream PR #2257) to a local
# TileLang checkout so ARLE V100 builds work today.
#
# REMOVE WHEN: PR https://github.com/tile-ai/tilelang/pull/2257 merges and a
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

if ! git diff --quiet -- src/backend/cuda/op/copy.cc src/tl_templates/cuda/instruction/mma_sm70.h 2>/dev/null; then
  echo "error: target files already have unstaged changes; abort" >&2
  git diff --stat -- src/backend/cuda/op/copy.cc src/tl_templates/cuda/instruction/mma_sm70.h >&2
  exit 1
fi

if grep -q "NeedsVoltaFragmentStaging" src/backend/cuda/op/copy.cc 2>/dev/null; then
  echo "patch already applied to $TILELANG_DIR (NeedsVoltaFragmentStaging found); skipping"
  exit 0
fi

git apply --check "$PATCH"
git apply "$PATCH"
echo "applied $PATCH to $TILELANG_DIR"
echo "next: cd $TILELANG_DIR && pip install --no-build-isolation -e ."
