#!/usr/bin/env bash
# CI-side apache-tvm-ffi patch. Vanilla apache-tvm-ffi 0.1.12 hard-aborts
# `import tilelang` (duplicate builtin TypeAttr registration vs tilelang<=0.1.11's
# bundled TVM — tile-ai/tilelang#2367). Rebuild tvm-ffi from sdist with the
# first-wins patch the repo carries, against the system Python (no venv).
# Mirrors scripts/pod-tilelang-env.sh, which does the same for the pod venv.
# Run AFTER `pip install -r requirements-build.txt` (and, for source-built
# TileLang lanes, after the TileLang install), BEFORE the cargo CUDA build.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH="$ROOT/crates/cuda-kernels/tools/tilelang/patches/tvm-ffi-legacy-tvm-compat.patch"
PIN="$(grep -oE 'apache-tvm-ffi==[0-9.]+' "$ROOT/requirements-build.txt" | cut -d= -f3)"

# Vanilla 0.1.12 aborts the process (SIGABRT) on import, so a clean import here
# means the patch already landed — nothing to do.
if python -c "import tilelang" 2>/dev/null; then
  echo "tilelang imports cleanly; tvm-ffi patch not needed"
  exit 0
fi

work="$(mktemp -d)"
pip download --no-cache-dir --no-deps --no-binary :all: -d "$work" "apache-tvm-ffi==$PIN"
tar -xzf "$work"/apache*tvm*ffi-*.tar.gz -C "$work"
src="$(ls -d "$work"/apache*tvm*ffi-*/)"
patch -p1 -d "$src" < "$PATCH"
# Build the wheel in THIS env with --no-build-isolation, so we control the
# build deps directly instead of pip's nested isolated builds (which recurse
# into the same old-packaging trap). The tvm-ffi sdist needs scikit-build-core
# + setuptools_scm + cython to configure, and a modern setuptools/packaging to
# parse its PEP 639 license expression (else "Cannot import packaging.licenses",
# setuptools>=77 / packaging>=24.2). Install the full set up front.
pip install --no-cache-dir -U \
  'setuptools>=77' 'packaging>=24.2' wheel scikit-build-core setuptools_scm cython ninja
pip wheel --no-cache-dir --no-deps --no-build-isolation -w "$work" "$src"
pip install --force-reinstall --no-deps "$work"/apache*tvm*ffi-*.whl
python -c "import tilelang; print('tilelang', tilelang.__version__, 'tvm-ffi patched-ok')"
