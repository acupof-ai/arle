#!/usr/bin/env bash
# CI-side apache-tvm-ffi patch. Vanilla apache-tvm-ffi 0.1.12 hard-aborts
# `import tilelang` (duplicate builtin TypeAttr registration vs tilelang's
# bundled TVM — tile-ai/tilelang#2367). Rebuild tvm-ffi from sdist with the
# first-wins patch the repo carries, against the system Python (no venv).
# Mirrors scripts/pod-tilelang-env.sh, which does the same for the pod venv.
# Run AFTER `pip install -r requirements-build.txt` (and, for source-built
# TileLang lanes, after the TileLang install), BEFORE the cargo CUDA build.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH="$ROOT/crates/cuda-kernels/tools/tilelang/patches/tvm-ffi-legacy-tvm-compat.patch"
PIN="$(grep -oE 'apache-tvm-ffi==[0-9.]+' "$ROOT/requirements-build.txt" | cut -d= -f3)"

# Skip only when the import is clean AND the installed tvm-ffi already matches
# the pin: the resolver pass installs tilelang's vanilla <=0.1.11 (which imports
# fine), so an import-only check would silently keep the un-pinned version.
# Use python3: the slim CUDA image has no bare `python` (only python3).
if python3 -c "
import importlib.metadata as m, tilelang
assert m.version('apache-tvm-ffi') == '$PIN'
" 2>/dev/null; then
  echo "tilelang imports cleanly with tvm-ffi $PIN; patch already landed"
  exit 0
fi

work="$(mktemp -d)"
# --no-binary ONLY for apache-tvm-ffi (we need its sdist to patch). `:all:` would
# force pip to source-build every build backend dep too — incl. cmake, which
# times out on CI when the env's pip cmake (4.x) falls outside scikit-build-core's
# accepted range and gets requested as a from-source backend dep.
pip download --no-cache-dir --no-deps --no-binary apache-tvm-ffi -d "$work" "apache-tvm-ffi==$PIN"
tar -xzf "$work"/apache*tvm*ffi-*.tar.gz -C "$work"
src="$(ls -d "$work"/apache*tvm*ffi-*/)"
patch -p1 -d "$src" < "$PATCH"
# Build the wheel with --no-build-isolation so THIS env's deps are authoritative
# (no nested isolated builds). tvm-ffi's scikit-build-core backend needs
# setuptools_scm + cython to configure and a cmake in its accepted range — the
# sm70 lane's `pip install cmake` pulls 4.x, which scikit-build-core rejects, so
# pin cmake<4. modern setuptools/packaging keeps license parsing quiet.
pip install --no-cache-dir -U \
  'setuptools>=77' 'packaging>=24.2' wheel scikit-build-core setuptools_scm cython ninja 'cmake<4'
pip wheel --no-cache-dir --no-deps --no-build-isolation -w "$work" "$src"
pip install --force-reinstall --no-deps "$work"/apache*tvm*ffi-*.whl
python3 -c "import tilelang; print('tilelang', tilelang.__version__, 'tvm-ffi patched-ok')"
