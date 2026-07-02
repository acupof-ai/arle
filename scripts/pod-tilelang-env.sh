#!/usr/bin/env bash
# Pod-side TileLang env for kernel AOT codegen — one gate: `import tilelang`.
#
# Builds the codegen venv from requirements-build.txt, then rebuilds
# apache-tvm-ffi from sdist with the ARLE first-wins TypeAttr patch: tvm-ffi
# >=0.1.12 registers builtin entries that the bundled TVM of tilelang <=0.1.11
# still registers at load, and vanilla 0.1.12 hard-aborts the import
# (tile-ai/tilelang#2367; upstream's only fix so far is pinning <=0.1.11).
# Idempotent — if the venv already imports tilelang, nothing happens.
#
# Pod PyPI is slow and the proxy is not always up: pre-push the sdist via tn
# and point TVM_FFI_SDIST at it to skip the download.
set -euo pipefail
TREE="${POD_TREE:-/host/arle-build}"
VENV="$TREE/crates/cuda-kernels/tools/tilelang/.venv"
PATCH="$TREE/crates/cuda-kernels/tools/tilelang/patches/tvm-ffi-legacy-tvm-compat.patch"
PY="$VENV/bin/python"
PIN="$(grep -oE 'apache-tvm-ffi==[0-9.]+' "$TREE/requirements-build.txt" | cut -d= -f3)"

ok() {
  "$PY" -c "
import importlib.metadata as m, tilelang
assert m.version('apache-tvm-ffi') == '$PIN', m.version('apache-tvm-ffi')
print('tilelang', tilelang.__version__, 'tvm-ffi', '$PIN', 'patched-ok')" 2>/dev/null
}

if ok; then exit 0; fi
[ -x "$PY" ] || python3 -m venv --system-site-packages "$VENV"
# Provision the pinned set only when tilelang itself is absent — a tvm-ffi
# mismatch alone must not re-pull the multi-GB torch chain over the pod's
# slow network; the patched-wheel step below satisfies the pin by itself.
"$PY" -m pip show tilelang >/dev/null 2>&1 \
  || "$PY" -m pip install --no-cache-dir -r "$TREE/requirements-build.txt"
if ! ok >/dev/null; then
  work="$(mktemp -d)"
  sdist="${TVM_FFI_SDIST:-}"
  if [ -z "$sdist" ]; then
    "$PY" -m pip download --no-deps --no-binary :all: -d "$work" "apache-tvm-ffi==$PIN"
    sdist="$(ls "$work"/apache*tvm*ffi-*.tar.gz)"
  fi
  tar -xzf "$sdist" -C "$work"
  src="$(ls -d "$work"/apache*tvm*ffi-*/)"
  patch -p1 -d "$src" < "$PATCH"
  # no-build-isolation reuses system cmake/ninja when the venv has
  # scikit-build-core; fall back to isolated (needs PyPI reachability).
  "$PY" -m pip wheel --no-deps --no-build-isolation -w "$work" "$src" \
    || "$PY" -m pip wheel --no-deps -w "$work" "$src"
  "$PY" -m pip install --force-reinstall --no-deps "$work"/apache*tvm*ffi-*.whl
fi
ok
