#!/usr/bin/env bash
# Pod-side TileLang env for kernel AOT codegen — one gate: `import tilelang`.
#
# tilelang >=0.1.13 imports cleanly against vanilla apache-tvm-ffi (the
# first-wins TypeAttr fix from tile-ai/tilelang#2367 landed upstream), so no
# ARLE tvm-ffi patch is needed.
# Idempotent — if the venv already imports tilelang, nothing happens.
set -euo pipefail
TREE="${POD_TREE:-/host/arle-build}"
VENV="$TREE/crates/cuda-kernels/tools/tilelang/.venv"
PY="$VENV/bin/python"

TILELANG_PIN="$(grep -oE 'tilelang==[0-9.]+' "$TREE/requirements-build.txt" | cut -d= -f3)"

ok() {
  "$PY" -c "
import importlib.metadata as m, tilelang
assert m.version('tilelang') == '$TILELANG_PIN', m.version('tilelang')
print('tilelang', tilelang.__version__, 'ok')" 2>/dev/null
}

if ok; then exit 0; fi
if [ ! -x "$PY" ]; then
  python3 -m venv "$VENV" \
    || python3 -m virtualenv "$VENV" \
    || { echo "venv creation failed: install virtualenv or ensurepip" >&2; exit 1; }
fi
"$PY" -m pip install --no-cache-dir -r "$TREE/requirements-build.txt"
ok
