#!/usr/bin/env bash
# CI-side TileLang import gate. tilelang >=0.1.13 imports cleanly against
# vanilla apache-tvm-ffi (the first-wins TypeAttr fix from
# tile-ai/tilelang#2367 landed upstream), so no ARLE tvm-ffi patch is needed.
# Run AFTER `pip install -r requirements-build.txt`, BEFORE the cargo CUDA build.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if python3 -c "import tilelang; print('tilelang', tilelang.__version__)" 2>/dev/null; then
  echo "tilelang imports cleanly; no tvm-ffi patch needed"
  exit 0
fi

echo "tilelang import failed; check requirements-build.txt" >&2
exit 1
