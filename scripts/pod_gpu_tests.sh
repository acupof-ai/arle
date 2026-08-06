#!/usr/bin/env bash
# Detached launcher for GPU test targets. `pod.sh build` validates its argv down
# to `cargo build --release --features … --bin <name>`, so nothing in the repo
# could run `cargo test --features cuda` — which is why the autograd and
# spec-train suites had never executed a CUDA backward. Mirrors pod_parity.sh:
# same build env, same re-attachable marker.
#   pod_gpu_tests.sh run <label> <devices> -- <cargo test argv...>
set -uo pipefail
TREE=/host/arle-build
# State lives outside the tree: `source_digest` hashes `git ls-files -co
# --exclude-standard`, so a run directory under $TREE makes the next build
# refuse with "source changed since receipt".
STATE_ROOT="${POD_STATE:-/root/arle-ops}"
MODE="${1:?usage: pod_gpu_tests.sh run <label> <devices> -- <cargo test argv...>}"
cd "$TREE" || exit 1
# shellcheck disable=SC1091
source "$TREE/scripts/pod-build-env.sh"

[ "$MODE" = run ] || { echo "usage: pod_gpu_tests.sh run <label> <devices> -- <cargo test argv...>" >&2; exit 2; }
LABEL="${2:?missing label}"; DEVICES="${3:?missing devices}"
shift 3
[ "${1:-}" = "--" ] && shift
[ $# -gt 0 ] || { echo "no cargo test arguments" >&2; exit 2; }

STATE="$STATE_ROOT/gpu-tests/$LABEL"
mkdir -p "$STATE"
LOG="$STATE/log"; MARKER="$STATE/marker"
rm -f "$MARKER"
{
  echo "=== gpu tests $LABEL devices=$DEVICES START $(date -u +%FT%TZ) ==="
  echo "cargo test $*"
  CUDA_VISIBLE_DEVICES="$DEVICES" cargo test "$@"
  rc=$?
  echo "RUN_EXIT=$rc"
  echo "RUN_EXIT=$rc" > "$MARKER"
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader || true
} >"$LOG" 2>&1
