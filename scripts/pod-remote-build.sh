#!/usr/bin/env bash
# Pod-side build runner — runs INSIDE the sglang-test container.
# Launched detached by scripts/pod.sh (setsid), so it survives the exec/SSH
# teardown. Writes its own PID for a clean exact-PID kill, and ends the log with
# a BUILD_EXIT=<n> marker — that marker, not process liveness, is the done-signal.
#
# Multi-agent safety (lock the racy steps, isolate the rest):
#   * TOOLCHAIN (~/.rustup, GLOBAL across all trees) — concurrent `rustup install`
#     corrupts it, so the idempotent ensure is wrapped in a global flock; it also
#     self-heals the "missing libstd" corruption (reinstall if the rlib is gone).
#   * CARGO BUILD (per-tree target/) — wrapped in a PER-TREE flock so two agents
#     on the SAME tree serialize cleanly, while agents on DIFFERENT POD_TREE dirs
#     build in parallel (separate target/, separate lock).
set -uo pipefail
TREE="${POD_TREE:-/host/arle-build}"
LABEL="${1:?usage: pod-remote-build.sh <label> <cargo-args...>}"; shift
echo $$ > "/root/build-$LABEL.pid"        # setsid made us session leader: PID == PGID
source "$TREE/scripts/pod-build-env.sh"
cd "$TREE"
LOG="/root/build-$LABEL.log"
TREE_LOCK="/tmp/arle-build$(echo "$TREE" | tr '/.' '__').lock"
{
  echo "=== BUILD START $(date -u) label=$LABEL tree=$TREE ==="
  # Racy step #1: ensure toolchain under a GLOBAL lock (idempotent + self-healing).
  flock /tmp/arle-toolchain.lock bash -c '
    toolchain_dir="${ARLE_RUST_TOOLCHAIN_DIR:-/root/.rustup/toolchains/1.95.0-x86_64-unknown-linux-gnu}"
    if ! [ -x "$toolchain_dir/bin/rustc" ] || ! ls "$toolchain_dir"/lib/rustlib/*/lib/libstd-*.rlib >/dev/null 2>&1; then
      echo "[setup] installing rust 1.95.0 via available network"
      rustup toolchain install 1.95.0 --profile minimal -c rustfmt -c clippy
    fi'
  source "$TREE/scripts/pod-build-env.sh"
  echo "rustc: $(rustc --version 2>&1)"
  echo "args : cargo build $*"
  # Racy step #2: build under a PER-TREE lock (same tree serializes, diff trees parallel).
  flock "$TREE_LOCK" cargo build "$@"
  echo "BUILD_EXIT=$?"
  echo "=== BUILD DONE $(date -u) ==="
} >"$LOG" 2>&1
