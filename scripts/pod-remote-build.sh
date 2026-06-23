#!/usr/bin/env bash
# Pod-side build runner — runs INSIDE the sglang-test container.
# Launched detached by scripts/pod.sh (setsid), so it survives the exec/SSH
# teardown. Writes its own PID for a clean exact-PID kill, and ends the log with
# a BUILD_EXIT=<n> marker — that marker, not process liveness, is the done-signal
# (a missing marker on a non-growing log == the build was killed, not finished).
set -uo pipefail
TREE="${POD_TREE:-/host/arle-build}"
LABEL="${1:?usage: pod-remote-build.sh <label> <cargo-args...>}"; shift
echo $$ > "/root/build-$LABEL.pid"        # setsid made us session leader: PID == PGID
source "$TREE/scripts/pod-build-env.sh"
cd "$TREE"
LOG="/root/build-$LABEL.log"
{
  echo "=== BUILD START $(date -u) label=$LABEL ==="
  echo "rustc: $(rustc --version 2>&1)"
  echo "args : cargo build $*"
  cargo build "$@"
  echo "BUILD_EXIT=$?"
  echo "=== BUILD DONE $(date -u) ==="
} >"$LOG" 2>&1
