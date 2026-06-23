#!/usr/bin/env bash
# pod.sh — local-authoritative / remote-verify devops for the H20 (sm_90) box.
#
#   Source of truth = THIS local git tree (edit + commit here, never on the pod).
#   Pod build tree  = /host/arle-build  — the node's /root/arle-build via the
#                     sglang-test container's host-root hostPath mount. Persistent
#                     (survives a static-pod restart) AND exactly where `tn push`
#                     lands, so one push updates the build tree with no copy-in hop.
#   Builds/runs are DETACHED (survive disconnect), logged with a BUILD_EXIT/RUN_EXIT
#   marker (the done-signal, not process liveness), and polled — never a foreground
#   `tn exec` a timeout can strand. Kills are by exact recorded PGID — never
#   `pkill -f <our own cmd>` (self-matches; once corrupted the toolchain mid-install).
#
# MULTI-AGENT (lock the racy steps, isolate the rest):
#   * Racy, LOCKED: the toolchain (~/.rustup, global) is installed under a global
#     flock + self-heals; each tree's `cargo build` is under a per-tree flock.
#     => agents on the SAME tree serialize safely; the toolchain never double-installs.
#   * Isolated, NO lock: give each agent its own POD_TREE (separate source+target =>
#     parallel builds), its own <label> (separate log/pid), and its own <gpu> for
#     `run` (INFER_CUDA_DEVICE). Runs on different GPUs never collide.
#   Typical fan-out: build the shared binary ONCE (`build`), then N agents each
#   `run <label-i> <gpu-i> -- <args>` on a free GPU (`gpus` to pick).
#
# Usage:
#   scripts/pod.sh push-scripts                       # push the 3 pod-side helpers
#   scripts/pod.sh setup                              # warm/repair the shared toolchain (flock)
#   scripts/pod.sh sync [paths...]                    # push files local->pod (default: git-changed)
#   scripts/pod.sh build <label> <cargo-args...>      # detached build (per-tree-flocked)
#   scripts/pod.sh run   <label> <gpu> [--] <args...> # detached arle run on ONE gpu
#   scripts/pod.sh gpus                               # per-GPU memory/util (pick a free one)
#   scripts/pod.sh status <label>                     # build- AND run-<label>: running? tail + marker
#   scripts/pod.sh log    <label>                     # full build/run log
#   scripts/pod.sh kill   <label>                     # kill build/run by recorded PGID
#
# Env overrides: POD (exec wrapper, default ~/bin/pod), TN (default tn),
#                NODE_TREE (tn push target), POD_TREE (container-view tree).
set -euo pipefail

POD="${POD:-$HOME/bin/pod}"                 # exec INTO the sglang-test container
TN="${TN:-tn}"                              # tunnel CLI for SFTP push (lands on node)
NODE_TREE="${NODE_TREE:-/root/arle-build}"  # tn push target (node view)
TREE="${POD_TREE:-/host/arle-build}"        # build tree (container view of the same dir)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cmd="${1:-help}"; shift || true

case "$cmd" in
  push-scripts)
    for s in pod-build-env.sh pod-remote-build.sh pod-remote-run.sh; do
      "$TN" push "$ROOT/scripts/$s" "$NODE_TREE/scripts/$s"
    done
    echo "pushed pod-side helpers -> $NODE_TREE/scripts/"
    ;;
  setup)
    # Warm/repair the shared toolchain once (so a build fan-out doesn't serialize
    # on the ensure-flock). Idempotent + self-healing; flock = no double-install.
    "$POD" "flock /tmp/arle-toolchain.lock bash -lc '
      if ls ~/.rustup/toolchains/1.95.0-*/lib/rustlib/*/lib/libstd-*.rlib >/dev/null 2>&1; then
        echo toolchain-1.95.0-OK
      else
        echo installing 1.95.0 via proxy; rustup toolchain install 1.95.0 --profile minimal -c rustfmt -c clippy
      fi'"
    ;;
  sync)
    paths=("$@")
    if [ ${#paths[@]} -eq 0 ]; then
      mapfile -t paths < <(cd "$ROOT" && git status --porcelain | awk '{print $2}')
    fi
    [ ${#paths[@]} -eq 0 ] && { echo "nothing to sync"; exit 0; }
    for p in "${paths[@]}"; do
      [ -f "$ROOT/$p" ] || { echo "skip (not a file): $p"; continue; }
      "$TN" push "$ROOT/$p" "$NODE_TREE/$p" && echo "synced  $p"
    done
    ;;
  build)
    label="${1:?usage: build <label> <cargo-args...>}"; shift
    "$POD" "setsid bash $TREE/scripts/pod-remote-build.sh $label $* </dev/null >/dev/null 2>&1 &"
    echo "build '$label' launched (detached). poll: scripts/pod.sh status $label"
    ;;
  run)
    label="${1:?usage: run <label> <gpu> [--] <arle args...>}"; shift
    gpu="${1:?usage: run <label> <gpu> [--] <arle args...>}"; shift
    [ "${1:-}" = "--" ] && shift
    "$POD" "setsid bash $TREE/scripts/pod-remote-run.sh $label $gpu $* </dev/null >/dev/null 2>&1 &"
    echo "run '$label' launched on GPU $gpu (detached). poll: scripts/pod.sh status $label"
    ;;
  gpus)
    "$POD" "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader"
    ;;
  status)
    label="${1:?usage: status <label>}"
    "$POD" "for k in build run; do f=/root/\$k-$label; [ -f \$f.log ] || continue; \
      p=\$(cat \$f.pid 2>/dev/null); \
      if [ -n \"\$p\" ] && kill -0 \$p 2>/dev/null; then echo \"[\$k] RUNNING pid=\$p\"; else echo \"[\$k] not running (done or never started)\"; fi; \
      echo '--- tail ---'; tail -20 \$f.log 2>/dev/null; \
      echo '--- marker ---'; grep -E 'BUILD_EXIT|RUN_EXIT|DONE' \$f.log 2>/dev/null | tail -2; done"
    ;;
  log)
    label="${1:?usage: log <label>}"
    "$POD" "for k in build run; do [ -f /root/\$k-$label.log ] && { echo \"==== \$k-$label.log ====\"; cat /root/\$k-$label.log; }; done"
    ;;
  kill)
    label="${1:?usage: kill <label>}"
    "$POD" "for k in build run; do p=\$(cat /root/\$k-$label.pid 2>/dev/null); \
      [ -n \"\$p\" ] && { kill -- -\$p 2>/dev/null; kill \$p 2>/dev/null; echo \"killed \$k pgid \$p\"; }; done"
    ;;
  *)
    sed -n '2,38p' "$0" | sed 's/^# \?//'
    ;;
esac
