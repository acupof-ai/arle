#!/usr/bin/env bash
# pod.sh — local-authoritative / remote-verify devops for the H20 (sm_90) box.
#
#   Source of truth = THIS local git tree (edit + commit here, never on the pod).
#   Pod build tree  = /host/arle-build  — the node's /root/arle-build via the
#                     sglang-test container's host-root hostPath mount. It is
#                     persistent (survives a static-pod restart, unlike the
#                     container overlay) AND it is exactly where `tn push` lands,
#                     so one push updates the build tree with no copy-in hop.
#   Builds are DETACHED (survive disconnect), logged with a BUILD_EXIT marker,
#   and polled — never run in a foreground `tn exec` that a timeout can strand.
#   Kills are by exact recorded PID/PGID — never `pkill -f <our own cmd>` (that
#   self-matches and once corrupted the toolchain mid-install).
#
# Usage:
#   scripts/pod.sh push-scripts                      # push the two pod-side helpers
#   scripts/pod.sh sync [paths...]                   # push files local->pod (default: git-changed)
#   scripts/pod.sh build <label> <cargo-args...>     # detached build; e.g.
#                  build autograd -p autograd --release --features cuda
#   scripts/pod.sh status <label>                    # running? tail log + exit marker
#   scripts/pod.sh log    <label>                    # full build log
#   scripts/pod.sh kill   <label>                    # kill that build by recorded PGID
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
    "$TN" push "$ROOT/scripts/pod-build-env.sh"   "$NODE_TREE/scripts/pod-build-env.sh"
    "$TN" push "$ROOT/scripts/pod-remote-build.sh" "$NODE_TREE/scripts/pod-remote-build.sh"
    echo "pushed pod-side build helpers -> $NODE_TREE/scripts/"
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
  status)
    label="${1:?usage: status <label>}"
    "$POD" "p=\$(cat /root/build-$label.pid 2>/dev/null); \
      if [ -n \"\$p\" ] && kill -0 \$p 2>/dev/null; then echo \"RUNNING pid=\$p\"; else echo 'not running (done or never started)'; fi; \
      echo '--- tail ---'; tail -25 /root/build-$label.log 2>/dev/null; \
      echo '--- marker ---'; grep -E 'BUILD_EXIT|BUILD DONE' /root/build-$label.log 2>/dev/null | tail -2"
    ;;
  log)
    label="${1:?usage: log <label>}"
    "$POD" "cat /root/build-$label.log"
    ;;
  kill)
    label="${1:?usage: kill <label>}"
    "$POD" "p=\$(cat /root/build-$label.pid 2>/dev/null); [ -n \"\$p\" ] && { kill -- -\$p 2>/dev/null; kill \$p 2>/dev/null; echo \"killed pgid \$p\"; } || echo 'no pid recorded'"
    ;;
  *)
    sed -n '2,30p' "$0" | sed 's/^# \?//'
    ;;
esac
