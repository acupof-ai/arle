#!/usr/bin/env bash
# Clean arle multiproc-serve lifecycle for the H20 pod, whose PID 1 is
# `sleep infinity` (never reaps orphans). Runs the serve under reap_run.py (a
# child-subreaper) so every TP worker is reaped HERE — SIGKILLing a hung serve
# no longer leaks permanent zombies. Always stop via this script, never a bare
# `pkill -9 arle` (that orphans the workers).
#
#   scripts/pod_serve.sh start <name> -- <serve args...>
#   scripts/pod_serve.sh stop  <name>      # graceful, then forced; drains the tree
#   scripts/pod_serve.sh status
#   scripts/pod_serve.sh log   <name>
set -u
ROOT="${ARLE_ROOT:-/root/arle}"
RUN="$ROOT/scripts/reap_run.py"
DIR=/tmp/arle_serve
mkdir -p "$DIR"

zombies() { ps -eo stat,comm 2>/dev/null | awk '$2=="arle" && $1 ~ /Z/' | wc -l; }

cmd="${1:-}"
shift || true
case "$cmd" in
  start)
    name="${1:?usage: start <name> -- <serve args...>}"; shift
    [ "${1:-}" = "--" ] && shift
    log="$DIR/$name.log"
    setsid python3 "$RUN" "$ROOT/target/release/arle" serve "$@" >"$log" 2>&1 &
    echo $! >"$DIR/$name.pid"
    echo "started '$name' reaper=$(cat "$DIR/$name.pid") log=$log"
    ;;
  stop)
    name="${1:?usage: stop <name>}"
    f="$DIR/$name.pid"
    [ -f "$f" ] || { echo "no such serve '$name'"; exit 1; }
    p=$(cat "$f")
    kill -TERM "$p" 2>/dev/null                                   # graceful: reaper -> SIGTERM serve group
    for _ in $(seq 1 16); do kill -0 "$p" 2>/dev/null || break; sleep 0.5; done
    if kill -0 "$p" 2>/dev/null; then
      kill -TERM "$p" 2>/dev/null                                 # 2nd TERM -> reaper SIGKILLs serve group
      for _ in $(seq 1 16); do kill -0 "$p" 2>/dev/null || break; sleep 0.5; done
    fi
    rm -f "$f"
    echo "stopped '$name' (reaper drained its tree; zombie arle now: $(zombies))"
    ;;
  status)
    for f in "$DIR"/*.pid; do
      [ -f "$f" ] || continue
      n=$(basename "$f" .pid); p=$(cat "$f")
      if kill -0 "$p" 2>/dev/null; then echo "$n: UP reaper=$p"; else echo "$n: DOWN (stale pidfile)"; fi
    done
    echo "live arle: $(pgrep -x arle 2>/dev/null | wc -l)   zombie arle: $(zombies)"
    ;;
  log)
    name="${1:?usage: log <name>}"
    tail -n "${2:-40}" "$DIR/$name.log"
    ;;
  *)
    echo "usage: pod_serve.sh {start <name> -- <serve args>|stop <name>|status|log <name> [n]}"
    exit 1
    ;;
esac
