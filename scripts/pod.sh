#!/usr/bin/env bash
# pod.sh — local-authoritative / remote-verify devops for the H20 (sm_90) box.
#
#   Source of truth = THIS local git tree (edit + commit here, never on the pod).
#   Pod build tree  = /host/arle-build  — the node's /root/arle-build via the
#                     sglang-test container's host-root hostPath mount. Persistent
#                     (survives a static-pod restart) AND exactly where `tn push`
#                     lands, so one push updates the build tree with no copy-in hop.
# Canonical sequence:
#   scripts/pod.sh push-scripts
#   scripts/pod.sh sync
#   scripts/pod.sh build <label>
#   scripts/pod.sh status <label>
#   scripts/pod.sh run <label> auto -- <args>
#
# Builds/runs are DETACHED (survive disconnect), logged with BUILD_EXIT/RUN_EXIT
# as the completion signal, and polled — never a foreground `tn exec` a timeout
# can strand. Kills are by exact recorded PGID — never `pkill -f <our own cmd>`.
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
#   scripts/pod.sh push-scripts
#   scripts/pod.sh sync
#   scripts/pod.sh build <label>
#   scripts/pod.sh status <label>
#   scripts/pod.sh run <label> auto -- <args>
#   scripts/pod.sh gpus                    # per-GPU memory/util
#   scripts/pod.sh ready 8000 900          # block until engine-ready
#   Other forms: build <label> <cargo-args> | run <label> <gpu> -- <args>
#                | status/log/kill <label> | setup (warm/repair toolchain)
#   Compile cache: setup-sccache (install binary; cache on /host persists) | sccache-stats
#   (RUSTC_WRAPPER=sccache auto-engages once installed — gives fresh-tree/cross-restart reuse)
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
    for s in pod-build-env.sh pod-remote-build.sh pod-remote-run.sh pod-tilelang-env.sh pick-gpu.sh; do
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
  setup-sccache)
    # Install the sccache binary (ephemeral /root/.cargo/bin — re-run after a pod
    # restart; the cache itself lives on persistent /host/sccache). Idempotent.
    "$POD" "if command -v sccache >/dev/null 2>&1; then sccache --version; else \
      V=v0.8.2; export all_proxy=socks5h://127.0.0.1:1080; \
      curl -fsSL --proxy socks5h://127.0.0.1:1080 -o /tmp/sccache.tgz \
        https://github.com/mozilla/sccache/releases/download/\$V/sccache-\$V-x86_64-unknown-linux-musl.tar.gz && \
      tar -C /tmp -xzf /tmp/sccache.tgz && \
      install -m755 /tmp/sccache-\$V-x86_64-unknown-linux-musl/sccache /root/.cargo/bin/sccache && \
      sccache --version; fi"
    ;;
  sccache-stats)
    "$POD" "sccache --show-stats 2>/dev/null | grep -iE 'compile requests|cache hits|cache misses|cache hit rate|stored|errors' | head"
    ;;
  setup-tilelang)
    # Codegen venv + ARLE-patched apache-tvm-ffi; the gate is `import tilelang`
    # in THE venv (a system tilelang no longer short-circuits — it was version-
    # blind). Idempotent; see scripts/pod-tilelang-env.sh.
    "$POD" "bash $TREE/scripts/pod-tilelang-env.sh"
    ;;
  sync)
    # Explicit paths => push exactly those. No args => make the pod tree EQUAL
    # this tree: committed state via pod-side git fetch+reset to our HEAD (a
    # `git status`-only push misses committed work), then dirty files on top.
    # git-reset also writes fresh mtimes, so cargo can't mistake the new
    # sources for "fresh" (tn push preserves local mtimes; see remote-build).
    if [ $# -gt 0 ]; then
      for p in "$@"; do
        [ -f "$ROOT/$p" ] || { echo "skip (not a file): $p"; continue; }
        "$TN" push "$ROOT/$p" "$NODE_TREE/$p" && echo "synced  $p"
      done
      exit 0
    fi
    head="$(git -C "$ROOT" rev-parse HEAD)"
    pod_head="$("$POD" "git -C $TREE rev-parse HEAD 2>/dev/null" | tr -d '\r\n')"
    if [ "$pod_head" = "$head" ]; then
      echo "pod tree already @ $head"
    else
      # Pod-side `git fetch origin` hangs when the proxy is down (observed
      # 2026-07-02); a git bundle rides the same tn lane as file pushes —
      # no pod-side network. git-reset also writes fresh mtimes.
      git -C "$ROOT" rev-parse -q --verify "$pod_head^{commit}" >/dev/null 2>&1 \
        || { echo "pod HEAD $pod_head unknown locally — re-provision the pod tree"; exit 1; }
      bundle="$(mktemp -t arle-sync-XXXX).bundle"
      git -C "$ROOT" bundle create "$bundle" "$pod_head..HEAD" >/dev/null 2>&1
      "$TN" push "$bundle" "$NODE_TREE/.arle-sync.bundle"
      rm -f "$bundle"
      "$POD" "cd $TREE && git fetch -q .arle-sync.bundle HEAD && \
        git reset --hard -q FETCH_HEAD && rm -f .arle-sync.bundle && \
        echo \"pod tree @ \$(git log --oneline -1)\"" \
        || { echo "pod bundle apply failed"; exit 1; }
    fi
    while IFS= read -r p; do
      [ -f "$ROOT/$p" ] || continue
      "$TN" push "$ROOT/$p" "$NODE_TREE/$p" && echo "synced  $p (dirty)"
    done < <(cd "$ROOT" && git status --porcelain | awk '{print $2}')
    ;;
  build)
    # First arg (if any) = label; remaining args = cargo args. Missing cargo
    # args ALWAYS fall back to the standard arle build — `build <label>` used
    # to silently launch a plain debug `cargo build` (unusable on GPU).
    if [ $# -eq 0 ]; then label="arle"; else label="$1"; shift; fi
    [ $# -eq 0 ] && set -- --release --features cuda --bin arle
    "$POD" "POD_TREE=$TREE setsid bash $TREE/scripts/pod-remote-build.sh $label $* </dev/null >/dev/null 2>&1 &"
    echo "build '$label' launched (detached). poll: scripts/pod.sh status $label"
    ;;
  run)
    # Tokens before `--`: 0 => auto GPU + auto label; 1 => label; 2 => label gpu.
    pre=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do pre+=("$1"); shift; done
    [ "${1:-}" = "--" ] && shift
    label="${pre[0]:-}"; gpu="${pre[1]:-auto}"
    if [ "$gpu" = "auto" ]; then
      gpu="$("$POD" "bash $TREE/scripts/pick-gpu.sh" 2>/dev/null | tail -1 | tr -d '\r\n ')"
      if ! [[ "$gpu" =~ ^[0-9]+$ ]]; then echo "no free GPU (all >2GB used or claimed)"; exit 1; fi
    fi
    [ -z "$label" ] && label="g$gpu"   # default label derived from the GPU => collision-free across agents
    "$POD" "POD_TREE=$TREE setsid bash $TREE/scripts/pod-remote-run.sh $label $gpu $* </dev/null >/dev/null 2>&1 &"
    echo "run '$label' launched on GPU $gpu. poll: scripts/pod.sh status $label"
    ;;
  gpus)
    # Per-GPU usage + per-holder attribution. A holder PID visible in our
    # /proc is ours (cmdline shown, killable here). An invisible one is NOT
    # proof it's foreign — nvidia-smi prints host PIDs, so it may be another
    # container's process OR our own from a pre-restart container instance
    # (observed 2026-07-16: our arm-A serve read as "foreign"). Attribute and
    # kill from the host: `tn exec 'tr "\0" " " </proc/<pid>/cmdline'`.
    "$POD" "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader; \
      map=\$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader); \
      apps=\$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader 2>/dev/null); \
      [ -n \"\$apps\" ] && echo '--- holders ---'; \
      echo \"\$apps\" | while IFS=',' read -r uuid pid mem; do \
        [ -n \"\$pid\" ] || continue; pid=\$(echo \"\$pid\" | tr -d ' '); uuid=\$(echo \"\$uuid\" | tr -d ' '); \
        idx=\$(echo \"\$map\" | grep \"\$uuid\" | cut -d',' -f1 | tr -d ' '); \
        if [ -d \"/proc/\$pid\" ]; then who=\"ours: \$(tr '\0' ' ' </proc/\$pid/cmdline | cut -c1-90)\"; \
        else who='NOT VISIBLE from this container (other container OR pre-restart leftover of ours) — attribute/kill from host via tn exec'; fi; \
        echo \"gpu \${idx:-?} pid \$pid\$mem — \$who\"; done"
    ;;
  ready)
    # Poll engine-ready (HTTP 200 on /v1/models). Usage: ready [port] [timeout_s].
    port="${1:-8000}"; timeout_s="${2:-1200}"
    case "$port$timeout_s" in *[!0-9]*) echo "usage: pod.sh ready [port] [timeout_s]"; exit 2;; esac
    "$POD" "code=; for i in \$(seq 1 $(( timeout_s / 10 ))); do \
      code=\$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$port/v1/models 2>/dev/null); \
      [ \"\$code\" = 200 ] && { echo \"ENGINE_READY port=$port after ~\$((i*10))s\"; exit 0; }; sleep 10; done; \
      echo \"NOT_READY after ${timeout_s}s (last http \${code:-none})\"; exit 1"
    ;;
  status)
    label="${1:-arle}"
    case "$label" in *[!A-Za-z0-9_.-]*) echo "invalid label: $label"; exit 2;; esac
    "$POD" "found=0; for k in build run; do f=/root/\$k-$label; [ -f \$f.log ] || continue; found=1; \
      p=\$(cat \$f.pid 2>/dev/null); \
      stat=; [ -n \"\$p\" ] && stat=\$(ps -p \$p -o stat= 2>/dev/null | tr -d ' ' || true); \
      if [ -n \"\$stat\" ] && ! echo \"\$stat\" | grep -q Z; then echo \"[\$k] RUNNING pid=\$p stat=\$stat\"; \
      elif [ -n \"\$stat\" ]; then echo \"[\$k] not running (zombie/done) pid=\$p stat=\$stat\"; \
      else echo \"[\$k] not running (done or never started)\"; fi; \
      echo '--- tail ---'; tail -20 \$f.log 2>/dev/null; \
      echo '--- marker ---'; grep -E 'BUILD_EXIT|RUN_EXIT|DONE' \$f.log 2>/dev/null | tail -2; done; \
      [ \$found -eq 1 ] || echo \"no build/run logs for label '$label'\""
    ;;
  log)
    label="${1:-arle}"
    case "$label" in *[!A-Za-z0-9_.-]*) echo "invalid label: $label"; exit 2;; esac
    "$POD" "for k in build run; do [ -f /root/\$k-$label.log ] && { echo \"==== \$k-$label.log ====\"; cat /root/\$k-$label.log; }; done"
    ;;
  kill)
    label="${1:-arle}"
    case "$label" in *[!A-Za-z0-9_.-]*) echo "invalid label: $label"; exit 2;; esac
    "$POD" "for k in build run; do p=\$(cat /root/\$k-$label.pid 2>/dev/null); \
      [ -n \"\$p\" ] && { kill -- -\$p 2>/dev/null; kill \$p 2>/dev/null; echo \"killed \$k pgid \$p\"; }; done"
    ;;
  *)
    sed -n '2,39p' "$0" | sed 's/^# \?//'
    ;;
esac
