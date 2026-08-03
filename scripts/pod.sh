#!/usr/bin/env bash
set -euo pipefail

POD="${POD:-$HOME/bin/pod}"
TN="${TN:-tn}"
# One tree seen from two sides: `tn push` writes NODE_TREE, the pod reads POD_TREE.
# Overriding one alone stages the sync tarball at the default node path and then
# tells the pod to read the overridden one — "incomplete sync stage", far from the
# cause (2026-08-04). Checked before the defaults land, or both always look set.
if { [ -n "${POD_TREE:-}" ] && [ -z "${NODE_TREE:-}" ]; } ||
   { [ -z "${POD_TREE:-}" ] && [ -n "${NODE_TREE:-}" ]; }; then
  echo "POD_TREE and NODE_TREE must be set together (same tree, pod side and node side); got POD_TREE='${POD_TREE:-}' NODE_TREE='${NODE_TREE:-}'" >&2
  exit 2
fi
NODE_TREE="${NODE_TREE:-/root/arle-build}"
TREE="${POD_TREE:-/host/arle-build}"
STATE="${POD_STATE:-/root/arle-ops}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cmd="${1:-help}"
shift || true

valid_label() {
  case "$1" in ""|*[!A-Za-z0-9_.-]*) echo "invalid label: $1" >&2; exit 2;; esac
}
valid_gpu() {
  local gpu="$1" seen="," device
  [ "$gpu" = auto ] && return
  IFS=',' read -r -a devices <<< "$gpu"
  [ "${#devices[@]}" -gt 0 ] || { echo "invalid GPU: $gpu" >&2; exit 2; }
  for device in "${devices[@]}"; do
    case "$device" in ""|*[!0-9]*) echo "invalid GPU: $gpu" >&2; exit 2;; esac
    case "$seen" in *",$device,"*) echo "invalid GPU: $gpu" >&2; exit 2;; esac
    seen="$seen$device,"
  done
}
new_label() {
  printf '%s-%s-%s\n' "$1" "$(date -u +%Y%m%dT%H%M%SZ)" "$$-$RANDOM"
}
args_file() {
  local out="$1"
  shift
  printf '%s\0' "$@" > "$out"
}
push_or_die() {
  "$TN" push "$1" "$2" || { echo "push failed; remote tree unchanged" >&2; exit 1; }
}
pod_path() {
  case "$1" in
    "$NODE_TREE"*) printf '%s%s\n' "$TREE" "${1#"$NODE_TREE"}" ;;
    *) echo "node path is outside NODE_TREE: $1" >&2; exit 2 ;;
  esac
}
# NUL-list of tarball contents: git-tracked + untracked (honors .gitignore), plus
# the AOT bundle in generated/ (gitignored → shipped explicitly so the pod reuses
# it, not outside source_digest so it can't trip apply-sync's guard).
tarball_files() {
  { git -C "$ROOT" ls-files -co --exclude-standard -z
    ( cd "$ROOT" && [ -d crates/cuda-kernels/generated ] && find crates/cuda-kernels/generated -type f -print0; ) || :; } |
    while IFS= read -r -d '' p; do [ -f "$ROOT/$p" ] && printf '%s\0' "$p"; done
}

case "$cmd" in
  push-scripts)
    for s in pod-build-env.sh pod-remote-build.sh pod-remote-run.sh pod-tilelang-env.sh pick-gpu.sh reap_run.py cuda_prebuilt_manifest.sh; do
      push_or_die "$ROOT/scripts/$s" "$NODE_TREE/scripts/$s"
    done
    echo "pushed pod helpers -> $NODE_TREE/scripts/"
    ;;
  setup)
    "$POD" "flock /tmp/arle-toolchain.lock bash -lc 'if ls ~/.rustup/toolchains/1.95.0-*/lib/rustlib/*/lib/libstd-*.rlib >/dev/null 2>&1; then echo toolchain-1.95.0-OK; else rustup toolchain install 1.95.0 --profile minimal -c rustfmt -c clippy; fi'"
    ;;
  setup-sccache)
    "$POD" "if command -v sccache >/dev/null 2>&1; then sccache --version; else V=v0.8.2; export all_proxy=socks5h://127.0.0.1:1080; curl -fsSL --proxy socks5h://127.0.0.1:1080 -o /tmp/sccache.tgz https://github.com/mozilla/sccache/releases/download/\$V/sccache-\$V-x86_64-unknown-linux-musl.tar.gz && tar -C /tmp -xzf /tmp/sccache.tgz && install -m755 /tmp/sccache-\$V-x86_64-unknown-linux-musl/sccache /root/.cargo/bin/sccache && sccache --version; fi"
    ;;
  sccache-stats)
    "$POD" "sccache --show-stats 2>/dev/null | grep -iE 'compile requests|cache hits|cache misses|cache hit rate|stored|errors' | head"
    ;;
  setup-tilelang)
    "$POD" "bash '$TREE/scripts/pod-tilelang-env.sh'"
    ;;
  sync)
    [ $# -eq 0 ] || { echo "sync is full-tree only" >&2; exit 2; }
    stage="$(mktemp -d -t arle-sync-XXXXXX)"
    trap 'rm -rf "$stage"' EXIT
    head="$(git -C "$ROOT" rev-parse HEAD)"
    dirty_digest="$(POD_TREE="$ROOT" bash "$ROOT/scripts/pod-remote-build.sh" source-digest "$ROOT")"
    bash "$ROOT/scripts/kernel_artifacts.sh" sync || true   # source-matched AOT bundle → generated/ (no-op offline/miss)
    tarball_files > "$stage/files"
    git -C "$ROOT" ls-files -d -z > "$stage/deletes"
    COPYFILE_DISABLE=1 tar -C "$ROOT" --null -T "$stage/files" -czf "$stage/tree.tgz"
    archive_sha="$(shasum -a 256 "$stage/tree.tgz" | cut -d' ' -f1)"
    git -C "$ROOT" bundle create "$stage/source.bundle" HEAD
    bundle_sha="$(shasum -a 256 "$stage/source.bundle" | cut -d' ' -f1)"
    printf 'schema=arle-source-stage-v1\nhead=%s\ndirty_digest=%s\narchive_sha=%s\nbundle_sha=%s\n' "$head" "$dirty_digest" "$archive_sha" "$bundle_sha" > "$stage/source.meta"
    remote_stage="$NODE_TREE.sync.$$.${RANDOM}"
    push_or_die "$stage/tree.tgz" "$remote_stage.tree.tgz"
    push_or_die "$stage/deletes" "$remote_stage.deletes"
    push_or_die "$stage/source.bundle" "$remote_stage.source.bundle"
    push_or_die "$stage/source.meta" "$remote_stage.source.meta"
    pod_stage="$(pod_path "$remote_stage")"
    "$POD" "POD_TREE='$TREE' POD_STATE='$STATE' bash '$TREE/scripts/pod-remote-build.sh' apply-sync '$pod_stage'" || { echo "remote sync apply failed" >&2; exit 1; }
    ;;
  build)
    label="${1:-}"
    if [ -n "$label" ]; then shift; else label="$(new_label build)"; fi
    valid_label "$label"
    [ $# -gt 0 ] || set -- --release --features cuda --bin arle
    tmp="$(mktemp -t arle-build-argv-XXXXXX)"
    trap 'rm -f "$tmp"' EXIT
    args_file "$tmp" "$@"
    bash "$ROOT/scripts/pod-remote-build.sh" validate-build-args "$tmp" >/dev/null
    op="build-$label-$(date +%s)-$$-$RANDOM"
    remote="$NODE_TREE.build-$op.argv"
    push_or_die "$tmp" "$remote"
    pod_remote="$(pod_path "$remote")"
    "$POD" "POD_TREE='$TREE' POD_STATE='$STATE' setsid bash '$TREE/scripts/pod-remote-build.sh' build '$label' '$op' '$pod_remote' </dev/null >/dev/null 2>&1 &"
    echo "build '$label' launched; receipt: build:$label"
    ;;
  run)
    build="${1:-}"
    [ -n "$build" ] || { echo "usage: pod.sh run <build-label> [run-label] [auto|GPU|GPU,...] -- <args>" >&2; exit 2; }
    shift
    valid_label "$build"
    pre=()
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do pre+=("$1"); shift; done
    [ "${1:-}" = "--" ] && shift
    label="${pre[0]:-$(new_label run)}"
    gpu="${pre[1]:-auto}"
    valid_label "$label"
    valid_gpu "$gpu"
    op="run-$label-$(date +%s)-$$-$RANDOM"
    tmp="$(mktemp -t arle-run-argv-XXXXXX)"
    trap 'rm -f "$tmp"' EXIT
    args_file "$tmp" "$@"
    remote="$NODE_TREE.run-$op.argv"
    push_or_die "$tmp" "$remote"
    pod_remote="$(pod_path "$remote")"
    "$POD" "POD_TREE='$TREE' POD_STATE='$STATE' setsid bash '$TREE/scripts/pod-remote-run.sh' run '$build' '$label' '$gpu' '$op' '$pod_remote' </dev/null >/dev/null 2>&1 &"
    echo "run '$label' launched from build:$build; GPU=$gpu"
    ;;
  gpus)
    "$POD" "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader"
    ;;
  ready)
    label="${1:-}"; timeout_s="${2:-1200}"
    valid_label "$label"
    case "$timeout_s" in ''|*[!0-9]*) echo "usage: pod.sh ready <run-label> [timeout_s]" >&2; exit 2;; esac
    "$POD" "POD_TREE='$TREE' POD_STATE='$STATE' bash '$TREE/scripts/pod-remote-run.sh' ready '$label' '$timeout_s'"
    ;;
  status|log|kill)
    label="${1:-}"
    valid_label "$label"
    "$POD" "POD_TREE='$TREE' POD_STATE='$STATE' bash '$TREE/scripts/pod-remote-run.sh' '$cmd' '$label'"
    ;;
  *)
    printf '%s\n' \
      'pod.sh sync' \
      'pod.sh build [label] [cargo argv...]' \
      'pod.sh run <build-label> [run-label] [auto|GPU|GPU,...] -- [arle argv...]' \
      'pod.sh status|ready|log|kill <run-label> [timeout]' \
      'pod.sh gpus | setup | setup-sccache | sccache-stats'
    ;;
esac
