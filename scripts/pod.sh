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
# Per-lane tree: running pod.sh from a lane worktree defaults to that lane's
# remote tree, so one lane's sync cannot land between another lane's sync and
# build. Override with POD_TREE/NODE_TREE (must be set together, checked above).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_lane_tree=""
if [ -z "${POD_TREE:-}" ] && [ -z "${NODE_TREE:-}" ]; then
  case "$ROOT" in
    */arle-lanes/*) _lane_tree="$(basename "$ROOT")" ;;
  esac
fi
if [ -n "$_lane_tree" ]; then
  NODE_TREE="/root/arle-build-$_lane_tree"
  TREE="/host/arle-build-$_lane_tree"
  # Per-lane state dir too: the shared /root/arle-ops keys builds/ and runs/ by
  # label, so two lanes with the same label clobbered one receipt dir and a run
  # could resolve another tree's build. Explicit POD_STATE still overrides.
  STATE="${POD_STATE:-/root/arle-ops-$_lane_tree}"
else
  NODE_TREE="${NODE_TREE:-/root/arle-build}"
  TREE="${POD_TREE:-/host/arle-build}"
  STATE="${POD_STATE:-/root/arle-ops}"
fi
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
# The held tn connection gives one transfer 60s before it reports the daemon
# gone; a 47 MB tree.tgz never finishes inside that. The constant encodes an
# assumed transfer rate, so a slower link needs a smaller PUSH_CHUNK_BYTES.
PUSH_CHUNK_BYTES="${PUSH_CHUNK_BYTES:-12582912}"
# One transfer, retried: the held tn connection drops often enough that a
# single-attempt push fails a whole sync for no lasting reason.
push_one() {
  local src="$1" dst="$2" attempt
  for attempt in 1 2 3; do
    "$TN" push "$src" "$dst" && return 0
    [ "$attempt" = 3 ] || { echo "push $dst failed (attempt $attempt), retrying" >&2; sleep 5; }
  done
  return 1
}
push_or_die() {
  local src="$1" dst="$2" size
  # stat, not `wc -c`: the latter reads all 47 MB just to report the length.
  size="$(stat -f%z "$src" 2>/dev/null || stat -c%s "$src")"
  if [ "$size" -le "$PUSH_CHUNK_BYTES" ]; then
    push_one "$src" "$dst" || { echo "push $dst failed; remote tree unchanged" >&2; exit 1; }
    return 0
  fi
  local parts
  parts="$(mktemp -d -t arle-push-XXXXXX)"
  trap 'rm -rf "$parts"' RETURN
  # split's own suffixes (part.aa, part.ab, ...) are already glob-ordered, so
  # the remote `cat` reassembles in the order they were written.
  split -b "$PUSH_CHUNK_BYTES" "$src" "$parts/part."
  for part in "$parts"/part.*; do
    push_one "$part" "$dst.$(basename "$part")" ||
      { echo "push $dst $(basename "$part") failed; remote tree unchanged" >&2; exit 1; }
  done
  "$TN" exec "cat '$dst'.part.* > '$dst' && rm -f '$dst'.part.*" >/dev/null || {
    echo "reassembling $dst on the node failed; remote tree unchanged" >&2; exit 1; }
}
# Ship the local helpers to the tree. sync does this before apply-sync because
# the remote apply runs the deployed tree's script: a digest-format change
# (e.g. the blob-identity hash) would otherwise verify with the old format and
# fail every sync until a human ran push-scripts by hand. push_scripts itself
# only transfers files (tn push); it never invokes the remote script, so no
# protocol change can deadlock it — the bootstrap is one-directional.
push_scripts() {
  for s in pod-build-env.sh pod-remote-build.sh pod-remote-run.sh pod-tilelang-env.sh pick-gpu.sh reap_run.py cuda_prebuilt_manifest.sh; do
    push_or_die "$ROOT/scripts/$s" "$NODE_TREE/scripts/$s"
  done
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
    push_scripts
    echo "pushed pod helpers -> $NODE_TREE/scripts/"
    ;;
  setup)
    "$POD" "flock /tmp/arle-toolchain.lock bash -lc 'if ls ~/.rustup/toolchains/1.98.0-*/lib/rustlib/*/lib/libstd-*.rlib >/dev/null 2>&1; then echo toolchain-1.98.0-OK; else rustup toolchain install 1.98.0 --profile minimal -c rustfmt -c clippy; fi'"
    ;;
  setup-sccache)
    "$POD" "if command -v sccache >/dev/null 2>&1; then sccache --version; else V=v0.8.2; export all_proxy=socks5h://127.0.0.1:1080; curl -fsSL --proxy socks5h://127.0.0.1:1080 -o /tmp/sccache.tgz https://github.com/mozilla/sccache/releases/download/\$V/sccache-\$V-x86_64-unknown-linux-musl.tar.gz && tar -C /tmp -xzf /tmp/sccache.tgz && install -m755 /tmp/sccache-\$V-x86_64-unknown-linux-musl/sccache /root/.cargo/bin/sccache && sccache --version; fi"
    ;;
  sccache-stats)
    "$POD" "sccache --show-stats 2>/dev/null | grep -iE 'compile requests|cache hits|cache misses|cache hit rate|stored|errors' | head"
    ;;
  setup-tilelang)
    "$POD" "POD_TREE='$TREE' bash '$TREE/scripts/pod-tilelang-env.sh'"
    ;;
  quantize-w4afp8)
    input="${1:?usage: pod.sh quantize-w4afp8 <input-dir> <output-dir>}"
    output="${2:?usage: pod.sh quantize-w4afp8 <input-dir> <output-dir>}"
    "$POD" "bash -lc 'cd $TREE && ${ARLE_TILELANG_VENV:-/root/arle-ops/tilelang-venv}/bin/python scripts/quantize_dsv4_w4afp8.py \"$input\" \"$output\"'"
    ;;
  sync)
    dirty=0; full=0
    while [ $# -gt 0 ]; do
      case "$1" in
        --dirty) dirty=1 ;;
        --full) full=1 ;;
        *) echo "sync: unknown arg $1" >&2; exit 2 ;;
      esac
      shift
    done
    stage="$(mktemp -d -t arle-sync-XXXXXX)"
    trap 'rm -rf "$stage"' EXIT
    head="$(git -C "$ROOT" rev-parse HEAD)"
    # The bundle lands in generated/ (inside the digest), so materialise it
    # before digesting or the remote apply-sync guard sees a different tree.
    bash "$ROOT/scripts/kernel_artifacts.sh" sync || true   # source-matched AOT bundle → generated/ (no-op offline/miss)
    if [ "$dirty" = 1 ]; then
      # Dirty sync: ship the working tree, including uncommitted and untracked.
      dirty_digest="$(POD_TREE="$ROOT" bash "$ROOT/scripts/pod-remote-build.sh" source-digest "$ROOT")"
      tarball_files > "$stage/files"
      git -C "$ROOT" ls-files -d -z > "$stage/deletes"
      COPYFILE_DISABLE=1 tar --no-xattrs -C "$ROOT" --null -T "$stage/files" -czf "$stage/tree.tgz"
    else
      # Clean sync: ship the committed tree. The tarball is `git archive HEAD`
      # plus the gitignored AOT bundle; no working-tree changes leave the box.
      dirty_digest="$(POD_TREE="$ROOT" CLEAN=1 bash "$ROOT/scripts/pod-remote-build.sh" source-digest "$ROOT")"
      mkdir -p "$stage/clean"
      git -C "$ROOT" archive HEAD | tar -x -C "$stage/clean"
      if [ -d "$ROOT/crates/cuda-kernels/generated" ]; then
        mkdir -p "$stage/clean/crates/cuda-kernels"
        cp -R "$ROOT/crates/cuda-kernels/generated" "$stage/clean/crates/cuda-kernels/"
      fi
      : > "$stage/deletes"
      # Tar from a file list, not `.`: `tar ... .` prefixes every entry with
      # `./`, and reconcile_untracked compares against `git ls-files` (no
      # prefix), so every tracked file read as a stray and got deleted.
      ( cd "$stage/clean" && find . \( -type f -o -type l \) | cut -c3- ) > "$stage/clean.files"
      COPYFILE_DISABLE=1 tar --no-xattrs -C "$stage/clean" -T "$stage/clean.files" -czf "$stage/tree.tgz"
    fi
    archive_sha="$(shasum -a 256 "$stage/tree.tgz" | cut -d' ' -f1)"
    pod_head="$("$POD" "git -C '$TREE' rev-parse HEAD" 2>/dev/null | tr -d '[:space:]' || true)"
    bundle_mode=full
    if [ "$full" = 0 ] && [ "$pod_head" = "$head" ]; then
      bundle_mode=none
    elif [ "$full" = 0 ] && [ -n "$pod_head" ] && git -C "$ROOT" merge-base --is-ancestor "$pod_head" HEAD 2>/dev/null; then
      bundle_mode=incremental
      git -C "$ROOT" bundle create "$stage/source.bundle" "$pod_head"..HEAD
    else
      git -C "$ROOT" bundle create "$stage/source.bundle" HEAD
    fi
    if [ "$bundle_mode" = none ]; then bundle_sha=none; else bundle_sha="$(shasum -a 256 "$stage/source.bundle" | cut -d' ' -f1)"; fi
    printf 'schema=arle-source-stage-v1\nhead=%s\ndirty=%s\ndirty_digest=%s\narchive_sha=%s\nbundle_sha=%s\nbundle_mode=%s\n' "$head" "$dirty" "$dirty_digest" "$archive_sha" "$bundle_sha" "$bundle_mode" > "$stage/source.meta"
    remote_stage="$NODE_TREE.sync.$$.${RANDOM}"
    push_or_die "$stage/tree.tgz" "$remote_stage.tree.tgz"
    push_or_die "$stage/deletes" "$remote_stage.deletes"
    [ "$bundle_mode" = none ] || push_or_die "$stage/source.bundle" "$remote_stage.source.bundle"
    push_or_die "$stage/source.meta" "$remote_stage.source.meta"
    pod_stage="$(pod_path "$remote_stage")"
    # ARLE_SKIP_PUSH_SCRIPTS is the negative-control hook for the bootstrap
    # fix (test_pod_flow.sh): a protocol-skewed remote must fail without it.
    [ -n "${ARLE_SKIP_PUSH_SCRIPTS:-}" ] || push_scripts
    "$POD" "POD_TREE='$TREE' POD_STATE='$STATE' bash '$TREE/scripts/pod-remote-build.sh' apply-sync '$pod_stage'" || { echo "remote sync apply failed" >&2; exit 1; }
    ;;
  build)
    label="${1:-}"
    if [ -n "$label" ]; then shift; else label="$(new_label build)"; fi
    valid_label "$label"
    [ $# -gt 0 ] || set -- --profile release-fast --features cuda,nccl --bin arle
    tmp="$(mktemp -t arle-build-argv-XXXXXX)"
    trap 'rm -f "$tmp"' EXIT
    args_file "$tmp" "$@"
    bash "$ROOT/scripts/pod-remote-build.sh" validate-build-args "$tmp" >/dev/null
    op="build-$label-$(date +%s)-$$-$RANDOM"
    remote="$NODE_TREE.build-$op.argv"
    push_or_die "$tmp" "$remote"
    pod_remote="$(pod_path "$remote")"
    "$POD" "POD_TREE='$TREE' POD_STATE='$STATE' ${ARLE_BUNDLE_DONOR:+ARLE_BUNDLE_DONOR='$ARLE_BUNDLE_DONOR' }setsid bash '$TREE/scripts/pod-remote-build.sh' build '$label' '$op' '$pod_remote' </dev/null >/dev/null 2>&1 &"
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
  kernel-ab)
    build="${1:-}"
    [ -n "$build" ] || { echo "usage: pod.sh kernel-ab <build-label> [run-label] [auto|GPU] [shape] [iters]" >&2; exit 2; }
    label="${2:-$(new_label kab)}"; gpu="${3:-auto}"; shape="${4:-1,34816,5120}"; iters="${5:-100}"
    valid_label "$build"; valid_label "$label"
    [ "$gpu" = auto ] || case "$gpu" in ''|*[!0-9]*) echo "invalid GPU: $gpu (auto or one index)" >&2; exit 2;; esac
    case "$iters" in ''|*[!0-9]*) echo "invalid iters: $iters" >&2; exit 2;; esac
    "$POD" "POD_TREE='$TREE' POD_STATE='$STATE' setsid bash '$TREE/scripts/pod-remote-run.sh' kernel-ab '$build' '$label' '$gpu' '$shape' '$iters' </dev/null >/dev/null 2>&1 &"
    echo "kernel-ab '$label' launched from build:$build; GPU=$gpu shape=$shape iters=$iters"
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
  status)
    # No arg: the tree + latest build sha (W4). With a run-label: run status.
    if [ $# -eq 0 ]; then
      "$POD" "POD_TREE='$TREE' POD_STATE='$STATE' bash '$TREE/scripts/pod-remote-build.sh' tree-status"
    else
      label="$1"; valid_label "$label"
      "$POD" "POD_TREE='$TREE' POD_STATE='$STATE' bash '$TREE/scripts/pod-remote-run.sh' '$cmd' '$label'"
    fi
    ;;
  log|kill)
    label="${1:-}"
    valid_label "$label"
    "$POD" "POD_TREE='$TREE' POD_STATE='$STATE' bash '$TREE/scripts/pod-remote-run.sh' '$cmd' '$label'"
    ;;
  *)
    printf '%s\n' \
      'pod.sh sync [--dirty] [--full]' \
      'pod.sh build [label] [cargo argv...]' \
      'pod.sh run <build-label> [run-label] [auto|GPU|GPU,...] -- [arle argv...]' \
      'pod.sh kernel-ab <build-label> [run-label] [auto|GPU] [shape] [iters]' \
      'pod.sh status [run-label] (no arg: tree head + latest build sha)' \
      'pod.sh ready|log|kill <run-label> [timeout]' \
      'pod.sh gpus | setup | setup-sccache | sccache-stats'
    ;;
esac
