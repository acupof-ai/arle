#!/usr/bin/env bash
set -uo pipefail

TREE="${POD_TREE:-/host/arle-build}"
STATE="${POD_STATE:-/root/arle-ops}"
TREE_LOCK="/tmp/arle-build$(printf '%s' "$TREE" | tr '/.' '__').lock"
mkdir -p "$STATE/builds" "$STATE/runs"

proc_start() { awk '{print $22}' "/proc/$1/stat" 2>/dev/null; }
sha256() { sha256sum "$1" | cut -d' ' -f1; }
dirty_digest() {
  local tmp
  tmp="$(mktemp)" || return 1
  git -C "$TREE" status --porcelain=v1 -z -- . ':(exclude).arle-source-receipt' > "$tmp" && sha256 "$tmp"
  rm -f "$tmp"
}
write_receipt() {
  local dst="$1" tmp="$1.tmp.$$"
  shift
  printf '%s\n' "$@" > "$tmp" && mv "$tmp" "$dst"
}

case "${1:-}" in
  apply-sync)
    stage="${2:?missing sync stage}"
    exec 9>"$TREE_LOCK"
    flock 9
    meta="$stage.source.meta"
    archive="$stage.tree.tgz"
    deletes="$stage.deletes"
    bundle="$stage.source.bundle"
    [ -f "$meta" ] && [ -f "$archive" ] && [ -f "$deletes" ] && [ -f "$bundle" ] || { echo "incomplete sync stage" >&2; exit 1; }
    archive_sha="$(awk -F= '$1=="archive_sha" {print $2}' "$meta")"
    bundle_sha="$(awk -F= '$1=="bundle_sha" {print $2}' "$meta")"
    [ "$(sha256 "$archive")" = "$archive_sha" ] && [ "$(sha256 "$bundle")" = "$bundle_sha" ] || { echo "sync digest mismatch" >&2; exit 1; }
    parent="$(dirname "$TREE")"
    incoming="$parent/.arle-incoming.$$"
    backup="$parent/.arle-backup.$$"
    git clone -q "$bundle" "$incoming"
    tar -C "$incoming" -xzf "$archive"
    while IFS= read -r -d '' path; do rm -f "$incoming/$path"; done < "$deletes"
    if ! mv "$TREE" "$backup" || ! mv "$incoming" "$TREE"; then
      [ -d "$backup" ] && mv "$backup" "$TREE"
      exit 1
    fi
    head="$(awk -F= '$1=="head" {print $2}' "$meta")"
    actual_head="$(git -C "$TREE" rev-parse HEAD)"
    expected_digest="$(awk -F= '$1=="dirty_digest" {print $2}' "$meta")"
    actual_digest="$(dirty_digest)"
    if [ "$actual_head" != "$head" ] || [ "$actual_digest" != "$expected_digest" ]; then
      rm -rf "$TREE"; mv "$backup" "$TREE"; echo "sync source mismatch" >&2; exit 1
    fi
    receipt="$TREE/.arle-source-receipt"
    write_receipt "$receipt" "schema=arle-source-v1" "head=$actual_head" "digest=$actual_digest" "archive_sha=$archive_sha" "bundle_sha=$bundle_sha" "applied_at=$(date -u +%FT%TZ)"
    rm -rf "$backup" "$archive" "$deletes" "$bundle" "$meta"
    echo "synced source head=$actual_head digest=$actual_digest"
    ;;
  build)
    LABEL="${2:?missing label}"; OP="${3:?missing operation}"; ARGV_FILE="${4:?missing argv file}"
    DIR="$STATE/builds/$LABEL"
    [ ! -e "$DIR" ] || { echo "build label exists: $LABEL" >&2; exit 1; }
    mkdir "$DIR" || exit 1
    mv "$ARGV_FILE" "$DIR/argv.nul" || exit 1
    LOG="$DIR/log"; RECEIPT="$DIR/receipt"
    echo $$ > "$DIR/pid"
    printf '%s\n' "op=$OP" "pid=$$" "pgid=$(ps -o pgid= -p $$ | tr -d ' ')" "start=$(proc_start $$)" > "$DIR/process"
    exec >"$LOG" 2>&1
    source "$TREE/scripts/pod-build-env.sh"
    cd "$TREE" || exit 1
    source_receipt="$TREE/.arle-source-receipt"
    rc=1; binary=""
    if [ ! -f "$source_receipt" ]; then
      echo "source receipt required"
    else
      source_head="$(awk -F= '$1=="head" {print $2}' "$source_receipt")"
      source_digest="$(awk -F= '$1=="digest" {print $2}' "$source_receipt")"
      argv_dump="$(python3 - "$DIR/argv.nul" <<'PY'
import shlex, sys
raw = open(sys.argv[1], "rb").read().split(b"\0")
if raw and not raw[-1]: raw.pop()
print(" ".join(shlex.quote(x.decode()) for x in raw))
PY
)"
      exec 9>"$TREE_LOCK"
      flock 9
      if [ "$(git rev-parse HEAD)" != "$source_head" ] || [ "$(dirty_digest)" != "$source_digest" ]; then
        echo "source changed since receipt"
      else
        # shellcheck disable=SC2016
        flock /tmp/arle-toolchain.lock bash -c 'toolchain_dir="${ARLE_RUST_TOOLCHAIN_DIR:-/root/.rustup/toolchains/1.95.0-x86_64-unknown-linux-gnu}"; [ -x "$toolchain_dir/bin/rustc" ] && ls "$toolchain_dir"/lib/rustlib/*/lib/libstd-*.rlib >/dev/null 2>&1 || rustup toolchain install 1.95.0 --profile minimal -c rustfmt -c clippy'
        python3 - "$DIR/argv.nul" <<'PY'
import os, sys
raw = open(sys.argv[1], "rb").read().split(b"\0")
if raw and not raw[-1]: raw.pop()
os.execvp("cargo", ["cargo", "build", *(os.fsdecode(x) for x in raw)])
PY
        rc=$?
        binary="$TREE/target/release/arle"
        case " $argv_dump " in *' --bin '*) binary="$TREE/target/release/$(python3 - "$DIR/argv.nul" <<'PY'
import os, sys
args = [os.fsdecode(x) for x in open(sys.argv[1], 'rb').read().split(b'\0') if x]
print(args[args.index('--bin') + 1] if '--bin' in args else 'arle')
PY
)";; esac
      fi
    fi
    binary_sha=""; [ "$rc" -eq 0 ] && [ -f "$binary" ] && binary_sha="$(sha256 "$binary")" || rc=1
    manifest="$(find "$TREE/target" -name arle-cuda-kernels.manifest -type f -print 2>/dev/null | sort | tail -1)"
    manifest_sha=""; kernel_id=""
    if [ -n "$manifest" ]; then manifest_sha="$(sha256 "$manifest")"; kernel_id="$(awk -F= '$1=="build_id" {print $2}' "$manifest" | tail -1)"; fi
    write_receipt "$RECEIPT" "schema=arle-build-v1" "operation=$OP" "label=$LABEL" "tree=$TREE" "source_head=${source_head:-}" "source_digest=${source_digest:-}" "argv_file=$DIR/argv.nul" "binary=$binary" "binary_sha=$binary_sha" "producer_manifest=$manifest" "producer_manifest_sha=$manifest_sha" "kernel_id=$kernel_id" "exit=$rc" "pid=$$" "pgid=$(ps -o pgid= -p $$ | tr -d ' ')" "start=$(proc_start $$)" "finished_at=$(date -u +%FT%TZ)"
    printf 'BUILD_EXIT=%s\n' "$rc"
    exit "$rc"
    ;;
  *) echo "usage: pod-remote-build.sh apply-sync|build ..." >&2; exit 2;;
esac
