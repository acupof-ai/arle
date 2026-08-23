#!/usr/bin/env bash
set -uo pipefail

TREE="${POD_TREE:-/host/arle-build}"
STATE="${POD_STATE:-/root/arle-ops}"
TREE_LOCK="/tmp/arle-build$(printf '%s' "$TREE" | tr '/.' '__').lock"

proc_start() { awk '{print $22}' "/proc/$1/stat" 2>/dev/null; }
sha256() { sha256sum "$1" | cut -d' ' -f1; }
source_digest() {
  python3 - "${1:-$TREE}" <<'PY'
import hashlib, os, stat, subprocess, sys

root = os.fsencode(os.path.abspath(sys.argv[1]))
def git(*args):
    return subprocess.check_output([b"git", b"-C", root, *args])

paths = set(git(b"ls-tree", b"-rz", b"--name-only", b"HEAD").split(b"\0"))
paths.update(git(b"ls-files", b"-co", b"--exclude-standard", b"-z").split(b"\0"))
paths.discard(b"")
paths.discard(b".arle-source-receipt")
digest = hashlib.sha256()
for path in sorted(paths):
    full = os.path.join(root, path)
    digest.update(len(path).to_bytes(8, "big")); digest.update(path)
    try:
        mode = os.lstat(full).st_mode
    except FileNotFoundError:
        digest.update(b"D")
        continue
    if stat.S_ISLNK(mode):
        data = os.fsencode(os.readlink(full)); kind = b"L"
    elif stat.S_ISREG(mode):
        with open(full, "rb") as f: data = f.read()
        kind = b"X" if mode & 0o111 else b"F"
    else:
        continue
    digest.update(kind); digest.update(len(data).to_bytes(8, "big")); digest.update(data)
print(digest.hexdigest())
PY
}
validate_build_args() {
  python3 - "$1" <<'PY'
import os, sys
args = [os.fsdecode(x) for x in open(sys.argv[1], "rb").read().split(b"\0") if x]
release = False
binary = None
i = 0
while i < len(args):
    arg = args[i]
    if arg == "--release": release = True; i += 1
    elif arg == "--no-default-features": i += 1
    elif arg in ("--features", "--bin"):
        if i + 1 == len(args) or args[i + 1].startswith("-"):
            raise SystemExit(f"missing value for {arg}")
        if arg == "--bin":
            if binary is not None: raise SystemExit("--bin must occur exactly once")
            binary = args[i + 1]
        i += 2
    elif arg.startswith("--features="):
        if not arg.removeprefix("--features="): raise SystemExit("missing value for --features")
        i += 1
    elif arg.startswith(("--profile", "--target-dir", "--target", "--message-format")):
        raise SystemExit(f"unsupported cargo build argument: {arg}")
    else:
        raise SystemExit(f"unsupported cargo build argument: {arg}")
if not release: raise SystemExit("cargo build arguments require --release")
if binary is None: raise SystemExit("cargo build arguments require --bin")
if "/" in binary or binary in ("", ".", ".."): raise SystemExit("invalid --bin value")
print(binary)
PY
}
write_receipt() {
  local dst="$1" tmp="$1.tmp.$$"
  shift
  printf '%s\n' "$@" > "$tmp" && mv "$tmp" "$dst"
}
argv_sha256() { sha256 "$1"; }
parse_cargo_events() {
  python3 - "$1" "$2" "$TREE/crates/cuda-kernels" <<'PY'
import json, pathlib, sys

path, binary, kernel_dir = sys.argv[1:]
kernel_package = "path+" + pathlib.Path(kernel_dir).resolve().as_uri() + "#"
executables = []
out_dirs = []
with open(path) as stream:
    for line in stream:
        try: event = json.loads(line)
        except json.JSONDecodeError: continue
        reason = event.get("reason")
        target = event.get("target", {})
        if reason == "compiler-artifact" and target.get("name") == binary and "bin" in target.get("kind", []):
            executable = event.get("executable")
            if executable: executables.append(executable)
        elif reason == "build-script-executed" and event.get("package_id", "").startswith(kernel_package):
            out_dir = event.get("out_dir")
            if out_dir: out_dirs.append(out_dir)
if len(executables) != 1: raise SystemExit(f"expected exactly one executable event for {binary}, found {len(executables)}")
if len(out_dirs) != 1: raise SystemExit(f"expected exactly one cuda-kernels build-script event, found {len(out_dirs)}")
print(executables[0]); print(out_dirs[0])
PY
}
# Delete files present in the remote tree but absent from the incoming tarball.
# Only untracked-unignored paths are considered: tracked deletions already
# arrive via `deletes`, and ignored paths (target/, .venv/, bench-output/) are
# build state the sync must not touch.
reconcile_untracked() {
  local tree="$1" archive="$2" tmp
  tmp="$(mktemp -d)" || return 0
  trap 'rm -rf "$tmp"' RETURN
  tar -tzf "$archive" 2>/dev/null | sed 's:/$::' | sort -u > "$tmp/incoming" || return 0
  # `.arle-source-receipt` is untracked and never in the tarball, and
  # `source_digest` discards it for the same reason — deleting it here made
  # every sync strip the receipt the next build requires.
  git -C "$tree" ls-files -co --exclude-standard -z 2>/dev/null |
    tr '\0' '\n' | grep -v '^\.arle-source-receipt$' | sort -u > "$tmp/present" || return 0
  comm -13 "$tmp/incoming" "$tmp/present" > "$tmp/strays"
  [ -s "$tmp/strays" ] || return 0
  echo "sync: removing $(wc -l < "$tmp/strays" | tr -d ' ') untracked file(s) absent from the pusher's tree" >&2
  while IFS= read -r path; do
    [ -n "$path" ] && rm -f "$tree/$path"
  done < "$tmp/strays"
}

restore_persistent() {
  local from="$1" to="$2" path
  for path in target crates/cuda-kernels/tools/tilelang/.venv bench-output; do
    [ -e "$from/$path" ] || continue
    mkdir -p "$(dirname "$to/$path")" || return 1
    mv "$from/$path" "$to/$path" || return 1
  done
}

case "${1:-}" in
  source-digest)
    source_digest "${2:-$TREE}"
    ;;
  validate-build-args)
    validate_build_args "${2:?missing argv file}"
    ;;
  apply-sync)
    stage="${2:?missing sync stage}"
    exec 9>"$TREE_LOCK"
    flock 9
    meta="$stage.source.meta"
    archive="$stage.tree.tgz"
    deletes="$stage.deletes"
    bundle="$stage.source.bundle"
    [ -f "$meta" ] && [ -f "$archive" ] && [ -f "$deletes" ] || { echo "incomplete sync stage" >&2; exit 1; }
    archive_sha="$(awk -F= '$1=="archive_sha" {print $2}' "$meta")"
    bundle_sha="$(awk -F= '$1=="bundle_sha" {print $2}' "$meta")"
    bundle_mode="$(awk -F= '$1=="bundle_mode" {print $2}' "$meta")"
    bundle_mode="${bundle_mode:-full}"
    head="$(awk -F= '$1=="head" {print $2}' "$meta")"
    [ "$(sha256 "$archive")" = "$archive_sha" ] || { echo "sync digest mismatch" >&2; exit 1; }
    backup=""
    if [ "$bundle_mode" = none ]; then
      [ "$bundle_sha" = none ] || { echo "sync digest mismatch" >&2; exit 1; }
      overlay="$TREE"
    elif [ "$bundle_mode" = incremental ]; then
      [ -f "$bundle" ] && [ "$(sha256 "$bundle")" = "$bundle_sha" ] || { echo "sync digest mismatch" >&2; exit 1; }
      git -C "$TREE" bundle unbundle "$bundle" >/dev/null || { echo "incremental unbundle failed" >&2; exit 1; }
      git -C "$TREE" reset --hard "$head" >/dev/null || { echo "incremental reset failed" >&2; exit 1; }
      overlay="$TREE"
    else
      [ -f "$bundle" ] && [ "$(sha256 "$bundle")" = "$bundle_sha" ] || { echo "sync digest mismatch" >&2; exit 1; }
      parent="$(dirname "$TREE")"
      incoming="$parent/.arle-incoming.$$"
      backup="$parent/.arle-backup.$$"
      git clone -q "$bundle" "$incoming"
      overlay="$incoming"
    fi
    tar -C "$overlay" -xzf "$archive"
    while IFS= read -r -d '' path; do rm -f "$overlay/$path"; done < "$deletes"
    # Reconcile untracked files too, not just tracked deletions. `source_digest`
    # counts every untracked-unignored file, so one left behind here (a debug
    # probe, a crash dump, an editor backup) puts the remote permanently out of
    # sync and fails EVERY later sync until a human removes it by hand. The
    # pusher's tarball is the authority on what the tree contains.
    reconcile_untracked "$overlay" "$archive"
    if [ "$bundle_mode" = full ]; then
      if ! mv "$TREE" "$backup" || ! mv "$incoming" "$TREE"; then
        [ -d "$backup" ] && mv "$backup" "$TREE"
        exit 1
      fi
      if ! restore_persistent "$backup" "$TREE"; then
        restore_persistent "$TREE" "$backup" || true
        rm -rf "$TREE"; mv "$backup" "$TREE"; exit 1
      fi
    fi
    actual_head="$(git -C "$TREE" rev-parse HEAD)"
    expected_digest="$(awk -F= '$1=="dirty_digest" {print $2}' "$meta")"
    actual_digest="$(source_digest)"
    if [ "$actual_head" != "$head" ] || [ "$actual_digest" != "$expected_digest" ]; then
      # Name the offenders before rolling back. A digest mismatch is almost
      # always a stray untracked file left in the remote tree (a probe, a dump,
      # an editor backup): it is counted by `git ls-files -co` here but absent
      # from the pusher's tree, and two bare hashes say nothing about which.
      if [ "$actual_digest" != "$expected_digest" ] && [ -n "${strays_tmp:=$(mktemp -d)}" ]; then
        tar -tzf "$archive" 2>/dev/null | sed 's:/$::' | sort -u > "$strays_tmp/incoming" || true
        git -C "$TREE" ls-files -co --exclude-standard -z 2>/dev/null |
          tr '\0' '\n' | grep -v '^\.arle-source-receipt$' | sort -u > "$strays_tmp/present" || true
        comm -13 "$strays_tmp/incoming" "$strays_tmp/present" | head -20 > "$strays_tmp/strays"
        [ -s "$strays_tmp/strays" ] && {
          echo "sync source mismatch: remote tree still carries files the pusher does not:" >&2
          sed 's/^/  /' "$strays_tmp/strays" >&2
        }
        rm -rf "$strays_tmp"
      fi
      [ -z "$backup" ] || { restore_persistent "$TREE" "$backup" || true; rm -rf "$TREE"; mv "$backup" "$TREE"; }
      echo "sync source mismatch: head=$actual_head expected_head=$head digest=$actual_digest expected_digest=$expected_digest" >&2; exit 1
    fi
    receipt="$TREE/.arle-source-receipt"
    write_receipt "$receipt" "schema=arle-source-v1" "head=$actual_head" "digest=$actual_digest" "archive_sha=$archive_sha" "bundle_sha=$bundle_sha" "applied_at=$(date -u +%FT%TZ)"
    if [ "$bundle_mode" = full ]; then
      rm -rf "$backup" "$archive" "$deletes" "$bundle" "$meta"
    else
      rm -f "$archive" "$deletes" "$meta"
      [ "$bundle_mode" = none ] || rm -f "$bundle"
    fi
    echo "synced source head=$actual_head digest=$actual_digest bundle_mode=$bundle_mode"
    ;;
  build)
    LABEL="${2:?missing label}"; OP="${3:?missing operation}"; ARGV_FILE="${4:?missing argv file}"
    binary_name="$(validate_build_args "$ARGV_FILE")" || exit 2
    # Only `build` writes under STATE. Creating it at file scope also fired for
    # `source-digest`, which pod.sh runs LOCALLY, so every sync from a Mac
    # printed two "mkdir: /host: Read-only file system" lines.
    mkdir -p "$STATE/builds" || exit 1
    DIR="$STATE/builds/$LABEL"
    [ ! -e "$DIR" ] || { echo "build label exists: $LABEL" >&2; exit 1; }
    mkdir "$DIR" || exit 1
    mv "$ARGV_FILE" "$DIR/argv.nul" || exit 1
    LOG="$DIR/log"; RECEIPT="$DIR/receipt"
    echo $$ > "$DIR/pid"
    printf '%s\n' "schema=arle-process-v1" "kind=build" "expected_helper=$TREE/scripts/pod-remote-build.sh" "operation=$OP" "pid=$$" "pgid=$(ps -o pgid= -p $$ | tr -d ' ')" "start=$(proc_start $$)" "expected_binary=" > "$DIR/process"
    exec >"$LOG" 2>&1
    # shellcheck disable=SC1091
    source "$TREE/scripts/pod-build-env.sh"
    # shellcheck disable=SC1091
    source "$TREE/scripts/cuda_prebuilt_manifest.sh"
    cd "$TREE" || exit 1
    source_receipt="$TREE/.arle-source-receipt"
    rc=1; binary=""
    if [ ! -f "$source_receipt" ]; then
      echo "source receipt required"
    else
      source_head="$(awk -F= '$1=="head" {print $2}' "$source_receipt")"
      source_digest_value="$(awk -F= '$1=="digest" {print $2}' "$source_receipt")"
      exec 9>"$TREE_LOCK"
      flock 9
      if [ "$(git rev-parse HEAD)" != "$source_head" ] || [ "$(source_digest)" != "$source_digest_value" ]; then
        echo "source changed since receipt"
      else
        # shellcheck disable=SC2016
        flock /tmp/arle-toolchain.lock bash -c 'toolchain_dir="${ARLE_RUST_TOOLCHAIN_DIR:-/root/.rustup/toolchains/1.98.0-x86_64-unknown-linux-gnu}"; [ -x "$toolchain_dir/bin/rustc" ] && ls "$toolchain_dir"/lib/rustlib/*/lib/libstd-*.rlib >/dev/null 2>&1 || rustup toolchain install 1.98.0 --profile minimal -c rustfmt -c clippy'
        events="$DIR/cargo.jsonl"
        export ARLE_CARGO_WORKSPACE_ROOT="$TREE"
        python3 - "$DIR/argv.nul" "$events" <<'PY'
import json, os, subprocess, sys
raw = open(sys.argv[1], "rb").read().split(b"\0")
if raw and not raw[-1]: raw.pop()
command = ["cargo", "build", "--message-format=json-render-diagnostics", *(os.fsdecode(x) for x in raw)]
with open(sys.argv[2], "w") as events:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
    for line in process.stdout:
        events.write(line)
        try: rendered = json.loads(line).get("message", {}).get("rendered")
        except json.JSONDecodeError: rendered = None
        if rendered: print(rendered, end="", file=sys.stderr)
raise SystemExit(process.wait())
PY
        rc=$?
        if [ "$rc" -eq 0 ]; then
          cargo_outputs="$(parse_cargo_events "$events" "$binary_name")" || rc=1
          if [ "$rc" -eq 0 ]; then
            binary="${cargo_outputs%%$'\n'*}"
            cargo_out_dir="${cargo_outputs#*$'\n'}"
          fi
        fi
      fi
    fi
    binary_sha=""; manifest=""; manifest_sha=""; producer_id=""; embedded_id=""
    [ "$rc" -eq 0 ] && [ -f "$binary" ] && binary_sha="$(sha256 "$binary")" || rc=1
    if [ "$rc" -eq 0 ]; then
      # Feature sets share target/release/<bin>; pin the artifact in the
      # build's own state dir, out of cargo clean/sweep's reach.
      artifact="$DIR/$binary_name"
      if { ln -f "$binary" "$artifact" 2>/dev/null || cp -f "$binary" "$artifact"; }; then
        binary="$artifact"
      else
        echo "failed to pin per-label binary: $artifact"; rc=1
      fi
    fi
    if [ -n "${ARLE_CUDA_KERNELS_PREBUILT_DIR:-}" ]; then
      manifest="$ARLE_CUDA_KERNELS_PREBUILT_DIR/arle-cuda-kernels.manifest"
    else
      manifest="${cargo_out_dir:-}/arle-cuda-kernels.manifest"
    fi
    if [ "$rc" -eq 0 ] && [ -f "$manifest" ]; then
      manifest_sha="$(sha256 "$manifest")"
      cuda_prebuilt_manifest_validate "$manifest" && producer_id="$(cuda_prebuilt_manifest_value "$manifest" kernel_build_id)" || rc=1
    else
      rc=1
    fi
    if [ "$rc" -eq 0 ]; then
      # First line only: `--kernel-build-id` also prints a `capabilities:` line
      # (06a27527e), and comparing the whole output failed every green build.
      embedded_id="$($binary --kernel-build-id | head -n1)" || rc=1
      [ "$embedded_id" != unreported ] && [ "$embedded_id" = "$producer_id" ] || rc=1
    fi
    write_receipt "$RECEIPT" "schema=arle-build-v1" "operation=$OP" "label=$LABEL" "tree=$TREE" "source_head=${source_head:-}" "source_digest=${source_digest_value:-}" "argv_file=$DIR/argv.nul" "argv_sha=$(argv_sha256 "$DIR/argv.nul")" "binary=$binary" "binary_sha=$binary_sha" "cargo_out_dir=${cargo_out_dir:-}" "producer_manifest=$manifest" "producer_manifest_sha=$manifest_sha" "producer_id=$producer_id" "embedded_id=$embedded_id" "kernel_id=$producer_id" "exit=$rc" "pid=$$" "pgid=$(ps -o pgid= -p $$ | tr -d ' ')" "start=$(proc_start $$)" "finished_at=$(date -u +%FT%TZ)"
    printf 'BUILD_EXIT=%s\n' "$rc"
    exit "$rc"
    ;;
  *) echo "usage: pod-remote-build.sh source-digest|validate-build-args|apply-sync|build ..." >&2; exit 2;;
esac
