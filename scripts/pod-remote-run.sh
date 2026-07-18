#!/usr/bin/env bash
set -uo pipefail

TREE="${POD_TREE:-/host/arle-build}"
STATE="${POD_STATE:-/root/arle-ops}"
PROC_ROOT="${PROC_ROOT:-/proc}"
mkdir -p "$STATE/runs"
sha256() { sha256sum "$1" | cut -d' ' -f1; }
source_digest() { bash "$TREE/scripts/pod-remote-build.sh" source-digest "$TREE"; }
field() { awk -F= -v key="$2" '$1==key {sub(/^[^=]*=/, ""); print; exit}' "$1"; }
proc_start() { awk '{print $22}' "$PROC_ROOT/$1/stat" 2>/dev/null; }
write_receipt() { local dst="$1" tmp="$1.tmp.$$"; shift; printf '%s\n' "$@" > "$tmp" && mv "$tmp" "$dst"; }
valid_process() {
  local file="$1" pid pgid start op cmd actual_pgid
  [ -f "$file" ] || return 1
  pid="$(field "$file" pid)"; pgid="$(field "$file" pgid)"; start="$(field "$file" start)"; op="$(field "$file" op)"
  [ -n "$pid" ] && [ -r "$PROC_ROOT/$pid/stat" ] || return 1
  [ "$(proc_start "$pid")" = "$start" ] || return 1
  actual_pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')"
  [ "$actual_pgid" = "$pgid" ] || return 1
  cmd="$(tr '\0' ' ' < "$PROC_ROOT/$pid/cmdline" 2>/dev/null)"
  case "$cmd" in *pod-remote-build.sh*"$op"*|*pod-remote-run.sh*"$op"*) return 0;; esac
  return 1
}
find_op() {
  local label="$1" d
  for d in "$STATE/builds/$label" "$STATE/runs/$label"; do [ -d "$d" ] && printf '%s\n' "$d"; done
}

case "${1:-}" in
  status|log|kill)
    action="$1"; label="${2:?missing label}"; found=0; failed=0
    while IFS= read -r dir; do
      [ -n "$dir" ] || continue; found=1
      if [ "$action" = log ]; then cat "$dir/log" 2>/dev/null; continue; fi
      if valid_process "$dir/process"; then
        if [ "$action" = kill ]; then pgid="$(field "$dir/process" pgid)"; "${KILL_CMD:-kill}" -- "-$pgid" && echo "killed $(basename "$dir") pgid=$pgid"
        else echo "$(basename "$dir"): RUNNING pid=$(field "$dir/process" pid)"; fi
      elif [ "$action" = kill ] && [ -f "$dir/receipt" ]; then
        echo "$(basename "$dir"): DONE exit=$(field "$dir/receipt" exit)"
      elif [ "$action" = kill ]; then
        echo "refuse stale or mismatched process: $(basename "$dir")" >&2
        failed=1
      else
        exit_code="$(field "$dir/receipt" exit 2>/dev/null)"; echo "$(basename "$dir"): DONE exit=${exit_code:-unknown}"
      fi
      [ "$action" = status ] && tail -20 "$dir/log" 2>/dev/null || true
    done < <(find_op "$label")
    [ "$found" -eq 1 ] || { echo "no operation: $label" >&2; exit 1; }
    exit "$failed"
    ;;
  run)
    BUILD="${2:?missing build label}"; LABEL="${3:?missing run label}"; GPU="${4:?missing GPU}"; OP="${5:?missing operation}"; ARGV_FILE="${6:?missing argv file}"
    DIR="$STATE/runs/$LABEL"; BUILD_DIR="$STATE/builds/$BUILD"; BUILD_RECEIPT="$BUILD_DIR/receipt"
    [ ! -e "$DIR" ] || { echo "run label exists: $LABEL" >&2; exit 1; }
    mkdir "$DIR" || exit 1
    mv "$ARGV_FILE" "$DIR/argv.nul" || exit 1
    LOG="$DIR/log"; RECEIPT="$DIR/receipt"; MARKER="$DIR/terminal"
    echo $$ > "$DIR/pid"
    printf '%s\n' "op=$OP" "pid=$$" "pgid=$(ps -o pgid= -p $$ | tr -d ' ')" "start=$(proc_start $$)" > "$DIR/process"
    exec >"$LOG" 2>&1
    rc=1; claim=""; binary=""; binary_sha=""; source_head=""; source_digest=""
    if [ ! -f "$BUILD_RECEIPT" ] || [ "$(field "$BUILD_RECEIPT" schema)" != arle-build-v1 ] || [ "$(field "$BUILD_RECEIPT" exit)" != 0 ]; then
      echo "successful build receipt required: build:$BUILD"
    else
      binary="$(field "$BUILD_RECEIPT" binary)"; binary_sha="$(field "$BUILD_RECEIPT" binary_sha)"
      source_head="$(field "$BUILD_RECEIPT" source_head)"; source_digest="$(field "$BUILD_RECEIPT" source_digest)"
      if [ "$(sha256 "$binary" 2>/dev/null)" != "$binary_sha" ]; then echo "binary SHA mismatch"
      elif [ ! -f "$TREE/.arle-source-receipt" ] || [ "$(field "$TREE/.arle-source-receipt" head)" != "$source_head" ] || [ "$(field "$TREE/.arle-source-receipt" digest)" != "$source_digest" ] || [ "$(git -C "$TREE" rev-parse HEAD)" != "$source_head" ] || [ "$(source_digest)" != "$source_digest" ]; then echo "source changed since build"
      else
        claim_env=(ARLE_OP_ID="$OP" ARLE_OWNER="$(id -u):$(id -un)" ARLE_CLAIM_PID="$$" ARLE_CLAIM_START="$(proc_start $$)")
        if [ "$GPU" = auto ]; then GPU="$(env "${claim_env[@]}" bash "$TREE/scripts/pick-gpu.sh")" || GPU=""; else ARLE_GPU="$GPU" env "${claim_env[@]}" bash "$TREE/scripts/pick-gpu.sh" >/dev/null || GPU=""; fi
        if [ -z "$GPU" ]; then echo "no free GPU"
        else
          claim="/tmp/arle-gpu-claims/$GPU"
          # shellcheck disable=SC1091
          source "$TREE/scripts/pod-build-env.sh"
          CUDA_VISIBLE_DEVICES="$GPU" INFER_CUDA_DEVICE=0 python3 "$TREE/scripts/reap_run.py" "$OP" --argv-file "$DIR/argv.nul" "$binary"; rc=$?
        fi
      fi
    fi
    [ -n "$claim" ] && [ "$(field "$claim" op 2>/dev/null)" = "$OP" ] && rm -f "$claim"
    write_receipt "$RECEIPT" "schema=arle-run-v1" "operation=$OP" "label=$LABEL" "build_label=$BUILD" "tree=$TREE" "source_head=$source_head" "source_digest=$source_digest" "binary=$binary" "binary_sha=$binary_sha" "argv_file=$DIR/argv.nul" "gpu=$GPU" "exit=$rc" "pid=$$" "pgid=$(ps -o pgid= -p $$ | tr -d ' ')" "start=$(proc_start $$)" "finished_at=$(date -u +%FT%TZ)"
    write_receipt "$MARKER" "RUN_EXIT=$rc" "operation=$OP"
    printf 'RUN_EXIT=%s\n' "$rc"
    exit "$rc"
    ;;
  *) echo "usage: pod-remote-run.sh run|status|log|kill ..." >&2; exit 2;;
esac
