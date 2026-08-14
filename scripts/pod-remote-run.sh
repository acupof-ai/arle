#!/usr/bin/env bash
set -uo pipefail

TREE="${POD_TREE:-/host/arle-build}"
STATE="${POD_STATE:-/root/arle-ops}"
PROC_ROOT="${PROC_ROOT:-/proc}"
mkdir -p "$STATE/runs"
sha256() { sha256sum "$1" | cut -d' ' -f1; }
source_digest() { bash "$TREE/scripts/pod-remote-build.sh" source-digest "$TREE"; }
field() { awk -F= -v key="$2" '$1==key {sub(/^[^=]*=/, ""); print; exit}' "$1" 2>/dev/null; }
proc_start() { awk '{print $22}' "$PROC_ROOT/$1/stat" 2>/dev/null; }
write_receipt() { local dst="$1" tmp="$1.tmp.$$"; shift; printf '%s\n' "$@" > "$tmp" && mv "$tmp" "$dst"; }
update_receipt() {
  python3 - "$@" <<'PY'
import os, sys, tempfile
path, *updates = sys.argv[1:]
values = {}
order = []
for line in open(path, encoding="utf-8"):
    key, sep, value = line.rstrip("\n").partition("=")
    if sep:
        if key not in values: order.append(key)
        values[key] = value
for update in updates:
    key, sep, value = update.partition("=")
    if not sep: raise SystemExit(f"invalid receipt update: {update}")
    if key not in values: order.append(key)
    values[key] = value
fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + ".tmp.", dir=os.path.dirname(path), text=True)
with os.fdopen(fd, "w", encoding="utf-8") as stream:
    for key in order: stream.write(f"{key}={values[key]}\n")
os.replace(tmp, path)
PY
}
process_cmd_matches() {
  python3 - "$PROC_ROOT/$1/cmdline" "$2" "$3" <<'PY'
import os, sys
path, helper, operation = sys.argv[1:]
try: args = [os.fsdecode(x) for x in open(path, "rb").read().split(b"\0") if x]
except OSError: raise SystemExit(1)
raise SystemExit(0 if helper in args and operation in args else 1)
PY
}
valid_process() {
  local dir="$1" file="$1/process" kind helper operation pid pgid start expected_binary actual_pgid receipt_binary
  [ -f "$file" ] || return 1
  kind="$(field "$file" kind)"; helper="$(field "$file" expected_helper)"; operation="$(field "$file" operation)"
  pid="$(field "$file" pid)"; pgid="$(field "$file" pgid)"; start="$(field "$file" start)"
  expected_binary="$(field "$file" expected_binary)"
  case "$kind:$helper" in
    build:"$TREE/scripts/pod-remote-build.sh"|run:"$TREE/scripts/pod-remote-run.sh") ;;
    *) return 1 ;;
  esac
  [ -n "$operation" ] && [ -n "$pid" ] && [ -n "$pgid" ] && [ -n "$start" ] || return 1
  [ -r "$PROC_ROOT/$pid/stat" ] && [ "$(proc_start "$pid")" = "$start" ] || return 1
  actual_pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')"
  [ "$actual_pgid" = "$pgid" ] || return 1
  process_cmd_matches "$pid" "$helper" "$operation" || return 1
  if [ "$kind" = run ]; then
    [ -n "$expected_binary" ] || return 1
    receipt_binary="$(field "$dir/receipt" binary)"
    [ -z "$receipt_binary" ] || [ "$receipt_binary" = "$expected_binary" ] || return 1
  fi
}
find_op() {
  local label="$1" d
  for d in "$STATE/builds/$label" "$STATE/runs/$label"; do [ -d "$d" ] && printf '%s\n' "$d"; done
}
terminal_receipt() {
  local receipt="$1"
  [ -f "$receipt" ] && [ -n "$(field "$receipt" exit)" ]
}
serve_port() {
  python3 - "$1" <<'PY'
import os, sys
args = [os.fsdecode(x) for x in open(sys.argv[1], "rb").read().split(b"\0") if x]
if not args or args[0] != "serve": raise SystemExit("recorded argv is not an arle serve command")
ports = []
i = 0
while i < len(args):
    arg = args[i]
    if arg == "--port":
        if i + 1 == len(args): raise SystemExit("recorded --port has no value")
        ports.append(args[i + 1]); i += 2; continue
    if arg.startswith("--port="): ports.append(arg.split("=", 1)[1])
    i += 1
if len(ports) > 1: raise SystemExit("recorded argv has duplicate --port")
port = ports[0] if ports else "8000"
if not port.isdigit() or not 1 <= int(port) <= 65535: raise SystemExit(f"invalid recorded port: {port}")
print(port)
PY
}
# Model dir of a recorded `serve` argv (empty for any other command), so the
# cold-boot auto-pin below fires only when we are about to load weights.
serve_model_dir() {
  python3 - "$1" <<'PY'
import os, sys
args = [os.fsdecode(x) for x in open(sys.argv[1], "rb").read().split(b"\0") if x]
if "serve" not in args: raise SystemExit(0)
i = 0
while i < len(args):
    a = args[i]
    if a == "--model-path" and i + 1 < len(args): print(args[i + 1]); break
    if a.startswith("--model-path="): print(a.split("=", 1)[1]); break
    i += 1
PY
}

case "${1:-}" in
  status|log|kill)
    action="$1"; label="${2:?missing label}"; found=0; failed=0
    while IFS= read -r dir; do
      [ -n "$dir" ] || continue; found=1
      if [ "$action" = log ]; then cat "$dir/log" 2>/dev/null; continue; fi
      if valid_process "$dir"; then
        if [ "$action" = kill ]; then
          pgid="$(field "$dir/process" pgid)"
          "${KILL_CMD:-kill}" -- "-$pgid" && echo "killed $(basename "$dir") pgid=$pgid" || failed=1
        else
          echo "$(basename "$dir"): RUNNING pid=$(field "$dir/process" pid) state=$(field "$dir/receipt" state)"
        fi
      elif terminal_receipt "$dir/receipt"; then
        echo "$(basename "$dir"): DONE exit=$(field "$dir/receipt" exit)"
      else
        echo "$(basename "$dir"): REFUSED process ownership mismatch" >&2
        failed=1
      fi
      [ "$action" = status ] && tail -20 "$dir/log" 2>/dev/null || true
    done < <(find_op "$label")
    [ "$found" -eq 1 ] || { echo "no operation: $label" >&2; exit 1; }
    exit "$failed"
    ;;
  ready)
    LABEL="${2:?missing run label}"; TIMEOUT="${3:-1200}"
    case "$TIMEOUT" in ''|*[!0-9]*) echo "invalid timeout: $TIMEOUT" >&2; exit 2;; esac
    DIR="$STATE/runs/$LABEL"; RECEIPT="$DIR/receipt"
    [ -d "$DIR" ] && [ "$(field "$RECEIPT" schema)" = arle-run-v1 ] || { echo "no run: $LABEL" >&2; exit 1; }
    valid_process "$DIR" || { echo "run process ownership mismatch: $LABEL" >&2; exit 1; }
    port="$(serve_port "$DIR/argv.nul")" || exit 2
    deadline=$((SECONDS + TIMEOUT)); models_ok=0
    while [ "$SECONDS" -le "$deadline" ]; do
      valid_process "$DIR" || { echo "run process ownership changed: $LABEL" >&2; exit 1; }
      # A dead serve otherwise makes this poll wait out the full timeout.
      [ -f "$DIR/terminal" ] && { echo "run exited before ready: $LABEL $(head -n1 "$DIR/terminal")" >&2; tail -n 20 "$DIR/log" >&2; exit 1; }
      remaining=$((deadline - SECONDS + 1))
      [ "$remaining" -gt 5 ] && request_timeout=5 || request_timeout="$remaining"
      if curl --max-time "$request_timeout" -sf "http://127.0.0.1:$port/v1/models" >/dev/null; then models_ok=1; break; fi
      sleep 1
    done
    [ "$models_ok" -eq 1 ] || { echo "NOT_READY run=$LABEL port=$port" >&2; exit 1; }
    stats_tmp="$DIR/stats.json.tmp.$$"
    remaining=$((deadline - SECONDS + 1))
    [ "$remaining" -gt 0 ] || { echo "stats deadline expired: run=$LABEL" >&2; exit 1; }
    [ "$remaining" -gt 5 ] && request_timeout=5 || request_timeout="$remaining"
    curl --max-time "$request_timeout" -sf "http://127.0.0.1:$port/v1/stats" -o "$stats_tmp" || { echo "stats unavailable: run=$LABEL" >&2; rm -f "$stats_tmp"; exit 1; }
    mv "$stats_tmp" "$DIR/stats.json"
    stats_sha="$(sha256 "$DIR/stats.json")"
    expected_sha="$(field "$RECEIPT" binary_sha)"; expected_kernel="$(field "$RECEIPT" kernel_id)"
    if ! python3 - "$DIR/stats.json" "$expected_sha" "$expected_kernel" <<'PY'
import json, sys
path, expected_sha, expected_kernel = sys.argv[1:]
try: identity = json.load(open(path))["build_identity"]
except (OSError, ValueError, KeyError, TypeError): raise SystemExit(1)
actual_sha = identity.get("product_binary_sha256", "").removeprefix("sha256:")
raise SystemExit(0 if actual_sha == expected_sha and identity.get("kernel_bundle_id") == expected_kernel else 1)
PY
    then
      update_receipt "$RECEIPT" "state=identity-mismatch" "stats_file=$DIR/stats.json" "stats_sha=$stats_sha" "identity_error=runtime-build-identity-mismatch" || {
        echo "failed to persist identity mismatch: run=$LABEL" >&2
        exit 1
      }
      echo "runtime build identity mismatch: run=$LABEL" >&2
      exit 1
    fi
    valid_process "$DIR" || { echo "run process ownership changed: $LABEL" >&2; exit 1; }
    update_receipt "$RECEIPT" "state=running-verified" "stats_file=$DIR/stats.json" "stats_sha=$stats_sha" "verified_at=$(date -u +%FT%TZ)" || {
      echo "failed to persist verified identity: run=$LABEL" >&2
      exit 1
    }
    echo "ENGINE_READY run=$LABEL port=$port state=running-verified"
    ;;
  run)
    BUILD="${2:?missing build label}"; LABEL="${3:?missing run label}"; GPU="${4:?missing GPU}"; OP="${5:?missing operation}"; ARGV_FILE="${6:?missing argv file}"
    DIR="$STATE/runs/$LABEL"; BUILD_DIR="$STATE/builds/$BUILD"; BUILD_RECEIPT="$BUILD_DIR/receipt"
    [ ! -e "$DIR" ] || { echo "run label exists: $LABEL" >&2; exit 1; }
    mkdir "$DIR" || exit 1
    mv "$ARGV_FILE" "$DIR/argv.nul" || exit 1
    LOG="$DIR/log"; RECEIPT="$DIR/receipt"; MARKER="$DIR/terminal"
    binary="$(field "$BUILD_RECEIPT" binary)"
    process_pgid="$(ps -o pgid= -p $$ | tr -d ' ')"; process_start="$(proc_start $$)"
    printf '%s\n' "schema=arle-process-v1" "kind=run" "expected_helper=$TREE/scripts/pod-remote-run.sh" "operation=$OP" "pid=$$" "pgid=$process_pgid" "start=$process_start" "expected_binary=$binary" > "$DIR/process"
    exec >"$LOG" 2>&1
    rc=1; claims=(); binary_sha=""; source_head=""; source_digest_value=""; kernel_id=""; producer_id=""; embedded_id=""; selected_gpu="$GPU"
    if [ ! -f "$BUILD_RECEIPT" ] || [ "$(field "$BUILD_RECEIPT" schema)" != arle-build-v1 ] || [ "$(field "$BUILD_RECEIPT" exit)" != 0 ]; then
      echo "successful build receipt required: build:$BUILD"
    else
      binary_sha="$(field "$BUILD_RECEIPT" binary_sha)"; source_head="$(field "$BUILD_RECEIPT" source_head)"; source_digest_value="$(field "$BUILD_RECEIPT" source_digest)"
      kernel_id="$(field "$BUILD_RECEIPT" kernel_id)"; producer_id="$(field "$BUILD_RECEIPT" producer_id)"; embedded_id="$(field "$BUILD_RECEIPT" embedded_id)"
      if [ "$(sha256 "$binary" 2>/dev/null)" != "$binary_sha" ]; then echo "binary SHA mismatch"
      elif [ ! -f "$TREE/.arle-source-receipt" ] || [ "$(field "$TREE/.arle-source-receipt" head)" != "$source_head" ] || [ "$(field "$TREE/.arle-source-receipt" digest)" != "$source_digest_value" ] || [ "$(git -C "$TREE" rev-parse HEAD)" != "$source_head" ] || [ "$(source_digest)" != "$source_digest_value" ]; then echo "source changed since build"
      else
        claim_env=(ARLE_OP_ID="$OP" ARLE_OWNER="$(id -u):$(id -un)" ARLE_CLAIM_PID="$$" ARLE_CLAIM_START="$process_start")
        if [ "$GPU" = auto ]; then
          selected_gpu="$(env "${claim_env[@]}" bash "$TREE/scripts/pick-gpu.sh")" || selected_gpu=""
        elif [[ "$GPU" = *,* ]]; then
          env "${claim_env[@]}" bash "$TREE/scripts/pick-gpu.sh" reserve-set "$GPU" >/dev/null || selected_gpu=""
        else
          ARLE_GPU="$GPU" env "${claim_env[@]}" bash "$TREE/scripts/pick-gpu.sh" >/dev/null || selected_gpu=""
        fi
        if [ -z "$selected_gpu" ]; then echo "no free GPU"
        else
          IFS=',' read -r -a selected_gpus <<< "$selected_gpu"
          for gpu in "${selected_gpus[@]}"; do claims+=("${ARLE_GPU_CLAIMS:-/tmp/arle-gpu-claims}/$gpu"); done
          if ! write_receipt "$RECEIPT" "schema=arle-run-v1" "state=running-unobserved" "operation=$OP" "label=$LABEL" "build_label=$BUILD" "tree=$TREE" "source_head=$source_head" "source_digest=$source_digest_value" "binary=$binary" "binary_sha=$binary_sha" "kernel_id=$kernel_id" "producer_id=$producer_id" "embedded_id=$embedded_id" "argv_file=$DIR/argv.nul" "argv_sha=$(sha256 "$DIR/argv.nul")" "gpu=$selected_gpu" "owner=$(id -u):$(id -un)" "claim_pid=$$" "claim_start=$process_start" "pid=$$" "pgid=$process_pgid" "start=$process_start" "launched_at=$(date -u +%FT%TZ)"; then
            echo "failed to persist launch receipt"
            rc=1
          else
            # shellcheck disable=SC1091
            source "$TREE/scripts/pod-build-env.sh"
            # Cold-boot auto-pin: /host reads at ~0.2 GB/s, so a re-read of the
            # weights is 25 min. mlock them once, detached and idempotent, so the
            # serve (and every later boot) hits the warm page cache. Local dir
            # only (HF-ID model-paths resolve elsewhere); skip if already pinned.
            model_dir="$(serve_model_dir "$DIR/argv.nul" 2>/dev/null || true)"
            # Match the exact dir at end-of-arg (cmdline-end OR a trailing flag
            # like --glob) so a pinned /host/DSv4 never suppresses /host/DS and a
            # manual pin with trailing flags still counts.
            # ponytail: no lock — two concurrent same-model runs could both pass
            # this pgrep and double-fork, but that's rare and MAP_SHARED means the
            # second mlock adds no physical pages. Add flock only if it bites.
            pin_dir_re="$(printf '%s' "$model_dir" | sed 's/[][\\.^$*+?(){}|]/\\&/g')"
            if [ -n "$model_dir" ] && [ -d "$model_dir" ] && ! pgrep -f "pin_model_cache\.py $pin_dir_re( |\$)" >/dev/null; then
              setsid nohup python3 "$TREE/scripts/pin_model_cache.py" "$model_dir" \
                >"$STATE/pin-model-cache.log" 2>&1 </dev/null &
              echo "auto-pin: launched for $model_dir (log: $STATE/pin-model-cache.log)"
            fi
            if [ "${#selected_gpus[@]}" -eq 1 ]; then
              CUDA_VISIBLE_DEVICES="$selected_gpu" INFER_CUDA_DEVICE=0 python3 "$TREE/scripts/reap_run.py" "$OP" --argv-file "$DIR/argv.nul" "$binary"; rc=$?
            else
              logical_gpus="$(printf '%s,' "${!selected_gpus[@]}")"
              logical_gpus="${logical_gpus%,}"
              logical_gpus="${logical_gpus// /,}"
              CUDA_VISIBLE_DEVICES="$selected_gpu" INFER_CUDA_DEVICES="$logical_gpus" INFER_TP_SIZE="${#selected_gpus[@]}" python3 "$TREE/scripts/reap_run.py" "$OP" --argv-file "$DIR/argv.nul" "$binary"; rc=$?
            fi
          fi
        fi
      fi
    fi
    for claim in "${claims[@]}"; do [ "$(field "$claim" op)" = "$OP" ] && rm -f "$claim"; done
    if [ -f "$RECEIPT" ]; then
      update_receipt "$RECEIPT" "state=exited" "exit=$rc" "finished_at=$(date -u +%FT%TZ)"
    else
      write_receipt "$RECEIPT" "schema=arle-run-v1" "state=exited" "operation=$OP" "label=$LABEL" "build_label=$BUILD" "tree=$TREE" "source_head=$source_head" "source_digest=$source_digest_value" "binary=$binary" "binary_sha=$binary_sha" "kernel_id=$kernel_id" "producer_id=$producer_id" "embedded_id=$embedded_id" "argv_file=$DIR/argv.nul" "argv_sha=$(sha256 "$DIR/argv.nul")" "gpu=$selected_gpu" "owner=$(id -u):$(id -un)" "pid=$$" "pgid=$process_pgid" "start=$process_start" "exit=$rc" "finished_at=$(date -u +%FT%TZ)"
    fi
    write_receipt "$MARKER" "RUN_EXIT=$rc" "operation=$OP"
    printf 'RUN_EXIT=%s\n' "$rc"
    exit "$rc"
    ;;
  *) echo "usage: pod-remote-run.sh run|ready|status|log|kill ..." >&2; exit 2;;
esac
