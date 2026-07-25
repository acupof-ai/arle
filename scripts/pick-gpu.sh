#!/usr/bin/env bash
set -uo pipefail

CLAIMS="${ARLE_GPU_CLAIMS:-/tmp/arle-gpu-claims}"
PROC_ROOT="${PROC_ROOT:-/proc}"
MEMORY_LIMIT_MIB="${ARLE_GPU_MEMORY_LIMIT_MIB:-2000}"
mkdir -p "$CLAIMS"

field() { awk -F= -v key="$2" '$1==key {sub(/^[^=]*=/, ""); print; exit}' "$1" 2>/dev/null; }
# A claim is live only if its pid still exists, matches the recorded start time,
# AND is not a zombie. Killed runs reparent to pid 1, which in this container is
# not an init that reaps, so a defunct holder keeps a /proc entry forever and
# would pin its GPU for good.
stale() {
  local c="$1" pid start stat
  pid="$(field "$c" pid)"; start="$(field "$c" start)"
  [ -n "$pid" ] && [ -r "$PROC_ROOT/$pid/stat" ] || return 0
  stat="$(cat "$PROC_ROOT/$pid/stat" 2>/dev/null)" || return 0
  [ "$(awk '{print $22}' <<< "$stat")" = "$start" ] || return 0
  [ "$(awk '{print $3}' <<< "$stat")" = "Z" ] && return 0
  return 1
}
load_gpus() {
  nvidia-smi --query-gpu=index,uuid,memory.used,compute_cap --format=csv,noheader,nounits
}
load_compute_uuids() {
  nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits 2>/dev/null || return 1
}
claim_available() {
  local gpu="$1" c
  c="$CLAIMS/$gpu"
  [ ! -f "$c" ] && return 0
  stale "$c" || return 1
  rm -f "$c"
}
physical_free() {
  local used="$1" uuid="$2" compute="$3"
  [ "$used" -le "$MEMORY_LIMIT_MIB" ] && ! grep -Fqx "$uuid" <<< "$compute"
}

case "${1:-}" in
  check-free-set|reserve-set)
    action="$1"; requested="${2:?usage: pick-gpu.sh $1 <csv>}"
    exec 9>"$CLAIMS/.lock"; flock 9
    compute="$(load_compute_uuids)" || { echo "cannot query compute applications" >&2; exit 1; }
    rows="$(load_gpus)" || exit 1
    python3 - "$requested" "$rows" "$compute" "$CLAIMS" "$MEMORY_LIMIT_MIB" "$PROC_ROOT" <<'PY'
import os, sys
requested, rows, compute, claims, limit, proc = sys.argv[1:]
indices = requested.split(",") if requested else []
if not indices or len(indices) != len(set(indices)) or any(not x.isdecimal() for x in indices): raise SystemExit("GPU set must contain unique decimal indices")
gpus = {}
for row in rows.splitlines():
    parts = [x.strip() for x in row.split(",")]
    if len(parts) != 4: continue
    idx, uuid, used, capability = parts
    try: gpus[idx] = (uuid, int(used), capability)
    except ValueError: continue
busy = set(compute.splitlines())
for idx in indices:
    if idx not in gpus: raise SystemExit(f"GPU {idx} is absent")
    uuid, used, capability = gpus[idx]
    if capability != "9.0": raise SystemExit(f"GPU {idx} is not SM90")
    claim = os.path.join(claims, idx)
    if os.path.isfile(claim):
        values = dict(line.rstrip("\n").split("=", 1) for line in open(claim) if "=" in line)
        pid, start = values.get("pid", ""), values.get("start", "")
        try:
            fields = open(os.path.join(proc, pid, "stat")).read().split()
            actual, state = fields[21], fields[2]
        except OSError: actual, state = "", ""
        # state "Z": the holder was killed and never reaped (pid 1 here is not an
        # init), so its /proc entry outlives it — that is a dead claim, not a live one.
        if pid and actual == start and state != "Z": raise SystemExit(f"GPU {idx} is claimed")
        os.unlink(claim)
    if used > int(limit): raise SystemExit(f"GPU {idx} memory is occupied")
    if uuid in busy: raise SystemExit(f"GPU {idx} has a compute application")
print(requested)
PY
    status=$?
    [ "$status" -eq 0 ] || exit "$status"
    if [ "$action" = reserve-set ]; then
      OP="${ARLE_OP_ID:?ARLE_OP_ID required}"; OWNER="${ARLE_OWNER:?ARLE_OWNER required}"
      CLAIM_PID="${ARLE_CLAIM_PID:-$$}"; CLAIM_START="${ARLE_CLAIM_START:-$(awk '{print $22}' "$PROC_ROOT/$CLAIM_PID/stat" 2>/dev/null)}"
      now="$(date +%s)"
      IFS=',' read -r -a indices <<< "$requested"
      reserved=()
      for idx in "${indices[@]}"; do
        c="$CLAIMS/$idx"; tmp="$c.tmp.$$"
        if ! printf 'schema=arle-gpu-claim-v1\nop=%s\nowner=%s\npid=%s\nstart=%s\ncreated=%s\n' "$OP" "$OWNER" "$CLAIM_PID" "$CLAIM_START" "$now" > "$tmp" || ! mv "$tmp" "$c"; then
          rm -f "$tmp"
          for gpu in "${reserved[@]}"; do rm -f "$CLAIMS/$gpu"; done
          exit 1
        fi
        reserved+=("$idx")
      done
    fi
    ;;
  "")
    OP="${ARLE_OP_ID:?ARLE_OP_ID required}"; OWNER="${ARLE_OWNER:?ARLE_OWNER required}"
    CLAIM_PID="${ARLE_CLAIM_PID:-$$}"; CLAIM_START="${ARLE_CLAIM_START-$(awk '{print $22}' "$PROC_ROOT/$$/stat" 2>/dev/null)}"; REQUESTED="${ARLE_GPU:-}"
    exec 9>"$CLAIMS/.lock"; flock 9
    compute="$(load_compute_uuids)" || { echo NONE; exit 1; }
    now="$(date +%s)"
    while IFS=',' read -r idx uuid used _; do
      idx="${idx//[[:space:]]/}"; uuid="${uuid//[[:space:]]/}"; used="${used//[!0-9]/}"
      [ -n "$idx" ] || continue
      [ -z "$REQUESTED" ] || [ "$idx" = "$REQUESTED" ] || continue
      physical_free "${used:-0}" "$uuid" "$compute" || continue
      claim_available "$idx" || continue
      c="$CLAIMS/$idx"; tmp="$c.tmp.$$"
      printf 'schema=arle-gpu-claim-v1\nop=%s\nowner=%s\npid=%s\nstart=%s\ncreated=%s\n' "$OP" "$OWNER" "$CLAIM_PID" "$CLAIM_START" "$now" > "$tmp" && mv "$tmp" "$c"
      echo "$idx"; exit 0
    done < <(load_gpus)
    echo NONE; exit 1
    ;;
  *) echo "usage: pick-gpu.sh [check-free-set|reserve-set <csv>]" >&2; exit 2;;
esac
