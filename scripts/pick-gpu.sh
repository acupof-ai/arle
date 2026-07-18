#!/usr/bin/env bash
set -uo pipefail

CLAIMS="${ARLE_GPU_CLAIMS:-/tmp/arle-gpu-claims}"
OP="${ARLE_OP_ID:?ARLE_OP_ID required}"
OWNER="${ARLE_OWNER:?ARLE_OWNER required}"
CLAIM_PID="${ARLE_CLAIM_PID:-$$}"
CLAIM_START="${ARLE_CLAIM_START:-$(awk '{print $22}' /proc/$$/stat)}"
REQUESTED="${ARLE_GPU:-}"
mkdir -p "$CLAIMS"
exec 9>"$CLAIMS/.lock"
flock 9
now="$(date +%s)"

field() { awk -F= -v key="$2" '$1==key {sub(/^[^=]*=/, ""); print; exit}' "$1" 2>/dev/null; }
stale() {
  local c="$1" pid start
  pid="$(field "$c" pid)"; start="$(field "$c" start)"
  [ -n "$pid" ] && [ -r "/proc/$pid/stat" ] && [ "$(awk '{print $22}' "/proc/$pid/stat")" = "$start" ] && return 1
  return 0
}
claim() {
  local gpu="$1"
  local c="$CLAIMS/$gpu"
  local tmp="$c.tmp.$$"
  if [ -f "$c" ]; then stale "$c" || return 1; rm -f "$c"; fi
  printf 'schema=arle-gpu-claim-v1\nop=%s\nowner=%s\npid=%s\nstart=%s\ncreated=%s\n' "$OP" "$OWNER" "$CLAIM_PID" "$CLAIM_START" "$now" > "$tmp" && mv "$tmp" "$c"
}

while IFS=',' read -r idx used _; do
  idx="$(printf '%s' "$idx" | tr -dc '0-9')"; used="$(printf '%s' "$used" | tr -dc '0-9')"
  [ -n "$idx" ] || continue
  [ -z "$REQUESTED" ] || [ "$idx" = "$REQUESTED" ] || continue
  [ "${used:-0}" -le 2000 ] || continue
  claim "$idx" || continue
  echo "$idx"
  exit 0
done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader)
echo NONE
exit 1
