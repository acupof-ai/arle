#!/usr/bin/env bash
# burst_curl.sh — pure curl escalation sweep against a running ARLE server
# No CUDA, no arle fork — safe from ELKEID kill
# Run detached: setsid bash burst_curl.sh </dev/null >/root/run-burst-curl.log 2>&1 &
set -uo pipefail

PORT="${BURST_PORT:-8088}"
GPU="${BURST_GPU:-3}"
BASE_URL="http://127.0.0.1:$PORT"

log() { echo "[burst $(date -u +%H:%M:%S)] $*"; }

# Verify server up
if ! curl -sf "$BASE_URL/v1/models" >/dev/null 2>&1; then
  log "ERROR: server not responding at $BASE_URL"
  exit 1
fi
log "server confirmed on $BASE_URL"

log "GPU $GPU memory baseline:"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader -i "$GPU"

# Stats baseline
log "stats baseline:"
curl -sf "$BASE_URL/v1/stats" 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    for k,v in sorted(d.items()):
        print(f'  {k}={v}')
except Exception as e:
    print('  parse error:', e)
" 2>/dev/null || log "  (no /v1/stats)"

run_burst() {
  local C=$1
  log "=== BURST c=$C starting ==="

  local TMPDIR_R
  TMPDIR_R=$(mktemp -d)
  local PIDS=()

  local START
  START=$(date +%s%3N)

  for i in $(seq 1 "$C"); do
    curl -s \
      -o "$TMPDIR_R/body_$i" \
      -w "%{http_code}" \
      -X POST "$BASE_URL/v1/completions" \
      -H "Content-Type: application/json" \
      -d '{"model":"default","prompt":"Count from 1 to 10 and explain each number.","max_tokens":256,"stream":false}' \
      --max-time 300 \
      > "$TMPDIR_R/code_$i" 2>/dev/null &
    PIDS+=($!)
  done

  log "  all $C requests fired, waiting..."
  for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done

  local END
  END=$(date +%s%3N)
  local ELAPSED=$(( END - START ))

  local OK=0 FAIL=0
  declare -A FAIL_CODES=()
  for i in $(seq 1 "$C"); do
    local CODE
    CODE=$(cat "$TMPDIR_R/code_$i" 2>/dev/null || echo "000")
    if [ "$CODE" = "200" ]; then
      OK=$(( OK + 1 ))
    else
      FAIL=$(( FAIL + 1 ))
      FAIL_CODES["$CODE"]=$(( ${FAIL_CODES["$CODE"]:-0} + 1 ))
      # Print first 3 failure bodies
      if [ $FAIL -le 3 ]; then
        local BODY
        BODY=$(cat "$TMPDIR_R/body_$i" 2>/dev/null | head -c 400 || echo "(empty)")
        log "  FAIL[$i] code=$CODE body=$BODY"
      fi
    fi
  done

  # Count actual tokens from successful 200 responses
  local TOTAL_TOKENS=0
  for i in $(seq 1 "$C"); do
    local CODE
    CODE=$(cat "$TMPDIR_R/code_$i" 2>/dev/null || echo "000")
    if [ "$CODE" = "200" ]; then
      local TOKS
      TOKS=$(python3 -c "
import json
try:
    d = json.load(open('$TMPDIR_R/body_$i'))
    print(d.get('usage', {}).get('completion_tokens', 0))
except:
    print(0)
" 2>/dev/null || echo "0")
      TOTAL_TOKENS=$(( TOTAL_TOKENS + TOKS ))
    fi
  done

  local TOKS_PER_S=0
  if [ "$ELAPSED" -gt 0 ]; then
    TOKS_PER_S=$(( TOTAL_TOKENS * 1000 / ELAPSED ))
  fi

  log "RESULT c=$C: OK=$OK FAIL=$FAIL wall=${ELAPSED}ms tokens=$TOTAL_TOKENS tok/s=$TOKS_PER_S"

  # Print failure code breakdown
  for code in "${!FAIL_CODES[@]}"; do
    log "  fail_code $code: ${FAIL_CODES[$code]} times"
  done

  # GPU memory after burst
  log "GPU $GPU memory after c=$C:"
  nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader -i "$GPU"

  # Stats after burst
  log "stats after c=$C:"
  curl -sf "$BASE_URL/v1/stats" 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    for k,v in sorted(d.items()):
        print(f'  {k}={v}')
except Exception as e:
    print('  parse error:', e)
" 2>/dev/null || log "  (no /v1/stats)"

  # Check server still alive
  if ! curl -sf "$BASE_URL/v1/models" >/dev/null 2>&1; then
    log "SERVER DOWN after c=$C — escalation complete"
    rm -rf "$TMPDIR_R"
    return 1
  fi

  rm -rf "$TMPDIR_R"

  if [ "$FAIL" -gt 0 ]; then
    log "FAILURES at c=$C — escalation stopping"
    return 2
  fi

  log "  recovery pause..."
  sleep 5
  return 0
}

# ---- Escalation sweep ----
LAST_OK_C=0

for C in 512 1024 2048; do
  run_burst "$C"
  RC=$?
  if [ $RC -eq 0 ]; then
    LAST_OK_C=$C
    log "c=$C CLEAN — continuing"
  elif [ $RC -eq 2 ]; then
    log "=== ESCALATION STOPPED: first failures at c=$C (last clean c=$LAST_OK_C) ==="
    break
  else
    log "=== SERVER CRASHED at c=$C (last clean c=$LAST_OK_C) ==="
    break
  fi
done

log "=== SWEEP DONE: last clean c=$LAST_OK_C ==="
log "GPU $GPU final memory:"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader -i "$GPU"
