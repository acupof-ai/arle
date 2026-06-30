#!/usr/bin/env bash
# burst_test.sh — escalation sweep to find ARLE OOM/failure point
# Runs inside the pod, launched via pod-remote-run.sh wrapper
# Usage: burst_test.sh (no args needed; server started internally)
set -uo pipefail

TREE="${POD_TREE:-/host/arle-build}"
PORT="${BURST_PORT:-8088}"
MODEL_PATH=/host/Qwen3-4B
NUM_SLOTS=1024
GPU="${BURST_GPU:-3}"  # override with BURST_GPU env var; default GPU 3 (free on shared box)

log() { echo "[burst] $*"; }

# ---- Start server ----
log "starting arle serve --num-slots $NUM_SLOTS --port $PORT"
CUDA_VISIBLE_DEVICES=$GPU INFER_CUDA_DEVICE=0 \
  "$TREE/target/release/arle" serve \
    --backend cuda \
    --model-path "$MODEL_PATH" \
    --num-slots "$NUM_SLOTS" \
    --port "$PORT" \
    &
SERVER_PID=$!
log "server pid=$SERVER_PID"

# ---- Wait for health ----
log "waiting for /health..."
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    log "server ready after ${i}s"
    break
  fi
  sleep 1
  if [ $i -eq 120 ]; then
    log "ERROR: server never became healthy in 120s"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
  fi
done

# ---- GPU baseline ----
log "GPU $GPU memory at server-ready:"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader -i "$GPU"

# ---- Stats baseline ----
log "stats at server-ready:"
curl -sf "http://127.0.0.1:$PORT/v1/stats" 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print('  kv_free_pages=' + str(d.get('kv_free_pages', d.get('free_pages', '?'))))
    print('  total_pages=' + str(d.get('kv_total_pages', d.get('total_pages', '?'))))
    print('  active_requests=' + str(d.get('active_requests', '?')))
    print('  full:', json.dumps(d, indent=2)[:500])
except Exception as e:
    print('  stats parse error:', e)
" 2>/dev/null || log "no stats endpoint"

# ---- Burst function ----
run_burst() {
  local C=$1
  log "=== burst c=$C ==="

  local TMPDIR_R
  TMPDIR_R=$(mktemp -d)
  local PIDS=()

  local START
  START=$(date +%s%3N)

  for i in $(seq 1 "$C"); do
    curl -s \
      -o "$TMPDIR_R/body_$i" \
      -w "%{http_code}" \
      -X POST "http://127.0.0.1:$PORT/v1/completions" \
      -H "Content-Type: application/json" \
      -d '{"model":"default","prompt":"Count from 1 to 10 and explain each number.","max_tokens":256,"stream":false}' \
      --max-time 120 \
      > "$TMPDIR_R/code_$i" 2>/dev/null &
    PIDS+=($!)
  done

  for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done

  local END
  END=$(date +%s%3N)
  local ELAPSED=$(( END - START ))

  local OK=0 FAIL=0
  for i in $(seq 1 "$C"); do
    local CODE
    CODE=$(cat "$TMPDIR_R/code_$i" 2>/dev/null || echo "000")
    if [ "$CODE" = "200" ]; then
      OK=$(( OK + 1 ))
    else
      FAIL=$(( FAIL + 1 ))
      # Print first few failures with body snippet
      if [ $FAIL -le 3 ]; then
        local BODY
        BODY=$(cat "$TMPDIR_R/body_$i" 2>/dev/null | head -c 300 || echo "(empty)")
        log "  FAIL code=$CODE body=$BODY"
      fi
    fi
  done

  # Count tokens from successful responses
  local TOTAL_TOKENS=0
  for i in $(seq 1 "$C"); do
    local CODE
    CODE=$(cat "$TMPDIR_R/code_$i" 2>/dev/null || echo "000")
    if [ "$CODE" = "200" ]; then
      local TOKS
      TOKS=$(python3 -c "
import sys, json
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

  # GPU memory
  log "GPU 1 memory after c=$C:"
  nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader -i "$GPU"

  # Stats
  log "stats after c=$C:"
  curl -sf "http://127.0.0.1:$PORT/v1/stats" 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print('  kv_free_pages=' + str(d.get('kv_free_pages', d.get('free_pages', '?'))))
    print('  active_requests=' + str(d.get('active_requests', '?')))
    print('  steps=' + str(d.get('steps', '?')))
    print('  queued=' + str(d.get('queued', '?')))
except Exception as e:
    print('  stats parse error:', e)
" 2>/dev/null || log "no stats endpoint"

  # Check server health
  if ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    log "SERVER DOWN after c=$C"
    rm -rf "$TMPDIR_R"
    return 1
  fi

  rm -rf "$TMPDIR_R"

  # If any failures, report and stop escalation
  if [ "$FAIL" -gt 0 ]; then
    log "FAILURES DETECTED at c=$C — stopping escalation"
    return 2
  fi

  sleep 5  # recovery window between bursts
  return 0
}

# ---- Escalation sweep ----
LAST_OK_C=0
LAST_OK_TOKS=0

for C in 256 512 1024 2048; do
  run_burst "$C"
  RC=$?
  if [ "$RC" -eq 0 ]; then
    LAST_OK_C=$C
  elif [ "$RC" -eq 2 ]; then
    # Failures but server alive
    log "ESCALATION STOPPED at c=$C (failures)"
    break
  else
    # Server down
    log "SERVER CRASHED at c=$C"
    break
  fi
done

log "=== SWEEP COMPLETE: last clean c=$LAST_OK_C ==="
log "GPU 1 final memory:"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader -i "$GPU"

# Graceful shutdown
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
log "server stopped"
