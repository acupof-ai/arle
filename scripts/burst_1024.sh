#!/usr/bin/env bash
# burst_1024.sh — single c=1024 burst with max_tokens=128 to beat ELKEID timeout
# Targeted run: find if c=1024 completes or hits a hard limit
set -uo pipefail

PORT="${BURST_PORT:-8088}"
GPU="${BURST_GPU:-3}"
BASE_URL="http://127.0.0.1:$PORT"
C=1024
MAX_TOKENS=128

log() { echo "[burst1024 $(date -u +%H:%M:%S)] $*"; }

if ! curl -sf "$BASE_URL/v1/models" >/dev/null 2>&1; then
  log "ERROR: server not responding at $BASE_URL"
  exit 1
fi
log "server OK on $BASE_URL"

log "GPU $GPU memory baseline:"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader -i "$GPU"

log "=== BURST c=$C max_tokens=$MAX_TOKENS ==="

TMPDIR_R=$(mktemp -d)
PIDS=()

START=$(date +%s%3N)

for i in $(seq 1 $C); do
  curl -s \
    -o "$TMPDIR_R/body_$i" \
    -w "%{http_code}" \
    -X POST "$BASE_URL/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"default\",\"prompt\":\"Count from 1 to 10 and explain each number.\",\"max_tokens\":$MAX_TOKENS,\"stream\":false}" \
    --max-time 300 \
    > "$TMPDIR_R/code_$i" 2>/dev/null &
  PIDS+=($!)
done

log "all $C requests fired, waiting..."

# Poll stats every 10s while waiting
CHECK_INTERVAL=10
while true; do
  DONE=0
  for pid in "${PIDS[@]}"; do
    kill -0 "$pid" 2>/dev/null || DONE=$((DONE+1))
  done
  STATS=$(curl -s "$BASE_URL/v1/stats" 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    s = d['scheduler']
    t = d['throughput']
    print(f\"active={s['active_requests']} queue={s['queue_depth']} free_pages={s['kv_free_pages']} steps={t['steps']} gen={t['generated_tokens']} completed={t['requests_completed']}\")
except:
    print('stats_error')
" 2>/dev/null || echo "server_down")
  log "  [poll] done_curl=$DONE/$C $STATS"
  [ "$DONE" -ge "$C" ] && break
  sleep "$CHECK_INTERVAL"
done

# Wait for remaining curl subprocesses
for pid in "${PIDS[@]}"; do
  wait "$pid" 2>/dev/null || true
done

END=$(date +%s%3N)
ELAPSED=$(( END - START ))

OK=0 FAIL=0
for i in $(seq 1 "$C"); do
  CODE=$(cat "$TMPDIR_R/code_$i" 2>/dev/null || echo "000")
  if [ "$CODE" = "200" ]; then
    OK=$((OK+1))
  else
    FAIL=$((FAIL+1))
    if [ $FAIL -le 3 ]; then
      BODY=$(cat "$TMPDIR_R/body_$i" 2>/dev/null | head -c 300 || echo "(empty)")
      log "  FAIL[$i] code=$CODE body=$BODY"
    fi
  fi
done

TOTAL_TOKENS=0
for i in $(seq 1 "$C"); do
  CODE=$(cat "$TMPDIR_R/code_$i" 2>/dev/null || echo "000")
  if [ "$CODE" = "200" ]; then
    TOKS=$(python3 -c "
import json
try:
    d = json.load(open('$TMPDIR_R/body_$i'))
    print(d.get('usage', {}).get('completion_tokens', 0))
except:
    print(0)
" 2>/dev/null || echo "0")
    TOTAL_TOKENS=$((TOTAL_TOKENS+TOKS))
  fi
done

TOKS_PER_S=0
[ "$ELAPSED" -gt 0 ] && TOKS_PER_S=$((TOTAL_TOKENS*1000/ELAPSED))

log "RESULT c=$C max_tokens=$MAX_TOKENS: OK=$OK FAIL=$FAIL wall=${ELAPSED}ms tokens=$TOTAL_TOKENS tok/s=$TOKS_PER_S"

log "GPU $GPU memory after:"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader -i "$GPU"

log "final stats:"
curl -sf "$BASE_URL/v1/stats" 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    for k,v in sorted(d.items()):
        print(f'  {k}={v}')
except Exception as e:
    print('  error:', e)
" 2>/dev/null || log "  server down"

# Check server alive
curl -sf "$BASE_URL/v1/models" >/dev/null 2>&1 && log "server ALIVE after c=$C" || log "server DEAD after c=$C"

rm -rf "$TMPDIR_R"
