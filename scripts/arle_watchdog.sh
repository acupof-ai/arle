#!/usr/bin/env bash
# ARLE serve watchdog: restarts the serve if the process exits or the engine
# becomes unresponsive. Logs to /tmp/arle_watchdog.log.
#
# Environment overrides:
#   ARLE_MODEL       model path (default: ThinkingCap W4A16)
#   ARLE_PORT        port (default: 8000)
#   ARLE_BIN         arle binary path
#   ARLE_EXTRA_ARGS  extra args appended to the serve command
#   ARLE_MAX_RESTARTS  max restarts per window (default: 10)
#   ARLE_RESTART_WINDOW  window in seconds (default: 300)

set -euo pipefail

MODEL="${ARLE_MODEL:-/data00/home/chenkailun.c/models/ThinkingCap-Qwen3.6-27B-W4A16}"
PORT="${ARLE_PORT:-8000}"
ARLE_BIN="${ARLE_BIN:-/data00/home/chenkailun.c/arle/target/release/arle}"
LOG="/tmp/arle_serve.log"
WATCH_LOG="/tmp/arle_watchdog.log"
MAX_RESTARTS="${ARLE_MAX_RESTARTS:-10}"
RESTART_WINDOW="${ARLE_RESTART_WINDOW:-300}"
STARTUP_GRACE="${ARLE_STARTUP_GRACE:-300}"  # seconds before health checks begin
MAX_RUNNING="${ARLE_MAX_RUNNING:-8}"  # cap slots; each slot reserves 146MB recurrent state
HEALTH_URL="http://127.0.0.1:${PORT}/health"

restart_count=0
window_start=$(date +%s)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$WATCH_LOG"; }

start_serve() {
    log "starting arle serve (model=$MODEL port=$PORT)"
    # --cuda-mempool-retain false: release pool at sync so VRAM doesn't grow
    # unboundedly on 32GB cards (pool retain mode caches everything, OOM after
    # enough requests on V100).
    "$ARLE_BIN" serve \
        --model-path "$MODEL" \
        --bind 0.0.0.0 \
        --port "$PORT" \
        --cuda-mempool-retain false \
        --max-running-requests "$MAX_RUNNING" \
        --mem-fraction-static 0.7 \
        ${ARLE_EXTRA_ARGS:-} \
        >> "$LOG" 2>&1 &
    SERVE_PID=$!
    SERVE_START=$(date +%s)
    log "serve pid=$SERVE_PID"
}

kill_serve() {
    kill "$SERVE_PID" 2>/dev/null || true
    wait "$SERVE_PID" 2>/dev/null || true
}

log "watchdog started (pid=$$)"
start_serve

while true; do
    sleep 10

    # Reset restart counter if the window has passed
    now=$(date +%s)
    if [ $((now - window_start)) -gt "$RESTART_WINDOW" ]; then
        restart_count=0
        window_start=$now
    fi

    # Check process alive
    if ! kill -0 "$SERVE_PID" 2>/dev/null; then
        wait "$SERVE_PID" 2>/dev/null || true
        exit_code=$?
        log "serve exited (code=$exit_code)"
    # Health check only after startup grace (model load takes ~60-90s)
    elif [ $((now - SERVE_START)) -lt "$STARTUP_GRACE" ]; then
        continue
    elif ! curl -sf --max-time 5 "$HEALTH_URL" | grep -q '"healthy"' 2>/dev/null; then
        log "health check failed, killing serve"
        kill_serve
    else
        continue
    fi

    restart_count=$((restart_count + 1))
    if [ "$restart_count" -gt "$MAX_RESTARTS" ]; then
        log "too many restarts ($restart_count in ${RESTART_WINDOW}s); giving up"
        exit 1
    fi

    backoff=$((5 * restart_count))
    [ $backoff -gt 60 ] && backoff=60
    log "restarting in ${backoff}s (attempt $restart_count/$MAX_RESTARTS)"
    sleep "$backoff"
    start_serve
done
