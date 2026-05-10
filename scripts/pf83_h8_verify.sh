#!/usr/bin/env bash
# PF8.3 H8 verify — post-build smoke that fires the diagnostic OR proves
# H8 disproven. Per docs/plans/M_pf83_h8_fix_patch.md.
#
# Pre-condition:
#   - cargo build --release -p infer --features cuda completed cleanly
#     after applying H8 diagnostic patch (81672c3) to marlin_w4_fp8_kernel.cu
#   - target/release/infer mtime > marlin_w4_fp8_kernel.cu mtime
#
# Outcome:
#   - "cleared pre-existing CUDA error" message in stderr → H8 CONFIRMED
#   - No such message + still code 2 → H8 disproven, pivot H1'

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PATH=$REPO_ROOT/.venv/bin:$PATH

# Cleanup any lingering server
pkill -f "target/release/infer.*--port 8000" 2>/dev/null
sleep 2

LOG=/tmp/pf83-h8-verify.log
RUST_MIN_STACK=33554432 \
  INFER_HYBRID_W4A8_PREFILL=1 \
  INFER_MARLIN_W4_FP8_PREFILL=1 \
  target/release/infer \
  --model-path infer/models/Qwen3-4B-W4-hybrid-zpfix \
  --port 8000 \
  > "$LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID $SERVER_PID, log $LOG"

# Wait for /healthz
READY=0
for i in $(seq 1 30); do
    sleep 2
    if curl -s --max-time 1 http://127.0.0.1:8000/healthz 2>&1 | grep -q '"status":"ok"'; then
        echo "Server ready after ${i}×2 sec"
        READY=1
        break
    fi
done

if [[ $READY -eq 0 ]]; then
    echo "ERROR: server never became ready"
    cat "$LOG" | tail -30
    kill -9 $SERVER_PID 2>/dev/null
    exit 1
fi

# Trigger 1 PF8 request
echo "=== curl /v1/completions ==="
curl -s --max-time 30 http://127.0.0.1:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"Qwen3-4B-W4-hybrid-zpfix","prompt":"The quick brown fox","max_tokens":10,"temperature":0,"stream":false}' \
    | head -3
echo
echo "=== curl second request ==="
curl -s --max-time 30 http://127.0.0.1:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"Qwen3-4B-W4-hybrid-zpfix","prompt":"In the beginning","max_tokens":10,"temperature":0,"stream":false}' \
    | head -3
echo

# Check for H8 diagnostic
sleep 1
echo "=== H8 diagnostic check ==="
if grep -q "cleared pre-existing CUDA error" "$LOG"; then
    echo "✅ H8 CONFIRMED: diagnostic fired"
    grep "cleared pre-existing CUDA error" "$LOG" | head -5
    H8=1
else
    echo "❌ H8 NOT confirmed: diagnostic never fired"
    H8=0
fi

# Check kernel error rate
echo
echo "=== gemm_w4_fp8_marlin_cuda failure count ==="
COUNT=$(grep -c "gemm_w4_fp8_marlin_cuda failed" "$LOG" 2>/dev/null || echo 0)
echo "Failures: $COUNT"

# Cleanup
echo
echo "=== cleanup ==="
kill -TERM $SERVER_PID 2>/dev/null
sleep 2
kill -9 $SERVER_PID 2>/dev/null

echo "=== verdict ==="
if [[ $H8 -eq 1 && $COUNT -eq 0 ]]; then
    echo "🎉 H8 CONFIRMED + kernel succeeded → apply Option B fix per 1b3f76c §2"
elif [[ $H8 -eq 1 && $COUNT -gt 0 ]]; then
    echo "⚠ H8 partially: diagnostic fires but kernel still failing"
    echo "  → real cause might be combination; investigate H6 (stream/ctx) next"
elif [[ $H8 -eq 0 && $COUNT -gt 0 ]]; then
    echo "❌ H8 DISPROVEN + kernel still failing → revert patch + pivot H1' static-scratch refactor"
else
    echo "🤔 No diagnostic + no failures → kernel working without prior errors? bench v11 to confirm"
fi
