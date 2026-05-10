#!/usr/bin/env bash
# task43_hypothesis_test.sh — single-shot 2-arm A/B for Task #43 hypothesis
# (cudarc allocator fragmentation under sustained W4A16 4k load).
#
# Per docs/research/2026-05-10-task43-w4a16-frag-hypothesis-confirmed-via-dispatch-audit.md
# §4: hypothesis is that W4A16 sustained-load failure (Task #43) shares root
# cause with PF8.3 KILL — both per-call alloc fragmentation when scratch
# fallback path is taken.
#
# Dispatch verified at linear.rs:2064-2095:
# - INFER_PREFILL_GRAPH=1  → marlin_scratch=Some → uses _with_scratch (no per-call alloc)
# - INFER_PREFILL_GRAPH    NOT set → marlin_scratch=None → falls back to per-call alloc
#
# Two arms:
# - Arm A (treatment): INFER_PREFILL_GRAPH=1 → predicted HEALTHY
# - Arm B (baseline):  no env var          → predicted SUBSTRATE-KILL or near-OOM
#
# If Arm B reproduces Task #43 + Arm A doesn't → hypothesis CONFIRMED.
# Fix direction: make scratch default-on OR add thread-local buffer fallback.
#
# Cross-refs:
#   - 1ba06f0 hypothesis source
#   - 2cc608a H1' design REVISION (linked Task #47 + #43)
#   - 868e147 pf83_bench_health.sh (verdict tool)
#   - 35fc3cf Task #24 (introduced INFER_PREFILL_GRAPH)
#
# Usage:
#   bash scripts/task43_hypothesis_test.sh
#
# Outputs:
#   /tmp/task43-A-scratch-enabled.log  Arm A server log
#   /tmp/task43-B-scratch-disabled.log Arm B server log
#   bench-output/2026-05-10-task43-A-scratch-enabled/  Arm A bench
#   bench-output/2026-05-10-task43-B-scratch-disabled/ Arm B bench
#   stdout                             A/B verdict + hypothesis result

set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

MODEL_PATH="infer/models/Qwen3-4B-W4A16-sym-g128-marlin"
PORT=8000
TARGET="http://127.0.0.1:$PORT"
DURATION=60
CONCURRENCY=4
PROMPT_TOKENS=4096
OUTPUT_TOKENS=128

# Pre-flight checks
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "ERROR: model checkpoint missing: $MODEL_PATH" >&2
  exit 1
fi

if [[ ! -x ".venv/bin/guidellm" ]]; then
  echo "ERROR: guidellm not in .venv. Run: pip install -e .[bench]" >&2
  exit 1
fi

if [[ ! -x "target/release/infer" ]]; then
  echo "ERROR: infer binary not built. Run: CUDA_HOME=/usr/local/cuda cargo build --release" >&2
  exit 1
fi

PATH=".venv/bin:$PATH"
export PATH

run_arm() {
  local arm="$1"        # "A-scratch-enabled" | "B-scratch-disabled"
  local env_setup="$2"  # full env var prefix string
  local server_log="/tmp/task43-${arm}.log"
  local output_dir="$REPO/bench-output/2026-05-10-task43-${arm}"

  # Cleanup any prior server (port-level, not command-string match per
  # codex's self-pkill misadventure 2026-05-10)
  fuser -k "$PORT/tcp" 2>/dev/null || true
  sleep 2

  mkdir -p "$output_dir"

  echo ""
  echo "==================== ARM $arm ===================="
  echo "[task43] env: $env_setup"
  echo "[task43] server log: $server_log"
  echo "[task43] bench output: $output_dir"

  RUST_MIN_STACK=33554432 \
    eval "$env_setup setsid target/release/infer \
      --model-path '$MODEL_PATH' \
      --port '$PORT' \
      > '$server_log' 2>&1 &"
  local server_pid=$!

  echo "[task43] Waiting up to 90s for server readiness..."
  local ready=0
  for i in $(seq 1 90); do
    if curl -fsS "$TARGET/v1/models" >/dev/null 2>&1; then
      echo "[task43] Server ready after ${i}s"
      ready=1
      break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "ERROR: server died during startup. Last log lines:" >&2
      tail -50 "$server_log" >&2
      return 2
    fi
    sleep 1
  done

  if [[ "$ready" -eq 0 ]]; then
    echo "ERROR: server readiness timeout after 90s. Last log lines:" >&2
    tail -50 "$server_log" >&2
    fuser -k "$PORT/tcp" 2>/dev/null || true
    return 3
  fi

  echo "[task43] Running guidellm conc=$CONCURRENCY ${DURATION}s sustained-load bench..."
  guidellm benchmark run \
      --target "$TARGET" \
      --model "$MODEL_PATH" \
      --processor "$MODEL_PATH" \
      --profile concurrent --rate "$CONCURRENCY" --max-seconds "$DURATION" --warmup 5 \
      --random-seed 20260416 \
      --data "prompt_tokens=$PROMPT_TOKENS,prompt_tokens_stdev=1,prompt_tokens_min=$PROMPT_TOKENS,prompt_tokens_max=$PROMPT_TOKENS,output_tokens=$OUTPUT_TOKENS,output_tokens_stdev=1,output_tokens_min=$OUTPUT_TOKENS,output_tokens_max=$OUTPUT_TOKENS" \
      --output-dir "$output_dir" \
      --backend openai_http \
      --backend-kwargs '{"validate_backend": "/v1/models", "request_format": "/v1/completions"}' \
      --disable-console-interactive \
      --outputs json --outputs csv --outputs html

  # Cleanup server before next arm
  kill -TERM "$server_pid" 2>/dev/null || true
  sleep 3
  fuser -k "$PORT/tcp" 2>/dev/null || true

  # Run health check on this arm
  echo ""
  echo "----- Arm $arm health check -----"
  bash "$REPO/scripts/pf83_bench_health.sh" "$output_dir" "$server_log"
  return $?
}

# Run both arms
echo "[task43] === Task #43 hypothesis test ==="
echo "[task43] Hypothesis: W4A16 sustained-load failure shares root cause with PF8.3 KILL"
echo "[task43]   Arm A (treatment): INFER_PREFILL_GRAPH=1 → marlin_scratch=Some → no per-call alloc"
echo "[task43]   Arm B (baseline):  no env var          → marlin_scratch=None → per-call alloc fallback"
echo ""

run_arm "A-scratch-enabled" "INFER_PREFILL_GRAPH=1"
ARM_A_HEALTH=$?
echo "[task43] Arm A exit: $ARM_A_HEALTH (0=HEALTHY, 1=SUBSTRATE-KILL, 2=TOOL-QUIRK, 3=NO-OUTPUT)"

run_arm "B-scratch-disabled" ""
ARM_B_HEALTH=$?
echo "[task43] Arm B exit: $ARM_B_HEALTH (0=HEALTHY, 1=SUBSTRATE-KILL, 2=TOOL-QUIRK, 3=NO-OUTPUT)"

echo ""
echo "==================== HYPOTHESIS VERDICT ===================="
if [[ "$ARM_A_HEALTH" -eq 0 && "$ARM_B_HEALTH" -eq 1 ]]; then
  echo "🎯 HYPOTHESIS CONFIRMED"
  echo "   Arm A (scratch enabled) HEALTHY + Arm B (scratch disabled) SUBSTRATE-KILL"
  echo "   = W4A16 Task #43 IS env-gated allocator fragmentation"
  echo ""
  echo "   FIX direction: make INFER_PREFILL_GRAPH=1 default-on (or add thread-local"
  echo "   alloc buffer for the per-call fallback path)."
  echo "   See docs/research/2026-05-10-task43-w4a16-frag-hypothesis-confirmed-via-dispatch-audit.md §5"
  exit 0
elif [[ "$ARM_A_HEALTH" -eq 0 && "$ARM_B_HEALTH" -eq 0 ]]; then
  echo "🚫 HYPOTHESIS DISPROVEN"
  echo "   Both arms HEALTHY = Task #43 root cause is NOT env-gated scratch fragmentation"
  echo "   Pivot to other Task #43 investigation paths"
  exit 0
elif [[ "$ARM_A_HEALTH" -ne 0 && "$ARM_B_HEALTH" -ne 0 ]]; then
  echo "⚠  AMBIGUOUS — both arms failed (Arm A: $ARM_A_HEALTH, Arm B: $ARM_B_HEALTH)"
  echo "   Investigate environmental issues before re-running"
  exit 1
else
  echo "⚠  PARTIAL signal — Arm A: $ARM_A_HEALTH, Arm B: $ARM_B_HEALTH"
  echo "   Inspect /tmp/task43-A-scratch-enabled.log and /tmp/task43-B-scratch-disabled.log"
  exit 1
fi
