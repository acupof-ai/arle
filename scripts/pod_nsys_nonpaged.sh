#!/usr/bin/env bash
# nsys duration capture for the base nonpaged_prefill_attention_kernel.
# ncu launch-and-attach deadlocks on the in-process async serve (per-pass
# memory save/restore over the KV pool); nsys only TRACES — no replay, no
# save/restore — so it captures real kernel durations cleanly.
#
# Usage: pod_nsys_nonpaged.sh <label> <gpu> <port>
set -uo pipefail
LABEL="${1:?label}"; GPU="${2:?gpu}"; PORT="${3:?port}"
ROOT=/host/arle-build
MODEL=/host/qwen35-08b-clean
OUT=/tmp/nsys_$LABEL
rm -f "$OUT".nsys-rep "$OUT".sqlite "$OUT.driver.log"
: > "$OUT.driver.log"

CUDA_VISIBLE_DEVICES="$GPU" setsid nsys profile \
  --trace=cuda --sample=none --cpuctxsw=none \
  --force-overwrite true \
  -o "$OUT" \
  "$ROOT/target/release/arle" serve --backend cuda \
    --model-path "$MODEL" --port "$PORT" --qwen35-fa3 false \
  > "$OUT.serve.log" 2>&1 < /dev/null &
NSYS_PID=$!
echo "nsys_pid=$NSYS_PID out=$OUT" | tee -a "$OUT.driver.log"

for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "server up after ${i}s" | tee -a "$OUT.driver.log"; break; fi
  kill -0 "$NSYS_PID" 2>/dev/null || { echo "nsys exited before server-ready" | tee -a "$OUT.driver.log"; break; }
  sleep 1
done

# Drive many prefills so the base kernel fires hundreds of times (6 full-attn
# layers × requests) — a stable duration distribution.
for r in $(seq 1 40); do
  curl -sf "http://127.0.0.1:$PORT/v1/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"x\",\"prompt\":\"Q$r: $(python3 -c 'print("data "*300)') One word.\",\"max_tokens\":2,\"temperature\":0}" \
    >/dev/null 2>&1
done
echo "drove 40 prefills" | tee -a "$OUT.driver.log"

# Stop nsys: SIGTERM the serve group root; nsys flushes the report on exit.
kill -TERM "$NSYS_PID" 2>/dev/null
for _ in $(seq 1 40); do kill -0 "$NSYS_PID" 2>/dev/null || break; sleep 1; done
kill -KILL "$NSYS_PID" 2>/dev/null
pkill -KILL -f "[a]rle serve.*$PORT" 2>/dev/null
sleep 2

echo "=== nsys stats: nonpaged_prefill duration ===" | tee -a "$OUT.driver.log"
nsys stats --report cuda_gpu_kern_sum --format table "$OUT.nsys-rep" 2>/dev/null \
  | grep -iE 'Time|nonpaged_prefill|Name|----' | tee -a "$OUT.driver.log"
echo "NSYS_PROFILE_DONE=$LABEL" | tee -a "$OUT.driver.log"
