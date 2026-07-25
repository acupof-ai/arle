#!/usr/bin/env bash
# Hand-rolled ncu profile for the base nonpaged_prefill_attention_kernel.
# ncu 2025.2 dropped --attach-pid, so profile_ncu_bench.sh's attach mode
# is dead here — launch arle serve UNDER ncu (launch-and-attach) instead,
# filter to the base kernel, drive a prefill load, capture N launches.
#
# Usage: pod_ncu_nonpaged.sh <label> <gpu> <port>
set -uo pipefail
LABEL="${1:?label}"; GPU="${2:?gpu}"; PORT="${3:?port}"
ROOT=/host/arle-build
MODEL=/host/qwen35-08b-clean
OUT=/tmp/ncu_$LABEL
rm -f "$OUT".ncu-rep "$OUT".log "$OUT".driver.log
: > "$OUT.driver.log"

# ncu launches + suspends arle serve; profiles only the base kernel
# (regex excludes devpos/ring, which are separate .cu files & names).
CUDA_VISIBLE_DEVICES="$GPU" setsid ncu \
  --mode launch-and-attach \
  --target-processes all \
  --kernel-name-base demangled \
  --kernel-name 'regex:nonpaged_prefill_attention_kernel' \
  --launch-skip 6 --launch-count 3 \
  --metrics gpu__time_duration.sum,sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread \
  --clock-control none --cache-control none \
  --kill yes --force-overwrite \
  --log-file "$OUT.log" -o "$OUT" \
  "$ROOT/target/release/arle" serve --backend cuda \
    --model-path "$MODEL" --port "$PORT" --qwen35-fa3 false \
    --max-total-tokens 8192 --mem-fraction-static 0.2 \
  > "$OUT.serve.log" 2>&1 < /dev/null &
NCU_PID=$!
echo "ncu_pid=$NCU_PID out=$OUT" | tee -a "$OUT.driver.log"

# Wait for the server to come up (model load is slow under ncu instrumentation).
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "server up after ${i}s" | tee -a "$OUT.driver.log"; break; fi
  kill -0 "$NCU_PID" 2>/dev/null || { echo "ncu exited before server-ready" | tee -a "$OUT.driver.log"; break; }
  sleep 1
done

# Drive prefill requests: each fires the base kernel once per full-attn layer.
for r in $(seq 1 8); do
  curl -sf "http://127.0.0.1:$PORT/v1/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"x\",\"prompt\":\"Question number $r: $(python3 -c 'print("data "*300)') Answer in one word.\",\"max_tokens\":2,\"temperature\":0}" \
    >/dev/null 2>&1
  echo "drove prefill $r" | tee -a "$OUT.driver.log"
  # stop once ncu has finished (it --kills the app after launch-count)
  kill -0 "$NCU_PID" 2>/dev/null || { echo "ncu done after prefill $r" | tee -a "$OUT.driver.log"; break; }
done

wait "$NCU_PID"; echo "ncu_rc=$?" | tee -a "$OUT.driver.log"
echo "=== metrics (csv from report) ===" | tee -a "$OUT.driver.log"
ncu --import "$OUT.ncu-rep" --csv --page raw 2>/dev/null \
  | grep -iE 'Kernel Name|nonpaged_prefill|gpu__time_duration|sm__warps_active|launch__registers_per_thread' \
  | tee -a "$OUT.driver.log"
ncu --import "$OUT.ncu-rep" --page details 2>/dev/null \
  | grep -iE 'nonpaged_prefill_attention_kernel|gpu__time_duration|sm__warps_active.avg.pct|launch__registers_per_thread' \
  | tee -a "$OUT.driver.log"
echo "NCU_PROFILE_DONE=$LABEL" | tee -a "$OUT.driver.log"