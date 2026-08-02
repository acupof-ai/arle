#!/usr/bin/env bash
# >65535 CP gate: ARLE_ND_SEQ=131072, cp=2, devices 0,1. Local shard 65536 ->
# ring path; head_dim=128 means single-card ref uses chunked-prefill (no seq^2
# f32 materialization). Samples host + GPU mem while running; captures dmesg on
# exit so a host-RAM cgroup OOM is attributable. Detached + RUN_EXIT marker.
set -uo pipefail
BIN="/host/arle-build/target/release/examples/nd_parallel_parity"
LOG="/host/nd-parity-gate.log"
MEM="/host/nd-parity-gate.mem.log"
: > "$LOG"; : > "$MEM"
# Pin to clean physical GPUs (shared box: 0/2/5 get polluted by other users).
# CUDA_VISIBLE_DEVICES remaps physical 1,3 -> logical 0,1 so BOTH the autograd
# backend (cudarc device 0) and the example's ranks land on idle GPUs.
export CUDA_VISIBLE_DEVICES=1,3
export ARLE_ND_CUDA_DEVICES=0,1
export ARLE_ND_SEQ=131072
export ARLE_ND_DIR="/host/arle_nd_parity_gate_$$"

# Background sampler: timestamp, total/used/avail host mem, this-cgroup anon RSS
# sum of any nd_parallel_parity proc, and GPU0/1 used MiB.
(
  while :; do
    ts=$(date +%s)
    read _ memtotal _ < <(grep -m1 MemTotal /proc/meminfo)
    read _ memavail _ < <(grep -m1 MemAvailable /proc/meminfo)
    rss=$(ps -eo rss,comm | awk '/nd_parallel_par/{s+=$1} END{print s+0}')
    gpu=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | awk -F', ' '$1==1||$1==3{printf "g%s=%sMiB ",$1,$2}')
    printf '%s memTotalKB=%s memAvailKB=%s parityRSSkB=%s %s\n' "$ts" "$memtotal" "$memavail" "$rss" "$gpu" >> "$MEM"
    sleep 3
  done
) &
sampler=$!

"$BIN" > "$LOG" 2>&1
rc=$?
kill "$sampler" 2>/dev/null

{
  echo "RUN_EXIT=$rc"
  echo "=== peak parity RSS (kB) ==="
  awk '{gsub("parityRSSkB=","",$4); if($4+0>m)m=$4+0} END{print m}' "$MEM"
  echo "=== min MemAvailable (kB) ==="
  awk '{for(i=1;i<=NF;i++) if($i ~ /^memAvailKB=/){gsub("memAvailKB=","",$i); if(n==0||$i+0<m){m=$i+0;n=1}}} END{print m}' "$MEM"
  echo "=== last mem sample ==="
  tail -1 "$MEM"
  echo "=== dmesg OOM (last 40) ==="
  dmesg 2>/dev/null | grep -iE "out of memory|killed process|oom|cgroup" | tail -40 || echo "(dmesg unreadable or no OOM lines)"
} >> "$LOG"
echo "RUN_EXIT=$rc"
exit "$rc"
