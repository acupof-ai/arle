#!/usr/bin/env bash
# Pod-side: atomically pick the freest idle GPU and soft-claim it, so concurrent
# agents auto-allocate DIFFERENT GPUs without passing one explicitly.
#   busy = memory.used > 2 GB  OR  a claim marker younger than 90 s (covers the
#   window between picking and the run's VRAM actually showing up).
# The flock serializes concurrent picks; the 90 s timestamp marker stops two
# back-to-back picks from choosing the same GPU before either has loaded.
# Prints the chosen index, or NONE (exit 1) if all are busy.
set -uo pipefail
mkdir -p /tmp/arle-gpu-claims
exec 9>/tmp/arle-gpu-pick.lock
flock 9
now=$(date +%s)
pick=""
while IFS=',' read -r idx used _rest; do
  idx=$(echo "$idx" | tr -dc '0-9'); used=$(echo "$used" | tr -dc '0-9')
  [ -z "$idx" ] && continue
  [ "${used:-0}" -gt 2000 ] && continue
  c="/tmp/arle-gpu-claims/$idx"
  if [ -f "$c" ]; then ts=$(cat "$c" 2>/dev/null || echo 0); [ $((now - ${ts:-0})) -lt 90 ] && continue; fi
  pick="$idx"; break
done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader)
[ -z "$pick" ] && { echo "NONE"; exit 1; }
echo "$now" > "/tmp/arle-gpu-claims/$pick"
echo "$pick"
