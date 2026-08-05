#!/usr/bin/env bash
# nsys GPU-kernel profile of ONE FlashQLA-on OPD step (fqgate perf_on).
# nsys traces the whole process tree, so wrapping fqgate captures the arle child.
set -uo pipefail
OUT=/host/fq-nsys-out
REP=$OUT/fqperf_on
mkdir -p "$OUT"
: > "$OUT/nsys-run.log"
{
  echo "NSYS_RUN_START=$(date -u +%FT%TZ)"
  nsys profile -t cuda \
    --sample=none --cpuctxsw=none \
    --force-overwrite true \
    -o "$REP" \
    bash /host/fqgate.sh perf_on
  echo "NSYS_PROFILE_EXIT=$?"
  echo "=== cuda_gpu_kern_sum ==="
  nsys stats --report cuda_gpu_kern_sum --format table "$REP.nsys-rep"
  echo "NSYS_STATS_EXIT=$?"
  echo "NSYS_RUN_END=$(date -u +%FT%TZ)"
} >> "$OUT/nsys-run.log" 2>&1
