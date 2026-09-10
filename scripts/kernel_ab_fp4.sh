#!/usr/bin/env bash
# Step 0b matched A/B: fp4-gemv vs marlin-fp4-gemm at M=1, both arms against
# the same CPU f32 reference (infer_quant::cpu_ref_fp4) in one process, so the
# comparison is matched. Prints both kernel lines and the winner; exits
# non-zero if either arm failed its correctness check.
#
# The keep/remove call for fp4-gemv: it has had no production caller since
# dispatch convergence, and this A/B decides whether the GEMV beats the
# tensor-core GEMM at M=1 (errors/2026-09-09-w4afp8-gemv-killed-item-dsv4-moe-only).
set -euo pipefail
BUILD="${1:?usage: kernel_ab_fp4.sh <build-label> [run-label] [auto|GPU] [shape] [iters]}"
LABEL="${2:-bench-kab-$(date -u +%Y%m%dT%H%M%SZ)}"
GPU="${3:-auto}"
SHAPE="${4:-1,34816,5120}"
ITERS="${5:-100}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$HERE/pod.sh" kernel-ab "$BUILD" "$LABEL" "$GPU" "$SHAPE" "$ITERS"
deadline=$((SECONDS + 1800)); out=""
while [ "$SECONDS" -le "$deadline" ]; do
  out="$(bash "$HERE/pod.sh" status "$LABEL" 2>/dev/null || true)"
  case "$out" in *DONE*) break;; esac
  sleep 5
done
case "$out" in
  *"DONE exit=0") ;;
  *DONE*) echo "$out" >&2; bash "$HERE/pod.sh" log "$LABEL" >&2; exit 1;;
  *) echo "kernel-ab '$LABEL' did not finish within 1800s: $out" >&2; exit 1;;
esac
log="$(bash "$HERE/pod.sh" log "$LABEL")"
printf '%s\n' "$log" | grep -E '^kernel=|^=== '
gemv="$(printf '%s\n' "$log" | sed -n 's/^kernel=fp4-gemv .*time_ms=\([0-9.]*\).*/\1/p')"
marlin="$(printf '%s\n' "$log" | sed -n 's/^kernel=marlin-fp4-gemm .*time_ms=\([0-9.]*\).*/\1/p')"
awk -v g="$gemv" -v m="$marlin" 'BEGIN {
  if (g=="" || m=="") { print "verdict: missing time_ms (gemv=" g " marlin=" m)" > "/dev/stderr"; exit 1 }
  winner = (g < m ? "fp4-gemv" : "marlin-fp4-gemm")
  ratio = (g < m ? m / g : g / m)
  printf "verdict: fp4-gemv %.4f ms vs marlin-fp4-gemm %.4f ms -> %s wins (%.2fx)\n", g, m, winner, ratio
}'
