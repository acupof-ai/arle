#!/usr/bin/env bash
# OPD capability A/B arm2:
# generalized beta-JSD + greedy rollout. Set KL_BETA=0.3 or KL_BETA=0.5.

set -euo pipefail

ROOT="${ARLE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT"

RUN_ROOT="${RUN_ROOT:-$ROOT/bench-output/opd-ab-math500-beta-jsd-${KL_BETA:-0.5}-$(date +%Y%m%d-%H%M%S)}" \
KL_DIRECTION="${KL_DIRECTION:-forward}" \
KL_BETA="${KL_BETA:-0.5}" \
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.0}" \
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}" \
ROLLOUT_TOP_K="${ROLLOUT_TOP_K:-0}" \
LOGITS_WINDOW_SIZE="${LOGITS_WINDOW_SIZE:-32}" \
GKD_LAMBDA="${GKD_LAMBDA:-0.0}" \
"$ROOT/examples/opd/run-math-r1-35b-to-4b.sh"
