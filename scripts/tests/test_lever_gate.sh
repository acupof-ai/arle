#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

cat >"$tmp/baseline.log" <<'EOF'
SUMMARY len=115 depth=0.00 exact=3 partial=0 miss=0 DET
SUMMARY len=300 depth=0.00 exact=0 partial=0 miss=3 DET
EOF
cat >"$tmp/pass.log" <<'EOF'
SUMMARY len=115 depth=0.00 exact=2 partial=1 miss=0 NONDET
SUMMARY len=300 depth=0.00 exact=1 partial=0 miss=2 NONDET
EOF
LEVER_GATE_VALIDATE_LOG="$tmp/pass.log" BASELINE_LOG="$tmp/baseline.log" \
    LENGTHS=115,300 RUNS=3 "$ROOT/scripts/lever_gate.sh" test >/dev/null

cat >"$tmp/fail.log" <<'EOF'
SUMMARY len=115 depth=0.00 exact=1 partial=0 miss=2 NONDET
SUMMARY len=300 depth=0.00 exact=0 partial=0 miss=3 DET
EOF
if LEVER_GATE_VALIDATE_LOG="$tmp/fail.log" BASELINE_LOG="$tmp/baseline.log" \
    LENGTHS=115,300 RUNS=3 "$ROOT/scripts/lever_gate.sh" test >/dev/null 2>&1; then
    echo "lever gate accepted a result outside the baseline envelope" >&2
    exit 1
fi


cat >"$tmp/miss-regression.log" <<'EOF'
SUMMARY len=115 depth=0.00 exact=2 partial=0 miss=1 NONDET
SUMMARY len=300 depth=0.00 exact=0 partial=0 miss=3 DET
EOF
if LEVER_GATE_VALIDATE_LOG="$tmp/miss-regression.log" BASELINE_LOG="$tmp/baseline.log" \
    LENGTHS=115,300 RUNS=3 "$ROOT/scripts/lever_gate.sh" test >/dev/null 2>&1; then
    echo "lever gate accepted a new miss above the baseline cap" >&2
    exit 1
fi

cat >"$tmp/incomplete.log" <<'EOF'
SUMMARY len=115 depth=0.00 exact=3 partial=0 miss=0 DET
EOF
if LEVER_GATE_VALIDATE_LOG="$tmp/incomplete.log" LENGTHS=115,300 RUNS=3 \
    "$ROOT/scripts/lever_gate.sh" test >/dev/null 2>&1; then
    echo "lever gate accepted an incomplete summary" >&2
    exit 1
fi
