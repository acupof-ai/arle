#!/usr/bin/env bash
# Assert gate — GPU-dispatch-governance Phase 3
# (docs/plans/gpu-dispatch-governance.md). Turns "链路不通" — an implemented
# dispatch path that never actually fired — from a bench-day surprise into a
# red gate: fail if an EXPECTED dispatch kernel has a runtime count of 0/absent.
#
# Reads the Observe-gate counter (`infer_dispatch_kernel_total`, from
# oplib::linear) off the Prometheus `/metrics` endpoint (NOT `/v1/stats`, which
# is the JSON service-stats). Run AFTER a bench/smoke against the served model.
#
# Usage:
#   scripts/assert_kernel_fired.sh <metrics_url> <expected_variant> [variant...]
#   METRICS_URL env overrides arg 1; default http://localhost:8000/metrics
#
# Exit: 0 = every expected variant fired (count>0); 1 = at least one never
# fired (链路不通); 2 = could not scrape /metrics.
#
# Example (DSv4 smoke):
#   scripts/assert_kernel_fired.sh http://localhost:18190/metrics Dsv4Fp8BatchGemv
set -uo pipefail

url="${METRICS_URL:-${1:-http://localhost:8000/metrics}}"
[[ "${1:-}" == "$url" ]] && shift || { [[ -n "${METRICS_URL:-}" ]] || shift; }

if [[ $# -eq 0 ]]; then
    echo "usage: $0 <metrics_url> <expected_variant> [variant...]" >&2
    exit 2
fi

metrics="$(curl -sf "$url" 2>/dev/null)" || {
    echo "FAIL: could not scrape Prometheus metrics at $url (is the server up? note: /metrics, not /v1/stats)" >&2
    exit 2
}

rc=0
for variant in "$@"; do
    # infer_dispatch_kernel_total{...,variant="<variant>",} <count>
    count="$(printf '%s\n' "$metrics" \
        | grep -E "^infer_dispatch_kernel_total\{[^}]*variant=\"${variant}\"" \
        | awk '{print $NF}' | head -1)"
    if [[ -z "$count" || "$count" == "0" ]]; then
        echo "FAIL: dispatch kernel '${variant}' never fired (count=${count:-absent}) — 链路不通: the path was implemented but not exercised on this workload" >&2
        rc=1
    else
        echo "OK:   dispatch kernel '${variant}' fired ${count} times"
    fi
done

exit $rc
