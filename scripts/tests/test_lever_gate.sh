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

if LEVER_GATE_REQUIRE_EXACT=1 LEVER_GATE_VALIDATE_LOG="$tmp/pass.log" \
    LENGTHS=115,300 RUNS=3 "$ROOT/scripts/lever_gate.sh" test >/dev/null 2>&1; then
    echo "lever gate exact mode accepted partial or missed outputs" >&2
    exit 1
fi
if LEVER_GATE_REQUIRE_EXACT=1 LEVER_GATE_VALIDATE_LOG="$tmp/baseline.log" \
    LENGTHS=115,300 RUNS=3 "$ROOT/scripts/lever_gate.sh" test >/dev/null 2>&1; then
    echo "lever gate exact mode accepted a non-exact length" >&2
    exit 1
fi
cat >"$tmp/exact.log" <<'EOF'
SUMMARY len=115 depth=0.00 exact=3 partial=0 miss=0 DET
SUMMARY len=300 depth=0.00 exact=3 partial=0 miss=0 DET
EOF
LEVER_GATE_REQUIRE_EXACT=1 LEVER_GATE_VALIDATE_LOG="$tmp/exact.log" \
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

# Missing baseline is loud unless explicitly allowed (silent-skip guard).
if LEVER_GATE_VALIDATE_LOG="$tmp/pass.log" LENGTHS=115,300 RUNS=3 \
    "$ROOT/scripts/lever_gate.sh" test >/dev/null 2>&1; then
    echo "lever gate accepted a run with no baseline to gate against" >&2
    exit 1
fi
LEVER_GATE_ALLOW_NO_BASELINE=1 LEVER_GATE_VALIDATE_LOG="$tmp/pass.log" \
    LENGTHS=115,300 RUNS=3 "$ROOT/scripts/lever_gate.sh" test >/dev/null

expect_fail() {
    if "$@" >/dev/null 2>&1; then
        echo "unexpected success: $*" >&2
        exit 1
    fi
}

SHA="$(printf product | shasum -a 256 | cut -d' ' -f1)"
KERNEL_ID="bundle:$(printf kernel | shasum -a 256 | cut -d' ' -f1)"
cat >"$tmp/server.py" <<'PY'
import json, os
from http.server import BaseHTTPRequestHandler, HTTPServer
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/v1/models': body = b'{}'
        elif self.path == '/v1/stats': body = os.environ['STATS'].encode()
        else: self.send_error(404); return
        self.send_response(200); self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
    def log_message(self, *_): pass
HTTPServer(('127.0.0.1', int(os.environ['PORT'])), Handler).serve_forever()
PY
cat >"$tmp/bin" <<'EOF'
#!/usr/bin/env bash
exec python3 "$SERVER"
EOF
chmod +x "$tmp/bin"
cat >"$tmp/needle.py" <<'EOF'
#!/usr/bin/env python3
print('SUMMARY len=115 depth=0.00 exact=1 partial=0 miss=0 DET')
EOF
cat >"$tmp/identity-baseline.log" <<'EOF'
SUMMARY len=115 depth=0.00 exact=1 partial=0 miss=0 DET
EOF
mkdir -p "$tmp/root/scripts"
cp "$ROOT/scripts/lever_gate.sh" "$tmp/root/scripts/lever_gate.sh"
cp "$ROOT/scripts/pick-gpu.sh" "$tmp/root/scripts/pick-gpu.sh"
cp "$ROOT/scripts/needle_summary.py" "$tmp/root/scripts/needle_summary.py"
cp "$tmp/needle.py" "$tmp/root/scripts/needle_gate.py"

run_gate() {
    local stats="$1" out="$2" sha="${3:-$SHA}" kernel="${4:-$KERNEL_ID}" port="$5"
    STATS="$stats" SERVER="$tmp/server.py" BIN="$tmp/bin" MODEL=model GATE_PROFILE=generic \
        PORT="$port" LENGTHS=115 RUNS=1 STATS_OUT="$out" EXPECTED_PRODUCT_SHA256="$sha" \
        EXPECTED_KERNEL_BUNDLE_ID="$kernel" BASELINE_LOG="$tmp/identity-baseline.log" \
        "$tmp/root/scripts/lever_gate.sh" identity >/dev/null
}

STATS="{\"build_identity\":{\"product_binary_sha256\":\"sha256:$SHA\",\"kernel_bundle_id\":\"$KERNEL_ID\"}}"
run_gate "$STATS" "$tmp/stats.json" "sha256:$SHA" "$KERNEL_ID" 29189
cmp <(printf '%s' "$STATS") "$tmp/stats.json"

for case in wrong-sha wrong-kernel malformed missing; do
    out="$tmp/$case.json"
    printf sentinel >"$out"
    stats="$STATS"; sha="$SHA"; kernel="$KERNEL_ID"
    case "$case" in
        wrong-sha) sha="$(printf wrong | shasum -a 256 | cut -d' ' -f1)" ;;
        wrong-kernel) kernel="bundle:$(printf wrong | shasum -a 256 | cut -d' ' -f1)" ;;
        malformed) stats='{' ;;
        missing) stats='{}' ;;
    esac
    expect_fail run_gate "$stats" "$out" "$sha" "$kernel" $((29189 + RANDOM % 1000))
    [[ "$(<"$out")" == sentinel ]]
done

expect_fail env EXPECTED_PRODUCT_SHA256="$SHA" STATS_OUT="$tmp/one.json" \
    "$ROOT/scripts/lever_gate.sh" one-sided
expect_fail env EXPECTED_KERNEL_BUNDLE_ID="$KERNEL_ID" STATS_OUT="$tmp/one.json" \
    "$ROOT/scripts/lever_gate.sh" one-sided

mkdir -p "$tmp/mock-bin" "$tmp/claims"
cat > "$tmp/mock-bin/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
case "$*" in
  *--query-compute-apps=gpu_uuid*) printf '%b' "${SMI_COMPUTE:-}" ;;
  *) printf '%b\n' "${SMI:-0, GPU-0, 0, 9.0}" ;;
esac
EOF
chmod +x "$tmp/mock-bin/nvidia-smi"
evidence="$tmp/dsv4-evidence.json"
printf sentinel > "$evidence"
expect_fail env PATH="$tmp/mock-bin:$PATH" ARLE_GPU_CLAIMS="$tmp/claims" \
    GATE_PROFILE=dsv4 INFER_CUDA_DEVICES=0 INFER_TP_SIZE=8 STATS_OUT="$evidence" \
    "$tmp/root/scripts/lever_gate.sh" dsv4-preflight
[[ "$(<"$evidence")" == sentinel ]]
expect_fail env PATH="$tmp/mock-bin:$PATH" ARLE_GPU_CLAIMS="$tmp/claims" \
    GATE_PROFILE=dsv4 INFER_CUDA_DEVICES=0,1,2,3,4,5,6,7 INFER_TP_SIZE=4 \
    "$tmp/root/scripts/lever_gate.sh" dsv4-preflight
