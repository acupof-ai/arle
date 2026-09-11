#!/usr/bin/env bash
# Mock-driven control-flow test for scripts/dspark_flashqla_verify.sh.
# No git, cargo, GPU, prereg ledger, or real serve: binaries and python tools
# are replaced with mocks through the script's test seams. A clean world exits
# 0 with a TSV+MD report; a failing lever forces exit 1.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

mkdir -p "$TMP/bin" "$TMP/tools"

# Mock arle: on `serve ... --port P`, run a tiny HTTP server answering 200 to
# every path so the script's ready-wait and mocks succeed.
cat > "$TMP/bin/arle" <<'SH'
#!/usr/bin/env bash
if [ "${1:-}" = serve ]; then
  port=""; shift
  while [ $# -gt 0 ]; do
    if [ "$1" = "--port" ]; then port="$2"; shift 2; else shift; fi
  done
  exec python3 - "$port" <<'PY'
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
class H(BaseHTTPRequestHandler):
    def do_GET(self): self.send_response(200); self.end_headers(); self.wfile.write(b"{}")
    def log_message(self, *a): pass
HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
PY
fi
exit 0
SH
chmod +x "$TMP/bin/arle"

# Mock lever: green world touches its OUT needle log; dirty world exits 1.
cat > "$TMP/lever-ok.sh" <<'SH'
#!/usr/bin/env bash
echo "mock needle PASS"; : > "${OUT:-/dev/null}"
exit 0
SH
cat > "$TMP/lever-bad.sh" <<'SH'
#!/usr/bin/env bash
echo "mock needle FAIL" >&2
exit 1
SH
chmod +x "$TMP/lever-ok.sh" "$TMP/lever-bad.sh"

# Mock bench_throughput.py: write a report shaped like the real one.
cat > "$TMP/tools/bench_throughput.py" <<'SH'
#!/usr/bin/env python3
import argparse, json
ap = argparse.ArgumentParser(); ap.add_argument("--output", required=True); a, _ = ap.parse_known_args()
rep = {"points": [{"summary": {"complete": 4, "prompt_tokens": 32000,
  "ttft": {"mean_ms": 120.5, "p50_ms": 118.0}}}]}
with open(a.output + ".json", "w") as f: json.dump(rep, f)
SH

# Mock bench_dspark_accept.py: c=1 ~13%, c=8 (8 concurrent clients) ~0% shape,
# both are recorded measurements, never gates.
cat > "$TMP/tools/bench_dspark_accept.py" <<'SH'
#!/usr/bin/env python3
import argparse, json
ap = argparse.ArgumentParser()
ap.add_argument("--port"); ap.add_argument("--measure-requests", type=int)
ap.add_argument("--max-tokens", type=int); ap.add_argument("--output")
a = ap.parse_args()
json.dump({"drafted": 100, "accepted": 13}, open(a.output, "w"))
SH
chmod +x "$TMP/tools/"*.py

common_env() {  # $1 = lever script
    printf '%s ' \
    ARLE_DSV_BIN_BASE="$TMP/bin/arle" \
    ARLE_DSV_BIN_TREAT="$TMP/bin/arle" \
    ARLE_DSV_TPS="1" ARLE_DSV_FREE_GPUS="0" ARLE_DSV_NO_CLAIM=1 \
    ARLE_DSV_LEVER="$1" ARLE_DSV_TOOLS_DIR="$TMP/tools" \
    ARLE_DSV_SKIP_PREREG=1 BENCH_SECONDS=1 ACCEPT_REQUESTS=1 \
    MODEL="$TMP/model" DRAFT_MODEL="$TMP/draft"
}

# ── Clean world: exit 0, full report ──
OUT1="$TMP/out-ok"
# shellcheck disable=SC2046  # common_env emits KEY=VAL words by design
if ! env $(common_env "$TMP/lever-ok.sh") \
    bash "$ROOT/scripts/dspark_flashqla_verify.sh" deadbeef cafebabe "$OUT1" \
    >"$TMP/ok.log" 2>&1; then
    echo "FAIL: clean world exited non-zero" >&2; cat "$TMP/ok.log" >&2; exit 1
fi
[ -f "$OUT1/results.tsv" ] || { echo "FAIL: results.tsv missing" >&2; exit 1; }
[ -f "$OUT1/results.md" ] || { echo "FAIL: results.md missing" >&2; exit 1; }
# 2 arms x (needle + prefill + accept-c1 + accept-c8) = 8 rows at tp 1.
nrows=$(tail -n +2 "$OUT1/results.tsv" | wc -l | tr -d ' ')
[ "$nrows" = 8 ] || { echo "FAIL: expected 8 rows, got $nrows" >&2; cat "$OUT1/results.tsv" >&2; exit 1; }
grep -qE '^base\t1\tneedle\t.*\tPASS\t' "$OUT1/results.tsv" \
    || { echo "FAIL: base needle row not PASS" >&2; cat "$OUT1/results.tsv" >&2; exit 1; }
grep -qE '^treatment\t1\tneedle\t.*\tPASS\t' "$OUT1/results.tsv" \
    || { echo "FAIL: treatment needle row not PASS" >&2; exit 1; }
grep -qE 'ttft_mean_ms=120.5' "$OUT1/results.tsv" \
    || { echo "FAIL: prefill TTFT metric missing" >&2; cat "$OUT1/results.tsv" >&2; exit 1; }
grep -qE '^base\t1\taccept-c1\t13/100\tPASS' "$OUT1/results.tsv" \
    || { echo "FAIL: accept-c1 metric missing" >&2; cat "$OUT1/results.tsv" >&2; exit 1; }
grep -q '13%' "$TMP/ok.log" || true
grep -q '| base | 1 | needle |' "$OUT1/results.md" \
    || { echo "FAIL: markdown table missing" >&2; exit 1; }

# ── Dirty world: failing lever forces non-zero ──
OUT2="$TMP/out-bad"
# shellcheck disable=SC2046  # intentional KEY=VAL word-split
if env $(common_env "$TMP/lever-bad.sh") \
    bash "$ROOT/scripts/dspark_flashqla_verify.sh" deadbeef cafebabe "$OUT2" \
    >"$TMP/bad.log" 2>&1; then
    echo "FAIL: needle-fail world exited 0" >&2; cat "$TMP/bad.log" >&2; exit 1
fi
grep -q 'correctness FAIL' "$TMP/bad.log" \
    || { echo "FAIL: missing correctness FAIL line" >&2; cat "$TMP/bad.log" >&2; exit 1; }
grep -qE '^base\t1\tneedle\t-\tFAIL\t' "$OUT2/results.tsv" \
    || { echo "FAIL: FAIL row not recorded" >&2; cat "$OUT2/results.tsv" >&2; exit 1; }

# ── Insufficient GPUs: tp 8 SKIPped, still exit 0 when nothing fails ──
OUT3="$TMP/out-skip"
# shellcheck disable=SC2046  # intentional KEY=VAL word-split
if ! env $(common_env "$TMP/lever-ok.sh") ARLE_DSV_TPS="8" \
    bash "$ROOT/scripts/dspark_flashqla_verify.sh" deadbeef cafebabe "$OUT3" \
    >"$TMP/skip.log" 2>&1; then
    echo "FAIL: skip world exited non-zero" >&2; cat "$TMP/skip.log" >&2; exit 1
fi
grep -qE '^base\t8\tall\t-\tSKIP\t' "$OUT3/results.tsv" \
    || { echo "FAIL: tp8 SKIP row missing" >&2; cat "$OUT3/results.tsv" >&2; exit 1; }

echo "test_dspark_flashqla_verify: PASS (clean exit0 + 8 rows, needle fail exit1, gpu SKIP)"
