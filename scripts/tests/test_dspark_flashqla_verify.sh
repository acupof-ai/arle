#!/usr/bin/env bash
# Mock-driven control-flow test for scripts/dspark_flashqla_verify.sh.
# No git, cargo, GPU, prereg ledger, or real serve: binaries and python tools
# are replaced with mocks through the script's test seams. A clean world exits
# 0 with a TSV+MD report; a failing lever forces exit 1.
#
# The mock arle is a global-counter stats server: every chat completion adds a
# fixed server-global drafted/accepted delta to /v1/stats. The mock acceptance
# tool takes ONE before/after pair around its thread pool, exactly like the
# fixed bench_dspark_accept.py. The old implementation (8 processes each
# snapshotting the global counter) would overcount c=8 8x; the assertion on the
# c=8 drafted total catches that regression.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

mkdir -p "$TMP/bin" "$TMP/tools"

# Mock arle: on `serve ... --port P`, print the single-process marker then run
# an HTTP server with GLOBAL spec counters: POST adds 100 drafted / 13 accepted.
cat > "$TMP/bin/arle" <<'SH'
#!/usr/bin/env bash
if [ "${1:-}" = serve ]; then
  port=""; shift
  while [ $# -gt 0 ]; do
    if [ "$1" = "--port" ]; then port="$2"; shift 2; else shift; fi
  done
  echo "[multiproc-coord] world_size=1; serving single-process (no workers)"
  exec python3 - "$port" <<'PY'
import sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
lock = threading.Lock(); drafted = 0; accepted = 0
class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def _send(self, body, ctype="application/json"):
        self.send_response(200); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)
    def do_GET(self):
        if self.path.endswith("/v1/models"):
            self._send(b"{}"); return
        with lock:
            fq = drafted // 100  # one gdr_fq op hit per processed request
            body = (
                '{"spec_decode":{"available":true,"drafted":%d,"accepted":%d},'
                '"op_timing":{"ops":[{"name":"linear/gdr_fq","total_micros":12,"count":%d},'
                '{"name":"linear/gdr_recurrent","total_micros":0,"count":0}]}}'
                % (drafted, accepted, fq)).encode()
        self._send(body)
    def do_POST(self):
        global drafted, accepted
        n = int(self.headers.get("Content-Length", 0))
        if n: self.rfile.read(n)  # drain body before replying, else client sees RST
        with lock: drafted += 100; accepted += 13
        self._send(b'{"choices":[]}')
    def log_message(self, *a): pass
ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
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

# Mock bench_dspark_accept.py: mirrors the fixed tool's contract — parse
# --concurrency, ONE global /v1/stats pair around an in-process thread pool.
# Records each invocation so the test can assert exactly one call per c.
cat > "$TMP/tools/bench_dspark_accept.py" <<'SH'
#!/usr/bin/env python3
import argparse, json, os, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
ap = argparse.ArgumentParser()
ap.add_argument("--port", default=8000); ap.add_argument("--measure-requests", type=int, default=1)
ap.add_argument("--concurrency", type=int, default=1); ap.add_argument("--max-tokens", type=int, default=1)
ap.add_argument("--output", required=True)
a = ap.parse_args()
base = f"http://127.0.0.1:{a.port}"
def stats():
    with urllib.request.urlopen(base + "/v1/stats", timeout=5) as r:
        return json.load(r)["spec_decode"]
def post(_):
    req = urllib.request.Request(base + "/v1/chat/completions", data=b"{}",
                                 headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=5).read()
with open(os.environ["ACCEPT_INVOC_LOG"], "a") as f:
    f.write(f"concurrency={a.concurrency} requests={a.measure_requests}\n")
before = stats()
with ThreadPoolExecutor(max_workers=a.concurrency) as pool:
    list(pool.map(post, range(a.concurrency * a.measure_requests)))
time.sleep(0.1)
after = stats()
json.dump({"drafted": after["drafted"] - before["drafted"],
           "accepted": after["accepted"] - before["accepted"],
           "concurrency": a.concurrency}, open(a.output, "w"))
SH
chmod +x "$TMP/tools/"*.py

export ACCEPT_INVOC_LOG="$TMP/accept-invocations.log"
: > "$ACCEPT_INVOC_LOG"

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
# 2 arms x (launch + needle + prefill + accept-c1 + accept-c8 + gdr-path) = 12.
nrows=$(tail -n +2 "$OUT1/results.tsv" | wc -l | tr -d ' ')
[ "$nrows" = 12 ] || { echo "FAIL: expected 12 rows, got $nrows" >&2; cat "$OUT1/results.tsv" >&2; exit 1; }
grep -qE '^base\t1\tgdr-path\tgdr_fq=[0-9]+ gdr_recurrent=0\tINFO\t' "$OUT1/results.tsv" \
    || { echo "FAIL: default gdr-path INFO row wrong" >&2; cat "$OUT1/results.tsv" >&2; exit 1; }
grep -qE '^base\t1\tneedle\t.*\tPASS\t' "$OUT1/results.tsv" \
    || { echo "FAIL: base needle row not PASS" >&2; cat "$OUT1/results.tsv" >&2; exit 1; }
grep -qE '^base\t1\tlaunch\tobserved workers=1 \(attn_tp=1\)\tPASS\t' "$OUT1/results.tsv" \
    || { echo "FAIL: launch/attn_tp row missing" >&2; cat "$OUT1/results.tsv" >&2; exit 1; }
grep -qE '^treatment\t1\tneedle\t.*\tPASS\t' "$OUT1/results.tsv" \
    || { echo "FAIL: treatment needle row not PASS" >&2; exit 1; }
grep -qE 'ttft_mean_ms=120.5' "$OUT1/results.tsv" \
    || { echo "FAIL: prefill TTFT metric missing" >&2; cat "$OUT1/results.tsv" >&2; exit 1; }
# Single global pair: c=1 drafted=100; c=8 (8 POSTs) drafted=800, NOT 8*800.
grep -qE '^base\t1\taccept-c1\t13/100\tPASS' "$OUT1/results.tsv" \
    || { echo "FAIL: accept-c1 global delta wrong" >&2; cat "$OUT1/results.tsv" >&2; exit 1; }
grep -qE '^base\t1\taccept-c8\t104/800\tPASS' "$OUT1/results.tsv" \
    || { echo "FAIL: accept-c8 must be single global pair 104/800 (overlap would be ~832/6400)" >&2
         cat "$OUT1/results.tsv" >&2; exit 1; }
# Exactly one tool invocation per c with the right --concurrency.
[ "$(grep -c '^concurrency=8 ' "$ACCEPT_INVOC_LOG")" = 2 ] \
    || { echo "FAIL: expected one c=8 acceptance call per arm" >&2; cat "$ACCEPT_INVOC_LOG" >&2; exit 1; }
grep -q '| base | 1 | needle |' "$OUT1/results.md" \
    || { echo "FAIL: markdown table missing" >&2; exit 1; }

# ── Dirty world: failing lever forces non-zero ──
OUT2="$TMP/out-bad"
: > "$ACCEPT_INVOC_LOG"
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

# ── GDR_CHUNKED=0: the path check enforces gdr_fq=0; the mock serve can't
# honor the flag (still reports gdr_fq>0), so the run must FAIL — proving a
# misspelled/no-op switch can't pass.
OUT4="$TMP/out-gdr"
: > "$ACCEPT_INVOC_LOG"
# shellcheck disable=SC2046  # intentional KEY=VAL word-split
if env $(common_env "$TMP/lever-ok.sh") GDR_CHUNKED=0 \
    bash "$ROOT/scripts/dspark_flashqla_verify.sh" deadbeef cafebabe "$OUT4" \
    >"$TMP/gdr.log" 2>&1; then
    echo "FAIL: GDR_CHUNKED=0 with gdr_fq>0 must fail (switch not enforced)" >&2
    cat "$TMP/gdr.log" >&2; exit 1
fi
grep -qE '^base\t1\tgdr-path\tgdr_fq=[1-9][0-9]* .*\tFAIL\t' "$OUT4/results.tsv" \
    || { echo "FAIL: gdr-path enforcement row missing" >&2; cat "$OUT4/results.tsv" >&2; exit 1; }

echo "test_dspark_flashqla_verify: PASS (clean exit0 + 12 rows + global c8=800, needle fail exit1, gpu SKIP, gdr-off enforced)"
