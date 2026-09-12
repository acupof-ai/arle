#!/usr/bin/env bash
# Thinking-model criterion: make BOTH arms' current behavior explicit without
# changing either verdict. The greedy arm judges message.content ONLY — a needle
# that surfaces only in reasoning_content is a miss (the caller received nothing),
# but the run is flagged NEEDLE_REASONING_ONLY so the open criterion question is
# visible in the record. The temp arm concatenates reasoning_content+content for
# its coherence check, so an empty content with non-empty reasoning PASSES it.
# Drives the real scripts against in-process servers; no model, no GPU.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP="$(mktemp -d)"
trap 'kill "${SRV_PID:-0}" 2>/dev/null || true; rm -rf "$TMP"' EXIT

# Response shapes, FAILMODE:
#   content -> needle in content, no reasoning
#   reason  -> content misses, needle only in reasoning_content
#   both    -> needle in both channels
#   neither -> miss text in both
#   raw     -> /v1/completions text carries the needle
#   thinklong -> content empty, reasoning is a long coherent needle answer
#                (completion_tokens=200): greedy miss, temp-arm pass
cat > "$TMP/server.py" <<'PY'
import json, os, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
mode = os.environ.get("FAILMODE", "content")
WORDS = ("access code retrieval context needle answer remember secret "
         "plain sentence coherent explanation hash map lookup insertion "
         "memory storage retrieval works when keys unique open addressing "
         "chaining buckets grow load factor resize rehash capacity double").split()
long_reason = "the secret access code is 738291 . " + " ".join(
    "%s%d" % (w, i) for i, w in enumerate(WORDS * 6))
shapes = {
    "content":   ("738291", ""),
    "reason":    ("I do not remember any code.", "738291"),
    "both":      ("738291", "the code is 738291"),
    "neither":   ("I do not remember any code.", "thinking about something else"),
    "thinklong": ("", long_reason),
}
class H(BaseHTTPRequestHandler):
    def _send(self, body):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)
    def do_GET(self):
        self._send(b"{}")
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); self.rfile.read(n)
        if self.path == "/v1/completions":
            payload = {"choices": [{"text": "738291" if mode == "raw" else shapes[mode][0]}],
                       "usage": {"prompt_tokens": 5, "completion_tokens": 5}}
        else:
            c, r = shapes[mode]
            toks = 200 if mode == "thinklong" else 5
            payload = {"choices": [{"message": {"content": c,
                                                "reasoning_content": r}}],
                       "usage": {"prompt_tokens": 5, "completion_tokens": toks}}
        self._send(json.dumps(payload).encode())
    def log_message(self, *a): pass
ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
PY

export PORT=19941
start() { FAILMODE="$1" python3 "$TMP/server.py" "$PORT" >/dev/null 2>&1 & SRV_PID=$!; sleep 1.2; }
stop() { kill "$SRV_PID" 2>/dev/null || true; wait "$SRV_PID" 2>/dev/null || true; }

check() { # $1=want rc $2=mode $3=label ; rest=gate args
    local want="$1" m="$2" label="$3"; shift 3
    local log="$TMP/out.log" rc=0
    start "$m"
    python3 "$ROOT/scripts/needle_gate.py" "$@" >"$log" 2>&1 || rc=$?
    stop
    if [ "$rc" -ne "$want" ]; then
        echo "FAIL: $label: want exit $want got $rc" >&2; cat "$log" >&2; exit 1
    fi
    cp "$log" "$TMP/last.log"; echo "ok: $label -> exit $rc"
}

grepq() { grep -q "$1" "$TMP/last.log" || { echo "FAIL: expected /$1/ in:" >&2; cat "$TMP/last.log" >&2; exit 1; }; }
noq() { ! grep -q "$1" "$TMP/last.log" || { echo "FAIL: unexpected /$1/ in:" >&2; cat "$TMP/last.log" >&2; exit 1; }; }

# Greedy arm: content is the verdict; reasoning location is diagnostic only.
check 0 content  "needle in content passes" 115 1
grepq "cls=exact loc=content"; noq "NEEDLE_REASONING_ONLY"; noq "reasoning_only=[1-9]"
check 1 reason   "needle only in reasoning_content is STILL a content miss" 115 1
grepq "NEEDLE_REASONING_ONLY"; grepq "cls=miss loc=reasoning_only"; grepq "reasoning_only=1"
check 0 both     "needle in both channels passes" 115 1
grepq "cls=exact loc=both"; noq "NEEDLE_REASONING_ONLY"
check 1 neither  "needle in neither channel fails clean" 115 1
grepq "cls=miss loc=neither"; noq "NEEDLE_REASONING_ONLY"

# Raw completions have no reasoning channel; loc must report content.
start raw
rc=0; RAW=1 python3 "$ROOT/scripts/needle_gate.py" 115 1 >"$TMP/raw.log" 2>&1 || rc=$?
stop
[ "$rc" = 0 ] || { echo "FAIL: raw want 0 got $rc" >&2; cat "$TMP/raw.log" >&2; exit 1; }
grep -q "cls=exact loc=content" "$TMP/raw.log" || { echo "FAIL: raw log" >&2; cat "$TMP/raw.log" >&2; exit 1; }
cp "$TMP/raw.log" "$TMP/last.log"
echo "ok: raw completions judged on text, loc=content"

# The diagnostic never changes the verdict under --report either: reasoning-only
# stays a recorded miss for the external comparator, exit stays 0.
check 0 reason   "--report keeps the miss verdict with the comparator, flagged" --report 115 1
grepq "NEEDLE_REASONING_ONLY"; grepq "exact=0 partial=0 miss=1 reasoning_only=1"

# Temp arm reads reasoning_content+content: the exact shape the greedy arm fails
# (empty content, long needle-bearing reasoning) PASSES the coherence arm.
start thinklong
rc=0; python3 "$ROOT/scripts/needle_gate.py" 115 1 >"$TMP/greedy.log" 2>&1 || rc=$?
stop
[ "$rc" = 1 ] || { echo "FAIL: greedy on thinklong want 1 got $rc" >&2; exit 1; }
grep -q "NEEDLE_REASONING_ONLY" "$TMP/greedy.log" || { echo "FAIL: greedy log missing marker" >&2; cat "$TMP/greedy.log" >&2; exit 1; }
start thinklong
rc=0; python3 "$ROOT/scripts/needle_gate.py" temp >"$TMP/temp.log" 2>&1 || rc=$?
stop
[ "$rc" = 0 ] || { echo "FAIL: temp arm on thinklong want 0 got $rc" >&2; cat "$TMP/temp.log" >&2; exit 1; }
grep -q "TEMP-ARM PASS" "$TMP/temp.log" || { echo "FAIL: no TEMP-ARM PASS" >&2; cat "$TMP/temp.log" >&2; exit 1; }
echo "ok: arms diverge on empty-content+needle-reasoning — greedy exit 1 (marker), temp exit 0"

# The enriched SUMMARY line still parses through the one shared parser, with the
# reasoning-only row excluded from exact and counted as miss (verdict intact).
python3 - "$ROOT" <<'PY'
import sys
sys.path.insert(0, sys.argv[1] + "/scripts")
from needle_summary import parse_summaries
line = "SUMMARY len=115 depth=0.00 exact=0 partial=0 miss=1 reasoning_only=1 DET kv=\n"
row = parse_summaries(line, 1)[115]
assert row == {"exact": 0, "partial": 0, "miss": 1, "det": "DET"}, row
assert parse_summaries(
    "SUMMARY len=115 depth=0.00 exact=1 partial=0 miss=0 DET kv=\n", 1)[115]["exact"] == 1
print("ok: needle_summary parses enriched + legacy SUMMARY, verdict fields intact")
PY

echo "test_needle_gate_reasoning_location: PASS (greedy judges content; temp joins reasoning; both explicit)"
