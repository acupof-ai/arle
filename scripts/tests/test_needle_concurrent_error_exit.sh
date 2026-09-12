#!/usr/bin/env bash
# needle_concurrent.py must separate a cross-row miss from a request error in
# both its exit code and its summary line. Regression for the false-record
# defect where a dead serve was folded into total_miss, so the artifact read
# "N rows missed the needle" when the server was simply not running.
#   0 = all rows retrieved their needle
#   1 = valid response, wrong needle (cross-row mix-up / retrieval miss)
#   2 = one or more request ERRORs (dead serve, refused, malformed transcript)
# Drives the real script against an in-process server; no model/GPU.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP="$(mktemp -d)"
trap 'kill "${SP:-0}" 2>/dev/null || true; rm -rf "$TMP"' EXIT

# good: echoes the per-row secret parsed from the prompt -> every row matches.
cat > "$TMP/good.py" <<'PY'
import sys, json, re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
class H(BaseHTTPRequestHandler):
    def _send(self, b):
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self): self._send(b"{}")
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); body = self.rfile.read(n).decode()
        m = re.search(r"secret access code is (\d+)", body)
        self._send(json.dumps({"choices": [{"text": m.group(1) if m else "none"}]}).encode())
    def log_message(self, *a): pass
ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
PY
# miss: returns a fixed wrong secret for every row.
cat > "$TMP/miss.py" <<'PY'
import sys, json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
class H(BaseHTTPRequestHandler):
    def _send(self, b):
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self): self._send(b"{}")
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); self.rfile.read(n)
        self._send(json.dumps({"choices": [{"text": "wrong secret entirely"}]}).encode())
    def log_message(self, *a): pass
ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
PY
# bad: valid HTTP/JSON but no choices[] (malformed transcript) -> request error.
cat > "$TMP/bad.py" <<'PY'
import sys, json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
class H(BaseHTTPRequestHandler):
    def _send(self, b):
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self): self._send(b"{}")
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); self.rfile.read(n)
        self._send(json.dumps({"unexpected": "shape"}).encode())
    def log_message(self, *a): pass
ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
PY

python3 "$TMP/good.py" 19931 >/dev/null 2>&1 & SP=$!; sleep 1.5
good_rc=0; python3 "$ROOT/scripts/needle_concurrent.py" 19931 3 100 1 0 >"$TMP/good.log" 2>&1 || good_rc=$?
kill "$SP" 2>/dev/null; wait "$SP" 2>/dev/null || true

python3 "$TMP/miss.py" 19932 >/dev/null 2>&1 & SP=$!; sleep 1.5
miss_rc=0; python3 "$ROOT/scripts/needle_concurrent.py" 19932 3 100 1 0 >"$TMP/miss.log" 2>&1 || miss_rc=$?
kill "$SP" 2>/dev/null; wait "$SP" 2>/dev/null || true

python3 "$TMP/bad.py" 19933 >/dev/null 2>&1 & SP=$!; sleep 1.5
bad_rc=0; python3 "$ROOT/scripts/needle_concurrent.py" 19933 3 100 1 0 >"$TMP/bad.log" 2>&1 || bad_rc=$?
kill "$SP" 2>/dev/null; wait "$SP" 2>/dev/null || true

dead_rc=0; python3 "$ROOT/scripts/needle_concurrent.py" 19934 2 100 1 0 >"$TMP/dead.log" 2>&1 || dead_rc=$?

fail() { echo "FAIL: $*" >&2; exit 1; }
[ "$good_rc" = 0 ] || fail "good run want 0 got $good_rc"
grep -q "CONCURRENT_NEEDLE PASS total_miss=0 total_error=0" "$TMP/good.log" \
  || { cat "$TMP/good.log" >&2; fail "good summary line"; }
[ "$miss_rc" = 1 ] || fail "wrong-secret run want 1 got $miss_rc"
grep -q "CONCURRENT_NEEDLE FAIL total_miss=3 total_error=0" "$TMP/miss.log" \
  || { cat "$TMP/miss.log" >&2; fail "miss must be counted separately from errors"; }
[ "$bad_rc" = 2 ] || fail "malformed transcript want 2 got $bad_rc"
grep -q "CONCURRENT_NEEDLE ERROR total_error=3 total_miss=0" "$TMP/bad.log" \
  || { cat "$TMP/bad.log" >&2; fail "bad-transcript summary must separate errors"; }
[ "$dead_rc" = 2 ] || fail "dead serve want 2 got $dead_rc"
grep -q "CONCURRENT_NEEDLE ERROR total_error=2 total_miss=0" "$TMP/dead.log" \
  || { cat "$TMP/dead.log" >&2; fail "dead-serve summary must not say needles were missed"; }

echo "test_needle_concurrent_error_exit: PASS (good=0, miss=1, malformed=2, dead=2; counts separated)"
