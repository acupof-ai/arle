#!/usr/bin/env bash
# needle_gate.py must express its verdict in its exit code. Regression for the
# defect where the standalone gate always exited 0 even when the needle was
# missed or the serve was unreachable (lever_gate.sh was the only thing that
# ever failed). Drives the real script against a tiny in-process HTTP server
# with switchable response shapes; no model, no GPU.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP="$(mktemp -d)"
trap 'kill "${SRV_PID:-0}" 2>/dev/null || true; rm -rf "$TMP"' EXIT

# Synthetic OpenAI-shaped server. FAILMODE selects the completion content:
#   ok       -> the needle is returned (chat + raw text)
#   miss     -> no needle fragment
#   partial  -> only the "738" prefix
#   think    -> needle present only in reasoning_content, content empty
#   malformed-> valid HTTP/JSON but no choices[] (transcript/shape defect)
cat > "$TMP/server.py" <<'PY'
import json, os, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
mode = os.environ.get("FAILMODE", "ok")
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
        if mode == "malformed":
            payload = {"unexpected": "shape", "usage": {}}
        else:
            text = {"ok": "738291", "miss": "I do not remember any code.",
                    "partial": "738 maybe", "think": ""}[mode]
            reason = "738291" if mode == "think" else text
            payload = {"choices": [{"text": text,
                                    "message": {"content": text,
                                                "reasoning_content": reason}}],
                       "usage": {"prompt_tokens": 5, "completion_tokens": 5}}
        self._send(json.dumps(payload).encode())
    def log_message(self, *a): pass
ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
PY

export PORT=19400
start() { FAILMODE="$1" python3 "$TMP/server.py" "$PORT" >/dev/null 2>&1 & SRV_PID=$!; sleep 1.2; }
stop() { kill "$SRV_PID" 2>/dev/null || true; wait "$SRV_PID" 2>/dev/null || true; }
gate() { python3 "$ROOT/scripts/needle_gate.py" "$@" >/dev/null 2>&1; }
expect_rc() { # $1=want $2=label ; rest=args
    local want="$1" label="$2"; shift 2
    local rc=0; gate "$@" || rc=$?
    if [ "$rc" -ne "$want" ]; then
        echo "FAIL: $label: want exit $want got $rc (args: $*)" >&2; exit 1
    fi
    echo "ok: $label -> exit $rc"
}

# Default (standalone gate): a model miss is a real failure, the needle a pass.
start ok;       expect_rc 0 "needle retrieved, default gate" 115 1; stop
start miss;     expect_rc 1 "needle missed, default gate must fail" 115 1; stop
start partial;  expect_rc 1 "partial-only is below the exact threshold" 115 1; stop
start ok;       expect_rc 0 "--check alias passes on needle" --check 115; stop
start miss;     expect_rc 1 "--check alias fails on miss" --check 115; stop
start ok;       expect_rc 1 "--min-exact above observed exact count" --min-exact 2 115 1; stop

# Infra / transcript failures are distinct (exit 2), never a model miss (1):
# a malformed JSON shape and an unreachable serve both abort the run.
start malformed; expect_rc 2 "malformed transcript is an infra error" 115 1; stop
gate 115 1 >/dev/null 2>&1 || rc=$?
[ "$rc" = 2 ] || { echo "FAIL: unreachable serve want exit 2 got ${rc:-0}" >&2; exit 1; }
echo "ok: unreachable serve -> exit 2"

# --report keeps the model-miss verdict with the caller (exit 0 on a miss so a
# baseline-envelope comparator can judge) but a request ERROR still aborts.
start miss;      expect_rc 0 "--report tolerates a model miss for external compare" --report 115 1; stop
start malformed; expect_rc 2 "--report still fatal on infra error" --report 115 1; stop

echo "test_needle_gate_exit_code: PASS (miss=1, needle=0, infra=2, report miss=0, report infra=2)"
