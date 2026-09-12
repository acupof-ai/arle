#!/usr/bin/env bash
# sampling_gate.py / longctx_numerical_gate.py must distinguish a dead serve
# (exit 2) from a reachable serve that genuinely fails the gate (exit 1).
# Drives the real scripts against in-process HTTP servers; no model, no GPU.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP="$(mktemp -d)"
trap 'kill "${SRV_PID:-0}" 2>/dev/null || true; rm -rf "$TMP"' EXIT
export PYTHONPATH="$ROOT/scripts"

# sampling_gate server. FAILMODE:
#   good      arms diverge as required, logit_bias dominated, liveness non-empty
#   miss      every arm returns the reference text verbatim -> gate FAIL
#   malformed valid HTTP/JSON, no choices[] -> transcript error
cat > "$TMP/srv_sampling.py" <<'PY'
import json, os, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
mode = os.environ.get("FAILMODE", "good")
REF = "apple banana cherry date elderberry fig grape honeydew"
DIV = "a clearly different completion about other fruits and vegetables"
BIAS = "zzzz zzzz zzzz zzzz zzzz zzzz"
class H(BaseHTTPRequestHandler):
    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)
    def do_GET(self):
        self._send({})
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); body = json.loads(self.rfile.read(n) or b"{}")
        if mode == "malformed":
            self._send({"unexpected": "shape"}); return
        if self.path.startswith("/v1/stats"):
            self._send({}); return
        if mode == "miss":
            text = REF
        elif "logit_bias" in body:
            text = BIAS
        elif any(k in body for k in ("repetition_penalty", "frequency_penalty", "presence_penalty")):
            text = DIV
        else:
            text = REF  # reference and liveness
        self._send({"choices": [{"message": {"content": text, "reasoning_content": None}}],
                    "usage": {"prompt_tokens": 9, "completion_tokens": 12}})
    def log_message(self, *a): pass
ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
PY

# longctx single-mode server (Raw /v1/completions). FAILMODE:
#   good      non-degenerate text
#   degrade   first five chars identical -> degenerate-output gate FAIL
#   malformed no choices[]
cat > "$TMP/srv_longctx.py" <<'PY'
import json, os, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
mode = os.environ.get("FAILMODE", "good")
class H(BaseHTTPRequestHandler):
    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); self.rfile.read(n)
        if mode == "malformed":
            self._send({"unexpected": "shape"}); return
        if mode == "http500":
            self._send({"error": "boom"}, code=500); return
        text = "aaaaa one two three four" if mode == "degrade" else "a coherent technical sentence about caching."
        self._send({"choices": [{"text": text}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 8}})
    def log_message(self, *a): pass
ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
PY

# Minimal real tokenizer (BPE with one unk) so longctx needs no model dir.
python3 - "$TMP" <<'PY'
import json, sys
from tokenizers import Tokenizer
from tokenizers.models import BPE
out = sys.argv[1]
tok = Tokenizer(BPE(vocab={"<unk>": 0}, merges=[], unk_token="<unk>"))
tok.save(out + "/tokenizer.json")
PY

P_SAMP=19951; P_LONG=19952
start() { FAILMODE="$1" python3 "$2" "$3" >/dev/null 2>&1 & SRV_PID=$!; sleep 1.2; }
stop() { kill "$SRV_PID" 2>/dev/null || true; wait "$SRV_PID" 2>/dev/null || true; }

expect() { # $1 want rc $2 label ; rest=cmd
    local want="$1" label="$2"; shift 2
    local rc=0; "$@" >/dev/null 2>&1 || rc=$?
    if [ "$rc" -ne "$want" ]; then
        echo "FAIL: $label: want exit $want got $rc" >&2; exit 1
    fi
    echo "ok: $label -> exit $rc"
}

# ── sampling_gate ──────────────────────────────────────────────────────────
start good "$TMP/srv_sampling.py" "$P_SAMP"
expect 0 "sampling healthy serve passes" python3 "$ROOT/scripts/sampling_gate.py" "$P_SAMP"
stop
start miss "$TMP/srv_sampling.py" "$P_SAMP"
expect 1 "reachable serve, arms identical -> genuine gate failure" python3 "$ROOT/scripts/sampling_gate.py" "$P_SAMP"
stop
start malformed "$TMP/srv_sampling.py" "$P_SAMP"
expect 2 "malformed transcript is infra, not regression" python3 "$ROOT/scripts/sampling_gate.py" "$P_SAMP"
stop
expect 2 "dead serve is infra, not regression" python3 "$ROOT/scripts/sampling_gate.py" "$((P_SAMP+100))"

# ── longctx_numerical_gate (single mode; tolerance avoids token-count coupling)
LONG=(python3 "$ROOT/scripts/longctx_numerical_gate.py" --label t
      --left-name left --prompt-count 1 --tokenizer "$TMP"
      --prompt-token-tolerance 100000 --out-dir "$TMP/lc")
start good "$TMP/srv_longctx.py" "$P_LONG"
expect 0 "longctx healthy serve passes" "${LONG[@]}" --left-url "http://127.0.0.1:$P_LONG"
stop
start degrade "$TMP/srv_longctx.py" "$P_LONG"
expect 1 "reachable serve, degenerate output -> gate failure" "${LONG[@]}" --left-url "http://127.0.0.1:$P_LONG"
stop
start malformed "$TMP/srv_longctx.py" "$P_LONG"
expect 2 "longctx malformed transcript is infra" "${LONG[@]}" --left-url "http://127.0.0.1:$P_LONG"
stop
start http500 "$TMP/srv_longctx.py" "$P_LONG"
expect 2 "longctx HTTP 500 is infra" "${LONG[@]}" --left-url "http://127.0.0.1:$P_LONG"
stop
expect 2 "longctx dead serve is infra" "${LONG[@]}" --left-url "http://127.0.0.1:$((P_LONG+100))"

echo "test_gate_infra_exit_codes: PASS (pass=0, real failure=1, request/transcript error=2)"
