#!/usr/bin/env bash
# End-to-end smoke test for the ARLE CLI and metal_serve HTTP surface.
#
# Drives the REPL via piped stdin (no TTY) and hits the OpenAI v1 surface on
# the background server. Exits non-zero on any assertion failure; echoes raw
# output so regressions are visible in CI logs.
#
# Usage: scripts/e2e_cli_smoke.sh [model-path]
#   default model: models/Qwen3-0.6B
#
# Prereqs:
#   cargo build --release --no-default-features --features metal
#   cargo build --release --no-default-features --features metal --bin metal_serve

set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO}"

MODEL="${1:-${REPO}/models/Qwen3-0.6B}"
BIN_CLI="${REPO}/target/release/arle"
BIN_SERVE="${REPO}/target/release/metal_serve"
OUT_DIR="$(mktemp -d -t arle-e2e-XXXXXX)"
PORT="${PORT:-8765}"
FAIL=0

pass() { printf '  \033[32mOK\033[0m    %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; FAIL=$((FAIL+1)); }
info() { printf '\033[1;34m==\033[0m %s\n' "$1"; }

assert_contains() {
    local name="$1" needle="$2" file="$3"
    if grep -qF -- "$needle" "$file"; then
        pass "$name — found: $needle"
    else
        fail "$name — missing: $needle (see $file)"
    fi
}

assert_not_contains() {
    local name="$1" needle="$2" file="$3"
    if grep -qF -- "$needle" "$file"; then
        fail "$name — unexpected: $needle (see $file)"
    else
        pass "$name — absent: $needle"
    fi
}

info "Artifacts: $OUT_DIR"
info "Model:     $MODEL"

for b in "$BIN_CLI" "$BIN_SERVE"; do
    if [[ ! -x "$b" ]]; then
        fail "binary missing: $b — build first"
        exit 2
    fi
done

# ----- Section 1: piped REPL -----------------------------------------------
info "Section 1 — ARLE piped REPL (Chat mode)"
SCRIPT_IN="$OUT_DIR/repl_in.txt"
REPL_OUT="$OUT_DIR/repl.out"
EXPORT_PATH="$OUT_DIR/exported.md"

cat > "$SCRIPT_IN" <<EOF
/help
/stats
say hi
/retry
/reset
line one \\
line two
/export $EXPORT_PATH
/models
/stats
/chat
/agent
/chat
/quit
EOF

"$BIN_CLI" \
    --model-path "$MODEL" \
    --max-tokens 32 \
    --temperature 0 \
    --non-interactive \
    < "$SCRIPT_IN" > "$REPL_OUT" 2>&1 &
REPL_PID=$!

# wait up to 180s for the model to load + drain scripted lines
WAITED=0
while kill -0 "$REPL_PID" 2>/dev/null; do
    sleep 2
    WAITED=$((WAITED+2))
    if (( WAITED > 240 )); then
        info "REPL still running after ${WAITED}s — killing"
        kill "$REPL_PID" 2>/dev/null || true
        break
    fi
done
wait "$REPL_PID" 2>/dev/null || true

assert_contains "banner"        "ARLE REPL" "$REPL_OUT"
assert_contains "model line"    "Model:"           "$REPL_OUT"
assert_contains "help listing"  "/help"            "$REPL_OUT"
assert_contains "stats turns"   "turns"            "$REPL_OUT"
assert_contains "chat→agent"    "agent"            "$REPL_OUT"
assert_contains "export note"   "$EXPORT_PATH"     "$REPL_OUT"
assert_contains "reset hint"    "reset"            "$REPL_OUT"
# /retry replays the last user turn; it should either succeed or fail with
# a user-friendly message (not a panic / backtrace).
assert_not_contains "no panic on /retry" "panicked at" "$REPL_OUT"
assert_not_contains "no backtrace"       "stack backtrace" "$REPL_OUT"

if [[ -f "$EXPORT_PATH" ]]; then
    pass "export file created at $EXPORT_PATH"
    assert_contains "export header" "#" "$EXPORT_PATH"
else
    fail "export file missing at $EXPORT_PATH"
fi

# ----- Section 2: metal_serve HTTP -----------------------------------------
info "Section 2 — metal_serve HTTP surface"
SERVE_OUT="$OUT_DIR/serve.out"

"$BIN_SERVE" \
    --model-path "$MODEL" \
    --port "$PORT" \
    --bind 127.0.0.1 \
    --warmup 0 \
    > "$SERVE_OUT" 2>&1 &
SERVE_PID=$!

# wait for the server to come up (look for the listen banner OR accept a /v1/models call)
READY=0
for i in $(seq 1 120); do
    if ! kill -0 "$SERVE_PID" 2>/dev/null; then
        info "metal_serve exited early (see $SERVE_OUT)"; break
    fi
    if curl -s -o /dev/null -w '%{http_code}' --max-time 2 "http://127.0.0.1:${PORT}/v1/models" | grep -q '^200$'; then
        READY=1; break
    fi
    sleep 2
done

if [[ $READY -eq 1 ]]; then
    pass "metal_serve listening on :$PORT"
else
    fail "metal_serve never became ready (see $SERVE_OUT)"
    kill "$SERVE_PID" 2>/dev/null || true
    wait "$SERVE_PID" 2>/dev/null || true
    info "Summary: $FAIL failure(s) — artifacts at $OUT_DIR"
    exit 1
fi

# 2a — /v1/models
MODELS_JSON="$OUT_DIR/models.json"
curl -s --max-time 5 "http://127.0.0.1:${PORT}/v1/models" > "$MODELS_JSON"
assert_contains "/v1/models has 'data'"   '"data"'   "$MODELS_JSON"
assert_contains "/v1/models has 'id'"     '"id"'     "$MODELS_JSON"

# 2b — non-streaming chat completion
CHAT_JSON="$OUT_DIR/chat.json"
curl -s --max-time 60 \
    -H 'Content-Type: application/json' \
    -d '{"model":"test","messages":[{"role":"user","content":"hello"}],"max_tokens":8,"stream":false}' \
    "http://127.0.0.1:${PORT}/v1/chat/completions" > "$CHAT_JSON"
assert_contains "chat (non-stream) has choices" '"choices"' "$CHAT_JSON"
assert_contains "chat (non-stream) has content" '"content"' "$CHAT_JSON"
assert_not_contains "chat (non-stream) no HTML error" "<html" "$CHAT_JSON"

# 2c — streaming chat completion (SSE)
CHAT_SSE="$OUT_DIR/chat.sse"
curl -s -N --max-time 60 \
    -H 'Content-Type: application/json' \
    -d '{"model":"test","messages":[{"role":"user","content":"hi"}],"max_tokens":8,"stream":true}' \
    "http://127.0.0.1:${PORT}/v1/chat/completions" > "$CHAT_SSE"
assert_contains "stream emits data: prefix"    "data: "    "$CHAT_SSE"
assert_contains "stream terminator [DONE]"     "[DONE]"    "$CHAT_SSE"

# 2d — completions endpoint (legacy prompt surface)
COMP_JSON="$OUT_DIR/comp.json"
curl -s --max-time 60 \
    -H 'Content-Type: application/json' \
    -d '{"model":"test","prompt":"The capital of France is","max_tokens":4,"stream":false}' \
    "http://127.0.0.1:${PORT}/v1/completions" > "$COMP_JSON"
# /v1/completions may or may not be wired; accept either a valid payload or a clean 404-ish body
if grep -q '"choices"' "$COMP_JSON"; then
    pass "/v1/completions returned choices"
else
    assert_not_contains "/v1/completions returns a sane body (no stack trace)" "panicked" "$COMP_JSON"
fi

kill "$SERVE_PID" 2>/dev/null || true
wait "$SERVE_PID" 2>/dev/null || true

# ----- Summary --------------------------------------------------------------
info "Summary: $FAIL failure(s) — artifacts at $OUT_DIR"
exit $FAIL
