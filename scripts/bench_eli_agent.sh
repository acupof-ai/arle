#!/usr/bin/env bash
# Eli-driven agent-workload bench harness.
#
# Boots agent-infer with the Metal backend, points an Eli client at it,
# replays a fixed multi-turn agentic trace (session reuse + tool-shaped
# turns), and emits guidellm-shape headline metrics into bench-output/.
#
# Tied to docs/plans/agent-workload-api.md §5. Until the §3.1/§3.3 P0
# extensions land, this harness is a scaffold:
#   - The eli driver mode produces real per-turn wall-clocks but no
#     server-side TTFT (nexil cannot stream tools yet — see G1).
#   - nexil's OpenAI adapter does not forward Eli's session_id to the
#     upstream server (crates/nexil/src/providers/openai.rs builds the
#     request body from messages/model/stream/tools/kwargs only — see
#     plan G4). Until that lands, /v1/stats will report
#     session_affinity_hit=0 even on a multi-turn Eli replay. The
#     harness reports the raw counter rather than papering over it.
#   - The openai-direct fallback mode delegates to bench_agent_trace.py
#     and produces the canonical TTFT/ITL/tok-s/req-s numbers today.
#
# Usage:
#   ./scripts/bench_eli_agent.sh <label>
#   ./scripts/bench_eli_agent.sh <label> --mode openai-direct
#   ./scripts/bench_eli_agent.sh <label> --port 8000 --model models/Qwen3.5-0.8B
#   ./scripts/bench_eli_agent.sh <label> --keep-server   # leave infer running on exit
#   ./scripts/bench_eli_agent.sh <label> --no-server     # use an already-running infer
#
# Required:
#   <label>   short tag; lands in the output dir name and results.json.
#
# Optional flags:
#   --mode {eli|openai-direct}     Default: eli. openai-direct skips Eli
#                                  and exercises the OpenAI surface
#                                  directly via bench_agent_trace.py.
#   --port N                       infer HTTP port. Default: 8765.
#   --model PATH                   metal_serve --model-path. Default:
#                                  models/default (matches every existing
#                                  bench_agent_trace.py call site, which
#                                  hardcodes the wire-format model id to
#                                  "default"). The wire-format id used in
#                                  every request is the basename of this
#                                  path; metal_serve.rs:195 derives it.
#   --model-id NAME                Override the wire-format model id sent in
#                                  every request. Default: basename(--model).
#                                  metal_serve validates request.model
#                                  against this; mismatch returns 404
#                                  model_not_found (openai_v1.rs:60).
#   --server-bin PATH              Override the server binary. Default:
#                                  target/release/metal_serve. The Metal
#                                  HTTP server lives in
#                                  infer/src/bin/metal_serve.rs and is
#                                  gated to required-features=["metal"];
#                                  the cuda-only `infer` binary will not
#                                  build here.
#   --trace JSONL                  Override the agent trace.
#                                  Default: scripts/data/eli_agent_trace.jsonl
#   --eli-bin PATH                 Path to the eli binary. Default: search
#                                  $ELI_REPO/target/release/eli, then PATH.
#   --eli-repo PATH                Eli checkout root. Default:
#                                  /path/to/code/eli (only used to
#                                  locate the binary in target/).
#   --max-tokens N                 Per-turn output cap. Default: 64.
#   --keep-server                  Do not stop metal_serve on exit.
#   --no-server                    Don't boot metal_serve; expect one
#                                  already running at --port.
#   --no-trace-poll                Skip the /v1/stats trace poller.
#
# Preconditions:
#   * Built metal_serve release binary at target/release/metal_serve.
#     If missing this script tries to build it with
#       cargo build --release -p infer --no-default-features \
#         --features metal --bin metal_serve
#     Pass --no-server to skip the build (e.g. against a remote server).
#   * On --mode eli: eli built (cargo build --release in the eli repo)
#     and reachable as described above.
#   * On --mode openai-direct: python3 with httpx (`pip install -e .[bench]`).
#     Also: bench_agent_trace.py hardcodes request.model="default"
#     (scripts/bench_agent_trace.py:727), so this mode requires --model-id
#     to be "default" — otherwise every request 404s. Use the default
#     --model models/default unless you have a good reason.
#
# Side effects:
#   * Writes raw artefacts to bench-output/<date>-bench-eli-agent-<label>/
#       results.json
#       headline.md
#       turns.jsonl
#       server.log              (only if we booted infer)
#       service_stats_before.json
#       service_stats_trace.jsonl  (5s poll, only if not --no-trace-poll)
#       service_stats_after.json
#       command.txt
#   * Does NOT seed a wins/ entry. Per docs/CLAUDE.md §Benchmarks, scaffold
#     additions don't require one until real numbers land. Once the §3.1/§3.3
#     P0 extensions ship and a labelled run produces canonical numbers,
#     copy results.json + headline.md into a wins/ entry from
#     docs/experience/wins/TEMPLATE-bench-agent-load.md.

set -euo pipefail

# ---- Defaults --------------------------------------------------------------

LABEL=""
MODE="eli"
PORT="8765"
MODEL_PATH="models/default"
MODEL_ID=""
SERVER_BIN_OVERRIDE=""
TRACE_PATH=""
ELI_BIN=""
ELI_REPO="${ELI_REPO_ROOT:-/path/to/code/eli}"
MAX_TOKENS="64"
KEEP_SERVER="0"
USE_SERVER="1"
TRACE_POLL="1"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATE_TAG="$(date +%Y-%m-%d)"

# BSD `date` on macOS does not support %N (would emit a literal "N" suffix
# and poison downstream arithmetic). Route every millisecond timestamp
# through python3 — the harness already requires it for the metric path.
now_ms() {
    python3 -c 'import time; print(int(time.time()*1000))'
}

START_EPOCH_MS="$(now_ms)"
WAIT_HEALTH_SECS="60"
TRACE_POLL_INTERVAL="5"

usage() {
    # Print only the leading comment block (header) — stops at the first
    # non-comment line so this stays correct when help text grows.
    awk '/^[^#]/ { exit } NR>1 { print }' "$0"
    exit "${1:-0}"
}

if [[ $# -lt 1 ]]; then usage 2; fi
LABEL="$1"; shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)             MODE="$2"; shift 2;;
        --port)             PORT="$2"; shift 2;;
        --model)            MODEL_PATH="$2"; shift 2;;
        --model-id)         MODEL_ID="$2"; shift 2;;
        --server-bin)       SERVER_BIN_OVERRIDE="$2"; shift 2;;
        --trace)            TRACE_PATH="$2"; shift 2;;
        --eli-bin)          ELI_BIN="$2"; shift 2;;
        --eli-repo)         ELI_REPO="$2"; shift 2;;
        --max-tokens)       MAX_TOKENS="$2"; shift 2;;
        --keep-server)      KEEP_SERVER="1"; shift;;
        --no-server)        USE_SERVER="0"; shift;;
        --no-trace-poll)    TRACE_POLL="0"; shift;;
        -h|--help)          usage 0;;
        *) echo "unknown flag: $1" >&2; usage 2;;
    esac
done

case "$MODE" in
    eli|openai-direct) ;;
    *) echo "--mode must be eli or openai-direct (got: $MODE)" >&2; exit 2;;
esac

if [[ -z "$TRACE_PATH" ]]; then
    TRACE_PATH="$REPO_ROOT/scripts/data/eli_agent_trace.jsonl"
fi
if [[ ! -f "$TRACE_PATH" ]]; then
    echo "trace not found: $TRACE_PATH" >&2; exit 2
fi

# Derive the wire-format model id from the model path basename when not
# explicitly overridden. metal_serve.rs:195 uses the same derivation, so
# this matches what the server will validate request.model against.
if [[ -z "$MODEL_ID" ]]; then
    MODEL_ID="$(basename "$MODEL_PATH")"
fi
if [[ -z "$MODEL_ID" ]]; then
    echo "could not derive model id from --model $MODEL_PATH; pass --model-id" >&2
    exit 2
fi

# bench_agent_trace.py hardcodes request.model="default". Refuse the
# openai-direct mode early when the resolved id won't match — silently
# producing 100% 404s would be worse than failing now.
if [[ "$MODE" == "openai-direct" && "$MODEL_ID" != "default" ]]; then
    cat >&2 <<EOF
[bench-eli-agent] --mode openai-direct requires --model-id default
  (bench_agent_trace.py:727 hardcodes "model": "default" in every request).
  Either pass --model models/default (so basename is "default"), or
  pass --model-id default to override the wire-format id, or use
  --mode eli (which honors ELI_MODEL).
EOF
    exit 2
fi

OUT_DIR="$REPO_ROOT/bench-output/${DATE_TAG}-bench-eli-agent-${LABEL}"
mkdir -p "$OUT_DIR"
TURNS_JSONL="$OUT_DIR/turns.jsonl"
RESULTS_JSON="$OUT_DIR/results.json"
HEADLINE_MD="$OUT_DIR/headline.md"
SERVER_LOG="$OUT_DIR/server.log"
TRACE_FILE="$OUT_DIR/service_stats_trace.jsonl"
STATS_BEFORE="$OUT_DIR/service_stats_before.json"
STATS_AFTER="$OUT_DIR/service_stats_after.json"
COMMAND_TXT="$OUT_DIR/command.txt"

: > "$TURNS_JSONL"

{
    echo "label=$LABEL"
    echo "mode=$MODE"
    echo "port=$PORT"
    echo "model=$MODEL_PATH"
    echo "model_id=$MODEL_ID"
    echo "trace=$TRACE_PATH"
    echo "max_tokens=$MAX_TOKENS"
    echo "use_server=$USE_SERVER"
    echo "keep_server=$KEEP_SERVER"
    echo "trace_poll=$TRACE_POLL"
    echo "date_tag=$DATE_TAG"
    echo "start_epoch_ms=$START_EPOCH_MS"
} > "$COMMAND_TXT"

# ---- Bootstrap helpers -----------------------------------------------------

# Metal HTTP server lives in infer/src/bin/metal_serve.rs (gated to
# required-features=["metal"]). The cuda-only `infer` binary will not
# build on this Mac.
SERVER_BIN="${SERVER_BIN_OVERRIDE:-$REPO_ROOT/target/release/metal_serve}"
SERVER_PID=""
TRACE_POLL_PID=""

cleanup() {
    if [[ -n "$TRACE_POLL_PID" ]] && kill -0 "$TRACE_POLL_PID" 2>/dev/null; then
        kill "$TRACE_POLL_PID" 2>/dev/null || true
        wait "$TRACE_POLL_PID" 2>/dev/null || true
    fi
    if [[ "$USE_SERVER" == "1" && "$KEEP_SERVER" == "0" && -n "$SERVER_PID" ]]; then
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            kill "$SERVER_PID" 2>/dev/null || true
            wait "$SERVER_PID" 2>/dev/null || true
        fi
    fi
}
trap cleanup EXIT

ensure_server_built() {
    if [[ -x "$SERVER_BIN" ]]; then return 0; fi
    if [[ -n "$SERVER_BIN_OVERRIDE" ]]; then
        echo "[bench-eli-agent] --server-bin not executable: $SERVER_BIN" >&2
        return 1
    fi
    echo "[bench-eli-agent] building metal_serve release (this can take a while on a cold cache)..."
    (cd "$REPO_ROOT" && cargo build --release -p infer \
        --no-default-features --features metal --bin metal_serve)
    if [[ ! -x "$SERVER_BIN" ]]; then
        echo "[bench-eli-agent] build finished but $SERVER_BIN is still missing" >&2
        return 1
    fi
}

start_infer() {
    if [[ "$USE_SERVER" != "1" ]]; then
        echo "[bench-eli-agent] --no-server: assuming metal_serve is already on :$PORT"
        return 0
    fi
    ensure_server_built
    echo "[bench-eli-agent] booting metal_serve on :$PORT (model=$MODEL_PATH, id=$MODEL_ID)"
    (
        cd "$REPO_ROOT"
        "$SERVER_BIN" \
            --model-path "$MODEL_PATH" \
            --port "$PORT" \
            > "$SERVER_LOG" 2>&1 &
        echo $! > "$OUT_DIR/.server.pid"
    )
    SERVER_PID="$(cat "$OUT_DIR/.server.pid")"
    rm -f "$OUT_DIR/.server.pid"
}

wait_for_server() {
    local deadline=$((SECONDS + WAIT_HEALTH_SECS))
    while (( SECONDS < deadline )); do
        if curl -fsS "http://127.0.0.1:$PORT/v1/stats?format=json" \
            -o "$STATS_BEFORE" 2>/dev/null; then
            echo "[bench-eli-agent] /v1/stats OK after ${SECONDS}s"
            return 0
        fi
        sleep 1
    done
    echo "[bench-eli-agent] timed out waiting for metal_serve on :$PORT" >&2
    if [[ -s "$SERVER_LOG" ]]; then tail -n 40 "$SERVER_LOG" >&2; fi
    return 1
}

start_trace_poll() {
    if [[ "$TRACE_POLL" != "1" ]]; then return 0; fi
    : > "$TRACE_FILE"
    (
        while true; do
            ts="$(date -Iseconds)"
            payload="$(curl -fsS "http://127.0.0.1:$PORT/v1/stats?format=json" 2>/dev/null || echo 'null')"
            printf '{"ts":"%s","stats":%s}\n' "$ts" "$payload" >> "$TRACE_FILE"
            sleep "$TRACE_POLL_INTERVAL"
        done
    ) &
    TRACE_POLL_PID=$!
}

# ---- eli driver -----------------------------------------------------------

ensure_eli_bin() {
    if [[ -n "$ELI_BIN" && -x "$ELI_BIN" ]]; then return 0; fi
    if [[ -x "$ELI_REPO/target/release/eli" ]]; then
        ELI_BIN="$ELI_REPO/target/release/eli"
        return 0
    fi
    if command -v eli >/dev/null 2>&1; then
        ELI_BIN="$(command -v eli)"
        return 0
    fi
    cat >&2 <<EOF
[bench-eli-agent] eli binary not found.
  Tried: \$ELI_BIN, $ELI_REPO/target/release/eli, PATH.
  Build it with: (cd $ELI_REPO && cargo build --release)
  Or pass --eli-bin /path/to/eli.
EOF
    return 1
}

run_eli_mode() {
    ensure_eli_bin
    local eli_home; eli_home="$(mktemp -d -t eli-bench-XXXXXX)"
    echo "[bench-eli-agent] driving eli; ELI_HOME=$eli_home"

    # ELI_MODEL=openai:<id> tells nexil to route through its OpenAI provider
    # and pass <id> in request.model. metal_serve.rs:195 derives the served
    # id from basename(--model-path); openai_v1.rs:60 returns 404 if the
    # request id doesn't match. We pass the resolved $MODEL_ID so the two
    # always agree.
    #
    # ELI_API_FORMAT defaults to "completion" (which on the openai adapter
    # is /v1/chat/completions — see crates/nexil/src/providers/openai.rs).
    # Despite env.example labelling "messages" as chat-completions, the
    # nexil openai adapter rejects the messages transport with
    # "messages format is only valid for Anthropic models".
    export ELI_HOME="$eli_home"
    export ELI_MODEL="${ELI_MODEL:-openai:$MODEL_ID}"
    export ELI_API_BASE="http://127.0.0.1:$PORT/v1"
    export ELI_API_KEY="${ELI_API_KEY:-bench-eli-agent}"
    export ELI_API_FORMAT="${ELI_API_FORMAT:-completion}"
    export ELI_MAX_TOKENS="$MAX_TOKENS"
    export ELI_MAX_STEPS="${ELI_MAX_STEPS:-2}"
    export ELI_EVOLUTION_DISABLED=1

    # Iterate sessions; for each, replay every user message via `eli run`,
    # threading a stable --chat-id and --session-id so nexil keeps the tape
    # and infer keeps slot affinity.
    python3 - "$TRACE_PATH" "$TURNS_JSONL" "$ELI_BIN" <<'PY'
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

trace_path, turns_path, eli_bin = sys.argv[1], sys.argv[2], sys.argv[3]
turns_out = open(turns_path, "a", encoding="utf-8")
sessions = [json.loads(line) for line in Path(trace_path).read_text().splitlines() if line.strip()]
total = 0
ok = 0

for session in sessions:
    sid = session["session_id"]
    user_turns = [t["content"] for t in session["turns"] if t["role"] == "user"]
    for turn_idx, user_msg in enumerate(user_turns):
        cmd = [
            eli_bin, "run",
            "--chat-id", sid,
            "--session-id", sid,
            "--channel", "cli",
            user_msg,
        ]
        t0 = time.perf_counter()
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        wall_ms = (time.perf_counter() - t0) * 1000.0
        total += 1
        # Eli exits 0 even when run_model fails (the pipeline catches the
        # error and renders it as a normal turn output). Detect the
        # failure markers from its stdout so we don't silently book
        # red turns as green.
        haystack = proc.stdout + proc.stderr
        eli_pipeline_failed = (
            "pipeline error" in haystack
            or "run_model failed" in haystack
            or "agent.run finished with error" in haystack
        )
        turn_ok = (proc.returncode == 0) and not eli_pipeline_failed
        if turn_ok:
            ok += 1
        record = {
            "session_id": sid,
            "turn_index": turn_idx,
            "wall_ms": round(wall_ms, 3),
            "exit_code": proc.returncode,
            "ok": turn_ok,
            "eli_pipeline_failed": eli_pipeline_failed,
            "stdout_bytes": len(proc.stdout.encode("utf-8")),
            "stderr_bytes": len(proc.stderr.encode("utf-8")),
            "stdout_first_chars": proc.stdout[:160],
            "stderr_first_chars": proc.stderr[:160],
            "command": " ".join(shlex.quote(c) for c in cmd),
        }
        turns_out.write(json.dumps(record) + "\n")
        turns_out.flush()
        marker = "OK" if turn_ok else "FAIL"
        print(
            f"  [{sid} turn {turn_idx}] wall={wall_ms:.1f}ms exit={proc.returncode} {marker}",
            flush=True,
        )

turns_out.close()
print(f"[bench-eli-agent] eli driver complete: {ok}/{total} turns OK")
sys.exit(0 if ok == total else 1)
PY
}

# ---- openai-direct driver -------------------------------------------------

run_openai_direct_mode() {
    echo "[bench-eli-agent] delegating to bench_agent_trace.py (--workload trace)"
    PYTHONUNBUFFERED=1 python3 "$REPO_ROOT/scripts/bench_agent_trace.py" \
        --workload trace \
        --trace "$TRACE_PATH" \
        --server "http://127.0.0.1:$PORT" \
        --label "$LABEL" \
        --max-tokens "$MAX_TOKENS" \
        --num-concurrent 4 \
        --out "$OUT_DIR/bench_agent_trace_summary.json" \
        --trace-out "$OUT_DIR/bench_agent_trace_replay.jsonl" \
        --no-probe-stats \
        2>&1 | tee "$OUT_DIR/bench_agent_trace.log"
}

# ---- Metric extraction ----------------------------------------------------

snapshot_after() {
    curl -fsS "http://127.0.0.1:$PORT/v1/stats?format=json" -o "$STATS_AFTER" || true
}

write_results() {
    local end_epoch_ms; end_epoch_ms="$(now_ms)"
    local elapsed_ms=$(( end_epoch_ms - START_EPOCH_MS ))
    python3 - \
        "$LABEL" "$MODE" "$elapsed_ms" "$TURNS_JSONL" \
        "$OUT_DIR/bench_agent_trace_summary.json" \
        "$RESULTS_JSON" "$HEADLINE_MD" <<'PY'
import json
import statistics
import sys
from pathlib import Path

label, mode, elapsed_ms, turns_path, fallback_summary, results_path, headline_path = sys.argv[1:]
elapsed_ms = float(elapsed_ms)

walls = []
ok = 0
total = 0
sessions_seen = set()
turns_path = Path(turns_path)
if turns_path.exists():
    for line in turns_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        total += 1
        sessions_seen.add(rec.get("session_id"))
        # Prefer the explicit "ok" field (added 2026-05-03 to catch Eli
        # turns that exit 0 with a render_model pipeline failure). Fall
        # back to exit_code for the openai-direct mode and for old
        # turns.jsonl files written before the field existed.
        if "ok" in rec:
            if rec["ok"]:
                ok += 1
        elif rec.get("exit_code") == 0:
            ok += 1
        if isinstance(rec.get("wall_ms"), (int, float)):
            walls.append(float(rec["wall_ms"]))

def pct(values, q):
    if not values:
        return None
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return s[idx]

# Optional fallback metrics from bench_agent_trace.py.
fallback = None
fp = Path(fallback_summary)
if fp.exists():
    try:
        fallback = json.loads(fp.read_text())
    except Exception:
        fallback = None

results = {
    "label": label,
    "mode": mode,
    "elapsed_ms": elapsed_ms,
    "turns_total": total,
    "turns_ok": ok,
    "sessions_total": len(sessions_seen),
    "client_wall_ms": {
        "p50": pct(walls, 0.50),
        "p90": pct(walls, 0.90),
        "p99": pct(walls, 0.99),
        "mean": (statistics.fmean(walls) if walls else None),
        "max": (max(walls) if walls else None),
    },
    "throughput_req_per_s": (ok / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else None,
    "fallback_bench_agent_trace": fallback,
}

Path(results_path).write_text(json.dumps(results, indent=2))

def fmt(v, suffix=""):
    if v is None:
        return "n/a"
    return f"{v:.2f}{suffix}"

lines = [
    f"# bench-eli-agent {label} ({mode})",
    "",
    f"- elapsed wall-clock: {fmt(elapsed_ms, ' ms')}",
    f"- turns OK / total: {ok} / {total}",
    f"- sessions: {len(sessions_seen)}",
    "",
    "## Client wall-clock per turn",
    f"- p50: {fmt(results['client_wall_ms']['p50'], ' ms')}",
    f"- p90: {fmt(results['client_wall_ms']['p90'], ' ms')}",
    f"- p99: {fmt(results['client_wall_ms']['p99'], ' ms')}",
    f"- max: {fmt(results['client_wall_ms']['max'], ' ms')}",
    f"- req/s: {fmt(results['throughput_req_per_s'])}",
    "",
]
if fallback is not None:
    lines.append("## OpenAI-direct fallback (bench_agent_trace.py)")
    lines.append(f"- snapshot: {Path(fallback_summary).name}")
    lines.append("")

Path(headline_path).write_text("\n".join(lines))
print(Path(headline_path).read_text())
PY
}

# ---- Main -----------------------------------------------------------------

start_infer
wait_for_server
start_trace_poll

case "$MODE" in
    eli)            run_eli_mode;;
    openai-direct)  run_openai_direct_mode;;
esac

snapshot_after
write_results

echo "[bench-eli-agent] done. results: $RESULTS_JSON"
