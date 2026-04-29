#!/usr/bin/env bash
# Canonical throughput/latency bench wrapper around vllm-project/guidellm.
#
# Usage:
#   ./scripts/bench_guidellm.sh <backend-label> [--target URL] [--model NAME]
#
# Required:
#   <backend-label>  e.g. cuda-h100, cuda-a100, metal-m3max
#                    used to name the output dir and wins file
#
# Optional flags (canonical run, produces a wins entry):
#   --target URL     inference server URL   (default: http://localhost:8000)
#   --model  NAME    model identifier       (default: Qwen/Qwen3-4B)
#   --processor PATH tokenizer path / HF id (default: local models/Qwen3-4B)
#   --trace-interval-ms N
#                    service trace polling interval for /v1/stats
#                    (default: 1000ms)
#
# Exploration mode (faster, non-canonical; DOES NOT produce a wins entry):
#   --fast               short c=16 preset: profile=concurrent, rate=16,
#                        data=4096-in/256-out, max-seconds=30.
#   --quick              ~4-minute matched-A/B preset: profile=concurrent,
#                        rate=1,2,4,8, data=512-in/128-out, max-seconds=60,
#                        warmup=5. Short dataset so requests complete in
#                        seconds on 4–8B models.
#   --concurrencies L    comma-separated concurrency list, e.g. "1,2,4,8".
#                        Switches profile to `concurrent`.
#   --profile TYPE       override profile (sweep|concurrent|synchronous|…).
#   --max-seconds N      override per-benchmark duration.
#   --warmup N           run guidellm's warmup phase (seconds or 0 < f < 1).
#
# Any of those overrides flips the run to exploration mode: raw artefacts
# still land under bench-output/, but no wins entry is seeded. Keep the
# wins pipeline reserved for canonical measurements.
#
# Preconditions:
#   * guidellm, curl, jq on PATH
#   * infer HTTP server is already running at --target
#     (start it with: scripts/start_infer.sh)
#
# Side effects:
#   * Writes raw artefacts to bench-output/<date>-<label>[-runN]/
#     (benchmarks.json / .csv / .html, plus guidellm.log and command.txt).
#   * Always writes service-side trace artefacts in the same output dir:
#     service_stats_before.txt / service_stats_after.txt /
#     service_stats_trace.jsonl / service_stats_trace_summary.md.
#     This dir is gitignored.
#   * Canonical mode only: seeds a new
#     docs/experience/wins/<date>-bench-guidellm-<label>.md from the
#     template with the commit sha, paths, and best-effort headline table.
#
# The canonical benchmark parameters are LOCKED here. Changing them is a
# deliberate commit, not a flag flip. See docs/plans/guidellm-integration.md §3.

set -euo pipefail

# ---- Canonical params (locked, see docs/plans/guidellm-integration.md §3) ----
PROFILE="sweep"
# guidellm 0.6.0's synthetic generator defaults to a wide normal
# distribution around `prompt_tokens` (saw min=22826, max=23133 with
# target mean=4096, stdev unset). When the server's --max-seq-len is
# tighter than the upper tail, the bench rejects most requests and the
# numbers are meaningless. Clamp stdev=0 + min=max=mean — apples-to-apples
# vs the wins-doc baselines, matches the bench_matrix.py fix landed
# 2026-04-28 (memory: project_bench_env_drift_2026-04-20).
DATA="prompt_tokens=4096,prompt_tokens_stdev=1,prompt_tokens_min=4096,prompt_tokens_max=4096,output_tokens=256,output_tokens_stdev=1,output_tokens_min=256,output_tokens_max=256"
MAX_SECONDS=60
RANDOM_SEED=20260416
OUTPUTS=(json csv html)
# **Pin the HTTP backend explicitly.** guidellm 0.6.0's default backend is
# `vllm_python` (an in-process vLLM import) which silently reports 0
# successful requests against our infer HTTP server and then crashes the
# sweep profile with "Invalid rates in sweep; aborting". We speak the
# OpenAI v1 HTTP API, so `openai_http` is the correct backend to use.
BACKEND="openai_http"
# guidellm's default backend validation probes GET /health, which
# metal_serve / cuda-infer do not expose. Point it at /v1/models instead
# (we already rely on that route being present in preflight below).
#
# Also pin the benchmark path to /v1/completions. The chat endpoint starts
# with a role-only delta that GuideLLM ignores for TTFT, so relying on its
# implicit request-format selection makes TTFT/ITL collection brittle.
BACKEND_KWARGS='{"validate_backend": "/v1/models", "request_format": "/v1/completions"}'
# ------------------------------------------------------------------------------

TARGET="http://localhost:8000"
MODEL="Qwen/Qwen3-4B"
# Local path used for tokenizer lookup during synthetic prompt generation.
# If the HF name isn't in the local HF cache, the synthetic_text dataset
# deserializer can't download it in sandboxed environments and bails with
# "OSError: Qwen3-4B is not a local folder". Defaults to a weights dir
# that already exists on CUDA and Metal bring-up boxes.
PROCESSOR_DEFAULT="infer/models/Qwen3-4B"
PROCESSOR=""
LABEL=""
# Exploration-mode overrides. Empty = use the canonical value above.
RATE_OVERRIDE=""
WARMUP_OVERRIDE=""
EXPLORATION_MODE=false
TRACE_INTERVAL_MS=1000

usage() {
    cat <<EOF
usage: $(basename "$0") <backend-label> [options]

  <backend-label>        required, e.g. cuda-h100, metal-m3max

Canonical run (produces a wins entry):
  --target URL           default: $TARGET
  --model NAME           default: $MODEL
  --processor PATH       tokenizer path / HF id (default: local $PROCESSOR_DEFAULT)
  --trace-interval-ms N  /v1/stats polling interval (default: $TRACE_INTERVAL_MS)

Exploration mode (faster, no wins entry):
  --fast                 short c=16 preset: profile=concurrent, rate=16,
                         data=4096-in/256-out, max-seconds=30
  --quick                 ~4-min preset: profile=concurrent rate=1,2,4,8
                          data=512-in/128-out max-seconds=60 warmup=5
  --concurrencies LIST    e.g. "1,2,4,8" (switches profile to concurrent)
  --profile TYPE          sweep|concurrent|synchronous|throughput|…
  --max-seconds N         override per-benchmark duration
  --warmup N              seconds (int >= 1) or fraction (0 < f < 1)

See docs/plans/guidellm-integration.md for the canonical parameters and
why this wrapper exists.
EOF
}

# ---- arg parsing -------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --target)
            [[ $# -ge 2 ]] || { echo "error: --target requires a value" >&2; exit 2; }
            TARGET="$2"; shift 2 ;;
        --model)
            [[ $# -ge 2 ]] || { echo "error: --model requires a value" >&2; exit 2; }
            MODEL="$2"; shift 2 ;;
        --fast)
            EXPLORATION_MODE=true
            PROFILE="concurrent"
            RATE_OVERRIDE="16"
            MAX_SECONDS=30
            shift ;;
        --processor)
            [[ $# -ge 2 ]] || { echo "error: --processor requires a value" >&2; exit 2; }
            PROCESSOR="$2"; shift 2 ;;
        --quick)
            # Exploration preset: short dataset so requests finish in
            # seconds even on 4–8B models; 60s window lets each stream
            # complete multiple requests per concurrency level.
            EXPLORATION_MODE=true
            PROFILE="concurrent"
            RATE_OVERRIDE="1,2,4,8"
            DATA="prompt_tokens=512,prompt_tokens_stdev=1,prompt_tokens_min=512,prompt_tokens_max=512,output_tokens=128,output_tokens_stdev=1,output_tokens_min=128,output_tokens_max=128"
            MAX_SECONDS=60
            WARMUP_OVERRIDE="5"
            shift ;;
        --concurrencies)
            [[ $# -ge 2 ]] || { echo "error: --concurrencies requires a value" >&2; exit 2; }
            EXPLORATION_MODE=true
            PROFILE="concurrent"
            RATE_OVERRIDE="$2"
            shift 2 ;;
        --profile)
            [[ $# -ge 2 ]] || { echo "error: --profile requires a value" >&2; exit 2; }
            EXPLORATION_MODE=true
            PROFILE="$2"; shift 2 ;;
        --max-seconds)
            [[ $# -ge 2 ]] || { echo "error: --max-seconds requires a value" >&2; exit 2; }
            EXPLORATION_MODE=true
            MAX_SECONDS="$2"; shift 2 ;;
        --warmup)
            [[ $# -ge 2 ]] || { echo "error: --warmup requires a value" >&2; exit 2; }
            EXPLORATION_MODE=true
            WARMUP_OVERRIDE="$2"; shift 2 ;;
        --trace-interval-ms)
            [[ $# -ge 2 ]] || { echo "error: --trace-interval-ms requires a value" >&2; exit 2; }
            TRACE_INTERVAL_MS="$2"; shift 2 ;;
        -h|--help)
            usage; exit 0 ;;
        --*)
            echo "error: unknown flag: $1" >&2
            usage >&2
            exit 2 ;;
        *)
            if [[ -z "$LABEL" ]]; then
                LABEL="$1"; shift
            else
                echo "error: unexpected positional arg: $1" >&2
                usage >&2
                exit 2
            fi
            ;;
    esac
done

if [[ -z "$LABEL" ]]; then
    echo "error: <backend-label> is required" >&2
    usage >&2
    exit 2
fi
if ! [[ "$TRACE_INTERVAL_MS" =~ ^[0-9]+$ ]] || [[ "$TRACE_INTERVAL_MS" -le 0 ]]; then
    echo "error: --trace-interval-ms must be a positive integer, got: $TRACE_INTERVAL_MS" >&2
    exit 2
fi

# ---- preflight: required tools on PATH ---------------------------------------
if ! command -v guidellm >/dev/null 2>&1; then
    echo "error: guidellm not on PATH — run: pip install -e .[bench]" >&2
    exit 2
fi
if ! command -v jq >/dev/null 2>&1; then
    echo "error: jq not on PATH — install jq (brew install jq / apt install jq)" >&2
    exit 2
fi
if ! command -v curl >/dev/null 2>&1; then
    echo "error: curl not on PATH" >&2
    exit 2
fi

# ---- preflight: server is up -------------------------------------------------
if ! curl -sS -f "$TARGET/v1/models" >/dev/null 2>&1; then
    echo "error: server not running at $TARGET — start it with scripts/start_infer.sh first" >&2
    exit 2
fi

probe_streaming_completions() {
    python3 - "$TARGET" "$MODEL" <<'PY'
import json
import sys
from urllib import error, request

target, model = sys.argv[1:]
payload = json.dumps(
    {
        "model": model,
        "prompt": "Hello world",
        "max_tokens": 8,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {
            "include_usage": True,
            "continuous_usage_stats": True,
        },
    }
).encode("utf-8")
req = request.Request(
    f"{target}/v1/completions",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    with request.urlopen(req, timeout=30) as resp:
        saw_text = False
        saw_usage = False
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            if line == "data: [DONE]":
                break
            if not line.startswith("data:"):
                continue
            data = json.loads(line[len("data:") :].strip())
            choice = (data.get("choices") or [{}])[0]
            text = choice.get("text")
            if isinstance(text, str) and text != "":
                saw_text = True
            usage = data.get("usage")
            if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
                saw_usage = True
            if saw_text and saw_usage:
                break
except error.HTTPError as exc:
    print(f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}", file=sys.stderr)
    sys.exit(1)

if not saw_text:
    print(
        "streaming completions probe failed: no non-empty text chunk arrived from /v1/completions",
        file=sys.stderr,
    )
    sys.exit(2)
if not saw_usage:
    print(
        "streaming completions probe failed: terminal usage chunk missing from /v1/completions",
        file=sys.stderr,
    )
    sys.exit(3)
PY
}

validate_guidellm_results() {
    python3 - "$1" <<'PY'
import json
import pathlib
import sys

json_path = pathlib.Path(sys.argv[1])
obj = json.loads(json_path.read_text())
benchmarks = obj.get("benchmarks") or []
if not benchmarks:
    print("guidellm validation failed: no benchmarks found in benchmarks.json", file=sys.stderr)
    sys.exit(1)

errors = []
for bm in benchmarks:
    strategy = bm.get("config", {}).get("strategy", {})
    rate = strategy.get("type_", "unknown")
    if rate == "concurrent":
        rate = f"conc{strategy.get('max_concurrency', '?')}"

    metrics = bm.get("metrics", {})
    request_totals = metrics.get("request_totals", {})
    successful = int(request_totals.get("successful") or 0)
    output_mean = (metrics.get("output_token_count", {}).get("successful", {}) or {}).get("mean") or 0.0
    ttft_p50 = (metrics.get("time_to_first_token_ms", {}).get("successful", {}).get("percentiles", {}) or {}).get("p50")
    itl_p50 = (metrics.get("inter_token_latency_ms", {}).get("successful", {}).get("percentiles", {}) or {}).get("p50")
    outputs = bm.get("requests", {}).get("successful", []) or []
    nonempty_outputs = sum(1 for req in outputs if (req.get("output") or "") != "")

    if successful <= 0:
        errors.append(f"{rate}: no successful requests recorded")
        continue

    if output_mean > 0.0 and nonempty_outputs == 0:
        errors.append(
            f"{rate}: successful requests reported {output_mean:.1f} output tokens on average but every sampled output was empty"
        )
    if output_mean > 0.0 and (ttft_p50 is None or ttft_p50 <= 0.0):
        errors.append(
            f"{rate}: TTFT p50 was {ttft_p50!r} despite successful requests with non-zero output tokens"
        )
    if output_mean > 1.0 and (itl_p50 is None or itl_p50 <= 0.0):
        errors.append(
            f"{rate}: ITL p50 was {itl_p50!r} despite successful requests averaging more than one output token"
        )

if errors:
    print("guidellm validation failed:", file=sys.stderr)
    for line in errors:
        print(f"  - {line}", file=sys.stderr)
    sys.exit(4)
PY
}

acquire_bench_lock() {
    local lock_file="$REPO_ROOT/bench-output/.bench_guidellm.lock"
    mkdir -p "$REPO_ROOT/bench-output"
    if command -v flock >/dev/null 2>&1; then
        exec 9>"$lock_file"
        if ! flock -n 9; then
            echo "error: another scripts/bench_guidellm.sh run is active (lock: $lock_file)" >&2
            echo "       bench runs are forced serial to keep service trace trustworthy." >&2
            exit 2
        fi
    else
        echo "warning: flock is not installed; serial bench lock is not enforced" >&2
    fi
}

capture_service_stats_snapshot() {
    local out_file="$1"
    if curl -sS --max-time 5 "$TARGET/v1/stats" > "$out_file"; then
        return 0
    fi
    echo "<unavailable>" > "$out_file"
    return 1
}

start_service_stats_trace() {
    local trace_file="$1"
    local interval_ms="$2"
    local interval_s
    interval_s="$(python3 - "$interval_ms" <<'PY'
import sys
ms = int(sys.argv[1])
print(f"{ms / 1000.0:.3f}")
PY
)"
    (
        set +e
        while true; do
            ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            stats="$(curl -sS --max-time 3 "$TARGET/v1/stats" 2>&1)"
            rc=$?
            if [[ $rc -eq 0 ]]; then
                printf '{"ts":"%s","ok":true,"stats":%s}\n' \
                    "$ts" "$(printf '%s' "$stats" | jq -Rs .)"
            else
                printf '{"ts":"%s","ok":false,"error":%s}\n' \
                    "$ts" "$(printf '%s' "$stats" | jq -Rs .)"
            fi
            sleep "$interval_s"
        done
    ) > "$trace_file" &
    SERVICE_TRACE_PID=$!
}

stop_service_stats_trace() {
    if [[ -n "${SERVICE_TRACE_PID:-}" ]]; then
        if kill -0 "$SERVICE_TRACE_PID" >/dev/null 2>&1; then
            kill "$SERVICE_TRACE_PID" >/dev/null 2>&1 || true
            wait "$SERVICE_TRACE_PID" >/dev/null 2>&1 || true
        fi
        SERVICE_TRACE_PID=""
    fi
}

write_service_stats_trace_summary() {
    local trace_file="$1"
    local before_file="$2"
    local after_file="$3"
    local summary_file="$4"
    local interval_ms="$5"
    python3 - "$trace_file" "$before_file" "$after_file" "$summary_file" "$interval_ms" <<'PY'
import json
import pathlib
import re
import sys

trace_path, before_path, after_path, out_path, interval_ms = sys.argv[1:]
trace_file = pathlib.Path(trace_path)
before = pathlib.Path(before_path).read_text().strip()
after = pathlib.Path(after_path).read_text().strip()
records = []
if trace_file.exists():
    for line in trace_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass

token_re = re.compile(r"([^=\s]+)=([^\s]+)")

def parse_fields(raw: str):
    return dict(token_re.findall(raw))

def parse_int(fields, key):
    val = fields.get(key)
    if val is None:
        return None
    val = val.rstrip("%")
    try:
        return int(float(val))
    except ValueError:
        return None

def parse_float(fields, key):
    val = fields.get(key)
    if val is None:
        return None
    val = val.rstrip("%")
    try:
        return float(val)
    except ValueError:
        return None

def quantile(vals, q):
    if not vals:
        return None
    ordered = sorted(vals)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac

def fmt_num(val, digits=0, suffix=""):
    if val is None:
        return "n/a"
    if digits == 0:
        return f"{val:.0f}{suffix}"
    return f"{val:.{digits}f}{suffix}"

def fmt_peak(vals, digits=0, suffix=""):
    return fmt_num(max(vals) if vals else None, digits, suffix)

def distribution_row(name, vals, digits=0, suffix=""):
    return (
        f"| {name} | {fmt_num(quantile(vals, 0.25), digits, suffix)} "
        f"| {fmt_num(quantile(vals, 0.50), digits, suffix)} "
        f"| {fmt_num(quantile(vals, 0.75), digits, suffix)} "
        f"| {fmt_num(quantile(vals, 0.99), digits, suffix)} "
        f"| {fmt_peak(vals, digits, suffix)} |"
    )

ok_records = [r for r in records if r.get("ok") is True and isinstance(r.get("stats"), str)]
fail_records = [r for r in records if r.get("ok") is False]

parsed = [parse_fields(r["stats"]) for r in ok_records]
waiting_vals = [v for f in parsed if (v := parse_int(f, "waiting")) is not None]
active_vals = [v for f in parsed if (v := parse_int(f, "active")) is not None]
running_batch_vals = [v for f in parsed if (v := parse_int(f, "running_batch")) is not None]
prefill_queue_vals = [v for f in parsed if (v := parse_int(f, "prefill_queue")) is not None]
kv_vals = [v for f in parsed if (v := parse_float(f, "kv_util")) is not None]
prefix_hit_vals = [v for f in parsed if (v := parse_float(f, "prefix_hit_rate")) is not None]
prefix_skip_vals = [v for f in parsed if (v := parse_float(f, "prefix_skip_rate")) is not None]
peak_mem_vals = [v for f in parsed if (v := parse_float(f, "peak_mem")) is not None]
active_mem_vals = [v for f in parsed if (v := parse_float(f, "active_mem")) is not None]
cache_mem_vals = [v for f in parsed if (v := parse_float(f, "cache_mem")) is not None]
queue_p50_vals = [v for f in parsed if (v := parse_float(f, "queue_p50")) is not None]
ttft_p50_vals = [v for f in parsed if (v := parse_float(f, "ttft_p50")) is not None]
ttft_p99_vals = [v for f in parsed if (v := parse_float(f, "ttft_p99")) is not None]
tpot_p50_vals = [v for f in parsed if (v := parse_float(f, "tpot_p50")) is not None]
service_p50_vals = [v for f in parsed if (v := parse_float(f, "service_p50")) is not None]
step_last_vals = [v for f in parsed if (v := parse_float(f, "step_last")) is not None]
step_p50_vals = [v for f in parsed if (v := parse_float(f, "step_p50")) is not None]
kv_fetch_q_vals = [v for f in parsed if (v := parse_int(f, "kv_fetch_q")) is not None]
kv_fetch_waiter_vals = [v for f in parsed if (v := parse_int(f, "kv_fetch_waiters")) is not None]
kv_store_q_vals = [v for f in parsed if (v := parse_int(f, "kv_store_q")) is not None]
tier_fetch_wait_vals = [v for f in parsed if (v := parse_float(f, "tier_fetch_wait")) is not None]
tier_store_wait_vals = [v for f in parsed if (v := parse_float(f, "tier_store_wait")) is not None]
decode_token_vals = [v for f in parsed if (v := parse_int(f, "decode_tokens")) is not None]
prefill_token_vals = [v for f in parsed if (v := parse_int(f, "prefill_tokens")) is not None]
tokens_out_vals = [v for f in parsed if (v := parse_int(f, "tokens_out")) is not None]

peak_waiting = max(waiting_vals) if waiting_vals else None
peak_active = max(active_vals) if active_vals else None
peak_running_batch = max(running_batch_vals) if running_batch_vals else None
peak_prefill_queue = max(prefill_queue_vals) if prefill_queue_vals else None
peak_kv = max(kv_vals) if kv_vals else None
peak_prefix_hit = max(prefix_hit_vals) if prefix_hit_vals else None
q75_prefix_hit = quantile(prefix_hit_vals, 0.75)
peak_prefix_skip = max(prefix_skip_vals) if prefix_skip_vals else None
peak_mem = max(peak_mem_vals) if peak_mem_vals else None
before_peak_mem = parse_float(parse_fields(before), "peak_mem")
peak_mem_delta = (peak_mem - before_peak_mem) if peak_mem is not None and before_peak_mem is not None else None
kv_fetch_q_saturated = sum(1 for v in kv_fetch_q_vals if v > 0)
kv_fetch_waiter_saturated = sum(1 for v in kv_fetch_waiter_vals if v > 0)
kv_store_q_saturated = sum(1 for v in kv_store_q_vals if v > 0)
service_distribution_rows = [
    distribution_row("waiting", waiting_vals),
    distribution_row("kv_util", kv_vals, 1, "%"),
    distribution_row("queue_p50", queue_p50_vals, 1),
    distribution_row("ttft_p50", ttft_p50_vals, 1),
    distribution_row("ttft_p99", ttft_p99_vals, 1),
    distribution_row("tpot_p50", tpot_p50_vals, 1),
    distribution_row("service_p50", service_p50_vals, 1),
    distribution_row("step_last", step_last_vals, 1),
    distribution_row("step_p50", step_p50_vals, 1),
    distribution_row("active_mem", active_mem_vals, 1),
    distribution_row("cache_mem", cache_mem_vals, 1),
]
service_distribution_rows = [row for row in service_distribution_rows if "n/a | n/a | n/a | n/a | n/a" not in row]
token_distribution_rows = [
    distribution_row("decode_tokens", decode_token_vals),
    distribution_row("prefill_tokens", prefill_token_vals),
    distribution_row("tokens_out", tokens_out_vals),
]
token_distribution_rows = [row for row in token_distribution_rows if "n/a | n/a | n/a | n/a | n/a" not in row]

lines = [
    "# Service Trace Summary",
    "",
    f"- Poll interval: `{interval_ms}ms`",
    f"- Samples: `{len(records)}` (ok: `{len(ok_records)}`, failed: `{len(fail_records)}`)",
    f"- Peak waiting: `{peak_waiting if peak_waiting is not None else 'n/a'}`",
    f"- Peak active: `{peak_active if peak_active is not None else 'n/a'}`",
    f"- Peak running_batch: `{peak_running_batch if peak_running_batch is not None else 'n/a'}`",
    f"- Peak prefill_queue: `{peak_prefill_queue if peak_prefill_queue is not None else 'n/a'}`",
    f"- Peak kv_util: `{f'{peak_kv:.1f}%' if peak_kv is not None else 'n/a'}`",
    f"- Prefix hit rate: peak `{fmt_num(peak_prefix_hit, 1, '%')}`, q75 `{fmt_num(q75_prefix_hit, 1, '%')}`",
    f"- Prefix skip rate peak: `{fmt_num(peak_prefix_skip, 1, '%')}`",
    f"- Peak mem: `{fmt_num(peak_mem, 1)}` (delta vs before: `{fmt_num(peak_mem_delta, 1)}`)",
    f"- Server ttft_p99 peak: `{fmt_peak(ttft_p99_vals, 1)}`",
    f"- KV fetch queue samples >0: `{kv_fetch_q_saturated}/{len(kv_fetch_q_vals)}`",
    f"- KV fetch waiter samples >0: `{kv_fetch_waiter_saturated}/{len(kv_fetch_waiter_vals)}`",
    f"- KV store queue samples >0: `{kv_store_q_saturated}/{len(kv_store_q_vals)}`",
    f"- Tier wait peaks: fetch `{fmt_peak(tier_fetch_wait_vals, 1)}`, store `{fmt_peak(tier_store_wait_vals, 1)}`",
    "",
    "## Trace Distributions",
    "",
    "| metric | q25 | q50 | q75 | q99 | peak |",
    "|---|---:|---:|---:|---:|---:|",
    *(service_distribution_rows or ["| n/a | n/a | n/a | n/a | n/a | n/a |"]),
    "",
    "## Token Counters",
    "",
    "| metric | q25 | q50 | q75 | q99 | peak |",
    "|---|---:|---:|---:|---:|---:|",
    *(token_distribution_rows or ["| n/a | n/a | n/a | n/a | n/a | n/a |"]),
    "",
    "## Before",
    "",
    "```text",
    before or "<empty>",
    "```",
    "",
    "## After",
    "",
    "```text",
    after or "<empty>",
    "```",
]
pathlib.Path(out_path).write_text("\n".join(lines) + "\n")
PY
}

# ---- resolve output paths, never overwrite -----------------------------------
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
acquire_bench_lock
DATE="$(date +%Y-%m-%d)"
SERVICE_TRACE_PID=""

cleanup() {
    stop_service_stats_trace
}
trap cleanup EXIT INT TERM

base_dir="$REPO_ROOT/bench-output/${DATE}-${LABEL}"
OUTPUT_DIR="$base_dir"
run=1
while [[ -e "$OUTPUT_DIR" ]]; do
    run=$((run + 1))
    OUTPUT_DIR="${base_dir}-run${run}"
done
mkdir -p "$OUTPUT_DIR"
GUIDELLM_LOG="$OUTPUT_DIR/guidellm.log"
GUIDELLM_CMD="$OUTPUT_DIR/command.txt"
SERVICE_STATS_BEFORE="$OUTPUT_DIR/service_stats_before.txt"
SERVICE_STATS_AFTER="$OUTPUT_DIR/service_stats_after.txt"
SERVICE_STATS_TRACE="$OUTPUT_DIR/service_stats_trace.jsonl"
SERVICE_STATS_SUMMARY="$OUTPUT_DIR/service_stats_trace_summary.md"

wins_dir="$REPO_ROOT/docs/experience/wins"
wins_base="$wins_dir/${DATE}-bench-guidellm-${LABEL}"
WINS_FILE="${wins_base}.md"
wrun=1
while [[ -e "$WINS_FILE" ]]; do
    wrun=$((wrun + 1))
    WINS_FILE="${wins_base}-run${wrun}.md"
done
TEMPLATE_FILE="$wins_dir/TEMPLATE-bench-guidellm.md"
if [[ ! -f "$TEMPLATE_FILE" ]]; then
    echo "error: missing template: $TEMPLATE_FILE" >&2
    exit 2
fi

COMMIT_SHA="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"

# ---- run guidellm ------------------------------------------------------------
echo ">>> guidellm benchmark"
echo "    target : $TARGET"
echo "    model  : $MODEL"
echo "    label  : $LABEL"
echo "    profile: $PROFILE"
echo "    data   : $DATA"
echo "    seconds: $MAX_SECONDS"
echo "    seed   : $RANDOM_SEED"
if [[ -n "$RATE_OVERRIDE" ]]; then
    echo "    rate   : $RATE_OVERRIDE"
fi
if [[ -n "$WARMUP_OVERRIDE" ]]; then
    echo "    warmup : $WARMUP_OVERRIDE"
fi
if [[ "$EXPLORATION_MODE" == true ]]; then
    echo "    mode   : exploration (no wins entry)"
else
    echo "    mode   : canonical"
fi
echo "    output : $OUTPUT_DIR"
echo "    formats: ${OUTPUTS[*]}"
echo "    log    : $GUIDELLM_LOG"
echo "    trace  : $SERVICE_STATS_TRACE (interval=${TRACE_INTERVAL_MS}ms)"
echo

# Tokenizer source: explicit --processor wins, else local path if it
# exists, else fall back to the HF model name (network required).
if [[ -z "$PROCESSOR" ]]; then
    if [[ -d "$REPO_ROOT/$PROCESSOR_DEFAULT" ]]; then
        PROCESSOR="$REPO_ROOT/$PROCESSOR_DEFAULT"
    else
        PROCESSOR="$MODEL"
    fi
fi
echo "    processor: $PROCESSOR"

if ! probe_streaming_completions; then
    echo "error: streaming probe failed before benchmark start" >&2
    echo "       target: $TARGET/v1/completions" >&2
    exit 2
fi
if ! capture_service_stats_snapshot "$SERVICE_STATS_BEFORE"; then
    echo "warning: failed to read pre-bench service stats from $TARGET/v1/stats" >&2
fi
start_service_stats_trace "$SERVICE_STATS_TRACE" "$TRACE_INTERVAL_MS"

# guidellm 0.6.0 hangs at "Setup complete, starting benchmarks..." on macOS
# under the default `fork` mp context (Python 3.11+ deprecates fork on darwin
# and the worker_group spawn deadlocks). `forkserver` boots cleanly. The
# guidellm pin lives in requirements-bench.txt, installed by setup.sh.
export GUIDELLM__MP_CONTEXT_TYPE="${GUIDELLM__MP_CONTEXT_TYPE:-forkserver}"

GUIDELLM_ARGS=(
    --target "$TARGET"
    --model "$MODEL"
    --processor "$PROCESSOR"
    --profile "$PROFILE"
    --data "$DATA"
    --max-seconds "$MAX_SECONDS"
    --random-seed "$RANDOM_SEED"
    --output-dir "$OUTPUT_DIR"
    --backend "$BACKEND"
    --backend-kwargs "$BACKEND_KWARGS"
    --disable-console-interactive
)
for output in "${OUTPUTS[@]}"; do
    GUIDELLM_ARGS+=(--outputs "$output")
done
if [[ -n "$RATE_OVERRIDE" ]]; then
    GUIDELLM_ARGS+=(--rate "$RATE_OVERRIDE")
fi
if [[ -n "$WARMUP_OVERRIDE" ]]; then
    GUIDELLM_ARGS+=(--warmup "$WARMUP_OVERRIDE")
fi

{
    echo "GUIDELLM__MP_CONTEXT_TYPE=${GUIDELLM__MP_CONTEXT_TYPE:-forkserver}"
    printf 'guidellm benchmark run'
    for arg in "${GUIDELLM_ARGS[@]}"; do
        printf ' %q' "$arg"
    done
    printf '\n'
} > "$GUIDELLM_CMD"

set +e
guidellm benchmark run "${GUIDELLM_ARGS[@]}" 2>&1 | tee "$GUIDELLM_LOG"
gdl_rc=${PIPESTATUS[0]}
set -e

stop_service_stats_trace
if ! capture_service_stats_snapshot "$SERVICE_STATS_AFTER"; then
    echo "warning: failed to read post-bench service stats from $TARGET/v1/stats" >&2
fi
write_service_stats_trace_summary \
    "$SERVICE_STATS_TRACE" \
    "$SERVICE_STATS_BEFORE" \
    "$SERVICE_STATS_AFTER" \
    "$SERVICE_STATS_SUMMARY" \
    "$TRACE_INTERVAL_MS"

if [[ $gdl_rc -ne 0 ]]; then
    echo "error: guidellm exited with status $gdl_rc" >&2
    echo "       raw artefacts (if any): $OUTPUT_DIR" >&2
    echo "       full log: $GUIDELLM_LOG" >&2
    echo "       service trace: $SERVICE_STATS_SUMMARY" >&2
    exit 3
fi

if ! validate_guidellm_results "$OUTPUT_DIR/benchmarks.json"; then
    echo "error: guidellm wrote benchmark files, but the result set is invalid" >&2
    echo "       raw artefacts: $OUTPUT_DIR" >&2
    echo "       full log: $GUIDELLM_LOG" >&2
    echo "       service trace: $SERVICE_STATS_SUMMARY" >&2
    exit 4
fi

# ---- metric extraction (schema pinned to guidellm 0.6.x) ---------------------
# Verified 2026-04-15 against Qwen3-0.6B on Metal. Path layout:
#   .benchmarks[n].metrics.<metric>.successful.{mean,std_dev,max,total_sum,percentiles.*}
#   .benchmarks[n].config.strategy.{type_,max_concurrency}
# Metrics used: time_to_first_token_ms, inter_token_latency_ms,
#               time_per_output_token_ms, request_latency, request_concurrency,
#               prompt/output/total tokens_per_second, requests_per_second,
#               prompt_token_count, output_token_count.
# If guidellm bumps the schema (see plan §9 trip wires), update this filter.
JSON_FILE="$OUTPUT_DIR/benchmarks.json"
TABLE_FILE="$OUTPUT_DIR/headline_table.md"

emit_header() {
    printf '| rate | TTFT mean | TTFT std | TTFT p50 | TTFT p99 | TPOT mean | ITL mean | ITL std | ITL p50 | ITL p95 | ITL p99 | ITL max | E2E mean | E2E p99 | conc p50 | out tok/s | total tok/s | in tok/s | total in | total out | req/s actual |\n'
    printf '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n'
}

extract_rows() {
    jq -r '
        def pctl($m): (.metrics[$m].successful.percentiles // {});
        def avg($m):  (.metrics[$m].successful.mean        // null);
        def std($m):  (.metrics[$m].successful.std_dev     // null);
        def mx($m):   (.metrics[$m].successful.max         // null);
        def tot($m):  (.metrics[$m].successful.total_sum   // null);
        def rnd(d): if . == null then "n/a" else (. * pow(10;d) | round / pow(10;d)) end;
        .benchmarks
        | map(
            {
              rate: (
                  .config.strategy
                  | if   .type_ == "synchronous" then "sync"
                    elif .type_ == "concurrent"  then "conc\(.max_concurrency // "?")"
                    elif .type_ == "throughput"  then "throughput"
                    elif (.rate // null) != null then "\(.rate)r/s"
                    else .type_
                    end
              ),
              ttft_mean: (avg("time_to_first_token_ms")      | rnd(1)),
              ttft_std:  (std("time_to_first_token_ms")      | rnd(1)),
              ttft_p50:  (pctl("time_to_first_token_ms").p50 | rnd(1)),
              ttft_p99:  (pctl("time_to_first_token_ms").p99 | rnd(1)),
              tpot_mean: (avg("time_per_output_token_ms")    | rnd(2)),
              itl_mean:  (avg("inter_token_latency_ms")      | rnd(2)),
              itl_std:   (std("inter_token_latency_ms")      | rnd(2)),
              itl_p50:   (pctl("inter_token_latency_ms").p50 | rnd(2)),
              itl_p95:   (pctl("inter_token_latency_ms").p95 | rnd(2)),
              itl_p99:   (pctl("inter_token_latency_ms").p99 | rnd(2)),
              itl_max:   (mx("inter_token_latency_ms")       | rnd(2)),
              e2e_mean:  (avg("request_latency")             | rnd(2)),
              e2e_p99:   (pctl("request_latency").p99        | rnd(2)),
              conc_p50:  (pctl("request_concurrency").p50    | rnd(1)),
              out_tok_s: (
                  if (((.metrics.request_totals.successful // 0) > 0)
                      and (((.metrics.request_streaming_iterations_count.successful.mean) // 0) == 0))
                  then "OOM/empty"
                  else (avg("output_tokens_per_second") | rnd(2) | tostring)
                  end
              ),
              total_tok_s: (avg("tokens_per_second")        | rnd(2)),
              in_tok_s:    (avg("prompt_tokens_per_second") | rnd(2)),
              total_in:    (tot("prompt_token_count")       | rnd(0)),
              total_out:   (tot("output_token_count")       | rnd(0)),
              req_s:       (avg("requests_per_second")      | rnd(3))
            }
          )
        | .[]
        | "| \(.rate) | \(.ttft_mean) | \(.ttft_std) | \(.ttft_p50) | \(.ttft_p99) | \(.tpot_mean) | \(.itl_mean) | \(.itl_std) | \(.itl_p50) | \(.itl_p95) | \(.itl_p99) | \(.itl_max) | \(.e2e_mean) | \(.e2e_p99) | \(.conc_p50) | \(.out_tok_s) | \(.total_tok_s) | \(.in_tok_s) | \(.total_in) | \(.total_out) | \(.req_s) |"
    ' "$JSON_FILE" 2>/dev/null || true
}

# K6 (2026-04-29): silent-OOM detector. When the server OOMs mid-bench,
# guidellm still records "successful" requests with zero streaming
# iterations (no tokens emitted). Surface these so they can't be confused
# with healthy runs. Prints one warning line per offending rate.
emit_oom_warnings() {
    jq -r '
        .benchmarks
        | map({
            rate: (
                .config.strategy
                | if   .type_ == "synchronous" then "sync"
                  elif .type_ == "concurrent"  then "conc\(.max_concurrency // "?")"
                  elif .type_ == "throughput"  then "throughput"
                  elif (.rate // null) != null then "\(.rate)r/s"
                  else .type_
                  end
            ),
            successful: (.metrics.request_totals.successful // 0),
            errored:    (.metrics.request_totals.errored    // 0),
            iter_mean:  ((.metrics.request_streaming_iterations_count.successful.mean) // 0)
          })
        | map(select(.successful > 0 and .iter_mean == 0))
        | map("  - \(.rate): successful=\(.successful) errored=\(.errored) iter_mean=0 — silent OOM / empty outputs")
        | .[]
    ' "$JSON_FILE" 2>/dev/null || true
}

{
    emit_header
    rows="$(extract_rows)"
    if [[ -n "$rows" ]]; then
        printf '%s\n' "$rows"
    else
        printf '| _extraction failed_ | see `benchmarks.html` | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |\n'
    fi
} > "$TABLE_FILE"

echo
OOM_WARN="$(emit_oom_warnings)"
if [[ -n "$OOM_WARN" ]]; then
    echo ">>> WARNING — silent-OOM runs detected (successful>0 but iter_mean=0):"
    echo "$OOM_WARN"
    echo "    These rows show tok/s = OOM/empty in the table below. Do not"
    echo "    treat as healthy results — the server emitted zero tokens."
    echo
fi
echo ">>> headline table"
cat "$TABLE_FILE"
echo

# ---- stability check: flag noisy ITL (p99 >> p50) ----------------------------
# Large gap between ITL p50 and p99 usually means thermal throttling, GC
# pause, or saturation. Flag anything where p99 > 2.0 × p50 — at that
# point the percentiles aren't stable enough for A/B comparison.
STABILITY_WARN="$(jq -r '
    .benchmarks
    | map({
        rate: (
            .config.strategy
            | if   .type_ == "synchronous" then "sync"
              elif .type_ == "concurrent"  then "conc\(.max_concurrency // "?")"
              else .type_
              end
        ),
        p50: (.metrics.inter_token_latency_ms.successful.percentiles.p50 // 0),
        p99: (.metrics.inter_token_latency_ms.successful.percentiles.p99 // 0)
    })
    | map(select(.p50 > 0 and .p99 / .p50 > 2.0))
    | map("  - \(.rate): ITL p99/p50 = \((.p99 / .p50) | .*100 | round / 100) (p50=\(.p50) ms, p99=\(.p99) ms)")
    | .[]
' "$JSON_FILE" 2>/dev/null || true)"
if [[ -n "$STABILITY_WARN" ]]; then
    echo ">>> stability warning — ITL p99 > 2× p50 at:"
    echo "$STABILITY_WARN"
    echo "    Consider re-running with a higher --warmup to let thermals settle."
    echo
fi

# ---- exploration mode: skip wins entry ---------------------------------------
if [[ "$EXPLORATION_MODE" == true ]]; then
    echo ">>> exploration mode — skipping wins entry seed"
    echo "    raw artefacts: $OUTPUT_DIR"
    echo "    service trace: $SERVICE_STATS_SUMMARY"
    exit 0
fi

# ---- seed the wins file ------------------------------------------------------
# Copy template, replace placeholders, append the real table into the
# "Results — sweep headline table" section.
python3 - "$TEMPLATE_FILE" "$WINS_FILE" "$TABLE_FILE" \
    "$LABEL" "$DATE" "$COMMIT_SHA" "$MODEL" "$TARGET" "$OUTPUT_DIR" <<'PY'
import sys, pathlib

template, out, table_file, label, date, sha, model, target, outdir = sys.argv[1:]
body = pathlib.Path(template).read_text()
table = pathlib.Path(table_file).read_text().rstrip() + "\n"

# Fill the top-of-doc placeholders (leave hardware/features as <TODO>).
body = body.replace("<SHORT TITLE>", f"guidellm sweep {label}")
body = body.replace("<BACKEND-LABEL>", label)
body = body.replace("<YYYY-MM-DD>", date)
body = body.replace("<short sha>", sha)
body = body.replace("<Qwen/Qwen3-4B | Qwen/Qwen3.5-4B | ...>", model)
body = body.replace("http://localhost:8000", target)
body = body.replace("bench-output/<date>-<label>/", outdir.rstrip('/') + "/")

# Replace the skeleton results table with the real one. The template has a
# header row then a "... sweep auto-steps ..." row; we swap the whole block.
marker = "## Results — sweep headline table"
parts = body.split(marker, 1)
if len(parts) == 2:
    tail = parts[1]
    # find next "## " heading to know where the table block ends
    next_h = tail.find("\n## ")
    if next_h == -1:
        new_tail = "\n\n" + table + "\n"
    else:
        new_tail = "\n\n" + table + "\n" + tail[next_h:]
    body = parts[0] + marker + new_tail

pathlib.Path(out).write_text(body)
PY

cat >> "$WINS_FILE" <<EOF

## Service Trace

- Poll interval: \`${TRACE_INTERVAL_MS}ms\`
- Before: \`${SERVICE_STATS_BEFORE}\`
- During: \`${SERVICE_STATS_TRACE}\`
- After: \`${SERVICE_STATS_AFTER}\`
- Summary: \`${SERVICE_STATS_SUMMARY}\`
EOF

echo ">>> artefacts"
echo "    raw  : $OUTPUT_DIR"
echo "    html : $OUTPUT_DIR/benchmarks.html"
echo "    trace: $SERVICE_STATS_SUMMARY"
echo "    wins : $WINS_FILE"
echo
echo "Next: fill the hardware / features / non-default flags in $WINS_FILE,"
echo "      diff against the previous $(basename "$wins_base").md snapshot,"
echo "      and commit both the wins entry and (if applicable) the code delta."
