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
DATA="prompt_tokens=4096,output_tokens=256"
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

usage() {
    cat <<EOF
usage: $(basename "$0") <backend-label> [options]

  <backend-label>        required, e.g. cuda-h100, metal-m3max

Canonical run (produces a wins entry):
  --target URL           default: $TARGET
  --model NAME           default: $MODEL
  --processor PATH       tokenizer path / HF id (default: local $PROCESSOR_DEFAULT)

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
            DATA="prompt_tokens=512,output_tokens=128"
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

# ---- resolve output paths, never overwrite -----------------------------------
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATE="$(date +%Y-%m-%d)"

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

# guidellm 0.6.0 hangs at "Setup complete, starting benchmarks..." on macOS
# under the default `fork` mp context (Python 3.11+ deprecates fork on darwin
# and the worker_group spawn deadlocks). `forkserver` boots cleanly. See
# scripts/setup_bench_toolchain.sh for the toolchain pin.
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

if [[ $gdl_rc -ne 0 ]]; then
    echo "error: guidellm exited with status $gdl_rc" >&2
    echo "       raw artefacts (if any): $OUTPUT_DIR" >&2
    echo "       full log: $GUIDELLM_LOG" >&2
    exit 3
fi

# ---- metric extraction (schema pinned to guidellm 0.6.x) ---------------------
# Verified 2026-04-15 against Qwen3-0.6B on Metal. Path layout:
#   .benchmarks[n].metrics.<metric>.successful.{mean,percentiles.p50,p99}
#   .benchmarks[n].config.strategy.{type_,max_concurrency}
# Metrics used: time_to_first_token_ms, inter_token_latency_ms,
#               output_tokens_per_second, requests_per_second.
# If guidellm bumps the schema (see plan §9 trip wires), update this filter.
JSON_FILE="$OUTPUT_DIR/benchmarks.json"
TABLE_FILE="$OUTPUT_DIR/headline_table.md"

emit_header() {
    printf '| rate | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |\n'
    printf '|---|---|---|---|---|---|---|\n'
}

extract_rows() {
    jq -r '
        def pctl($m): (.metrics[$m].successful.percentiles // {});
        def avg($m):  (.metrics[$m].successful.mean        // null);
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
              ttft_p50: (pctl("time_to_first_token_ms").p50 | rnd(1)),
              ttft_p99: (pctl("time_to_first_token_ms").p99 | rnd(1)),
              itl_p50:  (pctl("inter_token_latency_ms").p50 | rnd(2)),
              itl_p99:  (pctl("inter_token_latency_ms").p99 | rnd(2)),
              tok_s:    (avg("output_tokens_per_second") | rnd(2)),
              req_s:    (avg("requests_per_second")      | rnd(3))
            }
          )
        | .[]
        | "| \(.rate) | \(.ttft_p50) | \(.ttft_p99) | \(.itl_p50) | \(.itl_p99) | \(.tok_s) | \(.req_s) |"
    ' "$JSON_FILE" 2>/dev/null || true
}

{
    emit_header
    rows="$(extract_rows)"
    if [[ -n "$rows" ]]; then
        printf '%s\n' "$rows"
    else
        printf '| _extraction failed_ | see | `benchmarks.html` | for | full | results | . |\n'
    fi
} > "$TABLE_FILE"

echo
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
body = body.replace("<Qwen/Qwen3-4B | Qwen/Qwen3.5-4B | THUDM/GLM-4 | ...>", model)
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

echo ">>> artefacts"
echo "    raw  : $OUTPUT_DIR"
echo "    html : $OUTPUT_DIR/benchmarks.html"
echo "    wins : $WINS_FILE"
echo
echo "Next: fill the hardware / features / non-default flags in $WINS_FILE,"
echo "      diff against the previous $(basename "$wins_base").md snapshot,"
echo "      and commit both the wins entry and (if applicable) the code delta."
