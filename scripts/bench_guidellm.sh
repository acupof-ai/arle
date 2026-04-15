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
# Optional flags:
#   --target URL     inference server URL   (default: http://localhost:8000)
#   --model  NAME    model identifier       (default: Qwen/Qwen3-4B)
#
# Preconditions:
#   * guidellm, curl, jq on PATH
#   * infer HTTP server is already running at --target
#     (start it with: scripts/start_pegainfer.sh)
#
# Side effects:
#   * Writes raw artefacts to bench-output/<date>-<label>[-runN]/
#     (benchmarks.json / .csv / .html). This dir is gitignored.
#   * Seeds a new docs/experience/wins/<date>-bench-guidellm-<label>.md
#     from docs/experience/wins/TEMPLATE-bench-guidellm.md with the commit
#     sha, paths, and a best-effort headline metric table filled in.
#
# The canonical benchmark parameters are LOCKED here. Changing them is a
# deliberate commit, not a flag flip. See docs/plans/guidellm-integration.md §3.

set -euo pipefail

# ---- Canonical params (locked, see docs/plans/guidellm-integration.md §3) ----
PROFILE="sweep"
DATA="prompt_tokens=1024,output_tokens=256"
MAX_SECONDS=60
RANDOM_SEED=20260416
OUTPUTS="json,csv,html"
# guidellm's default backend validation probes GET /health, which
# metal_serve / cuda-infer do not expose. Point it at /v1/models instead
# (we already rely on that route being present in preflight below).
BACKEND_KWARGS='{"validate_backend": "/v1/models"}'
# ------------------------------------------------------------------------------

TARGET="http://localhost:8000"
MODEL="Qwen/Qwen3-4B"
LABEL=""

usage() {
    cat <<EOF
usage: $(basename "$0") <backend-label> [--target URL] [--model NAME]

  <backend-label>   required, e.g. cuda-h100, metal-m3max
  --target URL      default: $TARGET
  --model NAME      default: $MODEL

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
    echo "error: server not running at $TARGET — start it with scripts/start_pegainfer.sh first" >&2
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
echo "    output : $OUTPUT_DIR"
echo

set +e
guidellm benchmark run \
    --target "$TARGET" \
    --model "$MODEL" \
    --processor "$MODEL" \
    --profile "$PROFILE" \
    --data "$DATA" \
    --max-seconds "$MAX_SECONDS" \
    --random-seed "$RANDOM_SEED" \
    --output-dir "$OUTPUT_DIR" \
    --outputs "$OUTPUTS" \
    --backend-kwargs "$BACKEND_KWARGS"
gdl_rc=$?
set -e

if [[ $gdl_rc -ne 0 ]]; then
    echo "error: guidellm exited with status $gdl_rc" >&2
    echo "       raw artefacts (if any): $OUTPUT_DIR" >&2
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
