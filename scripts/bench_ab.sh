#!/usr/bin/env bash
# Matched A/B bench harness on top of bench_guidellm.sh.
#
# Why: any A/B comparison on one GPU must run A and B back-to-back on the
# same machine with the same workload. Manual start-server, bench, kill,
# start-other, bench, kill — easy to forget a step, easy to drift flags
# between runs. This script bundles it.
#
# Usage:
#   scripts/bench_ab.sh <label-a> <label-b> [--quick] [--concurrencies L] \
#       [--model NAME] [--processor PATH] \
#       --cmd-a "<shell cmd that starts server-A, backgrounds itself>" \
#       --cmd-b "<shell cmd that starts server-B, backgrounds itself>"
#
#   Commands must:
#     * launch a server that listens on $PORT (env var, default 8000)
#     * background themselves (trailing &) — the harness will read the PID
#       from $! inside eval
#     * be idempotent across kill (we pkill -f <cmd snippet> at cleanup)
#
# Exploration-mode flags (--quick, --concurrencies, --max-seconds, --warmup,
# --profile) are forwarded to bench_guidellm.sh. Artefacts land in
# bench-output/<date>-<label-a>/ and bench-output/<date>-<label-b>/. No
# wins entries are seeded (exploration mode).
#
# Example — DFlash vs no-DFlash on Qwen3.5-4B Metal, 2-minute /quick run:
#
#   MODEL=mlx-community/Qwen3.5-4B-MLX-4bit
#   BIN=target/release/metal_serve
#   scripts/bench_ab.sh \
#       qwen35-baseline \
#       qwen35-dflash \
#       --quick \
#       --model "$MODEL" \
#       --cmd-a "$BIN --model-path $MODEL --port 8000 --warmup 1 \
#                > /tmp/ab-a.log 2>&1 &" \
#       --cmd-b "$BIN --model-path $MODEL --port 8000 --warmup 1 \
#                --dflash-draft-model z-lab/Qwen3.5-4B-DFlash \
#                > /tmp/ab-b.log 2>&1 &"

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PORT="${PORT:-8000}"
TARGET="http://127.0.0.1:${PORT}"

LABEL_A=""
LABEL_B=""
CMD_A=""
CMD_B=""
# Flags forwarded to bench_guidellm.sh.
PASSTHROUGH=()

usage() {
    cat <<EOF
usage: $(basename "$0") <label-a> <label-b> --cmd-a "<launch>" --cmd-b "<launch>" [options]

  <label-a> / <label-b>   labels for A and B artefacts
  --cmd-a "..."           shell command that starts server A (trailing & required)
  --cmd-b "..."           shell command that starts server B (trailing & required)

Forwarded to bench_guidellm.sh (one exploration-mode flag required):
  --quick                 ~2-min preset
  --concurrencies LIST    e.g. "1,2,4,8"
  --max-seconds N         override per-benchmark duration
  --warmup N              warmup seconds/fraction
  --profile TYPE          override profile
  --model NAME            model identifier
  --processor PATH        tokenizer path / HF id

Env:
  PORT=8000               the port both servers bind (default)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cmd-a)
            [[ $# -ge 2 ]] || { echo "error: --cmd-a requires a value" >&2; exit 2; }
            CMD_A="$2"; shift 2 ;;
        --cmd-b)
            [[ $# -ge 2 ]] || { echo "error: --cmd-b requires a value" >&2; exit 2; }
            CMD_B="$2"; shift 2 ;;
        --quick|-h|--help)
            if [[ "$1" == "-h" || "$1" == "--help" ]]; then usage; exit 0; fi
            PASSTHROUGH+=("$1"); shift ;;
        --concurrencies|--max-seconds|--warmup|--profile|--model|--processor)
            [[ $# -ge 2 ]] || { echo "error: $1 requires a value" >&2; exit 2; }
            PASSTHROUGH+=("$1" "$2"); shift 2 ;;
        --*)
            echo "error: unknown flag: $1" >&2; usage >&2; exit 2 ;;
        *)
            if   [[ -z "$LABEL_A" ]]; then LABEL_A="$1"; shift
            elif [[ -z "$LABEL_B" ]]; then LABEL_B="$1"; shift
            else echo "error: unexpected positional arg: $1" >&2; usage >&2; exit 2; fi
            ;;
    esac
done

if [[ -z "$LABEL_A" || -z "$LABEL_B" || -z "$CMD_A" || -z "$CMD_B" ]]; then
    echo "error: <label-a>, <label-b>, --cmd-a, --cmd-b are all required" >&2
    usage >&2
    exit 2
fi

# Refuse to run without at least one exploration flag: sweep is ~10 min per
# side, too slow for rapid A/B. Force the user to opt in to canonical.
has_exploration=false
for f in "${PASSTHROUGH[@]}"; do
    case "$f" in
        --quick|--concurrencies|--max-seconds|--warmup|--profile)
            has_exploration=true; break ;;
    esac
done
if [[ "$has_exploration" == false ]]; then
    echo "error: bench_ab.sh requires exploration mode (--quick recommended)." >&2
    echo "       Canonical sweep A/B takes 15+ min and should be driven manually." >&2
    exit 2
fi

die() { echo "error: $*" >&2; exit 3; }

cleanup() {
    # Best-effort kill. Uses the full command string so we don't hit
    # unrelated processes.
    if [[ -n "${CURRENT_CMD:-}" ]]; then
        pkill -f "$(echo "$CURRENT_CMD" | awk '{print $1}')" 2>/dev/null || true
    fi
    pkill -f "metal_serve|cuda_serve|infer_serve" 2>/dev/null || true
    sleep 1
}
trap cleanup EXIT INT TERM

wait_for_server() {
    for i in $(seq 1 120); do
        if curl -s -o /dev/null -w '%{http_code}' --max-time 1 "$TARGET/v1/models" 2>/dev/null | grep -q '^200$'; then
            echo "ready after ${i}s"
            return 0
        fi
        sleep 1
    done
    return 1
}

run_side() {
    local label="$1" cmd="$2"
    CURRENT_CMD="$cmd"
    echo
    echo "=== $label ==="
    # Ensure the port is free before launching.
    pkill -f "metal_serve|cuda_serve|infer_serve" 2>/dev/null || true
    sleep 2
    echo "launch: $cmd"
    eval "$cmd" || die "failed to launch: $cmd"
    wait_for_server || die "server for $label never became ready"

    "$REPO_ROOT/scripts/bench_guidellm.sh" "$label" --target "$TARGET" "${PASSTHROUGH[@]}" \
        || die "bench run failed for $label"

    pkill -f "metal_serve|cuda_serve|infer_serve" 2>/dev/null || true
    sleep 2
    CURRENT_CMD=""
}

run_side "$LABEL_A" "$CMD_A"
run_side "$LABEL_B" "$CMD_B"

# ---- cross-label diff ---------------------------------------------------------
DATE="$(date +%Y-%m-%d)"
OUT_A="$REPO_ROOT/bench-output/${DATE}-${LABEL_A}"
OUT_B="$REPO_ROOT/bench-output/${DATE}-${LABEL_B}"
# If a run-N suffix was appended, take the latest matching dir.
pick_latest() { ls -d "${1}"* 2>/dev/null | sort | tail -1; }
OUT_A="$(pick_latest "$OUT_A")"
OUT_B="$(pick_latest "$OUT_B")"

DIFF_FILE="$REPO_ROOT/bench-output/${DATE}-${LABEL_A}-vs-${LABEL_B}-diff.md"
python3 - "$OUT_A" "$OUT_B" "$LABEL_A" "$LABEL_B" "$DIFF_FILE" <<'PY'
import sys, json, pathlib

a_dir, b_dir, label_a, label_b, out_path = sys.argv[1:]

def load(d):
    p = pathlib.Path(d) / "benchmarks.json"
    if not p.exists():
        return None
    j = json.loads(p.read_text())
    rows = {}
    for bench in j.get("benchmarks", []):
        strat = bench.get("config", {}).get("strategy", {})
        t = strat.get("type_", "?")
        if t == "synchronous":
            key = "sync"
        elif t == "concurrent":
            key = f"conc{strat.get('max_concurrency', '?')}"
        elif t == "throughput":
            key = "throughput"
        else:
            key = t
        m = bench.get("metrics", {})
        def _get(path):
            cur = m
            for part in path.split("."):
                if cur is None:
                    return None
                cur = cur.get(part)
            return cur
        rows[key] = {
            "ttft_p50": _get("time_to_first_token_ms.successful.percentiles.p50"),
            "itl_p50":  _get("inter_token_latency_ms.successful.percentiles.p50"),
            "tok_s":    _get("output_tokens_per_second.successful.mean"),
        }
    return rows

a = load(a_dir) or {}
b = load(b_dir) or {}
keys = sorted(set(a) | set(b), key=lambda k: (
    0 if k == "sync" else 1 if k.startswith("conc") else 2, k
))

def pct(x, y):
    if x is None or y is None or x == 0:
        return "n/a"
    return f"{((y - x) / x) * 100:+.1f}%"

def fmt(x, d=1):
    if x is None:
        return "n/a"
    return f"{x:.{d}f}"

lines = []
lines.append(f"# A/B diff — {label_a} vs {label_b}")
lines.append("")
lines.append(f"- A: {a_dir}")
lines.append(f"- B: {b_dir}")
lines.append("")
lines.append("| rate | A tok/s | B tok/s | Δ tok/s | A TTFT p50 | B TTFT p50 | Δ TTFT |")
lines.append("|---|---|---|---|---|---|---|")
for k in keys:
    av, bv = a.get(k, {}), b.get(k, {})
    lines.append(
        f"| {k} | {fmt(av.get('tok_s'),2)} | {fmt(bv.get('tok_s'),2)} "
        f"| {pct(av.get('tok_s'), bv.get('tok_s'))} "
        f"| {fmt(av.get('ttft_p50'),1)} | {fmt(bv.get('ttft_p50'),1)} "
        f"| {pct(av.get('ttft_p50'), bv.get('ttft_p50'))} |"
    )
lines.append("")
lines.append("Δ is (B - A) / A. Negative TTFT Δ is faster; positive tok/s Δ is faster.")
lines.append("")
lines.append("> Reminder: effects ≤10% in a single session are thermal noise. See")
lines.append("> memory/feedback_matched_ab_for_small_bench_effects.md.")

pathlib.Path(out_path).write_text("\n".join(lines) + "\n")
print("".join(f"{l}\n" for l in lines))
PY

echo ">>> diff: $DIFF_FILE"
