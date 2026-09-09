#!/usr/bin/env bash
# Matched A/B harness on top of bench_throughput.py.
#
# Why: any A/B comparison on one GPU must run A and B back-to-back on the
# same machine with the same workload. Manual start-server, bench, kill,
# start-other, bench, kill — easy to forget a step, easy to drift flags
# between runs. This script bundles it.
#
# Usage:
#   scripts/bench_ab.sh <label-a> <label-b> --seconds-per-concurrency N \
#       [--concurrency-grid L] [--model NAME] [--prompts-jsonl PATH] \
#       --cmd-a "<shell cmd that starts server-A, backgrounds itself>" \
#       --cmd-b "<shell cmd that starts server-B, backgrounds itself>"
#
#   Commands must:
#     * launch a server that listens on $PORT (env var, default 8000)
#     * background themselves (trailing &) — the harness will read the PID
#       from $! inside eval
#     * be idempotent across kill (cleanup SIGKILLs serves by comm + args)
#
# Native runner flags are forwarded to bench_throughput.py. Artefacts land in
# bench-output/<date>-<label-a>/ and bench-output/<date>-<label-b>/. No
# wins entries are seeded (exploration mode).
#
# Spec-decode readings need their own no-draft arm in the same session:
#
#   scripts/bench_ab.sh qwen36-dspark --vs-no-spec \
#       --cmd-treatment "$BIN --model-path $M --port 8000 \
#                        --spec-type dspark --mtp-draft-model $DRAFT &" \
#       --seconds-per-concurrency 120
#
# derives the control command by stripping the spec flags
# (--spec-type, --mtp-draft-model, --dspark-*) and runs both arms.
# A spec number from a single arm compared against an external row is
# refused: the diff step fails loud when either arm has no data.
#
# Example — MTP vs no-MTP on Qwen3.6 Metal, two-minute cells:
#
#   MODEL=mlx-community/Qwen3.6-35B-A3B-4bit
#   BIN="target/release/arle serve --backend metal"
#   scripts/bench_ab.sh \
#       qwen36-baseline \
#       qwen36-mtp \
#       --seconds-per-concurrency 120 \
#       --model "$MODEL" \
#       --cmd-a "$BIN --model-path $MODEL --port 8000 \
#                > /tmp/ab-a.log 2>&1 &" \
#       --cmd-b "$BIN --model-path $MODEL --port 8000 --spec-type mtp \
#                > /tmp/ab-b.log 2>&1 &"

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT" || exit

PORT="${PORT:-8000}"
TARGET="http://127.0.0.1:${PORT}"

LABEL_A=""
LABEL_B=""
CMD_A=""
CMD_B=""
# --vs-no-spec: one label + one treatment command; the control is derived.
VS_NO_SPEC=false
CMD_TREATMENT=""
# Flags forwarded to bench_throughput.py.
PASSTHROUGH=()

# Spec flags stripped from the treatment command to build the no-draft
# control arm. Each takes one value.
SPEC_FLAGS=(
    --spec-type
    --mtp-draft-model
    --dspark-sps-bias-ms
    --dspark-sps-row-ms
    --dspark-block-size
    --dspark-markov-init
)

strip_spec_flags() {
    local -a toks out
    read -ra toks <<< "$1"
    local skip=0 t
    for t in "${toks[@]}"; do
        if (( skip )); then skip=0; continue; fi
        local s
        for s in "${SPEC_FLAGS[@]}"; do
            if [[ "$t" == "$s" ]]; then skip=1; break; fi
        done
        (( skip )) || out+=("$t")
    done
    local IFS=' '
    printf '%s' "${out[*]}"
}

usage() {
    cat <<EOF
usage: $(basename "$0") <label-a> <label-b> --cmd-a "<launch>" --cmd-b "<launch>" [options]
       $(basename "$0") <label> --vs-no-spec --cmd-treatment "<launch with spec flags>" [options]

  <label-a> / <label-b>   labels for A and B artefacts
  --cmd-a "..."           shell command that starts server A (trailing & required)
  --cmd-b "..."           shell command that starts server B (trailing & required)
  --vs-no-spec            spec mode: derive the no-draft control from the
                          treatment command by stripping its spec flags
  --cmd-treatment "..."   treatment server command (with --spec-type etc.)

Forwarded to bench_throughput.py (one measurement bound required):
  --concurrency-grid LIST e.g. "1,2,4,8"
  --requests-per-concurrency N
  --seconds-per-concurrency N
  --max-tokens N
  --temperature F
  --model NAME            model identifier
  --prompts-jsonl PATH    prompt dataset

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
        --vs-no-spec)
            VS_NO_SPEC=true; shift ;;
        --cmd-treatment)
            [[ $# -ge 2 ]] || { echo "error: --cmd-treatment requires a value" >&2; exit 2; }
            CMD_TREATMENT="$2"; shift 2 ;;
        -h|--help)
            usage; exit 0 ;;
        --concurrency-grid|--requests-per-concurrency|--seconds-per-concurrency|--max-tokens|--temperature|--model|--prompts-jsonl|--synthetic-prompts|--timeout-seconds|--seed)
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

if [[ "$VS_NO_SPEC" == true ]]; then
    if [[ -n "$LABEL_B" || -n "$CMD_A" || -n "$CMD_B" ]]; then
        echo "error: --vs-no-spec takes one label and --cmd-treatment only" >&2
        usage >&2
        exit 2
    fi
    [[ -n "$CMD_TREATMENT" ]] || { echo "error: --vs-no-spec requires --cmd-treatment" >&2; exit 2; }
    [[ -n "$LABEL_A" ]] || { echo "error: --vs-no-spec requires one label" >&2; exit 2; }
    LABEL_B="${LABEL_A}-no-spec"
    CMD_A="$CMD_TREATMENT"
    CMD_B="$(strip_spec_flags "$CMD_TREATMENT")"
    [[ "$CMD_A" != "$CMD_B" ]] || die "no spec flags found in treatment command: $CMD_TREATMENT"
    echo "control arm derived by stripping spec flags:"
    echo "  B: $CMD_B"
fi

if [[ -z "$LABEL_A" || -z "$LABEL_B" || -z "$CMD_A" || -z "$CMD_B" ]]; then
    echo "error: <label-a>, <label-b>, --cmd-a, --cmd-b are all required" >&2
    usage >&2
    exit 2
fi

# Refuse unbounded A/B runs.
has_exploration=false
for f in "${PASSTHROUGH[@]}"; do
    case "$f" in
        --requests-per-concurrency|--seconds-per-concurrency)
            has_exploration=true; break ;;
    esac
done
if [[ "$has_exploration" == false ]]; then
    echo "error: bench_ab.sh requires --requests-per-concurrency or --seconds-per-concurrency." >&2
    exit 2
fi

die() { echo "error: $*" >&2; exit 3; }

# Kill only serves: comm=arle plus ' serve' in args leaves `arle kernel`
# and other subcommands alone. SIGKILL — serve ignores SIGTERM.
kill_arle_serve() {
    for p in $(pgrep -x arle 2>/dev/null); do
        ps -p "$p" -o args= 2>/dev/null | grep -q ' serve' && kill -9 "$p" 2>/dev/null
    done
}

cleanup() {
    kill_arle_serve
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
    local label="$1" cmd="$2" out_dir="$3"
    CURRENT_CMD="$cmd"
    echo
    echo "=== $label ==="
    # Ensure the port is free before launching.
    kill_arle_serve
    sleep 2
    echo "launch: $cmd"
    eval "$cmd" || die "failed to launch: $cmd"
    wait_for_server || die "server for $label never became ready"

    mkdir -p "$out_dir"
    python3 "$REPO_ROOT/scripts/bench_throughput.py" --url "$TARGET" \
        --output "$out_dir/bench_throughput" "${PASSTHROUGH[@]}" \
        || die "bench run failed for $label"

    kill_arle_serve
    sleep 2
    CURRENT_CMD=""
}

DATE="$(date +%Y-%m-%d)"
OUT_A="$REPO_ROOT/bench-output/${DATE}-${LABEL_A}"
OUT_B="$REPO_ROOT/bench-output/${DATE}-${LABEL_B}"
unique_dir() { local base="$1" out="$1" n=1; while [[ -e "$out" ]]; do n=$((n + 1)); out="${base}-run${n}"; done; printf '%s\n' "$out"; }
OUT_A="$(unique_dir "$OUT_A")"
OUT_B="$(unique_dir "$OUT_B")"

run_side "$LABEL_A" "$CMD_A" "$OUT_A"
run_side "$LABEL_B" "$CMD_B" "$OUT_B"

# ---- cross-label diff ---------------------------------------------------------

DIFF_FILE="$REPO_ROOT/bench-output/${DATE}-${LABEL_A}-vs-${LABEL_B}-diff.md"
python3 - "$OUT_A" "$OUT_B" "$LABEL_A" "$LABEL_B" "$DIFF_FILE" <<'PY' || die "diff refused (missing arm data)"
import sys, json, pathlib

a_dir, b_dir, label_a, label_b, out_path = sys.argv[1:]

def load(d):
    p = pathlib.Path(d) / "bench_throughput.json"
    if not p.exists():
        return None
    j = json.loads(p.read_text())
    rows = {}
    for point in j.get("points", []):
        m = point.get("summary", {})
        key = f"conc{m.get('concurrency', '?')}"
        rows[key] = {
            "ttft_p50": (m.get("ttft") or {}).get("p50_ms"),
            "itl_p50":  (m.get("itl") or {}).get("p50_ms"),
            "tok_s":    (1000.0 / im if (im := (m.get("itl") or {}).get("mean_ms")) else None),
        }
    return rows

a = load(a_dir) or {}
b = load(b_dir) or {}
if not a or not b:
    print(f"error: refusing to diff without both arms: {label_a}={len(a)} rows, "
          f"{label_b}={len(b)} rows", file=sys.stderr)
    sys.exit(1)
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
lines.append("| rate | A decode tok/s | B decode tok/s | Δ decode | A TTFT p50 | B TTFT p50 | Δ TTFT |")
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
lines.append("Reminder: effects <=10% in a single session are thermal noise; rerun")
lines.append("or extend the cell duration before trusting small deltas.")

pathlib.Path(out_path).write_text("\n".join(lines) + "\n")
print("".join(f"{l}\n" for l in lines))
PY

echo ">>> diff: $DIFF_FILE"
