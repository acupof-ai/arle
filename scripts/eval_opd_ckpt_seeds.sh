#!/usr/bin/env bash
# Multi-seed MMLU+GSM8K eval for a single OPD checkpoint (or base model).
# Usage: scripts/eval_opd_ckpt_seeds.sh <ckpt_path|base> <out_base> <seed1> [seed2 ...]
#
# Mechanism: boots `target/release/infer` once with INFER_LORA_PATH set to
# <ckpt_path> (or omitted when ckpt_path == "base"), then runs
# scripts/arle_capability_eval.py once per seed, emitting
# <out_base>/seed_<N>/{mmlu.json,gsm8k.json,summary.json,eval.log}.
# Reuses one serve across seeds — only sample-selection differs.
#
# Why a separate script vs eval_opd_ckpts.sh: that driver iterates ckpts
# (one serve per ckpt). For variance estimation at a fixed ckpt the
# per-seed re-serve is pure overhead, and we want the same model state
# across seeds so the only source of variance is the sample subset.
#
# Pass "base" as ckpt_path to eval the base model with no LoRA — used to
# get a matched-n multi-seed baseline for paired comparison against an
# OPD checkpoint.
set -uo pipefail

if [[ $# -lt 3 ]]; then
    echo "usage: $0 <ckpt_path|base> <out_base> <seed1> [seed2 ...]" >&2
    exit 2
fi

CKPT_PATH="$1"
OUT_BASE="$2"
shift 2
SEEDS=("$@")

if [[ "$CKPT_PATH" != "base" && ! -d "$CKPT_PATH" ]]; then
    echo "error: ckpt $CKPT_PATH not found (pass 'base' literal for no-LoRA baseline)" >&2
    exit 2
fi

STUDENT_BASE="${STUDENT_BASE:-/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base}"
PORT="${PORT:-8123}"
N_SAMPLES="${N_SAMPLES:-200}"
GSM8K_SHOTS="${GSM8K_SHOTS:-8}"
# Default 0.30 keeps 0.8B serve under ~5 GB on a 16 GB SKU (vs the 0.85
# infer default that grabs ~13.6 GB). Single-request eval doesn't need a
# big KV pool. Override for sweeps or larger ckpts.
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.30}"
mkdir -p "$OUT_BASE"

printf 'ckpt=%s\nout_base=%s\nseeds=%s\nn_samples=%s\n' \
    "$CKPT_PATH" "$OUT_BASE" "${SEEDS[*]}" "$N_SAMPLES" \
    | tee "$OUT_BASE/run.meta"

if [[ "$CKPT_PATH" == "base" ]]; then
    LORA_ENV=()
    echo "base-model eval (no INFER_LORA_PATH)"
else
    LORA_ENV=("INFER_LORA_PATH=$CKPT_PATH")
fi

env "${LORA_ENV[@]}" target/release/infer \
    --model-path "$STUDENT_BASE" \
    --port "$PORT" \
    --disable-cuda-graph \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    > "$OUT_BASE/serve.log" 2>&1 &
SERVE_PID=$!
trap 'kill "$SERVE_PID" 2>/dev/null || true; sleep 2; kill -9 "$SERVE_PID" 2>/dev/null || true' EXIT

tries=0
while ! curl -sf "http://127.0.0.1:$PORT/v1/models" -o "$OUT_BASE/models.json" 2>/dev/null; do
    tries=$((tries + 1))
    if [[ $tries -ge 180 ]]; then
        echo "serve TIMEOUT after 180s" >&2
        exit 3
    fi
    sleep 1
done
echo "serve ready after ${tries}s pid=$SERVE_PID"

for seed in "${SEEDS[@]}"; do
    seed_dir="$OUT_BASE/seed_$seed"
    mkdir -p "$seed_dir"
    printf '\n══════════ seed=%s ══════════\n' "$seed"

    ARLE_BASE_URL="http://127.0.0.1:$PORT" \
        .venv/bin/python scripts/arle_capability_eval.py \
            --backend arle \
            --base-url "http://127.0.0.1:$PORT" \
            --model-id "Qwen3___5-0___8B-Base" \
            --tasks mmlu,gsm8k \
            --n-samples "$N_SAMPLES" \
            --gsm8k-shots "$GSM8K_SHOTS" \
            --seed "$seed" \
            --output "$seed_dir/" \
            > "$seed_dir/eval.log" 2>&1 || {
        echo ">> seed=$seed eval EXITED nonzero, continuing"
        continue
    }

    if [[ -f "$seed_dir/summary.json" ]]; then
        .venv/bin/python -c "
import json
d = json.load(open('$seed_dir/summary.json'))
m = d['tasks']['mmlu'].get('accuracy', 'skip')
g = d['tasks']['gsm8k'].get('accuracy', 'skip')
print(f'>> seed=$seed mmlu={m} gsm8k={g}')"
    else
        echo ">> seed=$seed NO summary.json"
    fi
done

printf '\n══════════ MULTI-SEED SUMMARY ══════════\n'
printf '| %-6s | %8s | %8s |\n' 'seed' 'mmlu' 'gsm8k'
printf '| %-6s | %8s | %8s |\n' '------' '--------' '--------'
for d in "$OUT_BASE"/seed_*/; do
    name=$(basename "$d")
    seed=${name#seed_}
    if [[ -f "$d/summary.json" ]]; then
        readarray -t parts < <(.venv/bin/python -c "
import json
d = json.load(open('$d/summary.json'))
m = d['tasks']['mmlu'].get('accuracy', 0)
g = d['tasks']['gsm8k'].get('accuracy', 0)
print(f'{m:.4f}')
print(f'{g:.4f}')")
        printf '| %-6s | %8s | %8s |\n' "$seed" "${parts[0]}" "${parts[1]}"
    fi
done

# Wilson CI + mean / sample-σ + kill-criterion verdict via the dedicated
# analyzer (single source of truth vs. inline math). Add --paired-vs to
# the analyzer invocation manually when comparing two runs at matched
# seeds.
.venv/bin/python scripts/analyze_multi_seed.py "$OUT_BASE" || true
