#!/usr/bin/env bash
# Sequential per-checkpoint MMLU + GSM8K eval for an OPD train run dir.
# Usage: scripts/eval_opd_ckpts.sh <run_dir>
# Output: <run_dir>/capability/<ckpt>/{serve.log,eval.log,summary.json,...}
#
# Mechanism: for each ckpt under <run_dir>/student/step_NNN + final, boot
# `target/release/infer` with INFER_LORA_PATH set, wait for /v1/models,
# run scripts/arle_capability_eval.py via .venv/bin/python (needs `datasets`),
# kill serve, move on. Final pass prints a combined terminal table.
#
# Replaces the earlier eval_v3_ckpts.sh which hard-coded the v3 run dir.
set -uo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <run_dir>" >&2
    exit 2
fi

RUN_DIR="$1"
if [[ ! -d "$RUN_DIR/student" ]]; then
    echo "error: $RUN_DIR/student not found" >&2
    exit 2
fi

STUDENT_BASE="${STUDENT_BASE:-/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base}"
PORT="${PORT:-8123}"
OUT_BASE="$RUN_DIR/capability"
mkdir -p "$OUT_BASE"

eval_one() {
    local label="$1"
    local lora_path="$2"
    local out_dir="$OUT_BASE/$label"
    mkdir -p "$out_dir"
    printf '\n══════════ %s ══════════\n' "$label"
    printf 'lora=%s out=%s\n' "${lora_path:-<none>}" "$out_dir"

    local lora_env=""
    [[ -n "$lora_path" ]] && lora_env="INFER_LORA_PATH=$lora_path"
    env $lora_env target/release/infer \
        --model-path "$STUDENT_BASE" \
        --port "$PORT" \
        --disable-cuda-graph \
        > "$out_dir/serve.log" 2>&1 &
    local pid=$!
    local tries=0
    while ! curl -sf "http://127.0.0.1:$PORT/v1/models" -o "$out_dir/models.json" 2>/dev/null; do
        tries=$((tries+1))
        if [[ $tries -ge 120 ]]; then
            echo "serve TIMEOUT after 120s"
            kill "$pid" 2>/dev/null || true
            return 1
        fi
        sleep 1
    done
    echo "serve ready after ${tries}s pid=$pid"

    ARLE_BASE_URL="http://127.0.0.1:$PORT" \
        .venv/bin/python scripts/arle_capability_eval.py \
            --backend arle \
            --base-url "http://127.0.0.1:$PORT" \
            --model-id "Qwen3___5-0___8B-Base" \
            --tasks mmlu,gsm8k \
            --gsm8k-shots 8 \
            --output "$out_dir/" \
            > "$out_dir/eval.log" 2>&1

    kill "$pid" 2>/dev/null || true
    sleep 2
    kill -9 "$pid" 2>/dev/null || true

    if [[ -f "$out_dir/summary.json" ]]; then
        .venv/bin/python -c "
import json
d = json.load(open('$out_dir/summary.json'))
m = d['tasks']['mmlu'].get('accuracy', 'skip')
g = d['tasks']['gsm8k'].get('accuracy', 'skip')
print(f'>> $label mmlu={m} gsm8k={g}')"
    else
        printf '>> %s NO summary.json\n' "$label"
    fi
    sleep 3
}

eval_one "base_0p8b" ""
for step in 010 020 030 040 050 060 070 080 090 100; do
    ckpt="$RUN_DIR/student/step_000${step}"
    [[ -d "$ckpt" ]] && eval_one "step_000${step}" "$ckpt"
done
[[ -d "$RUN_DIR/student/final" ]] && eval_one "final" "$RUN_DIR/student/final"

printf '\n══════════ BENCH SUMMARY (%s) ══════════\n' "$RUN_DIR"
printf '| %-15s | %8s | %8s |\n' 'ckpt' 'mmlu' 'gsm8k'
printf '| %-15s | %8s | %8s |\n' '---------------' '--------' '--------'
for d in "$OUT_BASE"/*/; do
    name=$(basename "$d")
    if [[ -f "$d/summary.json" ]]; then
        readarray -t parts < <(.venv/bin/python -c "
import json
d = json.load(open('$d/summary.json'))
m = d['tasks']['mmlu'].get('accuracy', 0)
g = d['tasks']['gsm8k'].get('accuracy', 0)
print(f'{m:.4f}')
print(f'{g:.4f}')")
        printf '| %-15s | %8s | %8s |\n' "$name" "${parts[0]}" "${parts[1]}"
    fi
done
