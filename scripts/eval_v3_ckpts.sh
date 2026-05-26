#!/usr/bin/env bash
# Evaluate v3 OPD checkpoints (rollout=128 + NaN fix + completion-only KL).
# Sequential: serve one ckpt + arle_capability_eval.py + kill, repeat.
# Output: runs/2026-05-26-rollout128-v3-nanfix-train-60/capability/<ckpt>/summary.json
set -uo pipefail

RUN_DIR="runs/2026-05-26-rollout128-v3-nanfix-train-60"
STUDENT_BASE="/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base"
PORT=8123
OUT_BASE="$RUN_DIR/capability"

mkdir -p "$OUT_BASE"

eval_one() {
    local label="$1"   # e.g. "base_0p8b", "step_000010"
    local lora_path="$2"  # empty string for base, else ckpt dir
    local out_dir="$OUT_BASE/$label"
    mkdir -p "$out_dir"

    printf '\n══════════ %s ══════════\n' "$label"
    printf 'lora=%s\n' "${lora_path:-<none>}"
    printf 'out=%s\n' "$out_dir"

    # Launch serve in background
    local serve_log="$out_dir/serve.log"
    local lora_env=""
    if [[ -n "$lora_path" ]]; then
        lora_env="INFER_LORA_PATH=$lora_path"
    fi
    env $lora_env target/release/infer \
        --model-path "$STUDENT_BASE" \
        --port "$PORT" \
        --disable-cuda-graph \
        > "$serve_log" 2>&1 &
    local serve_pid=$!
    printf 'serve pid=%d\n' "$serve_pid"

    # Wait for serve ready
    local tries=0
    while true; do
        if curl -sf "http://127.0.0.1:$PORT/v1/models" -o "$out_dir/models.json" 2>/dev/null; then
            printf 'serve ready after %ds\n' "$tries"
            break
        fi
        tries=$((tries + 1))
        if [[ $tries -ge 120 ]]; then
            printf 'serve FAILED to start within 120s\n'
            kill "$serve_pid" 2>/dev/null || true
            return 1
        fi
        sleep 1
    done

    # Run capability eval
    ARLE_BASE_URL="http://127.0.0.1:$PORT" \
        .venv/bin/python scripts/arle_capability_eval.py \
            --backend arle \
            --base-url "http://127.0.0.1:$PORT" \
            --model-id "Qwen3___5-0___8B-Base" \
            --tasks mmlu,gsm8k \
            --gsm8k-shots 8 \
            --output "$out_dir/" \
            > "$out_dir/eval.log" 2>&1
    local eval_status=$?
    printf 'eval status=%d\n' "$eval_status"

    # Stop serve
    kill "$serve_pid" 2>/dev/null || true
    sleep 2
    kill -9 "$serve_pid" 2>/dev/null || true

    # Quick summary
    if [[ -f "$out_dir/summary.json" ]]; then
        python3 -c "
import json
d = json.load(open('$out_dir/summary.json'))
mmlu = d['tasks'].get('mmlu', {}).get('accuracy', 'n/a')
gsm = d['tasks'].get('gsm8k', {}).get('accuracy', 'n/a')
print(f'>> {\"$label\"} mmlu={mmlu} gsm8k={gsm}')
"
    else
        printf '>> %s NO summary.json\n' "$label"
    fi
    sleep 3  # let GPU release
}

# Order: base first as sanity, then ckpts in order
eval_one "base_0p8b" ""
for step in 010 020 030 040 050 060; do
    ckpt="$RUN_DIR/student/step_000${step}"
    if [[ -d "$ckpt" ]]; then
        eval_one "step_000${step}" "$ckpt"
    fi
done
if [[ -d "$RUN_DIR/student/final" ]]; then
    eval_one "final" "$RUN_DIR/student/final"
fi

# Print combined table
printf '\n══════════ V3 BENCH SUMMARY ══════════\n'
printf '| %-15s | %8s | %8s |\n' 'ckpt' 'mmlu' 'gsm8k'
printf '| %-15s | %8s | %8s |\n' '---------------' '--------' '--------'
for d in "$OUT_BASE"/*/; do
    name=$(basename "$d")
    if [[ -f "$d/summary.json" ]]; then
        line=$(python3 -c "
import json
d = json.load(open('$d/summary.json'))
mmlu = d['tasks'].get('mmlu', {}).get('accuracy', 0)
gsm = d['tasks'].get('gsm8k', {}).get('accuracy', 0)
print(f'{mmlu:.4f} {gsm:.4f}')
")
        mmlu=$(echo "$line" | awk '{print $1}')
        gsm=$(echo "$line" | awk '{print $2}')
        printf '| %-15s | %8s | %8s |\n' "$name" "$mmlu" "$gsm"
    fi
done
