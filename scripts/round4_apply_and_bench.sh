#!/usr/bin/env bash
# Round 4 #6 — Marlin hybrid dispatch one-shot apply + 3-arm A/B + multi-shape defense.
#
# Plan: docs/plans/M_quant-marlin-round4-hybrid-dispatch.md (8adc1e1).
# Trigger: codex's W4A8 substrate commit lands (linear.rs leaves WIP).
#
# This script:
#   1. Verifies linear.rs is clean (codex W4A8 already committed)
#   2. Applies the dispatch threshold edit (sed-safe, single-line constant)
#   3. Runs cargo build --release -p infer --features cuda
#   4. Runs cargo test --release greedy_consistency (correctness gate)
#   5. Starts ARLE serving (auto-FP8 KV, no --kv-cache-dtype override per
#      skill v1.2.0 isolation-motive callout)
#   6. Runs 4 benches:
#        - primary: longctx 4k/c=4 (Phase 5)
#        - defense 1: high-conc 1k/256/c=64
#        - defense 2: multi-tenant prefix-cache
#        - defense 3: longctx 8k/c=4
#   7. Kills ARLE, assembles result summary
#
# License-or-kill decision (per Phase 8) is Claude's manual call after reading
# results — this script does NOT auto-LAND or auto-revert.
#
# Usage:
#   scripts/round4_apply_and_bench.sh
#   scripts/round4_apply_and_bench.sh --dry-run    # print plan without executing
#   scripts/round4_apply_and_bench.sh --skip-baselines  # skip Arm A + B re-bench
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DRY_RUN=false
SKIP_BASELINES=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --skip-baselines) SKIP_BASELINES=true ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

run() {
    echo ">>> $*"
    if [[ "$DRY_RUN" != true ]]; then
        eval "$@"
    fi
}

# Step 1 — Gate check: linear.rs must NOT be in WIP (codex's W4A8 must have committed)
echo "=== Step 1: Conflict gate check ==="
if git status --short | grep -q "infer/src/ops/linear.rs"; then
    echo "GATE STILL ACTIVE: infer/src/ops/linear.rs is in WIP." >&2
    echo "       Wait for codex to commit + push W4A8 substrate." >&2
    exit 3
fi

CURRENT_BRANCH=$(git branch --show-current)
if [[ "$CURRENT_BRANCH" != "main" ]]; then
    echo "WARNING: not on main (branch=$CURRENT_BRANCH); aborting." >&2
    exit 4
fi

# Step 2 — Pull latest (codex's W4A8 commit should be on origin/main)
echo "=== Step 2: Sync with origin ==="
run "git pull --rebase --autostash origin main"

# Step 3 — Apply the threshold edit
echo "=== Step 3: Apply MARLIN_DECODE_BATCH_THRESHOLD edit ==="
LINEAR_RS="infer/src/ops/linear.rs"
if grep -q "MARLIN_DECODE_BATCH_THRESHOLD" "$LINEAR_RS"; then
    echo "Already applied (or codex W4A8 included it). Skipping edit."
else
    # Insert constant before fn batched (line 65 in current head)
    # Insert dispatch threshold guard inside fn batched.
    # Use a marker-aware patch via Python rather than fragile sed.
    if [[ "$DRY_RUN" != true ]]; then
        python3 <<'PY' || { echo "PATCH FAILED — re-inspect linear.rs (codex W4A8 may have renamed marlin_prefill_aligned)" >&2; exit 6; }
import re, pathlib, sys
p = pathlib.Path("infer/src/ops/linear.rs")
src = p.read_text()
orig = src

# Add constant before fn batched
src, n_const = re.subn(
    r"(    fn batched\(weight: &DeviceMatrix, batch: usize\) -> Self \{)",
    "const MARLIN_DECODE_BATCH_THRESHOLD: usize = 8;\n\n\\1",
    src, count=1)
if n_const != 1:
    print(f"FATAL: could not locate `fn batched(...)` to insert constant", file=sys.stderr)
    sys.exit(1)

# Replace marlin gate(s) — match any marlin_*aligned function name to be
# resilient against codex potential W4A8 dispatch addition + renames.
# Match: "if batch > 1 && marlin_<...>aligned(weight).is_ok() {"
gate_pat = r"if batch > 1 && (marlin\w*aligned)\(weight\)\.is_ok\(\) \{"
matches = re.findall(gate_pat, src)
if not matches:
    print(f"FATAL: no `if batch > 1 && marlin_*aligned(weight).is_ok()` gates found.", file=sys.stderr)
    print(f"       codex W4A8 dispatch may have changed shape; manual edit required.", file=sys.stderr)
    sys.exit(1)

src = re.sub(
    gate_pat,
    r"if batch > MARLIN_DECODE_BATCH_THRESHOLD\n            && \1(weight).is_ok()\n        {",
    src)

if src == orig:
    print("FATAL: regex matched but no substitution happened", file=sys.stderr)
    sys.exit(1)

p.write_text(src)
print(f"EDIT APPLIED: 1 constant + {len(matches)} marlin gate(s) patched ({matches})")
PY
    fi
fi

# Step 4 — Build
echo "=== Step 4: cargo build --release -p infer --features cuda ==="
run "CUDA_HOME=/opt/cuda TORCH_CUDA_ARCH_LIST=8.9 NVCC_CCBIN=/usr/bin/g++-14 \
     INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
     cargo build --release -p infer --features cuda"

# Step 5 — Correctness gate
echo "=== Step 5: greedy_consistency gate ==="
run "CUDA_HOME=/opt/cuda TORCH_CUDA_ARCH_LIST=8.9 NVCC_CCBIN=/usr/bin/g++-14 \
     INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
     cargo test --release -p infer --features cuda --test greedy_consistency"

# Step 6 — Start ARLE (production-default auto-FP8 KV per skill v1.2.0)
echo "=== Step 6: Start ARLE ==="
MODEL_PATH="infer/models/Qwen3-4B-W4A16-sym-g128-marlin"
PORT=8000
run "CUDA_HOME=/opt/cuda TORCH_CUDA_ARCH_LIST=8.9 \
     ./target/release/infer --model-path $MODEL_PATH \
     --port $PORT --num-slots 8 --max-seq-len 5120 \
     > /tmp/arle-r4-hybrid.log 2>&1 &"
ARLE_PID=$!
echo "ARLE PID: $ARLE_PID"
if [[ "$DRY_RUN" != true ]]; then
    sleep 35
    if ! curl -sS -m 5 "http://localhost:$PORT/v1/models" >/dev/null; then
        echo "ARLE failed to start; check /tmp/arle-r4-hybrid.log" >&2
        kill "$ARLE_PID" 2>/dev/null || true
        exit 5
    fi
fi

# Step 7 — Benches
echo "=== Step 7: 4-bench sequence ==="
COMMON_FLAGS="--model Qwen3-4B-W4A16-sym-g128-marlin \
  --processor $REPO_ROOT/$MODEL_PATH"

# Primary — longctx 4k/c=4
run "PATH=$REPO_ROOT/.venv/bin:\$PATH \
     scripts/bench_guidellm.sh marlin-w4a16-r4-hybrid-c4-4k $COMMON_FLAGS \
     --concurrencies 4 --max-seconds 120 --warmup 10 \
     --data 'prompt_tokens=4096,prompt_tokens_min=4096,prompt_tokens_max=4096,output_tokens=256,output_tokens_min=256,output_tokens_max=256'"

# Defense 1 — high-conc 1k/256/c=64
run "PATH=$REPO_ROOT/.venv/bin:\$PATH \
     scripts/bench_guidellm.sh marlin-w4a16-r4-defense-highconc $COMMON_FLAGS \
     --concurrencies 64 --max-seconds 120 --warmup 10 \
     --data 'prompt_tokens=1024,prompt_tokens_min=1024,prompt_tokens_max=1024,output_tokens=256,output_tokens_min=256,output_tokens_max=256'"

# Defense 2 — multi-tenant prefix-cache
run "PATH=$REPO_ROOT/.venv/bin:\$PATH \
     python scripts/bench_kv_cache_prefix.py --target http://localhost:$PORT \
     --concurrencies 4 --shared-prefix-tokens 2048 --tail-tokens 64 \
     --max-seconds 120 --warmup 10"

# Defense 3 — longctx 8k/c=4
run "PATH=$REPO_ROOT/.venv/bin:\$PATH \
     scripts/bench_guidellm.sh marlin-w4a16-r4-defense-longctx8k $COMMON_FLAGS \
     --concurrencies 4 --max-seconds 180 --warmup 10 \
     --data 'prompt_tokens=8192,prompt_tokens_min=8192,prompt_tokens_max=8192,output_tokens=256,output_tokens_min=256,output_tokens_max=256'"

# Step 8 — Kill ARLE
echo "=== Step 8: Kill ARLE ==="
run "kill $ARLE_PID 2>/dev/null || true"
if [[ "$DRY_RUN" != true ]]; then
    sleep 2
fi

# Step 9 — Result summary
echo "=== Step 9: Result summary ==="
echo
echo "Primary (Phase 5, longctx 4k/c=4):"
[[ "$DRY_RUN" != true ]] && head -3 "bench-output/$(date +%Y-%m-%d)-marlin-w4a16-r4-hybrid-c4-4k/headline_table.md" 2>/dev/null || echo "(dry-run)"
echo
echo "Defense 1 (high-conc 1k/256/c=64):"
[[ "$DRY_RUN" != true ]] && head -3 "bench-output/$(date +%Y-%m-%d)-marlin-w4a16-r4-defense-highconc/headline_table.md" 2>/dev/null || echo "(dry-run)"
echo
echo "Defense 3 (longctx 8k/c=4):"
[[ "$DRY_RUN" != true ]] && head -3 "bench-output/$(date +%Y-%m-%d)-marlin-w4a16-r4-defense-longctx8k/headline_table.md" 2>/dev/null || echo "(dry-run)"
echo
echo "License-or-kill decision is Claude's manual call:"
echo "  - Phase 8 thresholds: see docs/plans/M_quant-marlin-round4-hybrid-dispatch.md"
echo "  - Multi-shape defense gates: see same plan §Multi-shape defense gate"
echo
echo "If LAND: write wins entry, commit linear.rs change + wins, push."
echo "If KILL: revert linear.rs, write errors entry citing the failed phase, commit, push."
echo "If KILL by NULL band: try Phase 6 threshold sweep (4 / 16 / 32) before final KILL."
