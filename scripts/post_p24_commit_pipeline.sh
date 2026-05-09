#!/usr/bin/env bash
# Post-#24-commit orchestration pipeline. Run AFTER codex commits #24
# W4A8 prefill graph capture hoist. Chains the three pre-built actions:
#
#   1. Phase 0v3 5-gate validation (./scripts/validate_p24_phase0v3.sh)
#   2. Phase 2 step 3 RoPE patch apply to qwen3/weights.rs (per
#      docs/plans/2026-05-10-phase2-step3-qwen3-caller-optin-patch.md)
#   3. #37 throughput bench launch (matched-control vs codex baseline 1639ms)
#
# Each step gates the next. Failure at any step → stop + report.
#
# Usage:
#   ./scripts/post_p24_commit_pipeline.sh             # full pipeline
#   ./scripts/post_p24_commit_pipeline.sh --skip-bench
#   ./scripts/post_p24_commit_pipeline.sh --validate-only
#   ./scripts/post_p24_commit_pipeline.sh --apply-patch-only

set -euo pipefail

MODE="${1:-full}"
LOG_DIR="/tmp/post-p24-pipeline-$(date +%s)"
mkdir -p "$LOG_DIR"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

step() { echo -e "${BLUE}━━━ $* ━━━${NC}"; }
ok() { echo -e "${GREEN}✓${NC} $*"; }
fail_exit() { echo -e "${RED}✗${NC} $*" >&2; echo "logs: $LOG_DIR" >&2; exit 1; }
info() { echo -e "${YELLOW}→${NC} $*"; }

# ──────────────────── Step 0: pre-flight ────────────────────
step "Step 0: Pre-flight checks"

git fetch origin main 2>&1 | tail -3
LATEST=$(git log --oneline -1)
info "HEAD: $LATEST"

DIRTY=$(git status --short | grep -v '^??' | wc -l | tr -d ' ')
if [[ "$DIRTY" != "0" ]]; then
    fail_exit "working tree dirty ($DIRTY files modified). Aborting to avoid contaminating commits. Stash or commit first."
fi
ok "Working tree clean"

# Verify codex's expected #24 commit is present (look for the marlin_scratch key in their wins entry)
if ! grep -qE 'marlin_scratch=true|prefill graph capture' docs/experience/wins/*.md 2>/dev/null; then
    info "WARNING: no #24 commit / wins entry detected. Continue anyway? (Ctrl+C to abort)"
    sleep 5
fi

# ──────────────────── Step 1: Phase 0v3 validation ────────────────────
if [[ "$MODE" != "--apply-patch-only" ]]; then
    step "Step 1: Phase 0v3 5-gate validation"
    if [[ "$MODE" == "--validate-only" ]] || [[ "$MODE" == "--skip-bench" ]] || [[ "$MODE" == "full" ]]; then
        BENCH_FLAG=""
        if [[ "$MODE" == "--validate-only" ]] || [[ "$MODE" == "--skip-bench" ]]; then
            BENCH_FLAG="--skip-bench"
        fi
        if ! ./scripts/validate_p24_phase0v3.sh $BENCH_FLAG > "$LOG_DIR/step1-validate.log" 2>&1; then
            fail_exit "Phase 0v3 validation failed; see $LOG_DIR/step1-validate.log"
        fi
        ok "Phase 0v3 5-gate validation PASS"
    fi
fi

if [[ "$MODE" == "--validate-only" ]]; then
    echo
    echo "Validate-only mode → exit. Run without --validate-only for full pipeline."
    exit 0
fi

# ──────────────────── Step 2: Phase 2 step 3 RoPE patch ────────────────────
step "Step 2: Phase 2 step 3 — qwen3/weights.rs RoPE caller opt-in"

# Apply 4 hunks per docs/plans/2026-05-10-phase2-step3-qwen3-caller-optin-patch.md
WEIGHTS_RS="infer/src/model/qwen3/weights.rs"

# Pre-flight: ensure file is at expected baseline (no codex WIP overlap)
if grep -qE 'precompute_rope_with_scaling' "$WEIGHTS_RS"; then
    info "Already patched (precompute_rope_with_scaling found). Skipping."
else
    info "Patching $WEIGHTS_RS (4 hunks)..."
    # Use sed for direct edits (each hunk is small)
    # Hunk 1 + 3: import lines (precompute_rope → precompute_rope_with_scaling)
    sed -i 's/load_tensor_2d_sharded, precompute_rope,/load_tensor_2d_sharded,\n    precompute_rope_with_scaling,/g' "$WEIGHTS_RS"
    sed -i 's/load_tensor_2d_gguf_bf16, precompute_rope,/load_tensor_2d_gguf_bf16,\n            precompute_rope_with_scaling,/g' "$WEIGHTS_RS"

    # Hunk 2 + 4: call site rewrites
    # Need to be careful with multi-line. Use perl for safer multi-line replacement.
    perl -i -0777 -pe 's/precompute_rope\(\&ctx, config\.head_dim, rope_cache_len, config\.rope_theta\)\?/precompute_rope_with_scaling(\n            \&ctx,\n            config.head_dim,\n            rope_cache_len,\n            config.rope_theta,\n            config.rope_scaling.as_ref(),\n        )?/g' "$WEIGHTS_RS"
    perl -i -0777 -pe 's/precompute_rope\(ctx, config\.head_dim, rope_cache_len, config\.rope_theta\)\?/precompute_rope_with_scaling(\n            ctx,\n            config.head_dim,\n            rope_cache_len,\n            config.rope_theta,\n            config.rope_scaling.as_ref(),\n        )?/g' "$WEIGHTS_RS"

    # Verify
    if ! grep -q 'precompute_rope_with_scaling' "$WEIGHTS_RS"; then
        fail_exit "patch apply failed; sed/perl didn't match expected lines"
    fi
fi

# Cargo check
info "Verifying patched build (cargo check no-cuda)..."
if ! cargo check -p infer --no-default-features --features no-cuda > "$LOG_DIR/step2-cargo-check.log" 2>&1; then
    fail_exit "cargo check failed; see $LOG_DIR/step2-cargo-check.log"
fi
ok "Phase 2 step 3 RoPE patch applied + builds"

# Commit (by explicit path)
git add "$WEIGHTS_RS"
git commit -m "feat(infer): qwen3 caller opt-in for RoPE scaling — Phase 2 step 3

M_rope-yarn-scaling Phase 2 step 3 (per docs/plans/2026-05-10-phase2-step3-
qwen3-caller-optin-patch.md): qwen3/weights.rs callers (line 449 safetensors
loader, 750 GGUF loader) opt-in to precompute_rope_with_scaling, passing
config.rope_scaling.as_ref().

Type-direct: Qwen3Config::rope_scaling matches qwen3_spec::RopeScalingConfig
so no conversion shim needed (vs qwen35-spec mirror in cb80829).

Vanilla path bit-equivalent (config.rope_scaling=None default for current
Qwen3-4B); Phase 3 long-ctx YARN bench can now use --model-path
infer/models/Qwen3-4B-yarn-f2.0/ (per setup_qwen3_yarn_config.py).

Auto-applied via post_p24_commit_pipeline.sh after codex's #24 commit landed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>" 2>&1 | tail -3
git push origin main 2>&1 | tail -3
ok "Phase 2 step 3 commit pushed"

if [[ "$MODE" == "--apply-patch-only" ]]; then
    echo
    echo "Apply-patch-only mode → exit."
    exit 0
fi

# ──────────────────── Step 3: #37 throughput bench ────────────────────
if [[ "$MODE" == "--skip-bench" ]]; then
    echo
    echo "Skip-bench mode → exit after Phase 2 step 3."
    exit 0
fi

step "Step 3: #37 throughput bench (matched-control 4k/c=4)"

# Build release first for bench
info "cargo build --release --features cuda..."
if ! env CUDA_HOME=/opt/cuda NVCC_CCBIN=/usr/bin/g++-14 \
    INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
    TORCH_CUDA_ARCH_LIST=8.9 \
    cargo build --release -p infer --features cuda > "$LOG_DIR/step3-build.log" 2>&1; then
    fail_exit "cargo build release failed; see $LOG_DIR/step3-build.log"
fi
ok "release build complete"

# Bench A: graph OFF baseline
info "Bench A: INFER_PREFILL_GRAPH OFF (baseline)..."
if ! INFER_HYBRID_W4A8_PREFILL=1 \
    PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
    scripts/bench_guidellm.sh p37-graph-off-baseline \
    --concurrencies 4 --max-seconds 90 --warmup 15 \
    --data 'prompt_tokens=4096,prompt_tokens_stdev=1,prompt_tokens_min=4096,prompt_tokens_max=4096,output_tokens=256,output_tokens_stdev=1,output_tokens_min=256,output_tokens_max=256' \
    > "$LOG_DIR/step3-bench-off.log" 2>&1; then
    fail_exit "bench A (graph off) failed; see $LOG_DIR/step3-bench-off.log"
fi
ok "Bench A (graph OFF) complete"

# Bench B: graph ON
info "Bench B: INFER_PREFILL_GRAPH=1 (treatment)..."
if ! INFER_PREFILL_GRAPH=1 INFER_HYBRID_W4A8_PREFILL=1 \
    PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
    scripts/bench_guidellm.sh p37-graph-on-treatment \
    --concurrencies 4 --max-seconds 90 --warmup 15 \
    --data 'prompt_tokens=4096,prompt_tokens_stdev=1,prompt_tokens_min=4096,prompt_tokens_max=4096,output_tokens=256,output_tokens_stdev=1,output_tokens_min=256,output_tokens_max=256' \
    > "$LOG_DIR/step3-bench-on.log" 2>&1; then
    fail_exit "bench B (graph on) failed; see $LOG_DIR/step3-bench-on.log"
fi
ok "Bench B (graph ON) complete"

# ──────────────────── Summary ────────────────────
echo
echo "============================================="
echo -e "${GREEN}✓ FULL PIPELINE COMPLETE${NC}"
echo "Logs: $LOG_DIR"
echo
echo "Next: write wins entry comparing bench A vs bench B"
echo "  A (off): bench-output/2026-*p37-graph-off-baseline*/"
echo "  B (on) : bench-output/2026-*p37-graph-on-treatment*/"
echo "License: TTFT 4k/c=4 Δ ≥ +10% σ < 5% n=3 → wins"
echo "         Δ < +5% → KILL with errors entry"
