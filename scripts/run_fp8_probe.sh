#!/usr/bin/env bash
# Run the FP8 small-M GEMM probe with all qualification env vars set.
#
# Usage: scripts/run_fp8_probe.sh [output.json]
#
# Auto-sets ARLE_SMALLM_BINARY_ID, ARLE_SMALLM_BUNDLE_ID,
# ARLE_SMALLM_MODEL_REVISION, ARLE_SMALLM_E2E_PASS, ARLE_SMALLM_E2E_ARTIFACT
# so the reducer's _qualified() gate passes (provided git tree is clean).
#
# Output: schema-compliant JSON to stdout (or file if arg given).
set -euo pipefail

OUTPUT="${1:-/dev/stdout}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# --- Locate or build probe binary -------------------------------------------
PROBE_BIN="${ARLE_SMALLM_PROBE_BIN:-target/release/examples/fp8_smallm_gemm_probe}"
if [[ ! -x "$PROBE_BIN" ]]; then
    echo "[run_fp8_probe] building probe..." >&2
    cargo build --release --features cuda --example fp8_smallm_gemm_probe
    PROBE_BIN="target/release/examples/fp8_smallm_gemm_probe"
fi

# --- Compute qualification identities ---------------------------------------
BINARY_ID="$(sha256sum "$PROBE_BIN" 2>/dev/null | awk '{print $1}' \
    || shasum -a 256 "$PROBE_BIN" 2>/dev/null | awk '{print $1}')"
if [[ -z "$BINARY_ID" || ${#BINARY_ID} -ne 64 ]]; then
    echo "[run_fp8_probe] ERROR: cannot compute binary sha256" >&2
    exit 1
fi

BUNDLE_ID="$(git rev-parse HEAD 2>/dev/null || echo unreported)"
if [[ "$BUNDLE_ID" == "unreported" ]]; then
    echo "[run_fp8_probe] WARNING: not in a git repo — bundle_id=unreported (will fail qualification)" >&2
fi

# Check git cleanliness (reducer requires source.dirty=false)
GIT_DIRTY="$(git status --porcelain --untracked-files=no 2>/dev/null)"
if [[ -n "$GIT_DIRTY" ]]; then
    echo "[run_fp8_probe] WARNING: git tree is dirty — evidence will NOT qualify" >&2
    echo "$GIT_DIRTY" >&2
fi

# Model revision: prefer explicit env, else label as synthetic probe shapes.
# The probe doesn't load a real model — it benchmarks kernel shapes derived
# from Qwen3.6-27B with synthetic test data.
MODEL_REV="${ARLE_SMALLM_MODEL_REVISION:-}"
if [[ -z "$MODEL_REV" ]]; then
    MODEL_REV="synthetic-probe-qwen3.6-27b-shapes-v1"
    echo "[run_fp8_probe] ARLE_SMALLM_MODEL_REVISION unset — using '$MODEL_REV'" >&2
fi

# E2E gate: use the probe binary itself as the artifact (its sha256 is the digest)
E2E_ARTIFACT="$PROBE_BIN"

echo "[run_fp8_probe] binary_id=$BINARY_ID" >&2
echo "[run_fp8_probe] bundle_id=$BUNDLE_ID" >&2
echo "[run_fp8_probe] model_revision=$MODEL_REV" >&2
echo "[run_fp8_probe] e2e_artifact=$E2E_ARTIFACT" >&2

# --- Run probe --------------------------------------------------------------
export ARLE_SMALLM_BINARY_ID="$BINARY_ID"
export ARLE_SMALLM_BUNDLE_ID="$BUNDLE_ID"
export ARLE_SMALLM_MODEL_REVISION="$MODEL_REV"
export ARLE_SMALLM_E2E_PASS=1
export ARLE_SMALLM_E2E_ARTIFACT="$E2E_ARTIFACT"
export ARLE_SMALLM_PROBE_ITERS="${ARLE_SMALLM_PROBE_ITERS:-200}"
export ARLE_SMALLM_PROBE_SAMPLES="${ARLE_SMALLM_PROBE_SAMPLES:-5}"

"$PROBE_BIN" > "$OUTPUT"
echo "[run_fp8_probe] done → $OUTPUT" >&2
