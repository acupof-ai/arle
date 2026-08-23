#!/usr/bin/env bash
#
# CI-aligned local validation to run before `git push`.
#
# Usage:
#   scripts/pre_push_checks.sh
#
# Speed: HEAD is exported into a STABLE snapshot dir and refreshed with
# rsync --checksum, so unchanged files keep their mtimes and cargo's
# incremental cache (target/pre-push-quick) stays warm across runs. The
# previous mktemp-per-run snapshot changed every source path on every
# push, invalidating all workspace-crate fingerprints — a full cold
# rebuild (~5-8 min) per push, which is why the .githooks/pre-push hook
# got disabled. Warm runs are now sub-minute.
#
# The snapshot lives OUTSIDE the repo on purpose: inside the repo tree,
# git commands run by the hygiene check would discover the parent repo
# and operate on it instead of the snapshot.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SNAPSHOT_ROOT="${TMPDIR:-/tmp}/arle-pre-push-snapshot"
STAGE_ROOT=""

info() { echo "[pre-push] $*"; }

run() {
    info "$*"
    "$@"
}

cleanup() {
    if [[ -n "${STAGE_ROOT}" && -d "${STAGE_ROOT}" ]]; then
        rm -rf "${STAGE_ROOT}"
    fi
}

trap cleanup EXIT

STAGE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/arle-pre-push-stage.XXXXXX")"
info "refreshing HEAD snapshot at ${SNAPSHOT_ROOT}"
git -C "${REPO_ROOT}" archive HEAD | tar -x -C "${STAGE_ROOT}"
mkdir -p "${SNAPSHOT_ROOT}"
# --checksum keeps mtimes of content-identical files untouched (cargo sees
# them as unchanged); --delete drops files removed from HEAD.
rsync -a --delete --checksum "${STAGE_ROOT}/" "${SNAPSHOT_ROOT}/"
cd "${SNAPSHOT_ROOT}"

export CARGO_TERM_COLOR=always
export RUSTFLAGS="-D warnings"
export CARGO_TARGET_DIR="${REPO_ROOT}/target/pre-push-quick"
# cudarc probes the CUDA version at build time; pin it so the cuda,no-cuda
# typecheck works on hosts without nvcc.
export CUDARC_CUDA_VERSION="${CUDARC_CUDA_VERSION:-12080}"

run python3 scripts/check_repo_hygiene.py
run cargo fmt --all -- --check
for test in \
    test_cuda_prebuilt_export.sh \
    test_lever_gate.sh \
    test_kernel_artifact_qualification.sh \
    test_validate_release.sh \
    test_pod_flow.sh; do
    run bash "scripts/tests/${test}"
done
run cargo check -p arle --no-default-features --features cpu,no-cuda,cli --bin arle
run cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib
run cargo test -p chat -p tools -p qwen3-spec -p qwen35-spec -p spec-train -p kv-native-sys --release
run cargo test --release \
    -p infer-core -p infer-server -p infer-plan -p infer-seam \
    -p infer-moe -p infer-topo -p infer-util -p deepseek-spec -p agent
run cargo clippy -p kv-native-sys --release --all-targets -- -D warnings

METAL_CHECKS="${ARLE_PRE_PUSH_METAL:-${AGENT_INFER_PRE_PUSH_METAL:-0}}"

if [[ "${METAL_CHECKS}" == "1" && "$(uname -s)" == "Darwin" ]]; then
    run cargo check -p infer-api --no-default-features --features metal,no-cuda --lib --release
    run cargo build --no-default-features --features metal,no-cuda,cli -p arle --release --bin arle
    # Metal correctness gate: needle ladder on the local 0.8B test model.
    GATE_BIN="${CARGO_TARGET_DIR}/release/arle"
    GATE_MODEL="${REPO_ROOT}/models/Qwen3.5-0.8B-MLX-4bit"
    if [[ -x "$GATE_BIN" && -d "$GATE_MODEL" ]]; then
        info "Metal needle gate (Qwen3.5-0.8B-MLX-4bit, lengths 115/300/446)"
        BIN="$GATE_BIN" MODEL="$GATE_MODEL" \
        GATE_PROFILE=metal LENGTHS=115,300,446 RUNS=1 \
        PORT=18189 LEVER_GATE_ALLOW_NO_BASELINE=1 LEVER_GATE_SKIP_TEMP=1 LEVER_GATE_SKIP_CONCURRENT=1 \
        RUST_LOG=warn \
        bash scripts/lever_gate.sh "prepush-$$" || {
            echo "[pre-push] Metal needle gate FAIL" >&2
            exit 1
        }
    else
        info "skipping Metal needle gate (binary or model missing)"
    fi
elif [[ "${METAL_CHECKS}" == "1" ]]; then
    info "skipping Metal-only checks on non-macOS host"
else
    info "skipping Metal checks; set ARLE_PRE_PUSH_METAL=1 (legacy AGENT_INFER_PRE_PUSH_METAL also works) to enable"
fi

info "quick pre-push checks passed"
