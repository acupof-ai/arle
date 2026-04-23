#!/usr/bin/env bash
#
# CI-aligned local validation to run before `git push`.
#
# Usage:
#   scripts/pre_push_checks.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SNAPSHOT_ROOT=""

info() { echo "[pre-push] $*"; }

run() {
    info "$*"
    "$@"
}

cleanup() {
    if [[ -n "${SNAPSHOT_ROOT}" && -d "${SNAPSHOT_ROOT}" ]]; then
        rm -rf "${SNAPSHOT_ROOT}"
    fi
}

trap cleanup EXIT

SNAPSHOT_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/arle-pre-push.XXXXXX")"
info "exporting HEAD snapshot to ${SNAPSHOT_ROOT}"
git -C "${REPO_ROOT}" archive HEAD | tar -x -C "${SNAPSHOT_ROOT}"
cd "${SNAPSHOT_ROOT}"

export ZIG
ZIG="$(./scripts/setup_zig_toolchain.sh --print-zig)"
info "using ZIG=${ZIG}"

export CARGO_TERM_COLOR=always
export RUSTFLAGS="-D warnings"
export CARGO_TARGET_DIR="${REPO_ROOT}/target/pre-push-quick"

run python3 scripts/check_repo_hygiene.py
run cargo fmt --manifest-path infer/Cargo.toml --all -- --check
run cargo check --manifest-path infer/Cargo.toml --no-default-features --features no-cuda --lib
run cargo check -p agent-infer --no-default-features --features cpu,no-cuda,cli --bin arle
run cargo test -p chat -p tools -p qwen3-spec -p qwen35-spec -p kv-native-sys --release
run cargo clippy --manifest-path infer/Cargo.toml --no-default-features --features no-cuda --lib -- -D warnings

METAL_CHECKS="${ARLE_PRE_PUSH_METAL:-${AGENT_INFER_PRE_PUSH_METAL:-0}}"

if [[ "${METAL_CHECKS}" == "1" && "$(uname -s)" == "Darwin" ]]; then
    run cargo check --manifest-path infer/Cargo.toml --no-default-features --features metal,no-cuda --lib --release
    run cargo check --no-default-features --features metal,no-cuda,cli -p agent-infer --release --bin arle
elif [[ "${METAL_CHECKS}" == "1" ]]; then
    info "skipping Metal-only checks on non-macOS host"
else
    info "skipping Metal checks; set ARLE_PRE_PUSH_METAL=1 (legacy AGENT_INFER_PRE_PUSH_METAL also works) to enable"
fi

info "quick pre-push checks passed"
