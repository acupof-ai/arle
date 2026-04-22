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

SNAPSHOT_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/agent-infer-pre-push.XXXXXX")"
info "exporting HEAD snapshot to ${SNAPSHOT_ROOT}"
git -C "${REPO_ROOT}" archive HEAD | tar -x -C "${SNAPSHOT_ROOT}"
cd "${SNAPSHOT_ROOT}"

export ZIG
ZIG="$(./scripts/setup_zig_toolchain.sh --print-zig)"
info "using ZIG=${ZIG}"

export CARGO_TERM_COLOR=always
export RUSTFLAGS="-D warnings"
export CARGO_TARGET_DIR="${REPO_ROOT}/target/pre-push-quick"

run cargo fmt --manifest-path infer/Cargo.toml --all -- --check
run cargo check --manifest-path infer/Cargo.toml --no-default-features --features no-cuda --lib

if [[ "$(uname -s)" == "Darwin" ]]; then
    run cargo check --manifest-path infer/Cargo.toml --no-default-features --features metal,no-cuda --lib --release
    run cargo check --no-default-features --features metal,no-cuda,cli -p agent-infer --release
else
    info "skipping Metal-only checks on non-macOS host"
fi

info "quick pre-push checks passed"
