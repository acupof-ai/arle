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

run cargo fmt --manifest-path infer/Cargo.toml --all -- --check
run env CHECK_KV_ZIG_SCOPE=kv-only ./scripts/check_kv_zig.sh
run cargo check --manifest-path infer/Cargo.toml --no-default-features --features no-cuda --lib
run cargo clippy --manifest-path infer/Cargo.toml --no-default-features --features no-cuda --lib -- -D warnings
run cargo test --manifest-path infer/Cargo.toml --no-default-features --features no-cuda --lib
run python -m pytest tests/ -v

if [[ "$(uname -s)" == "Darwin" ]]; then
    run cargo check --manifest-path infer/Cargo.toml --no-default-features --features metal,no-cuda --lib --release
    run cargo check --no-default-features --features metal,no-cuda,cli -p agent-infer --release
    run cargo test --manifest-path infer/Cargo.toml --no-default-features --features metal,no-cuda --lib --release -- --test-threads 1
    run cargo test --no-default-features --features metal,no-cuda,cli -p agent-infer --release
    run cargo build --manifest-path infer/Cargo.toml --no-default-features --features metal,no-cuda --bin metal_serve --release
    run cargo build --manifest-path infer/Cargo.toml --no-default-features --features metal,no-cuda --bin metal_bench --release
    run python3 -c "import json; json.load(open('benchmarks/metal_baseline.json'))"
else
    info "skipping Metal-only checks on non-macOS host"
fi

info "all pre-push checks passed"
