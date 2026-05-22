#!/usr/bin/env bash
# Local pre-push smoke that mirrors the GitHub CI commands.
#
# Why: the cudarc-in-CPU-build leak (root cause #1 in
# docs/experience/errors/2026-05-22-ci-cudarc-leak-and-hygiene-drift.md)
# only surfaced on a fresh CI machine because local target/ already had
# cudarc built. This script uses an isolated CARGO_TARGET_DIR so the
# leak (and others like it) become visible locally.
#
# Run before push:   bash scripts/ci-smoke.sh
# Cleanup target:    rm -rf /tmp/arle-ci-smoke
#
# What it checks (in order, fail-fast):
#   1. python3 scripts/check_repo_hygiene.py
#   2. scoped fmt:  bash scripts/ci-fmt-check-changed.sh
#   3. infer CPU-only check  (the surface that caught cudarc leak)
#   4. infer CPU-only tests
#   5. workspace clippy (CPU-only features)
#   6. workspace tests (cli + agent-infer cpu smoke + support crates)
#
# Skipped vs GitHub CI:
#   - macOS Metal check (needs macOS-arm64)
#   - GitHub Actions caching (we use a stable local target dir instead)
#   - Cargo Deny (separate workflow, independent)
#
# Timing reference (on a warm cargo registry / cold local target):
#   - Steps 1-2:        ~5 s
#   - Step 3 (check):   ~30-60 s
#   - Steps 4-5-6:      ~3-6 min combined
# Total cold:           ~5-8 min
# Total warm:           ~30 s

set -euo pipefail

export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/tmp/arle-ci-smoke}"
export CARGO_TERM_COLOR="${CARGO_TERM_COLOR:-always}"
export RUSTFLAGS="${RUSTFLAGS:--D warnings}"

cd "$(dirname "$0")/.."

step() {
    printf '\n──── %s ────\n' "$1"
}

step "1/6 repo hygiene"
python3 scripts/check_repo_hygiene.py

step "2/6 scoped fmt check"
bash scripts/ci-fmt-check-changed.sh

step "3/6 cargo check (infer CPU-only)"
(cd infer && cargo check --no-default-features --features no-cuda --lib)

step "4/6 cargo test (infer CPU-only --lib --no-run)"
# --no-run avoids actually running CUDA-gated tests on a CPU box. The compile
# step is what caught the cudarc dev-dep leak.
(cd infer && cargo test --no-default-features --features no-cuda --lib --no-run)

step "5/6 cargo clippy (workspace CPU-only surface)"
(cd infer && cargo clippy --no-default-features --features no-cuda --lib -- -D warnings)
cargo clippy -p cli --release --no-default-features --features no-cuda -- -D warnings
cargo clippy -p agent-infer --no-default-features --features cpu,no-cuda,cli --bin arle -- -D warnings

step "6/6 cargo test (workspace support crates, --no-run for speed)"
cargo test -p autograd --release --features no-cuda --lib --no-run
cargo test -p train --release --features no-cuda --lib --no-run
cargo test -p chat -p tools -p qwen3-spec -p qwen35-spec -p kv-native-sys --release --no-run

printf '\nci-smoke: all checks passed ✓\n'
printf 'target dir: %s\n' "$CARGO_TARGET_DIR"
