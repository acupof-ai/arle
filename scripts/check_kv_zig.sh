#!/usr/bin/env bash
#
# Local validation helper for the Zig-backed kv substrate work.
#
# Usage:
#   scripts/check_kv_zig.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

info() { echo "[check_kv_zig] $*"; }

info "validating Zig toolchain"
./scripts/setup_zig_toolchain.sh

info "cargo check -p kv-native-sys"
cargo check -p kv-native-sys

info "cargo clippy -p kv-native-sys -- -D warnings"
cargo clippy -p kv-native-sys -- -D warnings

info "cargo check -p infer --no-default-features --features no-cuda"
cargo check -p infer --no-default-features --features no-cuda

info "cargo check -p infer --no-default-features --features metal"
cargo check -p infer --no-default-features --features metal

info "cargo clippy -p infer --no-default-features --features no-cuda -- -D warnings"
cargo clippy -p infer --no-default-features --features no-cuda -- -D warnings

info "cargo test -p infer --no-default-features --features no-cuda kv_tier::transport::disk -- --nocapture"
cargo test -p infer --no-default-features --features no-cuda kv_tier::transport::disk -- --nocapture

info "OK"
