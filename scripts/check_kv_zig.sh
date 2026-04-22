#!/usr/bin/env bash
#
# Local validation helper for the Zig-backed kv substrate work.
#
# Usage:
#   scripts/check_kv_zig.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
SCOPE="${CHECK_KV_ZIG_SCOPE:-full}"

info() { echo "[check_kv_zig] $*"; }

case "${SCOPE}" in
    full | kv-only) ;;
    *)
        echo "[check_kv_zig] unsupported CHECK_KV_ZIG_SCOPE=${SCOPE}" >&2
        exit 64
        ;;
esac

info "validating Zig toolchain"
export ZIG
ZIG="$(./scripts/setup_zig_toolchain.sh --print-zig)"
info "using ZIG=${ZIG}"
info "scope=${SCOPE}"

info "cargo check -p kv-native-sys"
cargo check -p kv-native-sys

info "cargo test -p kv-native-sys"
cargo test -p kv-native-sys

info "cargo clippy -p kv-native-sys -- -D warnings"
cargo clippy -p kv-native-sys -- -D warnings

if [[ "${SCOPE}" == "full" ]]; then
    info "cargo check -p infer --no-default-features --features no-cuda"
    cargo check -p infer --no-default-features --features no-cuda

    info "cargo check -p infer --no-default-features --features metal"
    cargo check -p infer --no-default-features --features metal

    info "cargo clippy -p infer --no-default-features --features no-cuda -- -D warnings"
    cargo clippy -p infer --no-default-features --features no-cuda -- -D warnings

    info "cargo test -p infer --no-default-features --features no-cuda kv_tier::transport::disk -- --nocapture"
    cargo test -p infer --no-default-features --features no-cuda kv_tier::transport::disk -- --nocapture
fi

info "OK"
