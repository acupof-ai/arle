#!/usr/bin/env bash
# Clean the stale cuda-kernels build artifacts (persisted target/ across syncs
# left a partial out/ tree), then rebuild the parity example. Kernel object
# cache (/host/arle-kernel-cache/objects) stays warm, so regen is a fast copy.
# Detached + BUILD_EXIT marker. Ops-only: touches no source.
set -uo pipefail
export TREE="${POD_TREE:-/host/arle-build}"
LOG="/host/nd-parity-build.log"
: > "$LOG"
exec >"$LOG" 2>&1
# shellcheck disable=SC1091
source "$TREE/scripts/pod-build-env.sh"
# shellcheck disable=SC1091
source "$TREE/scripts/cuda_prebuilt_manifest.sh" 2>/dev/null || true
cd "$TREE" || { echo "BUILD_EXIT=97"; exit 97; }
export ARLE_CARGO_WORKSPACE_ROOT="$TREE"
flock /tmp/arle-toolchain.lock bash -c 'toolchain_dir="${ARLE_RUST_TOOLCHAIN_DIR:-/root/.rustup/toolchains/1.95.0-x86_64-unknown-linux-gnu}"; [ -x "$toolchain_dir/bin/rustc" ] && ls "$toolchain_dir"/lib/rustlib/*/lib/libstd-*.rlib >/dev/null 2>&1 || rustup toolchain install 1.95.0 --profile minimal -c rustfmt -c clippy'
exec 9>"/tmp/arle-build_host_arle-build.lock"
flock 9
echo "=== cargo clean -p cuda-kernels (drop stale out/ trees) ==="
cargo clean -p cuda-kernels --release
echo "=== rebuild example ==="
cargo build -p train --release --no-default-features --features cuda,nccl --example nd_parallel_parity
rc=$?
echo "BUILD_EXIT=$rc"
exit "$rc"
