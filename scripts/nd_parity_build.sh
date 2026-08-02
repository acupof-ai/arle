#!/usr/bin/env bash
# One-off: build the nd_parallel_parity example on the pod, mirroring
# pod-remote-build.sh's env (pod.sh build can't target --example). Detached +
# BUILD_EXIT marker so it is re-attachable from the log alone.
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
# self-healing toolchain (same guard as pod-remote-build.sh)
flock /tmp/arle-toolchain.lock bash -c 'toolchain_dir="${ARLE_RUST_TOOLCHAIN_DIR:-/root/.rustup/toolchains/1.95.0-x86_64-unknown-linux-gnu}"; [ -x "$toolchain_dir/bin/rustc" ] && ls "$toolchain_dir"/lib/rustlib/*/lib/libstd-*.rlib >/dev/null 2>&1 || rustup toolchain install 1.95.0 --profile minimal -c rustfmt -c clippy'
exec 9>"/tmp/arle-build_host_arle-build.lock"
flock 9
cargo build -p train --release --no-default-features --features cuda,nccl --example nd_parallel_parity
rc=$?
echo "BUILD_EXIT=$rc"
exit "$rc"
