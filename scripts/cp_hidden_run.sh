#!/usr/bin/env bash
# CP hidden-state parity: full forward per shard vs single-card, 2-rank on GPUs
# 1,3 (dodge foreign occupant on GPU0). The example spawns its own 2 child
# ranks. Mirrors pod-build-env.sh. Detached + RUN_EXIT marker.
set -uo pipefail
export TREE="${POD_TREE:-/host/arle-build}"
LOG="/host/cp-hidden.log"
: > "$LOG"
exec >"$LOG" 2>&1
# shellcheck disable=SC1091
source "$TREE/scripts/pod-build-env.sh"
# shellcheck disable=SC1091
source "$TREE/scripts/cuda_prebuilt_manifest.sh" 2>/dev/null || true
cd "$TREE" || { echo "RUN_EXIT=97"; exit 97; }
export ARLE_CARGO_WORKSPACE_ROOT="$TREE"
export ARLE_CPH_CUDA_DEVICES=1,3
flock /tmp/arle-toolchain.lock bash -c 'toolchain_dir="${ARLE_RUST_TOOLCHAIN_DIR:-/root/.rustup/toolchains/1.95.0-x86_64-unknown-linux-gnu}"; [ -x "$toolchain_dir/bin/rustc" ] && ls "$toolchain_dir"/lib/rustlib/*/lib/libstd-*.rlib >/dev/null 2>&1 || rustup toolchain install 1.95.0 --profile minimal -c rustfmt -c clippy'
exec 9>"/tmp/arle-build_host_arle-build.lock"
flock 9
cargo run -p train --release --no-default-features \
  --features cuda,nccl --example cp_hidden_parity
rc=$?
echo "RUN_EXIT=$rc"
exit "$rc"
