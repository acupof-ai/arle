#!/usr/bin/env bash
# Isolation unit test: device ring kernel vs host reference at the failing
# GQA hd128 zigzag config. Single GPU, no NCCL. Mirrors pod-build-env.sh.
# Detached + TEST_EXIT marker. Ops-only: touches no source.
set -uo pipefail
export TREE="${POD_TREE:-/host/arle-build}"
LOG="/host/nd-ring-test.log"
: > "$LOG"
exec >"$LOG" 2>&1
# shellcheck disable=SC1091
source "$TREE/scripts/pod-build-env.sh"
# shellcheck disable=SC1091
source "$TREE/scripts/cuda_prebuilt_manifest.sh" 2>/dev/null || true
cd "$TREE" || { echo "TEST_EXIT=97"; exit 97; }
export ARLE_CARGO_WORKSPACE_ROOT="$TREE"
export CUDA_VISIBLE_DEVICES=1   # dodge the foreign occupant on GPU0
flock /tmp/arle-toolchain.lock bash -c 'toolchain_dir="${ARLE_RUST_TOOLCHAIN_DIR:-/root/.rustup/toolchains/1.95.0-x86_64-unknown-linux-gnu}"; [ -x "$toolchain_dir/bin/rustc" ] && ls "$toolchain_dir"/lib/rustlib/*/lib/libstd-*.rlib >/dev/null 2>&1 || rustup toolchain install 1.95.0 --profile minimal -c rustfmt -c clippy'
exec 9>"/tmp/arle-build_host_arle-build.lock"
flock 9
cargo test -p autograd --release --no-default-features --features cuda,nccl \
  device_ring_two_blocks_matches_host_reference_gqa_hd128 -- --nocapture
rc=$?
echo "TEST_EXIT=$rc"
exit "$rc"
