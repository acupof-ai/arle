#!/usr/bin/env bash
# Final CP-parity confirmation on the recalibrated gate: baseline (seq 16) then
# the 256K liveness gate (seq 131072), both cp=2 on GPUs 1,3 (dodge foreign
# occupant on GPU0). Mirrors pod-build-env.sh. Detached; two exit markers so
# each run is attributable from the log alone.
set -uo pipefail
export TREE="${POD_TREE:-/host/arle-build}"
LOG="/host/nd-final.log"
: > "$LOG"
exec >"$LOG" 2>&1
# shellcheck disable=SC1091
source "$TREE/scripts/pod-build-env.sh"
# shellcheck disable=SC1091
source "$TREE/scripts/cuda_prebuilt_manifest.sh" 2>/dev/null || true
cd "$TREE" || { echo "BASELINE_EXIT=97"; echo "GATE_EXIT=97"; exit 97; }
export ARLE_CARGO_WORKSPACE_ROOT="$TREE"
export ARLE_ND_CUDA_DEVICES=1,3
flock /tmp/arle-toolchain.lock bash -c 'toolchain_dir="${ARLE_RUST_TOOLCHAIN_DIR:-/root/.rustup/toolchains/1.95.0-x86_64-unknown-linux-gnu}"; [ -x "$toolchain_dir/bin/rustc" ] && ls "$toolchain_dir"/lib/rustlib/*/lib/libstd-*.rlib >/dev/null 2>&1 || rustup toolchain install 1.95.0 --profile minimal -c rustfmt -c clippy'
exec 9>"/tmp/arle-build_host_arle-build.lock"
flock 9

echo "=== build once ==="
cargo build -p train --release --no-default-features --features cuda,nccl --example nd_parallel_parity
bexit=$?
echo "BUILD_EXIT=$bexit"
[ "$bexit" -eq 0 ] || { echo "BASELINE_EXIT=$bexit"; echo "GATE_EXIT=$bexit"; exit "$bexit"; }
BIN="$TREE/target/release/examples/nd_parallel_parity"

echo "=== BASELINE (seq=16, cp=2, devices 1,3) ==="
ARLE_ND_DIR="/host/arle_nd_final_baseline_$$" "$BIN"
echo "BASELINE_EXIT=$?"

echo "=== GATE (seq=131072, cp=2, devices 1,3) ==="
ARLE_ND_SEQ=131072 ARLE_ND_DIR="/host/arle_nd_final_gate_$$" "$BIN"
echo "GATE_EXIT=$?"
