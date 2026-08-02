#!/usr/bin/env bash
# One-off compile-only gate for the new marlin_w8a16 kernel. Detached build:
# sources the canonical pod-build-env.sh, emits BUILD_EXIT markers, re-attachable log.
export TREE=/host/arle-build
cd "$TREE" || { echo "MARLIN_BUILD_FATAL: cannot cd $TREE"; echo "FINAL_BUILD_EXIT step1=99 step2=99"; exit 99; }
# Idempotent toolchain guard (mirrors pod-remote-build.sh line 189).
flock /tmp/arle-toolchain.lock bash -c 'd="${ARLE_RUST_TOOLCHAIN_DIR:-/root/.rustup/toolchains/1.95.0-x86_64-unknown-linux-gnu}"; [ -x "$d/bin/rustc" ] && ls "$d"/lib/rustlib/*/lib/libstd-*.rlib >/dev/null 2>&1 || rustup toolchain install 1.95.0 --profile minimal -c rustfmt -c clippy'
# shellcheck disable=SC1091
source "$TREE/scripts/pod-build-env.sh"
export ARLE_CARGO_WORKSPACE_ROOT="$TREE"
echo "=== ENV cargo=$(command -v cargo) INFER_TILELANG_PYTHON=${INFER_TILELANG_PYTHON:-<unset>} RUSTC_WRAPPER=${RUSTC_WRAPPER:-<unset>} ==="
echo "=== STEP1 $(date -u +%FT%TZ): cargo build --release --features cuda -p cuda-kernels ==="
cargo build --release --features cuda -p cuda-kernels
rc1=$?
echo "STEP1_BUILD_EXIT=$rc1"
rc2=127
if [ "$rc1" -eq 0 ]; then
  echo "=== STEP2 $(date -u +%FT%TZ): cargo build --release --features cuda --example marlin_w8a16_parity -p infer-cuda ==="
  cargo build --release --features cuda --example marlin_w8a16_parity -p infer-cuda
  rc2=$?
  echo "STEP2_BUILD_EXIT=$rc2"
  if [ "$rc2" -eq 0 ]; then
    bin="$TREE/target/release/examples/marlin_w8a16_parity"
    if [ -x "$bin" ]; then echo "PARITY_BIN_PRESENT=$bin"; else echo "PARITY_BIN_MISSING=$bin"; rc2=98; fi
  fi
else
  echo "STEP2_SKIPPED_STEP1_FAILED"
fi
echo "FINAL_BUILD_EXIT step1=$rc1 step2=$rc2 $(date -u +%FT%TZ)"
