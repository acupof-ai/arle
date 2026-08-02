#!/usr/bin/env bash
# One-off: build+run the train crate's cp_hidden_parity example on the H20 box.
# pod.sh build is --bin-only; this replicates pod-remote-build.sh's env for an example.
set -uo pipefail
TREE=/host/arle-build
export POD_TREE="$TREE"
TREE_LOCK="/tmp/arle-build$(printf '%s' "$TREE" | tr '/.' '__').lock"

case "${1:-}" in
  build)
    LOG=/host/cph-build.log
    {
      echo "BUILD_START=$(date -u +%FT%TZ)"
      cd "$TREE" || exit 1
      echo "HEAD=$(git rev-parse HEAD)"
      # shellcheck disable=SC1091
      source "$TREE/scripts/pod-build-env.sh"
      export ARLE_CARGO_WORKSPACE_ROOT="$TREE"
      # toolchain self-heal (verbatim from pod-remote-build.sh)
      # shellcheck disable=SC2016
      flock /tmp/arle-toolchain.lock bash -c 'toolchain_dir="${ARLE_RUST_TOOLCHAIN_DIR:-/root/.rustup/toolchains/1.95.0-x86_64-unknown-linux-gnu}"; [ -x "$toolchain_dir/bin/rustc" ] && ls "$toolchain_dir"/lib/rustlib/*/lib/libstd-*.rlib >/dev/null 2>&1 || rustup toolchain install 1.95.0 --profile minimal -c rustfmt -c clippy'
      exec 9>"$TREE_LOCK"; flock 9
      cargo build -p train --release --no-default-features --features cuda,nccl --example cp_hidden_parity
      rc=$?
      echo "BUILD_EXIT=$rc"
    } >"$LOG" 2>&1
    ;;
  run)
    LOG=/host/cph-run.log
    {
      echo "RUN_START=$(date -u +%FT%TZ)"
      cd "$TREE" || exit 1
      echo "HEAD=$(git rev-parse HEAD)"
      # shellcheck disable=SC1091
      source "$TREE/scripts/pod-build-env.sh"
      export ARLE_CARGO_WORKSPACE_ROOT="$TREE"
      export ARLE_CPH_CUDA_DEVICES=1,3
      cargo run -p train --release --no-default-features --features cuda,nccl --example cp_hidden_parity
      rc=$?
      echo "RUN_EXIT=$rc"
    } >"$LOG" 2>&1
    ;;
  *) echo "usage: cph_parity.sh build|run" >&2; exit 2;;
esac
