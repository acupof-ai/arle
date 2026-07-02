#!/usr/bin/env bash
# TileLang kernel-artifact bundle — GitHub Releases as the registry, the
# gitignored crates/cuda-kernels/generated/ vendored tier as the local endpoint.
# build.rs verifies each kernel's SRC_HASH at consume time, so a stale bundle
# is ignored per-kernel (and the build demands a regen env), never silently used.
#
#   pack           tar generated/ -> arle-kernels-<hash>.tar.gz
#   publish        pack + upload to the rolling release (gh CLI; needs repo write)
#   fetch [tag]    download the bundle and extract into generated/
#
# Producer (env with tilelang — the kernels-publish workflow or the pod):
#   TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" ARLE_KERNEL_VENDOR=1 \
#     cargo build --release --features cuda -p cuda-kernels
#   scripts/kernel_artifacts.sh publish
# Consumer (zero Python):
#   scripts/kernel_artifacts.sh fetch && cargo build --release --features cuda
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GEN="$ROOT/crates/cuda-kernels/generated"
REL="${ARLE_KERNEL_RELEASE_TAG:-kernel-artifacts}"

sha() { command -v sha256sum >/dev/null && sha256sum | cut -c1-16 || shasum -a 256 | cut -c1-16; }
bundle_hash() { find "$GEN" -name meta.txt -exec grep -h '^SRC_HASH=' {} + | sort | sha; }

case "${1:-help}" in
  pack)
    [ -d "$GEN" ] || { echo "no $GEN — build with ARLE_KERNEL_VENDOR=1 first" >&2; exit 1; }
    f="arle-kernels-$(bundle_hash).tar.gz"
    tar -C "$GEN" -czf "$f" .
    echo "$f"
    ;;
  publish)
    f="$("$0" pack)"
    gh release view "$REL" >/dev/null 2>&1 || gh release create "$REL" --prerelease \
      --title "TileLang kernel artifacts (rolling)" \
      --notes "Prebuilt TileLang AOT kernels. Consume: scripts/kernel_artifacts.sh fetch"
    cp "$f" arle-kernels-latest.tar.gz
    gh release upload "$REL" "$f" arle-kernels-latest.tar.gz --clobber
    rm -f arle-kernels-latest.tar.gz
    echo "published $f -> release $REL"
    ;;
  fetch)
    mkdir -p "$GEN"
    tmp="$(mktemp -d)"
    gh release download "${2:-$REL}" -p 'arle-kernels-latest.tar.gz' -D "$tmp" \
      -R "${GITHUB_REPOSITORY:-cklxx/arle}"
    tar -C "$GEN" -xzf "$tmp/arle-kernels-latest.tar.gz"
    rm -rf "$tmp"
    echo "kernel bundle -> $GEN ($(ls "$GEN" | wc -l | tr -d ' ') artifact dirs; build.rs verifies SRC_HASH per kernel)"
    ;;
  *) sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//' ;;
esac
