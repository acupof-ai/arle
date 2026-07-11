#!/usr/bin/env bash
#
# Harvest prebuilt CUDA kernel archives from a built target tree into a
# directory consumable by ARLE_CUDA_KERNELS_PREBUILT_DIR (the build.rs fast
# path that skips ALL nvcc + TileLang AOT work — link-only, ~5s).
#
# This is the missing producer for the existing consumer in
# crates/cuda-kernels/build.rs::link_prebuilt_cuda_artifacts. Typical uses:
#   - freeze kernels for .rs-only iteration loops (pod or local)
#   - tn-push a kernel pack to a pod so its first build skips nvcc entirely
#
# Usage:
#   scripts/export_prebuilt_cuda_kernels.sh <dest-dir> [target-dir] [profile]
#     target-dir defaults to ./target, profile to release.
#
# Consume with:
#   export ARLE_CUDA_KERNELS_PREBUILT_DIR=<dest-dir>
#
# build.rs rejects a pack unless arle-cuda-kernels.manifest byte-matches the
# current canonical source/toolchain manifest and required symbols are present.
# manifest.json records human-readable provenance. See
# errors/2026-05-28-dsv4-flashmla-decode-parity-precond-fail.md for why.

set -euo pipefail

DEST="${1:?usage: export_prebuilt_cuda_kernels.sh <dest-dir> [target-dir] [profile]}"
TARGET_DIR="${2:-target}"
PROFILE="${3:-release}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/cuda_prebuilt_manifest.sh"

# Newest cuda-kernels OUT_DIR that actually contains both archives (several
# fingerprint hashes can coexist after feature/env flips).
SRC_OUT=""
for candidate in $(ls -td "${TARGET_DIR}/${PROFILE}"/build/cuda-kernels-*/out 2>/dev/null); do
    if [[ -f "${candidate}/libkernels_cuda.a" && -f "${candidate}/libtilelang_kernels_aot.a" ]]; then
        SRC_OUT="${candidate}"
        break
    fi
done

if [[ -z "${SRC_OUT}" ]]; then
    echo "error: no cuda-kernels OUT_DIR with libkernels_cuda.a + libtilelang_kernels_aot.a under ${TARGET_DIR}/${PROFILE}/build/" >&2
    echo "       run a CUDA build first: CUDA_HOME=... cargo build --release --features cuda" >&2
    exit 1
fi

mkdir -p "${DEST}"
cp "${SRC_OUT}/libkernels_cuda.a" "${DEST}/"
cp "${SRC_OUT}/libtilelang_kernels_aot.a" "${DEST}/"

SIDECAR=""
if [[ -f "${SRC_OUT}/arle_deepep_sidecar" ]]; then
    cp "${SRC_OUT}/arle_deepep_sidecar" "${DEST}/"
    SIDECAR="arle_deepep_sidecar"
fi

# Provenance manifest — the cache key docs/environment.md tells consumers to
# respect: source tree object + toolkit + SM list + flags.
KERNELS_TREE="$(git rev-parse "HEAD:crates/cuda-kernels" 2>/dev/null || echo unknown)"
HEAD_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
DIRTY="clean"
if [[ -n "$(git status --porcelain -- crates/cuda-kernels 2>/dev/null)" ]]; then
    DIRTY="dirty"
fi
NVCC_VERSION="$("${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1 || true)"

cat > "${DEST}/manifest.json" <<EOF
{
  "exported_at": "$(date -u +%FT%TZ)",
  "head": "${HEAD_SHA}",
  "cuda_kernels_tree": "${KERNELS_TREE}",
  "cuda_kernels_worktree": "${DIRTY}",
  "nvcc_version": "${NVCC_VERSION:-unknown}",
  "torch_cuda_arch_list": "${TORCH_CUDA_ARCH_LIST:-unset (nvidia-smi autodetect)}",
  "source_out_dir": "${SRC_OUT}",
  "sidecar": "${SIDECAR:-none}"
}
EOF

cuda_prebuilt_manifest > "${DEST}/arle-cuda-kernels.manifest"

echo "[export] ${SRC_OUT} -> ${DEST}"
echo "[export] archives: libkernels_cuda.a libtilelang_kernels_aot.a ${SIDECAR}"
echo "[export] manifest: arle-cuda-kernels.manifest (consumer key), manifest.json (provenance)"
echo "[export] consume with: export ARLE_CUDA_KERNELS_PREBUILT_DIR=$(cd "${DEST}" && pwd)"
