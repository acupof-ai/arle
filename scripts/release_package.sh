#!/usr/bin/env bash
# release_package.sh — produce a tagged Docker image + binary tarball for a release.
#
# Inputs:
#   --version <v0.X.Y>       — release version tag (default: Cargo.toml workspace.version with v prefix)
#   --cuda <X.Y>             — CUDA toolkit version baked into the image (default: 12.8)
#   --os <distro-tag>        — base OS in tag, e.g. ubuntu22.04 (default: ubuntu22.04)
#   --arch <x86_64|aarch64>  — target arch (default: x86_64)
#   --commit <sha>           — short SHA suffix (default: git rev-parse --short HEAD)
#   --no-image               — skip Docker build, only produce binary tarball from current target/
#   --no-tarball             — skip tarball, only build image
#   --push <registry>        — after build, docker push to <registry>/arle-infer:<tag>
#
# Outputs (in `dist/`):
#   arle-infer-<version>-cuda<X.Y>-<os>-<arch>-<sha>.tar.gz
#   arle-infer-<version>-cuda<X.Y>-<os>-<arch>-<sha>.image.tar.gz   (docker save, optional)
#   arle-infer-<version>-cuda<X.Y>-<os>-<arch>-<sha>.sha256
#   arle-infer-<version>-cuda<X.Y>-<os>-<arch>-<sha>.manifest.json
#
# Tag schema:
#   docker:  arle-infer:<version>-cuda<X.Y>-<os>-<arch>-<sha>
#   tarball: arle-infer-<version>-cuda<X.Y>-<os>-<arch>-<sha>.tar.gz
#
# Convention follows nvidia/cuda image tagging: <cuda><runtime|devel>-<distro>
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VERSION="$(grep '^version' Cargo.toml | head -1 | cut -d'"' -f2)"
VERSION="v${VERSION}"
CUDA_VER="12.8"
OS_TAG="ubuntu22.04"
ARCH="x86_64"
COMMIT="$(git rev-parse --short HEAD)"
DO_IMAGE=1
DO_TARBALL=1
PUSH_REGISTRY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version) VERSION="$2"; shift 2;;
    --cuda) CUDA_VER="$2"; shift 2;;
    --os) OS_TAG="$2"; shift 2;;
    --arch) ARCH="$2"; shift 2;;
    --commit) COMMIT="$2"; shift 2;;
    --no-image) DO_IMAGE=0; shift;;
    --no-tarball) DO_TARBALL=0; shift;;
    --push) PUSH_REGISTRY="$2"; shift 2;;
    -h|--help) sed -n '2,28p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

TAG="${VERSION}-cuda${CUDA_VER}-${OS_TAG}-${ARCH}-${COMMIT}"
DIST_DIR="${REPO_ROOT}/dist"
mkdir -p "$DIST_DIR"
DATE_UTC="$(date -u +%FT%TZ)"

write_manifest() {
  cat >"${DIST_DIR}/arle-infer-${TAG}.manifest.json" <<EOF
{
  "version": "${VERSION}",
  "cuda_toolkit_version": "${CUDA_VER}",
  "os": "${OS_TAG}",
  "arch": "${ARCH}",
  "commit": "${COMMIT}",
  "branch": "$(git rev-parse --abbrev-ref HEAD)",
  "build_date_utc": "${DATE_UTC}",
  "rust_toolchain": "$(rustc --version 2>/dev/null | cut -d' ' -f1-2 || echo unknown)",
  "tag": "${TAG}",
  "release_notes_anchors": [
    "docs/experience/wins/2026-05-28-cuda-gap-b-dsv4-route-block-parallel.md",
    "docs/experience/wins/2026-05-28-cuda-gap-c-cheap-fp8-cpasync.md",
    "docs/experience/wins/2026-05-28-int4-kv-two-level-k.md",
    "docs/experience/wins/2026-05-28-gap-a-phase3-mma-kernel-partial.md"
  ]
}
EOF
  echo "[manifest] ${DIST_DIR}/arle-infer-${TAG}.manifest.json"
}

if [[ $DO_TARBALL -eq 1 ]]; then
  echo "== Building binary tarball =="
  if [[ ! -x target/release/infer ]]; then
    echo "[fatal] target/release/infer not present — build first with:"
    echo "  ARLE_CUDA_DISABLE_MARLIN_W4_FP8=1 CUDA_HOME=/path/to/cuda-${CUDA_VER} \\"
    echo "    cargo build --release --features cuda,nccl -p infer --bin infer"
    exit 2
  fi
  TBOX="${DIST_DIR}/arle-infer-${TAG}"
  rm -rf "$TBOX"
  mkdir -p "$TBOX"
  cp target/release/infer "$TBOX/"
  [[ -x target/release/arle ]] && cp target/release/arle "$TBOX/"
  # Ship the LICENSE + a tag-keyed README so the package is self-documenting.
  cp LICENSE "$TBOX/" 2>/dev/null || true
  cat >"${TBOX}/README.md" <<EOF
# arle-infer — release package ${TAG}

- Binary: \`infer\` (CUDA inference server)
- CLI: \`arle\` (if present)
- Built against CUDA ${CUDA_VER} on ${OS_TAG} / ${ARCH}
- Commit: ${COMMIT}
- Build date: ${DATE_UTC}

## Runtime requirements

- NVIDIA driver compatible with CUDA ${CUDA_VER} (see https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/)
- Compatible \`libcuda.so.1\` from the host driver
- libcudart and other CUDA runtime libs at \`/usr/local/cuda-${CUDA_VER}/lib64/\` OR ship statically

## Quick start

\`\`\`bash
./infer serve --port 8000
\`\`\`
EOF
  tar -czf "${TBOX}.tar.gz" -C "$DIST_DIR" "arle-infer-${TAG}"
  rm -rf "$TBOX"
  (cd "$DIST_DIR" && sha256sum "arle-infer-${TAG}.tar.gz" >"arle-infer-${TAG}.sha256" 2>/dev/null \
    || shasum -a 256 "arle-infer-${TAG}.tar.gz" >"arle-infer-${TAG}.sha256")
  echo "[tarball] ${TBOX}.tar.gz"
  echo "[sha256]  ${DIST_DIR}/arle-infer-${TAG}.sha256"
fi

if [[ $DO_IMAGE -eq 1 ]]; then
  echo "== Building Docker image =="
  IMAGE_NAME="arle-infer:${TAG}"
  CUDA_BASE_DEVEL="nvidia/cuda:${CUDA_VER}.0-devel-${OS_TAG}"
  CUDA_BASE_RUNTIME="nvidia/cuda:${CUDA_VER}.0-runtime-${OS_TAG}"
  docker build \
    --build-arg CUDA_IMAGE="${CUDA_BASE_DEVEL}" \
    --tag "${IMAGE_NAME}" \
    --label "org.opencontainers.image.version=${VERSION}" \
    --label "org.opencontainers.image.revision=${COMMIT}" \
    --label "org.opencontainers.image.source=$(git config --get remote.origin.url || echo unknown)" \
    --label "arle.cuda_version=${CUDA_VER}" \
    --label "arle.os=${OS_TAG}" \
    --label "arle.arch=${ARCH}" \
    --label "arle.build_date=${DATE_UTC}" \
    -f Dockerfile . 2>&1 | tail -30
  echo "[image] ${IMAGE_NAME}"
  docker save "${IMAGE_NAME}" | gzip > "${DIST_DIR}/arle-infer-${TAG}.image.tar.gz"
  echo "[image-tarball] ${DIST_DIR}/arle-infer-${TAG}.image.tar.gz"

  if [[ -n "$PUSH_REGISTRY" ]]; then
    REMOTE_TAG="${PUSH_REGISTRY}/arle-infer:${TAG}"
    docker tag "${IMAGE_NAME}" "${REMOTE_TAG}"
    docker push "${REMOTE_TAG}"
    echo "[pushed] ${REMOTE_TAG}"
  fi
fi

write_manifest

echo
echo "== Release artifacts in ${DIST_DIR}/ =="
ls -lh "${DIST_DIR}/arle-infer-${TAG}"*
