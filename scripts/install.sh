#!/bin/sh
# ARLE installer.
#
# Usage:
#   curl -fsSL https://github.com/cklxx/arle/releases/latest/download/install.sh | sh
#
# Environment overrides:
#   ARLE_VERSION   Tag to install (default: latest). Example: v0.1.0
#   INSTALL_DIR    Where to drop the binary (default: $HOME/.local/bin).
#   ARLE_NO_VERIFY If set, skip SHA256SUMS verification.

set -eu

REPO="cklxx/arle"
VERSION="${ARLE_VERSION:-latest}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/bin}"

err() { printf 'error: %s\n' "$*" >&2; exit 1; }
info() { printf '==> %s\n' "$*"; }

need() { command -v "$1" >/dev/null 2>&1 || err "missing required command: $1"; }
need curl
need tar
need uname

OS="$(uname -s)"
ARCH="$(uname -m)"
case "${OS}-${ARCH}" in
  Darwin-arm64)        PLATFORM="macos-arm64"; BINARIES="arle metal_serve" ;;
  Linux-x86_64|Linux-amd64) PLATFORM="linux-x86_64"; BINARIES="arle infer bench_serving" ;;
  *) err "unsupported platform: ${OS}-${ARCH} (supported: Darwin-arm64, Linux-x86_64)" ;;
esac

# Resolve "latest" via GH's redirect so we don't need jq.
if [ "$VERSION" = "latest" ]; then
  VERSION="$(curl -fsSLI -o /dev/null -w '%{url_effective}' \
    "https://github.com/${REPO}/releases/latest" \
    | sed -E 's|.*/tag/||')"
  [ -n "$VERSION" ] || err "could not resolve latest release tag"
fi

TARBALL="arle-${VERSION}-${PLATFORM}.tar.gz"
BASE_URL="https://github.com/${REPO}/releases/download/${VERSION}"

info "Installing ARLE ${VERSION} for ${PLATFORM} into ${INSTALL_DIR}"

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

info "Downloading ${TARBALL}"
curl -fsSL -o "${TMPDIR}/${TARBALL}" "${BASE_URL}/${TARBALL}" \
  || err "failed to download ${BASE_URL}/${TARBALL}"

if [ -z "${ARLE_NO_VERIFY:-}" ]; then
  info "Verifying SHA256"
  curl -fsSL -o "${TMPDIR}/SHA256SUMS.txt" "${BASE_URL}/SHA256SUMS.txt" \
    || err "failed to download SHA256SUMS.txt"
  expected="$(grep " ${TARBALL}\$" "${TMPDIR}/SHA256SUMS.txt" | awk '{print $1}')"
  [ -n "$expected" ] || err "no SHA256 entry for ${TARBALL}"
  if command -v sha256sum >/dev/null 2>&1; then
    actual="$(sha256sum "${TMPDIR}/${TARBALL}" | awk '{print $1}')"
  elif command -v shasum >/dev/null 2>&1; then
    actual="$(shasum -a 256 "${TMPDIR}/${TARBALL}" | awk '{print $1}')"
  else
    err "no sha256sum or shasum available"
  fi
  [ "$expected" = "$actual" ] || err "SHA256 mismatch (expected $expected, got $actual)"
fi

info "Extracting"
tar -xzf "${TMPDIR}/${TARBALL}" -C "${TMPDIR}"

mkdir -p "${INSTALL_DIR}"
for bin in $BINARIES; do
  src="${TMPDIR}/${bin}"
  [ -f "$src" ] || continue
  install -m 0755 "$src" "${INSTALL_DIR}/${bin}"
  info "Installed ${INSTALL_DIR}/${bin}"
done

# Stage MLX kernels next to the macOS binaries. MLX searches for
# `mlx.metallib` colocated with the running binary; without this copy
# `metal_serve` fails with "Failed to load the default metallib".
if [ "$PLATFORM" = "macos-arm64" ] && [ -f "${TMPDIR}/mlx.metallib" ]; then
  install -m 0644 "${TMPDIR}/mlx.metallib" "${INSTALL_DIR}/mlx.metallib"
  info "Installed ${INSTALL_DIR}/mlx.metallib"
fi

case ":${PATH}:" in
  *":${INSTALL_DIR}:"*) ;;
  *)
    cat <<EOF

⚠  ${INSTALL_DIR} is not on your PATH. Add this to your shell profile:

    export PATH="${INSTALL_DIR}:\$PATH"

EOF
    ;;
esac

info "Done. Run: arle --doctor"
