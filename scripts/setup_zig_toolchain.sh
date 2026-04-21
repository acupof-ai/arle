#!/usr/bin/env bash
#
# Install or validate the Zig toolchain used by `crates/kv-native-sys`.
#
# Usage:
#   scripts/setup_zig_toolchain.sh [--print-zig] [--check-only] [EXPECTED_VERSION]
#
# Env:
#   ZIG                   explicit zig binary to validate first
#   KV_ZIG_VERSION        alternate version override
#   KV_ZIG_INSTALL_ROOT   repo-local install root (default: .toolchains/zig)
#   KV_ZIG_DOWNLOAD_BASE  download base (default: https://ziglang.org/download)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INSTALL_ROOT="${KV_ZIG_INSTALL_ROOT:-${REPO_ROOT}/.toolchains/zig}"
DOWNLOAD_BASE="${KV_ZIG_DOWNLOAD_BASE:-https://ziglang.org/download}"
PRINT_ZIG=0
CHECK_ONLY=0
EXPECTED_VERSION="${KV_ZIG_VERSION:-0.16.0}"

info() {
  if [[ "${PRINT_ZIG}" -eq 0 ]]; then
    echo "[setup_zig_toolchain] $*"
  fi
}

die() {
  echo "[setup_zig_toolchain] error: $*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage: scripts/setup_zig_toolchain.sh [--print-zig] [--check-only] [EXPECTED_VERSION]

Options:
  --print-zig   Print the resolved Zig binary path after validation/install.
  --check-only  Validate an existing Zig binary only; do not install.
  -h, --help    Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --print-zig)
      PRINT_ZIG=1
      shift
      ;;
    --check-only)
      CHECK_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      EXPECTED_VERSION="$1"
      shift
      ;;
  esac
done

emit_result() {
  local zig_bin="$1"
  if [[ "${PRINT_ZIG}" -eq 1 ]]; then
    echo "${zig_bin}"
  else
    info "using Zig $("${zig_bin}" version) at ${zig_bin}"
  fi
}

validate_version() {
  local zig_bin="$1"
  local actual
  actual="$("${zig_bin}" version)"
  [[ "${actual}" == "${EXPECTED_VERSION}" ]]
}

pick_zig() {
  if [[ -n "${ZIG:-}" ]] && [[ -x "${ZIG}" ]]; then
    echo "${ZIG}"
    return 0
  fi
  if command -v zig >/dev/null 2>&1; then
    command -v zig
    return 0
  fi
  return 1
}

platform_triple() {
  local os arch
  os="$(uname -s)"
  arch="$(uname -m)"

  case "${os}/${arch}" in
    Darwin/arm64|Darwin/aarch64) echo "aarch64-macos" ;;
    Darwin/x86_64) echo "x86_64-macos" ;;
    Linux/x86_64) echo "x86_64-linux" ;;
    Linux/aarch64|Linux/arm64) echo "aarch64-linux" ;;
    *)
      die "unsupported host platform ${os}/${arch} for Zig auto-install"
      ;;
  esac
}

install_local() {
  local triple archive url install_dir archive_path tmp_dir
  triple="$(platform_triple)"
  archive="zig-${triple}-${EXPECTED_VERSION}.tar.xz"
  url="${DOWNLOAD_BASE}/${EXPECTED_VERSION}/${archive}"
  install_dir="${INSTALL_ROOT}/zig-${triple}-${EXPECTED_VERSION}"

  if [[ -x "${install_dir}/zig" ]]; then
    echo "${install_dir}/zig"
    return 0
  fi

  mkdir -p "${INSTALL_ROOT}"
  tmp_dir="$(mktemp -d)"
  archive_path="${tmp_dir}/${archive}"
  trap 'rm -rf "${tmp_dir}"' RETURN

  info "downloading Zig ${EXPECTED_VERSION} for ${triple}"
  curl --fail --location --silent --show-error "${url}" --output "${archive_path}"

  info "extracting ${archive} into ${INSTALL_ROOT}"
  tar -xJf "${archive_path}" -C "${INSTALL_ROOT}"

  [[ -x "${install_dir}/zig" ]] || die "downloaded Zig but ${install_dir}/zig was not created"
  echo "${install_dir}/zig"
}

if zig_bin="$(pick_zig)"; then
  if validate_version "${zig_bin}"; then
    emit_result "${zig_bin}"
    exit 0
  fi

  if [[ "${CHECK_ONLY}" -eq 1 ]]; then
    die "expected Zig ${EXPECTED_VERSION}, found $("${zig_bin}" version) at ${zig_bin}"
  fi

  info "ignoring mismatched Zig $("${zig_bin}" version) at ${zig_bin}; installing ${EXPECTED_VERSION}"
elif [[ "${CHECK_ONLY}" -eq 1 ]]; then
  die "zig not found; expected ${EXPECTED_VERSION}"
fi

zig_bin="$(install_local)"
validate_version "${zig_bin}" || die "expected Zig ${EXPECTED_VERSION}, found $("${zig_bin}" version) at ${zig_bin}"
emit_result "${zig_bin}"
