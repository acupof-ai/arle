#!/usr/bin/env bash
#
# Install or validate the Zig toolchain used by `crates/kv-native-sys`.
#
# Usage:
#   scripts/setup_zig_toolchain.sh [EXPECTED_VERSION]
#
# Env:
#   ZIG                   explicit zig binary to validate first
#   KV_ZIG_VERSION        alternate version override
#   HOMEBREW_NO_AUTO_UPDATE=1  recommended when using brew on macOS CI/dev boxes

set -euo pipefail

EXPECTED_VERSION="${1:-${KV_ZIG_VERSION:-0.16.0}}"

info() { echo "[setup_zig_toolchain] $*"; }
die() { echo "[setup_zig_toolchain] error: $*" >&2; exit 1; }

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

validate_version() {
  local zig_bin="$1"
  local actual
  actual="$("${zig_bin}" version)"
  if [[ "${actual}" != "${EXPECTED_VERSION}" ]]; then
    die "expected Zig ${EXPECTED_VERSION}, found ${actual} at ${zig_bin}"
  fi
  info "using Zig ${actual} at ${zig_bin}"
}

if zig_bin="$(pick_zig)"; then
  validate_version "${zig_bin}"
  exit 0
fi

case "$(uname -s)" in
  Darwin)
    command -v brew >/dev/null 2>&1 || die "zig not found and Homebrew is unavailable"
    info "installing Zig ${EXPECTED_VERSION} via Homebrew"
    brew list zig >/dev/null 2>&1 || brew install zig
    zig_bin="$(command -v zig || true)"
    [[ -n "${zig_bin}" ]] || die "brew install zig completed but zig is still not on PATH"
    validate_version "${zig_bin}"
    ;;
  *)
    die "zig not found; install Zig ${EXPECTED_VERSION} and re-run (or set ZIG=/absolute/path/to/zig)"
    ;;
esac
