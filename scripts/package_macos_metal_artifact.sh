#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  cat >&2 <<'EOF'
usage: package_macos_metal_artifact.sh <metal_serve-binary> <output-tarball>
EOF
  exit 1
fi

binary_path="$1"
output_tarball="$2"

if [[ ! -f "$binary_path" ]]; then
  echo "package_macos_metal_artifact.sh: missing binary: $binary_path" >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

mkdir -p "$tmpdir/dist"
cp "$binary_path" "$tmpdir/dist/metal_serve"
tar -czf "$output_tarball" -C "$tmpdir/dist" metal_serve
