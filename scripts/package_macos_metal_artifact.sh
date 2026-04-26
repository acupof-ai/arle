#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat >&2 <<'EOF'
usage:
  package_macos_metal_artifact.sh <output-tarball> <binary>...
  package_macos_metal_artifact.sh <binary> <output-tarball>   # legacy form
EOF
  exit 1
fi

if [[ "$1" == *.tar.gz || "$1" == *.tgz ]]; then
  output_tarball="$1"
  shift
  binary_paths=("$@")
else
  output_tarball="${@: -1}"
  binary_paths=("${@:1:$(($# - 1))}")
fi

if [[ "${#binary_paths[@]}" -eq 0 ]]; then
  echo "package_macos_metal_artifact.sh: no binaries provided" >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

mkdir -p "$tmpdir/dist"
for binary_path in "${binary_paths[@]}"; do
  if [[ ! -f "$binary_path" ]]; then
    echo "package_macos_metal_artifact.sh: missing binary: $binary_path" >&2
    exit 1
  fi
  cp "$binary_path" "$tmpdir/dist/$(basename "$binary_path")"
done

tar -czf "$output_tarball" -C "$tmpdir/dist" .
