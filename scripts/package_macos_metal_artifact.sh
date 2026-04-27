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

# Ship MLX's metallib alongside the binaries. MLX's `load_default_library`
# searches binary-colocated paths first; without it the binaries fail with
# "Failed to load the default metallib" on every fresh install.
metallib_src="${ARLE_MLX_METALLIB:-}"
if [[ -z "$metallib_src" ]]; then
  # Pick the newest metallib the local cargo build produced. Build.rs already
  # copies it next to the binary in `target/<profile>/`, but that copy may be
  # outside the binary list passed to this script.
  for binary_path in "${binary_paths[@]}"; do
    candidate="$(dirname "$binary_path")/mlx.metallib"
    if [[ -f "$candidate" ]]; then
      metallib_src="$candidate"
      break
    fi
  done
fi
if [[ -z "$metallib_src" ]]; then
  metallib_src="$(ls -t target/release/build/mlx-sys-*/out/build/mlx/backend/metal/kernels/mlx.metallib 2>/dev/null | head -1 || true)"
fi
if [[ -z "$metallib_src" || ! -f "$metallib_src" ]]; then
  echo "package_macos_metal_artifact.sh: cannot locate mlx.metallib (set ARLE_MLX_METALLIB to override)" >&2
  exit 1
fi
cp "$metallib_src" "$tmpdir/dist/mlx.metallib"

tar -czf "$output_tarball" -C "$tmpdir/dist" .
