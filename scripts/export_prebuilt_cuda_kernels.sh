#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/cuda_prebuilt_manifest.sh"
CUDA_PREBUILT_MANIFEST="arle-cuda-kernels.manifest"
CUDA_PREBUILT_ARTIFACTS=(libkernels_cuda.a libtilelang_kernels_aot.a arle_deepep_sidecar)

cuda_prebuilt_validate_bundle() {
    local dir="$1" manifest name size hash
    manifest="$dir/$CUDA_PREBUILT_MANIFEST"
    [[ -f "$manifest" ]] || { echo "CUDA prebuilt manifest missing: $manifest" >&2; return 1; }
    cuda_prebuilt_manifest_validate "$manifest" || return 1
    [[ "$(cuda_prebuilt_manifest_value "$manifest" schema)" == "3" ]] || {
        echo "unsupported CUDA prebuilt manifest schema" >&2
        return 1
    }
    for name in "${CUDA_PREBUILT_ARTIFACTS[@]}"; do
        size="$(cuda_prebuilt_manifest_value "$manifest" "artifact.$name.size" 2>/dev/null || true)"
        hash="$(cuda_prebuilt_manifest_value "$manifest" "artifact.$name.sha256" 2>/dev/null || true)"
        if [[ -z "$size" ]]; then
            if [[ "$name" == arle_deepep_sidecar ]]; then
                [[ -z "$hash" && ! -e "$dir/$name" ]] || {
                    echo "CUDA prebuilt sidecar manifest/file half-state" >&2
                    return 1
                }
                continue
            fi
            echo "CUDA prebuilt manifest lacks artifact.$name.size" >&2
            return 1
        fi
        [[ -f "$dir/$name" ]] || { echo "CUDA prebuilt artifact missing: $dir/$name" >&2; return 1; }
        [[ "$(wc -c <"$dir/$name" | tr -d ' ')" == "$size" ]] || {
            echo "CUDA prebuilt artifact size mismatch: $name" >&2
            return 1
        }
        [[ "$(cuda_prebuilt_hash_file "$dir/$name")" == "$hash" ]] || {
            echo "CUDA prebuilt artifact hash mismatch: $name" >&2
            return 1
        }
    done
}

cuda_prebuilt_export() {
    local dest="$1" out_dir="$2" name
    cuda_prebuilt_validate_bundle "$out_dir"
    mkdir -p "$dest"
    rm -f "$dest/$CUDA_PREBUILT_MANIFEST"
    for name in "${CUDA_PREBUILT_ARTIFACTS[@]}"; do
        rm -f "$dest/$name"
    done
    for name in "$CUDA_PREBUILT_MANIFEST" "${CUDA_PREBUILT_ARTIFACTS[@]}"; do
        [[ -f "$out_dir/$name" ]] && cp -p "$out_dir/$name" "$dest/$name"
    done
    cuda_prebuilt_validate_bundle "$dest"
    printf '[export] %s -> %s\n' "$out_dir" "$dest"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    [[ $# == 2 ]] || {
        echo "usage: export_prebuilt_cuda_kernels.sh <dest-dir> <cuda-kernels-out-dir>" >&2
        exit 2
    }
    cuda_prebuilt_export "$1" "$2"
    echo "[export] consume with: export ARLE_CUDA_KERNELS_PREBUILT_DIR=$(cd "$1" && pwd)"
fi
