#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${PROFILE:-release-fast}"
FEATURES="${FEATURES:-cuda,nccl}"
BIN="${BIN:-infer}"
PREBUILT_DIR="${ARLE_CUDA_KERNELS_PREBUILT_DIR:-$ROOT/target/dsv4-cuda-kernels-prebuilt}"
TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"

prefer_sccache() {
    if command -v sccache >/dev/null 2>&1; then
        export RUSTC_WRAPPER="${RUSTC_WRAPPER:-sccache}"
        export ARLE_NVCC_WRAPPER="${ARLE_NVCC_WRAPPER:-sccache}"
    fi
}

prebuilt_ready() {
    [[ -f "$PREBUILT_DIR/libkernels_cuda.a" ]] &&
        [[ -f "$PREBUILT_DIR/libtilelang_kernels_aot.a" ]]
}

find_latest_cuda_out() {
    find "$TARGET_DIR/$PROFILE/build" "$TARGET_DIR/release/build" \
        -maxdepth 3 -path '*/cuda-kernels-*/out/libkernels_cuda.a' \
        -print 2>/dev/null |
        while IFS= read -r lib; do
            local mtime
            mtime="$(stat -c %Y "$lib" 2>/dev/null || stat -f %m "$lib")"
            printf '%s %s\n' "$mtime" "$(dirname "$lib")"
        done |
        sort -nr |
        awk 'NR == 1 {print $2}'
}

harvest_prebuilt() {
    local out_dir
    out_dir="$(find_latest_cuda_out || true)"
    [[ -n "$out_dir" ]] || return 0
    [[ -f "$out_dir/libkernels_cuda.a" ]] || return 0
    [[ -f "$out_dir/libtilelang_kernels_aot.a" ]] || return 0

    mkdir -p "$PREBUILT_DIR"
    cp -f "$out_dir/libkernels_cuda.a" "$PREBUILT_DIR/"
    cp -f "$out_dir/libtilelang_kernels_aot.a" "$PREBUILT_DIR/"
    if [[ -f "$out_dir/arle_deepep_sidecar" ]]; then
        cp -f "$out_dir/arle_deepep_sidecar" "$PREBUILT_DIR/"
        chmod +x "$PREBUILT_DIR/arle_deepep_sidecar" || true
    fi
    echo "harvested CUDA prebuilt artifacts from $out_dir -> $PREBUILT_DIR"
}

cd "$ROOT"
prefer_sccache

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export ARLE_CUDA_KERNEL_SET="${ARLE_CUDA_KERNEL_SET:-dsv4_flash}"
export ARLE_NVCC_SPLIT_COMPILE="${ARLE_NVCC_SPLIT_COMPILE:-8}"

if prebuilt_ready; then
    export ARLE_CUDA_KERNELS_PREBUILT_DIR="$PREBUILT_DIR"
    if [[ -f "$PREBUILT_DIR/arle_deepep_sidecar" ]]; then
        export ARLE_DEEPEP_SIDECAR_PREBUILT="$PREBUILT_DIR/arle_deepep_sidecar"
    fi
    echo "using CUDA prebuilt artifacts from $PREBUILT_DIR"
else
    unset ARLE_CUDA_KERNELS_PREBUILT_DIR
    echo "no CUDA prebuilt artifacts at $PREBUILT_DIR; first run will build and harvest them"
fi

echo "profile=$PROFILE features=$FEATURES bin=$BIN"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "ARLE_CUDA_KERNEL_SET=$ARLE_CUDA_KERNEL_SET"
echo "RUSTC_WRAPPER=${RUSTC_WRAPPER:-}"
echo "ARLE_NVCC_WRAPPER=${ARLE_NVCC_WRAPPER:-}"
echo "ARLE_NVCC_SPLIT_COMPILE=$ARLE_NVCC_SPLIT_COMPILE"

time cargo build --profile "$PROFILE" -p infer --features "$FEATURES" --bin "$BIN"
harvest_prebuilt
