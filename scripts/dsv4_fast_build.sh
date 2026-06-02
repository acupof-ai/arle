#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${PROFILE:-release-fast}"
FEATURES="${FEATURES:-cuda,nccl}"
BIN="${BIN:-infer}"
PREBUILT_DIR="${ARLE_CUDA_KERNELS_PREBUILT_DIR:-$ROOT/target/dsv4-cuda-kernels-prebuilt}"
TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"
MANIFEST_NAME="arle-cuda-kernels.manifest"

detect_cuda() {
    local nvcc=""
    if [[ -n "${CUDA_HOME:-}" && -x "$CUDA_HOME/bin/nvcc" ]]; then
        nvcc="$CUDA_HOME/bin/nvcc"
    else
        for version in 13.1 13.0 12.9 12.8 12.7 12.6 12.5 12.4 12.3 12.2 12.1 12.0; do
            local candidate="/usr/local/cuda-$version/bin/nvcc"
            if [[ -x "$candidate" ]]; then
                nvcc="$candidate"
                break
            fi
        done
        for candidate in /usr/local/cuda/bin/nvcc /opt/cuda/bin/nvcc; do
            [[ -n "$nvcc" ]] && break
            if [[ -x "$candidate" ]]; then
                nvcc="$candidate"
                break
            fi
        done
    fi

    if [[ -n "$nvcc" ]]; then
        export CUDA_HOME="$(cd "$(dirname "$nvcc")/.." && pwd)"
        export PATH="$CUDA_HOME/bin:$PATH"
        if [[ -z "${CUDARC_CUDA_VERSION:-}" ]]; then
            local major_minor major minor
            major_minor="$("$nvcc" --version |
                sed -n 's/.*release \([0-9][0-9]*\)\.\([0-9][0-9]*\).*/\1 \2/p' |
                head -1)"
            if [[ -n "$major_minor" ]]; then
                read -r major minor <<<"$major_minor"
                export CUDARC_CUDA_VERSION="$((major * 1000 + minor * 10))"
            fi
        fi
    fi

    export CUDARC_CUDA_VERSION="${CUDARC_CUDA_VERSION:-12080}"
}

prefer_sccache() {
    if command -v sccache >/dev/null 2>&1; then
        export RUSTC_WRAPPER="${RUSTC_WRAPPER:-sccache}"
        export ARLE_NVCC_WRAPPER="${ARLE_NVCC_WRAPPER:-sccache}"
    fi
}

abs_path() {
    local path="$1"
    if [[ "$path" = /* ]]; then
        printf '%s\n' "$path"
    else
        printf '%s\n' "$ROOT/$path"
    fi
}

resolve_deepgemm_env() {
    local native="${ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE:-${ARLE_CUDA_ENABLE_DEEPGEMM_TORCH:-}}"
    if [[ "$native" != "1" && "$native" != "true" && "$native" != "TRUE" && "$native" != "yes" && "$native" != "YES" ]]; then
        return 0
    fi

    local root="${ARLE_DEEPGEMM_ROOT:-$ROOT/crates/cuda-kernels/vendor/deepgemm}"
    root="$(abs_path "$root")"
    local library_root="${ARLE_DEEPGEMM_LIBRARY_ROOT:-$root/deep_gemm}"
    library_root="$(abs_path "$library_root")"
    export ARLE_DEEPGEMM_ROOT="$root"
    export ARLE_DEEPGEMM_LIBRARY_ROOT="$library_root"

    if [[ -z "${ARLE_DEEPGEMM_CUTLASS_INCLUDE:-}" ]]; then
        local bundled="$root/third-party/cutlass/include"
        local flashmla="$ROOT/crates/cuda-kernels/vendor/flashmla/csrc/cutlass/include"
        if [[ -f "$bundled/cutlass/arch/barrier.h" ]]; then
            export ARLE_DEEPGEMM_CUTLASS_INCLUDE="$bundled"
        elif [[ -f "$flashmla/cutlass/arch/barrier.h" ]]; then
            export ARLE_DEEPGEMM_CUTLASS_INCLUDE="$flashmla"
        fi
    else
        export ARLE_DEEPGEMM_CUTLASS_INCLUDE="$(abs_path "$ARLE_DEEPGEMM_CUTLASS_INCLUDE")"
    fi
}

prebuilt_ready() {
    [[ -f "$PREBUILT_DIR/libkernels_cuda.a" ]] &&
        [[ -f "$PREBUILT_DIR/libtilelang_kernels_aot.a" ]] &&
        manifest_matches
}

hash_stream() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum | awk '{print $1}'
    else
        shasum -a 256 | awk '{print $1}'
    fi
}

tree_hash() {
    local path="$1"
    git rev-parse "HEAD:$path" 2>/dev/null || printf 'missing'
}

dirty_hash() {
    {
        git diff --binary HEAD -- Cargo.toml crates/cuda-kernels crates/deepep-sys 2>/dev/null || true
        git diff --binary --cached HEAD -- Cargo.toml crates/cuda-kernels crates/deepep-sys 2>/dev/null || true
    } | hash_stream
}

artifact_manifest() {
    cat <<EOF
cargo_toml=$(tree_hash Cargo.toml)
cuda_kernels_tree=$(tree_hash crates/cuda-kernels)
deepep_sys_tree=$(tree_hash crates/deepep-sys)
dirty_hash=$(dirty_hash)
cuda_home=${CUDA_HOME:-}
cudarc_cuda_version=${CUDARC_CUDA_VERSION:-}
torch_cuda_arch_list=${TORCH_CUDA_ARCH_LIST:-}
kernel_set=${ARLE_CUDA_KERNEL_SET:-}
deepgemm_native=${ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE:-}
deepgemm_root=${ARLE_DEEPGEMM_ROOT:-}
deepgemm_library_root=${ARLE_DEEPGEMM_LIBRARY_ROOT:-}
deepgemm_cutlass_include=${ARLE_DEEPGEMM_CUTLASS_INCLUDE:-}
deepep_dir=${ARLE_DEEPEP_DIR:-}
disable_flashmla=${ARLE_CUDA_DISABLE_FLASHMLA:-}
enable_flashmla_decode=${ARLE_CUDA_ENABLE_FLASHMLA_DECODE:-}
disable_flashmla_decode=${ARLE_CUDA_DISABLE_FLASHMLA_DECODE:-}
disable_marlin_w4_fp8=${ARLE_CUDA_DISABLE_MARLIN_W4_FP8:-}
nvcc_ccbin=${NVCC_CCBIN:-}
tilelang_python=${INFER_TILELANG_PYTHON:-}
EOF
}

manifest_matches() {
    local manifest="$PREBUILT_DIR/$MANIFEST_NAME"
    if [[ ! -f "$manifest" ]]; then
        echo "prebuilt artifacts missing $MANIFEST_NAME; ignoring stale/manual cache"
        return 1
    fi
    if ! diff -u "$manifest" <(artifact_manifest) >/tmp/arle-cuda-kernels-manifest.diff 2>/dev/null; then
        echo "prebuilt artifact manifest mismatch; ignoring $PREBUILT_DIR"
        cat /tmp/arle-cuda-kernels-manifest.diff || true
        return 1
    fi
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
    artifact_manifest >"$PREBUILT_DIR/$MANIFEST_NAME"
    echo "harvested CUDA prebuilt artifacts from $out_dir -> $PREBUILT_DIR"
}

cd "$ROOT"
detect_cuda
prefer_sccache
resolve_deepgemm_env

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export ARLE_CUDA_KERNEL_SET="${ARLE_CUDA_KERNEL_SET:-dsv4_flash}"
if [[ "$ARLE_CUDA_KERNEL_SET" == "dsv4_flash" ]]; then
    export ARLE_CUDA_ENABLE_FLASHMLA_DECODE="${ARLE_CUDA_ENABLE_FLASHMLA_DECODE:-1}"
fi
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
echo "CUDA_HOME=${CUDA_HOME:-}"
echo "CUDARC_CUDA_VERSION=$CUDARC_CUDA_VERSION"
echo "ARLE_CUDA_KERNEL_SET=$ARLE_CUDA_KERNEL_SET"
echo "RUSTC_WRAPPER=${RUSTC_WRAPPER:-}"
echo "ARLE_NVCC_WRAPPER=${ARLE_NVCC_WRAPPER:-}"
echo "ARLE_NVCC_SPLIT_COMPILE=$ARLE_NVCC_SPLIT_COMPILE"
echo "ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE=${ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE:-}"
echo "ARLE_DEEPGEMM_ROOT=${ARLE_DEEPGEMM_ROOT:-}"
echo "ARLE_DEEPGEMM_LIBRARY_ROOT=${ARLE_DEEPGEMM_LIBRARY_ROOT:-}"
echo "ARLE_DEEPGEMM_CUTLASS_INCLUDE=${ARLE_DEEPGEMM_CUTLASS_INCLUDE:-}"

time cargo build --profile "$PROFILE" -p infer --features "$FEATURES" --bin "$BIN"
harvest_prebuilt
