#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/export_prebuilt_cuda_kernels.sh"
PROFILE="${PROFILE:-release-fast}"
FEATURES="${FEATURES:-cuda,nccl}"
BIN="${BIN:-arle}"
PREBUILT_DIR="${ARLE_CUDA_KERNELS_PREBUILT_DIR:-$ROOT/target/dsv4-cuda-kernels-prebuilt}"
USED_PREBUILT=0

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
        CUDA_HOME="$(cd "$(dirname "$nvcc")/.." && pwd)"
        export CUDA_HOME
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
        ARLE_DEEPGEMM_CUTLASS_INCLUDE="$(abs_path "$ARLE_DEEPGEMM_CUTLASS_INCLUDE")"
        export ARLE_DEEPGEMM_CUTLASS_INCLUDE
    fi
}

deepep_dir_valid() {
    local dir="$1"
    [[ -d "$dir/csrc/kernels" ]] || return 1
    [[ -f "$dir/csrc/kernels/api.cuh" || -f "$dir/csrc/kernels/legacy/api.cuh" ]]
}

resolve_deepep_env() {
    if [[ -n "${ARLE_DEEPEP_DIR:-}" ]]; then
        ARLE_DEEPEP_DIR="$(abs_path "$ARLE_DEEPEP_DIR")"
        export ARLE_DEEPEP_DIR
        deepep_dir_valid "$ARLE_DEEPEP_DIR" || {
            echo "ARLE_DEEPEP_DIR=$ARLE_DEEPEP_DIR is not a supported DeepEP source tree; expected csrc/kernels/api.cuh or csrc/kernels/legacy/api.cuh" >&2
            return 1
        }
        return 0
    fi

    local candidate
    for candidate in \
        "$ROOT/../DeepEP" \
        "$ROOT/../deepep" \
        "/data01/build/DeepEP" \
        "/workspace/DeepEP" \
        "/workspace/deepep"; do
        if deepep_dir_valid "$candidate"; then
            export ARLE_DEEPEP_DIR="$candidate"
            echo "using DeepEP source tree from $ARLE_DEEPEP_DIR"
            return 0
        fi
    done

    if [[ "${ARLE_DSV4_MOE_BACKEND:-}" == "native-deepep" ||
          "${ARLE_DSV4_MOE_BACKEND:-}" == "native_deepep" ||
          "${ARLE_DSV4_PERFORMANCE_PROFILE:-}" == "sglang" ||
          "${ARLE_DSV4_PERFORMANCE_PROFILE:-}" == "sglang-best-practice" ]]; then
        echo "DeepEP source tree not found; native-deepep / sglang profile requires ARLE_DEEPEP_DIR" >&2
        return 1
    fi

    echo "DeepEP source tree not found; building without arle_deepep_sidecar"
}

prebuilt_ready() {
    cuda_prebuilt_validate_bundle "$PREBUILT_DIR" || return 1
    ARLE_CUDA_KERNELS_PREBUILT_DIR="$PREBUILT_DIR" \
        cargo check -q -p cuda-kernels --profile "$PROFILE" --features "$FEATURES" 2>/dev/null
}

harvest_prebuilt() {
    local out_dir="$1"
    [[ -n "$out_dir" ]] || {
        echo "cargo did not report cuda-kernels OUT_DIR" >&2
        return 1
    }
    cuda_prebuilt_export "$PREBUILT_DIR" "$out_dir"
}

cd "$ROOT"
detect_cuda
prefer_sccache
resolve_deepgemm_env
resolve_deepep_env

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export ARLE_NVCC_SPLIT_COMPILE="${ARLE_NVCC_SPLIT_COMPILE:-8}"

if prebuilt_ready; then
    export ARLE_CUDA_KERNELS_PREBUILT_DIR="$PREBUILT_DIR"
    USED_PREBUILT=1
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
echo "RUSTC_WRAPPER=${RUSTC_WRAPPER:-}"
echo "ARLE_NVCC_WRAPPER=${ARLE_NVCC_WRAPPER:-}"
echo "ARLE_NVCC_SPLIT_COMPILE=$ARLE_NVCC_SPLIT_COMPILE"
echo "ARLE_CUDA_DISABLE_DEEPGEMM_NATIVE=${ARLE_CUDA_DISABLE_DEEPGEMM_NATIVE:-}"
echo "ARLE_DEEPGEMM_ROOT=${ARLE_DEEPGEMM_ROOT:-}"
echo "ARLE_DEEPGEMM_LIBRARY_ROOT=${ARLE_DEEPGEMM_LIBRARY_ROOT:-}"
echo "ARLE_DEEPGEMM_CUTLASS_INCLUDE=${ARLE_DEEPGEMM_CUTLASS_INCLUDE:-}"
echo "ARLE_DEEPEP_DIR=${ARLE_DEEPEP_DIR:-}"

BUILD_JSON="$(mktemp)"
trap 'rm -f "$BUILD_JSON"' EXIT
time cargo build --profile "$PROFILE" --features "$FEATURES" --bin "$BIN" \
    --message-format=json-render-diagnostics | tee "$BUILD_JSON"
if [[ "$USED_PREBUILT" == "1" ]]; then
    echo "prebuilt fast path used; not harvesting source artifacts"
else
    mapfile -t CUDA_OUT_DIRS < <(jq -r '
        select(.reason == "build-script-executed") |
        select(.package_id | test("/crates/cuda-kernels#[^/]+$")) |
        .out_dir
    ' "$BUILD_JSON" | LC_ALL=C sort -u)
    [[ ${#CUDA_OUT_DIRS[@]} == 1 ]] || {
        printf 'expected exactly one cuda-kernels OUT_DIR, found %s:
' "${#CUDA_OUT_DIRS[@]}" >&2
        printf '  %s
' "${CUDA_OUT_DIRS[@]}" >&2
        exit 1
    }
    harvest_prebuilt "${CUDA_OUT_DIRS[0]}"
fi
