#!/usr/bin/env bash

cuda_prebuilt_hash_stream() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum | awk '{print $1}'
    else
        shasum -a 256 | awk '{print $1}'
    fi
}

cuda_prebuilt_hash_file() {
    local path="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$path" | awk '{print $1}'
    else
        shasum -a 256 "$path" | awk '{print $1}'
    fi
}

cuda_prebuilt_files_hash() {
    local path
    {
        for path in "$@"; do
            if [[ -f "$path" ]]; then
                printf 'file\t%s\t%s\n' "$path" "$(cuda_prebuilt_hash_file "$path")"
            elif [[ -d "$path" ]]; then
                find "$path" -type f -print0 |
                    LC_ALL=C sort -z |
                    while IFS= read -r -d '' file; do
                        printf 'file\t%s\t%s\n' "$file" "$(cuda_prebuilt_hash_file "$file")"
                    done
            else
                printf 'missing\t%s\n' "$path"
            fi
        done
    } | cuda_prebuilt_hash_stream
}

cuda_prebuilt_command_id() {
    local command="$1"
    shift
    if ! command -v "$command" >/dev/null 2>&1; then
        printf 'missing'
        return
    fi
    {
        command -v "$command"
        "$command" "$@" 2>&1 || true
    } | cuda_prebuilt_hash_stream
}

cuda_prebuilt_tree_hash() {
    local path="$1"
    git rev-parse "HEAD:$path" 2>/dev/null || printf 'missing'
}

cuda_prebuilt_source_hash() {
    local path="$1"
    local root relative
    root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
    if [[ -n "$root" ]]; then
        if [[ "$path" = /* ]]; then
            relative="${path#"$root"/}"
        else
            relative="$path"
        fi
        if [[ "$relative" != "$path" || "$path" != /* ]] &&
            git cat-file -e "HEAD:$relative" 2>/dev/null; then
            git rev-parse "HEAD:$relative"
            return
        fi
    fi
    if git -C "$path" rev-parse HEAD >/dev/null 2>&1; then
        {
            git -C "$path" rev-parse HEAD
            git -C "$path" diff --binary HEAD -- . 2>/dev/null || true
            git -C "$path" diff --binary --cached HEAD -- . 2>/dev/null || true
            git -C "$path" ls-files --others --exclude-standard -z 2>/dev/null |
                LC_ALL=C sort -z |
                while IFS= read -r -d '' file; do
                    printf 'untracked\t%s\t%s\n' "$file" "$(cuda_prebuilt_hash_file "$path/$file")"
                done
        } | cuda_prebuilt_hash_stream
        return
    fi
    cuda_prebuilt_files_hash "$path"
}

cuda_prebuilt_dirty_hash() {
    {
        git diff --binary HEAD -- Cargo.toml Cargo.lock requirements-build.txt crates/cuda-kernels crates/deepep-sys 2>/dev/null || true
        git diff --binary --cached HEAD -- Cargo.toml Cargo.lock requirements-build.txt crates/cuda-kernels crates/deepep-sys 2>/dev/null || true
        git ls-files --others --exclude-standard -- Cargo.toml Cargo.lock requirements-build.txt crates/cuda-kernels crates/deepep-sys 2>/dev/null |
            LC_ALL=C sort |
            while IFS= read -r file; do
                printf 'untracked\t%s\t%s\n' "$file" "$(cuda_prebuilt_hash_file "$file")"
            done
    } | cuda_prebuilt_hash_stream
}

cuda_prebuilt_manifest() {
    local cuda_home="${CUDA_HOME:-${CUDA_PATH:-/usr/local/cuda}}"
    local nvcc="$cuda_home/bin/nvcc"
    local ccbin="${NVCC_CCBIN:-g++}"
    local deepgemm_root="${ARLE_DEEPGEMM_ROOT:-crates/cuda-kernels/vendor/deepgemm}"
    local deepgemm_library_root="${ARLE_DEEPGEMM_LIBRARY_ROOT:-$deepgemm_root/deep_gemm}"
    local deepgemm_cutlass="${ARLE_DEEPGEMM_CUTLASS_INCLUDE:-$deepgemm_root/third-party/cutlass/include}"
    local deepep_root="${ARLE_DEEPEP_DIR:-crates/deepep-sys/vendor/deepep}"
    local nvshmem_root="${ARLE_DEEPEP_NVSHMEM_DIR:-}"
    local sidecar="${ARLE_DEEPEP_SIDECAR_PREBUILT:-}"

    cat <<EOF
schema=2
cargo_toml=$(cuda_prebuilt_tree_hash Cargo.toml)
cargo_lock=$(cuda_prebuilt_tree_hash Cargo.lock)
requirements_build=$(cuda_prebuilt_tree_hash requirements-build.txt)
cuda_kernels_tree=$(cuda_prebuilt_tree_hash crates/cuda-kernels)
deepep_sys_tree=$(cuda_prebuilt_tree_hash crates/deepep-sys)
dirty_hash=$(cuda_prebuilt_dirty_hash)
target=${TARGET:-${CARGO_BUILD_TARGET:-}}
profile=${PROFILE:-}
features=${FEATURES:-}
cuda_home=$cuda_home
nvcc_id=$(cuda_prebuilt_command_id "$nvcc" --version)
nvcc_ccbin_id=$(cuda_prebuilt_command_id "$ccbin" --version)
ar_id=$(cuda_prebuilt_command_id ar --version)
rustc_id=$(cuda_prebuilt_command_id rustc -vV)
python_id=$(cuda_prebuilt_command_id "${INFER_TILELANG_PYTHON:-python3}" --version)
cudarc_cuda_version=${CUDARC_CUDA_VERSION:-}
torch_cuda_arch_list=${TORCH_CUDA_ARCH_LIST:-}
cmake_cuda_architectures=${CMAKE_CUDA_ARCHITECTURES:-}
disable_deepgemm_native=${ARLE_CUDA_DISABLE_DEEPGEMM_NATIVE:-}
deepgemm_root=$deepgemm_root
deepgemm_root_hash=$(cuda_prebuilt_source_hash "$deepgemm_root")
deepgemm_library_root=$deepgemm_library_root
deepgemm_library_hash=$(cuda_prebuilt_source_hash "$deepgemm_library_root")
deepgemm_cutlass_include=$deepgemm_cutlass
deepgemm_cutlass_hash=$(cuda_prebuilt_source_hash "$deepgemm_cutlass")
deepep_dir=$deepep_root
deepep_hash=$(cuda_prebuilt_source_hash "$deepep_root")
deepep_nvshmem_dir=$nvshmem_root
deepep_nvshmem_hash=$(if [[ -n "$nvshmem_root" ]]; then cuda_prebuilt_source_hash "$nvshmem_root"; else printf 'disabled'; fi)
deepep_sidecar_prebuilt=$sidecar
deepep_sidecar_hash=$(if [[ -n "$sidecar" ]]; then cuda_prebuilt_hash_file "$sidecar"; else printf 'disabled'; fi)
disable_flashmla=${ARLE_CUDA_DISABLE_FLASHMLA:-}
disable_flashmla_decode=${ARLE_CUDA_DISABLE_FLASHMLA_DECODE:-}
enable_fa3=${ARLE_CUDA_ENABLE_FA3:-}
enable_flashqla_gdr=${ARLE_CUDA_ENABLE_FLASHQLA_GDR:-}
disable_marlin_w4_fp8=${ARLE_CUDA_DISABLE_MARLIN_W4_FP8:-}
nvcc_split_compile=${ARLE_NVCC_SPLIT_COMPILE:-}
nvcc_ccbin=${NVCC_CCBIN:-}
cc=${CC:-}
cflags=${CFLAGS:-}
cxxflags=${CXXFLAGS:-}
EOF
}
