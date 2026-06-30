#!/usr/bin/env bash

cuda_prebuilt_hash_stream() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum | awk '{print $1}'
    else
        shasum -a 256 | awk '{print $1}'
    fi
}

cuda_prebuilt_tree_hash() {
    local path="$1"
    git rev-parse "HEAD:$path" 2>/dev/null || printf 'missing'
}

cuda_prebuilt_dirty_hash() {
    {
        git diff --binary HEAD -- Cargo.toml crates/cuda-kernels crates/deepep-sys 2>/dev/null || true
        git diff --binary --cached HEAD -- Cargo.toml crates/cuda-kernels crates/deepep-sys 2>/dev/null || true
    } | cuda_prebuilt_hash_stream
}

cuda_prebuilt_manifest() {
    cat <<EOF
cargo_toml=$(cuda_prebuilt_tree_hash Cargo.toml)
cuda_kernels_tree=$(cuda_prebuilt_tree_hash crates/cuda-kernels)
deepep_sys_tree=$(cuda_prebuilt_tree_hash crates/deepep-sys)
dirty_hash=$(cuda_prebuilt_dirty_hash)
cuda_home=${CUDA_HOME:-}
cudarc_cuda_version=${CUDARC_CUDA_VERSION:-}
torch_cuda_arch_list=${TORCH_CUDA_ARCH_LIST:-}
kernel_set=${ARLE_CUDA_KERNEL_SET:-}
disable_deepgemm_native=${ARLE_CUDA_DISABLE_DEEPGEMM_NATIVE:-}
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
