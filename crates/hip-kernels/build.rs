//! Compiles the AIPC HIP kernel objects with hipcc when the `hip` feature
//! is on (#76/#77).
//!
//! No hipcc on the machine (this Mac) => warn and skip: the crate still
//! typechecks, mirroring the `cuda,no-cuda` lane. The kernel sources are
//! `csrc/iq2_mmvq.cu` plus the six shim-portable cuda-kernels sources,
//! compiled as HIP with `csrc/arle_hip_shim.h` force-included.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Workspace-relative paths of every source compiled into the archive.
///
/// Basic-op additions (stage A, 2026-06-10) all scan clean on the strict
/// SM90 grep (`wgmma|cp.async|mbarrier|__grid_constant__|setmaxnreg|cluster|
/// asm volatile`) and use only shim-covered CUDA APIs:
/// - `misc/norm.cu`: bf16 conversions + `__ldg` only.
/// - `misc/sampling.cu`: `__shfl_down_sync` full-mask reductions (shim-mapped).
/// - `gemm/quantized_gemv.cu`: `__shfl_xor_sync` + bf16/fp8 includes.
///
/// Excluded: `gemm/gemv.cu` — depends on cuBLAS/cuBLASLt (`cublasGemmEx`,
/// `cublasLtMatmul`, autotune cache); the HIP GEMM lane is hipBLASLt per the
/// plan §3, not a shim port.
const SOURCES: &[&str] = &[
    "crates/hip-kernels/csrc/iq2_mmvq.cu",
    "crates/cuda-kernels/csrc/misc/dsv4_attention.cu",
    "crates/cuda-kernels/csrc/misc/dsv4_mhc.cu",
    "crates/cuda-kernels/csrc/misc/elementwise_basic.cu",
    "crates/cuda-kernels/csrc/misc/norm.cu",
    "crates/cuda-kernels/csrc/misc/sampling.cu",
    "crates/cuda-kernels/csrc/attention/decode_prep_paged.cu",
    "crates/cuda-kernels/csrc/gemm/moe_grouped_gemm.cu",
    "crates/cuda-kernels/csrc/gemm/quantized_gemv.cu",
];

const SHIM: &str = "crates/hip-kernels/csrc/arle_hip_shim.h";

/// hipcc-only stand-ins for the literal `#include <cuda*.h>` lines in the
/// unmodified cuda-kernels sources. Listed file-by-file because
/// rerun-if-changed on a directory is not recursive.
const CUDA_COMPAT_DIR: &str = "crates/hip-kernels/csrc/cuda_compat";
const CUDA_COMPAT_HEADERS: &[&str] = &["cuda.h", "cuda_runtime.h", "cuda_bf16.h", "cuda_fp8.h"];

fn main() {
    if std::env::var_os("CARGO_FEATURE_HIP").is_none() {
        return;
    }

    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=HIP_PATH");
    println!("cargo:rerun-if-env-changed=AMDGPU_TARGETS");

    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_root = manifest_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();

    for src in SOURCES {
        println!(
            "cargo:rerun-if-changed={}",
            workspace_root.join(src).display()
        );
    }
    println!(
        "cargo:rerun-if-changed={}",
        workspace_root.join(SHIM).display()
    );
    for header in CUDA_COMPAT_HEADERS {
        let path = workspace_root.join(CUDA_COMPAT_DIR).join(header);
        println!("cargo:rerun-if-changed={}", path.display());
    }

    let rocm_root = std::env::var("ROCM_PATH")
        .or_else(|_| std::env::var("HIP_PATH"))
        .unwrap_or_else(|_| "/opt/rocm".to_string());
    let hipcc = Path::new(&rocm_root).join("bin/hipcc");
    if !hipcc.exists() {
        println!(
            "cargo:warning=hip-kernels: hipcc not found at {} (ROCM_PATH/HIP_PATH//opt/rocm); \
             skipping kernel compilation — typecheck-only lane, kernel objects are built on the \
             ROCm box (pending-remote)",
            hipcc.display()
        );
        return;
    }

    // First entry of AMDGPU_TARGETS (`;`/`,`-separated, cmake convention).
    let arch = std::env::var("AMDGPU_TARGETS")
        .ok()
        .and_then(|t| t.split([';', ',']).next().map(str::to_string))
        .filter(|t| !t.is_empty())
        .unwrap_or_else(|| "gfx1151".to_string());

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let mut objects = Vec::new();

    for src in SOURCES {
        let src_path = workspace_root.join(src);
        let obj = out_dir.join(format!(
            "{}.o",
            src_path.file_stem().unwrap().to_string_lossy()
        ));
        let status = Command::new(&hipcc)
            .arg("-c")
            .arg("-O3")
            .arg("-x")
            .arg("hip")
            .arg(format!("--offload-arch={arch}"))
            .arg("-include")
            .arg(workspace_root.join(SHIM))
            .arg("-I")
            .arg(workspace_root.join("vendor/llama.cpp"))
            .arg("-I")
            .arg(workspace_root.join("crates/cuda-kernels/csrc"))
            .arg("-I")
            .arg(workspace_root.join(CUDA_COMPAT_DIR))
            .arg("-o")
            .arg(&obj)
            .arg(&src_path)
            .status()
            .unwrap_or_else(|e| panic!("failed to run {}: {e}", hipcc.display()));
        assert!(status.success(), "hipcc failed on {src} (arch {arch})");
        objects.push(obj);
    }

    let archive = out_dir.join("libarle_hip_kernels.a");
    let status = Command::new("ar")
        .arg("crs")
        .arg(&archive)
        .args(&objects)
        .status()
        .expect("failed to run ar");
    assert!(
        status.success(),
        "ar failed to create {}",
        archive.display()
    );

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=arle_hip_kernels");
    // The fat-binary objects self-register through the HIP runtime.
    println!("cargo:rustc-link-search=native={rocm_root}/lib");
    println!("cargo:rustc-link-lib=dylib=amdhip64");
}
