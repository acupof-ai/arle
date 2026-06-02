use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/scheduler/cuda/nvtx_scopes.c");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    emit_build_git_sha();

    let cuda_enabled = env::var_os("CARGO_FEATURE_CUDA").is_some();
    let no_cuda_enabled = env::var_os("CARGO_FEATURE_NO_CUDA").is_some();
    if cuda_enabled && !no_cuda_enabled {
        build_nvtx_scopes();
        emit_deepep_sidecar_path();
    }
}

/// M_d.1 §3: emit `BUILD_GIT_SHA` so `derive_radix_namespace` can mix it
/// into the RadixCache namespace alongside the tokenizer fingerprint and
/// `CARGO_PKG_VERSION`. Falls back to "unknown" outside a git checkout
/// (tarball / sandbox builds) — the namespace then collapses to
/// `(fingerprint, version, "unknown")` which still flips on tokenizer or
/// version changes; only inter-build-from-same-tag drift is undetectable
/// in that case.
fn emit_build_git_sha() {
    println!("cargo:rerun-if-changed=.git/HEAD");
    let sha = Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=BUILD_GIT_SHA={sha}");
}

fn build_nvtx_scopes() {
    let cuda_root = cuda_root().expect(
        "CUDA feature enabled but CUDA headers were not found; set CUDA_HOME or install CUDA under /opt/cuda or /usr/local/cuda",
    );
    let include = cuda_include_dir(&cuda_root).unwrap_or_else(|| {
        panic!(
            "CUDA feature enabled but nvtx3/nvToolsExt.h was not found under {}",
            cuda_root.display()
        )
    });

    cc::Build::new()
        .file("src/scheduler/cuda/nvtx_scopes.c")
        .include(include)
        .warnings(false)
        .compile("arle_nvtx_scopes");

    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-lib=dl");
    }
}

fn emit_deepep_sidecar_path() {
    println!("cargo:rerun-if-env-changed=ARLE_DEEPEP_SIDECAR_PREBUILT");
    println!("cargo:rerun-if-env-changed=ARLE_DEEPEP_SIDECAR_PATH");
    println!("cargo:rerun-if-env-changed=ARLE_CUDA_KERNELS_PREBUILT_DIR");
    println!("cargo:rerun-if-env-changed=ARLE_DEEPEP_DIR");

    if let Some(path) = env_path("ARLE_DEEPEP_SIDECAR_PREBUILT") {
        emit_existing_deepep_sidecar("ARLE_DEEPEP_SIDECAR_PREBUILT", &path);
        return;
    }
    if let Some(path) = env_path("ARLE_DEEPEP_SIDECAR_PATH") {
        emit_existing_deepep_sidecar("ARLE_DEEPEP_SIDECAR_PATH", &path);
        return;
    }
    if let Some(prebuilt_dir) = env_path("ARLE_CUDA_KERNELS_PREBUILT_DIR") {
        let sidecar = prebuilt_dir.join("arle_deepep_sidecar");
        if sidecar.is_file() {
            emit_existing_deepep_sidecar("ARLE_CUDA_KERNELS_PREBUILT_DIR", &sidecar);
        } else if env::var_os("ARLE_DEEPEP_DIR").is_some() {
            panic!(
                "ARLE_CUDA_KERNELS_PREBUILT_DIR={} is missing arle_deepep_sidecar while ARLE_DEEPEP_DIR is set",
                prebuilt_dir.display()
            );
        }
    }
}

fn emit_existing_deepep_sidecar(source: &str, path: &Path) {
    if !path.is_file() {
        panic!("{source}={} is not a file", path.display());
    }
    println!("cargo:rerun-if-changed={}", path.display());
    println!(
        "cargo:rustc-env=ARLE_DEEPEP_SIDECAR_PATH={}",
        path.display()
    );
}

fn env_path(name: &str) -> Option<PathBuf> {
    env::var_os(name)
        .map(PathBuf::from)
        .filter(|path| !path.as_os_str().is_empty())
}

fn cuda_root() -> Option<PathBuf> {
    env::var_os("CUDA_HOME")
        .or_else(|| env::var_os("CUDA_PATH"))
        .map(PathBuf::from)
        .filter(|path| path.exists())
        .or_else(|| first_existing(["/opt/cuda", "/usr/local/cuda"]))
}

fn first_existing<const N: usize>(paths: [&str; N]) -> Option<PathBuf> {
    paths
        .into_iter()
        .map(PathBuf::from)
        .find(|path| path.exists())
}

fn cuda_include_dir(cuda_root: &Path) -> Option<PathBuf> {
    [
        cuda_root.join("targets/x86_64-linux/include"),
        cuda_root.join("include"),
    ]
    .into_iter()
    .find(|include| include.join("nvtx3/nvToolsExt.h").exists())
}
