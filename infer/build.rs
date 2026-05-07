use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/scheduler/cuda/nvtx_scopes.c");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    let cuda_enabled = env::var_os("CARGO_FEATURE_CUDA").is_some();
    let no_cuda_enabled = env::var_os("CARGO_FEATURE_NO_CUDA").is_some();
    if cuda_enabled && !no_cuda_enabled {
        build_nvtx_scopes();
    }
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
