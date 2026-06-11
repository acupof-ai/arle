//! Links the ROCm HIP runtime when the `hip` feature is on and libamdhip64 is
//! available.
//!
//! Detection follows the industry env-var convention: `ROCM_PATH`, then
//! `HIP_PATH`, then `/opt/rocm`. Off-box builds keep compiling with the
//! `hip` feature by falling back to the Rust stub surface.

use std::path::Path;

fn has_amdhip64(dir: &Path) -> bool {
    dir.join("libamdhip64.so").exists()
        || dir.join("libamdhip64.dylib").exists()
        || dir.join("amdhip64.lib").exists()
        || std::fs::read_dir(dir)
            .ok()
            .into_iter()
            .flat_map(|entries| entries.filter_map(|entry| entry.ok()))
            .any(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with("libamdhip64.so.")
            })
}

fn main() {
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=HIP_PATH");
    println!("cargo:rustc-check-cfg=cfg(hip_runtime_available)");
    if std::env::var_os("CARGO_FEATURE_HIP").is_none() {
        return;
    }
    let root = std::env::var("ROCM_PATH")
        .or_else(|_| std::env::var("HIP_PATH"))
        .unwrap_or_else(|_| "/opt/rocm".to_string());

    let lib_dir = ["lib", "lib64"]
        .iter()
        .map(|subdir| Path::new(&root).join(subdir))
        .find(|dir| has_amdhip64(dir));

    let Some(lib_dir) = lib_dir else {
        println!(
            "cargo:warning=hip-sys: libamdhip64 not found under {root}/lib or {root}/lib64; compiling HIP runtime stubs"
        );
        return;
    };

    println!("cargo:rustc-cfg=hip_runtime_available");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=amdhip64");
}
