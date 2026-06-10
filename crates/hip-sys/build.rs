//! Links the ROCm HIP runtime when the `hip` feature is on.
//!
//! Detection follows the industry env-var convention: `ROCM_PATH`, then
//! `HIP_PATH`, then `/opt/rocm`. Link directives are the only thing gated
//! on the feature — the crate always typechecks without ROCm installed.

fn main() {
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=HIP_PATH");
    if std::env::var_os("CARGO_FEATURE_HIP").is_none() {
        return;
    }
    let root = std::env::var("ROCM_PATH")
        .or_else(|_| std::env::var("HIP_PATH"))
        .unwrap_or_else(|_| "/opt/rocm".to_string());
    println!("cargo:rustc-link-search=native={root}/lib");
    println!("cargo:rustc-link-lib=dylib=amdhip64");
}
