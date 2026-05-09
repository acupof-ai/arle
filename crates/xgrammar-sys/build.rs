use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=XGRAMMAR_SOURCE_DIR");
    println!("cargo:rerun-if-env-changed=ARLE_XGRAMMAR_SOURCE_DIR");

    if env::var_os("CARGO_FEATURE_REAL").is_none() {
        return;
    }

    let source_dir = find_source_dir();
    validate_source_dir(&source_dir);

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .include("src")
        .include(source_dir.join("include"))
        .include(source_dir.join("cpp"))
        .include(source_dir.join("3rdparty/picojson"))
        .include(source_dir.join("3rdparty/dlpack/include"))
        .file("src/xgrammar_ffi.cpp")
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-sign-compare")
        .flag_if_supported("-Wno-free-nonheap-object")
        .define("XGRAMMAR_ENABLE_CPPTRACE", "0")
        .define("XGRAMMAR_ENABLE_INTERNAL_CHECK", "0");

    add_xgrammar_sources(&mut build, &source_dir.join("cpp"));
    build.compile("arle_xgrammar_ffi");

    println!("cargo:rerun-if-changed=src/xgrammar_ffi.cpp");
    println!("cargo:rerun-if-changed=src/xgrammar_ffi.h");
}

fn find_source_dir() -> PathBuf {
    for key in ["XGRAMMAR_SOURCE_DIR", "ARLE_XGRAMMAR_SOURCE_DIR"] {
        if let Some(path) = env::var_os(key) {
            return PathBuf::from(path);
        }
    }
    PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"))
        .join("vendor/xgrammar")
}

fn validate_source_dir(source_dir: &Path) {
    for rel in [
        "include/xgrammar/xgrammar.h",
        "cpp/grammar_compiler.cc",
        "cpp/grammar_matcher.cc",
        "3rdparty/dlpack/include/dlpack/dlpack.h",
        "3rdparty/picojson/picojson.h",
    ] {
        let path = source_dir.join(rel);
        assert!(
            path.exists(),
            "xgrammar-sys real feature requires a pinned xgrammar checkout at {}; missing {}. \
             Set XGRAMMAR_SOURCE_DIR=/path/to/xgrammar (validated against mlc-ai/xgrammar v0.1.34) \
             and initialize the dlpack submodule.",
            source_dir.display(),
            rel
        );
    }
}

fn add_xgrammar_sources(build: &mut cc::Build, dir: &Path) {
    for entry in fs::read_dir(dir).expect("read xgrammar cpp dir") {
        let entry = entry.expect("read xgrammar cpp entry");
        let path = entry.path();
        if path.is_dir() {
            if path.file_name().and_then(|name| name.to_str()) == Some("tvm_ffi") {
                continue;
            }
            add_xgrammar_sources(build, &path);
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("cc") {
            build.file(path);
        }
    }
}
