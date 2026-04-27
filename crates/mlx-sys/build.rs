use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let vendor_dir = manifest_dir.join("vendor");
    let mlx_vendor_dir = vendor_dir.join("mlx");
    let metal_cpp_vendor_dir = vendor_dir.join("metal-cpp");
    let json_vendor_dir = vendor_dir.join("json");
    let fmt_vendor_dir = vendor_dir.join("fmt");
    let gguflib_vendor_dir = vendor_dir.join("gguflib");

    for required_dir in [
        &mlx_vendor_dir,
        &metal_cpp_vendor_dir,
        &json_vendor_dir,
        &fmt_vendor_dir,
        &gguflib_vendor_dir,
    ] {
        assert!(
            required_dir.exists(),
            "Missing vendored dependency: {}",
            required_dir.display()
        );
    }

    // mlx C++ uses `__builtin_available(macOS 26, ...)` and
    // `__builtin_available(macOS 15, ...)` checks. clang lowers those to
    // `__isPlatformVersionAtLeast(...)` calls, which live in the clang
    // runtime (`libclang_rt.osx.a`). Rust's default link line doesn't pull
    // in compiler-rt, so the symbol fails to resolve at link time. Two
    // mitigations together:
    //
    //   1. Pin the C++ compile deployment target via env (default 14.0)
    //      so any future `@available(macOS 14, *)` checks are elided.
    //   2. On macOS, locate the active clang's `libclang_rt.osx.a` and
    //      add it to the Rust link line so the symbol resolves regardless
    //      of which @available macro version mlx uses.
    let macos_deployment_target =
        env::var("MACOSX_DEPLOYMENT_TARGET").unwrap_or_else(|_| "14.0".to_string());
    let on_macos = env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos");
    if on_macos {
        if let Some(rt) = clang_compiler_rt() {
            println!(
                "cargo:rustc-link-search=native={}",
                rt.parent().unwrap().display()
            );
            println!("cargo:rustc-link-lib=static=clang_rt.osx");
        } else {
            println!(
                "cargo:warning=mlx-sys: could not locate libclang_rt.osx.a — Metal link may fail with `__isPlatformVersionAtLeast` undefined symbol"
            );
        }
    }

    // Step 1: Build MLX from the vendored source tree using cmake.
    // The upstream MLX CMakeLists still references FetchContent, but every
    // dependency is pinned to a repository-local source tree and network
    // access is disabled.
    let mlx_dst = cmake::Config::new(&mlx_vendor_dir)
        .define("MLX_BUILD_METAL", "ON")
        .define("MLX_BUILD_ACCELERATE", "ON")
        .define("MLX_BUILD_TESTS", "OFF")
        .define("MLX_BUILD_EXAMPLES", "OFF")
        .define("MLX_BUILD_BENCHMARKS", "OFF")
        .define("MLX_BUILD_PYTHON_BINDINGS", "OFF")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("CMAKE_CXX_STANDARD", "17")
        // Pin the C++ deployment target so `@available(macOS 14, *)` branches
        // are elided at compile time (see comment above).
        .define("CMAKE_OSX_DEPLOYMENT_TARGET", &macos_deployment_target)
        .define("FETCHCONTENT_FULLY_DISCONNECTED", "ON")
        .define(
            "FETCHCONTENT_SOURCE_DIR_METAL_CPP",
            metal_cpp_vendor_dir.as_os_str().to_string_lossy().as_ref(),
        )
        .define(
            "FETCHCONTENT_SOURCE_DIR_JSON",
            json_vendor_dir.as_os_str().to_string_lossy().as_ref(),
        )
        .define(
            "FETCHCONTENT_SOURCE_DIR_FMT",
            fmt_vendor_dir.as_os_str().to_string_lossy().as_ref(),
        )
        .define(
            "FETCHCONTENT_SOURCE_DIR_GGUFLIB",
            gguflib_vendor_dir.as_os_str().to_string_lossy().as_ref(),
        )
        .build_target("mlx")
        .build();

    let mlx_build = mlx_dst.join("build");

    // Find MLX include directories.
    // MLX sources are vendored under this crate, while any generated headers
    // live in the cmake build tree.

    // Step 2: Compile C++ bridge files with cc.
    let mut bridge = cc::Build::new();
    bridge
        .cpp(true)
        .std("c++17")
        .file("src/mlx_bridge.cpp")
        .file("src/mlx_dflash_draft_model.cpp")
        .file("src/mlx_qwen35_model.cpp")
        .file("src/mlx_qwen35_moe_block.cpp")
        .include("src")
        .include(&mlx_vendor_dir)
        .include(&mlx_build) // for generated headers
        .flag("-Wno-deprecated-copy")
        .flag("-Wno-unused-parameter")
        .flag("-Wno-sign-compare");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        bridge.flag(format!("-mmacosx-version-min={macos_deployment_target}"));
    }
    bridge.compile("mlx_ffi");

    // Step 2b: Compile the Objective-C++ Metal capture hook separately as its
    // own static lib. Keeping it out of the C++17 `mlx_ffi` build avoids
    // forcing `-x objective-c++` onto MLX sources. This file is env-gated and
    // costs a single relaxed atomic load on the hot path when disabled.
    let mut capture = cc::Build::new();
    capture
        .cpp(true)
        .file("src/mlx_metal_capture.mm")
        .flag("-fobjc-arc")
        .flag("-std=c++17")
        .flag("-Wno-unused-parameter");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        capture.flag(format!("-mmacosx-version-min={macos_deployment_target}"));
    }
    capture.compile("mlx_metal_capture");

    // Step 3: Link MLX static library.
    // MLX builds as libmlx.a in the build directory.
    println!(
        "cargo:rustc-link-search=native={}",
        mlx_dst.join("lib").display()
    );
    // CMake's build tree places libmlx.a at the root build dir when MLX is
    // built as the top-level project.
    println!("cargo:rustc-link-search=native={}", mlx_build.display());
    println!("cargo:rustc-link-lib=static=mlx");

    // Step 4: Link the bridge static libs.
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=mlx_ffi");
    println!("cargo:rustc-link-lib=static=mlx_metal_capture");

    // Step 5: Link macOS frameworks.
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Accelerate");
    println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");

    // Link C++ standard library.
    println!("cargo:rustc-link-lib=c++");

    // Rerun if bridge files change.
    println!("cargo:rerun-if-changed=src/mlx_bridge.cpp");
    println!("cargo:rerun-if-changed=src/mlx_dflash_draft_model.cpp");
    println!("cargo:rerun-if-changed=src/mlx_qwen35_model.cpp");
    println!("cargo:rerun-if-changed=src/mlx_qwen35_moe_block.cpp");
    println!("cargo:rerun-if-changed=src/mlx_metal_capture.mm");
    println!("cargo:rerun-if-changed=src/mlx_common.h");
    println!("cargo:rerun-if-changed=vendor");
    println!("cargo:rerun-if-env-changed=MACOSX_DEPLOYMENT_TARGET");
}

/// Locate the active clang's `libclang_rt.osx.a` via
/// `clang -print-resource-dir`. Returns `None` if clang isn't on PATH or the
/// library isn't where we expect (e.g. on a non-macOS host).
fn clang_compiler_rt() -> Option<PathBuf> {
    let output = Command::new("clang")
        .arg("-print-resource-dir")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let resource_dir = String::from_utf8(output.stdout).ok()?;
    let resource_dir = resource_dir.trim();
    let candidate = Path::new(resource_dir).join("lib/darwin/libclang_rt.osx.a");
    if candidate.is_file() {
        Some(candidate)
    } else {
        None
    }
}
