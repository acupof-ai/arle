use std::env;
use std::path::PathBuf;

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
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("src/mlx_bridge.cpp")
        .file("src/mlx_dflash_draft_model.cpp")
        .file("src/mlx_qwen35_model.cpp")
        .include("src")
        .include(&mlx_vendor_dir)
        .include(&mlx_build) // for generated headers
        .flag("-Wno-deprecated-copy")
        .flag("-Wno-unused-parameter")
        .flag("-Wno-sign-compare")
        .compile("mlx_ffi");

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

    // Step 4: Link the bridge static lib.
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=mlx_ffi");

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
    println!("cargo:rerun-if-changed=src/mlx_common.h");
    println!("cargo:rerun-if-changed=vendor");
}
