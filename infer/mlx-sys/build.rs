use std::env;
use std::path::PathBuf;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Step 1: Build MLX from source using cmake.
    // The CMakeLists.txt at mlx/ uses FetchContent to download MLX v0.31.1.
    let mlx_dst = cmake::Config::new("mlx")
        .define("MLX_BUILD_METAL", "ON")
        .define("MLX_BUILD_ACCELERATE", "ON")
        .define("MLX_BUILD_TESTS", "OFF")
        .define("MLX_BUILD_EXAMPLES", "OFF")
        .define("MLX_BUILD_BENCHMARKS", "OFF")
        .define("MLX_BUILD_PYTHON_BINDINGS", "OFF")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("CMAKE_CXX_STANDARD", "17")
        .build_target("mlx")
        .build();

    let mlx_build = mlx_dst.join("build");

    // Find MLX include directories.
    // After FetchContent, headers are in the fetched source tree.
    let mlx_fetch_src = mlx_build.join("_deps/mlx-src");
    let mlx_fetch_build = mlx_build.join("_deps/mlx-build");

    // Step 2: Compile C++ bridge files with cc.
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("src/mlx_bridge.cpp")
        .file("src/mlx_qwen35_model.cpp")
        .include("src")
        .include(&mlx_fetch_src)
        .include(&mlx_fetch_build) // for generated headers
        .flag("-Wno-unused-parameter")
        .flag("-Wno-sign-compare")
        .compile("mlx_ffi");

    // Step 3: Link MLX static library.
    // MLX builds as libmlx.a in the build directory.
    println!(
        "cargo:rustc-link-search=native={}",
        mlx_fetch_build.join("lib").display()
    );
    // Also check directly in the build dir
    println!(
        "cargo:rustc-link-search=native={}",
        mlx_fetch_build.display()
    );
    // And the top-level build/lib
    println!(
        "cargo:rustc-link-search=native={}",
        mlx_build.join("lib").display()
    );
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
    println!("cargo:rerun-if-changed=src/mlx_qwen35_model.cpp");
    println!("cargo:rerun-if-changed=src/mlx_common.h");
    println!("cargo:rerun-if-changed=mlx/CMakeLists.txt");
}
