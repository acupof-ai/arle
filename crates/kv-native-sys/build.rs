use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let target = env::var("TARGET").expect("TARGET");
    let zig = env::var("ZIG").unwrap_or_else(|_| "zig".to_string());
    let src = manifest_dir.join("zig").join("src").join("kv_native.zig");
    let emit = out_dir.join("libkv_native.a");
    let cache_dir = out_dir.join("zig-cache");
    let global_cache_dir = out_dir.join("zig-global-cache");

    let status = Command::new(&zig)
        .arg("build-lib")
        .arg("-O")
        .arg("ReleaseSafe")
        .arg("-static")
        // `kv_native.zig` uses `@cImport` plus `std.heap.c_allocator`,
        // so Zig must compile it with an explicit libc dependency.
        .arg("-lc")
        // Glibc's fortified `fcntl.h` wrappers trip Zig's C importer on
        // valid `open(2)` uses; compile the substrate without fortify.
        .arg("-D_FORTIFY_SOURCE=0")
        // The Rust link step only needs the exported C ABI symbols from the
        // static archive; disabling Zig stack probing avoids a dangling
        // `__zig_probe_stack` reference in downstream cargo test/build.
        .arg("-fno-stack-check")
        .arg(format!("-femit-bin={}", emit.display()))
        .arg("--cache-dir")
        .arg(cache_dir.as_os_str())
        .arg("--global-cache-dir")
        .arg(global_cache_dir.as_os_str())
        .arg(src.as_os_str())
        .status()
        .unwrap_or_else(|err| {
            panic!(
                "failed to spawn Zig compiler `{zig}` for kv-native-sys: {err}. \
set ZIG=/absolute/path/to/zig if it is installed outside PATH"
            )
        });

    assert!(
        status.success(),
        "zig build-lib failed for kv-native-sys with status {status}"
    );

    if target.contains("apple-darwin") {
        let ranlib_status = Command::new("ranlib")
            .arg(&emit)
            .status()
            .unwrap_or_else(|err| {
                panic!("failed to spawn ranlib while fixing kv-native-sys archive: {err}")
            });
        assert!(
            ranlib_status.success(),
            "ranlib failed for kv-native-sys with status {ranlib_status}"
        );
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=kv_native");
    println!("cargo:rerun-if-changed={}", src.display());
    println!("cargo:rerun-if-env-changed=ZIG");
}
