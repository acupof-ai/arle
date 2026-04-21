use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let target = env::var("TARGET").expect("TARGET");
    let zig = env::var("ZIG").unwrap_or_else(|_| "zig".to_string());
    let src = manifest_dir.join("zig").join("src").join("kv_native.zig");
    let obj = out_dir.join("kv_native.o");
    let emit = out_dir.join("libkv_native.a");
    let cache_dir = out_dir.join("zig-cache");
    let global_cache_dir = out_dir.join("zig-global-cache");

    let status = Command::new(&zig)
        .arg("build-obj")
        .arg("-O")
        .arg("ReleaseSafe")
        .arg(format!("-femit-bin={}", obj.display()))
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

    let archive_status = if target.contains("apple-darwin") {
        Command::new("libtool")
            .arg("-static")
            .arg("-o")
            .arg(&emit)
            .arg(&obj)
            .status()
            .unwrap_or_else(|err| {
                panic!("failed to spawn libtool while archiving kv-native-sys: {err}")
            })
    } else {
        Command::new("ar")
            .arg("crs")
            .arg(&emit)
            .arg(&obj)
            .status()
            .unwrap_or_else(|err| panic!("failed to spawn ar while archiving kv-native-sys: {err}"))
    };

    assert!(
        archive_status.success(),
        "archiving kv-native-sys failed with status {archive_status}"
    );

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=kv_native");
    println!("cargo:rerun-if-changed={}", src.display());
    println!("cargo:rerun-if-env-changed=ZIG");
}
