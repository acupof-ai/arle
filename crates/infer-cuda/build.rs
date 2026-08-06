use std::process::Command;

fn main() {
    println!("cargo::rustc-check-cfg=cfg(infer_cuda_cuda_12)");
    println!("cargo:rerun-if-env-changed=CUDARC_CUDA_VERSION");

    // Prefer CUDARC_CUDA_VERSION (set by cudarc's build script / CI) over nvcc
    // probing: in CI typecheck lanes there is no nvcc, but cudarc is compiled
    // against a specific CUDA version and only exposes matching APIs.
    let is_cuda_12 = if let Ok(ver_str) = std::env::var("CUDARC_CUDA_VERSION") {
        ver_str.parse::<u32>().map(|v| v >= 12000).unwrap_or(false)
    } else if let Ok(output) = Command::new("nvcc").arg("--version").output()
        && output.status.success()
    {
        let s = String::from_utf8_lossy(&output.stdout);
        s.find("release ")
            .and_then(|rel| {
                let ver = &s[rel + 8..];
                ver.find(".").and_then(|dot| ver[..dot].parse::<u32>().ok())
            })
            .map(|major| major >= 12)
            .unwrap_or(false)
    } else {
        false
    };

    if is_cuda_12 {
        println!("cargo:rustc-cfg=infer_cuda_cuda_12");
    }
}
