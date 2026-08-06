use std::process::Command;

fn main() {
    println!("cargo::rustc-check-cfg=cfg(infer_cuda_cuda_12)");
    if let Ok(output) = Command::new("nvcc").arg("--version").output()
        && output.status.success()
    {
        let s = String::from_utf8_lossy(&output.stdout);
        if let Some(rel) = s.find("release ") {
            let ver = &s[rel + 8..];
            if let Some(dot) = ver.find(".")
                && let Ok(major) = ver[..dot].parse::<u32>()
                && major >= 12
            {
                println!("cargo:rustc-cfg=infer_cuda_cuda_12");
            }
        }
    }
}
