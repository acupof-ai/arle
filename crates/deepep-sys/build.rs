// SPDX-License-Identifier: Apache-2.0
// Phase B-2 build script for deepep-sys.
//
// When ARLE_DEEPEP_DIR is set + cuda toolchain present, compile our
// torch-free C wrapper + DeepEP's csrc/kernels/{intranode,layout,runtime}
// .cu files into a static archive and link against libcudart. When
// either is missing, emit a "deepep not built" cargo warning and rely
// on src/lib.rs's stub returning DeepEpError::NotBuilt at runtime.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=ARLE_DEEPEP_DIR");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-changed=csrc/deepep_buffer.cpp");
    println!("cargo:rerun-if-changed=csrc/deepep_buffer.hpp");
    println!("cargo:rerun-if-changed=build.rs");

    let Ok(deepep_dir) = std::env::var("ARLE_DEEPEP_DIR") else {
        println!(
            "cargo:warning=ARLE_DEEPEP_DIR unset — deepep-sys stub only \
             (set to the deepseek-ai/DeepEP source tree to enable)."
        );
        println!("cargo:rustc-cfg=deepep_stub");
        return;
    };
    let deepep_root = PathBuf::from(&deepep_dir);
    // DeepEP upstream refactored the intranode kernels into csrc/kernels/legacy/
    // after the phase 1.0a-iv spike was written against the old flat layout.
    // Probe both layouts and use whichever has api.cuh present.
    let (kernels_dir, is_legacy_layout) = {
        let flat = deepep_root.join("csrc").join("kernels");
        let legacy = flat.join("legacy");
        if flat.join("api.cuh").exists() {
            (flat, false)
        } else if legacy.join("api.cuh").exists() {
            (legacy, true)
        } else {
            println!(
                "cargo:warning=ARLE_DEEPEP_DIR={} missing csrc/kernels/api.cuh or \
                 csrc/kernels/legacy/api.cuh — skipping deepep-sys native build.",
                deepep_root.display()
            );
            println!("cargo:rustc-cfg=deepep_stub");
            return;
        }
    };

    let cuda_home = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".to_string());
    let nvcc = format!("{cuda_home}/bin/nvcc");
    if !PathBuf::from(&nvcc).exists() {
        println!("cargo:warning=nvcc not found at {nvcc} — skipping deepep-sys native build.");
        println!("cargo:rustc-cfg=deepep_stub");
        return;
    }

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let archive = out_dir.join("libarle_deepep.a");

    // SM90 only — DSv4 SLO target H100/H20. Other SMs would need TMA-
    // disabled variants of the DeepEP kernels.
    let arch = "-gencode=arch=compute_90,code=sm_90";

    let csrc_dir = PathBuf::from("csrc");

    // 1. nvcc-compile each source to a .o
    let mut sources: Vec<(PathBuf, &str)> = vec![
        (kernels_dir.join("intranode.cu"), "intranode.o"),
        (kernels_dir.join("layout.cu"), "layout.o"),
    ];
    // runtime.cu existed in DeepEP's flat layout but was removed/inlined in
    // the legacy subdir refactor. Skip when absent — our cpp wrapper only
    // calls intranode::{barrier,notify_dispatch,dispatch,cached_notify_
    // combine,combine} + layout::get_dispatch_layout, all defined in
    // intranode.cu + layout.cu.
    let runtime_cu = kernels_dir.join("runtime.cu");
    if runtime_cu.exists() {
        sources.push((runtime_cu, "runtime.o"));
    }
    sources.push((csrc_dir.join("deepep_buffer.cpp"), "deepep_buffer.o"));
    let mut objs = Vec::with_capacity(sources.len());
    for (src, name) in sources {
        let obj = out_dir.join(name);
        let mut cmd = Command::new(&nvcc);
        cmd.arg("-ccbin")
            .arg("g++")
            .arg("-std=c++17")
            .arg("-O2")
            .arg("-DDISABLE_NVSHMEM")
            .arg("--expt-relaxed-constexpr")
            .arg("--expt-extended-lambda");
        if is_legacy_layout {
            cmd.arg("-DARLE_DEEPEP_LEGACY_LAYOUT=1");
        }
        cmd.arg("-Xcompiler")
            .arg("-fPIC")
            .arg(arch)
            .arg("-I")
            .arg(deepep_root.join("csrc"))
            // DeepEP refactor: legacy kernels include <deep_ep/common/...>
            // from the new namespaced header tree at deep_ep/include/.
            .arg("-I")
            .arg(deepep_root.join("deep_ep").join("include"))
            .arg("-I")
            .arg(&csrc_dir)
            .arg("-c")
            .arg(&src)
            .arg("-o")
            .arg(&obj);
        let status = cmd.status().expect("spawn nvcc");
        if !status.success() {
            panic!(
                "nvcc failed for {} (status {:?}). Unset ARLE_DEEPEP_DIR \
                 to skip the deepep-sys native build.",
                src.display(),
                status.code()
            );
        }
        objs.push(obj);
    }

    // 2. Archive objects into a static lib.
    let _ = std::fs::remove_file(&archive);
    let mut ar = Command::new("ar");
    ar.arg("rcs").arg(&archive);
    for o in &objs {
        ar.arg(o);
    }
    let status = ar.status().expect("spawn ar");
    if !status.success() {
        panic!("ar failed building {}", archive.display());
    }

    println!(
        "cargo:warning=deepep-sys native build OK (archive={}, DeepEP={})",
        archive.display(),
        deepep_root.display()
    );
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=arle_deepep");
    println!("cargo:rustc-link-search=native={cuda_home}/lib64");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=stdc++");
}
