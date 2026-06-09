// SPDX-License-Identifier: Apache-2.0
// Phase B-2 build script for deepep-sys.
//
// When ARLE_DEEPEP_DIR is set + cuda toolchain present, compile our
// torch-free C wrapper + DeepEP's csrc/kernels/legacy/{intranode,layout}
// .cu files into a static archive and link against libcudart. When
// either is missing, emit a "deepep not built" cargo warning and rely
// on src/lib.rs's stub returning DeepEpError::NotBuilt at runtime.
//
// DeepEP is pinned at d4f41e4, whose intranode/layout kernels live under
// csrc/kernels/legacy/ (upstream tree name — that is literally where the
// sources are). We compile against that single layout only.
//
// NVSHMEM low-latency (internode_ll) path — T4-LL de-risk (2026-06-09):
// when ARLE_DEEPEP_NVSHMEM_DIR points at an NVSHMEM install (host
// libnvshmem_host.so.3 + device libnvshmem_device.a + include/nvshmem.h),
// additionally compile DeepEP's internode.cu + internode_ll.cu +
// backend/nvshmem.cu WITHOUT -DDISABLE_NVSHMEM, with
// -rdc=true (separable compilation for device-side NVSHMEM), run the
// nvcc -dlink device-link step against libnvshmem_device.a, archive the
// dlink object alongside, and emit the NVSHMEM host/device link
// directives. Mirrors DeepEP setup.py ~line 100-118. Unset
// ARLE_DEEPEP_NVSHMEM_DIR ⇒ byte-identical intranode-only build.

use std::path::{Path, PathBuf};
use std::process::Command;

fn env_nonempty(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn tool_command(tool: &str, wrapper: Option<&str>) -> Command {
    if let Some(wrapper) = wrapper {
        let mut parts = wrapper.split_whitespace();
        let program = parts
            .next()
            .expect("ARLE_NVCC_WRAPPER was filtered to non-empty");
        let mut command = Command::new(program);
        command.args(parts);
        command.arg(tool);
        command
    } else {
        Command::new(tool)
    }
}

/// Resolve the real on-disk filename of the first `{prefix}.so*` under
/// `<root>/lib`, preferring an unversioned `.so` symlink. NVIDIA pip
/// wheels ship only the SONAME file (e.g. `libnvshmem_host.so.3`) with no
/// `.so` symlink, so `-l:NAME` must use the real name. Mirrors DeepEP
/// setup.py `_find_versioned_so`.
fn find_versioned_so(lib_dir: &Path, prefix: &str) -> Option<String> {
    let unversioned = lib_dir.join(format!("{prefix}.so"));
    if unversioned.exists() {
        return Some(format!("{prefix}.so"));
    }
    let entries = std::fs::read_dir(lib_dir).ok()?;
    let needle = format!("{prefix}.so.");
    let mut candidates: Vec<String> = entries
        .filter_map(|e| e.ok())
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|name| name.starts_with(&needle))
        .collect();
    candidates.sort();
    candidates.into_iter().next()
}

fn main() {
    println!("cargo:rerun-if-env-changed=ARLE_DEEPEP_DIR");
    println!("cargo:rerun-if-env-changed=ARLE_DEEPEP_NVSHMEM_DIR");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=ARLE_NVCC_WRAPPER");
    println!("cargo:rerun-if-env-changed=ARLE_NVCC_SPLIT_COMPILE");
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
    // DeepEP intranode/layout kernels (upstream csrc/kernels/legacy tree,
    // pinned d4f41e4). The pinned checkout is the only layout we support;
    // bail clearly if api.cuh is missing there.
    let kernels_dir = deepep_root.join("csrc").join("kernels").join("legacy");
    if !kernels_dir.join("api.cuh").exists() {
        println!(
            "cargo:warning=ARLE_DEEPEP_DIR={} does not point at a DeepEP checkout with the \
             kernels/legacy layout (csrc/kernels/legacy/api.cuh absent) — re-pin to d4f41e4. \
             Skipping deepep-sys native build.",
            deepep_root.display()
        );
        println!("cargo:rustc-cfg=deepep_stub");
        return;
    }

    let cuda_home = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".to_string());
    let nvcc = format!("{cuda_home}/bin/nvcc");
    if !PathBuf::from(&nvcc).exists() {
        println!("cargo:warning=nvcc not found at {nvcc} — skipping deepep-sys native build.");
        println!("cargo:rustc-cfg=deepep_stub");
        return;
    }

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let archive = out_dir.join("libarle_deepep.a");
    let nvcc_wrapper = env_nonempty("ARLE_NVCC_WRAPPER");
    let nvcc_split_compile = env_nonempty("ARLE_NVCC_SPLIT_COMPILE");

    // SM90 only — DSv4 SLO target H100/H20. Other SMs would need TMA-
    // disabled variants of the DeepEP kernels.
    let arch = "-gencode=arch=compute_90,code=sm_90";

    let csrc_dir = PathBuf::from("csrc");

    // --- NVSHMEM low-latency (internode_ll) discovery -----------------
    // backend/nvshmem.cu lives under csrc/kernels/backend/. When
    // ARLE_DEEPEP_NVSHMEM_DIR resolves to a valid NVSHMEM install AND the
    // backend + internode sources exist, build the LL path with NVSHMEM
    // enabled.
    let backend_dir = deepep_root.join("csrc").join("kernels").join("backend");
    let nvshmem_dir = env_nonempty("ARLE_DEEPEP_NVSHMEM_DIR").map(PathBuf::from);
    let nvshmem_enabled = nvshmem_dir
        .as_ref()
        .map(|nv| {
            let inc_ok = nv.join("include").join("nvshmem.h").exists();
            let lib_dir = nv.join("lib");
            let dev_ok = lib_dir.join("libnvshmem_device.a").exists();
            let host_ok = find_versioned_so(&lib_dir, "libnvshmem_host").is_some();
            let src_ok = backend_dir.join("nvshmem.cu").exists()
                && kernels_dir.join("internode.cu").exists()
                && kernels_dir.join("internode_ll.cu").exists();
            if !(inc_ok && dev_ok && host_ok && src_ok) {
                println!(
                    "cargo:warning=ARLE_DEEPEP_NVSHMEM_DIR={} set but LL prerequisites \
                     missing (inc={inc_ok} dev={dev_ok} host={host_ok} src={src_ok}) — \
                     building intranode-only (no internode_ll).",
                    nv.display()
                );
            }
            inc_ok && dev_ok && host_ok && src_ok
        })
        .unwrap_or(false);

    // 1. nvcc-compile each NVSHMEM-free source to a .o (intranode/layout
    //    + the cpp wrapper), all with -DDISABLE_NVSHMEM.
    let mut sources: Vec<(PathBuf, &str)> = vec![
        (kernels_dir.join("intranode.cu"), "intranode.o"),
        (kernels_dir.join("layout.cu"), "layout.o"),
    ];
    // runtime.cu existed in DeepEP's old flat layout but was removed/inlined
    // in the kernels/legacy tree. Skip when absent — our cpp wrapper only
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
        let mut cmd = tool_command(&nvcc, nvcc_wrapper.as_deref());
        cmd.arg("-ccbin")
            .arg("g++")
            .arg("-std=c++17")
            .arg("-O2")
            .arg("-DDISABLE_NVSHMEM")
            .arg("--expt-relaxed-constexpr")
            .arg("--expt-extended-lambda");
        if let Some(split_compile) = nvcc_split_compile.as_deref() {
            cmd.arg(format!("--split-compile={split_compile}"));
        }
        if nvshmem_enabled {
            // Wrapper references arle_deepep_nvshmem_built() force-link
            // shim only when NVSHMEM objects are in the archive.
            cmd.arg("-DARLE_DEEPEP_HAVE_NVSHMEM=1");
        }
        cmd.arg("-Xcompiler")
            .arg("-fPIC")
            .arg(arch)
            .arg("-I")
            .arg(deepep_root.join("csrc"))
            // DeepEP kernels include <deep_ep/common/...> from the
            // namespaced header tree at deep_ep/include/.
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

    // 1b. NVSHMEM LL sources — compiled WITHOUT -DDISABLE_NVSHMEM, WITH
    //     -I <nvshmem>/include and -rdc=true (device-side NVSHMEM needs
    //     separable compilation). Mirrors DeepEP setup.py nvcc_flags.
    let mut rdc_objs: Vec<PathBuf> = Vec::new();
    if nvshmem_enabled {
        let nv = nvshmem_dir.as_ref().expect("nvshmem_enabled implies dir");
        let nv_inc = nv.join("include");
        // backend/{nccl,cuda_driver}.cu are DELIBERATELY EXCLUDED: they pull
        // <pybind11/...> + Python.h (DeepEP's python_api glue) and fail to
        // compile in our torch-free build. Verified 2026-06-09 that
        // internode.cu / internode_ll.cu / nvshmem.cu reference ZERO
        // deep_ep::nccl / deep_ep::cuda_driver symbols (`nm -uC` on the 3
        // objects is empty), so the LL kernel + NVSHMEM device-link path
        // does not need them.
        let ll_sources: Vec<(PathBuf, &str)> = vec![
            (kernels_dir.join("internode.cu"), "internode.o"),
            (kernels_dir.join("internode_ll.cu"), "internode_ll.o"),
            (backend_dir.join("nvshmem.cu"), "backend_nvshmem.o"),
        ];
        for (src, name) in ll_sources {
            let obj = out_dir.join(name);
            let mut cmd = tool_command(&nvcc, nvcc_wrapper.as_deref());
            cmd.arg("-ccbin")
                .arg("g++")
                .arg("-std=c++17")
                .arg("-O3")
                .arg("--expt-relaxed-constexpr")
                .arg("--expt-extended-lambda")
                // DeepEP setup.py CUDA-12 flags for the H800/H20 path:
                .arg("-rdc=true")
                .arg("--ptxas-options=--register-usage-level=10")
                .arg("--diag-suppress=128,2417")
                .arg("-DDISABLE_AGGRESSIVE_PTX_INSTRS");
            if let Some(split_compile) = nvcc_split_compile.as_deref() {
                cmd.arg(format!("--split-compile={split_compile}"));
            }
            cmd.arg("-Xcompiler")
                .arg("-fPIC")
                .arg(arch)
                .arg("-I")
                .arg(deepep_root.join("csrc"))
                .arg("-I")
                .arg(deepep_root.join("deep_ep").join("include"))
                .arg("-I")
                .arg(&csrc_dir)
                .arg("-I")
                .arg(&nv_inc)
                .arg("-c")
                .arg(&src)
                .arg("-o")
                .arg(&obj);
            let status = cmd.status().expect("spawn nvcc (nvshmem LL)");
            if !status.success() {
                panic!(
                    "nvcc failed for NVSHMEM LL source {} (status {:?}).",
                    src.display(),
                    status.code()
                );
            }
            rdc_objs.push(obj);
        }

        // 1c. Device-link step: gather all -rdc=true objects + the device
        //     NVSHMEM static lib into a single relocatable device object.
        //     Mirrors DeepEP setup.py nvcc_dlink:
        //       nvcc -dlink -L<nvshmem>/lib -lnvshmem_device ...
        let dlink_obj = out_dir.join("deepep_nvshmem_dlink.o");
        let nv_lib = nv.join("lib");
        let mut dlink = tool_command(&nvcc, nvcc_wrapper.as_deref());
        dlink.arg("-dlink").arg(arch).arg("-Xcompiler").arg("-fPIC");
        for o in &rdc_objs {
            dlink.arg(o);
        }
        dlink
            .arg(format!("-L{}", nv_lib.display()))
            .arg("-lnvshmem_device")
            .arg("-o")
            .arg(&dlink_obj);
        let status = dlink.status().expect("spawn nvcc -dlink");
        if !status.success() {
            panic!(
                "nvcc -dlink failed for NVSHMEM device-link (status {:?}).",
                status.code()
            );
        }
        // The rdc objects AND the dlink object all go into the archive.
        objs.extend(rdc_objs.iter().cloned());
        objs.push(dlink_obj);
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
        "cargo:warning=deepep-sys native build OK (archive={}, DeepEP={}, nvshmem={})",
        archive.display(),
        deepep_root.display(),
        nvshmem_enabled
    );
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=arle_deepep");
    println!("cargo:rustc-link-search=native={cuda_home}/lib64");
    println!("cargo:rustc-link-lib=cudart");

    if nvshmem_enabled {
        let nv = nvshmem_dir.as_ref().expect("nvshmem_enabled implies dir");
        let nv_lib = nv.join("lib");
        let host_so = find_versioned_so(&nv_lib, "libnvshmem_host")
            .expect("nvshmem_enabled checked host lib exists");
        println!("cargo:rustc-link-search=native={}", nv_lib.display());
        // Host lib ships as SONAME-only (libnvshmem_host.so.3) with no
        // `.so` symlink, so the plain `-lnvshmem_host` form cannot resolve.
        // Use `rustc-link-lib=dylib:+verbatim=<realname>` — `+verbatim`
        // passes the exact filename to the linker (equivalent to `-l:NAME`,
        // mirroring DeepEP setup.py `-l:libnvshmem_host.so.3`) AND, unlike
        // `rustc-link-arg`, `rustc-link-lib` PROPAGATES from a dependency
        // build script to the final binary link. (rustc-link-arg from a dep
        // is silently dropped at the bin link — the first attempt's bug.)
        // The device static lib likewise links verbatim. Both are demanded
        // by the force-link probe in deepep_buffer.cpp, so no
        // --no-as-needed is required — the symbols are genuinely undefined.
        println!("cargo:rustc-link-lib=dylib:+verbatim={host_so}");
        println!("cargo:rustc-link-lib=static:+verbatim=libnvshmem_device.a");
        // Device-link emits references to the CUDA driver API; -lcuda.
        println!("cargo:rustc-link-lib=cuda");
        // The force-link shim in deepep_buffer.cpp references
        // deep_ep::nvshmem::get_unique_id, which pulls backend_nvshmem.o →
        // its undefined nvshmem_* host syms → libnvshmem_host. Expose a cfg
        // so dependent crates can compile-gate LL-only code.
        println!("cargo:rustc-cfg=deepep_nvshmem");
    }

    println!("cargo:rustc-link-lib=stdc++");
}
