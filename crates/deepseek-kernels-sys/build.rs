//! Builds `libdeepseek_kernels.a`: the vendored FlashMLA / DeepGEMM /
//! FA3-hopper kernels plus the thin C-ABI shims over them. Dependents read the
//! `links = "deepseek_kernels"` metadata (`DEP_DEEPSEEK_KERNELS_*`) for the
//! enable flags and vendored include roots.

use std::path::{Path, PathBuf};
use std::process::Command;

mod cuda_build;
use cuda_build::*;

/// Resolve the CUTLASS include tree for DeepGEMM. The producer (compile path)
/// and the consumer (prebuilt path) must agree, or the prebuilt producer
/// contract mismatches on `deepgemm-native`.
fn resolve_deepgemm_cutlass_include(deepgemm_root: &Path) -> PathBuf {
    if let Some(dir) = env_nonempty("ARLE_DEEPGEMM_CUTLASS_INCLUDE") {
        return PathBuf::from(dir);
    }
    let bundled = deepgemm_root.join("third-party/cutlass/include");
    if bundled.join("cutlass/arch/barrier.h").is_file() {
        return bundled;
    }
    // DeepGEMM's own cutlass submodule is not vendored; fall back to the
    // FlashMLA vendored cutlass, which carries the same Hopper barrier header.
    let flashmla_cutlass = Path::new("vendor/flashmla/csrc/cutlass/include");
    if flashmla_cutlass.join("cutlass/arch/barrier.h").is_file() {
        return flashmla_cutlass.to_path_buf();
    }
    bundled
}

fn env_flag(name: &str) -> bool {
    matches!(
        std::env::var(name).as_deref(),
        Ok("1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON")
    )
}

fn emit_flag(key: &str, enabled: bool) {
    println!("cargo:{key}={}", u8::from(enabled));
}

fn main() {
    // Relative paths below (csrc/, vendor/) are crate-relative.
    let manifest_dir =
        PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    std::env::set_current_dir(&manifest_dir).expect("chdir to CARGO_MANIFEST_DIR");

    let flashmla_root = Path::new("vendor/flashmla");
    let fa3_root = Path::new("vendor/flash-attention");
    println!("cargo:root={}", manifest_dir.display());
    println!(
        "cargo:cutlass_include={}",
        manifest_dir
            .join(flashmla_root)
            .join("csrc/cutlass/include")
            .display()
    );
    println!("cargo:fa3_root={}", manifest_dir.join(fa3_root).display());

    if std::env::var("CARGO_FEATURE_CUDA").is_err() {
        println!("cargo:rerun-if-env-changed=CARGO_FEATURE_CUDA");
        return;
    }
    if std::env::var("CARGO_FEATURE_NO_CUDA").is_ok() {
        println!("cargo:rerun-if-env-changed=CARGO_FEATURE_NO_CUDA");
        return;
    }

    let cuda_path = std::env::var("CUDA_HOME")
        .or_else(|_| std::env::var("CUDA_PATH"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=ARLE_CUDA_KERNELS_PREBUILT_DIR");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let sm_targets = detect_sm_targets();
    validate_sm_set(&sm_targets);
    let legacy_volta_build = has_legacy_volta(&sm_targets);

    // FlashMLA SM90 sparse prefill — vendored at `vendor/flashmla/` (pin
    // df022ebafb88578eab9f0300606ee765608d8b5c). Add the 5 .cu files needed
    // by `arle_flashmla_shim.cu`: fwd.cu + 4 phase1 instantiations.
    // Hopper only (DSv4 target = H20 / SM90a); SM100 sources are skipped.
    // CUTLASS ships inside vendor/flashmla/csrc/cutlass/ (NVIDIA tag
    // 147f5673 — FlashMLA submodule pin). Refs sgl-kernel/cmake/flashmla.cmake.
    println!("cargo:rerun-if-env-changed=ARLE_CUDA_DISABLE_FLASHMLA");
    println!("cargo:rerun-if-env-changed=ARLE_CUDA_DISABLE_FLASHMLA_DECODE");
    let flashmla_stub = Path::new("csrc/attention/arle_flashmla_decode_stubs.cu");
    // FlashMLA is SM90-only sparse-FP8 (prefill + decode). Legacy Volta (sm_70)
    // has no FP8, so the SM90 instantiations fail to compile there — fall back
    // to the cudaErrorNotSupported stub path, same as an explicit opt-out.
    let enable_flashmla =
        flashmla_root.is_dir() && !env_flag("ARLE_CUDA_DISABLE_FLASHMLA") && !legacy_volta_build;
    if enable_flashmla && env_flag("ARLE_CUDA_DISABLE_FLASHMLA_DECODE") {
        panic!(
            "ARLE_CUDA_DISABLE_FLASHMLA_DECODE would create a FlashMLA half-state. \
             Disable FlashMLA entirely with ARLE_CUDA_DISABLE_FLASHMLA=1, or build the real decode shim."
        );
    }
    let enable_flashmla_decode = enable_flashmla;

    // FA3 hopper fwd + bwd (hdim256/bf16/sm90) — vendored at
    // `vendor/flash-attention/` (Dao-AILab/flash-attention @ fc8cbad6, cutlass
    // pin 71275920). The vendored tree plus an sm_90 target is the whole gate:
    // the instantiation units are nvcc-heavy, but only an sm_90 build compiles
    // them and that is exactly the build that wants FA3. The C-ABI shim over
    // these units (`arle_fa3_shim.cu`) lives in `cuda-kernels`: its quantized
    // path calls cuda-kernels' paged-KV dequant kernels.
    let enable_fa3 =
        fa3_root.join("hopper").is_dir() && sm_targets.iter().any(|target| target.sm == "90");

    println!("cargo:rerun-if-env-changed=NVCC_CCBIN");
    println!("cargo:rerun-if-env-changed=ARLE_CUDA_DISABLE_DEEPGEMM_NATIVE");
    println!("cargo:rerun-if-env-changed=ARLE_DEEPGEMM_ROOT");
    println!("cargo:rerun-if-env-changed=ARLE_DEEPGEMM_CUTLASS_INCLUDE");
    println!("cargo:rerun-if-env-changed=DG_JIT_USE_RUNTIME_API");
    // DeepGEMM availability is also a RUNTIME preflight probe (`cuda_kernels::
    // has_deepgemm_native`), not a cfg — the non-native stub exports the same
    // bridge symbols, so it can't be build-determined.
    let deepgemm_root = std::env::var("ARLE_DEEPGEMM_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("vendor/deepgemm"));
    let deepgemm_root = if deepgemm_root.is_absolute() {
        deepgemm_root
    } else {
        std::env::current_dir()
            .expect("failed to resolve deepseek-kernels-sys build cwd")
            .join(deepgemm_root)
    };
    let deepgemm_library_root = deepgemm_root.join("deep_gemm");
    let deepgemm_cutlass_include = resolve_deepgemm_cutlass_include(&deepgemm_root);
    let deepgemm_cutlass_include = if deepgemm_cutlass_include.is_absolute() {
        deepgemm_cutlass_include
    } else {
        std::env::current_dir()
            .expect("failed to resolve deepseek-kernels-sys build cwd")
            .join(deepgemm_cutlass_include)
    };
    // DeepGEMM FP8-native dense/grouped GEMM: DEFAULT-ON when the build can support it
    // — a Hopper sm_90 target AND the vendored source present — so FP8 prefill takes the
    // fastest path with no manual flag. Mirrors the FlashMLA auto-detect above. Opt out with
    // ARLE_CUDA_DISABLE_DEEPGEMM_NATIVE=1. If unbuilt/unsupported, the runtime preflight
    // falls back to dequant→GEMM (never GEMV for prefill — see infer-cuda quant_linear).
    let deepgemm_buildable = sm_targets.iter().any(|s| s.sm.starts_with("90"))
        && deepgemm_library_root.is_dir()
        && deepgemm_cutlass_include
            .join("cutlass/arch/barrier.h")
            .is_file();
    let enable_deepgemm_native =
        !env_flag("ARLE_CUDA_DISABLE_DEEPGEMM_NATIVE") && deepgemm_buildable;

    emit_flag("flashmla", enable_flashmla);
    emit_flag("fa3", enable_fa3);
    emit_flag("deepgemm_native", enable_deepgemm_native);

    // Prebuilt bundle: the archive ships next to cuda-kernels' archives and is
    // hash-verified against the producer manifest by cuda-kernels' build.rs.
    if let Some(prebuilt_dir) = env_nonempty("ARLE_CUDA_KERNELS_PREBUILT_DIR") {
        let prebuilt_dir = PathBuf::from(prebuilt_dir);
        let lib = prebuilt_dir.join("libdeepseek_kernels.a");
        assert!(
            lib.is_file(),
            "ARLE_CUDA_KERNELS_PREBUILT_DIR={} is missing libdeepseek_kernels.a (bundle predates the \
             deepseek-kernels-sys split, producer schema < 4); rebuild the bundle",
            prebuilt_dir.display()
        );
        println!("cargo:rerun-if-changed={}", lib.display());
        println!("cargo:lib={}", lib.display());
        println!("cargo:rustc-link-search=native={}", prebuilt_dir.display());
        println!("cargo:rustc-link-lib=static=deepseek_kernels");
        emit_cuda_system_link_libs(&cuda_path);
        return;
    }

    println!("cargo:rerun-if-env-changed=ARLE_NVCC_WRAPPER");
    println!("cargo:rerun-if-env-changed=ARLE_NVCC_SPLIT_COMPILE");
    let nvcc_wrapper = env_nonempty("ARLE_NVCC_WRAPPER");
    let nvcc_split_compile = env_nonempty("ARLE_NVCC_SPLIT_COMPILE");
    let nvcc = format!("{}/bin/nvcc", cuda_path);
    let arch_args = nvcc_arch_args(&sm_targets);

    let csrc_dir = Path::new("csrc");
    let mut cu_files: Vec<PathBuf> = Vec::new();
    collect_cu_files(csrc_dir, &mut cu_files);

    // `collect_cu_files` sees the fallback stub because it lives under csrc/.
    // Drop it first so FlashMLA builds link exactly one implementation of the
    // prefill/decode FFI symbols. Otherwise the archive order can satisfy
    // `arle_flashmla_sm90_sparse_prefill_fwd` from the stub before the real
    // shim object is considered, turning default-on FlashMLA into a runtime
    // cudaErrorNotSupported.
    cu_files.retain(|p| p != flashmla_stub);
    if !enable_flashmla {
        // FlashMLA SM90 disabled (likely SM89-only box or explicit opt-out).
        // Drop the SM90-coupled shims from cu_files — they include vendored
        // SM90 templates that won't compile without the FlashMLA tree, and
        // they emit symbols that the stubs below will substitute for.
        cu_files.retain(|p| {
            let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or_default();
            !matches!(stem, "arle_flashmla_shim" | "arle_flashmla_decode_shim")
        });
        // Compile a stub that satisfies the `arle_flashmla_sm90_sparse_decode_*`
        // symbol set with `cudaErrorNotSupported` returns, so the Rust crate
        // links. The runtime gate `dsv4_flashmla_decode_enabled` defaults OFF
        // so this path is never actually called in practice.
        cu_files.push(flashmla_stub.to_path_buf());
    }
    if enable_flashmla {
        assert!(
            !cu_files.iter().any(|p| p == flashmla_stub),
            "FlashMLA vendor tree is present, but the decode stub is still scheduled for nvcc"
        );
        let sparse = flashmla_root.join("csrc/sm90/prefill/sparse");
        for entry in [
            "fwd.cu",
            "instantiations/phase1_k512.cu",
            "instantiations/phase1_k512_topklen.cu",
            "instantiations/phase1_k576.cu",
            "instantiations/phase1_k576_topklen.cu",
        ] {
            cu_files.push(sparse.join(entry));
        }

        if enable_flashmla_decode {
            // FlashMLA SM90 sparse decode — fp8 KV cache + split-KV combine +
            // CPU-side decode scheduler. 4 model×head instantiations
            // (MODEL1×{64,128}, V32×{64,128}) + combine + sched-meta kernel.
            // Requires CUDA headers with __nv_fp8_e8m0 and is runtime-gated by
            // `dsv4_flashmla_decode_enabled` (compile-gated on HAS_FLASHMLA).
            let decode_sparse_fp8 = flashmla_root.join("csrc/sm90/decode/sparse_fp8");
            for entry in [
                "instantiations/model1_persistent_h64.cu",
                "instantiations/model1_persistent_h128.cu",
                "instantiations/v32_persistent_h64.cu",
                "instantiations/v32_persistent_h128.cu",
            ] {
                cu_files.push(decode_sparse_fp8.join(entry));
            }
            cu_files.push(flashmla_root.join("csrc/smxx/decode/combine/combine.cu"));
            cu_files.push(
                flashmla_root
                    .join("csrc/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu"),
            );
        }
    }

    if enable_fa3 {
        for entry in [
            "instantiations/flash_fwd_hdim256_bf16_sm90.cu",
            "instantiations/flash_fwd_hdim256_bf16_split_sm90.cu",
            "instantiations/flash_fwd_hdim256_bf16_packgqa_sm90.cu",
            "instantiations/flash_fwd_hdim256_bf16_paged_sm90.cu",
            "instantiations/flash_fwd_hdim256_bf16_paged_split_sm90.cu",
            // fp8 operands for the quantized-pool prefill shim (paged only).
            "instantiations/flash_fwd_hdim256_e4m3_paged_sm90.cu",
            "instantiations/flash_fwd_hdim256_e4m3_paged_split_sm90.cu",
            "instantiations/flash_bwd_hdim256_bf16_sm90.cu",
            "flash_fwd_combine.cu",
            // Defines prepare_varlen_num_blocks — referenced by the launch
            // template's runtime VARLEN_SWITCH even on the non-varlen path,
            // so it must link whenever any fwd instantiation does.
            "flash_prepare_scheduler.cu",
        ] {
            cu_files.push(fa3_root.join("hopper").join(entry));
        }
    }

    // Keep a stable compile order independent of filesystem iteration order.
    cu_files.sort();

    if enable_deepgemm_native {
        println!(
            "cargo:warning=DeepGEMM native enabled (sm_90 + vendored source; set ARLE_CUDA_DISABLE_DEEPGEMM_NATIVE=1 to opt out)"
        );
    }

    let ccbin = std::env::var("NVCC_CCBIN").ok();
    let mut obj_files = Vec::new();
    let mut nvcc_jobs = Vec::new();
    for cu_file in &cu_files {
        let stem = cu_file.file_stem().unwrap().to_str().unwrap();
        let obj_file = out_dir.join(format!("{}_cuda.o", stem));

        let mut nvcc_args = vec![
            "-c".to_string(),
            cu_file.to_string_lossy().to_string(),
            "-o".to_string(),
            obj_file.to_string_lossy().to_string(),
            "-O3".to_string(),
        ];
        if let Some(bin) = ccbin.as_deref() {
            nvcc_args.push(format!("-ccbin={bin}"));
        }
        if enable_deepgemm_native {
            nvcc_args.push("-DARLE_ENABLE_DEEPGEMM_NATIVE=1".to_string());
        }
        if let Some(split_compile) = nvcc_split_compile.as_deref() {
            nvcc_args.push(format!("--split-compile={split_compile}"));
        }
        // FlashMLA (sparse prefill/decode) and FA3 hopper kernels use
        // thread-block clusters + WGMMA that require the sm_90a arch variant.
        // Compiling them for the rest of the global arch list (sm_80/86/89, or
        // plain sm_90) hard-fails ("cannot specify max blocks per cluster").
        // Force sm_90a-ONLY for these TUs, independent of TORCH_CUDA_ARCH_LIST,
        // so a T1 release binary (8.0;8.6;8.9;9.0) carries FlashMLA-sm_90a
        // alongside the full T1 arch set for every other kernel — the FlashMLA
        // path is dispatched only on sm_90 hardware by the runtime gate
        // (dsv4_flashmla_decode_enabled), dormant elsewhere. Mirrors upstream
        // FlashMLA/FA3, which ship sm_90a-only.
        let is_flashmla_kernel = cu_file.components().any(|c| c.as_os_str() == "flashmla");
        let is_fa3_kernel = cu_file
            .components()
            .any(|c| c.as_os_str() == "flash-attention");
        let is_sm90a_only = is_flashmla_kernel
            || is_fa3_kernel
            || matches!(stem, "arle_flashmla_shim" | "arle_flashmla_decode_shim");
        if is_sm90a_only {
            nvcc_args.push("-gencode=arch=compute_90a,code=sm_90a".to_string());
        } else {
            nvcc_args.extend(arch_args.clone());
        }
        nvcc_args.extend(["--compiler-options".to_string(), "-fPIC".to_string()]);
        nvcc_args.push("-Icsrc".to_string());

        if enable_deepgemm_native && stem == "deepgemm_native" {
            nvcc_args.extend([
                "-std=c++17".to_string(),
                "--expt-relaxed-constexpr".to_string(),
                "-Wno-deprecated-declarations".to_string(),
                format!("-I{}/include", cuda_path),
                format!("-I{}", deepgemm_root.display()),
                format!("-I{}", deepgemm_root.join("csrc").display()),
                format!("-I{}", deepgemm_library_root.join("include").display()),
                format!("-I{}", deepgemm_cutlass_include.display()),
                format!(
                    "-I{}",
                    deepgemm_root.join("third-party/fmt/include").display()
                ),
                format!(
                    "-DARLE_DEEPGEMM_DEFAULT_LIBRARY_ROOT=\"{}\"",
                    deepgemm_library_root.display()
                ),
                format!("-DARLE_DEEPGEMM_DEFAULT_CUDA_HOME=\"{}\"", cuda_path),
                format!(
                    "-DARLE_DEEPGEMM_DEFAULT_CUTLASS_INCLUDE=\"{}\"",
                    deepgemm_cutlass_include.display()
                ),
            ]);
            if env_flag("DG_JIT_USE_RUNTIME_API") {
                nvcc_args.push("-DDG_JIT_USE_RUNTIME_API=1".to_string());
            }
        }

        // FlashMLA SM90 sparse prefill kernels + shim. Mirror
        // sgl-kernel/cmake/flashmla.cmake flags so we inherit upstream's
        // tuning. CUTLASS include is FlashMLA's vendored copy (NVIDIA
        // CUTLASS tag 147f5673 from the FlashMLA submodule). The sm_90a
        // gencode is set above (is_sm90a_only) — FlashMLA's WGMMA/cluster
        // primitives require the arch variant and only sm_90a.
        if is_sm90a_only
            && (is_flashmla_kernel
                || stem == "arle_flashmla_shim"
                || stem == "arle_flashmla_decode_shim")
        {
            nvcc_args.extend([
                "-std=c++17".to_string(),
                "--expt-relaxed-constexpr".to_string(),
                "--expt-extended-lambda".to_string(),
                "--use_fast_math".to_string(),
                "-Xcudafe=--diag_suppress=177".to_string(),
                format!("-I{}", flashmla_root.join("csrc").display()),
                format!("-I{}", flashmla_root.join("csrc/cutlass/include").display()),
                format!(
                    "-I{}",
                    flashmla_root.join("csrc/kerutils/include").display()
                ),
            ]);
        }

        // FA3 hopper units. Flag set mirrors hopper/setup.py and MUST stay
        // identical to cuda-kernels' `arle_fa3_shim` flags (same templates on
        // both sides of the archive boundary): NDEBUG is upstream-marked
        // "otherwise performance is severely impacted"; EXTENDED_MMA_SHAPES is
        // required for FA3's WGMMA tiles. DISABLE_{LOCAL,APPENDKV} prune
        // template combinations never dispatched (causal/full only).
        if is_sm90a_only && is_fa3_kernel {
            nvcc_args.extend([
                "-std=c++17".to_string(),
                "--expt-relaxed-constexpr".to_string(),
                "--expt-extended-lambda".to_string(),
                "--use_fast_math".to_string(),
                "-DNDEBUG".to_string(),
                "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED".to_string(),
                "-DCUTLASS_ENABLE_GDC_FOR_SM90".to_string(),
                "-DCUTLASS_DEBUG_TRACE_LEVEL=0".to_string(),
                "-DFLASHATTENTION_DISABLE_LOCAL".to_string(),
                "-DFLASHATTENTION_DISABLE_APPENDKV".to_string(),
                "-Xcudafe=--diag_suppress=177".to_string(),
                format!("-I{}", fa3_root.join("hopper").display()),
                format!("-I{}", fa3_root.join("csrc/cutlass/include").display()),
            ]);
        }

        nvcc_jobs.push(NvccJob {
            cu_file: cu_file.clone(),
            args: nvcc_args,
        });
        obj_files.push(obj_file);
    }

    run_nvcc_jobs(&nvcc, nvcc_wrapper.as_deref(), &nvcc_jobs);

    if enable_flashmla {
        let stub_obj = "arle_flashmla_decode_stubs_cuda.o";
        assert!(
            !obj_files.iter().any(|path| path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name == stub_obj)),
            "FlashMLA vendor tree is present, but {stub_obj} would be archived"
        );
    }

    let lib = out_dir.join("libdeepseek_kernels.a");
    match std::fs::remove_file(&lib) {
        Ok(()) => {}
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => panic!("Failed to remove stale {}: {}", lib.display(), err),
    }
    let mut ar_args = vec!["rcs".to_string(), lib.to_string_lossy().to_string()];
    ar_args.extend(
        obj_files
            .into_iter()
            .map(|path| path.to_string_lossy().to_string()),
    );
    let status = Command::new("ar")
        .args(&ar_args)
        .status()
        .expect("Failed to run ar");
    assert!(status.success(), "ar failed");

    println!("cargo:lib={}", lib.display());
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=deepseek_kernels");
    emit_cuda_system_link_libs(&cuda_path);
    if enable_deepgemm_native {
        println!(
            "cargo:warning=DeepGEMM native bridge enabled, root={}",
            deepgemm_root.display()
        );
    }

    // Recursive watch — `rerun-if-changed=csrc/` alone only watches the
    // immediate dir entries.
    println!("cargo:rerun-if-changed=csrc/");
    emit_rerun_recursive(Path::new("csrc"));
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=cuda_build.rs");
    println!("cargo:rerun-if-env-changed=TORCH_CUDA_ARCH_LIST");
    println!("cargo:rerun-if-env-changed=CMAKE_CUDA_ARCHITECTURES");
}
