//! CUDA build helpers shared by the `deepseek-kernels-sys` and `cuda-kernels`
//! build scripts (the latter includes this file via `#[path]`): SM-target
//! detection + tier policy, nvcc job pool, csrc walk, and the CUDA system link
//! set. Both crates must resolve the same SM targets from the same env.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Tier-1 SMs: default-compiled fat-binary set. A100 / A10·3090 / L4·4090 / H100.
pub const T1_SMS: &[&str] = &["80", "86", "89", "90"];

/// Tier-2 SMs: opt-in via TORCH_CUDA_ARCH_LIST. B100·B200 / RTX 5090.
pub const T2_SMS: &[&str] = &["100", "120"];

/// Legacy Volta SMs: opt-in / auto-detect only, built as a separate SM-pinned binary.
pub const LEGACY_VOLTA_SMS: &[&str] = &["70"];

pub fn is_supported_sm(sm: &str) -> bool {
    T1_SMS.contains(&sm) || T2_SMS.contains(&sm) || LEGACY_VOLTA_SMS.contains(&sm)
}

pub fn is_legacy_volta_sm(sm: &str) -> bool {
    LEGACY_VOLTA_SMS.contains(&sm)
}

pub fn has_legacy_volta(sm_targets: &[SmSpec]) -> bool {
    sm_targets.iter().any(|spec| is_legacy_volta_sm(&spec.sm))
}

#[derive(Clone, Debug)]
pub struct SmSpec {
    pub sm: String,
    /// `+PTX` requested for this SM (per PyTorch TORCH_CUDA_ARCH_LIST convention).
    pub ptx: bool,
}

/// Parse a single SM token. Accepts:
///   - PyTorch:  `8.0`, `9.0`, `12.0+PTX`
///   - CMake:    `80`, `90`, `120`
///   - nvcc:     `sm_80`, `compute_90`
pub fn parse_sm_token(raw: &str) -> Option<SmSpec> {
    let token = raw.trim().trim_matches('"');
    if token.is_empty() {
        return None;
    }

    let (token, ptx) = if let Some(stem) = token
        .strip_suffix("+PTX")
        .or_else(|| token.strip_suffix("+ptx"))
    {
        (stem.trim_end(), true)
    } else {
        (token, false)
    };

    let token = token
        .strip_prefix("sm_")
        .or_else(|| token.strip_prefix("compute_"))
        .unwrap_or(token);

    let sm = if let Some((major, minor)) = token.split_once('.') {
        if major.chars().all(|c| c.is_ascii_digit()) && minor.chars().all(|c| c.is_ascii_digit()) {
            format!("{major}{minor}")
        } else {
            return None;
        }
    } else if token.chars().all(|c| c.is_ascii_digit()) {
        if token.len() == 1 {
            format!("{token}0")
        } else {
            token.to_string()
        }
    } else {
        return None;
    };

    Some(SmSpec { sm, ptx })
}

/// Reject SMs outside the explicit whitelist. Turing/Pascal/older stay unsupported.
pub fn validate_sm(spec: &SmSpec, source: &str) {
    if !is_supported_sm(&spec.sm) {
        panic!(
            "Unsupported CUDA compute capability 'sm_{}' from {}. \
             ARLE supports T1={{80,86,89,90}} (default), T2={{100,120}} (opt-in), \
             and legacy Volta={{70}} as a separate SM-pinned build. \
             Turing/Pascal/unknown SMs are rejected. \
             See docs/environment.md and docs/support-matrix.md. \
             To restrict targets explicitly: TORCH_CUDA_ARCH_LIST=\"8.0;8.6;8.9;9.0\".",
            spec.sm, source
        );
    }
}

pub fn validate_sm_set(sm_targets: &[SmSpec]) {
    if has_legacy_volta(sm_targets) && sm_targets.len() != 1 {
        panic!(
            "sm_70 legacy Volta builds must be SM-pinned and cannot be mixed with T1/T2 targets. \
             Build V100 with TORCH_CUDA_ARCH_LIST=\"7.0\"; build the T1 release binary separately \
             with TORCH_CUDA_ARCH_LIST=\"8.0;8.6;8.9;9.0\". \
             This keeps T1 cubins free of sm_70 fallback code and keeps sm_70 binaries free of \
             T1/Hopper-only kernels. See docs/environment.md."
        );
    }
}

/// Parse TORCH_CUDA_ARCH_LIST / CMAKE_CUDA_ARCHITECTURES.
/// Separators: `;`, `,`, whitespace. Empty tokens skipped. Each token validated.
///
/// Empty result panics: an empty / whitespace / separators-only env var is
/// almost always a typo (e.g. `TORCH_CUDA_ARCH_LIST=""`), and silently
/// continuing would emit AOT dispatch wrappers with zero `case` arms — every
/// runtime call would then return `CUDA_ERROR_NOT_SUPPORTED`. Fail fast.
pub fn parse_arch_list(raw: &str, source: &str) -> Vec<SmSpec> {
    let mut sms: BTreeSet<String> = BTreeSet::new();
    let mut ptx_for: BTreeSet<String> = BTreeSet::new();

    for token in raw.split(|c: char| c == ';' || c == ',' || c.is_whitespace()) {
        if token.is_empty() {
            continue;
        }
        let spec = parse_sm_token(token).unwrap_or_else(|| {
            panic!(
                "Failed to parse SM token '{token}' from {source} (raw='{raw}'). \
                 Expected format e.g. '8.0', '8.0+PTX', '80', 'sm_80'."
            )
        });
        validate_sm(&spec, source);
        if spec.ptx {
            ptx_for.insert(spec.sm.clone());
        }
        sms.insert(spec.sm);
    }

    if sms.is_empty() {
        panic!(
            "{source} is set but parsed to zero SM targets (raw='{raw}'). \
             Either unset {source} (auto-detect via nvidia-smi or T1 default) \
             or pass a non-empty list, e.g. '8.0;8.6;8.9;9.0' (T1) or '9.0' (H100 only)."
        );
    }

    sms.into_iter()
        .map(|sm| SmSpec {
            ptx: ptx_for.contains(&sm),
            sm,
        })
        .collect()
}

pub fn sm_targets_from_nvidia_smi() -> Option<Vec<SmSpec>> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut sms: BTreeSet<String> = BTreeSet::new();
    for line in stdout.lines() {
        let cap = line.split(',').next().unwrap_or(line).trim();
        if cap.is_empty() {
            continue;
        }
        let spec = parse_sm_token(cap)
            .unwrap_or_else(|| panic!("nvidia-smi reported unparseable compute_cap '{cap}'."));
        validate_sm(&spec, "nvidia-smi --query-gpu=compute_cap");
        sms.insert(spec.sm);
    }

    if sms.is_empty() {
        None
    } else {
        Some(
            sms.into_iter()
                .map(|sm| SmSpec { sm, ptx: false })
                .collect(),
        )
    }
}

pub fn detect_sm_targets() -> Vec<SmSpec> {
    if let Ok(env) = std::env::var("TORCH_CUDA_ARCH_LIST") {
        return parse_arch_list(&env, "TORCH_CUDA_ARCH_LIST");
    }
    if let Ok(env) = std::env::var("CMAKE_CUDA_ARCHITECTURES") {
        return parse_arch_list(&env, "CMAKE_CUDA_ARCHITECTURES");
    }

    if let Some(sms) = sm_targets_from_nvidia_smi() {
        return sms;
    }

    println!(
        "cargo:warning=No GPU detected and TORCH_CUDA_ARCH_LIST not set; defaulting to T1 SMs (sm_80, sm_86, sm_89, sm_90). \
         To target Blackwell (sm_100, sm_120), set TORCH_CUDA_ARCH_LIST=\"...;10.0\" or \"...;12.0\". \
         See docs/environment.md."
    );
    T1_SMS
        .iter()
        .map(|s| SmSpec {
            sm: (*s).to_string(),
            ptx: false,
        })
        .collect()
}

pub fn nvcc_arch_args(sm_targets: &[SmSpec]) -> Vec<String> {
    let mut args = Vec::new();
    for spec in sm_targets {
        // SASS for this SM.
        args.push("-gencode".to_string());
        args.push(format!("arch=compute_{sm},code=sm_{sm}", sm = spec.sm));
        // Per-SM PTX requested via `+PTX` suffix.
        if spec.ptx {
            args.push("-gencode".to_string());
            args.push(format!("arch=compute_{sm},code=compute_{sm}", sm = spec.sm));
        }
    }

    // Always emit PTX for the highest SM as a forward-compat JIT fallback for
    // newer hardware (e.g. T2 sm_120 when only T1 is built). Skip if that SM
    // already requested `+PTX`.
    if let Some(max_spec) = sm_targets
        .iter()
        .max_by_key(|s| s.sm.parse::<u32>().unwrap_or(0))
        && !max_spec.ptx
    {
        args.push("-gencode".to_string());
        args.push(format!(
            "arch=compute_{sm},code=compute_{sm}",
            sm = max_spec.sm
        ));
    }

    args
}

// Recursively collect every `.cu` file under `dir` so domain subdirs
// (attention/, gemm/, moe/, kv/, quant/, sampling/, norm/, recurrent/,
// elementwise/, ...) are picked up automatically.
pub fn collect_cu_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(err) => panic!("Failed to read {}: {}", dir.display(), err),
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_cu_files(&path, out);
            continue;
        }
        if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("._"))
        {
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) == Some("cu") {
            out.push(path);
        }
    }
}

// Recursively walk `dir` and emit `cargo:rerun-if-changed` for every file.
// Cargo's `rerun-if-changed=<dir>` directive only watches the *immediate*
// directory entries, NOT subdirectories — so `rerun-if-changed=csrc/` alone
// silently misses changes to `csrc/kv/*.cu`, `csrc/attention/*.cu`, etc., and
// stale cubins ship while source diffs sit dormant. Emit one directive per
// file so every `.cu`/`.cuh`/`.h` edit invalidates the build.
pub fn emit_rerun_recursive(dir: &Path) {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(err) => panic!("Failed to read {}: {}", dir.display(), err),
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("._"))
        {
            continue;
        }
        if path.is_dir() {
            println!("cargo:rerun-if-changed={}", path.display());
            emit_rerun_recursive(&path);
        } else {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}

pub fn env_nonempty(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

pub fn emit_cuda_system_link_libs(cuda_path: &str) {
    if cfg!(target_os = "windows") {
        println!("cargo:rustc-link-search=native={}/lib/x64", cuda_path);
    } else {
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    }
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cublas");
    println!("cargo:rustc-link-lib=cublasLt");
    // NCCL feature: link libnccl so multi-rank TP/EP builds don't need a manual
    // `RUSTFLAGS=-lnccl` workaround. cargo sets CARGO_FEATURE_NCCL when the
    // `nccl` feature (=> `cuda`) is active, so this is inert on no-cuda builds.
    if std::env::var_os("CARGO_FEATURE_NCCL").is_some() {
        // NCCL ships either with CUDA (lib64, already searched above) or as a
        // system package; add the common Linux multiarch dir + an NCCL_HOME
        // override so `-lnccl` resolves without per-invocation link flags.
        if let Some(nccl_home) = env_nonempty("NCCL_HOME") {
            println!("cargo:rustc-link-search=native={nccl_home}/lib");
        }
        if !cfg!(target_os = "windows") {
            println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
        }
        println!("cargo:rustc-link-lib=nccl");
    }
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=c++");
    } else if !cfg!(target_os = "windows") {
        println!("cargo:rustc-link-lib=stdc++");
        let gcc_major = Command::new("gcc")
            .arg("-dumpfullversion")
            .output()
            .ok()
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .and_then(|version| version.split('.').next()?.trim().parse::<u32>().ok())
            .unwrap_or(99);
        if gcc_major < 9 {
            println!("cargo:rustc-link-lib=stdc++fs");
        }
    }
}

pub fn tool_command(tool: &str, wrapper: Option<&str>) -> Command {
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

/// One queued nvcc invocation: the source (for error messages) plus the full
/// pre-built argv. The output path is already baked into `args`.
pub struct NvccJob {
    pub cu_file: PathBuf,
    pub args: Vec<String>,
}

pub fn run_nvcc_job(nvcc: &str, wrapper: Option<&str>, job: &NvccJob) {
    let status = tool_command(nvcc, wrapper)
        .args(&job.args)
        .status()
        .unwrap_or_else(|_| panic!("Failed to run nvcc for {}", job.cu_file.display()));
    assert!(
        status.success(),
        "nvcc compilation failed for {}",
        job.cu_file.display()
    );
}

/// Bounded worker count for the nvcc pool. Capped at 8 because a single
/// multi-arch nvcc invocation can take 1-2 GB of RAM; `ARLE_NVCC_PARALLEL=1`
/// restores the previous serial behavior.
pub fn nvcc_parallelism(jobs: usize) -> usize {
    println!("cargo:rerun-if-env-changed=ARLE_NVCC_PARALLEL");
    let configured = env_nonempty("ARLE_NVCC_PARALLEL").and_then(|v| v.parse::<usize>().ok());
    let default = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(8);
    configured.unwrap_or(default).clamp(1, jobs.max(1))
}

/// Run all queued nvcc jobs through a bounded pool. Archive order is decided
/// by the caller's `obj_files` (queue order), not completion order, so the
/// `ar` symbol-resolution order is identical to the old serial loop. A panic
/// in any worker is re-raised when the scope joins, failing the build.
pub fn run_nvcc_jobs(nvcc: &str, wrapper: Option<&str>, jobs: &[NvccJob]) {
    let workers = nvcc_parallelism(jobs.len());
    if workers <= 1 {
        for job in jobs {
            run_nvcc_job(nvcc, wrapper, job);
        }
        return;
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                loop {
                    let idx = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(job) = jobs.get(idx) else { break };
                    run_nvcc_job(nvcc, wrapper, job);
                }
            });
        }
    });
}
