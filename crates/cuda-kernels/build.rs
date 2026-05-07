use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Tier-1 SMs: default-compiled fat-binary set. A100 / A10·3090 / L4·4090 / H100.
const T1_SMS: &[&str] = &["80", "86", "89", "90"];

/// Tier-2 SMs: opt-in via TORCH_CUDA_ARCH_LIST. B100·B200 / RTX 5090.
const T2_SMS: &[&str] = &["100", "120"];

fn is_supported_sm(sm: &str) -> bool {
    T1_SMS.contains(&sm) || T2_SMS.contains(&sm)
}

#[derive(Clone, Debug)]
struct SmSpec {
    sm: String,
    /// `+PTX` requested for this SM (per PyTorch TORCH_CUDA_ARCH_LIST convention).
    ptx: bool,
}

/// Parse a single SM token. Accepts:
///   - PyTorch:  `8.0`, `9.0`, `12.0+PTX`
///   - CMake:    `80`, `90`, `120`
///   - nvcc:     `sm_80`, `compute_90`
fn parse_sm_token(raw: &str) -> Option<SmSpec> {
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

/// Reject SMs outside the T1∪T2 whitelist. T3 (Volta/Turing/older) is unsupported.
fn validate_sm(spec: &SmSpec, source: &str) {
    if !is_supported_sm(&spec.sm) {
        panic!(
            "Unsupported CUDA compute capability 'sm_{}' from {}. \
             ARLE supports T1={{80,86,89,90}} (default) and T2={{100,120}} (opt-in). \
             T3 (sm < 80) and unknown SMs are rejected. \
             See docs/plans/sm-coverage.md and docs/support-matrix.md. \
             To restrict targets explicitly: TORCH_CUDA_ARCH_LIST=\"8.0;8.6;8.9;9.0\".",
            spec.sm, source
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
fn parse_arch_list(raw: &str, source: &str) -> Vec<SmSpec> {
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

fn sm_targets_from_nvidia_smi() -> Option<Vec<SmSpec>> {
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

fn detect_sm_targets() -> Vec<SmSpec> {
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
         See docs/plans/sm-coverage.md."
    );
    T1_SMS
        .iter()
        .map(|s| SmSpec {
            sm: (*s).to_string(),
            ptx: false,
        })
        .collect()
}

fn nvcc_arch_args(sm_targets: &[SmSpec]) -> Vec<String> {
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
    {
        if !max_spec.ptx {
            args.push("-gencode".to_string());
            args.push(format!(
                "arch=compute_{sm},code=compute_{sm}",
                sm = max_spec.sm
            ));
        }
    }

    args
}

/// Convert "80" -> "8.0", "120" -> "12.0", for inclusion in TORCH_CUDA_ARCH_LIST hint strings.
fn sm_to_arch_list_token(sm: &str) -> String {
    let len = sm.len();
    if len < 2 {
        return sm.to_string();
    }
    let (head, tail) = sm.split_at(len - 1);
    format!("{head}.{tail}")
}

/// Format a SM-dispatching C wrapper. The generated wrapper:
///   - Caches `compute_capability_major * 10 + minor` per-thread via
///     `__thread` TLS — multi-GPU runtimes (where rank threads bind
///     different devices) get their own value; single-thread/single-GPU
///     hits the cache after first call.
///   - extern-declares each per-SM AOT func with `extern_signature`.
///   - Defines `<public_decl>` whose body is a switch over the SM.
///   - Returns `CUDA_ERROR_NOT_SUPPORTED` for SMs not in the build (T1
///     hard-fail policy makes this branch unreachable for any SM the
///     binary was built to support; the branch exists only as a guard
///     for the T2-opt-in case where the user excluded a SM via
///     `TORCH_CUDA_ARCH_LIST`, and as the failure path when no CUDA
///     context is current).
///
/// **Why __thread, not pthread_once + global static.** The first design
/// (pthread_once) was a foot-gun for multi-GPU code: thread A on GPU 0
/// (sm_80) and thread B on GPU 1 (sm_90) would race on a shared global
/// and the loser would silently dispatch to the wrong cubin. With
/// `__thread` storage, each rank thread caches its own bound device's
/// SM independently — the standard CUDA convention is "one thread, one
/// device" for the lifetime of a kernel-launch chain, which matches
/// thread-local storage semantics exactly. A transient
/// missing-context first call returns NOT_SUPPORTED but does not
/// poison subsequent calls (we re-read `g_sm_pack` each invocation
/// and re-probe when it is `-1`).
fn format_dispatch_wrapper(
    public_decl: &str,
    extern_signature: &str,
    call_args: &str,
    per_sm_funcs: &[(String, String)],
) -> String {
    let externs = per_sm_funcs
        .iter()
        .map(|(_, func)| format!("CUresult {func}({extern_signature});"))
        .collect::<Vec<_>>()
        .join("\n");
    let cases = per_sm_funcs
        .iter()
        .map(|(sm, func)| format!("        case {sm}: return {func}({call_args});"))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "#include <cuda.h>\n\
         #include <stdint.h>\n\
         \n\
         {externs}\n\
         \n\
         static __thread int g_sm_pack = -1;\n\
         \n\
         static int load_sm_pack(void) {{\n\
         \x20   int major = 0, minor = 0;\n\
         \x20   CUdevice dev = 0;\n\
         \x20   if (cuCtxGetDevice(&dev) != CUDA_SUCCESS) return -1;\n\
         \x20   if (cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev) != CUDA_SUCCESS) return -1;\n\
         \x20   if (cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev) != CUDA_SUCCESS) return -1;\n\
         \x20   return major * 10 + minor;\n\
         }}\n\
         \n\
         CUresult {public_decl} {{\n\
         \x20   int sm = g_sm_pack;\n\
         \x20   if (sm < 0) {{\n\
         \x20       sm = load_sm_pack();\n\
         \x20       if (sm < 0) return CUDA_ERROR_NOT_SUPPORTED;\n\
         \x20       g_sm_pack = sm;\n\
         \x20   }}\n\
         \x20   switch (sm) {{\n\
         {cases}\n\
         \x20       default: return CUDA_ERROR_NOT_SUPPORTED;\n\
         \x20   }}\n\
         }}\n"
    )
}

/// One AOT-specialized prefill HD128 kernel per (num_q_heads, num_kv_heads).
/// Mirrors `SUPPORTED_HEADS` in `tools/tilelang/batch_prefill_paged_hd128.py`
/// — when adding a new Qwen3 head config, extend both lists in lockstep
/// AND add the matching FFI extern + dispatch arm in
/// `crates/cuda-kernels/src/ffi/attention.rs` and
/// `infer/src/ops/attention.rs`.
const TILELANG_PREFILL_HD128_HEAD_CONFIGS: &[(u32, u32)] = &[(16, 8), (32, 8), (40, 8), (64, 8)];

/// One AOT-specialized prefill HD256 kernel per (num_q_heads, num_kv_heads).
/// Mirrors `SUPPORTED_HEADS` in `tools/tilelang/batch_prefill_paged_hd256.py`
/// — when adding a new Qwen3.5 full-attn head config, extend both lists in
/// lockstep AND add the matching FFI extern + dispatch arm in
/// `crates/cuda-kernels/src/ffi/attention.rs` and
/// `infer/src/ops/attention.rs`.
const TILELANG_PREFILL_HD256_HEAD_CONFIGS: &[(u32, u32)] = &[(8, 2), (16, 2), (16, 4)];

/// One AOT-specialized decode HD256 kernel per (num_q_heads, num_kv_heads).
/// Mirrors `SUPPORTED_HEADS` in `tools/tilelang/batch_decode_paged_hd256.py`
/// — when adding a new Qwen3.5 full-attn head config, extend both lists in
/// lockstep AND add the matching FFI extern + dispatch arm in
/// `crates/cuda-kernels/src/ffi/attention.rs` and
/// `infer/src/ops/attention.rs`.
const TILELANG_DECODE_HD256_HEAD_CONFIGS: &[(u32, u32)] = &[(8, 2), (16, 2), (16, 4)];

/// One AOT-specialized decode HD128 kernel per (num_q_heads, num_kv_heads).
/// Mirrors `SUPPORTED_HEADS` in `tools/tilelang/batch_decode_paged_hd128.py`
/// — when adding a new Qwen3 full-attn head config, extend both lists in
/// lockstep AND add the matching FFI extern + dispatch arm in
/// `crates/cuda-kernels/src/ffi/attention.rs` and
/// `infer/src/ops/attention.rs`.
const TILELANG_DECODE_HD128_HEAD_CONFIGS: &[(u32, u32)] = &[(16, 8), (32, 8), (40, 8), (64, 8)];

struct TileLangKernelSpec {
    artifact_dir: String,
    kernel_path: &'static str,
    kernel_name: String,
    out_name: String,
    kernel_family: &'static str,
    kernel_key: Option<&'static str>,
    num_q_heads: Option<u32>,
    num_kv_heads: Option<u32>,
    public_decl: &'static str,
    extern_decl: &'static str,
    call_args: &'static str,
}

fn probe_tilelang_python(candidate: &str) -> Result<String, String> {
    let output = Command::new(candidate)
        .args(["-c", "import tilelang"])
        .output()
        .map_err(|err| format!("{candidate}: {err}"))?;

    if output.status.success() {
        Ok(candidate.to_string())
    } else {
        Err(format!(
            "{candidate}: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ))
    }
}

fn find_tilelang_python() -> Result<String, String> {
    if let Ok(candidate) = std::env::var("INFER_TILELANG_PYTHON") {
        let candidate = candidate.trim();
        if candidate.is_empty() {
            return Err(
                "INFER_TILELANG_PYTHON is set but empty. See tools/tilelang/README.md.".to_string(),
            );
        }
        return probe_tilelang_python(candidate).map_err(|message| {
            format!(
                "INFER_TILELANG_PYTHON=`{candidate}` could not import tilelang. {message}. See tools/tilelang/README.md."
            )
        });
    }

    let tool_venv = PathBuf::from("tools/tilelang/.venv/bin/python");
    let local_venv = PathBuf::from(".venv/bin/python");
    let mut diagnostics = Vec::new();
    let mut candidates = Vec::new();
    if tool_venv.exists() {
        candidates.push(tool_venv.to_string_lossy().to_string());
    }
    if local_venv.exists() {
        candidates.push(local_venv.to_string_lossy().to_string());
    }
    candidates.extend(["python3".to_string(), "python".to_string()]);

    for candidate in candidates {
        match probe_tilelang_python(&candidate) {
            Ok(path) => return Ok(path),
            Err(message) => diagnostics.push(message),
        }
    }

    Err(format!(
        "Could not find a Python interpreter with TileLang installed. Set INFER_TILELANG_PYTHON, bootstrap tools/tilelang/.venv, or `pip install -e .[tilelang]`. Probe results: {}.",
        diagnostics.join(" | ")
    ))
}

/// Per-SM TileLang AOT artifact: (sm token, exported func name, generated .c path).
type TileLangPerSmArtifact = (String, String, PathBuf);

/// 18-arg public C signature shared by all three TileLang families
/// (HD128 prefill / HD256 prefill / HD256 decode). Matches the FFI
/// macros in `crates/cuda-kernels/src/ffi/attention.rs`.
const TILELANG_DISPATCH_PUBLIC_DECL: &str = "uint16_t *q, const int32_t *q_indptr, uint16_t *k_pool, uint16_t *v_pool, \
     const int32_t *kv_indptr, const int32_t *kv_indices, const int32_t *kv_last_page_len, \
     uint16_t *o, int32_t batch_size, int32_t total_q_tokens, int32_t max_qlen, \
     int32_t num_pages, int32_t total_pages, int32_t num_q_heads, int32_t num_kv_heads, \
     int32_t page_size, float sm_scale, CUstream stream";
const TILELANG_DISPATCH_EXTERN_DECL: &str = TILELANG_DISPATCH_PUBLIC_DECL;
const TILELANG_DISPATCH_CALL_ARGS: &str = "q, q_indptr, k_pool, v_pool, kv_indptr, kv_indices, kv_last_page_len, o, \
     batch_size, total_q_tokens, max_qlen, num_pages, total_pages, num_q_heads, \
     num_kv_heads, page_size, sm_scale, stream";

const GDR_PREPARE_PUBLIC_DECL: &str = "const uint16_t* qkv, const uint16_t* b_proj, const uint16_t* a_proj, const uint16_t* dt_bias, const float* a_log, uint16_t* q_out, uint16_t* k_out, uint16_t* v_out, float* g_out, float* beta_out, int32_t num_key_heads, int32_t num_value_heads, int32_t qkv_dim, int32_t seq_len, CUstream stream";
const GDR_PREPARE_EXTERN_DECL: &str = GDR_PREPARE_PUBLIC_DECL;
const GDR_PREPARE_CALL_ARGS: &str = "qkv, b_proj, a_proj, dt_bias, a_log, q_out, k_out, v_out, g_out, beta_out, num_key_heads, num_value_heads, qkv_dim, seq_len, stream";

const GDR_CUMSUM_PUBLIC_DECL: &str =
    "const float* g_in, float* g_out, int32_t seq_len, int32_t num_value_heads, CUstream stream";
const GDR_CUMSUM_EXTERN_DECL: &str = GDR_CUMSUM_PUBLIC_DECL;
const GDR_CUMSUM_CALL_ARGS: &str = "g_in, g_out, seq_len, num_value_heads, stream";

const GDR_A_PUBLIC_DECL: &str = "const uint16_t* k, const float* g_cumsum, const float* beta, float* a_tril, int32_t seq_len, int32_t num_value_heads, CUstream stream";
const GDR_A_EXTERN_DECL: &str = GDR_A_PUBLIC_DECL;
const GDR_A_CALL_ARGS: &str = "k, g_cumsum, beta, a_tril, seq_len, num_value_heads, stream";

const GDR_RECOMPUTE_PUBLIC_DECL: &str = "const uint16_t* k, const uint16_t* v, const float* beta, uint16_t* w, uint16_t* u, const uint16_t* a_inv, const float* g_cumsum, int32_t seq_len, int32_t num_value_heads, CUstream stream";
const GDR_RECOMPUTE_EXTERN_DECL: &str = GDR_RECOMPUTE_PUBLIC_DECL;
const GDR_RECOMPUTE_CALL_ARGS: &str =
    "k, v, beta, w, u, a_inv, g_cumsum, seq_len, num_value_heads, stream";

const GDR_STATE_PUBLIC_DECL: &str = "const uint16_t* k, const uint16_t* w, const uint16_t* u, const float* g_cumsum, const float* initial_state, float* chunk_state, uint16_t* v_new, float* final_state, int32_t seq_len, int32_t num_value_heads, CUstream stream";
const GDR_STATE_EXTERN_DECL: &str = GDR_STATE_PUBLIC_DECL;
const GDR_STATE_CALL_ARGS: &str = "k, w, u, g_cumsum, initial_state, chunk_state, v_new, final_state, seq_len, num_value_heads, stream";

const GDR_O_PUBLIC_DECL: &str = "const uint16_t* q, const uint16_t* k, const uint16_t* v_new, const float* chunk_state, const float* g_cumsum, uint16_t* output, int32_t seq_len, int32_t num_value_heads, float scale, CUstream stream";
const GDR_O_EXTERN_DECL: &str = GDR_O_PUBLIC_DECL;
const GDR_O_CALL_ARGS: &str =
    "q, k, v_new, chunk_state, g_cumsum, output, seq_len, num_value_heads, scale, stream";

/// Locate the directories the TileLang AOT generator needs for nvcc to
/// compile `device_kernel.cu`: TileLang's `src/` (for `tl_templates/`),
/// the cutlass headers it bundles, and the active CUDA toolkit include.
fn tilelang_include_dirs(python: &str) -> (PathBuf, PathBuf) {
    let probe = r#"
import importlib.util, json, sys
spec = importlib.util.find_spec("tilelang")
if spec is None or not spec.submodule_search_locations:
    print("ERR_NOT_INSTALLED")
    sys.exit(0)
pkg = spec.submodule_search_locations[0]
print(json.dumps({
    "src": f"{pkg}/src",
    "cutlass_include": f"{pkg}/3rdparty/cutlass/include",
}))
"#;
    let output = Command::new(python)
        .arg("-c")
        .arg(probe)
        .output()
        .expect("failed to probe tilelang install path");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stdout = stdout.trim();
    if stdout == "ERR_NOT_INSTALLED" {
        panic!("tilelang Python package not installed for the chosen interpreter");
    }
    // Tiny hand-rolled parse — JSON has only two known keys, no need to add a dep.
    let src = stdout
        .split("\"src\":")
        .nth(1)
        .and_then(|s| s.split('"').nth(1))
        .expect("tilelang probe: src field missing");
    let cutlass_include = stdout
        .split("\"cutlass_include\":")
        .nth(1)
        .and_then(|s| s.split('"').nth(1))
        .expect("tilelang probe: cutlass_include field missing");
    (PathBuf::from(src), PathBuf::from(cutlass_include))
}

/// Run gen_tilelang_aot.py once per SM target. Each per-SM invocation:
///   - artifact_dir = `<base>_sm{sm}` so cubin/.c paths are unique;
///   - out_name = `<base>_sm{sm}` (drives the .cubin / .c filenames);
///   - kernel_name = `<base_kernel_name>_sm{sm}` so the exported symbol is
///     `<base_kernel_name>_sm{sm}_cuda` (TileLang's gen script appends
///     `_cuda`). The base symbol (`<base_kernel_name>_cuda`) is reserved
///     for the dispatch wrapper that this driver writes.
///
/// Hard-fail on any (kernel, SM) compile failure; suggest a
/// `TORCH_CUDA_ARCH_LIST=...` value that excludes the failing SM.
fn generate_tilelang_artifacts_per_sm(
    python: &str,
    out_dir: &Path,
    sm_targets: &[SmSpec],
    cuda_path: &str,
    tilelang_src: &Path,
    cutlass_include: &Path,
    base_spec: &TileLangKernelSpec,
) -> Vec<TileLangPerSmArtifact> {
    let generator_path = PathBuf::from("tools/tilelang/gen_tilelang_aot.py");
    let mut results = Vec::new();

    for sm in sm_targets {
        let sm_token = &sm.sm;
        let cuda_arch: u32 = sm_token
            .parse()
            .expect("SmSpec.sm passed whitelist; must parse as u32");
        let target = format!("cuda -arch=sm_{sm_token}");

        let per_sm_artifact_dir = format!("{}_sm{sm_token}", base_spec.artifact_dir);
        let per_sm_out_name = format!("{}_sm{sm_token}", base_spec.out_name);
        let per_sm_kernel_name = format!("{}_sm{sm_token}", base_spec.kernel_name);
        let artifact_dir = out_dir.join("tilelang_aot").join(&per_sm_artifact_dir);

        let output = Command::new(python)
            .arg(&generator_path)
            .arg("--kernel-path")
            .arg(base_spec.kernel_path)
            .arg("--kernel-name")
            .arg(&per_sm_kernel_name)
            .arg("--out-name")
            .arg(&per_sm_out_name)
            .arg("--out-dir")
            .arg(&artifact_dir)
            .arg("--target")
            .arg(&target)
            .arg("--kernel-family")
            .arg(base_spec.kernel_family)
            .arg("--cuda-arch")
            .arg(cuda_arch.to_string())
            .arg("--tilelang-src")
            .arg(tilelang_src)
            .arg("--cutlass-include")
            .arg(cutlass_include)
            .arg("--cuda-include")
            .arg(format!("{cuda_path}/include"))
            .args(
                base_spec
                    .kernel_key
                    .into_iter()
                    .flat_map(|key| ["--kernel-key".to_string(), key.to_string()]),
            )
            .args(
                base_spec
                    .num_q_heads
                    .into_iter()
                    .flat_map(|heads| ["--num-q-heads".to_string(), heads.to_string()]),
            )
            .args(
                base_spec
                    .num_kv_heads
                    .into_iter()
                    .flat_map(|heads| ["--num-kv-heads".to_string(), heads.to_string()]),
            )
            .output()
            .unwrap_or_else(|err| {
                panic!(
                    "failed to spawn TileLang AOT generator for {} on sm_{sm_token}: {err}",
                    base_spec.kernel_name
                )
            });

        if !output.status.success() {
            let other_sms: Vec<String> = sm_targets
                .iter()
                .filter(|s| s.sm != *sm_token)
                .map(|s| sm_to_arch_list_token(&s.sm))
                .collect();
            let suggestion = if other_sms.is_empty() {
                "all targets failed; bump tilelang in pyproject.toml or pin a working version"
                    .to_string()
            } else {
                format!("TORCH_CUDA_ARCH_LIST=\"{}\"", other_sms.join(";"))
            };
            panic!(
                "TileLang AOT failed to compile {} for sm_{sm_token}.\n\
                 stdout: {}\n\
                 stderr: {}\n\n\
                 Hint: bump tilelang (pin lives in pyproject.toml) OR exclude sm_{sm_token} via:\n  \
                 {suggestion}\n\
                 See docs/plans/sm-coverage.md.",
                base_spec.kernel_name,
                String::from_utf8_lossy(&output.stdout).trim(),
                String::from_utf8_lossy(&output.stderr).trim(),
            );
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut func_name = None;
        let mut c_path = None;
        for line in stdout.lines() {
            if let Some(value) = line.strip_prefix("FUNC_NAME=") {
                func_name = Some(value.trim().to_string());
            } else if let Some(value) = line.strip_prefix("C_PATH=") {
                c_path = Some(PathBuf::from(value.trim()));
            }
        }

        results.push((
            sm_token.clone(),
            func_name.expect("TileLang generator did not print FUNC_NAME"),
            c_path.expect("TileLang generator did not print C_PATH"),
        ));
    }

    results
}

/// Build one TileLang head-config kernel for every SM target: generate
/// per-SM artifacts, write a single dispatch wrapper exposing
/// `<base_kernel_name>_cuda`, and append all sources to
/// `generated_sources` for cc::Build to compile.
fn build_tilelang_kernel(
    python: &str,
    out_dir: &Path,
    sm_targets: &[SmSpec],
    cuda_path: &str,
    tilelang_src: &Path,
    cutlass_include: &Path,
    base_spec: &TileLangKernelSpec,
    generated_sources: &mut Vec<PathBuf>,
) {
    let per_sm = generate_tilelang_artifacts_per_sm(
        python,
        out_dir,
        sm_targets,
        cuda_path,
        tilelang_src,
        cutlass_include,
        base_spec,
    );
    let pairs: Vec<(String, String)> = per_sm
        .iter()
        .map(|(sm, func, _)| (sm.clone(), func.clone()))
        .collect();

    let public_name = format!("{}_cuda", base_spec.kernel_name);
    let public_decl = format!("{public_name}({})", base_spec.public_decl);
    let wrapper_src = format_dispatch_wrapper(
        &public_decl,
        base_spec.extern_decl,
        base_spec.call_args,
        &pairs,
    );

    let dispatch_dir = out_dir
        .join("tilelang_aot")
        .join(format!("{}_dispatch", base_spec.artifact_dir));
    std::fs::create_dir_all(&dispatch_dir).expect("create TileLang dispatch directory");
    let wrapper_path = dispatch_dir.join(format!("{}_dispatch.c", base_spec.out_name));
    std::fs::write(&wrapper_path, wrapper_src).expect("write TileLang dispatch wrapper");

    for (_, _, c) in per_sm {
        generated_sources.push(c);
    }
    generated_sources.push(wrapper_path);
}

fn compile_tilelang_aot_kernels(cuda_path: &str, out_dir: &Path, sm_targets: &[SmSpec]) {
    let python = find_tilelang_python().unwrap_or_else(|message| panic!("{message}"));
    let (tilelang_src, cutlass_include) = tilelang_include_dirs(&python);
    let mut generated_sources = Vec::new();

    for &(q, kv) in TILELANG_PREFILL_HD128_HEAD_CONFIGS {
        let suffix = format!("q{q}_kv{kv}");
        let spec = TileLangKernelSpec {
            artifact_dir: format!("batch_prefill_paged_hd128_{suffix}"),
            kernel_path: "tools/tilelang/batch_prefill_paged_hd128.py",
            kernel_name: format!("tilelang_batch_prefill_paged_hd128_{suffix}_run"),
            out_name: format!("tilelang_batch_prefill_paged_hd128_{suffix}"),
            kernel_family: "attention",
            kernel_key: None,
            num_q_heads: Some(q),
            num_kv_heads: Some(kv),
            public_decl: TILELANG_DISPATCH_PUBLIC_DECL,
            extern_decl: TILELANG_DISPATCH_EXTERN_DECL,
            call_args: TILELANG_DISPATCH_CALL_ARGS,
        };
        build_tilelang_kernel(
            &python,
            out_dir,
            sm_targets,
            cuda_path,
            &tilelang_src,
            &cutlass_include,
            &spec,
            &mut generated_sources,
        );
    }

    for &(q, kv) in TILELANG_PREFILL_HD256_HEAD_CONFIGS {
        let suffix = format!("q{q}_kv{kv}");
        let spec = TileLangKernelSpec {
            artifact_dir: format!("batch_prefill_paged_hd256_{suffix}"),
            kernel_path: "tools/tilelang/batch_prefill_paged_hd256.py",
            kernel_name: format!("tilelang_batch_prefill_paged_hd256_{suffix}_run"),
            out_name: format!("tilelang_batch_prefill_paged_hd256_{suffix}"),
            kernel_family: "attention",
            kernel_key: None,
            num_q_heads: Some(q),
            num_kv_heads: Some(kv),
            public_decl: TILELANG_DISPATCH_PUBLIC_DECL,
            extern_decl: TILELANG_DISPATCH_EXTERN_DECL,
            call_args: TILELANG_DISPATCH_CALL_ARGS,
        };
        build_tilelang_kernel(
            &python,
            out_dir,
            sm_targets,
            cuda_path,
            &tilelang_src,
            &cutlass_include,
            &spec,
            &mut generated_sources,
        );
    }

    for &(q, kv) in TILELANG_DECODE_HD256_HEAD_CONFIGS {
        let suffix = format!("q{q}_kv{kv}");
        let spec = TileLangKernelSpec {
            artifact_dir: format!("batch_decode_paged_hd256_{suffix}"),
            kernel_path: "tools/tilelang/batch_decode_paged_hd256.py",
            kernel_name: format!("tilelang_batch_decode_paged_hd256_{suffix}_run"),
            out_name: format!("tilelang_batch_decode_paged_hd256_{suffix}"),
            kernel_family: "attention",
            kernel_key: None,
            num_q_heads: Some(q),
            num_kv_heads: Some(kv),
            public_decl: TILELANG_DISPATCH_PUBLIC_DECL,
            extern_decl: TILELANG_DISPATCH_EXTERN_DECL,
            call_args: TILELANG_DISPATCH_CALL_ARGS,
        };
        build_tilelang_kernel(
            &python,
            out_dir,
            sm_targets,
            cuda_path,
            &tilelang_src,
            &cutlass_include,
            &spec,
            &mut generated_sources,
        );
    }

    for &(q, kv) in TILELANG_DECODE_HD128_HEAD_CONFIGS {
        let suffix = format!("q{q}_kv{kv}");
        let spec = TileLangKernelSpec {
            artifact_dir: format!("batch_decode_paged_hd128_{suffix}"),
            kernel_path: "tools/tilelang/batch_decode_paged_hd128.py",
            kernel_name: format!("tilelang_batch_decode_paged_hd128_{suffix}_run"),
            out_name: format!("tilelang_batch_decode_paged_hd128_{suffix}"),
            kernel_family: "attention",
            kernel_key: None,
            num_q_heads: Some(q),
            num_kv_heads: Some(kv),
            public_decl: TILELANG_DISPATCH_PUBLIC_DECL,
            extern_decl: TILELANG_DISPATCH_EXTERN_DECL,
            call_args: TILELANG_DISPATCH_CALL_ARGS,
        };
        build_tilelang_kernel(
            &python,
            out_dir,
            sm_targets,
            cuda_path,
            &tilelang_src,
            &cutlass_include,
            &spec,
            &mut generated_sources,
        );
    }

    let gdr_specs = [
        TileLangKernelSpec {
            artifact_dir: "gated_delta_rule_chunk_prepare".to_string(),
            kernel_path: "tools/tilelang/gated_delta_rule.py",
            kernel_name: "gated_delta_rule_prefill_chunk_prepare".to_string(),
            out_name: "tilelang_gated_delta_rule_chunk_prepare".to_string(),
            kernel_family: "gdr",
            kernel_key: Some("gdr_chunk_prepare"),
            num_q_heads: None,
            num_kv_heads: None,
            public_decl: GDR_PREPARE_PUBLIC_DECL,
            extern_decl: GDR_PREPARE_EXTERN_DECL,
            call_args: GDR_PREPARE_CALL_ARGS,
        },
        TileLangKernelSpec {
            artifact_dir: "gated_delta_rule_chunk_cumsum".to_string(),
            kernel_path: "tools/tilelang/gated_delta_rule.py",
            kernel_name: "gated_delta_rule_prefill_chunk_cumsum".to_string(),
            out_name: "tilelang_gated_delta_rule_chunk_cumsum".to_string(),
            kernel_family: "gdr",
            kernel_key: Some("gdr_chunk_cumsum"),
            num_q_heads: None,
            num_kv_heads: None,
            public_decl: GDR_CUMSUM_PUBLIC_DECL,
            extern_decl: GDR_CUMSUM_EXTERN_DECL,
            call_args: GDR_CUMSUM_CALL_ARGS,
        },
        TileLangKernelSpec {
            artifact_dir: "gated_delta_rule_chunk_a".to_string(),
            kernel_path: "tools/tilelang/gated_delta_rule.py",
            kernel_name: "gated_delta_rule_prefill_chunk_a".to_string(),
            out_name: "tilelang_gated_delta_rule_chunk_a".to_string(),
            kernel_family: "gdr",
            kernel_key: Some("gdr_chunk_a"),
            num_q_heads: None,
            num_kv_heads: None,
            public_decl: GDR_A_PUBLIC_DECL,
            extern_decl: GDR_A_EXTERN_DECL,
            call_args: GDR_A_CALL_ARGS,
        },
        TileLangKernelSpec {
            artifact_dir: "gated_delta_rule_chunk_recompute".to_string(),
            kernel_path: "tools/tilelang/gated_delta_rule.py",
            kernel_name: "gated_delta_rule_prefill_chunk_recompute".to_string(),
            out_name: "tilelang_gated_delta_rule_chunk_recompute".to_string(),
            kernel_family: "gdr",
            kernel_key: Some("gdr_chunk_recompute"),
            num_q_heads: None,
            num_kv_heads: None,
            public_decl: GDR_RECOMPUTE_PUBLIC_DECL,
            extern_decl: GDR_RECOMPUTE_EXTERN_DECL,
            call_args: GDR_RECOMPUTE_CALL_ARGS,
        },
        TileLangKernelSpec {
            artifact_dir: "gated_delta_rule_chunk_state".to_string(),
            kernel_path: "tools/tilelang/gated_delta_rule.py",
            kernel_name: "gated_delta_rule_prefill_chunk_state".to_string(),
            out_name: "tilelang_gated_delta_rule_chunk_state".to_string(),
            kernel_family: "gdr",
            kernel_key: Some("gdr_chunk_state"),
            num_q_heads: None,
            num_kv_heads: None,
            public_decl: GDR_STATE_PUBLIC_DECL,
            extern_decl: GDR_STATE_EXTERN_DECL,
            call_args: GDR_STATE_CALL_ARGS,
        },
        TileLangKernelSpec {
            artifact_dir: "gated_delta_rule_chunk_o".to_string(),
            kernel_path: "tools/tilelang/gated_delta_rule.py",
            kernel_name: "gated_delta_rule_prefill_chunk_o".to_string(),
            out_name: "tilelang_gated_delta_rule_chunk_o".to_string(),
            kernel_family: "gdr",
            kernel_key: Some("gdr_chunk_o"),
            num_q_heads: None,
            num_kv_heads: None,
            public_decl: GDR_O_PUBLIC_DECL,
            extern_decl: GDR_O_EXTERN_DECL,
            call_args: GDR_O_CALL_ARGS,
        },
    ];
    for spec in &gdr_specs {
        build_tilelang_kernel(
            &python,
            out_dir,
            sm_targets,
            cuda_path,
            &tilelang_src,
            &cutlass_include,
            spec,
            &mut generated_sources,
        );
    }

    let mut build = cc::Build::new();
    build
        .cuda(false)
        .include(format!("{}/include", cuda_path))
        .flag("-std=c11")
        .warnings(false);
    for source in &generated_sources {
        build.file(source);
    }
    build.compile("tilelang_kernels_aot");

    println!("cargo:rustc-link-lib=cuda");
    println!(
        "cargo:warning=TileLang AOT: built per-SM cubins for {} target(s) across HD128/HD256 prefill, HD128/HD256 decode, and Qwen3.5 GDR; SM dispatch via __thread cache + cuDeviceGetAttribute. See docs/plans/sm-coverage.md.",
        sm_targets.len()
    );
    for entry in std::fs::read_dir("tools/tilelang")
        .expect("tools/tilelang directory must exist")
        .flatten()
    {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("py") {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
    println!("cargo:rerun-if-changed=tools/tilelang");
    println!("cargo:rerun-if-env-changed=INFER_TILELANG_PYTHON");
}

// Recursively collect every `.cu` file under `dir` so domain subdirs
// (attention/, gemm/, kv/, quant/, misc/) are picked up automatically.
fn collect_cu_files(dir: &Path, out: &mut Vec<PathBuf>) {
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
        if path.extension().and_then(|e| e.to_str()) == Some("cu") {
            out.push(path);
        }
    }
}

fn main() {
    if std::env::var("CARGO_FEATURE_METAL").is_ok() {
        println!("cargo:warning=metal feature active: relying on mlx-sys bridge only.");
    }

    if std::env::var("CARGO_FEATURE_CUDA").is_err() {
        println!("cargo:warning=cuda feature inactive: skipping CUDA/TileLang kernel compilation.");
        println!("cargo:rerun-if-env-changed=CARGO_FEATURE_CUDA");
        return;
    }

    // When the `no-cuda` feature is active (e.g. macOS dev machines without a GPU),
    // skip all CUDA/TileLang compilation. GPU ops will panic at runtime.
    if std::env::var("CARGO_FEATURE_NO_CUDA").is_ok() {
        println!(
            "cargo:warning=no-cuda feature active: skipping CUDA/TileLang kernel compilation."
        );
        println!("cargo:rerun-if-env-changed=CARGO_FEATURE_NO_CUDA");
        return;
    }

    let cuda_path = std::env::var("CUDA_HOME")
        .or_else(|_| std::env::var("CUDA_PATH"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());

    let nvcc = format!("{}/bin/nvcc", cuda_path);
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let sm_targets = detect_sm_targets();
    let arch_args = nvcc_arch_args(&sm_targets);
    println!(
        "cargo:warning=Compiling CUDA kernels for targets: {}",
        sm_targets
            .iter()
            .map(|s| if s.ptx {
                format!("sm_{}+PTX", s.sm)
            } else {
                format!("sm_{}", s.sm)
            })
            .collect::<Vec<_>>()
            .join(",")
    );

    let csrc_dir = Path::new("csrc");
    let mut cu_files: Vec<PathBuf> = Vec::new();
    collect_cu_files(csrc_dir, &mut cu_files);
    // Keep a stable compile order independent of filesystem iteration order.
    cu_files.sort();

    println!("cargo:rerun-if-env-changed=NVCC_CCBIN");
    let ccbin = std::env::var("NVCC_CCBIN").ok();

    let mut obj_files = Vec::new();
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
        nvcc_args.extend(arch_args.clone());
        nvcc_args.extend(["--compiler-options".to_string(), "-fPIC".to_string()]);
        // Ensure `#include "common.cuh"` resolves from any domain subdir
        // (attention/, gemm/, kv/, quant/, misc/).
        nvcc_args.push("-Icsrc".to_string());

        // Marlin kernel needs C++17 + relaxed constexpr
        if stem.starts_with("marlin_") {
            nvcc_args.extend([
                "-std=c++17".to_string(),
                "--expt-relaxed-constexpr".to_string(),
            ]);
        }

        let status = Command::new(&nvcc)
            .args(&nvcc_args)
            .status()
            .unwrap_or_else(|_| panic!("Failed to run nvcc for {}", cu_file.display()));

        assert!(
            status.success(),
            "nvcc compilation failed for {}",
            cu_file.display()
        );

        obj_files.push(obj_file);
    }

    let cuda_lib = out_dir.join("libkernels_cuda.a");
    let mut ar_args = vec!["rcs".to_string(), cuda_lib.to_string_lossy().to_string()];
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

    compile_tilelang_aot_kernels(&cuda_path, &out_dir, &sm_targets);

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    if cfg!(target_os = "windows") {
        println!("cargo:rustc-link-search=native={}/lib/x64", cuda_path);
    } else {
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    }
    println!("cargo:rustc-link-lib=static=kernels_cuda");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cublas");
    println!("cargo:rustc-link-lib=cublasLt");
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=c++");
    } else if !cfg!(target_os = "windows") {
        println!("cargo:rustc-link-lib=stdc++");
    }

    println!("cargo:rerun-if-changed=csrc/");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=TORCH_CUDA_ARCH_LIST");
    println!("cargo:rerun-if-env-changed=CMAKE_CUDA_ARCHITECTURES");
}
