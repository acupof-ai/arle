use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Tier-1 SMs: default-compiled fat-binary set. A100 / A10·3090 / L4·4090 / H100.
const T1_SMS: &[&str] = &["80", "86", "89", "90"];

/// Tier-2 SMs: opt-in via TORCH_CUDA_ARCH_LIST. B100·B200 / RTX 5090.
const T2_SMS: &[&str] = &["100", "120"];

/// Legacy Volta SMs: opt-in / auto-detect only, built as a separate SM-pinned binary.
const LEGACY_VOLTA_SMS: &[&str] = &["70"];

fn is_supported_sm(sm: &str) -> bool {
    T1_SMS.contains(&sm) || T2_SMS.contains(&sm) || LEGACY_VOLTA_SMS.contains(&sm)
}

fn is_legacy_volta_sm(sm: &str) -> bool {
    LEGACY_VOLTA_SMS.contains(&sm)
}

fn has_legacy_volta(sm_targets: &[SmSpec]) -> bool {
    sm_targets.iter().any(|spec| is_legacy_volta_sm(&spec.sm))
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

/// Reject SMs outside the explicit whitelist. Turing/Pascal/older stay unsupported.
fn validate_sm(spec: &SmSpec, source: &str) {
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

fn validate_sm_set(sm_targets: &[SmSpec]) {
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

// The TileLang AOT kernel matrix (head configs, ABIs, allow_sm70, gates) is the
// single source of truth in `crates/cuda-kernels/kernels.toml`, parsed at build
// time by `load_registry()` (see the registry model + emitters below). The old
// hand-duplicated `TILELANG_*_HEAD_CONFIGS` consts + the `*_PUBLIC_DECL` /
// `*_EXTERN_DECL` / `*_CALL_ARGS` C-signature consts were deleted in favor of
// the registry rows + `[abi.*]` blocks.

#[derive(Clone)]
struct TileLangKernelSpec {
    artifact_dir: String,
    kernel_path: String,
    kernel_name: String,
    out_name: String,
    kernel_family: String,
    kernel_key: Option<String>,
    num_q_heads: Option<u32>,
    num_kv_heads: Option<u32>,
    public_decl: String,
    extern_decl: String,
    call_args: String,
    allow_sm70: bool,
}

// ---- registry data model (kernels.toml) -----------------------------------

#[derive(Clone)]
struct AbiSig {
    fn_ty: String,                    // resolve() fn-ptr alias name
    c_decl: String,                   // C param list for the dispatch wrapper
    call_args: String,                // C call args
    rust_args: Vec<(String, String)>, // (name, rust_ty); empty => build-only
}

#[derive(Clone)]
struct RegKernel {
    head_dim: Option<u32>,
    q_heads: Option<u32>,
    kv_heads: Option<u32>,
    phase: String,
    abi: String,
    kernel_family: String, // python --kernel-family tag
    kernel_key: Option<String>,
    py_module: String, // --kernel-path (manifest-relative)
    artifact_dir: String,
    kernel_name: String,
    out_name: String,
    allow_sm70: bool,
    gate: String, // "default" | "flashqla"
    ffi: bool,
}

struct Registry {
    abis: std::collections::BTreeMap<String, AbiSig>,
    kernels: Vec<RegKernel>,
}

/// Parse `crates/cuda-kernels/kernels.toml` (manifest-relative). Re-runs the
/// build on edit via `rerun-if-changed`.
fn load_registry() -> Registry {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let path = manifest.join("kernels.toml");
    println!("cargo:rerun-if-changed={}", path.display());
    // The tilelang pin folds into the kernel cache key (tilelang_kernel_src_hash);
    // re-run the build when it changes so a tilelang bump busts the cache.
    println!(
        "cargo:rerun-if-changed={}",
        manifest.join("../../requirements-build.txt").display()
    );
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read kernels.toml ({}): {e}", path.display()));
    parse_registry(&text)
}

/// Parse kernels.toml into the registry model using the `toml` crate. Walks
/// `doc["abi"]` (table of ABI signatures) + `doc["kernel"]` (array of tables).
fn parse_registry(text: &str) -> Registry {
    let doc: toml::Value =
        toml::from_str(text).unwrap_or_else(|e| panic!("parse kernels.toml: {e}"));

    let mut abis = std::collections::BTreeMap::new();
    if let Some(abi_tbl) = doc.get("abi").and_then(|v| v.as_table()) {
        for (name, body) in abi_tbl {
            let body = body
                .as_table()
                .unwrap_or_else(|| panic!("abi.{name} must be a table"));
            let fn_ty = body
                .get("fn_ty")
                .and_then(|v| v.as_str())
                .unwrap_or_else(|| panic!("abi.{name} missing fn_ty"))
                .to_string();
            let c_decl = normalize_ws(
                body.get("c")
                    .and_then(|v| v.as_str())
                    .unwrap_or_else(|| panic!("abi.{name} missing c")),
            );
            let call_args = normalize_ws(
                body.get("call")
                    .and_then(|v| v.as_str())
                    .unwrap_or_else(|| panic!("abi.{name} missing call")),
            );
            let rust_args: Vec<(String, String)> = body
                .get("rust")
                .and_then(|v| v.as_array())
                .map(|rows| {
                    rows.iter()
                        .map(|row| {
                            let pair = row.as_array().unwrap_or_else(|| {
                                panic!("abi.{name}.rust rows must be [name, ty] arrays")
                            });
                            let n = pair[0]
                                .as_str()
                                .unwrap_or_else(|| {
                                    panic!("abi.{name}.rust arg name must be a string")
                                })
                                .to_string();
                            let t = pair[1]
                                .as_str()
                                .unwrap_or_else(|| {
                                    panic!("abi.{name}.rust arg ty must be a string")
                                })
                                .to_string();
                            (n, t)
                        })
                        .collect()
                })
                .unwrap_or_default();
            abis.insert(
                name.clone(),
                AbiSig {
                    fn_ty,
                    c_decl,
                    call_args,
                    rust_args,
                },
            );
        }
    }

    let mut kernels = Vec::new();
    if let Some(rows) = doc.get("kernel").and_then(|v| v.as_array()) {
        for row in rows {
            let t = row.as_table().expect("[[kernel]] must be a table");
            let getstr = |key: &str| t.get(key).and_then(|v| v.as_str()).map(|s| s.to_string());
            let getu32 = |key: &str| t.get(key).and_then(|v| v.as_integer()).map(|n| n as u32);
            let kernel_name = getstr("kernel_name").expect("[[kernel]] missing kernel_name");
            kernels.push(RegKernel {
                head_dim: getu32("head_dim"),
                q_heads: getu32("q_heads"),
                kv_heads: getu32("kv_heads"),
                phase: getstr("phase").unwrap_or_default(),
                abi: getstr("abi").unwrap_or_else(|| panic!("kernel {kernel_name} missing abi")),
                kernel_family: getstr("kernel_family")
                    .unwrap_or_else(|| panic!("kernel {kernel_name} missing kernel_family")),
                kernel_key: getstr("kernel_key"),
                py_module: getstr("py_module")
                    .unwrap_or_else(|| panic!("kernel {kernel_name} missing py_module")),
                artifact_dir: getstr("artifact_dir")
                    .unwrap_or_else(|| panic!("kernel {kernel_name} missing artifact_dir")),
                out_name: getstr("out_name")
                    .unwrap_or_else(|| panic!("kernel {kernel_name} missing out_name")),
                kernel_name,
                allow_sm70: t
                    .get("allow_sm70")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false),
                gate: getstr("gate").unwrap_or_else(|| "default".to_string()),
                ffi: t.get("ffi").and_then(|v| v.as_bool()).unwrap_or(false),
            });
        }
    }

    Registry { abis, kernels }
}

/// Collapse runs of whitespace (incl. the `\`-continuation newlines TOML keeps
/// in multi-line basic strings) into single spaces, matching the old
/// `&str` consts after rustfmt joined their `\`-wrapped literals.
fn normalize_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Build the `TileLangKernelSpec` list from every registry row. The flashqla
/// gate (sm90 + env flag) is applied per-row by the caller; SM-tier stubbing
/// (e.g. sm_70 legacy) happens inside `build_tilelang_kernel`.
fn registry_to_specs(reg: &Registry) -> Vec<(TileLangKernelSpec, &RegKernel)> {
    reg.kernels
        .iter()
        .map(|k| {
            let abi = reg.abis.get(&k.abi).unwrap_or_else(|| {
                panic!("kernel {} references unknown abi {}", k.kernel_name, k.abi)
            });
            let spec = TileLangKernelSpec {
                artifact_dir: k.artifact_dir.clone(),
                kernel_path: k.py_module.clone(),
                kernel_name: k.kernel_name.clone(),
                out_name: k.out_name.clone(),
                kernel_family: k.kernel_family.clone(),
                kernel_key: k.kernel_key.clone(),
                num_q_heads: k.q_heads,
                num_kv_heads: k.kv_heads,
                public_decl: abi.c_decl.clone(),
                extern_decl: abi.c_decl.clone(),
                call_args: abi.call_args.clone(),
                allow_sm70: k.allow_sm70,
            };
            (spec, k)
        })
        .collect()
}

// ---- FFI emitter: registry -> OUT_DIR/ffi_tilelang_generated.rs ------------

/// Emit the extern blocks (one per ffi=true ABI) + fn-ptr aliases + the
/// `AttnPhase` enum + `resolve_*()`. The head-config matrix lives entirely in
/// kernels.toml; this is pure Rust codegen and runs even on a no-cuda Mac check.
fn emit_ffi_generated(reg: &Registry, out_dir: &Path) {
    use std::fmt::Write as _;
    let mut s = String::new();
    s.push_str("// @generated by build.rs from kernels.toml — DO NOT EDIT.\n\n");

    // (A) extern blocks, grouped by ABI, only for ffi=true rows.
    for (abi_name, abi) in &reg.abis {
        if abi.rust_args.is_empty() {
            continue; // build-only ABI -> no extern
        }
        let syms: Vec<&RegKernel> = reg
            .kernels
            .iter()
            .filter(|k| k.ffi && &k.abi == abi_name)
            .collect();
        if syms.is_empty() {
            continue;
        }
        writeln!(s, "// abi {abi_name} — {} symbol(s)", syms.len()).unwrap();
        s.push_str("unsafe extern \"C\" {\n");
        for k in &syms {
            s.push_str("    #[allow(dead_code)]\n");
            writeln!(s, "    pub fn {}_cuda(", k.kernel_name).unwrap();
            for (name, ty) in &abi.rust_args {
                writeln!(s, "        {name}: {ty},").unwrap();
            }
            s.push_str("    ) -> CUresult;\n");
        }
        s.push_str("}\n\n");
    }

    // (B) fn-ptr type aliases (one per ffi=true ABI).
    for (abi_name, abi) in &reg.abis {
        if abi.rust_args.is_empty() {
            continue;
        }
        if !reg.kernels.iter().any(|k| k.ffi && &k.abi == abi_name) {
            continue;
        }
        let tys: Vec<&str> = abi.rust_args.iter().map(|(_, t)| t.as_str()).collect();
        writeln!(s, "#[allow(dead_code)]").unwrap();
        writeln!(
            s,
            "pub type {} = unsafe extern \"C\" fn(\n    {},\n) -> CUresult;\n",
            abi.fn_ty,
            tys.join(", ")
        )
        .unwrap();
    }

    // AttnPhase enum (the distinct ffi=true attention phases).
    s.push_str(
        "#[allow(dead_code)]\n#[derive(Clone, Copy, PartialEq, Eq, Debug)]\n\
         pub enum AttnPhase { Prefill, Decode }\n\n",
    );

    // (C) resolve_* for the ffi=true paged_attn_v1 ABI, keyed on (hd,q,kv,phase).
    emit_resolve_paged_attn_v1(&mut s, reg);

    let out = out_dir.join("ffi_tilelang_generated.rs");
    std::fs::write(&out, s).expect("write ffi_tilelang_generated.rs");
}

// ---- flashqla emitter: registry -> OUT_DIR/flashqla_gdr_generated.rs -------

/// The five AOT phases of one flashqla geometry: kernels.toml `phase`, the
/// generated struct field, and the hand-written fn-ptr alias in
/// `src/ffi/recurrent.rs`.
const FLASHQLA_PHASES: [(&str, &str, &str); 5] = [
    ("fq_cumsum", "cumsum", "FqCumsumFn"),
    ("fq_kkt", "kkt", "FqKktFn"),
    ("fq_fwd", "fwd", "FqFwdFn"),
    ("fq_prepare_h", "prepare_h", "FqPrepareHFn"),
    ("fq_bwd", "bwd", "FqBwdFn"),
];

/// Emit the (H, Hg) -> symbol-set table that autograd's GDN dispatch reads, so
/// the built-geometry set has exactly one source (kernels.toml). Included by
/// `src/ffi/recurrent.rs`: a row whose extern is missing fails to compile.
fn emit_flashqla_gdr_generated(reg: &Registry, out_dir: &Path) {
    use std::fmt::Write as _;
    let mut geoms: Vec<(u32, u32)> = Vec::new();
    for k in reg.kernels.iter().filter(|k| k.kernel_family == "flashqla") {
        let g = (
            k.q_heads
                .unwrap_or_else(|| panic!("flashqla row {} missing q_heads", k.kernel_name)),
            k.kv_heads
                .unwrap_or_else(|| panic!("flashqla row {} missing kv_heads", k.kernel_name)),
        );
        if !geoms.contains(&g) {
            geoms.push(g);
        }
    }

    let mut s = String::from("// @generated by build.rs from kernels.toml — DO NOT EDIT.\n\n");
    s.push_str("#[allow(dead_code)]\n#[derive(Clone, Copy)]\npub struct FlashqlaGdrSyms {\n");
    s.push_str("    pub q_heads: u32,\n    pub kv_heads: u32,\n");
    for (_, field, fn_ty) in FLASHQLA_PHASES {
        writeln!(s, "    pub {field}: {fn_ty},").unwrap();
    }
    s.push_str(
        "}\n\n#[allow(dead_code)]\npub static FLASHQLA_GDR_TABLE: &[FlashqlaGdrSyms] = &[\n",
    );
    for (q, kv) in geoms {
        writeln!(s, "    FlashqlaGdrSyms {{ q_heads: {q}, kv_heads: {kv},").unwrap();
        for (phase, field, _) in FLASHQLA_PHASES {
            let row = reg
                .kernels
                .iter()
                .find(|k| {
                    k.kernel_family == "flashqla"
                        && k.phase == phase
                        && k.q_heads == Some(q)
                        && k.kv_heads == Some(kv)
                })
                .unwrap_or_else(|| {
                    panic!("flashqla geometry {q}/{kv} has no {phase} row in kernels.toml")
                });
            writeln!(s, "        {field}: {}_cuda,", row.kernel_name).unwrap();
        }
        s.push_str("    },\n");
    }
    s.push_str("];\n");

    let out = out_dir.join("flashqla_gdr_generated.rs");
    std::fs::write(&out, s).expect("write flashqla_gdr_generated.rs");
}

fn phase_variant(phase: &str) -> &str {
    match phase {
        "prefill" => "AttnPhase::Prefill",
        "decode" => "AttnPhase::Decode",
        other => panic!("paged_attn_v1 row has unexpected phase {other:?}"),
    }
}

fn emit_resolve_paged_attn_v1(s: &mut String, reg: &Registry) {
    use std::fmt::Write as _;
    s.push_str(
        "#[allow(dead_code)]\n\
         pub fn resolve_paged_attn_v1(head_dim: u32, q_heads: u32, kv_heads: u32, \
         phase: AttnPhase) -> Option<PagedAttnV1Fn> {\n\
         \x20   Some(match (head_dim, q_heads, kv_heads, phase) {\n",
    );
    for k in reg
        .kernels
        .iter()
        .filter(|k| k.ffi && k.abi == "paged_attn_v1")
    {
        let (hd, q, kv) = (k.head_dim.unwrap(), k.q_heads.unwrap(), k.kv_heads.unwrap());
        writeln!(
            s,
            "        ({hd}, {q}, {kv}, {}) => {}_cuda,",
            phase_variant(&k.phase),
            k.kernel_name
        )
        .unwrap();
    }
    s.push_str("        _ => return None,\n    })\n}\n\n");
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
    let candidates: Vec<String> = [
        tool_venv
            .exists()
            .then(|| tool_venv.to_string_lossy().to_string()),
        local_venv
            .exists()
            .then(|| local_venv.to_string_lossy().to_string()),
    ]
    .into_iter()
    .flatten()
    .chain(["python3".to_string(), "python".to_string()])
    .collect();

    for candidate in candidates {
        match probe_tilelang_python(&candidate) {
            Ok(path) => return Ok(path),
            Err(message) => diagnostics.push(message),
        }
    }

    Err(format!(
        "Could not find a Python interpreter with TileLang installed. Either install TileLang \
(set INFER_TILELANG_PYTHON, bootstrap tools/tilelang/.venv, or `pip install -e .[tilelang]`), \
or avoid regeneration entirely: run `scripts/kernel_artifacts.sh fetch` to restore the vendored \
generated/ tier, or set ARLE_CUDA_KERNELS_PREBUILT_DIR to a prebuilt kernel bundle. \
Probe results: {}.",
        diagnostics.join(" | ")
    ))
}

/// Lazy TileLang toolchain resolver. Resolves the Python interpreter and the
/// TileLang/cutlass include dirs ONCE, on first actual (re)generation, and
/// caches the result. `generated/` is gitignored now (no committed `.c`), so a
/// from-source build regenerates and DOES need `tilelang` installed — `get()` is
/// only skipped when every per-(kernel, SM) artifact is restored from the
/// persistent kernel cache or an externally-provided `generated/`/prebuilt dir.
struct TilelangToolchain {
    python: PathBuf,
    package: PathBuf,
    tvm_runtime: PathBuf,
    tvm_ffi: PathBuf,
    src: PathBuf,
    cutlass_include: PathBuf,
}

struct LazyTilelang {
    cell: std::cell::OnceCell<TilelangToolchain>,
}

impl LazyTilelang {
    fn new() -> Self {
        Self {
            cell: std::cell::OnceCell::new(),
        }
    }

    /// Resolve python + include dirs lazily — called ONLY when a kernel must be
    /// regenerated. A fully-vendored build never calls this, so it never needs
    /// tilelang.
    fn get(&self) -> &TilelangToolchain {
        self.cell.get_or_init(|| {
            let python = find_tilelang_python().unwrap_or_else(|m| {
                panic!(
                    "TileLang is required to (re)generate a missing or source-changed AOT kernel, but it is unavailable: {m}"
                )
            });
            tilelang_toolchain(&python)
        })
    }
}

fn sha256_bytes(bytes: &[u8]) -> String {
    for (program, args) in [("sha256sum", &[][..]), ("shasum", &["-a", "256"][..])] {
        let Ok(mut child) = Command::new(program)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
        else {
            continue;
        };
        child
            .stdin
            .take()
            .expect("SHA-256 stdin is piped")
            .write_all(bytes)
            .expect("write SHA-256 input");
        let output = child.wait_with_output().expect("wait for SHA-256 tool");
        if output.status.success()
            && let Some(hash) = String::from_utf8_lossy(&output.stdout)
                .split_whitespace()
                .next()
            && hash.len() == 64
            && hash.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return hash.to_ascii_lowercase();
        }
    }
    panic!("SHA-256 tool unavailable: install sha256sum or shasum")
}

fn identity_file(path: impl AsRef<Path>) -> Vec<u8> {
    let path = path.as_ref();
    std::fs::read(path)
        .unwrap_or_else(|err| panic!("read identity input {}: {err}", path.display()))
}

fn add_identity_part(identity: &mut Vec<u8>, name: &str, value: &[u8]) {
    writeln!(identity, "{name}\t{}", value.len()).expect("write identity header");
    identity.extend_from_slice(value);
    identity.push(b'\n');
}

fn collect_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir)
        .unwrap_or_else(|err| panic!("read identity directory {}: {err}", dir.display()))
    {
        let path = entry.expect("read identity directory entry").path();
        if path
            .file_name()
            .is_some_and(|name| name == "__pycache__" || name.to_string_lossy().starts_with('.'))
        {
            continue;
        }
        if path.is_dir() {
            collect_files(&path, out);
        } else if path.is_file() && path.extension().is_none_or(|extension| extension != "pyc") {
            out.push(path);
        }
    }
}

fn add_identity_tree(identity: &mut Vec<u8>, root: &Path) {
    let mut files = Vec::new();
    collect_files(root, &mut files);
    files.sort();
    for file in files {
        add_identity_part(identity, &file.to_string_lossy(), &identity_file(&file));
    }
}

fn tilelang_nvcc_argv(sm_token: &str) -> String {
    let arch = if sm_token == "90" {
        "90a".to_string()
    } else {
        sm_token.to_string()
    };
    format!(
        "nvcc -cubin -O3 -gencode=arch=compute_{arch},code=sm_{arch} -std=c++17 \
         --expt-relaxed-constexpr -Xcompiler=-fPIC \
         -I$TILELANG_SRC -I$CUTLASS_INCLUDE -I$CUDA_INCLUDE \
         -DENABLE_BF16 -DCUDA_ARCH={sm_token}0 $DEVICE_CU -o $CUBIN"
    )
}

fn tilelang_kernel_src_hash(base_spec: &TileLangKernelSpec, sm_token: &str) -> String {
    let mut identity = Vec::new();
    for path in [
        base_spec.kernel_path.as_str(),
        "tools/tilelang/gen_tilelang_aot.py",
        "kernels.toml",
        "build.rs",
        "../../requirements-build.txt",
    ] {
        add_identity_part(&mut identity, path, &identity_file(path));
    }
    add_identity_tree(&mut identity, Path::new("tools/tilelang/patches"));
    for (name, value) in [
        ("kernel_name", base_spec.kernel_name.as_str()),
        ("out_name", base_spec.out_name.as_str()),
        ("kernel_family", base_spec.kernel_family.as_str()),
        ("kernel_key", base_spec.kernel_key.as_deref().unwrap_or("")),
        ("sm", sm_token),
        ("public_decl", base_spec.public_decl.as_str()),
        ("extern_decl", base_spec.extern_decl.as_str()),
        ("call_args", base_spec.call_args.as_str()),
    ] {
        add_identity_part(&mut identity, name, value.as_bytes());
    }
    add_identity_part(
        &mut identity,
        "num_q_heads",
        base_spec
            .num_q_heads
            .unwrap_or_default()
            .to_string()
            .as_bytes(),
    );
    add_identity_part(
        &mut identity,
        "num_kv_heads",
        base_spec
            .num_kv_heads
            .unwrap_or_default()
            .to_string()
            .as_bytes(),
    );
    add_identity_part(
        &mut identity,
        "nvcc_argv",
        tilelang_nvcc_argv(sm_token).as_bytes(),
    );
    sha256_bytes(&identity)
}

/// Per-SM TileLang AOT artifact: (sm token, exported func name, generated .c path).
type TileLangPerSmArtifact = (String, String, PathBuf);

/// Persistent kernel-artifact cache root, OUTSIDE `target/` so `cargo clean` and
/// fresh clones reuse it. `generated/` is gitignored now, so a clean/CI build
/// regenerates every kernel (TileLang + nvcc, ~60s); this cache restores a prior
/// build's identical `(kernel source × SM × nvcc)` artifact instead of regenning.
/// Disable with `ARLE_CUDA_KERNEL_CACHE=0`; relocate with `ARLE_CUDA_KERNEL_CACHE_DIR`.
fn kernel_cache_root() -> Option<PathBuf> {
    if std::env::var("ARLE_CUDA_KERNEL_CACHE").ok().as_deref() == Some("0") {
        return None;
    }
    let root = std::env::var("ARLE_CUDA_KERNEL_CACHE_DIR")
        .ok()
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var("HOME")
                .ok()
                .map(|h| PathBuf::from(h).join(".cache/arle-cuda-kernels"))
        })?;
    std::fs::create_dir_all(&root).ok()?;
    Some(root)
}

fn resolve_executable(program: &str) -> PathBuf {
    let path = Path::new(program);
    if path.components().count() > 1 && path.is_file() {
        return path.to_path_buf();
    }
    std::env::var_os("PATH")
        .into_iter()
        .flat_map(|paths| std::env::split_paths(&paths).collect::<Vec<_>>())
        .map(|dir| dir.join(program))
        .find(|candidate| candidate.is_file())
        .unwrap_or_else(|| panic!("compiler executable not found: {program}"))
}

fn executable_identity(program: &Path) -> Vec<u8> {
    // Resolve bare names (e.g. "python3", "g++") to their full PATH so the
    // identity hash reads the actual binary, not a cwd-relative file that
    // doesn't exist (panics with "No such file or directory").
    let resolved = if program.components().count() > 1 && program.is_file() {
        program.to_path_buf()
    } else {
        resolve_executable(&program.to_string_lossy())
    };
    let output = Command::new(&resolved)
        .arg("--version")
        .output()
        .unwrap_or_else(|err| panic!("run executable {}: {err}", resolved.display()));
    let mut identity = Vec::new();
    add_identity_part(&mut identity, "path", resolved.to_string_lossy().as_bytes());
    add_identity_part(&mut identity, "binary", &identity_file(&resolved));
    add_identity_part(&mut identity, "stdout", &output.stdout);
    add_identity_part(&mut identity, "stderr", &output.stderr);
    identity
}

fn compiler_identity(program: &str) -> Vec<u8> {
    executable_identity(&resolve_executable(program))
}

fn tilelang_toolchain_id(
    toolchain: &TilelangToolchain,
    nvcc: &Path,
    wrapper: &[String],
    cuda_include: &Path,
) -> String {
    let mut identity = Vec::new();
    add_identity_part(
        &mut identity,
        "python",
        &executable_identity(&toolchain.python),
    );
    add_identity_tree(&mut identity, &toolchain.package);
    add_identity_tree(&mut identity, &toolchain.tvm_runtime);
    add_identity_tree(&mut identity, &toolchain.tvm_ffi);
    add_identity_tree(&mut identity, &toolchain.src);
    add_identity_tree(&mut identity, &toolchain.cutlass_include);
    add_identity_tree(&mut identity, cuda_include);
    add_identity_part(&mut identity, "nvcc", &executable_identity(nvcc));
    add_identity_part(
        &mut identity,
        "host_compiler",
        &compiler_identity(&std::env::var("NVCC_CCBIN").unwrap_or_else(|_| "g++".to_string())),
    );
    for (index, arg) in wrapper.iter().enumerate() {
        add_identity_part(
            &mut identity,
            &format!("wrapper_argv_{index}"),
            arg.as_bytes(),
        );
        if index == 0 {
            add_identity_part(
                &mut identity,
                "wrapper_executable",
                &executable_identity(&resolve_executable(arg)),
            );
        }
    }
    sha256_bytes(&identity)
}

fn tilelang_cache_id(src_hash: &str, sm_token: &str, toolchain_id: &str) -> String {
    let mut identity = Vec::new();
    add_identity_part(&mut identity, "source", src_hash.as_bytes());
    add_identity_part(&mut identity, "toolchain", toolchain_id.as_bytes());
    add_identity_part(
        &mut identity,
        "nvcc_argv",
        tilelang_nvcc_argv(sm_token).as_bytes(),
    );
    sha256_bytes(&identity)
}

fn metadata_value<'a>(metadata: &'a str, key: &str) -> Option<&'a str> {
    metadata
        .lines()
        .find_map(|line| line.strip_prefix(key).map(str::trim))
        .filter(|value| !value.is_empty())
}

fn validate_tilelang_artifact(
    dir: &Path,
    out_name: &str,
    src_hash: &str,
    cache_id: Option<&str>,
) -> Result<String, String> {
    let metadata = std::fs::read_to_string(dir.join("meta.txt"))
        .map_err(|err| format!("read meta.txt: {err}"))?;
    let value =
        |key| metadata_value(&metadata, key).ok_or_else(|| format!("meta.txt missing {key}"));
    if value("SRC_HASH=")? != src_hash {
        return Err("source hash mismatch".to_string());
    }
    let recorded_cache_id = value("CACHE_ID=")?;
    if cache_id.is_some_and(|expected| recorded_cache_id != expected) {
        return Err("cache ID mismatch".to_string());
    }
    let c_path = dir.join(format!("{out_name}.cc"));
    let output_hash = sha256_bytes(
        &std::fs::read(&c_path).map_err(|err| format!("read {}: {err}", c_path.display()))?,
    );
    if value("OUTPUT_SHA256=")? != output_hash {
        return Err("output hash mismatch".to_string());
    }
    if let Some(expected_cache_id) = cache_id {
        let marker = std::fs::read_to_string(dir.join(".complete"))
            .map_err(|err| format!("read .complete: {err}"))?;
        if metadata_value(&marker, "CACHE_ID=") != Some(expected_cache_id)
            || metadata_value(&marker, "OUTPUT_SHA256=") != Some(output_hash.as_str())
        {
            return Err("completion marker mismatch".to_string());
        }
    }
    Ok(value("FUNC_NAME=")?.to_string())
}

fn lock_tilelang_cache(root: &Path, cache_id: &str) -> File {
    let lock_dir = root.join("locks");
    std::fs::create_dir_all(&lock_dir)
        .unwrap_or_else(|err| panic!("create cache lock directory {}: {err}", lock_dir.display()));
    let path = lock_dir.join(format!("{cache_id}.lock"));
    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&path)
        .unwrap_or_else(|err| panic!("open cache lock {}: {err}", path.display()));
    file.lock()
        .unwrap_or_else(|err| panic!("lock TileLang cache {}: {err}", path.display()));
    file
}

fn run_tilelang_cache_identity_self_test(registry: &Registry, out_dir: &Path) {
    assert_eq!(
        sha256_bytes(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    let (spec, _) = registry_to_specs(registry)
        .into_iter()
        .next()
        .expect("registry has a TileLang kernel");
    let base = tilelang_kernel_src_hash(&spec, "90");
    let mut changed = spec.clone();
    changed.public_decl.push_str(" changed");
    assert_ne!(base, tilelang_kernel_src_hash(&changed, "90"));
    assert_ne!(base, tilelang_kernel_src_hash(&spec, "89"));

    let dir = out_dir.join("tilelang_cache_identity_self_test");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create cache identity self-test dir");
    let out_name = "probe";
    let output = b"int probe(void) { return 0; }\n";
    let output_hash = sha256_bytes(output);
    std::fs::write(dir.join("probe.cc"), output).expect("write self-test output");
    std::fs::write(
        dir.join("meta.txt"),
        format!(
            "FUNC_NAME=probe_cuda\nSRC_HASH={base}\nCACHE_ID=cache\nOUTPUT_SHA256={output_hash}\n"
        ),
    )
    .expect("write self-test metadata");
    std::fs::write(
        dir.join(".complete"),
        format!("CACHE_ID=cache\nOUTPUT_SHA256={output_hash}\n"),
    )
    .expect("write self-test marker");
    validate_tilelang_artifact(&dir, out_name, &base, Some("cache")).expect("valid cache fixture");
    std::fs::write(dir.join("probe.cc"), b"corrupt").expect("corrupt self-test output");
    assert!(validate_tilelang_artifact(&dir, out_name, &base, Some("cache")).is_err());
    std::fs::remove_dir_all(&dir).expect("remove cache identity self-test dir");
}

// The shared C ABI signatures (the old TILELANG_DISPATCH_* / GDR_* / FQ_*
// *_PUBLIC_DECL / *_EXTERN_DECL / *_CALL_ARGS consts) now live in the
// `[abi.*]` blocks of kernels.toml and reach build code via the parsed
// `TileLangKernelSpec.public_decl/extern_decl/call_args` fields.

fn tilelang_toolchain_probe() -> &'static str {
    r#"
import importlib.util
import sys
from pathlib import Path

def package(name):
    spec = importlib.util.find_spec(name)
    if spec is None:
        raise SystemExit("ERR_NOT_INSTALLED:" + name)
    if spec.submodule_search_locations:
        return Path(next(iter(spec.submodule_search_locations))).resolve()
    return Path(spec.origin).resolve()

pkg = package("tilelang")
tvm_ffi = package("tvm_ffi")
roots = [pkg, pkg.parent]
tvm_runtime = next((root / "lib" for root in roots if (root / "lib").is_dir()), None)
src = next((root / "src" for root in roots if (root / "src" / "tl_templates").exists()), None)
cutlass = next((root / "3rdparty" / "cutlass" / "include" for root in roots if (root / "3rdparty" / "cutlass" / "include").exists()), None)
if tvm_runtime is None or src is None or cutlass is None:
    raise SystemExit("ERR_LAYOUT:" + str(pkg))
print(pkg)
print(tvm_runtime)
print(tvm_ffi)
print(src)
print(cutlass)
"#
}

fn run_tilelang_toolchain_probe_self_test(out_dir: &Path) {
    let root = out_dir.join("tilelang_toolchain_probe_self_test");
    let _ = std::fs::remove_dir_all(&root);
    for path in [
        "tilelang/lib",
        "tilelang/src/tl_templates",
        "tilelang/3rdparty/cutlass/include",
        "tvm_ffi",
    ] {
        std::fs::create_dir_all(root.join(path)).expect("create TileLang probe fixture");
    }
    std::fs::write(root.join("tilelang/__init__.py"), "").expect("write tilelang fixture");
    std::fs::write(root.join("tvm_ffi/__init__.py"), "").expect("write tvm-ffi fixture");
    let output = Command::new("python3")
        .arg("-c")
        .arg(tilelang_toolchain_probe())
        .env("PYTHONPATH", &root)
        .output()
        .expect("run TileLang toolchain probe self-test");
    assert!(
        output.status.success(),
        "TileLang toolchain probe self-test failed: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    let paths = String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    assert_eq!(paths.len(), 5);
    assert_eq!(paths[0], root.join("tilelang"));
    assert_eq!(paths[1], root.join("tilelang/lib"));
    assert_eq!(paths[2], root.join("tvm_ffi"));
    assert_eq!(paths[3], root.join("tilelang/src"));
    assert_eq!(paths[4], root.join("tilelang/3rdparty/cutlass/include"));
    assert!(!root.join("tvm").exists());
    std::fs::remove_dir_all(&root).expect("remove TileLang probe fixture");
}

/// Locate the installed codegen and runtime trees that affect TileLang output.
fn tilelang_toolchain(python: &str) -> TilelangToolchain {
    let output = Command::new(python)
        .arg("-c")
        .arg(tilelang_toolchain_probe())
        .output()
        .expect("failed to probe tilelang install path");
    assert!(
        output.status.success(),
        "tilelang toolchain probe failed: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    let paths = String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    assert_eq!(
        paths.len(),
        5,
        "tilelang toolchain probe returned invalid output"
    );
    TilelangToolchain {
        python: PathBuf::from(python),
        package: paths[0].clone(),
        tvm_runtime: paths[1].clone(),
        tvm_ffi: paths[2].clone(),
        src: paths[3].clone(),
        cutlass_include: paths[4].clone(),
    }
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
    tl: &LazyTilelang,
    out_dir: &Path,
    sm_targets: &[SmSpec],
    cuda_path: &str,
    base_spec: &TileLangKernelSpec,
) -> Vec<TileLangPerSmArtifact> {
    let generator_path = PathBuf::from("tools/tilelang/gen_tilelang_aot.py");
    let manifest_dir = PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR set by cargo"),
    );
    let force_regen = env_truthy("ARLE_TILELANG_REGEN");
    let cache_root = kernel_cache_root();
    let nvcc = resolve_executable(&format!("{cuda_path}/bin/nvcc"));
    let cuda_include = PathBuf::from(cuda_path).join("include");
    let wrapper: Vec<String> = env_nonempty("ARLE_NVCC_WRAPPER")
        .map(|value| {
            value
                .split_whitespace()
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    // Lazily resolved — only when a vendored/cached artifact is unavailable and
    // a kernel must be (re)generated. A fully-vendored build never touches this,
    // so it never needs TileLang installed.
    let toolchain_id_cell: std::cell::OnceCell<String> = std::cell::OnceCell::new();
    let mut results = Vec::new();

    for sm in sm_targets {
        let sm_token = &sm.sm;
        let cuda_arch: u32 = sm_token
            .parse()
            .expect("SmSpec.sm passed whitelist; must parse as u32");
        // FlashQLA's warp-specialized kernels emit `setmaxnreg`, which ptxas takes
        // only under the sm_90a variant — same constraint that forces sm_90a on the
        // FlashMLA/FA3 TUs below. Artifact names keep `sm90`; only the target moves.
        let arch_token = if base_spec.kernel_family == "flashqla" && sm_token == "90" {
            "90a"
        } else {
            sm_token.as_str()
        };
        let target = format!("cuda -arch=sm_{arch_token}");

        let src_hash = tilelang_kernel_src_hash(base_spec, sm_token);
        let per_sm_artifact_dir = format!("{}_sm{sm_token}", base_spec.artifact_dir);
        let per_sm_out_name = format!("{}_sm{sm_token}", base_spec.out_name);
        let per_sm_kernel_name = format!("{}_sm{sm_token}", base_spec.kernel_name);
        let vendored_dir = manifest_dir.join("generated").join(&per_sm_artifact_dir);
        let out_artifact_dir = out_dir.join("tilelang_aot").join(&per_sm_artifact_dir);
        let vendored_c = vendored_dir.join(format!("{per_sm_out_name}.cc"));

        let vendored_func = if !force_regen && vendored_dir.exists() {
            match validate_tilelang_artifact(&vendored_dir, &per_sm_out_name, &src_hash, None) {
                Ok(func) => Some(func),
                Err(err) => {
                    println!(
                        "cargo:warning=rejecting invalid vendored TileLang artifact {}: {err}",
                        vendored_dir.display()
                    );
                    None
                }
            }
        } else {
            None
        };
        if let Some(func) = vendored_func {
            let _ = std::fs::remove_dir_all(&out_artifact_dir);
            copy_dir_recursive(&vendored_dir, &out_artifact_dir).unwrap_or_else(|err| {
                panic!(
                    "consume vendored TileLang artifact {} -> {}: {err}",
                    vendored_dir.display(),
                    out_artifact_dir.display()
                )
            });
            let consumed_c = out_artifact_dir.join(format!("{per_sm_out_name}.cc"));
            println!("cargo:rerun-if-changed={}", vendored_c.display());
            println!("cargo:rerun-if-changed={}", base_spec.kernel_path);
            results.push((sm_token.clone(), func, consumed_c));
            continue;
        }

        // Reached only when the vendored artifact is missing/invalid — now we
        // may need the toolchain (cache lookup or regeneration).
        let toolchain_id = toolchain_id_cell
            .get_or_init(|| tilelang_toolchain_id(tl.get(), &nvcc, &wrapper, &cuda_include));
        let cache_id = tilelang_cache_id(&src_hash, sm_token, toolchain_id);

        let cache_entry = cache_root
            .as_ref()
            .map(|root| root.join("objects").join(&cache_id));
        let _cache_lock = cache_root
            .as_ref()
            .map(|root| lock_tilelang_cache(root, &cache_id));
        if let Some(entry) = &cache_entry
            && !force_regen
        {
            match validate_tilelang_artifact(entry, &per_sm_out_name, &src_hash, Some(&cache_id)) {
                Ok(func) => {
                    let _ = std::fs::remove_dir_all(&out_artifact_dir);
                    copy_dir_recursive(entry, &out_artifact_dir).unwrap_or_else(|err| {
                        panic!(
                            "restore TileLang cache {} -> {}: {err}",
                            entry.display(),
                            out_artifact_dir.display()
                        )
                    });
                    println!("cargo:rerun-if-changed={}", base_spec.kernel_path);
                    results.push((
                        sm_token.clone(),
                        func,
                        out_artifact_dir.join(format!("{per_sm_out_name}.cc")),
                    ));
                    continue;
                }
                Err(_) if !entry.exists() => {}
                Err(err) => {
                    println!(
                        "cargo:warning=discarding invalid TileLang cache {}: {err}",
                        entry.display()
                    );
                    std::fs::remove_dir_all(entry).unwrap_or_else(|remove_err| {
                        panic!("remove invalid cache {}: {remove_err}", entry.display())
                    });
                }
            }
        }

        let _ = std::fs::remove_dir_all(&out_artifact_dir);

        let toolchain = tl.get();
        let mut command = Command::new(&toolchain.python);
        let output = command
            .arg(&generator_path)
            .arg("--kernel-path")
            .arg(&base_spec.kernel_path)
            .arg("--kernel-name")
            .arg(&per_sm_kernel_name)
            .arg("--out-name")
            .arg(&per_sm_out_name)
            .arg("--out-dir")
            .arg(&out_artifact_dir)
            .arg("--target")
            .arg(&target)
            .arg("--kernel-family")
            .arg(&base_spec.kernel_family)
            .arg("--cuda-arch")
            .arg(cuda_arch.to_string())
            .arg("--tilelang-src")
            .arg(&toolchain.src)
            .arg("--cutlass-include")
            .arg(&toolchain.cutlass_include)
            .arg("--cuda-include")
            .arg(&cuda_include)
            .arg("--nvcc")
            .arg(&nvcc)
            .args(
                wrapper
                    .iter()
                    .flat_map(|arg| ["--nvcc-wrapper".to_string(), arg.clone()]),
            )
            .args(
                base_spec
                    .kernel_key
                    .as_deref()
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
                 See docs/environment.md.",
                base_spec.kernel_name,
                String::from_utf8_lossy(&output.stdout).trim(),
                String::from_utf8_lossy(&output.stderr).trim(),
            );
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut gen_func_name = None;
        let mut c_path = None;
        for line in stdout.lines() {
            if let Some(value) = line.strip_prefix("FUNC_NAME=") {
                gen_func_name = Some(value.trim().to_string());
            } else if let Some(value) = line.strip_prefix("C_PATH=") {
                c_path = Some(PathBuf::from(value.trim()));
            }
        }
        let gen_func_name = gen_func_name.expect("TileLang generator did not print FUNC_NAME");
        let c_path = c_path.expect("TileLang generator did not print C_PATH");
        let output_hash = sha256_bytes(
            &std::fs::read(&c_path)
                .unwrap_or_else(|err| panic!("read generated output {}: {err}", c_path.display())),
        );

        std::fs::write(
            out_artifact_dir.join("meta.txt"),
            format!(
                "FUNC_NAME={gen_func_name}\nSRC_HASH={src_hash}\nCACHE_ID={cache_id}\nOUTPUT_SHA256={output_hash}\n"
            ),
        )
        .unwrap_or_else(|err| panic!("write meta.txt for {}: {err}", out_artifact_dir.display()));
        validate_tilelang_artifact(&out_artifact_dir, &per_sm_out_name, &src_hash, None)
            .unwrap_or_else(|err| panic!("validate generated artifact: {err}"));

        if let Some(entry) = &cache_entry {
            let parent = entry.parent().expect("cache entry has parent");
            std::fs::create_dir_all(parent)
                .unwrap_or_else(|err| panic!("create cache directory {}: {err}", parent.display()));
            let staging =
                entry.with_file_name(format!("{cache_id}.staging.{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&staging);
            copy_dir_recursive(&out_artifact_dir, &staging)
                .unwrap_or_else(|err| panic!("stage TileLang cache {}: {err}", staging.display()));
            std::fs::write(
                staging.join(".complete"),
                format!("CACHE_ID={cache_id}\nOUTPUT_SHA256={output_hash}\n"),
            )
            .unwrap_or_else(|err| panic!("write cache marker {}: {err}", staging.display()));
            validate_tilelang_artifact(&staging, &per_sm_out_name, &src_hash, Some(&cache_id))
                .unwrap_or_else(|err| panic!("validate staged cache {}: {err}", staging.display()));
            let _ = std::fs::remove_dir_all(entry);
            std::fs::rename(&staging, entry).unwrap_or_else(|err| {
                panic!(
                    "publish TileLang cache {} -> {}: {err}",
                    staging.display(),
                    entry.display()
                )
            });
        }

        results.push((sm_token.clone(), gen_func_name, c_path));
    }

    // ARLE_KERNEL_VENDOR=1: mirror every consumed/restored/regenerated artifact
    // into the (gitignored) vendored tier — `scripts/kernel_artifacts.sh pack`
    // tars that directory into the GitHub-Release kernel bundle, and `fetch`
    // extracts back into it so consumers build with zero Python.
    if env_truthy("ARLE_KERNEL_VENDOR") {
        for (_, _, c_path) in &results {
            let src = c_path.parent().expect("artifact .c has a parent dir");
            let dst = manifest_dir
                .join("generated")
                .join(src.file_name().expect("artifact dir has a name"));
            if src != dst {
                let _ = std::fs::remove_dir_all(&dst);
                copy_dir_recursive(src, &dst)
                    .unwrap_or_else(|err| panic!("vendor export {}: {err}", dst.display()));
            }
        }
    }

    results
}

/// Build one TileLang head-config kernel for every SM target: generate
/// per-SM artifacts, write a single dispatch wrapper exposing
/// `<base_kernel_name>_cuda`, and append all sources to
/// `generated_sources` for cc::Build to compile.
fn build_tilelang_kernel(
    tl: &LazyTilelang,
    out_dir: &Path,
    sm_targets: &[SmSpec],
    cuda_path: &str,
    base_spec: &TileLangKernelSpec,
    generated_sources: &mut Vec<PathBuf>,
) {
    let eligible_targets: Vec<SmSpec> = sm_targets
        .iter()
        .filter(|sm| base_spec.allow_sm70 || !is_legacy_volta_sm(&sm.sm))
        .cloned()
        .collect();
    let per_sm =
        generate_tilelang_artifacts_per_sm(tl, out_dir, &eligible_targets, cuda_path, base_spec);
    let pairs: Vec<(String, String)> = per_sm
        .iter()
        .map(|(sm, func, _)| (sm.clone(), func.clone()))
        .collect();

    let public_name = format!("{}_cuda", base_spec.kernel_name);
    let public_decl = format!("{public_name}({})", base_spec.public_decl);
    let wrapper_src = format_dispatch_wrapper(
        &public_decl,
        &base_spec.extern_decl,
        &base_spec.call_args,
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

fn write_tilelang_unsupported_stub(
    out_dir: &Path,
    artifact_dir: &str,
    out_name: &str,
    kernel_name: &str,
    public_decl: &str,
    generated_sources: &mut Vec<PathBuf>,
) {
    let public_name = format!("{kernel_name}_cuda");
    let public_decl = format!("{public_name}({public_decl})");
    let src = format!(
        "#include <cuda.h>\n\
         #include <stdint.h>\n\
         \n\
         CUresult {public_decl} {{\n\
         \x20   return CUDA_ERROR_NOT_SUPPORTED;\n\
         }}\n"
    );
    let stub_dir = out_dir.join("tilelang_stub").join(artifact_dir);
    std::fs::create_dir_all(&stub_dir).expect("create TileLang stub directory");
    let stub_path = stub_dir.join(format!("{out_name}_stub.c"));
    std::fs::write(&stub_path, src).expect("write TileLang unsupported stub");
    generated_sources.push(stub_path);
}

fn compile_tilelang_aot_kernels(
    reg: &Registry,
    cuda_path: &str,
    out_dir: &Path,
    sm_targets: &[SmSpec],
) {
    let tl = LazyTilelang::new();
    let mut generated_sources = Vec::new();

    // Registry-driven: one TileLangKernelSpec per kernels.toml row. Every SM
    // target gets every kernel; the flashqla rows need sm_90, which is the
    // whole gate — an env opt-in made `--qwen35-gdr-chunked` (default true) a
    // silent no-op for three days.
    let sm90_targets: Vec<SmSpec> = sm_targets
        .iter()
        .filter(|sm| sm.sm == "90")
        .cloned()
        .collect();
    for (spec, k) in registry_to_specs(reg) {
        if k.gate == "flashqla" {
            if sm90_targets.is_empty() {
                write_tilelang_unsupported_stub(
                    out_dir,
                    &spec.artifact_dir,
                    &spec.out_name,
                    &spec.kernel_name,
                    &spec.public_decl,
                    &mut generated_sources,
                );
            } else {
                build_tilelang_kernel(
                    &tl,
                    out_dir,
                    &sm90_targets,
                    cuda_path,
                    &spec,
                    &mut generated_sources,
                );
            }
        } else {
            build_tilelang_kernel(
                &tl,
                out_dir,
                sm_targets,
                cuda_path,
                &spec,
                &mut generated_sources,
            );
        }
    }

    // Vendor the freshly-generated per-SM TileLang artifacts back into the
    // source tree BEFORE cc compiles them, so a regen run captures the exact
    // `.cu`/`.c`/`.cubin` triples that just compiled clean. Opt-in via
    // ARLE_TILELANG_REGEN=1 — the default build never writes into the tree.
    println!("cargo:rerun-if-env-changed=ARLE_TILELANG_REGEN");
    if env_truthy("ARLE_TILELANG_REGEN") {
        vendor_tilelang_generated(out_dir, sm_targets);
    }

    let mut build = cc::Build::new();
    build
        .cuda(false)
        .include(format!("{}/include", cuda_path))
        .flag("-std=c++17")
        .warnings(false);
    for source in &generated_sources {
        build.file(source);
    }
    build.compile("tilelang_kernels_aot");

    println!("cargo:rustc-link-lib=cuda");
    println!(
        "cargo:warning=TileLang AOT: built per-SM cubins for {} target(s) across HD64/HD256 prefill, HD64/HD256 decode, and Qwen3.5 GDR; SM dispatch via __thread cache + cuDeviceGetAttribute. See docs/environment.md.",
        sm_targets.len()
    );
    if has_legacy_volta(sm_targets) {
        println!(
            "cargo:warning=sm_70 legacy Volta build: TileLang AOT emits BF16 Qwen3.5/3.6 full-attention (all HD256 BF16 configs) and GDR cubins; FP8 KV (sm_80+), DSv4 HD64, and HD64 wrappers return CUDA_ERROR_NOT_SUPPORTED."
        );
    }
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
// (attention/, gemm/, moe/, kv/, quant/, sampling/, norm/, recurrent/,
// elementwise/, ...) are picked up automatically.
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
fn emit_rerun_recursive(dir: &Path) {
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

fn env_flag(name: &str) -> bool {
    matches!(
        std::env::var(name).as_deref(),
        Ok("1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON")
    )
}

fn env_nonempty(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

/// True when `name` is set to a truthy value (`1`, `true`, `yes`, `on`,
/// case-insensitive). Used by opt-in build toggles like ARLE_TILELANG_REGEN.
fn env_truthy(name: &str) -> bool {
    match env_nonempty(name) {
        Some(value) => matches!(
            value.to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        None => false,
    }
}

/// Recursively copy `src` into `dst`, creating `dst` and any parents.
fn copy_dir_recursive(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if from.is_dir() {
            copy_dir_recursive(&from, &to)?;
        } else {
            std::fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

/// Vendor the per-SM TileLang AOT artifacts (the `*_device_kernel.cu` device
/// sources, their `.c` host launchers, `.cubin` blobs, and the `*_dispatch`
/// wrappers) from the ephemeral `$OUT_DIR/tilelang_aot/` into the source tree
/// at `crates/cuda-kernels/generated/`, one subdirectory per artifact_dir
/// (which is already suffixed `_sm{token}` / `_dispatch`).
///
/// Opt-in (ARLE_TILELANG_REGEN=1). The freshly-generated tree is the source of
/// truth — every regen mirrors exactly what just compiled clean for the
/// requested SM targets, so the checked-in `.cu` is reproducible. A vendoring
/// run on a Blackwell box (TORCH_CUDA_ARCH_LIST="10.0;12.0") populates the
/// `*_sm100/` and `*_sm120/` directories; a Hopper/Ampere box populates its own.
fn vendor_tilelang_generated(out_dir: &Path, sm_targets: &[SmSpec]) {
    let aot_root = out_dir.join("tilelang_aot");
    if !aot_root.is_dir() {
        println!(
            "cargo:warning=ARLE_TILELANG_REGEN set but {} is absent; nothing to vendor.",
            aot_root.display()
        );
        return;
    }
    let manifest_dir = PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR set by cargo"),
    );
    let generated_root = manifest_dir.join("generated");
    let sm_tokens: Vec<String> = sm_targets.iter().map(|s| format!("sm{}", s.sm)).collect();

    let mut copied = 0usize;
    for entry in std::fs::read_dir(&aot_root)
        .unwrap_or_else(|err| panic!("read {} for regen: {err}", aot_root.display()))
        .flatten()
    {
        let from = entry.path();
        if !from.is_dir() {
            continue;
        }
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        // Dispatch wrappers are ephemeral C++ glue, never cache artifacts.
        if name_str.ends_with("_dispatch") {
            continue;
        }
        let is_target_sm = sm_tokens.iter().any(|tok| name_str.ends_with(tok));
        if !is_target_sm {
            continue;
        }
        let to = generated_root.join(name);
        copy_dir_recursive(&from, &to)
            .unwrap_or_else(|err| panic!("vendor {} -> {}: {err}", from.display(), to.display()));
        copied += 1;
    }
    println!(
        "cargo:warning=ARLE_TILELANG_REGEN: vendored {copied} TileLang artifact dir(s) for SM target(s) [{}] into {}.",
        sm_tokens.join(","),
        generated_root.display()
    );
}

fn emit_cuda_system_link_libs(cuda_path: &str) {
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

fn emit_prebuilt_deepep_sidecar(path: &Path) {
    if !path.is_file() {
        panic!(
            "ARLE_DEEPEP_SIDECAR_PREBUILT={} is not a file",
            path.display()
        );
    }
    println!("cargo:rerun-if-changed={}", path.display());
    println!(
        "cargo:rustc-env=ARLE_DEEPEP_SIDECAR_PATH={}",
        path.display()
    );
    println!(
        "cargo:warning=Using prebuilt ARLE DeepEP sidecar at {}.",
        path.display()
    );
}

const PREBUILT_REQUIRED_DSV4_SYMBOLS: &[&str] = &[
    "dsv4_deepgemm_native_preflight_cuda",
    "dsv4_sm90_mega_moe_workspace_layout_cuda",
    "dsv4_sm90_mega_moe_pre_dispatch_cuda",
    "dsv4_sm90_mega_moe_launch_cuda",
    "arle_dsv4_fp8_kv_fill_one_sw_slot_from_start_pos_cuda",
    "arle_dsv4_flashmla_decode_build_indices_start_pos_ptr_cuda",
    "arle_dsv4_flashmla_decode_build_indices_batched_cuda",
    "dsv4_prepare_qk_start_pos_ptr_cuda",
    "dsv4_prepare_qk_fused_start_pos_ptr_cuda",
    "arle_dsv4_output_inverse_rope_cuda",
    "arle_dsv4_output_inverse_rope_start_pos_ptr_cuda",
    "dsv4_update_window_cache_start_pos_ptr_cuda",
    "dsv4_compressor_update_start_pos_ptr_cuda",
    "arle_dsv4_fp8_kv_pack_completed_compressor_row_start_pos_cuda",
    "arle_dsv4_v32_fp8_kv_pack_strided_cuda",
    "dsv4_mtp_add_eproj_hproj_cuda",
];

const PREBUILT_MANIFEST: &str = "arle-cuda-kernels.manifest";
const PREBUILT_SCHEMA: &str = "3";
const PREBUILT_ARTIFACTS: &[&str] = &[
    "libkernels_cuda.a",
    "libtilelang_kernels_aot.a",
    "arle_deepep_sidecar",
];

fn command_id(program: &str, args: &[&str]) -> String {
    let mut identity = program.as_bytes().to_vec();
    if let Ok(output) = Command::new(program).args(args).output() {
        identity.extend(output.stdout);
        identity.extend(output.stderr);
    } else {
        identity.extend(b"missing");
    }
    sha256_bytes(&identity)
}

fn cargo_features() -> String {
    std::env::vars_os()
        .filter_map(|(key, _)| {
            key.to_str()?
                .strip_prefix("CARGO_FEATURE_")
                .map(str::to_owned)
        })
        .map(|feature| feature.to_ascii_lowercase().replace('_', "-"))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>()
        .join(",")
}

fn resolved_sms(sm_targets: &[SmSpec]) -> String {
    sm_targets
        .iter()
        .map(|spec| format!("sm_{}{}", spec.sm, if spec.ptx { "+ptx" } else { "" }))
        .collect::<Vec<_>>()
        .join(",")
}

fn required_symbols(capabilities: &BTreeSet<String>) -> Vec<&'static str> {
    let mut symbols = PREBUILT_REQUIRED_DSV4_SYMBOLS.to_vec();
    if capabilities.contains("flashmla") {
        symbols.push("arle_flashmla_sm90_sparse_decode_real_kernel_marker_cuda");
    }
    if capabilities.contains("fa3") {
        symbols.push("arle_fa3_real_kernel_marker_cuda");
    }
    symbols.sort_unstable();
    symbols
}

fn archive_symbols(archive: &Path) -> BTreeSet<String> {
    let output = Command::new("nm")
        .args(["-g", "--defined-only"])
        .arg(archive)
        .output()
        .unwrap_or_else(|err| panic!("Failed to run nm for {}: {err}", archive.display()));
    if !output.status.success() {
        panic!(
            "Failed to inspect CUDA archive {} with nm: {}",
            archive.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| line.split_whitespace().last())
        .map(str::to_string)
        .collect()
}

fn validate_archive_symbols(archive: &Path, required: &[&str]) {
    let symbols = archive_symbols(archive);
    let missing = required
        .iter()
        .copied()
        .filter(|symbol| !symbols.contains(*symbol))
        .collect::<Vec<_>>();
    assert!(
        missing.is_empty(),
        "CUDA archive {} is missing required symbols: {}",
        archive.display(),
        missing.join(", ")
    );
}

fn collect_input_files(path: &Path, files: &mut Vec<PathBuf>) {
    if path.is_file() {
        files.push(path.to_path_buf());
        return;
    }
    let Ok(entries) = std::fs::read_dir(path) else {
        return;
    };
    for entry in entries.flatten() {
        let entry_path = entry.path();
        // Exclude generated/ (vendored artifacts) and Python bytecode caches:
        // the producer computes this hash AFTER the build (when __pycache__
        // exists), the consumer BEFORE it — the two must agree. Matches the
        // exclusion in collect_files (TileLang cache identity).
        if entry_path.file_name().is_some_and(|name| {
            name == "generated" || name == "__pycache__" || name.to_string_lossy().starts_with('.')
        }) {
            continue;
        }
        if entry_path.is_dir() {
            collect_input_files(&entry_path, files);
        } else if entry_path.is_file()
            && entry_path
                .extension()
                .is_none_or(|extension| extension != "pyc")
        {
            files.push(entry_path);
        }
    }
}

fn hash_input_root(digest: &mut ring::digest::Context, label: &str, root: &Path) {
    let mut files = Vec::new();
    collect_input_files(root, &mut files);
    files.sort();
    for file in files {
        let relative = file.strip_prefix(root).unwrap_or(&file);
        digest.update(label.as_bytes());
        digest.update(b"/");
        digest.update(relative.to_string_lossy().as_bytes());
        digest.update(b"\0");
        digest.update(sha256_file(&file).as_bytes());
        digest.update(b"\n");
    }
}

fn producer_inputs_sha256() -> String {
    let crate_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let repo_root = crate_dir.parent().and_then(Path::parent).unwrap();
    let mut roots = vec![
        ("workspace-cargo", repo_root.join("Cargo.toml")),
        ("workspace-lock", repo_root.join("Cargo.lock")),
        (
            "requirements-build",
            repo_root.join("requirements-build.txt"),
        ),
        ("cuda-cargo", crate_dir.join("Cargo.toml")),
        ("cuda-build", crate_dir.join("build.rs")),
        ("cuda-registry", crate_dir.join("kernels.toml")),
        ("cuda-csrc", crate_dir.join("csrc")),
        ("cuda-rust", crate_dir.join("src")),
        ("cuda-tools", crate_dir.join("tools")),
        ("cuda-vendor", crate_dir.join("vendor")),
    ];
    for (env, label) in [
        ("ARLE_DEEPGEMM_ROOT", "deepgemm"),
        ("ARLE_DEEPGEMM_LIBRARY_ROOT", "deepgemm-library"),
        ("ARLE_DEEPGEMM_CUTLASS_INCLUDE", "deepgemm-cutlass"),
        ("ARLE_DEEPEP_DIR", "deepep"),
        ("ARLE_DEEPEP_NVSHMEM_DIR", "nvshmem"),
    ] {
        if let Some(path) = env_nonempty(env) {
            roots.push((label, PathBuf::from(path)));
        }
    }
    let mut digest = ring::digest::Context::new(&ring::digest::SHA256);
    for (label, root) in roots {
        hash_input_root(&mut digest, label, &root);
    }
    hex_digest(digest.finish().as_ref())
}

/// Resolve the CUTLASS include tree for DeepGEMM. The producer (compile path)
/// and the consumer (`configured_capabilities`) must agree, or the prebuilt
/// producer contract mismatches on `deepgemm-native`.
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

fn configured_capabilities(sm_targets: &[SmSpec]) -> BTreeSet<String> {
    let legacy_volta = has_legacy_volta(sm_targets);
    let mut capabilities = BTreeSet::new();
    if Path::new("vendor/flashmla").is_dir()
        && !env_flag("ARLE_CUDA_DISABLE_FLASHMLA")
        && !legacy_volta
    {
        capabilities.insert("flashmla".into());
    }
    let sm90 = sm_targets.iter().any(|spec| spec.sm == "90");
    if sm90 && Path::new("vendor/flash-attention/hopper").is_dir() {
        capabilities.insert("fa3".into());
    }
    if sm90 {
        capabilities.insert("flashqla".into());
    }
    let deepgemm_root = env_nonempty("ARLE_DEEPGEMM_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("vendor/deepgemm"));
    let deepgemm_library = env_nonempty("ARLE_DEEPGEMM_LIBRARY_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| deepgemm_root.join("deep_gemm"));
    let deepgemm_cutlass = resolve_deepgemm_cutlass_include(&deepgemm_root);
    if !env_flag("ARLE_CUDA_DISABLE_DEEPGEMM_NATIVE")
        && sm_targets.iter().any(|spec| spec.sm == "90")
        && deepgemm_library.is_dir()
        && deepgemm_cutlass.join("cutlass/arch/barrier.h").is_file()
    {
        capabilities.insert("deepgemm-native".into());
    }
    if std::env::var_os("CARGO_FEATURE_NCCL").is_some() {
        capabilities.insert("nccl".into());
    }
    if env_nonempty("ARLE_DEEPEP_DIR").is_some() {
        capabilities.insert("deepep-sidecar".into());
    }
    capabilities
}

fn producer_contract(
    cuda_path: &str,
    sm_targets: &[SmSpec],
    capabilities: &BTreeSet<String>,
) -> BTreeMap<String, String> {
    let nvcc = format!("{cuda_path}/bin/nvcc");
    BTreeMap::from([
        ("schema".into(), PREBUILT_SCHEMA.into()),
        ("package".into(), "cuda-kernels".into()),
        ("cargo_features".into(), cargo_features()),
        (
            "profile".into(),
            std::env::var("PROFILE").unwrap_or_default(),
        ),
        ("target".into(), std::env::var("TARGET").unwrap_or_default()),
        ("host".into(), std::env::var("HOST").unwrap_or_default()),
        ("resolved_sm".into(), resolved_sms(sm_targets)),
        (
            "capabilities".into(),
            capabilities.iter().cloned().collect::<Vec<_>>().join(","),
        ),
        ("inputs_sha256".into(), producer_inputs_sha256()),
        ("toolchain.nvcc".into(), command_id(&nvcc, &["--version"])),
        (
            "toolchain.host_compiler".into(),
            command_id(
                &std::env::var("NVCC_CCBIN").unwrap_or_else(|_| "g++".into()),
                &["--version"],
            ),
        ),
        ("toolchain.ar".into(), command_id("ar", &["--version"])),
        ("toolchain.rustc".into(), command_id("rustc", &["-vV"])),
        (
            "required_symbols".into(),
            required_symbols(capabilities).join(","),
        ),
    ])
}

fn parse_manifest(path: &Path) -> BTreeMap<String, String> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|err| panic!("Failed to read {}: {err}", path.display()));
    let mut fields = BTreeMap::new();
    for (index, line) in text.lines().enumerate() {
        let (key, value) = line.split_once('=').unwrap_or_else(|| {
            panic!(
                "Malformed CUDA prebuilt manifest line {}:{}: {line}",
                path.display(),
                index + 1
            )
        });
        assert!(
            !key.is_empty() && fields.insert(key.to_string(), value.to_string()).is_none(),
            "Duplicate or empty CUDA prebuilt manifest key {}:{}: {key}",
            path.display(),
            index + 1
        );
    }
    fields
}

fn write_producer_manifest(
    out_dir: &Path,
    cuda_path: &str,
    sm_targets: &[SmSpec],
    capabilities: &BTreeSet<String>,
) {
    let required = required_symbols(capabilities);
    validate_archive_symbols(&out_dir.join("libkernels_cuda.a"), &required);
    let mut manifest = producer_contract(cuda_path, sm_targets, capabilities);
    for name in PREBUILT_ARTIFACTS {
        let path = out_dir.join(name);
        if path.is_file() {
            manifest.insert(
                format!("artifact.{name}.size"),
                path.metadata().unwrap().len().to_string(),
            );
            manifest.insert(format!("artifact.{name}.sha256"), sha256_file(&path));
        }
    }
    let identity = manifest
        .iter()
        .map(|(key, value)| format!("{key}={value}\n"))
        .collect::<String>();
    manifest.insert(
        "kernel_build_id".into(),
        format!("bundle:{}", sha256_bytes(identity.as_bytes())),
    );
    let text = manifest
        .iter()
        .map(|(key, value)| format!("{key}={value}\n"))
        .collect::<String>();
    std::fs::write(out_dir.join(PREBUILT_MANIFEST), text).expect("write CUDA producer manifest");
    // ARLE_KERNEL_VENDOR=1: pack tars generated/ into the bundle, and the
    // qualification path requires the producer manifest inside the candidate
    // tree — mirror it and the prebuilt static libs next to the vendored
    // artifact dirs so consumers can link the bundle directly
    // (ARLE_CUDA_KERNELS_PREBUILT_DIR) instead of recompiling every csrc TU.
    if env_truthy("ARLE_KERNEL_VENDOR") {
        let vendor_tier =
            PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("generated");
        std::fs::create_dir_all(&vendor_tier).expect("create vendor tier");
        std::fs::copy(
            out_dir.join(PREBUILT_MANIFEST),
            vendor_tier.join(PREBUILT_MANIFEST),
        )
        .expect("mirror producer manifest into vendor tier");
        for name in PREBUILT_ARTIFACTS {
            let src = out_dir.join(name);
            if src.is_file() {
                std::fs::copy(&src, vendor_tier.join(name))
                    .unwrap_or_else(|err| panic!("vendor export {name}: {err}"));
            }
        }
    }
    emit_kernel_build_identity(out_dir, &manifest);
}

fn verify_prebuilt_manifest(
    prebuilt_dir: &Path,
    cuda_path: &str,
    sm_targets: &[SmSpec],
    capabilities: &BTreeSet<String>,
) -> BTreeMap<String, String> {
    let path = prebuilt_dir.join(PREBUILT_MANIFEST);
    if !path.is_file() {
        panic!(
            "ARLE_CUDA_KERNELS_PREBUILT_DIR={} is missing {PREBUILT_MANIFEST}",
            prebuilt_dir.display()
        );
    }
    println!("cargo:rerun-if-changed={}", path.display());
    let manifest = parse_manifest(&path);
    let expected = producer_contract(cuda_path, sm_targets, capabilities);
    for (key, value) in expected {
        assert_eq!(
            manifest.get(&key),
            Some(&value),
            "CUDA prebuilt producer contract mismatch for {key}; rebuild the bundle"
        );
    }
    for name in PREBUILT_ARTIFACTS {
        let size_key = format!("artifact.{name}.size");
        let hash_key = format!("artifact.{name}.sha256");
        let expected_size = manifest.get(&size_key);
        let expected_hash = manifest.get(&hash_key);
        let path = prebuilt_dir.join(name);
        if *name == "arle_deepep_sidecar" && expected_size.is_none() && expected_hash.is_none() {
            assert!(
                !path.exists(),
                "CUDA prebuilt sidecar exists but is absent from the manifest"
            );
            continue;
        }
        let expected_size =
            expected_size.unwrap_or_else(|| panic!("CUDA prebuilt manifest lacks {size_key}"));
        assert!(
            expected_hash.is_some(),
            "CUDA prebuilt manifest lacks {hash_key}"
        );
        assert!(
            path.is_file(),
            "CUDA prebuilt artifact missing: {}",
            path.display()
        );
        assert_eq!(
            path.metadata().unwrap().len().to_string(),
            *expected_size,
            "CUDA prebuilt artifact size mismatch: {name}"
        );
        assert_eq!(
            manifest.get(&hash_key),
            Some(&sha256_file(&path)),
            "CUDA prebuilt artifact hash mismatch: {name}"
        );
        println!("cargo:rerun-if-changed={}", path.display());
    }
    let symbols = manifest
        .get("required_symbols")
        .map(|value| {
            value
                .split(',')
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    validate_archive_symbols(&prebuilt_dir.join("libkernels_cuda.a"), &symbols);
    manifest
}

fn link_prebuilt_cuda_artifacts(
    prebuilt_dir: &Path,
    cuda_path: &str,
    sm_targets: &[SmSpec],
    capabilities: &BTreeSet<String>,
    out_dir: &Path,
) {
    println!("cargo:rerun-if-changed=build.rs");
    let manifest = verify_prebuilt_manifest(prebuilt_dir, cuda_path, sm_targets, capabilities);
    emit_kernel_build_identity(out_dir, &manifest);
    if capabilities.contains("flashmla") {
        println!("cargo:rustc-cfg=arle_flashmla");
    }
    println!("cargo:rustc-link-search=native={}", prebuilt_dir.display());
    println!("cargo:rustc-link-lib=static=kernels_cuda");
    println!("cargo:rustc-link-lib=static=tilelang_kernels_aot");
    emit_cuda_system_link_libs(cuda_path);
    let sidecar = env_nonempty("ARLE_DEEPEP_SIDECAR_PREBUILT")
        .map(PathBuf::from)
        .unwrap_or_else(|| prebuilt_dir.join("arle_deepep_sidecar"));
    if sidecar.is_file() {
        emit_prebuilt_deepep_sidecar(&sidecar);
    } else if env_nonempty("ARLE_DEEPEP_DIR").is_some() {
        panic!(
            "ARLE_CUDA_KERNELS_PREBUILT_DIR={} is missing arle_deepep_sidecar while ARLE_DEEPEP_DIR is set",
            prebuilt_dir.display()
        );
    }
    println!(
        "cargo:warning=Using verified prebuilt CUDA kernel artifacts from {}; skipping nvcc and TileLang AOT.",
        prebuilt_dir.display()
    );
}

fn hex_digest(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes.iter().fold(
        String::with_capacity(bytes.len() * 2),
        |mut output, byte| {
            write!(output, "{byte:02x}").expect("writing to String is infallible");
            output
        },
    )
}

fn sha256_file(path: &Path) -> String {
    let mut file = File::open(path)
        .unwrap_or_else(|err| panic!("Failed to open {} for hashing: {err}", path.display()));
    let mut digest = ring::digest::Context::new(&ring::digest::SHA256);
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .unwrap_or_else(|err| panic!("Failed to hash {}: {err}", path.display()));
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    hex_digest(digest.finish().as_ref())
}

fn emit_kernel_build_identity(out_dir: &Path, manifest: &BTreeMap<String, String>) {
    let id = manifest
        .get("kernel_build_id")
        .expect("CUDA prebuilt manifest lacks kernel_build_id");
    let capabilities = manifest.get("capabilities").map_or("", String::as_str);
    std::fs::write(
        out_dir.join("kernel_build_identity.rs"),
        format!(
            "pub const KERNEL_BUILD_ID: &str = {id:?};\n\
             pub const KERNEL_CAPABILITIES: &str = {capabilities:?};\n"
        ),
    )
    .expect("write kernel build identity");
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

/// One queued nvcc invocation: the source (for error messages) plus the full
/// pre-built argv. The output path is already baked into `args`.
struct NvccJob {
    cu_file: PathBuf,
    args: Vec<String>,
}

fn run_nvcc_job(nvcc: &str, wrapper: Option<&str>, job: &NvccJob) {
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
fn nvcc_parallelism(jobs: usize) -> usize {
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
fn run_nvcc_jobs(nvcc: &str, wrapper: Option<&str>, jobs: &[NvccJob]) {
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

fn main() {
    // build.rs uses relative paths (tools/tilelang/..., kernels.toml) that are
    // relative to this crate's directory. Cargo runs build scripts from the
    // workspace root, so chdir to CARGO_MANIFEST_DIR first.
    if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        let _ = std::env::set_current_dir(manifest_dir);
    }

    // Compile-capability cfgs, consumed cross-crate via the `cuda_kernels::HAS_*`
    // consts in lib.rs (a `cfg!` set here is not visible in `infer-cuda`).
    // Declared unconditionally — before the no-cuda early-return — so the lib.rs
    // `#[cfg(...)]` arms never trip the unexpected-cfg lint.
    println!("cargo:rustc-check-cfg=cfg(arle_flashmla)");
    if std::env::var("CARGO_FEATURE_METAL").is_ok() {
        println!("cargo:warning=metal feature active: relying on mlx-sys bridge only.");
    }

    // Emit the generated Rust FFI glue (extern block + resolve_* + AttnPhase)
    // UNCONDITIONALLY and EARLY: `src/ffi/attention.rs` `include!`s it from
    // OUT_DIR, so the file must exist before any cuda/metal/no-cuda branch
    // returns (the Mac `cuda,no-cuda` typecheck hits the no-cuda return below).
    // This is pure Rust codegen from kernels.toml — it needs no CUDA toolchain.
    let registry = load_registry();
    let early_out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    emit_ffi_generated(&registry, &early_out_dir);
    emit_flashqla_gdr_generated(&registry, &early_out_dir);
    std::fs::write(
        early_out_dir.join("kernel_build_identity.rs"),
        "pub const KERNEL_BUILD_ID: &str = \"not-built\";\n\
         pub const KERNEL_CAPABILITIES: &str = \"\";\n",
    )
    .expect("write initial kernel build identity");
    println!("cargo:rerun-if-env-changed=ARLE_TILELANG_TOOLCHAIN_PROBE_SELF_TEST");
    if env_truthy("ARLE_TILELANG_TOOLCHAIN_PROBE_SELF_TEST") {
        run_tilelang_toolchain_probe_self_test(&early_out_dir);
        println!("cargo:warning=TileLang toolchain probe self-test passed");
    }
    println!("cargo:rerun-if-env-changed=ARLE_TILELANG_CACHE_ID_SELF_TEST");
    if env_truthy("ARLE_TILELANG_CACHE_ID_SELF_TEST") {
        run_tilelang_cache_identity_self_test(&registry, &early_out_dir);
        println!("cargo:warning=TileLang cache identity self-test passed");
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

    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=ARLE_CUDA_KERNELS_PREBUILT_DIR");
    println!("cargo:rerun-if-env-changed=ARLE_DEEPEP_SIDECAR_PREBUILT");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let sm_targets = detect_sm_targets();
    validate_sm_set(&sm_targets);
    if let Some(prebuilt_dir) = env_nonempty("ARLE_CUDA_KERNELS_PREBUILT_DIR") {
        let capabilities = configured_capabilities(&sm_targets);
        link_prebuilt_cuda_artifacts(
            Path::new(&prebuilt_dir),
            &cuda_path,
            &sm_targets,
            &capabilities,
            &out_dir,
        );
        return;
    }

    println!("cargo:rerun-if-env-changed=ARLE_NVCC_WRAPPER");
    println!("cargo:rerun-if-env-changed=ARLE_NVCC_SPLIT_COMPILE");
    let nvcc_wrapper = env_nonempty("ARLE_NVCC_WRAPPER");
    let nvcc_split_compile = env_nonempty("ARLE_NVCC_SPLIT_COMPILE");

    let nvcc = format!("{}/bin/nvcc", cuda_path);
    let legacy_volta_build = has_legacy_volta(&sm_targets);
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

    // custom_all_reduce.cu is a multi-GPU one-shot collective (sgl-kernel/vLLM
    // lineage) consumed ONLY under the `nccl` feature — every `arle_car_*`
    // reference lives in tp.rs `mod oneshot` (`#[cfg(all(cuda, nccl))]`) or the
    // comm_bench example (`required-features = [cuda, nccl]`). Its bf16 path
    // uses Ampere+ HW intrinsics (custom_all_reduce.cuh gates the bf16 scalar
    // ops to `__CUDA_ARCH__ >= 800`), so a single-GPU sm_70 (V100) release —
    // which never links the collective — would still fail to COMPILE the .cu
    // (generic `upcast<bf16,N>` instantiates with no sm_70 bf16 overload).
    // Skip it without nccl: the archive needs no custom-AR object, so the
    // non-nccl V100/T1/Blackwell release binaries build clean.
    if std::env::var_os("CARGO_FEATURE_NCCL").is_none() {
        cu_files.retain(|p| p.file_name().and_then(|n| n.to_str()) != Some("custom_all_reduce.cu"));
    }
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_NCCL");

    // FlashMLA SM90 sparse prefill — vendored at `vendor/flashmla/` (pin
    // df022ebafb88578eab9f0300606ee765608d8b5c). Add the 5 .cu files needed
    // by ARLE's `arle_flashmla_shim.cu`: fwd.cu + 4 phase1 instantiations.
    // Hopper only (DSv4 target = H20 / SM90a); SM100 sources are skipped.
    // CUTLASS ships inside vendor/flashmla/csrc/cutlass/ (NVIDIA tag
    // 147f5673 — FlashMLA submodule pin). Refs sgl-kernel/cmake/flashmla.cmake.
    println!("cargo:rerun-if-env-changed=ARLE_CUDA_DISABLE_FLASHMLA");
    println!("cargo:rerun-if-env-changed=ARLE_CUDA_DISABLE_FLASHMLA_DECODE");
    let flashmla_root = Path::new("vendor/flashmla");
    let flashmla_stub = Path::new("csrc/attention/arle_flashmla_decode_stubs.cu");
    // FlashMLA is SM90-only sparse-FP8 (prefill + decode). Legacy Volta (sm_70)
    // has no FP8, so the SM90 instantiations fail to compile there — fall back
    // to the cudaErrorNotSupported stub path, same as an explicit opt-out.
    let enable_flashmla =
        flashmla_root.is_dir() && !env_flag("ARLE_CUDA_DISABLE_FLASHMLA") && !legacy_volta_build;
    if enable_flashmla {
        // Runtime FlashMLA gates read `cuda_kernels::HAS_FLASHMLA` (this cfg); a
        // build without the FlashMLA kernels falls back to scalar — no env var.
        println!("cargo:rustc-cfg=arle_flashmla");
    }
    if enable_flashmla && env_flag("ARLE_CUDA_DISABLE_FLASHMLA_DECODE") {
        panic!(
            "ARLE_CUDA_DISABLE_FLASHMLA_DECODE would create a FlashMLA half-state. \
             Disable FlashMLA entirely with ARLE_CUDA_DISABLE_FLASHMLA=1, or build the real decode shim."
        );
    }
    let enable_flashmla_decode = enable_flashmla;
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
            // --dsv4-flashmla-decode, default OFF.
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

    // FA3 hopper fwd + bwd (hdim256/bf16/sm90) — vendored at
    // `vendor/flash-attention/` (Dao-AILab/flash-attention @ fc8cbad6, cutlass
    // pin 71275920). The vendored tree plus an sm_90 target is the whole gate:
    // the instantiation units are nvcc-heavy, but only an sm_90 build compiles
    // them and that is exactly the build that wants FA3.
    let fa3_root = Path::new("vendor/flash-attention");
    let fa3_stub = Path::new("csrc/attention/arle_fa3_stubs.cu");
    let fa3_shim = Path::new("csrc/attention/arle_fa3_shim.cu");
    let enable_fa3 =
        fa3_root.join("hopper").is_dir() && sm_targets.iter().any(|target| target.sm == "90");
    // Exactly one implementation of the FA3 FFI symbols may reach the archive
    // (same single-definition rule as the FlashMLA stub handling above).
    cu_files.retain(|p| p != fa3_stub);
    if !enable_fa3 {
        cu_files.retain(|p| p != fa3_shim);
        cu_files.push(fa3_stub.to_path_buf());
    } else {
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

    println!("cargo:rerun-if-env-changed=NVCC_CCBIN");
    println!("cargo:rerun-if-env-changed=ARLE_CUDA_DISABLE_DEEPGEMM_NATIVE");
    println!("cargo:rerun-if-env-changed=ARLE_DEEPGEMM_ROOT");
    println!("cargo:rerun-if-env-changed=ARLE_DEEPGEMM_CUTLASS_INCLUDE");
    println!("cargo:rerun-if-env-changed=DG_JIT_USE_RUNTIME_API");
    // `enable_deepgemm_native` is computed below (after the vendored paths +
    // sm_targets resolve) so it can DEFAULT-ON via auto-detection.
    // DeepGEMM availability is also a RUNTIME preflight probe (`cuda_kernels::
    // has_deepgemm_native`), not a cfg — the non-native stub exports the same
    // bridge symbols, so it can't be build-determined. No cfg emitted here.
    let deepgemm_root = std::env::var("ARLE_DEEPGEMM_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("vendor/deepgemm"));
    let deepgemm_root = if deepgemm_root.is_absolute() {
        deepgemm_root
    } else {
        std::env::current_dir()
            .expect("failed to resolve cuda-kernels build cwd")
            .join(deepgemm_root)
    };
    let deepgemm_library_root = deepgemm_root.join("deep_gemm");
    let deepgemm_cutlass_include = resolve_deepgemm_cutlass_include(&deepgemm_root);
    let deepgemm_cutlass_include = if deepgemm_cutlass_include.is_absolute() {
        deepgemm_cutlass_include
    } else {
        std::env::current_dir()
            .expect("failed to resolve cuda-kernels build cwd")
            .join(deepgemm_cutlass_include)
    };
    // DeepGEMM FP8-native dense/grouped GEMM: DEFAULT-ON when the build can support it
    // — a Hopper sm_90 target AND the vendored source present — so FP8 prefill takes the
    // fastest path with no manual flag (it was opt-in, leaving production on the ~20×
    // slower dequant→GEMM fallback). Mirrors the FlashMLA auto-detect above. Opt out with
    // ARLE_CUDA_DISABLE_DEEPGEMM_NATIVE=1. If unbuilt/unsupported, the runtime preflight
    // falls back to dequant→GEMM (never GEMV for prefill — see infer-cuda quant_linear).
    let deepgemm_buildable = sm_targets.iter().any(|s| s.sm.starts_with("90"))
        && deepgemm_library_root.is_dir()
        && deepgemm_cutlass_include
            .join("cutlass/arch/barrier.h")
            .is_file();
    let enable_deepgemm_native =
        !env_flag("ARLE_CUDA_DISABLE_DEEPGEMM_NATIVE") && deepgemm_buildable;
    if enable_deepgemm_native {
        println!(
            "cargo:warning=DeepGEMM native enabled (sm_90 + vendored source; set ARLE_CUDA_DISABLE_DEEPGEMM_NATIVE=1 to opt out)"
        );
    }
    // sm_120 grouped blockwise-scaled FP8 MoE GEMM (CUTLASS 4.3.5 collective).
    // Only instantiate the CUTLASS sm_120a collective when the build targets
    // sm_120 — the TU carries `-DARLE_SM120_GROUPED_FP8` + forced
    // compute_120a,sm_120a gencode only then; every other build compiles the
    // cheap `#else` stub (symbol present, returns cudaErrorNotSupported). This
    // is the sm_120 replacement for the Hopper-only DeepGEMM grouped GEMM.
    let sm120_grouped_fp8_target = sm_targets.iter().any(|s| s.sm.starts_with("120"));

    let ccbin = std::env::var("NVCC_CCBIN").ok();
    println!("cargo:rerun-if-env-changed=ARLE_CUDA_DISABLE_MARLIN_W4_FP8");
    let disable_marlin_w4_fp8 = legacy_volta_build
        || matches!(
            std::env::var("ARLE_CUDA_DISABLE_MARLIN_W4_FP8").as_deref(),
            Ok("1" | "true" | "TRUE" | "yes" | "YES")
        );

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
        if disable_marlin_w4_fp8 && stem == "marlin_w4_fp8_kernel" {
            nvcc_args.push("-DARLE_DISABLE_MARLIN_W4_FP8=1".to_string());
        }
        if enable_deepgemm_native {
            nvcc_args.push("-DARLE_ENABLE_DEEPGEMM_NATIVE=1".to_string());
        }
        if let Some(split_compile) = nvcc_split_compile.as_deref() {
            nvcc_args.push(format!("--split-compile={split_compile}"));
        }
        if legacy_volta_build
            && matches!(
                stem,
                "marlin_kernel" | "marlin_repack" | "marlin_w4a8_kernel" | "marlin_gemm"
            )
        {
            nvcc_args.push("-DARLE_DISABLE_MARLIN_SM70=1".to_string());
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
            || matches!(
                stem,
                "arle_flashmla_shim"
                    | "arle_flashmla_decode_shim"
                    | "arle_fa3_shim"
                    | "arle_q8kv8_prefill_shim"
                    | "w4a8_grouped_gemm"
                    | "nvfp4_to_w4afp8"
            );
        // The sm_120 grouped-FP8 TU that also has an sm_120 gencode target: the
        // only build that forces sm_120a gencode + defines ARLE_SM120_GROUPED_FP8.
        let build_sm120_grouped_fp8 =
            stem == "fp8_moe_grouped_cutlass_sm120" && sm120_grouped_fp8_target;
        if is_sm90a_only {
            nvcc_args.push("-gencode=arch=compute_90a,code=sm_90a".to_string());
        } else if build_sm120_grouped_fp8 {
            // CUTLASS sm_120 blockwise MMA needs the accelerated 'a' variant.
            nvcc_args.push("-gencode=arch=compute_120a,code=sm_120a".to_string());
        } else {
            nvcc_args.extend(arch_args.clone());
        }
        nvcc_args.extend(["--compiler-options".to_string(), "-fPIC".to_string()]);
        // Ensure `#include "common.cuh"` resolves from any domain subdir
        // (attention/, gemm/, moe/, kv/, sampling/, recurrent/, ...).
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

        // sm_120 grouped FP8 MoE GEMM: CUTLASS 4.3.5 collective (FlashMLA's
        // vendored cutlass tree). `--expt-relaxed-constexpr` for the collective's
        // device `std::min`; `-lcuda` (device TMA `cuDriverGetVersion`) is already
        // linked via emit_cuda_system_link_libs. Only the sm_120 build defines the
        // macro that instantiates the collective.
        if build_sm120_grouped_fp8 {
            nvcc_args.extend([
                "-std=c++17".to_string(),
                "--expt-relaxed-constexpr".to_string(),
                "-DARLE_SM120_GROUPED_FP8=1".to_string(),
                format!("-I{}/include", cuda_path),
                format!("-I{}", flashmla_root.join("csrc/cutlass/include").display()),
            ]);
        }

        // W4A8 MoE grouped GEMM (SGLang CUTLASS kernel, Apache-2.0): SM90-only.
        // Uses the FlashMLA vendored CUTLASS 4.x — SGLang's mixed-input extensions
        // were written for CUTLASS 4.x; the 3.7.0 backport introduced a TMA
        // descriptor lifecycle bug (-729 crash on first MoE forward).
        if stem == "w4a8_grouped_gemm" {
            nvcc_args.extend([
                "-std=c++17".to_string(),
                "--expt-relaxed-constexpr".to_string(),
                "-Wno-deprecated-declarations".to_string(),
                format!("-I{}/include", cuda_path),
                format!("-I{}", flashmla_root.join("csrc/cutlass/include").display()),
                "-Icsrc/moe/w4a8".to_string(),
            ]);
        }

        // Marlin kernel needs C++17 + relaxed constexpr
        if stem.starts_with("marlin_") {
            nvcc_args.extend([
                "-std=c++17".to_string(),
                "--expt-relaxed-constexpr".to_string(),
            ]);
        }

        // FlashMLA SM90 sparse prefill kernels + ARLE shim. Mirror
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

        // SGLang sparse_mla_q8kv8_prefill_sm90 shim: native FP8 (q8 x kv8)
        // sparse MLA prefill. Reuses FlashMLA's vendored CUTLASS tree for the
        // CuTe/WGMMA headers the kernel pulls in. sm_90a gencode set above.
        if is_sm90a_only && stem == "arle_q8kv8_prefill_shim" {
            nvcc_args.extend([
                "-std=c++17".to_string(),
                "--expt-relaxed-constexpr".to_string(),
                "--expt-extended-lambda".to_string(),
                "--use_fast_math".to_string(),
                "-DNDEBUG".to_string(),
                "-DCUTE_USE_PACKED_TUPLE=1".to_string(),
                "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1".to_string(),
                "-Xcudafe=--diag_suppress=177".to_string(),
                format!("-I{}", flashmla_root.join("csrc/cutlass/include").display()),
                "-Ivendor/q8kv8_prefill".to_string(),
            ]);
        }

        // FA3 hopper units + ARLE shim. Flag set mirrors hopper/setup.py:
        // NDEBUG is upstream-marked "otherwise performance is severely
        // impacted"; EXTENDED_MMA_SHAPES is required for FA3's WGMMA tiles.
        // DISABLE_{LOCAL,APPENDKV} prune template combinations ARLE never
        // dispatches (causal/full only, KV written by ARLE's own prep).
        // The sm_90a gencode is set above (is_sm90a_only).
        if is_sm90a_only && (is_fa3_kernel || stem == "arle_fa3_shim") {
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

    let cuda_lib = out_dir.join("libkernels_cuda.a");
    match std::fs::remove_file(&cuda_lib) {
        Ok(()) => {}
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => panic!("Failed to remove stale {}: {}", cuda_lib.display(), err),
    }
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

    let mut capabilities = BTreeSet::new();
    if enable_flashmla {
        capabilities.insert("flashmla".into());
    }
    if enable_fa3 {
        capabilities.insert("fa3".into());
    }
    // Same condition `compile_tilelang_aot_kernels` gates the flashqla rows on.
    if sm_targets.iter().any(|spec| spec.sm == "90") {
        capabilities.insert("flashqla".into());
    }

    compile_tilelang_aot_kernels(&registry, &cuda_path, &out_dir, &sm_targets);

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=kernels_cuda");
    emit_cuda_system_link_libs(&cuda_path);
    if enable_deepgemm_native {
        println!(
            "cargo:warning=DeepGEMM native bridge enabled, root={}",
            deepgemm_root.display()
        );
    }

    build_deepep_sidecar(
        &cuda_path,
        &nvcc,
        &out_dir,
        &sm_targets,
        nvcc_wrapper.as_deref(),
        nvcc_split_compile.as_deref(),
    );

    if enable_deepgemm_native {
        capabilities.insert("deepgemm-native".into());
    }
    if std::env::var_os("CARGO_FEATURE_NCCL").is_some() {
        capabilities.insert("nccl".into());
    }
    if out_dir.join("arle_deepep_sidecar").is_file() {
        capabilities.insert("deepep-sidecar".into());
    }
    write_producer_manifest(&out_dir, &cuda_path, &sm_targets, &capabilities);

    // Recursive watch — `rerun-if-changed=csrc/` alone only watches the
    // immediate dir entries; subdirectory `.cu`/`.cuh`/`.h` edits would be
    // missed by cargo's incremental detector without this walk.
    println!("cargo:rerun-if-changed=csrc/");
    emit_rerun_recursive(Path::new("csrc"));
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=TORCH_CUDA_ARCH_LIST");
    println!("cargo:rerun-if-env-changed=CMAKE_CUDA_ARCHITECTURES");
    println!("cargo:rerun-if-env-changed=ARLE_DEEPEP_DIR");
}

/// Compile the ARLE DeepEP sidecar binary if `ARLE_DEEPEP_DIR` points at an
/// upstream DeepEP source tree (the deepseek-ai/DeepEP repo). Skipped silently
/// when unset — the sidecar is opt-in until phase 2 default flip.
///
/// Output: `$OUT_DIR/arle_deepep_sidecar`. The path is published via
/// `cargo:rustc-env=ARLE_DEEPEP_SIDECAR_PATH=<path>` for runtime discovery.
fn build_deepep_sidecar(
    cuda_path: &str,
    nvcc: &str,
    out_dir: &Path,
    sm_targets: &[SmSpec],
    nvcc_wrapper: Option<&str>,
    nvcc_split_compile: Option<&str>,
) {
    if let Some(sidecar) = env_nonempty("ARLE_DEEPEP_SIDECAR_PREBUILT") {
        emit_prebuilt_deepep_sidecar(Path::new(&sidecar));
        return;
    }

    let Ok(deepep_dir) = std::env::var("ARLE_DEEPEP_DIR") else {
        println!(
            "cargo:warning=ARLE_DEEPEP_DIR unset — skipping arle_deepep_sidecar build (set to the deepseek-ai/DeepEP source tree to enable native-deepep backend)."
        );
        return;
    };
    let deepep_root = PathBuf::from(&deepep_dir);
    let kernels_root = deepep_root.join("csrc").join("kernels");
    let flat_api_header = kernels_root.join("api.cuh");
    let legacy_kernels_dir = kernels_root.join("legacy");
    let legacy_api_header = legacy_kernels_dir.join("api.cuh");
    let (kernels_dir, compile_units, legacy_layout) = if flat_api_header.exists() {
        (
            kernels_root.clone(),
            vec!["intranode.cu", "layout.cu", "runtime.cu"],
            false,
        )
    } else if legacy_api_header.exists() {
        (legacy_kernels_dir, vec!["intranode.cu", "layout.cu"], true)
    } else {
        println!(
            "cargo:warning=ARLE_DEEPEP_DIR={} does not contain csrc/kernels/api.cuh or csrc/kernels/legacy/api.cuh — skipping sidecar build.",
            deepep_root.display()
        );
        return;
    };
    for unit in &compile_units {
        let path = kernels_dir.join(unit);
        if !path.exists() {
            println!(
                "cargo:warning=ARLE_DEEPEP_DIR={} missing DeepEP sidecar compile unit {} — skipping sidecar build.",
                deepep_root.display(),
                path.display()
            );
            return;
        }
    }

    // The sidecar only supports SM90 (H100/H20) today — that's where the DSv4
    // 32K/1.5K SLO bench lives. Earlier SMs would need TMA-disabled paths.
    let sidecar_archs: Vec<String> = sm_targets
        .iter()
        .filter(|spec| spec.sm == "90")
        .map(|spec| format!("-gencode=arch=compute_{},code=sm_{}", spec.sm, spec.sm))
        .collect();
    if sidecar_archs.is_empty() {
        println!(
            "cargo:warning=SM90 not in current TORCH_CUDA_ARCH_LIST — skipping arle_deepep_sidecar build (sidecar targets H100/H20 only)."
        );
        return;
    }

    let csrc = Path::new("csrc").join("deepep_sidecar");
    let sidecar_main = csrc.join("sidecar_main.cpp");
    let sidecar_bin = out_dir.join("arle_deepep_sidecar");

    let mut cmd = tool_command(nvcc, nvcc_wrapper);
    cmd.arg("-ccbin")
        .arg("g++")
        .arg("-std=c++17")
        .arg("-O2")
        .arg("-DDISABLE_NVSHMEM")
        .arg("--expt-relaxed-constexpr")
        .arg("--expt-extended-lambda")
        .arg("-I")
        .arg(deepep_root.join("csrc"))
        .arg("-I")
        .arg(&csrc);
    let deepep_include = deepep_root.join("deep_ep").join("include");
    if deepep_include.is_dir() {
        cmd.arg("-I").arg(deepep_include);
    }
    if legacy_layout {
        cmd.arg("-DARLE_DEEPEP_LEGACY_LAYOUT=1");
    }
    if let Some(split_compile) = nvcc_split_compile {
        cmd.arg(format!("--split-compile={split_compile}"));
    }
    for a in &sidecar_archs {
        cmd.arg(a);
    }
    for unit in &compile_units {
        cmd.arg(kernels_dir.join(unit));
    }
    cmd.arg(&sidecar_main)
        .arg("-lcudart")
        .arg(format!("-L{}/lib64", cuda_path))
        .arg("-o")
        .arg(&sidecar_bin);

    let status = cmd
        .status()
        .expect("Failed to spawn nvcc for arle_deepep_sidecar");
    if !status.success() {
        panic!(
            "nvcc failed to build arle_deepep_sidecar (status {:?}). Set ARLE_DEEPEP_DIR=<DeepEP source tree> or unset to skip.",
            status.code()
        );
    }

    println!(
        "cargo:warning=arle_deepep_sidecar built at {} (DeepEP={} layout={})",
        sidecar_bin.display(),
        deepep_root.display(),
        if legacy_layout { "legacy" } else { "flat" }
    );
    println!(
        "cargo:rustc-env=ARLE_DEEPEP_SIDECAR_PATH={}",
        sidecar_bin.display()
    );
}
