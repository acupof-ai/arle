//! In-process OpenAI v1 HTTP serving entry.
//!
//! [`serve_http`] is the server-START the legacy `infer` / `metal_serve` /
//! `cpu_serve` bins used to own: build the axum router for the compiled backend,
//! bind a `TcpListener`, and drive `axum::serve` to graceful (Ctrl-C) shutdown.
//! Those bins were deleted with the `infer/` crate (PR #53); the runtime now
//! ships only the `arle` binary, so the CLI calls this in-process instead of
//! exec'ing a standalone serve binary.
//!
//! The CLI is synchronous, so this owns its tokio multi-thread runtime rather
//! than relying on `#[tokio::main]` (which is what the deleted bins used).
//!
//! Backend is selected at compile time (`metal`/`cuda`/`hip`/`vulkan`/`cpu`) and the router is
//! built by [`LoadedInferenceEngine::router_for_backend`], which spawns the same
//! `ServeHandle` the `load_*` constructors spawn. On a build with no backend
//! compiled in, [`serve_http`] returns a clear error (mirrors `--doctor`).

use std::path::PathBuf;

use anyhow::{Context, Result};

use crate::loaded::EngineLoadConfig;

/// Options for the in-process [`serve_http`] entry.
///
/// Mirrors the arg surface the CLI `serve` subcommand forwards. Fields that the
/// rewrite serve stack does not yet honor (`train_control_url`, `pool_models`,
/// `spec_type`, `mtp_*`) are carried so the CLI can validate + warn at one site;
/// `serve_http` rejects any that the active backend cannot satisfy.
#[derive(Debug, Clone)]
pub struct ServeHttpOptions {
    /// Model directory or HuggingFace model id.
    pub model_path: String,
    /// Host or IP address to bind.
    pub bind: String,
    /// Port to listen on.
    pub port: u16,
    /// Honored by the CUDA decode-graph default only.
    pub enable_cuda_graph: bool,
    /// Engine slot / page / token-budget configuration.
    pub engine_config: EngineLoadConfig,
    /// Speculative-decode request surface. The rewrite server keeps this
    /// fail-closed until a backend actually consumes it.
    pub spec: ServeSpecOptions,
    /// Optional SSD-backed KV tier request. The option is backend-neutral at the
    /// service boundary; current rewrite backends fail closed until they expose a
    /// real recall path below the executor seam.
    pub kv_ssd: ServeKvSsdOptions,
}

impl ServeHttpOptions {
    /// Construct serve options with the default [`EngineLoadConfig`].
    #[must_use]
    pub fn new(model_path: impl Into<String>, bind: impl Into<String>, port: u16) -> Self {
        Self {
            model_path: model_path.into(),
            bind: bind.into(),
            port,
            enable_cuda_graph: true,
            engine_config: EngineLoadConfig::default(),
            spec: ServeSpecOptions::default(),
            kv_ssd: ServeKvSsdOptions::default(),
        }
    }
}

/// SSD-backed KV tier request carried at the serve boundary.
#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub struct ServeKvSsdOptions {
    /// Root directory for local SSD/NVMe KV blocks.
    pub root: Option<PathBuf>,
    /// Optional capacity guard for this serve process.
    pub max_bytes: Option<usize>,
    /// The only intended SSD serving mode in the rewrite stack: synchronous,
    /// high-throughput promotion without background preemption.
    pub high_performance_non_preemptive: bool,
}

impl ServeKvSsdOptions {
    #[must_use]
    pub fn requested(&self) -> bool {
        self.root.is_some() || self.max_bytes.is_some() || self.high_performance_non_preemptive
    }

    #[must_use]
    pub fn default_root() -> PathBuf {
        if let Some(path) = std::env::var_os("ARLE_KV_SSD_PATH")
            && !path.is_empty()
        {
            return PathBuf::from(path);
        }
        default_cache_root().join("arle").join("kv-ssd")
    }

    pub fn fill_default_root(&mut self) {
        if self.requested() && self.root.is_none() {
            self.root = Some(Self::default_root());
            self.high_performance_non_preemptive = true;
        }
    }
}

fn default_cache_root() -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        if let Some(path) = std::env::var_os("LOCALAPPDATA") {
            return PathBuf::from(path);
        }
        if let Some(path) = std::env::var_os("USERPROFILE") {
            return PathBuf::from(path).join("AppData").join("Local");
        }
    }

    #[cfg(target_os = "macos")]
    {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join("Library").join("Caches");
        }
    }

    #[cfg(not(target_os = "windows"))]
    {
        if let Some(path) = std::env::var_os("XDG_CACHE_HOME")
            && !path.is_empty()
        {
            return PathBuf::from(path);
        }
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join(".cache");
        }
    }

    std::env::temp_dir()
}

/// Speculative decode mode requested at the serve boundary.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub enum ServeSpecType {
    /// Standard target-only decode.
    #[default]
    None,
    /// Backend chooses the available speculative route.
    Auto,
    /// Multi-token prediction / MTP route.
    Mtp,
}

impl ServeSpecType {
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Auto => "auto",
            Self::Mtp => "mtp",
        }
    }
}

/// Speculative decode options carried by [`ServeHttpOptions`].
#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub struct ServeSpecOptions {
    pub spec_type: ServeSpecType,
    pub mtp_draft_model: Option<String>,
    pub mtp_draft_tokens: Option<usize>,
    pub mtp_draft_topk: Option<usize>,
}

pub const DEFAULT_MTP_DRAFT_TOKENS: usize = 2;
pub const DEFAULT_MTP_DRAFT_TOPK: usize = 1;

impl ServeSpecOptions {
    #[must_use]
    pub fn requested(&self) -> bool {
        self.spec_type != ServeSpecType::None
            || self.mtp_draft_model.is_some()
            || self.mtp_draft_tokens.is_some()
            || self.mtp_draft_topk.is_some()
    }
}

/// Build the backend router, bind `bind:port`, and serve OpenAI v1 traffic until
/// Ctrl-C. Blocks the calling (sync) thread on an owned tokio runtime.
///
/// Errors before binding if no backend was compiled in, if the model / tokenizer
/// fails to load, or if the address is already in use.
#[cfg(any(
    feature = "metal",
    feature = "cuda",
    feature = "hip",
    feature = "vulkan",
    feature = "cpu"
))]
pub fn serve_http(mut opts: ServeHttpOptions) -> Result<()> {
    opts.kv_ssd.fill_default_root();
    validate_kv_ssd_options(&opts.kv_ssd)?;

    // Lower the requested spec surface into the engine config. The blanket
    // fail-close is now narrowed: an external draft model and `auto` are still
    // unimplemented and error, but `--spec-type mtp` drives the CUDA DSv4
    // checkpoint-native MTP head via `mtp_draft_tokens`. The per-backend
    // fail-close (MTP is CUDA-only) lives in `router_for_backend`'s `load_*`.
    let mut engine_config = opts.engine_config;
    if opts.spec.mtp_draft_model.is_some() {
        anyhow::bail!(
            "--mtp-draft-model (external draft model) is not supported on this serve path; \
             CUDA DSv4 uses the checkpoint-native MTP head"
        );
    }
    let spec_type = if opts.spec.spec_type == ServeSpecType::None
        && (opts.spec.mtp_draft_tokens.is_some() || opts.spec.mtp_draft_topk.is_some())
    {
        ServeSpecType::Mtp
    } else {
        opts.spec.spec_type
    };
    match spec_type {
        ServeSpecType::None => {}
        ServeSpecType::Auto => {
            anyhow::bail!("--spec-type auto is not implemented; use mtp");
        }
        ServeSpecType::Mtp => {
            engine_config.mtp_draft_tokens = Some(
                opts.spec
                    .mtp_draft_tokens
                    .unwrap_or(DEFAULT_MTP_DRAFT_TOKENS),
            );
            engine_config.mtp_draft_topk =
                Some(opts.spec.mtp_draft_topk.unwrap_or(DEFAULT_MTP_DRAFT_TOPK));
        }
    }

    let shutdown = infer_server::ServeShutdown::new();
    let router = crate::loaded::router_for_backend(
        &opts.model_path,
        opts.enable_cuda_graph,
        engine_config,
        &opts.kv_ssd,
        shutdown.clone(),
    )
    .with_context(|| format!("failed to build serve router for {}", opts.model_path))?;

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("failed to build serve tokio runtime")?;

    runtime.block_on(async move {
        let listener = tokio::net::TcpListener::bind((opts.bind.as_str(), opts.port))
            .await
            .with_context(|| format!("failed to bind {}:{}", opts.bind, opts.port))?;
        let addr = listener
            .local_addr()
            .context("failed to read listener local address")?;
        log::info!("serving OpenAI v1 on http://{} ({})", addr, opts.model_path);
        axum::serve(listener, router)
            .with_graceful_shutdown(shutdown_signal(shutdown))
            .await
            .context("serve loop error")
    })
}

fn validate_kv_ssd_options(opts: &ServeKvSsdOptions) -> Result<()> {
    if !opts.requested() {
        return Ok(());
    }

    let root = opts
        .root
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("KV SSD tier requested without a resolved root"))?;
    anyhow::ensure!(
        !root.as_os_str().is_empty(),
        "--kv-ssd-path must not be empty"
    );
    anyhow::ensure!(
        root.is_absolute(),
        "--kv-ssd-path must be absolute for serving; got {}",
        root.display()
    );
    std::fs::create_dir_all(root)
        .with_context(|| format!("create --kv-ssd-path {}", root.display()))?;
    let meta = std::fs::metadata(root)
        .with_context(|| format!("inspect --kv-ssd-path {}", root.display()))?;
    anyhow::ensure!(
        meta.is_dir(),
        "--kv-ssd-path must be a directory; got {}",
        root.display()
    );
    anyhow::ensure!(
        opts.max_bytes.is_none_or(|value| value > 0),
        "--kv-ssd-max-bytes must be positive"
    );
    anyhow::ensure!(
        opts.high_performance_non_preemptive,
        "--kv-ssd-path currently only supports the high-performance non-preemptive mode"
    );

    // Structural validation only — whether the loaded backend/model can
    // actually consume the T2 tier is decided at engine build
    // (`router_for_backend` / `cuda_serve_handle`), which fails closed for
    // arms without a page-addressable tier store.
    Ok(())
}

/// Backend-absent build: report the same way `--doctor` does and return an error.
#[cfg(not(any(
    feature = "metal",
    feature = "cuda",
    feature = "hip",
    feature = "vulkan",
    feature = "cpu"
)))]
pub fn serve_http(opts: ServeHttpOptions) -> Result<()> {
    validate_kv_ssd_options(&opts.kv_ssd)?;
    anyhow::bail!(
        "serve requires a backend build; rebuild with cuda, metal/no-cuda, vulkan/no-cuda, or cpu/no-cuda"
    )
}

#[cfg(test)]
mod tests {
    use super::{ServeKvSsdOptions, validate_kv_ssd_options};

    #[test]
    fn kv_ssd_capacity_without_root_uses_default_root() {
        let mut opts = ServeKvSsdOptions {
            root: None,
            max_bytes: Some(1),
            high_performance_non_preemptive: true,
        };
        opts.fill_default_root();

        assert_eq!(opts.root, Some(ServeKvSsdOptions::default_root()));
        validate_kv_ssd_options(&opts).expect("default root should pass structural validation");
    }

    #[test]
    fn kv_ssd_rejects_relative_root() {
        let err = validate_kv_ssd_options(&ServeKvSsdOptions {
            root: Some("relative/kv".into()),
            max_bytes: None,
            high_performance_non_preemptive: true,
        })
        .expect_err("relative root should fail");

        assert!(err.to_string().contains("must be absolute"));
    }

    #[test]
    fn kv_ssd_valid_root_passes_structural_validation() {
        // Backend consumption is gated at engine build (CUDA-only today,
        // fails closed there); structural validation accepts a valid root.
        let dir = tempfile::tempdir().expect("tempdir");
        validate_kv_ssd_options(&ServeKvSsdOptions {
            root: Some(dir.path().to_path_buf()),
            max_bytes: Some(1 << 30),
            high_performance_non_preemptive: true,
        })
        .expect("valid absolute root should pass structural validation");
    }
}

#[cfg(any(
    feature = "metal",
    feature = "cuda",
    feature = "hip",
    feature = "vulkan",
    feature = "cpu"
))]
async fn shutdown_signal(shutdown: infer_server::ServeShutdown) {
    if tokio::signal::ctrl_c().await.is_ok() {
        shutdown.request();
        log::info!("shutdown signal received");
    }
}
