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
        }
    }
}

/// The L3 (NVMe) KV spill root a bare `--kv-disk` resolves to:
/// `ARLE_KV_SSD_PATH`, else the platform cache dir.
#[must_use]
pub fn default_kv_ssd_root() -> PathBuf {
    if let Some(path) = std::env::var_os("ARLE_KV_SSD_PATH")
        && !path.is_empty()
    {
        return PathBuf::from(path);
    }
    default_cache_root().join("arle").join("kv-ssd")
}

/// Structural validation of the L3 spill request carried by
/// [`EngineLoadConfig`] (`kv_ssd_root` / `kv_disk_limit`): the root must be
/// an absolute, creatable directory, and a limit needs a root. Budget
/// positivity is enforced where the budget resolves (`kv_ssd_spill`); whether
/// the loaded backend/model can actually consume the tier is decided at
/// engine build, which fails closed for arms without a tier store.
pub fn validate_kv_ssd_config(config: &EngineLoadConfig) -> Result<()> {
    let Some(root) = config.kv_ssd_root.as_ref() else {
        anyhow::ensure!(
            config.kv_disk_limit.is_none(),
            "--kv-disk-limit requires --kv-disk"
        );
        return Ok(());
    };
    anyhow::ensure!(!root.as_os_str().is_empty(), "--kv-disk must not be empty");
    anyhow::ensure!(
        root.is_absolute(),
        "--kv-disk must be absolute for serving; got {}",
        root.display()
    );
    std::fs::create_dir_all(root)
        .with_context(|| format!("create --kv-disk {}", root.display()))?;
    let meta =
        std::fs::metadata(root).with_context(|| format!("inspect --kv-disk {}", root.display()))?;
    anyhow::ensure!(
        meta.is_dir(),
        "--kv-disk must be a directory; got {}",
        root.display()
    );
    Ok(())
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
    pub fn mtp_enabled(&self) -> bool {
        self.mtp_draft_tokens.is_some() || self.mtp_draft_topk.is_some()
    }

    #[must_use]
    pub fn requested(&self) -> bool {
        self.spec_type != ServeSpecType::None
            || self.mtp_draft_model.is_some()
            || self.mtp_enabled()
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
pub fn serve_http(opts: ServeHttpOptions) -> Result<()> {
    validate_kv_ssd_config(&opts.engine_config)?;

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
    let spec_type = if opts.spec.spec_type == ServeSpecType::None && opts.spec.mtp_enabled() {
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
        shutdown.clone(),
    )
    .with_context(|| format!("failed to build serve router for {}", opts.model_path))?;

    bind_and_serve(
        opts.bind.as_str(),
        opts.port,
        router,
        &opts.model_path,
        shutdown,
    )
}

/// Serve the multiproc SPMD coordinator HTTP front door (parent process, no TP
/// rank). Serves OpenAI v1 via [`infer_server::coordinator_router`] over the
/// already-accepted `relay`; blocks on an owned tokio runtime until Ctrl-C, like
/// [`serve_http`]. CUDA-only; TP=1 never reaches here.
#[cfg(feature = "cuda")]
pub fn serve_coordinator_http(
    model_path: &str,
    bind: &str,
    port: u16,
    max_thinking_tokens: usize,
    relay: infer_server::RelayCoordinator,
) -> Result<()> {
    let tokenizer = infer_server::OpenAiTokenizer::from_model_dir(model_path)
        .with_context(|| format!("coordinator tokenizer load for {model_path}"))?;
    let model_id = crate::serve_engine::model_id_from_path(model_path);
    let shutdown = infer_server::ServeShutdown::new();
    let router = infer_server::coordinator_router(
        relay,
        tokenizer,
        model_id,
        max_thinking_tokens,
        None,
        Some(shutdown.clone()),
    );
    bind_and_serve(bind, port, router, model_path, shutdown)
}

#[cfg(any(
    feature = "metal",
    feature = "cuda",
    feature = "hip",
    feature = "vulkan",
    feature = "cpu"
))]
fn bind_and_serve(
    bind: &str,
    port: u16,
    router: axum::Router,
    label: &str,
    shutdown: infer_server::ServeShutdown,
) -> Result<()> {
    let bind = bind.to_owned();
    let label = label.to_owned();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("failed to build serve tokio runtime")?;
    runtime.block_on(async move {
        let listener = tokio::net::TcpListener::bind((bind.as_str(), port))
            .await
            .with_context(|| format!("failed to bind {bind}:{port}"))?;
        let addr = listener
            .local_addr()
            .context("failed to read listener local address")?;
        log::info!("serving OpenAI v1 on http://{addr} ({label})");
        axum::serve(listener, router)
            .with_graceful_shutdown(shutdown_signal(shutdown))
            .await
            .context("serve loop error")
    })
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
    validate_kv_ssd_config(&opts.engine_config)?;
    anyhow::bail!(
        "serve requires a backend build; rebuild with cuda, metal/no-cuda, vulkan/no-cuda, or cpu/no-cuda"
    )
}

#[cfg(test)]
mod tests {
    use super::{EngineLoadConfig, validate_kv_ssd_config};
    use infer_seam::KvTierBudget;

    #[test]
    fn kv_disk_limit_without_root_is_rejected() {
        let err = validate_kv_ssd_config(&EngineLoadConfig {
            kv_disk_limit: Some(KvTierBudget::Bytes(1)),
            ..EngineLoadConfig::default()
        })
        .expect_err("limit without a root should fail");
        assert!(err.to_string().contains("requires --kv-disk"));
    }

    #[test]
    fn kv_ssd_rejects_relative_root() {
        let err = validate_kv_ssd_config(&EngineLoadConfig {
            kv_ssd_root: Some("relative/kv".into()),
            ..EngineLoadConfig::default()
        })
        .expect_err("relative root should fail");
        assert!(err.to_string().contains("must be absolute"));
    }

    #[test]
    fn kv_ssd_valid_root_passes_structural_validation() {
        // Backend consumption is gated at engine build (fails closed there);
        // structural validation accepts a valid absolute root.
        let dir = tempfile::tempdir().expect("tempdir");
        validate_kv_ssd_config(&EngineLoadConfig {
            kv_ssd_root: Some(dir.path().to_path_buf()),
            kv_disk_limit: Some(KvTierBudget::Bytes(1 << 30)),
            ..EngineLoadConfig::default()
        })
        .expect("valid absolute root should pass structural validation");
    }

    /// The multiproc coordinator ships its resolved config to worker ranks as
    /// JSON (`ARLE_WORKER_ENGINE_CONFIG`); the KV tier request must survive
    /// that trip, and configs serialized before the fields existed must parse.
    #[test]
    fn kv_ssd_fields_round_trip_worker_config_json() {
        let config = EngineLoadConfig {
            kv_dram: KvTierBudget::Bytes(64 << 30),
            kv_ssd_root: Some(std::path::PathBuf::from("/data/kv-spill")),
            kv_disk_limit: Some(KvTierBudget::Fraction(0.25)),
            ..EngineLoadConfig::default()
        };
        let json = serde_json::to_string(&config).expect("serialize");
        let back: EngineLoadConfig = serde_json::from_str(&json).expect("parse");
        assert_eq!(back.kv_dram, KvTierBudget::Bytes(64 << 30));
        assert_eq!(back.kv_ssd_root, config.kv_ssd_root);
        assert_eq!(back.kv_disk_limit, Some(KvTierBudget::Fraction(0.25)));

        let mut legacy = serde_json::to_value(EngineLoadConfig::default()).expect("to_value");
        let fields = legacy.as_object_mut().expect("object");
        fields.remove("kv_dram");
        fields.remove("kv_ssd_root");
        fields.remove("kv_disk_limit");
        let parsed: EngineLoadConfig =
            serde_json::from_value(legacy).expect("pre-field worker JSON parses");
        assert_eq!(parsed.kv_dram, KvTierBudget::default());
        assert_eq!(parsed.kv_ssd_root, None);
        assert_eq!(parsed.kv_disk_limit, None);
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
    // SIGINT or SIGTERM (kill/pod_serve.sh/orchestrators send SIGTERM): without
    // handling SIGTERM the coordinator dies un-gracefully (drop(guard) skipped) → TP
    // workers reaped mid-NCCL-collective → wedged/leaked GPU contexts.
    #[cfg(unix)]
    let terminate = async {
        use tokio::signal::unix::{SignalKind, signal};
        match signal(SignalKind::terminate()) {
            Ok(mut term) => {
                term.recv().await;
            }
            Err(e) => {
                log::warn!(
                    "SIGTERM handler install failed ({e}); SIGTERM won't shut down gracefully"
                );
                std::future::pending::<()>().await;
            }
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = tokio::signal::ctrl_c() => {}
        _ = terminate => {}
    }
    shutdown.request();
    log::info!("shutdown signal received");
}
