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
#[cfg(any(
    feature = "metal",
    feature = "cuda",
    feature = "hip",
    feature = "vulkan",
    feature = "cpu"
))]
use std::sync::Arc;

use anyhow::{Context, Result};

use crate::loaded::EngineLoadConfig;
#[cfg(any(
    feature = "metal",
    feature = "cuda",
    feature = "hip",
    feature = "vulkan",
    feature = "cpu"
))]
use crate::loaded::LoadedInferenceEngine;

/// Options for the in-process [`serve_http`] entry.
///
/// Mirrors the arg surface the CLI `serve` subcommand forwards. Fields that the
/// rewrite serve stack does not yet honor (`pool_models`,
/// `spec_type`, `mtp_*`) are carried so the CLI can validate + warn at one site;
/// `serve_http` rejects any that the active backend cannot satisfy.
#[derive(Debug, Clone)]
pub struct ServeHttpOptions {
    pub model_path: String,
    pub bind: String,
    pub port: u16,
    /// Honored by the CUDA decode-graph default only.
    pub enable_cuda_graph: bool,
    pub engine_config: EngineLoadConfig,
    /// Speculative-decode request surface. The rewrite server keeps this
    /// fail-closed until a backend actually consumes it.
    pub spec: ServeSpecOptions,
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
    #[default]
    None,
    Auto,
    Mtp,
    /// DSpark/DFlash block drafter (external draft checkpoint dir via
    /// `--mtp-draft-model`; CUDA Qwen3.5/3.6 only).
    Dspark,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ServeSpecOptions {
    pub spec_type: ServeSpecType,
    pub mtp_draft_model: Option<String>,
    pub dspark_sps_bias_ms: f32,
    pub dspark_sps_row_ms: f32,
    pub mtp_draft_tokens: Option<usize>,
    pub mtp_draft_topk: Option<usize>,
    pub dspark_block_size: Option<usize>,
    /// A saved Markov head to install over the draft checkpoint's at startup.
    pub dspark_markov_init: Option<std::path::PathBuf>,
}

pub const DEFAULT_MTP_DRAFT_TOKENS: usize = 2;

/// Whether the checkpoint ships a multi-token-prediction head, which is what
/// `--spec-type auto` routes on. Both families declare it in `config.json` and
/// neither parses it into its typed config, so read the key directly: Qwen3.5
/// nests under `text_config`, DeepSeek-V4 and GLM use the DeepSeek name (GLM
/// ships 0). A checkpoint that declares no head stays on the plain decode path.
#[must_use]
pub fn checkpoint_has_mtp_head(model_path: &str) -> bool {
    let Ok(raw) = std::fs::read_to_string(std::path::Path::new(model_path).join("config.json"))
    else {
        return false;
    };
    let Ok(cfg) = serde_json::from_str::<serde_json::Value>(&raw) else {
        return false;
    };
    ["mtp_num_hidden_layers", "num_nextn_predict_layers"]
        .iter()
        .any(|key| {
            [cfg.get(key), cfg.get("text_config").and_then(|t| t.get(key))]
                .into_iter()
                .flatten()
                .any(|v| v.as_u64().is_some_and(|n| n > 0))
        })
}
pub const DEFAULT_MTP_DRAFT_TOPK: usize = 1;

impl ServeSpecOptions {
    #[must_use]
    pub fn mtp_enabled(&self) -> bool {
        self.mtp_draft_tokens.is_some() || self.mtp_draft_topk.is_some()
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
#[allow(clippy::type_complexity)]
pub fn serve_http(
    opts: ServeHttpOptions,
    on_engine_loaded: Option<Box<dyn Fn(&Arc<LoadedInferenceEngine>) -> Result<()> + Send + Sync>>,
) -> Result<()> {
    validate_kv_ssd_config(&opts.engine_config)?;

    // Lower the requested spec surface into the engine config. The blanket
    // fail-close is now narrowed: an external draft model and `auto` are still
    // unimplemented and error, but `--spec-type mtp` drives the CUDA DSv4
    // checkpoint-native MTP head via `mtp_draft_tokens`. The per-backend
    // fail-close (MTP is CUDA-only) lives in `router_for_backend`'s `load_*`.
    let mut engine_config = opts.engine_config;
    let spec_type = match opts.spec.spec_type {
        ServeSpecType::None if opts.spec.mtp_enabled() => ServeSpecType::Mtp,
        // `auto` speculates whenever the checkpoint carries the head. Measured on
        // Qwen3.8-27B-NVFP4 at 32K: 20.50 -> 11.94 ms per committed token and
        // +21.6% end-to-end tok/s at c=1, inert above it (the MTP branch is gated
        // to a single decode row), needle ladder exact x3 DET at every length.
        ServeSpecType::Auto => {
            let head = checkpoint_has_mtp_head(&opts.model_path);
            log::info!(
                "--spec-type auto: checkpoint {} an MTP head -> {}",
                if head { "declares" } else { "declares no" },
                if head { "mtp" } else { "no speculation" }
            );
            if head { ServeSpecType::Mtp } else { ServeSpecType::None }
        }
        other => other,
    };
    if opts.spec.mtp_draft_model.is_some() && spec_type != ServeSpecType::Dspark {
        anyhow::bail!(
            "--mtp-draft-model (external draft model) is only consumed by --spec-type dspark; \
             CUDA DSv4 MTP uses the checkpoint-native head"
        );
    }
    match spec_type {
        ServeSpecType::None => {}
        // Resolved above into Mtp or None.
        ServeSpecType::Auto => unreachable!("auto is resolved before this match"),
        ServeSpecType::Mtp => {
            engine_config.mtp_draft_tokens = Some(
                opts.spec
                    .mtp_draft_tokens
                    .unwrap_or(DEFAULT_MTP_DRAFT_TOKENS),
            );
            engine_config.mtp_draft_topk =
                Some(opts.spec.mtp_draft_topk.unwrap_or(DEFAULT_MTP_DRAFT_TOPK));
        }
        ServeSpecType::Dspark => {
            let dir = opts.spec.mtp_draft_model.as_deref().ok_or_else(|| {
                anyhow::anyhow!(
                    "--spec-type dspark requires --mtp-draft-model <DSpark/DFlash checkpoint dir>"
                )
            })?;
            engine_config.dspark_draft_model = Some(std::path::PathBuf::from(dir));
            engine_config.dspark_sps_bias_ms = opts.spec.dspark_sps_bias_ms;
            engine_config.dspark_sps_row_ms = opts.spec.dspark_sps_row_ms;
            engine_config.dspark_block_size = opts.spec.dspark_block_size;
            // `--dspark-markov-init` needs a head slot to install over; a DFlash
            // backbone ships none. Shape only — both halves are overwritten — but
            // the rank comes from the file, so a mismatched head fails at load
            // with its own shape rather than against a guessed one.
            engine_config.markov_head_rank = match &opts.spec.dspark_markov_init {
                Some(path) => Some(spec_train::markov_head::shape(path)?.1),
                None => None,
            };
        }
    }

    let shutdown = infer_server::ServeShutdown::new();
    let (router, engine) = crate::loaded::router_for_backend(
        &opts.model_path,
        opts.enable_cuda_graph,
        engine_config,
        shutdown.clone(),
    )
    .with_context(|| format!("failed to build serve router for {}", opts.model_path))?;

    if let (Some(engine), Some(hook)) = (engine.as_ref(), on_engine_loaded.as_ref()) {
        hook(engine)?;
    }

    bind_and_serve(
        opts.bind.as_str(),
        opts.port,
        router,
        &opts.model_path,
        shutdown,
    )
}

/// Multi-group DP variant: one relay per TP group, routed by a
/// least-in-flight [`infer_server::DpCoordinator`].
#[cfg(feature = "cuda")]
pub fn serve_coordinator_http_dp(
    model_path: &str,
    bind: &str,
    port: u16,
    max_thinking_tokens: usize,
    relays: Vec<infer_server::RelayCoordinator>,
) -> Result<()> {
    let tokenizer = infer_server::OpenAiTokenizer::from_model_dir(model_path)
        .with_context(|| format!("coordinator tokenizer load for {model_path}"))?;
    let model_id = crate::serve_engine::model_id_from_path(model_path);
    let shutdown = infer_server::ServeShutdown::new();
    let router = infer_server::dp_coordinator_router(
        relays,
        tokenizer,
        model_id,
        max_thinking_tokens,
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
pub fn bind_and_serve(
    bind: &str,
    port: u16,
    router: axum::Router,
    label: &str,
    shutdown: infer_server::ServeShutdown,
) -> Result<()> {
    let listener = std::net::TcpListener::bind((bind, port))
        .with_context(|| format!("failed to bind {bind}:{port}"))?;
    serve_listener(listener, router, label, shutdown, true)
}

#[cfg(any(
    feature = "metal",
    feature = "cuda",
    feature = "hip",
    feature = "vulkan",
    feature = "cpu"
))]
fn serve_listener(
    listener: std::net::TcpListener,
    router: axum::Router,
    label: &str,
    shutdown: infer_server::ServeShutdown,
    process_signals: bool,
) -> Result<()> {
    let label = label.to_owned();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("failed to build serve tokio runtime")?;
    runtime.block_on(async move {
        listener
            .set_nonblocking(true)
            .context("failed to set listener non-blocking")?;
        let listener = tokio::net::TcpListener::from_std(listener)
            .context("failed to adopt listener into tokio")?;
        let addr = listener
            .local_addr()
            .context("failed to read listener local address")?;
        log::info!("serving OpenAI v1 on http://{addr} ({label})");
        axum::serve(listener, router)
            .with_graceful_shutdown(shutdown_signal(shutdown, process_signals))
            .await
            .context("serve loop error")
    })
}

#[cfg(any(
    feature = "metal",
    feature = "cuda",
    feature = "hip",
    feature = "vulkan",
    feature = "cpu"
))]
pub struct ServeThread {
    join: std::thread::JoinHandle<Result<()>>,
    shutdown: infer_server::ServeShutdown,
}

#[cfg(any(
    feature = "metal",
    feature = "cuda",
    feature = "hip",
    feature = "vulkan",
    feature = "cpu"
))]
impl ServeThread {
    pub fn shutdown(self) -> Result<()> {
        self.shutdown.request();
        self.join
            .join()
            .map_err(|_| anyhow::anyhow!("serve thread panicked"))?
    }
}

/// Serve an already-built router (e.g. [`crate::LoadedInferenceEngine::local_router`])
/// on a background thread, leaving the caller free to keep driving the engine.
/// Binds in the caller so a port-in-use error surfaces here, not at `shutdown()`.
/// Never touches process signals — the caller owns process lifecycle; the ONLY
/// stop path is the [`ServeThread`] token (so a background server can't swallow
/// the SIGTERM that should kill the whole training process).
#[cfg(any(
    feature = "metal",
    feature = "cuda",
    feature = "hip",
    feature = "vulkan",
    feature = "cpu"
))]
pub fn serve_router_on_thread(router: axum::Router, bind: &str, port: u16) -> Result<ServeThread> {
    let shutdown = infer_server::ServeShutdown::new();
    let listener = std::net::TcpListener::bind((bind, port))
        .with_context(|| format!("failed to bind {bind}:{port}"))?;
    let thread_shutdown = shutdown.clone();
    let join = std::thread::Builder::new()
        .name("arle-http-serve".to_string())
        .spawn(move || serve_listener(listener, router, "local router", thread_shutdown, false))
        .context("spawn arle-http-serve thread")?;
    Ok(ServeThread { join, shutdown })
}

/// Backend-absent build: report the same way `--doctor` does and return an error.
#[cfg(not(any(
    feature = "metal",
    feature = "cuda",
    feature = "hip",
    feature = "vulkan",
    feature = "cpu"
)))]
pub fn serve_http(opts: ServeHttpOptions, _on_engine_loaded: Option<()>) -> Result<()> {
    validate_kv_ssd_config(&opts.engine_config)?;
    anyhow::bail!(
        "serve requires a backend build; rebuild with cuda, metal/no-cuda, vulkan/no-cuda, or cpu/no-cuda"
    )
}

#[cfg(any(
    feature = "metal",
    feature = "cuda",
    feature = "hip",
    feature = "vulkan",
    feature = "cpu"
))]
async fn shutdown_signal(shutdown: infer_server::ServeShutdown, process_signals: bool) {
    // SIGINT or SIGTERM (kill/pod_serve.sh/orchestrators send SIGTERM): without
    // handling SIGTERM the coordinator dies un-gracefully (drop(guard) skipped) → TP
    // workers reaped mid-NCCL-collective → wedged/leaked GPU contexts.
    // `process_signals: false` (background server inside a training process):
    // installing a handler would swallow the default-kill SIGTERM while only the
    // HTTP loop exits — wait exclusively on the programmatic token instead.
    #[cfg(unix)]
    let terminate = async {
        if !process_signals {
            std::future::pending::<()>().await;
        }
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
    let ctrl_c = async {
        if !process_signals {
            std::future::pending::<()>().await;
        }
        let _ = tokio::signal::ctrl_c().await;
    };

    // Programmatic teardown (#135): `ServeShutdown::request()` from the
    // coordinator's fatal lockstep path must also unwind this loop — signals
    // alone would leave the HTTP server (and thus the worker guard) alive.
    let requested = async {
        while !shutdown.is_requested() {
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        }
    };

    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
        _ = requested => {}
    }
    shutdown.request();
    log::info!("shutdown signal received");
}

#[cfg(test)]
mod spec_auto_tests {
    use super::checkpoint_has_mtp_head;

    fn dir_with(config: &str) -> tempfile::TempDir {
        let d = tempfile::tempdir().expect("tempdir");
        std::fs::write(d.path().join("config.json"), config).expect("write config");
        d
    }

    /// A head this returns false for is a silently disabled `--spec-type auto`,
    /// which is why the shapes are pinned rather than trusted.
    #[test]
    fn detects_every_shape_that_ships_a_head() {
        for (label, config) in [
            ("qwen3.5 nested", r#"{"text_config":{"mtp_num_hidden_layers":1}}"#),
            ("qwen top level", r#"{"mtp_num_hidden_layers":1}"#),
            ("deepseek-v4", r#"{"num_nextn_predict_layers":1}"#),
        ] {
            let d = dir_with(config);
            assert!(
                checkpoint_has_mtp_head(d.path().to_str().expect("utf8")),
                "{label} declares a head"
            );
        }
        for (label, config) in [
            ("glm ships zero", r#"{"num_nextn_predict_layers":0}"#),
            ("qwen zero nested", r#"{"text_config":{"mtp_num_hidden_layers":0}}"#),
            ("no key at all", r#"{"num_hidden_layers":64}"#),
        ] {
            let d = dir_with(config);
            assert!(
                !checkpoint_has_mtp_head(d.path().to_str().expect("utf8")),
                "{label} must not speculate"
            );
        }
    }

    #[test]
    fn missing_or_unparsable_config_does_not_speculate() {
        assert!(!checkpoint_has_mtp_head("/nonexistent/checkpoint"));
        let d = dir_with("{ not json");
        assert!(!checkpoint_has_mtp_head(d.path().to_str().expect("utf8")));
    }
}
