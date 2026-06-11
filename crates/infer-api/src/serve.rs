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
//! Backend is selected at compile time (`metal`/`cuda`/`cpu`) and the router is
//! built by [`LoadedInferenceEngine::router_for_backend`], which spawns the same
//! `ServeHandle` the `load_*` constructors spawn. On a build with no backend
//! compiled in, [`serve_http`] returns a clear error (mirrors `--doctor`).

use anyhow::Result;

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
        }
    }
}

/// Build the backend router, bind `bind:port`, and serve OpenAI v1 traffic until
/// Ctrl-C. Blocks the calling (sync) thread on an owned tokio runtime.
///
/// Errors before binding if no backend was compiled in, if the model / tokenizer
/// fails to load, or if the address is already in use.
#[cfg(any(feature = "metal", feature = "cuda", feature = "hip", feature = "cpu"))]
pub fn serve_http(opts: ServeHttpOptions) -> Result<()> {
    use anyhow::Context;

    let router = crate::loaded::router_for_backend(
        &opts.model_path,
        opts.enable_cuda_graph,
        opts.engine_config,
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
            .with_graceful_shutdown(shutdown_signal())
            .await
            .context("serve loop error")
    })
}

/// Backend-absent build: report the same way `--doctor` does and return an error.
#[cfg(not(any(feature = "metal", feature = "cuda", feature = "hip", feature = "cpu")))]
pub fn serve_http(_opts: ServeHttpOptions) -> Result<()> {
    anyhow::bail!(
        "serve requires a backend build; rebuild with cuda, metal/no-cuda, or cpu/no-cuda"
    )
}

#[cfg(any(feature = "metal", feature = "cuda", feature = "hip", feature = "cpu"))]
async fn shutdown_signal() {
    if tokio::signal::ctrl_c().await.is_ok() {
        log::info!("shutdown signal received");
    }
}
