//! Metal-backed OpenAI-compatible inference server.
//!
//! All traffic goes through the live `MetalScheduler` runtime with chunked
//! prefill, decode-priority interleaving, variable-length Qwen3.5 packed
//! decode, and DFlash speculative decode (Qwen3, token-buffer pattern).

#![cfg(feature = "metal")]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use clap::{ArgAction, Parser};
use infer::backend::metal::{
    MetalBackendOptions, MetalDflashOptions, MetalKvDiskOptions, MetalRuntimeLimits,
    spawn_metal_scheduler_handle_from_path_with_options_and_metrics,
};
use infer::http_server::{HttpServerConfig, TrainControlTarget, build_app_with_config};
use infer::logging;
use infer::metrics::ServerMetrics;
use infer::request_handle::RequestHandle;
use infer::sampler::SamplingParams;
use infer::scheduler::{IncomingRequest, RequestPriority};
use infer::server_engine::{CompletionStreamDelta, EnginePoolModelSpec};
use log::info;

const DEFAULT_WARMUP_PROMPT: &str = "Write one short sentence about Metal inference.";
const WARMUP_TIMEOUT: Duration = Duration::from_mins(5);

#[derive(Parser)]
#[command(
    name = "metal_serve",
    about = "Metal-backed OpenAI-compatible server (live Metal scheduler; DFlash on scheduler runtime)"
)]
struct Args {
    /// Model directory or HuggingFace model ID.
    #[arg(long)]
    model_path: String,

    /// Port to listen on.
    #[arg(long, default_value_t = 8000)]
    port: u16,

    /// Host or IP address to bind to.
    #[arg(long, default_value = "127.0.0.1")]
    bind: String,

    /// Optional Bearer API key required for `/v1/*` endpoints.
    ///
    /// If omitted, `AGENT_INFER_API_KEY` is used when present.
    #[arg(long)]
    api_key: Option<String>,

    /// Maximum waiting requests before rejecting new submissions.
    /// Single explicit backlog cap for the scheduler runtime.
    /// Keep this bounded so long-prompt throughput sweeps shed load instead of
    /// draining an unbounded queue for minutes after the measurement window.
    #[arg(long, default_value_t = 256)]
    max_waiting: usize,

    /// Enable Metal DFlash with the given draft model path or HuggingFace repo.
    #[arg(long, value_name = "PATH_OR_REPO")]
    dflash_draft_model: Option<String>,

    /// Enable the experimental Metal KV pool for the Qwen3 fallback path.
    #[arg(long, action = ArgAction::SetTrue, conflicts_with = "no_kv_pool")]
    kv_pool: bool,

    /// Disable the experimental Metal KV pool even if the env fallback is set.
    #[arg(long, action = ArgAction::SetTrue, conflicts_with = "kv_pool")]
    no_kv_pool: bool,

    /// Directory for experimental Metal SSD KV cache persistence.
    #[arg(long, value_name = "DIR")]
    kv_disk_dir: Option<PathBuf>,

    /// Maximum bytes for the experimental Metal SSD KV cache.
    #[arg(long, value_name = "BYTES", requires = "kv_disk_dir")]
    kv_disk_max_bytes: Option<u64>,

    /// High watermark for Metal SSD KV cache reclamation.
    #[arg(long, requires = "kv_disk_dir")]
    kv_disk_high_watermark: Option<f64>,

    /// Low watermark for Metal SSD KV cache reclamation.
    #[arg(long, requires = "kv_disk_dir")]
    kv_disk_low_watermark: Option<f64>,

    /// Fsync each experimental Metal SSD KV cache block write.
    #[arg(long, action = ArgAction::SetTrue, requires = "kv_disk_dir")]
    kv_disk_fsync_each_block: bool,

    /// Override the MLX allocator memory limit in bytes before model load.
    #[arg(long, value_name = "BYTES")]
    memory_limit_bytes: Option<usize>,

    /// Override the MLX allocator cache limit in bytes before model load.
    #[arg(long, value_name = "BYTES")]
    cache_limit_bytes: Option<usize>,

    /// Override the MLX allocator wired limit in bytes before model load.
    #[arg(long, value_name = "BYTES")]
    wired_limit_bytes: Option<usize>,

    /// Override the DFlash speculative block size.
    /// Defaults to the draft config; lower values can reduce throughput.
    #[arg(long)]
    speculative_tokens: Option<usize>,

    /// Number of startup warmup requests to run before serving traffic.
    #[arg(long, default_value_t = 1)]
    warmup: usize,

    /// Prompt used for startup warmup requests.
    #[arg(long, default_value = DEFAULT_WARMUP_PROMPT)]
    warmup_prompt: String,

    /// Maximum generated tokens per startup warmup request.
    #[arg(long, default_value_t = 1)]
    warmup_max_new_tokens: usize,

    /// Optional upstream train control-plane URL to expose under `/v1/train/*`.
    #[arg(long)]
    train_control_url: Option<String>,

    /// Additional engine-pool model metadata to expose from `/v1/models`.
    #[arg(long = "pool-model", value_name = "SPEC")]
    pool_models: Vec<String>,
}

impl Args {
    fn kv_pool_override(&self) -> Option<bool> {
        if self.kv_pool {
            Some(true)
        } else if self.no_kv_pool {
            Some(false)
        } else {
            None
        }
    }

    fn runtime_limits(&self) -> MetalRuntimeLimits {
        MetalRuntimeLimits {
            memory_limit_bytes: self.memory_limit_bytes,
            cache_limit_bytes: self.cache_limit_bytes,
            wired_limit_bytes: self.wired_limit_bytes,
        }
    }

    fn kv_disk_options(&self) -> Result<Option<MetalKvDiskOptions>> {
        let Some(dir) = self.kv_disk_dir.clone() else {
            return Ok(None);
        };
        let options = MetalKvDiskOptions {
            dir,
            max_bytes: self.kv_disk_max_bytes,
            high_watermark: self
                .kv_disk_high_watermark
                .unwrap_or(MetalKvDiskOptions::DEFAULT_HIGH_WATERMARK),
            low_watermark: self
                .kv_disk_low_watermark
                .unwrap_or(MetalKvDiskOptions::DEFAULT_LOW_WATERMARK),
            fsync_each_block: self.kv_disk_fsync_each_block,
        };
        options.validate()?;
        Ok(Some(options))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    logging::init_default();

    let args = Args::parse();
    if args.warmup > 0 && args.warmup_prompt.trim().is_empty() {
        bail!("--warmup-prompt must not be empty when --warmup > 0");
    }
    if args.warmup > 0 && args.warmup_max_new_tokens == 0 {
        bail!("--warmup-max-new-tokens must be >= 1 when --warmup > 0");
    }

    let backend_options = MetalBackendOptions {
        dflash: args
            .dflash_draft_model
            .as_ref()
            .map(|draft_model| MetalDflashOptions {
                draft_model: draft_model.clone(),
                speculative_tokens: args.speculative_tokens,
            }),
        kv_pool: args.kv_pool_override(),
        kv_disk: args.kv_disk_options()?,
        runtime_limits: args.runtime_limits(),
    };
    let model_id = std::path::Path::new(&args.model_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(&args.model_path)
        .to_string();
    let metrics = ServerMetrics::new(&model_id);
    // Both DFlash and non-DFlash traffic now goes through the scheduler
    // runtime. DFlash uses the token-buffer pattern inside Qwen3StepDriver
    // (speculative blocks are transparent to the scheduler).
    let handle: Arc<dyn RequestHandle> = Arc::new(
        spawn_metal_scheduler_handle_from_path_with_options_and_metrics(
            &args.model_path,
            backend_options,
            args.max_waiting,
            metrics.clone(),
        )
        .with_context(|| {
            format!(
                "failed to start Metal scheduler runtime for {}",
                args.model_path
            )
        })?,
    );

    if let Some(draft_model) = &args.dflash_draft_model {
        info!(
            "Metal DFlash enabled: draft_model={} speculative_tokens={}",
            draft_model,
            args.speculative_tokens
                .map_or_else(|| "draft-default".to_string(), |value| value.to_string(),)
        );
    }

    let api_key = resolve_api_key(args.api_key.as_deref());
    if api_key.is_some() {
        info!("Metal server API auth enabled for /v1/* endpoints");
    }
    let train_control_target = args
        .train_control_url
        .as_deref()
        .map(TrainControlTarget::parse)
        .transpose()
        .unwrap_or_else(|err| panic!("invalid --train-control-url: {err}"));

    run_startup_warmup(
        &handle,
        args.warmup,
        &args.warmup_prompt,
        args.warmup_max_new_tokens,
    )
    .await?;

    let app = build_app_with_config(
        handle,
        metrics,
        HttpServerConfig {
            api_key: api_key.map(Arc::<str>::from),
            train_control_target,
            pool_models: parse_pool_models(&args.pool_models)?,
        },
    );
    let listener = tokio::net::TcpListener::bind((args.bind.as_str(), args.port))
        .await
        .with_context(|| format!("failed to bind {}:{}", args.bind, args.port))?;
    let addr = listener
        .local_addr()
        .context("failed to read listener local address")?;
    info!("Metal server listening on {} ({})", addr, args.model_path);

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("server error")?;
    Ok(())
}

fn parse_pool_models(raw: &[String]) -> Result<Vec<EnginePoolModelSpec>> {
    raw.iter()
        .map(|spec| {
            EnginePoolModelSpec::parse_cli(spec)
                .map_err(|err| anyhow::anyhow!("invalid --pool-model `{spec}`: {err}"))
        })
        .collect()
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to install CTRL+C handler");
    info!("shutdown signal received");
}

fn resolve_api_key(explicit: Option<&str>) -> Option<String> {
    let candidate = explicit
        .map(ToOwned::to_owned)
        .or_else(|| std::env::var("AGENT_INFER_API_KEY").ok())?;
    let trimmed = candidate.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

async fn run_startup_warmup(
    handle: &dyn RequestHandle,
    runs: usize,
    prompt: &str,
    max_new_tokens: usize,
) -> Result<()> {
    if runs == 0 {
        return Ok(());
    }

    info!(
        "Running {} startup warmup request(s) (prompt_chars={}, max_new_tokens={})",
        runs,
        prompt.chars().count(),
        max_new_tokens
    );

    for run_idx in 0..runs {
        let started = Instant::now();
        let mut delta_rx = submit_warmup_request(handle, prompt, max_new_tokens)
            .with_context(|| format!("failed to submit warmup request {}", run_idx + 1))?;

        let outcome = tokio::time::timeout(WARMUP_TIMEOUT, async move {
            let mut completion: Option<usize> = None;
            let mut prompt_tokens: Option<usize> = None;
            let mut saw_terminal_delta = false;
            while let Some(delta) = delta_rx.recv().await {
                if delta.finish_reason.is_some() {
                    saw_terminal_delta = true;
                }
                if let Some(usage) = delta.usage {
                    prompt_tokens = Some(usage.prompt_tokens);
                    completion = Some(usage.completion_tokens);
                }
            }
            (saw_terminal_delta, prompt_tokens, completion)
        })
        .await;

        match outcome {
            Ok((true, prompt_tokens, completion_tokens)) => {
                info!(
                    "Warmup {}/{} finished in {:.0}ms (prompt_tokens={}, completion_tokens={})",
                    run_idx + 1,
                    runs,
                    started.elapsed().as_secs_f64() * 1000.0,
                    prompt_tokens.unwrap_or(0),
                    completion_tokens.unwrap_or(0)
                );
            }
            Ok((false, _, _)) => {
                bail!(
                    "startup warmup {} failed before the backend emitted a terminal delta",
                    run_idx + 1
                );
            }
            Err(_) => {
                bail!(
                    "startup warmup {} timed out after {}s",
                    run_idx + 1,
                    WARMUP_TIMEOUT.as_secs()
                );
            }
        }
    }

    Ok(())
}

fn submit_warmup_request(
    handle: &dyn RequestHandle,
    prompt: &str,
    max_new_tokens: usize,
) -> Result<tokio::sync::mpsc::UnboundedReceiver<CompletionStreamDelta>> {
    let (delta_tx, delta_rx) = tokio::sync::mpsc::unbounded_channel();
    handle
        .submit(IncomingRequest {
            prompt: prompt.to_string(),
            prompt_tokens: None,
            max_tokens: max_new_tokens,
            sampling: SamplingParams {
                temperature: 0.0,
                top_k: 1,
                ..Default::default()
            },
            stop: None,
            speculative: None,
            priority: RequestPriority::High,
            session_id: None,
            delta_tx,
            trace_context: None,
        })
        .map_err(|_| anyhow::anyhow!("backend warmup queue rejected the request"))?;
    Ok(delta_rx)
}
