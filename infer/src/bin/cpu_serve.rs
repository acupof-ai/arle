//! CPU-backed OpenAI-compatible development server.
//!
//! This server reuses the serial backend runtime abstraction so API handlers,
//! request submission, and streaming behavior can be exercised on machines
//! without CUDA or Metal.

#![cfg(feature = "cpu")]

use clap::Parser;
use infer::backend::runtime::spawn_cpu_runtime_handle_from_path;
use infer::http_server::{HttpServerConfig, TrainControlTarget, build_app_with_config};
use infer::logging;
use infer::server_engine::EnginePoolModelSpec;
use log::info;

#[derive(Parser)]
#[command(
    name = "cpu_serve",
    about = "CPU development server (serial runtime, smoke-test backend)"
)]
struct Args {
    /// Model directory or HuggingFace model ID.
    #[arg(long)]
    model_path: String,

    /// Port to listen on.
    #[arg(long, default_value_t = 8000)]
    port: u16,

    /// Maximum waiting requests before rejecting new submissions.
    #[arg(long, default_value_t = 256)]
    max_waiting: usize,

    /// Optional upstream train control-plane URL to expose under `/v1/train/*`.
    #[arg(long)]
    train_control_url: Option<String>,

    /// Additional engine-pool model metadata to expose from `/v1/models`.
    #[arg(long = "pool-model", value_name = "SPEC")]
    pool_models: Vec<String>,
}

#[tokio::main]
async fn main() {
    logging::init_default();

    let args = Args::parse();
    let handle = spawn_cpu_runtime_handle_from_path(&args.model_path, args.max_waiting)
        .expect("failed to start CPU runtime");

    let train_control_target = args
        .train_control_url
        .as_deref()
        .map(TrainControlTarget::parse)
        .transpose()
        .unwrap_or_else(|err| panic!("invalid --train-control-url: {err}"));
    let app = build_app_with_config(
        handle,
        infer::metrics::ServerMetrics::new(&args.model_path),
        HttpServerConfig {
            train_control_target,
            pool_models: parse_pool_models(&args.pool_models),
            ..Default::default()
        },
    );
    let addr = format!("0.0.0.0:{}", args.port);
    info!("CPU server listening on {} ({})", addr, args.model_path);

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .unwrap_or_else(|e| panic!("failed to bind {addr}: {e}"));
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect("server error");
}

fn parse_pool_models(raw: &[String]) -> Vec<EnginePoolModelSpec> {
    raw.iter()
        .map(|spec| {
            EnginePoolModelSpec::parse_cli(spec)
                .unwrap_or_else(|err| panic!("invalid --pool-model `{spec}`: {err}"))
        })
        .collect()
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to install CTRL+C handler");
    info!("shutdown signal received");
}
