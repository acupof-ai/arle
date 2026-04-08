#![cfg(feature = "metal")]

use clap::Parser;
use infer::backend_runtime::spawn_metal_runtime_handle_from_path;
use infer::http_server::build_app;
use infer::logging;
use log::info;

#[derive(Parser)]
#[command(name = "metal_serve", about = "Metal-backed OpenAI-compatible server")]
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
}

#[tokio::main]
async fn main() {
    logging::init_default();

    let args = Args::parse();
    let handle = spawn_metal_runtime_handle_from_path(&args.model_path, args.max_waiting)
        .expect("failed to start Metal runtime");

    let app = build_app(handle);
    let addr = format!("0.0.0.0:{}", args.port);
    info!("Metal server listening on {} ({})", addr, args.model_path);

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .unwrap_or_else(|e| panic!("failed to bind {addr}: {e}"));
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect("server error");
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to install CTRL+C handler");
    info!("shutdown signal received");
}
