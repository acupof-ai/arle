//! `build_app*` constructors that wire handlers, middleware, and routes.
//!
//! Split out of `http_server.rs` (pure structural refactor — no behavior change).

use std::sync::Arc;

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::middleware;
use axum::routing::{get, post};

use super::handlers::{
    attach_request_id, chat_completions, completions, healthz_handler, method_not_allowed_handler,
    metrics_handler, models_handler, readyz_handler, responses_handler, route_not_found_handler,
    stats_handler, train_events_handler, train_save_handler, train_status_handler,
    train_stop_handler,
};
use super::types::{AppState, HTTP_REQUEST_BODY_LIMIT_BYTES, HttpServerConfig, ServingIdentity};
use crate::metrics::ServerMetrics;
use crate::request_handle::RequestHandle;

/// Build the Axum router with default (empty) metrics.
pub fn build_app<H>(handle: H) -> Router
where
    H: RequestHandle + 'static,
{
    build_app_inner(handle, ServerMetrics::new(""), HttpServerConfig::default())
}

/// Build the Axum router with a pre-configured `ServerMetrics` instance.
pub fn build_app_with_metrics<H>(handle: H, metrics: ServerMetrics) -> Router
where
    H: RequestHandle + 'static,
{
    build_app_inner(handle, metrics, HttpServerConfig::default())
}

/// Build the Axum router with explicit metrics and server configuration.
pub fn build_app_with_config<H>(
    handle: H,
    metrics: ServerMetrics,
    config: HttpServerConfig,
) -> Router
where
    H: RequestHandle + 'static,
{
    build_app_inner(handle, metrics, config)
}

fn build_app_inner<H>(handle: H, metrics: ServerMetrics, config: HttpServerConfig) -> Router
where
    H: RequestHandle + 'static,
{
    let tokenizer = handle.tokenizer_clone();
    let identity = ServingIdentity {
        model_id: handle.model_id().to_string(),
        dflash_status: handle.dflash_status(),
    };
    let state = Arc::new(AppState {
        handle: Arc::new(handle),
        tokenizer,
        identity,
        metrics,
        config,
    });

    Router::new()
        .route("/healthz", get(healthz_handler))
        .route("/readyz", get(readyz_handler))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/responses", post(responses_handler))
        .route("/v1/models", get(models_handler))
        .route("/v1/train/status", get(train_status_handler))
        .route("/v1/train/events", get(train_events_handler))
        .route("/v1/train/stop", post(train_stop_handler))
        .route("/v1/train/save", post(train_save_handler))
        .route("/metrics", get(metrics_handler))
        .route("/v1/stats", get(stats_handler))
        .method_not_allowed_fallback(method_not_allowed_handler)
        .fallback(route_not_found_handler)
        .layer(DefaultBodyLimit::max(HTTP_REQUEST_BODY_LIMIT_BYTES))
        .layer(middleware::from_fn(attach_request_id))
        .with_state(state)
}
