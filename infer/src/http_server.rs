//! HTTP server coordinator.
//!
//! This file used to hold the entire OpenAI-compatible HTTP surface
//! (~2979 LOC). It was split into responsibility-sized submodules in
//! 2026-04-27 (pure structural refactor — no behavior change). The
//! submodules are:
//!
//! - `types` — `HttpServerConfig`, `TrainControlTarget`, `AppState`,
//!   `ServingIdentity`, request/response containers, byte-limit + timeout
//!   constants.
//! - `handlers` — request handlers (`completions`, `chat_completions`,
//!   `responses_handler`, `models_handler`, health/ready/stats/metrics,
//!   train_* proxies), JSON/route helpers, SSE streaming machinery,
//!   the `attach_request_id` middleware, and the train control TCP proxy.
//! - `router` — the `build_app*` family that wires handlers + middleware.
//! - `tests` — end-to-end Axum tests for every route.
//!
//! Pre-existing siblings `openai_v1` (request/response DTOs) and
//! `sessions` (the session subtree router) remain unchanged.

#[allow(clippy::struct_field_names, clippy::needless_pass_by_value)]
mod openai_v1;
pub mod sessions;

#[path = "http_server/types.rs"]
pub(in crate::http_server) mod types;

#[path = "http_server/handlers.rs"]
pub(in crate::http_server) mod handlers;

#[path = "http_server/router.rs"]
pub(in crate::http_server) mod router;

#[path = "http_server/tests.rs"]
mod tests_mod;

// Public surface preserved verbatim for `main.rs`, `bin/metal_serve.rs`,
// `bin/cpu_serve.rs`, and `session_persistence.rs`.
pub use router::{
    build_app, build_app_with_config, build_app_with_config_and_session_engine,
    build_app_with_metrics, build_app_with_session_engine,
};
pub use types::{HttpServerConfig, TrainControlTarget};

// `sessions::session_router` reads `super::HTTP_REQUEST_BODY_LIMIT_BYTES`;
// keep that path resolvable from the coordinator.
pub(in crate::http_server) use types::HTTP_REQUEST_BODY_LIMIT_BYTES;
