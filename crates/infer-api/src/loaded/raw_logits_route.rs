//! `POST /v1/raw_logits` — the OPD API-teacher scoring surface.
//!
//! An OPD run distilling into an FP8 student can't hold the full FP8 teacher on
//! the same card, so the teacher runs as a separate `arle serve` and the trainer
//! reaches it over HTTP (`--teacher-runtime api`). The request carries the exact
//! token/position sequence; the response is the full `[seq, vocab]` teacher
//! logits, bf16-encoded (the teacher is bf16 anyway, so it's lossless and halves
//! the wire vs f32). The client decoder is `train::teacher_infer::ApiTeacher`.
//!
//! CUDA-only, and it lives here (infer-api) rather than infer-server's
//! device-neutral coordinator: `forward_token_logits` is a CUDA-typed engine
//! method, so exposing it keeps the backend type out of the neutral router.

use std::sync::Arc;

use axum::{Json, Router, extract::State, http::StatusCode, response::IntoResponse, routing::post};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use half::bf16;
use serde::{Deserialize, Serialize};

use crate::loaded::LoadedInferenceEngine;

#[derive(Deserialize)]
struct RawLogitsRequest {
    input_ids: Vec<u32>,
    positions: Vec<u32>,
    // Client's requested dtype. We always answer bf16 (teacher is bf16; lossless,
    // half the wire), so this is accepted-and-ignored — kept so an f32 request
    // deserializes rather than 400s.
    #[serde(default)]
    #[allow(dead_code)]
    dtype: Option<String>,
}

#[derive(Serialize)]
struct RawLogitsResponse {
    shape: [usize; 2],
    dtype: &'static str,
    logits_b64: String,
}

/// A raw-logits sub-router bound to the CUDA engine, ready to `.merge()` into the
/// serve router. Keeps `forward_token_logits` (CUDA-typed) out of infer-server.
pub(crate) fn raw_logits_router(engine: Arc<LoadedInferenceEngine>) -> Router {
    Router::new()
        .route("/v1/raw_logits", post(handle_raw_logits))
        .with_state(engine)
}

async fn handle_raw_logits(
    State(engine): State<Arc<LoadedInferenceEngine>>,
    Json(req): Json<RawLogitsRequest>,
) -> impl IntoResponse {
    if req.input_ids.is_empty() || req.input_ids.len() != req.positions.len() {
        return (
            StatusCode::BAD_REQUEST,
            format!(
                "input_ids/positions must be non-empty and equal length: ids={} pos={}",
                req.input_ids.len(),
                req.positions.len()
            ),
        )
            .into_response();
    }
    // The forward is a blocking CUDA call; keep it off the async worker.
    let result = tokio::task::block_in_place(|| {
        let raw = engine.forward_token_logits(&req.input_ids, &req.positions)?;
        let shape = [raw.seq_len(), raw.vocab_size()];
        let host = raw.to_host_f32()?;
        anyhow::Ok((shape, host))
    });
    match result {
        Ok((shape, host)) => {
            // bf16-LE bytes -> base64. ApiTeacher decodes exactly this (dtype=bf16).
            let mut bytes = Vec::with_capacity(host.len() * 2);
            for v in &host {
                bytes.extend_from_slice(&bf16::from_f32(*v).to_bits().to_le_bytes());
            }
            Json(RawLogitsResponse {
                shape,
                dtype: "bf16",
                logits_b64: STANDARD.encode(&bytes),
            })
            .into_response()
        }
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("raw-logits forward failed: {err}"),
        )
            .into_response(),
    }
}
