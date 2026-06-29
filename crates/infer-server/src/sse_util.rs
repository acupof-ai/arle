//! Shared SSE formatting helpers used by both the in-process and coordinator paths.
use std::time::{SystemTime, UNIX_EPOCH};

use infer_plan::FinishReason;
use serde_json::json;

pub(crate) fn completion_stream_chunk(
    id: &str,
    created: u64,
    model: &str,
    text: String,
    finish: Option<&str>,
    usage: Option<serde_json::Value>,
) -> serde_json::Value {
    json!({
        "id": id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{"text": text, "index": 0, "logprobs": null, "finish_reason": finish}],
        "usage": usage,
    })
}

pub(crate) fn finish_reason(reason: Option<&FinishReason>) -> &'static str {
    match reason {
        Some(FinishReason::Stop) => "stop",
        Some(FinishReason::Length) | None => "length",
        Some(FinishReason::Abort) => "abort",
    }
}

pub(crate) fn unix_time_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}
