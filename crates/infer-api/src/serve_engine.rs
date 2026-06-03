//! Generic adapter wrapping a rewrite-stack [`ServeHandle`] + tokenizer as an
//! [`InferenceEngine`].
//!
//! This is the byte-for-byte bridge described in the cutover map:
//! `tokenize -> submit -> collect -> detokenize`, projecting the rewrite
//! `CompletedRequest` back into the legacy [`CompletionOutput`] shape.

use anyhow::{Result, anyhow};
use infer_core::CompletedRequest;
use infer_seam::{BackendExecutor, KvPool};
use infer_server::{OpenAiTokenizer, ServeHandle};
use tokio::sync::mpsc::UnboundedSender;

use crate::types::{
    CompletionOutput, CompletionRequest, CompletionStreamDelta, EngineTelemetry, FinishReason,
    InferenceEngine, TokenUsage,
};

/// Adapter over one running [`ServeHandle`] + its tokenizer.
///
/// Generic over the rewrite executor / KV pool so the same wiring serves Metal,
/// CUDA, and the CPU placeholder. Construct a backend-specific instance through
/// [`crate::LoadedInferenceEngine::load`]; this type is the shared body behind
/// every variant's [`InferenceEngine`] impl.
pub struct ServeInferenceEngine<E: BackendExecutor, K: KvPool> {
    model_id: String,
    tokenizer: OpenAiTokenizer,
    serve: ServeHandle<E, K>,
}

impl<E, K> ServeInferenceEngine<E, K>
where
    E: BackendExecutor + 'static,
    K: KvPool + 'static,
{
    /// Adopt a spawned [`ServeHandle`] plus the matching tokenizer.
    #[must_use]
    pub fn new(model_id: String, tokenizer: OpenAiTokenizer, serve: ServeHandle<E, K>) -> Self {
        Self {
            model_id,
            tokenizer,
            serve,
        }
    }

    /// Shared `tokenize -> submit -> collect -> detokenize` body.
    ///
    /// Returns the projected [`CompletionOutput`]; both `complete` and
    /// `complete_stream` build on it (the rewrite `ServeHandle` exposes blocking
    /// `collect` only, so `complete_stream` emits a single terminal delta — see
    /// the streaming gap in the crate docs).
    fn run(&self, req: &CompletionRequest) -> Result<CompletionOutput> {
        let prompt_token_ids = self
            .tokenizer
            .encode(&req.prompt)
            .map_err(|err| anyhow!("tokenize prompt failed: {err}"))?;

        let ticket = self
            .serve
            .submit(
                prompt_token_ids.clone(),
                req.max_tokens,
                req.sampling.clone(),
            )
            .map_err(|err| anyhow!("request submission failed: {err}"))?;
        let completed: CompletedRequest = ticket.collect()?;

        let response_token_ids = completed.generated_tokens.clone();
        let raw_text = self
            .tokenizer
            .decode(&response_token_ids)
            .map_err(|err| anyhow!("decode generated tokens failed: {err}"))?;

        // Host-side stop-string truncation (the rewrite engine stops on token
        // ids, not strings; OpenAI `stop` strings are applied here, matching the
        // legacy backend's `truncate_at_first_stop`).
        let (text, stop_truncated) = match req.stop.as_deref() {
            Some(stops) => match truncate_at_first_stop(&raw_text, stops) {
                Some(truncated) => (truncated, true),
                None => (raw_text, false),
            },
            None => (raw_text, false),
        };

        let finish_reason = if stop_truncated {
            FinishReason::Stop
        } else {
            completed
                .finish
                .as_ref()
                .map_or(FinishReason::Length, FinishReason::from_plan)
        };

        let prompt_tokens = completed.prompt_tokens.len();
        let completion_tokens = response_token_ids.len();
        Ok(CompletionOutput {
            text,
            finish_reason,
            usage: TokenUsage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
            // The rewrite stack does not surface per-token logprobs yet.
            token_logprobs: Vec::new(),
            prompt_token_ids,
            response_token_ids,
        })
    }
}

// The `InferenceEngine: Send` supertrait is satisfied via the auto-trait:
// `ServeInferenceEngine` only stores a `ServeHandle<E, K>` (a `Send` channel +
// `JoinHandle` + `PhantomData<fn() -> (E, K)>`), never an `E`/`K` value, so it
// is `Send` even for a `!Send` executor like the MLX-backed `MetalExecutor`
// (the executor lives on the engine thread and never crosses the seam).
impl<E, K> InferenceEngine for ServeInferenceEngine<E, K>
where
    E: BackendExecutor + 'static,
    K: KvPool + 'static,
{
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn complete(&mut self, req: CompletionRequest) -> Result<CompletionOutput> {
        self.run(&req)
    }

    fn complete_stream(
        &mut self,
        req: CompletionRequest,
        tx: UnboundedSender<CompletionStreamDelta>,
    ) -> Result<()> {
        // GAP (documented): the rewrite `ServeHandle` exposes blocking `collect`
        // only — no per-token streaming hook. The adapter runs the request to
        // completion, then emits the full text plus a terminal finish delta so
        // streaming consumers receive a correct (if not incremental) result.
        // True token-granular streaming requires a `ServeHandle` streaming hook
        // (follow-up).
        match self.run(&req) {
            Ok(output) => {
                if !output.text.is_empty() {
                    let _ = tx.send(CompletionStreamDelta {
                        text_delta: output.text,
                        finish_reason: None,
                        usage: None,
                        logprob: None,
                        token_ids: output.response_token_ids,
                        error: None,
                    });
                }
                let _ = tx.send(CompletionStreamDelta {
                    text_delta: String::new(),
                    finish_reason: Some(output.finish_reason),
                    usage: Some(output.usage),
                    logprob: None,
                    token_ids: Vec::new(),
                    error: None,
                });
                Ok(())
            }
            Err(err) => {
                let _ = tx.send(CompletionStreamDelta::error(
                    "inference_failed",
                    vec![err.to_string()],
                ));
                Ok(())
            }
        }
    }

    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        self.tokenizer
            .encode(text)
            .map_err(|err| anyhow!("tokenize failed: {err}"))
    }

    fn telemetry(&self) -> EngineTelemetry {
        // GAP (documented): the rewrite `ServeHandle` does not surface scheduler
        // counters (queue depth / occupancy / TTFT / ITL) across its thread
        // channel today. Returns the empty shape; callers treat empty as
        // "unavailable", never zero.
        EngineTelemetry::default()
    }
}

/// Truncate at the first occurrence of any non-empty stop string.
///
/// Returns the prefix of `text` up to (but not including) the earliest stop, or
/// `None` if no stop matched. Mirrors the legacy
/// `infer::server_engine::stream::truncate_at_first_stop`.
fn truncate_at_first_stop(text: &str, stops: &[String]) -> Option<String> {
    let mut earliest = None::<usize>;
    for stop in stops {
        if stop.is_empty() {
            continue;
        }
        if let Some(pos) = text.find(stop.as_str()) {
            earliest = Some(match earliest {
                None => pos,
                Some(existing) => existing.min(pos),
            });
        }
    }
    earliest.map(|pos| text[..pos].to_string())
}

/// Derive a model id from the final path segment of a model path or HF id.
///
/// Mirrors the legacy `model_id_from_path` helper. Used by the backend variants
/// (`#[cfg(any(metal, cuda, cpu))]`); unused on a no-backend lib build.
#[must_use]
#[cfg_attr(
    not(any(feature = "metal", feature = "cuda", feature = "cpu")),
    allow(dead_code)
)]
pub(crate) fn model_id_from_path(model_path: &str) -> String {
    std::path::Path::new(model_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(model_path)
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::{model_id_from_path, truncate_at_first_stop};

    #[test]
    fn truncate_at_first_stop_picks_earliest() {
        let stops: Vec<String> = vec!["\n\n".into(), "END".into()];
        assert_eq!(
            truncate_at_first_stop("4\n\nand more", &stops),
            Some("4".to_string())
        );
        assert_eq!(
            truncate_at_first_stop("helloEND", &stops),
            Some("hello".to_string())
        );
        assert_eq!(truncate_at_first_stop("hello", &stops), None);
        assert_eq!(
            truncate_at_first_stop("a\n\nbEND", &stops),
            Some("a".to_string())
        );
    }

    #[test]
    fn model_id_uses_final_path_segment() {
        assert_eq!(
            model_id_from_path("mlx-community/Qwen3-0.6B-4bit"),
            "Qwen3-0.6B-4bit"
        );
        assert_eq!(model_id_from_path("/tmp/models/Qwen3-4B"), "Qwen3-4B");
    }
}
