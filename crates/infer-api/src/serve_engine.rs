//! Generic adapter wrapping a [`ServeHandle`] + tokenizer as an
//! [`InferenceEngine`]: `tokenize -> submit -> collect -> detokenize`, projecting
//! the rewrite `CompletedRequest` back into [`CompletionOutput`].

use anyhow::{Result, anyhow};
use infer_core::CompletedRequest;
use infer_seam::{BackendExecutor, KvPool};
use infer_server::{OpenAiTokenizer, ServeHandle, StreamItem};
use tokio::sync::mpsc::UnboundedSender;

use crate::types::{
    CompletionOutput, CompletionRequest, CompletionStreamDelta, EngineTelemetry, FinishReason,
    InferenceEngine, TokenUsage,
};

/// Adapter over one running [`ServeHandle`] + its tokenizer, generic over the
/// executor / KV pool so the same wiring serves every backend. The shared body
/// behind each [`crate::LoadedInferenceEngine`] variant's impl.
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

    /// Shared `tokenize -> submit -> collect -> detokenize` body returning the
    /// projected [`CompletionOutput`]; both `complete` and `complete_stream`
    /// build on it.
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

        // Host-side stop-string truncation (the engine stops on token ids; OpenAI
        // `stop` strings are applied here).
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
            token_logprobs: Vec::new(), // not yet surfaced
            prompt_token_ids,
            response_token_ids,
        })
    }
}

// `Send` holds via the auto-trait: `ServeInferenceEngine` stores only a
// `ServeHandle<E, K>` (Send channel + JoinHandle + PhantomData), never an `E`/`K`
// value, so a `!Send` executor (e.g. MLX `MetalExecutor`, which lives on the
// engine thread) is fine.
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
        let prompt_token_ids = match self.tokenizer.encode(&req.prompt) {
            Ok(ids) => ids,
            Err(err) => {
                let _ = tx.send(CompletionStreamDelta::error(
                    "tokenize_failed",
                    vec![err.to_string()],
                ));
                return Ok(());
            }
        };
        let (ticket, stream_rx) = match self.serve.submit_streaming(
            prompt_token_ids.clone(),
            req.max_tokens,
            req.sampling.clone(),
        ) {
            Ok(v) => v,
            Err(err) => {
                let _ = tx.send(CompletionStreamDelta::error(
                    "inference_failed",
                    vec![err.to_string()],
                ));
                return Ok(());
            }
        };

        // Stream tokens live. Decode the accumulated ids each step (so multi-byte
        // / BPE-merge boundaries resolve correctly) and emit only the newly-stable
        // prefix, holding back a tail that could be the start of a stop string.
        let holdback = req.stop.as_deref().map_or(0, stop_holdback);
        let mut acc_ids: Vec<u32> = Vec::new();
        let mut emitted = String::new();
        let mut completed = None;
        while let Ok(item) = stream_rx.recv() {
            match item {
                StreamItem::Token { token, .. } => {
                    acc_ids.push(token);
                    let full = self.tokenizer.decode(&acc_ids).unwrap_or_default();
                    if let Some(delta) = deliverable_delta(&full, &emitted, holdback) {
                        emitted.push_str(&delta);
                        let _ = tx.send(CompletionStreamDelta {
                            text_delta: delta,
                            finish_reason: None,
                            usage: None,
                            logprob: None,
                            token_ids: vec![token],
                            error: None,
                        });
                    }
                }
                StreamItem::Done(c) => {
                    completed = Some(c);
                    break;
                }
            }
        }

        // Terminal: re-decode the full output, apply host-side stop truncation,
        // flush any text not yet emitted (the held-back tail, up to the stop),
        // then a finish delta with the reason + usage.
        let completed = match completed {
            Some(c) => c,
            None => ticket.collect()?,
        };
        let full = self
            .tokenizer
            .decode(&completed.generated_tokens)
            .unwrap_or_default();
        let (final_text, stop_truncated) = match req.stop.as_deref() {
            Some(stops) => match truncate_at_first_stop(&full, stops) {
                Some(truncated) => (truncated, true),
                None => (full, false),
            },
            None => (full, false),
        };
        if final_text.len() > emitted.len() && final_text.starts_with(&emitted) {
            let _ = tx.send(CompletionStreamDelta {
                text_delta: final_text[emitted.len()..].to_string(),
                finish_reason: None,
                usage: None,
                logprob: None,
                token_ids: Vec::new(),
                error: None,
            });
        }
        let finish_reason = if stop_truncated {
            FinishReason::Stop
        } else {
            completed
                .finish
                .as_ref()
                .map_or(FinishReason::Length, FinishReason::from_plan)
        };
        let prompt_tokens = completed.prompt_tokens.len();
        let completion_tokens = completed.generated_tokens.len();
        let _ = tx.send(CompletionStreamDelta {
            text_delta: String::new(),
            finish_reason: Some(finish_reason),
            usage: Some(TokenUsage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            }),
            logprob: None,
            token_ids: Vec::new(),
            error: None,
        });
        Ok(())
    }

    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        self.tokenizer
            .encode(text)
            .map_err(|err| anyhow!("tokenize failed: {err}"))
    }

    fn telemetry(&self) -> EngineTelemetry {
        // GAP: `ServeHandle` surfaces no scheduler counters; empty means
        // "unavailable", not zero.
        EngineTelemetry::default()
    }
}

/// Truncate at the first occurrence of any non-empty stop string, returning the
/// prefix before it (or `None` if none matched).
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

/// Bytes to hold back from the live stream so a stop string spanning token
/// boundaries is detected before any of its prefix is emitted: one less than the
/// longest stop (0 when there are no stops).
fn stop_holdback(stops: &[String]) -> usize {
    stops
        .iter()
        .map(String::len)
        .max()
        .unwrap_or(0)
        .saturating_sub(1)
}

/// The newly-stable text to emit given the full decode so far, what has already
/// been emitted, and the stop-boundary `holdback`. Returns the slice of `full`
/// after `emitted` up to `full.len() - holdback` (rounded to a char boundary), or
/// `None` if nothing new is stable yet. Returns `None` if `full` no longer starts
/// with `emitted` (a rare retroactive decode change — wait for the next token).
fn deliverable_delta(full: &str, emitted: &str, holdback: usize) -> Option<String> {
    if !full.starts_with(emitted) {
        return None;
    }
    let mut end = full.len().saturating_sub(holdback).max(emitted.len());
    while end > emitted.len() && !full.is_char_boundary(end) {
        end -= 1;
    }
    if end <= emitted.len() {
        return None;
    }
    Some(full[emitted.len()..end].to_string())
}

/// Derive a model id from the final path segment of a model path or HF id.
/// Used by the backend variants; unused on a no-backend lib build.
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
    use super::{deliverable_delta, model_id_from_path, stop_holdback, truncate_at_first_stop};

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

    #[test]
    fn stop_holdback_is_longest_stop_minus_one() {
        assert_eq!(stop_holdback(&[]), 0);
        assert_eq!(stop_holdback(&["END".into(), "\n\n".into()]), 2); // max len 3 -> 2
        assert_eq!(stop_holdback(&["x".into()]), 0);
    }

    #[test]
    fn deliverable_delta_emits_new_stable_prefix() {
        // No holdback: every newly-decoded char is deliverable immediately.
        assert_eq!(deliverable_delta("Paris", "", 0).as_deref(), Some("Paris"));
        assert_eq!(
            deliverable_delta("Paris.", "Paris", 0).as_deref(),
            Some(".")
        );
        // Nothing new beyond what's emitted.
        assert_eq!(deliverable_delta("Paris", "Paris", 0), None);
    }

    #[test]
    fn deliverable_delta_holds_back_stop_boundary() {
        // holdback=2 ("END" stop): hold back the last 2 bytes so "EN" of a
        // potential "END" is never emitted before the full stop is seen.
        assert_eq!(deliverable_delta("hello", "", 2).as_deref(), Some("hel"));
        // After "hel" emitted, "hello" still holds back "lo".
        assert_eq!(deliverable_delta("hello", "hel", 2), None);
        // The completion grows; "lo" becomes safe once 2 more bytes follow
        // ("XY" is now the held-back tail).
        assert_eq!(
            deliverable_delta("helloXY", "hel", 2).as_deref(),
            Some("lo")
        );
    }

    #[test]
    fn deliverable_delta_waits_on_retroactive_change() {
        // `full` no longer starts with `emitted` (a decode that changed under us):
        // emit nothing and wait for the next token.
        assert_eq!(deliverable_delta("Pari", "Paris", 0), None);
    }
}
