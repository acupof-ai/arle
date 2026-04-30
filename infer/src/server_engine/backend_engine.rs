use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;

use anyhow::Result;
use tokio::sync::mpsc::UnboundedSender;

#[cfg(feature = "cpu")]
use crate::backend::cpu::CpuBackend;
#[cfg(feature = "metal")]
use crate::backend::metal::MetalBackend;
use crate::backend::runtime::StopChunkProcessor;
use crate::backend::{InferenceBackend, StreamStopMatched, StreamingInferenceBackend};

use super::stream::{
    model_id_from_path, panic_message, parse_finish_reason, truncate_at_first_stop,
};
use super::{
    CompletionOutput, CompletionRequest, CompletionStreamDelta, FinishReason, InferenceEngine,
    TokenUsage,
};

pub struct BackendInferenceEngine<B: InferenceBackend> {
    pub(super) model_id: String,
    pub(super) backend: B,
}

#[cfg(feature = "metal")]
impl BackendInferenceEngine<MetalBackend> {
    #[allow(dead_code)]
    pub(super) fn load(model_path: &str) -> Result<Self> {
        let mut backend = MetalBackend::new();
        backend.load(Path::new(model_path))?;
        Ok(Self {
            model_id: model_id_from_path(model_path),
            backend,
        })
    }
}

#[cfg(feature = "cpu")]
impl BackendInferenceEngine<CpuBackend> {
    pub(super) fn load(model_path: &str) -> Result<Self> {
        let mut backend = CpuBackend::new();
        backend.load(Path::new(model_path))?;
        Ok(Self {
            model_id: model_id_from_path(model_path),
            backend,
        })
    }
}

impl<B: InferenceBackend + StreamingInferenceBackend> InferenceEngine
    for BackendInferenceEngine<B>
{
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn complete(&mut self, req: CompletionRequest) -> Result<CompletionOutput> {
        let mut sampling = req.sampling;
        sampling.max_new_tokens = Some(req.max_tokens);

        // Phase 2 trajectory: tokenize prompt+response so the agent loop
        // can populate `tokens.prompt_ids` / `tokens.response_ids`. Errors
        // are tolerated — the agent loop treats empty vectors as
        // "unavailable" and downgrades `tokens = None`.
        let prompt_token_ids = self.backend.tokenize(&req.prompt).unwrap_or_default();

        let generated = catch_unwind(AssertUnwindSafe(|| {
            self.backend.generate(&req.prompt, &sampling)
        }))
        .map_err(|panic| {
            anyhow::anyhow!(
                "{} backend panicked during completion: {}",
                self.backend.name(),
                panic_message(panic)
            )
        })??;

        let mut text = generated.text;
        let mut finish_reason = parse_finish_reason(&generated.finish_reason);

        if let Some(stops) = req.stop
            && let Some(truncated) = truncate_at_first_stop(&text, &stops)
        {
            text = truncated;
            finish_reason = FinishReason::Stop;
        }

        let response_token_ids = if text.is_empty() {
            Vec::new()
        } else {
            self.backend.tokenize(&text).unwrap_or_default()
        };

        Ok(CompletionOutput {
            text,
            finish_reason,
            usage: TokenUsage {
                prompt_tokens: generated.prompt_tokens,
                completion_tokens: generated.completion_tokens,
                total_tokens: generated.prompt_tokens + generated.completion_tokens,
            },
            token_logprobs: Vec::new(),
            prompt_token_ids,
            response_token_ids,
        })
    }

    /// Chunk-by-chunk streaming over `StreamingInferenceBackend`.
    fn complete_stream(
        &mut self,
        req: CompletionRequest,
        tx: UnboundedSender<CompletionStreamDelta>,
    ) -> Result<()> {
        let mut sampling = req.sampling;
        sampling.max_new_tokens = Some(req.max_tokens);

        let stops: Vec<String> = req
            .stop
            .unwrap_or_default()
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect();

        let processor = std::cell::RefCell::new(StopChunkProcessor::new(stops));
        let consumer_dropped = std::cell::Cell::new(false);
        // Phase 2 trajectory: collate the visible text we actually
        // forwarded to the consumer so the final delta can carry its
        // tokenized form as `token_ids`. Per-chunk token IDs aren't
        // observable here (the backend hands us raw text), so we emit
        // the full response on the FINAL delta — the collator's
        // `response_token_ids = concat(deltas.token_ids)` then sees the
        // complete tokenized output.
        let emitted_text = std::cell::RefCell::new(String::new());

        let backend_name = self.backend.name();
        let generated = catch_unwind(AssertUnwindSafe(|| {
            self.backend
                .generate_stream(&req.prompt, &sampling, |chunk: &str| -> Result<()> {
                    let (delta, stop_hit) = {
                        let mut processor = processor.borrow_mut();
                        let delta = processor.push_chunk(chunk);
                        let stop_hit = processor.hit_stop();
                        (delta, stop_hit)
                    };
                    if let Some(delta) = delta
                        && !delta.is_empty()
                    {
                        emitted_text.borrow_mut().push_str(&delta);
                        if tx
                            .send(CompletionStreamDelta {
                                text_delta: delta,
                                finish_reason: None,
                                usage: None,
                                logprob: None,
                                token_ids: Vec::new(),
                            })
                            .is_err()
                        {
                            consumer_dropped.set(true);
                            return Err(anyhow::anyhow!("consumer dropped"));
                        }
                    }
                    if stop_hit {
                        return Err(StreamStopMatched.into());
                    }
                    Ok(())
                })
        }))
        .map_err(|panic| {
            anyhow::anyhow!(
                "{} backend panicked during completion: {}",
                backend_name,
                panic_message(panic)
            )
        })?;

        if consumer_dropped.get() {
            return Ok(());
        }

        let generated = generated?;

        if let Some(trailing) = processor.borrow_mut().finish()
            && !trailing.is_empty()
        {
            emitted_text.borrow_mut().push_str(&trailing);
            let _ = tx.send(CompletionStreamDelta {
                text_delta: trailing,
                finish_reason: None,
                usage: None,
                logprob: None,
                token_ids: Vec::new(),
            });
        }

        let finish_reason = if processor.borrow().hit_stop() {
            FinishReason::Stop
        } else {
            parse_finish_reason(&generated.finish_reason)
        };
        let usage = TokenUsage {
            prompt_tokens: generated.prompt_tokens,
            completion_tokens: generated.completion_tokens,
            total_tokens: generated.prompt_tokens + generated.completion_tokens,
        };

        // Tokenize the visible text we actually emitted (post stop-trim)
        // and ride it on the final delta. Empty Vec on tokenize failure
        // is honest signal — the agent loop downgrades `tokens = None`
        // when any sub-turn produces an empty `response_token_ids`.
        let final_text = emitted_text.borrow();
        let response_token_ids = if final_text.is_empty() {
            Vec::new()
        } else {
            self.backend.tokenize(&final_text).unwrap_or_default()
        };
        drop(final_text);

        let _ = tx.send(CompletionStreamDelta {
            text_delta: String::new(),
            finish_reason: Some(finish_reason),
            usage: Some(usage),
            logprob: None,
            token_ids: response_token_ids,
        });
        Ok(())
    }

    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        self.backend.tokenize(text)
    }
}
