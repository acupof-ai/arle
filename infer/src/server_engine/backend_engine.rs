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
use crate::session_persistence::SessionPersistence;

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

        Ok(CompletionOutput {
            text,
            finish_reason,
            usage: TokenUsage {
                prompt_tokens: generated.prompt_tokens,
                completion_tokens: generated.completion_tokens,
                total_tokens: generated.prompt_tokens + generated.completion_tokens,
            },
            token_logprobs: Vec::new(),
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
                        && tx
                            .send(CompletionStreamDelta {
                                text_delta: delta,
                                finish_reason: None,
                                usage: None,
                                logprob: None,
                            })
                            .is_err()
                    {
                        consumer_dropped.set(true);
                        return Err(anyhow::anyhow!("consumer dropped"));
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
            let _ = tx.send(CompletionStreamDelta {
                text_delta: trailing,
                finish_reason: None,
                usage: None,
                logprob: None,
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

        let _ = tx.send(CompletionStreamDelta {
            text_delta: String::new(),
            finish_reason: Some(finish_reason),
            usage: Some(usage),
            logprob: None,
        });
        Ok(())
    }
}

impl<B: InferenceBackend> SessionPersistence for BackendInferenceEngine<B> {}
