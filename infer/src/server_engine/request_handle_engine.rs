use anyhow::Result;
use tokio::sync::mpsc::UnboundedSender;

use crate::request_handle::RequestHandle;
use crate::scheduler::{IncomingRequest, RequestPriority};

use super::{CompletionOutput, CompletionRequest, CompletionStreamDelta, InferenceEngine};

pub struct RequestHandleInferenceEngine<H: RequestHandle> {
    pub(super) model_id: String,
    pub(super) handle: H,
}

impl<H: RequestHandle> RequestHandleInferenceEngine<H> {
    /// Adopt a previously-spawned `RequestHandle` (e.g. the CUDA scheduler
    /// or the Metal runtime). Caller owns any thread join handle / guard
    /// that backs the underlying scheduler.
    pub fn from_handle(model_id: String, handle: H) -> Self {
        Self { model_id, handle }
    }

    fn submit_request(
        &self,
        req: CompletionRequest,
        delta_tx: UnboundedSender<CompletionStreamDelta>,
    ) -> Result<()> {
        self.handle
            .submit(IncomingRequest {
                prompt: req.prompt,
                prompt_tokens: None,
                max_tokens: req.max_tokens,
                sampling: req.sampling,
                stop: req.stop,
                priority: RequestPriority::Normal,
                session_id: req.session_id,
                delta_tx,
                trace_context: req.trace_context,
            })
            .map_err(|err| anyhow::anyhow!("request submission failed: {err}"))
    }
}

impl<H: RequestHandle> InferenceEngine for RequestHandleInferenceEngine<H> {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn complete(&mut self, req: CompletionRequest) -> Result<CompletionOutput> {
        // Phase 2 trajectory: snapshot the prompt before submission so
        // we can tokenize it after the worker takes ownership of `req`.
        // Failures collapse to empty Vec — the agent loop treats empty
        // as "unavailable" and downgrades `tokens = None`.
        let prompt_token_ids = self.tokenize(&req.prompt).unwrap_or_default();

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        self.submit_request(req, tx)?;

        let mut text = String::new();
        let mut finish_reason = None;
        let mut usage = None;
        let mut response_token_ids: Vec<u32> = Vec::new();

        while let Some(delta) = rx.blocking_recv() {
            if !delta.text_delta.is_empty() {
                text.push_str(&delta.text_delta);
            }
            if !delta.token_ids.is_empty() {
                response_token_ids.extend(delta.token_ids);
            }
            if let Some(final_usage) = delta.usage {
                usage = Some(final_usage);
            }
            if let Some(reason) = delta.finish_reason {
                finish_reason = Some(reason);
                break;
            }
        }

        Ok(CompletionOutput {
            text,
            finish_reason: finish_reason
                .ok_or_else(|| anyhow::anyhow!("stream ended without finish reason"))?,
            usage: usage.ok_or_else(|| anyhow::anyhow!("stream ended without token usage"))?,
            token_logprobs: Vec::new(),
            prompt_token_ids,
            response_token_ids,
        })
    }

    fn complete_stream(
        &mut self,
        req: CompletionRequest,
        tx: UnboundedSender<CompletionStreamDelta>,
    ) -> Result<()> {
        self.submit_request(req, tx)
    }

    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        let tokenizer = self
            .handle
            .tokenizer_clone()
            .ok_or_else(|| anyhow::anyhow!("backend has no tokenizer to tokenize() with"))?;
        tokenizer.encode(text)
    }
}
