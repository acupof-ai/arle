use anyhow::Result;
use tokio::sync::mpsc::UnboundedSender;

use crate::request_handle::RequestHandle;
use crate::scheduler::{IncomingRequest, RequestPriority};
use crate::session_persistence::SessionPersistence;

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
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        self.submit_request(req, tx)?;

        let mut text = String::new();
        let mut finish_reason = None;
        let mut usage = None;

        while let Some(delta) = rx.blocking_recv() {
            if !delta.text_delta.is_empty() {
                text.push_str(&delta.text_delta);
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
        })
    }

    fn complete_stream(
        &mut self,
        req: CompletionRequest,
        tx: UnboundedSender<CompletionStreamDelta>,
    ) -> Result<()> {
        self.submit_request(req, tx)
    }
}

impl<H: RequestHandle> SessionPersistence for RequestHandleInferenceEngine<H> {}
