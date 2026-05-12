use anyhow::Result;
use tokio::sync::mpsc::UnboundedSender;

use crate::request_handle::RequestHandle;
use crate::scheduler::{IncomingRequest, RequestPriority};

use super::{
    CompletionOutput, CompletionRequest, CompletionStreamDelta, EngineTelemetry, InferenceEngine,
};

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
    ) -> Result<Option<Vec<u32>>> {
        let prompt_tokens = self.preprocess_prompt_tokens(&req);
        self.handle
            .submit(IncomingRequest {
                prompt: req.prompt,
                prompt_tokens: prompt_tokens.clone(),
                max_tokens: req.max_tokens,
                sampling: req.sampling,
                stop: req.stop,
                speculative: None,
                priority: RequestPriority::Normal,
                session_id: req.session_id,
                delta_tx,
                trace_context: req.trace_context,
            })
            .map_err(|err| anyhow::anyhow!("request submission failed: {err}"))?;
        Ok(prompt_tokens)
    }

    fn preprocess_prompt_tokens(&self, req: &CompletionRequest) -> Option<Vec<u32>> {
        let tokenizer = self.handle.tokenizer_clone()?;
        tokenizer.encode(&req.prompt).ok()
    }
}

impl<H: RequestHandle> InferenceEngine for RequestHandleInferenceEngine<H> {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn complete(&mut self, req: CompletionRequest) -> Result<CompletionOutput> {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let prompt_token_ids = self.submit_request(req, tx)?.unwrap_or_default();

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
        // Backend contract (matches BackendInferenceEngine): complete_stream
        // blocks until the request finishes, with all deltas already on `tx`.
        // Forward via an internal channel so we can wait for the finish
        // marker before returning.
        let (inner_tx, mut inner_rx) = tokio::sync::mpsc::unbounded_channel();
        let _ = self.submit_request(req, inner_tx)?;
        while let Some(delta) = inner_rx.blocking_recv() {
            let finished = delta.finish_reason.is_some();
            if tx.send(delta).is_err() {
                // Consumer dropped — drain remaining deltas silently.
                while inner_rx.blocking_recv().is_some() {}
                return Ok(());
            }
            if finished {
                break;
            }
        }
        Ok(())
    }

    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        let tokenizer = self
            .handle
            .tokenizer_clone()
            .ok_or_else(|| anyhow::anyhow!("backend has no tokenizer to tokenize() with"))?;
        tokenizer.encode(text)
    }

    fn telemetry(&self) -> EngineTelemetry {
        // Both CUDA `SchedulerHandle` and Metal `MetalSchedulerHandle`
        // expose a clone of the shared `ServerMetrics` instance via
        // `RequestHandle::server_metrics()`. Empty default for handles
        // that don't carry one (mocks/tests).
        self.handle
            .server_metrics()
            .map(crate::metrics::ServerMetrics::snapshot_engine_telemetry)
            .unwrap_or_default()
    }
}
