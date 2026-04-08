use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Result, anyhow};
use log::error;
use tokio::sync::mpsc;

use crate::backend::{GenerateResult, InferenceBackend, StreamingInferenceBackend};
#[cfg(feature = "metal")]
use crate::metal_backend::MetalBackend;
use crate::request_handle::{RequestHandle, SubmitError};
use crate::scheduler::IncomingRequest;
use crate::server_engine::{FinishReason, StreamDelta, Usage};

/// Serial runtime handle for backends that only support one in-flight request.
#[derive(Clone)]
pub struct BackendRuntimeHandle {
    tx: mpsc::UnboundedSender<IncomingRequest>,
    model_id: Arc<str>,
    waiting_count: Arc<AtomicUsize>,
    max_waiting: usize,
}

impl BackendRuntimeHandle {
    pub fn new(
        tx: mpsc::UnboundedSender<IncomingRequest>,
        model_id: Arc<str>,
        waiting_count: Arc<AtomicUsize>,
        max_waiting: usize,
    ) -> Self {
        Self {
            tx,
            model_id,
            waiting_count,
            max_waiting,
        }
    }
}

impl RequestHandle for BackendRuntimeHandle {
    fn submit(&self, req: IncomingRequest) -> Result<(), SubmitError> {
        if self.max_waiting > 0 {
            let current = self.waiting_count.load(Ordering::Relaxed);
            if current >= self.max_waiting {
                return Err(SubmitError);
            }
        }

        self.waiting_count.fetch_add(1, Ordering::Relaxed);
        self.tx.send(req).map_err(|_| {
            self.waiting_count.fetch_sub(1, Ordering::Relaxed);
            SubmitError
        })
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

pub fn spawn_backend_runtime_handle<B>(
    backend: B,
    model_id: impl Into<Arc<str>>,
    max_waiting: usize,
) -> BackendRuntimeHandle
where
    B: StreamingInferenceBackend + 'static,
{
    let (tx, rx) = mpsc::unbounded_channel();
    let waiting_count = Arc::new(AtomicUsize::new(0));
    let handle = BackendRuntimeHandle::new(tx, model_id.into(), waiting_count.clone(), max_waiting);

    std::thread::spawn(move || run_backend_runtime(backend, rx, waiting_count));
    handle
}

#[cfg(feature = "metal")]
pub fn spawn_metal_runtime_handle_from_path(
    model_path: &str,
    max_waiting: usize,
) -> Result<BackendRuntimeHandle> {
    let mut backend = MetalBackend::new();
    backend.load(Path::new(model_path))?;

    let model_id = Path::new(model_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(model_path)
        .to_string();

    Ok(spawn_backend_runtime_handle(
        backend,
        Arc::<str>::from(model_id),
        max_waiting,
    ))
}

fn run_backend_runtime<B>(
    backend: B,
    mut rx: mpsc::UnboundedReceiver<IncomingRequest>,
    waiting_count: Arc<AtomicUsize>,
) where
    B: StreamingInferenceBackend,
{
    while let Some(req) = rx.blocking_recv() {
        waiting_count.fetch_sub(1, Ordering::Relaxed);
        if let Err(err) = execute_request(&backend, req) {
            error!("backend runtime request failed: {err:#}");
        }
    }
}

fn execute_request<B>(backend: &B, req: IncomingRequest) -> Result<()>
where
    B: StreamingInferenceBackend,
{
    let mut sampling = req.sampling.clone();
    sampling.max_new_tokens = Some(req.max_tokens);

    let mut stop_processor = StopChunkProcessor::new(req.stop.unwrap_or_default());
    let delta_tx = req.delta_tx;

    let generated = backend.generate_stream(&req.prompt, &sampling, |chunk| {
        if let Some(delta) = stop_processor.push_chunk(chunk) {
            send_text_delta(&delta_tx, delta)?;
        }
        Ok(())
    })?;

    if let Some(final_delta) = stop_processor.finish() {
        send_text_delta(&delta_tx, final_delta)?;
    }

    let finish_reason = if stop_processor.hit_stop() {
        FinishReason::Stop
    } else {
        parse_finish_reason(&generated)
    };

    let usage = Usage {
        prompt_tokens: generated.prompt_tokens,
        completion_tokens: generated.completion_tokens,
        total_tokens: generated.prompt_tokens + generated.completion_tokens,
    };

    let _ = delta_tx.send(StreamDelta {
        text_delta: String::new(),
        finish_reason: Some(finish_reason),
        usage: Some(usage),
        logprob: None,
    });

    Ok(())
}

fn parse_finish_reason(generated: &GenerateResult) -> FinishReason {
    match generated.finish_reason.as_str() {
        "length" => FinishReason::Length,
        _ => FinishReason::Stop,
    }
}

fn send_text_delta(
    delta_tx: &mpsc::UnboundedSender<StreamDelta>,
    text_delta: String,
) -> Result<()> {
    if text_delta.is_empty() {
        return Ok(());
    }

    delta_tx
        .send(StreamDelta {
            text_delta,
            finish_reason: None,
            usage: None,
            logprob: None,
        })
        .map_err(|_| anyhow!("stream consumer dropped"))
}

struct StopChunkProcessor {
    accumulated: String,
    sent_len: usize,
    stops: Vec<String>,
    max_stop_len: usize,
    hit_stop: bool,
}

impl StopChunkProcessor {
    fn new(stops: Vec<String>) -> Self {
        let max_stop_len = stops.iter().map(String::len).max().unwrap_or(0);
        Self {
            accumulated: String::new(),
            sent_len: 0,
            stops,
            max_stop_len,
            hit_stop: false,
        }
    }

    fn push_chunk(&mut self, chunk: &str) -> Option<String> {
        if self.hit_stop {
            return None;
        }

        self.accumulated.push_str(chunk);

        if let Some(stop_pos) = self.find_earliest_stop() {
            let delta = self.accumulated[self.sent_len..stop_pos].to_string();
            self.sent_len = stop_pos;
            self.hit_stop = true;
            return Some(delta);
        }

        if self.max_stop_len <= 1 {
            return self.flush_ready(self.accumulated.len());
        }

        let safe_end = self
            .accumulated
            .len()
            .saturating_sub(self.max_stop_len.saturating_sub(1));
        let safe_end = clamp_char_boundary(&self.accumulated, safe_end);
        self.flush_ready(safe_end)
    }

    fn finish(&mut self) -> Option<String> {
        if self.hit_stop {
            return None;
        }
        self.flush_ready(self.accumulated.len())
    }

    fn hit_stop(&self) -> bool {
        self.hit_stop
    }

    fn flush_ready(&mut self, end: usize) -> Option<String> {
        if end <= self.sent_len {
            return None;
        }
        let delta = self.accumulated[self.sent_len..end].to_string();
        self.sent_len = end;
        Some(delta)
    }

    fn find_earliest_stop(&self) -> Option<usize> {
        let unsent = &self.accumulated[self.sent_len..];
        let mut earliest = None::<usize>;

        for stop in &self.stops {
            if stop.is_empty() {
                continue;
            }
            if let Some(pos) = unsent.find(stop) {
                let absolute = self.sent_len + pos;
                earliest = Some(match earliest {
                    None => absolute,
                    Some(existing) => existing.min(absolute),
                });
            }
        }

        earliest
    }
}

fn clamp_char_boundary(text: &str, mut idx: usize) -> usize {
    idx = idx.min(text.len());
    while idx > 0 && !text.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::GenerateResult;
    use crate::request_handle::RequestHandle;
    use crate::sampler::SamplingParams;

    #[derive(Clone)]
    struct MockStreamingBackend {
        chunks: Vec<String>,
        finish_reason: String,
    }

    impl InferenceBackend for MockStreamingBackend {
        fn load(&mut self, _model_path: &Path) -> Result<()> {
            Ok(())
        }

        fn generate(&self, _prompt: &str, _params: &SamplingParams) -> Result<GenerateResult> {
            Ok(GenerateResult {
                text: self.chunks.concat(),
                prompt_tokens: 3,
                completion_tokens: 2,
                finish_reason: self.finish_reason.clone(),
                ttft_ms: 1.0,
                prompt_tps: 10.0,
                generation_tps: 20.0,
                total_time_ms: 2.0,
            })
        }

        fn name(&self) -> &'static str {
            "mock"
        }
    }

    impl StreamingInferenceBackend for MockStreamingBackend {
        fn generate_stream<F>(
            &self,
            _prompt: &str,
            _params: &SamplingParams,
            mut on_chunk: F,
        ) -> Result<GenerateResult>
        where
            F: FnMut(&str) -> Result<()>,
        {
            for chunk in &self.chunks {
                on_chunk(chunk)?;
            }
            self.generate("", &SamplingParams::default())
        }
    }

    fn make_request(
        prompt: &str,
        stop: Option<Vec<String>>,
    ) -> (IncomingRequest, mpsc::UnboundedReceiver<StreamDelta>) {
        let (delta_tx, delta_rx) = mpsc::unbounded_channel();
        (
            IncomingRequest {
                prompt: prompt.to_string(),
                max_tokens: 8,
                sampling: SamplingParams::default(),
                stop,
                priority: crate::scheduler::RequestPriority::default(),
                delta_tx,
            },
            delta_rx,
        )
    }

    #[tokio::test]
    async fn backend_runtime_streams_chunks_and_usage() {
        let handle = spawn_backend_runtime_handle(
            MockStreamingBackend {
                chunks: vec!["hel".into(), "lo".into()],
                finish_reason: "stop".into(),
            },
            Arc::<str>::from("mock-model"),
            8,
        );

        let (req, mut delta_rx) = make_request("hi", None);
        handle.submit(req).unwrap();

        let mut deltas = Vec::new();
        while let Some(delta) = delta_rx.recv().await {
            deltas.push(delta);
            if deltas
                .last()
                .and_then(|delta| delta.finish_reason)
                .is_some()
            {
                break;
            }
        }

        assert_eq!(deltas[0].text_delta, "hel");
        assert_eq!(deltas[1].text_delta, "lo");
        assert_eq!(deltas[2].finish_reason, Some(FinishReason::Stop));
        assert_eq!(
            deltas[2].usage,
            Some(Usage {
                prompt_tokens: 3,
                completion_tokens: 2,
                total_tokens: 5,
            })
        );
    }

    #[tokio::test]
    async fn backend_runtime_applies_stop_strings() {
        let handle = spawn_backend_runtime_handle(
            MockStreamingBackend {
                chunks: vec!["hel".into(), "loENDworld".into()],
                finish_reason: "length".into(),
            },
            Arc::<str>::from("mock-model"),
            8,
        );

        let (req, mut delta_rx) = make_request("hi", Some(vec!["END".into()]));
        handle.submit(req).unwrap();

        let mut text = String::new();
        let mut finish_reason = None;
        while let Some(delta) = delta_rx.recv().await {
            text.push_str(&delta.text_delta);
            if delta.finish_reason.is_some() {
                finish_reason = delta.finish_reason;
                break;
            }
        }

        assert_eq!(text, "hello");
        assert_eq!(finish_reason, Some(FinishReason::Stop));
    }
}
