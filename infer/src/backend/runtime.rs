use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Result, anyhow};
use log::error;
use tokio::sync::mpsc;

#[cfg(feature = "cpu")]
use super::cpu::CpuBackend;
#[cfg(feature = "metal")]
use super::metal::MetalBackend;
#[cfg(feature = "metal")]
use super::metal::MetalBackendOptions;
#[cfg(any(feature = "metal", feature = "cpu"))]
use crate::backend::InferenceBackend;
use crate::backend::{GenerateResult, StreamStopMatched, StreamingInferenceBackend};
use crate::metrics::ServerMetrics;
use crate::request_handle::{RequestHandle, SubmitError};
use crate::scheduler::IncomingRequest;
use crate::server_engine::{CompletionStreamDelta, FinishReason, TokenUsage};

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
        // Atomically increment the waiting count only if below the limit.
        // A CAS loop prevents concurrent submits from racing past the cap.
        if self.max_waiting > 0 {
            loop {
                let current = self.waiting_count.load(Ordering::Acquire);
                if current >= self.max_waiting {
                    return Err(SubmitError);
                }
                if self
                    .waiting_count
                    .compare_exchange_weak(
                        current,
                        current + 1,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .is_ok()
                {
                    break;
                }
            }
        } else {
            self.waiting_count.fetch_add(1, Ordering::AcqRel);
        }

        self.tx.send(req).map_err(|_| {
            self.waiting_count.fetch_sub(1, Ordering::AcqRel);
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
    spawn_backend_runtime_handle_with_metrics(
        backend,
        model_id,
        max_waiting,
        ServerMetrics::new(""),
    )
}

pub fn spawn_backend_runtime_handle_with_metrics<B>(
    backend: B,
    model_id: impl Into<Arc<str>>,
    max_waiting: usize,
    metrics: ServerMetrics,
) -> BackendRuntimeHandle
where
    B: StreamingInferenceBackend + 'static,
{
    let (tx, rx) = mpsc::unbounded_channel();
    let waiting_count = Arc::new(AtomicUsize::new(0));
    let handle = BackendRuntimeHandle::new(tx, model_id.into(), waiting_count.clone(), max_waiting);

    std::thread::spawn(move || run_backend_runtime(backend, rx, waiting_count, metrics));
    handle
}

#[cfg(feature = "metal")]
pub fn spawn_metal_runtime_handle_from_path(
    model_path: &str,
    max_waiting: usize,
) -> Result<BackendRuntimeHandle> {
    spawn_metal_runtime_handle_from_path_with_options(
        model_path,
        MetalBackendOptions::default(),
        max_waiting,
    )
}

#[cfg(feature = "metal")]
pub fn spawn_metal_runtime_handle_from_path_with_options(
    model_path: &str,
    options: MetalBackendOptions,
    max_waiting: usize,
) -> Result<BackendRuntimeHandle> {
    use std::path::Path;

    let model_id = Path::new(model_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(model_path)
        .to_string();
    spawn_metal_runtime_handle_from_path_with_options_and_metrics(
        model_path,
        options,
        max_waiting,
        ServerMetrics::new(&model_id),
    )
}

#[cfg(feature = "metal")]
pub fn spawn_metal_runtime_handle_from_path_with_options_and_metrics(
    model_path: &str,
    options: MetalBackendOptions,
    max_waiting: usize,
    metrics: ServerMetrics,
) -> Result<BackendRuntimeHandle> {
    use std::path::Path;

    let mut backend = MetalBackend::with_options(options);
    backend.load(Path::new(model_path))?;

    let model_id = Path::new(model_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(model_path)
        .to_string();

    Ok(spawn_backend_runtime_handle_with_metrics(
        backend,
        Arc::<str>::from(model_id),
        max_waiting,
        metrics,
    ))
}

#[cfg(feature = "cpu")]
pub fn spawn_cpu_runtime_handle_from_path(
    model_path: &str,
    max_waiting: usize,
) -> Result<BackendRuntimeHandle> {
    use std::path::Path;

    let mut backend = CpuBackend::new();
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

#[allow(clippy::needless_pass_by_value)]
fn run_backend_runtime<B>(
    backend: B,
    mut rx: mpsc::UnboundedReceiver<IncomingRequest>,
    waiting_count: Arc<AtomicUsize>,
    metrics: ServerMetrics,
) where
    B: StreamingInferenceBackend,
{
    while let Some(req) = rx.blocking_recv() {
        waiting_count.fetch_sub(1, Ordering::AcqRel);
        metrics.set_waiting(waiting_count.load(Ordering::Acquire) as u64);
        metrics.set_active(1);
        let result = catch_unwind(AssertUnwindSafe(|| {
            execute_request(&backend, req, &metrics)
        }));
        let outcome = match result {
            Ok(result) => result,
            Err(panic) => Err(anyhow!(
                "backend runtime panicked: {}",
                panic_message(panic)
            )),
        };
        if let Err(err) = outcome {
            error!("backend runtime request failed: {err:#}");
            metrics.record_request_failed();
        }
        metrics.set_active(0);
        metrics.set_waiting(waiting_count.load(Ordering::Acquire) as u64);
    }
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    match payload.downcast::<String>() {
        Ok(msg) => *msg,
        Err(payload) => match payload.downcast::<&'static str>() {
            Ok(msg) => (*msg).to_string(),
            Err(_) => "unknown panic payload".to_string(),
        },
    }
}

fn execute_request<B>(backend: &B, req: IncomingRequest, metrics: &ServerMetrics) -> Result<()>
where
    B: StreamingInferenceBackend,
{
    let mut sampling = req.sampling.clone();
    sampling.max_new_tokens = Some(req.max_tokens);

    let mut stop_processor = StopChunkProcessor::new(req.stop.unwrap_or_default());
    let delta_tx = req.delta_tx;

    // Phase 2 trajectory: collate visible text we forward so the final
    // delta can ride its tokenized form as `token_ids`.
    let emitted_text = std::cell::RefCell::new(String::new());

    let generated = backend.generate_stream(&req.prompt, &sampling, |chunk| {
        if let Some(delta) = stop_processor.push_chunk(chunk) {
            emitted_text.borrow_mut().push_str(&delta);
            send_text_delta(&delta_tx, delta)?;
        }
        if stop_processor.hit_stop() {
            return Err(StreamStopMatched.into());
        }
        Ok(())
    })?;

    let ttft_s = generated.ttft_ms / 1000.0;
    let e2e_s = generated.total_time_ms / 1000.0;
    let tpot_s = if generated.completion_tokens > 1 {
        (e2e_s - ttft_s).max(0.0) / (generated.completion_tokens - 1) as f64
    } else {
        0.0
    };
    metrics.record_request_completed(
        generated.prompt_tokens as u64,
        generated.completion_tokens as u64,
        ttft_s,
        tpot_s,
        e2e_s,
    );

    if let Some(final_delta) = stop_processor.finish() {
        emitted_text.borrow_mut().push_str(&final_delta);
        send_text_delta(&delta_tx, final_delta)?;
    }

    let finish_reason = if stop_processor.hit_stop() {
        FinishReason::Stop
    } else {
        parse_finish_reason(&generated)
    };
    let usage = TokenUsage {
        prompt_tokens: generated.prompt_tokens,
        completion_tokens: generated.completion_tokens,
        total_tokens: generated.prompt_tokens + generated.completion_tokens,
    };

    let final_text = emitted_text.borrow();
    let response_token_ids = if final_text.is_empty() {
        Vec::new()
    } else {
        backend.tokenize(&final_text).unwrap_or_default()
    };
    drop(final_text);

    let _ = delta_tx.send(CompletionStreamDelta {
        text_delta: String::new(),
        finish_reason: Some(finish_reason),
        usage: Some(usage),
        logprob: None,
        token_ids: response_token_ids,
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
    delta_tx: &mpsc::UnboundedSender<CompletionStreamDelta>,
    text_delta: String,
) -> Result<()> {
    if text_delta.is_empty() {
        return Ok(());
    }

    delta_tx
        .send(CompletionStreamDelta {
            text_delta,
            finish_reason: None,
            usage: None,
            logprob: None,
            // Per-chunk token IDs aren't available on the
            // text-callback streaming path; the cumulative list rides
            // on the final delta in `execute_request`.
            token_ids: Vec::new(),
        })
        .map_err(|_| anyhow!("stream consumer dropped"))
}

/// Chunk-level stop-sequence guard.
///
/// Buffers streamed text and emits ready-to-send deltas while withholding
/// the last `max_stop_len - 1` bytes — that tail could still complete a
/// stop marker on the next chunk. When a stop is matched (anywhere in
/// the unsent suffix, earliest wins), emits the delta up to the marker,
/// flips `hit_stop`, and thereafter absorbs further chunks silently so
/// the raw stop bytes never reach the consumer.
///
/// Shared by `BackendInferenceEngine::complete_stream` in
/// `server_engine.rs` and the Metal runtime stream path. Keep the stop
/// buffering logic here so all streaming callers agree on the same
/// boundary and tail-withholding behavior.
pub(crate) struct StopChunkProcessor {
    accumulated: String,
    sent_len: usize,
    stops: Vec<String>,
    max_stop_len: usize,
    hit_stop: bool,
}

impl StopChunkProcessor {
    pub(crate) fn new(stops: Vec<String>) -> Self {
        let max_stop_len = stops.iter().map(String::len).max().unwrap_or(0);
        Self {
            accumulated: String::new(),
            sent_len: 0,
            stops,
            max_stop_len,
            hit_stop: false,
        }
    }

    pub(crate) fn push_chunk(&mut self, chunk: &str) -> Option<String> {
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

    pub(crate) fn finish(&mut self) -> Option<String> {
        if self.hit_stop {
            return None;
        }
        self.flush_ready(self.accumulated.len())
    }

    pub(crate) fn hit_stop(&self) -> bool {
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
    use std::path::Path;

    use super::*;
    use crate::backend::{GenerateResult, InferenceBackend};
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
                if let Err(err) = on_chunk(chunk) {
                    if crate::backend::is_stream_stop_matched(&err) {
                        return self.generate("", &SamplingParams::default());
                    }
                    return Err(err);
                }
            }
            self.generate("", &SamplingParams::default())
        }
    }

    fn make_request(
        prompt: &str,
        stop: Option<Vec<String>>,
    ) -> (
        IncomingRequest,
        mpsc::UnboundedReceiver<CompletionStreamDelta>,
    ) {
        let (delta_tx, delta_rx) = mpsc::unbounded_channel();
        (
            IncomingRequest {
                prompt: prompt.to_string(),
                prompt_tokens: None,
                max_tokens: 8,
                sampling: SamplingParams::default(),
                stop,
                priority: crate::scheduler::RequestPriority::default(),
                session_id: None,
                delta_tx,
                trace_context: None,
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
            Some(TokenUsage {
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

    #[tokio::test]
    async fn backend_runtime_short_circuits_after_text_stop_match() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        #[derive(Clone)]
        struct CountingStopBackend {
            chunks: Vec<String>,
            chunks_attempted: Arc<AtomicUsize>,
        }

        impl InferenceBackend for CountingStopBackend {
            fn load(&mut self, _model_path: &Path) -> Result<()> {
                Ok(())
            }

            fn generate(&self, _prompt: &str, _params: &SamplingParams) -> Result<GenerateResult> {
                unreachable!("test exercises streaming path only")
            }

            fn name(&self) -> &'static str {
                "counting-stop"
            }
        }

        impl StreamingInferenceBackend for CountingStopBackend {
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
                    self.chunks_attempted.fetch_add(1, Ordering::Relaxed);
                    if let Err(err) = on_chunk(chunk) {
                        if err
                            .downcast_ref::<crate::backend::StreamStopMatched>()
                            .is_some()
                        {
                            return Ok(GenerateResult {
                                text: "helloENDwaste".into(),
                                prompt_tokens: 11,
                                completion_tokens: 7,
                                finish_reason: "stop".into(),
                                ttft_ms: 12.0,
                                prompt_tps: 1.0,
                                generation_tps: 2.0,
                                total_time_ms: 45.0,
                            });
                        }
                        return Err(err);
                    }
                }
                Ok(GenerateResult {
                    text: "helloENDwaste never-sent".into(),
                    prompt_tokens: 11,
                    completion_tokens: 13,
                    finish_reason: "length".into(),
                    ttft_ms: 12.0,
                    prompt_tps: 1.0,
                    generation_tps: 2.0,
                    total_time_ms: 45.0,
                })
            }
        }

        let chunks_attempted = Arc::new(AtomicUsize::new(0));
        let handle = spawn_backend_runtime_handle(
            CountingStopBackend {
                chunks: vec!["helloENDwaste".into(), "never-sent".into()],
                chunks_attempted: Arc::clone(&chunks_attempted),
            },
            Arc::<str>::from("mock-model"),
            8,
        );

        let (req, mut delta_rx) = make_request("hi", Some(vec!["END".into()]));
        handle.submit(req).unwrap();

        let mut text = String::new();
        let mut finish_reason = None;
        let mut usage = None;
        while let Some(delta) = delta_rx.recv().await {
            text.push_str(&delta.text_delta);
            if delta.finish_reason.is_some() {
                finish_reason = delta.finish_reason;
                usage = delta.usage;
                break;
            }
        }

        assert_eq!(text, "hello");
        assert_eq!(finish_reason, Some(FinishReason::Stop));
        assert_eq!(
            usage,
            Some(TokenUsage {
                prompt_tokens: 11,
                completion_tokens: 7,
                total_tokens: 18,
            })
        );
        assert_eq!(chunks_attempted.load(Ordering::Relaxed), 1);
    }

    /// A mock backend that blocks for a configurable duration, used to test
    /// backpressure under concurrent submits.
    #[derive(Clone)]
    struct SlowBackend {
        delay: std::time::Duration,
    }

    impl InferenceBackend for SlowBackend {
        fn load(&mut self, _model_path: &Path) -> Result<()> {
            Ok(())
        }
        fn generate(&self, _prompt: &str, _params: &SamplingParams) -> Result<GenerateResult> {
            std::thread::sleep(self.delay);
            Ok(GenerateResult {
                text: "ok".into(),
                prompt_tokens: 1,
                completion_tokens: 1,
                finish_reason: "stop".into(),
                ttft_ms: 1.0,
                prompt_tps: 1.0,
                generation_tps: 1.0,
                total_time_ms: 1.0,
            })
        }
        fn name(&self) -> &'static str {
            "slow"
        }
    }

    impl StreamingInferenceBackend for SlowBackend {
        fn generate_stream<F>(
            &self,
            prompt: &str,
            params: &SamplingParams,
            _on_chunk: F,
        ) -> Result<GenerateResult>
        where
            F: FnMut(&str) -> Result<()>,
        {
            self.generate(prompt, params)
        }
    }

    #[tokio::test]
    async fn backpressure_rejects_when_at_capacity() {
        let (tx, _rx) = mpsc::unbounded_channel();
        let waiting_count = Arc::new(AtomicUsize::new(0));
        let handle = BackendRuntimeHandle::new(tx, Arc::<str>::from("slow"), waiting_count, 2);

        // Fill the queue to capacity.
        let (req1, _rx1) = make_request("a", None);
        let (req2, _rx2) = make_request("b", None);
        handle.submit(req1).unwrap();
        handle.submit(req2).unwrap();

        // Third submit should be rejected (waiting_count == 2 == max_waiting).
        let (req3, _rx3) = make_request("c", None);
        assert!(handle.submit(req3).is_err(), "should reject at capacity");
    }

    #[tokio::test]
    async fn backpressure_concurrent_submits_respect_limit() {
        use std::sync::atomic::Ordering;

        let handle = spawn_backend_runtime_handle(
            SlowBackend {
                delay: std::time::Duration::from_millis(500),
            },
            Arc::<str>::from("slow"),
            4, // max_waiting = 4
        );

        // Spawn 8 concurrent submits. The queue may admit one extra request if
        // the worker thread has already popped one into the active slot, but
        // the waiting queue itself must never exceed `max_waiting`.
        let mut tasks = Vec::new();
        for _ in 0..8 {
            let h = handle.clone();
            tasks.push(tokio::spawn(async move {
                let (req, _rx) = make_request("x", None);
                h.submit(req)
            }));
        }

        let mut accepted = 0usize;
        let mut rejected = 0usize;
        for task in tasks {
            match task.await.unwrap() {
                Ok(()) => accepted += 1,
                Err(_) => rejected += 1,
            }
        }

        let waiting = handle.waiting_count.load(Ordering::Acquire);
        assert!(waiting <= 4, "waiting queue should cap at 4, got {waiting}");
        assert!(
            accepted <= 5,
            "should accept at most 1 active + 4 waiting, got {accepted}"
        );
        assert!(rejected >= 3, "should reject at least 3, got {rejected}");
    }

    #[cfg(feature = "cpu")]
    #[tokio::test]
    async fn cpu_runtime_handle_loads_and_streams() {
        let model_dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            model_dir.path().join("config.json"),
            r#"{"architectures":["Qwen3ForCausalLM"],"model_type":"qwen3"}"#,
        )
        .expect("config");

        let handle =
            spawn_cpu_runtime_handle_from_path(model_dir.path().to_str().expect("utf8"), 8)
                .expect("cpu runtime");

        let (mut req, mut delta_rx) = make_request("smoke request", None);
        req.max_tokens = 64;
        handle.submit(req).expect("submit");

        let mut text = String::new();
        let mut finish_reason = None;
        while let Some(delta) = delta_rx.recv().await {
            text.push_str(&delta.text_delta);
            if delta.finish_reason.is_some() {
                finish_reason = delta.finish_reason;
                break;
            }
        }

        assert!(text.contains("CPU backend development response"));
        assert_eq!(finish_reason, Some(FinishReason::Stop));
    }
}
