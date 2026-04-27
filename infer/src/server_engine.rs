#[cfg(any(feature = "metal", feature = "cpu"))]
use std::panic::{AssertUnwindSafe, catch_unwind};
#[cfg(any(feature = "metal", feature = "cpu", test))]
use std::path::Path;

use anyhow::Result;
use tokio::sync::mpsc::UnboundedSender;

#[cfg(feature = "cpu")]
use crate::backend::cpu::CpuBackend;
#[cfg(feature = "cuda")]
pub use crate::backend::cuda::bootstrap::{
    InferenceEngineOptions, ModelType, ServerRuntimeConfig, detect_model_type,
};
#[cfg(feature = "metal")]
use crate::backend::metal::{
    MetalBackend, MetalSchedulerHandle, spawn_metal_scheduler_handle_from_path,
};
#[cfg(any(feature = "metal", feature = "cpu"))]
use crate::backend::runtime::StopChunkProcessor;
#[cfg(any(feature = "metal", feature = "cpu"))]
use crate::backend::{InferenceBackend, StreamStopMatched, StreamingInferenceBackend};
use crate::request_handle::RequestHandle;
use crate::sampler::SamplingParams;
use crate::scheduler::{IncomingRequest, RequestPriority};
use crate::session_persistence::SessionPersistence;

/// Truncate at the first occurrence of any stop string (OpenAI-compatible).
/// Returns the prefix of `text` up to (but not including) the earliest stop.
#[cfg(any(feature = "metal", feature = "cpu", test))]
fn truncate_at_first_stop(text: &str, stops: &[String]) -> Option<String> {
    let mut earliest = None::<usize>;
    for s in stops {
        let s = s.as_str();
        if s.is_empty() {
            continue;
        }
        if let Some(pos) = text.find(s) {
            earliest = Some(match earliest {
                None => pos,
                Some(e) => std::cmp::min(e, pos),
            });
        }
    }
    earliest.map(|pos| text[..pos].to_string())
}

#[cfg(any(feature = "metal", feature = "cpu", test))]
fn model_id_from_path(model_path: &str) -> String {
    Path::new(model_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(model_path)
        .to_string()
}

impl<H: RequestHandle> RequestHandleInferenceEngine<H> {
    /// Adopt a previously-spawned `RequestHandle` (e.g. the CUDA scheduler
    /// or the Metal runtime). Caller owns any thread join handle / guard
    /// that backs the underlying scheduler.
    pub fn from_handle(model_id: String, handle: H) -> Self {
        Self { model_id, handle }
    }
}

#[cfg(any(feature = "metal", feature = "cpu"))]
fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    match payload.downcast::<String>() {
        Ok(msg) => *msg,
        Err(payload) => match payload.downcast::<&'static str>() {
            Ok(msg) => (*msg).to_string(),
            Err(_) => "unknown panic payload".to_string(),
        },
    }
}

#[cfg(any(feature = "metal", feature = "cpu", test))]
fn parse_finish_reason(finish_reason: &str) -> FinishReason {
    match finish_reason {
        "length" => FinishReason::Length,
        _ => FinishReason::Stop,
    }
}

pub struct CompletionRequest {
    pub prompt: String,
    pub max_tokens: usize,
    pub sampling: SamplingParams,
    /// Stop generation when output ends with any of these strings (OpenAI-compatible).
    pub stop: Option<Vec<String>>,
    /// Return per-token log-probabilities (greedy sampling only).
    pub logprobs: bool,
    /// Optional client-supplied session identifier used for sticky routing /
    /// prefix-cache affinity. Forwarded onto `IncomingRequest::session_id`
    /// when this request is routed through a `RequestHandle`. CLI agent
    /// callers may populate this; otherwise leave `None`.
    pub session_id: Option<crate::types::SessionId>,
    /// Parent tracing context to attach to the scheduler-side request.
    /// Forwarded onto `IncomingRequest::trace_context`. `None` for
    /// non-traced callers.
    pub trace_context: Option<fastrace::collector::SpanContext>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FinishReason {
    Length,
    Stop,
}

impl FinishReason {
    pub(crate) fn as_openai_str(self) -> &'static str {
        match self {
            Self::Length => "length",
            Self::Stop => "stop",
        }
    }
}

pub struct CompletionOutput {
    pub text: String,
    pub finish_reason: FinishReason,
    pub usage: TokenUsage,
    /// Per-token log-probabilities (greedy only). Empty if logprobs not requested.
    pub token_logprobs: Vec<f32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

pub struct CompletionStreamDelta {
    pub text_delta: String,
    pub finish_reason: Option<FinishReason>,
    pub usage: Option<TokenUsage>,
    /// Log-probability of the generated token (greedy only, None otherwise).
    #[allow(dead_code)]
    pub logprob: Option<f32>,
}

impl CompletionStreamDelta {
    /// Create a text delta (no finish, no logprob).
    pub fn text(s: String) -> Self {
        Self {
            text_delta: s,
            finish_reason: None,
            usage: None,
            logprob: None,
        }
    }
}

pub trait InferenceEngine: Send {
    /// Returns the model identifier (e.g. `"Qwen3-8B"`).
    fn model_id(&self) -> &str;

    /// Run a complete generation request synchronously and return the full output.
    fn complete(&mut self, req: CompletionRequest) -> Result<CompletionOutput>;

    /// Run a generation request, streaming token deltas through `tx` as they are produced.
    fn complete_stream(
        &mut self,
        req: CompletionRequest,
        tx: UnboundedSender<CompletionStreamDelta>,
    ) -> Result<()>;
}

// ============================================================================
// Public engine constructors
// ============================================================================

#[cfg(any(feature = "metal", feature = "cpu"))]
pub struct BackendInferenceEngine<B: InferenceBackend> {
    model_id: String,
    backend: B,
}

pub struct RequestHandleInferenceEngine<H: RequestHandle> {
    model_id: String,
    handle: H,
}

#[cfg(feature = "metal")]
impl BackendInferenceEngine<MetalBackend> {
    #[allow(dead_code)]
    fn load(model_path: &str) -> Result<Self> {
        let mut backend = MetalBackend::new();
        backend.load(Path::new(model_path))?;
        Ok(Self {
            model_id: model_id_from_path(model_path),
            backend,
        })
    }
}

#[cfg(feature = "metal")]
impl RequestHandleInferenceEngine<MetalSchedulerHandle> {
    fn load(model_path: &str) -> Result<Self> {
        let handle = spawn_metal_scheduler_handle_from_path(model_path, 0)?;
        Ok(Self {
            model_id: model_id_from_path(model_path),
            handle,
        })
    }
}

#[cfg(feature = "cpu")]
impl BackendInferenceEngine<CpuBackend> {
    fn load(model_path: &str) -> Result<Self> {
        let mut backend = CpuBackend::new();
        backend.load(Path::new(model_path))?;
        Ok(Self {
            model_id: model_id_from_path(model_path),
            backend,
        })
    }
}

#[cfg(any(feature = "metal", feature = "cpu"))]
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
    ///
    /// Two design points the previous revision (70e2776) got wrong:
    ///
    /// 1. **Stop detection must scan the unsent suffix, not just the
    ///    accumulated tail.** The default `StreamingInferenceBackend`
    ///    impl emits the entire completion as one chunk ([backend.rs:62]),
    ///    and real backends can produce stops that span chunk boundaries.
    ///    We route through [`StopChunkProcessor`] — the shared helper
    ///    used by the serial-runtime and Metal paths ([backend/runtime.rs:253]) —
    ///    which scans for the earliest stop and withholds `max_stop_len - 1`
    ///    bytes so a marker crossing a chunk boundary isn't leaked.
    ///
    /// 2. **Matched text stops must short-circuit decode and still carry
    ///    real usage.** The callback returns `Err(StreamStopMatched)` once
    ///    the shared processor flips `hit_stop()`. Backends treat that
    ///    sentinel as graceful termination and still return their final
    ///    `GenerateResult`, so the REPL keeps seeing exact usage in the
    ///    terminal delta ([crates/cli/src/repl.rs:787]).
    ///
    /// Dropping `tx` (rx disconnected) still propagates back through the
    /// callback as an `Err` — the REPL's Ctrl-C path relies on that
    /// (see [crates/cli/src/repl.rs:646]).
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

        // Cancel path: rx dropped mid-generation. Backend exited via the
        // `Err` we threaded through `on_chunk`. Return Ok — the REPL
        // (or any other tokio consumer) already handled the cancel
        // signal itself.
        if consumer_dropped.get() {
            return Ok(());
        }

        // Backend completed naturally or short-circuited on a matched
        // text stop and still returned its final usage counters. Flush any
        // bytes still held back by the max_stop_len - 1 safety window, then
        // emit the final delta.
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

impl<H: RequestHandle> RequestHandleInferenceEngine<H> {
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

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
pub enum LoadedInferenceEngine {
    /// Unified CUDA variant: drives the multi-request scheduler runtime
    /// through the same `RequestHandle` contract as Metal. The held
    /// `SchedulerRuntimeGuard` keeps the scheduler thread joined on drop.
    #[cfg(feature = "cuda")]
    Cuda {
        engine: RequestHandleInferenceEngine<crate::scheduler::SchedulerHandle>,
        _guard: crate::backend::cuda::bootstrap::SchedulerRuntimeGuard,
    },
    #[cfg(feature = "metal")]
    Metal(RequestHandleInferenceEngine<MetalSchedulerHandle>),
    #[cfg(feature = "cpu")]
    Cpu(BackendInferenceEngine<CpuBackend>),
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
impl LoadedInferenceEngine {
    pub fn load(model_path: &str, enable_cuda_graph: bool) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            return Self::load_with_options(
                model_path,
                42,
                InferenceEngineOptions { enable_cuda_graph },
            );
        }

        #[cfg(all(not(feature = "cuda"), feature = "metal"))]
        {
            let _ = enable_cuda_graph;
            return Ok(Self::Metal(RequestHandleInferenceEngine::load(model_path)?));
        }

        #[cfg(all(not(feature = "cuda"), not(feature = "metal"), feature = "cpu"))]
        {
            let _ = enable_cuda_graph;
            return Ok(Self::Cpu(BackendInferenceEngine::load(model_path)?));
        }

        #[allow(unreachable_code)]
        {
            let _ = (model_path, enable_cuda_graph);
            anyhow::bail!("no inference backend enabled")
        }
    }

    #[cfg(feature = "cuda")]
    pub fn load_with_options(
        model_path: &str,
        seed: u64,
        options: InferenceEngineOptions,
    ) -> Result<Self> {
        // All CUDA loads route through the unified multi-request scheduler
        // runtime via the `RequestHandle` contract — same path the HTTP
        // server uses through `spawn_scheduler_handle_from_path`.
        let runtime = crate::backend::cuda::bootstrap::ServerRuntimeConfig {
            engine: options,
            seed,
            ..Default::default()
        };
        let metrics = crate::metrics::ServerMetrics::new("");
        let (handle, guard) = crate::backend::cuda::bootstrap::spawn_scheduler_handle_from_path(
            model_path, runtime, metrics,
        )?;
        let model_id = handle.model_id().to_string();
        Ok(Self::Cuda {
            engine: RequestHandleInferenceEngine::from_handle(model_id, handle),
            _guard: guard,
        })
    }

    pub fn backend_name(&self) -> &'static str {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda { .. } => "cuda",
            #[cfg(feature = "metal")]
            Self::Metal(_) => "metal",
            #[cfg(feature = "cpu")]
            Self::Cpu(_) => "cpu",
        }
    }
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
impl InferenceEngine for LoadedInferenceEngine {
    fn model_id(&self) -> &str {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda { engine, .. } => engine.model_id(),
            #[cfg(feature = "metal")]
            Self::Metal(engine) => engine.model_id(),
            #[cfg(feature = "cpu")]
            Self::Cpu(engine) => engine.model_id(),
        }
    }

    fn complete(&mut self, req: CompletionRequest) -> Result<CompletionOutput> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda { engine, .. } => engine.complete(req),
            #[cfg(feature = "metal")]
            Self::Metal(engine) => engine.complete(req),
            #[cfg(feature = "cpu")]
            Self::Cpu(engine) => engine.complete(req),
        }
    }

    fn complete_stream(
        &mut self,
        req: CompletionRequest,
        tx: UnboundedSender<CompletionStreamDelta>,
    ) -> Result<()> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda { engine, .. } => engine.complete_stream(req, tx),
            #[cfg(feature = "metal")]
            Self::Metal(engine) => engine.complete_stream(req, tx),
            #[cfg(feature = "cpu")]
            Self::Cpu(engine) => engine.complete_stream(req, tx),
        }
    }
}

#[cfg(any(feature = "metal", feature = "cpu"))]
impl<B: InferenceBackend> SessionPersistence for BackendInferenceEngine<B> {}

impl<H: RequestHandle> SessionPersistence for RequestHandleInferenceEngine<H> {}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
impl SessionPersistence for LoadedInferenceEngine {}

#[cfg(test)]
mod tests {
    use super::{FinishReason, model_id_from_path, parse_finish_reason, truncate_at_first_stop};

    #[test]
    fn test_truncate_at_first_stop() {
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
        assert_eq!(truncate_at_first_stop("", &stops), None);
        assert_eq!(
            truncate_at_first_stop("a\n\nbEND", &stops),
            Some("a".to_string())
        );
        let stops_nl: Vec<String> = vec!["\n".into()];
        assert_eq!(
            truncate_at_first_stop("hello\nworld", &stops_nl),
            Some("hello".to_string())
        );
        assert_eq!(
            truncate_at_first_stop("ab", &["ab".to_string()]),
            Some(String::new())
        );
    }

    /// Regression test for codex review 0da212f/97c1a95 High: before this
    /// fix, `BackendInferenceEngine<Metal|Cpu>::complete_stream` called
    /// `self.complete(req)?` — a blocking full generation — and only
    /// touched `tx` at the end. Dropping `rx` mid-generation had no
    /// effect: the worker thread blocked until completion. The REPL's
    /// Ctrl-C path at `crates/cli/src/repl.rs:646` (which relies on
    /// `tx.send` failing when `rx` is dropped) was a lie on Metal + CPU.
    ///
    /// This test drops the receiver after zero chunks read, then
    /// asserts the mock backend's chunk counter stops at 1 (not 10) —
    /// i.e. the `on_chunk` callback propagated the `rx-disconnected`
    /// error back through `generate_stream`, short-circuiting the loop.
    #[cfg(any(feature = "metal", feature = "cpu"))]
    #[test]
    fn backend_complete_stream_short_circuits_when_rx_dropped() {
        use super::{BackendInferenceEngine, CompletionRequest, InferenceEngine};
        use crate::backend::{GenerateResult, InferenceBackend, StreamingInferenceBackend};
        use crate::sampler::SamplingParams;
        use std::path::Path;
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use tokio::sync::mpsc;

        #[derive(Clone)]
        struct CountingMock {
            chunks_attempted: Arc<AtomicUsize>,
        }

        impl InferenceBackend for CountingMock {
            fn load(&mut self, _p: &Path) -> anyhow::Result<()> {
                Ok(())
            }
            fn generate(
                &self,
                _prompt: &str,
                _params: &SamplingParams,
            ) -> anyhow::Result<GenerateResult> {
                unreachable!("test exercises streaming path only")
            }
            fn name(&self) -> &'static str {
                "counting-mock"
            }
        }

        impl StreamingInferenceBackend for CountingMock {
            fn generate_stream<F>(
                &self,
                _prompt: &str,
                _params: &SamplingParams,
                mut on_chunk: F,
            ) -> anyhow::Result<GenerateResult>
            where
                F: FnMut(&str) -> anyhow::Result<()>,
            {
                for _ in 0..10 {
                    self.chunks_attempted.fetch_add(1, Ordering::Relaxed);
                    on_chunk("x")?; // returns Err the instant tx fails
                }
                Ok(GenerateResult {
                    text: "xxxxxxxxxx".into(),
                    prompt_tokens: 1,
                    completion_tokens: 10,
                    finish_reason: "length".into(),
                    ttft_ms: 0.0,
                    prompt_tps: 0.0,
                    generation_tps: 0.0,
                    total_time_ms: 0.0,
                })
            }
        }

        let counter = Arc::new(AtomicUsize::new(0));
        let mut engine = BackendInferenceEngine {
            model_id: "counting-mock".into(),
            backend: CountingMock {
                chunks_attempted: counter.clone(),
            },
        };

        let (tx, rx) = mpsc::unbounded_channel();
        drop(rx); // simulate REPL cancel before the first chunk.

        let res = engine.complete_stream(
            CompletionRequest {
                prompt: "hi".into(),
                max_tokens: 10,
                sampling: SamplingParams::default(),
                stop: None,
                logprobs: false,
                session_id: None,
                trace_context: None,
            },
            tx,
        );

        assert!(res.is_ok(), "consumer-dropped is not an error");
        assert_eq!(
            counter.load(Ordering::Relaxed),
            1,
            "generate_stream must exit after the first failed tx.send, \
             not keep looping through all 10 chunks"
        );
    }

    /// Normal completion: no stop sequences, reader intact → backend
    /// runs to completion, each chunk flows through, and the final
    /// delta carries `finish_reason` + `usage`.
    #[cfg(any(feature = "metal", feature = "cpu"))]
    #[tokio::test]
    async fn backend_complete_stream_emits_all_chunks_and_finish_marker() {
        use super::{BackendInferenceEngine, CompletionRequest, InferenceEngine};
        use crate::backend::{GenerateResult, InferenceBackend, StreamingInferenceBackend};
        use crate::sampler::SamplingParams;
        use std::path::Path;
        use tokio::sync::mpsc;

        struct FullRunMock;
        impl InferenceBackend for FullRunMock {
            fn load(&mut self, _p: &Path) -> anyhow::Result<()> {
                Ok(())
            }
            fn generate(
                &self,
                _prompt: &str,
                _params: &SamplingParams,
            ) -> anyhow::Result<GenerateResult> {
                unreachable!()
            }
            fn name(&self) -> &'static str {
                "full-run-mock"
            }
        }
        impl StreamingInferenceBackend for FullRunMock {
            fn generate_stream<F>(
                &self,
                _prompt: &str,
                _params: &SamplingParams,
                mut on_chunk: F,
            ) -> anyhow::Result<GenerateResult>
            where
                F: FnMut(&str) -> anyhow::Result<()>,
            {
                on_chunk("hel")?;
                on_chunk("lo")?;
                Ok(GenerateResult {
                    text: "hello".into(),
                    prompt_tokens: 1,
                    completion_tokens: 2,
                    finish_reason: "length".into(),
                    ttft_ms: 0.0,
                    prompt_tps: 0.0,
                    generation_tps: 0.0,
                    total_time_ms: 0.0,
                })
            }
        }

        let mut engine = BackendInferenceEngine {
            model_id: "full-run-mock".into(),
            backend: FullRunMock,
        };

        let (tx, mut rx) = mpsc::unbounded_channel();
        let res = engine.complete_stream(
            CompletionRequest {
                prompt: "p".into(),
                max_tokens: 8,
                sampling: SamplingParams::default(),
                stop: None,
                logprobs: false,
                session_id: None,
                trace_context: None,
            },
            tx,
        );
        assert!(res.is_ok());

        let mut text_parts: Vec<String> = Vec::new();
        let mut finish: Option<FinishReason> = None;
        while let Ok(chunk) = rx.try_recv() {
            if chunk.finish_reason.is_some() {
                finish = chunk.finish_reason;
            }
            if !chunk.text_delta.is_empty() {
                text_parts.push(chunk.text_delta);
            }
        }
        assert_eq!(text_parts.concat(), "hello");
        assert_eq!(finish, Some(FinishReason::Length));
    }

    #[cfg(any(feature = "metal", test))]
    #[test]
    fn request_handle_engine_complete_collects_streamed_deltas() {
        use super::{
            CompletionOutput, CompletionRequest, CompletionStreamDelta, InferenceEngine,
            RequestHandleInferenceEngine, TokenUsage,
        };
        use crate::request_handle::{RequestHandle, SubmitError};
        use crate::sampler::SamplingParams;
        use crate::scheduler::IncomingRequest;

        struct MockHandle;

        impl RequestHandle for MockHandle {
            fn submit(&self, req: IncomingRequest) -> Result<(), SubmitError> {
                let _ = req.delta_tx.send(CompletionStreamDelta {
                    text_delta: "hel".into(),
                    finish_reason: None,
                    usage: None,
                    logprob: None,
                });
                let _ = req.delta_tx.send(CompletionStreamDelta {
                    text_delta: "lo".into(),
                    finish_reason: None,
                    usage: None,
                    logprob: None,
                });
                let _ = req.delta_tx.send(CompletionStreamDelta {
                    text_delta: String::new(),
                    finish_reason: Some(FinishReason::Stop),
                    usage: Some(TokenUsage {
                        prompt_tokens: 3,
                        completion_tokens: 2,
                        total_tokens: 5,
                    }),
                    logprob: None,
                });
                Ok(())
            }

            fn model_id(&self) -> &str {
                "mock-handle"
            }
        }

        let mut engine = RequestHandleInferenceEngine {
            model_id: "mock-handle".into(),
            handle: MockHandle,
        };
        let output = engine
            .complete(CompletionRequest {
                prompt: "hi".into(),
                max_tokens: 2,
                sampling: SamplingParams::default(),
                stop: None,
                logprobs: false,
                session_id: None,
                trace_context: None,
            })
            .expect("complete");

        let CompletionOutput {
            text,
            finish_reason,
            usage,
            token_logprobs,
        } = output;
        assert_eq!(text, "hello");
        assert_eq!(finish_reason, FinishReason::Stop);
        assert_eq!(
            usage,
            TokenUsage {
                prompt_tokens: 3,
                completion_tokens: 2,
                total_tokens: 5,
            }
        );
        assert!(token_logprobs.is_empty());
    }

    /// Regression for codex review 70e2776 High #1 — stop *inside* a
    /// single chunk. The default `StreamingInferenceBackend` impl sends
    /// the whole completion as one chunk; the old end-of-buffer check
    /// would only fire when the chunk *ended* with the stop, so a stop
    /// mid-chunk leaked the raw marker + trailing bytes. With
    /// `StopChunkProcessor::push_chunk` scanning the unsent suffix,
    /// everything after the stop is withheld.
    #[cfg(any(feature = "metal", feature = "cpu"))]
    #[tokio::test]
    async fn backend_complete_stream_stop_inside_single_chunk() {
        use super::{BackendInferenceEngine, CompletionRequest, InferenceEngine, TokenUsage};
        use crate::backend::{GenerateResult, InferenceBackend, StreamingInferenceBackend};
        use crate::sampler::SamplingParams;
        use std::path::Path;
        use tokio::sync::mpsc;

        struct SingleChunkMock;
        impl InferenceBackend for SingleChunkMock {
            fn load(&mut self, _p: &Path) -> anyhow::Result<()> {
                Ok(())
            }
            fn generate(
                &self,
                _prompt: &str,
                _params: &SamplingParams,
            ) -> anyhow::Result<GenerateResult> {
                unreachable!()
            }
            fn name(&self) -> &'static str {
                "single-chunk-mock"
            }
        }
        impl StreamingInferenceBackend for SingleChunkMock {
            fn generate_stream<F>(
                &self,
                _prompt: &str,
                _params: &SamplingParams,
                mut on_chunk: F,
            ) -> anyhow::Result<GenerateResult>
            where
                F: FnMut(&str) -> anyhow::Result<()>,
            {
                match on_chunk("hello<|im_end|>trailing") {
                    Ok(()) => {}
                    Err(err)
                        if err
                            .downcast_ref::<crate::backend::StreamStopMatched>()
                            .is_some() => {}
                    Err(err) => return Err(err),
                }
                Ok(GenerateResult {
                    text: "hello<|im_end|>trailing".into(),
                    prompt_tokens: 3,
                    completion_tokens: 7,
                    finish_reason: "stop".into(),
                    ttft_ms: 0.0,
                    prompt_tps: 0.0,
                    generation_tps: 0.0,
                    total_time_ms: 0.0,
                })
            }
        }

        let mut engine = BackendInferenceEngine {
            model_id: "single-chunk-mock".into(),
            backend: SingleChunkMock,
        };

        let (tx, mut rx) = mpsc::unbounded_channel();
        engine
            .complete_stream(
                CompletionRequest {
                    prompt: "p".into(),
                    max_tokens: 32,
                    sampling: SamplingParams::default(),
                    stop: Some(vec!["<|im_end|>".into()]),
                    logprobs: false,
                    session_id: None,
                    trace_context: None,
                },
                tx,
            )
            .unwrap();

        let mut text_parts: Vec<String> = Vec::new();
        let mut finish: Option<FinishReason> = None;
        let mut usage: Option<TokenUsage> = None;
        while let Ok(chunk) = rx.try_recv() {
            if chunk.finish_reason.is_some() {
                finish = chunk.finish_reason;
            }
            if chunk.usage.is_some() {
                usage = chunk.usage;
            }
            if !chunk.text_delta.is_empty() {
                text_parts.push(chunk.text_delta);
            }
        }
        let joined = text_parts.concat();
        assert_eq!(joined, "hello", "stop marker + trailing must be withheld");
        assert!(
            !joined.contains("<|im_end|>"),
            "raw stop marker must never reach the consumer"
        );
        assert_eq!(finish, Some(FinishReason::Stop));
        assert_eq!(
            usage,
            Some(TokenUsage {
                prompt_tokens: 3,
                completion_tokens: 7,
                total_tokens: 10,
            })
        );
    }

    /// Regression for codex review 70e2776 High #2 — stop *spanning*
    /// chunk boundaries. Before the fix, the first chunk's bytes were
    /// forwarded immediately and the stop was only detected on the
    /// chunk that completed it — by then the prefix had already been
    /// leaked. `StopChunkProcessor` withholds the last `max_stop_len-1`
    /// bytes of each chunk until the next one arrives.
    #[cfg(any(feature = "metal", feature = "cpu"))]
    #[tokio::test]
    async fn backend_complete_stream_stop_spanning_chunks() {
        use super::{BackendInferenceEngine, CompletionRequest, InferenceEngine, TokenUsage};
        use crate::backend::{GenerateResult, InferenceBackend, StreamingInferenceBackend};
        use crate::sampler::SamplingParams;
        use std::path::Path;
        use tokio::sync::mpsc;

        struct SplitChunkMock;
        impl InferenceBackend for SplitChunkMock {
            fn load(&mut self, _p: &Path) -> anyhow::Result<()> {
                Ok(())
            }
            fn generate(
                &self,
                _prompt: &str,
                _params: &SamplingParams,
            ) -> anyhow::Result<GenerateResult> {
                unreachable!()
            }
            fn name(&self) -> &'static str {
                "split-chunk-mock"
            }
        }
        impl StreamingInferenceBackend for SplitChunkMock {
            fn generate_stream<F>(
                &self,
                _prompt: &str,
                _params: &SamplingParams,
                mut on_chunk: F,
            ) -> anyhow::Result<GenerateResult>
            where
                F: FnMut(&str) -> anyhow::Result<()>,
            {
                on_chunk("hello<|im_")?;
                match on_chunk("end|>trail") {
                    Ok(()) => {}
                    Err(err)
                        if err
                            .downcast_ref::<crate::backend::StreamStopMatched>()
                            .is_some() => {}
                    Err(err) => return Err(err),
                }
                Ok(GenerateResult {
                    text: "hello<|im_end|>trail".into(),
                    prompt_tokens: 2,
                    completion_tokens: 5,
                    finish_reason: "stop".into(),
                    ttft_ms: 0.0,
                    prompt_tps: 0.0,
                    generation_tps: 0.0,
                    total_time_ms: 0.0,
                })
            }
        }

        let mut engine = BackendInferenceEngine {
            model_id: "split-chunk-mock".into(),
            backend: SplitChunkMock,
        };

        let (tx, mut rx) = mpsc::unbounded_channel();
        engine
            .complete_stream(
                CompletionRequest {
                    prompt: "p".into(),
                    max_tokens: 32,
                    sampling: SamplingParams::default(),
                    stop: Some(vec!["<|im_end|>".into()]),
                    logprobs: false,
                    session_id: None,
                    trace_context: None,
                },
                tx,
            )
            .unwrap();

        let mut text_parts: Vec<String> = Vec::new();
        let mut finish: Option<FinishReason> = None;
        let mut usage: Option<TokenUsage> = None;
        while let Ok(chunk) = rx.try_recv() {
            if chunk.finish_reason.is_some() {
                finish = chunk.finish_reason;
            }
            if chunk.usage.is_some() {
                usage = chunk.usage;
            }
            if !chunk.text_delta.is_empty() {
                text_parts.push(chunk.text_delta);
            }
        }
        let joined = text_parts.concat();
        assert_eq!(
            joined, "hello",
            "stop split across chunks must still strip the marker"
        );
        assert!(
            !joined.contains("<|im_") && !joined.contains("im_end") && !joined.contains("|>"),
            "no partial stop-marker bytes may leak (got {joined:?})",
        );
        assert_eq!(finish, Some(FinishReason::Stop));
        assert_eq!(
            usage,
            Some(TokenUsage {
                prompt_tokens: 2,
                completion_tokens: 5,
                total_tokens: 7,
            })
        );
    }

    /// Regression for codex review 2026-04-20 P1 — once a streamed text
    /// stop is matched, the consumer must stop seeing bytes, but final
    /// usage must still come from the backend's real completion result.
    #[cfg(any(feature = "metal", feature = "cpu"))]
    #[tokio::test]
    async fn backend_complete_stream_text_stop_keeps_real_usage() {
        use super::{BackendInferenceEngine, CompletionRequest, InferenceEngine, TokenUsage};
        use crate::backend::{GenerateResult, InferenceBackend, StreamingInferenceBackend};
        use crate::sampler::SamplingParams;
        use std::path::Path;
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use tokio::sync::mpsc;

        #[derive(Clone)]
        struct CountingStopMock {
            chunks_attempted: Arc<AtomicUsize>,
        }

        impl InferenceBackend for CountingStopMock {
            fn load(&mut self, _p: &Path) -> anyhow::Result<()> {
                Ok(())
            }
            fn generate(
                &self,
                _prompt: &str,
                _params: &SamplingParams,
            ) -> anyhow::Result<GenerateResult> {
                unreachable!()
            }
            fn name(&self) -> &'static str {
                "counting-stop-mock"
            }
        }

        impl StreamingInferenceBackend for CountingStopMock {
            fn generate_stream<F>(
                &self,
                _prompt: &str,
                _params: &SamplingParams,
                mut on_chunk: F,
            ) -> anyhow::Result<GenerateResult>
            where
                F: FnMut(&str) -> anyhow::Result<()>,
            {
                for chunk in ["hello<|im_end|>waste", "never-sent"] {
                    self.chunks_attempted.fetch_add(1, Ordering::Relaxed);
                    if let Err(err) = on_chunk(chunk) {
                        if err
                            .downcast_ref::<crate::backend::StreamStopMatched>()
                            .is_some()
                        {
                            return Ok(GenerateResult {
                                text: "hello<|im_end|>waste".into(),
                                prompt_tokens: 4,
                                completion_tokens: 5,
                                finish_reason: "stop".into(),
                                ttft_ms: 0.0,
                                prompt_tps: 0.0,
                                generation_tps: 0.0,
                                total_time_ms: 0.0,
                            });
                        }
                        return Err(err);
                    }
                }
                Ok(GenerateResult {
                    text: "hello<|im_end|>waste never-sent".into(),
                    prompt_tokens: 4,
                    completion_tokens: 9,
                    finish_reason: "length".into(),
                    ttft_ms: 0.0,
                    prompt_tps: 0.0,
                    generation_tps: 0.0,
                    total_time_ms: 0.0,
                })
            }
        }

        let chunks_attempted = Arc::new(AtomicUsize::new(0));
        let mut engine = BackendInferenceEngine {
            model_id: "counting-stop-mock".into(),
            backend: CountingStopMock {
                chunks_attempted: Arc::clone(&chunks_attempted),
            },
        };

        let (tx, mut rx) = mpsc::unbounded_channel();
        engine
            .complete_stream(
                CompletionRequest {
                    prompt: "p".into(),
                    max_tokens: 32,
                    sampling: SamplingParams::default(),
                    stop: Some(vec!["<|im_end|>".into()]),
                    logprobs: false,
                    session_id: None,
                    trace_context: None,
                },
                tx,
            )
            .unwrap();

        let mut text = String::new();
        let mut finish = None;
        let mut usage = None;
        while let Ok(chunk) = rx.try_recv() {
            text.push_str(&chunk.text_delta);
            if chunk.finish_reason.is_some() {
                finish = chunk.finish_reason;
                usage = chunk.usage;
                break;
            }
        }

        assert_eq!(text, "hello");
        assert_eq!(finish, Some(FinishReason::Stop));
        assert_eq!(
            usage,
            Some(TokenUsage {
                prompt_tokens: 4,
                completion_tokens: 5,
                total_tokens: 9,
            })
        );
        assert_eq!(chunks_attempted.load(Ordering::Relaxed), 1);
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
    fn parse_finish_reason_defaults_to_stop() {
        assert_eq!(parse_finish_reason("length"), FinishReason::Length);
        assert_eq!(parse_finish_reason("stop"), FinishReason::Stop);
        assert_eq!(parse_finish_reason("tool_calls"), FinishReason::Stop);
    }
}
