//! Frontend serve layer for [`infer_core::Engine`].
//!
//! This crate implements the vLLM-V1 / ideal-architecture **frontend-||-engine**
//! split: CPU frontend work (request ingress, completion fan-out) is decoupled
//! from the engine loop, which owns the [`Engine`] on a dedicated background
//! thread. The frontend talks to the engine thread only through channels, so a
//! caller never blocks the engine while preparing or collecting a request.
//!
//! The API is intentionally `std`-only (threads + `mpsc`), no async runtime:
//!
//! - [`ServeHandle::spawn`] starts the engine thread owning an `Engine<E, K>`.
//! - [`ServeHandle::submit`] hands a prompt to the engine and returns a
//!   [`RequestTicket`] carrying the engine-assigned [`RequestHandle`] plus a
//!   private back-channel for that request's completion.
//! - [`ServeHandle::collect`] / [`RequestTicket::collect`] blocks until the
//!   request finishes and returns its [`CompletedRequest`] (generated tokens,
//!   finish reason).
//!
//! The engine thread is an OS good citizen: it only spins while there is work,
//! and parks on the submit channel (`recv_timeout`) when fully idle instead of
//! busy-looping.
//!
//! Internal layout (pure-reorganization split, same numerics):
//!
//! - [`execution`] — the HOT engine loop (`engine_loop` + `Submission`).
//! - [`http`] — the axum OpenAI v1 router and route handlers.
//! - [`tokenizer`] — the tokenizer / chat-template adapter (COLD).
//! - [`schema`] — the OpenAI wire types and `ApiError` (COLD).

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow};
use infer_core::{CompletedRequest, Engine, RequestHandle, SchedulerConfig};
use infer_plan::{ForwardPlan, SamplingParams, SlotToken, StepOutput};
use infer_seam::{BackendExecutor, KvPool, PollResult};

mod coordinator;
mod execution;
mod metrics;
pub mod multimodal;
pub mod multiproc_relay;
mod schema;
mod sse_util;
mod tokenizer;

use execution::{ControlMessage, Submission, engine_loop};
pub use execution::{CounterSnapshot, StreamItem};

pub use coordinator::coordinator_router;

/// In-process channel for multimodal requests from coordinator → relay driver.
pub struct LocalMultimodalRequest {
    pub prompt_tokens: Vec<u32>,
    pub images: Vec<infer_plan::MultimodalImage>,
    pub max_tokens: usize,
    pub sampling: infer_plan::SamplingParams,
    pub response_tx: tokio::sync::oneshot::Sender<multiproc_relay::RelayCompletionDelta>,
}

pub type LocalMultimodalTx = std::sync::mpsc::SyncSender<LocalMultimodalRequest>;
pub(crate) type LocalMultimodalRx = std::sync::mpsc::Receiver<LocalMultimodalRequest>;
pub use multiproc_relay::{
    PendingRelayCoordinator, RelayChannel, RelayCompletionDelta, RelayCoordinator, RelayEnvelope,
    RelayWorker, TcpChannel, WireRequest, WireStats, broadcast_tick, set_tick_broadcaster,
    tick_broadcaster_installed,
};
pub use schema::{
    ChatCompletionRequest, ChatCompletionResponse, ChatContent, ChatContentPart, ChatMessage,
    CompletionRequest, CompletionResponse,
};
pub use tokenizer::OpenAiTokenizer;

/// Handle to a running engine thread.
///
/// Owns the engine thread's `JoinHandle` and the submit channel. Dropping the
/// handle closes the submit channel; the engine thread then drains any
/// in-flight work, exits its loop, and joins. A [`ServeShutdown`] token can
/// request a process-level abort instead, used by HTTP Ctrl-C shutdown. The
/// generic parameters mirror [`Engine`]; both must be `Send + 'static` because
/// the engine lives on a separate thread.
pub struct ServeHandle<E: BackendExecutor, K: KvPool> {
    submit_tx: Option<Sender<Submission>>,
    /// Out-of-band control channel: runs a `FnOnce(&mut Engine<E, K>)` on the
    /// engine thread between steps. The OPD control surface (raw-logits forward,
    /// weight offload/reload, student-LoRA re-merge + prefix-cache invalidation)
    /// reaches the thread-owned engine through here without crossing the request
    /// hot path.
    control_tx: Option<Sender<ControlMessage<E, K>>>,
    join: Option<JoinHandle<()>>,
    /// Latest scheduler counters, republished by the engine loop each tick.
    counters: Arc<Mutex<CounterSnapshot>>,
    /// Backend-requested frontend live-request cap.
    max_live_requests: usize,
    /// Request tickets currently handed out and not yet dropped.
    live_requests: Arc<AtomicUsize>,
    _backend: std::marker::PhantomData<fn() -> (E, K)>,
}

/// Shared server shutdown token observed by the HTTP signal handler and the
/// engine thread.
#[derive(Clone, Debug)]
pub struct ServeShutdown {
    requested: Arc<AtomicBool>,
}

impl ServeShutdown {
    #[must_use]
    pub fn new() -> Self {
        Self {
            requested: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn request(&self) {
        self.requested.store(true, Ordering::Release);
    }

    #[must_use]
    pub fn is_requested(&self) -> bool {
        self.requested.load(Ordering::Acquire)
    }

    #[must_use]
    pub fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.requested)
    }
}

impl Default for ServeShutdown {
    fn default() -> Self {
        Self::new()
    }
}

/// Receipt for a submitted request.
///
/// Holds the engine-assigned [`RequestHandle`] and the private back-channel on
/// which that request's [`CompletedRequest`] will arrive. Collect via [`RequestTicket::collect`].
pub struct RequestTicket {
    handle: RequestHandle,
    completion_rx: Receiver<CompletedRequest>,
    live_requests: Arc<AtomicUsize>,
}

impl RequestTicket {
    /// Return the engine-assigned handle for this request.
    #[must_use]
    pub fn handle(&self) -> RequestHandle {
        self.handle
    }

    /// Block until this request completes and return its result.
    ///
    /// Returns an error if the engine thread exited before delivering the
    /// completion (e.g. the engine panicked or was shut down).
    pub fn collect(self) -> Result<CompletedRequest> {
        self.completion_rx.recv().map_err(|_| {
            anyhow!(
                "engine thread closed before request {} completed",
                self.handle.id()
            )
        })
    }

    /// Block up to `timeout` for this request to complete.
    ///
    /// On timeout the ticket is returned in the `Err` so the caller can retry.
    pub fn collect_timeout(
        self,
        timeout: Duration,
    ) -> std::result::Result<CompletedRequest, CollectTimeout> {
        match self.completion_rx.recv_timeout(timeout) {
            Ok(completed) => Ok(completed),
            Err(RecvTimeoutError::Timeout) => Err(CollectTimeout::Pending(self)),
            Err(RecvTimeoutError::Disconnected) => Err(CollectTimeout::Closed(self.handle)),
        }
    }
}

impl Drop for RequestTicket {
    fn drop(&mut self) {
        self.live_requests.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Outcome of a [`RequestTicket::collect_timeout`] that did not deliver a result.
pub enum CollectTimeout {
    /// The request is still running; the ticket is returned to retry.
    Pending(RequestTicket),
    /// The engine thread closed before delivering the completion.
    Closed(RequestHandle),
}

fn submit_trace_enabled() -> bool {
    std::env::var_os("ARLE_SERVE_SUBMIT_TRACE").is_some()
}

impl<E, K> ServeHandle<E, K>
where
    E: BackendExecutor + Send + 'static,
    K: KvPool + Send + 'static,
{
    /// Spawn an engine thread owning `Engine::with_config(executor, kv, config)`.
    ///
    /// The returned handle is the only way to reach the engine: submit requests
    /// with [`ServeHandle::submit`] and collect results with
    /// [`ServeHandle::collect`].
    pub fn spawn(executor: E, kv: K, config: SchedulerConfig) -> Self {
        Self::spawn_with_shutdown(executor, kv, config, ServeShutdown::new())
    }

    /// Spawn an engine thread that also observes `shutdown` for process-level
    /// aborts.
    pub fn spawn_with_shutdown(
        executor: E,
        kv: K,
        config: SchedulerConfig,
        shutdown: ServeShutdown,
    ) -> Self {
        let (submit_tx, submit_rx) = mpsc::channel::<Submission>();
        let (control_tx, control_rx) = mpsc::channel::<ControlMessage<E, K>>();
        let counters = Arc::new(Mutex::new(CounterSnapshot::default()));
        let loop_counters = Arc::clone(&counters);
        let max_live_requests = executor.max_live_requests().max(1);
        let join = thread::Builder::new()
            .name("infer-engine".to_string())
            .spawn(move || {
                engine_loop(
                    Engine::with_config(executor, kv, config),
                    submit_rx,
                    control_rx,
                    loop_counters,
                    shutdown,
                )
            })
            .expect("spawn infer-engine thread");
        Self {
            submit_tx: Some(submit_tx),
            control_tx: Some(control_tx),
            join: Some(join),
            counters,
            max_live_requests,
            live_requests: Arc::new(AtomicUsize::new(0)),
            _backend: std::marker::PhantomData,
        }
    }
}

impl<E, K> ServeHandle<E, K>
where
    E: BackendExecutor + 'static,
    K: KvPool + 'static,
{
    /// Spawn an engine thread and build the engine inside that thread.
    ///
    /// This is for backends whose executor owns thread-affine handles. The
    /// builder itself must be sendable, but the constructed `E` and `K` never
    /// cross a thread boundary; they are created and consumed by the engine
    /// thread in place.
    pub fn spawn_with_engine_builder<B>(builder: B) -> Result<Self>
    where
        B: FnOnce() -> Result<Engine<E, K>> + Send + 'static,
    {
        Self::spawn_with_engine_builder_and_shutdown(builder, ServeShutdown::new())
    }

    /// Spawn an engine thread, build the engine inside that thread, and observe
    /// `shutdown` for process-level aborts.
    pub fn spawn_with_engine_builder_and_shutdown<B>(
        builder: B,
        shutdown: ServeShutdown,
    ) -> Result<Self>
    where
        B: FnOnce() -> Result<Engine<E, K>> + Send + 'static,
    {
        let (submit_tx, submit_rx) = mpsc::channel::<Submission>();
        let (control_tx, control_rx) = mpsc::channel::<ControlMessage<E, K>>();
        let (ready_tx, ready_rx) = mpsc::sync_channel::<std::result::Result<usize, String>>(1);
        let counters = Arc::new(Mutex::new(CounterSnapshot::default()));
        let loop_counters = Arc::clone(&counters);
        let join = thread::Builder::new()
            .name("infer-engine".to_string())
            .spawn(move || match builder() {
                Ok(mut engine) => {
                    if let Err(err) = engine.warmup() {
                        let _ = ready_tx.send(Err(err.to_string()));
                        return;
                    }
                    let max_live_requests = engine.max_live_requests();
                    let _ = ready_tx.send(Ok(max_live_requests));
                    engine_loop(engine, submit_rx, control_rx, loop_counters, shutdown);
                }
                Err(err) => {
                    let _ = ready_tx.send(Err(err.to_string()));
                }
            })
            .expect("spawn infer-engine thread");

        match ready_rx
            .recv()
            .map_err(|_| anyhow!("engine thread exited before signalling readiness"))?
        {
            Ok(max_live_requests) => Ok(Self {
                submit_tx: Some(submit_tx),
                control_tx: Some(control_tx),
                join: Some(join),
                counters,
                max_live_requests: max_live_requests.max(1),
                live_requests: Arc::new(AtomicUsize::new(0)),
                _backend: std::marker::PhantomData,
            }),
            Err(err) => {
                let _ = join.join();
                Err(anyhow!("engine build failed: {err}"))
            }
        }
    }

    /// Submit a prompt for generation and return a [`RequestTicket`].
    ///
    /// Blocks only until the engine thread assigns a [`RequestHandle`] (a single
    /// channel round-trip), not until the request finishes. Use the returned
    /// ticket to [`collect`](RequestTicket::collect) the result.
    pub fn submit(
        &self,
        prompt: Vec<u32>,
        max_tokens: usize,
        sampling: SamplingParams,
    ) -> Result<RequestTicket> {
        self.do_submit(prompt, max_tokens, sampling, None)
    }

    pub fn submit_streaming(
        &self,
        prompt: Vec<u32>,
        max_tokens: usize,
        sampling: SamplingParams,
    ) -> Result<(RequestTicket, Receiver<StreamItem>)> {
        let (stream_tx, stream_rx) = mpsc::channel::<StreamItem>();
        let ticket = self.do_submit(prompt, max_tokens, sampling, Some(stream_tx))?;
        Ok((ticket, stream_rx))
    }

    fn do_submit(
        &self,
        prompt: Vec<u32>,
        max_tokens: usize,
        sampling: SamplingParams,
        stream_tx: Option<mpsc::Sender<StreamItem>>,
    ) -> Result<RequestTicket> {
        let mode = if stream_tx.is_some() {
            "streaming"
        } else {
            "blocking"
        };
        self.acquire_live_request()?;
        let submit_tx = self.submit_tx.as_ref().ok_or_else(|| {
            self.release_live_request();
            anyhow!("ServeHandle already shut down")
        })?;
        let (handle_tx, handle_rx) = mpsc::channel::<RequestHandle>();
        let (completion_tx, completion_rx) = mpsc::channel::<CompletedRequest>();
        if submit_tx
            .send(Submission {
                prompt,
                max_tokens,
                sampling,
                handle_tx,
                completion_tx,
                stream_tx,
            })
            .is_err()
        {
            self.release_live_request();
            return Err(anyhow!("engine thread closed; cannot submit"));
        }
        let trace_start = submit_trace_enabled().then(Instant::now);
        let handle = match handle_rx.recv() {
            Ok(handle) => handle,
            Err(_) => {
                self.release_live_request();
                return Err(anyhow!(
                    "engine thread closed before assigning a request handle"
                ));
            }
        };
        if let Some(start) = trace_start {
            let wait_ms = start.elapsed().as_secs_f64() * 1000.0;
            log::info!(
                "[serve-submit] mode={mode} handle={} wait_ms={wait_ms:.1} live={} max_live={}",
                handle.id(),
                self.live_requests.load(Ordering::Acquire),
                self.max_live_requests
            );
        }
        Ok(RequestTicket {
            handle,
            completion_rx,
            live_requests: Arc::clone(&self.live_requests),
        })
    }

    /// Snapshot of the engine's live scheduler counters (active requests, queue
    /// depth, free KV pages), republished by the engine loop each tick. Defaults
    /// to zero before the first tick (or if the engine thread has gone).
    #[must_use]
    pub fn counters(&self) -> CounterSnapshot {
        self.counters.lock().map(|c| *c).unwrap_or_default()
    }

    fn acquire_live_request(&self) -> Result<()> {
        loop {
            let current = self.live_requests.load(Ordering::Acquire);
            anyhow::ensure!(
                current < self.max_live_requests,
                "server is busy: backend allows at most {} live request(s)",
                self.max_live_requests
            );
            if self
                .live_requests
                .compare_exchange(current, current + 1, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return Ok(());
            }
        }
    }

    fn release_live_request(&self) {
        self.live_requests.fetch_sub(1, Ordering::AcqRel);
    }

    /// Run `f` against the engine-thread-owned [`Engine`] and return its result.
    ///
    /// The closure executes on the engine thread (between scheduler steps), so it
    /// has exclusive `&mut Engine<E, K>` access — scheduler, RadixCache, *and*
    /// executor — without racing the request hot path. This is the engine-level
    /// out-of-band control seam: the OPD surface uses it when a control op must
    /// touch engine state the executor cannot reach, e.g. dropping the now-stale
    /// prefix cache atomically with a resident-weight LoRA re-merge. Blocks until
    /// the engine thread runs the closure and returns its value.
    pub fn run_on_engine<R, F>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&mut Engine<E, K>) -> R + Send + 'static,
        R: Send + 'static,
    {
        let control_tx = self
            .control_tx
            .as_ref()
            .ok_or_else(|| anyhow!("ServeHandle already shut down"))?;
        let (response_tx, response_rx) = mpsc::channel::<R>();
        control_tx
            .send(Box::new(move |engine: &mut Engine<E, K>| {
                // If the caller dropped the receiver, discard the result.
                let _ = response_tx.send(f(engine));
            }))
            .map_err(|_| anyhow!("engine thread closed; cannot run control closure"))?;
        response_rx
            .recv()
            .map_err(|_| anyhow!("engine thread closed before running control closure"))
    }

    /// Run `f` against the engine-thread-owned executor and return its result.
    ///
    /// Thin wrapper over [`Self::run_on_engine`] for the control surface that only
    /// needs `&mut E` (raw-logits forward, weight offload/reload, LoRA re-merge):
    /// the executor's non-serving methods the request/response channels cannot
    /// express. Blocks until the engine thread runs the closure and returns it.
    pub fn run_on_executor<R, F>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&mut E) -> R + Send + 'static,
        R: Send + 'static,
    {
        self.run_on_engine(move |engine| f(engine.executor_mut()))
    }

    /// Offload the engine's device weights to host RAM (OPD teacher weight
    /// time-share), returning the device bytes freed.
    ///
    /// Runs on the engine thread (between scheduler steps) via the out-of-band
    /// control seam, so the weight movement never races an in-flight forward
    /// step. Blocks until the round-trip completes.
    pub fn offload_engine_weights(&self) -> Result<usize> {
        self.run_on_executor(|executor| executor.offload_weights())?
    }

    /// Reload the engine's device weights from the host snapshot (OPD teacher
    /// weight time-share). Blocks until the H2D round-trip completes.
    pub fn reload_engine_weights(&self) -> Result<()> {
        self.run_on_executor(|executor| executor.reload_weights())?
    }

    /// Release the engine's inference forward scratch WITHOUT offloading weights or
    /// evicting KV (OPD rollout->writeback VRAM reclaim). Runs on the engine thread
    /// (between scheduler steps) via the out-of-band control seam, so the release
    /// never races an in-flight forward step. Blocks until the round-trip completes.
    pub fn release_inference_scratch(&self) -> Result<()> {
        self.run_on_executor(|executor| executor.release_inference_scratch())?
    }

    /// Drop the engine's KV pool WITHOUT offloading weights (OPD writeback
    /// headroom). Runs on the engine thread via the control seam, so it never
    /// races an in-flight forward. Blocks until the round-trip completes.
    pub fn release_kv_pool(&self) -> Result<()> {
        self.run_on_executor(|executor| executor.release_kv_pool())?
    }

    /// Re-acquire the engine's KV pool before the next rollout (after
    /// [`Self::release_kv_pool`]). Runs on the engine thread; blocks until done.
    pub fn ensure_kv_pool(&self) -> Result<()> {
        self.run_on_executor(|executor| executor.ensure_kv_pool())?
    }

    /// Close the submit channel and join the engine thread.
    ///
    /// The engine drains its in-flight and waiting work to completion before the
    /// thread exits, so any tickets already submitted still resolve. Called
    /// automatically on drop; exposed so callers can observe a panic in the
    /// engine thread.
    pub fn shutdown(mut self) -> thread::Result<()> {
        self.submit_tx.take();
        self.control_tx.take();
        match self.join.take() {
            Some(join) => join.join(),
            None => Ok(()),
        }
    }
}

impl<E: BackendExecutor, K: KvPool> Drop for ServeHandle<E, K> {
    fn drop(&mut self) {
        // Close the submit + control channels so the engine loop can observe
        // shutdown, then join so the engine thread fully drains before we return.
        self.submit_tx.take();
        self.control_tx.take();
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

// ServeHandle<E, K> is Send regardless of whether E or K are Send: every field
// is independently Send (channels, Arc, AtomicUsize, PhantomData<fn()->(E,K)>).
// The E/K values themselves live exclusively on the engine thread and are never
// moved out through the public ServeHandle API. Rust's conservative auto-trait
// inference denies Send because of the generic parameters, but the send-safety
// proof is field-by-field: Sender<Submission>, Sender<ControlMessage<E,K>>
// (ControlMessage is Box<dyn FnOnce+Send>, always Send), Arc<Mutex<_>>, etc.
unsafe impl<E: BackendExecutor, K: KvPool> Send for ServeHandle<E, K> {}

/// A tiny in-crate executor that echoes each row's input forward by `+1`.
///
/// Mirrors the engine-core test mock's token rule so the serve layer can be
/// exercised end-to-end without a real backend (no MLX, no CUDA): a decode row
/// emits `last_token + 1`, and a prefill chunk's committed token (on the final
/// chunk) is `last_prompt_token + 1`. Completion is therefore length-bound by
/// `max_tokens`, which makes token-count assertions deterministic.
#[derive(Debug, Clone, Copy, Default)]
pub struct EchoExecutor;

impl BackendExecutor for EchoExecutor {
    type Inflight = StepOutput;

    fn submit(&mut self, plan: &ForwardPlan, _kv: &mut dyn KvPool) -> Result<Self::Inflight> {
        let tokens = plan
            .prefill_rows
            .iter()
            .map(|row| SlotToken {
                slot: row.slot,
                token: row.tokens.last().copied().map_or(1, |last| last + 1),
                logprob: None,
                finish: None,
            })
            .chain(plan.decode_rows.iter().map(|row| SlotToken {
                slot: row.slot,
                token: row.last_token + 1,
                logprob: None,
                finish: None,
            }))
            .collect();
        Ok(StepOutput { tokens })
    }

    fn poll(&mut self, inflight: Self::Inflight) -> Result<PollResult<Self::Inflight>> {
        Ok(PollResult::Ready(inflight))
    }
}

/// Single-process coordinator HTTP router using a local in-process relay.
///
/// Identical route surface as [`coordinator_router`] (single HTTP implementation);
/// request handling goes through the relay protocol over an in-process channel
/// rather than TCP. Pass `multimodal_kind` for VLM backends (Gemma4, DeepseekOcr,
/// DiffusionGemma); text-only backends pass `None`.
pub fn coordinator_local_router<E, K>(
    serve: ServeHandle<E, K>,
    tokenizer: tokenizer::OpenAiTokenizer,
    model: impl Into<String>,
    max_thinking_tokens: usize,
    multimodal_kind: Option<infer_plan::MultimodalKind>,
) -> axum::Router
where
    E: infer_seam::BackendExecutor + 'static,
    K: infer_seam::KvPool + 'static,
{
    use multiproc_relay::RelayCoordinator;
    use std::sync::Arc;

    let (relay, engine_recv, engine_tx) = RelayCoordinator::new_local();
    let serve = Arc::new(serve);

    let (multimodal_rx, coord_multimodal) = match multimodal_kind {
        Some(kind) => {
            let (tx, rx) = std::sync::mpsc::sync_channel::<LocalMultimodalRequest>(8);
            (Some(rx), Some((tx, kind)))
        }
        None => (None, None),
    };
    std::thread::Builder::new()
        .name("arle-local-relay-driver".to_string())
        .spawn(move || serve_handle_relay_driver(serve, engine_recv, engine_tx, multimodal_rx))
        .expect("spawn arle-local-relay-driver");

    coordinator::coordinator_router(
        relay,
        tokenizer,
        model.into(),
        max_thinking_tokens,
        coord_multimodal,
    )
}

fn relay_stream(
    request_id: u64,
    rx: std::sync::mpsc::Receiver<execution::StreamItem>,
    tx: &std::sync::mpsc::SyncSender<multiproc_relay::RelayEnvelope>,
) {
    use multiproc_relay::RelayEnvelope;
    for item in rx {
        let delta = match item {
            execution::StreamItem::Token { token, .. } => multiproc_relay::RelayCompletionDelta {
                token_ids: vec![token],
                ..Default::default()
            },
            execution::StreamItem::Done(completed) => multiproc_relay::RelayCompletionDelta {
                finish: true,
                finish_reason: completed.finish.clone(),
                ..Default::default()
            },
        };
        if tx
            .send(RelayEnvelope::Completion { request_id, delta })
            .is_err()
        {
            break; // coordinator gone
        }
    }
}

fn serve_handle_relay_driver<E, K>(
    serve: std::sync::Arc<ServeHandle<E, K>>,
    mut engine_recv: multiproc_relay::LocalChannelRecv,
    engine_tx: std::sync::mpsc::SyncSender<multiproc_relay::RelayEnvelope>,
    multimodal_rx: Option<LocalMultimodalRx>,
) where
    E: infer_seam::BackendExecutor + 'static,
    K: infer_seam::KvPool + 'static,
{
    use multiproc_relay::{RelayChannel, RelayEnvelope, WireStats};

    if let Some(rx) = multimodal_rx {
        let serve_mm = std::sync::Arc::clone(&serve);
        std::thread::Builder::new()
            .name("arle-local-multimodal".to_string())
            .spawn(move || {
                for req in rx {
                    let serve_clone = std::sync::Arc::clone(&serve_mm);
                    std::thread::spawn(move || {
                        let result = serve_clone.run_on_executor(move |e| {
                            e.generate_multimodal(
                                &req.prompt_tokens,
                                &req.images,
                                req.max_tokens,
                                &req.sampling,
                            )
                        });
                        let delta = match result {
                            Ok(Ok(Some(out))) => multiproc_relay::RelayCompletionDelta {
                                text_delta: String::new(),
                                token_ids: out.generated_tokens,
                                finish: true,
                                finish_reason: Some(out.finish),
                                error: None,
                            },
                            Ok(Ok(None)) => multiproc_relay::RelayCompletionDelta {
                                text_delta: String::new(),
                                token_ids: Vec::new(),
                                finish: true,
                                finish_reason: None,
                                error: Some("generate_multimodal returned None".to_string()),
                            },
                            Ok(Err(e)) | Err(e) => multiproc_relay::RelayCompletionDelta {
                                text_delta: String::new(),
                                token_ids: Vec::new(),
                                finish: true,
                                finish_reason: None,
                                error: Some(e.to_string()),
                            },
                        };
                        let _ = req.response_tx.send(delta);
                    });
                }
            })
            .expect("spawn arle-local-multimodal");
    }

    // Pre-spawn a fixed pool of relay worker threads — one per engine slot.
    // This replaces per-request thread::spawn which created O(c) threads in
    // a burst and triggered ELKEID SIGKILL at c=1024.
    // Ticket held alongside rx so live_requests is decremented only after the
    // worker finishes, not the moment submit_streaming returns.
    type WorkItem = (
        u64,
        RequestTicket,
        std::sync::mpsc::Receiver<execution::StreamItem>,
    );
    let (work_tx, work_rx) = std::sync::mpsc::channel::<WorkItem>();
    let work_rx = std::sync::Arc::new(std::sync::Mutex::new(work_rx));
    let n_workers = serve.max_live_requests.clamp(1, 1024);
    for _ in 0..n_workers {
        let work_rx = std::sync::Arc::clone(&work_rx);
        let tx = engine_tx.clone();
        std::thread::Builder::new()
            .name("arle-relay-worker".into())
            .spawn(move || {
                while let Ok((request_id, _ticket, rx)) = work_rx.lock().unwrap().recv() {
                    relay_stream(request_id, rx, &tx);
                    // _ticket drops here — live_requests decremented after relay completes
                }
            })
            .expect("spawn relay worker");
    }

    loop {
        match engine_recv.recv() {
            Ok(Some(RelayEnvelope::TickAdmissions { seq: _, requests })) => {
                for wire in requests {
                    let request_id = wire.request_id;
                    let (prompt_tokens, max_tokens, sampling) = wire.submit_args();
                    match serve.submit_streaming(prompt_tokens, max_tokens, sampling) {
                        Ok((ticket, rx)) => {
                            let _ = work_tx.send((request_id, ticket, rx));
                        }
                        Err(e) => {
                            let _ = engine_tx.send(RelayEnvelope::Completion {
                                request_id,
                                delta: multiproc_relay::RelayCompletionDelta {
                                    finish: true,
                                    error: Some(e.to_string()),
                                    ..Default::default()
                                },
                            });
                        }
                    }
                }
            }
            Ok(Some(RelayEnvelope::StatsQuery { request_id })) => {
                let counters = serve.counters();
                let data = Box::new(WireStats::from_counters(&counters));
                let _ = engine_tx.send(RelayEnvelope::StatsResponse { request_id, data });
            }
            Ok(Some(RelayEnvelope::Shutdown)) | Ok(None) => {
                log::info!("[local-relay-driver] shutdown");
                return;
            }
            Ok(Some(other)) => {
                log::debug!("[local-relay-driver] unexpected envelope: {other:?}");
            }
            Err(e) => {
                log::error!("[local-relay-driver] recv error: {e:#}");
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use infer_seam::HostPagedKvPool;

    #[derive(Debug, Clone, Copy, Default)]
    struct SingleFlightEchoExecutor;

    impl BackendExecutor for SingleFlightEchoExecutor {
        type Inflight = StepOutput;

        fn submit(&mut self, plan: &ForwardPlan, kv: &mut dyn KvPool) -> Result<Self::Inflight> {
            let mut inner = EchoExecutor;
            inner.submit(plan, kv)
        }

        fn poll(&mut self, inflight: Self::Inflight) -> Result<PollResult<Self::Inflight>> {
            let mut inner = EchoExecutor;
            inner.poll(inflight)
        }

        fn max_live_requests(&self) -> usize {
            1
        }
    }

    #[derive(Debug, Clone, Copy, Default)]
    struct NeverReadyEchoExecutor;

    impl BackendExecutor for NeverReadyEchoExecutor {
        type Inflight = StepOutput;

        fn submit(&mut self, plan: &ForwardPlan, kv: &mut dyn KvPool) -> Result<Self::Inflight> {
            let mut inner = EchoExecutor;
            inner.submit(plan, kv)
        }

        fn poll(&mut self, inflight: Self::Inflight) -> Result<PollResult<Self::Inflight>> {
            Ok(PollResult::NotReady(inflight))
        }
    }

    /// End-to-end frontend-||-engine smoke test with no GPU/MLX dependency.
    ///
    /// Spawns the engine thread on an `EchoExecutor` + real host paged KV pool,
    /// submits two requests, collects both off their private back-channels, and
    /// asserts the generated token counts match each request's `max_tokens`
    /// (the echo executor never emits a stop token, so finish is length-bound).
    #[test]
    fn submit_two_requests_and_collect_both() -> Result<()> {
        let config = SchedulerConfig {
            num_slots: 2,
            max_prompt_tokens: 4096,
            max_total_tokens: 8192,
            ..SchedulerConfig::default()
        };
        // Real host-side pool: 2 slots, ample pages, page_size 16.
        let kv = HostPagedKvPool::new(2, 256, 16);
        let serve = ServeHandle::spawn(EchoExecutor, kv, config);

        let first = serve.submit(vec![10, 11, 12], 5, SamplingParams::default())?;
        let second = serve.submit(vec![100, 101], 3, SamplingParams::default())?;

        // Handles are distinct and ordered by submission.
        assert_ne!(first.handle(), second.handle());

        let first_done = first.collect()?;
        let second_done = second.collect()?;

        assert_eq!(
            first_done.generated_tokens.len(),
            5,
            "first request should generate exactly max_tokens=5 tokens"
        );
        assert_eq!(
            second_done.generated_tokens.len(),
            3,
            "second request should generate exactly max_tokens=3 tokens"
        );
        // Echo rule: final prefill chunk commits last_prompt + 1, then +1/decode.
        assert_eq!(first_done.generated_tokens, vec![13, 14, 15, 16, 17]);
        assert_eq!(second_done.generated_tokens, vec![102, 103, 104]);

        serve.shutdown().expect("engine thread joins cleanly");
        Ok(())
    }

    #[test]
    fn single_flight_backend_rejects_second_live_request() -> Result<()> {
        let config = SchedulerConfig {
            num_slots: 1,
            max_prompt_tokens: 4096,
            max_total_tokens: 8192,
            ..SchedulerConfig::default()
        };
        let kv = HostPagedKvPool::new(1, 256, 16);
        let serve = ServeHandle::spawn(SingleFlightEchoExecutor, kv, config);

        let first = serve.submit(vec![10, 11, 12], 5, SamplingParams::default())?;
        let err = match serve.submit(vec![100, 101], 3, SamplingParams::default()) {
            Ok(_) => panic!("single-flight backend must reject a second live request"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("server is busy"),
            "unexpected error: {err:#}"
        );

        assert_eq!(first.collect()?.generated_tokens, vec![13, 14, 15, 16, 17]);
        let second = serve.submit(vec![100, 101], 3, SamplingParams::default())?;
        assert_eq!(second.collect()?.generated_tokens, vec![102, 103, 104]);

        serve.shutdown().expect("engine thread joins cleanly");
        Ok(())
    }

    /// `submit_streaming` yields each token live in commit order, then exactly one
    /// terminal `Done`; the ticket still resolves the same full completion.
    #[test]
    fn submit_streaming_yields_each_token_then_done() -> Result<()> {
        let config = SchedulerConfig {
            num_slots: 1,
            max_prompt_tokens: 4096,
            max_total_tokens: 8192,
            ..SchedulerConfig::default()
        };
        let kv = HostPagedKvPool::new(1, 256, 16);
        let serve = ServeHandle::spawn(EchoExecutor, kv, config);

        let (ticket, stream_rx) =
            serve.submit_streaming(vec![10, 11, 12], 5, SamplingParams::default())?;

        let mut streamed = Vec::new();
        let mut done = None;
        while let Ok(item) = stream_rx.recv() {
            match item {
                StreamItem::Token { token, .. } => streamed.push(token),
                StreamItem::Done(completed) => {
                    done = Some(completed);
                    break;
                }
            }
        }
        // Echo rule: final prefill chunk commits last_prompt+1, then +1/decode.
        assert_eq!(
            streamed,
            vec![13, 14, 15, 16, 17],
            "tokens stream in commit order"
        );
        let done = done.expect("terminal Done item");
        assert_eq!(done.generated_tokens, streamed);
        assert_eq!(ticket.collect()?.generated_tokens, streamed);

        serve.shutdown().expect("engine thread joins cleanly");
        Ok(())
    }

    /// The engine loop republishes scheduler counters each tick; after a request
    /// runs, `counters()` reflects a live (non-default) snapshot — proving the
    /// publish path, vs the all-zero default before the first tick.
    #[test]
    fn counters_are_published_each_tick() -> Result<()> {
        let config = SchedulerConfig {
            num_slots: 1,
            max_prompt_tokens: 4096,
            max_total_tokens: 8192,
            ..SchedulerConfig::default()
        };
        let kv = HostPagedKvPool::new(1, 256, 16);
        let serve = ServeHandle::spawn(EchoExecutor, kv, config);

        serve
            .submit(vec![10, 11, 12], 4, SamplingParams::default())?
            .collect()?;

        // After ticks ran, the snapshot reflects the (now idle) engine: requests
        // drained and the KV pool restored to its non-zero free-page count.
        // The completion reaches the collector BEFORE the engine's next tick
        // publishes the post-drain snapshot, so poll briefly instead of racing
        // the engine thread (one-shot read flaked on slow CI runners).
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        let snap = loop {
            let snap = serve.counters();
            let drained = snap.active_requests == 0 && snap.queue_depth == 0;
            if drained || std::time::Instant::now() >= deadline {
                break snap;
            }
            std::thread::sleep(Duration::from_millis(10));
        };
        assert_eq!(snap.active_requests, 0, "request drained");
        assert_eq!(snap.queue_depth, 0, "queue drained");
        assert!(
            snap.kv_free_pages > 0,
            "free-page count published, not default"
        );

        serve.shutdown().expect("engine thread joins cleanly");
        Ok(())
    }

    /// A request the engine completes synchronously at ingress (prompt too long)
    /// must still deliver its completion to the collector without hanging.
    #[test]
    fn rejected_request_completes_without_hanging() -> Result<()> {
        let config = SchedulerConfig {
            num_slots: 1,
            max_prompt_tokens: 2,
            max_total_tokens: 16,
            ..SchedulerConfig::default()
        };
        let kv = HostPagedKvPool::new(1, 64, 16);
        let serve = ServeHandle::spawn(EchoExecutor, kv, config);

        let ticket = serve.submit(vec![1, 2, 3], 4, SamplingParams::default())?;
        let done = ticket
            .collect_timeout(Duration::from_secs(5))
            .map_err(|_| anyhow!("rejected request did not complete in time"))?;
        assert!(done.generated_tokens.is_empty());
        assert!(done.finish.is_some());

        serve.shutdown().expect("engine thread joins cleanly");
        Ok(())
    }

    #[test]
    fn shutdown_token_aborts_inflight_without_drain() -> Result<()> {
        let config = SchedulerConfig {
            num_slots: 1,
            max_prompt_tokens: 4096,
            max_total_tokens: 8192,
            ..SchedulerConfig::default()
        };
        let shutdown = ServeShutdown::new();
        let kv = HostPagedKvPool::new(1, 256, 16);
        let serve =
            ServeHandle::spawn_with_shutdown(NeverReadyEchoExecutor, kv, config, shutdown.clone());

        let ticket = serve.submit(vec![10, 11, 12], 5, SamplingParams::default())?;
        let handle = ticket.handle();
        shutdown.request();

        match ticket.collect_timeout(Duration::from_secs(2)) {
            Err(CollectTimeout::Closed(closed)) => assert_eq!(closed, handle),
            Err(CollectTimeout::Pending(_)) => panic!("shutdown did not close in-flight request"),
            Ok(done) => panic!("never-ready request unexpectedly completed: {done:?}"),
        }

        serve.shutdown().expect("engine thread joins cleanly");
        Ok(())
    }
}
