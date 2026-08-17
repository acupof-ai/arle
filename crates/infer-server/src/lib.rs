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
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::Instant;

use anyhow::{Result, anyhow};
use infer_core::{CompletedRequest, Engine, RequestHandle, SchedulerConfig};
use infer_plan::{ForwardPlan, SamplingParams, SlotToken, StepOutput};
use infer_seam::{BackendExecutor, KvPool, PollResult};

mod anthropic;
mod coordinator;
mod execution;
mod grammar;
mod metrics;
pub mod multimodal;
pub mod multiproc_relay;
mod observe;
mod schema;
mod sse_util;
mod tokenizer;

use execution::{ControlMessage, Submission, engine_loop};
pub use execution::{CounterSnapshot, StreamItem};

pub use anthropic::messages_body_to_chat_request;
pub use coordinator::{
    TokensSidecar, coordinator_router, dp_coordinator_router, set_messages_dump_dir,
    tokens_sidecar_path,
};

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
    RelayWorker, TcpChannel, WireRequest, WireStats, set_tick_broadcaster,
};
pub use schema::{
    ChatContent, ChatContentPart, ChatMessage, CompletionRequest, SamplingDefaults,
    set_sampling_defaults,
};
pub use tokenizer::OpenAiTokenizer;

static PRODUCT_BINARY_ID: OnceLock<String> = OnceLock::new();

/// Server-owned product and kernel-bundle identities for `/v1/stats`.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct BuildIdentity {
    pub product_binary_sha256: String,
    pub kernel_bundle_id: String,
}

/// Materialize build identity only at an explicit stats request boundary.
#[must_use]
pub fn build_identity(artifact: infer_seam::BackendArtifactIdentity) -> BuildIdentity {
    let kernel_bundle_id = if artifact.kernel_bundle_id.is_empty() {
        "unreported".to_string()
    } else {
        artifact.kernel_bundle_id
    };
    BuildIdentity {
        product_binary_sha256: PRODUCT_BINARY_ID
            .get_or_init(|| product_binary_sha256().unwrap_or_else(|_| "unreported".to_string()))
            .clone(),
        kernel_bundle_id,
    }
}

fn product_binary_sha256() -> Result<String> {
    use std::fmt::Write;
    use std::io::Read;

    let path = std::env::current_exe()?;
    let mut file = std::fs::File::open(path)?;
    let mut digest = ring::digest::Context::new(&ring::digest::SHA256);
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    let mut id = String::with_capacity(71);
    id.push_str("sha256:");
    for byte in digest.finish().as_ref() {
        write!(id, "{byte:02x}").expect("writing to String is infallible");
    }
    Ok(id)
}

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
}

impl Drop for RequestTicket {
    fn drop(&mut self) {
        self.live_requests.fetch_sub(1, Ordering::AcqRel);
    }
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
        let observe_counters = Arc::clone(&counters);
        observe::spawn_observe_task(move || observe_counters.lock().ok().map(|s| s.clone()));
        let max_live_requests = executor.step_limits().max_live_requests.max(1);
        let join = thread::Builder::new()
            .name("infer-engine".to_string())
            .spawn(move || {
                engine_loop(
                    Engine::with_config(executor, kv, config).expect("engine config rejected"),
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

fn quiesce_engine<E: BackendExecutor, K: KvPool>(engine: &mut Engine<E, K>) -> Result<usize> {
    let handles = engine.quiesce();
    engine.run_to_idle()?;
    for handle in &handles {
        log::warn!("[serve-engine] cancelled orphaned request {}", handle.id());
    }
    Ok(handles.len())
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
        let observe_counters = Arc::clone(&counters);
        observe::spawn_observe_task(move || observe_counters.lock().ok().map(|s| s.clone()));
        let join = thread::Builder::new()
            .name("infer-engine".to_string())
            .spawn(move || match builder() {
                Ok(mut engine) => {
                    if let Err(err) = engine.warmup() {
                        let _ = ready_tx.send(Err(format!("{err:#}")));
                        return;
                    }
                    let max_live_requests = engine.max_live_requests();
                    let _ = ready_tx.send(Ok(max_live_requests));
                    engine_loop(engine, submit_rx, control_rx, loop_counters, shutdown);
                }
                Err(err) => {
                    // {err:#} keeps the context chain across the String channel.
                    let _ = ready_tx.send(Err(format!("{err:#}")));
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
                // {err:#} keeps the full context chain: "{err}" flattened it and
                // hid an OOM root cause behind "row fuse + <tensor>".
                Err(anyhow!("engine build failed: {err:#}"))
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
        self.do_submit(prompt, max_tokens, sampling, None, None)
    }

    pub fn submit_streaming(
        &self,
        prompt: Vec<u32>,
        max_tokens: usize,
        sampling: SamplingParams,
    ) -> Result<(RequestTicket, Receiver<StreamItem>)> {
        let (stream_tx, stream_rx) = mpsc::channel::<StreamItem>();
        let ticket = self.do_submit(prompt, max_tokens, sampling, Some(stream_tx), None)?;
        Ok((ticket, stream_rx))
    }

    pub fn submit_streaming_constrained(
        &self,
        prompt: Vec<u32>,
        max_tokens: usize,
        sampling: SamplingParams,
        grammar: Option<infer_core::GrammarHook>,
    ) -> Result<(RequestTicket, Receiver<StreamItem>)> {
        let (stream_tx, stream_rx) = mpsc::channel::<StreamItem>();
        let ticket = self.do_submit(prompt, max_tokens, sampling, Some(stream_tx), grammar)?;
        Ok((ticket, stream_rx))
    }

    fn do_submit(
        &self,
        prompt: Vec<u32>,
        max_tokens: usize,
        sampling: SamplingParams,
        stream_tx: Option<mpsc::Sender<StreamItem>>,
        grammar: Option<infer_core::GrammarHook>,
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
                grammar,
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
        self.counters.lock().map(|c| c.clone()).unwrap_or_default()
    }

    /// Materialize operator dispatch counters + backend artifact identity in one
    /// engine-thread round-trip, only at a stats request boundary.
    pub fn operator_stats(
        &self,
    ) -> Result<(
        infer_seam::OperatorDispatchStats,
        infer_seam::BackendArtifactIdentity,
    )> {
        self.run_on_engine(|engine| {
            let stats = engine.backend_stats();
            (stats.operator_dispatch, stats.artifact)
        })
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
        self.run_on_engine(|engine| engine.offload_engine_weights())?
    }

    /// Reload the engine's device weights from the host snapshot (OPD teacher
    /// weight time-share). Blocks until the H2D round-trip completes.
    pub fn reload_engine_weights(&self) -> Result<()> {
        self.run_on_engine(|engine| engine.reload_engine_weights())?
    }

    /// Release the engine's inference forward scratch WITHOUT offloading weights or
    /// evicting KV (OPD rollout->writeback VRAM reclaim). Runs on the engine thread
    /// (between scheduler steps) via the out-of-band control seam, so the release
    /// never races an in-flight forward step. Blocks until the round-trip completes.
    pub fn release_inference_scratch(&self) -> Result<()> {
        self.run_on_engine(|engine| engine.release_inference_scratch())?
    }

    /// Drop the engine's KV pool WITHOUT offloading weights (OPD writeback
    /// headroom). Runs on the engine thread via the control seam, so it never
    /// races an in-flight forward. Blocks until the round-trip completes.
    pub fn release_kv_pool(&self) -> Result<()> {
        self.run_on_engine(|engine| engine.release_kv_pool())?
    }

    pub fn ensure_kv_pool(&self) -> Result<()> {
        self.run_on_engine(|engine| engine.ensure_kv_pool())?
    }

    /// Re-acquire the engine's KV pool, then resume admission atomically on the
    /// engine thread. A failed ensure leaves the engine quiesced.
    pub fn ensure_kv_pool_and_resume_admissions(&self) -> Result<()> {
        self.run_on_engine(|engine| {
            engine.ensure_kv_pool()?;
            engine.resume_serving();
            Ok(())
        })?
    }

    /// OPD round-loop quiesce: switch the engine to Quiesced (the serve loop
    /// defers new admission) and cancel every in-flight (waiting + active)
    /// request, atomically on the engine thread. Returns how many were
    /// cancelled. Pairs with [`Self::resume_admissions`], called after the KV
    /// pool is re-acquired.
    pub fn quiesce_admissions(&self) -> Result<usize> {
        self.run_on_engine(quiesce_engine)?
    }

    /// Re-arm engine serving after the OPD writeback bracket (KV pool
    /// re-acquired). Idempotent.
    pub fn resume_admissions(&self) -> Result<()> {
        self.run_on_engine(|engine| engine.resume_serving())
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
// SAFETY: Field-by-field Send proof: Sender<Submission>, Sender<ControlMessage<E,K>>
// (ControlMessage is Box<dyn FnOnce+Send>, always Send), Arc<Mutex<_>>, etc.
// E/K values live exclusively on the engine thread, never moved through the public API.
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
                top_logprobs: Vec::new(),
                finish: None,
            })
            .chain(plan.decode_rows.iter().map(|row| SlotToken {
                slot: row.slot,
                token: row.last_token + 1,
                logprob: None,
                top_logprobs: Vec::new(),
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
    serve: Arc<ServeHandle<E, K>>,
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

    let (relay, engine_recv, engine_tx) = RelayCoordinator::new_local();

    let (multimodal_rx, coord_multimodal) = match multimodal_kind {
        Some(kind) => {
            let (tx, rx) = std::sync::mpsc::sync_channel::<LocalMultimodalRequest>(8);
            (Some(rx), Some((tx, kind)))
        }
        None => (None, None),
    };
    let grammars = grammar::GrammarCache::new(&tokenizer)
        .map(std::sync::Arc::new)
        .map_err(|e| log::warn!("structured output unavailable: {e}"))
        .ok();
    std::thread::Builder::new()
        .name("arle-local-relay-driver".to_string())
        .spawn(move || {
            serve_handle_relay_driver(serve, engine_recv, engine_tx, multimodal_rx, grammars)
        })
        .expect("spawn arle-local-relay-driver");

    coordinator::coordinator_router(
        relay,
        tokenizer,
        model.into(),
        max_thinking_tokens,
        coord_multimodal,
        // Local lane: the engine thread and this process share fate already.
        None,
        false,
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
            execution::StreamItem::Token {
                token,
                logprob,
                top_logprobs,
            } => multiproc_relay::RelayCompletionDelta {
                token_ids: vec![token],
                logprobs: logprob.into_iter().collect(),
                top_logprobs: if top_logprobs.is_empty() {
                    Vec::new()
                } else {
                    vec![top_logprobs]
                },
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
    grammars: Option<std::sync::Arc<grammar::GrammarCache>>,
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
                        let result = serve_clone.run_on_executor(move |e| match e.multimodal() {
                            Some(mm) => mm.generate_multimodal(
                                &req.prompt_tokens,
                                &req.images,
                                req.max_tokens,
                                &req.sampling,
                            ),
                            None => Ok(None),
                        });
                        let delta = match result {
                            Ok(Ok(Some(out))) => multiproc_relay::RelayCompletionDelta {
                                token_ids: out.generated_tokens,
                                finish: true,
                                finish_reason: Some(out.finish),
                                ..Default::default()
                            },
                            Ok(Ok(None)) => multiproc_relay::RelayCompletionDelta {
                                finish: true,
                                error: Some("generate_multimodal returned None".to_string()),
                                ..Default::default()
                            },
                            Ok(Err(e)) | Err(e) => multiproc_relay::RelayCompletionDelta {
                                finish: true,
                                error: Some(e.to_string()),
                                ..Default::default()
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
    // request_id -> engine handle for `CancelRequest` lookups (client
    // disconnect); inserted at submit, removed by the worker at stream end.
    let handles: std::sync::Arc<Mutex<std::collections::HashMap<u64, RequestHandle>>> =
        std::sync::Arc::default();
    let n_workers = serve.max_live_requests.clamp(1, 1024);
    for _ in 0..n_workers {
        let work_rx = std::sync::Arc::clone(&work_rx);
        let handles = std::sync::Arc::clone(&handles);
        let tx = engine_tx.clone();
        std::thread::Builder::new()
            .name("arle-relay-worker".into())
            .spawn(move || {
                loop {
                    // Bind recv() in its own scope so the work_rx guard drops
                    // BEFORE relay_stream's long blocking loop — else one worker
                    // holds the lock for its whole stream, collapsing the pool.
                    let next = { work_rx.lock().unwrap().recv() };
                    let Ok((request_id, _ticket, rx)) = next else {
                        break;
                    };
                    relay_stream(request_id, rx, &tx);
                    handles.lock().unwrap().remove(&request_id);
                    // _ticket drops here — live_requests decremented after relay completes
                }
            })
            .expect("spawn relay worker");
    }

    // Engine-touching work (submit / cancel / stats) runs on a dedicated thread
    // so the pump loop below can ALWAYS ack ticks. `submit_streaming` blocks on
    // an engine-thread round-trip; with it on the ack path, one long engine step
    // (first-touch JIT, giant prefill chunk) starved the coordinator's
    // TICK_WINDOW for >ACK_STALL_TIMEOUT and its watchdog tore down the serve
    // (2026-07-23 single-GPU OPD rollout). Acks are liveness, not admission.
    let (engine_work_tx, engine_work_rx) = std::sync::mpsc::channel::<RelayEnvelope>();
    {
        let serve = std::sync::Arc::clone(&serve);
        let engine_tx = engine_tx.clone();
        std::thread::Builder::new()
            .name("arle-local-relay-submitter".to_string())
            .spawn(move || {
                for envelope in engine_work_rx {
                    match envelope {
                        RelayEnvelope::TickAdmissions { requests, .. } => {
                            for wire in requests {
                                let request_id = wire.request_id;
                                let (prompt_tokens, max_tokens, sampling, format) =
                                    wire.submit_args();
                                let hook = match grammar::resolve(grammars.as_deref(), format) {
                                    Ok(hook) => hook,
                                    Err(e) => {
                                        let _ = engine_tx.send(RelayEnvelope::Completion {
                                            request_id,
                                            delta: multiproc_relay::RelayCompletionDelta {
                                                finish: true,
                                                error: Some(e.to_string()),
                                                ..Default::default()
                                            },
                                        });
                                        continue;
                                    }
                                };
                                match serve.submit_streaming_constrained(
                                    prompt_tokens,
                                    max_tokens,
                                    sampling,
                                    hook,
                                ) {
                                    Ok((ticket, rx)) => {
                                        // Insert BEFORE handing to a worker so the
                                        // worker's remove-at-stream-end can never
                                        // race the insert.
                                        handles.lock().unwrap().insert(request_id, ticket.handle());
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
                        // Client disconnected (`InFlightGuard::drop` -> coordinator
                        // broadcast): stop decoding the orphan so it frees its KV
                        // slot. Ordered behind the submit that inserted the handle;
                        // a missing/finished handle is a benign race with natural
                        // finish.
                        RelayEnvelope::CancelRequest { request_id } => {
                            let handle = handles.lock().unwrap().get(&request_id).copied();
                            if let Some(handle) = handle
                                && serve
                                    .run_on_engine(move |engine| engine.cancel_request(handle))
                                    .unwrap_or(false)
                            {
                                log::info!(
                                    "[local-relay-driver] cancelled req#{request_id} (client gone)"
                                );
                            }
                        }
                        RelayEnvelope::StatsQuery { request_id } => {
                            let counters = serve.counters();
                            let (operator_dispatch, artifact) =
                                serve.operator_stats().unwrap_or_default();
                            let data = Box::new(WireStats::from_counters(
                                &counters,
                                build_identity(artifact),
                                operator_dispatch,
                            ));
                            let _ =
                                engine_tx.send(RelayEnvelope::StatsResponse { request_id, data });
                        }
                        other => {
                            log::debug!("[local-relay-submitter] unexpected envelope: {other:?}");
                        }
                    }
                }
            })
            .expect("spawn arle-local-relay-submitter");
    }

    loop {
        match engine_recv.recv() {
            Ok(Some(RelayEnvelope::TickAdmissions { seq, requests })) => {
                if !requests.is_empty() {
                    let _ = engine_work_tx.send(RelayEnvelope::TickAdmissions { seq, requests });
                }
                // Flow-control ack (rank 0 = the only local worker); without it
                // the lockstep loop's ack window stalls at TICK_WINDOW. A send
                // failure means the coordinator is gone — recv EOFs us out next.
                let _ = engine_tx.send(RelayEnvelope::TickAck { rank: 0, seq });
            }
            Ok(Some(
                envelope @ (RelayEnvelope::CancelRequest { .. } | RelayEnvelope::StatsQuery { .. }),
            )) => {
                let _ = engine_work_tx.send(envelope);
            }
            Ok(Some(RelayEnvelope::Shutdown)) | Ok(None) => {
                // Dropping engine_work_tx ends the submitter's for-loop.
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
