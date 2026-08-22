//! SPMD multiproc coordinator: the thin control plane (B split).
//!
//! The parent owns NO TP rank — all `world_size` ranks are symmetric child
//! workers — so it joins no collective and can never be the missing rank at a
//! barrier. It bridges async HTTP (tokio) to the sync relay (blocking TCP): a
//! dedicated-thread lockstep loop broadcasts one `TickAdmissions` per step to
//! every worker, and per-rank completion readers route rank-0's deltas into the
//! per-request sinks the async handlers await. TP=1 never reaches here.

use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver as SyncReceiver, RecvTimeoutError, Sender as SyncSender};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use crate::CounterSnapshot;

use axum::extract::{DefaultBodyLimit, Query, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use infer_plan::{FinishReason, MultimodalKind, SamplingParams};
use uuid::Uuid;

use crate::anthropic::{self, MessagesError, MessagesRequest, StreamEncoder};
use crate::multiproc_relay::{
    CompletionSinks, RelayCompletionDelta, RelayCoordinator, RelayEnvelope, WireRequest,
};
use crate::schema::{
    ApiError, AssistantMessage, ChatChoice, ChatCompletionRequest, ChatCompletionResponse,
    CompletionChoice, CompletionRequest, CompletionResponse, ModelsResponse, ResponseToolCall,
    SYSTEM_FINGERPRINT, StatsResponse, Usage, split_reasoning,
};
use crate::sse_util::{
    ChatDelta, StreamPipeline, chat_stream_chunk, completion_stream_chunk, finish_reason,
    stream_error_chunk, stream_usage_chunk, unix_time_secs,
};
use crate::tokenizer::{IncrementalDetokenizer, OpenAiTokenizer};

/// Idle park on the submit channel; matches the in-process engine loop.
const IDLE_PARK: Duration = Duration::from_millis(2);

/// `--dump-messages-dir`: raw `/v1/messages` bodies land here as
/// `<epoch_ms>_<seq>.json` (CC-trajectory capture). Unset = zero cost.
static MESSAGES_DUMP: std::sync::OnceLock<(std::path::PathBuf, AtomicU64)> =
    std::sync::OnceLock::new();

/// Enable raw `/v1/messages` request dumping. Creates `dir`; call once at
/// startup, before the router serves traffic. A second call is a no-op.
pub fn set_messages_dump_dir(dir: impl Into<std::path::PathBuf>) -> std::io::Result<()> {
    let dir = dir.into();
    std::fs::create_dir_all(&dir)?;
    let _ = MESSAGES_DUMP.set((dir, AtomicU64::new(0)));
    Ok(())
}

/// Serialize `value` to JSON and offload the blocking write of `path` to a
/// `spawn_blocking` task (fire-and-forget; log-and-continue on serialize OR write
/// error). `what` names the artifact in the warn logs.
fn spawn_json_dump(path: PathBuf, value: &impl serde::Serialize, what: &str) {
    let bytes = match serde_json::to_vec(value) {
        Ok(bytes) => bytes,
        Err(err) => {
            log::warn!("serialize {what} {} failed: {err}", path.display());
            return;
        }
    };
    let what = what.to_string(); // owned: the spawn_blocking closure is 'static
    tokio::task::spawn_blocking(move || {
        if let Err(err) = std::fs::write(&path, bytes) {
            log::warn!("write {what} {} failed: {err}", path.display());
        }
    });
}

/// A 5s keep-alive ticker with the immediate first tick already consumed, so each
/// later `.tick()` marks a real idle interval.
async fn keepalive_ticker() -> tokio::time::Interval {
    let mut keepalive = tokio::time::interval(Duration::from_secs(5));
    keepalive.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    keepalive.tick().await; // the first tick fires immediately — consume it
    keepalive
}

/// Await the next stream delta, emitting an SSE keep-alive comment on each idle
/// tick (a long prefill still writes to the socket, and a client disconnect is
/// seen before the first token). `None` = stream ended or client gone — the
/// caller stops.
async fn recv_or_keepalive(
    rx: &mut tokio::sync::mpsc::UnboundedReceiver<RelayCompletionDelta>,
    keepalive: &mut tokio::time::Interval,
    chunk_tx: &tokio::sync::mpsc::Sender<Result<Vec<u8>, std::convert::Infallible>>,
) -> Option<RelayCompletionDelta> {
    loop {
        tokio::select! {
            maybe = rx.recv() => return maybe,
            _ = keepalive.tick() => {
                // SSE comment: OpenAI clients ignore it; a failed send means the
                // client is gone.
                if chunk_tx.send(Ok(b": keep-alive\n\n".to_vec())).await.is_err() {
                    return None;
                }
            }
        }
    }
}

/// Fire-and-forget dump of one `/v1/messages` body (log-and-continue on
/// error). Returns the dump path so the handler can write its token sidecar.
fn dump_messages_body(body: &serde_json::Value) -> Option<PathBuf> {
    let (dir, seq) = MESSAGES_DUMP.get()?;
    let epoch_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis());
    // pid-tagged: multiple serve processes (cp rollout fleet) share one dir.
    let path = dir.join(format!(
        "{epoch_ms}_{}_{}.json",
        std::process::id(),
        seq.fetch_add(1, Ordering::Relaxed)
    ));
    spawn_json_dump(path.clone(), body, "/v1/messages body");
    Some(path)
}

/// Request-keyed token truth written beside a `/v1/messages` dump: the exact
/// prompt tokens the serve rendered + the tokens the engine generated for THAT
/// request — no re-render drift. Assistant-span byte offsets are not captured:
/// the serve renders via the checkpoint's Jinja template, which exposes no
/// spans.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct TokensSidecar {
    pub prompt_token_ids: Vec<u32>,
    pub gen_token_ids: Vec<u32>,
    /// Generation-time behavior logprobs (P6b), one per `gen_token_ids` entry:
    /// log p under the SAME filtered dist the sampler drew from, captured at
    /// commit time. Empty when uncaptured (greedy, Metal, pre-P6 serves).
    #[serde(default)]
    pub gen_logprobs: Vec<f32>,
}

/// Sidecar path for a dump: `<epoch_ms>_<seq>.tokens.json`.
#[must_use]
pub fn tokens_sidecar_path(dump_path: &Path) -> PathBuf {
    dump_path.with_extension("tokens.json")
}

/// Fire-and-forget sidecar write (log-and-continue on error). `logprobs` is
/// all-or-nothing: kept only when it covers every generated token (a partial
/// vector would silently misalign the F.6 ratio diagnostic).
fn write_tokens_sidecar(
    dump_path: &Path,
    prompt: Vec<u32>,
    generated: Vec<u32>,
    logprobs: Vec<f32>,
) {
    let path = tokens_sidecar_path(dump_path);
    let gen_logprobs = if logprobs.len() == generated.len() {
        logprobs
    } else {
        Vec::new()
    };
    let sidecar = TokensSidecar {
        prompt_token_ids: prompt,
        gen_token_ids: generated,
        gen_logprobs,
    };
    spawn_json_dump(path, &sidecar, "tokens sidecar");
}

/// Cap on ticks broadcast beyond the slowest rank's [`RelayEnvelope::TickAck`].
/// Pacing the tick stream to engine speed bounds the worker FIFO depth, so a
/// mid-decode submission (or `StatsQuery`, which rides the same FIFO) waits
/// ≤ window × step time instead of queue-depth × step time (measured pre-fix:
/// ~608k queued ticks for ~600 engine steps).
const TICK_WINDOW: u64 = 4;

/// A worker can stop acking without closing its socket, leaving
/// `any_worker_dead()` false forever (2026-07-05 TP=4 livelock — see
/// docs/experience/errors/2026-07-05-multiproc-lockstep-ack-hang-no-timeout.md).
/// Bounds [`wait_for_ack_window`] so that case fails like a crash instead of
/// hanging every request permanently. Generous vs. any real step time.
const ACK_STALL_TIMEOUT: Duration = Duration::from_secs(120);

/// Block until `seq` is within [`TICK_WINDOW`] of the slowest rank's tick ack.
/// `None` = safe to proceed. `Some(reason)` = the lockstep group is broken
/// (a crashed worker, or one stalled past [`ACK_STALL_TIMEOUT`]) and the
/// caller must tear down. Never aborts on a merely SLOW worker.
fn wait_for_ack_window(relay: &Arc<Mutex<RelayCoordinator>>, seq: u64) -> Option<String> {
    let started = Instant::now();
    let mut next_warn = Duration::from_secs(10);
    loop {
        let (min_acked, any_dead) = {
            let coord = relay
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            (coord.min_acked_ticks(), coord.any_worker_dead())
        };
        if any_dead {
            return Some("a worker's relay reader observed EOF/error".to_string());
        }
        if seq < min_acked + TICK_WINDOW {
            return None;
        }
        let elapsed = started.elapsed();
        if elapsed >= ACK_STALL_TIMEOUT {
            log::error!(
                "[coordinator] lockstep ack wait exceeded {ACK_STALL_TIMEOUT:?} at tick #{seq} \
                 (min_acked={min_acked}); tearing down instead of hanging forever"
            );
            return Some(format!(
                "tick #{seq} ack wait exceeded {ACK_STALL_TIMEOUT:?}"
            ));
        }
        if elapsed >= next_warn {
            log::warn!(
                "[coordinator] lockstep stalled: tick #{seq} awaiting acks (min_acked={min_acked}, \
                 elapsed={elapsed:?})"
            );
            next_warn += Duration::from_secs(10);
        }
        std::thread::sleep(Duration::from_micros(500));
    }
}

enum CoordSubmission {
    Submit(Box<WireRequest>),
    /// A request's HTTP client disconnected/timed out (`InFlightGuard::drop`).
    /// Sent unconditionally on every drop, including normal completions —
    /// `RelayEnvelope::CancelRequest` is a no-op on a rank where the request
    /// already finished, so there's no need to distinguish "aborted early"
    /// from "finished normally" here.
    Cancel(u64),
}

/// The coordinator's HTTP-facing state (lives behind `Arc` in the axum router).
pub struct CoordinatorHandle {
    model: String,
    tokenizer: Mutex<OpenAiTokenizer>,
    max_thinking_tokens: usize,
    /// Resolved once at load: does this checkpoint's template default to
    /// thinking-on? True only for DeepSeek-V4-Flash (reasoning-trained). Cached
    /// here so the hot path never locks the tokenizer just to read it.
    template_defaults_thinking: bool,
    /// Cached think-start/end token IDs for reasoning models.
    think_token_ids: Option<(u32, u32)>,
    next_request_id: AtomicU64,
    /// Submitted but not yet terminally completed; gates decode-only ticks.
    in_flight: Arc<AtomicUsize>,
    submit_tx: SyncSender<CoordSubmission>,
    /// Registered/unregistered directly (not via the `RelayCoordinator` mutex) so
    /// HTTP (un)register never contends with the lockstep loop's blocking broadcast.
    sinks: CompletionSinks,
    /// Relay handle for stats queries to rank-0.
    relay: Arc<Mutex<RelayCoordinator>>,
    /// Monotonic counter for stats query request IDs (separate from request IDs to
    /// avoid collisions with completion sink IDs).
    stats_request_id: AtomicU64,
    /// In-process channel to the multimodal driver thread (VLM backends only).
    multimodal_tx: Option<crate::LocalMultimodalTx>,
    /// Multimodal kind for the current backend (VLM backends only).
    multimodal_kind: Option<MultimodalKind>,
}

impl CoordinatorHandle {
    fn alloc_request_id(&self) -> u64 {
        self.next_request_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Resolve the thinking budget and wire think token IDs into `sampling`.
    /// Precedence: reasoning_effort > client budget_tokens > server config > DSv4 default.
    /// Returns the budget for the max_tokens default; `None` when not thinking.
    fn resolve_think_budget(
        &self,
        thinking: bool,
        effort: Option<&str>,
        client_budget: Option<usize>,
        sampling: &mut SamplingParams,
    ) -> Option<usize> {
        if !thinking {
            return None;
        }
        let budget = effort
            .and_then(think_budget_for_effort)
            .or(client_budget)
            .or_else(|| (self.max_thinking_tokens > 0).then_some(self.max_thinking_tokens))
            .or_else(|| {
                self.template_defaults_thinking
                    .then_some(DEFAULT_THINK_BUDGET)
            });
        if let (Some((start, end)), Some(budget)) = (self.think_token_ids, budget) {
            sampling.think_start_token_id = Some(start);
            sampling.think_end_token_id = Some(end);
            sampling.max_thinking_tokens = Some(budget);
        }
        budget
    }

    /// Broadcast a stats query and collect per-rank [`WireStats`] responses.
    /// Returns an empty vec on send failure; on timeout, returns whatever
    /// ranks responded (partial snapshot — counters underestimate, gauges
    /// may overestimate).
    async fn collect_wire_stats(
        &self,
        timeout: Duration,
    ) -> Vec<crate::multiproc_relay::WireStats> {
        let request_id = self.stats_request_id.fetch_add(1, Ordering::Relaxed);
        let (mut rx, expected) = {
            let mut r = self
                .relay
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let expected = r.worker_count();
            let rx = r.register_stats_awaiter(request_id);
            if r.send_stats_query(request_id).is_err() {
                r.unregister_stats_awaiter(request_id);
                return Vec::new();
            }
            (rx, expected)
        };
        let mut ranks = Vec::with_capacity(expected);
        let deadline = tokio::time::Instant::now() + timeout;
        while ranks.len() < expected {
            match tokio::time::timeout_at(deadline, rx.recv()).await {
                Ok(Some(w)) => ranks.push(w),
                _ => break,
            }
        }
        {
            let r = self
                .relay
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            r.unregister_stats_awaiter(request_id);
        }
        if ranks.len() < expected {
            // Partial: counters underestimate safely; gauges may overestimate,
            // but all-zero would hide real traffic.
            log::warn!(
                "stats query: {}/{} ranks responded before timeout; using partial snapshot",
                ranks.len(),
                expected
            );
        }
        ranks
    }
}

/// Data-parallel coordinator: wraps M independent TP groups and routes each
/// request to the least-in-flight one. `Deref` selects the target group so
/// handler bodies stay unchanged — shared fields (model, tokenizer) are
/// identical across groups, and per-group operations (sinks, submit_tx) are
/// consistent within a single function call (the Deref coercion happens once
/// at the call site, fixing the group for that call).
pub struct DpCoordinator {
    groups: Vec<Arc<CoordinatorHandle>>,
    cached_stats: Arc<RwLock<Option<CounterSnapshot>>>,
}

impl DpCoordinator {
    pub fn new(groups: Vec<Arc<CoordinatorHandle>>) -> Self {
        Self {
            groups,
            cached_stats: Arc::new(RwLock::new(None)),
        }
    }

    /// Latest stats snapshot from the background observer poll.
    /// `None` until the first poll completes (or when observe is disabled).
    pub(crate) fn cached_stats(&self) -> Option<CounterSnapshot> {
        self.cached_stats.read().ok()?.clone()
    }

    fn select(&self) -> &Arc<CoordinatorHandle> {
        self.groups
            .iter()
            .min_by_key(|g| g.in_flight.load(Ordering::Acquire))
            .unwrap_or(&self.groups[0])
    }

    /// Aggregate stats from every TP group into a deployment-level
    /// [`CounterSnapshot`]. Groups serve disjoint requests, so counters and
    /// gauges sum across groups.
    pub(crate) async fn query_stats_all(&self, timeout: Duration) -> CounterSnapshot {
        let groups = self.collect_wire_stats_all(timeout).await;
        crate::multiproc_relay::aggregate_wire_stats_dp(groups).into_counter_snapshot()
    }

    /// Per-group aggregated [`WireStats`] from every TP group, queried
    /// concurrently. A group where no rank responded is excluded.
    async fn collect_wire_stats_all(
        &self,
        timeout: Duration,
    ) -> Vec<crate::multiproc_relay::WireStats> {
        let mut set = tokio::task::JoinSet::new();
        for group in &self.groups {
            let group = Arc::clone(group);
            set.spawn(async move { group.collect_wire_stats(timeout).await });
        }
        let mut groups = Vec::with_capacity(self.groups.len());
        while let Some(joined) = set.join_next().await {
            match joined {
                Ok(ranks) if !ranks.is_empty() => {
                    groups.push(crate::multiproc_relay::aggregate_wire_stats(ranks));
                }
                Ok(_) => log::warn!(
                    "stats query: a TP group returned no ranks; excluding it from deployment stats"
                ),
                Err(e) => log::warn!("stats query: TP group stats task failed: {e}"),
            }
        }
        groups
    }
}

impl Deref for DpCoordinator {
    type Target = CoordinatorHandle;
    fn deref(&self) -> &Self::Target {
        self.select()
    }
}

/// Default thinking budget for DeepSeek-V4 (high tier).
const DEFAULT_THINK_BUDGET: usize = 32768;
/// Content headroom added to the thinking budget for the max_tokens default.
const THINK_CONTENT_HEADROOM: usize = 4096;

/// Map OpenAI `reasoning_effort` to a thinking token budget.
fn think_budget_for_effort(effort: &str) -> Option<usize> {
    match effort {
        "minimal" => Some(256),
        "low" => Some(2048),
        "medium" => Some(8192),
        "high" => Some(DEFAULT_THINK_BUDGET),
        _ => None,
    }
}

/// Count reasoning tokens in a generated token stream. The chat template
/// pre-fills `<think>` into the prompt, so thinking starts immediately
/// (`start_in_thinking = true` when thinking is on). Think start/end tokens
/// themselves are not counted — matches the engine's `update_think_state`.
fn count_reasoning_tokens(
    tokens: &[u32],
    think_ids: Option<(u32, u32)>,
    start_in_thinking: bool,
) -> usize {
    let Some((start, end)) = think_ids else {
        return 0;
    };
    let mut count = 0;
    let mut in_thinking = start_in_thinking;
    for &token in tokens {
        if token == start {
            in_thinking = true;
        } else if token == end {
            in_thinking = false;
        } else if in_thinking {
            count += 1;
        }
    }
    count
}

/// Build the coordinator router + spawn the lockstep loop thread. `relay` is the
/// accepted [`RelayCoordinator`] (all N ranks connected via `accept_symmetric`).
/// The lockstep loop runs for the process lifetime. Pass `multimodal` for VLM
/// backends; text-only backends pass `None`. `observe` spawns the background
/// observe task — true for the engine-less multiproc coordinator, false for
/// local relay (ServeHandle owns sampling).
#[allow(private_interfaces)]
pub fn coordinator_router(
    relay: RelayCoordinator,
    tokenizer: OpenAiTokenizer,
    model: impl Into<String>,
    max_thinking_tokens: usize,
    multimodal: Option<(crate::LocalMultimodalTx, MultimodalKind)>,
    shutdown: Option<crate::ServeShutdown>,
    observe: bool,
) -> Router {
    let handle = coordinator_handle(
        relay,
        tokenizer,
        model,
        max_thinking_tokens,
        multimodal,
        shutdown,
    );
    let dp = Arc::new(DpCoordinator::new(vec![handle]));
    if observe {
        spawn_coordinator_observe(Arc::clone(&dp));
    }
    build_router(dp)
}

/// Multi-group DP coordinator router: one [`CoordinatorHandle`] per relay (each
/// with its own lockstep loop), wrapped in a [`DpCoordinator`] that routes
/// requests to the least-in-flight group.
#[allow(private_interfaces)]
pub fn dp_coordinator_router(
    relays: Vec<RelayCoordinator>,
    tokenizer: OpenAiTokenizer,
    model: impl Into<String>,
    max_thinking_tokens: usize,
    shutdown: Option<crate::ServeShutdown>,
) -> Router {
    let model = model.into();
    let handles: Vec<Arc<CoordinatorHandle>> = relays
        .into_iter()
        .map(|relay| {
            coordinator_handle(
                relay,
                tokenizer.clone(),
                model.clone(),
                max_thinking_tokens,
                None,
                shutdown.clone(),
            )
        })
        .collect();
    let dp = Arc::new(DpCoordinator::new(handles));
    spawn_coordinator_observe(Arc::clone(&dp));
    build_router(dp)
}

fn spawn_coordinator_observe(dp: Arc<DpCoordinator>) {
    let Ok(rt) = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    else {
        log::error!("observe: tokio runtime build failed");
        return;
    };
    let cached = Arc::clone(&dp.cached_stats);
    crate::observe::spawn_observe_task(move || {
        let snap = rt.block_on(dp.query_stats_all(Duration::from_secs(5)));
        if let Ok(mut guard) = cached.write() {
            *guard = Some(snap.clone());
        }
        Some(snap)
    });
}

fn coordinator_handle(
    relay: RelayCoordinator,
    tokenizer: OpenAiTokenizer,
    model: impl Into<String>,
    max_thinking_tokens: usize,
    multimodal: Option<(crate::LocalMultimodalTx, MultimodalKind)>,
    shutdown: Option<crate::ServeShutdown>,
) -> Arc<CoordinatorHandle> {
    let sinks = relay.completion_sinks();
    let relay = Arc::new(Mutex::new(relay));
    let in_flight = Arc::new(AtomicUsize::new(0));
    let (submit_tx, submit_rx) = std::sync::mpsc::channel::<CoordSubmission>();

    {
        let relay = Arc::clone(&relay);
        let in_flight = Arc::clone(&in_flight);
        let sinks = sinks.clone();
        std::thread::Builder::new()
            .name("arle-coordinator-lockstep".to_string())
            .spawn(move || lockstep_loop(relay, in_flight, sinks, submit_rx, shutdown))
            .expect("spawn coordinator lockstep thread");
    }

    let (multimodal_tx, multimodal_kind) = multimodal.unzip();

    let template_defaults_thinking = tokenizer.defaults_thinking_on();
    let think_token_ids = tokenizer.think_token_ids();
    Arc::new(CoordinatorHandle {
        model: model.into(),
        tokenizer: Mutex::new(tokenizer),
        max_thinking_tokens,
        template_defaults_thinking,
        think_token_ids,
        next_request_id: AtomicU64::new(1),
        in_flight,
        submit_tx,
        sinks,
        relay,
        stats_request_id: AtomicU64::new(0),
        multimodal_tx,
        multimodal_kind,
    })
}

fn build_router(state: Arc<DpCoordinator>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/messages", post(anthropic_messages))
        .route("/v1/messages/count_tokens", post(anthropic_count_tokens))
        .route("/v1/models", get(list_models))
        .route("/v1/embeddings", post(embeddings_not_implemented))
        .route("/v1/stats", get(stats))
        .route("/metrics", get(metrics))
        .route("/v1/observe/query", get(observe_query))
        .route("/dashboard", get(dashboard_page))
        .fallback(fallback_404)
        .layer(DefaultBodyLimit::max(256 * 1024 * 1024))
        .with_state(state)
}

/// Broadcast one `CancelRequest` per id, in order, before the caller's next
/// broadcast on the same locked `coord` — every rank applies a tick's
/// cancellations before that tick's admissions/step.
fn broadcast_cancellations(
    coord: &mut RelayCoordinator,
    cancellations: &[u64],
) -> anyhow::Result<()> {
    for &request_id in cancellations {
        coord.broadcast(&RelayEnvelope::CancelRequest { request_id })?;
    }
    Ok(())
}

/// Broadcasts exactly one `TickAdmissions` per step to every worker (carrying
/// that step's drained submissions, empty on pure-decode ticks), stepping while
/// `in_flight` remains and parking when idle. Per-rank tick acks pace the busy
/// path to engine speed ([`wait_for_ack_window`]), so ticks never flood the
/// worker FIFOs. A broadcast failure breaks the lockstep group, so it logs at
/// error and stops (workers then EOF and exit).
fn lockstep_loop(
    relay: Arc<Mutex<RelayCoordinator>>,
    in_flight: Arc<AtomicUsize>,
    sinks: CompletionSinks,
    submit_rx: SyncReceiver<CoordSubmission>,
    shutdown: Option<crate::ServeShutdown>,
) {
    // Group teardown on a fatal lockstep error (#135): failing the sinks stops
    // the hang, but surviving workers spin inside the broken NCCL collective
    // until killed. Requesting serve shutdown unwinds the coordinator HTTP
    // loop, whose exit drops the worker guard — pipe EOF, 5s grace, SIGKILL.
    let teardown = |sinks: &CompletionSinks, reason: &str| {
        sinks.fail_all(reason);
        if let Some(shutdown) = &shutdown {
            log::error!(
                "[coordinator] tearing down the serve (worker group unwound by the child guard)"
            );
            shutdown.request();
        }
    };
    let mut seq: u64 = 0;
    let mut submit_open = true;
    loop {
        // LOCKSTEP INVARIANT (load-bearing): `in_flight > 0` must strictly contain
        // every rank's non-idle window — increment BEFORE submit, decrement only
        // AFTER the terminal delta. Then the coordinator can only OVER-send
        // decode-only ticks (idle workers skip them), never UNDER-send one a worker
        // needs for an NCCL collective. Reordering the inc/dec can deadlock NCCL.
        let busy = in_flight.load(Ordering::Acquire) > 0;
        if busy {
            // Flow control: wait for the ack window BEFORE the drain, so a
            // submission landing during the wait coalesces into the very next
            // allowed tick instead of queueing behind unacked ones.
            if let Some(reason) = wait_for_ack_window(&relay, seq) {
                teardown(
                    &sinks,
                    &format!("multiproc worker group stalled ({reason})"),
                );
                return;
            }
        }

        let mut drained: Vec<WireRequest> = Vec::new();
        let mut cancellations: Vec<u64> = Vec::new();
        loop {
            match submit_rx.try_recv() {
                Ok(CoordSubmission::Submit(request)) => drained.push(*request),
                Ok(CoordSubmission::Cancel(request_id)) => cancellations.push(request_id),
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    submit_open = false;
                    break;
                }
            }
        }

        if !drained.is_empty() || !cancellations.is_empty() || busy {
            let requests = drained;
            let send = {
                let mut coord = relay
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                // Cancellations broadcast BEFORE this tick's admissions, same
                // lock scope, so every rank applies them at the identical
                // point relative to the step they gate.
                broadcast_cancellations(&mut coord, &cancellations).and_then(|()| {
                    coord.broadcast(&RelayEnvelope::TickAdmissions { seq, requests })
                })
            };
            if let Err(err) = send {
                log::error!(
                    "[coordinator] tick #{seq} admission broadcast failed; stopping lockstep \
                     loop (workers will EOF and exit): {err:#}"
                );
                // Worker group broken: fail every in-flight sink so awaiting
                // handlers return an error (and drop their guard) instead of hanging.
                teardown(
                    &sinks,
                    "multiproc worker group failed (admission broadcast error)",
                );
                return;
            }
            seq += 1;
            continue;
        }

        if !submit_open {
            return;
        }
        match submit_rx.recv_timeout(IDLE_PARK) {
            Ok(CoordSubmission::Submit(request)) => {
                let send = {
                    let mut coord = relay
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    coord.broadcast(&RelayEnvelope::TickAdmissions {
                        seq,
                        requests: vec![*request],
                    })
                };
                if let Err(err) = send {
                    log::error!(
                        "[coordinator] tick #{seq} admission broadcast failed; stopping lockstep \
                         loop: {err:#}"
                    );
                    teardown(
                        &sinks,
                        "multiproc worker group failed (admission broadcast error)",
                    );
                    return;
                }
                seq += 1;
            }
            // `in_flight == 0` here (the idle branch) means every rank's
            // engine is provably idle (the LOCKSTEP INVARIANT above), so this
            // is always a no-op — broadcast anyway rather than special-case
            // "known no-op", the coordinator has no visibility into engine
            // state to verify that itself. No seq/tick consumed: nothing to
            // admit, no step for workers to run.
            Ok(CoordSubmission::Cancel(request_id)) => {
                let send = {
                    let mut coord = relay
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    coord.broadcast(&RelayEnvelope::CancelRequest { request_id })
                };
                if let Err(err) = send {
                    log::error!(
                        "[coordinator] idle cancel broadcast failed; stopping lockstep loop: {err:#}"
                    );
                    teardown(
                        &sinks,
                        "multiproc worker group failed (cancel broadcast error)",
                    );
                    return;
                }
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => submit_open = false,
        }
    }
}

/// One coordinator request's result, reconstructed from relay completion deltas.
struct CollectedGeneration {
    prompt_tokens: usize,
    generated_tokens: Vec<u32>,
    /// Behavior logprobs accumulated from the deltas (empty when uncaptured).
    gen_logprobs: Vec<f32>,
    /// OpenAI logprobs capture, one entry per generated token (entry 0 =
    /// sampled token's full-dist logprob, 1.. = top-N alternatives). Empty
    /// when the request did not ask or the backend does not surface it.
    top_logprobs: Vec<Vec<(u32, f32)>>,
    finish: Option<FinishReason>,
}

/// RAII guard for one in-flight request: `Drop` does the `in_flight` decrement +
/// sink unregister, so a cancelled HTTP task still releases both (a leaked
/// `in_flight` would pin the lockstep loop into a forever busy-spin). Armed after
/// sink registration and before the submit await to cover the whole window.
struct InFlightGuard {
    in_flight: Arc<AtomicUsize>,
    sinks: CompletionSinks,
    request_id: u64,
    submit_tx: SyncSender<CoordSubmission>,
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.in_flight.fetch_sub(1, Ordering::AcqRel);
        self.sinks.unregister(self.request_id);
        // Tell every rank's engine to stop working on this request if it's
        // still queued/active — closes the gap where a client disconnecting
        // only released coordinator-side bookkeeping and left the request a
        // permanent zombie in the engine (2026-07-05 multiproc hang
        // investigation's last open item — docs/experience/errors/2026-07-05-multiproc-lockstep-ack-hang-no-timeout.md).
        // Best-effort: a closed channel means the lockstep loop already exited.
        let _ = self
            .submit_tx
            .send(CoordSubmission::Cancel(self.request_id));
    }
}

/// The last SSE frame of a stream: the finish-reason chunk, an optional
/// `stream_options.include_usage` trailer (empty `choices`, populated
/// `usage`), then `[DONE]`.
fn sse_final_frame(
    final_chunk: &serde_json::Value,
    usage_chunk: Option<serde_json::Value>,
) -> Vec<u8> {
    let mut out = format!(
        "data: {}\n\n",
        serde_json::to_string(final_chunk).unwrap_or_default()
    );
    if let Some(usage_chunk) = usage_chunk {
        out.push_str(&format!(
            "data: {}\n\n",
            serde_json::to_string(&usage_chunk).unwrap_or_default()
        ));
    }
    out.push_str("data: [DONE]\n\n");
    out.into_bytes()
}

/// Set up the relay sink + in_flight guard + submit for a streaming request.
/// Returns the per-token delta receiver and the guard (caller must keep alive
/// until streaming ends so `in_flight` is decremented only after the last delta).
fn streaming_submit(
    state: &CoordinatorHandle,
    prompt_tokens: Vec<u32>,
    max_tokens: usize,
    sampling: SamplingParams,
    response_format: Option<crate::grammar::ResponseFormat>,
) -> Result<
    (
        tokio::sync::mpsc::UnboundedReceiver<RelayCompletionDelta>,
        InFlightGuard,
    ),
    ApiError,
> {
    let request_id = state.alloc_request_id();
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<RelayCompletionDelta>();
    state
        .sinks
        .register(request_id, tx)
        .map_err(ApiError::from)?;
    state.in_flight.fetch_add(1, Ordering::AcqRel);
    let guard = InFlightGuard {
        in_flight: Arc::clone(&state.in_flight),
        sinks: state.sinks.clone(),
        request_id,
        submit_tx: state.submit_tx.clone(),
    };
    state
        .submit_tx
        .send(CoordSubmission::Submit(Box::new(WireRequest {
            request_id,
            prompt_tokens,
            max_tokens,
            sampling,
            response_format,
        })))
        .map_err(|_| ApiError::internal("coordinator lockstep loop closed; cannot submit"))?;
    Ok((rx, guard))
}

async fn submit_and_collect(
    state: &CoordinatorHandle,
    prompt_tokens: Vec<u32>,
    max_tokens: usize,
    sampling: SamplingParams,
    response_format: Option<crate::grammar::ResponseFormat>,
) -> Result<CollectedGeneration, ApiError> {
    let prompt_len = prompt_tokens.len();
    let (mut rx, _guard) =
        streaming_submit(state, prompt_tokens, max_tokens, sampling, response_format)?;
    let mut generated_tokens: Vec<u32> = Vec::new();
    let mut gen_logprobs: Vec<f32> = Vec::new();
    let mut top_logprobs: Vec<Vec<(u32, f32)>> = Vec::new();
    let mut finish: Option<FinishReason> = None;
    let mut error: Option<String> = None;
    while let Some(mut delta) = rx.recv().await {
        generated_tokens.extend_from_slice(&delta.token_ids);
        gen_logprobs.extend_from_slice(&delta.logprobs);
        top_logprobs.append(&mut delta.top_logprobs);
        let done = delta.is_done();
        error = error.or(delta.error);
        if done {
            finish = delta.finish_reason;
            break;
        }
    }
    if let Some(message) = error {
        return Err(ApiError::internal(message));
    }
    Ok(CollectedGeneration {
        prompt_tokens: prompt_len,
        generated_tokens,
        gen_logprobs,
        top_logprobs,
        finish,
    })
}

fn encode(state: &CoordinatorHandle, text: &str) -> Result<Vec<u32>, ApiError> {
    let tokenizer = state
        .tokenizer
        .lock()
        .map_err(|_| ApiError::internal("tokenizer lock poisoned"))?;
    Ok(tokenizer.encode(text)?)
}

fn decode(state: &CoordinatorHandle, tokens: &[u32]) -> Result<String, ApiError> {
    let tokenizer = state
        .tokenizer
        .lock()
        .map_err(|_| ApiError::internal("tokenizer lock poisoned"))?;
    Ok(tokenizer.decode(tokens)?)
}

/// Strip trailing stop sequences from the generated text. The engine stops on
/// the stop token but includes it in the output; OpenAI omits the stop
/// sequence from the response.
fn strip_stop_strings(text: &str, stop: &[String]) -> String {
    let mut result = text.to_string();
    for s in stop {
        if s.is_empty() {
            continue;
        }
        if let Some(idx) = result.rfind(s) {
            result.truncate(idx);
        }
    }
    result
}

/// Split a decoded chat completion into visible content + response tool calls
/// (the non-streaming twin of [`crate::sse_util::StreamPipeline`]).
///
/// Tool-less requests pass the text straight through so [`from_parts`]' standard
/// reasoning split runs (byte-identical). When tools are active, the canonical
/// [`split_reasoning`] runs FIRST: the chat template pre-fills `<think>` into
/// the *prompt*, so thinking output arrives as `reasoning</think>answer` with
/// no opening tag — `openai_parse_tool_calls`' paired-tag strip misses that
/// form and leaks reasoning + a stray `</think>` into content. Only the
/// content half then feeds tool parsing; reasoning is dropped (no wire lane in
/// tools mode) and the response splits with thinking OFF (third return value)
/// — re-scanning already-split text would move all content into reasoning.
///
/// [`from_parts`]: ChatCompletionResponse::from_parts
/// [`split_reasoning`]: crate::schema::split_reasoning
fn finalize_chat_content(
    decoded: String,
    tools_active: bool,
    thinking: bool,
) -> (String, Vec<ResponseToolCall>, bool) {
    if !tools_active {
        return (decoded, Vec::new(), thinking);
    }
    let (_reasoning, content) = crate::schema::split_reasoning(&decoded, thinking);
    let (content, calls) = chat::openai_parse_tool_calls(&content);
    let tool_calls = calls
        .iter()
        .enumerate()
        .map(|(index, call)| ResponseToolCall::from_parsed(call, index))
        .collect();
    (content, tool_calls, false)
}

/// `true` when the rendered prompt ends with an open `<think>` — the chat
/// template prefilled thinking (e.g. Qwen3.6's Jinja defaults `enable_thinking`
/// on), so output arrives as `reasoning</think>answer` regardless of what the
/// request or server flags said. The parse-side reasoning gate must key off
/// this rendered truth: flags alone leak the bare `</think>` form.
fn prompt_prefills_think(prompt: &str) -> bool {
    prompt.trim_end().ends_with("<think>")
}

/// OpenAI completions `logprobs` object from the engine capture
/// (`SlotToken::top_logprobs`: entry 0 = the sampled token's full-distribution
/// logprob, 1.. = the top-N alternatives). 501 when the capture is missing —
/// the backend/model path does not surface logprobs.
fn completion_logprobs_value(
    state: &CoordinatorHandle,
    token_ids: &[u32],
    captures: &[Vec<(u32, f32)>],
) -> Result<serde_json::Value, ApiError> {
    if captures.len() != token_ids.len() || captures.iter().any(Vec::is_empty) {
        return Err(ApiError::not_implemented(
            "logprobs are not surfaced by this backend/model path \
             (supported: the CUDA Qwen3.5/3.6 executor)",
        ));
    }
    let tokenizer = state
        .tokenizer
        .lock()
        .map_err(|_| ApiError::internal("tokenizer lock poisoned"))?;
    let mut tokens: Vec<String> = Vec::with_capacity(token_ids.len());
    let mut token_logprobs: Vec<f32> = Vec::with_capacity(token_ids.len());
    let mut top_logprobs: Vec<serde_json::Value> = Vec::with_capacity(token_ids.len());
    let mut text_offset: Vec<usize> = Vec::with_capacity(token_ids.len());
    let mut offset = 0usize;
    for (&tid, cap) in token_ids.iter().zip(captures) {
        let piece = tokenizer.decode(&[tid])?;
        text_offset.push(offset);
        offset += piece.len();
        tokens.push(piece);
        token_logprobs.push(cap[0].1);
        let mut alts = serde_json::Map::with_capacity(cap.len() - 1);
        for &(alt, lp) in &cap[1..] {
            let key = tokenizer.decode(&[alt])?;
            // Distinct token ids can decode to one string; keep the more
            // probable (first, entries are probability-descending).
            alts.entry(key).or_insert_with(|| lp.into());
        }
        top_logprobs.push(serde_json::Value::Object(alts));
    }
    Ok(serde_json::json!({
        "tokens": tokens,
        "token_logprobs": token_logprobs,
        "top_logprobs": top_logprobs,
        "text_offset": text_offset,
    }))
}

/// OpenAI chat `logprobs` object (`content` entries). Uses the full capture
/// when present; otherwise falls back to the legacy behavior-logprob shape
/// (token-id strings, empty `top_logprobs`) so backends without the capture
/// keep their previous wire behavior.
fn chat_logprobs_value(
    state: &CoordinatorHandle,
    token_ids: &[u32],
    behavior_logprobs: &[f32],
    captures: &[Vec<(u32, f32)>],
) -> Result<serde_json::Value, ApiError> {
    if captures.len() == token_ids.len() && !captures.iter().any(Vec::is_empty) {
        let tokenizer = state
            .tokenizer
            .lock()
            .map_err(|_| ApiError::internal("tokenizer lock poisoned"))?;
        let mut content: Vec<serde_json::Value> = Vec::with_capacity(token_ids.len());
        for (&tid, cap) in token_ids.iter().zip(captures) {
            let mut alts: Vec<serde_json::Value> = Vec::with_capacity(cap.len() - 1);
            for &(alt, lp) in &cap[1..] {
                alts.push(serde_json::json!({
                    "token": tokenizer.decode(&[alt])?,
                    "logprob": lp,
                }));
            }
            content.push(serde_json::json!({
                "token": tokenizer.decode(&[tid])?,
                "logprob": cap[0].1,
                "top_logprobs": alts,
            }));
        }
        return Ok(serde_json::json!({ "content": content }));
    }
    if behavior_logprobs.len() != token_ids.len() {
        return Ok(serde_json::Value::Null);
    }
    let content: Vec<serde_json::Value> = token_ids
        .iter()
        .zip(behavior_logprobs)
        .map(|(&t, &lp)| {
            serde_json::json!({"token": t.to_string(), "logprob": lp, "top_logprobs": []})
        })
        .collect();
    Ok(serde_json::json!({ "content": content }))
}

async fn completions(
    State(state): State<Arc<DpCoordinator>>,
    request: Result<Json<CompletionRequest>, axum::extract::rejection::JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(request) = request.map_err(|rejection| {
        ApiError::bad_request(format!("Failed to parse request body as JSON: {rejection}"))
    })?;
    request.validate()?;
    let mut sampling = request.sampling_params();
    // Convert string `stop` sequences to token ids. The engine only supports
    // token-id stops, so try the string as-is and with a leading space/newline
    // (models often emit a leading space before a word).
    if let Some(stop_strings) = &request.stop {
        for s in stop_strings {
            if s.is_empty() {
                continue;
            }
            for variant in [s.clone(), format!(" {s}"), format!("\n{s}")] {
                let ids = encode(&state, &variant)?;
                sampling.stop_token_ids.extend(ids);
            }
        }
    }
    let max_tokens = sampling.max_new_tokens.unwrap_or(THINK_CONTENT_HEADROOM);
    // Token-id prompt → feed verbatim (exact-token multi-turn); text → tokenize.
    let prompt_tokens = match &request.prompt {
        crate::schema::PromptInput::Tokens(ids) => ids.clone(),
        crate::schema::PromptInput::Text(text) => encode(&state, text)?,
    };
    // Local engine: send usage by default (a client may opt out with
    // stream_options.include_usage=false). Cloud providers gate this behind
    // the flag; a local server has no reason to withhold token counts.
    let include_usage = request
        .stream_options
        .as_ref()
        .is_none_or(|o| o.include_usage);

    if request.stream.unwrap_or(false) {
        if request.logprobs.is_some() {
            return Err(ApiError::bad_request(
                "logprobs with stream=true is not supported yet",
            ));
        }
        let prompt_len = prompt_tokens.len();
        let (mut rx, guard) = streaming_submit(
            &state,
            prompt_tokens,
            max_tokens,
            sampling,
            request.response_format.clone(),
        )?;
        let id = format!("cmpl-{}", Uuid::new_v4().simple());
        let created = unix_time_secs();
        let model = state.model.clone();
        let state_clone = Arc::clone(&state);
        // Bounded channel: backpressure keeps the task from racing too far ahead.
        let (chunk_tx, chunk_rx) =
            tokio::sync::mpsc::channel::<Result<Vec<u8>, std::convert::Infallible>>(64);
        tokio::spawn(async move {
            // `guard` dropped when this task exits, decrementing in_flight + unregistering sink.
            let _guard = guard;
            let mut completion_count = 0usize;
            let mut detok = IncrementalDetokenizer::default();
            // Keep-alive pings while generation runs (long prefills emit no tokens),
            // which also surfaces a client disconnect before the first token.
            let mut keepalive = keepalive_ticker().await;
            while let Some(delta) = recv_or_keepalive(&mut rx, &mut keepalive, &chunk_tx).await {
                if let Some(err) = delta.error {
                    let chunk =
                        serde_json::to_string(&stream_error_chunk(&id, created, &model, &err))
                            .unwrap_or_default();
                    let _ = chunk_tx
                        .send(Ok(format!("data: {chunk}\n\ndata: [DONE]\n\n").into_bytes()))
                        .await;
                    break;
                }
                if !delta.token_ids.is_empty() {
                    completion_count += delta.token_ids.len();
                    let text = {
                        let tok = state_clone
                            .tokenizer
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        detok.push(&tok, &delta.token_ids)
                    };
                    let chunk = serde_json::to_string(&completion_stream_chunk(
                        &id, created, &model, text, None, None,
                    ))
                    .unwrap_or_default();
                    if chunk_tx
                        .send(Ok(format!("data: {chunk}\n\n").into_bytes()))
                        .await
                        .is_err()
                    {
                        break; // Client disconnected; guard drops, releasing in_flight.
                    }
                }
                if delta.finish {
                    let fr = finish_reason(delta.finish_reason.as_ref());
                    // Bytes still held for an unfinished codepoint: the stream is
                    // over, so they will never complete.
                    let tail = {
                        let tok = state_clone
                            .tokenizer
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        detok.flush(&tok)
                    };
                    let final_chunk =
                        completion_stream_chunk(&id, created, &model, tail, Some(fr), None);
                    let usage_chunk = include_usage.then(|| {
                        let usage = serde_json::to_value(Usage::new(prompt_len, completion_count))
                            .unwrap_or_default();
                        stream_usage_chunk(&id, created, &model, "text_completion", usage)
                    });
                    let _ = chunk_tx
                        .send(Ok(sse_final_frame(&final_chunk, usage_chunk)))
                        .await;
                    break;
                }
            }
        });
        let body =
            axum::body::Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(chunk_rx));
        return Ok((
            [
                (header::CONTENT_TYPE, "text/event-stream"),
                (header::CACHE_CONTROL, "no-cache"),
                (header::HeaderName::from_static("x-accel-buffering"), "no"),
            ],
            body,
        )
            .into_response());
    }

    let return_token_ids = request.return_token_ids.unwrap_or(false);
    let prompt_token_ids = return_token_ids.then(|| prompt_tokens.clone());
    let n = sampling.n.max(1);

    if n > 1 {
        let mut choices = Vec::with_capacity(n);
        for i in 0..n {
            let mut params = sampling.clone();
            params.seed = Some(sampling.seed.unwrap_or(0).wrapping_add(i as u64));
            let outcome = submit_and_collect(
                &state,
                prompt_tokens.clone(),
                max_tokens,
                params,
                request.response_format.clone(),
            )
            .await?;
            let text = decode(&state, &outcome.generated_tokens)?;
            let text = request
                .stop
                .as_deref()
                .map_or_else(|| text.clone(), |stop| strip_stop_strings(&text, stop));
            let logprobs_value = match request.logprobs {
                Some(_) => Some(completion_logprobs_value(
                    &state,
                    &outcome.generated_tokens,
                    &outcome.top_logprobs,
                )?),
                None => None,
            };
            choices.push(CompletionChoice {
                text,
                index: i,
                logprobs: logprobs_value,
                finish_reason: finish_reason(outcome.finish.as_ref()).to_string(),
                token_ids: return_token_ids.then_some(outcome.generated_tokens.clone()),
                prompt_token_ids: prompt_token_ids.clone(),
            });
        }
        let total_completion: usize = choices.iter().map(|c| c.text.len()).sum();
        return Ok(Json(CompletionResponse {
            id: format!("cmpl-{}", uuid::Uuid::new_v4().simple()),
            object: "text_completion",
            created: unix_time_secs(),
            model: state.model.clone(),
            choices,
            usage: Usage::new(prompt_tokens.len(), total_completion),
            system_fingerprint: SYSTEM_FINGERPRINT,
        })
        .into_response());
    }

    let outcome = submit_and_collect(
        &state,
        prompt_tokens,
        max_tokens,
        sampling,
        request.response_format.clone(),
    )
    .await?;
    let text = decode(&state, &outcome.generated_tokens)?;
    let text = request
        .stop
        .as_deref()
        .map_or_else(|| text.clone(), |stop| strip_stop_strings(&text, stop));
    let logprobs_value = match request.logprobs {
        Some(_) => Some(completion_logprobs_value(
            &state,
            &outcome.generated_tokens,
            &outcome.top_logprobs,
        )?),
        None => None,
    };
    Ok(Json(CompletionResponse::from_parts(
        state.model.clone(),
        text,
        outcome.prompt_tokens,
        outcome.generated_tokens.len(),
        outcome.finish.as_ref(),
        return_token_ids.then_some(outcome.generated_tokens),
        prompt_token_ids,
        logprobs_value,
    ))
    .into_response())
}

async fn chat_completions(
    State(state): State<Arc<DpCoordinator>>,
    request: Result<Json<ChatCompletionRequest>, axum::extract::rejection::JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(request) = request.map_err(|rejection| {
        ApiError::bad_request(format!("Failed to parse request body as JSON: {rejection}"))
    })?;
    request.validate()?;
    let mut sampling = request.sampling_params();
    // Convert string `stop` sequences to token ids (engine only supports
    // token-id stops). Try as-is and with leading space/newline.
    if let Some(stop_strings) = &request.stop {
        for s in stop_strings {
            if s.is_empty() {
                continue;
            }
            for variant in [s.clone(), format!(" {s}"), format!("\n{s}")] {
                let ids = encode(&state, &variant)?;
                sampling.stop_token_ids.extend(ids);
            }
        }
    }
    let stream = request.stream.unwrap_or(false);
    if stream && request.logprobs.unwrap_or(false) {
        return Err(ApiError::bad_request(
            "logprobs with stream=true is not supported yet",
        ));
    }
    // Local engine: send usage by default (a client may opt out with
    // stream_options.include_usage=false). Cloud providers gate this behind
    // the flag; a local server has no reason to withhold token counts.
    let include_usage = request
        .stream_options
        .as_ref()
        .is_none_or(|o| o.include_usage);
    // Thinking defaults on when a budget is configured OR the checkpoint is a
    // reasoning model (DeepSeek-V4-Flash) that degenerates when forced
    // non-thinking; `0` + non-reasoning keeps it off and byte-identical.
    let thinking =
        request.enable_thinking(state.max_thinking_tokens > 0 || state.template_defaults_thinking);
    // Tool definitions the model may call — empty when none supplied or
    // `tool_choice=none`; gates prompt rendering AND response parsing so a
    // tool-less request stays byte-identical on the wire.
    let tools_active = request.wants_tools();
    let tools: &[chat::OpenAiToolDefinition] = if tools_active { &request.tools } else { &[] };
    let reasoning_effort = request
        .chat_template_kwargs
        .as_ref()
        .and_then(|kwargs| kwargs.get("reasoning_effort"))
        .and_then(serde_json::Value::as_str);
    // Pass tool_choice to the template so it can instruct the model (required /
    // forced function). Templates that ignore it stay byte-identical.
    let mut template_kwargs = request.chat_template_kwargs.clone().unwrap_or_default();
    if let Some(tc) = &request.tool_choice {
        template_kwargs.insert("tool_choice".to_string(), tc.clone());
    }
    let template_kwargs = if template_kwargs.is_empty() {
        None
    } else {
        Some(template_kwargs)
    };
    let think_budget = state.resolve_think_budget(thinking, reasoning_effort, None, &mut sampling);
    let max_tokens = sampling.max_new_tokens.unwrap_or_else(|| {
        think_budget.map_or(THINK_CONTENT_HEADROOM, |budget| {
            budget + THINK_CONTENT_HEADROOM
        })
    });

    // Multimodal dispatch: if the backend is a VLM and the request has images,
    // route through the in-process multimodal channel instead of the relay.
    if let Some(kind) = state.multimodal_kind {
        let images = crate::multimodal::extract_images(&request.messages, Some(kind))?;
        if !images.is_empty() {
            if stream {
                return Err(ApiError::bad_request(
                    "stream=true is not supported for multimodal chat yet",
                ));
            }
            let prompt = if kind == MultimodalKind::DeepseekOcr {
                crate::multimodal::build_deepseek_ocr_prompt(&request.messages)
            } else {
                let tokenizer = state
                    .tokenizer
                    .lock()
                    .map_err(|_| ApiError::internal("tokenizer lock poisoned"))?;
                tokenizer.render_chat_full(
                    &request.messages,
                    template_kwargs.as_ref(),
                    tools,
                    thinking,
                    reasoning_effort,
                )?
            };
            let thinking = thinking || prompt_prefills_think(&prompt);
            let prompt = crate::multimodal::expand_image_markers(&prompt, &images, Some(kind))?;
            let prompt_tokens = encode(&state, &prompt)?;
            let prompt_token_count = prompt_tokens.len();
            let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
            state
                .multimodal_tx
                .as_ref()
                .ok_or_else(|| ApiError::internal("multimodal channel not initialized"))?
                .send(crate::LocalMultimodalRequest {
                    prompt_tokens,
                    images,
                    max_tokens,
                    sampling,
                    response_tx: resp_tx,
                })
                .map_err(|_| ApiError::internal("multimodal channel closed"))?;
            let delta = tokio::time::timeout(std::time::Duration::from_secs(300), resp_rx)
                .await
                .map_err(|_| ApiError::internal("multimodal request timed out"))?
                .map_err(|_| ApiError::internal("multimodal response channel closed"))?;
            if let Some(err) = delta.error {
                return Err(ApiError::internal(err));
            }
            let decoded = decode(&state, &delta.token_ids)?;
            let (content, tool_calls, split_thinking) =
                finalize_chat_content(decoded, tools_active, thinking);
            let reasoning_tokens =
                count_reasoning_tokens(&delta.token_ids, state.think_token_ids, thinking);
            return Ok(Json(ChatCompletionResponse::from_parts(
                state.model.clone(),
                content,
                prompt_token_count,
                delta.token_ids.len(),
                reasoning_tokens,
                delta.finish_reason.as_ref(),
                split_thinking,
                tool_calls,
                None,
            ))
            .into_response());
        }
    }

    let prompt = {
        let tokenizer = state
            .tokenizer
            .lock()
            .map_err(|_| ApiError::internal("tokenizer lock poisoned"))?;
        tokenizer.render_chat_full(
            &request.messages,
            request.chat_template_kwargs.as_ref(),
            tools,
            thinking,
            reasoning_effort,
        )?
    };
    let prompt_tokens = encode(&state, &prompt)?;
    let thinking = thinking || prompt_prefills_think(&prompt);

    if stream {
        let prompt_len = prompt_tokens.len();
        let (mut rx, guard) = streaming_submit(
            &state,
            prompt_tokens,
            max_tokens,
            sampling,
            request.response_format.clone(),
        )?;
        let id = format!("chatcmpl-{}", Uuid::new_v4().simple());
        let created = unix_time_secs();
        let model = state.model.clone();
        let state_clone = Arc::clone(&state);
        // Bounded channel: backpressure keeps the task from racing too far ahead.
        let (chunk_tx, chunk_rx) =
            tokio::sync::mpsc::channel::<Result<Vec<u8>, std::convert::Infallible>>(64);
        tokio::spawn(async move {
            // `guard` dropped when this task exits, decrementing in_flight + unregistering sink.
            let _guard = guard;
            let mut completion_count = 0usize;
            let mut reasoning_count = 0usize;
            let mut in_thinking = thinking;
            // Converged reasoning-then-tools pipeline (shared with /v1/messages).
            let mut pipeline = StreamPipeline::new(thinking, tools_active);
            let mut completed_calls: Vec<chat::ToolCall> = Vec::new();
            // OpenAI convention: the first emitted chunk's delta carries `role`.
            let mut role_sent = false;
            let mut with_role = move |mut delta: serde_json::Value| {
                if !std::mem::replace(&mut role_sent, true) {
                    delta["role"] = serde_json::json!("assistant");
                }
                delta
            };
            // Keep-alive pings while generation runs (long prefills emit no tokens),
            // which also surfaces a client disconnect before the first token.
            let mut keepalive = keepalive_ticker().await;
            let mut detok = IncrementalDetokenizer::default();
            'stream: while let Some(delta) =
                recv_or_keepalive(&mut rx, &mut keepalive, &chunk_tx).await
            {
                if let Some(err) = delta.error {
                    let chunk =
                        serde_json::to_string(&stream_error_chunk(&id, created, &model, &err))
                            .unwrap_or_default();
                    let _ = chunk_tx
                        .send(Ok(format!("data: {chunk}\n\ndata: [DONE]\n\n").into_bytes()))
                        .await;
                    break;
                }
                if !delta.token_ids.is_empty() {
                    completion_count += delta.token_ids.len();
                    if let Some((start, end)) = state_clone.think_token_ids {
                        for &token in &delta.token_ids {
                            if token == start {
                                in_thinking = true;
                            } else if token == end {
                                in_thinking = false;
                            } else if in_thinking {
                                reasoning_count += 1;
                            }
                        }
                    }
                    let text = {
                        let tok = state_clone
                            .tokenizer
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        detok.push(&tok, &delta.token_ids)
                    };
                    let (pieces, calls) = pipeline.push(&text);
                    completed_calls.extend(calls);
                    for piece in pieces {
                        let chunk = serde_json::to_string(&chat_stream_chunk(
                            &id,
                            created,
                            &model,
                            with_role(piece.into_delta()),
                            None,
                        ))
                        .unwrap_or_default();
                        if chunk_tx
                            .send(Ok(format!("data: {chunk}\n\n").into_bytes()))
                            .await
                            .is_err()
                        {
                            break 'stream; // Client disconnected; guard drops, releasing in_flight.
                        }
                    }
                }
                if delta.finish {
                    // Bytes held for an unfinished codepoint go through the pipeline
                    // before it closes — the stream is over, they never complete.
                    let tail = {
                        let tok = state_clone
                            .tokenizer
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        detok.flush(&tok)
                    };
                    let mut pieces = Vec::new();
                    if !tail.is_empty() {
                        let (tail_pieces, calls) = pipeline.push(&tail);
                        pieces.extend(tail_pieces);
                        completed_calls.extend(calls);
                    }
                    // Flush the pipeline: truncated thinking + buffered tool tail.
                    let (finish_pieces, calls) = pipeline.finish();
                    pieces.extend(finish_pieces);
                    completed_calls.extend(calls);
                    for piece in pieces {
                        let chunk = serde_json::to_string(&chat_stream_chunk(
                            &id,
                            created,
                            &model,
                            with_role(piece.into_delta()),
                            None,
                        ))
                        .unwrap_or_default();
                        if chunk_tx
                            .send(Ok(format!("data: {chunk}\n\n").into_bytes()))
                            .await
                            .is_err()
                        {
                            break 'stream;
                        }
                    }
                    // Each completed tool call rides one OpenAI streaming delta.
                    for (index, call) in completed_calls.iter().enumerate() {
                        let tool_call = ResponseToolCall::from_parsed(call, index);
                        let delta = with_role(serde_json::json!({
                            "tool_calls": [{
                                "index": index,
                                "id": tool_call.id,
                                "type": tool_call.call_type,
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments,
                                },
                            }],
                        }));
                        let chunk = serde_json::to_string(&chat_stream_chunk(
                            &id, created, &model, delta, None,
                        ))
                        .unwrap_or_default();
                        if chunk_tx
                            .send(Ok(format!("data: {chunk}\n\n").into_bytes()))
                            .await
                            .is_err()
                        {
                            break 'stream;
                        }
                    }
                    let fr = if completed_calls.is_empty() {
                        finish_reason(delta.finish_reason.as_ref())
                    } else {
                        "tool_calls"
                    };
                    let final_chunk = chat_stream_chunk(
                        &id,
                        created,
                        &model,
                        with_role(serde_json::json!({})),
                        Some(fr),
                    );
                    let usage_chunk = include_usage.then(|| {
                        let usage = if thinking {
                            Usage::with_reasoning(prompt_len, completion_count, reasoning_count)
                        } else {
                            Usage::new(prompt_len, completion_count)
                        };
                        let usage = serde_json::to_value(usage).unwrap_or_default();
                        stream_usage_chunk(&id, created, &model, "chat.completion.chunk", usage)
                    });
                    let _ = chunk_tx
                        .send(Ok(sse_final_frame(&final_chunk, usage_chunk)))
                        .await;
                    break;
                }
            }
        });
        let body =
            axum::body::Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(chunk_rx));
        return Ok((
            [
                (header::CONTENT_TYPE, "text/event-stream"),
                (header::CACHE_CONTROL, "no-cache"),
                (header::HeaderName::from_static("x-accel-buffering"), "no"),
            ],
            body,
        )
            .into_response());
    }

    let n = sampling.n.max(1);

    if n > 1 {
        let mut choices = Vec::with_capacity(n);
        let want_lps = request.logprobs.unwrap_or(false);
        let mut total_completion_tokens = 0usize;
        let mut total_reasoning_tokens = 0usize;
        for i in 0..n {
            let mut params = sampling.clone();
            params.seed = Some(sampling.seed.unwrap_or(0).wrapping_add(i as u64));
            let outcome = submit_and_collect(
                &state,
                prompt_tokens.clone(),
                max_tokens,
                params,
                request.response_format.clone(),
            )
            .await?;
            total_completion_tokens += outcome.generated_tokens.len();
            total_reasoning_tokens +=
                count_reasoning_tokens(&outcome.generated_tokens, state.think_token_ids, thinking);
            let decoded = decode(&state, &outcome.generated_tokens)?;
            let decoded = request.stop.as_deref().map_or_else(
                || decoded.clone(),
                |stop| strip_stop_strings(&decoded, stop),
            );
            let (content, tool_calls, split_thinking) =
                finalize_chat_content(decoded, tools_active, thinking);
            let (reasoning_content, content) = split_reasoning(&content, split_thinking);
            let finish_reason = if tool_calls.is_empty() {
                finish_reason(outcome.finish.as_ref()).to_string()
            } else {
                "tool_calls".to_string()
            };
            let logprobs_value = if want_lps {
                Some(chat_logprobs_value(
                    &state,
                    &outcome.generated_tokens,
                    &outcome.gen_logprobs,
                    &outcome.top_logprobs,
                )?)
            } else {
                None
            };
            choices.push(ChatChoice {
                index: i,
                message: AssistantMessage {
                    role: "assistant",
                    content,
                    reasoning_content,
                    tool_calls,
                },
                logprobs: logprobs_value,
                finish_reason,
            });
        }
        let usage = if thinking {
            Usage::with_reasoning(
                prompt_tokens.len(),
                total_completion_tokens,
                total_reasoning_tokens,
            )
        } else {
            Usage::new(prompt_tokens.len(), total_completion_tokens)
        };
        return Ok(Json(ChatCompletionResponse {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4().simple()),
            object: "chat.completion",
            created: unix_time_secs(),
            model: state.model.clone(),
            choices,
            usage,
            system_fingerprint: SYSTEM_FINGERPRINT,
        })
        .into_response());
    }

    let outcome = submit_and_collect(
        &state,
        prompt_tokens,
        max_tokens,
        sampling,
        request.response_format.clone(),
    )
    .await?;
    let decoded = decode(&state, &outcome.generated_tokens)?;
    let decoded = request.stop.as_deref().map_or_else(
        || decoded.clone(),
        |stop| strip_stop_strings(&decoded, stop),
    );
    let (content, tool_calls, split_thinking) =
        finalize_chat_content(decoded, tools_active, thinking);
    let logprobs_value = if request.logprobs.unwrap_or(false) {
        Some(chat_logprobs_value(
            &state,
            &outcome.generated_tokens,
            &outcome.gen_logprobs,
            &outcome.top_logprobs,
        )?)
    } else {
        None
    };
    let reasoning_tokens =
        count_reasoning_tokens(&outcome.generated_tokens, state.think_token_ids, thinking);
    Ok(Json(ChatCompletionResponse::from_parts(
        state.model.clone(),
        content,
        outcome.prompt_tokens,
        outcome.generated_tokens.len(),
        reasoning_tokens,
        outcome.finish.as_ref(),
        split_thinking,
        tool_calls,
        logprobs_value,
    ))
    .into_response())
}

/// Anthropic → internal prompt: map the Messages request onto the shared chat
/// machinery (same template render / thinking / tools gating as
/// [`chat_completions`]) and tokenize. Returns the mapped chat request plus
/// the resolved `(thinking, tools_active)` switches and prompt tokens.
fn anthropic_prompt(
    state: &CoordinatorHandle,
    request: &MessagesRequest,
) -> Result<(ChatCompletionRequest, bool, bool, Vec<u32>), ApiError> {
    let chat_request = request.to_chat_request();
    let thinking = chat_request
        .enable_thinking(state.max_thinking_tokens > 0 || state.template_defaults_thinking);
    let tools_active = chat_request.wants_tools();
    let tools: &[chat::OpenAiToolDefinition] = if tools_active {
        &chat_request.tools
    } else {
        &[]
    };
    let mut anthropic_kwargs = chat_request
        .chat_template_kwargs
        .clone()
        .unwrap_or_default();
    if let Some(tc) = &chat_request.tool_choice {
        anthropic_kwargs.insert("tool_choice".to_string(), tc.clone());
    }
    let anthropic_kwargs = if anthropic_kwargs.is_empty() {
        None
    } else {
        Some(anthropic_kwargs)
    };
    let prompt = {
        let tokenizer = state
            .tokenizer
            .lock()
            .map_err(|_| ApiError::internal("tokenizer lock poisoned"))?;
        tokenizer.render_chat_full(
            &chat_request.messages,
            anthropic_kwargs.as_ref(),
            tools,
            thinking,
            None,
        )?
    };
    let prompt_tokens = encode(state, &prompt)?;
    // /v1/messages carries no chat_template_kwargs, so the rendered-prompt gate
    // is its ONLY route to the reasoning split for template-default thinking.
    let thinking = thinking || prompt_prefills_think(&prompt);
    Ok((chat_request, thinking, tools_active, prompt_tokens))
}

/// Encode routed pipeline output as Anthropic events: content becomes text
/// deltas (reasoning has no Anthropic lane — dropped), completed calls become
/// tool_use blocks. Returns whether any tool call was emitted.
fn push_anthropic_events(
    events: &mut String,
    encoder: &mut StreamEncoder,
    pieces: Vec<ChatDelta>,
    calls: &[chat::ToolCall],
) -> bool {
    for piece in pieces {
        match piece {
            ChatDelta::Reasoning(text) => events.push_str(&encoder.thinking_delta(&text)),
            ChatDelta::Content(text) => events.push_str(&encoder.text_delta(&text)),
        }
    }
    for call in calls {
        events.push_str(&encoder.tool_use(&call.name, &call.arguments));
    }
    !calls.is_empty()
}

/// `POST /v1/messages` — Anthropic Messages API onto the OpenAI chat machinery.
async fn anthropic_messages(
    State(state): State<Arc<DpCoordinator>>,
    Json(body): Json<serde_json::Value>,
) -> Result<Response, MessagesError> {
    let dump_path = dump_messages_body(&body);
    let request: MessagesRequest = serde_json::from_value(body)
        .map_err(|err| MessagesError::invalid_request(err.to_string()))?;
    request.validate()?;
    let (chat_request, thinking, tools_active, prompt_tokens) = anthropic_prompt(&state, &request)?;
    let mut sampling = chat_request.sampling_params();
    // Convert string `stop` sequences to token ids (engine only supports
    // token-id stops). Try as-is and with leading space/newline.
    if let Some(stop_strings) = &chat_request.stop {
        for s in stop_strings {
            if s.is_empty() {
                continue;
            }
            for variant in [s.clone(), format!(" {s}"), format!("\n{s}")] {
                let ids = encode(&state, &variant)?;
                sampling.stop_token_ids.extend(ids);
            }
        }
    }
    let client_budget = request.thinking.as_ref().and_then(|t| t.budget_tokens);
    let think_budget = state.resolve_think_budget(thinking, None, client_budget, &mut sampling);
    let max_tokens = sampling.max_new_tokens.unwrap_or_else(|| {
        think_budget.map_or(THINK_CONTENT_HEADROOM, |budget| {
            budget + THINK_CONTENT_HEADROOM
        })
    });
    // Echo the request's model string back (Anthropic contract).
    let model = request.model.clone().unwrap_or_else(|| state.model.clone());
    let prompt_token_count = prompt_tokens.len();

    if request.stream.unwrap_or(false) {
        // Token sidecar rides the dump: carry the rendered prompt into the task.
        let mut sidecar = dump_path.map(|path| (path, prompt_tokens.clone()));
        let (mut rx, guard) = streaming_submit(&state, prompt_tokens, max_tokens, sampling, None)?;
        let state_clone = Arc::clone(&state);
        // Bounded channel: backpressure keeps the task from racing too far ahead.
        let (chunk_tx, chunk_rx) =
            tokio::sync::mpsc::channel::<Result<Vec<u8>, std::convert::Infallible>>(64);
        tokio::spawn(async move {
            // `guard` dropped when this task exits, decrementing in_flight + unregistering sink.
            let _guard = guard;
            let mut encoder = StreamEncoder::new(format!("msg_{}", Uuid::new_v4().simple()), model);
            if chunk_tx
                .send(Ok(encoder.message_start(prompt_token_count).into_bytes()))
                .await
                .is_err()
            {
                return;
            }
            // Anthropic pipeline: reasoning is emitted as `thinking` blocks even
            // in tools mode (unlike OpenAI which drops it).
            let mut pipeline = StreamPipeline::new(thinking, tools_active);
            let mut saw_tool_use = false;
            let mut output_tokens = 0usize;
            let mut gen_ids: Vec<u32> = Vec::new();
            let mut gen_lps: Vec<f32> = Vec::new();
            let mut detok = IncrementalDetokenizer::default();
            // Keep-alive pings while generation runs (long prefills).
            let mut ping = tokio::time::interval(Duration::from_secs(5));
            ping.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            ping.tick().await; // the first tick fires immediately — consume it
            loop {
                tokio::select! {
                    maybe = rx.recv() => {
                        let Some(delta) = maybe else {
                            let _ = chunk_tx
                                .send(Ok(StreamEncoder::error("engine stream closed unexpectedly")
                                    .into_bytes()))
                                .await;
                            break;
                        };
                        if let Some(err) = delta.error {
                            let _ = chunk_tx.send(Ok(StreamEncoder::error(&err).into_bytes())).await;
                            break;
                        }
                        let mut events = String::new();
                        if !delta.token_ids.is_empty() {
                            output_tokens += delta.token_ids.len();
                            if sidecar.is_some() {
                                gen_ids.extend_from_slice(&delta.token_ids);
                                gen_lps.extend_from_slice(&delta.logprobs);
                            }
                            let text = {
                                let tok = state_clone
                                    .tokenizer
                                    .lock()
                                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                                detok.push(&tok, &delta.token_ids)
                            };
                            let (pieces, calls) = pipeline.push(&text);
                            saw_tool_use |=
                                push_anthropic_events(&mut events, &mut encoder, pieces, &calls);
                        }
                        if delta.finish {
                            // Bytes held for an unfinished codepoint: the stream is
                            // over, so they never complete.
                            let tail = {
                                let tok = state_clone
                                    .tokenizer
                                    .lock()
                                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                                detok.flush(&tok)
                            };
                            if !tail.is_empty() {
                                let (pieces, calls) = pipeline.push(&tail);
                                saw_tool_use |=
                                    push_anthropic_events(&mut events, &mut encoder, pieces, &calls);
                            }
                            if let Some((path, prompt)) = sidecar.take() {
                                write_tokens_sidecar(
                                    &path,
                                    prompt,
                                    std::mem::take(&mut gen_ids),
                                    std::mem::take(&mut gen_lps),
                                );
                            }
                            // Flush the pipeline: truncated thinking + buffered tool tail.
                            let (pieces, calls) = pipeline.finish();
                            saw_tool_use |=
                                push_anthropic_events(&mut events, &mut encoder, pieces, &calls);
                            events.push_str(&encoder.finish(
                                anthropic::stop_reason(delta.finish_reason.as_ref(), saw_tool_use),
                                output_tokens,
                            ));
                            let _ = chunk_tx.send(Ok(events.into_bytes())).await;
                            break;
                        }
                        if !events.is_empty()
                            && chunk_tx.send(Ok(events.into_bytes())).await.is_err()
                        {
                            break; // Client disconnected; guard drops, releasing in_flight.
                        }
                    }
                    _ = ping.tick() => {
                        if chunk_tx.send(Ok(StreamEncoder::ping().into_bytes())).await.is_err() {
                            break;
                        }
                    }
                }
            }
        });
        let body =
            axum::body::Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(chunk_rx));
        return Ok((
            [
                (header::CONTENT_TYPE, "text/event-stream"),
                (header::CACHE_CONTROL, "no-cache"),
                (header::HeaderName::from_static("x-accel-buffering"), "no"),
            ],
            body,
        )
            .into_response());
    }

    let sidecar_prompt = dump_path.as_ref().map(|_| prompt_tokens.clone());
    let outcome = submit_and_collect(&state, prompt_tokens, max_tokens, sampling, None).await?;
    if let (Some(path), Some(prompt)) = (dump_path, sidecar_prompt) {
        write_tokens_sidecar(
            &path,
            prompt,
            outcome.generated_tokens.clone(),
            outcome.gen_logprobs.clone(),
        );
    }
    let decoded = decode(&state, &outcome.generated_tokens)?;
    let decoded = chat_request.stop.as_deref().map_or_else(
        || decoded.clone(),
        |stop| strip_stop_strings(&decoded, stop),
    );
    // Anthropic keeps reasoning even when tools are active (thinking blocks
    // precede text/tool_use blocks). Parse tool calls only when tools are
    // active (the prompt advertised them), then hand the result to
    // `from_parts` so it splits reasoning into `reasoning_content`.
    let (content, calls) = if tools_active {
        chat::openai_parse_tool_calls(&decoded)
    } else {
        (decoded.clone(), Vec::new())
    };
    let tool_calls: Vec<ResponseToolCall> = calls
        .iter()
        .enumerate()
        .map(|(index, call)| ResponseToolCall::from_parsed(call, index))
        .collect();
    let reasoning_tokens =
        count_reasoning_tokens(&outcome.generated_tokens, state.think_token_ids, thinking);
    let chat = ChatCompletionResponse::from_parts(
        model,
        content,
        outcome.prompt_tokens,
        outcome.generated_tokens.len(),
        reasoning_tokens,
        outcome.finish.as_ref(),
        thinking,
        tool_calls,
        None,
    );
    Ok(Json(anthropic::MessagesResponse::from_chat(&chat)).into_response())
}

/// `POST /v1/messages/count_tokens` — render the prompt exactly like
/// [`anthropic_messages`] would and return its token count.
async fn anthropic_count_tokens(
    State(state): State<Arc<DpCoordinator>>,
    Json(body): Json<serde_json::Value>,
) -> Result<Response, MessagesError> {
    let request: MessagesRequest = serde_json::from_value(body)
        .map_err(|err| MessagesError::invalid_request(err.to_string()))?;
    request.validate_for_count()?;
    let (_, _, _, prompt_tokens) = anthropic_prompt(&state, &request)?;
    Ok(Json(serde_json::json!({"input_tokens": prompt_tokens.len()})).into_response())
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "healthy"}))
}

async fn list_models(State(state): State<Arc<DpCoordinator>>) -> Json<ModelsResponse> {
    Json(ModelsResponse::single(state.model.clone()))
}

/// `/v1/embeddings` is not implemented — the runtime has no embedding model.
/// Returns a 501 in the OpenAI error envelope so clients see a structured
/// error instead of a plain 404.
async fn embeddings_not_implemented() -> ApiError {
    ApiError::not_implemented("embeddings are not supported by this model")
}

/// Fallback for unmatched routes: return an OpenAI-shaped 404 (or 405 for a
/// wrong method on a known path) instead of axum's plain-text default.
async fn fallback_404(req: axum::extract::Request) -> (StatusCode, Json<serde_json::Value>) {
    use axum::http::StatusCode;
    let message = format!("No such route: {} {}", req.method(), req.uri().path());
    (
        StatusCode::NOT_FOUND,
        Json(serde_json::json!({
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": null,
                "code": null
            }
        })),
    )
}

async fn metrics(
    State(state): State<Arc<DpCoordinator>>,
) -> ([(header::HeaderName, &'static str); 1], String) {
    let counters = match state.cached_stats() {
        Some(snap) => snap,
        None => state.query_stats_all(Duration::from_secs(5)).await,
    };
    (
        [(
            header::CONTENT_TYPE,
            crate::metrics::PROMETHEUS_CONTENT_TYPE,
        )],
        crate::metrics::render_prometheus(&counters, &state.model),
    )
}

async fn stats(State(state): State<Arc<DpCoordinator>>) -> Result<Json<StatsResponse>, ApiError> {
    let groups = state
        .collect_wire_stats_all(std::time::Duration::from_secs(5))
        .await;
    if groups.is_empty() {
        return Err(ApiError::internal(
            "stats query failed (no TP group responded)",
        ));
    }
    let wire = crate::multiproc_relay::aggregate_wire_stats_dp(groups);
    Ok(Json(StatsResponse::from_wire(wire)))
}

#[derive(serde::Deserialize)]
struct ObserveQueryParams {
    range: Option<String>,
}

fn parse_range_ms(s: &str) -> Option<u64> {
    let s = s.trim();
    let unit = s.chars().last()?;
    let num = &s[..s.len() - unit.len_utf8()];
    let n: u64 = num.parse().ok()?;
    match unit {
        's' => Some(n * 1_000),
        'm' => Some(n * 60_000),
        'h' => Some(n * 3_600_000),
        'd' => Some(n * 86_400_000),
        _ => None,
    }
}

async fn observe_query(Query(params): Query<ObserveQueryParams>) -> Json<serde_json::Value> {
    let range_ms = params
        .range
        .as_deref()
        .and_then(parse_range_ms)
        .unwrap_or(3_600_000);
    let samples = tokio::task::spawn_blocking(move || crate::observe::query(range_ms))
        .await
        .unwrap_or_default();
    Json(serde_json::json!({ "samples": samples }))
}

async fn dashboard_page() -> ([(header::HeaderName, &'static str); 1], &'static str) {
    (
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        include_str!("dashboard.html"),
    )
}
