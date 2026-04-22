use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use log::{error, info, warn};
use tokio::sync::mpsc;

use super::request_state::{
    DflashBatchOutcome, MetalMixedBatchResult, MetalRequestPhase as RuntimePhase,
    MetalRequestState, Qwen35PackedDecodeBatch, Qwen35PrefixSnapshot,
};
use super::scheduler::{
    MetalRequestPriority, MetalRuntimeRequestState, MetalScheduleStep, MetalScheduler,
    MetalSchedulerConfig,
};
use super::weights::MetalWeights;
use super::{MetalBackend, MetalBackendOptions};
use crate::backend::InferenceBackend;
use crate::backend::runtime::StopChunkProcessor;
use crate::metrics::ServerMetrics;
use crate::sampler::SamplingParams;
use crate::scheduler::{IncomingRequest, RequestPriority, SchedulerHandle};
use crate::server_engine::{CompletionStreamDelta, FinishReason, TokenUsage};
use crate::tokenizer::{IncrementalDecoder, Tokenizer};
use crate::types::{InferenceMode, RequestId};

struct PendingMetalRequest {
    delta_tx: mpsc::UnboundedSender<CompletionStreamDelta>,
    prompt_tokens: Vec<u32>,
    max_tokens: usize,
    sampling: SamplingParams,
    stop: Option<Vec<String>>,
    enqueued_at: Instant,
}

impl PendingMetalRequest {
    fn from_incoming(
        tokenizer: &Tokenizer,
        incoming: IncomingRequest,
    ) -> Result<(Self, MetalRequestPriority)> {
        let prompt_tokens = tokenizer.encode(&incoming.prompt)?;
        if prompt_tokens.is_empty() {
            bail!("Metal scheduler request requires at least one prompt token");
        }
        Ok((
            Self {
                delta_tx: incoming.delta_tx,
                prompt_tokens,
                max_tokens: incoming.max_tokens,
                sampling: incoming.sampling,
                stop: incoming.stop,
                enqueued_at: Instant::now(),
            },
            map_request_priority(incoming.priority),
        ))
    }

    fn delta_closed(&self) -> bool {
        self.delta_tx.is_closed()
    }

    fn activate(
        self,
        backend: &'static MetalBackend,
        tokenizer: &'static Tokenizer,
        enable_dflash: bool,
    ) -> Result<ActiveMetalRequest> {
        ActiveMetalRequest::from_pending(backend, tokenizer, self, enable_dflash)
    }
}

struct ActiveMetalRequest {
    delta_tx: mpsc::UnboundedSender<CompletionStreamDelta>,
    request_state: MetalRequestState<'static>,
    decoder: IncrementalDecoder<'static>,
    stop_processor: StopChunkProcessor,
    prompt_tokens: Vec<u32>,
    enqueued_at: Instant,
    admitted_at: Instant,
    first_token_at: Option<Instant>,
}

impl ActiveMetalRequest {
    fn from_pending(
        backend: &'static MetalBackend,
        tokenizer: &'static Tokenizer,
        pending: PendingMetalRequest,
        enable_dflash: bool,
    ) -> Result<Self> {
        let prompt_tokens = pending.prompt_tokens;
        let max_tokens = pending.max_tokens;
        let mut sampling = pending.sampling;
        sampling.max_new_tokens = Some(max_tokens);
        // Thread DFlash runtime into the request state so Qwen3StepDriver
        // can initialize speculative-decode state. Both refs are 'static
        // because the backend is leaked into the scheduler runtime thread.
        // SAFETY: `backend` was leaked to `'static` at runtime.rs:591 before
        // this function is called. The ptr-cast inside is sound.
        //
        // `enable_dflash=false` (caller sees concurrent sessions already
        // queued) skips the DFlash hidden-capture prefill too, saving the
        // full-prompt single-shot prefill cost — the request would have
        // been downgraded at the first decode step anyway.
        let dflash_ref = if enable_dflash {
            unsafe { backend.dflash_runtime_static() }
        } else {
            None
        };
        let request_state =
            backend.create_request_state_with_dflash(&prompt_tokens, &sampling, dflash_ref)?;
        Ok(Self {
            delta_tx: pending.delta_tx,
            request_state,
            decoder: tokenizer.incremental_decoder(),
            stop_processor: StopChunkProcessor::new(pending.stop.unwrap_or_default()),
            prompt_tokens,
            enqueued_at: pending.enqueued_at,
            admitted_at: Instant::now(),
            first_token_at: None,
        })
    }

    fn delta_closed(&self) -> bool {
        self.delta_tx.is_closed()
    }

    fn phase(&self) -> RuntimePhase {
        self.request_state.phase()
    }

    fn stop_hit(&self) -> bool {
        self.stop_processor.hit_stop()
    }

    fn prefill_chunk(&mut self, budget: usize) -> Result<Option<u32>> {
        let result = self.request_state.prefill_chunk(budget)?;
        if let Some(token) = result.emitted_token {
            self.process_token(token)?;
            Ok(Some(token))
        } else {
            Ok(None)
        }
    }

    fn decode_step(&mut self) -> Result<u32> {
        let token = self
            .request_state
            .decode_step()?
            .context("decode_step did not emit a token")?;
        self.process_token(token)?;
        Ok(token)
    }

    fn cancel(&mut self) -> Result<()> {
        self.request_state.cancel()
    }

    fn send_final_delta(&mut self) -> Result<()> {
        if let Some(tail) = self.decoder.finish()? {
            self.push_text_chunk(&tail)?;
        }
        if let Some(final_delta) = self.stop_processor.finish() {
            send_text_delta(&self.delta_tx, final_delta)?;
        }

        let finish_reason = if self.stop_processor.hit_stop() {
            FinishReason::Stop
        } else {
            map_finish_reason(self.request_state.finish_reason())
        };
        let completion_tokens = self.request_state.generated_tokens();
        let prompt_tokens = self.prompt_tokens.len();
        let usage = TokenUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        };

        let _ = self.delta_tx.send(CompletionStreamDelta {
            text_delta: String::new(),
            finish_reason: Some(finish_reason),
            usage: Some(usage),
            logprob: None,
        });
        Ok(())
    }

    fn process_token(&mut self, token_id: u32) -> Result<()> {
        if self.first_token_at.is_none() {
            self.first_token_at = Some(Instant::now());
        }
        if let Some(chunk) = self.decoder.step(token_id)? {
            self.push_text_chunk(&chunk)?;
        }
        Ok(())
    }

    fn push_text_chunk(&mut self, chunk: &str) -> Result<()> {
        if let Some(delta) = self.stop_processor.push_chunk(chunk) {
            send_text_delta(&self.delta_tx, delta)?;
        }
        Ok(())
    }

    fn prompt_len(&self) -> usize {
        self.prompt_tokens.len()
    }
}

const METAL_PREFIX_BLOCK_SIZE: usize = 16;
const METAL_PREFIX_POOL_MULTIPLIER: usize = 4;
const METRICS_REFRESH_INTERVAL: Duration = Duration::from_millis(40);

enum PrefillChunkOutcome {
    Progress {
        emitted_token: Option<u32>,
        runtime_finished: bool,
        stop_hit: bool,
    },
    ClientDropped,
    Failed(anyhow::Error),
}

enum MetalLivePrefixRuntime {
    Qwen35(MetalQwen35PrefixRuntime),
}

struct MetalQwen35CachedPrefix {
    snapshot: Qwen35PrefixSnapshot,
    last_used_tick: u64,
}

struct MetalQwen35PrefixRuntime {
    entries: HashMap<Vec<u32>, MetalQwen35CachedPrefix>,
    max_cached_tokens: usize,
    cached_tokens: usize,
    next_tick: u64,
    block_size: usize,
}

struct CachedQwen35DecodeBatch {
    req_ids: Vec<RequestId>,
    batch: Qwen35PackedDecodeBatch<'static>,
}

impl MetalLivePrefixRuntime {
    fn new(backend: &'static MetalBackend, config: &MetalSchedulerConfig) -> Result<Option<Self>> {
        let weights = backend.weights.as_ref().context("weights not loaded")?;
        let max_total_tokens = (config
            .max_running_requests
            .saturating_mul(config.max_batch_tokens)
            .saturating_mul(METAL_PREFIX_POOL_MULTIPLIER))
        .max(METAL_PREFIX_BLOCK_SIZE * 8);
        match weights {
            MetalWeights::Qwen3(_) => {
                info!(
                    "Metal live prefix cache disabled for Qwen3: long-prompt allocator stability takes priority over prompt-prefix reuse"
                );
                Ok(None)
            }
            MetalWeights::Qwen35(weights) => {
                if weights.cpp_model.is_none() {
                    info!(
                        "Metal live prefix cache disabled for Qwen3.6/Qwen3.5-MoE: snapshot replay requires the compiled Qwen3.5 step path"
                    );
                    return Ok(None);
                }
                info!(
                    "Metal live prefix cache enabled for Qwen3.5 snapshot replay: block_size={}, max_cached_tokens={}",
                    METAL_PREFIX_BLOCK_SIZE, max_total_tokens
                );
                Ok(Some(Self::Qwen35(MetalQwen35PrefixRuntime::new(
                    max_total_tokens,
                    METAL_PREFIX_BLOCK_SIZE,
                ))))
            }
        }
    }

    fn prepare_request(
        &mut self,
        request: &mut ActiveMetalRequest,
        metrics: &ServerMetrics,
    ) -> Result<()> {
        match self {
            MetalLivePrefixRuntime::Qwen35(runtime) => runtime.prepare_request(request, metrics),
        }
    }

    fn publish_prompt_prefix(&mut self, request: &ActiveMetalRequest) -> Result<()> {
        match self {
            MetalLivePrefixRuntime::Qwen35(runtime) => runtime.publish_prompt_prefix(request),
        }
    }
}

impl MetalQwen35PrefixRuntime {
    fn new(max_cached_tokens: usize, block_size: usize) -> Self {
        Self {
            entries: HashMap::new(),
            max_cached_tokens,
            cached_tokens: 0,
            next_tick: 1,
            block_size,
        }
    }

    fn prepare_request(
        &mut self,
        request: &mut ActiveMetalRequest,
        metrics: &ServerMetrics,
    ) -> Result<()> {
        let prompt_tokens = &request.prompt_tokens;
        if prompt_tokens.len() < self.block_size {
            metrics.record_prefix_lookup(false);
            return Ok(());
        }

        let Some(prefix_key) = self.lookup_longest_prefix(prompt_tokens) else {
            metrics.record_prefix_lookup(false);
            return Ok(());
        };

        let imported =
            if let Some(snapshot) = self.entries.get(&prefix_key).map(|entry| &entry.snapshot) {
                request
                    .request_state
                    .import_qwen35_prefix_snapshot(snapshot, prefix_key.len())
                    .context("import matched Qwen3.5 prefix snapshot into request state")?
            } else {
                false
            };

        metrics.record_prefix_lookup(imported);
        if !imported {
            return Ok(());
        }

        self.touch(&prefix_key);
        Ok(())
    }

    fn publish_prompt_prefix(&mut self, request: &ActiveMetalRequest) -> Result<()> {
        let snapshots = request
            .request_state
            .export_qwen35_prompt_prefixes(self.block_size)
            .context("export Qwen3.5 prompt prefix snapshots")?;
        for snapshot in snapshots {
            self.insert_snapshot(snapshot);
        }
        Ok(())
    }

    fn lookup_longest_prefix(&self, prompt_tokens: &[u32]) -> Option<Vec<u32>> {
        self.entries
            .keys()
            .filter(|tokens| {
                let prefix_len = tokens.len();
                prefix_len >= self.block_size
                    && prefix_len < prompt_tokens.len()
                    && prompt_tokens.starts_with(tokens.as_slice())
            })
            .max_by_key(|tokens| tokens.len())
            .cloned()
    }

    fn insert_snapshot(&mut self, snapshot: Qwen35PrefixSnapshot) {
        let token_count = snapshot.token_ids.len();
        if token_count < self.block_size || !token_count.is_multiple_of(self.block_size) {
            return;
        }
        let tick = self.bump_tick();
        if let Some(existing) = self.entries.get_mut(&snapshot.token_ids) {
            existing.last_used_tick = tick;
            return;
        }
        if token_count > self.max_cached_tokens {
            return;
        }

        self.ensure_capacity_for(token_count);
        let key = snapshot.token_ids.clone();
        self.cached_tokens += token_count;
        self.entries.insert(
            key,
            MetalQwen35CachedPrefix {
                snapshot,
                last_used_tick: tick,
            },
        );
    }

    fn ensure_capacity_for(&mut self, needed_tokens: usize) {
        while self.cached_tokens.saturating_add(needed_tokens) > self.max_cached_tokens {
            let Some((lru_key, lru_tokens)) = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used_tick)
                .map(|(tokens, entry)| (tokens.clone(), entry.snapshot.token_ids.len()))
            else {
                break;
            };
            self.entries.remove(&lru_key);
            self.cached_tokens = self.cached_tokens.saturating_sub(lru_tokens);
        }
    }

    fn touch(&mut self, key: &[u32]) {
        let tick = self.bump_tick();
        if let Some(entry) = self.entries.get_mut(key) {
            entry.last_used_tick = tick;
        }
    }

    fn bump_tick(&mut self) -> u64 {
        let tick = self.next_tick;
        self.next_tick = self.next_tick.saturating_add(1);
        tick
    }
}

/// Spawn the first live Metal scheduler runtime.
///
/// This runtime uses the request-state API to interleave chunked prefill and
/// decode scheduling. Qwen3 decode batches are executed as one cross-request
/// GPU graph; unsupported decode batches fall back to request-by-request
/// execution inside the scheduler loop.
pub fn spawn_metal_scheduler_handle_from_path_with_options(
    model_path: &str,
    options: MetalBackendOptions,
    max_waiting: usize,
) -> Result<MetalSchedulerHandle> {
    let model_id = Path::new(model_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(model_path)
        .to_string();
    spawn_metal_scheduler_handle_from_path_with_options_and_metrics(
        model_path,
        options,
        max_waiting,
        ServerMetrics::new(&model_id),
    )
}

/// Wrapper that pairs a `SchedulerHandle` with DFlash init-time metadata for
/// HTTP-layer introspection (`/v1/models`).
///
/// The inner scheduler handle submits work exactly as the raw
/// `SchedulerHandle` does — this struct only adds a read-only side channel
/// for the DFlash draft id and speculative block size, captured at backend
/// load time. Acceptance rate is NOT stored here; it is read from the shared
/// `ServerMetrics` at response time (rolling counter).
#[derive(Clone)]
pub struct MetalSchedulerHandle {
    inner: SchedulerHandle,
    dflash_status: Option<crate::request_handle::DflashStatus>,
}

impl MetalSchedulerHandle {
    /// Borrow the underlying `SchedulerHandle` for callers that still expect
    /// the raw scheduler type (e.g. bench harness token-counter plumbing).
    pub fn inner(&self) -> &SchedulerHandle {
        &self.inner
    }
}

impl crate::request_handle::RequestHandle for MetalSchedulerHandle {
    fn submit(
        &self,
        req: IncomingRequest,
    ) -> std::result::Result<(), crate::request_handle::SubmitError> {
        SchedulerHandle::submit(&self.inner, req).map_err(|_| crate::request_handle::SubmitError)
    }

    fn model_id(&self) -> &str {
        SchedulerHandle::model_id(&self.inner)
    }

    fn dflash_status(&self) -> Option<crate::request_handle::DflashStatus> {
        self.dflash_status.clone()
    }
}

pub fn spawn_metal_scheduler_handle_from_path_with_options_and_metrics(
    model_path: &str,
    options: MetalBackendOptions,
    max_waiting: usize,
    metrics: ServerMetrics,
) -> Result<MetalSchedulerHandle> {
    // DFlash is now supported: Qwen3StepDriver's token-buffer pattern runs
    // speculative blocks inside decode_token, transparent to the scheduler.
    let mut backend = MetalBackend::with_options(options);
    backend.load(Path::new(model_path))?;

    // Snapshot DFlash metadata BEFORE the backend is leaked into the
    // scheduler thread. When DFlash is disabled at load time (either no
    // draft requested, or a compatibility check failed and fell back),
    // this reads `None` and the HTTP layer reports DFlash disabled —
    // matching the actual runtime state.
    let dflash_status =
        backend
            .dflash_runtime_ref()
            .map(|rt| crate::request_handle::DflashStatus {
                draft_model: rt.draft_model_id().to_string(),
                speculative_tokens: rt.block_size(),
            });

    let model_id = Path::new(model_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(model_path)
        .to_string();

    let (tx, rx) = mpsc::unbounded_channel();
    let waiting_count = Arc::new(AtomicUsize::new(0));
    let handle =
        SchedulerHandle::with_shared_waiting_count(tx, &model_id, max_waiting, waiting_count);

    let runtime_handle = handle.clone();
    std::thread::spawn(move || {
        // The runtime owns one backend instance for the process lifetime. The
        // request-state API currently borrows backend internals, so keep the
        // loaded backend stable inside the worker thread until the server exits.
        let backend: &'static MetalBackend = Box::leak(Box::new(backend));
        let Some(tokenizer) = backend.tokenizer.as_ref() else {
            error!("Metal scheduler runtime failed: model tokenizer not loaded");
            return;
        };
        let tokenizer: &'static Tokenizer = tokenizer;

        let result = catch_unwind(AssertUnwindSafe(|| {
            run_metal_scheduler_runtime(
                backend,
                tokenizer,
                rx,
                &runtime_handle,
                &metrics,
                MetalSchedulerConfig::default(),
            )
        }));

        match result {
            Ok(Ok(())) => {}
            Ok(Err(err)) => error!("Metal scheduler runtime failed: {err:#}"),
            Err(panic) => error!(
                "Metal scheduler runtime panicked: {}",
                super::panic_message(panic)
            ),
        }
    });

    Ok(MetalSchedulerHandle {
        inner: handle,
        dflash_status,
    })
}

pub fn spawn_metal_scheduler_handle_from_path(
    model_path: &str,
    max_waiting: usize,
) -> Result<MetalSchedulerHandle> {
    spawn_metal_scheduler_handle_from_path_with_options(
        model_path,
        MetalBackendOptions::default(),
        max_waiting,
    )
}

fn run_metal_scheduler_runtime(
    backend: &'static MetalBackend,
    tokenizer: &'static Tokenizer,
    mut request_rx: mpsc::UnboundedReceiver<IncomingRequest>,
    handle: &SchedulerHandle,
    metrics: &ServerMetrics,
    config: MetalSchedulerConfig,
) -> Result<()> {
    let mut prefix_runtime = MetalLivePrefixRuntime::new(backend, &config)?;
    let mut scheduler = MetalScheduler::new(config)?;
    let mut pending = HashMap::<RequestId, PendingMetalRequest>::new();
    let mut active = HashMap::<RequestId, ActiveMetalRequest>::new();
    let mut qwen35_decode_batch_cache: Option<CachedQwen35DecodeBatch> = None;
    let mut request_rx_closed = false;
    let mut last_metrics_refresh: Option<Instant> = None;

    info!("Metal scheduler runtime started");

    loop {
        drain_incoming_requests(
            tokenizer,
            handle,
            metrics,
            &mut request_rx,
            &mut request_rx_closed,
            &mut scheduler,
            &mut pending,
        );
        reap_closed_clients(handle, &mut scheduler, &mut pending, &mut active);
        maybe_refresh_runtime_metrics(
            metrics,
            handle,
            &scheduler,
            &pending,
            &active,
            &mut last_metrics_refresh,
            METRICS_REFRESH_INTERVAL,
        );

        if request_rx_closed && active.is_empty() && scheduler.waiting_len() == 0 {
            info!("Metal scheduler runtime shutting down: all handles dropped");
            break;
        }

        if active.is_empty() && scheduler.waiting_len() == 0 {
            if let Some(incoming) = request_rx.blocking_recv() {
                enqueue_request(
                    metrics,
                    tokenizer,
                    incoming,
                    handle,
                    &mut scheduler,
                    &mut pending,
                );
                // Admission is rare enough that an unconditional refresh
                // is fine — helps the first metrics scrape after idle.
                refresh_runtime_metrics(metrics, handle, &scheduler, &pending, &active);
                last_metrics_refresh = Some(Instant::now());
            } else {
                request_rx_closed = true;
                continue;
            }
        }

        let runtime_states = scheduler_runtime_states(&active);
        let step = scheduler.step(&runtime_states);
        if step.is_idle() {
            metrics.set_scheduler_step(0, 0, 0, 0, 0, 0);
            continue;
        }

        let scheduled_decode_rows =
            step.decode.as_ref().map_or(0, |batch| batch.req_ids.len()) as u64;
        let scheduled_prefill_rows = u64::from(step.prefill.is_some());
        let scheduled_prefill_tokens = step
            .prefill
            .as_ref()
            .map_or(0, |prefill| prefill.input_tokens.len() as u64);
        let scheduled_rows = scheduled_decode_rows + scheduled_prefill_rows;
        metrics.set_scheduler_step(
            scheduled_rows,
            scheduled_decode_rows,
            scheduled_prefill_rows,
            scheduled_decode_rows,
            scheduled_prefill_tokens,
            scheduled_rows,
        );
        let step_started_at = Instant::now();

        guard_schedule_step(
            step,
            backend,
            tokenizer,
            handle,
            metrics,
            &mut prefix_runtime,
            &mut scheduler,
            &mut pending,
            &mut active,
            &mut qwen35_decode_batch_cache,
        );
        metrics.observe_scheduler_step(step_started_at.elapsed().as_secs_f64());

        maybe_refresh_runtime_metrics(
            metrics,
            handle,
            &scheduler,
            &pending,
            &active,
            &mut last_metrics_refresh,
            METRICS_REFRESH_INTERVAL,
        );
    }

    Ok(())
}

fn guard_prefill_chunk(
    req_id: RequestId,
    budget: usize,
    backend: &'static MetalBackend,
    tokenizer: &'static Tokenizer,
    handle: &SchedulerHandle,
    metrics: &ServerMetrics,
    prefix_runtime: &mut Option<MetalLivePrefixRuntime>,
    scheduler: &mut MetalScheduler,
    pending: &mut HashMap<RequestId, PendingMetalRequest>,
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
) {
    let result = catch_unwind(AssertUnwindSafe(|| {
        execute_prefill_chunk(
            req_id,
            budget,
            backend,
            tokenizer,
            handle,
            metrics,
            prefix_runtime,
            scheduler,
            pending,
            active,
        );
    }));

    if let Err(panic) = result {
        error!(
            "Metal prefill chunk panicked for {:?}: {}",
            req_id,
            super::panic_message(panic)
        );
        metrics.record_request_failed();
        *prefix_runtime = None;
        abort_runtime_requests(&[req_id], scheduler, active);
    }
}

#[allow(clippy::too_many_arguments)]
fn guard_schedule_step(
    step: MetalScheduleStep,
    backend: &'static MetalBackend,
    tokenizer: &'static Tokenizer,
    handle: &SchedulerHandle,
    metrics: &ServerMetrics,
    prefix_runtime: &mut Option<MetalLivePrefixRuntime>,
    scheduler: &mut MetalScheduler,
    pending: &mut HashMap<RequestId, PendingMetalRequest>,
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
    qwen35_decode_batch_cache: &mut Option<CachedQwen35DecodeBatch>,
) {
    match (step.decode, step.prefill) {
        (Some(batch), Some(prefill)) => {
            if !guard_mixed_batch(
                batch.req_ids.clone(),
                prefill.req_id,
                prefill.input_tokens.len(),
                backend,
                tokenizer,
                handle,
                metrics,
                prefix_runtime,
                scheduler,
                pending,
                active,
            ) {
                guard_decode_batch(
                    batch.req_ids,
                    metrics,
                    scheduler,
                    active,
                    qwen35_decode_batch_cache,
                );
                guard_prefill_chunk(
                    prefill.req_id,
                    prefill.input_tokens.len(),
                    backend,
                    tokenizer,
                    handle,
                    metrics,
                    prefix_runtime,
                    scheduler,
                    pending,
                    active,
                );
            }
        }
        (Some(batch), None) => {
            guard_decode_batch(
                batch.req_ids,
                metrics,
                scheduler,
                active,
                qwen35_decode_batch_cache,
            );
        }
        (None, Some(prefill)) => {
            guard_prefill_chunk(
                prefill.req_id,
                prefill.input_tokens.len(),
                backend,
                tokenizer,
                handle,
                metrics,
                prefix_runtime,
                scheduler,
                pending,
                active,
            );
        }
        (None, None) => {}
    }
}

fn guard_mixed_batch(
    decode_req_ids: Vec<RequestId>,
    prefill_req_id: RequestId,
    prefill_budget: usize,
    backend: &'static MetalBackend,
    tokenizer: &'static Tokenizer,
    handle: &SchedulerHandle,
    metrics: &ServerMetrics,
    prefix_runtime: &mut Option<MetalLivePrefixRuntime>,
    scheduler: &mut MetalScheduler,
    pending: &mut HashMap<RequestId, PendingMetalRequest>,
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
) -> bool {
    let mut panic_req_ids = decode_req_ids.clone();
    panic_req_ids.push(prefill_req_id);
    let result = catch_unwind(AssertUnwindSafe(|| {
        execute_mixed_batch(
            decode_req_ids,
            prefill_req_id,
            prefill_budget,
            backend,
            tokenizer,
            handle,
            metrics,
            prefix_runtime,
            scheduler,
            pending,
            active,
        )
    }));

    match result {
        Ok(handled) => handled,
        Err(panic) => {
            error!(
                "Metal mixed batch panicked for {:?}: {}",
                panic_req_ids,
                super::panic_message(panic)
            );
            metrics.record_request_failed();
            *prefix_runtime = None;
            abort_runtime_requests(&panic_req_ids, scheduler, active);
            true
        }
    }
}

fn guard_decode_batch(
    req_ids: Vec<RequestId>,
    metrics: &ServerMetrics,
    scheduler: &mut MetalScheduler,
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
    qwen35_decode_batch_cache: &mut Option<CachedQwen35DecodeBatch>,
) {
    let panic_req_ids = req_ids.clone();
    let result = catch_unwind(AssertUnwindSafe(|| {
        execute_decode_batch(
            req_ids,
            metrics,
            scheduler,
            active,
            qwen35_decode_batch_cache,
        );
    }));

    if let Err(panic) = result {
        error!(
            "Metal decode batch panicked for {:?}: {}",
            panic_req_ids,
            super::panic_message(panic)
        );
        metrics.record_request_failed();
        *qwen35_decode_batch_cache = None;
        abort_runtime_requests(&panic_req_ids, scheduler, active);
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_mixed_batch(
    decode_req_ids: Vec<RequestId>,
    prefill_req_id: RequestId,
    prefill_budget: usize,
    backend: &'static MetalBackend,
    tokenizer: &'static Tokenizer,
    handle: &SchedulerHandle,
    metrics: &ServerMetrics,
    prefix_runtime: &mut Option<MetalLivePrefixRuntime>,
    scheduler: &mut MetalScheduler,
    pending: &mut HashMap<RequestId, PendingMetalRequest>,
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
) -> bool {
    if !active.contains_key(&prefill_req_id) {
        activate_pending_request(
            prefill_req_id,
            backend,
            tokenizer,
            handle,
            metrics,
            prefix_runtime,
            scheduler,
            pending,
            active,
        );
    }
    let Some(prefill_snapshot) = active.get(&prefill_req_id) else {
        return false;
    };
    if prefill_snapshot.delta_closed()
        || !prefill_snapshot.request_state.is_qwen3()
        || prefill_snapshot.request_state.is_dflash_enabled()
    {
        return false;
    }
    if !decode_req_ids.iter().all(|req_id| {
        active.get(req_id).is_some_and(|request| {
            !request.delta_closed()
                && request.request_state.is_qwen3()
                && !request.request_state.is_dflash_enabled()
        })
    }) {
        return false;
    }

    let mut decode_rows = Vec::with_capacity(decode_req_ids.len());
    for req_id in decode_req_ids {
        let Some(request) = active.remove(&req_id) else {
            warn!(
                "Metal mixed batch referenced missing decode request {:?}",
                req_id
            );
            scheduler.finish_request(req_id, None);
            continue;
        };
        if request.delta_closed() {
            scheduler.finish_request(req_id, request_mode(&request));
            continue;
        }
        decode_rows.push((req_id, request));
    }

    let Some(mut prefill_request) = active.remove(&prefill_req_id) else {
        for (req_id, request) in decode_rows {
            active.insert(req_id, request);
        }
        return false;
    };
    if prefill_request.delta_closed() {
        scheduler.finish_request(prefill_req_id, request_mode(&prefill_request));
        if let Err(err) = prefill_request.cancel() {
            warn!(
                "Metal request cancel failed for {:?}: {err:#}",
                prefill_req_id
            );
        }
        for (req_id, request) in decode_rows {
            active.insert(req_id, request);
        }
        return true;
    }

    let outcome = {
        let mut decode_refs: Vec<&mut MetalRequestState<'static>> = decode_rows
            .iter_mut()
            .map(|(_, request)| &mut request.request_state)
            .collect();
        MetalRequestState::try_mixed_batch(
            &mut decode_refs,
            &mut prefill_request.request_state,
            prefill_budget,
        )
    };

    let Some(MetalMixedBatchResult {
        decode_tokens,
        prefill,
    }) = (match outcome {
        Ok(result) => result,
        Err(err) => {
            error!("Metal mixed batch failed: {err:#}");
            metrics.record_request_failed();
            cancel_detached_request(prefill_req_id, prefill_request, scheduler);
            for (req_id, request) in decode_rows {
                cancel_detached_request(req_id, request, scheduler);
            }
            return true;
        }
    })
    else {
        active.insert(prefill_req_id, prefill_request);
        for (req_id, request) in decode_rows {
            active.insert(req_id, request);
        }
        return false;
    };

    for ((req_id, mut request), sampled_token) in decode_rows.into_iter().zip(decode_tokens) {
        if let Err(err) = request.process_token(sampled_token) {
            error!(
                "Metal mixed decode post-process failed for {:?}: {err:#}",
                req_id
            );
            metrics.record_request_failed();
            cancel_detached_request(req_id, request, scheduler);
            continue;
        }
        finish_or_requeue_decoded_request(req_id, request, metrics, scheduler, active);
    }

    if let Some(sampled_token) = prefill.emitted_token {
        if let Err(err) = prefill_request.process_token(sampled_token) {
            error!(
                "Metal mixed prefill post-process failed for {:?}: {err:#}",
                prefill_req_id
            );
            metrics.record_request_failed();
            cancel_detached_request(prefill_req_id, prefill_request, scheduler);
            return true;
        }
        if let Some(prefix_runtime) = prefix_runtime.as_mut()
            && let Err(err) = prefix_runtime.publish_prompt_prefix(&prefill_request)
        {
            warn!(
                "Metal live prefix publish failed for {:?}: {err:#}",
                prefill_req_id
            );
        }
    }

    if prefill_request.phase() == RuntimePhase::Finished || prefill_request.stop_hit() {
        finalize_detached_request(prefill_req_id, prefill_request, metrics, scheduler);
    } else {
        active.insert(prefill_req_id, prefill_request);
    }

    true
}

fn abort_runtime_requests(
    req_ids: &[RequestId],
    scheduler: &mut MetalScheduler,
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
) {
    for &req_id in req_ids {
        let mode = active.get(&req_id).and_then(request_mode);
        let _ = scheduler.finish_request(req_id, mode);
        if let Some(mut request) = active.remove(&req_id) {
            if let Err(err) = request.cancel() {
                warn!("Metal panic cleanup failed for {:?}: {err:#}", req_id);
            }
            drop(request);
        }
    }
}

fn maybe_refresh_runtime_metrics(
    metrics: &ServerMetrics,
    handle: &SchedulerHandle,
    scheduler: &MetalScheduler,
    pending: &HashMap<RequestId, PendingMetalRequest>,
    active: &HashMap<RequestId, ActiveMetalRequest>,
    last: &mut Option<Instant>,
    interval: Duration,
) {
    let now = Instant::now();
    if let Some(prev) = *last {
        if now.duration_since(prev) < interval {
            return;
        }
    }
    refresh_runtime_metrics(metrics, handle, scheduler, pending, active);
    *last = Some(now);
}

fn drain_incoming_requests(
    tokenizer: &'static Tokenizer,
    handle: &SchedulerHandle,
    metrics: &ServerMetrics,
    request_rx: &mut mpsc::UnboundedReceiver<IncomingRequest>,
    request_rx_closed: &mut bool,
    scheduler: &mut MetalScheduler,
    pending: &mut HashMap<RequestId, PendingMetalRequest>,
) {
    loop {
        match request_rx.try_recv() {
            Ok(incoming) => {
                enqueue_request(metrics, tokenizer, incoming, handle, scheduler, pending);
            }
            Err(mpsc::error::TryRecvError::Empty) => break,
            Err(mpsc::error::TryRecvError::Disconnected) => {
                *request_rx_closed = true;
                break;
            }
        }
    }
}

fn enqueue_request(
    metrics: &ServerMetrics,
    tokenizer: &'static Tokenizer,
    incoming: IncomingRequest,
    handle: &SchedulerHandle,
    scheduler: &mut MetalScheduler,
    pending: &mut HashMap<RequestId, PendingMetalRequest>,
) {
    if incoming.delta_tx.is_closed() {
        handle.consume_one();
        return;
    }

    let (pending_request, priority) = match PendingMetalRequest::from_incoming(tokenizer, incoming)
    {
        Ok(request) => request,
        Err(err) => {
            error!("Metal scheduler request init failed: {err:#}");
            metrics.record_request_failed();
            handle.consume_one();
            return;
        }
    };

    let req_id = match scheduler.submit(
        pending_request.prompt_tokens.clone(),
        pending_request.max_tokens,
        priority,
    ) {
        Ok(req_id) => req_id,
        Err(err) => {
            error!("Metal scheduler submit failed: {err}");
            metrics.record_request_failed();
            handle.consume_one();
            return;
        }
    };

    if pending.insert(req_id, pending_request).is_some() {
        warn!("Metal scheduler request id collision for {:?}", req_id);
    }
}

fn activate_pending_request(
    req_id: RequestId,
    backend: &'static MetalBackend,
    tokenizer: &'static Tokenizer,
    handle: &SchedulerHandle,
    metrics: &ServerMetrics,
    prefix_runtime: &mut Option<MetalLivePrefixRuntime>,
    scheduler: &mut MetalScheduler,
    pending: &mut HashMap<RequestId, PendingMetalRequest>,
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
) {
    let Some(pending_request) = pending.remove(&req_id) else {
        warn!(
            "Metal prefill chunk referenced missing pending request {:?}",
            req_id
        );
        scheduler.finish_request(req_id, None);
        return;
    };

    if pending_request.delta_closed() {
        handle.consume_one();
        scheduler.finish_request(req_id, None);
        return;
    }

    // Always initialize DFlash when the backend has a draft model loaded;
    // concurrent DFlash rows are handled later in decode batching.
    let enable_dflash = true;
    let mut request = match pending_request.activate(backend, tokenizer, enable_dflash) {
        Ok(request) => request,
        Err(err) => {
            error!(
                "Metal scheduler activation failed for {:?}: {err:#}",
                req_id
            );
            metrics.record_request_failed();
            handle.consume_one();
            scheduler.finish_request(req_id, None);
            return;
        }
    };

    if let Some(prefix_runtime) = prefix_runtime.as_mut() {
        if let Err(err) = prefix_runtime.prepare_request(&mut request, metrics) {
            error!(
                "Metal prefix-cache activation failed for {:?}: {err:#}",
                req_id
            );
            metrics.record_request_failed();
            handle.consume_one();
            scheduler.finish_request(req_id, None);
            return;
        }
    }

    handle.consume_one();
    if active.insert(req_id, request).is_some() {
        warn!(
            "Metal scheduler activation overwrote an existing active request {:?}",
            req_id
        );
    }
}

fn execute_prefill_chunk(
    req_id: RequestId,
    mut budget: usize,
    backend: &'static MetalBackend,
    tokenizer: &'static Tokenizer,
    handle: &SchedulerHandle,
    metrics: &ServerMetrics,
    prefix_runtime: &mut Option<MetalLivePrefixRuntime>,
    scheduler: &mut MetalScheduler,
    pending: &mut HashMap<RequestId, PendingMetalRequest>,
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
) {
    if !active.contains_key(&req_id) {
        activate_pending_request(
            req_id,
            backend,
            tokenizer,
            handle,
            metrics,
            prefix_runtime,
            scheduler,
            pending,
            active,
        );
    }
    if !active.contains_key(&req_id) {
        return;
    }

    // DFlash requires full-prompt prefill in one shot because
    // `qwen3_forward_with_hidden_states` captures hidden states for all
    // positions — chunked KV-only prefill can't produce them. Override the
    // scheduler's chunk budget to process the entire remaining prompt.
    if let Some(request) = active.get(&req_id) {
        if request.request_state.is_dflash_enabled() {
            let remaining = request
                .request_state
                .prompt_len()
                .saturating_sub(request.request_state.prompt_progress());
            budget = budget.max(remaining);
        }
    }

    let outcome = {
        let Some(request) = active.get_mut(&req_id) else {
            warn!(
                "Metal prefill chunk referenced missing request {:?}",
                req_id
            );
            scheduler.finish_request(req_id, None);
            return;
        };

        if request.delta_closed() {
            PrefillChunkOutcome::ClientDropped
        } else {
            match request.prefill_chunk(budget) {
                Ok(emitted_token) => PrefillChunkOutcome::Progress {
                    emitted_token,
                    runtime_finished: request.phase() == RuntimePhase::Finished,
                    stop_hit: request.stop_hit(),
                },
                Err(err) => {
                    if request.delta_closed() {
                        PrefillChunkOutcome::ClientDropped
                    } else {
                        PrefillChunkOutcome::Failed(err)
                    }
                }
            }
        }
    };

    match outcome {
        PrefillChunkOutcome::Progress {
            emitted_token,
            runtime_finished,
            stop_hit,
        } => {
            if let Some(_token) = emitted_token {
                if let Some(prefix_runtime) = prefix_runtime.as_mut()
                    && let Some(request) = active.get(&req_id)
                    && let Err(err) = prefix_runtime.publish_prompt_prefix(request)
                {
                    warn!("Metal live prefix publish failed for {:?}: {err:#}", req_id);
                }
            }

            if runtime_finished || stop_hit {
                finalize_request(req_id, metrics, scheduler, active);
            }
        }
        PrefillChunkOutcome::ClientDropped => cancel_request(req_id, scheduler, active),
        PrefillChunkOutcome::Failed(err) => {
            error!("Metal prefill chunk failed for {:?}: {err:#}", req_id);
            metrics.record_request_failed();
            cancel_request(req_id, scheduler, active);
        }
    }
}

fn execute_decode_batch(
    req_ids: Vec<RequestId>,
    metrics: &ServerMetrics,
    scheduler: &mut MetalScheduler,
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
    qwen35_decode_batch_cache: &mut Option<CachedQwen35DecodeBatch>,
) {
    if req_ids.is_empty() {
        return;
    }

    let mut staged = Vec::with_capacity(req_ids.len());
    for req_id in req_ids {
        let Some(request) = active.remove(&req_id) else {
            warn!("Metal decode batch referenced missing request {:?}", req_id);
            scheduler.finish_request(req_id, None);
            continue;
        };
        staged.push((req_id, request));
    }

    let mut open = Vec::with_capacity(staged.len());
    for (req_id, request) in staged {
        if request.delta_closed() {
            scheduler.finish_request(req_id, request_mode(&request));
            continue;
        }
        open.push((req_id, request));
    }

    // Round-3 codex findings on the partitioner are both closed:
    //   - [P2] "all-or-nothing DFlash demotion on buffered-speculative
    //     rows" — fixed at
    //     `request_state.rs::try_decode_qwen35_dflash_speculative_batch`
    //     (majority-equivalence-class per-row partition).
    //   - [P1] "plain-decode cache rollback on singleton fallback" —
    //     retracted; the `invalidate_*` sync on the `Ok(None)` arm is the
    //     only path that propagates `packed_kv_flat`/`packed_gdr_flat`
    //     updates into per-request state.
    // Partition into dflash_rows and plain_rows. Dispatch:
    //   - plain_rows (≥1): existing `execute_qwen35_packed_decode_batch`.
    //   - dflash_rows (≥2): new `execute_qwen35_dflash_packed_batch`.
    //   - dflash_rows (==1): fall through to the existing per-row
    //     `execute_decode_single` path (batched-stack overhead not worth it).
    let (dflash_requests, non_dflash): (Vec<_>, Vec<_>) = open
        .into_iter()
        .partition(|(_, request)| request.request_state.is_dflash_enabled());

    if dflash_requests.len() >= 2 {
        execute_qwen35_dflash_packed_batch(dflash_requests, metrics, scheduler, active);
    } else {
        for (req_id, request) in dflash_requests {
            execute_decode_single(req_id, request, metrics, scheduler, active);
        }
    }
    let mut open = non_dflash;

    let batch_result =
        match execute_qwen35_packed_decode_batch(&mut open, active, qwen35_decode_batch_cache) {
            Ok(Some(result)) => Some(result),
            Ok(None) => {
                invalidate_qwen35_decode_batch_cache(qwen35_decode_batch_cache, active, &mut open);
                if open.len() >= 2 {
                    let mut request_refs: Vec<&mut MetalRequestState<'static>> = open
                        .iter_mut()
                        .map(|(_, request)| &mut request.request_state)
                        .collect();
                    match MetalRequestState::decode_batch(&mut request_refs) {
                        Ok(result) => result,
                        Err(err) => {
                            error!("Metal batched decode failed: {err:#}");
                            metrics.record_request_failed();
                            for (req_id, request) in open {
                                cancel_detached_request(req_id, request, scheduler);
                            }
                            return;
                        }
                    }
                } else {
                    None
                }
            }
            Err(err) => {
                error!("Metal packed Qwen3.5 decode failed: {err:#}");
                metrics.record_request_failed();
                invalidate_qwen35_decode_batch_cache(qwen35_decode_batch_cache, active, &mut open);
                for (req_id, request) in open {
                    cancel_detached_request(req_id, request, scheduler);
                }
                return;
            }
        };

    if let Some(sampled_tokens) = batch_result {
        for ((req_id, mut request), sampled_token) in open.into_iter().zip(sampled_tokens) {
            if let Err(err) = request.process_token(sampled_token) {
                error!(
                    "Metal batched decode post-process failed for {:?}: {err:#}",
                    req_id
                );
                metrics.record_request_failed();
                cancel_detached_request(req_id, request, scheduler);
                continue;
            }
            finish_or_requeue_decoded_request(req_id, request, metrics, scheduler, active);
        }
        return;
    }

    for (req_id, request) in open {
        execute_decode_single(req_id, request, metrics, scheduler, active);
    }
}

/// Dispatch ≥2 DFlash-enabled Qwen3.5 rows through the batched speculative
/// block kernel. Mirrors `execute_qwen35_packed_decode_batch` in how sampled
/// tokens get fanned back into the scheduler via `process_token` +
/// `finish_or_requeue_decoded_request`.
///
/// No persistent cache struct (unlike the plain-decode path): the DFlash
/// verify batch re-stacks per-row target KV / GDR every tick, and the
/// scalar draft state already lives inside each `MetalRequestState`.
fn execute_qwen35_dflash_packed_batch(
    mut rows: Vec<(RequestId, ActiveMetalRequest)>,
    metrics: &ServerMetrics,
    scheduler: &mut MetalScheduler,
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
) {
    if rows.len() < 2 {
        // Partition guard already filters on ≥2; defensive fallthrough only.
        for (req_id, request) in rows {
            execute_decode_single(req_id, request, metrics, scheduler, active);
        }
        return;
    }

    let outcome = {
        let mut request_refs: Vec<&mut MetalRequestState<'static>> = rows
            .iter_mut()
            .map(|(_, request)| &mut request.request_state)
            .collect();
        match MetalRequestState::try_decode_qwen35_dflash_speculative_batch(&mut request_refs) {
            Ok(Some(outcome)) => outcome,
            Ok(None) => {
                // <2 rows ready (wrong mode / phase / target_hidden not
                // captured / non-empty token_buffer / cross-row disagreement):
                // every row falls back to per-row single-path decode. Scalar
                // `decode_token` handles the stale-target_hidden, Rust-mode,
                // and buffered-drain cases cleanly.
                for (req_id, request) in rows {
                    execute_decode_single(req_id, request, metrics, scheduler, active);
                }
                return;
            }
            Err(err) => {
                error!("Metal Qwen3.5 DFlash batched decode failed: {err:#}");
                metrics.record_request_failed();
                for (req_id, request) in rows {
                    cancel_detached_request(req_id, request, scheduler);
                }
                return;
            }
        }
    };

    let DflashBatchOutcome {
        ready_indices,
        tokens: sampled,
    } = outcome;

    if sampled.len() != ready_indices.len() {
        error!(
            "Metal Qwen3.5 DFlash batched decode: expected {} sampled tokens, got {}",
            ready_indices.len(),
            sampled.len()
        );
        metrics.record_request_failed();
        for (req_id, request) in rows {
            cancel_detached_request(req_id, request, scheduler);
        }
        return;
    }

    // Commit ready-row tokens and dispatch stale rows in the original
    // scheduler order (priority/arrival established by `build_decode_batch`).
    // `ready_indices` is sorted ascending, so one linear pass suffices.
    let mut sampled_iter = sampled.into_iter();
    let mut ready_cursor = 0usize;
    for (idx, (req_id, mut request)) in rows.into_iter().enumerate() {
        let is_ready = ready_cursor < ready_indices.len() && idx == ready_indices[ready_cursor];
        if is_ready {
            ready_cursor += 1;
            let sampled_token = sampled_iter
                .next()
                .expect("sampled.len() == ready_indices.len() validated above");
            if let Err(err) = request.process_token(sampled_token) {
                error!(
                    "Metal DFlash batched decode post-process failed for {:?}: {err:#}",
                    req_id
                );
                metrics.record_request_failed();
                cancel_detached_request(req_id, request, scheduler);
                continue;
            }
            finish_or_requeue_decoded_request(req_id, request, metrics, scheduler, active);
        } else {
            execute_decode_single(req_id, request, metrics, scheduler, active);
        }
    }
}

fn execute_qwen35_packed_decode_batch(
    open: &mut [(RequestId, ActiveMetalRequest)],
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
    cache: &mut Option<CachedQwen35DecodeBatch>,
) -> Result<Option<Vec<u32>>> {
    if open.len() < 2 {
        return Ok(None);
    }

    let current_req_ids: Vec<RequestId> = open.iter().map(|(req_id, _)| *req_id).collect();

    if let Some(cached) = cache.as_mut() {
        if cached.req_ids != current_req_ids {
            if let Some(retained_rows) = retained_row_indices(&cached.req_ids, &current_req_ids) {
                cached.batch.retain_rows(&retained_rows)?;
                cached.req_ids.clone_from(&current_req_ids);
            } else if let Some(new_indices) = admit_row_indices(&cached.req_ids, &current_req_ids) {
                // Prefix-preserving grow: existing rows still first (in
                // order), new rows appended at the end. Admit when every new
                // row's own `cache_len` is `<= batch_cursor`. A row with
                // `cache_len < batch_cursor` gets left-padded up to the
                // cursor and receives its per-row RoPE offset via the
                // `rope_offsets` array passed through the bridge — so both
                // the attention mask and positional encoding stay correct.
                // A row with `cache_len > batch_cursor` would force the
                // cursor to bump and re-pad every existing row, which costs
                // more than a full rebuild, so we fall through to invalidate
                // in that case.
                let batch_cursor = cached.batch.batch_cache_len();
                let admittable = new_indices.iter().all(|&idx| {
                    open.get(idx)
                        .and_then(|(_, request)| request.request_state.qwen35_decode_cursor())
                        .is_some_and(|cache_len| cache_len <= batch_cursor)
                });
                if admittable {
                    let mut request_refs: Vec<&mut MetalRequestState<'static>> = open
                        .iter_mut()
                        .map(|(_, request)| &mut request.request_state)
                        .collect();
                    cached.batch.admit_rows(&mut request_refs, &new_indices)?;
                    cached.req_ids.clone_from(&current_req_ids);
                } else {
                    invalidate_qwen35_decode_batch_cache(cache, active, open);
                }
            } else {
                invalidate_qwen35_decode_batch_cache(cache, active, open);
            }
        }
    }

    if cache.is_none() {
        let mut request_refs: Vec<&mut MetalRequestState<'static>> = open
            .iter_mut()
            .map(|(_, request)| &mut request.request_state)
            .collect();
        let Some(batch) =
            MetalRequestState::try_build_qwen35_packed_decode_batch(&mut request_refs)?
        else {
            return Ok(None);
        };
        *cache = Some(CachedQwen35DecodeBatch {
            req_ids: current_req_ids.clone(),
            batch,
        });
    }

    let cached = cache
        .as_mut()
        .context("Qwen3.5 packed decode cache missing after build")?;
    if cached.req_ids != current_req_ids {
        invalidate_qwen35_decode_batch_cache(cache, active, open);
        return Ok(None);
    }

    let mut request_refs: Vec<&mut MetalRequestState<'static>> = open
        .iter_mut()
        .map(|(_, request)| &mut request.request_state)
        .collect();
    MetalRequestState::try_decode_qwen35_packed_batch(&mut request_refs, &mut cached.batch)
}

fn invalidate_qwen35_decode_batch_cache(
    cache: &mut Option<CachedQwen35DecodeBatch>,
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
    open: &mut [(RequestId, ActiveMetalRequest)],
) {
    let Some(mut cached) = cache.take() else {
        return;
    };

    let mut row_indices = Vec::new();
    let mut state_ptrs = Vec::new();
    for (row_idx, req_id) in cached.req_ids.iter().enumerate() {
        if let Some((_, request)) = open.iter_mut().find(|(candidate, _)| candidate == req_id) {
            row_indices.push(row_idx);
            state_ptrs.push(&raw mut request.request_state);
            continue;
        }
        if let Some(request) = active.get_mut(req_id) {
            row_indices.push(row_idx);
            state_ptrs.push(&raw mut request.request_state);
        }
    }

    if row_indices.is_empty() {
        return;
    }

    if row_indices.len() != cached.req_ids.len() {
        if let Err(err) = cached.batch.retain_rows(&row_indices) {
            error!("Metal packed Qwen3.5 cache retain_rows failed during invalidate: {err:#}");
            return;
        }
    }

    let mut request_refs: Vec<&mut MetalRequestState<'static>> = state_ptrs
        .into_iter()
        .map(|ptr| unsafe { &mut *ptr })
        .collect();
    if let Err(err) =
        MetalRequestState::sync_qwen35_packed_decode_batch(&mut request_refs, &cached.batch)
    {
        error!("Metal packed Qwen3.5 cache sync failed during invalidate: {err:#}");
    }
}

fn retained_row_indices(
    previous_req_ids: &[RequestId],
    current_req_ids: &[RequestId],
) -> Option<Vec<usize>> {
    let mut indices = Vec::with_capacity(current_req_ids.len());
    let mut cursor = 0usize;
    for req_id in current_req_ids {
        let relative = previous_req_ids[cursor..]
            .iter()
            .position(|candidate| candidate == req_id)?;
        let absolute = cursor + relative;
        indices.push(absolute);
        cursor = absolute + 1;
    }
    Some(indices)
}

/// Prefix-preserving grow detector: if `current_req_ids` starts with
/// `previous_req_ids` in the exact same order, return the indices of the
/// new rows (the tail of `current_req_ids`). Otherwise return `None` and
/// the caller falls back to full invalidate.
///
/// We deliberately restrict to the prefix case rather than any supersequence
/// because `Qwen35PackedDecodeBatch::admit_rows` appends the new rows at the
/// end of the packed KV tensors — arbitrary splicing would require extra
/// `take_axis` reorders.
fn admit_row_indices(
    previous_req_ids: &[RequestId],
    current_req_ids: &[RequestId],
) -> Option<Vec<usize>> {
    if current_req_ids.len() <= previous_req_ids.len() {
        return None;
    }
    if &current_req_ids[..previous_req_ids.len()] != previous_req_ids {
        return None;
    }
    Some((previous_req_ids.len()..current_req_ids.len()).collect())
}

fn execute_decode_single(
    req_id: RequestId,
    mut request: ActiveMetalRequest,
    metrics: &ServerMetrics,
    scheduler: &mut MetalScheduler,
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
) {
    enum Outcome {
        Progress {
            runtime_finished: bool,
            stop_hit: bool,
        },
        ClientDropped,
        Failed(anyhow::Error),
    }

    let outcome = if request.delta_closed() {
        Outcome::ClientDropped
    } else {
        match request.decode_step() {
            Ok(_sampled_token) => Outcome::Progress {
                runtime_finished: request.phase() == RuntimePhase::Finished,
                stop_hit: request.stop_hit(),
            },
            Err(err) => {
                if request.delta_closed() {
                    Outcome::ClientDropped
                } else {
                    Outcome::Failed(err)
                }
            }
        }
    };

    match outcome {
        Outcome::Progress {
            runtime_finished,
            stop_hit,
        } => {
            if runtime_finished || stop_hit {
                finalize_detached_request(req_id, request, metrics, scheduler);
            } else {
                active.insert(req_id, request);
            }
        }
        Outcome::ClientDropped => {
            scheduler.finish_request(req_id, request_mode(&request));
            if let Err(err) = request.cancel() {
                warn!("Metal request cancel failed for {:?}: {err:#}", req_id);
            }
            drop(request);
        }
        Outcome::Failed(err) => {
            error!("Metal decode step failed for {:?}: {err:#}", req_id);
            metrics.record_request_failed();
            cancel_detached_request(req_id, request, scheduler);
        }
    }
}

fn finish_or_requeue_decoded_request(
    req_id: RequestId,
    request: ActiveMetalRequest,
    metrics: &ServerMetrics,
    scheduler: &mut MetalScheduler,
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
) {
    let runtime_finished = request.phase() == RuntimePhase::Finished;
    let stop_hit = request.stop_hit();
    if runtime_finished || stop_hit {
        finalize_detached_request(req_id, request, metrics, scheduler);
    } else {
        active.insert(req_id, request);
    }
}

fn cancel_detached_request(
    req_id: RequestId,
    mut request: ActiveMetalRequest,
    scheduler: &mut MetalScheduler,
) {
    scheduler.finish_request(req_id, request_mode(&request));
    if let Err(err) = request.cancel() {
        warn!("Metal request cancel failed for {:?}: {err:#}", req_id);
    }
    drop(request);
}

fn finalize_detached_request(
    req_id: RequestId,
    mut request: ActiveMetalRequest,
    metrics: &ServerMetrics,
    scheduler: &mut MetalScheduler,
) {
    scheduler.finish_request(req_id, Some(InferenceMode::Decode));
    record_request_completed(metrics, &request);
    if let Err(err) = request.send_final_delta() {
        warn!("Metal request final delta failed for {:?}: {err:#}", req_id);
    }
    drop(request);
}

fn reap_closed_clients(
    handle: &SchedulerHandle,
    scheduler: &mut MetalScheduler,
    pending: &mut HashMap<RequestId, PendingMetalRequest>,
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
) {
    let pending_closed: Vec<_> = pending
        .iter()
        .filter_map(|(req_id, request)| request.delta_closed().then_some(*req_id))
        .collect();
    for req_id in pending_closed {
        handle.consume_one();
        scheduler.finish_request(req_id, None);
        pending.remove(&req_id);
    }

    let closed: Vec<_> = active
        .iter()
        .filter_map(|(req_id, request)| request.delta_closed().then_some(*req_id))
        .collect();

    for req_id in closed {
        cancel_request(req_id, scheduler, active);
    }
}

fn cancel_request(
    req_id: RequestId,
    scheduler: &mut MetalScheduler,
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
) {
    let mode = active.get(&req_id).map(request_mode);
    scheduler.finish_request(req_id, mode.flatten());
    if let Some(mut request) = active.remove(&req_id) {
        if let Err(err) = request.cancel() {
            warn!("Metal request cancel failed for {:?}: {err:#}", req_id);
        }
        drop(request);
    }
}

fn finalize_request(
    req_id: RequestId,
    metrics: &ServerMetrics,
    scheduler: &mut MetalScheduler,
    active: &mut HashMap<RequestId, ActiveMetalRequest>,
) {
    scheduler.finish_request(req_id, Some(InferenceMode::Decode));
    let Some(mut request) = active.remove(&req_id) else {
        return;
    };
    record_request_completed(metrics, &request);
    if let Err(err) = request.cancel() {
        warn!("Metal request cleanup failed for {:?}: {err:#}", req_id);
    }
    if let Err(err) = request.send_final_delta()
        && !request.delta_closed()
    {
        warn!("Metal request final delta failed for {:?}: {err:#}", req_id);
    }
    drop(request);
}

fn record_request_completed(metrics: &ServerMetrics, request: &ActiveMetalRequest) {
    let completion_tokens = request.request_state.generated_tokens() as u64;
    let completed_at = Instant::now();
    let queue_wait_s = request
        .admitted_at
        .duration_since(request.enqueued_at)
        .as_secs_f64();
    let e2e_s = completed_at
        .duration_since(request.enqueued_at)
        .as_secs_f64();
    let active_ttft_s = request.first_token_at.map_or(0.0, |first| {
        first.duration_since(request.admitted_at).as_secs_f64()
    });
    let ttft_s = request.first_token_at.map_or(e2e_s, |first| {
        first.duration_since(request.enqueued_at).as_secs_f64()
    });
    let tpot_s = if completion_tokens > 1 {
        (e2e_s - ttft_s).max(0.0) / (completion_tokens - 1) as f64
    } else {
        0.0
    };
    metrics.record_request_completed_detailed(
        request.prompt_len() as u64,
        completion_tokens,
        queue_wait_s,
        active_ttft_s,
        ttft_s,
        tpot_s,
        e2e_s,
    );

    // Flush DFlash speculative decode metrics if this was a DFlash request.
    if let Some((blocks, accepted, drafted)) = request.request_state.dflash_block_stats() {
        for i in 0..blocks {
            metrics.record_dflash_block(accepted.get(i).copied().unwrap_or(0), drafted);
        }
    }
}

fn request_mode(request: &ActiveMetalRequest) -> Option<InferenceMode> {
    match request.phase() {
        RuntimePhase::Prefill => Some(InferenceMode::Prefill),
        RuntimePhase::Decode => Some(InferenceMode::Decode),
        RuntimePhase::Finished => None,
    }
}

fn scheduler_runtime_states(
    active: &HashMap<RequestId, ActiveMetalRequest>,
) -> Vec<MetalRuntimeRequestState> {
    active
        .iter()
        .filter(|(_, request)| request.phase() != RuntimePhase::Finished)
        .map(|(req_id, request)| MetalRuntimeRequestState {
            req_id: *req_id,
            phase: match request.phase() {
                RuntimePhase::Prefill => super::scheduler::MetalRequestPhase::Prefilling,
                RuntimePhase::Decode | RuntimePhase::Finished => {
                    super::scheduler::MetalRequestPhase::Decoding
                }
            },
            prompt_progress: request.request_state.prompt_progress(),
            last_token: request.request_state.last_token(),
        })
        .collect()
}

fn refresh_runtime_metrics(
    metrics: &ServerMetrics,
    handle: &SchedulerHandle,
    _scheduler: &MetalScheduler,
    _pending: &HashMap<RequestId, PendingMetalRequest>,
    active: &HashMap<RequestId, ActiveMetalRequest>,
) {
    metrics.set_active(active.len() as u64);
    metrics.set_waiting(handle.waiting_count() as u64);
    let running_batch = active
        .values()
        .filter(|request| request.phase() == RuntimePhase::Decode)
        .count() as u64;
    let prefill_queue = active
        .values()
        .filter(|request| request.phase() == RuntimePhase::Prefill)
        .count() as u64;
    metrics.set_scheduler_occupancy(running_batch, prefill_queue);
    metrics.set_kv_coordinator(0, 0, 0, 0, false, false);
    metrics.set_tier_wait_seconds(0.0, 0.0);

    let (kv_used, kv_total) = active.values().fold((0u64, 0u64), |acc, request| {
        if let Some((used, total)) = request.request_state.kv_pool_usage() {
            (acc.0 + used as u64, acc.1 + total as u64)
        } else {
            acc
        }
    });
    metrics.set_kv_gpu_blocks(kv_total.saturating_sub(kv_used), kv_total);
    metrics.set_memory_bytes(
        super::mlx::active_memory_bytes(),
        super::mlx::peak_memory_bytes(),
        super::mlx::cache_memory_bytes(),
    );
}

fn map_request_priority(priority: RequestPriority) -> MetalRequestPriority {
    match priority {
        RequestPriority::Low => MetalRequestPriority::Low,
        RequestPriority::Normal => MetalRequestPriority::Normal,
        RequestPriority::High => MetalRequestPriority::High,
    }
}

fn map_finish_reason(reason: Option<&str>) -> FinishReason {
    match reason {
        Some("length") => FinishReason::Length,
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
        })
        .map_err(|_| anyhow!("stream consumer dropped"))
}
