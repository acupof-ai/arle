//! Backend-agnostic inference engine loop.
//!
//! This crate depends only on the plan and seam contracts. Runtime-specific
//! buffers and model implementation details stay below `infer-seam`.

use std::collections::{BTreeMap, VecDeque};

mod planner;
mod prefix;
mod radix;
mod recall;
mod writethrough;

pub use radix::{BlockId, PrefixMatch, RadixCache};
pub use recall::{RecallConfig, RecallPlan, plan_recall};
pub use writethrough::{
    cap_rep_pool, evict_drop_pages, plan_working_set, prefetch_blocks, prefetch_query,
};

use anyhow::{Result, bail};
use infer_plan::{FinishReason, ForwardPlan, SamplingParams, SlotToken, StepOutput};
use infer_seam::{
    AdmissionVerdict, BackendExecutor, KvPool, PermissiveGovernor, PollResult, ResourceGovernor,
    StepBudget,
};

/// Stable handle returned to callers when a request is submitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RequestHandle(u64);

impl RequestHandle {
    /// Return the numeric id backing this handle.
    #[must_use]
    pub fn id(self) -> u64 {
        self.0
    }
}

/// Request priority level. Higher-priority requests are admitted first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum RequestPriority {
    /// Background or batch work.
    Low = 0,
    /// Normal API work.
    #[default]
    Normal = 1,
    /// Interactive or latency-sensitive work.
    High = 2,
}

/// Scheduler admission knobs used by `infer-core`.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Maximum number of concurrently active request slots.
    pub num_slots: usize,
    /// Maximum total tokens assigned to one scheduler tick.
    pub max_num_batched_tokens: usize,
    /// Maximum prompt tokens admitted for prefill in one scheduler tick.
    pub max_prefill_tokens: usize,
    /// Maximum number of requests allowed to be prefilling at once.
    pub prefill_max_requests: Option<usize>,
    /// Maximum prompt length accepted at ingress.
    pub max_prompt_tokens: usize,
    /// Maximum prompt plus generated length for one request.
    pub max_total_tokens: usize,
    /// Target minimum free pages after prefix-cache reclamation.
    pub prefix_cache_low_water_pages: usize,
    /// Per-request prefill chunk size. A prompt longer than this is prefilled
    /// across multiple ticks so decode rows can interleave (chunked prefill).
    pub chunked_prefill_size: usize,
    /// Whether cross-request prompt-prefix reuse via the host radix cache is
    /// enabled. Backends should only enable this when every attached prefix page
    /// is a complete restore boundary, or when they can clamp matches to such
    /// boundaries via `BackendExecutor::reusable_prefix_blocks`.
    pub enable_prefix_cache: bool,
    /// Maximum total plan tokens (decode rows + prefill chunk tokens) per
    /// forward, from `BackendExecutor::max_tokens_per_step`. `usize::MAX` (the
    /// default) means unbounded; the deepep_ll DSv4 arm reports its LL dispatch
    /// cap so the planner never builds a forward the executor would reject.
    pub max_tokens_per_step: usize,
    /// Opt-in slot oversubscription: admit more waiters than `num_slots` by
    /// demoting (parking) the longest-running decode's whole-slot image to the
    /// DRAM tier to free a slot, round-robin. Requires a whole-slot tier
    /// executor (`kv_slot_tier_enabled`). Default false → byte-identical to a
    /// build without this trigger; only changes WHEN the proven demote fires.
    pub slot_oversubscription: bool,
}

impl SchedulerConfig {
    /// Build runtime defaults for a fixed logical slot count.
    #[must_use]
    pub fn for_slots(num_slots: usize) -> Self {
        Self {
            num_slots,
            ..Self::default()
        }
    }

    fn max_concurrent_prefill(&self) -> usize {
        self.prefill_max_requests.unwrap_or(self.num_slots).max(1)
    }

    fn prefill_step_budget(&self) -> usize {
        self.max_num_batched_tokens.min(self.max_prefill_tokens)
    }

    fn prefill_chunk_size(&self) -> usize {
        self.chunked_prefill_size.max(1)
    }
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            num_slots: 4,
            max_num_batched_tokens: 16_384,
            max_prefill_tokens: 16_384,
            prefill_max_requests: None,
            max_prompt_tokens: 16_384,
            max_total_tokens: 32_768,
            prefix_cache_low_water_pages: 0,
            chunked_prefill_size: 2_048,
            enable_prefix_cache: true,
            max_tokens_per_step: usize::MAX,
            slot_oversubscription: false,
        }
    }
}

/// Options accepted at request ingress.
#[derive(Debug, Clone, Default)]
pub struct RequestOptions {
    /// Admission priority.
    pub priority: RequestPriority,
    /// Cooperative cancellation observed before queue insertion.
    pub cancelled: bool,
    /// Per-request sampling parameters (default: greedy / argmax).
    pub sampling: SamplingParams,
}

/// Completed request state retained after its slot has been freed.
#[derive(Debug, Clone)]
pub struct CompletedRequest {
    /// Request handle.
    pub handle: RequestHandle,
    /// Prompt tokens submitted by the caller.
    pub prompt_tokens: Vec<u32>,
    /// Tokens generated by executor steps.
    pub generated_tokens: Vec<u32>,
    /// Final finish reason.
    pub finish: Option<FinishReason>,
}

/// Engine throughput counters, monotonic since engine start.
///
/// Maintained on the apply path of the backend-neutral engine so every
/// backend reports the same semantics; surfaced through `/v1/stats` and
/// `/metrics` for QPS/TPS computation by scrape tooling.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ThroughputStats {
    /// Executor steps whose output was applied.
    pub steps: u64,
    /// Prompt tokens advanced through (chunked) prefill.
    pub prefill_tokens: u64,
    /// Tokens committed to requests (final prefill chunk + decode).
    pub generated_tokens: u64,
    /// Requests finished after holding a slot, any finish reason (immediate
    /// ingress rejections are not counted — they did no engine work).
    pub requests_completed: u64,
}

/// KV host-demoted counters maintained by the backend-neutral
/// scheduler. All zero unless the executor reports a nonzero
/// [`infer_seam::BackendExecutor::kv_tier_capacity_pages`] (page route) or
/// [`infer_seam::BackendExecutor::kv_slot_tier_enabled`] (whole-slot route).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KvTierStats {
    /// Prefix pages demoted to the backend host tier instead of dropped.
    pub demoted_pages: u64,
    /// Demoted pages promoted back into device pages on a prefix hit.
    pub promoted_pages: u64,
    /// Promotions that failed (entry severed, tail re-prefilled instead).
    pub promote_failures: u64,
    /// Blocks currently resident in the host tier.
    pub resident_blocks: usize,
    /// Whole-slot restore images demoted on preemption.
    pub demoted_slots: u64,
    /// Whole-slot restore images promoted back on re-admission.
    pub promoted_slots: u64,
    /// Whole-slot restore failures that fell back to recompute.
    pub slot_promote_failures: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KvSystemMetrics {
    pub resident_pages: usize,
    pub resident_evictable_pages: usize,
    pub host_demoted_pages: usize,
    pub host_demoted_pending_inflight: usize,
    pub disk_pages: usize,
    pub reuse_hit_resident: u64,
    pub reuse_hit_host_demoted: u64,
    pub reuse_hit_disk: u64,
    pub reuse_miss: u64,
    pub demote_mset_count: u64,
    pub demote_mset_copy_bytes: u64,
    pub demote_mset_copy_ms: u64,
    pub promote_mget_count: u64,
    pub promote_mget_copy_bytes: u64,
    pub promote_mget_copy_ms: u64,
    pub fetch_wait_ms: u64,
    pub fallback_recompute: u64,
    pub prefix_match_full_blocks: u64,
    pub prefix_match_clamped_blocks: u64,
}

/// Prefix-cache counters maintained by the backend-neutral scheduler.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrefixCacheStats {
    /// Requests admitted while prefix caching was enabled.
    pub lookups: u64,
    /// Lookups that attached at least one backend-reusable prefix page.
    pub hits: u64,
    /// Total prompt tokens skipped by attached prefix pages.
    pub hit_tokens: u64,
    /// Total prefix pages attached to admitted requests.
    pub hit_pages: u64,
    /// Prompt pages newly retained in the radix cache.
    pub published_pages: u64,
    /// Pages currently retained by the radix cache.
    pub cached_pages: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct WaitingRequestHint {
    session_affinity_tokens: usize,
    immediate_reuse_tokens: usize,
    total_reuse_tokens: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WaitingInsertBias {
    BeforeEqual,
    AfterEqual,
}

/// Minimum tokens an oversubscription-admitted request must decode before it
/// is eligible to be parked again. A just-resumed request runs at least this
/// many steps before yielding, bounding park/resume ping-pong (most binding at
/// num_slots=1, where two requests would otherwise swap every step).
const OVERSUBSCRIPTION_MIN_SLICE: usize = 8;

/// Result of attempting to admit the front waiter onto one slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AdmitOutcome {
    /// Admitted onto the slot; the slot is consumed.
    Admitted,
    /// No waiting request to admit.
    NoWaiter,
    /// The front waiter does not fit the current per-tick budget.
    Throttled,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum RequestPhase {
    Prefilling { progress: usize },
    Decoding,
    Finished,
}

#[derive(Debug, Clone)]
struct RequestState {
    handle: RequestHandle,
    prompt_tokens: Vec<u32>,
    generated_tokens: Vec<u32>,
    priority: RequestPriority,
    max_tokens: usize,
    sampling: SamplingParams,
    phase: RequestPhase,
    prefill_start_pos: usize,
    reused_prefix_pages: Vec<BlockId>,
    /// Whole-slot tier key while this request's complete restore image is
    /// swapped out. Re-admission promotes and resumes decode; generated tokens
    /// are kept.
    swap_key: Option<u64>,
    /// Materialized sequence length captured at demote time.
    /// Captured (not derived from `prompt + generated`) because the last
    /// committed token's KV is not materialized yet — it is the next decode
    /// step's input — so the materialized length is one short of the token
    /// count and deriving it would desync host accounting from the restored
    /// device state.
    swap_seq_len: usize,
    finish: Option<FinishReason>,
    waiting_hint: WaitingRequestHint,
    /// Monotonic stamp set each time this request is admitted/resumed into a
    /// slot. The oversubscription victim selector picks the smallest stamp
    /// (longest continuous run since its last admit); a just-resumed request
    /// gets the largest stamp so it is never the immediate victim (no thrash).
    admit_seq: u64,
    /// `generated_tokens.len()` captured at the last admit. The oversubscription
    /// victim must have decoded at least `OVERSUBSCRIPTION_MIN_SLICE` tokens
    /// since then, so a just-resumed request runs a bit before it can be parked
    /// again — bounding ping-pong churn at num_slots=1.
    admit_gen_mark: usize,
}

impl RequestState {
    fn new(
        handle: RequestHandle,
        prompt_tokens: Vec<u32>,
        priority: RequestPriority,
        max_tokens: usize,
        sampling: SamplingParams,
    ) -> Self {
        Self {
            handle,
            prompt_tokens,
            generated_tokens: Vec::new(),
            priority,
            max_tokens,
            sampling,
            phase: RequestPhase::Prefilling { progress: 0 },
            prefill_start_pos: 0,
            reused_prefix_pages: Vec::new(),
            swap_key: None,
            swap_seq_len: 0,
            finish: None,
            waiting_hint: WaitingRequestHint::default(),
            admit_seq: 0,
            admit_gen_mark: 0,
        }
    }

    fn complete_immediately(mut self, finish: FinishReason) -> Self {
        self.phase = RequestPhase::Finished;
        self.finish = Some(finish);
        self
    }

    fn reset_for_recompute(mut self) -> Self {
        self.generated_tokens.clear();
        self.phase = RequestPhase::Prefilling { progress: 0 };
        self.prefill_start_pos = 0;
        self.reused_prefix_pages.clear();
        // Store-entry lifetime is the caller's job (drop before reset); the
        // key is cleared so a recomputed request can never promote stale state.
        self.swap_key = None;
        self.swap_seq_len = 0;
        self.finish = None;
        self.waiting_hint = WaitingRequestHint::default();
        self
    }

    fn prompt_len(&self) -> usize {
        self.prompt_tokens.len()
    }
}

impl From<RequestState> for CompletedRequest {
    fn from(request: RequestState) -> Self {
        Self {
            handle: request.handle,
            prompt_tokens: request.prompt_tokens,
            generated_tokens: request.generated_tokens,
            finish: request.finish,
        }
    }
}

/// Backend-agnostic engine loop.
///
/// Prefix reuse is host-indexed: the engine carries token blocks and page ids,
/// while executor/model-specific storage stays below the seam.
pub struct Engine<E: BackendExecutor, K: KvPool> {
    executor: E,
    kv: K,
    config: SchedulerConfig,
    governor: Box<dyn ResourceGovernor>,
    radix: RadixCache,
    next_request_id: u64,
    active: BTreeMap<usize, RequestState>,
    waiting: VecDeque<RequestState>,
    completed: BTreeMap<RequestHandle, CompletedRequest>,
    inflight: Option<E::Inflight>,
    /// Plan submitted with `inflight`, kept so `apply_output` can advance chunked
    /// prefill progress and resolve which rows produced tokens.
    pending_plan: Option<ForwardPlan>,
    /// Set once [`Engine::warmup`] has run the backend warmup. The first
    /// [`Engine::step`] triggers it lazily so callers need not warm up by hand.
    warmed_up: bool,
    /// Model-default stop token ids (EOS + configured stops) read once from the
    /// executor. Used as the fallback stop set for requests that supply none.
    model_stop_token_ids: Vec<u32>,
    /// Prefix-cache accounting exposed to serving telemetry.
    prefix_cache_stats: PrefixCacheStats,
    throughput_stats: ThroughputStats,
    kv_tier_stats: KvTierStats,
    kv_system_metrics: KvSystemMetrics,
    /// Next backend tier-store key; monotonically burned (failures included).
    next_tier_key: u64,
    /// Monotonic admit stamp source for oversubscription victim selection;
    /// each admit/resume into a slot burns one (see `RequestState::admit_seq`).
    next_admit_seq: u64,
    /// Optional per-token observer invoked as each token is committed to a
    /// request, so a serving layer can stream tokens live. `None` by default —
    /// when unset, token commit behavior is byte-identical to before.
    on_token: Option<TokenObserver>,
}

/// Per-token observer: invoked with `(handle, &token)` as each token is committed
/// to its request. The seam a serving layer installs to stream tokens live.
pub type TokenObserver = Box<dyn FnMut(RequestHandle, &SlotToken)>;

impl<E: BackendExecutor, K: KvPool> Engine<E, K> {
    /// Create an engine with permissive resource governance.
    #[must_use]
    pub fn new(executor: E, kv: K, max_slots: usize) -> Self {
        Self::with_config(executor, kv, SchedulerConfig::for_slots(max_slots))
    }

    /// Create an engine with explicit scheduler config.
    #[must_use]
    pub fn with_config(executor: E, kv: K, config: SchedulerConfig) -> Self {
        Self::with_config_and_governor(executor, kv, config, Box::new(PermissiveGovernor))
    }

    /// Create an engine with explicit scheduler config and resource governor.
    #[must_use]
    pub fn with_config_and_governor(
        executor: E,
        kv: K,
        mut config: SchedulerConfig,
        governor: Box<dyn ResourceGovernor>,
    ) -> Self {
        let max_rows = executor.max_rows_per_step().max(1);
        if config.num_slots > max_rows {
            log::warn!(
                "executor caps rows per step at {max_rows}; scheduler slots {} -> {max_rows}",
                config.num_slots
            );
            config.num_slots = max_rows;
        }
        // Per-forward token cap (deepep_ll LL dispatch buffer): clamp num_slots so
        // a pure-decode forward (one token per slot) never exceeds it.
        config.max_tokens_per_step = executor.max_tokens_per_step().max(1);
        config.num_slots = config.num_slots.min(config.max_tokens_per_step);
        config.num_slots = config.num_slots.max(1);
        let radix = RadixCache::new(kv.page_size().max(1));
        let model_stop_token_ids = executor.model_stop_token_ids();
        Self {
            executor,
            kv,
            config,
            governor,
            radix,
            next_request_id: 0,
            active: BTreeMap::new(),
            waiting: VecDeque::new(),
            completed: BTreeMap::new(),
            inflight: None,
            pending_plan: None,
            warmed_up: false,
            model_stop_token_ids,
            prefix_cache_stats: PrefixCacheStats::default(),
            throughput_stats: ThroughputStats::default(),
            kv_tier_stats: KvTierStats::default(),
            kv_system_metrics: KvSystemMetrics::default(),
            next_tier_key: 0,
            next_admit_seq: 0,
            on_token: None,
        }
    }

    /// Install a per-token observer invoked with `(handle, &token)` as each token
    /// is committed to its request — the seam a serving layer uses to stream
    /// tokens live. Replaces any previously installed observer; the default
    /// (no observer) leaves token-commit behavior byte-identical.
    pub fn set_token_observer(&mut self, observer: TokenObserver) {
        self.on_token = Some(observer);
    }

    /// Mutable access to the backend executor.
    ///
    /// The serving layer runs out-of-band control closures (OPD raw-logits
    /// forward, weight offload/reload, LoRA re-merge) against the executor
    /// between scheduler steps. Not for the request hot path — the scheduler
    /// drives the executor through [`Engine::step`].
    pub fn executor_mut(&mut self) -> &mut E {
        &mut self.executor
    }

    /// Frontend live-request capacity requested by the backend executor.
    #[must_use]
    pub fn max_live_requests(&self) -> usize {
        self.executor.max_live_requests().max(1)
    }

    /// Run backend warmup exactly once.
    ///
    /// Delegates to [`BackendExecutor::warmup`] (default no-op for Metal/mock).
    /// Idempotent: subsequent calls are a no-op so the first [`Engine::step`]
    /// can call it lazily without re-warming.
    ///
    /// # Errors
    /// Propagates any error returned by the backend executor's warmup.
    pub fn warmup(&mut self) -> Result<()> {
        if self.warmed_up {
            return Ok(());
        }
        self.executor.warmup()?;
        self.warmed_up = true;
        Ok(())
    }

    /// Move the backend's device weights to host RAM (OPD teacher time-share),
    /// returning the device bytes freed.
    ///
    /// Delegates to [`BackendExecutor::offload_weights`] (default no-op for
    /// backends without weight offload). The engine must be idle — no in-flight
    /// step — when this is called; the serving loop drains its work before
    /// dispatching the control request.
    ///
    /// # Errors
    /// Propagates any error returned by the backend executor's offload.
    pub fn offload_engine_weights(&mut self) -> Result<usize> {
        self.executor.offload_weights()
    }

    /// Restore the backend's device weights from the host snapshot (OPD teacher
    /// time-share). Delegates to [`BackendExecutor::reload_weights`].
    ///
    /// # Errors
    /// Propagates any error returned by the backend executor's reload.
    pub fn reload_engine_weights(&mut self) -> Result<()> {
        self.executor.reload_weights()
    }

    /// Release the backend's inference forward scratch (workspace / batched-decode /
    /// captured graphs) WITHOUT offloading weights or evicting KV, so a co-resident
    /// OPD writeback reuses the VRAM. Delegates to
    /// [`BackendExecutor::release_inference_scratch`] (default no-op). The engine
    /// must be idle (no in-flight step); the rollout has synced before this is called.
    ///
    /// # Errors
    /// Propagates any error returned by the backend executor's scratch release.
    pub fn release_inference_scratch(&mut self) -> Result<()> {
        self.executor.release_inference_scratch()
    }

    /// Drop the backend's KV pool WITHOUT offloading weights (OPD writeback
    /// headroom: the writeback's fresh autograd forward never reads this engine's
    /// KV). Delegates to [`BackendExecutor::release_kv_pool`] (default no-op). The
    /// engine must be idle (all rollouts synced before this is called).
    ///
    /// # Errors
    /// Propagates any error returned by the backend executor's pool release.
    pub fn release_kv_pool(&mut self) -> Result<()> {
        self.executor.release_kv_pool()
    }

    /// Re-acquire the KV pool dropped by [`Self::release_kv_pool`] before the next
    /// rollout. Delegates to [`BackendExecutor::ensure_kv_pool`] (default no-op).
    ///
    /// # Errors
    /// Propagates any error returned by the backend executor's pool re-acquire.
    pub fn ensure_kv_pool(&mut self) -> Result<()> {
        self.executor.ensure_kv_pool()
    }

    /// Submit a normal-priority request into the waiting queue.
    pub fn submit_request(&mut self, prompt_tokens: Vec<u32>, max_tokens: usize) -> RequestHandle {
        self.submit_request_with_options(prompt_tokens, max_tokens, RequestOptions::default())
    }

    /// Submit a request with a specific priority.
    pub fn submit_request_with_priority(
        &mut self,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        priority: RequestPriority,
    ) -> RequestHandle {
        self.submit_request_with_options(
            prompt_tokens,
            max_tokens,
            RequestOptions {
                priority,
                ..RequestOptions::default()
            },
        )
    }

    /// Submit a request with full ingress options.
    pub fn submit_request_with_options(
        &mut self,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        options: RequestOptions,
    ) -> RequestHandle {
        let handle = self.next_handle();
        let request = self.normalize_request(handle, prompt_tokens, max_tokens, options);
        match request {
            NormalizedRequest::Waiting(request) => {
                self.enqueue_waiting_request(request, WaitingInsertBias::AfterEqual);
            }
            NormalizedRequest::Completed(request) => {
                self.completed.insert(handle, request.into());
            }
            NormalizedRequest::Skipped => {}
        }
        handle
    }

    /// Run one scheduler tick.
    ///
    /// Overlap ordering: the previous executor step is polled first; if it is
    /// still running, this tick exits and keeps the in-flight handle. Once the
    /// previous step completes, the CPU side admits requests and builds plan
    /// N+1 while the executor had already run plan N behind the seam.
    pub fn step(&mut self) -> Result<()> {
        // Lazily warm the backend before the first step does any real work.
        self.warmup()?;

        let diag = std::env::var_os("ARLE_STEP_DIAG").is_some();
        if let Some(inflight) = self.inflight.take() {
            match self.executor.poll(inflight)? {
                PollResult::Ready(output) => {
                    let plan = self.pending_plan.take().unwrap_or_else(ForwardPlan::idle);
                    self.apply_output(&plan, output)?;
                }
                PollResult::NotReady(inflight) => {
                    self.inflight = Some(inflight);
                    if diag {
                        eprintln!("[STEP-DIAG] inflight NotReady (forward submitted, not done)");
                    }
                    return Ok(());
                }
            }
        }

        let budget = self.governor.step_budget();
        if self.governor.should_yield() || budget.max_tokens == 0 || budget.max_micros == 0 {
            if diag {
                eprintln!(
                    "[STEP-DIAG] gate: should_yield={} max_tokens={} max_micros={}",
                    self.governor.should_yield(),
                    budget.max_tokens,
                    budget.max_micros
                );
            }
            std::thread::yield_now();
            return Ok(());
        }

        self.admit_waiting()?;
        if diag {
            eprintln!(
                "[STEP-DIAG] post-admit: active={} waiting={}",
                self.active.len(),
                self.waiting.len()
            );
        }
        let mut plan = self.build_forward_plan();
        self.apply_step_budget(&mut plan, budget);
        if plan.is_idle() {
            if diag {
                eprintln!(
                    "[STEP-DIAG] plan idle after build (active={})",
                    self.active.len()
                );
            }
            return Ok(());
        }

        self.retract_decode_to_fit(&mut plan);
        if plan.is_idle() {
            return Ok(());
        }

        self.allocate_for_plan(&plan)?;
        log::trace!("infer-core submit plan: mode={:?}", plan.mode);
        if diag {
            eprintln!("[STEP-DIAG] SUBMIT plan mode={:?}", plan.mode);
        }
        self.inflight = Some(self.executor.submit(&plan, &mut self.kv)?);
        self.pending_plan = Some(plan);
        Ok(())
    }

    fn apply_step_budget(&self, plan: &mut ForwardPlan, budget: StepBudget) {
        if budget.max_tokens == usize::MAX {
            return;
        }

        let mut remaining = budget.max_tokens;
        if remaining == 0 {
            plan.decode_rows.clear();
            plan.prefill_rows.clear();
            plan.mode = infer_plan::ForwardMode::Idle;
            return;
        }

        if plan.decode_rows.len() > remaining {
            plan.decode_rows.truncate(remaining);
            plan.prefill_rows.clear();
            plan.mode =
                planner::plan_mode(plan.prefill_rows.is_empty(), plan.decode_rows.is_empty());
            return;
        }
        remaining -= plan.decode_rows.len();

        let mut keep_prefills = 0;
        for row in &mut plan.prefill_rows {
            if remaining == 0 {
                break;
            }
            if row.tokens.len() > remaining {
                row.tokens.truncate(remaining);
            }
            remaining = remaining.saturating_sub(row.tokens.len());
            keep_prefills += 1;
        }
        plan.prefill_rows.truncate(keep_prefills);
        plan.mode = planner::plan_mode(plan.prefill_rows.is_empty(), plan.decode_rows.is_empty());
    }

    /// Run scheduler ticks until there is no waiting, active, or in-flight work.
    pub fn run_to_idle(&mut self) -> Result<()> {
        while !self.is_idle() {
            self.step()?;
        }
        Ok(())
    }

    /// Return whether the engine has no queued, active, or in-flight work.
    #[must_use]
    pub fn is_idle(&self) -> bool {
        self.waiting.is_empty() && self.active.is_empty() && self.inflight.is_none()
    }

    /// Return whether an executor step is currently in flight.
    #[must_use]
    pub fn has_inflight(&self) -> bool {
        self.inflight.is_some()
    }

    /// Return the number of active slots.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.active.len()
    }

    /// Return the number of waiting requests.
    #[must_use]
    pub fn waiting_count(&self) -> usize {
        self.waiting.len()
    }

    /// Return currently free host-indexed KV pages.
    #[must_use]
    pub fn kv_free_pages(&self) -> usize {
        self.kv.free_pages()
    }

    /// Return prefix-cache counters plus the current retained cache size.
    #[must_use]
    pub fn prefix_cache_stats(&self) -> PrefixCacheStats {
        let mut stats = self.prefix_cache_stats;
        stats.cached_pages = self.radix.cached_page_count();
        stats
    }

    /// Return engine throughput counters (steps, tokens, completions).
    #[must_use]
    pub fn throughput_stats(&self) -> ThroughputStats {
        self.throughput_stats
    }

    /// Return KV host-tier counters plus the current tier-resident size.
    #[must_use]
    pub fn kv_tier_stats(&self) -> KvTierStats {
        let mut stats = self.kv_tier_stats;
        stats.resident_blocks = self.radix.demoted_block_count();
        stats
    }

    #[must_use]
    pub fn kv_system_metrics(&self) -> KvSystemMetrics {
        let mut metrics = self.kv_system_metrics;
        metrics.resident_pages = self.kv.resident_pages();
        metrics.resident_evictable_pages = self.kv.resident_evictable_pages();
        metrics.host_demoted_pages = self.executor.kv_tier_host_demoted_pages();
        metrics.host_demoted_pending_inflight = 0;
        metrics.disk_pages = self.executor.kv_tier_disk_pages();
        metrics
    }

    /// Return a completed request by handle.
    #[must_use]
    pub fn completed(&self, handle: RequestHandle) -> Option<&CompletedRequest> {
        self.completed.get(&handle)
    }

    fn next_handle(&mut self) -> RequestHandle {
        let handle = RequestHandle(self.next_request_id);
        self.next_request_id = self.next_request_id.saturating_add(1);
        handle
    }

    fn normalize_request(
        &self,
        handle: RequestHandle,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        options: RequestOptions,
    ) -> NormalizedRequest {
        if options.cancelled {
            return NormalizedRequest::Skipped;
        }

        if prompt_tokens.is_empty() || prompt_tokens.len() > self.config.max_prompt_tokens {
            return NormalizedRequest::Completed(
                RequestState::new(handle, prompt_tokens, options.priority, 0, options.sampling)
                    .complete_immediately(FinishReason::Abort),
            );
        }

        let max_tokens = max_tokens.min(
            self.config
                .max_total_tokens
                .saturating_sub(prompt_tokens.len()),
        );
        let mut sampling = options.sampling;
        // `max_tokens` is the scheduler-admitted budget after max-total clamping.
        // Keep the sampling copy normalized too: diffusion executors consume it
        // directly as their generation length.
        sampling.max_new_tokens = Some(max_tokens);
        if max_tokens == 0 {
            return NormalizedRequest::Completed(
                RequestState::new(handle, prompt_tokens, options.priority, 0, sampling)
                    .complete_immediately(FinishReason::Length),
            );
        }

        NormalizedRequest::Waiting(RequestState::new(
            handle,
            prompt_tokens,
            options.priority,
            max_tokens,
            sampling,
        ))
    }

    fn enqueue_waiting_request(&mut self, request: RequestState, bias: WaitingInsertBias) {
        let insert_at = waiting_insert_position(&self.waiting, &request, bias);
        self.waiting.insert(insert_at, request);
    }

    fn apply_output(&mut self, plan: &ForwardPlan, output: StepOutput) -> Result<()> {
        self.throughput_stats.steps = self.throughput_stats.steps.saturating_add(1);
        let mut tokens_by_slot: BTreeMap<usize, VecDeque<SlotToken>> = BTreeMap::new();
        for token in output.tokens {
            tokens_by_slot
                .entry(token.slot)
                .or_default()
                .push_back(token);
        }
        let mut finished_slots = Vec::new();
        // Slots whose prefill completed this step: capture their position-0
        // prefix image AFTER the prefill loop so the capture's `&mut self` does
        // not collide with the `self.active` request borrow inside the loop.
        let mut prefill_completed_slots: Vec<usize> = Vec::new();
        // Tokens committed this step, in commit order (prefill rows then decode
        // rows), forwarded to `on_token` after the request-borrowing loops so the
        // `self.active` and `self.on_token` borrows stay disjoint.
        let mut committed: Vec<(RequestHandle, SlotToken)> = Vec::new();
        let model_stops = self.model_stop_token_ids.clone();

        // Advance chunked prefill. A non-final chunk only moves progress; the
        // final chunk transitions the slot to decode and consumes its first token.
        for row in &plan.prefill_rows {
            let Some(request) = self.active.get_mut(&row.slot) else {
                continue;
            };
            if !matches!(request.phase, RequestPhase::Prefilling { .. }) {
                continue;
            }
            let prompt_len = request.prompt_tokens.len();
            let new_start = (row.start_pos + row.tokens.len()).min(prompt_len);
            self.throughput_stats.prefill_tokens = self
                .throughput_stats
                .prefill_tokens
                .saturating_add(new_start.saturating_sub(row.start_pos) as u64);
            request.prefill_start_pos = new_start;
            if new_start >= prompt_len {
                request.phase = RequestPhase::Decoding;
                // Capture-on-prefill-complete for the position-0 prefix store.
                // At this exact moment the slot's KV holds the full prompt at
                // absolute positions [0, prompt_len): the final prefill chunk is
                // committed and no decode token's KV is written yet. This is the
                // only moment the slot is a clean position-0 prompt image — a
                // later finish_slot would see prompt+generated KV. Deferred to
                // after the loop to keep the `&mut self` capture call disjoint
                // from the request borrow; the executor re-asserts
                // `slot.seq_len == prompt_len` and stores nothing if the slot is
                // misaligned (no-op when the store is disabled).
                prefill_completed_slots.push(row.slot);
                if let Some(token) = tokens_by_slot
                    .get_mut(&row.slot)
                    .and_then(VecDeque::pop_front)
                {
                    request.generated_tokens.push(token.token);
                    if let Some(finish) = finish_reason_for(request, &token, &model_stops) {
                        finished_slots.push((row.slot, finish));
                    }
                    committed.push((request.handle, token));
                }
            } else {
                request.phase = RequestPhase::Prefilling {
                    progress: new_start,
                };
                // A non-final chunk produces no committed token.
                tokens_by_slot.remove(&row.slot);
            }
        }

        // Capture the just-completed prefills into the position-0 prefix store.
        // The request borrow from the prefill loop has ended, so the capture's
        // `&mut self` is free. Clone the prompt tokens (the store needs the key)
        // and skip vanished slots defensively.
        for slot in prefill_completed_slots {
            let Some(prompt_tokens) = self
                .active
                .get(&slot)
                .map(|request| request.prompt_tokens.clone())
            else {
                continue;
            };
            if let Err(err) = self.executor.capture_cached_prefix(slot, &prompt_tokens) {
                log::warn!("position-0 prefix capture failed for slot {slot}: {err:#}");
            }
        }

        // Decode rows: append sampled token(s) and check stop/length after each.
        // Speculative backends may return multiple committed tokens for one
        // decode row; the host KV pool pre-allocated the first one in
        // `allocate_for_plan`, so each extra token gets one more logical append
        // before it is exposed to scheduling.
        for row in &plan.decode_rows {
            let Some(mut tokens) = tokens_by_slot.remove(&row.slot) else {
                continue;
            };
            let mut token_idx = 0usize;
            while let Some(token) = tokens.pop_front() {
                if token_idx > 0 {
                    self.alloc_with_prefix_reclaim(row.slot, 1)?;
                }
                let Some(request) = self.active.get_mut(&row.slot) else {
                    break;
                };
                if !matches!(request.phase, RequestPhase::Decoding) {
                    break;
                }
                request.generated_tokens.push(token.token);
                let finished = finish_reason_for(request, &token, &model_stops);
                committed.push((request.handle, token));
                token_idx += 1;
                if let Some(finish) = finished {
                    finished_slots.push((row.slot, finish));
                    break;
                }
            }
        }

        self.throughput_stats.generated_tokens = self
            .throughput_stats
            .generated_tokens
            .saturating_add(committed.len() as u64);

        // Stream committed tokens to the observer (if any) before finishing slots,
        // so a serving layer sees the terminal token ahead of completion.
        if let Some(observer) = &mut self.on_token {
            for (handle, token) in &committed {
                observer(*handle, token);
            }
        }

        for (slot, reason) in finished_slots {
            self.finish_slot(slot, reason);
        }
        Ok(())
    }

    fn finish_slot(&mut self, slot: usize, reason: FinishReason) {
        let Some(mut request) = self.active.remove(&slot) else {
            return;
        };
        self.throughput_stats.requests_completed =
            self.throughput_stats.requests_completed.saturating_add(1);
        request.phase = RequestPhase::Finished;
        request.finish = Some(reason);
        // Capture sidecar at final seq_len (prompt + decode tokens) before publishing
        // radix pages. Without this, decode tokens advance the radix cache past the
        // last prefill-time sidecar capture, causing full re-prefill fallbacks on every
        // subsequent agentic turn that generates ≥16 tokens.
        let full_tokens: Vec<u32> = request
            .prompt_tokens
            .iter()
            .chain(request.generated_tokens.iter())
            .copied()
            .collect();
        if let Err(err) = self.executor.capture_cached_prefix(slot, &full_tokens) {
            log::debug!("finish_slot sidecar capture failed for slot {slot}: {err:#}");
        }
        self.publish_prefix_blocks(slot, &request);
        // free_slot BEFORE release_reused_prefix: reclaim_page sees page_refs>0
        // and skips retained prefix pages; release_reused_prefix then drops the
        // last ref and the page enters the free pool exactly once. Reversed order
        // caused a double-push: release_pages dropped page_refs to 0 and pushed
        // to free, then reclaim_page (page_refs absent → 0) pushed again.
        self.kv.free_slot(slot);
        self.release_reused_prefix(&request.reused_prefix_pages);
        self.evict_prefix_cache_if_below_low_water();
        self.completed.insert(request.handle, request.into());
    }

    fn record_attached_prefix_metrics(&mut self, attached_pages: usize) {
        if !self.config.enable_prefix_cache {
            return;
        }
        if attached_pages == 0 {
            self.kv_system_metrics.reuse_miss = self.kv_system_metrics.reuse_miss.saturating_add(1);
            return;
        }
        let pages = attached_pages as u64;
        self.kv_system_metrics.prefix_match_full_blocks = self
            .kv_system_metrics
            .prefix_match_full_blocks
            .saturating_add(pages);
        self.kv_system_metrics.prefix_match_clamped_blocks = self
            .kv_system_metrics
            .prefix_match_clamped_blocks
            .saturating_add(pages);
        if self.kv_tier_capacity() == 0 {
            self.kv_system_metrics.reuse_hit_resident = self
                .kv_system_metrics
                .reuse_hit_resident
                .saturating_add(pages);
        }
    }

    fn admit_waiting(&mut self) -> Result<()> {
        match self.governor.admission_gate() {
            AdmissionVerdict::Admit | AdmissionVerdict::ShedTo(_) => {}
            AdmissionVerdict::Hold => return Ok(()),
        }

        if self.config.num_slots == 0 && !self.waiting.is_empty() {
            bail!("infer-core engine has waiting requests but zero slots");
        }

        let mut free_slots = self.free_slots();
        let mut remaining_prefill_tokens = self.config.prefill_step_budget();
        let mut active_prefills = self.active_prefill_count();
        let max_prefills = self.config.max_concurrent_prefill();
        let mut remaining_pages = self.kv.free_pages();
        self.evict_prefix_cache_if_below_low_water();

        while let Some(&slot) = free_slots.first() {
            match self.try_admit_front_waiter(
                slot,
                &mut remaining_prefill_tokens,
                &mut active_prefills,
                max_prefills,
                &mut remaining_pages,
            )? {
                AdmitOutcome::Admitted => {
                    free_slots.remove(0);
                }
                AdmitOutcome::NoWaiter | AdmitOutcome::Throttled => break,
            }
        }

        // P5 — slot-pressure oversubscription. After the budget-bounded admit
        // loop above drains every free slot, if waiters remain we may demote
        // (park) the longest-running decode's whole-slot image to free a slot
        // and admit a waiter in its place, round-robin. Default-off and the
        // whole-slot tier gate keep this byte-identical to before when unused;
        // it only adds WHEN the proven demote/resume fires.
        if self.config.slot_oversubscription && self.executor.kv_slot_tier_enabled() {
            self.admit_via_oversubscription(
                &mut remaining_prefill_tokens,
                &mut active_prefills,
                max_prefills,
                &mut remaining_pages,
            )?;
        }

        Ok(())
    }

    /// Admit the front waiter onto `slot`, advancing the per-tick budgets.
    /// Returns `Admitted` on success, `NoWaiter` if the queue is empty, or
    /// `Throttled` if the front waiter does not fit the current per-tick budget
    /// (prefill concurrency / token budget / KV pages). The single canonical
    /// admit body, shared by the main loop and the oversubscription trigger.
    fn try_admit_front_waiter(
        &mut self,
        slot: usize,
        remaining_prefill_tokens: &mut usize,
        active_prefills: &mut usize,
        max_prefills: usize,
        remaining_pages: &mut usize,
    ) -> Result<AdmitOutcome> {
        let Some(candidate) = self.waiting.front() else {
            return Ok(AdmitOutcome::NoWaiter);
        };
        let prefix_match = if self.config.enable_prefix_cache {
            let matched = self
                .radix
                .peek_longest_prefix_match(&candidate.prompt_tokens);
            self.clamp_prefix_to_backend(matched)
        } else {
            PrefixMatch::empty()
        };
        // Backends without page-radix reuse (DSv4) may still hold a
        // position-0 whole-slot prefix image. The page route reports
        // `matched_len == 0` for them, so budget the prefill/pages against
        // the executor's cached prefix length when it is longer. This only
        // affects budgeting; the actual restore happens at attach below.
        // Skipped for swap re-admissions: they restore the FULL sequence via
        // `restore_swapped_slot` (consuming the slot) and never take the
        // cached-prefix attach path, so the cached length is irrelevant to
        // their prefill (which is 0).
        let reuse_matched_len = if self.config.enable_prefix_cache && candidate.swap_key.is_none() {
            let cached = self
                .executor
                .cached_prefix_match_len(&candidate.prompt_tokens)?
                .min(candidate.prompt_len());
            prefix_match.matched_len.max(cached)
        } else {
            prefix_match.matched_len
        };
        let prefill_tokens = candidate.prompt_len().saturating_sub(reuse_matched_len);
        if prefill_tokens > 0 && *active_prefills >= max_prefills {
            return Ok(AdmitOutcome::Throttled);
        }
        // Long prompts are admitted and chunked across ticks (the planner
        // caps per-tick prefill tokens). The per-tick budget only throttles
        // how many NEW requests we admit at once: stop once it is consumed.
        if prefill_tokens > 0 && *remaining_prefill_tokens == 0 {
            return Ok(AdmitOutcome::Throttled);
        }

        let pages_needed = self.request_pages_needed_after_prefix(candidate, reuse_matched_len);
        if self.kv.is_active() && pages_needed > *remaining_pages {
            let reclaimed = self.evict_prefix_cache_for_pages(pages_needed - *remaining_pages);
            *remaining_pages = remaining_pages.saturating_add(reclaimed);
            if pages_needed > *remaining_pages {
                return Ok(AdmitOutcome::Throttled);
            }
        }

        let mut request = self
            .waiting
            .pop_front()
            .expect("waiting.front() was Some above");
        if let Some(key) = request.swap_key.take() {
            // Whole-slot re-admission: restore the swapped device image
            // and resume decode (falls back to recompute internally).
            self.restore_swapped_slot(slot, &mut request, key)?;
        } else if self.attach_cached_prefix(slot, &mut request)? > 0 {
            // Position-0 whole-slot prefix reuse (DSv4): the executor
            // restored the cached prefix image and set `prefill_start_pos`;
            // the tail re-prefills. No page-radix attach for this request.
        } else {
            let prefix_match = if self.config.enable_prefix_cache {
                // Tier-aware: demoted blocks in the match are promoted
                // back into fresh pages here, so attach sees a
                // resident-only match.
                self.lookup_prefix_for_attach(&request.prompt_tokens)
            } else {
                PrefixMatch::empty()
            };
            self.attach_prefix_to_request(slot, &mut request, prefix_match)?;
            self.record_attached_prefix_metrics(request.reused_prefix_pages.len());
        }

        *remaining_prefill_tokens = remaining_prefill_tokens.saturating_sub(
            request
                .prompt_len()
                .saturating_sub(request.prefill_start_pos),
        );
        *remaining_pages = remaining_pages.saturating_sub(pages_needed);
        if matches!(request.phase, RequestPhase::Prefilling { .. }) {
            *active_prefills += 1;
        }
        // Fresh admit stamp + generation mark: a just-admitted/resumed request
        // is the youngest (never the immediate victim) and must decode a min
        // slice before it can be parked again (bounds ping-pong churn).
        request.admit_seq = self.next_admit_seq;
        self.next_admit_seq = self.next_admit_seq.wrapping_add(1);
        request.admit_gen_mark = request.generated_tokens.len();
        self.active.insert(slot, request);
        Ok(AdmitOutcome::Admitted)
    }

    /// P5 oversubscription trigger: free slots by parking the longest-running
    /// decode (smallest `admit_seq`) and admit a waiter in its place, capped at
    /// `num_slots/4` (>=1) preemptions per call to bound swap churn. Terminates:
    /// each iteration either demotes+admits one (decrementing the bounded cap)
    /// or breaks (no eligible victim / waiter throttled / queue empty).
    fn admit_via_oversubscription(
        &mut self,
        remaining_prefill_tokens: &mut usize,
        active_prefills: &mut usize,
        max_prefills: usize,
        remaining_pages: &mut usize,
    ) -> Result<()> {
        let cap = self.config.num_slots.div_ceil(4).max(1);
        let mut preempted = 0usize;
        while preempted < cap && !self.waiting.is_empty() {
            let Some(victim_slot) = self.oversubscription_victim() else {
                break;
            };
            // Park `AfterEqual`: the victim yields to the existing equal-priority
            // waiter so we admit THAT one, not re-admit the just-parked victim.
            self.requeue_preempted_decode_with_bias(victim_slot, WaitingInsertBias::AfterEqual);
            *remaining_pages = self.kv.free_pages();
            // The freed slot may not match `victim_slot` if requeue reshuffled;
            // admit onto whichever slot is now free. There is exactly one.
            let Some(free_slot) = self.free_slots().into_iter().next() else {
                break;
            };
            match self.try_admit_front_waiter(
                free_slot,
                remaining_prefill_tokens,
                active_prefills,
                max_prefills,
                remaining_pages,
            )? {
                AdmitOutcome::Admitted => preempted += 1,
                // Demoted but the waiter still won't fit (throttle) or vanished:
                // stop rather than churn the parked victim back and forth.
                AdmitOutcome::Throttled | AdmitOutcome::NoWaiter => break,
            }
        }
        Ok(())
    }

    /// The oversubscription victim: among `Decoding` requests that have decoded
    /// at least `OVERSUBSCRIPTION_MIN_SLICE` tokens since their last admit, the
    /// one with the smallest `admit_seq` (longest continuous run). Prefilling /
    /// just-admitted requests are skipped — only a materialized decode has a
    /// whole-slot image to park; the min-slice floor + oldest-first selection
    /// prevent demote-resume thrash (a just-resumed request runs a slice first).
    fn oversubscription_victim(&self) -> Option<usize> {
        self.active
            .iter()
            .filter(|(_, request)| {
                matches!(request.phase, RequestPhase::Decoding)
                    && request
                        .generated_tokens
                        .len()
                        .saturating_sub(request.admit_gen_mark)
                        >= OVERSUBSCRIPTION_MIN_SLICE
            })
            .min_by_key(|(_, request)| request.admit_seq)
            .map(|(&slot, _)| slot)
    }

    fn free_slots(&self) -> Vec<usize> {
        (0..self.config.num_slots)
            .filter(|slot| !self.active.contains_key(slot))
            .collect()
    }

    fn active_prefill_count(&self) -> usize {
        self.active
            .values()
            .filter(|request| matches!(request.phase, RequestPhase::Prefilling { .. }))
            .count()
    }

    fn allocate_for_plan(&mut self, plan: &ForwardPlan) -> Result<()> {
        for row in &plan.prefill_rows {
            self.alloc_with_prefix_reclaim(row.slot, row.tokens.len())?;
        }
        for row in &plan.decode_rows {
            self.alloc_with_prefix_reclaim(row.slot, 1)?;
        }
        Ok(())
    }

    fn evict_prefix_cache_if_below_low_water(&mut self) -> usize {
        let low_water = self.config.prefix_cache_low_water_pages;
        if low_water <= self.kv.free_pages() {
            return 0;
        }
        self.evict_prefix_cache_for_pages(low_water - self.kv.free_pages())
    }
}

enum NormalizedRequest {
    Waiting(RequestState),
    Completed(RequestState),
    Skipped,
}

fn waiting_insert_position(
    waiting: &VecDeque<RequestState>,
    incoming: &RequestState,
    bias: WaitingInsertBias,
) -> usize {
    waiting
        .iter()
        .position(|queued| waiting_request_precedes(incoming, queued, bias))
        .unwrap_or(waiting.len())
}

fn waiting_request_precedes(
    incoming: &RequestState,
    queued: &RequestState,
    bias: WaitingInsertBias,
) -> bool {
    // Admission order is a single lexicographic key: higher priority first, then
    // more reusable KV (session-affinity, then immediate, then total reuse). On a
    // full tie the bias decides whether `incoming` sorts before an equal `queued`.
    let key = |r: &RequestState| {
        (
            r.priority,
            r.waiting_hint.session_affinity_tokens,
            r.waiting_hint.immediate_reuse_tokens,
            r.waiting_hint.total_reuse_tokens,
        )
    };
    match key(incoming).cmp(&key(queued)) {
        std::cmp::Ordering::Greater => true,
        std::cmp::Ordering::Less => false,
        std::cmp::Ordering::Equal => matches!(bias, WaitingInsertBias::BeforeEqual),
    }
}

fn finish_reason_for(
    request: &RequestState,
    token: &SlotToken,
    model_stop_token_ids: &[u32],
) -> Option<FinishReason> {
    if let Some(finish) = token.finish.clone() {
        return Some(finish);
    }
    let sampling = &request.sampling;
    // Request-supplied stops take priority; the model defaults are the fallback
    // used only when the request supplies none. Both honor `ignore_eos`.
    let stop_ids: &[u32] = if sampling.stop_token_ids.is_empty() {
        model_stop_token_ids
    } else {
        &sampling.stop_token_ids
    };
    if !sampling.ignore_eos && stop_ids.contains(&token.token) {
        Some(FinishReason::Stop)
    } else if request.generated_tokens.len() >= request.max_tokens {
        Some(FinishReason::Length)
    } else {
        None
    }
}

#[cfg(test)]
mod testing {
    use std::collections::BTreeMap;

    use anyhow::{Result, bail};
    use infer_plan::{SamplingParams, SlotToken, StepOutput, sample_token};
    use infer_seam::{
        AdmissionVerdict, BackendExecutor, KvAllocator, KvPool, KvPrefixStore, KvQuery, PollResult,
        PrefixBlock, ResourceGovernor, StepBudget, pages_only_reusable_prefix_blocks,
    };

    use super::ForwardPlan;

    #[derive(Debug, Clone)]
    pub(super) struct MockKvPool {
        page_size: usize,
        seq_lens: Vec<usize>,
        pages: Vec<Vec<u32>>,
        slot_epochs: Vec<u64>,
        free_pages: Vec<u32>,
        next_page: u32,
        total_pages: usize,
        page_ref_counts: BTreeMap<u32, usize>,
        page_attach_counts: BTreeMap<u32, usize>,
    }

    impl MockKvPool {
        pub(super) fn new(num_slots: usize) -> Self {
            Self::with_capacity(num_slots, 16, 4096)
        }

        pub(super) fn with_capacity(
            num_slots: usize,
            page_size: usize,
            total_pages: usize,
        ) -> Self {
            let capped_pages = total_pages.min(u32::MAX as usize);
            Self {
                page_size,
                seq_lens: vec![0; num_slots],
                pages: vec![Vec::new(); num_slots],
                slot_epochs: vec![0; num_slots],
                free_pages: (1..=capped_pages as u32).rev().collect(),
                next_page: capped_pages as u32 + 1,
                total_pages: capped_pages,
                page_ref_counts: BTreeMap::new(),
                page_attach_counts: BTreeMap::new(),
            }
        }

        pub(super) fn total_pages(&self) -> usize {
            self.total_pages
        }

        fn ensure_slot(&self, slot: usize) -> Result<()> {
            if slot >= self.seq_lens.len() {
                bail!("slot {slot} out of range");
            }
            Ok(())
        }

        fn alloc_pages(&mut self, count: usize) -> Result<Vec<u32>> {
            if count > self.free_pages.len() {
                bail!(
                    "mock pool out of pages: requested {count}, free {}",
                    self.free_pages.len()
                );
            }
            let pages = (0..count)
                .map(|_| self.free_pages.pop().expect("free page count checked"))
                .collect();
            Ok(pages)
        }

        fn ensure_total_capacity_for_detached(&mut self, count: usize) {
            while self.free_pages.len() < count && self.next_page < u32::MAX {
                self.free_pages.push(self.next_page);
                self.next_page = self.next_page.saturating_add(1);
                self.total_pages = self.total_pages.saturating_add(1);
            }
        }

        fn remove_from_free_list(&mut self, page: u32) {
            if let Some(pos) = self.free_pages.iter().position(|&free| free == page) {
                self.free_pages.swap_remove(pos);
            }
        }

        fn retain_page(&mut self, page: u32) {
            self.remove_from_free_list(page);
            *self.page_ref_counts.entry(page).or_insert(0) += 1;
        }

        fn release_page(&mut self, page: u32) {
            if let Some(count) = self.page_ref_counts.get_mut(&page) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    self.page_ref_counts.remove(&page);
                }
            }
            self.recycle_if_unused(page);
        }

        fn attach_page(&mut self, page: u32) {
            self.remove_from_free_list(page);
            *self.page_attach_counts.entry(page).or_insert(0) += 1;
        }

        fn detach_page(&mut self, page: u32) {
            if let Some(count) = self.page_attach_counts.get_mut(&page) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    self.page_attach_counts.remove(&page);
                }
            }
            self.recycle_if_unused(page);
        }

        fn recycle_if_unused(&mut self, page: u32) {
            let retained = self.page_ref_counts.get(&page).copied().unwrap_or(0);
            let attached = self.page_attach_counts.get(&page).copied().unwrap_or(0);
            if retained == 0 && attached == 0 && !self.free_pages.contains(&page) {
                self.free_pages.push(page);
            }
        }
    }

    impl KvQuery for MockKvPool {
        fn is_active(&self) -> bool {
            true
        }

        fn page_size(&self) -> usize {
            self.page_size
        }

        fn free_pages(&self) -> usize {
            self.free_pages.len()
        }

        fn free_tokens(&self) -> usize {
            let partial_capacity = self
                .seq_lens
                .iter()
                .map(|seq_len| {
                    let tail = seq_len % self.page_size;
                    if tail == 0 { 0 } else { self.page_size - tail }
                })
                .sum::<usize>();
            self.free_pages.len() * self.page_size + partial_capacity
        }

        fn resident_pages(&self) -> usize {
            self.total_pages.saturating_sub(self.free_pages.len())
        }

        fn resident_evictable_pages(&self) -> usize {
            self.page_ref_counts
                .iter()
                .filter(|entry| {
                    let (page, count) = *entry;
                    *count > 0 && self.page_attach_counts.get(page).copied().unwrap_or(0) == 0
                })
                .count()
        }

        fn seq_len(&self, slot: usize) -> usize {
            self.seq_lens[slot]
        }

        fn slot_epoch(&self, slot: usize) -> u64 {
            self.slot_epochs[slot]
        }

        fn append_pages_needed(&self, slot: usize, tokens: usize) -> usize {
            if tokens == 0 {
                return 0;
            }
            let tail = self.seq_lens[slot] % self.page_size;
            let available = if tail == 0 { 0 } else { self.page_size - tail };
            tokens.saturating_sub(available).div_ceil(self.page_size)
        }

        fn page_indices(&self, slot: usize) -> &[u32] {
            &self.pages[slot]
        }

        fn page_indices_for_token_range(&self, slot: usize, start: usize, len: usize) -> &[u32] {
            let start_page = start / self.page_size;
            let end_page = (start + len).div_ceil(self.page_size);
            &self.pages[slot][start_page..end_page]
        }
    }

    impl KvAllocator for MockKvPool {
        fn alloc(&mut self, slot: usize, tokens: usize) -> Result<()> {
            self.ensure_slot(slot)?;
            let new_pages = self.append_pages_needed(slot, tokens);
            let pages = self.alloc_pages(new_pages)?;
            for &page in &pages {
                self.attach_page(page);
            }
            self.pages[slot].extend_from_slice(&pages);
            self.seq_lens[slot] += tokens;
            Ok(())
        }

        fn alloc_detached_pages(&mut self, pages: usize) -> Result<Vec<u32>> {
            self.ensure_total_capacity_for_detached(pages);
            let pages = self.alloc_pages(pages)?;
            for &page in &pages {
                self.retain_page(page);
            }
            Ok(pages)
        }

        fn free_detached_pages(&mut self, pages: &[u32]) {
            // Mock detached pages carry a retain ref (see alloc_detached_pages);
            // releasing it returns them to the free pool.
            KvPrefixStore::release_pages(self, pages);
        }

        fn free_slot(&mut self, slot: usize) {
            let pages = std::mem::take(&mut self.pages[slot]);
            for page in pages {
                self.detach_page(page);
            }
            self.seq_lens[slot] = 0;
            self.slot_epochs[slot] = self.slot_epochs[slot].saturating_add(1);
        }

        fn truncate_slot(&mut self, slot: usize, new_len: usize) -> Result<()> {
            self.ensure_slot(slot)?;
            if new_len > self.seq_lens[slot] {
                bail!("cannot grow slot {slot} via truncate");
            }
            self.seq_lens[slot] = new_len;
            let keep_pages = new_len.div_ceil(self.page_size);
            let slot_page_len = self.pages[slot].len();
            let removed = self.pages[slot].split_off(keep_pages.min(slot_page_len));
            for page in removed {
                self.detach_page(page);
            }
            self.slot_epochs[slot] = self.slot_epochs[slot].saturating_add(1);
            Ok(())
        }

        fn migrate(&mut self, slot: usize, start: usize, len: usize) -> Result<()> {
            self.ensure_slot(slot)?;
            let target = start.saturating_add(len);
            if target > self.seq_lens[slot] {
                self.alloc(slot, target - self.seq_lens[slot])?;
            }
            Ok(())
        }
    }

    impl KvPrefixStore for MockKvPool {
        fn retain_pages(&mut self, pages: &[u32]) {
            for page in pages {
                self.retain_page(*page);
            }
        }

        fn release_pages(&mut self, pages: &[u32]) {
            for page in pages {
                self.release_page(*page);
            }
        }

        fn retained_count(&self) -> usize {
            self.page_ref_counts.len()
        }

        fn attach_pages(&mut self, slot: usize, pages: &[u32], token_count: usize) -> Result<()> {
            self.ensure_slot(slot)?;
            let old_pages = std::mem::take(&mut self.pages[slot]);
            for page in old_pages {
                self.detach_page(page);
            }
            self.pages[slot].extend_from_slice(pages);
            for &page in pages {
                self.attach_page(page);
            }
            self.seq_lens[slot] = token_count;
            self.slot_epochs[slot] = self.slot_epochs[slot].saturating_add(1);
            Ok(())
        }
    }

    #[derive(Debug, Clone)]
    pub(super) struct MockInflight {
        output: StepOutput,
        return_not_ready_once: bool,
    }

    #[derive(Debug, Clone)]
    pub(super) struct MockExecutor {
        not_ready_once_per_submit: bool,
    }

    impl MockExecutor {
        pub(super) fn ready() -> Self {
            Self {
                not_ready_once_per_submit: false,
            }
        }

        pub(super) fn not_ready_once() -> Self {
            Self {
                not_ready_once_per_submit: true,
            }
        }
    }

    /// Echo executor token rule shared by the mock executors: each prefill row
    /// emits `last_prompt_token + 1` (or `1` when empty) and each decode row emits
    /// `last_token + 1`. Real backends sample; the mocks just need a deterministic
    /// monotonically-advancing token so the engine's slot bookkeeping is exercised.
    fn echo_tokens(plan: &ForwardPlan) -> Vec<SlotToken> {
        let prefill = plan.prefill_rows.iter().map(|row| SlotToken {
            slot: row.slot,
            token: row.tokens.last().copied().map_or(1, |last| last + 1),
            logprob: None,
            finish: None,
        });
        let decode = plan.decode_rows.iter().map(|row| SlotToken {
            slot: row.slot,
            token: row.last_token + 1,
            logprob: None,
            finish: None,
        });
        prefill.chain(decode).collect()
    }

    impl BackendExecutor for MockExecutor {
        type Inflight = MockInflight;

        fn submit(&mut self, plan: &ForwardPlan, _kv: &mut dyn KvPool) -> Result<Self::Inflight> {
            Ok(MockInflight {
                output: StepOutput {
                    tokens: echo_tokens(plan),
                },
                return_not_ready_once: self.not_ready_once_per_submit,
            })
        }

        fn poll(&mut self, mut inflight: Self::Inflight) -> Result<PollResult<Self::Inflight>> {
            if inflight.return_not_ready_once {
                inflight.return_not_ready_once = false;
                Ok(PollResult::NotReady(inflight))
            } else {
                Ok(PollResult::Ready(inflight.output))
            }
        }

        fn reusable_prefix_blocks(&self, blocks: &[PrefixBlock]) -> usize {
            pages_only_reusable_prefix_blocks(blocks, |_| false)
        }
    }

    /// Executor emulating the DSv4 prefix-reuse seam: the page-radix route is
    /// always empty (`reusable_prefix_blocks == 0`, like DSv4), and reuse rides
    /// the position-0 whole-slot prefix store instead. `store` maps a captured
    /// full prompt to a marker; `restored`/`captured` record the wiring so the
    /// scheduler test can assert the engine drove the right hooks.
    pub(super) struct PrefixStoreMockExecutor {
        inner: MockExecutor,
        store: std::cell::RefCell<Vec<Vec<u32>>>,
        pub(super) captured: std::rc::Rc<std::cell::RefCell<Vec<Vec<u32>>>>,
        pub(super) restored: std::rc::Rc<std::cell::RefCell<Vec<(usize, usize)>>>,
    }

    impl PrefixStoreMockExecutor {
        pub(super) fn new() -> Self {
            Self {
                inner: MockExecutor::ready(),
                store: std::cell::RefCell::new(Vec::new()),
                captured: std::rc::Rc::new(std::cell::RefCell::new(Vec::new())),
                restored: std::rc::Rc::new(std::cell::RefCell::new(Vec::new())),
            }
        }
    }

    impl BackendExecutor for PrefixStoreMockExecutor {
        type Inflight = MockInflight;

        fn submit(&mut self, plan: &ForwardPlan, kv: &mut dyn KvPool) -> Result<Self::Inflight> {
            self.inner.submit(plan, kv)
        }

        fn poll(&mut self, inflight: Self::Inflight) -> Result<PollResult<Self::Inflight>> {
            self.inner.poll(inflight)
        }

        // Page-radix reuse is unavailable (DSv4 shape): every block clamps to 0.
        fn reusable_prefix_blocks(&self, _blocks: &[PrefixBlock]) -> usize {
            0
        }

        fn cached_prefix_match_len(&self, tokens: &[u32]) -> Result<usize> {
            Ok(self
                .store
                .borrow()
                .iter()
                .filter(|stored| {
                    stored.len() <= tokens.len() && tokens[..stored.len()] == stored[..]
                })
                .map(Vec::len)
                .max()
                .unwrap_or(0))
        }

        fn capture_cached_prefix(&mut self, _slot: usize, tokens: &[u32]) -> Result<()> {
            self.captured.borrow_mut().push(tokens.to_vec());
            let mut store = self.store.borrow_mut();
            if !store.iter().any(|s| s == tokens) {
                store.push(tokens.to_vec());
            }
            Ok(())
        }

        fn restore_cached_prefix(
            &mut self,
            slot: usize,
            tokens: &[u32],
            matched_len: usize,
            _slot_pages: &[u32],
        ) -> Result<()> {
            let key = &tokens[..matched_len];
            if !self.store.borrow().iter().any(|s| s == key) {
                bail!("restore of unknown prefix len {matched_len}");
            }
            self.restored.borrow_mut().push((slot, matched_len));
            Ok(())
        }
    }

    #[derive(Debug, Clone)]
    pub(super) struct SingleRowExecutor {
        inner: MockExecutor,
        pub(super) max_rows_seen: std::rc::Rc<std::cell::Cell<usize>>,
    }

    impl SingleRowExecutor {
        pub(super) fn new() -> Self {
            Self {
                inner: MockExecutor::ready(),
                max_rows_seen: std::rc::Rc::new(std::cell::Cell::new(0)),
            }
        }
    }

    impl BackendExecutor for SingleRowExecutor {
        type Inflight = MockInflight;

        fn submit(&mut self, plan: &ForwardPlan, kv: &mut dyn KvPool) -> Result<Self::Inflight> {
            let rows = plan.prefill_rows.len() + plan.decode_rows.len();
            self.max_rows_seen.set(self.max_rows_seen.get().max(rows));
            if rows > 1 {
                bail!("single-row executor received {rows} rows");
            }
            self.inner.submit(plan, kv)
        }

        fn poll(&mut self, inflight: Self::Inflight) -> Result<PollResult<Self::Inflight>> {
            self.inner.poll(inflight)
        }

        fn max_rows_per_step(&self) -> usize {
            1
        }
    }

    /// Mock executor that reports model-default stop tokens, mirroring how a
    /// real backend (e.g. Metal) exposes its config EOS/stop ids to the engine.
    #[derive(Debug, Clone)]
    pub(super) struct StopTokenExecutor {
        inner: MockExecutor,
        model_stops: Vec<u32>,
    }

    impl StopTokenExecutor {
        pub(super) fn with_model_stops(model_stops: Vec<u32>) -> Self {
            Self {
                inner: MockExecutor::ready(),
                model_stops,
            }
        }
    }

    impl BackendExecutor for StopTokenExecutor {
        type Inflight = MockInflight;

        fn submit(&mut self, plan: &ForwardPlan, kv: &mut dyn KvPool) -> Result<Self::Inflight> {
            self.inner.submit(plan, kv)
        }

        fn poll(&mut self, inflight: Self::Inflight) -> Result<PollResult<Self::Inflight>> {
            self.inner.poll(inflight)
        }

        fn model_stop_token_ids(&self) -> Vec<u32> {
            self.model_stops.clone()
        }
    }

    /// Mock executor that can only reuse a bounded number of leading prefix
    /// pages, mirroring a backend whose complete restore state exists at only
    /// some page boundaries. The engine must clamp the radix-offered prefix down
    /// to this count before attaching, or it would ask the executor for a
    /// boundary it cannot serve.
    #[derive(Debug, Clone)]
    pub(super) struct LimitedPrefixExecutor {
        inner: MockExecutor,
        max_reuse_blocks: usize,
    }

    impl LimitedPrefixExecutor {
        pub(super) fn with_max_reuse_blocks(max_reuse_blocks: usize) -> Self {
            Self {
                inner: MockExecutor::ready(),
                max_reuse_blocks,
            }
        }
    }

    impl BackendExecutor for LimitedPrefixExecutor {
        type Inflight = MockInflight;

        fn submit(&mut self, plan: &ForwardPlan, kv: &mut dyn KvPool) -> Result<Self::Inflight> {
            self.inner.submit(plan, kv)
        }

        fn poll(&mut self, inflight: Self::Inflight) -> Result<PollResult<Self::Inflight>> {
            self.inner.poll(inflight)
        }

        fn reusable_prefix_blocks(&self, blocks: &[PrefixBlock]) -> usize {
            pages_only_reusable_prefix_blocks(blocks, |_| false).min(self.max_reuse_blocks)
        }
    }

    /// Mock executor that records how many times `warmup` was called, so a test
    /// can assert the engine warms the backend exactly once.
    #[derive(Debug, Clone, Default)]
    pub(super) struct WarmupCountingExecutor {
        pub(super) warmup_calls: std::rc::Rc<std::cell::Cell<u32>>,
    }

    impl BackendExecutor for WarmupCountingExecutor {
        type Inflight = MockInflight;

        fn submit(&mut self, plan: &ForwardPlan, _kv: &mut dyn KvPool) -> Result<Self::Inflight> {
            Ok(MockInflight {
                output: StepOutput {
                    tokens: echo_tokens(plan),
                },
                return_not_ready_once: false,
            })
        }

        fn poll(&mut self, inflight: Self::Inflight) -> Result<PollResult<Self::Inflight>> {
            Ok(PollResult::Ready(inflight.output))
        }

        fn warmup(&mut self) -> Result<()> {
            self.warmup_calls.set(self.warmup_calls.get() + 1);
            Ok(())
        }
    }

    /// Mock executor that exercises the full host sampling seam instead of
    /// echoing input tokens: it samples from a fixed logits row using the
    /// `SamplingParams` and logical position supplied by each plan row.
    #[derive(Debug, Clone)]
    pub(super) struct SamplingExecutor {
        logits: std::rc::Rc<Vec<f32>>,
        observed: std::rc::Rc<std::cell::RefCell<Vec<(usize, u64, SamplingParams)>>>,
    }

    impl SamplingExecutor {
        pub(super) fn new(logits: Vec<f32>) -> Self {
            Self {
                logits: std::rc::Rc::new(logits),
                observed: std::rc::Rc::new(std::cell::RefCell::new(Vec::new())),
            }
        }

        pub(super) fn observed(&self) -> Vec<(usize, u64, SamplingParams)> {
            self.observed.borrow().clone()
        }
    }

    impl BackendExecutor for SamplingExecutor {
        type Inflight = MockInflight;

        fn submit(&mut self, plan: &ForwardPlan, _kv: &mut dyn KvPool) -> Result<Self::Inflight> {
            let tokens = plan
                .prefill_rows
                .iter()
                .map(|row| {
                    let position = (row.start_pos + row.tokens.len()) as u64;
                    self.observed
                        .borrow_mut()
                        .push((row.slot, position, row.params.clone()));
                    SlotToken {
                        slot: row.slot,
                        token: sample_token(&self.logits, &row.params, position),
                        logprob: None,
                        finish: None,
                    }
                })
                .chain(plan.decode_rows.iter().map(|row| {
                    let position = row.kv_seq_len.saturating_add(1) as u64;
                    self.observed
                        .borrow_mut()
                        .push((row.slot, position, row.params.clone()));
                    SlotToken {
                        slot: row.slot,
                        token: sample_token(&self.logits, &row.params, position),
                        logprob: None,
                        finish: None,
                    }
                }))
                .collect();
            Ok(MockInflight {
                output: StepOutput { tokens },
                return_not_ready_once: false,
            })
        }

        fn poll(&mut self, inflight: Self::Inflight) -> Result<PollResult<Self::Inflight>> {
            Ok(PollResult::Ready(inflight.output))
        }
    }

    /// Mirrors the real CUDA executor's device KV advancement + its decode-step
    /// length invariant (`device.seq_len(slot) == DecodeRow.kv_seq_len`), so a
    /// host-side `kv_seq_len` off-by-one fails on CPU rather than only on an H20.
    #[derive(Debug, Clone, Default)]
    pub(super) struct DeviceMirrorExecutor {
        /// Per-slot materialized device KV length, mirroring the executor pool.
        materialized: std::rc::Rc<std::cell::RefCell<std::collections::HashMap<usize, usize>>>,
        /// Recorded `(materialized_before, kv_seq_len)` for every decode row.
        decode_log: std::rc::Rc<std::cell::RefCell<Vec<(usize, usize, usize)>>>,
    }

    impl DeviceMirrorExecutor {
        pub(super) fn materialized(&self, slot: usize) -> usize {
            self.materialized.borrow().get(&slot).copied().unwrap_or(0)
        }

        /// `(slot, materialized_before_decode, decode_row.kv_seq_len)` per decode.
        pub(super) fn decode_log(&self) -> Vec<(usize, usize, usize)> {
            self.decode_log.borrow().clone()
        }
    }

    impl BackendExecutor for DeviceMirrorExecutor {
        type Inflight = MockInflight;

        fn submit(&mut self, plan: &ForwardPlan, _kv: &mut dyn KvPool) -> Result<Self::Inflight> {
            let mut tokens = Vec::new();
            let mut materialized = self.materialized.borrow_mut();

            for row in &plan.prefill_rows {
                // Mirrors RealCudaExecutor::ensure_slot_ready_for_prefill: a fresh
                // prefill (start_pos == 0) of a reused slot drops the stale device
                // pages before re-allocating.
                let entry = materialized.entry(row.slot).or_insert(0);
                if row.start_pos == 0 {
                    *entry = 0;
                } else if *entry != row.start_pos {
                    bail!(
                        "device mirror: chunked prefill expects materialized {} == start_pos {} for slot {}",
                        *entry,
                        row.start_pos,
                        row.slot
                    );
                }
                *entry += row.tokens.len();
                let token = row.tokens.last().copied().map_or(1, |last| last + 1);
                tokens.push(SlotToken {
                    slot: row.slot,
                    token,
                    logprob: None,
                    finish: None,
                });
            }

            for row in &plan.decode_rows {
                let entry = materialized.entry(row.slot).or_insert(0);
                self.decode_log
                    .borrow_mut()
                    .push((row.slot, *entry, row.kv_seq_len));
                // The invariant the real executor asserts at executor.rs:139.
                if *entry != row.kv_seq_len {
                    bail!(
                        "CUDA materialized cache_len {} != DecodeRow.kv_seq_len {} for slot {}",
                        *entry,
                        row.kv_seq_len,
                        row.slot
                    );
                }
                *entry += 1;
                tokens.push(SlotToken {
                    slot: row.slot,
                    token: row.last_token + 1,
                    logprob: None,
                    finish: None,
                });
            }

            Ok(MockInflight {
                output: StepOutput { tokens },
                return_not_ready_once: false,
            })
        }

        fn poll(&mut self, inflight: Self::Inflight) -> Result<PollResult<Self::Inflight>> {
            Ok(PollResult::Ready(inflight.output))
        }
    }

    /// Mirrors a depth-1 speculative backend: each decode forward materializes
    /// two device-token positions and returns two committed tokens. This catches
    /// host-side `kv_seq_len` drift when the scheduler has to advance by the real
    /// accepted-token count, not by a hard-coded one token per row.
    #[derive(Debug, Clone, Default)]
    pub(super) struct SpecMirrorExecutor {
        materialized: std::rc::Rc<std::cell::RefCell<std::collections::HashMap<usize, usize>>>,
        decode_log: std::rc::Rc<std::cell::RefCell<Vec<(usize, usize, usize)>>>,
    }

    impl SpecMirrorExecutor {
        pub(super) fn materialized(&self, slot: usize) -> usize {
            self.materialized.borrow().get(&slot).copied().unwrap_or(0)
        }

        pub(super) fn decode_log(&self) -> Vec<(usize, usize, usize)> {
            self.decode_log.borrow().clone()
        }
    }

    impl BackendExecutor for SpecMirrorExecutor {
        type Inflight = MockInflight;

        fn submit(&mut self, plan: &ForwardPlan, _kv: &mut dyn KvPool) -> Result<Self::Inflight> {
            let mut tokens = Vec::new();
            let mut materialized = self.materialized.borrow_mut();

            for row in &plan.prefill_rows {
                let entry = materialized.entry(row.slot).or_insert(0);
                if row.start_pos == 0 {
                    *entry = 0;
                } else if *entry != row.start_pos {
                    bail!(
                        "spec mirror: chunked prefill expects materialized {} == start_pos {} for slot {}",
                        *entry,
                        row.start_pos,
                        row.slot
                    );
                }
                *entry += row.tokens.len();
                tokens.push(SlotToken {
                    slot: row.slot,
                    token: row.tokens.last().copied().map_or(1, |last| last + 1),
                    logprob: None,
                    finish: None,
                });
            }

            for row in &plan.decode_rows {
                let entry = materialized.entry(row.slot).or_insert(0);
                self.decode_log
                    .borrow_mut()
                    .push((row.slot, *entry, row.kv_seq_len));
                if *entry != row.kv_seq_len {
                    bail!(
                        "spec mirror: materialized cache_len {} != DecodeRow.kv_seq_len {} for slot {}",
                        *entry,
                        row.kv_seq_len,
                        row.slot
                    );
                }
                *entry += 2;
                tokens.push(SlotToken {
                    slot: row.slot,
                    token: row.last_token + 1,
                    logprob: None,
                    finish: None,
                });
                tokens.push(SlotToken {
                    slot: row.slot,
                    token: row.last_token + 2,
                    logprob: None,
                    finish: None,
                });
            }

            Ok(MockInflight {
                output: StepOutput { tokens },
                return_not_ready_once: false,
            })
        }

        fn poll(&mut self, inflight: Self::Inflight) -> Result<PollResult<Self::Inflight>> {
            Ok(PollResult::Ready(inflight.output))
        }
    }

    /// Mirrors the Qwen3.6 hybrid (recurrent + full-attn paged) executor across an
    /// AGENTIC RE-PREFILL turn boundary — the path that produced the
    /// `materialized state len != DecodeRow.kv_seq_len` (Δ=42) crash at ~4.5K
    /// tokens. Each agent turn submits a NEW request whose prompt is a growing
    /// superset of the prior turn (prior context + tool output), so it lands on
    /// the same slot and reuses the published page-radix prefix at `start_pos =
    /// matched_len`. The decisive detail: the slot's device counter carries the
    /// PRIOR turn's `prompt + generated` length (HIGHER than this turn's
    /// `matched_len`), so the attach must REWIND it to `matched_len`. The bug was
    /// `restore_recurrent_sidecar` skipping that rewind (the slot stayed at the
    /// stale higher value while the host pool reset to `matched_len`).
    ///
    /// `rewind_on_attach=true` models the fix (`set_seq_len(matched_len)` inside
    /// `restore_prefix_sidecar`); `false` reproduces the pre-fix drift so the test
    /// proves it actually catches the regression rather than passing vacuously.
    #[derive(Debug, Clone)]
    pub(super) struct HybridReprefillMirror {
        materialized: std::rc::Rc<std::cell::RefCell<std::collections::HashMap<usize, usize>>>,
        decode_log: std::rc::Rc<std::cell::RefCell<Vec<(usize, usize, usize)>>>,
        /// The fix: rewind the device counter to `matched_len` on prefix attach.
        rewind_on_attach: bool,
    }

    impl HybridReprefillMirror {
        pub(super) fn new(rewind_on_attach: bool) -> Self {
            Self {
                materialized: std::rc::Rc::new(std::cell::RefCell::new(
                    std::collections::HashMap::new(),
                )),
                decode_log: std::rc::Rc::new(std::cell::RefCell::new(Vec::new())),
                rewind_on_attach,
            }
        }

        pub(super) fn decode_log(&self) -> Vec<(usize, usize, usize)> {
            self.decode_log.borrow().clone()
        }
    }

    impl BackendExecutor for HybridReprefillMirror {
        type Inflight = MockInflight;

        // Opt into page-radix prefix reuse exactly like the real Qwen3.5/3.6
        // executor — the default seam impl is fail-closed (0), which would route
        // every turn through a fresh `start_pos == 0` prefill and never exercise
        // the re-prefill rewind this test targets.
        fn reusable_prefix_blocks(&self, blocks: &[PrefixBlock]) -> usize {
            pages_only_reusable_prefix_blocks(blocks, |_| false)
        }

        // The hybrid override: `attach_prefix_to_request` calls this after
        // `kv.attach_pages()` reset the HOST pool's seq_len to `matched_len`. The
        // device-side counter MUST be rewound to match (the `1b0f0459` fix). With
        // `rewind_on_attach == false` we leave the stale higher value in place,
        // reproducing the drift.
        fn restore_prefix_sidecar(
            &mut self,
            slot: usize,
            _tokens: &[u32],
            matched_len: usize,
            _prefix_pages: &[u32],
        ) -> Result<()> {
            if self.rewind_on_attach {
                self.materialized.borrow_mut().insert(slot, matched_len);
            }
            Ok(())
        }

        fn submit(&mut self, plan: &ForwardPlan, _kv: &mut dyn KvPool) -> Result<Self::Inflight> {
            let mut tokens = Vec::new();
            let mut materialized = self.materialized.borrow_mut();

            for row in &plan.prefill_rows {
                let entry = materialized.entry(row.slot).or_insert(0);
                if row.start_pos == 0 {
                    *entry = 0;
                } else if *entry != row.start_pos {
                    // The real `prefill_row_paged_default` readiness guard
                    // (`pool.seq_len(slot) == start_pos`): a stale (un-rewound)
                    // device counter trips here on the tail prefill.
                    bail!(
                        "hybrid mirror: re-prefill expects materialized {} == start_pos {} for slot {} \
                         (device counter not rewound to matched_len on prefix attach)",
                        *entry,
                        row.start_pos,
                        row.slot
                    );
                }
                *entry += row.tokens.len();
                tokens.push(SlotToken {
                    slot: row.slot,
                    token: row.tokens.last().copied().map_or(1, |last| last + 1),
                    logprob: None,
                    finish: None,
                });
            }

            for row in &plan.decode_rows {
                let entry = materialized.entry(row.slot).or_insert(0);
                self.decode_log
                    .borrow_mut()
                    .push((row.slot, *entry, row.kv_seq_len));
                // The invariant asserted at executor.rs `submit_decode_row`.
                if *entry != row.kv_seq_len {
                    bail!(
                        "hybrid mirror: materialized state len {} != DecodeRow.kv_seq_len {} for slot {}",
                        *entry,
                        row.kv_seq_len,
                        row.slot
                    );
                }
                *entry += 1;
                tokens.push(SlotToken {
                    slot: row.slot,
                    token: row.last_token + 1,
                    logprob: None,
                    finish: None,
                });
            }

            Ok(MockInflight {
                output: StepOutput { tokens },
                return_not_ready_once: false,
            })
        }

        fn poll(&mut self, inflight: Self::Inflight) -> Result<PollResult<Self::Inflight>> {
            Ok(PollResult::Ready(inflight.output))
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub(super) struct HoldGovernor;

    impl ResourceGovernor for HoldGovernor {
        fn admission_gate(&self) -> AdmissionVerdict {
            AdmissionVerdict::Hold
        }

        fn step_budget(&self) -> StepBudget {
            StepBudget::UNBOUNDED
        }

        fn should_yield(&self) -> bool {
            false
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub(super) struct TokenBudgetGovernor {
        pub(super) max_tokens: usize,
    }

    impl ResourceGovernor for TokenBudgetGovernor {
        fn admission_gate(&self) -> AdmissionVerdict {
            AdmissionVerdict::Admit
        }

        fn step_budget(&self) -> StepBudget {
            StepBudget {
                max_tokens: self.max_tokens,
                max_micros: u64::MAX,
            }
        }

        fn should_yield(&self) -> bool {
            false
        }
    }

    /// Tier-store mock: forwards delegate to [`MockExecutor`]; demoted page
    /// payloads live in a host map with a capacity cap, drops are recorded.
    #[derive(Debug, Clone)]
    pub(super) struct TierMockExecutor {
        inner: MockExecutor,
        capacity: usize,
        pub(super) store: BTreeMap<u64, u32>,
        pub(super) dropped: Vec<u64>,
        pub(super) demote_batches: Vec<usize>,
        pub(super) promote_batches: Vec<usize>,
        fail_promotes: bool,
        pub(super) max_reuse_blocks: Option<usize>,
    }

    impl TierMockExecutor {
        pub(super) fn with_capacity(capacity: usize) -> Self {
            Self {
                inner: MockExecutor::ready(),
                capacity,
                store: BTreeMap::new(),
                dropped: Vec::new(),
                demote_batches: Vec::new(),
                promote_batches: Vec::new(),
                fail_promotes: false,
                max_reuse_blocks: None,
            }
        }

        pub(super) fn failing_promotes(capacity: usize) -> Self {
            Self {
                fail_promotes: true,
                ..Self::with_capacity(capacity)
            }
        }
    }

    impl BackendExecutor for TierMockExecutor {
        type Inflight = MockInflight;

        fn submit(&mut self, plan: &ForwardPlan, kv: &mut dyn KvPool) -> Result<Self::Inflight> {
            self.inner.submit(plan, kv)
        }

        fn poll(&mut self, inflight: Self::Inflight) -> Result<PollResult<Self::Inflight>> {
            self.inner.poll(inflight)
        }

        fn kv_tier_capacity_pages(&self) -> usize {
            self.capacity
        }

        fn kv_tier_page_bytes(&self) -> usize {
            16
        }

        fn kv_tier_host_demoted_pages(&self) -> usize {
            self.store.len()
        }

        fn kv_tier_location(&self, key: u64) -> Option<infer_seam::KvTierLocation> {
            self.store
                .contains_key(&key)
                .then_some(infer_seam::KvTierLocation::HostDemoted)
        }

        fn reusable_prefix_blocks(&self, blocks: &[PrefixBlock]) -> usize {
            let reusable =
                pages_only_reusable_prefix_blocks(blocks, |key| self.store.contains_key(&key));
            self.max_reuse_blocks
                .map_or(reusable, |max| reusable.min(max))
        }

        fn demote_prefix_pages(&mut self, entries: &[(u32, u64)]) -> Result<usize> {
            self.demote_batches.push(entries.len());
            let mut accepted = 0;
            for &(page, key) in entries {
                if self.store.len() >= self.capacity {
                    break;
                }
                self.store.insert(key, page);
                accepted += 1;
            }
            Ok(accepted)
        }

        fn promote_prefix_pages(&mut self, entries: &[(u64, u32)]) -> Result<()> {
            self.promote_batches.push(entries.len());
            if self.fail_promotes {
                bail!("mock promote failure");
            }
            for (key, _page) in entries {
                if !self.store.contains_key(key) {
                    bail!("promote of unknown tier key {key}");
                }
            }
            Ok(())
        }

        fn drop_kv_tier_entries(&mut self, keys: &[u64]) {
            for key in keys {
                self.store.remove(key);
                self.dropped.push(*key);
            }
        }
    }

    /// Whole-slot tier mock: records demoted complete restore images by key;
    /// forwards delegate to [`MockExecutor`].
    #[derive(Debug, Clone)]
    pub(super) struct SlotTierMockExecutor {
        inner: MockExecutor,
        pub(super) store: BTreeMap<u64, usize>,
        pub(super) dropped: Vec<u64>,
        reject_demotes: bool,
        fail_promotes: bool,
    }

    impl SlotTierMockExecutor {
        pub(super) fn enabled() -> Self {
            Self {
                inner: MockExecutor::ready(),
                store: BTreeMap::new(),
                dropped: Vec::new(),
                reject_demotes: false,
                fail_promotes: false,
            }
        }

        pub(super) fn rejecting_demotes() -> Self {
            Self {
                reject_demotes: true,
                ..Self::enabled()
            }
        }

        pub(super) fn failing_promotes() -> Self {
            Self {
                fail_promotes: true,
                ..Self::enabled()
            }
        }
    }

    impl BackendExecutor for SlotTierMockExecutor {
        type Inflight = MockInflight;

        fn submit(&mut self, plan: &ForwardPlan, kv: &mut dyn KvPool) -> Result<Self::Inflight> {
            self.inner.submit(plan, kv)
        }

        fn poll(&mut self, inflight: Self::Inflight) -> Result<PollResult<Self::Inflight>> {
            self.inner.poll(inflight)
        }

        fn kv_slot_tier_enabled(&self) -> bool {
            true
        }

        fn demote_slot(&mut self, slot: usize, key: u64) -> Result<bool> {
            if self.reject_demotes {
                return Ok(false);
            }
            self.store.insert(key, slot);
            Ok(true)
        }

        fn promote_slot(&mut self, key: u64, _slot: usize, _slot_pages: &[u32]) -> Result<()> {
            if self.fail_promotes {
                bail!("mock slot promote failure");
            }
            if !self.store.contains_key(&key) {
                bail!("promote of unknown slot key {key}");
            }
            Ok(())
        }

        fn drop_kv_slot_entries(&mut self, keys: &[u64]) {
            for key in keys {
                self.store.remove(key);
                self.dropped.push(*key);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use infer_plan::{
        DiffusionBlockModel, DiffusionCanvasPrediction, DiffusionGenerationConfig,
        DiffusionModelError, FinishReason, argmax_logit, sample_token,
    };
    use infer_seam::{BufferedDiffusionExecutor, HostPagedKvPool, KvAllocator, KvQuery};

    use super::testing::{
        DeviceMirrorExecutor, HoldGovernor, HybridReprefillMirror, LimitedPrefixExecutor,
        MockExecutor, MockKvPool, PrefixStoreMockExecutor, SamplingExecutor, SingleRowExecutor,
        SlotTierMockExecutor, SpecMirrorExecutor, StopTokenExecutor, TierMockExecutor,
        TokenBudgetGovernor, WarmupCountingExecutor,
    };
    use super::*;

    /// Drive `turns` agentic re-prefills on one slot: each turn's prompt is the
    /// prior turn's full text (prompt + the deterministic generated tokens) plus
    /// `tool_tokens` appended "tool output", so the page-radix prefix grows and
    /// every turn after the first re-prefills from `start_pos = matched_len`.
    /// Returns the executor probe for invariant inspection. The page size is 16,
    /// so a multi-page prompt forces a non-trivial block-aligned `matched_len`.
    fn run_agentic_reprefill(
        rewind_on_attach: bool,
        turns: usize,
        base_prompt: usize,
        gen_per_turn: usize,
        tool_tokens: usize,
    ) -> (HybridReprefillMirror, Result<()>) {
        let executor = HybridReprefillMirror::new(rewind_on_attach);
        let probe = executor.clone();
        let mut config = test_config(1);
        config.chunked_prefill_size = 64;
        config.enable_prefix_cache = true;
        // 16-token pages, ample capacity for a ~few-thousand-token conversation.
        let mut engine =
            Engine::with_config(executor, MockKvPool::with_capacity(1, 16, 4096), config);

        // Conversation token stream; each turn appends generated + tool tokens.
        let mut convo: Vec<u32> = (0..base_prompt as u32).map(|t| t + 1).collect();
        let result = (|| -> Result<()> {
            for turn in 0..turns {
                let handle = engine.submit_request(convo.clone(), gen_per_turn.max(1));
                engine.run_to_idle()?;
                let completed = engine
                    .completed(handle)
                    .ok_or_else(|| anyhow::anyhow!("turn {turn} did not complete"))?;
                // Grow the conversation: append this turn's generated tokens, then
                // the next "tool output" — the next turn re-prefills this superset.
                convo.extend_from_slice(&completed.generated_tokens);
                convo.extend((0..tool_tokens as u32).map(|t| 50_000 + turn as u32 * 100 + t));
            }
            Ok(())
        })();
        (probe, result)
    }

    /// The fix (`rewind_on_attach = true`): a multi-turn agentic re-prefill never
    /// drifts the decode counter — `materialized == kv_seq_len` at every decode
    /// across every turn boundary, even as the conversation grows past several
    /// pages (the ~4.5K-token shape that crashed on the pre-`1b0f0459` tree).
    #[test]
    fn agentic_reprefill_keeps_seq_len_in_lockstep() {
        let (probe, result) = run_agentic_reprefill(true, 6, 40, 5, 24);
        result.expect("agentic re-prefill must not drift seq_len with the rewind fix");
        let log = probe.decode_log();
        assert!(!log.is_empty(), "expected decode rows across the turns");
        for (k, &(slot, materialized, kv_seq_len)) in log.iter().enumerate() {
            assert_eq!(slot, 0);
            assert_eq!(
                materialized, kv_seq_len,
                "decode {k}: materialized {materialized} != kv_seq_len {kv_seq_len} \
                 (re-prefill seq_len drift)"
            );
        }
    }

    /// The drift the fix prevents: skipping the device-counter rewind on prefix
    /// attach reproduces the exact failure — the tail re-prefill (or the first
    /// decode) trips the readiness/invariant guard with the stale higher counter.
    /// This proves the lockstep test above is not vacuous.
    #[test]
    fn agentic_reprefill_without_rewind_fails_loud() {
        let (_probe, result) = run_agentic_reprefill(false, 6, 40, 5, 24);
        let err = result.expect_err(
            "without the device-counter rewind, the re-prefill MUST surface the drift, not pass",
        );
        let msg = err.to_string();
        assert!(
            msg.contains("start_pos") || msg.contains("kv_seq_len") || msg.contains("materialized"),
            "expected the seq_len drift guard, got: {msg}"
        );
    }

    fn submit_with_sampling(
        engine: &mut Engine<StopTokenExecutor, MockKvPool>,
        prompt: Vec<u32>,
        max_tokens: usize,
        sampling: SamplingParams,
    ) -> RequestHandle {
        engine.submit_request_with_options(
            prompt,
            max_tokens,
            RequestOptions {
                sampling,
                ..RequestOptions::default()
            },
        )
    }

    /// A request that supplies no stop tokens falls back to the model EOS:
    /// MockExecutor emits 11, 12, 13, ... for prompt [10]; model EOS 12 halts
    /// generation at the second token with a Stop reason, not Length.
    #[test]
    fn request_without_stops_falls_back_to_model_eos() -> Result<()> {
        let executor = StopTokenExecutor::with_model_stops(vec![12]);
        let mut engine = Engine::new(executor, MockKvPool::new(1), 1);
        let handle = engine.submit_request(vec![10], 8);

        engine.run_to_idle()?;

        let completed = engine.completed(handle).expect("request completed");
        assert_eq!(completed.generated_tokens, vec![11, 12]);
        assert!(matches!(completed.finish, Some(FinishReason::Stop)));
        Ok(())
    }

    /// Request-supplied stops take priority over the model EOS: stop id 11 halts
    /// before the model EOS 12 is ever produced.
    #[test]
    fn request_stops_take_priority_over_model_eos() -> Result<()> {
        let executor = StopTokenExecutor::with_model_stops(vec![12]);
        let mut engine = Engine::new(executor, MockKvPool::new(1), 1);
        let sampling = SamplingParams {
            stop_token_ids: vec![11],
            ..SamplingParams::default()
        };
        let handle = submit_with_sampling(&mut engine, vec![10], 8, sampling);

        engine.run_to_idle()?;

        let completed = engine.completed(handle).expect("request completed");
        assert_eq!(completed.generated_tokens, vec![11]);
        assert!(matches!(completed.finish, Some(FinishReason::Stop)));
        Ok(())
    }

    /// `ignore_eos` disables the model EOS fallback: the request runs to its
    /// length limit instead of stopping on the model stop token.
    #[test]
    fn ignore_eos_disables_model_eos_fallback() -> Result<()> {
        let executor = StopTokenExecutor::with_model_stops(vec![12]);
        let mut engine = Engine::new(executor, MockKvPool::new(1), 1);
        let sampling = SamplingParams {
            ignore_eos: true,
            ..SamplingParams::default()
        };
        let handle = submit_with_sampling(&mut engine, vec![10], 4, sampling);

        engine.run_to_idle()?;

        let completed = engine.completed(handle).expect("request completed");
        assert_eq!(completed.generated_tokens, vec![11, 12, 13, 14]);
        assert!(matches!(completed.finish, Some(FinishReason::Length)));
        Ok(())
    }

    /// An executor that exposes no model stops (the CUDA/mock default) keeps the
    /// prior behavior: a request with no stops runs to its length limit.
    #[test]
    fn no_model_stops_runs_to_length() -> Result<()> {
        let mut engine = Engine::new(MockExecutor::ready(), MockKvPool::new(1), 1);
        let handle = engine.submit_request(vec![10], 3);

        engine.run_to_idle()?;

        let completed = engine.completed(handle).expect("request completed");
        assert_eq!(completed.generated_tokens, vec![11, 12, 13]);
        assert!(matches!(completed.finish, Some(FinishReason::Length)));
        Ok(())
    }

    #[test]
    fn token_observer_streams_prefill_and_decode_tokens_before_completion() -> Result<()> {
        let executor = StopTokenExecutor::with_model_stops(vec![12]);
        let mut engine = Engine::new(executor, MockKvPool::new(1), 1);
        let observed = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let observed_sink = observed.clone();
        engine.set_token_observer(Box::new(move |handle, token| {
            observed_sink.borrow_mut().push((handle.id(), token.token));
        }));

        let handle = engine.submit_request(vec![10], 8);
        engine.run_to_idle()?;

        let expected = vec![(handle.id(), 11), (handle.id(), 12)];
        assert_eq!(
            *observed.borrow(),
            expected,
            "serving observer must see the prefill token and terminal decode token in commit order"
        );
        let completed = engine.completed(handle).expect("request completed");
        assert_eq!(completed.generated_tokens, vec![11, 12]);
        assert!(matches!(completed.finish, Some(FinishReason::Stop)));
        Ok(())
    }

    #[test]
    fn engine_warms_backend_exactly_once_across_steps() -> Result<()> {
        let executor = WarmupCountingExecutor::default();
        let warmup_calls = executor.warmup_calls.clone();
        let mut engine = Engine::new(executor, MockKvPool::new(1), 1);

        // No warmup until the first step runs.
        assert_eq!(warmup_calls.get(), 0);

        let handle = engine.submit_request(vec![10], 3);
        engine.run_to_idle()?;

        // The backend was warmed exactly once even though many steps ran.
        assert_eq!(warmup_calls.get(), 1);
        assert!(engine.completed(handle).is_some());

        // An explicit warmup after the lazy one is a no-op.
        engine.warmup()?;
        assert_eq!(warmup_calls.get(), 1);
        Ok(())
    }

    #[test]
    fn sampling_params_flow_through_prefill_and_decode_to_executor() -> Result<()> {
        let logits: Vec<f32> = (0..32).map(|i| i as f32 * 0.05).collect();
        let sampling = SamplingParams {
            temperature: 1.0,
            top_k: 8,
            top_p: 0.85,
            min_p: 0.0,
            seed: Some(0xC0FFEE),
            ..SamplingParams::default()
        };
        let executor = SamplingExecutor::new(logits.clone());
        let probe = executor.clone();
        let mut engine = Engine::new(executor, MockKvPool::new(1), 1);
        let prompt = vec![10, 11, 12];
        let max_tokens = 3;
        let handle = engine.submit_request_with_options(
            prompt.clone(),
            max_tokens,
            RequestOptions {
                sampling: sampling.clone(),
                ..RequestOptions::default()
            },
        );

        engine.run_to_idle()?;

        let expected: Vec<u32> = (0..max_tokens)
            .map(|step| sample_token(&logits, &sampling, (prompt.len() + step) as u64))
            .collect();
        assert_ne!(
            expected,
            vec![argmax_logit(&logits); max_tokens],
            "test fixture must exercise non-greedy sampling, not collapse to argmax"
        );
        let completed = engine.completed(handle).expect("request completed");
        assert_eq!(
            completed.generated_tokens, expected,
            "completed output must come from the request sampling params carried through the plan"
        );

        let observed = probe.observed();
        assert_eq!(observed.len(), max_tokens);
        for (step, (slot, position, params)) in observed.iter().enumerate() {
            assert_eq!(*slot, 0);
            assert_eq!(*position, (prompt.len() + step) as u64);
            assert_eq!(params.temperature, sampling.temperature);
            assert_eq!(params.top_k, sampling.top_k);
            assert_eq!(params.top_p, sampling.top_p);
            assert_eq!(params.seed, sampling.seed);
        }
        Ok(())
    }

    #[test]
    fn tier_demotes_on_eviction_and_promotes_on_prefix_hit() -> Result<()> {
        // 4 pages of 4 tokens; each 8-token prompt + 1 generated token needs 3.
        let mut engine = Engine::with_config(
            TierMockExecutor::with_capacity(8),
            MockKvPool::with_capacity(1, 4, 4),
            test_config(1),
        );

        let first = engine.submit_request((1..=8).collect(), 1);
        engine.run_to_idle()?;
        assert_finished(engine.completed(first).expect("first completed"));
        assert_eq!(engine.prefix_cache_stats().published_pages, 2);

        // A disjoint prompt cannot fit without evicting; with a tier store the
        // evicted block demotes instead of dropping.
        let second = engine.submit_request((50..=57).collect(), 1);
        engine.run_to_idle()?;
        assert_finished(engine.completed(second).expect("second completed"));
        let tier = engine.kv_tier_stats();
        assert!(
            tier.demoted_pages >= 1,
            "eviction demoted into the store: {tier:?}"
        );
        assert!(!engine.executor.store.is_empty());

        // Re-running the first prompt promotes the demoted block back instead
        // of re-prefilling it.
        let third = engine.submit_request((1..=8).collect(), 1);
        engine.run_to_idle()?;
        assert_finished(engine.completed(third).expect("third completed"));
        let tier = engine.kv_tier_stats();
        assert!(
            tier.promoted_pages >= 1,
            "prefix hit promoted from the store: {tier:?}"
        );
        assert_eq!(tier.promote_failures, 0);
        assert!(
            engine.prefix_cache_stats().hits >= 1,
            "tiered match counts as a prefix hit"
        );
        // Promoted entries were dropped from the store by the key drain.
        assert!(
            engine.executor.dropped.len() as u64 >= tier.promoted_pages,
            "store entries dropped after promote: {:?}",
            engine.executor.dropped
        );
        Ok(())
    }

    #[test]
    fn position0_prefix_store_captures_then_restores_leading_prefix() -> Result<()> {
        // DSv4-shaped reuse: page-radix is always empty, so reuse rides the
        // executor's position-0 whole-slot prefix store. 8 pages of 4 tokens.
        let mut engine = Engine::with_config(
            PrefixStoreMockExecutor::new(),
            MockKvPool::with_capacity(2, 4, 16),
            test_config(2),
        );
        let captured = engine.executor.captured.clone();
        let restored = engine.executor.restored.clone();

        // First request: an 8-token prompt prefills from position 0 and on
        // prefill completion the engine captures it into the position-0 store.
        let first = engine.submit_request((1..=8).collect(), 2);
        engine.run_to_idle()?;
        assert_finished(engine.completed(first).expect("first completed"));
        // Two captures per request: prefill (prompt only) + finish-slot sidecar
        // (prompt + generated echo tokens). Echo rule: last_prompt+1 = 9, then
        // last_decode+1 = 10 for max_new_tokens=2.
        assert_eq!(
            captured.borrow().as_slice(),
            &[
                vec![1, 2, 3, 4, 5, 6, 7, 8],
                vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            ],
            "prefill + finish-sidecar captures the position-0 prefix"
        );
        assert!(
            restored.borrow().is_empty(),
            "no restore on the first (cold) request"
        );

        // Second request: leading 10 tokens match the sidecar-captured sequence
        // [1..10] (prefill captured [1..8]; finish-sidecar extended to [1..10]).
        let before_hits = engine.kv_system_metrics().reuse_hit_resident;
        let second = engine.submit_request((1..=12).collect(), 2);
        engine.run_to_idle()?;
        assert_finished(engine.completed(second).expect("second completed"));
        let restores = restored.borrow();
        assert_eq!(restores.len(), 1, "exactly one cached-prefix restore fired");
        assert_eq!(
            restores[0].1, 10,
            "restored the sidecar-captured 10-token leading prefix"
        );
        assert!(
            engine.kv_system_metrics().reuse_hit_resident > before_hits,
            "DSv4 cached-prefix reuse bumps reuse_hit_resident (closes the WS2 counter gap)"
        );
        assert!(
            engine.prefix_cache_stats().hits >= 1,
            "cached-prefix reuse counts as a prefix hit"
        );
        Ok(())
    }

    #[test]
    fn position0_prefix_store_no_match_reprefills_whole_prompt() -> Result<()> {
        let mut engine = Engine::with_config(
            PrefixStoreMockExecutor::new(),
            MockKvPool::with_capacity(2, 4, 16),
            test_config(2),
        );
        let restored = engine.executor.restored.clone();

        let first = engine.submit_request((1..=8).collect(), 1);
        engine.run_to_idle()?;
        assert_finished(engine.completed(first).expect("first completed"));

        // A prompt that shares no leading prefix never matches the store.
        let second = engine.submit_request((50..=57).collect(), 1);
        engine.run_to_idle()?;
        assert_finished(engine.completed(second).expect("second completed"));
        assert!(
            restored.borrow().is_empty(),
            "divergent prompt drove no cached-prefix restore"
        );
        Ok(())
    }

    #[test]
    fn tier_promote_failure_truncates_match_and_reprefills() -> Result<()> {
        let mut engine = Engine::with_config(
            TierMockExecutor::failing_promotes(8),
            MockKvPool::with_capacity(1, 4, 4),
            test_config(1),
        );

        let first = engine.submit_request((1..=8).collect(), 1);
        engine.run_to_idle()?;
        assert_finished(engine.completed(first).expect("first completed"));

        let second = engine.submit_request((50..=57).collect(), 1);
        engine.run_to_idle()?;
        assert_finished(engine.completed(second).expect("second completed"));
        assert!(engine.kv_tier_stats().demoted_pages >= 1);

        // Promotion fails -> demoted entry is severed, the tail re-prefills,
        // and the request still completes.
        let third = engine.submit_request((1..=8).collect(), 1);
        engine.run_to_idle()?;
        assert_finished(engine.completed(third).expect("third completed"));
        let tier = engine.kv_tier_stats();
        assert!(tier.promote_failures >= 1, "failure counted: {tier:?}");
        assert_eq!(tier.promoted_pages, 0);
        assert!(
            engine.kv_system_metrics().fallback_recompute >= 1,
            "failed promote should count recompute fallback"
        );
        assert!(
            engine.executor.dropped.contains(&0),
            "failed tier entry dropped: {:?}",
            engine.executor.dropped
        );
        assert!(
            !engine.executor.store.contains_key(&0),
            "failed tier entry must not remain in the store: {:?}",
            engine.executor.store
        );
        Ok(())
    }

    #[test]
    fn tier_lookup_checks_restore_boundary_before_promote() -> Result<()> {
        let mut engine = Engine::with_config(
            TierMockExecutor::with_capacity(8),
            MockKvPool::with_capacity(1, 4, 4),
            test_config(1),
        );

        let first = engine.submit_request((1..=8).collect(), 1);
        engine.run_to_idle()?;
        assert_finished(engine.completed(first).expect("first completed"));
        assert_eq!(engine.prefix_cache_stats().published_pages, 2);
        engine.executor.max_reuse_blocks = Some(1);

        let reclaimed = engine.evict_prefix_cache_for_pages(2);
        assert_eq!(reclaimed, 2);
        assert_eq!(engine.kv_tier_stats().demoted_pages, 2);
        assert_eq!(engine.executor.store.len(), 2);

        let second = engine.submit_request((1..=8).collect(), 1);
        engine.step()?;

        let request = engine
            .active
            .values()
            .find(|request| request.handle == second)
            .expect("second admitted");
        assert_eq!(request.prefill_start_pos, 4);
        assert_eq!(request.reused_prefix_pages.len(), 1);
        assert_eq!(
            engine.kv_tier_stats().promoted_pages,
            1,
            "only the mcheck-approved leading block should be promoted"
        );
        assert_eq!(
            engine.executor.store.len(),
            1,
            "unusable demoted tail stays in the store, not promoted then discarded"
        );

        engine.run_to_idle()?;
        assert_finished(engine.completed(second).expect("second completed"));
        Ok(())
    }

    #[test]
    fn preemption_with_tier_swaps_prompt_and_promotes_on_readmission() -> Result<()> {
        let mut engine = Engine::with_config(
            TierMockExecutor::with_capacity(8),
            MockKvPool::with_capacity(2, 4, 16),
            test_config(2),
        );
        let handle = engine.submit_request((1..=8).collect(), 4);
        engine.step()?;
        engine.step()?;
        let (&slot, request) = engine.active.iter().next().expect("request active");
        assert!(matches!(request.phase, RequestPhase::Decoding));

        engine.requeue_preempted_decode(slot);
        assert!(engine.active.is_empty(), "victim re-queued");
        let tier = engine.kv_tier_stats();
        assert_eq!(
            tier.demoted_pages, 2,
            "both sealed prompt blocks swapped to the host tier: {tier:?}"
        );
        assert!(
            engine.executor.demote_batches.iter().any(|&n| n == 2),
            "prompt pages should demote through one mset batch: {:?}",
            engine.executor.demote_batches
        );

        // Re-admission promotes the prompt back instead of re-prefilling.
        engine.run_to_idle()?;
        assert_finished(engine.completed(handle).expect("completed"));
        let tier = engine.kv_tier_stats();
        assert!(tier.promoted_pages >= 2, "prompt promoted back: {tier:?}");
        let kv_system = engine.kv_system_metrics();
        assert!(kv_system.demote_mset_count >= 1, "{kv_system:?}");
        assert!(kv_system.demote_mset_copy_bytes >= 32, "{kv_system:?}");
        assert!(kv_system.promote_mget_count >= 1, "{kv_system:?}");
        assert!(kv_system.promote_mget_copy_bytes >= 32, "{kv_system:?}");
        assert!(kv_system.reuse_hit_host_demoted >= 1, "{kv_system:?}");
        assert!(
            engine.executor.promote_batches.iter().any(|&n| n == 2),
            "prompt pages should promote through one mget batch: {:?}",
            engine.executor.promote_batches
        );
        assert!(engine.prefix_cache_stats().hits >= 1);
        Ok(())
    }

    #[test]
    fn slot_swap_resumes_decode_with_generation_intact() -> Result<()> {
        let mut engine = Engine::with_config(
            SlotTierMockExecutor::enabled(),
            MockKvPool::with_capacity(2, 4, 32),
            test_config(2),
        );
        let handle = engine.submit_request((1..=8).collect(), 4);
        engine.step()?;
        engine.step()?;
        let (&slot, request) = engine.active.iter().next().expect("request active");
        assert!(matches!(request.phase, RequestPhase::Decoding));
        assert_eq!(request.generated_tokens, vec![9]);
        // Between steps the engine invariant is host seq_len == device
        // seq_len == the next plan's kv_seq_len; capture it for the
        // restore-exactness assertion below.
        let demoted_seq_len = engine.kv.seq_len(slot);
        assert_eq!(demoted_seq_len, 9, "prompt 8 + the committed token's row");

        engine.requeue_preempted_decode(slot);
        let tier = engine.kv_tier_stats();
        assert_eq!(tier.demoted_slots, 1, "whole slot swapped out: {tier:?}");
        assert_eq!(engine.executor.store.len(), 1);

        // One tick: re-admission promotes (restoring EXACTLY the demoted
        // length) and the resumed decode step appends its one token — host
        // accounting must line up with the restored device image, not run
        // one ahead (codex P2 on 6ec31f53).
        engine.step()?;
        let (&new_slot, _) = engine.active.iter().next().expect("resumed");
        assert_eq!(
            engine.kv.seq_len(new_slot),
            demoted_seq_len + 1,
            "restored materialized length + the resumed step's single append"
        );

        engine.run_to_idle()?;
        let done = engine.completed(handle).expect("completed");
        assert_finished(done);
        assert_eq!(done.generated_tokens, vec![9, 10, 11, 12]);
        let tier = engine.kv_tier_stats();
        assert_eq!(tier.promoted_slots, 1, "decode resumed via promote");
        assert_eq!(tier.slot_promote_failures, 0);
        assert_eq!(engine.executor.dropped.len(), 1, "store entry dropped");
        // The discriminator: the prompt prefilled exactly once — a recompute
        // fallback would have prefilled it twice.
        assert_eq!(engine.throughput_stats().prefill_tokens, 8);
        Ok(())
    }

    /// P5 slot-pressure trigger: with `slot_oversubscription` on and a single
    /// physical slot, admitting a second request parks the running decode's
    /// whole-slot image (demote) to free the slot; the waiter runs, then the
    /// parked request resumes via promote with its generation intact.
    #[test]
    fn oversubscription_parks_running_decode_to_admit_waiter() -> Result<()> {
        let config = SchedulerConfig {
            slot_oversubscription: true,
            ..test_config(1)
        };
        let mut engine = Engine::with_config(
            SlotTierMockExecutor::enabled(),
            MockKvPool::with_capacity(1, 4, 64),
            config,
        );

        // Request A occupies the only slot and decodes past the min slice so it
        // is an eligible park victim (a just-resumed request would not be).
        let a = engine.submit_request((1..=8).collect(), 20);
        for _ in 0..(OVERSUBSCRIPTION_MIN_SLICE + 3) {
            engine.step()?;
        }
        let (&slot, request) = engine.active.iter().next().expect("A active");
        assert_eq!(slot, 0);
        assert!(matches!(request.phase, RequestPhase::Decoding));
        let a_gen_at_arrival = request.generated_tokens.len();
        assert!(
            a_gen_at_arrival >= OVERSUBSCRIPTION_MIN_SLICE,
            "A decoded past the min slice: {a_gen_at_arrival}"
        );
        // A's monotonic echo generation is exactly [9, 10, ...].
        let a_gen_prefix: Vec<u32> = (9..9 + a_gen_at_arrival as u32).collect();
        assert_eq!(request.generated_tokens, a_gen_prefix);

        // Request B arrives; no free slot, so the next admit oversubscribes:
        // A (the only eligible Decoding request) is parked.
        let b = engine.submit_request((20..=24).collect(), 3);
        engine.step()?;
        let tier = engine.kv_tier_stats();
        assert_eq!(tier.demoted_slots, 1, "A parked to free the slot: {tier:?}");
        let parked = engine
            .waiting
            .iter()
            .find(|r| r.handle == a)
            .expect("A parked");
        // Generation preserved (not reset to recompute) and the whole-slot key
        // is carried so re-admission promotes instead of re-prefilling.
        assert!(
            parked.generated_tokens.starts_with(&a_gen_prefix),
            "generation preserved, not reset: {:?}",
            parked.generated_tokens
        );
        assert!(parked.swap_key.is_some(), "A carries its whole-slot key");
        // B holds the freed slot — A genuinely yielded it.
        assert!(engine.active.values().any(|r| r.handle == b), "B admitted");

        // B runs to completion on the freed slot, then A resumes via promote
        // and runs to its length cap. B's short run (< min slice) never makes B
        // a victim, so there is no park ping-pong: exactly one demote/promote.
        engine.run_to_idle()?;
        let done_b = engine.completed(b).expect("B completed");
        assert_finished(done_b);
        assert_eq!(done_b.generated_tokens, vec![25, 26, 27]);
        let done_a = engine.completed(a).expect("A completed");
        assert_finished(done_a);
        // A resumed (promote), did NOT restart: full monotonic [9..=28], 20 tokens.
        assert_eq!(done_a.generated_tokens, (9..=28).collect::<Vec<u32>>());
        let tier = engine.kv_tier_stats();
        assert_eq!(
            tier.demoted_slots, 1,
            "exactly one park (no thrash): {tier:?}"
        );
        assert_eq!(tier.promoted_slots, 1, "A resumed via promote: {tier:?}");
        assert_eq!(tier.slot_promote_failures, 0);
        // A's prompt prefilled exactly once (8) + B's prompt (5): a recompute
        // fallback for A would have re-prefilled its 8.
        assert_eq!(engine.throughput_stats().prefill_tokens, 13);
        Ok(())
    }

    /// Default-off byte-identity guard: with `slot_oversubscription` false the
    /// trigger never fires — a second request simply waits for the slot, no
    /// demote, identical to before this knob.
    #[test]
    fn oversubscription_off_never_parks() -> Result<()> {
        let mut engine = Engine::with_config(
            SlotTierMockExecutor::enabled(),
            MockKvPool::with_capacity(1, 4, 64),
            test_config(1), // slot_oversubscription defaults false
        );
        let a = engine.submit_request((1..=8).collect(), 4);
        engine.step()?;
        engine.step()?;
        let b = engine.submit_request((20..=24).collect(), 3);
        engine.run_to_idle()?;

        assert_eq!(engine.kv_tier_stats().demoted_slots, 0, "no park when off");
        let done_a = engine.completed(a).expect("A completed");
        assert_eq!(done_a.generated_tokens, vec![9, 10, 11, 12]);
        let done_b = engine.completed(b).expect("B completed");
        assert_eq!(done_b.generated_tokens, vec![25, 26, 27]);
        Ok(())
    }

    #[test]
    fn slot_promote_failure_falls_back_to_recompute() -> Result<()> {
        let mut engine = Engine::with_config(
            SlotTierMockExecutor::failing_promotes(),
            MockKvPool::with_capacity(2, 4, 32),
            test_config(2),
        );
        let handle = engine.submit_request((1..=8).collect(), 4);
        engine.step()?;
        engine.step()?;
        let (&slot, _) = engine.active.iter().next().expect("request active");

        engine.requeue_preempted_decode(slot);
        engine.run_to_idle()?;
        let done = engine.completed(handle).expect("completed");
        assert_finished(done);
        assert_eq!(done.generated_tokens, vec![9, 10, 11, 12]);
        let tier = engine.kv_tier_stats();
        assert_eq!(tier.slot_promote_failures, 1, "{tier:?}");
        assert_eq!(tier.promoted_slots, 0);
        assert_eq!(engine.executor.dropped.len(), 1, "failed entry dropped");
        assert_eq!(
            engine.throughput_stats().prefill_tokens,
            16,
            "fallback re-prefilled the prompt"
        );
        Ok(())
    }

    #[test]
    fn slot_store_rejection_recomputes_without_counters() -> Result<()> {
        let mut engine = Engine::with_config(
            SlotTierMockExecutor::rejecting_demotes(),
            MockKvPool::with_capacity(2, 4, 32),
            test_config(2),
        );
        let handle = engine.submit_request((1..=8).collect(), 4);
        engine.step()?;
        engine.step()?;
        let (&slot, _) = engine.active.iter().next().expect("request active");

        engine.requeue_preempted_decode(slot);
        engine.run_to_idle()?;
        let done = engine.completed(handle).expect("completed");
        assert_finished(done);
        assert_eq!(done.generated_tokens, vec![9, 10, 11, 12]);
        let tier = engine.kv_tier_stats();
        assert_eq!(tier.demoted_slots, 0);
        assert_eq!(
            engine.throughput_stats().prefill_tokens,
            16,
            "plain recompute path"
        );
        Ok(())
    }

    #[test]
    fn preemption_without_tier_is_unchanged() -> Result<()> {
        let mut engine = Engine::with_config(
            MockExecutor::ready(),
            MockKvPool::with_capacity(2, 4, 16),
            test_config(2),
        );
        let free_before = engine.kv_free_pages();
        let handle = engine.submit_request((1..=8).collect(), 4);
        engine.step()?;
        engine.step()?;
        let (&slot, _) = engine.active.iter().next().expect("request active");

        engine.requeue_preempted_decode(slot);
        assert_eq!(
            engine.radix.cached_page_count(),
            0,
            "no publish on the plain recompute path"
        );
        assert_eq!(engine.kv_free_pages(), free_before, "all pages freed");

        engine.run_to_idle()?;
        assert_finished(engine.completed(handle).expect("completed"));
        Ok(())
    }

    #[test]
    fn tier_full_rotates_out_the_coldest_entry() -> Result<()> {
        let mut engine = Engine::with_config(
            TierMockExecutor::with_capacity(1),
            MockKvPool::with_capacity(1, 4, 8),
            test_config(1),
        );

        let first = engine.submit_request((1..=8).collect(), 1);
        engine.run_to_idle()?;
        assert_finished(engine.completed(first).expect("first completed"));
        assert_eq!(engine.prefix_cache_stats().published_pages, 2);

        // Force-demote both published blocks through a capacity-1 store: the
        // second demotion must rotate the first entry out (LRU), not fail.
        let reclaimed = engine.evict_prefix_cache_for_pages(2);
        assert_eq!(reclaimed, 2);
        let tier = engine.kv_tier_stats();
        assert_eq!(tier.demoted_pages, 2);
        assert_eq!(tier.resident_blocks, 1, "capacity-1 store holds one block");
        assert_eq!(engine.executor.store.len(), 1);
        assert_eq!(
            engine.executor.dropped.len(),
            1,
            "coldest entry rotated out"
        );
        Ok(())
    }

    fn test_config(num_slots: usize) -> SchedulerConfig {
        SchedulerConfig {
            num_slots,
            max_prompt_tokens: 128,
            max_total_tokens: 512,
            ..SchedulerConfig::default()
        }
    }

    fn assert_finished(completed: &CompletedRequest) {
        assert!(matches!(
            completed.finish,
            Some(FinishReason::Length | FinishReason::Stop | FinishReason::Abort)
        ));
    }

    #[test]
    fn test_single_request_decodes_to_max_tokens() -> Result<()> {
        let mut engine = Engine::new(MockExecutor::ready(), MockKvPool::new(1), 1);
        let handle = engine.submit_request(vec![10], 3);

        engine.run_to_idle()?;

        let completed = engine.completed(handle).expect("request completed");
        assert_eq!(completed.generated_tokens, vec![11, 12, 13]);
        assert!(matches!(completed.finish, Some(FinishReason::Length)));
        assert!(engine.is_idle());
        Ok(())
    }

    /// Decode-path off-by-one regression: at decode step k the materialized device
    /// length == that row's `kv_seq_len` == `prompt_len + k` (the prefill forward
    /// emits the first token).
    fn assert_decode_kv_seq_len_matches_materialized(prompt_len: usize, max_new: usize) {
        // Single-chunk prefill: the whole prompt lands in one prefill row.
        assert_decode_kv_seq_len_with_chunk(prompt_len, max_new, prompt_len.max(1));
    }

    fn assert_decode_kv_seq_len_with_chunk(prompt_len: usize, max_new: usize, chunk: usize) {
        let executor = DeviceMirrorExecutor::default();
        let probe = executor.clone();
        let mut config = test_config(1);
        config.chunked_prefill_size = chunk.max(1);
        let mut engine = Engine::with_config(executor, MockKvPool::new(1), config);

        let prompt: Vec<u32> = (0..prompt_len as u32).map(|t| t + 1).collect();
        let handle = engine.submit_request(prompt, max_new);
        engine
            .run_to_idle()
            .unwrap_or_else(|e| panic!("kv_seq_len drift for prompt_len={prompt_len}: {e}"));

        let completed = engine.completed(handle).expect("request completed");
        assert_eq!(
            completed.generated_tokens.len(),
            max_new,
            "expected {max_new} generated tokens for prompt_len={prompt_len}"
        );
        // The prefill forward emits the first token, so the number of *decode*
        // forwards is max_new - 1, and the device KV ends at prompt_len + decodes.
        let decode_steps = max_new.saturating_sub(1);
        let log = probe.decode_log();
        assert_eq!(
            log.len(),
            decode_steps,
            "expected {decode_steps} decode rows for prompt_len={prompt_len}, got {log:?}"
        );
        for (k, &(slot, materialized, kv_seq_len)) in log.iter().enumerate() {
            assert_eq!(slot, 0);
            assert_eq!(
                materialized, kv_seq_len,
                "decode step {k}: device materialized {materialized} != kv_seq_len {kv_seq_len}"
            );
            assert_eq!(
                kv_seq_len,
                prompt_len + k,
                "decode step {k}: kv_seq_len {kv_seq_len} != prompt_len+{k} ({})",
                prompt_len + k
            );
        }
        assert_eq!(probe.materialized(0), prompt_len + decode_steps);
    }

    #[test]
    fn decode_kv_seq_len_matches_materialized_one_token_prompt() {
        // The exact H20 repro shape: 1-token prompt, several decode steps.
        assert_decode_kv_seq_len_matches_materialized(1, 4);
    }

    #[test]
    fn decode_kv_seq_len_matches_materialized_multi_token_prompt() {
        assert_decode_kv_seq_len_matches_materialized(8, 5);
        assert_decode_kv_seq_len_matches_materialized(16, 3);
    }

    #[test]
    fn decode_kv_seq_len_advances_by_speculative_output_count() {
        let executor = SpecMirrorExecutor::default();
        let probe = executor.clone();
        let prompt_len = 8usize;
        let max_new = 5usize;
        let mut engine = Engine::with_config(executor, MockKvPool::new(1), test_config(1));
        let prompt: Vec<u32> = (0..prompt_len as u32).map(|t| t + 1).collect();

        let handle = engine.submit_request(prompt, max_new);
        engine
            .run_to_idle()
            .unwrap_or_else(|e| panic!("speculative kv_seq_len drift: {e}"));

        let completed = engine.completed(handle).expect("request completed");
        assert_eq!(completed.generated_tokens, vec![9, 10, 11, 12, 13]);
        assert!(matches!(completed.finish, Some(FinishReason::Length)));
        assert_eq!(
            probe.decode_log(),
            vec![
                (0, prompt_len, prompt_len),
                (0, prompt_len + 2, prompt_len + 2)
            ]
        );
        assert_eq!(probe.materialized(0), prompt_len + max_new - 1);
    }

    /// Chunked prefill (prompt split across ticks) must still hand the first
    /// decode the correct `kv_seq_len`: the device materializes the prompt one
    /// chunk per tick, so the prefill→decode boundary is the off-by-one's most
    /// likely hiding spot.
    #[test]
    fn decode_kv_seq_len_matches_materialized_chunked_prefill() {
        assert_decode_kv_seq_len_with_chunk(8, 4, 3);
        assert_decode_kv_seq_len_with_chunk(10, 3, 4);
    }

    /// Prefix reuse on the single-row executor must either run cleanly or fail
    /// loud at the prefill-readiness guard (device `materialized == start_pos`),
    /// never silently drift the decode counter.
    #[test]
    fn prefix_reuse_either_decodes_cleanly_or_fails_loud_at_prefill() {
        let executor = DeviceMirrorExecutor::default();
        let mut config = test_config(2);
        config.chunked_prefill_size = 64;
        let mut engine = Engine::with_config(executor, MockKvPool::new(2), config);

        // Prime a prefix, then submit a request that shares it.
        let first = engine.submit_request(vec![1, 2, 3, 4], 2);
        engine.run_to_idle().expect("primer drains");
        let second = engine.submit_request(vec![1, 2, 3, 4, 5, 6], 2);

        // The R6 executor never silently inherits a device/host counter gap: it
        // either runs the request to completion (prefix not reused on the device
        // path) or surfaces the explicit prefill-readiness guard.
        match engine.run_to_idle() {
            Ok(()) => {
                assert!(engine.completed(first).is_some());
                assert!(engine.completed(second).is_some());
            }
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    msg.contains("start_pos") || msg.contains("materialized"),
                    "expected explicit prefill-readiness guard, got: {msg}"
                );
            }
        }
    }

    /// Slot reuse: a second request lands on slot 0 after the first frees it.
    /// The device pool must be reset on the fresh prefill (start_pos == 0) so the
    /// second request's decode kv_seq_len does not inherit the first's count.
    #[test]
    fn decode_kv_seq_len_matches_materialized_across_slot_reuse() {
        let executor = DeviceMirrorExecutor::default();
        let probe = executor.clone();
        let mut config = test_config(1);
        config.chunked_prefill_size = 64;
        let mut engine = Engine::with_config(executor, MockKvPool::new(1), config);

        let first = engine.submit_request(vec![1, 2, 3], 3);
        engine.run_to_idle().expect("first request drains");
        let second = engine.submit_request(vec![7, 7, 7, 7], 3);
        engine.run_to_idle().expect("second request drains");

        assert_eq!(
            engine
                .completed(first)
                .expect("first done")
                .generated_tokens
                .len(),
            3
        );
        assert_eq!(
            engine
                .completed(second)
                .expect("second done")
                .generated_tokens
                .len(),
            3
        );
        // Every recorded decode kept materialized == kv_seq_len (no drift / no
        // cross-request inheritance).
        for (k, &(_, materialized, kv_seq_len)) in probe.decode_log().iter().enumerate() {
            assert_eq!(
                materialized, kv_seq_len,
                "decode {k}: materialized {materialized} != kv_seq_len {kv_seq_len}"
            );
        }
    }

    #[test]
    fn test_two_requests_interleave() -> Result<()> {
        let mut engine = Engine::new(MockExecutor::ready(), MockKvPool::new(2), 2);
        let left = engine.submit_request(vec![10], 3);
        let right = engine.submit_request(vec![100], 2);

        engine.run_to_idle()?;

        let left = engine.completed(left).expect("left request completed");
        let right = engine.completed(right).expect("right request completed");
        assert_eq!(left.generated_tokens, vec![11, 12, 13]);
        assert_eq!(right.generated_tokens, vec![101, 102]);
        assert!(matches!(left.finish, Some(FinishReason::Length)));
        assert!(matches!(right.finish, Some(FinishReason::Length)));
        Ok(())
    }

    #[test]
    fn test_overlap_not_ready_then_ready() -> Result<()> {
        let mut engine = Engine::new(MockExecutor::not_ready_once(), MockKvPool::new(1), 1);
        let handle = engine.submit_request(vec![5], 2);

        engine.step()?;
        assert!(engine.has_inflight());
        assert_eq!(engine.active_count(), 1);

        engine.step()?;
        assert!(engine.has_inflight());
        assert!(engine.completed(handle).is_none());

        engine.run_to_idle()?;

        let completed = engine.completed(handle).expect("request completed");
        assert_eq!(completed.generated_tokens, vec![6, 7]);
        assert!(matches!(completed.finish, Some(FinishReason::Length)));
        Ok(())
    }

    #[test]
    fn test_idle_plan_when_no_work() -> Result<()> {
        let mut engine = Engine::new(MockExecutor::ready(), MockKvPool::new(1), 1);

        engine.step()?;

        assert!(engine.is_idle());
        assert!(!engine.has_inflight());
        assert_eq!(engine.active_count(), 0);
        assert_eq!(engine.waiting_count(), 0);
        Ok(())
    }

    #[test]
    fn admit_respects_priority() -> Result<()> {
        let mut engine =
            Engine::with_config(MockExecutor::ready(), MockKvPool::new(1), test_config(1));
        let low = engine.submit_request_with_priority(vec![10], 1, RequestPriority::Low);
        let high = engine.submit_request_with_priority(vec![100], 1, RequestPriority::High);

        engine.step()?;

        assert_eq!(engine.active_count(), 1);
        assert_eq!(engine.waiting_count(), 1);
        assert!(engine.active.values().any(|request| request.handle == high));
        assert!(engine.waiting.iter().any(|request| request.handle == low));
        Ok(())
    }

    #[test]
    fn admission_holds_when_pool_full_then_admits_after_free() -> Result<()> {
        let config = SchedulerConfig {
            num_slots: 2,
            max_prompt_tokens: 64,
            max_total_tokens: 128,
            ..SchedulerConfig::default()
        };
        let mut engine = Engine::with_config(
            MockExecutor::ready(),
            MockKvPool::with_capacity(2, 16, 2),
            config,
        );
        let first = engine.submit_request(vec![1; 16], 1);
        let second = engine.submit_request(vec![2; 16], 1);

        engine.step()?;
        assert_eq!(engine.active_count(), 1);
        assert_eq!(engine.waiting_count(), 1);
        assert!(
            engine
                .active
                .values()
                .any(|request| request.handle == first)
        );

        engine.step()?;
        assert_eq!(engine.active_count(), 1);
        assert_eq!(engine.waiting_count(), 0);
        assert!(
            engine
                .active
                .values()
                .any(|request| request.handle == second)
        );
        assert!(engine.completed(first).is_some());

        engine.run_to_idle()?;
        assert_finished(engine.completed(second).expect("second completed"));
        Ok(())
    }

    #[test]
    fn prompt_too_long_is_rejected() -> Result<()> {
        let config = SchedulerConfig {
            num_slots: 1,
            max_prompt_tokens: 2,
            max_total_tokens: 16,
            ..SchedulerConfig::default()
        };
        let mut engine = Engine::with_config(MockExecutor::ready(), MockKvPool::new(1), config);

        let handle = engine.submit_request(vec![1, 2, 3], 4);
        engine.step()?;

        assert_eq!(engine.active_count(), 0);
        assert_eq!(engine.waiting_count(), 0);
        let completed = engine.completed(handle).expect("request rejected");
        assert!(matches!(completed.finish, Some(FinishReason::Abort)));
        assert!(completed.generated_tokens.is_empty());
        Ok(())
    }

    #[test]
    fn preempt_requeues_least_progressed() -> Result<()> {
        let mut engine = Engine::with_config(
            MockExecutor::ready(),
            MockKvPool::with_capacity(3, 1, 8),
            test_config(3),
        );
        let longer = engine.next_handle();
        let shorter = engine.next_handle();

        let mut long_req = RequestState::new(
            longer,
            vec![1, 1, 1, 1, 1],
            RequestPriority::Normal,
            8,
            SamplingParams::default(),
        );
        long_req.generated_tokens = vec![9, 10];
        long_req.phase = RequestPhase::Decoding;
        let mut short_req = RequestState::new(
            shorter,
            vec![2, 2],
            RequestPriority::Normal,
            8,
            SamplingParams::default(),
        );
        short_req.generated_tokens = vec![7, 8];
        short_req.phase = RequestPhase::Decoding;

        engine.kv.alloc(0, 5)?;
        engine.kv.alloc(1, 2)?;
        engine.active.insert(0, long_req);
        engine.active.insert(1, short_req);

        engine.step()?;

        assert_eq!(engine.active_count(), 1);
        assert_eq!(engine.waiting_count(), 1);
        assert!(
            engine
                .active
                .values()
                .any(|request| request.handle == shorter)
        );
        let requeued = engine.waiting.front().expect("victim requeued");
        assert_eq!(requeued.handle, longer);
        assert!(requeued.generated_tokens.is_empty());
        assert!(matches!(
            requeued.phase,
            RequestPhase::Prefilling { progress: 0 }
        ));
        Ok(())
    }

    #[test]
    fn finish_frees_slot_and_pages() -> Result<()> {
        let mut engine = Engine::with_config(
            MockExecutor::ready(),
            MockKvPool::with_capacity(1, 16, 2),
            test_config(1),
        );
        let handle = engine.submit_request(vec![3; 15], 1);
        let initial_free = engine.kv_free_pages();

        engine.step()?;
        assert!(engine.kv_free_pages() < initial_free);

        engine.step()?;
        assert_eq!(engine.kv_free_pages(), initial_free);
        assert_finished(engine.completed(handle).expect("request completed"));
        Ok(())
    }

    #[test]
    fn admission_respects_governor_hold() -> Result<()> {
        let mut engine = Engine::with_config_and_governor(
            MockExecutor::ready(),
            MockKvPool::new(1),
            test_config(1),
            Box::new(HoldGovernor),
        );
        engine.submit_request(vec![1, 2, 3], 1);

        engine.step()?;

        assert_eq!(engine.active_count(), 0);
        assert_eq!(engine.waiting_count(), 1);
        assert!(!engine.has_inflight());
        Ok(())
    }

    #[test]
    fn backend_row_cap_clamps_scheduler_slots() -> Result<()> {
        let executor = SingleRowExecutor::new();
        let max_rows_seen = executor.max_rows_seen.clone();
        let mut engine = Engine::with_config(executor, MockKvPool::new(4), test_config(4));
        engine.submit_request(vec![1, 2, 3], 1);
        engine.submit_request(vec![10, 11, 12], 1);

        engine.step()?;

        assert_eq!(engine.active_count(), 1);
        assert_eq!(engine.waiting_count(), 1);
        assert_eq!(max_rows_seen.get(), 1);
        Ok(())
    }

    #[test]
    fn step_budget_clamps_prefill_tokens() -> Result<()> {
        let mut config = test_config(1);
        config.chunked_prefill_size = 8;
        let mut engine = Engine::with_config_and_governor(
            MockExecutor::ready(),
            MockKvPool::new(1),
            config,
            Box::new(TokenBudgetGovernor { max_tokens: 3 }),
        );
        engine.submit_request(vec![1, 2, 3, 4, 5, 6, 7, 8], 1);

        // First tick submits a 3-token prefill chunk; second tick polls and
        // applies it, proving the governor budget was enforced before submit.
        engine.step()?;
        engine.step()?;

        let request = engine.active.values().next().expect("request still active");
        assert_eq!(request.prefill_start_pos, 3);
        assert!(matches!(request.phase, RequestPhase::Prefilling { .. }));
        Ok(())
    }

    #[test]
    fn throughput_counters_track_steps_tokens_and_completions() -> Result<()> {
        let mut engine = Engine::with_config(
            MockExecutor::ready(),
            MockKvPool::with_capacity(1, 4, 64),
            test_config(1),
        );
        let before = engine.throughput_stats();
        assert_eq!(before.steps, 0);
        assert_eq!(before.prefill_tokens, 0);
        assert_eq!(before.generated_tokens, 0);
        assert_eq!(before.requests_completed, 0);

        let handle = engine.submit_request(vec![1, 2, 3, 4, 9], 3);
        engine.run_to_idle()?;
        assert_finished(engine.completed(handle).expect("completed"));

        let stats = engine.throughput_stats();
        assert_eq!(stats.prefill_tokens, 5, "whole prompt prefilled once");
        assert_eq!(
            stats.generated_tokens, 3,
            "length-bound run commits max_tokens tokens"
        );
        assert_eq!(stats.requests_completed, 1);
        assert!(
            stats.steps >= 2,
            "at least one prefill and one decode step, got {}",
            stats.steps
        );
        Ok(())
    }

    #[test]
    fn prefix_hit_no_double_free_on_finish() -> Result<()> {
        // Regression: release_reused_prefix (kv.release_pages) then kv.free_slot
        // (reclaim_page) both push the prefix page to the free list when page_refs
        // hits 0 before the slot page table is cleared. The page ends up in the
        // free list twice → two future slots get the same physical page → KV
        // corruption / heap corruption in workers.
        let mut engine = Engine::with_config(
            MockExecutor::ready(),
            MockKvPool::with_capacity(2, 4, 16),
            test_config(2),
        );
        // First request: fills one sealed block (tokens 1..=4) → published to radix.
        let _first = engine.submit_request(vec![1, 2, 3, 4, 9], 1);
        engine.run_to_idle()?;
        let free_after_first = engine.kv_free_pages();

        // Second request: prefix hit on the sealed block → the shared page gets
        // retain_pages called (page_refs=1). On finish, the page must enter the
        // free list exactly once.
        let second = engine.submit_request(vec![1, 2, 3, 4, 10, 11], 1);
        engine.run_to_idle()?;
        assert_finished(engine.completed(second).expect("second completed"));
        let free_after_second = engine.kv_free_pages();

        // A third allocation must not see the prefix page listed twice.
        // If double-pushed, two distinct allocs would get the same page.
        let third = engine.submit_request(vec![20, 21, 22, 23, 24], 1);
        let fourth = engine.submit_request(vec![30, 31, 32, 33, 34], 1);
        engine.step()?;
        let slots: Vec<_> = engine.active.iter().collect();
        // Each admitted slot must have a distinct set of pages.
        if slots.len() >= 2 {
            let pages0: std::collections::HashSet<u32> = engine
                .kv
                .page_indices(*slots[0].0)
                .iter()
                .copied()
                .collect();
            let pages1: std::collections::HashSet<u32> = engine
                .kv
                .page_indices(*slots[1].0)
                .iter()
                .copied()
                .collect();
            assert!(
                pages0.is_disjoint(&pages1),
                "two active slots share a physical page — double-free regression: \
                 slot0={pages0:?} slot1={pages1:?}"
            );
        }
        let _ = (free_after_first, free_after_second, third, fourth);
        Ok(())
    }

    #[test]
    fn prefix_hit_reuses_pages() -> Result<()> {
        let mut engine = Engine::with_config(
            MockExecutor::ready(),
            MockKvPool::with_capacity(1, 4, 8),
            test_config(1),
        );
        let first = engine.submit_request(vec![1, 2, 3, 4, 9], 1);

        engine.run_to_idle()?;
        assert_finished(engine.completed(first).expect("first completed"));
        assert_eq!(engine.prefix_cache_stats().lookups, 1);
        assert_eq!(engine.prefix_cache_stats().hits, 0);
        assert_eq!(engine.prefix_cache_stats().published_pages, 1);
        assert!(
            engine.kv_system_metrics().resident_pages > 0,
            "published prefix page should stay resident"
        );

        let hit = engine.radix.peek_longest_prefix_match(&[1, 2, 3, 4]);
        assert_eq!(hit.matched_len, 4);
        assert_eq!(hit.block_ids.len(), 1);
        let cached_page = hit.block_ids[0];
        let free_after_publish = engine.kv_free_pages();

        let second = engine.submit_request(vec![1, 2, 3, 4, 10, 11], 1);
        engine.step()?;

        let (&slot, request) = engine
            .active
            .iter()
            .find(|(_, request)| request.handle == second)
            .expect("second admitted");
        assert_eq!(request.prefill_start_pos, 4);
        assert_eq!(request.reused_prefix_pages, vec![cached_page]);
        assert_eq!(engine.kv.page_indices(slot)[0], cached_page);
        assert_eq!(engine.kv_free_pages(), free_after_publish - 1);
        let stats = engine.prefix_cache_stats();
        assert_eq!(stats.lookups, 2);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.hit_tokens, 4);
        assert_eq!(stats.hit_pages, 1);
        assert_eq!(stats.cached_pages, 1);
        let kv_system = engine.kv_system_metrics();
        assert_eq!(kv_system.reuse_hit_resident, 1, "{kv_system:?}");
        assert_eq!(kv_system.prefix_match_full_blocks, 1, "{kv_system:?}");
        assert_eq!(kv_system.prefix_match_clamped_blocks, 1, "{kv_system:?}");
        assert!(kv_system.resident_pages > 0, "{kv_system:?}");

        engine.run_to_idle()?;
        assert_finished(engine.completed(second).expect("second completed"));
        Ok(())
    }

    #[test]
    fn prefix_match_clamped_to_backend_reusable_blocks() -> Result<()> {
        // Regression for false prefix attach: the backend reports only one
        // complete restore boundary, so both publish and attach stay at one
        // block instead of caching an unusable tail.
        let mut engine = Engine::with_config(
            LimitedPrefixExecutor::with_max_reuse_blocks(1),
            MockKvPool::with_capacity(1, 4, 16),
            test_config(1),
        );
        let first = engine.submit_request(vec![1, 2, 3, 4, 5, 6, 7, 8, 99], 1);
        engine.run_to_idle()?;
        assert_finished(engine.completed(first).expect("first completed"));

        let hit = engine
            .radix
            .peek_longest_prefix_match(&[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(hit.matched_len, 4);
        assert_eq!(hit.block_ids.len(), 1);

        let second = engine.submit_request(vec![1, 2, 3, 4, 5, 6, 7, 8, 100, 101], 1);
        engine.step()?;
        let request = engine
            .active
            .values()
            .find(|request| request.handle == second)
            .expect("second admitted");
        assert_eq!(
            request.prefill_start_pos, 4,
            "prefix attach must clamp to the backend's 1 reusable page"
        );
        assert_eq!(request.reused_prefix_pages.len(), 1);

        engine.run_to_idle()?;
        assert_finished(engine.completed(second).expect("second completed"));
        Ok(())
    }

    #[test]
    fn cross_turn_reuse() -> Result<()> {
        let mut engine = Engine::with_config(
            MockExecutor::ready(),
            MockKvPool::with_capacity(1, 4, 4),
            test_config(1),
        );
        let first = engine.submit_request(vec![7, 7, 7, 7], 1);
        engine.run_to_idle()?;
        assert_eq!(
            engine
                .completed(first)
                .expect("first completed")
                .generated_tokens,
            vec![8]
        );

        let second = engine.submit_request(vec![7, 7, 7, 7], 1);
        engine.run_to_idle()?;
        assert_eq!(
            engine
                .completed(second)
                .expect("second completed")
                .generated_tokens,
            vec![8]
        );

        let third = engine.submit_request(vec![7, 7, 7, 7], 1);
        engine.run_to_idle()?;
        assert_eq!(
            engine
                .completed(third)
                .expect("third completed")
                .generated_tokens,
            vec![8]
        );
        assert_eq!(engine.radix.cached_page_count(), 1);
        assert_eq!(engine.kv_free_pages(), engine.kv.total_pages() - 1);
        Ok(())
    }

    #[test]
    fn eviction_frees_lru_not_active() -> Result<()> {
        let mut engine = Engine::with_config(
            MockExecutor::ready(),
            MockKvPool::with_capacity(1, 2, 5),
            test_config(1),
        );
        let first = engine.submit_request(vec![1, 1], 1);
        engine.run_to_idle()?;
        assert_finished(engine.completed(first).expect("first completed"));

        let second = engine.submit_request(vec![2, 2], 1);
        engine.run_to_idle()?;
        assert_finished(engine.completed(second).expect("second completed"));

        let active = engine.submit_request(vec![1, 1, 3], 1);
        engine.step()?;
        assert!(
            engine
                .active
                .values()
                .any(|request| request.handle == active)
        );

        let evicted = engine.evict_prefix_cache_for_pages(1);
        assert_eq!(evicted, 1);
        assert_eq!(
            engine.radix.peek_longest_prefix_match(&[2, 2]).matched_len,
            0
        );
        assert_eq!(
            engine.radix.peek_longest_prefix_match(&[1, 1]).matched_len,
            2
        );

        engine.run_to_idle()?;
        assert_finished(engine.completed(active).expect("active completed"));
        Ok(())
    }

    #[test]
    fn invalidate_prefix_cache_drops_all_idle_cached_pages() -> Result<()> {
        let mut engine = Engine::with_config(
            MockExecutor::ready(),
            MockKvPool::with_capacity(1, 2, 6),
            test_config(1),
        );
        // Seal two distinct single-block prefixes into the cache.
        let a = engine.submit_request(vec![5, 5], 1);
        engine.run_to_idle()?;
        assert_finished(engine.completed(a).expect("a completed"));
        let b = engine.submit_request(vec![7, 7], 1);
        engine.run_to_idle()?;
        assert_finished(engine.completed(b).expect("b completed"));
        assert_eq!(engine.radix.cached_page_count(), 2);
        let total = engine.kv.total_pages();
        assert_eq!(engine.kv_free_pages(), total - 2);

        // A live re-merge invalidates every cached block; with no in-flight
        // request pinning a page, all idle pages return to the pool.
        engine.invalidate_prefix_cache();
        assert_eq!(engine.radix.cached_page_count(), 0);
        assert_eq!(engine.kv_free_pages(), total, "every idle page reclaimed");
        assert_eq!(
            engine.radix.peek_longest_prefix_match(&[5, 5]).matched_len,
            0
        );
        assert_eq!(
            engine.radix.peek_longest_prefix_match(&[7, 7]).matched_len,
            0
        );
        Ok(())
    }

    #[test]
    fn invalidate_prefix_cache_keeps_pinned_drops_idle() -> Result<()> {
        let mut engine = Engine::with_config(
            MockExecutor::ready(),
            MockKvPool::with_capacity(1, 2, 5),
            test_config(1),
        );
        let first = engine.submit_request(vec![1, 1], 1);
        engine.run_to_idle()?;
        assert_finished(engine.completed(first).expect("first completed"));
        let second = engine.submit_request(vec![2, 2], 1);
        engine.run_to_idle()?;
        assert_finished(engine.completed(second).expect("second completed"));
        assert_eq!(engine.radix.cached_page_count(), 2);

        // Pin [1,1] via an in-flight request that reuses it (ref_count > 0).
        let active = engine.submit_request(vec![1, 1, 3], 1);
        engine.step()?;
        assert!(
            engine
                .active
                .values()
                .any(|request| request.handle == active)
        );

        // Live re-merge invalidation: the idle [2,2] block is dropped, the
        // pinned [1,1] block is left in place (never freed under a live reader).
        engine.invalidate_prefix_cache();
        assert_eq!(
            engine.radix.peek_longest_prefix_match(&[2, 2]).matched_len,
            0
        );
        assert_eq!(
            engine.radix.peek_longest_prefix_match(&[1, 1]).matched_len,
            2,
            "pinned prefix survives invalidation"
        );

        engine.run_to_idle()?;
        assert_finished(engine.completed(active).expect("active completed"));
        Ok(())
    }

    #[test]
    fn prefix_refcount_no_double_free() -> Result<()> {
        let mut engine = Engine::with_config(
            MockExecutor::ready(),
            MockKvPool::with_capacity(1, 2, 4),
            test_config(1),
        );
        let first = engine.submit_request(vec![5, 5], 1);
        engine.run_to_idle()?;
        assert_finished(engine.completed(first).expect("first completed"));
        assert_eq!(engine.radix.cached_page_count(), 1);
        assert_eq!(engine.kv_free_pages(), engine.kv.total_pages() - 1);

        let second = engine.submit_request(vec![5, 5], 1);
        engine.run_to_idle()?;
        assert_finished(engine.completed(second).expect("second completed"));
        assert_eq!(engine.radix.cached_page_count(), 1);
        assert_eq!(engine.kv_free_pages(), engine.kv.total_pages() - 1);

        assert_eq!(engine.evict_prefix_cache_for_pages(1), 1);
        assert_eq!(engine.kv_free_pages(), engine.kv.total_pages());
        assert_eq!(engine.evict_prefix_cache_for_pages(1), 0);
        assert_eq!(engine.kv_free_pages(), engine.kv.total_pages());
        Ok(())
    }

    #[test]
    fn partial_block_not_published() -> Result<()> {
        let mut engine = Engine::with_config(
            MockExecutor::ready(),
            MockKvPool::with_capacity(1, 4, 8),
            test_config(1),
        );
        let partial = engine.submit_request(vec![9, 9, 9], 1);
        engine.run_to_idle()?;
        assert_finished(engine.completed(partial).expect("partial completed"));
        assert_eq!(engine.radix.cached_page_count(), 0);

        let with_tail = engine.submit_request(vec![1, 2, 3, 4, 5, 6], 1);
        engine.run_to_idle()?;
        assert_finished(engine.completed(with_tail).expect("with tail completed"));
        let hit = engine.radix.peek_longest_prefix_match(&[1, 2, 3, 4, 5, 6]);
        assert_eq!(hit.matched_len, 4);
        assert_eq!(hit.block_ids.len(), 1);
        Ok(())
    }

    #[test]
    fn r1d_admits_long_prompt_over_budget_and_chunks() -> Result<()> {
        let mut config = test_config(1);
        config.chunked_prefill_size = 2;
        // Per-tick budget < prompt: relies on the chunked-prefill admit relaxation.
        config.max_num_batched_tokens = 3;
        config.max_prefill_tokens = 3;
        let mut engine = Engine::with_config(MockExecutor::ready(), MockKvPool::new(1), config);
        engine.submit_request(vec![1, 2, 3, 4, 5], 4);

        engine.admit_waiting()?;
        assert_eq!(
            engine.active.len(),
            1,
            "a prompt longer than the per-tick budget must still admit (chunked across ticks)"
        );

        let plan = engine.build_forward_plan();
        assert_eq!(plan.prefill_rows.len(), 1);
        assert_eq!(
            plan.prefill_rows[0].tokens,
            vec![1, 2],
            "prefill must be chunked to chunked_prefill_size, not the whole prompt"
        );
        assert_eq!(plan.prefill_rows[0].start_pos, 0);
        assert_eq!(plan.prefill_rows[0].total_tokens, 5);
        Ok(())
    }

    #[test]
    fn r1d_chunked_prefill_completes_long_prompt() -> Result<()> {
        let mut config = test_config(1);
        config.chunked_prefill_size = 2;
        let mut engine = Engine::with_config(MockExecutor::ready(), MockKvPool::new(1), config);
        let handle = engine.submit_request(vec![1, 2, 3, 4, 5], 2);

        engine.run_to_idle()?;

        let completed = engine.completed(handle).expect("request completed");
        // chunks [1,2] [3,4] [5]; the final chunk's committed token is 5 + 1 = 6, then decode -> 7.
        assert_eq!(completed.generated_tokens, vec![6, 7]);
        assert_finished(completed);
        Ok(())
    }

    #[test]
    fn prefill_chunk_stops_on_page_boundary() -> Result<()> {
        // page_size 4, 10-token prompt, ample budget: the first chunk must stop
        // at the last page boundary (8) instead of crossing it to the prompt end
        // (10), so a restore-boundary-limited backend can snapshot state at the
        // same page boundary the radix later caches. The 2-token sub-page tail
        // follows next tick. Pairs with
        // `prefix_match_clamped_to_backend_reusable_blocks`: the clamp is the
        // safety floor, this alignment is the reuse-coverage win.
        let mut engine = Engine::with_config(
            MockExecutor::ready(),
            MockKvPool::with_capacity(1, 4, 16),
            test_config(1),
        );
        engine.submit_request(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1);

        let mut prefilling_positions = Vec::new();
        for _ in 0..6 {
            engine.step()?;
            if let Some(request) = engine.active.values().next() {
                if matches!(request.phase, RequestPhase::Prefilling { .. }) {
                    prefilling_positions.push(request.prefill_start_pos);
                }
            }
        }
        // The chunk landed on page boundary 8 (a Prefilling progress value).
        // Without alignment the single chunk [0,10) would jump straight to
        // Decoding and 8 would never appear.
        assert!(
            prefilling_positions.contains(&8),
            "prefill chunk must stop on page boundary 8, got {prefilling_positions:?}"
        );
        Ok(())
    }

    #[test]
    fn r1d_chunked_prefill_advances_progress_across_ticks() -> Result<()> {
        let mut config = test_config(1);
        config.chunked_prefill_size = 2;
        let mut engine = Engine::with_config(MockExecutor::ready(), MockKvPool::new(1), config);
        engine.submit_request(vec![1, 2, 3, 4, 5], 4);

        let mut prefill_start_positions = Vec::new();
        for _ in 0..6 {
            engine.step()?;
            if let Some(request) = engine.active.values().next() {
                if matches!(request.phase, RequestPhase::Prefilling { .. }) {
                    prefill_start_positions.push(request.prefill_start_pos);
                }
            }
        }
        // start_pos advances in chunk-sized steps (0 -> 2 -> 4) across ticks,
        // proving the prompt was prefilled over multiple ticks, not all at once.
        assert!(
            prefill_start_positions.contains(&2) && prefill_start_positions.contains(&4),
            "expected chunked progress across ticks, got {prefill_start_positions:?}"
        );
        Ok(())
    }

    #[derive(Default)]
    struct FakeDiffusionModel {
        prompts: Vec<Vec<u32>>,
        commits: Vec<Vec<u32>>,
        begin_configs: Vec<DiffusionGenerationConfig>,
        predictions: Vec<DiffusionCanvasPrediction>,
        calls: usize,
    }

    impl DiffusionBlockModel for FakeDiffusionModel {
        fn begin_request(
            &mut self,
            config: &DiffusionGenerationConfig,
        ) -> std::result::Result<(), DiffusionModelError> {
            self.begin_configs.push(config.clone());
            Ok(())
        }

        fn prefill(&mut self, prompt_tokens: &[u32]) -> Result<(), DiffusionModelError> {
            self.prompts.push(prompt_tokens.to_vec());
            Ok(())
        }

        fn predict_canvas(
            &mut self,
            _canvas: &[u32],
            _valid_len: usize,
            _step: usize,
            _temperature: f32,
        ) -> Result<DiffusionCanvasPrediction, DiffusionModelError> {
            let idx = self.calls.min(self.predictions.len().saturating_sub(1));
            self.calls += 1;
            self.predictions
                .get(idx)
                .cloned()
                .ok_or_else(|| DiffusionModelError::new("no prediction"))
        }

        fn commit(&mut self, tokens: &[u32]) -> Result<(), DiffusionModelError> {
            self.commits.push(tokens.to_vec());
            Ok(())
        }
    }

    fn diffusion_prediction(tokens: &[u32], canvas_len: usize) -> DiffusionCanvasPrediction {
        let mut sampled_tokens = vec![0; canvas_len];
        let mut argmax_tokens = vec![0; canvas_len];
        let entropies = vec![0.0; canvas_len];
        for (idx, &token) in tokens.iter().enumerate() {
            sampled_tokens[idx] = token;
            argmax_tokens[idx] = token;
        }
        DiffusionCanvasPrediction {
            sampled_tokens,
            argmax_tokens,
            entropies,
        }
    }

    fn diffusion_config(max_new_tokens: usize) -> DiffusionGenerationConfig {
        DiffusionGenerationConfig {
            canvas_length: 4,
            max_denoising_steps: 1,
            max_new_tokens,
            vocab_size: 128,
            stop_token_ids: vec![99],
            pad_token_id: 0,
            entropy_bound: 0.1,
            confidence_threshold: 0.01,
            t_min: 0.4,
            t_max: 0.8,
            stability_threshold: 1,
            seed: 0,
        }
    }

    #[test]
    fn diffusion_buffered_executor_runs_through_engine_completion() -> Result<()> {
        let model = FakeDiffusionModel {
            predictions: vec![diffusion_prediction(&[10, 11, 12], 4)],
            ..FakeDiffusionModel::default()
        };
        let executor = BufferedDiffusionExecutor::new(model, diffusion_config(3));
        let mut config = test_config(1);
        config.chunked_prefill_size = 2;
        let mut engine = Engine::with_config(executor, HostPagedKvPool::new(1, 8, 4), config);

        let handle = engine.submit_request_with_options(
            vec![1, 2, 3, 4],
            3,
            RequestOptions {
                sampling: SamplingParams {
                    max_new_tokens: Some(3),
                    ..SamplingParams::default()
                },
                ..RequestOptions::default()
            },
        );
        engine.run_to_idle()?;

        let completed = engine.completed(handle).expect("request completed");
        assert_eq!(completed.prompt_tokens, vec![1, 2, 3, 4]);
        assert_eq!(completed.generated_tokens, vec![10, 11, 12]);
        assert!(matches!(completed.finish, Some(FinishReason::Length)));
        Ok(())
    }

    #[test]
    fn diffusion_executor_uses_engine_max_tokens_without_sampling_override() -> Result<()> {
        let model = FakeDiffusionModel {
            predictions: vec![diffusion_prediction(&[10, 11], 4)],
            ..FakeDiffusionModel::default()
        };
        let executor = BufferedDiffusionExecutor::new(model, diffusion_config(8));
        let mut engine =
            Engine::with_config(executor, HostPagedKvPool::new(1, 8, 4), test_config(1));

        let handle = engine.submit_request(vec![1, 2], 2);
        engine.run_to_idle()?;

        let completed = engine.completed(handle).expect("request completed");
        assert_eq!(completed.generated_tokens, vec![10, 11]);
        let model = engine.executor.into_inner();
        assert_eq!(model.begin_configs.len(), 1);
        assert_eq!(model.begin_configs[0].max_new_tokens, 2);
        Ok(())
    }

    #[test]
    fn diffusion_executor_uses_clamped_budget_with_sampling_override() -> Result<()> {
        let model = FakeDiffusionModel {
            predictions: vec![diffusion_prediction(&[10, 11, 12], 4)],
            ..FakeDiffusionModel::default()
        };
        let executor = BufferedDiffusionExecutor::new(model, diffusion_config(8));
        let mut config = test_config(1);
        config.max_total_tokens = 5;
        let mut engine = Engine::with_config(executor, HostPagedKvPool::new(1, 8, 4), config);

        let handle = engine.submit_request_with_options(
            vec![1, 2],
            100,
            RequestOptions {
                sampling: SamplingParams {
                    max_new_tokens: Some(100),
                    ..SamplingParams::default()
                },
                ..RequestOptions::default()
            },
        );
        engine.run_to_idle()?;

        let completed = engine.completed(handle).expect("request completed");
        assert_eq!(completed.generated_tokens, vec![10, 11, 12]);
        let model = engine.executor.into_inner();
        assert_eq!(model.begin_configs.len(), 1);
        assert_eq!(model.begin_configs[0].max_new_tokens, 3);
        Ok(())
    }

    #[test]
    fn diffusion_executor_disables_prefix_reuse_for_repeated_prompt() -> Result<()> {
        let model = FakeDiffusionModel {
            predictions: vec![
                diffusion_prediction(&[10, 11], 4),
                diffusion_prediction(&[20, 21], 4),
            ],
            ..FakeDiffusionModel::default()
        };
        let executor = BufferedDiffusionExecutor::new(model, diffusion_config(2));
        let mut engine =
            Engine::with_config(executor, HostPagedKvPool::new(1, 8, 4), test_config(1));

        let prompt = vec![1, 2, 3, 4];
        let first = engine.submit_request(prompt.clone(), 2);
        engine.run_to_idle()?;
        let second = engine.submit_request(prompt.clone(), 2);
        engine.run_to_idle()?;

        assert_eq!(
            engine
                .completed(first)
                .expect("first request completed")
                .generated_tokens,
            vec![10, 11]
        );
        assert_eq!(
            engine
                .completed(second)
                .expect("second request completed")
                .generated_tokens,
            vec![20, 21]
        );
        let stats = engine.prefix_cache_stats();
        assert_eq!(stats.lookups, 2);
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.published_pages, 0);
        let model = engine.executor.into_inner();
        assert_eq!(model.prompts, vec![prompt.clone(), prompt]);
        Ok(())
    }

    /// Micro-benchmark of the device-neutral engine-core scheduler (mock backend,
    /// synchronous — so this isolates the CPU-side scheduling cost: admission,
    /// radix, chunked planning, apply_output). Run:
    ///   CUDARC_CUDA_VERSION=12060 cargo test --release -p infer-core \
    ///     bench_engine_core_scheduler_throughput -- --ignored --nocapture
    #[test]
    #[ignore = "benchmark; run with --release -- --ignored --nocapture"]
    fn bench_engine_core_scheduler_throughput() -> Result<()> {
        use std::time::Instant;

        // Scenario 1: c=1 single long request (AI-PC focus — per-decode-tick cost).
        {
            let mut config = test_config(1);
            config.max_prompt_tokens = 8192;
            config.max_total_tokens = 16_384;
            let mut engine = Engine::with_config(MockExecutor::ready(), MockKvPool::new(1), config);
            let prompt: Vec<u32> = (0..1024).map(|i| (i % 50_000) as u32 + 2).collect();
            let max_tokens = 512usize;
            let handle = engine.submit_request(prompt, max_tokens);
            let mut ticks = 0u64;
            let start = Instant::now();
            while !engine.is_idle() {
                engine.step()?;
                ticks += 1;
            }
            let elapsed = start.elapsed();
            let generated = engine
                .completed(handle)
                .map_or(0, |c| c.generated_tokens.len());
            eprintln!(
                "[engine-core c=1 long] gen={generated} ticks={ticks} wall={elapsed:?} \
                 us_per_tick={:.3} sched_tok_per_s={:.0}",
                elapsed.as_micros() as f64 / ticks.max(1) as f64,
                generated as f64 / elapsed.as_secs_f64(),
            );
        }

        // Scenario 2: batched concurrency (8 slots, 64 distinct requests).
        {
            let mut config = test_config(8);
            config.max_prompt_tokens = 8192;
            config.max_total_tokens = 16_384;
            config.chunked_prefill_size = 256;
            let mut engine = Engine::with_config(MockExecutor::ready(), MockKvPool::new(8), config);
            let n = 64usize;
            let max_tokens = 128usize;
            let handles: Vec<_> = (0..n)
                .map(|r| {
                    let mut prompt = vec![1u32; 384];
                    prompt[0] = r as u32 + 2; // distinct prefix -> no radix reuse
                    engine.submit_request(prompt, max_tokens)
                })
                .collect();
            let mut ticks = 0u64;
            let start = Instant::now();
            while !engine.is_idle() {
                engine.step()?;
                ticks += 1;
            }
            let elapsed = start.elapsed();
            let total_gen: usize = handles
                .iter()
                .map(|h| engine.completed(*h).map_or(0, |c| c.generated_tokens.len()))
                .sum();
            eprintln!(
                "[engine-core batched c=8 n=64] gen={total_gen} ticks={ticks} wall={elapsed:?} \
                 us_per_tick={:.3} sched_tok_per_s={:.0}",
                elapsed.as_micros() as f64 / ticks.max(1) as f64,
                total_gen as f64 / elapsed.as_secs_f64(),
            );
        }
        Ok(())
    }
}
