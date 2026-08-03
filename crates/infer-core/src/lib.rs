//! Backend-agnostic inference engine loop.
//!
//! This crate depends only on the plan and seam contracts. Runtime-specific
//! buffers and model implementation details stay below `infer-seam`.

use std::collections::{BTreeMap, VecDeque};
use std::sync::OnceLock;

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
    AdmissionVerdict, BackendExecutor, DeviceRowDemand, KvPool, PermissiveGovernor, PollResult,
    ResourceGovernor, StepBudget,
};

/// Stable handle returned to callers when a request is submitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RequestHandle(u64);

impl RequestHandle {
    #[must_use]
    pub fn id(self) -> u64 {
        self.0
    }
}

/// Request priority level. Higher-priority requests are admitted first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum RequestPriority {
    Low = 0,
    #[default]
    Normal = 1,
    High = 2,
}

/// Scheduler admission knobs used by `infer-core`.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub num_slots: usize,
    /// Scheduler-only cap on active requests. `None` means `num_slots`.
    pub max_running_requests: Option<usize>,
    pub max_num_batched_tokens: usize,
    pub max_prefill_tokens: usize,
    pub prefill_max_requests: Option<usize>,
    pub max_prompt_tokens: usize,
    pub max_total_tokens: usize,
    pub prefix_cache_low_water_pages: usize,
    /// Per-request prefill chunk size; prompts longer than this span multiple ticks.
    pub chunked_prefill_size: usize,
    /// Cross-request prompt-prefix reuse via the host radix cache.
    pub enable_prefix_cache: bool,
    /// Admit waiters beyond `max_running_requests` by parking longest-running decode.
    pub slot_oversubscription: bool,
}

impl SchedulerConfig {
    #[must_use]
    pub fn for_slots(num_slots: usize) -> Self {
        Self {
            num_slots,
            ..Self::default()
        }
    }

    fn max_concurrent_prefill(&self) -> usize {
        self.prefill_max_requests
            .unwrap_or_else(|| self.running_cap())
            .max(1)
    }

    fn running_cap(&self) -> usize {
        self.max_running_requests
            .unwrap_or(self.num_slots)
            .min(self.num_slots)
            .max(1)
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
            max_running_requests: None,
            max_num_batched_tokens: 16_384,
            max_prefill_tokens: 16_384,
            prefill_max_requests: None,
            max_prompt_tokens: 16_384,
            max_total_tokens: 32_768,
            prefix_cache_low_water_pages: 0,
            chunked_prefill_size: 2_048,
            enable_prefix_cache: true,
            slot_oversubscription: false,
        }
    }
}

/// Per-request next-token constraint. Called with `None` once at admit and with
/// each committed token after; the returned bitmask (bit set = allowed) rides on
/// the next step's `SamplingParams`. `None` = unconstrained from here on.
///
/// A callback rather than a matcher because the tokenizer lives above the
/// engine — this keeps the grammar backend out of `infer-core` entirely.
#[derive(Clone)]
pub struct GrammarHook(pub std::sync::Arc<GrammarFn>);

/// Committed token in, next-step bitmask out.
pub type GrammarFn = dyn Fn(Option<u32>) -> Option<std::sync::Arc<[u32]>> + Send + Sync;

impl std::fmt::Debug for GrammarHook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("GrammarHook")
    }
}

/// Options accepted at request ingress.
#[derive(Debug, Clone, Default)]
pub struct RequestOptions {
    pub priority: RequestPriority,
    /// Cooperative cancellation observed before queue insertion.
    pub cancelled: bool,
    /// Default: greedy / argmax.
    pub sampling: SamplingParams,
    pub grammar: Option<GrammarHook>,
}

/// Completed request state retained after its slot has been freed.
#[derive(Debug, Clone)]
pub struct CompletedRequest {
    pub handle: RequestHandle,
    pub prompt_tokens: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    pub finish: Option<FinishReason>,
}

/// Engine throughput counters, monotonic since engine start.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ThroughputStats {
    pub steps: u64,
    pub prefill_tokens: u64,
    /// Tokens committed to requests (final prefill chunk + decode).
    pub generated_tokens: u64,
    /// Requests that finished after holding a slot.
    pub requests_completed: u64,
}

/// Process-global GPU-busy micros: cumulative forward wall (`submit`→`poll` Ready)
/// summed across every [`Engine`] step in this process, excluding the idle between
/// steps when nothing is in flight. Read the delta over a window (e.g. a rollout)
/// to split it into decode-active vs host-latency-idle. Monotonic; one engine per
/// process makes the delta that engine's busy time.
static ENGINE_FORWARD_BUSY_MICROS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Snapshot the running forward-busy micros (submit→ready wall, summed over
/// `Engine` steps); the delta over a window is that engine's GPU-busy time.
#[must_use]
pub fn engine_forward_busy_micros() -> u64 {
    ENGINE_FORWARD_BUSY_MICROS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Per-phase host micros inside [`Engine::step`], accumulated under
/// `ARLE_STEP_PHASE=1`. The between-steps host section is invisible to a CUDA
/// profile — the whole-step decode graph makes the GPU side one contiguous
/// replay, so everything left shows up as a single per-step stall with no API
/// calls in it (2026-08-03 ledger). This is the only instrument that splits it.
/// Order matches [`STEP_PHASE_NAMES`].
static STEP_PHASE_MICROS: [std::sync::atomic::AtomicU64; 6] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];
static STEP_PHASE_STEPS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
const STEP_PHASE_NAMES: [&str; 6] = ["poll", "apply_out", "poll_bg", "admit", "plan", "submit"];

/// KV host-demoted counters. All zero unless the executor reports nonzero tier capacity.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KvTierStats {
    pub demoted_pages: u64,
    pub promoted_pages: u64,
    /// Promotions that failed (entry severed, tail re-prefilled instead).
    pub promote_failures: u64,
    pub resident_blocks: usize,
    pub demoted_slots: u64,
    pub promoted_slots: u64,
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
    pub tier_io_mode: infer_seam::KvTierIoMode,
    pub tier_io_useful_read_bytes: u64,
    pub tier_io_useful_write_bytes: u64,
    pub tier_io_submitted_read_bytes: u64,
    pub tier_io_submitted_write_bytes: u64,
    pub tier_io_metadata_write_bytes: u64,
    pub tier_io_failures: u64,
    pub tier_io_completion_wait_ns: u64,
}

/// Prefix-cache counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrefixCacheStats {
    pub lookups: u64,
    /// Attached at least one backend-reusable prefix page.
    pub hits: u64,
    pub hit_tokens: u64,
    pub hit_pages: u64,
    pub published_pages: u64,
    pub cached_pages: usize,
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

/// Cap on retained completions. The in-process consumer drains each engine step
/// and reads its own completion synchronously, so only recent (largest,
/// monotonic) handles are ever queried; beyond this the oldest — long since read
/// — are dropped to keep `completed` from growing without bound.
const COMPLETED_CAP: usize = 1 << 16;

/// Result of attempting to admit the front waiter onto one slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AdmitOutcome {
    /// Admitted onto the slot; the slot is consumed.
    Admitted,
    /// No waiting request to admit.
    NoWaiter,
    /// The front waiter does not fit the current per-tick budget, but might
    /// once other active requests finish and free pages — keep it in
    /// `waiting` and retry on a later tick.
    Throttled,
    /// The front waiter was popped and failed: it needs more pages than the
    /// pool can EVER provide (checked only when the pool is completely idle,
    /// so no other request could still free more) — retrying would wait
    /// forever. Not a slot consumption; the caller should keep admitting
    /// whatever is now at the front of `waiting`.
    Rejected,
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
    grammar: Option<GrammarHook>,
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
            admit_seq: 0,
            admit_gen_mark: 0,
            grammar: None,
        }
    }

    fn with_grammar(mut self, grammar: Option<GrammarHook>) -> Self {
        if let Some(g) = &grammar {
            self.sampling.grammar_bitmask = (g.0)(None);
        }
        self.grammar = grammar;
        self
    }

    fn advance_grammar(&mut self, token: u32) {
        let Some(g) = self.grammar.clone() else {
            return;
        };
        self.sampling.grammar_bitmask = (g.0)(Some(token));
    }

    fn complete_immediately(mut self, finish: FinishReason) -> Self {
        self.phase = RequestPhase::Finished;
        self.finish = Some(finish);
        self
    }

    /// Generated tokens are kept (#156): recompute re-prefills the committed
    /// stream and decode resumes after the last committed token, so the
    /// observer never sees a token twice.
    fn reset_for_recompute(mut self) -> Self {
        self.phase = RequestPhase::Prefilling { progress: 0 };
        self.prefill_start_pos = 0;
        self.reused_prefix_pages.clear();
        // Store-entry lifetime is the caller's job (drop before reset); the
        // key is cleared so a recomputed request can never promote stale state.
        self.swap_key = None;
        self.swap_seq_len = 0;
        self.finish = None;
        self
    }

    fn prompt_len(&self) -> usize {
        self.prompt_tokens.len()
    }

    /// Prompt + generated: the committed stream a prefill must materialize.
    fn committed_len(&self) -> usize {
        self.prompt_tokens.len() + self.generated_tokens.len()
    }

    fn committed_tokens(&self) -> Vec<u32> {
        self.committed_slice(0, self.committed_len())
    }

    /// Borrowed for the common fresh-request case (nothing generated yet).
    fn committed_cow(&self) -> std::borrow::Cow<'_, [u32]> {
        if self.generated_tokens.is_empty() {
            std::borrow::Cow::Borrowed(&self.prompt_tokens)
        } else {
            std::borrow::Cow::Owned(self.committed_tokens())
        }
    }

    fn committed_slice(&self, start: usize, len: usize) -> Vec<u32> {
        self.prompt_tokens
            .iter()
            .chain(&self.generated_tokens)
            .skip(start)
            .take(len)
            .copied()
            .collect()
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

/// Engine admission mode. `Quiesced` marks the OPD writeback bracket (KV pool
/// released); the serve loop defers submission admission while it is set so no
/// request is prefilled onto the dropped pool (qwen35 full_attn_kv=None panic,
/// f6d 2026-07-18). Cleared by `resume_serving` after the pool is re-acquired.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
enum EngineMode {
    #[default]
    Serving,
    Quiesced,
}

/// Backend-agnostic engine loop.
///
/// Prefix reuse is host-indexed: the engine carries token blocks and page ids,
/// while executor/model-specific storage stays below the seam.
pub struct Engine<E: BackendExecutor, K: KvPool> {
    executor: E,
    kv: K,
    config: SchedulerConfig,
    max_tokens_per_step: usize,
    governor: Box<dyn ResourceGovernor>,
    radix: RadixCache,
    next_request_id: u64,
    active: BTreeMap<usize, RequestState>,
    waiting: VecDeque<RequestState>,
    completed: BTreeMap<RequestHandle, CompletedRequest>,
    inflight: Option<E::Inflight>,
    /// Wall clock captured just before the in-flight forward's `submit`, consumed
    /// when its `poll` returns Ready to accrue [`ENGINE_FORWARD_BUSY_MICROS`].
    inflight_submit_at: Option<std::time::Instant>,
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
    /// Admission mode; `Quiesced` during the OPD writeback bracket. See [`EngineMode`].
    mode: EngineMode,
    /// Scratch for the per-step `kv_device_fit` gate (demand rows + unfit
    /// indices), reused so an active gate never heap-allocs per step.
    device_demand_scratch: Vec<DeviceRowDemand>,
    device_unfit_scratch: Vec<usize>,
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
        let max_tokens_per_step = executor.max_tokens_per_step().max(1);
        config.num_slots = config.num_slots.min(max_tokens_per_step);
        config.num_slots = config.num_slots.max(1);
        if let Some(cap) = config.max_running_requests
            && cap > config.num_slots
        {
            log::warn!(
                "max_running_requests={cap} exceeds executor hot workspace slots {}; active cap follows capacity",
                config.num_slots
            );
        }
        let radix = RadixCache::new(kv.page_size().max(1));
        let model_stop_token_ids = executor.model_stop_token_ids();
        Self {
            executor,
            kv,
            config,
            max_tokens_per_step,
            governor,
            radix,
            next_request_id: 0,
            active: BTreeMap::new(),
            waiting: VecDeque::new(),
            completed: BTreeMap::new(),
            inflight: None,
            inflight_submit_at: None,
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
            mode: EngineMode::Serving,
            device_demand_scratch: Vec::new(),
            device_unfit_scratch: Vec::new(),
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
                self.record_completed(handle, request.into());
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

        // Cached: `step` runs per decode token and an env read takes a global lock.
        static STEP_DIAG: OnceLock<bool> = OnceLock::new();
        let diag = *STEP_DIAG.get_or_init(|| std::env::var_os("ARLE_STEP_DIAG").is_some());
        static PHASE: OnceLock<bool> = OnceLock::new();
        let mut mark = PHASE
            .get_or_init(|| std::env::var_os("ARLE_STEP_PHASE").is_some())
            .then(std::time::Instant::now);
        // Buffered, then committed only for decode-only steps: one 32K prefill
        // costs ~25 s and would swamp a 19 ms decode step in a flat average.
        let mut phase_buf = [0u64; 6];
        macro_rules! phase {
            ($i:expr) => {
                if let Some(t) = mark.as_mut() {
                    let now = std::time::Instant::now();
                    phase_buf[$i] += now.duration_since(*t).as_micros() as u64;
                    *t = now;
                }
            };
        }
        if let Some(inflight) = self.inflight.take() {
            match self.executor.poll(inflight)? {
                PollResult::Ready(output) => {
                    if let Some(submitted_at) = self.inflight_submit_at.take() {
                        ENGINE_FORWARD_BUSY_MICROS.fetch_add(
                            submitted_at.elapsed().as_micros() as u64,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                    }
                    let plan = self.pending_plan.take().unwrap_or_else(ForwardPlan::idle);
                    phase!(0);
                    self.apply_output(&plan, output)?;
                    phase!(1);
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

        self.executor.poll_background()?;
        phase!(2);

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
        phase!(3);
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

        self.fit_plan_to_kv_pages(&mut plan)?;
        if plan.is_idle() {
            return Ok(());
        }

        self.allocate_for_plan(&mut plan);
        phase!(4);
        if plan.is_idle() {
            return Ok(());
        }
        log::trace!("infer-core submit plan: mode={:?}", plan.mode);
        if diag {
            eprintln!("[STEP-DIAG] SUBMIT plan mode={:?}", plan.mode);
        }
        let submit_at = std::time::Instant::now();
        self.inflight = Some(self.executor.submit(&plan, &mut self.kv)?);
        self.inflight_submit_at = Some(submit_at);
        phase!(5);
        let decode_only = plan.prefill_rows.is_empty();
        self.pending_plan = Some(plan);
        if mark.is_some() && decode_only {
            for (acc, v) in STEP_PHASE_MICROS.iter().zip(phase_buf) {
                acc.fetch_add(v, std::sync::atomic::Ordering::Relaxed);
            }
            let n = STEP_PHASE_STEPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if n.is_multiple_of(500) {
                let parts: Vec<String> = STEP_PHASE_NAMES
                    .iter()
                    .zip(STEP_PHASE_MICROS.iter())
                    .map(|(name, acc)| {
                        let us = acc.load(std::sync::atomic::Ordering::Relaxed);
                        format!("{name}={:.3}ms", us as f64 / 1000.0 / n as f64)
                    })
                    .collect();
                log::info!("[step-phase] steps={n} {}", parts.join(" "));
            }
        }
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

    /// Cumulative speculative-decode counters from the backend executor.
    #[must_use]
    pub fn spec_decode_stats(&self) -> infer_seam::SpecDecodeStats {
        self.executor.spec_decode_stats()
    }

    /// Cumulative operator-policy identity and dispatch counters from the backend.
    #[must_use]
    pub fn operator_dispatch_stats(&self) -> infer_seam::OperatorDispatchStats {
        self.executor.operator_dispatch_stats()
    }

    /// Exact backend artifact identity, if the build verified one.
    #[must_use]
    pub fn artifact_identity(&self) -> infer_seam::BackendArtifactIdentity {
        self.executor.artifact_identity()
    }

    #[must_use]
    pub fn kv_system_metrics(&self) -> KvSystemMetrics {
        let mut metrics = self.kv_system_metrics;
        metrics.resident_pages = self.kv.resident_pages();
        metrics.resident_evictable_pages = self.kv.resident_evictable_pages();
        metrics.host_demoted_pages = self.executor.kv_tier_host_demoted_pages();
        metrics.host_demoted_pending_inflight = 0;
        metrics.disk_pages = self.executor.kv_tier_disk_pages();
        let tier_hits = self.executor.kv_tier_read_hits();
        metrics.reuse_hit_host_demoted = metrics
            .reuse_hit_host_demoted
            .saturating_add(tier_hits.host_demoted);
        metrics.reuse_hit_disk = metrics.reuse_hit_disk.saturating_add(tier_hits.disk);
        let io = self.executor.kv_tier_io_stats();
        metrics.tier_io_mode = io.mode;
        metrics.tier_io_useful_read_bytes = io.useful_read_bytes;
        metrics.tier_io_useful_write_bytes = io.useful_write_bytes;
        metrics.tier_io_submitted_read_bytes = io.submitted_read_bytes;
        metrics.tier_io_submitted_write_bytes = io.submitted_write_bytes;
        metrics.tier_io_metadata_write_bytes = io.metadata_write_bytes;
        metrics.tier_io_failures = io.failures;
        metrics.tier_io_completion_wait_ns = io.completion_wait_ns;
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

    /// Record a completion, evicting the oldest (smallest-handle) entries beyond
    /// `COMPLETED_CAP` so host RAM stays bounded. Handles are monotonic, so key
    /// order is submission order and the dropped entries are the coldest.
    fn record_completed(&mut self, handle: RequestHandle, completed: CompletedRequest) {
        self.completed.insert(handle, completed);
        while self.completed.len() > COMPLETED_CAP {
            self.completed.pop_first();
        }
    }

    /// Abort a waiting/parked request: release its whole-slot tier image (if any)
    /// so it cannot leak — restore_swapped_slot (planner.rs) is the only other
    /// release path — then record the Abort completion.
    fn abort_waiter(&mut self, request: RequestState) {
        if let Some(key) = request.swap_key {
            self.executor.drop_kv_slot_entries(&[key]);
        }
        self.record_completed(
            request.handle,
            request.complete_immediately(FinishReason::Abort).into(),
        );
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
            // A silent Abort reads as an empty completion to the client — say why.
            log::warn!(
                "aborting request: prompt {} tokens outside (0, max_prompt_tokens={}]",
                prompt_tokens.len(),
                self.config.max_prompt_tokens
            );
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
        // Normalize sampling max_new_tokens too — diffusion executors consume it directly.
        sampling.max_new_tokens = Some(max_tokens);
        if max_tokens == 0 {
            return NormalizedRequest::Completed(
                RequestState::new(handle, prompt_tokens, options.priority, 0, sampling)
                    .complete_immediately(FinishReason::Length),
            );
        }

        NormalizedRequest::Waiting(
            RequestState::new(
                handle,
                prompt_tokens,
                options.priority,
                max_tokens,
                sampling,
            )
            .with_grammar(options.grammar),
        )
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
        // Committed tokens (prefill then decode), forwarded to `on_token` after request loops.
        let mut committed: Vec<(RequestHandle, SlotToken)> = Vec::new();

        // Advance chunked prefill. Non-final chunk only moves progress; final chunk transitions to decode.
        let mut prompt_sealed_slots: Vec<usize> = Vec::new();
        for row in &plan.prefill_rows {
            let Some(request) = self.active.get_mut(&row.slot) else {
                continue;
            };
            if !matches!(request.phase, RequestPhase::Prefilling { .. }) {
                continue;
            }
            let target = request.committed_len();
            let new_start = (row.start_pos + row.tokens.len()).min(target);
            self.throughput_stats.prefill_tokens = self
                .throughput_stats
                .prefill_tokens
                .saturating_add(new_start.saturating_sub(row.start_pos) as u64);
            request.prefill_start_pos = new_start;
            if new_start >= target {
                request.phase = RequestPhase::Decoding;
                prompt_sealed_slots.push(row.slot);
                if let Some(token) = tokens_by_slot
                    .get_mut(&row.slot)
                    .and_then(VecDeque::pop_front)
                {
                    request.generated_tokens.push(token.token);
                    request.advance_grammar(token.token);
                    if let Some(finish) =
                        finish_reason_for(request, &token, &self.model_stop_token_ids)
                    {
                        finished_slots.push((row.slot, finish));
                    }
                    committed.push((request.handle, token));
                }
            } else {
                request.phase = RequestPhase::Prefilling {
                    progress: new_start,
                };
                tokens_by_slot.remove(&row.slot);
            }
        }

        // Seal just-prefilled prompts into radix at PROMPT boundary — a chat/messages-resend restores at
        // prompt_len, not post-generation. Without this, prompt-only resends full-recompute. `radix.insert` dedups.
        for slot in prompt_sealed_slots {
            let Some(prompt_tokens) = self.active.get(&slot).map(|r| r.prompt_tokens.clone())
            else {
                continue;
            };
            self.publish_prefix_blocks(slot, &prompt_tokens);
        }

        // Decode rows: append token(s) + check stop/length. Speculative backends may return multiple tokens
        // per row; the first was pre-allocated in `allocate_for_plan`, extras grow the slot to the
        // committed length (a no-op for a backend that already grew it to draft its chain).
        for row in &plan.decode_rows {
            let Some(mut tokens) = tokens_by_slot.remove(&row.slot) else {
                continue;
            };
            let mut token_idx = 0usize;
            while let Some(token) = tokens.pop_front() {
                // Speculative extras run BEFORE any plan repair; the engine
                // has no spec-depth to pre-budget, so on true exhaustion
                // degrade like the repair does — park the request (#162 path)
                // and drop the unappended tail — instead of a fatal unwind.
                if token_idx > 0
                    && self
                        .alloc_to_len_with_prefix_reclaim(row.slot, row.kv_seq_len + 1 + token_idx)
                        .is_err()
                {
                    self.requeue_preempted_decode(row.slot);
                    break;
                }
                let Some(request) = self.active.get_mut(&row.slot) else {
                    break;
                };
                if !matches!(request.phase, RequestPhase::Decoding) {
                    break;
                }
                request.generated_tokens.push(token.token);
                request.advance_grammar(token.token);
                let finished = finish_reason_for(request, &token, &self.model_stop_token_ids);
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

        // Stream tokens before finishing — serving layer sees terminal token ahead of completion.
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

    /// Free a slot's pages, then drop the backend's provisional prefix-state
    /// entries for them — a freed id recycles, and a stale write-only entry
    /// under a recycled id could later be confirmed as new content. The page
    /// list is snapshotted BEFORE the free (free_slot clears it). Radix-retained
    /// pages don't actually free (refcount) and their confirmed entries stay.
    fn free_slot_pages(&mut self, slot: usize) {
        let pages = self.kv.page_indices(slot).to_vec();
        self.kv.free_slot(slot);
        self.executor.release_provisional_prefix_pages(&pages);
        self.executor.release_kv_slot(slot);
    }

    fn finish_slot(&mut self, slot: usize, reason: FinishReason) {
        let Some(mut request) = self.active.remove(&slot) else {
            return;
        };
        self.throughput_stats.requests_completed =
            self.throughput_stats.requests_completed.saturating_add(1);
        request.phase = RequestPhase::Finished;
        request.finish = Some(reason);
        // Publish the full sequence (prompt + decode tokens) into the radix, which
        // now also captures the recurrent sidecar at that exact boundary (see
        // `publish_prefix_blocks`). Publishing runs unconditionally — including a
        // restore-derived turn — so an agentic follow-up matches THROUGH the
        // previous turn's generated tokens and restores instead of re-prefilling.
        let full_tokens = request.committed_tokens();
        // Write the finish frontier (generated content + live carry at the exact
        // finish position) through to the content-keyed prefix store while the
        // slot's device state is still resident. BEFORE publish_prefix_blocks:
        // it publishes PROVISIONAL entries that publish's save_prefix_sidecar
        // confirm/repair then reconciles to the radix's canonical ids (the same
        // path the prefill publish rides). Best-effort; default no-op (only DSv4
        // under --dsv4-decode-reuse captures).
        let slot_pages = self.kv.page_indices(slot).to_vec();
        if let Err(err) = self
            .executor
            .capture_finish_frontier(slot, &full_tokens, &slot_pages)
        {
            log::warn!("finish frontier capture failed for slot {slot}: {err:#}");
        }
        self.publish_prefix_blocks(slot, &full_tokens);
        // free_slot BEFORE release_reused_prefix: reclaim_page sees page_refs>0
        // and skips retained prefix pages; release_reused_prefix then drops the
        // last ref and the page enters the free pool exactly once. Reversed order
        // caused a double-push: release_pages dropped page_refs to 0 and pushed
        // to free, then reclaim_page (page_refs absent → 0) pushed again.
        self.free_slot_pages(slot);
        self.release_reused_prefix(&request.reused_prefix_pages);
        self.evict_prefix_cache_if_below_low_water();
        self.record_completed(request.handle, request.into());
    }

    /// Cancel `handle`: drop it from `waiting` if not yet admitted, or free
    /// its slot (same release path as a natural finish) if active/decoding.
    /// A no-op if `handle` already finished or is unknown — safe to call
    /// unconditionally on a request that may have already completed on its
    /// own (e.g. a client-disconnect guard that fires after the stream
    /// already ended naturally).
    ///
    /// MULTIPROC INVARIANT: like admission, this mutates scheduler-visible
    /// state (`waiting`, `active`, KV page counts) — call it identically, on
    /// the same tick, on every rank in an SPMD group, or ranks desync (the
    /// same hazard class as the 2026-07-05 TP=4 admission livelock; see
    /// docs/experience/errors/2026-07-05-multiproc-lockstep-ack-hang-no-timeout.md).
    /// The multiproc driver carries cancellations through the same
    /// `TickAdmissions` broadcast as new admissions for exactly this reason.
    pub fn cancel_request(&mut self, handle: RequestHandle) -> bool {
        if let Some(pos) = self.waiting.iter().position(|r| r.handle == handle) {
            let request = self.waiting.remove(pos).expect("position found above");
            self.abort_waiter(request);
            return true;
        }
        if let Some(&slot) = self
            .active
            .iter()
            .find(|(_, r)| r.handle == handle)
            .map(|(slot, _)| slot)
        {
            self.finish_slot(slot, FinishReason::Abort);
            return true;
        }
        false
    }

    /// Cancel every waiting + active request ([`Self::cancel_request`] per
    /// handle, same release path). Orphan sweep for a caller that knows all
    /// clients are gone (e.g. the OPD round-loop quiesce after its cc
    /// children exited). Same multiproc invariant as `cancel_request`.
    pub fn cancel_all_requests(&mut self) -> Vec<RequestHandle> {
        let handles: Vec<RequestHandle> = self
            .waiting
            .iter()
            .map(|r| r.handle)
            .chain(self.active.values().map(|r| r.handle))
            .collect();
        for &handle in &handles {
            self.cancel_request(handle);
        }
        handles
    }

    /// OPD writeback bracket: switch to `Quiesced` and cancel all in-flight
    /// work, atomically on the engine thread. The serve loop reads the mode to
    /// defer draining submissions until [`Self::resume_serving`]. Returns the
    /// cancelled handles (as `cancel_all_requests`).
    pub fn quiesce(&mut self) -> Vec<RequestHandle> {
        self.mode = EngineMode::Quiesced;
        self.cancel_all_requests()
    }

    /// Re-arm serving after the OPD writeback bracket (once the KV pool is
    /// re-acquired). Idempotent.
    pub fn resume_serving(&mut self) {
        self.mode = EngineMode::Serving;
    }

    /// True while in the OPD writeback bracket (KV pool released). The serve
    /// loop defers submission admission until [`Self::resume_serving`].
    #[must_use]
    pub fn is_quiesced(&self) -> bool {
        self.mode == EngineMode::Quiesced
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

        // TP-synced: a rank-local `free_pages()` can differ across ranks (e.g.
        // per-rank KV-tier host-demote residuals), and this value gates the
        // same Admit/Throttle decision as the `cached_prefix_match_len`
        // collective below — a diverging decision means one rank stops
        // calling that collective while another keeps calling it every tick,
        // a permanent cross-rank admission livelock (2026-07-05 TP=4 hang,
        // docs/experience/errors/2026-07-05-multiproc-lockstep-ack-hang-no-timeout.md).
        let mut remaining_pages = self.executor.tp_sync_min(self.kv.free_pages())?;
        self.evict_prefix_cache_if_below_low_water();
        // Nothing to admit — both loops below are waiter-driven, so skip the
        // per-tick free-slot scan. Placed AFTER the collective so every rank
        // still issues it on every tick.
        if self.waiting.is_empty() {
            return Ok(());
        }

        let mut free_slots = self.free_slots();
        let running_cap = self.config.running_cap();
        let mut remaining_prefill_tokens = self.config.prefill_step_budget();
        let mut active_prefills = self.active_prefill_count();
        let max_prefills = self.config.max_concurrent_prefill();

        while self.active.len() < running_cap {
            let Some(&slot) = free_slots.first() else {
                break;
            };
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
                // Rejected: the front waiter is gone (failed, not admitted) —
                // no slot consumed, keep going against the new front.
                AdmitOutcome::Rejected => {}
                AdmitOutcome::NoWaiter | AdmitOutcome::Throttled => break,
            }
        }

        // P5 — running-cap oversubscription. Once the scheduler cap is full,
        // waiters may rotate in by parking the longest-running decode's
        // whole-slot image. Executor capacity stays independent from this cap.
        if self.config.slot_oversubscription
            && self.executor.kv_slot_tier_enabled()
            && self.active.len() >= running_cap
        {
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
        // Matching runs over the committed stream (#156); borrowed unless the
        // candidate is a recompute-resumed victim with generated tokens.
        let reuse_matched_len = if self.config.enable_prefix_cache {
            let committed = candidate.committed_cow();
            let matched = self.radix.peek_longest_prefix_match(&committed);
            let prefix_match = self.clamp_prefix_to_backend(matched, &committed);
            // Backends without page-radix reuse (DSv4) may still hold a
            // position-0 whole-slot prefix image. The page route reports
            // `matched_len == 0` for them, so budget the prefill/pages against
            // the executor's cached prefix length when it is longer. This only
            // affects budgeting; the actual restore happens at attach below.
            // Skipped for swap re-admissions: they restore the FULL sequence
            // via `restore_swapped_slot` (consuming the slot) and never take
            // the cached-prefix attach path, so the cached length is
            // irrelevant to their prefill (which is 0).
            let cached = if candidate.swap_key.is_none() {
                self.executor
                    .cached_prefix_match_len(&committed)?
                    .min(committed.len())
            } else {
                0
            };
            prefix_match.matched_len.max(cached)
        } else {
            0
        };
        let prefill_tokens = candidate.committed_len().saturating_sub(reuse_matched_len);
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
                // `self.active.is_empty()`: the pool is completely idle right
                // now, so nothing else could ever finish and free more pages
                // — this candidate structurally exceeds the pool's total
                // capacity and retrying it forever would hang every later
                // request queued behind it (2026-07-05 round 5 — see
                // docs/experience/errors/2026-07-05-multiproc-lockstep-ack-hang-no-timeout.md).
                // Conservative on purpose: with other requests still active,
                // Throttle as before — they may free enough pages on finish.
                if self.active.is_empty() {
                    let request = self
                        .waiting
                        .pop_front()
                        .expect("waiting.front() was Some above");
                    log::warn!(
                        "admission reject: request needs {pages_needed} KV pages, pool has \
                         {remaining_pages} free with no other request active (prompt_len={})",
                        request.prompt_len()
                    );
                    self.abort_waiter(request);
                    return Ok(AdmitOutcome::Rejected);
                }
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
        } else {
            let committed = request.committed_tokens();
            let prefix_match = if self.config.enable_prefix_cache {
                // Tier-aware: demoted blocks in the match are promoted
                // back into fresh pages here, so attach sees a
                // resident-only match.
                self.lookup_prefix_for_attach(&committed)
            } else {
                PrefixMatch::empty()
            };
            self.attach_prefix_to_request(slot, &mut request, &committed, prefix_match)?;
            self.record_attached_prefix_metrics(request.reused_prefix_pages.len());
        }

        *remaining_pages = remaining_pages.saturating_sub(pages_needed);
        if matches!(request.phase, RequestPhase::Prefilling { .. }) {
            *remaining_prefill_tokens = remaining_prefill_tokens.saturating_sub(
                request
                    .committed_len()
                    .saturating_sub(request.prefill_start_pos),
            );
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

    /// P5 oversubscription trigger: free a running slot by parking the
    /// longest-running decode (smallest `admit_seq`) and admit a waiter in its
    /// place, capped at `running_cap/4` (>=1) preemptions per call. Terminates:
    /// each iteration either demotes+admits one (decrementing the bounded cap)
    /// or breaks (no eligible victim / waiter throttled / queue empty).
    fn admit_via_oversubscription(
        &mut self,
        remaining_prefill_tokens: &mut usize,
        active_prefills: &mut usize,
        max_prefills: usize,
        remaining_pages: &mut usize,
    ) -> Result<()> {
        let cap = self.config.running_cap().div_ceil(4).max(1);
        let mut preempted = 0usize;
        while preempted < cap && !self.waiting.is_empty() {
            let Some(victim_slot) = self.oversubscription_victim() else {
                break;
            };
            // PARK-OR-NOTHING: demote first; a refused park leaves the victim
            // running and stops the rotation this tick (retrying next tick is
            // the same refusal — the old reset-to-recompute path livelocked).
            if !self.try_park_for_oversubscription(victim_slot) {
                break;
            }
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
                // Demoted but the waiter still won't fit (throttle), vanished,
                // or was rejected outright: stop rather than churn the parked
                // victim back and forth.
                AdmitOutcome::Throttled | AdmitOutcome::NoWaiter | AdmitOutcome::Rejected => break,
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

    /// Allocate KV for every plan row. `fit_plan_to_kv_pages` sizes the plan
    /// so this succeeds; if an alloc still fails (capacity drift), degrade
    /// the row — shed a prefill chunk (it retries next tick) or park the
    /// decode victim (#162 path) — never propagate: a step-loop alloc error
    /// unwinds the whole TP worker group (#164). Lockstep-safe: pool state
    /// is identical across ranks (same admissions + plans every tick), so
    /// every rank degrades the same rows.
    fn allocate_for_plan(&mut self, plan: &mut ForwardPlan) {
        // Device-pool budget gate: backends with device pools separate from
        // `self.kv` (Qwen3.6 `full_attn_kv` + recall keepalive, DSv4
        // demand-paged FlashMLA bands #160) alloc per row inside `submit`,
        // where a failure is engine-fatal. `kv_device_fit` gives a PER-ROW
        // verdict in shed-priority order (decode first: keeping a running
        // decode alive is the cheap row, a shed prefill chunk retries next
        // tick for free) BEFORE the host alloc, so device exhaustion
        // degrades here exactly like host exhaustion — exactly the unfit
        // rows park/shed; later fitting rows keep running (a stuck row must
        // not starve the rest of the batch). Inert backends skip the gate —
        // no demand rows are built.
        if self.executor.kv_device_gate_active() {
            let page_size = self.kv.page_size();
            self.device_demand_scratch.clear();
            self.device_unfit_scratch.clear();
            self.device_demand_scratch
                .extend(plan.decode_rows.iter().map(|row| DeviceRowDemand {
                    slot: row.slot,
                    target_tokens: row.kv_seq_len + 1,
                    pages_hint: 1,
                }));
            self.device_demand_scratch
                .extend(plan.prefill_rows.iter().map(|row| DeviceRowDemand {
                    slot: row.slot,
                    target_tokens: row.total_tokens,
                    pages_hint: row.tokens.len().div_ceil(page_size) + 1,
                }));
            self.executor
                .kv_device_fit(&self.device_demand_scratch, &mut self.device_unfit_scratch);
            if !self.device_unfit_scratch.is_empty() {
                let mut idx = 0;
                plan.decode_rows.retain(|row| {
                    let keep = !self.device_unfit_scratch.contains(&idx);
                    idx += 1;
                    if !keep {
                        log::warn!(
                            "device KV pool exhausted for decode row (slot {}); \
                             parking the request (#164 backstop)",
                            row.slot
                        );
                        self.requeue_preempted_decode(row.slot);
                    }
                    keep
                });
                plan.prefill_rows.retain(|row| {
                    let keep = !self.device_unfit_scratch.contains(&idx);
                    idx += 1;
                    if !keep {
                        log::warn!(
                            "device KV pool exhausted for prefill chunk (slot {}); \
                             shedding the chunk (#164 backstop)",
                            row.slot
                        );
                    }
                    keep
                });
            }
        }
        plan.decode_rows
            .retain(|row| match self.alloc_with_prefix_reclaim(row.slot, 1) {
                Ok(()) => true,
                Err(err) => {
                    log::warn!(
                        "KV alloc failed for decode row (slot {}): {err:#}; \
                         parking the request (#164 backstop)",
                        row.slot
                    );
                    self.requeue_preempted_decode(row.slot);
                    false
                }
            });
        plan.prefill_rows.retain(|row| {
            match self.alloc_with_prefix_reclaim(row.slot, row.tokens.len()) {
                Ok(()) => true,
                Err(err) => {
                    log::warn!(
                        "KV alloc failed for prefill chunk (slot {}): {err:#}; \
                         shedding the chunk (#164 backstop)",
                        row.slot
                    );
                    false
                }
            }
        });
        plan.mode = planner::plan_mode(plan.prefill_rows.is_empty(), plan.decode_rows.is_empty());
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
    // Higher priority sorts first; on a tie the bias decides whether `incoming`
    // precedes an equal `queued`. (Reuse-based tiebreaks were dead: the reuse
    // hint is only known post-admit, so every waiter compares as default here.)
    match incoming.priority.cmp(&queued.priority) {
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
    } else if !sampling.ignore_eos && tail_is_degenerate_loop(&request.generated_tokens) {
        Some(FinishReason::Stop)
    } else {
        None
    }
}

/// Early-stop a runaway: the last [`LOOP_TAIL`] tokens collapsed into an exact
/// period-`p` (`p ≤ 8`) repetition — a token-0 spin, a `a b a b …`, etc. A real
/// generation of that length is never a pure short cycle, so the false-positive
/// floor is high; gated off with `ignore_eos` (benchmarks that want raw length).
fn tail_is_degenerate_loop(tokens: &[u32]) -> bool {
    const LOOP_TAIL: usize = 48;
    if tokens.len() < LOOP_TAIL {
        return false;
    }
    let tail = &tokens[tokens.len() - LOOP_TAIL..];
    (1..=8).any(|p| tail.chunks_exact(p).all(|c| c == &tail[..p]))
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
                .keys()
                .filter(|&&page| self.page_is_evictable(page))
                .count()
        }

        fn page_is_evictable(&self, page: u32) -> bool {
            self.page_ref_counts.get(&page).copied().unwrap_or(0) > 0
                && self.page_attach_counts.get(&page).copied().unwrap_or(0) == 0
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

    pub(super) struct BackgroundPublishExecutor {
        pub(super) inner: MockExecutor,
        pub(super) ready: std::rc::Rc<std::cell::Cell<bool>>,
        pub(super) pending: std::rc::Rc<std::cell::Cell<bool>>,
    }

    impl BackendExecutor for BackgroundPublishExecutor {
        type Inflight = MockInflight;

        fn submit(&mut self, plan: &ForwardPlan, kv: &mut dyn KvPool) -> Result<Self::Inflight> {
            self.inner.submit(plan, kv)
        }

        fn poll(&mut self, inflight: Self::Inflight) -> Result<PollResult<Self::Inflight>> {
            self.inner.poll(inflight)
        }

        fn poll_background(&mut self) -> Result<()> {
            if self.pending.replace(false) {
                self.ready.set(true);
            }
            Ok(())
        }

        fn reusable_prefix_blocks(&self, blocks: &[PrefixBlock]) -> usize {
            if self.ready.get() {
                pages_only_reusable_prefix_blocks(blocks, |_| false)
            } else {
                0
            }
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

    #[derive(Debug, Clone)]
    pub(super) struct PlanTokenCapExecutor {
        inner: MockExecutor,
        cap: usize,
        pub(super) capability_reads: std::rc::Rc<std::cell::Cell<usize>>,
        pub(super) plans: std::rc::Rc<std::cell::RefCell<Vec<(bool, usize)>>>,
    }

    impl PlanTokenCapExecutor {
        pub(super) fn with_cap(cap: usize) -> Self {
            Self {
                inner: MockExecutor::ready(),
                cap,
                capability_reads: std::rc::Rc::new(std::cell::Cell::new(0)),
                plans: std::rc::Rc::new(std::cell::RefCell::new(Vec::new())),
            }
        }
    }

    impl BackendExecutor for PlanTokenCapExecutor {
        type Inflight = MockInflight;

        fn submit(&mut self, plan: &ForwardPlan, kv: &mut dyn KvPool) -> Result<Self::Inflight> {
            let tokens = plan.decode_rows.len()
                + plan
                    .prefill_rows
                    .iter()
                    .map(|row| row.tokens.len())
                    .sum::<usize>();
            self.plans.borrow_mut().push((
                !plan.decode_rows.is_empty() && !plan.prefill_rows.is_empty(),
                tokens,
            ));
            self.inner.submit(plan, kv)
        }

        fn poll(&mut self, inflight: Self::Inflight) -> Result<PollResult<Self::Inflight>> {
            self.inner.poll(inflight)
        }

        fn max_tokens_per_step(&self) -> usize {
            self.capability_reads
                .set(self.capability_reads.get().saturating_add(1));
            self.cap
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

    /// Mock executor whose restore boundaries exist only at every
    /// `align_blocks` pages, mirroring DSv4's ring-aligned commit points:
    /// `reusable_prefix_blocks` floors the leading run to that alignment.
    #[derive(Debug, Clone)]
    pub(super) struct AlignedPrefixExecutor {
        inner: MockExecutor,
        align_blocks: usize,
    }

    impl AlignedPrefixExecutor {
        pub(super) fn with_align_blocks(align_blocks: usize) -> Self {
            Self {
                inner: MockExecutor::ready(),
                align_blocks,
            }
        }
    }

    impl BackendExecutor for AlignedPrefixExecutor {
        type Inflight = MockInflight;

        fn submit(&mut self, plan: &ForwardPlan, kv: &mut dyn KvPool) -> Result<Self::Inflight> {
            self.inner.submit(plan, kv)
        }

        fn poll(&mut self, inflight: Self::Inflight) -> Result<PollResult<Self::Inflight>> {
            self.inner.poll(inflight)
        }

        fn reusable_prefix_blocks(&self, blocks: &[PrefixBlock]) -> usize {
            let n = pages_only_reusable_prefix_blocks(blocks, |_| false);
            n - n % self.align_blocks.max(1)
        }
    }

    /// `(prefix_pages, slot_pages, newly_cached)` recorded per
    /// `save_prefix_sidecar` call.
    pub(super) type SidecarSaves =
        std::rc::Rc<std::cell::RefCell<Vec<(Vec<u32>, Vec<u32>, Vec<u32>)>>>;

    /// Hybrid-sidecar mirror for the #155 lifetime bug: `restore_prefix_sidecar`
    /// always MISSES (the engine falls back to full recompute), and every
    /// `save_prefix_sidecar` call is recorded so a test can assert which pages
    /// the sidecar's eviction lifetime was keyed to.
    #[derive(Debug, Clone)]
    pub(super) struct SidecarMissExecutor {
        inner: MockExecutor,
        pub(super) saves: SidecarSaves,
    }

    impl SidecarMissExecutor {
        pub(super) fn new() -> Self {
            Self {
                inner: MockExecutor::ready(),
                saves: std::rc::Rc::new(std::cell::RefCell::new(Vec::new())),
            }
        }
    }

    impl BackendExecutor for SidecarMissExecutor {
        type Inflight = MockInflight;

        fn submit(&mut self, plan: &ForwardPlan, kv: &mut dyn KvPool) -> Result<Self::Inflight> {
            self.inner.submit(plan, kv)
        }

        fn poll(&mut self, inflight: Self::Inflight) -> Result<PollResult<Self::Inflight>> {
            self.inner.poll(inflight)
        }

        fn reusable_prefix_blocks(&self, blocks: &[PrefixBlock]) -> usize {
            pages_only_reusable_prefix_blocks(blocks, |_| false)
        }

        fn restore_prefix_sidecar(
            &mut self,
            _slot: usize,
            _tokens: &[u32],
            _matched_len: usize,
            _prefix_pages: &[u32],
        ) -> Result<usize> {
            bail!("sidecar miss")
        }

        fn save_prefix_sidecar(
            &mut self,
            _slot: usize,
            _tokens: &[u32],
            _matched_len: usize,
            prefix_pages: &[u32],
            slot_pages: &[u32],
            newly_cached: &[u32],
        ) -> Result<()> {
            self.saves.borrow_mut().push((
                prefix_pages.to_vec(),
                slot_pages.to_vec(),
                newly_cached.to_vec(),
            ));
            Ok(())
        }
    }

    /// Mock executor reporting a coarser-than-page restore alignment plus a
    /// per-forward chunk capability, mirroring DSv4's ring-snapshot
    /// `sliding_window` unit — proves the planner combines the alignment with
    /// KV page size via LCM, not `.max()`, and caps chunks to the capability.
    #[derive(Debug, Clone)]
    pub(super) struct RestoreAlignmentExecutor {
        inner: MockExecutor,
        alignment: usize,
        max_chunk: usize,
    }

    impl RestoreAlignmentExecutor {
        pub(super) fn with_alignment_and_chunk(alignment: usize, max_chunk: usize) -> Self {
            Self {
                inner: MockExecutor::ready(),
                alignment,
                max_chunk,
            }
        }
    }

    impl BackendExecutor for RestoreAlignmentExecutor {
        type Inflight = MockInflight;

        fn submit(&mut self, plan: &ForwardPlan, kv: &mut dyn KvPool) -> Result<Self::Inflight> {
            self.inner.submit(plan, kv)
        }

        fn poll(&mut self, inflight: Self::Inflight) -> Result<PollResult<Self::Inflight>> {
            self.inner.poll(inflight)
        }

        fn prefill_restore_boundary_alignment(&self) -> usize {
            self.alignment
        }

        fn max_prefill_chunk(&self) -> usize {
            self.max_chunk
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
        logits: std::rc::Rc<[f32]>,
        observed: std::rc::Rc<std::cell::RefCell<Vec<(usize, u64, SamplingParams)>>>,
    }

    impl SamplingExecutor {
        pub(super) fn new(logits: Vec<f32>) -> Self {
            Self {
                logits: std::rc::Rc::from(logits),
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

    /// Echo executor with settable per-pool device headroom and per-row need,
    /// mirroring a backend (Qwen3.6 `full_attn_kv`, DSv4 per-layer band
    /// pools) whose device pools run dry while the host pool still has
    /// pages. Exercises the `allocate_for_plan` `kv_device_fit` gate: each
    /// `(free, need)` pool pair is checked independently per row (#160 —
    /// pairing, not scalar extrema); `need = None` charges the engine's
    /// `pages_hint`. Empty pools = gate inert.
    #[derive(Clone, Default)]
    pub(super) struct DeviceBudgetExecutor {
        #[allow(clippy::type_complexity)]
        pub(super) pools: std::rc::Rc<std::cell::RefCell<Vec<(usize, Option<usize>)>>>,
    }

    impl DeviceBudgetExecutor {
        pub(super) fn set_pools(&self, pools: &[(usize, Option<usize>)]) {
            *self.pools.borrow_mut() = pools.to_vec();
        }
    }

    impl BackendExecutor for DeviceBudgetExecutor {
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

        fn kv_device_gate_active(&self) -> bool {
            !self.pools.borrow().is_empty()
        }

        fn kv_device_fit(&self, rows: &[infer_seam::DeviceRowDemand], unfit: &mut Vec<usize>) {
            let mut pools = self.pools.borrow().clone();
            for (idx, row) in rows.iter().enumerate() {
                let need = |pool_need: Option<usize>| pool_need.unwrap_or(row.pages_hint);
                if pools.iter().any(|&(free, n)| need(n) > free) {
                    unfit.push(idx);
                    continue;
                }
                for (free, n) in &mut pools {
                    *free -= need(*n);
                }
            }
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

        /// Pre-seed the device-side materialized length for a directly
        /// injected (already-prefilled) request.
        pub(super) fn seed_materialized(&self, slot: usize, len: usize) {
            self.materialized.borrow_mut().insert(slot, len);
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
        /// Periodic-snapshot PARTIAL restore: when `> 0`, `restore_prefix_sidecar`
        /// restores at a boundary `B = matched_len - partial_gap < matched_len`
        /// instead of `matched_len`, modelling a cross-conversation hit where no
        /// sidecar existed exactly at the radix match.
        partial_gap: usize,
        /// `(matched_len, B)` of the most recent partial restore — the test asserts
        /// the engine set `prefill_start_pos` to `B`.
        last_restore: std::rc::Rc<std::cell::RefCell<Option<(usize, usize)>>>,
    }

    impl HybridReprefillMirror {
        pub(super) fn with_partial_gap(rewind_on_attach: bool, partial_gap: usize) -> Self {
            Self {
                materialized: std::rc::Rc::new(std::cell::RefCell::new(
                    std::collections::HashMap::new(),
                )),
                decode_log: std::rc::Rc::new(std::cell::RefCell::new(Vec::new())),
                rewind_on_attach,
                partial_gap,
                last_restore: std::rc::Rc::new(std::cell::RefCell::new(None)),
            }
        }

        pub(super) fn decode_log(&self) -> Vec<(usize, usize, usize)> {
            self.decode_log.borrow().clone()
        }

        pub(super) fn last_restore(&self) -> Option<(usize, usize)> {
            *self.last_restore.borrow()
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
        ) -> Result<usize> {
            // Periodic-snapshot partial restore: land on a lower boundary `B =
            // matched_len - partial_gap` (the caller keeps `partial_gap` a page
            // multiple), so the engine truncates the over-attached pages and
            // re-prefills [B..prompt].
            let restored = if self.partial_gap > 0 && matched_len > self.partial_gap {
                let b = matched_len - self.partial_gap;
                *self.last_restore.borrow_mut() = Some((matched_len, b));
                b
            } else {
                matched_len
            };
            if self.rewind_on_attach {
                self.materialized.borrow_mut().insert(slot, restored);
            }
            Ok(restored)
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
        AlignedPrefixExecutor, BackgroundPublishExecutor, DeviceBudgetExecutor,
        DeviceMirrorExecutor, HoldGovernor, HybridReprefillMirror, LimitedPrefixExecutor,
        MockExecutor, MockKvPool, PlanTokenCapExecutor, RestoreAlignmentExecutor, SamplingExecutor,
        SidecarMissExecutor, SingleRowExecutor, SlotTierMockExecutor, SpecMirrorExecutor,
        StopTokenExecutor, TierMockExecutor, TokenBudgetGovernor, WarmupCountingExecutor,
    };
    use super::*;

    /// The grammar hook must be pumped once at admit and once per committed
    /// token, and its mask must reach the plan the executor samples from —
    /// break any link and generation silently ignores the constraint.
    #[test]
    fn grammar_hook_pumps_at_admit_and_every_token() -> Result<()> {
        use std::sync::{Arc, Mutex};
        let seen: Arc<Mutex<Vec<Option<u32>>>> = Arc::new(Mutex::new(Vec::new()));
        let log = Arc::clone(&seen);
        let hook = GrammarHook(Arc::new(move |t| {
            log.lock().unwrap().push(t);
            Some(Arc::from([0xABCDu32].as_slice()))
        }));

        let mut engine = Engine::new(MockExecutor::ready(), MockKvPool::new(1), 1);
        let handle = engine.submit_request_with_options(
            vec![1, 2, 3],
            3,
            RequestOptions {
                grammar: Some(hook),
                ..RequestOptions::default()
            },
        );
        engine.step()?;
        let plan = engine.build_forward_plan();
        if let Some(row) = plan.decode_rows.first() {
            assert_eq!(
                row.params.grammar_bitmask.as_deref(),
                Some([0xABCDu32].as_slice()),
                "the refreshed mask must ride the decode row"
            );
        }
        engine.run_to_idle()?;
        let generated = engine
            .completed(handle)
            .expect("completes")
            .generated_tokens
            .len();

        let calls = seen.lock().unwrap();
        assert_eq!(calls[0], None, "admit primes the opening mask");
        assert_eq!(calls.len(), generated + 1);
        assert!(calls[1..].iter().all(Option::is_some));
        Ok(())
    }

    #[test]
    fn degenerate_loop_tail_is_detected() {
        // Below the window: never fires.
        assert!(!tail_is_degenerate_loop(&[0u32; 47]));
        // Token-0 spin (period 1) and a 2-cycle: both caught at 48.
        assert!(tail_is_degenerate_loop(&[0u32; 48]));
        let two_cycle: Vec<u32> = (0..48).map(|i| (i % 2) as u32).collect();
        assert!(tail_is_degenerate_loop(&two_cycle));
        // A real varied tail (strictly increasing) is not a short cycle.
        let varied: Vec<u32> = (0..48).collect();
        assert!(!tail_is_degenerate_loop(&varied));
        // Only the last 48 matter: a healthy prefix + degenerate tail still fires.
        let mut mixed: Vec<u32> = (0..100).collect();
        mixed.extend(std::iter::repeat_n(7u32, 48));
        assert!(tail_is_degenerate_loop(&mixed));
    }

    /// Device-pool exhaustion (the backend pool running dry while the host pool
    /// still has pages — Qwen3.6 recall keepalive) must degrade like host
    /// exhaustion: shed prefills / park decodes, never a fatal step error
    /// (2026-07-23 pod: TokenKVPool out-of-pages inside submit killed the
    /// engine thread mid-rollout).
    #[test]
    fn device_pool_exhaustion_degrades_instead_of_fatal() -> Result<()> {
        let executor = DeviceBudgetExecutor::default();
        let pools = executor.clone();
        pools.set_pools(&[(0, None)]);
        let mut engine =
            Engine::with_config(executor, HostPagedKvPool::new(1, 256, 16), test_config(1));

        let handle = engine.submit_request((1..=20).collect(), 4);
        // Zero device pages: the prefill chunk sheds every tick — no progress,
        // no error, request stays live.
        for _ in 0..3 {
            engine.step()?;
        }
        assert!(engine.completed(handle).is_none());
        assert!(!engine.is_idle());

        // Pages free up mid-prefill: the request completes normally.
        pools.set_pools(&[(256, None)]);
        engine.run_to_idle()?;
        let completed = engine
            .completed(handle)
            .expect("completes after the device budget lifts");
        assert_eq!(completed.generated_tokens.len(), 4);

        // Decode side: a running decode degrades (#162 park / re-admit) when
        // the device pool dries up mid-generation — steps stay Ok with no
        // progress — and resumes to completion when it refills.
        let handle2 = engine.submit_request((1..=20).collect(), 8);
        engine.step()?;
        pools.set_pools(&[(0, None)]);
        for _ in 0..3 {
            engine.step()?;
        }
        assert!(engine.completed(handle2).is_none());
        assert!(!engine.is_idle());
        pools.set_pools(&[(256, None)]);
        engine.run_to_idle()?;
        let resumed = engine
            .completed(handle2)
            .expect("gated decode resumes and completes");
        assert_eq!(resumed.generated_tokens.len(), 8);
        Ok(())
    }

    /// A backend-projected per-row demand (DSv4 demand-paged band growth,
    /// #160) must override the engine's `pages_hint`: a row whose projected
    /// growth exceeds the device headroom parks even though the naive chunk
    /// formula would fit, and admits once the projection drops (band already
    /// resident).
    #[test]
    fn device_row_projection_overrides_engine_formula() -> Result<()> {
        let executor = DeviceBudgetExecutor::default();
        let pools = executor.clone();
        pools.set_pools(&[(8, Some(64))]);
        let mut engine =
            Engine::with_config(executor, HostPagedKvPool::new(1, 256, 16), test_config(1));

        // 20-token prompt: engine formula needs ceil(20/16)+1 = 3 ≤ 8 pages,
        // but the projected band growth (64) exceeds the headroom — the chunk
        // sheds every tick instead of exhausting `band_extend` inside submit.
        let handle = engine.submit_request((1..=20).collect(), 4);
        for _ in 0..3 {
            engine.step()?;
        }
        assert!(engine.completed(handle).is_none());
        assert!(!engine.is_idle());

        // Band now resident (projection 0): the request completes normally.
        pools.set_pools(&[(8, Some(0))]);
        engine.run_to_idle()?;
        let completed = engine
            .completed(handle)
            .expect("completes once the projected band growth fits");
        assert_eq!(completed.generated_tokens.len(), 4);
        Ok(())
    }

    /// Codex P1 regression (#160): heterogeneous pools must pair need and
    /// headroom PER POOL. A saturated sliding-window layer (free=0, need=0)
    /// next to a compressed layer with headroom (free>0, need=1) must ADMIT —
    /// the old scalar min(free)/max(need) projection read this as permanent
    /// exhaustion and parked every row forever.
    #[test]
    fn device_fit_pairs_need_with_pool_not_extrema() -> Result<()> {
        let executor = DeviceBudgetExecutor::default();
        let pools = executor.clone();
        pools.set_pools(&[(0, Some(0)), (16, Some(1))]);
        let mut engine =
            Engine::with_config(executor, HostPagedKvPool::new(1, 256, 16), test_config(1));

        let handle = engine.submit_request((1..=20).collect(), 4);
        engine.run_to_idle()?;
        let completed = engine
            .completed(handle)
            .expect("saturated zero-need pool must not read as exhaustion");
        assert_eq!(completed.generated_tokens.len(), 4);
        Ok(())
    }

    /// Codex P1 regression (#160): verdicts are per-row, not first-unfit +
    /// drain-tail. A stuck big-need row (prefill A, 8 pages > 4 free) must
    /// not starve a later fitting row (prefill B, 3 pages) — B completes
    /// while A sheds, and A resumes when headroom returns.
    #[test]
    fn device_fit_unfit_row_does_not_starve_later_fitting_rows() -> Result<()> {
        let executor = DeviceBudgetExecutor::default();
        let pools = executor.clone();
        pools.set_pools(&[(4, None)]);
        let mut engine =
            Engine::with_config(executor, HostPagedKvPool::new(2, 256, 16), test_config(2));

        let big = engine.submit_request((1..=100).collect(), 4);
        let small = engine.submit_request((1..=20).collect(), 4);
        for _ in 0..8 {
            engine.step()?;
        }
        let completed = engine
            .completed(small)
            .expect("later fitting row must keep running past a stuck unfit row");
        assert_eq!(completed.generated_tokens.len(), 4);
        assert!(engine.completed(big).is_none());
        assert!(!engine.is_idle());

        pools.set_pools(&[(256, None)]);
        engine.run_to_idle()?;
        let resumed = engine
            .completed(big)
            .expect("stuck row resumes once headroom returns");
        assert_eq!(resumed.generated_tokens.len(), 4);
        Ok(())
    }

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
        run_agentic_reprefill_gap(
            rewind_on_attach,
            turns,
            base_prompt,
            gen_per_turn,
            tool_tokens,
            0,
        )
    }

    fn run_agentic_reprefill_gap(
        rewind_on_attach: bool,
        turns: usize,
        base_prompt: usize,
        gen_per_turn: usize,
        tool_tokens: usize,
        partial_gap: usize,
    ) -> (HybridReprefillMirror, Result<()>) {
        let executor = HybridReprefillMirror::with_partial_gap(rewind_on_attach, partial_gap);
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
        let (probe, result) = run_agentic_reprefill(true, 6, 128, 5, 24);
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

    /// Periodic-snapshot PARTIAL restore: when the backend restores at a boundary
    /// `B < matched_len` (no sidecar existed exactly at the radix match — a
    /// cross-conversation hit), the engine must set `prefill_start_pos = B` and
    /// truncate the over-attached pages so the tail prefills `[B..prompt]`.
    #[test]
    fn partial_restore_reprefills_from_returned_boundary() -> Result<()> {
        // page_size 4. First prompt publishes an 8-token (2-page) prefix; the
        // second shares it. The hybrid mock restores PARTIALLY at
        // `B = matched_len - partial_gap = 8 - 4 = 4` (one page below the radix
        // match), modelling a cross-conversation hit with no sidecar exactly at
        // `matched_len`. The engine must truncate the over-attached second page
        // and set `prefill_start_pos = B`, re-prefilling `[B..prompt]`.
        let executor = HybridReprefillMirror::with_partial_gap(true, 4);
        let probe = executor.clone();
        let mut engine = Engine::with_config(
            executor,
            MockKvPool::with_capacity(2, 4, 16),
            test_config(2),
        );

        let first = engine.submit_request(vec![1, 2, 3, 4, 5, 6, 7, 8, 99], 1);
        engine.run_to_idle()?;
        assert_finished(engine.completed(first).expect("first completed"));
        let hit = engine
            .radix
            .peek_longest_prefix_match(&[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(hit.matched_len, 8, "2-page prefix published");

        let second = engine.submit_request(vec![1, 2, 3, 4, 5, 6, 7, 8, 100, 101], 1);
        engine.step()?;

        let (_, request) = engine
            .active
            .iter()
            .find(|(_, request)| request.handle == second)
            .expect("second admitted");
        // Partial restore fired (B = 8 - 4 = 4) and the engine re-enters prefill at B.
        let (matched_len, b) = probe.last_restore().expect("a partial restore occurred");
        assert_eq!(matched_len, 8);
        assert_eq!(b, 4);
        assert_eq!(
            request.prefill_start_pos, 4,
            "prefill_start_pos must be the restored boundary B, not matched_len"
        );

        engine.run_to_idle()?;
        assert_finished(engine.completed(second).expect("second completed"));
        Ok(())
    }

    /// cc-as-harness shape: an 8-page CONSTANT system prompt (cc's large prompt,
    /// scaled to the mock's regime) re-matched every turn plus multi-turn growth
    /// — 3.2× the base of `agentic_reprefill_keeps_seq_len_in_lockstep`. Drift is
    /// scale-independent (device-counter rewind), proven identical at base
    /// 40/64/96/128; this pins the multi-page-prefix re-match shape cc drives.
    /// Confirms `materialized == kv_seq_len` so cc rollouts feed the OPD writeback
    /// a non-drifting KV. Per-round serve RESTART separately keeps the cross-round
    /// adapter epoch fresh (token-keyed RadixCache has no epoch — issue #92).
    #[test]
    fn agentic_reprefill_cc_large_prompt_shape() {
        let (probe, result) = run_agentic_reprefill(true, 6, 128, 5, 24);
        result.expect("cc-shape big-prefix multi-turn must not drift seq_len");
        let log = probe.decode_log();
        assert!(
            !log.is_empty(),
            "expected decode rows across the cc-shape turns"
        );
        for (k, &(slot, materialized, kv_seq_len)) in log.iter().enumerate() {
            assert_eq!(slot, 0);
            assert_eq!(
                materialized, kv_seq_len,
                "decode {k}: materialized {materialized} != kv_seq_len {kv_seq_len} \
                 (cc large-prefix re-prefill drift)"
            );
        }
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
            engine.executor.demote_batches.contains(&2),
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
            engine.executor.promote_batches.contains(&2),
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

    /// Pod round-5 livelock regression: with a store that REFUSES every demote
    /// (page-size violation / full / zero budget), oversubscription must be
    /// park-or-nothing. The old path reset the failed park to recompute, and
    /// the running pair ping-ponged at the 8-token min slice forever (~2,060
    /// park→refuse→recompute cycles, zero completions). Now a refused park
    /// leaves the victim running: A completes untouched, then B admits and
    /// completes, with zero demotes recorded.
    #[test]
    fn oversubscription_refused_park_keeps_victim_running() -> Result<()> {
        let config = SchedulerConfig {
            slot_oversubscription: true,
            max_running_requests: Some(1),
            ..test_config(2)
        };
        let mut engine = Engine::with_config(
            SlotTierMockExecutor::rejecting_demotes(),
            MockKvPool::with_capacity(2, 8, 64),
            config,
        );
        let a = engine.submit_request((1..=8).collect(), OVERSUBSCRIPTION_MIN_SLICE + 6);
        for _ in 0..(OVERSUBSCRIPTION_MIN_SLICE + 3) {
            engine.step()?;
        }
        let b = engine.submit_request((20..=24).collect(), 3);
        // Old code never terminated here (reset→re-admit→reset). Bounded run.
        engine.run_to_idle()?;
        let done_a = engine.completed(a).expect("A completed");
        assert_finished(done_a);
        assert_eq!(
            done_a.generated_tokens.len(),
            OVERSUBSCRIPTION_MIN_SLICE + 6,
            "A ran to its cap without ever being reset"
        );
        let done_b = engine.completed(b).expect("B completed after A");
        assert_finished(done_b);
        let tier = engine.kv_tier_stats();
        assert_eq!(tier.demoted_slots, 0, "no successful park: {tier:?}");
        assert_eq!(tier.promoted_slots, 0);
        Ok(())
    }

    /// P5 running-cap trigger: with two physical slots but
    /// `max_running_requests=1`, admitting a second request parks the running
    /// decode's whole-slot image; the waiter runs, then the parked request
    /// resumes via promote with its generation intact. This proves the
    /// scheduler cap is independent from executor hot-workspace capacity.
    #[test]
    fn oversubscription_parks_running_decode_to_admit_waiter() -> Result<()> {
        let config = SchedulerConfig {
            slot_oversubscription: true,
            max_running_requests: Some(1),
            ..test_config(2)
        };
        let mut engine = Engine::with_config(
            SlotTierMockExecutor::enabled(),
            MockKvPool::with_capacity(2, 8, 64),
            config,
        );

        // Request A fills the scheduler cap and decodes past the min slice so it
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

        // Request B arrives; a physical slot is free, but the scheduler cap is
        // full, so the next admit oversubscribes: A is parked, then B is admitted.
        let b = engine.submit_request((20..=24).collect(), 3);
        engine.step()?;
        let tier = engine.kv_tier_stats();
        assert_eq!(
            tier.demoted_slots, 1,
            "A parked to free the running cap: {tier:?}"
        );
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
        assert_eq!(engine.active_count(), 1, "scheduler cap remains 1");
        // B holds the active lane — A genuinely yielded it.
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
            17,
            "fallback re-prefilled the committed stream (prompt + 1 generated)"
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
            17,
            "plain recompute re-prefilled the committed stream (prompt + 1 generated)"
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
        // Prompt was sealed into radix (L1 cache) during normal step — that's
        // always-on, independent of tier store. No tier demotion should occur.
        assert_eq!(
            engine.radix.cached_page_count(),
            2,
            "prompt sealed into L1 radix"
        );
        assert_eq!(
            engine.kv_tier_stats().demoted_pages,
            0,
            "no tier demote without store"
        );
        // Radix retains 2 cached prompt pages (L1 always-on); the rest freed.
        let cached = engine.radix.cached_page_count();
        assert_eq!(
            engine.kv_free_pages(),
            free_before - cached,
            "all pages freed except radix-cached"
        );

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

    /// 2026-07-05 round 5: a prompt that fits `max_prompt_tokens` but needs
    /// more KV pages than the pool could EVER provide must be rejected once
    /// the pool is idle, not retried (Throttled) forever — an unfittable
    /// request stuck in `waiting` would hang every request queued behind it.
    #[test]
    fn request_exceeding_total_pool_capacity_is_rejected_not_throttled_forever() -> Result<()> {
        let config = test_config(1);
        // page_size=1, total_pages=2: 5 prompt tokens + 1 max_tokens needs 6
        // pages, more than the pool could ever hold, even fully idle.
        let mut engine = Engine::with_config(
            MockExecutor::ready(),
            MockKvPool::with_capacity(1, 1, 2),
            config,
        );

        let handle = engine.submit_request(vec![1, 2, 3, 4, 5], 1);
        engine.admit_waiting()?;

        assert_eq!(engine.active_count(), 0, "must not have been admitted");
        assert_eq!(
            engine.waiting_count(),
            0,
            "must be rejected out of waiting, not left to retry forever"
        );
        let completed = engine.completed(handle).expect("request rejected");
        assert!(matches!(completed.finish, Some(FinishReason::Abort)));
        Ok(())
    }

    /// `InFlightGuard`'s cancellation propagation (2026-07-05 follow-up):
    /// a still-queued request must be droppable without ever occupying a
    /// slot.
    #[test]
    fn cancel_waiting_request_completes_it_aborted() -> Result<()> {
        let config = test_config(1);
        let mut engine = Engine::with_config(MockExecutor::ready(), MockKvPool::new(1), config);
        let first = engine.submit_request(vec![1], 4);
        let second = engine.submit_request(vec![2], 4);
        engine.admit_waiting()?;
        assert_eq!(engine.active_count(), 1, "only one slot");
        assert_eq!(engine.waiting_count(), 1, "second is still queued");

        assert!(engine.cancel_request(second));

        assert_eq!(engine.waiting_count(), 0);
        assert!(matches!(
            engine.completed(second).expect("cancelled").finish,
            Some(FinishReason::Abort)
        ));
        // Unaffected: the admitted request keeps running.
        assert_eq!(engine.active_count(), 1);
        assert!(engine.completed(first).is_none());
        Ok(())
    }

    /// Cancelling an ACTIVE (decoding) request must free its slot/KV pages
    /// through the same release path a natural finish uses, not leak them.
    #[test]
    fn cancel_active_request_frees_its_slot() -> Result<()> {
        let mut engine = Engine::new(MockExecutor::ready(), MockKvPool::new(1), 1);
        let handle = engine.submit_request(vec![1, 2, 3], 4);
        engine.step()?; // admit + run at least one step so pages are actually held
        assert_eq!(engine.active_count(), 1);
        let free_before = engine.kv_free_pages();

        assert!(engine.cancel_request(handle));

        assert_eq!(engine.active_count(), 0, "slot must be released");
        assert!(
            engine.kv_free_pages() > free_before,
            "cancelling must free the pages the active request held"
        );
        assert!(matches!(
            engine.completed(handle).expect("cancelled").finish,
            Some(FinishReason::Abort)
        ));
        Ok(())
    }

    /// Cancelling a handle that already finished, or was never submitted,
    /// must be a safe no-op — the client-disconnect guard fires unconditionally
    /// and cannot know whether the stream already ended naturally.
    #[test]
    fn cancel_already_completed_or_unknown_handle_is_a_noop() -> Result<()> {
        let mut engine = Engine::new(MockExecutor::ready(), MockKvPool::new(1), 1);
        let handle = engine.submit_request(vec![1], 1);
        engine.run_to_idle()?;
        assert!(engine.completed(handle).is_some(), "finished naturally");

        assert!(!engine.cancel_request(handle), "already completed");
        let unknown = engine.next_handle();
        assert!(!engine.cancel_request(unknown), "unknown handle");
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
        assert_eq!(
            requeued.generated_tokens,
            vec![9, 10],
            "recompute keeps committed tokens (#156)"
        );
        assert!(matches!(
            requeued.phase,
            RequestPhase::Prefilling { progress: 0 }
        ));
        Ok(())
    }

    /// #156: a decode preempted onto the PLAIN recompute path (no tier store,
    /// no whole-slot image) must resume at the committed position — the
    /// observer stream stays byte-continuous, with no token ever re-emitted.
    #[test]
    fn recompute_preemption_never_reemits_committed_tokens() -> Result<()> {
        let mut engine = Engine::with_config(
            MockExecutor::ready(),
            MockKvPool::with_capacity(1, 4, 32),
            test_config(1),
        );
        let observed = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let observed_sink = observed.clone();
        engine.set_token_observer(Box::new(move |_, token| {
            observed_sink.borrow_mut().push(token.token);
        }));

        let handle = engine.submit_request(vec![1, 2, 3, 4], 6);
        for _ in 0..3 {
            engine.step()?;
        }
        let (&slot, request) = engine.active.iter().next().expect("request active");
        assert!(matches!(request.phase, RequestPhase::Decoding));
        let committed_at_preempt = request.generated_tokens.clone();
        assert_eq!(committed_at_preempt, vec![5, 6]);

        engine.requeue_preempted_decode(slot);
        let requeued = engine.waiting.front().expect("victim requeued");
        assert!(requeued.swap_key.is_none(), "plain recompute path");
        assert_eq!(requeued.generated_tokens, committed_at_preempt);
        assert_eq!(engine.kv_system_metrics().fallback_recompute, 1);

        engine.run_to_idle()?;
        let done = engine.completed(handle).expect("completed");
        assert_finished(done);
        assert!(done.generated_tokens.starts_with(&committed_at_preempt));
        let streamed = observed.borrow().clone();
        assert_eq!(
            streamed, done.generated_tokens,
            "observer stream equals the final generation — nothing re-emitted"
        );
        Ok(())
    }

    /// #164: pool exhausted (free 0) with a mixed plan must never reach a
    /// failing `allocate_for_plan` — that error is fatal (TP group unwind).
    /// The prefill chunk is shed and the LAST decode row parks (requeues).
    #[test]
    fn kv_exhaustion_sheds_prefill_and_parks_last_decode_instead_of_fatal() -> Result<()> {
        let mut engine = Engine::with_config(
            MockExecutor::ready(),
            MockKvPool::with_capacity(2, 4, 1),
            test_config(2),
        );
        let decoding = engine.next_handle();
        let prefilling = engine.next_handle();

        let mut decode_req = RequestState::new(
            decoding,
            vec![1, 1, 1],
            RequestPriority::Normal,
            8,
            SamplingParams::default(),
        );
        decode_req.generated_tokens = vec![9];
        decode_req.phase = RequestPhase::Decoding;
        let mut prefill_req = RequestState::new(
            prefilling,
            vec![2; 8],
            RequestPriority::Normal,
            8,
            SamplingParams::default(),
        );
        prefill_req.phase = RequestPhase::Prefilling { progress: 0 };

        engine.kv.alloc(0, 4)?; // the only page: next decode token needs a new one
        engine.active.insert(0, decode_req);
        engine.active.insert(1, prefill_req);
        assert_eq!(engine.kv_free_pages(), 0);

        engine.step()?; // pre-#164 this returned the fatal alloc error

        let requeued = engine.waiting.front().expect("decode victim requeued");
        assert_eq!(requeued.handle, decoding);
        assert!(
            engine.active.contains_key(&1),
            "shed prefill stays active for the next tick"
        );
        // The victim's page is reclaimable: freed outright or parked in the
        // prefix cache, where `alloc_with_prefix_reclaim` evicts it on demand.
        assert!(
            engine.kv_free_pages() + engine.kv.resident_evictable_pages() > 0,
            "retraction made the victim's pages reclaimable"
        );
        Ok(())
    }

    /// free=0 but the radix cache holds reclaimable pages: `fit_plan_to_kv_pages`
    /// must evict the shortfall instead of shedding every prefill row — an
    /// all-trimmed plan is idle, re-enters the same state next tick, and the
    /// request stalls forever (`alloc_with_prefix_reclaim` never reached).
    #[test]
    fn plan_repair_reclaims_cached_pages_before_trimming_prefill() -> Result<()> {
        let mut engine = Engine::with_config(
            MockExecutor::ready(),
            MockKvPool::with_capacity(2, 4, 2),
            test_config(2),
        );
        // Cache one of the two pages: 3 prompt + 2 generated tokens materialize
        // 4 KV positions (the newest token's KV is unwritten), sealing one full
        // block at finish → that page stays cache-retained, one page free.
        let warm = engine.submit_request(vec![1, 2, 3], 2);
        engine.run_to_idle()?;
        assert_finished(engine.completed(warm).expect("warm-up completed"));
        assert_eq!(engine.kv_free_pages(), 1, "one page free, one cached");
        assert_eq!(engine.kv.resident_evictable_pages(), 1);

        // 8-token prompt needs 2 pages > 1 free; the cached page must be
        // reclaimed instead of shedding the row into a permanently idle plan.
        let handle = engine.next_handle();
        let mut request = RequestState::new(
            handle,
            vec![9; 8],
            RequestPriority::Normal,
            1,
            SamplingParams::default(),
        );
        request.phase = RequestPhase::Prefilling { progress: 0 };
        engine.active.insert(0, request);

        for _ in 0..8 {
            engine.step()?;
        }
        let done = engine.completed(handle).expect("request must not stall");
        assert!(!done.generated_tokens.is_empty());
        Ok(())
    }

    /// Evictable-but-not-free pages are repair capacity: with the whole pool
    /// radix-cached (free=0), a plan the cache can satisfy survives untouched
    /// — `alloc_with_prefix_reclaim` evicts on demand at alloc time.
    #[test]
    fn plan_repair_keeps_rows_when_evictable_pages_cover_demand() -> Result<()> {
        let mut engine = Engine::with_config(
            MockExecutor::ready(),
            MockKvPool::with_capacity(2, 4, 2),
            test_config(2),
        );
        // Seal both pages into the radix, then drop the slot: cache-retained
        // pages with no live reference — evictable, not free.
        engine.kv.alloc(0, 8)?;
        engine.publish_prefix_blocks(0, &[1, 2, 3, 4, 5, 6, 7, 8]);
        engine.free_slot_pages(0);
        assert_eq!(engine.kv_free_pages(), 0);
        assert_eq!(engine.kv.resident_evictable_pages(), 2);

        let mut plan = ForwardPlan::idle();
        plan.prefill_rows.push(infer_plan::PrefillRow {
            slot: 1,
            tokens: vec![7; 8],
            start_pos: 0,
            total_tokens: 8,
            params: SamplingParams::default(),
        });
        plan.mode = infer_plan::ForwardMode::Prefill;

        engine.fit_plan_to_kv_pages(&mut plan)?;
        assert_eq!(
            plan.prefill_rows.len(),
            1,
            "evictable capacity covers demand: nothing shed"
        );
        Ok(())
    }

    /// P4 pod smoke: with the pool nearly exhausted by long sessions, a queued
    /// chunked-prefill CONTINUATION chunk (start_pos > 0) reached the executor
    /// needing 128 pages with 6 free — engine-fatal at submit. Pins the host
    /// contract: `fit_plan_to_kv_pages` defers the chunk (it retries next tick)
    /// instead of over-committing; the un-repaired plan provably over-commits.
    #[test]
    fn plan_repair_defers_prefill_continuation_when_pool_nearly_exhausted() -> Result<()> {
        let mut engine = Engine::with_config(
            MockExecutor::ready(),
            MockKvPool::with_capacity(2, 2, 4),
            test_config(2),
        );
        engine.kv.alloc(0, 6)?; // long-lived occupant: 3 of 4 pages
        engine.kv.alloc(1, 2)?; // mid-prefill occupant: first chunk done
        assert_eq!(engine.kv_free_pages(), 0);

        let mut plan = ForwardPlan::idle();
        plan.prefill_rows.push(infer_plan::PrefillRow {
            slot: 1,
            tokens: vec![7; 4], // continuation chunk: 2 new pages, none available
            start_pos: 2,
            total_tokens: 6,
            params: SamplingParams::default(),
        });
        plan.mode = infer_plan::ForwardMode::Prefill;

        let capacity = engine.kv_free_pages() + engine.kv.resident_evictable_pages();
        assert!(
            engine.plan_new_pages_needed(&plan) > capacity,
            "un-repaired plan over-commits the pool"
        );

        engine.fit_plan_to_kv_pages(&mut plan)?;
        assert!(
            plan.prefill_rows.is_empty(),
            "continuation chunk must be deferred, not over-committed"
        );
        assert!(plan.is_idle());
        Ok(())
    }

    /// #164 residual, hole 1: the LRU HEAD is a live slot's mid-flight
    /// published page (retained once, still attached). Eviction must skip it
    /// — severing it frees nothing — and continue into the deeper LRU to a
    /// genuinely freeable page. Uses the production `HostPagedKvPool`.
    #[test]
    fn evict_skips_live_attached_lru_head_and_frees_deeper_page() -> Result<()> {
        let mut engine = Engine::with_config(
            MockExecutor::ready(),
            HostPagedKvPool::new(2, 2, 4),
            test_config(2),
        );
        // Slot 0 LIVE with its page published (prompt-seal): LRU-oldest.
        engine.kv.alloc(0, 4)?;
        engine.publish_prefix_blocks(0, &[1, 2, 3, 4]);
        // Slot 1 finished: published then freed — evictable, newer access.
        engine.kv.alloc(1, 4)?;
        engine.publish_prefix_blocks(1, &[5, 6, 7, 8]);
        engine.free_slot_pages(1);
        assert_eq!(engine.kv_free_pages(), 0);
        assert_eq!(engine.kv.resident_evictable_pages(), 1);

        let freed = engine.evict_prefix_cache_for_pages(2);
        assert_eq!(freed, 1, "skipped the live-attached head, freed deeper");
        assert_eq!(engine.kv_free_pages(), 1);
        // The live slot kept its page AND its cached prefix.
        assert_eq!(engine.kv.page_indices(0).len(), 1);
        assert!(
            !engine
                .radix
                .peek_longest_prefix_match(&[1, 2, 3, 4])
                .is_empty(),
            "live-attached prefix must not be severed for zero gain"
        );
        Ok(())
    }

    /// #164 residual, hole 2 (tick #8340 mirror): free=0 and every cached
    /// page is still attached to a live slot. The capacity model must not
    /// count them (`page_is_evictable` — the evictor's own predicate), so the
    /// repair parks the decode victim instead of reaching a fatal
    /// `allocate_for_plan` failure. Uses the production `HostPagedKvPool`.
    #[test]
    fn plan_repair_sees_shortfall_when_cached_pages_are_live_attached() -> Result<()> {
        let mut engine = Engine::with_config(
            MockExecutor::ready(),
            HostPagedKvPool::new(2, 2, 4),
            test_config(2),
        );
        let handle = engine.next_handle();
        let mut decode_req = RequestState::new(
            handle,
            vec![1, 2, 3, 4, 5, 6, 7, 8],
            RequestPriority::Normal,
            8,
            SamplingParams::default(),
        );
        decode_req.generated_tokens = vec![9];
        decode_req.phase = RequestPhase::Decoding;
        engine.kv.alloc(0, 8)?; // both pool pages, full
        engine.publish_prefix_blocks(0, &[1, 2, 3, 4, 5, 6, 7, 8]);
        engine.active.insert(0, decode_req);
        assert_eq!(engine.kv_free_pages(), 0);
        assert_eq!(
            engine.kv.resident_evictable_pages(),
            0,
            "live-attached cached pages are not repair capacity"
        );

        engine.step()?; // pre-fix: phantom capacity -> fatal alloc at tick time

        let parked = engine.waiting.front().expect("decode victim parked");
        assert_eq!(parked.handle, handle);
        assert_eq!(
            engine.kv.resident_evictable_pages(),
            2,
            "the park made the pages genuinely evictable"
        );
        Ok(())
    }

    /// Shedding must skip zero-demand rows: a fully prefix-reused (or
    /// tail-room) chunk frees no pages, so deferring it gains nothing.
    #[test]
    fn plan_repair_sheds_only_demand_reducing_prefill_rows() -> Result<()> {
        let mut engine = Engine::with_config(
            MockExecutor::ready(),
            MockKvPool::with_capacity(2, 4, 1),
            test_config(2),
        );
        // Slot 0 holds the only page with tail room: appending 2 tokens
        // demands zero new pages. Nothing free, nothing cached.
        engine.kv.alloc(0, 2)?;
        assert_eq!(engine.kv_free_pages(), 0);

        let mut plan = ForwardPlan::idle();
        plan.prefill_rows.push(infer_plan::PrefillRow {
            slot: 1,
            tokens: vec![7; 4],
            start_pos: 0,
            total_tokens: 4,
            params: SamplingParams::default(),
        });
        // Zero-demand row LAST: the pre-fix blind end-pop shed it first.
        plan.prefill_rows.push(infer_plan::PrefillRow {
            slot: 0,
            tokens: vec![5, 6],
            start_pos: 2,
            total_tokens: 4,
            params: SamplingParams::default(),
        });
        plan.mode = infer_plan::ForwardMode::Prefill;

        engine.fit_plan_to_kv_pages(&mut plan)?;
        assert_eq!(plan.prefill_rows.len(), 1);
        assert_eq!(plan.prefill_rows[0].slot, 0, "zero-demand row survives");
        assert!(matches!(plan.mode, infer_plan::ForwardMode::Prefill));
        Ok(())
    }

    /// True exhaustion — nothing free AND nothing evictable — still empties
    /// the plan without error: prefill shed, decode victim parked (#164).
    #[test]
    fn plan_repair_true_exhaustion_empties_plan_without_error() -> Result<()> {
        let mut engine = Engine::with_config(
            MockExecutor::ready(),
            MockKvPool::with_capacity(2, 4, 1),
            test_config(2),
        );
        let handle = engine.next_handle();
        let mut decode_req = RequestState::new(
            handle,
            vec![1, 1, 1],
            RequestPriority::Normal,
            8,
            SamplingParams::default(),
        );
        decode_req.generated_tokens = vec![9];
        decode_req.phase = RequestPhase::Decoding;
        engine.kv.alloc(0, 4)?; // page full: the next decode token needs a new page
        engine.active.insert(0, decode_req);

        let mut plan = ForwardPlan::idle();
        plan.decode_rows.push(infer_plan::DecodeRow {
            slot: 0,
            last_token: 9,
            kv_seq_len: 4,
            params: SamplingParams::default(),
        });
        plan.prefill_rows.push(infer_plan::PrefillRow {
            slot: 1,
            tokens: vec![7; 4],
            start_pos: 0,
            total_tokens: 4,
            params: SamplingParams::default(),
        });
        plan.mode = infer_plan::ForwardMode::Mixed;

        engine.fit_plan_to_kv_pages(&mut plan)?;
        assert!(
            plan.is_idle(),
            "nothing reclaimable: plan empties instead of erroring"
        );
        let parked = engine.waiting.front().expect("decode victim parked");
        assert_eq!(parked.handle, handle);
        Ok(())
    }

    /// #164 adjacent hole: a speculative backend's EXTRA decode token is
    /// appended in `apply_output` before any repair can run. On true
    /// exhaustion the append must park the request, not unwind the group.
    #[test]
    fn spec_extra_token_exhaustion_parks_request_instead_of_fatal() -> Result<()> {
        let executor = SpecMirrorExecutor::default();
        executor.seed_materialized(0, 3);
        // HoldGovernor keeps the parked victim in `waiting` (no same-tick
        // re-admission), so the degrade outcome stays observable.
        let mut engine = Engine::with_config_and_governor(
            executor,
            MockKvPool::with_capacity(1, 2, 2),
            test_config(1),
            Box::new(HoldGovernor),
        );
        let handle = engine.next_handle();
        let mut request = RequestState::new(
            handle,
            vec![1, 1, 1],
            RequestPriority::Normal,
            8,
            SamplingParams::default(),
        );
        request.generated_tokens = vec![9];
        request.phase = RequestPhase::Decoding;
        engine.kv.alloc(0, 3)?; // both pages used; the first decode token fits the tail
        engine.active.insert(0, request);
        assert_eq!(engine.kv_free_pages(), 0);

        engine.step()?; // submit the decode forward (two-token spec output)
        engine.step()?; // apply: the extra's append finds nothing reclaimable — park, not fatal

        assert_eq!(
            engine.waiting.front().expect("request parked").handle,
            handle
        );
        assert!(engine.active.is_empty());
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
    fn full_prefix_match_still_prefills_the_last_block() -> Result<()> {
        // Regression for the wrong-seed-token bug
        // (docs/experience/errors/2026-07-06-dsv4-concurrent-decode-digit-corruption-unresolved.md,
        // "Layer-0-15 residual bisection"): a full-prompt radix match must
        // never jump straight to `Decoding` with an empty `generated_tokens`,
        // since the planner would then silently re-feed the prompt's own
        // last token as the decode seed, duplicating it into KV. Two
        // full 4-token blocks give the radix trie a full match on repeat.
        let mut engine = Engine::with_config(
            MockExecutor::ready(),
            MockKvPool::with_capacity(1, 4, 8),
            test_config(1),
        );
        let first = engine.submit_request(vec![1, 2, 3, 4, 5, 6, 7, 8], 1);
        engine.run_to_idle()?;
        assert_finished(engine.completed(first).expect("first completed"));
        assert_eq!(engine.radix.cached_page_count(), 2, "both blocks published");

        let second = engine.submit_request(vec![1, 2, 3, 4, 5, 6, 7, 8], 1);
        engine.step()?; // admission + attach only; output not yet applied.

        let (_, request) = engine
            .active
            .iter()
            .find(|(_, request)| request.handle == second)
            .expect("second admitted");
        assert_eq!(
            request.phase,
            RequestPhase::Prefilling { progress: 4 },
            "the last matched block must still be re-prefilled for real, \
             never a direct jump to Decoding"
        );
        assert_eq!(
            request.reused_prefix_pages.len(),
            1,
            "only the non-final matched block is reused from cache"
        );
        Ok(())
    }

    #[test]
    fn frontier_tail_restore_still_prefills_the_last_token() -> Result<()> {
        // A DSv4 finish-write-through sidecar can restore PAST the trimmed
        // radix match back to the full committed stream; entering Decoding
        // there re-feeds the last token's KV. The restore clamp must hold one
        // token back so the tail re-seeds from real logits.
        use infer_seam::{PrefixBlock, pages_only_reusable_prefix_blocks};

        struct FrontierRestoreExecutor(MockExecutor);
        impl BackendExecutor for FrontierRestoreExecutor {
            type Inflight = super::testing::MockInflight;
            fn submit(
                &mut self,
                plan: &ForwardPlan,
                kv: &mut dyn KvPool,
            ) -> Result<Self::Inflight> {
                self.0.submit(plan, kv)
            }
            fn poll(&mut self, inflight: Self::Inflight) -> Result<PollResult<Self::Inflight>> {
                self.0.poll(inflight)
            }
            fn reusable_prefix_blocks(&self, blocks: &[PrefixBlock]) -> usize {
                pages_only_reusable_prefix_blocks(blocks, |_| false)
            }
            fn restore_prefix_sidecar(
                &mut self,
                _slot: usize,
                tokens: &[u32],
                _matched_len: usize,
                _prefix_pages: &[u32],
            ) -> Result<usize> {
                Ok(tokens.len())
            }
        }

        let mut engine = Engine::with_config(
            FrontierRestoreExecutor(MockExecutor::ready()),
            MockKvPool::with_capacity(1, 4, 8),
            test_config(1),
        );
        let first = engine.submit_request(vec![1, 2, 3, 4, 5, 6, 7, 8], 1);
        engine.run_to_idle()?;
        assert_finished(engine.completed(first).expect("first completed"));

        let second = engine.submit_request(vec![1, 2, 3, 4, 5, 6, 7, 8], 1);
        engine.step()?; // admission + attach only.
        let (_, request) = engine
            .active
            .iter()
            .find(|(_, request)| request.handle == second)
            .expect("second admitted");
        assert_eq!(
            request.phase,
            RequestPhase::Prefilling { progress: 7 },
            "a full frontier restore must hold the last token back for prefill"
        );
        Ok(())
    }

    #[test]
    fn full_prefix_match_trim_runs_before_backend_clamp() -> Result<()> {
        // #154 D4 regression: the full-match one-block trim must run BEFORE
        // `clamp_prefix_to_backend`. With an alignment-flooring backend
        // (commit points every 2 blocks), the old clamp-then-trim order left
        // an UNALIGNED 3-block match (12 tokens) that the backend restore
        // predicate rejects; trim-then-clamp floors it to 2 blocks (8).
        let mut engine = Engine::with_config(
            AlignedPrefixExecutor::with_align_blocks(2),
            MockKvPool::with_capacity(1, 4, 8),
            test_config(1),
        );
        let prompt: Vec<u32> = (1..=16).collect();
        let first = engine.submit_request(prompt.clone(), 1);
        engine.run_to_idle()?;
        assert_finished(engine.completed(first).expect("first completed"));
        assert_eq!(
            engine.radix.cached_page_count(),
            4,
            "all 4 blocks published"
        );

        let second = engine.submit_request(prompt, 1);
        engine.step()?; // admission + attach only.
        let (_, request) = engine
            .active
            .iter()
            .find(|(_, request)| request.handle == second)
            .expect("second admitted");
        assert_eq!(
            request.phase,
            RequestPhase::Prefilling { progress: 8 },
            "trim-then-clamp must land on the 2-block-aligned boundary"
        );
        assert_eq!(request.reused_prefix_pages.len(), 2);
        Ok(())
    }

    #[test]
    fn background_completion_is_published_before_admission_lookup() -> Result<()> {
        let ready = std::rc::Rc::new(std::cell::Cell::new(true));
        let pending = std::rc::Rc::new(std::cell::Cell::new(false));
        let mut engine = Engine::with_config(
            BackgroundPublishExecutor {
                inner: MockExecutor::ready(),
                ready: ready.clone(),
                pending: pending.clone(),
            },
            MockKvPool::with_capacity(1, 4, 8),
            test_config(1),
        );
        let prompt: Vec<u32> = (1..=8).collect();
        let first = engine.submit_request(prompt.clone(), 1);
        engine.run_to_idle()?;
        assert_finished(engine.completed(first).expect("first completed"));

        ready.set(false);
        pending.set(true);
        let second = engine.submit_request(prompt, 1);
        engine.step()?;
        let (_, request) = engine
            .active
            .iter()
            .find(|(_, request)| request.handle == second)
            .expect("second admitted");
        assert_eq!(request.reused_prefix_pages.len(), 1);
        assert!(!pending.get());
        Ok(())
    }

    #[test]
    fn resend_sidecar_save_keys_to_radix_pages_not_recomputed_slot_pages() -> Result<()> {
        // #155: turn 2 resends turn 1's prompt, the sidecar restore misses and
        // the request full-recomputes on fresh slot pages. Its publishes fully
        // dedupe (`newly_cached` empty), so the sidecar save must be keyed to
        // the radix's ORIGINAL chain — keying it to the recomputed slot pages
        // (freed at finish, never radix-evicted) leaks the blob, then drops it
        // on page-id reuse while the cached prefix still expects it.
        let executor = SidecarMissExecutor::new();
        let saves = executor.saves.clone();
        let mut engine = Engine::with_config(
            executor,
            MockKvPool::with_capacity(1, 4, 16),
            test_config(1),
        );

        // Turn 1: publish prompt (2 blocks) at prefill-seal + the sealed
        // prompt+generated boundary (3 blocks: echo mock generates 9..=13,
        // KV holds 12) at finish.
        let prompt: Vec<u32> = (1..=8).collect();
        let first = engine.submit_request(prompt.clone(), 5);
        engine.run_to_idle()?;
        assert_finished(engine.completed(first).expect("first completed"));
        let full_sequence: Vec<u32> = (1..=12).collect();
        let chain = engine
            .radix
            .peek_longest_prefix_match(&full_sequence)
            .block_ids;
        assert_eq!(chain.len(), 3, "turn 1 published the full sequence");
        saves.borrow_mut().clear();

        // Turn 2: radix match + forced restore miss → full recompute.
        let second = engine.submit_request(prompt, 5);
        engine.run_to_idle()?;
        assert_finished(engine.completed(second).expect("second completed"));

        let saves = saves.borrow();
        assert!(
            !saves.is_empty(),
            "resend must still re-publish the sidecar"
        );
        for (prefix_pages, slot_pages, newly_cached) in saves.iter() {
            assert!(
                newly_cached.is_empty(),
                "resend publishes must fully dedupe, got {newly_cached:?}"
            );
            assert_eq!(
                prefix_pages[..],
                chain[..prefix_pages.len()],
                "sidecar keyed to recomputed slot pages instead of the radix \
                 chain — those pages free at finish, so the blob's lifetime \
                 no longer rides the radix (#155)"
            );
            // #157 repair plumbing: the save also carries the slot's OWN chain
            // position-aligned with the canonical one, so a backend can adopt
            // the recomputed entry where a canonical pool entry evicted.
            assert_eq!(
                slot_pages.len(),
                prefix_pages.len(),
                "slot chain must be position-aligned with the canonical chain"
            );
        }
        Ok(())
    }

    /// #157 H2: a preempted decode must seal prompt+GENERATED (the finish
    /// boundary), not prompt-only — otherwise every generated page drops as a
    /// provisional backend entry at the free and the whole generated region
    /// recomputes on resume / follow-up turns.
    #[test]
    fn requeue_publishes_generated_pages() -> Result<()> {
        let mut engine = Engine::with_config(
            MockExecutor::ready(),
            MockKvPool::with_capacity(1, 4, 16),
            test_config(1),
        );
        let prompt: Vec<u32> = (1..=8).collect();
        let handle = engine.submit_request(prompt, 8);
        // Step until 5 tokens committed: KV holds 12 tokens = 3 sealed blocks.
        for _ in 0..16 {
            engine.step()?;
            let generated = engine
                .active
                .values()
                .find(|r| r.handle == handle)
                .map_or(0, |r| r.generated_tokens.len());
            if generated == 5 {
                break;
            }
        }
        let (&slot, request) = engine
            .active
            .iter()
            .find(|(_, r)| r.handle == handle)
            .expect("victim still active");
        assert_eq!(request.generated_tokens.len(), 5);
        engine.requeue_preempted_decode(slot);
        let committed: Vec<u32> = (1..=12).collect();
        assert_eq!(
            engine
                .radix
                .peek_longest_prefix_match(&committed)
                .block_ids
                .len(),
            3,
            "requeue sealed the committed sequence through the generated region"
        );
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
    fn backend_plan_token_cap_bounds_mixed_plans_and_long_prompt_completes() -> Result<()> {
        let executor = PlanTokenCapExecutor::with_cap(3);
        let capability_reads = executor.capability_reads.clone();
        let plans = executor.plans.clone();
        let mut config = test_config(2);
        config.chunked_prefill_size = 128;
        let mut engine = Engine::with_config(executor, MockKvPool::new(2), config);
        let decode = engine.submit_request(vec![10], 8);

        engine.step()?;
        engine.step()?;
        let long_prompt = engine.submit_request((1u32..=17).collect(), 2);
        engine.run_to_idle()?;

        assert_eq!(
            capability_reads.get(),
            1,
            "backend plan-token capability must be snapshotted at construction"
        );
        assert!(
            plans.borrow().iter().all(|(_, tokens)| *tokens <= 3),
            "every submitted plan must fit the snapshotted backend cap: {plans:?}"
        );
        assert!(
            plans.borrow().iter().any(|(mixed, _)| *mixed),
            "test must exercise mixed decode+prefill planning: {plans:?}"
        );
        assert_finished(engine.completed(decode).expect("decode completed"));
        assert_finished(
            engine
                .completed(long_prompt)
                .expect("long prompt completed"),
        );
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
            if let Some(request) = engine.active.values().next()
                && matches!(request.phase, RequestPhase::Prefilling { .. })
            {
                prefilling_positions.push(request.prefill_start_pos);
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
    fn prefill_chunk_stops_on_lcm_of_page_size_and_restore_alignment() -> Result<()> {
        // page_size=4, restore_alignment=6: neither divides the other, so
        // lcm(4,6)=12 is required (max(4,6)=6 would land on 30, NOT a
        // multiple of page_size 4). max_prefill_chunk=12 (one alignment
        // unit) caps each chunk, so a 32-token prompt hits EVERY boundary —
        // 12, then 24 — not just the deepest one a single oversized chunk
        // could reach (28, page-only) or (30, max-only).
        let mut engine = Engine::with_config(
            RestoreAlignmentExecutor::with_alignment_and_chunk(6, 12),
            MockKvPool::with_capacity(1, 4, 16),
            test_config(1),
        );
        engine.submit_request((1u32..=32).collect(), 1);

        let mut prefilling_positions = Vec::new();
        for _ in 0..8 {
            engine.step()?;
            if let Some(request) = engine.active.values().next()
                && matches!(request.phase, RequestPhase::Prefilling { .. })
            {
                prefilling_positions.push(request.prefill_start_pos);
            }
        }
        assert!(
            prefilling_positions.contains(&12) && prefilling_positions.contains(&24),
            "prefill must stop at EVERY lcm(4,6)=12 boundary (12, 24), got {prefilling_positions:?}"
        );
        assert!(
            !prefilling_positions.contains(&28) && !prefilling_positions.contains(&30),
            "chunk must not skip ahead to page-size-only (28) or max-only (30) boundaries, \
             got {prefilling_positions:?}"
        );
        Ok(())
    }

    #[test]
    fn dsv4_shaped_prefill_chunking_is_unchanged_by_capability() -> Result<()> {
        // DSv4 Phase-1 shape: page 16, restore_alignment 128, max_prefill_chunk
        // 128, config chunk 4096. Plans must be byte-identical to the removed
        // one-alignment-unit cap: a 300-token prompt chunks [128, 128, 44] —
        // start positions 0 -> 128 -> 256, tail unaligned, nothing else.
        let mut config = test_config(1);
        config.max_prompt_tokens = 512;
        config.chunked_prefill_size = 4096;
        let mut engine = Engine::with_config(
            RestoreAlignmentExecutor::with_alignment_and_chunk(128, 128),
            MockKvPool::with_capacity(1, 16, 32),
            config,
        );
        engine.submit_request((0u32..300).collect(), 1);

        let mut prefilling_positions = Vec::new();
        for _ in 0..6 {
            engine.step()?;
            if let Some(request) = engine.active.values().next()
                && matches!(request.phase, RequestPhase::Prefilling { .. })
            {
                prefilling_positions.push(request.prefill_start_pos);
            }
        }
        // First step admits (position still 0), then chunks advance 0 -> 128
        // -> 256 -> decode: exactly the 128/128/44 split the one-unit cap gave.
        assert_eq!(
            prefilling_positions,
            vec![0, 128, 256],
            "DSv4-shaped chunking must stay 128/128/44 (old one-unit-cap behavior)"
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
            if let Some(request) = engine.active.values().next()
                && matches!(request.phase, RequestPhase::Prefilling { .. })
            {
                prefill_start_positions.push(request.prefill_start_pos);
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
