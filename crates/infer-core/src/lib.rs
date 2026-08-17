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

pub use radix::{BlockId, PrefixMatch, RadixCache};
pub use recall::{RecallConfig, RecallPlan, plan_recall};

use anyhow::Result;
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
    /// Minimum decode tokens before an oversubscribed request may be parked again.
    pub oversubscription_min_slice: usize,
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
            oversubscription_min_slice: DEFAULT_OVERSUBSCRIPTION_MIN_SLICE,
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
    pub submitted_at: std::time::Instant,
    pub first_token_at: Option<std::time::Instant>,
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
    pub requests_succeeded: u64,
    pub requests_failed: u64,
    pub ttft_micros_total: u64,
    pub ttft_count: u64,
    pub tpot_micros_total: u64,
    pub tpot_count: u64,
    pub e2e_micros_total: u64,
    pub e2e_count: u64,
    /// Submit-to-ready wall, split by the submitted plan shape.
    pub forward_busy_micros: u64,
    pub prefill_forward_steps: u64,
    pub prefill_forward_busy_micros: u64,
    pub decode_forward_steps: u64,
    pub decode_forward_busy_micros: u64,
    pub mixed_forward_steps: u64,
    pub mixed_forward_busy_micros: u64,
    /// Decode-only host phases. These overlap `forward_busy_micros` at submit.
    pub decode_step_phase: StepPhaseStats,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StepPhaseStats {
    pub steps: u64,
    pub poll_micros: u64,
    pub apply_output_micros: u64,
    pub poll_background_micros: u64,
    pub admit_micros: u64,
    pub plan_micros: u64,
    pub submit_micros: u64,
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

fn step_phase_stats() -> StepPhaseStats {
    let load = |index: usize| STEP_PHASE_MICROS[index].load(std::sync::atomic::Ordering::Relaxed);
    StepPhaseStats {
        steps: STEP_PHASE_STEPS.load(std::sync::atomic::Ordering::Relaxed),
        poll_micros: load(0),
        apply_output_micros: load(1),
        poll_background_micros: load(2),
        admit_micros: load(3),
        plan_micros: load(4),
        submit_micros: load(5),
    }
}

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
    /// Restored at least one backend-reusable prefix token.
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
pub const DEFAULT_OVERSUBSCRIPTION_MIN_SLICE: usize = 8;

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
    /// victim must have decoded the configured minimum slice
    /// since then, so a just-resumed request runs a bit before it can be parked
    /// again — bounding ping-pong churn at num_slots=1.
    admit_gen_mark: usize,
    grammar: Option<GrammarHook>,
    submitted_at: std::time::Instant,
    first_token_at: Option<std::time::Instant>,
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
            submitted_at: std::time::Instant::now(),
            first_token_at: None,
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

    /// Penalty state lives nowhere but here: the engine ships a snapshot per
    /// row so preemption, prefix restore and spec rollback cannot desync a
    /// mirrored copy below the seam. Default requests pay nothing.
    pub(crate) fn penalty_history(&self) -> (Option<std::sync::Arc<[u32]>>, usize) {
        if !self.sampling.has_penalty() {
            return (None, 0);
        }
        let history: std::sync::Arc<[u32]> = self.committed_tokens().into();
        (Some(history), self.prompt_tokens.len())
    }
}

impl From<RequestState> for CompletedRequest {
    fn from(request: RequestState) -> Self {
        Self {
            handle: request.handle,
            prompt_tokens: request.prompt_tokens,
            generated_tokens: request.generated_tokens,
            finish: request.finish,
            submitted_at: request.submitted_at,
            first_token_at: request.first_token_at,
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
    ///
    /// # Errors
    /// Rejects a config that requests a capability this backend does not have.
    pub fn new(executor: E, kv: K, max_slots: usize) -> Result<Self> {
        Self::with_config(executor, kv, SchedulerConfig::for_slots(max_slots))
    }

    /// Create an engine with explicit scheduler config.
    ///
    /// # Errors
    /// Rejects a config that requests a capability this backend does not have.
    pub fn with_config(executor: E, kv: K, config: SchedulerConfig) -> Result<Self> {
        Self::with_config_and_governor(executor, kv, config, Box::new(PermissiveGovernor))
    }

    /// Create an engine with explicit scheduler config and resource governor.
    ///
    /// # Errors
    /// Rejects a config that requests a capability this backend does not have.
    pub fn with_config_and_governor(
        mut executor: E,
        kv: K,
        mut config: SchedulerConfig,
        governor: Box<dyn ResourceGovernor>,
    ) -> Result<Self> {
        // Backend-neutral flag check at the one place config and executor meet;
        // without it a tier-less backend serves with the flag silently ignored.
        if config.slot_oversubscription {
            anyhow::ensure!(
                executor.kv_slot_tier().is_some(),
                "--kv-oversubscription is set, but backend {} has no whole-slot \
                 KV tier, so the flag would do nothing",
                std::any::type_name::<E>()
            );
        }
        let limits = executor.step_limits();
        let max_rows = limits.max_rows_per_step.max(1);
        if config.num_slots > max_rows {
            log::warn!(
                "executor caps rows per step at {max_rows}; scheduler slots {} -> {max_rows}",
                config.num_slots
            );
            config.num_slots = max_rows;
        }
        // Per-forward token cap (deepep_ll LL dispatch buffer): clamp num_slots so
        // a pure-decode forward (one token per slot) never exceeds it.
        let max_tokens_per_step = limits.max_tokens_per_step.max(1);
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
        Ok(Self {
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
        })
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
        self.executor.step_limits().max_live_requests.max(1)
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
        match self.executor.weight_residency() {
            Some(residency) => residency.offload_weights(),
            None => Ok(0),
        }
    }

    /// Restore the backend's device weights from the host snapshot (OPD teacher
    /// time-share). Delegates to [`BackendExecutor::reload_weights`].
    ///
    /// # Errors
    /// Propagates any error returned by the backend executor's reload.
    pub fn reload_engine_weights(&mut self) -> Result<()> {
        match self.executor.weight_residency() {
            Some(residency) => residency.reload_weights(),
            None => Ok(()),
        }
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
        match self.executor.weight_residency() {
            Some(residency) => residency.release_inference_scratch(),
            None => Ok(()),
        }
    }

    /// Drop the backend's KV pool WITHOUT offloading weights (OPD writeback
    /// headroom: the writeback's fresh autograd forward never reads this engine's
    /// KV). Delegates to [`BackendExecutor::release_kv_pool`] (default no-op). The
    /// engine must be idle (all rollouts synced before this is called).
    ///
    /// # Errors
    /// Propagates any error returned by the backend executor's pool release.
    pub fn release_kv_pool(&mut self) -> Result<()> {
        match self.executor.weight_residency() {
            Some(residency) => residency.release_kv_pool(),
            None => Ok(()),
        }
    }

    /// Re-acquire the KV pool dropped by [`Self::release_kv_pool`] before the next
    /// rollout. Delegates to [`BackendExecutor::ensure_kv_pool`] (default no-op).
    ///
    /// # Errors
    /// Propagates any error returned by the backend executor's pool re-acquire.
    pub fn ensure_kv_pool(&mut self) -> Result<()> {
        match self.executor.weight_residency() {
            Some(residency) => residency.ensure_kv_pool(),
            None => Ok(()),
        }
    }

    /// Submit a normal-priority request into the waiting queue.
    pub fn submit_request(&mut self, prompt_tokens: Vec<u32>, max_tokens: usize) -> RequestHandle {
        self.submit_request_with_options(prompt_tokens, max_tokens, RequestOptions::default())
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
        // The plan apply_output PROCESSES is the previous step's; gating only on
        // the plan being SUBMITTED books a post-prefill radix seal (~1 s over a
        // 33K prompt) into the decode bucket.
        let mut applied_decode_only = true;
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
                    let plan = self.pending_plan.take().unwrap_or_else(ForwardPlan::idle);
                    if let Some(submitted_at) = self.inflight_submit_at.take() {
                        let busy = submitted_at.elapsed().as_micros() as u64;
                        ENGINE_FORWARD_BUSY_MICROS
                            .fetch_add(busy, std::sync::atomic::Ordering::Relaxed);
                        self.throughput_stats.forward_busy_micros = self
                            .throughput_stats
                            .forward_busy_micros
                            .saturating_add(busy);
                        let (steps, micros) =
                            match (plan.prefill_rows.is_empty(), plan.decode_rows.is_empty()) {
                                (false, true) => {
                                    let stats = &mut self.throughput_stats;
                                    (
                                        &mut stats.prefill_forward_steps,
                                        &mut stats.prefill_forward_busy_micros,
                                    )
                                }
                                (true, false) => {
                                    let stats = &mut self.throughput_stats;
                                    (
                                        &mut stats.decode_forward_steps,
                                        &mut stats.decode_forward_busy_micros,
                                    )
                                }
                                _ => {
                                    let stats = &mut self.throughput_stats;
                                    (
                                        &mut stats.mixed_forward_steps,
                                        &mut stats.mixed_forward_busy_micros,
                                    )
                                }
                            };
                        *steps = steps.saturating_add(1);
                        *micros = micros.saturating_add(busy);
                    }
                    applied_decode_only = plan.prefill_rows.is_empty();
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
        let decode_only = plan.prefill_rows.is_empty() && applied_decode_only;
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

    #[must_use]
    pub fn active_count(&self) -> usize {
        self.active.len()
    }

    #[must_use]
    pub fn waiting_count(&self) -> usize {
        self.waiting.len()
    }

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

    #[must_use]
    pub fn throughput_stats(&self) -> ThroughputStats {
        ThroughputStats {
            decode_step_phase: step_phase_stats(),
            ..self.throughput_stats
        }
    }

    /// Return KV host-tier counters plus the current tier-resident size.
    #[must_use]
    pub fn kv_tier_stats(&self) -> KvTierStats {
        let mut stats = self.kv_tier_stats;
        stats.resident_blocks = self.radix.demoted_block_count();
        stats
    }

    /// Backend counters and artifact identity, in one executor round-trip.
    #[must_use]
    pub fn backend_stats(&self) -> infer_seam::BackendStats {
        self.executor.stats()
    }

    #[must_use]
    pub fn kv_system_metrics(&self) -> KvSystemMetrics {
        let mut metrics = self.kv_system_metrics;
        metrics.resident_pages = self.kv.resident_pages();
        metrics.resident_evictable_pages = self.kv.resident_evictable_pages();
        let (host_demoted_pages, disk_pages, tier_hits, io) =
            match self.executor.kv_page_tier_view() {
                Some(tier) => (
                    tier.kv_tier_host_demoted_pages(),
                    tier.kv_tier_disk_pages(),
                    tier.kv_tier_read_hits(),
                    tier.kv_tier_io_stats(),
                ),
                None => (
                    0,
                    0,
                    infer_seam::KvTierReadHits::default(),
                    infer_seam::KvTierIoStats::default(),
                ),
            };
        metrics.host_demoted_pages = host_demoted_pages;
        metrics.host_demoted_pending_inflight = 0;
        metrics.disk_pages = disk_pages;
        metrics.reuse_hit_host_demoted = metrics
            .reuse_hit_host_demoted
            .saturating_add(tier_hits.host_demoted);
        metrics.reuse_hit_disk = metrics.reuse_hit_disk.saturating_add(tier_hits.disk);
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
        let now = std::time::Instant::now();
        let stats = &mut self.throughput_stats;
        let failed = matches!(completed.finish, Some(FinishReason::Abort));
        if failed {
            stats.requests_failed = stats.requests_failed.saturating_add(1);
        } else {
            stats.requests_succeeded = stats.requests_succeeded.saturating_add(1);
        }
        let e2e = now.duration_since(completed.submitted_at).as_micros() as u64;
        stats.e2e_micros_total = stats.e2e_micros_total.saturating_add(e2e);
        stats.e2e_count = stats.e2e_count.saturating_add(1);
        if let Some(first_token_at) = completed.first_token_at {
            let ttft = first_token_at
                .duration_since(completed.submitted_at)
                .as_micros() as u64;
            stats.ttft_micros_total = stats.ttft_micros_total.saturating_add(ttft);
            stats.ttft_count = stats.ttft_count.saturating_add(1);
            let n = completed.generated_tokens.len() as u64;
            if let Some(tpot) = now
                .duration_since(first_token_at)
                .as_micros()
                .checked_div(n as u128)
            {
                let tpot = tpot as u64;
                stats.tpot_micros_total = stats.tpot_micros_total.saturating_add(tpot);
                stats.tpot_count = stats.tpot_count.saturating_add(1);
            }
        }
        self.completed.insert(handle, completed);
        while self.completed.len() > COMPLETED_CAP {
            self.completed.pop_first();
        }
    }

    /// Abort a waiting/parked request: release its whole-slot tier image (if any)
    /// so it cannot leak — restore_swapped_slot (planner.rs) is the only other
    /// release path — then record the Abort completion.
    fn abort_waiter(&mut self, request: RequestState) {
        if let Some(key) = request.swap_key
            && let Some(tier) = self.executor.kv_slot_tier()
        {
            tier.drop_kv_slot_entries(&[key]);
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
                    if request.first_token_at.is_none() {
                        request.first_token_at = Some(std::time::Instant::now());
                    }
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
        // prompt_len, not post-generation. Without this, prompt-only resends full-recompute. `radix.insert_replicated` dedups.
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
                if request.first_token_at.is_none() {
                    request.first_token_at = Some(std::time::Instant::now());
                }
                request.advance_grammar(token.token);
                let finished = finish_reason_for(request, &token, &self.model_stop_token_ids);
                committed.push((request.handle, token));
                token_idx += 1;
                if let Some(finish) = finished {
                    finished_slots.push((row.slot, finish));
                    break;
                }
            }
            // Return the #197 spec pre-budget the committed chain did not use (#205).
            let committed_len = row.kv_seq_len + token_idx;
            if token_idx > 0
                && self.active.contains_key(&row.slot)
                && self.kv.seq_len(row.slot) > committed_len
            {
                self.kv.truncate_slot(row.slot, committed_len)?;
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
        if let Some(reuse) = self.executor.prefix_reuse() {
            reuse.release_provisional_prefix_pages(&pages);
        }
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
        if let Some(reuse) = self.executor.prefix_reuse()
            && let Err(err) = reuse.capture_finish_frontier(slot, &full_tokens, &slot_pages)
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

    fn record_prefix_match_metrics(&mut self, raw_pages: usize, licensed_pages: usize) {
        if !self.config.enable_prefix_cache {
            return;
        }
        self.kv_system_metrics.prefix_match_full_blocks = self
            .kv_system_metrics
            .prefix_match_full_blocks
            .saturating_add(raw_pages as u64);
        self.kv_system_metrics.prefix_match_clamped_blocks = self
            .kv_system_metrics
            .prefix_match_clamped_blocks
            .saturating_add(licensed_pages as u64);
    }

    fn record_prefix_restore_metrics(&mut self, restored_tokens: usize) {
        if !self.config.enable_prefix_cache {
            return;
        }
        if restored_tokens == 0 {
            self.kv_system_metrics.reuse_miss = self.kv_system_metrics.reuse_miss.saturating_add(1);
            return;
        }
        self.prefix_cache_stats.hits = self.prefix_cache_stats.hits.saturating_add(1);
        self.prefix_cache_stats.hit_tokens = self
            .prefix_cache_stats
            .hit_tokens
            .saturating_add(restored_tokens as u64);
        let restored_pages = restored_tokens.div_ceil(self.radix.block_size()) as u64;
        self.prefix_cache_stats.hit_pages = self
            .prefix_cache_stats
            .hit_pages
            .saturating_add(restored_pages);
        if self.kv_tier_capacity() == 0 {
            self.kv_system_metrics.reuse_hit_resident = self
                .kv_system_metrics
                .reuse_hit_resident
                .saturating_add(restored_pages);
        }
    }

    fn admit_waiting(&mut self) -> Result<()> {
        match self.governor.admission_gate() {
            AdmissionVerdict::Admit | AdmissionVerdict::ShedTo(_) => {}
            AdmissionVerdict::Hold => return Ok(()),
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
            && self.executor.kv_slot_tier().is_some()
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
        // CP sharding: the ring pass recomputes the whole prompt, so there is
        // no prefix reuse to budget. The radix match + clamp issued a
        // tp_sync_min per candidate; divergent admission-loop iteration counts
        // deadlocked cross-communicator (global TP vs attn_cp).
        let reuse_matched_len =
            if self.config.enable_prefix_cache && self.executor.kv_shard_spec().is_none() {
                let committed = candidate.committed_cow();
                let matched = self.radix.peek_longest_prefix_match(&committed);
                let prefix_match = Self::clamp_prefix_to_backend(
                    &mut self.executor,
                    self.radix.block_size(),
                    matched,
                    &committed,
                );
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
                    match self.executor.prefix_reuse() {
                        Some(reuse) => reuse
                            .cached_prefix_match_len(&committed)?
                            .min(committed.len()),
                        None => 0,
                    }
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
        } else if self.executor.kv_shard_spec().is_none() {
            // C.3: under 2D the ring pass recomputes the whole prompt — the
            // ring path does not gather the paged cached-prefix shard into
            // the ring block. Skip match+attach so prefill starts at 0.
            // The empty-match attach issued two tp_sync_min collectives per
            // request; divergent admission-loop iteration counts deadlocked
            // cross-communicator. Fresh requests already carry the
            // prefill_start_pos=0 / Prefilling{0} state the empty-attach sets.
            let committed = request.committed_tokens();
            let prefix_match = if self.config.enable_prefix_cache {
                // Tier-aware: demoted blocks in the match are promoted
                // back into fresh pages here, so attach sees a
                // resident-only match.
                self.lookup_prefix_for_attach(&committed)?
            } else {
                PrefixMatch::empty()
            };
            self.attach_prefix_to_request(slot, &mut request, &committed, prefix_match)?;
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
    /// at least `oversubscription_min_slice` tokens since their last admit, the
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
                        >= self.config.oversubscription_min_slice.max(1)
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
        if let Some(fit) = self.executor.device_kv_fit() {
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
            fit.kv_device_fit(&self.device_demand_scratch, &mut self.device_unfit_scratch);
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
        let spec_row_tokens = self.executor.step_limits().spec_row_tokens;
        plan.decode_rows.retain(|row| {
            match self.alloc_with_prefix_reclaim(row.slot, spec_row_tokens) {
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
