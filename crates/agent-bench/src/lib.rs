//! Agent-workflow benchmark harness — the AI-PC north-star metric.
//!
//! Unlike a tok/s sweep, this drives the engine with a **multi-turn agent
//! workflow** (a shared system prompt + a sequence of user turns, each growing
//! the shared context so cross-turn KV reuse via the radix cache applies — the
//! realistic agent shape) and measures end-to-end task behavior: per-turn
//! latency / ticks and aggregate throughput.
//!
//! It is generic over the backend ([`infer_seam::BackendExecutor`] +
//! [`infer_seam::KvPool`]). With the bundled [`EchoExecutor`] it benchmarks the
//! device-neutral **scheduler layer** today (CPU only); pointed at
//! `infer_metal::MetalExecutor` (once R3b lands the real generation loop) it
//! produces the real on-device agent-workflow + OS-impact report (G3 of the
//! rewrite verification targets).

use std::cell::RefCell;
use std::rc::Rc;
use std::time::{Duration, Instant};

use anyhow::Result;
use infer_core::{Engine, SchedulerConfig};
use infer_plan::{ForwardPlan, SlotToken, StepOutput};
use infer_seam::{BackendExecutor, KvPool, PollResult};

pub use infer_metal::MetalKvPool;

/// OS-impact accounting for a workflow run.
///
/// On the EchoExecutor (scheduler-only) path this stays all-zero — there is no
/// device memory pressure or foreground-responsiveness cost to attribute. On a
/// real-backend run a probe ([`PeakMemProbe`]) fills these in so the north-star
/// report can answer "did serving this agent workflow degrade the AI-PC?".
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OsImpactReport {
    /// Number of times the probe was sampled (once per turn).
    pub samples: u64,
    /// Peak resident set size observed across samples, in bytes.
    pub peak_rss_bytes: u64,
}

/// A pluggable sampler of OS-level impact, driven once per workflow turn.
///
/// `sample` is called by [`run_agent_workflow`] after each turn completes;
/// `report` is folded into [`WorkflowMetrics`] at the end. The trait is
/// deliberately tiny so the scheduler-layer bench can run with a zero-cost
/// [`NoopProbe`] while a real-backend harness swaps in [`PeakMemProbe`].
pub trait OsImpactProbe {
    /// Take one OS-impact sample (called once per turn).
    fn sample(&mut self);
    /// Summarize all samples taken so far.
    fn report(&self) -> OsImpactReport;
}

/// Default probe: records nothing. Used for EchoExecutor / scheduler-layer runs
/// where there is no device memory or responsiveness cost to attribute.
#[derive(Debug, Default)]
pub struct NoopProbe;

impl OsImpactProbe for NoopProbe {
    fn sample(&mut self) {}
    fn report(&self) -> OsImpactReport {
        OsImpactReport::default()
    }
}

/// Real-backend OS-impact probe.
///
/// Once per turn it reads this process's *peak* resident set size — the OS
/// high-water mark, so a discrete poll still captures the true peak reached
/// between samples — and folds it into [`OsImpactReport::peak_rss_bytes`] via
/// `.max(..)`. This is the north-star "did serving this agent workflow degrade
/// the AI-PC?" memory metric, measured against the engine driving a real
/// backend forward. Platform source: `mach_task_basic_info.resident_size_max`
/// on macOS, `/proc/self/status` `VmHWM` on Linux; other targets report 0.
#[derive(Debug, Default)]
pub struct PeakMemProbe {
    samples: u64,
    peak_rss_bytes: u64,
}

impl PeakMemProbe {
    /// Build a fresh peak-memory probe.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl OsImpactProbe for PeakMemProbe {
    fn sample(&mut self) {
        self.samples += 1;
        self.peak_rss_bytes = self.peak_rss_bytes.max(current_peak_rss_bytes());
    }

    fn report(&self) -> OsImpactReport {
        OsImpactReport {
            samples: self.samples,
            peak_rss_bytes: self.peak_rss_bytes,
        }
    }
}

/// Read this process's peak resident set size in bytes (OS high-water mark).
///
/// macOS: `task_info(MACH_TASK_BASIC_INFO).resident_size_max`. The kernel tracks
/// the peak continuously, so a single poll reflects the max reached so far.
#[cfg(target_os = "macos")]
// libc 0.2 deprecates its `mach` bindings in favor of the `mach2` crate; the
// symbols are stable within 0.2 (semver) and identical, so a scoped allow beats
// pulling a whole extra crate into a bench harness for one syscall.
#[allow(deprecated)]
fn current_peak_rss_bytes() -> u64 {
    // SAFETY: `task_info` writes into a correctly-sized, zeroed
    // `mach_task_basic_info` and reads `count` for the buffer length; on
    // `KERN_SUCCESS` the struct is fully initialized.
    unsafe {
        let mut info: libc::mach_task_basic_info = std::mem::zeroed();
        let mut count: libc::mach_msg_type_number_t = libc::MACH_TASK_BASIC_INFO_COUNT;
        let kr = libc::task_info(
            libc::mach_task_self(),
            libc::MACH_TASK_BASIC_INFO as libc::task_flavor_t,
            std::ptr::addr_of_mut!(info).cast::<libc::integer_t>(),
            &mut count,
        );
        if kr == libc::KERN_SUCCESS {
            info.resident_size_max as u64
        } else {
            0
        }
    }
}

/// Linux: `/proc/self/status` `VmHWM` (peak RSS), reported in kB.
#[cfg(target_os = "linux")]
fn current_peak_rss_bytes() -> u64 {
    let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
        return 0;
    };
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kb: u64 = rest
                .split_whitespace()
                .next()
                .and_then(|n| n.parse().ok())
                .unwrap_or(0);
            return kb.saturating_mul(1024);
        }
    }
    0
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn current_peak_rss_bytes() -> u64 {
    0
}

/// Deterministic, synchronous executor for scheduler-layer benchmarking.
///
/// Produces one placeholder token per scheduled row (decode: last+1; prefill:
/// last-prompt-token+1). It isolates the engine's CPU scheduling cost from any
/// model compute, so workflow timings here are the scheduler's contribution.
#[derive(Debug, Default)]
pub struct EchoExecutor;

impl BackendExecutor for EchoExecutor {
    type Inflight = StepOutput;

    fn submit(&mut self, plan: &ForwardPlan, _kv: &mut dyn KvPool) -> Result<Self::Inflight> {
        let tokens = plan
            .prefill_rows
            .iter()
            .map(|row| SlotToken {
                slot: row.slot,
                token: row.tokens.last().copied().unwrap_or(0).wrapping_add(1),
                logprob: None,
                top_logprobs: Vec::new(),
                finish: None,
            })
            .chain(plan.decode_rows.iter().map(|row| SlotToken {
                slot: row.slot,
                token: row.last_token.wrapping_add(1),
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

/// One turn of an agent workflow: the user message and how many tokens the
/// assistant generates in reply.
#[derive(Debug, Clone)]
pub struct AgentTurn {
    pub user_tokens: Vec<u32>,
    pub gen_tokens: usize,
}

/// A multi-turn agent workflow with a shared system prompt.
#[derive(Debug, Clone)]
pub struct AgentWorkflow {
    pub system_tokens: Vec<u32>,
    pub turns: Vec<AgentTurn>,
}

impl AgentWorkflow {
    /// A representative coding-agent shape: a `system_len` system prompt and
    /// `num_turns` turns of (`user_len` user tokens -> `gen_len` reply tokens).
    /// Each turn appends to the shared context, so later turns reuse the radix
    /// prefix of earlier ones — the agent KV-reuse case.
    #[must_use]
    pub fn synthetic(system_len: usize, num_turns: usize, user_len: usize, gen_len: usize) -> Self {
        let system_tokens: Vec<u32> = (0..system_len).map(|i| (i as u32 % 30_000) + 2).collect();
        let turns = (0..num_turns)
            .map(|t| AgentTurn {
                // distinct per-turn user tokens (offset by turn so they differ)
                user_tokens: (0..user_len)
                    .map(|i| ((t * user_len + i) as u32 % 30_000) + 40_000)
                    .collect(),
                gen_tokens: gen_len,
            })
            .collect();
        Self {
            system_tokens,
            turns,
        }
    }

    /// Total user-message tokens across all turns.
    #[must_use]
    pub fn total_user_tokens(&self) -> usize {
        self.turns.iter().map(|t| t.user_tokens.len()).sum()
    }
}

/// Per-turn measurement.
#[derive(Debug, Clone)]
pub struct TurnMetric {
    pub turn: usize,
    pub prompt_len: usize,
    pub generated: usize,
    pub ticks: u64,
    /// Time-to-first-token, measured in `engine.step()` calls: the number of
    /// ticks from submitting this turn's request until it first holds a
    /// generated token (i.e. prefill completed and decode produced token #1).
    /// `0` means no token was ever generated for this turn.
    pub ticks_to_first_token: u64,
    pub wall: Duration,
    /// Wall time attributed to the prefill phase of this turn: the sum of
    /// `engine.step()` walls from request submit up to and including the step
    /// that commits the first generated token. This is the turn-local analog of
    /// `metal_bench`'s `ttft_ms`.
    pub prefill_wall: Duration,
    /// Wall time attributed to the decode phase of this turn: the sum of
    /// `engine.step()` walls after the first generated token is committed. This
    /// is the turn-local analog of `metal_bench`'s decode window
    /// (`total_time - ttft`), so `generated / decode_wall` is the rewrite's
    /// PURE steady decode tok/s — the apples-to-apples comparison to the legacy
    /// `generation_tps`.
    pub decode_wall: Duration,
}

/// Aggregate workflow measurement.
#[derive(Debug, Clone)]
pub struct WorkflowMetrics {
    pub turns: Vec<TurnMetric>,
    pub total_wall: Duration,
    pub total_generated: usize,
    pub total_ticks: u64,
    /// OS-level impact folded from the [`OsImpactProbe`] (all-zero for
    /// EchoExecutor / scheduler-only runs).
    pub os_impact: OsImpactReport,
}

impl WorkflowMetrics {
    /// Scheduler ticks per generated token (lower is leaner).
    #[must_use]
    pub fn ticks_per_token(&self) -> f64 {
        self.total_ticks as f64 / self.total_generated.max(1) as f64
    }

    /// Total prefill-phase wall across all turns.
    #[must_use]
    pub fn total_prefill_wall(&self) -> Duration {
        self.turns.iter().map(|t| t.prefill_wall).sum()
    }

    /// Total decode-phase wall across all turns.
    #[must_use]
    pub fn total_decode_wall(&self) -> Duration {
        self.turns.iter().map(|t| t.decode_wall).sum()
    }

    /// Turn-wall tok/s: `total_generated / total_wall`. Folds in per-turn
    /// prefill + scheduler/poll ticks — the confounded framing.
    #[must_use]
    pub fn turn_wall_tok_s(&self) -> f64 {
        self.total_generated as f64 / self.total_wall.as_secs_f64().max(1e-9)
    }

    /// PURE decode tok/s: `total_generated / total_decode_wall`. The
    /// apples-to-apples analog of `metal_bench`'s `generation_tps` — excludes
    /// prefill and the to-first-token window.
    #[must_use]
    pub fn pure_decode_tok_s(&self) -> f64 {
        self.total_generated as f64 / self.total_decode_wall().as_secs_f64().max(1e-9)
    }
}

/// Shared, single-threaded observation cell for [`TtftObserver`].
///
/// The bench keeps a clone of this handle so it can drive the per-step tick and
/// read back the first-token tick without an `Engine::executor_mut` accessor
/// (engine-core exposes none, and this crate may not add one).
#[derive(Debug, Default, Clone)]
pub struct TtftHandle(Rc<RefCell<TtftState>>);

#[derive(Debug, Default)]
struct TtftState {
    /// Tick the bench is currently on (1-based; set before each `step()`).
    current_tick: u64,
    /// Tick at which the first generated token first exists, once observed.
    first_token_tick: Option<u64>,
}

impl TtftHandle {
    fn set_tick(&self, tick: u64) {
        self.0.borrow_mut().current_tick = tick;
    }

    fn reset_turn(&self) {
        let mut s = self.0.borrow_mut();
        s.current_tick = 0;
        s.first_token_tick = None;
    }

    fn take_first_token_tick(&self) -> u64 {
        self.0.borrow_mut().first_token_tick.take().unwrap_or(0)
    }

    fn note_first_token(&self) {
        let mut s = self.0.borrow_mut();
        if s.first_token_tick.is_none() {
            // Token is committed when this plan's output is applied, which is at
            // the top of the *next* step — so it first exists one tick later.
            s.first_token_tick = Some(s.current_tick + 1);
        }
    }

    fn observed(&self) -> bool {
        self.0.borrow().first_token_tick.is_some()
    }
}

/// Executor wrapper that observes per-request time-to-first-token in scheduler
/// ticks, without reaching into engine-core internals.
///
/// The engine commits a request's **first** generated token exactly when its
/// final prefill chunk lands (the chunk whose `start_pos + tokens.len()` reaches
/// `total_tokens`) or, after that, on any decode row. That fact is visible in
/// the [`ForwardPlan`] the engine hands the executor on `submit`. The bench tags
/// each tick via the shared [`TtftHandle`]; the first plan that will commit a
/// token records `tick + 1` — the token is applied at the *start* of the
/// following `step()`, so it first exists one tick later.
///
/// This is correct for the single-request-per-turn drive regardless of how many
/// chunked-prefill steps or poll-drain (`NotReady`) ticks the wrapped backend
/// inserts, because it keys off the plan content, not a tick count.
#[derive(Debug)]
pub struct TtftObserver<E> {
    inner: E,
    handle: TtftHandle,
}

impl<E> TtftObserver<E> {
    /// Wrap a backend executor for TTFT observation, returning the executor and
    /// a shared [`TtftHandle`] the bench drives.
    pub fn new(inner: E) -> (Self, TtftHandle) {
        let handle = TtftHandle::default();
        (
            Self {
                inner,
                handle: handle.clone(),
            },
            handle,
        )
    }

    /// Whether `plan` commits the request's first generated token: any decode
    /// row (already past first token), or a final prefill chunk.
    fn plan_commits_first_token(plan: &ForwardPlan) -> bool {
        if !plan.decode_rows.is_empty() {
            return true;
        }
        plan.prefill_rows
            .iter()
            .any(|row| row.start_pos + row.tokens.len() >= row.total_tokens)
    }
}

impl<E: BackendExecutor> BackendExecutor for TtftObserver<E> {
    type Inflight = E::Inflight;

    fn submit(&mut self, plan: &ForwardPlan, kv: &mut dyn KvPool) -> Result<Self::Inflight> {
        if !self.handle.observed() && Self::plan_commits_first_token(plan) {
            self.handle.note_first_token();
        }
        self.inner.submit(plan, kv)
    }

    fn poll(&mut self, inflight: Self::Inflight) -> Result<PollResult<Self::Inflight>> {
        self.inner.poll(inflight)
    }

    // The observer must be transparent: forward the remaining seam methods to the
    // inner executor so wrapped benches exercise the real backend's warmup
    // (graph JIT / prewarm) and stop-token contract, not the seam's no-op default.
    fn warmup(&mut self) -> Result<()> {
        self.inner.warmup()
    }

    fn model_stop_token_ids(&self) -> Vec<u32> {
        self.inner.model_stop_token_ids()
    }
}

/// Drive a single-agent (c=1) workflow turn by turn through `engine`, returning
/// per-turn and aggregate metrics with an all-zero OS-impact report (the
/// scheduler-layer default). The context grows each turn so the engine's radix
/// cache reuses the shared prefix across turns.
///
/// `engine`'s executor must be a [`TtftObserver`]; pass the [`TtftHandle`] it
/// returned so per-turn TTFT can be read from the forward plan without
/// engine-core internals.
pub fn run_agent_workflow<E, K>(
    engine: &mut Engine<TtftObserver<E>, K>,
    ttft: &TtftHandle,
    workflow: &AgentWorkflow,
) -> Result<WorkflowMetrics>
where
    E: BackendExecutor,
    K: KvPool,
{
    run_agent_workflow_with_probe(engine, ttft, workflow, &mut NoopProbe)
}

/// Drive a single-agent (c=1) workflow turn by turn through `engine`, sampling
/// `probe` once after each turn and folding its report into the result.
///
/// For EchoExecutor runs the probe is a [`NoopProbe`] (no OS impact to
/// attribute); a real-backend harness threads a [`PeakMemProbe`].
pub fn run_agent_workflow_with_probe<E, K, P>(
    engine: &mut Engine<TtftObserver<E>, K>,
    ttft: &TtftHandle,
    workflow: &AgentWorkflow,
    probe: &mut P,
) -> Result<WorkflowMetrics>
where
    E: BackendExecutor,
    K: KvPool,
    P: OsImpactProbe + ?Sized,
{
    let mut context: Vec<u32> = workflow.system_tokens.clone();
    let mut turns = Vec::with_capacity(workflow.turns.len());
    let mut total_generated = 0usize;
    let mut total_ticks = 0u64;
    let total_start = Instant::now();

    for (i, turn) in workflow.turns.iter().enumerate() {
        let mut prompt = context.clone();
        prompt.extend_from_slice(&turn.user_tokens);
        let prompt_len = prompt.len();

        ttft.reset_turn();
        let handle = engine.submit_request(prompt, turn.gen_tokens);
        let start = Instant::now();
        let mut ticks = 0u64;
        // Split per-step wall into prefill vs decode phases. The first generated
        // token is committed at the *start* of the step following the
        // first-token-committing plan (see `TtftHandle::note_first_token`), so a
        // step belongs to the prefill phase iff no first token had been observed
        // when the step began. Summing per-phase step walls gives the turn-local
        // analog of `metal_bench`'s ttft / decode windows.
        let mut prefill_wall = Duration::ZERO;
        let mut decode_wall = Duration::ZERO;
        while !engine.is_idle() {
            ticks += 1;
            // Attribute this step's submit to `ticks` so the observer can map
            // the first-token-committing plan to a tick.
            ttft.set_tick(ticks);
            let in_prefill_phase = !ttft.observed();
            let step_start = Instant::now();
            engine.step()?;
            let step_wall = step_start.elapsed();
            if in_prefill_phase {
                prefill_wall += step_wall;
            } else {
                decode_wall += step_wall;
            }
        }
        let wall = start.elapsed();
        let ticks_to_first_token = ttft.take_first_token_tick();

        let generated = engine
            .completed(handle)
            .map(|c| c.generated_tokens.clone())
            .unwrap_or_default();

        // Append the user turn + assistant reply to the shared context so the
        // next turn reuses this prefix (radix KV reuse).
        context.extend_from_slice(&turn.user_tokens);
        context.extend_from_slice(&generated);

        total_generated += generated.len();
        total_ticks += ticks;
        turns.push(TurnMetric {
            turn: i,
            prompt_len,
            generated: generated.len(),
            ticks,
            ticks_to_first_token,
            wall,
            prefill_wall,
            decode_wall,
        });

        // One OS-impact sample per turn.
        probe.sample();
    }

    Ok(WorkflowMetrics {
        turns,
        total_wall: total_start.elapsed(),
        total_generated,
        total_ticks,
        os_impact: probe.report(),
    })
}

/// Build the default (MLX-free) scheduler-layer engine: an [`EchoExecutor`]
/// wrapped in a [`TtftObserver`], backed by a host-side [`MetalKvPool`].
///
/// Returns the engine and the shared [`TtftHandle`] to pass into
/// [`run_agent_workflow`].
#[must_use]
pub fn echo_engine() -> (Engine<TtftObserver<EchoExecutor>, MetalKvPool>, TtftHandle) {
    let mut config = SchedulerConfig::for_slots(4);
    config.max_prompt_tokens = 32_768;
    config.max_total_tokens = 65_536;
    config.chunked_prefill_size = 512;
    let (executor, ttft) = TtftObserver::new(EchoExecutor);
    // page_size 16, 8192 pages -> 131072 token capacity
    let engine = Engine::with_config(executor, MetalKvPool::new(4, 8192, 16), config)
        .expect("echo engine config is always accepted");
    (engine, ttft)
}

/// Build the REAL Metal (MLX) engine from a model path or HuggingFace id,
/// wrapping `infer_metal::MetalExecutor` in a [`TtftObserver`] backed by an
/// `infer_metal::MetalKvPool`. Available only under the `metal` feature; the
/// default build stays EchoExecutor + MLX-free.
///
/// This is the harness entry point for driving the on-device agent workflow
/// once R3 lands the real MLX generation loop. Returns the engine and the
/// shared [`TtftHandle`] to pass into [`run_agent_workflow_with_probe`]
/// together with a [`PeakMemProbe`].
#[cfg(feature = "metal")]
#[allow(clippy::type_complexity)]
pub fn metal_engine_from_model_path(
    model_path: impl AsRef<std::path::Path>,
) -> Result<(
    Engine<TtftObserver<infer_metal::MetalExecutor>, MetalKvPool>,
    TtftHandle,
)> {
    let mut config = SchedulerConfig::for_slots(4);
    config.max_prompt_tokens = 32_768;
    config.max_total_tokens = 65_536;
    // Keep the real Metal agent bench visibly chunked: turn 1 spans several
    // prefill chunks, while later radix hits only prefill the uncached suffix.
    config.chunked_prefill_size = 64;
    let executor = infer_metal::MetalExecutor::from_model_path(model_path)?;
    let (executor, ttft) = TtftObserver::new(executor);
    // page_size 16, 8192 pages -> 131072 token capacity
    let engine = Engine::with_config(executor, MetalKvPool::new(4, 8192, 16), config)?;
    Ok((engine, ttft))
}

/// Build an `Engine` driving the real CUDA executor (clean BF16 Qwen3) over
/// `infer_cuda::CudaKvPool`. Available only under the `cuda` feature; this is the
/// H20 greedy-parity harness entry point. The clean R6 CUDA forward is single
/// scheduled row, so the engine runs one request at a time (`num_slots = 1`).
#[cfg(feature = "cuda")]
pub fn cuda_engine_from_model_path(
    model_path: impl AsRef<std::path::Path>,
) -> Result<Engine<infer_cuda::CudaExecutor, infer_cuda::CudaKvPool>> {
    let requested_pages = 8192; // page_size 16 -> 131072 token capacity (floor)
    let mut config = SchedulerConfig::for_slots(1);
    config.max_prompt_tokens = 32_768;
    config.max_total_tokens = 65_536;
    // Dense Qwen3 profiles its shared paged pool from measured free VRAM
    // (mem_fraction_static 0.9, the serve default); the host KV pool must mirror
    // the ACTUAL device page count, not the requested floor.
    let executor = infer_cuda::CudaExecutor::from_qwen3_bf16_safetensors(
        model_path,
        1,
        requested_pages,
        infer_cuda::CudaKvCacheDtype::default(),
        0.9,
    )?;
    let total_pages = executor.effective_total_pages().unwrap_or(requested_pages);
    let engine = Engine::with_config(
        executor,
        infer_cuda::CudaKvPool::new(1, total_pages, 16),
        config,
    )?;
    Ok(engine)
}

/// Build a CUDA engine over the Qwen3.5/3.6 HYBRID forward (gated-delta linear +
/// periodic full attention, BF16 MoE / dense MLP — `crate::qwen35`). Same
/// `Engine<CudaExecutor, CudaKvPool>` type as the dense path; only the executor
/// constructor differs. The Qwen3.5 executor owns its KV state internally, so
/// the host `CudaKvPool` just paginates the logical token budget.
#[cfg(feature = "cuda")]
pub fn cuda_qwen35_engine_from_model_path(
    model_path: impl AsRef<std::path::Path>,
) -> Result<Engine<infer_cuda::CudaExecutor, infer_cuda::CudaKvPool>> {
    let total_pages = 8192; // page_size 16 -> 131072 token capacity
    let mut config = SchedulerConfig::for_slots(1);
    config.max_prompt_tokens = 32_768;
    config.max_total_tokens = 65_536;
    // mem_fraction_static 0.9 (the serve default): the full-attn KV is a shared
    // profile-sized paged pool; the host CudaKvPool below admits the requested
    // `total_pages` (a valid subset of the profiled device pool ≥ admission floor).
    let executor = infer_cuda::CudaExecutor::from_qwen35_safetensors(
        model_path,
        1,
        total_pages,
        config.max_total_tokens,
        infer_cuda::CudaKvCacheDtype::Bf16,
        0.9,
        None,
        0.0,
        0.0,
        None,
        None,
        None,
        None,
    )?;
    let engine = Engine::with_config(
        executor,
        infer_cuda::CudaKvPool::new(1, total_pages, 16),
        config,
    )?;
    Ok(engine)
}

// Concurrent (c>=2) drive — validates scheduler/executor behavior when multiple
// requests are submitted at once.

/// Result of driving N concurrent requests through an engine to completion.
#[derive(Debug, Clone)]
pub struct ConcurrentResult {
    /// Per-request generated token ids, in submission order.
    pub generated: Vec<Vec<u32>>,
    /// `Err` message if any `engine.step()` failed, else `None`.
    pub step_error: Option<String>,
    /// Scheduler ticks taken to drain (0 if a step errored).
    pub ticks: u64,
}

impl ConcurrentResult {
    /// FNV-1a fingerprint of every request's token stream (order-stable), for a
    /// cross-process matched A/B of greedy determinism.
    #[must_use]
    pub fn fingerprint(&self) -> u64 {
        let mut hash: u64 = 0xcbf29ce484222325;
        for tokens in &self.generated {
            for &t in tokens {
                for byte in t.to_le_bytes() {
                    hash ^= u64::from(byte);
                    hash = hash.wrapping_mul(0x100000001b3);
                }
            }
            // separator so [[1],[2]] != [[1,2]]
            hash ^= 0xff;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }
}

/// Submit all `requests` (`(prompt, gen_tokens)`) into `engine` at once, then
/// drive to idle. Returns each request's generated tokens, the first step error
/// (if any), and the tick count. A step error is captured rather than
/// propagated so tests can assert either successful scheduling or a loud backend
/// failure depending on the executor capability under test.
pub fn drive_concurrent<E, K>(
    engine: &mut Engine<E, K>,
    requests: &[(Vec<u32>, usize)],
) -> ConcurrentResult
where
    E: BackendExecutor,
    K: KvPool,
{
    let handles: Vec<_> = requests
        .iter()
        .map(|(prompt, gen_tokens)| engine.submit_request(prompt.clone(), *gen_tokens))
        .collect();
    let mut ticks = 0u64;
    let mut step_error = None;
    while !engine.is_idle() {
        ticks += 1;
        if let Err(e) = engine.step() {
            step_error = Some(e.to_string());
            break;
        }
        // Guard against a runaway loop if a step error leaves work stuck.
        if ticks > 1_000_000 {
            step_error = Some("drive_concurrent exceeded tick cap".to_string());
            break;
        }
    }
    let generated = handles
        .iter()
        .map(|h| {
            engine
                .completed(*h)
                .map(|c| c.generated_tokens.clone())
                .unwrap_or_default()
        })
        .collect();
    let ticks = if step_error.is_some() { 0 } else { ticks };
    ConcurrentResult {
        generated,
        step_error,
        ticks,
    }
}
