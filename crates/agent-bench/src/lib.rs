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
                finish: None,
            })
            .chain(plan.decode_rows.iter().map(|row| SlotToken {
                slot: row.slot,
                token: row.last_token.wrapping_add(1),
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
    let engine = Engine::with_config(executor, MetalKvPool::new(4, 8192, 16), config);
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
    let engine = Engine::with_config(executor, MetalKvPool::new(4, 8192, 16), config);
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
    );
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
    )?;
    let engine = Engine::with_config(
        executor,
        infer_cuda::CudaKvPool::new(1, total_pages, 16),
        config,
    );
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agent_workflow_runs_and_grows_context() -> Result<()> {
        let (mut engine, ttft) = echo_engine();
        let wf = AgentWorkflow::synthetic(64, 3, 8, 4);
        let m = run_agent_workflow(&mut engine, &ttft, &wf)?;
        assert_eq!(m.turns.len(), 3);
        // Each turn's prompt is longer than the previous (context grows).
        assert!(m.turns[1].prompt_len > m.turns[0].prompt_len);
        assert!(m.turns[2].prompt_len > m.turns[1].prompt_len);
        // Each turn generated its requested reply length.
        assert!(m.turns.iter().all(|t| t.generated == 4));
        assert_eq!(m.total_generated, 12);
        Ok(())
    }

    #[test]
    fn ttft_is_observed_and_precedes_decode() -> Result<()> {
        let (mut engine, ttft) = echo_engine();
        // Small prompts that fit one prefill chunk (chunked_prefill_size=512).
        let wf = AgentWorkflow::synthetic(64, 3, 8, 4);
        let m = run_agent_workflow(&mut engine, &ttft, &wf)?;
        for t in &m.turns {
            // A token was produced, so TTFT is a real (non-zero) tick.
            assert!(
                t.ticks_to_first_token > 0,
                "turn {} never observed a first token",
                t.turn
            );
            // First token cannot arrive after the turn finished decoding.
            assert!(
                t.ticks_to_first_token <= t.ticks,
                "turn {} TTFT {} exceeds total ticks {}",
                t.turn,
                t.ticks_to_first_token,
                t.ticks
            );
            // With 4 generated tokens, decode adds 3 ticks after first token,
            // so TTFT is at most ticks - (generated - 1).
            assert!(
                t.ticks_to_first_token <= t.ticks - (t.generated as u64 - 1),
                "turn {} TTFT {} leaves no room for {} decode ticks (total {})",
                t.turn,
                t.ticks_to_first_token,
                t.generated - 1,
                t.ticks
            );
        }
        Ok(())
    }

    #[test]
    fn ttft_counts_chunked_prefill_steps() -> Result<()> {
        let (mut engine, ttft) = echo_engine();
        // System prompt (1024) far exceeds chunked_prefill_size (512), so the
        // first turn's prefill spans multiple ticks before the first token.
        let wf = AgentWorkflow::synthetic(1024, 1, 8, 2);
        let m = run_agent_workflow(&mut engine, &ttft, &wf)?;
        let t = &m.turns[0];
        // > 512 prompt tokens => at least 2 prefill ticks before first token.
        assert!(
            t.ticks_to_first_token >= 2,
            "chunked prefill should take >=2 ticks to first token, got {}",
            t.ticks_to_first_token
        );
        Ok(())
    }

    #[test]
    fn noop_probe_yields_zero_os_impact() -> Result<()> {
        let (mut engine, ttft) = echo_engine();
        let wf = AgentWorkflow::synthetic(64, 3, 8, 4);
        // Default path uses NoopProbe.
        let m = run_agent_workflow(&mut engine, &ttft, &wf)?;
        assert_eq!(m.os_impact, OsImpactReport::default());
        assert_eq!(m.os_impact.peak_rss_bytes, 0);
        Ok(())
    }

    #[test]
    fn probe_is_sampled_once_per_turn() -> Result<()> {
        let (mut engine, ttft) = echo_engine();
        let wf = AgentWorkflow::synthetic(64, 5, 8, 4);
        let mut probe = PeakMemProbe::new();
        let m = run_agent_workflow_with_probe(&mut engine, &ttft, &wf, &mut probe)?;
        // One sample per turn folded into the report.
        assert_eq!(m.os_impact.samples, 5);
        // On macOS/Linux the peak-RSS syscall returns this running test binary's
        // high-water mark, always > 0. Other targets are a documented no-op.
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        assert!(
            m.os_impact.peak_rss_bytes > 0,
            "peak RSS should be measured on this platform"
        );
        Ok(())
    }

    #[test]
    #[ignore = "benchmark; run with --release -- --ignored --nocapture"]
    fn bench_agent_workflow_scheduler() -> Result<()> {
        // Representative coding-agent shape: 512-token system prompt, 6 turns,
        // 64-token user msgs, 96-token replies. Context grows each turn.
        let wf = AgentWorkflow::synthetic(512, 6, 64, 96);
        let (mut engine, ttft) = echo_engine();
        let m = run_agent_workflow(&mut engine, &ttft, &wf)?;
        eprintln!(
            "[agent-workflow scheduler] turns={} system={} total_user={} total_gen={} \
             total_wall={:?} total_ticks={} ticks_per_token={:.3} os_impact={:?}",
            wf.turns.len(),
            wf.system_tokens.len(),
            wf.total_user_tokens(),
            m.total_generated,
            m.total_wall,
            m.total_ticks,
            m.ticks_per_token(),
            m.os_impact,
        );
        for t in &m.turns {
            eprintln!(
                "  turn {} prompt_len={} gen={} ticks={} ttft_ticks={} wall={:?} \
                 (per-turn task latency)",
                t.turn, t.prompt_len, t.generated, t.ticks, t.ticks_to_first_token, t.wall
            );
        }
        Ok(())
    }

    /// First REAL end-to-end agent-workflow bench on the new engine: drives
    /// `Engine<MetalExecutor, MetalKvPool>` over a multi-turn workflow with the
    /// real MLX Qwen3.5-0.8B forward. Run:
    ///   CUDARC_CUDA_VERSION=12060 cargo test --release -p agent-bench --features metal \
    ///     bench_agent_workflow_metal_qwen35_08b -- --ignored --nocapture
    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "real Metal e2e bench; needs --features metal + the cached model"]
    fn bench_agent_workflow_metal_qwen35_08b() -> Result<()> {
        let model = "mlx-community/Qwen3.5-0.8B-MLX-4bit";
        let (mut engine, ttft) = metal_engine_from_model_path(model)?;
        // Multi-turn growing context: turn 1 chunks the full prompt; later
        // turns attach the page-aligned radix prefix and prefill only the new
        // suffix before greedy decode.
        let wf = AgentWorkflow::synthetic(256, 3, 32, 48);
        let mut probe = PeakMemProbe::new();
        let m = run_agent_workflow_with_probe(&mut engine, &ttft, &wf, &mut probe)?;
        eprintln!(
            "[agent-workflow METAL Qwen3.5-0.8B] turns={} total_gen={} total_wall={:?} \
             turn_wall_tok_s={:.1} prefill_wall={:?} decode_wall={:?} PURE_decode_tok_s={:.1} \
             peak_rss_gb={:.2} os_impact={:?}",
            wf.turns.len(),
            m.total_generated,
            m.total_wall,
            m.turn_wall_tok_s(),
            m.total_prefill_wall(),
            m.total_decode_wall(),
            m.pure_decode_tok_s(),
            m.os_impact.peak_rss_bytes as f64 / (1u64 << 30) as f64,
            m.os_impact
        );
        for t in &m.turns {
            let turn_decode_tok_s = t.generated as f64 / t.decode_wall.as_secs_f64().max(1e-9);
            eprintln!(
                "  turn {} prompt_len={} gen={} ttft_ticks={} wall={:?} \
                 prefill_wall={:?} decode_wall={:?} decode_tok_s={:.1}",
                t.turn,
                t.prompt_len,
                t.generated,
                t.ticks_to_first_token,
                t.wall,
                t.prefill_wall,
                t.decode_wall,
                turn_decode_tok_s,
            );
        }
        assert!(m.turns.len() >= 2);
        assert!(
            m.turns[1].ticks_to_first_token < m.turns[0].ticks_to_first_token,
            "turn 2 should reuse turn 1's prefix and reach first token faster: turn1={} turn2={}",
            m.turns[0].ticks_to_first_token,
            m.turns[1].ticks_to_first_token
        );
        Ok(())
    }

    /// CANONICAL Metal verification on the project's production model
    /// (`mlx-community/Qwen3.6-35B-A3B-4bit`, MoE — per CLAUDE.md the unified
    /// Metal target). Drives the rewrite `Engine<MetalExecutor, MetalKvPool>`
    /// over a multi-turn agent workflow: confirms the rewrite Metal MoE forward
    /// runs end-to-end + prefix reuse holds on the real MoE shape, and reports
    /// tok/s. Run:
    ///   cargo test --release -p agent-bench --no-default-features --features metal,no-cuda \
    ///     bench_agent_workflow_metal_qwen36_canonical -- --ignored --nocapture
    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "real Metal MoE e2e bench; needs --features metal + the ~19GB cached Qwen3.6-35B-A3B-4bit"]
    fn bench_agent_workflow_metal_qwen36_canonical() -> Result<()> {
        let model = "mlx-community/Qwen3.6-35B-A3B-4bit";
        let (mut engine, ttft) = metal_engine_from_model_path(model)?;
        let wf = AgentWorkflow::synthetic(256, 3, 32, 48);
        let mut probe = PeakMemProbe::new();
        let m = run_agent_workflow_with_probe(&mut engine, &ttft, &wf, &mut probe)?;
        eprintln!(
            "[agent-workflow METAL Qwen3.6-35B-A3B-4bit] turns={} total_gen={} \
             total_wall={:?} turn_wall_tok_s={:.1} prefill_wall={:?} decode_wall={:?} \
             PURE_decode_tok_s={:.1} peak_rss_gb={:.2} os_impact={:?}",
            wf.turns.len(),
            m.total_generated,
            m.total_wall,
            m.turn_wall_tok_s(),
            m.total_prefill_wall(),
            m.total_decode_wall(),
            m.pure_decode_tok_s(),
            m.os_impact.peak_rss_bytes as f64 / (1u64 << 30) as f64,
            m.os_impact
        );
        for t in &m.turns {
            let turn_decode_tok_s = t.generated as f64 / t.decode_wall.as_secs_f64().max(1e-9);
            eprintln!(
                "  turn {} prompt_len={} gen={} ttft_ticks={} wall={:?} \
                 prefill_wall={:?} decode_wall={:?} decode_tok_s={:.1}",
                t.turn,
                t.prompt_len,
                t.generated,
                t.ticks_to_first_token,
                t.wall,
                t.prefill_wall,
                t.decode_wall,
                turn_decode_tok_s,
            );
        }
        assert!(m.turns.len() >= 2);
        assert!(
            m.turns[1].ticks_to_first_token < m.turns[0].ticks_to_first_token,
            "turn 2 should reuse turn 1's prefix: turn1={} turn2={}",
            m.turns[0].ticks_to_first_token,
            m.turns[1].ticks_to_first_token
        );
        Ok(())
    }

    /// c>=2 plan-shape fact (CPU, no MLX): with two requests admitted at once
    /// and `num_slots=2`, the planner batches both into a single tick's plan —
    /// proving a genuine concurrent workload produces a multi-row `ForwardPlan`.
    /// Scalar backends report a row cap to the scheduler so this multi-row shape
    /// is only produced for backends that can accept it.
    #[test]
    fn concurrent_plan_batches_multiple_rows() -> Result<()> {
        // Probe executor that records the max rows seen in any single plan.
        #[derive(Default)]
        struct MaxRowProbe {
            max_rows: std::rc::Rc<std::cell::Cell<usize>>,
        }
        impl BackendExecutor for MaxRowProbe {
            type Inflight = StepOutput;
            fn submit(
                &mut self,
                plan: &ForwardPlan,
                _kv: &mut dyn KvPool,
            ) -> Result<Self::Inflight> {
                let rows = plan.prefill_rows.len() + plan.decode_rows.len();
                self.max_rows.set(self.max_rows.get().max(rows));
                let tokens = plan
                    .prefill_rows
                    .iter()
                    .map(|row| SlotToken {
                        slot: row.slot,
                        token: row.tokens.last().copied().unwrap_or(0).wrapping_add(1),
                        logprob: None,
                        finish: None,
                    })
                    .chain(plan.decode_rows.iter().map(|row| SlotToken {
                        slot: row.slot,
                        token: row.last_token.wrapping_add(1),
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

        let executor = MaxRowProbe::default();
        let max_rows = executor.max_rows.clone();
        let mut config = SchedulerConfig::for_slots(2);
        config.chunked_prefill_size = 64;
        let mut engine = Engine::with_config(executor, MetalKvPool::new(2, 256, 16), config);
        let res = drive_concurrent(
            &mut engine,
            &[(vec![10, 11, 12, 13], 4), (vec![20, 21, 22, 23], 4)],
        );
        assert!(res.step_error.is_none(), "echo path should not error");
        assert_eq!(res.generated.len(), 2);
        assert!(
            max_rows.get() >= 2,
            "two concurrent requests must batch >=2 rows into one plan, saw max {}",
            max_rows.get()
        );
        Ok(())
    }

    /// REAL Metal c=1 greedy fingerprint (Qwen3.5-0.8B). Drives a single greedy
    /// request to a fixed length and prints the FNV fingerprint of the generated
    /// ids + the pipeline fast-path hit count. Run once with the pipeline on
    /// (default) and once off (`infer_metal::apply_runtime_flags` with
    /// `pipeline: false` before engine build); greedy is
    /// deterministic given the prompt, so the fingerprints MUST match
    /// bit-for-bit — the pipeline path feeds the same argmax token into the next
    /// `step_session` the HEAD path would. The hit count proves the fast path
    /// fired (on) / stayed silent (off) at c=1.
    ///   CUDARC_CUDA_VERSION=12060 cargo test --release -p agent-bench \
    ///     --no-default-features --features metal,no-cuda \
    ///     metal_c1_greedy_fingerprint_qwen35_08b -- --ignored --nocapture
    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "real Metal c=1 greedy fingerprint; needs --features metal + cached Qwen3.5-0.8B-MLX-4bit"]
    fn metal_c1_greedy_fingerprint_qwen35_08b() -> Result<()> {
        let model = "mlx-community/Qwen3.5-0.8B-MLX-4bit";
        let hits_before = infer_metal::pipeline_fast_path_hits();
        let (mut engine, _ttft) = metal_engine_from_model_path(model)?;
        let prompt: Vec<u32> = vec![9707, 11, 1879, 358, 1079, 264, 6722, 13];
        let res = drive_concurrent(&mut engine, &[(prompt, 32)]);
        let hits_after = infer_metal::pipeline_fast_path_hits();
        eprintln!(
            "[metal c=1 greedy Qwen3.5-0.8B] step_error={:?} \
             gen_len={} fingerprint={:#018x} pipeline_hits_delta={} gen={:?}",
            res.step_error,
            res.generated.first().map_or(0, Vec::len),
            res.fingerprint(),
            hits_after - hits_before,
            res.generated.first(),
        );
        assert!(res.step_error.is_none(), "c=1 must not error: {res:?}");
        assert_eq!(
            res.generated.first().map_or(0, Vec::len),
            32,
            "expected 32 greedy tokens"
        );
        Ok(())
    }

    /// REAL Metal prefix-reuse CORRECTNESS gate on the canonical Qwen3.6 MoE
    /// (gated-delta linear-attention recurrent state + conv ring — the model
    /// whose cross-request reuse `702454fe` disabled on CUDA as unsound).
    ///
    /// Decisive question: does `materialize_slot_from_prefix` reconstruct the
    /// recurrent/GDR state so a slot resumed at `start_pos > 0` decodes
    /// identically to a fresh cold prefill? Metal greedy is deterministic (see
    /// `metal_c1_greedy_fingerprint_qwen35_08b`), so the floor is exact match.
    ///
    /// One model load: drive prompt `P` cold (caches `P`), then drive `P` again
    /// — the radix now serves a partial prefix, so the second request
    /// reconstructs the slot at the deepest cached page boundary, prefills only
    /// the sub-page tail, and decodes. `G_reuse == G_cold` bit-for-bit ⇒ reuse
    /// is sound (keep the clamp + page-align fix); a mismatch ⇒ recurrent reuse
    /// corrupts state (revert and disable it on Metal too, mirroring 702454fe).
    /// `ticks_reuse < ticks_cold` proves reuse actually engaged (skipped prefill).
    ///   CUDARC_CUDA_VERSION=12060 cargo test --release -p agent-bench \
    ///     --no-default-features --features metal,no-cuda \
    ///     metal_prefix_reuse_parity_qwen36 -- --ignored --nocapture
    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "real Metal prefix-reuse parity; needs --features metal + cached Qwen3.6-35B-A3B-4bit"]
    fn metal_prefix_reuse_parity_qwen36() -> Result<()> {
        let model = "mlx-community/Qwen3.6-35B-A3B-4bit";
        // 200 raw ids (avoid 0 = engine STOP). 200 tokens = 12 full pages + an
        // 8-token tail at page_size 16, so the second request reuses 192 tokens
        // and re-prefills only [192,200): the real partial-reuse path.
        let prompt: Vec<u32> = (10u32..210).collect();
        let n_gen = 24usize;

        let (mut engine, _ttft) = metal_engine_from_model_path(model)?;
        let cold = drive_concurrent(&mut engine, &[(prompt.clone(), n_gen)]);
        let reuse = drive_concurrent(&mut engine, &[(prompt.clone(), n_gen)]);
        eprintln!(
            "[metal prefix-reuse parity Qwen3.6] cold(err={:?} ticks={} fp={:#018x}) \
             reuse(err={:?} ticks={} fp={:#018x})",
            cold.step_error,
            cold.ticks,
            cold.fingerprint(),
            reuse.step_error,
            reuse.ticks,
            reuse.fingerprint(),
        );

        assert!(cold.step_error.is_none(), "cold run errored: {cold:?}");
        assert!(reuse.step_error.is_none(), "reuse run errored: {reuse:?}");
        let g_cold = cold.generated.first().cloned().unwrap_or_default();
        let g_reuse = reuse.generated.first().cloned().unwrap_or_default();
        assert_eq!(g_cold.len(), n_gen, "cold run must decode {n_gen} tokens");
        assert_eq!(g_reuse.len(), n_gen, "reuse run must decode {n_gen} tokens");
        assert!(
            reuse.ticks < cold.ticks,
            "reuse must skip prefill (engaged): cold ticks {} vs reuse ticks {}",
            cold.ticks,
            reuse.ticks
        );
        assert_eq!(
            g_reuse, g_cold,
            "prefix-reuse greedy continuation must match cold prefill bit-for-bit \
             — a mismatch means recurrent/GDR state reconstruction is unsound"
        );
        Ok(())
    }

    /// H20 greedy-parity harness for the clean CUDA BF16 Qwen3 forward.
    ///
    /// Raw token ids (no tokenizer): greedy decode is deterministic given the ids,
    /// so the legacy CUDA path on the SAME ids + model must produce the SAME
    /// continuation. Run this (prints NEW token ids), then run the legacy CUDA
    /// greedy path on the same `PARITY_MODEL` + prompt and compare the arrays.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "real CUDA H20 greedy parity; needs --features cuda + a BF16 Qwen3 safetensors model + GPU"]
    fn cuda_qwen3_greedy_parity() -> Result<()> {
        let model = std::env::var("PARITY_MODEL")
            .unwrap_or_else(|_| "/data01/models/Qwen3-0.6B".to_string());
        let max_new: usize = std::env::var("PARITY_MAX_NEW")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(16);
        // Fixed prompt token ids (avoid 0 = engine STOP id). Greedy continuation
        // of these exact ids is the parity fingerprint.
        let prompt: Vec<u32> = vec![9707, 11, 1879, 358, 1079, 264, 6722, 13];
        let mut engine = cuda_engine_from_model_path(&model)?;
        let handle = engine.submit_request(prompt.clone(), max_new);
        engine.run_to_idle()?;
        let completed = engine
            .completed(handle)
            .ok_or_else(|| anyhow::anyhow!("request did not complete"))?;
        eprintln!(
            "[cuda-parity NEW] model={model} prompt={prompt:?} gen={:?} finish={:?}",
            completed.generated_tokens, completed.finish
        );
        assert!(
            !completed.generated_tokens.is_empty(),
            "expected at least one generated token"
        );
        Ok(())
    }

    /// Greedy-parity harness for the clean CUDA Qwen3.5/3.6 HYBRID BF16 forward
    /// (gated-delta linear + periodic full attention). Same contract as
    /// [`cuda_qwen3_greedy_parity`] but drives the `qwen35` executor path.
    ///
    /// Env: `QWEN35_PARITY_MODEL` (safetensors dir), `PARITY_MAX_NEW` (default
    /// 16), `PARITY_PROMPT_IDS` (comma-separated u32; default below). Prints
    /// `clean_tokens=[...]`; compare against the HF/legacy greedy reference on
    /// the SAME ids + model (raw-id greedy decode is deterministic).
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "real CUDA greedy parity; needs --features cuda + a BF16 Qwen3.5 safetensors model + GPU"]
    fn cuda_qwen35_greedy_parity() -> Result<()> {
        let model = std::env::var("QWEN35_PARITY_MODEL")
            .map_err(|_| anyhow::anyhow!("set QWEN35_PARITY_MODEL to a Qwen3.5 safetensors dir"))?;
        let max_new: usize = std::env::var("PARITY_MAX_NEW")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(16);
        let prompt: Vec<u32> = match std::env::var("PARITY_PROMPT_IDS") {
            Ok(s) => s
                .split(',')
                .map(|t| t.trim().parse::<u32>())
                .collect::<std::result::Result<_, _>>()
                .map_err(|e| anyhow::anyhow!("bad PARITY_PROMPT_IDS: {e}"))?,
            Err(_) => vec![9707, 11, 1879, 358, 1079, 264, 6722, 13],
        };
        let mut engine = cuda_qwen35_engine_from_model_path(&model)?;
        let handle = engine.submit_request(prompt.clone(), max_new);
        engine.run_to_idle()?;
        let completed = engine
            .completed(handle)
            .ok_or_else(|| anyhow::anyhow!("request did not complete"))?;
        eprintln!(
            "[qwen35-cuda-parity] model={model} prompt={prompt:?} finish={:?}",
            completed.finish
        );
        eprintln!("clean_tokens={:?}", completed.generated_tokens);
        assert!(
            !completed.generated_tokens.is_empty(),
            "expected at least one generated token"
        );
        Ok(())
    }

    // -----------------------------------------------------------------------
    // DSv4 KV-precision-parity gate (re-port of the deleted
    // `infer/tests/kv_precision_parity.rs`).
    //
    // WHY: every DSv4 decode win this session (FlashMLA +18%, fused wqkv +5%,
    // contiguous MoE +12.78%) ships default-OFF because there was no re-ported
    // cross-precision gate to license a default flip. This is that gate.
    //
    // DSv4 KV PRECISION SURFACE (verified against `infer-cuda/src/attention.rs`
    // + `dsv4.rs`, 2026-06): the DSv4 decode attention path exposes exactly two
    // KV precisions, selected at runtime — not via a `KVCacheDtype`/`KVFormat`
    // enum (those were `infer::model::*` types deleted in the R5 cutover), but
    // via process-local dispatch overrides + gate envs:
    //   * scalar bf16 REFERENCE — `set_dsv4_flashmla_decode_override(Some(false))`
    //     (== `--dsv4-flashmla-decode false`). Decode reads the BF16 window cache.
    //   * FlashMLA FP8-KV     — `set_dsv4_flashmla_decode_override(Some(true))`
    //     (== `--dsv4-flashmla-decode true`). Decode packs the KV into the FP8
    //     arena (`dsv4.rs` `fp8_kv_pool`, `dsv4_fp8_kv_pack`).
    // There is NO INT8 / TQ4 / TurboQuant KV path for DSv4 — the legacy
    // `KVFormat::{INT8, FP8E4M3, TurboQuant}` matrix targeted the dense Qwen3
    // paged pool, which DSv4 does not use. So INT8 ≥ 99% / TQ4 ≥ 80% gates have
    // no precision to bind to here and are intentionally OMITTED (documented,
    // not faked). The fused-wqkv / contiguous-MoE wins are decode *compute*
    // variants on top of the FP8-KV path; they are exercised as extra
    // report-only rows so a single audit covers all three default-flip
    // candidates against the same scalar reference.
    //
    // GATE (legacy thresholds, mapped to the precisions that exist):
    //   * scalar self-parity = 100% (sanity: the reference vs itself).
    //   * FlashMLA FP8-KV ≥ 95% trajectory match vs scalar bf16 — THIS is the
    //     gate that licenses the `--dsv4-flashmla-decode` default flip.
    //   * fused-wqkv / contig-MoE rows are report-only (compute fusions, not a
    //     new KV precision; license via the perf A/B + this trajectory monitor).
    //
    // DEGENERATE-BASELINE GUARD (per
    // `errors/2026-05-26-fp8-kv-catastrophic-was-test-artifact.md`): if the
    // scalar reference collapses to a single-token repetition loop, trajectory
    // match measures noise-fidelity, not quality — warn loudly and skip the
    // gate assertion rather than draw a false conclusion.
    //
    // OUTPUT: per-precision metrics to `target/kv-parity-dsv4-<unix>.json`.
    //
    // ENV KNOBS (mirroring the legacy gate where they map):
    //   * INFER_DSV4_MODEL_PATH  required — DSv4 FP8 safetensors dir.
    //   * INFER_DSV4_PROMPT_IDS  comma-separated DeepSeek ids per prompt; repeat
    //                            the flag-less list separated by `;` for multiple
    //                            prompts. Defaults to one ORACLE-anchored prompt.
    //   * KV_PARITY_PROMPTS      cap the number of prompts used (default = all).
    //   * KV_PARITY_MAX_TOKENS / KV_PARITY_PROFILE  decode horizon (64 full /
    //                            4 smoke), same semantics as the legacy gate.
    // -----------------------------------------------------------------------

    /// One DSv4 KV/attention precision (or compute variant) under audit.
    #[cfg(feature = "cuda")]
    #[derive(Clone, Copy, Debug)]
    struct Dsv4PrecisionCase {
        name: &'static str,
        /// `--dsv4-flashmla-decode` dispatch (false = scalar bf16 reference).
        flashmla: bool,
        /// Fused wqkv decode linear (compute variant on the FP8-KV path).
        fused_wqkv: bool,
        /// Contiguous-MoE decode (compute variant on the FP8-KV path).
        contig_moe: bool,
        /// Minimum trajectory match (mean over prompts) to pass the gate.
        /// `None` = report-only.
        gate_trajectory: Option<f32>,
    }

    #[cfg(feature = "cuda")]
    fn dsv4_precision_matrix() -> Vec<Dsv4PrecisionCase> {
        vec![
            Dsv4PrecisionCase {
                name: "scalar_bf16",
                flashmla: false,
                fused_wqkv: false,
                contig_moe: false,
                gate_trajectory: Some(1.0), // self-parity reference
            },
            Dsv4PrecisionCase {
                name: "flashmla_fp8",
                flashmla: true,
                fused_wqkv: false,
                contig_moe: false,
                // The FP8-KV gate. ≥ 0.95 licenses the FlashMLA default flip.
                gate_trajectory: Some(0.95),
            },
            // Compute fusions on top of FP8-KV — report-only (no new KV
            // precision; the perf A/B + this trajectory monitor license them).
            Dsv4PrecisionCase {
                name: "flashmla_fp8_fused_wqkv",
                flashmla: true,
                fused_wqkv: true,
                contig_moe: false,
                gate_trajectory: None,
            },
            Dsv4PrecisionCase {
                name: "flashmla_fp8_fused_wqkv_contig_moe",
                flashmla: true,
                fused_wqkv: true,
                contig_moe: true,
                gate_trajectory: None,
            },
        ]
    }

    /// Result of decoding the prompt set under one precision.
    #[cfg(feature = "cuda")]
    #[derive(Debug)]
    struct Dsv4PrecisionResult {
        name: &'static str,
        /// Per-prompt generated token id sequences.
        sequences: Vec<Vec<u32>>,
    }

    /// Trajectory diff of a candidate precision against the scalar reference.
    #[cfg(feature = "cuda")]
    #[derive(Debug)]
    struct Dsv4DiffRow {
        name: &'static str,
        per_prompt_match: Vec<f32>,
        mean_match: f32,
        first_diverging_prompt: Option<usize>,
        first_diverging_step: Option<usize>,
        gate: Option<f32>,
        gate_passed: Option<bool>,
    }

    /// Decode horizon: explicit `KV_PARITY_MAX_TOKENS`, else 64 (full) / 4
    /// (smoke) via `KV_PARITY_PROFILE` — same semantics as the legacy gate's
    /// `support/kv_parity_config.rs`.
    #[cfg(feature = "cuda")]
    fn dsv4_kv_parity_max_tokens() -> Result<usize> {
        if let Ok(raw) = std::env::var("KV_PARITY_MAX_TOKENS") {
            let parsed: usize = raw
                .parse()
                .map_err(|e| anyhow::anyhow!("parse KV_PARITY_MAX_TOKENS={raw:?}: {e}"))?;
            anyhow::ensure!(parsed > 0, "KV_PARITY_MAX_TOKENS must be > 0");
            return Ok(parsed);
        }
        Ok(match std::env::var("KV_PARITY_PROFILE").ok().as_deref() {
            None | Some("") | Some("full") | Some("FULL") => 64,
            Some("smoke") | Some("SMOKE") => 4,
            Some(other) => {
                anyhow::bail!("unsupported KV_PARITY_PROFILE={other:?}; expected 'full' or 'smoke'")
            }
        })
    }

    /// Default prompt set as DeepSeek token ids. Anchored on the ORACLE prompt
    /// used by the resident A/B example so a base-model greedy continuation is
    /// coherent (avoids the degenerate `!`-loop regime). Override with
    /// `INFER_DSV4_PROMPT_IDS` (`;`-separated lists for multiple prompts).
    #[cfg(feature = "cuda")]
    fn dsv4_default_prompts() -> Vec<Vec<u32>> {
        vec![
            vec![671, 6102, 294, 8760, 344],
            vec![603, 671, 6102, 294, 8760, 344, 11111],
        ]
    }

    #[cfg(feature = "cuda")]
    fn dsv4_parse_prompts(raw: &str) -> Result<Vec<Vec<u32>>> {
        let prompts = raw
            .split(';')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|chunk| {
                let ids: Vec<u32> = chunk
                    .split(',')
                    .map(str::trim)
                    .filter(|t| !t.is_empty())
                    .map(|t| {
                        t.parse::<u32>().map_err(|e| {
                            anyhow::anyhow!("bad token id `{t}` in INFER_DSV4_PROMPT_IDS: {e}")
                        })
                    })
                    .collect::<Result<_>>()?;
                anyhow::ensure!(!ids.is_empty(), "empty prompt in INFER_DSV4_PROMPT_IDS");
                Ok(ids)
            })
            .collect::<Result<Vec<_>>>()?;
        anyhow::ensure!(!prompts.is_empty(), "INFER_DSV4_PROMPT_IDS resolved empty");
        Ok(prompts)
    }

    /// Greedy sampling params (temperature 0 = argmax). Same contract as the
    /// resident A/B example's `greedy()`.
    #[cfg(feature = "cuda")]
    fn dsv4_greedy() -> infer_plan::SamplingParams {
        infer_plan::SamplingParams::default()
    }

    #[cfg(feature = "cuda")]
    fn dsv4_prefill_plan(tokens: &[u32]) -> ForwardPlan {
        ForwardPlan {
            mode: infer_plan::ForwardMode::Prefill,
            decode_rows: Vec::new(),
            prefill_rows: vec![infer_plan::PrefillRow {
                slot: 0,
                tokens: tokens.to_vec(),
                start_pos: 0,
                total_tokens: tokens.len(),
                params: dsv4_greedy(),
            }],
            microbatch: None,
            spec: None,
        }
    }

    #[cfg(feature = "cuda")]
    fn dsv4_decode_plan(last_token: u32, kv_seq_len: usize) -> ForwardPlan {
        ForwardPlan {
            mode: infer_plan::ForwardMode::Decode,
            decode_rows: vec![infer_plan::DecodeRow {
                slot: 0,
                last_token,
                kv_seq_len,
                params: dsv4_greedy(),
            }],
            prefill_rows: Vec::new(),
            microbatch: None,
            spec: None,
        }
    }

    /// One forward step through the resident DSv4 executor (mirrors the
    /// `forward_once` helper in `infer-cuda/examples/dsv4_resident_ab.rs`).
    #[cfg(feature = "cuda")]
    fn dsv4_forward_once(
        exec: &mut infer_cuda::CudaExecutor,
        kv: &mut infer_cuda::CudaKvPool,
        plan: ForwardPlan,
    ) -> Result<u32> {
        let inflight = exec.submit(&plan, kv as &mut dyn KvPool)?;
        match exec.poll(inflight)? {
            PollResult::Ready(out) => out
                .tokens
                .first()
                .map(|t| t.token)
                .ok_or_else(|| anyhow::anyhow!("DSv4 step produced no token")),
            PollResult::NotReady(_) => {
                anyhow::bail!("DSv4 executor resolves synchronously; got NotReady")
            }
        }
    }

    /// Run the prompt set under one precision: select dispatch via the process-
    /// local overrides + contig-MoE env, then greedy-decode each prompt to
    /// `max_tokens`. A fresh `CudaKvPool` per prompt resets host bookkeeping
    /// alongside the executor's prefill `start_pos=0`.
    #[cfg(feature = "cuda")]
    fn dsv4_run_precision(
        exec: &mut infer_cuda::CudaExecutor,
        case: Dsv4PrecisionCase,
        prompts: &[Vec<u32>],
        max_tokens: usize,
    ) -> Result<Dsv4PrecisionResult> {
        infer_cuda::set_dsv4_flashmla_decode_override(Some(case.flashmla));
        infer_cuda::set_dsv4_fused_wqkv_decode_override(Some(case.fused_wqkv));
        infer_cuda::set_dsv4_moe_contig_decode(case.contig_moe);

        let mut sequences = Vec::with_capacity(prompts.len());
        for (idx, prompt) in prompts.iter().enumerate() {
            // One slot, page_size 16; DSv4 owns device KV internally, the host
            // pool only paginates the logical token budget.
            let mut kv = infer_cuda::CudaKvPool::new(1, 8192, 16);
            let first = dsv4_forward_once(exec, &mut kv, dsv4_prefill_plan(prompt))?;
            let mut tokens = vec![first];
            for step in 1..max_tokens {
                let kv_seq_len = prompt.len() + step - 1;
                let last = *tokens.last().expect("tokens is non-empty");
                let tok = dsv4_forward_once(exec, &mut kv, dsv4_decode_plan(last, kv_seq_len))?;
                tokens.push(tok);
            }
            eprintln!(
                "[kv-parity-dsv4] precision={} prompt[{}/{}] tokens={}",
                case.name,
                idx + 1,
                prompts.len(),
                tokens.len()
            );
            sequences.push(tokens);
        }
        Ok(Dsv4PrecisionResult {
            name: case.name,
            sequences,
        })
    }

    /// Trajectory match = common-prefix-length / max(reference_len, 1), averaged
    /// over prompts (the legacy `diff_against_reference` logic).
    #[cfg(feature = "cuda")]
    fn dsv4_diff_against_reference(
        reference: &Dsv4PrecisionResult,
        candidate: &Dsv4PrecisionResult,
        gate: Option<f32>,
    ) -> Dsv4DiffRow {
        assert_eq!(
            reference.sequences.len(),
            candidate.sequences.len(),
            "reference and candidate must share prompt count"
        );
        let mut first_diverging_prompt = None;
        let mut first_diverging_step = None;
        let per_prompt_match: Vec<f32> = reference
            .sequences
            .iter()
            .zip(candidate.sequences.iter())
            .enumerate()
            .map(|(idx, (ref_seq, cand_seq))| {
                let common = ref_seq
                    .iter()
                    .zip(cand_seq.iter())
                    .take_while(|(r, c)| r == c)
                    .count();
                let denom = ref_seq.len().max(1);
                if first_diverging_prompt.is_none()
                    && (common < ref_seq.len().min(cand_seq.len())
                        || ref_seq.len() != cand_seq.len())
                {
                    first_diverging_prompt = Some(idx);
                    first_diverging_step = Some(common);
                }
                common as f32 / denom as f32
            })
            .collect();
        let mean_match = if per_prompt_match.is_empty() {
            0.0
        } else {
            per_prompt_match.iter().sum::<f32>() / per_prompt_match.len() as f32
        };
        let gate_passed = gate.map(|g| mean_match >= g - 1e-6);
        Dsv4DiffRow {
            name: candidate.name,
            per_prompt_match,
            mean_match,
            first_diverging_prompt,
            first_diverging_step,
            gate,
            gate_passed,
        }
    }

    /// Emit `target/kv-parity-dsv4-<unix>.json` (hand-rolled, no serde dep —
    /// matches the bench crate's `metal_greedy_parity_gold` style).
    #[cfg(feature = "cuda")]
    fn dsv4_write_json_report(
        max_tokens: usize,
        num_prompts: usize,
        degenerate: bool,
        rows: &[Dsv4DiffRow],
    ) -> Result<std::path::PathBuf> {
        let unix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("target");
        std::fs::create_dir_all(&dir).ok();
        let out = dir.join(format!("kv-parity-dsv4-{unix}.json"));

        let mut buf = String::new();
        buf.push_str("{\n");
        buf.push_str("  \"model\": \"dsv4\",\n");
        buf.push_str(&format!("  \"unix_ts\": {unix},\n"));
        buf.push_str(&format!("  \"max_tokens\": {max_tokens},\n"));
        buf.push_str(&format!("  \"num_prompts\": {num_prompts},\n"));
        buf.push_str(&format!("  \"degenerate_baseline\": {degenerate},\n"));
        buf.push_str("  \"precisions\": [\n");
        for (i, row) in rows.iter().enumerate() {
            let trailing = if i + 1 == rows.len() { "" } else { "," };
            buf.push_str("    {\n");
            buf.push_str(&format!("      \"name\": \"{}\",\n", row.name));
            buf.push_str(&format!("      \"mean_match\": {:.6},\n", row.mean_match));
            buf.push_str(&format!(
                "      \"per_prompt_match\": [{}],\n",
                row.per_prompt_match
                    .iter()
                    .map(|v| format!("{v:.4}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
            match (row.first_diverging_prompt, row.first_diverging_step) {
                (Some(p), Some(s)) => {
                    buf.push_str(&format!("      \"first_diverging_prompt\": {p},\n"));
                    buf.push_str(&format!("      \"first_diverging_step\": {s},\n"));
                }
                _ => {
                    buf.push_str("      \"first_diverging_prompt\": null,\n");
                    buf.push_str("      \"first_diverging_step\": null,\n");
                }
            }
            match (row.gate, row.gate_passed) {
                (Some(g), Some(p)) => {
                    buf.push_str(&format!("      \"gate\": {g:.4},\n"));
                    buf.push_str(&format!("      \"gate_passed\": {p}\n"));
                }
                _ => {
                    buf.push_str("      \"gate\": null,\n");
                    buf.push_str("      \"gate_passed\": null\n");
                }
            }
            buf.push_str(&format!("    }}{trailing}\n"));
        }
        buf.push_str("  ]\n}\n");
        std::fs::write(&out, buf).map_err(|e| anyhow::anyhow!("write kv-parity report: {e}"))?;
        Ok(out)
    }

    /// DSv4 KV-precision-parity gate. Loads the executor ONCE, runs the same
    /// prompt set through every precision, computes trajectory match vs the
    /// scalar bf16 reference, and asserts the FP8 ≥ 95% gate that licenses the
    /// FlashMLA default flip. Needs an 8×H20 pod + DSv4 weights — `#[ignore]`d
    /// so CI stays green (mirrors `cuda_*_greedy_parity` / `metal_*_greedy_parity`).
    ///
    /// Run (pod):
    ///   CUDARC_CUDA_VERSION=12090 INFER_DSV4_MODEL_PATH=/path/to/dsv4-fp8 \
    ///     cargo test --release -p agent-bench --features cuda \
    ///     dsv4_kv_precision_parity -- --ignored --nocapture --test-threads=1
    /// (multi-rank TP=8: add `--features cuda,nccl` + INFER_NCCL_ID_FILE per the
    /// resident A/B harness; this single-rank form is the default driving shape.)
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "DSv4 KV-precision-parity gate; needs --features cuda + 8xH20 pod + DSv4 FP8 weights"]
    fn dsv4_kv_precision_parity() -> Result<()> {
        let model_path = std::env::var("INFER_DSV4_MODEL_PATH").map_err(|_| {
            anyhow::anyhow!(
                "INFER_DSV4_MODEL_PATH must point at the DSv4 FP8 safetensors directory"
            )
        })?;

        let max_tokens = dsv4_kv_parity_max_tokens()?;
        let mut prompts = match std::env::var("INFER_DSV4_PROMPT_IDS") {
            Ok(raw) => dsv4_parse_prompts(&raw)?,
            Err(_) => dsv4_default_prompts(),
        };
        if let Some(cap) = std::env::var("KV_PARITY_PROMPTS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
        {
            prompts.truncate(cap.max(1));
        }
        let cases = dsv4_precision_matrix();

        eprintln!(
            "[kv-parity-dsv4] model={model_path} tokens/prompt={max_tokens} \
             prompts={} precisions={}",
            prompts.len(),
            cases.len()
        );

        // The FlashMLA FP8 arena is a load-time capability gate independent of
        // per-step dispatch; allocate it so the scalar-first ordering still lets
        // the later FlashMLA precision pack KV (mirrors the resident A/B).
        // SAFETY: process startup, before the executor builds CUDA/NCCL state.
        unsafe {
            std::env::set_var("ARLE_DSV4_FLASHMLA_DECODE_ALLOC", "1");
            std::env::set_var("ARLE_DSV4_FUSED_WQKV_DECODE_ALLOC", "1");
            // Decode graph capture can mask step-ordering bugs; keep it off for
            // the audit (the resident A/B defaults the same way).
            std::env::set_var("ARLE_DSV4_DECODE_GRAPH", "0");
        }

        // KV parity/precision audit: short prompts, <=64 decode steps
        // (`dsv4_kv_parity_max_tokens`) — 4096 is generous headroom, not a
        // tunable knob.
        let mut exec = infer_cuda::CudaExecutor::from_dsv4_fp8_safetensors(
            &model_path,
            1,
            4096,
            None,
            None,
            None,
            0.5,
            0.0,
        )
        .map_err(|e| anyhow::anyhow!("from_dsv4_fp8_safetensors failed: {e:#}"))?;

        // Reference = scalar bf16.
        let ref_case = cases
            .iter()
            .find(|c| c.name == "scalar_bf16")
            .copied()
            .expect("scalar_bf16 must be in the matrix");
        let reference = dsv4_run_precision(&mut exec, ref_case, &prompts, max_tokens)?;

        // Degenerate-baseline guard: a scalar reference that repeats a single
        // token makes trajectory match a noise-fidelity metric, not quality.
        let degenerate_baseline = reference
            .sequences
            .iter()
            .any(|seq| seq.len() >= 8 && seq.iter().take(8).all(|&t| t == seq[0]));
        if degenerate_baseline {
            let dump: Vec<&[u32]> = reference
                .sequences
                .iter()
                .map(|s| &s[..s.len().min(8)])
                .collect();
            eprintln!(
                "[kv-parity-dsv4] WARNING degenerate scalar reference detected (one or more \
                 prompts repeat a single token for the first 8 generated tokens). \
                 FP8 quality conclusions from this run are INVALID — trajectory match is \
                 measuring noise-fidelity, not quality. Reference first-8 per prompt: {dump:?}. \
                 SKIPPING the gate assertion (see \
                 errors/2026-05-26-fp8-kv-catastrophic-was-test-artifact.md)."
            );
        }

        let mut rows = vec![dsv4_diff_against_reference(
            &reference,
            &reference,
            ref_case.gate_trajectory,
        )];
        for case in cases.iter().filter(|c| c.name != "scalar_bf16") {
            let result = dsv4_run_precision(&mut exec, *case, &prompts, max_tokens)?;
            // Token-level divergence dump for prompt 0 — catastrophic-vs-noise.
            if let (Some(ref_seq), Some(cand_seq)) =
                (reference.sequences.first(), result.sequences.first())
            {
                let n = ref_seq.len().min(cand_seq.len()).min(8);
                eprintln!(
                    "[kv-parity-dsv4] {:<34} prompt0 first{} tokens: ref={:?} cand={:?}",
                    result.name,
                    n,
                    &ref_seq[..n],
                    &cand_seq[..n]
                );
            }
            rows.push(dsv4_diff_against_reference(
                &reference,
                &result,
                case.gate_trajectory,
            ));
        }

        // Restore env-driven dispatch.
        infer_cuda::set_dsv4_flashmla_decode_override(None);
        infer_cuda::set_dsv4_fused_wqkv_decode_override(None);

        let report = dsv4_write_json_report(max_tokens, prompts.len(), degenerate_baseline, &rows)?;
        eprintln!("[kv-parity-dsv4] report: {}", report.display());
        for row in &rows {
            eprintln!(
                "[kv-parity-dsv4] {:<34} mean_match={:.4} first_div={:?}/{:?} gate={:?} passed={:?}",
                row.name,
                row.mean_match,
                row.first_diverging_prompt,
                row.first_diverging_step,
                row.gate,
                row.gate_passed,
            );
        }

        if degenerate_baseline {
            // Audit invalid — do not assert (already warned loudly above).
            return Ok(());
        }

        // Gather every gate failure and assert once so a single run surfaces all
        // regressed precisions (the legacy multi-line panic strategy).
        let failed: Vec<&Dsv4DiffRow> = rows
            .iter()
            .filter(|r| matches!(r.gate_passed, Some(false)))
            .collect();
        if !failed.is_empty() {
            let mut msg = String::from("DSv4 KV precision parity gate failures:\n");
            for row in &failed {
                msg.push_str(&format!(
                    "  - {}: mean_match={:.4} < gate={:.4} (first divergence prompt={:?} step={:?})\n",
                    row.name,
                    row.mean_match,
                    row.gate.unwrap_or(0.0),
                    row.first_diverging_prompt,
                    row.first_diverging_step,
                ));
            }
            msg.push_str(&format!("Full report: {}\n", report.display()));
            panic!("{msg}");
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Metal greedy-parity regression harness.
    //
    // The legacy `infer/src/backend/metal` path was deleted, so Metal numeric
    // changes shipped with NO oracle (CUDA has `cuda_qwen3_greedy_parity` /
    // `cuda_qwen35_greedy_parity`; Metal had none). These tests close that gap.
    //
    // ORACLE MODEL: there is no live legacy Metal path to diff against, so the
    // oracle is a PINNED snapshot — a committed gold continuation + FNV-1a
    // fingerprint per (model, prompt, max_new). Raw-id greedy decode is
    // deterministic, so the MLX Qwen3.5 forward must reproduce the exact same
    // ids across refactors. The gold lives at
    // `crates/agent-bench/test_data/metal_greedy_parity_gold.json`.
    //
    // To FIRST establish the gold (or re-bless after an intended numeric
    // change), run with `METAL_PARITY_BLESS=1`: the test prints the fresh
    // continuation + fingerprint and SKIPS the assertion. Paste those into the
    // JSON, commit, and from then on the test is a hard regression gate.
    //
    // Cross-check against an *independent* oracle (HF transformers greedy on
    // the same ids) once at bless time — see the module doc on the JSON file —
    // so the pinned snapshot is anchored to ground truth, not just to itself.
    // -----------------------------------------------------------------------

    /// Parse one model's gold entry (`gen`, `fingerprint`) out of the committed
    /// `metal_greedy_parity_gold.json` without pulling a JSON dep into the bench
    /// crate. Returns `(prompt_ids, max_new, gold_gen, gold_fingerprint)`.
    /// `gold_gen` empty / fingerprint 0 means the entry is still PENDING.
    #[cfg(feature = "metal")]
    fn load_metal_gold(model: &str) -> Result<(Vec<u32>, usize, Vec<u32>, u64)> {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/test_data/metal_greedy_parity_gold.json"
        );
        let raw =
            std::fs::read_to_string(path).map_err(|e| anyhow::anyhow!("read gold {path}: {e}"))?;

        // Tiny hand-rolled extraction (top-level `prompt_ids`, `max_new`, and
        // the per-model `gen` / `fingerprint`) — avoids a serde_json dep in the
        // bench crate while staying robust to whitespace/formatting.
        fn u32_array_after(haystack: &str, key: &str) -> Option<Vec<u32>> {
            let start = haystack.find(key)? + key.len();
            let open = haystack[start..].find('[')? + start;
            let close = haystack[open..].find(']')? + open;
            let inner = &haystack[open + 1..close];
            let out = inner
                .split(',')
                .map(str::trim)
                .filter(|t| !t.is_empty())
                .map(|t| t.parse::<u32>().ok())
                .collect::<Option<Vec<_>>>()?;
            Some(out)
        }

        let prompt_ids = u32_array_after(&raw, "\"prompt_ids\":")
            .ok_or_else(|| anyhow::anyhow!("missing prompt_ids in gold"))?;
        let max_new = {
            let k = "\"max_new\":";
            let s = raw
                .find(k)
                .ok_or_else(|| anyhow::anyhow!("missing max_new"))?
                + k.len();
            raw[s..]
                .trim_start()
                .split([',', '\n', '}'])
                .next()
                .and_then(|v| v.trim().parse::<usize>().ok())
                .ok_or_else(|| anyhow::anyhow!("bad max_new"))?
        };

        // Scope to the model's object: slice from the model key to the next `}`.
        let model_key = format!("\"{model}\"");
        let mstart = raw
            .find(&model_key)
            .ok_or_else(|| anyhow::anyhow!("model '{model}' absent from gold"))?;
        let mslice = &raw[mstart..];
        let mend = mslice.find('}').unwrap_or(mslice.len());
        let mslice = &mslice[..mend];

        let gold_gen = u32_array_after(mslice, "\"gen\":").unwrap_or_default();
        let gold_fp = {
            let k = "\"fingerprint\":";
            mslice
                .find(k)
                .and_then(|i| {
                    let s = &mslice[i + k.len()..];
                    let q1 = s.find('"')? + 1;
                    let q2 = s[q1..].find('"')? + q1;
                    let hex = s[q1..q2].trim_start_matches("0x");
                    u64::from_str_radix(hex, 16).ok()
                })
                .unwrap_or(0)
        };

        Ok((prompt_ids, max_new, gold_gen, gold_fp))
    }

    /// Drive `max_new` greedy tokens through the real Metal engine for `model`
    /// against the committed gold continuation. Shared body for the small-model
    /// CI gate and the canonical-MoE gate.
    ///
    /// `METAL_PARITY_BLESS=1` prints the fresh continuation + fingerprint and
    /// skips the assertion (use to establish / re-bless the gold).
    #[cfg(feature = "metal")]
    fn run_metal_greedy_parity(model: &str) -> Result<()> {
        let (prompt, max_new, gold_gen, gold_fp) = load_metal_gold(model)?;

        let (mut engine, _ttft) = metal_engine_from_model_path(model)?;
        // Single greedy request to a fixed length. `drive_concurrent` with one
        // request gives us the FNV fingerprint helper for free and captures any
        // step error instead of panicking.
        let res = drive_concurrent(&mut engine, &[(prompt.clone(), max_new)]);
        assert!(
            res.step_error.is_none(),
            "c=1 greedy step errored: {:?}",
            res.step_error
        );
        let gen_tokens = res.generated.first().cloned().unwrap_or_default();
        let fp = res.fingerprint();

        let bless = std::env::var("METAL_PARITY_BLESS").as_deref() == Ok("1");
        eprintln!(
            "[metal-greedy-parity] model={model} bless={bless} prompt={prompt:?} \
             max_new={max_new} gen_len={} fingerprint={fp:#018x} gen={gen_tokens:?}",
            gen_tokens.len()
        );

        if bless {
            eprintln!(
                "[metal-greedy-parity BLESS] paste into test_data/metal_greedy_parity_gold.json \
                 under \"{model}\":\n  \"gen\": {gen_tokens:?},\n  \"fingerprint\": \"{fp:#018x}\""
            );
            // Bless mode is generative, not a gate.
            return Ok(());
        }

        assert!(
            !gold_gen.is_empty() && gold_fp != 0,
            "no gold committed for '{model}' yet — run once with METAL_PARITY_BLESS=1, \
             paste the printed gen/fingerprint into \
             crates/agent-bench/test_data/metal_greedy_parity_gold.json, then commit"
        );
        // gen_len must hit max_new (no early STOP) for the snapshot to be
        // meaningful; a short gen means the prompt hit an EOS — pick a prompt
        // that doesn't, or shorten max_new at bless time.
        assert_eq!(
            gen_tokens.len(),
            max_new,
            "expected {max_new} greedy tokens, got {} (early stop?)",
            gen_tokens.len()
        );
        // Hard regression gate: token-exact + fingerprint match.
        assert_eq!(
            fp, gold_fp,
            "Metal greedy fingerprint drifted for '{model}': got {fp:#018x}, gold {gold_fp:#018x} \
             — the MLX forward changed numerically. If intended, re-bless via METAL_PARITY_BLESS=1."
        );
        assert_eq!(
            gen_tokens, gold_gen,
            "Metal greedy continuation drifted for '{model}' (fingerprint collision or partial \
             change): got {gen_tokens:?}, gold {gold_gen:?}"
        );
        Ok(())
    }

    /// CHEAP CI GATE — Metal greedy parity on the small model
    /// (`Qwen3.5-0.8B-MLX-4bit`, ~0.5 GB). Token-exact regression gate against
    /// the committed gold: the MLX Qwen3.5 forward must reproduce the pinned
    /// greedy continuation bit-for-bit. This is the oracle that was missing —
    /// any numeric change to `infer-metal` / `mlx-sys` Qwen3.5 ops now fails
    /// here instead of shipping silently.
    ///   CUDARC_CUDA_VERSION=12060 cargo test --release -p agent-bench \
    ///     --no-default-features --features metal,no-cuda \
    ///     metal_qwen35_greedy_parity -- --ignored --nocapture
    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "real Metal greedy parity; needs --features metal + cached Qwen3.5-0.8B-MLX-4bit"]
    fn metal_qwen35_greedy_parity() -> Result<()> {
        run_metal_greedy_parity("mlx-community/Qwen3.5-0.8B-MLX-4bit")
    }

    /// CANONICAL GATE — Metal greedy parity on the production MoE model
    /// (`Qwen3.6-35B-A3B-4bit`, ~19 GB — per CLAUDE.md the unified Metal
    /// target). Catches MoE-specific numeric regressions (router / expert /
    /// gated-delta paths) that the dense 0.8B model cannot surface. Heavier;
    /// run on demand, not every CI tick.
    ///   cargo test --release -p agent-bench --no-default-features \
    ///     --features metal,no-cuda \
    ///     metal_qwen36_greedy_parity -- --ignored --nocapture
    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "real Metal MoE greedy parity; needs --features metal + ~19GB cached Qwen3.6-35B-A3B-4bit"]
    fn metal_qwen36_greedy_parity() -> Result<()> {
        run_metal_greedy_parity("mlx-community/Qwen3.6-35B-A3B-4bit")
    }
}
