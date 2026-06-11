//! Metal backend executor + session machinery.
//!
//! `new()` keeps a CPU placeholder so the submit/poll seam stays testable without
//! the `metal` feature; `from_model_path()` builds the real MLX Qwen3.5 executor.
//! `RealMetalExecutor` and all MLX-touching session state are gated behind
//! `#[cfg(feature = "metal")]`.

#[cfg(feature = "metal")]
use std::collections::HashMap;
#[cfg(feature = "metal")]
use std::path::Path;

use infer_plan::{ForwardPlan, SlotToken, StepOutput};
use infer_seam::{BackendExecutor, KvPool, PollResult};

#[cfg(feature = "metal")]
use crate::{config, mlx, model_source, qwen35, wired_limit};

#[cfg(feature = "metal")]
const KV_CACHE_CHUNK: i32 = 256;

/// Metal KV-cache storage dtype. The host `MetalKvPool` remains a logical page
/// allocator; this controls the MLX arrays inside each Metal slot.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MetalKvCacheDtype {
    Bf16,
    #[default]
    Int8,
}

impl MetalKvCacheDtype {
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Int8 => "int8",
        }
    }
}

/// Cross-step decode pipelining (env-gated, default ON since
/// `wins/2026-06-04-metal-decode-pipeline-c2-safe-default-on.md`).
///
/// HEAD decode is strictly submit(N) → poll(N) blocks on `eval` → apply(N) →
/// submit(N+1): the GPU idles for the host gap between poll(N)'s eval finishing
/// and submit(N+1) kicking `step_session` again (apply_output + admission +
/// plan-N+1 build + a fresh `begin_session`). With the pipeline on the decode
/// session is held open across steps and `submit_decode` eagerly issues the
/// NEXT greedy step's `step_session` (async) inside the current submit, so step
/// N+1's GPU forward overlaps step N's host token materialization — the proven
/// legacy `pending_sampled` shape, kept one step deep. Single-slot greedy only;
/// a non-greedy or recycled-slot single-row decode drains and takes the cold
/// (HEAD) path via the `pending_matches_live_slot` guard.
///
/// Serve safety: Metal reports one live request and one plan row to the shared
/// layers. The HTTP frontend rejects a second live request, while the executor's
/// single-row submit guard remains an internal fail-closed fence before any
/// pipeline logic. The default-on flip therefore changes only the c=1 greedy
/// path. Opt OUT with `INFER_METAL_PIPELINE=0`.
#[cfg(feature = "metal")]
fn pipeline_decode_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let on = std::env::var("INFER_METAL_PIPELINE")
            .map(|v| !(v == "0" || v.eq_ignore_ascii_case("false")))
            .unwrap_or(true);
        eprintln!("[infer-metal] decode pipeline (INFER_METAL_PIPELINE) = {on}");
        on
    })
}

/// One-shot probe printed the first time the pipeline fast path runs, so a bench
/// can prove the overlapped path is actually live (not just enabled).
#[cfg(feature = "metal")]
fn probe_pipeline_fast_path() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| eprintln!("[infer-metal] pipeline fast path LIVE (overlapped decode)"));
    PIPELINE_FAST_PATH_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Monotonic count of pipeline fast-path firings (process-wide). A test or bench
/// reads this to prove which decode path each step took. Harmless in production:
/// a single relaxed counter on an already-rare event.
#[cfg(feature = "metal")]
pub(crate) static PIPELINE_FAST_PATH_HITS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Read the pipeline fast-path firing count (test/bench observability).
#[cfg(feature = "metal")]
#[must_use]
pub fn pipeline_fast_path_hits() -> u64 {
    PIPELINE_FAST_PATH_HITS.load(std::sync::atomic::Ordering::Relaxed)
}

/// In-flight handle for a submitted Metal step.
pub enum MetalInflight {
    /// CPU placeholder output.
    Ready(StepOutput),
    /// Real MLX greedy sample. `poll` materializes this scalar token.
    #[cfg(feature = "metal")]
    Sampled { slot: usize, sampled: mlx::MlxArray },
}

impl std::fmt::Debug for MetalInflight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ready(output) => f.debug_tuple("Ready").field(output).finish(),
            #[cfg(feature = "metal")]
            Self::Sampled { slot, sampled } => f
                .debug_struct("Sampled")
                .field("slot", slot)
                .field("sampled", sampled)
                .finish(),
        }
    }
}

/// Turn a logits array into an in-flight result under `params`.
///
/// Greedy keeps the device `argmax` + async path. Non-greedy Metal sampling
/// used to materialize host f32 logits and sample on CPU every token, which
/// creates synchronous D2H stalls on the local desktop path. Default behavior is
/// therefore constrained to device greedy; opt back into the old host sampler
/// with `INFER_METAL_HOST_SAMPLING=1` for debugging.
#[cfg(feature = "metal")]
fn sample_inflight(
    slot: usize,
    logits: &mlx::MlxArray,
    params: &infer_plan::SamplingParams,
    position: u64,
) -> MetalInflight {
    if params.is_greedy() || !host_sampling_enabled() {
        if !params.is_greedy() {
            warn_host_sampling_downgrade();
        }
        let sampled = mlx::argmax(logits);
        mlx::async_eval(&[&sampled]);
        return MetalInflight::Sampled { slot, sampled };
    }
    let logits_f32 = mlx::as_dtype(logits, mlx::Dtype::Float32);
    mlx::eval(&[&logits_f32]);
    let token = infer_plan::sample_token(logits_f32.as_slice_f32(), params, position);
    MetalInflight::Ready(StepOutput {
        tokens: vec![SlotToken {
            slot,
            token,
            logprob: None,
            finish: None,
        }],
    })
}

#[cfg(feature = "metal")]
fn host_sampling_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("INFER_METAL_HOST_SAMPLING")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

#[cfg(feature = "metal")]
fn warn_host_sampling_downgrade() {
    use std::sync::atomic::{AtomicBool, Ordering};
    static WARNED: AtomicBool = AtomicBool::new(false);
    if !WARNED.swap(true, Ordering::Relaxed) {
        log::warn!(
            "Metal non-greedy sampling requested, but host logits sampling is disabled; \
             using device greedy argmax. Set INFER_METAL_HOST_SAMPLING=1 to opt into \
             the blocking D2H sampler."
        );
    }
}

/// Metal backend executor.
#[derive(Default)]
pub struct MetalExecutor {
    #[cfg(feature = "metal")]
    real: Option<RealMetalExecutor>,
}

impl std::fmt::Debug for MetalExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug = f.debug_struct("MetalExecutor");
        #[cfg(feature = "metal")]
        debug.field("real", &self.real.is_some());
        debug.finish()
    }
}

impl MetalExecutor {
    /// Build a Metal executor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "metal")]
            real: None,
        }
    }

    /// Build a real single-row greedy MLX Qwen3.5 executor from a local model
    /// path or HuggingFace id.
    #[cfg(feature = "metal")]
    pub fn from_model_path(model_path: impl AsRef<Path>) -> anyhow::Result<Self> {
        Self::from_model_path_with_kv_cache_dtype(model_path, MetalKvCacheDtype::default())
    }

    /// Build a real MLX Qwen3.5/Qwen3.6 executor with an explicit Metal KV dtype.
    #[cfg(feature = "metal")]
    pub fn from_model_path_with_kv_cache_dtype(
        model_path: impl AsRef<Path>,
        kv_cache_dtype: MetalKvCacheDtype,
    ) -> anyhow::Result<Self> {
        let model_source = model_path.as_ref().to_string_lossy();
        let resolved = model_source::resolve_model_path(&model_source)?;
        let _guard = mlx_sys::mlx_guard();
        if let Some(limit) = wired_limit::auto_wired_limit_bytes(&resolved) {
            let previous = mlx::set_wired_limit_bytes(limit as u64);
            log::info!(
                "Metal executor wired limit set to {} bytes (previous {})",
                limit,
                previous
            );
        }
        let config = config::load_metal_config(&resolved)?;
        if kv_cache_dtype == MetalKvCacheDtype::Int8 {
            validate_int8_kv_config(&config)?;
        }
        eprintln!("[infer-metal] kv cache dtype = {}", kv_cache_dtype.label());
        let weights = qwen35::load_qwen35_metal_weights(&resolved, &config)?;
        Ok(Self {
            real: Some(RealMetalExecutor {
                config,
                kv_cache_dtype,
                weights,
                slots: HashMap::new(),
                page_store: MetalPageStore::default(),
                active_session_slot: None,
                pending: None,
            }),
        })
    }

    /// Feature-free placeholder forward: one deterministic token per scheduled
    /// row, so the submit/poll seam is exercisable on CPU without MLX.
    fn placeholder_forward(plan: &ForwardPlan) -> StepOutput {
        let mut tokens = Vec::with_capacity(plan.decode_rows.len() + plan.prefill_rows.len());
        for row in &plan.decode_rows {
            tokens.push(SlotToken {
                slot: row.slot,
                token: row.last_token.wrapping_add(1),
                logprob: None,
                finish: None,
            });
        }
        for row in &plan.prefill_rows {
            let token = row.tokens.last().copied().unwrap_or(0).wrapping_add(1);
            tokens.push(SlotToken {
                slot: row.slot,
                token,
                logprob: None,
                finish: None,
            });
        }
        StepOutput { tokens }
    }
}

impl BackendExecutor for MetalExecutor {
    type Inflight = MetalInflight;

    fn submit(
        &mut self,
        plan: &ForwardPlan,
        kv: &mut dyn KvPool,
    ) -> anyhow::Result<Self::Inflight> {
        #[cfg(feature = "metal")]
        if let Some(real) = self.real.as_mut() {
            return real.submit(plan, kv);
        }
        #[cfg(not(feature = "metal"))]
        let _ = kv;

        Ok(MetalInflight::Ready(Self::placeholder_forward(plan)))
    }

    fn poll(&mut self, inflight: Self::Inflight) -> anyhow::Result<PollResult<Self::Inflight>> {
        match inflight {
            MetalInflight::Ready(output) => Ok(PollResult::Ready(output)),
            #[cfg(feature = "metal")]
            MetalInflight::Sampled { slot, sampled } => {
                let _guard = mlx_sys::mlx_guard();
                mlx::eval(&[&sampled]);
                let token = sampled.item_i32() as u32;
                Ok(PollResult::Ready(StepOutput {
                    tokens: vec![SlotToken {
                        slot,
                        token,
                        logprob: None,
                        finish: None,
                    }],
                }))
            }
        }
    }

    fn model_stop_token_ids(&self) -> Vec<u32> {
        #[cfg(feature = "metal")]
        if let Some(real) = self.real.as_ref() {
            return real.config.stop_token_ids.clone();
        }
        Vec::new()
    }

    fn max_rows_per_step(&self) -> usize {
        1
    }

    fn max_live_requests(&self) -> usize {
        1
    }

    fn reusable_prefix_pages(&self, block_ids: &[u32]) -> usize {
        #[cfg(feature = "metal")]
        if let Some(real) = self.real.as_ref() {
            return real.page_store.reusable_prefix_pages(block_ids);
        }
        block_ids.len()
    }

    fn release_prefix_pages(&mut self, _pages: &[u32]) {
        #[cfg(feature = "metal")]
        if let Some(real) = self.real.as_mut() {
            real.page_store.release_pages(_pages);
        }
    }

    fn warmup(&mut self) -> anyhow::Result<()> {
        #[cfg(feature = "metal")]
        if let Some(real) = self.real.as_mut() {
            return real.warmup();
        }
        Ok(())
    }
}

/// A greedy decode step whose `step_session` was already issued (async) for the
/// slot's next token inside the previous submit. `submit_decode` returns this on
/// the following tick without re-running the forward, so the GPU stayed busy
/// across the host gap. At most one is outstanding; single-slot greedy only.
#[cfg(feature = "metal")]
struct PendingStep {
    slot: usize,
    sampled: mlx::MlxArray,
}

#[cfg(feature = "metal")]
struct RealMetalExecutor {
    config: config::MetalModelConfig,
    kv_cache_dtype: MetalKvCacheDtype,
    weights: qwen35::Qwen35MetalWeights,
    slots: HashMap<usize, MetalSlotState>,
    page_store: MetalPageStore,
    active_session_slot: Option<usize>,
    /// Cross-step decode prequeue (see `pipeline_decode_enabled`).
    pending: Option<PendingStep>,
}

#[cfg(feature = "metal")]
impl RealMetalExecutor {
    /// Pre-build (JIT-compile) the prefill + decode MLX graphs at load so turn-0
    /// is not cold. After the steady-decode pipeline recovery
    /// (`wins/2026-06-04-metal-rewrite-decode-pipeline-recovery`), the residual
    /// turn-wall gap is turn-0's lazy graph build + first MoE encode landing on
    /// the first real request. A tiny throwaway forward on a reserved warmup slot
    /// (never published to the kv pool) pre-pays that JIT at load instead. Opt
    /// out with `INFER_METAL_WARMUP=0`.
    fn warmup(&mut self) -> anyhow::Result<()> {
        let on = std::env::var("INFER_METAL_WARMUP")
            .map(|v| !(v == "0" || v.eq_ignore_ascii_case("false")))
            .unwrap_or(true);
        eprintln!("[infer-metal] warmup (INFER_METAL_WARMUP) = {on}");
        if !on {
            return Ok(());
        }
        let _guard = mlx_sys::mlx_guard();
        let model = self.weights.cpp_model()?;
        // Throwaway warmup state: a reserved slot id, tiny cache, never inserted
        // into `self.slots` or published to the kv pool. Token id 0 is a valid
        // vocab index; the output is discarded — only the graph JIT matters.
        let mut state = MetalSlotState::new(usize::MAX, 0, &self.config, self.kv_cache_dtype, 8);
        state.ensure_session_active(model)?;
        // Tiny prefill → JIT the prefill graph + first MoE encode.
        let prefill = mlx::MlxArray::from_slice_i32(&[0, 0], &[2]);
        let logits = model.prefill_session(&prefill, 2, 0)?;
        mlx::async_eval(&[&logits]);
        state.cache_len = 2;
        // One decode step → JIT the decode-step graph.
        let step = mlx::MlxArray::from_slice_i32(&[0], &[1]);
        let logits = model.step_session(&step, state.cache_len as i32)?;
        mlx::async_eval(&[&logits]);
        state.cache_len += 1;
        // Blocking materialize so the JIT completes before the first request.
        state.drain_session(model)?;
        Ok(())
    }

    fn submit(&mut self, plan: &ForwardPlan, kv: &mut dyn KvPool) -> anyhow::Result<MetalInflight> {
        let _guard = mlx_sys::mlx_guard();
        let row_count = plan.prefill_rows.len() + plan.decode_rows.len();
        anyhow::ensure!(
            row_count == 1,
            "R3a MetalExecutor supports exactly one prefill or decode row, got {row_count}"
        );

        if let Some(row) = plan.prefill_rows.first() {
            return self.submit_prefill(row, kv);
        }
        if let Some(row) = plan.decode_rows.first() {
            return self.submit_decode(row, kv);
        }
        anyhow::bail!("R3a MetalExecutor received a non-idle plan with no rows")
    }

    fn submit_prefill(
        &mut self,
        row: &infer_plan::PrefillRow,
        kv: &mut dyn KvPool,
    ) -> anyhow::Result<MetalInflight> {
        anyhow::ensure!(
            !row.tokens.is_empty(),
            "MetalExecutor prefill row must contain at least one token"
        );
        self.ensure_no_other_active_session(row.slot)?;

        self.reset_slot_if_epoch_changed(row.slot, kv)?;
        if !self.slots.contains_key(&row.slot) {
            let reservation = kv
                .seq_len(row.slot)
                .max(row.total_tokens.saturating_add(512))
                .max(row.tokens.len().saturating_add(1));
            let state = if row.start_pos == 0 {
                MetalSlotState::new(
                    row.slot,
                    kv.slot_epoch(row.slot),
                    &self.config,
                    self.kv_cache_dtype,
                    reservation,
                )
            } else {
                self.page_store.materialize_slot_from_prefix(
                    row.slot,
                    kv.slot_epoch(row.slot),
                    kv,
                    row.start_pos,
                    reservation,
                )?
            };
            self.slots.insert(row.slot, state);
        }

        let model = self.weights.cpp_model()?;
        let slot = self.slots.get_mut(&row.slot).expect("slot inserted above");
        anyhow::ensure!(
            row.start_pos == slot.cache_len,
            "prefill start_pos mismatch for slot {}: plan={}, metal_state={}",
            row.slot,
            row.start_pos,
            slot.cache_len
        );
        // Reservation normally covers the whole prompt; guard against a chunk that
        // would write past it so prefill shares the decode growth invariant.
        slot.ensure_kv_capacity(model, row.tokens.len())?;
        slot.ensure_session_active(model)?;
        self.active_session_slot = Some(row.slot);
        let token_values: Vec<i32> = row.tokens.iter().map(|&token| token as i32).collect();
        let token_arr = mlx::MlxArray::from_slice_i32(&token_values, &[token_values.len() as i32]);
        let logits =
            model.prefill_session(&token_arr, token_values.len() as i32, row.start_pos as i32)?;
        mlx::async_eval(&[&logits]);
        slot.cache_len = row.start_pos + row.tokens.len();
        slot.committed_len = slot.cache_len;
        slot.last_sampled = None;
        let position = slot.cache_len as u64;
        slot.drain_session(model)?;
        self.active_session_slot = None;
        // Publish is prefill-only by design: engine-core's radix cache only ever
        // offers PROMPT pages for attach (`infer-core` prefix.rs:
        // `publishable_tokens = request.prompt_len().min(self.kv.seq_len(slot))`),
        // so pages/snapshots covering generated tokens are unreachable. The old
        // decode-time publishes were a per-token O(full_pages) re-slice plus an
        // unbounded GDR-snapshot leak (`prefixes` is never evicted).
        self.page_store.publish_slot(slot, kv)?;
        // A new prefill restarts this slot's token stream; any decode prequeue
        // from a prior turn is stale.
        self.pending = None;

        Ok(sample_inflight(row.slot, &logits, &row.params, position))
    }

    fn submit_decode(
        &mut self,
        row: &infer_plan::DecodeRow,
        kv: &mut dyn KvPool,
    ) -> anyhow::Result<MetalInflight> {
        // Pipeline fast path: this step's `step_session` was already issued
        // (async) inside the previous submit, with the session left open one step
        // ahead. Drain that now-committed step, prequeue the next one, and
        // return the already-sampled token —
        // no forward on the engine's critical poll path. Greedy + single-slot
        // only. The guard validates the pending against the LIVE slot before
        // reuse: the slot must still be the SAME live state we prequeued from
        // (same epoch — not recycled by `finish_slot` into a different request),
        // its session must still be open, and the engine's committed length must
        // match ours. An exact prefix-cache hit can admit a NEW request straight
        // into `Decoding` on a recycled slot index, which would otherwise return
        // the prior request's stale token; these checks send that case to the
        // cold path (which resets the slot and drops the stale pending).
        if pipeline_decode_enabled()
            && row.params.is_greedy()
            && self.pending_matches_live_slot(row, kv)
        {
            probe_pipeline_fast_path();
            let ready = self.pending.take().expect("pending checked above");
            self.commit_pending_then_prequeue(row, kv)?;
            return Ok(MetalInflight::Sampled {
                slot: ready.slot,
                sampled: ready.sampled,
            });
        }
        // A pending that did not pass the live-slot guard is stale (slot
        // recycled, length drift, …); drop it before the cold path rebuilds.
        if self.pending.as_ref().is_some_and(|p| p.slot == row.slot) {
            self.pending = None;
        }

        self.ensure_no_other_active_session(row.slot)?;
        self.reset_slot_if_epoch_changed(row.slot, kv)?;
        let model = self.weights.cpp_model()?;
        if !self.slots.contains_key(&row.slot) {
            anyhow::ensure!(
                row.kv_seq_len > 0,
                "decode for slot {} before prefill with empty host prefix",
                row.slot
            );
            let reservation = kv.seq_len(row.slot).max(row.kv_seq_len.saturating_add(512));
            let state = self.page_store.materialize_slot_from_prefix(
                row.slot,
                kv.slot_epoch(row.slot),
                kv,
                row.kv_seq_len,
                reservation,
            )?;
            self.slots.insert(row.slot, state);
        }
        let slot = self
            .slots
            .get_mut(&row.slot)
            .ok_or_else(|| anyhow::anyhow!("decode for slot {} before prefill", row.slot))?;
        anyhow::ensure!(
            row.kv_seq_len == slot.committed_len && slot.committed_len == slot.cache_len,
            "decode kv_seq_len mismatch for slot {}: plan={}, committed={}, metal_state={}",
            row.slot,
            row.kv_seq_len,
            slot.committed_len,
            slot.cache_len
        );
        // Grow the flat K/V before this step would write past the reservation —
        // the host pool already grew its pages for this length; the executor
        // must keep pace or `slice_update` silently drops the write.
        slot.ensure_kv_capacity(model, 1)?;
        slot.ensure_session_active(model)?;
        self.active_session_slot = Some(row.slot);
        let token_arr = mlx::MlxArray::from_slice_i32(&[row.last_token as i32], &[1]);
        let logits = model.step_session(&token_arr, slot.cache_len as i32)?;
        mlx::async_eval(&[&logits]);
        slot.cache_len = slot.cache_len.saturating_add(1);
        slot.committed_len = slot.cache_len;
        let position = slot.cache_len as u64;
        slot.drain_session(model)?;
        self.active_session_slot = None;

        let inflight = sample_inflight(row.slot, &logits, &row.params, position);

        // Cold start of a greedy decode run: seed the pipeline. Record this
        // step's sampled token and issue the next step's forward so subsequent
        // ticks take the fast path and overlap.
        if pipeline_decode_enabled()
            && row.params.is_greedy()
            && let MetalInflight::Sampled { sampled, .. } = &inflight
        {
            if let Some(slot) = self.slots.get_mut(&row.slot) {
                slot.last_sampled = Some(sampled.clone());
            }
            self.prequeue_decode(row.slot, kv)?;
        }

        Ok(inflight)
    }

    /// Whether the outstanding `pending` decode genuinely belongs to `row`'s
    /// LIVE slot and may be returned. Guards against a recycled slot index: an
    /// exact prefix-cache hit can admit a fresh request directly into decode on
    /// the same slot number a finished request left a `pending` on. We require
    /// the same slot, an unchanged epoch (the host has not freed/reallocated the
    /// slot), a still-open one-ahead session, and a matching committed length.
    fn pending_matches_live_slot(&self, row: &infer_plan::DecodeRow, kv: &dyn KvPool) -> bool {
        let Some(pending) = self.pending.as_ref() else {
            return false;
        };
        if pending.slot != row.slot {
            return false;
        }
        let Some(slot) = self.slots.get(&row.slot) else {
            return false;
        };
        slot.session_active
            && slot.slot_epoch == kv.slot_epoch(row.slot)
            && row.kv_seq_len == slot.committed_len
            && slot.cache_len == slot.committed_len + 1
    }

    /// Pipeline fast path: the slot's session is open one step ahead (the step
    /// whose token we are about to return). Drain it to extract the committed
    /// K/V + gdr, then prequeue the following step (leaving the session open
    /// again).
    fn commit_pending_then_prequeue(
        &mut self,
        row: &infer_plan::DecodeRow,
        kv: &mut dyn KvPool,
    ) -> anyhow::Result<()> {
        let model = self.weights.cpp_model()?;
        {
            let slot = self
                .slots
                .get_mut(&row.slot)
                .ok_or_else(|| anyhow::anyhow!("pipeline commit missing slot {}", row.slot))?;
            // The just-completed prequeue advanced `cache_len` one past the
            // committed length; that step is now the committed token.
            debug_assert_eq!(
                row.kv_seq_len, slot.committed_len,
                "pipeline decode committed_len drift on slot {}",
                row.slot
            );
            slot.committed_len = slot.cache_len;
            slot.drain_session(model)?;
            self.active_session_slot = None;
        }
        self.prequeue_decode(row.slot, kv)
    }

    /// Issue (async) the next greedy step on `slot`'s session, feeding the slot's
    /// `last_sampled` deferred token straight into `step_session` (no host token
    /// round-trip), and stash the resulting sampled token as `pending`. The
    /// session is left OPEN one step ahead so the following submit can drain +
    /// publish it. Capacity-bounded: if the slot's reserved K/V is full, the
    /// prequeue is skipped and the next submit falls back to the cold path.
    fn prequeue_decode(&mut self, slot_idx: usize, kv: &mut dyn KvPool) -> anyhow::Result<()> {
        let _ = kv;
        let seed = self
            .slots
            .get(&slot_idx)
            .and_then(|s| s.last_sampled.clone());
        let Some(seed) = seed else {
            return Ok(());
        };
        let model = self.weights.cpp_model()?;
        let token_arr = mlx::reshape(&seed, &[1]);
        let slot = self
            .slots
            .get_mut(&slot_idx)
            .ok_or_else(|| anyhow::anyhow!("prequeue missing slot {slot_idx}"))?;
        // Bound the prequeue to the slot's reserved cache (kv_flat capacity).
        let capacity = slot
            .kv_flat
            .first()
            .map(|a| a.shape().get(2).copied().unwrap_or(0) as usize)
            .unwrap_or(0);
        if capacity != 0 && slot.cache_len + 1 > capacity {
            return Ok(());
        }
        slot.ensure_session_active(model)?;
        self.active_session_slot = Some(slot_idx);
        let logits = model.step_session(&token_arr, slot.cache_len as i32)?;
        mlx::async_eval(&[&logits]);
        slot.cache_len = slot.cache_len.saturating_add(1);
        let next = mlx::argmax(&logits);
        mlx::async_eval(&[&next]);
        slot.last_sampled = Some(next.clone());
        self.pending = Some(PendingStep {
            slot: slot_idx,
            sampled: next,
        });
        Ok(())
    }

    fn ensure_no_other_active_session(&self, slot: usize) -> anyhow::Result<()> {
        if let Some(active) = self.active_session_slot {
            anyhow::ensure!(
                active == slot,
                "scalar Qwen3.5 C++ sessions support only one active slot"
            );
        }
        Ok(())
    }

    fn reset_slot_if_epoch_changed(&mut self, slot: usize, kv: &dyn KvPool) -> anyhow::Result<()> {
        let epoch = kv.slot_epoch(slot);
        let stale = self
            .slots
            .get(&slot)
            .is_some_and(|state| state.slot_epoch != epoch);
        if stale {
            // Host-epoch bump is the slot-release signal until the seam grows an
            // explicit executor release callback.
            if let Some(mut state) = self.slots.remove(&slot)
                && state.session_active
            {
                let model = self.weights.cpp_model()?;
                state.drain_session(model)?;
            }
            if self.active_session_slot == Some(slot) {
                self.active_session_slot = None;
            }
            // The discarded slot's prequeued step is gone.
            if self.pending.as_ref().is_some_and(|p| p.slot == slot) {
                self.pending = None;
            }
        }
        Ok(())
    }
}

#[cfg(feature = "metal")]
#[derive(Default)]
struct MetalPageStore {
    pages: HashMap<u32, MetalPageBlock>,
    prefixes: HashMap<Vec<u32>, MetalPrefixSnapshot>,
}

#[cfg(feature = "metal")]
struct MetalPageBlock {
    kv_flat: Vec<mlx::MlxArray>,
}

#[cfg(feature = "metal")]
struct MetalPrefixSnapshot {
    cache_len: usize,
    gdr_flat: Vec<mlx::MlxArray>,
}

#[cfg(feature = "metal")]
impl MetalPageStore {
    /// Largest leading page count of `block_ids` (in prompt order) for which a
    /// GDR prefix snapshot exists. The host radix caches every page boundary,
    /// but linear-attention (GDR) recurrent/conv state is only snapshotted at
    /// the page boundaries a forward pass landed on; attaching at any other
    /// boundary fails in `materialize_slot_from_prefix`. Engine-core clamps the
    /// offered prefix to this so it never asks for a boundary we can't serve.
    fn reusable_prefix_pages(&self, block_ids: &[u32]) -> usize {
        (1..=block_ids.len())
            .rev()
            .find(|&k| self.prefixes.contains_key(&block_ids[..k]))
            .unwrap_or(0)
    }

    fn release_pages(&mut self, pages: &[u32]) {
        if pages.is_empty() {
            return;
        }
        for page in pages {
            self.pages.remove(page);
        }
        self.prefixes
            .retain(|key, _| !pages.iter().any(|page| key.contains(page)));
    }

    fn publish_slot(&mut self, slot: &MetalSlotState, kv: &dyn KvPool) -> anyhow::Result<()> {
        let page_size = kv.page_size().max(1);
        let full_pages = slot.cache_len / page_size;
        if full_pages == 0 {
            return Ok(());
        }

        let page_ids = kv.page_indices(slot.slot);
        let publish_pages = full_pages.min(page_ids.len());
        for (page_idx, page_id) in page_ids.iter().take(publish_pages).enumerate() {
            let start = page_idx * page_size;
            let end = start + page_size;
            let mut kv_flat = Vec::with_capacity(slot.kv_flat.len());
            for array in &slot.kv_flat {
                kv_flat.push(slice_kv_tokens(array, start, end)?);
            }
            // Host page ids may be reused after the seam frees a slot. Overwrite
            // with the current slot's contents; retained/shared pages cannot be
            // reallocated by the host pool, so this does not corrupt live reuse.
            if self
                .pages
                .insert(*page_id, MetalPageBlock { kv_flat })
                .is_some()
            {
                // Alias hazard: overwriting a page block means this page id was
                // recycled to a new occupant (or republished). Any surviving
                // prefix key containing it would pair the NEW page contents with
                // a STALE GDR snapshot in `materialize_slot_from_prefix` —
                // silently corrupt linear-attention state. Prune every such key
                // except exact prefixes of the live occupant's own page list,
                // which the boundary insert below keeps coherent.
                let overwritten = *page_id;
                self.prefixes.retain(|key, _| {
                    !key.contains(&overwritten)
                        || (key.len() <= page_ids.len() && page_ids[..key.len()] == key[..])
                });
            }
        }

        // GDR state is prefix-wide, not page-local. Only publish a hot-prefix
        // snapshot at an exact page boundary where the exported recurrent/conv
        // state corresponds to the same token length as the page-id prefix.
        if slot.cache_len.is_multiple_of(page_size) && publish_pages == full_pages {
            let key = page_ids[..full_pages].to_vec();
            if key.iter().all(|page| self.pages.contains_key(page)) {
                self.prefixes.insert(
                    key,
                    MetalPrefixSnapshot {
                        cache_len: slot.cache_len,
                        gdr_flat: slot.gdr_flat.clone(),
                    },
                );
            }
        }

        Ok(())
    }

    fn materialize_slot_from_prefix(
        &self,
        slot: usize,
        slot_epoch: u64,
        kv: &dyn KvPool,
        prefix_tokens: usize,
        capacity_tokens: usize,
    ) -> anyhow::Result<MetalSlotState> {
        let page_size = kv.page_size().max(1);
        anyhow::ensure!(
            prefix_tokens.is_multiple_of(page_size),
            "Metal prefix attach requires page-aligned prefix: prefix_tokens={}, page_size={}",
            prefix_tokens,
            page_size
        );
        let prefix_pages = prefix_tokens / page_size;
        let slot_pages = kv.page_indices(slot);
        anyhow::ensure!(
            slot_pages.len() >= prefix_pages,
            "Metal prefix attach for slot {slot} needs {prefix_pages} pages, host slot has {}",
            slot_pages.len()
        );
        let key = slot_pages[..prefix_pages].to_vec();
        let snapshot = self.prefixes.get(&key).ok_or_else(|| {
            anyhow::anyhow!(
                "Metal prefix attach missing GDR snapshot for slot {slot}, prefix_tokens={prefix_tokens}, pages={key:?}"
            )
        })?;
        anyhow::ensure!(
            snapshot.cache_len == prefix_tokens,
            "Metal prefix snapshot length mismatch for slot {slot}: requested={}, snapshot={}",
            prefix_tokens,
            snapshot.cache_len
        );

        let first_page = key
            .first()
            .ok_or_else(|| anyhow::anyhow!("Metal prefix attach got empty page key"))?;
        let first_block = self.pages.get(first_page).ok_or_else(|| {
            anyhow::anyhow!("Metal prefix attach missing K/V page {first_page} for slot {slot}")
        })?;

        let mut kv_flat = Vec::with_capacity(first_block.kv_flat.len());
        let capacity = round_up_capacity(capacity_tokens.max(prefix_tokens)) as usize;
        for array_idx in 0..first_block.kv_flat.len() {
            let mut page_arrays = Vec::with_capacity(key.len());
            for page in &key {
                let block = self.pages.get(page).ok_or_else(|| {
                    anyhow::anyhow!("Metal prefix attach missing K/V page {page} for slot {slot}")
                })?;
                let array = block.kv_flat.get(array_idx).ok_or_else(|| {
                    anyhow::anyhow!(
                        "Metal prefix attach K/V page {page} is missing array index {array_idx}"
                    )
                })?;
                page_arrays.push(array.clone());
            }
            let prefix_array = concatenate_or_single(page_arrays);
            let shape = prefix_array.shape().to_vec();
            anyhow::ensure!(
                shape.len() == 4 && shape[2] as usize == prefix_tokens,
                "Metal prefix K/V materialization shape mismatch for slot {slot}: shape={shape:?}, prefix_tokens={prefix_tokens}"
            );
            if capacity > prefix_tokens {
                let mut zero_shape = shape;
                zero_shape[2] = usize_to_i32(capacity - prefix_tokens)?;
                let zeros = mlx::zeros(&zero_shape, prefix_array.dtype());
                kv_flat.push(mlx::concatenate_axis(&[prefix_array, zeros], 2));
            } else {
                kv_flat.push(prefix_array);
            }
        }

        Ok(MetalSlotState::from_arrays(
            slot,
            slot_epoch,
            prefix_tokens,
            kv_flat,
            snapshot.gdr_flat.clone(),
        ))
    }
}

#[cfg(feature = "metal")]
struct MetalSlotState {
    slot: usize,
    slot_epoch: u64,
    /// Session position: number of tokens whose `step_session` has been issued.
    /// In pipeline mode this runs one ahead of `committed_len` (the prequeued
    /// step). In HEAD mode the two stay equal.
    cache_len: usize,
    /// Tokens the engine has committed for this slot (its `kv_seq_len`). Decode
    /// admission is validated against this, not `cache_len`, so the prequeued
    /// step does not trip the seam's length invariant.
    committed_len: usize,
    kv_flat: Vec<mlx::MlxArray>,
    gdr_flat: Vec<mlx::MlxArray>,
    session_active: bool,
    /// Deferred sampled token (greedy argmax, async-evaluated) from the most
    /// recent step issued on this slot — the input the next prequeue feeds into
    /// `step_session`. `None` outside pipeline mode.
    last_sampled: Option<mlx::MlxArray>,
}

#[cfg(feature = "metal")]
impl MetalSlotState {
    fn new(
        slot: usize,
        slot_epoch: u64,
        config: &config::MetalModelConfig,
        kv_cache_dtype: MetalKvCacheDtype,
        capacity_tokens: usize,
    ) -> Self {
        let capacity = round_up_capacity(capacity_tokens);
        let kv_flat = allocate_kv_flat(config, kv_cache_dtype, capacity);

        let mut gdr_flat = Vec::with_capacity(config.arch.num_linear_attention_layers() * 2);
        for _ in 0..config.arch.num_linear_attention_layers() {
            gdr_flat.push(mlx::zeros(
                &[
                    1,
                    config.arch.linear.num_value_heads as i32,
                    config.arch.linear.value_dim as i32,
                    config.arch.linear.key_dim as i32,
                ],
                mlx::Dtype::Float32,
            ));
            gdr_flat.push(mlx::zeros(
                &[
                    1,
                    (config.arch.linear.conv_kernel - 1) as i32,
                    config.arch.linear.qkv_dim() as i32,
                ],
                mlx::Dtype::Bfloat16,
            ));
        }

        Self {
            slot,
            slot_epoch,
            cache_len: 0,
            committed_len: 0,
            kv_flat,
            gdr_flat,
            session_active: false,
            last_sampled: None,
        }
    }

    fn from_arrays(
        slot: usize,
        slot_epoch: u64,
        cache_len: usize,
        kv_flat: Vec<mlx::MlxArray>,
        gdr_flat: Vec<mlx::MlxArray>,
    ) -> Self {
        Self {
            slot,
            slot_epoch,
            cache_len,
            committed_len: cache_len,
            kv_flat,
            gdr_flat,
            session_active: false,
            last_sampled: None,
        }
    }

    fn ensure_session_active(&mut self, model: &qwen35::CppQwen35Model) -> anyhow::Result<()> {
        if self.session_active {
            return Ok(());
        }
        model.begin_session(&self.kv_flat, &self.gdr_flat)?;
        self.session_active = true;
        Ok(())
    }

    fn drain_session(&mut self, model: &qwen35::CppQwen35Model) -> anyhow::Result<()> {
        if !self.session_active {
            return Ok(());
        }
        let (kv_flat, gdr_flat) = model.end_session(self.kv_flat.len(), self.gdr_flat.len())?;
        self.kv_flat = kv_flat;
        self.gdr_flat = gdr_flat;
        self.session_active = false;
        Ok(())
    }

    /// Guarantee the flat K/V cache can hold `cache_len + needed` tokens, growing
    /// the seq axis with zeros when the prefill reservation is exhausted.
    ///
    /// The C++ session writes each step's K/V with `slice_update`, which returns a
    /// *same-shape* array — so the session's capacity is frozen at `begin_session`
    /// and never grows on its own. The host KV pool already grows page-by-page for
    /// arbitrarily long generations; without this the executor's `kv_flat` lags
    /// behind, `slice_update` silently drops out-of-range writes (corrupt output),
    /// and `publish_slot` eventually hard-errors at a page boundary
    /// (`K/V slice token range [..] exceeds shape=[..]`). The GDR recurrent/conv
    /// state is sequence-independent (see `MetalSlotState::new`) and is left
    /// untouched, exactly as `materialize_slot_from_prefix` treats it. Growing
    /// mutates `kv_flat`, which an open session owns, so the session is drained
    /// first; the caller re-activates it via `ensure_session_active`.
    fn ensure_kv_capacity(
        &mut self,
        model: &qwen35::CppQwen35Model,
        needed: usize,
    ) -> anyhow::Result<()> {
        let capacity = self
            .kv_flat
            .first()
            .map(|array| array.shape().get(2).copied().unwrap_or(0) as usize)
            .unwrap_or(0);
        let required = self.cache_len.saturating_add(needed);
        if capacity == 0 || required <= capacity {
            return Ok(());
        }
        // The open session holds these arrays; drain before reallocating so the
        // grown buffers are the ones the next `begin_session` binds.
        self.drain_session(model)?;
        let new_capacity = round_up_capacity(required.max(capacity.saturating_mul(2))) as usize;
        let mut grown = Vec::with_capacity(self.kv_flat.len());
        for array in &self.kv_flat {
            grown.push(grow_kv_seq_axis(array, new_capacity)?);
        }
        // Materialize before re-binding so the concatenation is not replayed
        // lazily on every subsequent step's forward graph.
        let refs: Vec<&mlx::MlxArray> = grown.iter().collect();
        mlx::eval(&refs);
        self.kv_flat = grown;
        Ok(())
    }
}

/// Extend a rank-4 `[B, n_kv, seq, head_dim]` K/V cache array along the seq axis
/// (index 2) to `new_capacity`, padding the new tail with zeros. The leading
/// tokens are preserved bit-for-bit; the C++ session then writes future tokens
/// into the zero tail via `slice_update`. A no-op (cheap clone) when the array
/// already meets the capacity.
#[cfg(feature = "metal")]
fn grow_kv_seq_axis(array: &mlx::MlxArray, new_capacity: usize) -> anyhow::Result<mlx::MlxArray> {
    let shape = array.shape().to_vec();
    anyhow::ensure!(
        shape.len() == 4,
        "expected rank-4 K/V array to grow, got shape={shape:?}"
    );
    let current = shape[2] as usize;
    if new_capacity <= current {
        return Ok(array.clone());
    }
    let mut tail_shape = shape;
    tail_shape[2] = usize_to_i32(new_capacity - current)?;
    let zeros = mlx::zeros(&tail_shape, array.dtype());
    Ok(mlx::concatenate_axis(&[array.clone(), zeros], 2))
}

#[cfg(feature = "metal")]
fn slice_kv_tokens(
    array: &mlx::MlxArray,
    start_token: usize,
    end_token: usize,
) -> anyhow::Result<mlx::MlxArray> {
    let shape = array.shape().to_vec();
    anyhow::ensure!(
        shape.len() == 4,
        "expected Qwen3.5 flat K/V array to be rank-4, got shape={shape:?}"
    );
    anyhow::ensure!(
        start_token <= end_token && end_token <= shape[2] as usize,
        "K/V slice token range [{start_token}, {end_token}) exceeds shape={shape:?}"
    );
    let start = [0, 0, usize_to_i32(start_token)?, 0];
    let stop = [shape[0], shape[1], usize_to_i32(end_token)?, shape[3]];
    let strides = [1, 1, 1, 1];
    Ok(mlx::slice(array, &start, &stop, &strides))
}

#[cfg(feature = "metal")]
fn concatenate_or_single(mut arrays: Vec<mlx::MlxArray>) -> mlx::MlxArray {
    debug_assert!(!arrays.is_empty());
    if arrays.len() == 1 {
        arrays.pop().expect("len checked")
    } else {
        mlx::concatenate_axis(&arrays, 2)
    }
}

#[cfg(feature = "metal")]
fn usize_to_i32(value: usize) -> anyhow::Result<i32> {
    i32::try_from(value).map_err(|_| anyhow::anyhow!("value {value} exceeds i32::MAX"))
}

#[cfg(feature = "metal")]
fn round_up_capacity(tokens: usize) -> i32 {
    let tokens = tokens.max(1) as i32;
    ((tokens + KV_CACHE_CHUNK - 1) / KV_CACHE_CHUNK) * KV_CACHE_CHUNK
}

#[cfg(feature = "metal")]
fn validate_int8_kv_config(config: &config::MetalModelConfig) -> anyhow::Result<()> {
    let group_size = int8_kv_group_size(config.head_dim)?;
    anyhow::ensure!(
        config.head_dim.is_multiple_of(group_size),
        "Metal int8 KV requires head_dim divisible by group_size: head_dim={}, group_size={group_size}",
        config.head_dim
    );
    Ok(())
}

#[cfg(feature = "metal")]
fn int8_kv_group_size(head_dim: usize) -> anyhow::Result<usize> {
    if head_dim.is_multiple_of(128) {
        Ok(128)
    } else if head_dim.is_multiple_of(64) {
        Ok(64)
    } else if head_dim.is_multiple_of(32) {
        Ok(32)
    } else {
        anyhow::bail!("Metal int8 KV requires head_dim divisible by 32/64/128, got {head_dim}")
    }
}

#[cfg(feature = "metal")]
fn allocate_kv_flat(
    config: &config::MetalModelConfig,
    kv_cache_dtype: MetalKvCacheDtype,
    capacity: i32,
) -> Vec<mlx::MlxArray> {
    let full_layers = config.arch.num_full_attention_layers();
    let nkv = config.num_key_value_heads as i32;
    let hd = config.head_dim as i32;
    match kv_cache_dtype {
        MetalKvCacheDtype::Bf16 => {
            let cache_shape = [1, nkv, capacity, hd];
            let mut kv_flat = Vec::with_capacity(full_layers * 2);
            for _ in 0..full_layers {
                kv_flat.push(mlx::zeros(&cache_shape, mlx::Dtype::Bfloat16));
                kv_flat.push(mlx::zeros(&cache_shape, mlx::Dtype::Bfloat16));
            }
            kv_flat
        }
        MetalKvCacheDtype::Int8 => {
            let group_size = int8_kv_group_size(config.head_dim)
                .expect("validated before slot allocation") as i32;
            let packed_shape = [1, nkv, capacity, hd / 4];
            let scale_shape = [1, nkv, capacity, hd / group_size];
            let mut kv_flat = Vec::with_capacity(full_layers * 6);
            for _ in 0..full_layers {
                // K: packed uint32 data + bf16 scale/bias, then V with the same
                // layout. C++ session code interprets n_kv=6*full_layers as
                // quantized KV.
                kv_flat.push(mlx::zeros(&packed_shape, mlx::Dtype::Uint32));
                kv_flat.push(mlx::zeros(&scale_shape, mlx::Dtype::Bfloat16));
                kv_flat.push(mlx::zeros(&scale_shape, mlx::Dtype::Bfloat16));
                kv_flat.push(mlx::zeros(&packed_shape, mlx::Dtype::Uint32));
                kv_flat.push(mlx::zeros(&scale_shape, mlx::Dtype::Bfloat16));
                kv_flat.push(mlx::zeros(&scale_shape, mlx::Dtype::Bfloat16));
            }
            kv_flat
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_pool::MetalKvPool;
    use infer_plan::{DecodeRow, ForwardMode, PrefillRow};

    // Regression guard for the "K/V slice token range [..] exceeds shape=[..]"
    // crash: a long generation outgrows the prefill reservation, so `kv_flat`
    // must grow along the seq axis while preserving every prior token. Mirrors
    // the exact operation `ensure_kv_capacity` performs on each cache array.
    #[cfg(feature = "metal")]
    #[test]
    fn grow_kv_seq_axis_preserves_tokens_and_zero_pads_tail() {
        let _guard = mlx_sys::mlx_guard();
        // [B=1, n_kv=1, seq=2, head_dim=2] with distinct, known values.
        let src = mlx::MlxArray::from_slice_i32(&[10, 11, 20, 21], &[1, 1, 2, 2]);
        let src = mlx::as_dtype(&src, mlx::Dtype::Float32);
        let grown = grow_kv_seq_axis(&src, 4).unwrap();
        mlx::eval(&[&grown]);
        assert_eq!(grown.shape(), &[1, 1, 4, 2], "seq axis must extend to 4");
        let vals = grown.as_slice_f32();
        // Tokens 0,1 preserved bit-for-bit.
        assert_eq!(&vals[0..4], &[10.0, 11.0, 20.0, 21.0]);
        // Tokens 2,3 (the grown tail) are zero — the slice_update write target.
        assert_eq!(&vals[4..8], &[0.0, 0.0, 0.0, 0.0]);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn grow_kv_seq_axis_is_noop_when_capacity_met() {
        let _guard = mlx_sys::mlx_guard();
        let src = mlx::MlxArray::from_slice_i32(&[1, 2, 3, 4], &[1, 1, 2, 2]);
        let src = mlx::as_dtype(&src, mlx::Dtype::Float32);
        let same = grow_kv_seq_axis(&src, 2).unwrap();
        assert_eq!(same.shape(), &[1, 1, 2, 2]);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn int8_kv_layout_allocates_packed_triples_per_kv_axis() {
        let _guard = mlx_sys::mlx_guard();
        let config = config::MetalModelConfig {
            hidden_size: 16,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            num_hidden_layers: 1,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            head_dim: 128,
            stop_token_ids: vec![0],
            quantization: None,
            arch: config::MetalQwen35ArchConfig {
                layer_types: vec![config::MetalQwen35LayerType::FullAttention],
                rotary_dim: 128,
                linear: config::MetalGdrConfig {
                    num_key_heads: 0,
                    key_dim: 0,
                    num_value_heads: 0,
                    value_dim: 0,
                    conv_kernel: 4,
                    rms_norm_eps: 1e-6,
                },
                moe: None,
            },
        };
        let arrays = allocate_kv_flat(&config, MetalKvCacheDtype::Int8, 256);
        assert_eq!(arrays.len(), 6);
        assert_eq!(arrays[0].shape(), &[1, 1, 256, 32]);
        assert_eq!(arrays[0].dtype(), mlx::Dtype::Uint32);
        assert_eq!(arrays[1].shape(), &[1, 1, 256, 1]);
        assert_eq!(arrays[1].dtype(), mlx::Dtype::Bfloat16);
        assert_eq!(arrays[2].shape(), &[1, 1, 256, 1]);
        assert_eq!(arrays[3].shape(), &[1, 1, 256, 32]);
        assert_eq!(arrays[3].dtype(), mlx::Dtype::Uint32);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn int8_kv_group_size_prefers_largest_supported_divisor() {
        assert_eq!(int8_kv_group_size(256).unwrap(), 128);
        assert_eq!(int8_kv_group_size(96).unwrap(), 32);
        assert!(int8_kv_group_size(80).is_err());
    }

    /// Rank-4 `[1, 1, seq, 2]` f32 K/V array filled with `fill` — the minimal
    /// shape `slice_kv_tokens` accepts, no model load needed.
    #[cfg(feature = "metal")]
    fn kv_array(seq: usize, fill: i32) -> mlx::MlxArray {
        let vals = vec![fill; seq * 2];
        let arr = mlx::MlxArray::from_slice_i32(&vals, &[1, 1, seq as i32, 2]);
        mlx::as_dtype(&arr, mlx::Dtype::Float32)
    }

    /// Tiny stand-in GDR state array carrying a distinguishable `fill` value so a
    /// test can tell WHICH occupant's snapshot survives in `prefixes`.
    #[cfg(feature = "metal")]
    fn gdr_array(fill: i32) -> mlx::MlxArray {
        let arr = mlx::MlxArray::from_slice_i32(&[fill], &[1]);
        mlx::as_dtype(&arr, mlx::Dtype::Float32)
    }

    // Defect-2 regression guard (stale prefix snapshot aliasing): host page ids
    // are recycled LIFO after radix eviction, but `prefixes` keys used to live
    // forever. A later radix match colliding with a stale key would serve the
    // NEW occupant's K/V pages with the OLD occupant's GDR snapshot. Publishing
    // a second occupant under the same recycled page ids must prune the first
    // occupant's prefix key.
    #[cfg(feature = "metal")]
    #[test]
    fn page_reuse_prunes_stale_prefix_snapshot() {
        use infer_seam::{KvAllocator, KvQuery};
        let _guard = mlx_sys::mlx_guard();
        let mut store = MetalPageStore::default();
        let mut pool = MetalKvPool::new(2, 8, 4);

        // First occupant: slot 0, 8 tokens = 2 full pages, exact page boundary
        // -> publishes both page blocks and a GDR prefix snapshot.
        pool.alloc(0, 8).unwrap();
        let first_pages: Vec<u32> = pool.page_indices(0).to_vec();
        let state_a = MetalSlotState::from_arrays(
            0,
            pool.slot_epoch(0),
            8,
            vec![kv_array(8, 10)],
            vec![gdr_array(1)],
        );
        store.publish_slot(&state_a, &pool).unwrap();
        assert_eq!(store.reusable_prefix_pages(&first_pages), 2);

        // Free slot 0 and allocate slot 1: the LIFO free list recycles the SAME
        // physical page ids (in reversed order) to the new occupant.
        pool.free_slot(0);
        pool.alloc(1, 8).unwrap();
        let second_pages: Vec<u32> = pool.page_indices(1).to_vec();
        let sorted = |mut v: Vec<u32>| {
            v.sort_unstable();
            v
        };
        assert_eq!(
            sorted(first_pages.clone()),
            sorted(second_pages.clone()),
            "test premise: the pool must recycle the freed page ids"
        );
        assert_ne!(
            first_pages, second_pages,
            "test premise: the recycled order must differ so the stale key is not \
             the new occupant's own prefix"
        );

        let state_b = MetalSlotState::from_arrays(
            1,
            pool.slot_epoch(1),
            8,
            vec![kv_array(8, 20)],
            vec![gdr_array(2)],
        );
        store.publish_slot(&state_b, &pool).unwrap();

        // The first occupant's prefix key is pruned: it contains overwritten
        // page ids and is not a prefix of the new occupant's page list.
        assert!(
            !store.prefixes.contains_key(&first_pages),
            "stale prefix key {first_pages:?} must be pruned on page reuse"
        );
        assert_eq!(store.reusable_prefix_pages(&first_pages), 0);

        // The new occupant's own boundary snapshot survives and carries ITS GDR
        // state, not the first occupant's.
        assert_eq!(store.reusable_prefix_pages(&second_pages), 2);
        let snap = store
            .prefixes
            .get(&second_pages)
            .expect("new occupant's boundary snapshot must survive");
        assert_eq!(snap.cache_len, 8);
        mlx::eval(&[&snap.gdr_flat[0]]);
        assert_eq!(snap.gdr_flat[0].as_slice_f32(), &[2.0]);
    }

    // A slot republishing its own pages (e.g. the next prefill chunk) overwrites
    // its earlier page blocks, but its earlier boundary snapshots are exact
    // prefixes of the live page list and must NOT be pruned.
    #[cfg(feature = "metal")]
    #[test]
    fn republish_same_slot_keeps_own_prefix_snapshots() {
        use infer_seam::{KvAllocator, KvQuery};
        let _guard = mlx_sys::mlx_guard();
        let mut store = MetalPageStore::default();
        let mut pool = MetalKvPool::new(1, 8, 4);

        // First chunk: 4 tokens = 1 full page -> snapshot at [p0].
        pool.alloc(0, 4).unwrap();
        let one_page: Vec<u32> = pool.page_indices(0).to_vec();
        let state = MetalSlotState::from_arrays(
            0,
            pool.slot_epoch(0),
            4,
            vec![kv_array(4, 10)],
            vec![gdr_array(1)],
        );
        store.publish_slot(&state, &pool).unwrap();
        assert_eq!(store.reusable_prefix_pages(&one_page), 1);

        // Second chunk: 8 tokens = 2 pages. Page p0's block is overwritten
        // (insert returns Some) but [p0] is an exact prefix of the live
        // occupant's page list, so its snapshot survives.
        pool.alloc(0, 4).unwrap();
        let two_pages: Vec<u32> = pool.page_indices(0).to_vec();
        let state = MetalSlotState::from_arrays(
            0,
            pool.slot_epoch(0),
            8,
            vec![kv_array(8, 10)],
            vec![gdr_array(1)],
        );
        store.publish_slot(&state, &pool).unwrap();

        assert!(store.prefixes.contains_key(&one_page));
        assert!(store.prefixes.contains_key(&two_pages));
        assert_eq!(store.reusable_prefix_pages(&one_page), 1);
        assert_eq!(store.reusable_prefix_pages(&two_pages), 2);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn release_pages_drops_mirrors_and_prefix_snapshots() {
        use infer_seam::{KvAllocator, KvQuery};
        let _guard = mlx_sys::mlx_guard();
        let mut store = MetalPageStore::default();
        let mut pool = MetalKvPool::new(1, 8, 4);

        pool.alloc(0, 8).unwrap();
        let pages: Vec<u32> = pool.page_indices(0).to_vec();
        let state = MetalSlotState::from_arrays(
            0,
            pool.slot_epoch(0),
            8,
            vec![kv_array(8, 10)],
            vec![gdr_array(1)],
        );
        store.publish_slot(&state, &pool).unwrap();
        assert_eq!(store.pages.len(), 2);
        assert_eq!(store.reusable_prefix_pages(&pages), 2);

        store.release_pages(&[pages[0]]);

        assert!(
            !store.pages.contains_key(&pages[0]),
            "evicted page mirror must be dropped"
        );
        assert!(
            !store.prefixes.contains_key(&pages),
            "prefix snapshot containing the evicted page must be pruned"
        );
        assert_eq!(store.reusable_prefix_pages(&pages), 0);
    }

    #[test]
    fn executor_decode_plumbing_returns_one_token_per_row() {
        let mut exec = MetalExecutor::new();
        let mut pool = MetalKvPool::new(2, 8, 16);
        let plan = ForwardPlan {
            mode: ForwardMode::Decode,
            decode_rows: vec![
                DecodeRow {
                    slot: 0,
                    last_token: 10,
                    kv_seq_len: 4,
                    params: infer_plan::SamplingParams::default(),
                },
                DecodeRow {
                    slot: 1,
                    last_token: 20,
                    kv_seq_len: 7,
                    params: infer_plan::SamplingParams::default(),
                },
            ],
            prefill_rows: Vec::new(),
            microbatch: None,
            spec: None,
        };
        let inflight = exec.submit(&plan, &mut pool).unwrap();
        match exec.poll(inflight).unwrap() {
            PollResult::Ready(out) => {
                assert_eq!(out.tokens.len(), 2);
                assert_eq!(out.tokens[0].token, 11);
                assert_eq!(out.tokens[1].token, 21);
            }
            PollResult::NotReady(_) => panic!("skeleton resolves synchronously"),
        }
    }

    #[test]
    fn executor_prefill_plumbing_returns_completion_token() {
        let mut exec = MetalExecutor::new();
        let mut pool = MetalKvPool::new(1, 8, 16);
        let plan = ForwardPlan {
            mode: ForwardMode::Prefill,
            decode_rows: Vec::new(),
            prefill_rows: vec![PrefillRow {
                slot: 0,
                tokens: vec![1, 2, 3],
                start_pos: 0,
                total_tokens: 3,
                params: infer_plan::SamplingParams::default(),
            }],
            microbatch: None,
            spec: None,
        };
        let inflight = exec.submit(&plan, &mut pool).unwrap();
        match exec.poll(inflight).unwrap() {
            PollResult::Ready(out) => {
                assert_eq!(out.tokens.len(), 1);
                assert_eq!(out.tokens[0].slot, 0);
                assert_eq!(out.tokens[0].token, 4); // last prompt token (3) + 1
            }
            PollResult::NotReady(_) => panic!("skeleton resolves synchronously"),
        }
    }
}
