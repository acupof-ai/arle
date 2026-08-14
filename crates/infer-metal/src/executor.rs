//! Metal backend executor + session machinery.
//!
//! `new()` keeps a CPU placeholder so the submit/poll seam stays testable without
//! the `metal` feature; `from_model_path()` builds the real MLX Qwen3.5 executor.
//! `RealMetalExecutor` and all MLX-touching session state are gated behind
//! `#[cfg(feature = "metal")]`.

#[cfg(feature = "metal")]
use std::collections::{BTreeSet, HashMap};
#[cfg(feature = "metal")]
use std::path::{Path, PathBuf};

use infer_plan::{ForwardPlan, SlotToken, StepOutput};
use infer_seam::{BackendExecutor, KvPool, PollResult, PrefixBlock};

#[cfg(feature = "metal")]
use crate::{config, dflash, mlx, model_source, qwen35};

#[cfg(feature = "metal")]
const KV_CACHE_CHUNK: i32 = 256;

/// Machine-derived **L3 (NVMe)** spill budget for the Metal disk tier
/// (unified with the CUDA policy — same probe, same clamp).
#[cfg(feature = "metal")]
pub use kv_native_sys::default_t2_budget_bytes;

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

    /// Resolve a backend-neutral requested dtype against the Metal support
    /// matrix. `Auto` resolves to INT8 (the Metal default after the int8 gate);
    /// `Fp8`/`Tq4` are CUDA-only paged-KV quant modes and fail loud here rather
    /// than silently downgrading.
    pub fn resolve(requested: infer_seam::KvCacheDtype) -> anyhow::Result<Self> {
        use infer_seam::KvCacheDtype;
        match requested {
            KvCacheDtype::Auto | KvCacheDtype::Int8 => Ok(Self::Int8),
            KvCacheDtype::Bf16 => Ok(Self::Bf16),
            other => anyhow::bail!(
                "Metal KV cache supports bf16/int8; requested {} is CUDA-only",
                other.label()
            ),
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
/// path. Opt OUT with `--metal-pipeline false`.
#[cfg(feature = "metal")]
fn pipeline_decode_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let on = crate::runtime_flags::pipeline();
        eprintln!("[infer-metal] decode pipeline (--metal-pipeline) = {on}");
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

#[cfg(feature = "metal")]
#[must_use]
pub fn pipeline_fast_path_hits() -> u64 {
    PIPELINE_FAST_PATH_HITS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Paged-prefix read path for single-token decode. The C++ session still owns
/// K/V writes; only SDPA's prefix read source changes. Default on after BF16
/// and INT8 reachability gates; opt out with `--metal-paged-kv-read false`.
#[cfg(feature = "metal")]
fn paged_kv_read_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let on = crate::runtime_flags::paged_kv_read();
        eprintln!("[infer-metal] paged KV read (--metal-paged-kv-read) = {on}");
        on
    })
}

#[cfg(feature = "metal")]
pub(crate) static PAGED_KV_READ_HITS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

#[cfg(feature = "metal")]
pub(crate) static PAGED_KV_READ_FALLBACKS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

#[cfg(feature = "metal")]
fn probe_paged_kv_read_hit() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| eprintln!("[infer-metal] paged KV read LIVE (single-token decode)"));
    PAGED_KV_READ_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(feature = "metal")]
fn probe_paged_kv_read_fallback() {
    PAGED_KV_READ_FALLBACKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(feature = "metal")]
#[must_use]
pub fn paged_kv_read_hits() -> u64 {
    PAGED_KV_READ_HITS.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(feature = "metal")]
#[must_use]
pub fn paged_kv_read_fallbacks() -> u64 {
    PAGED_KV_READ_FALLBACKS.load(std::sync::atomic::Ordering::Relaxed)
}

pub enum MetalInflight {
    Ready(StepOutput),
    #[cfg(feature = "metal")]
    Sampled {
        slot: usize,
        sampled: mlx::MlxArray,
    },
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
/// The raw-argmax fast path keeps the device `argmax` + async path. Non-greedy
/// Metal sampling used to materialize host f32 logits and sample on CPU every
/// token, which creates synchronous D2H stalls on the local desktop path;
/// the temperature path is therefore downgraded to device greedy unless
/// `--metal-host-sampling` opts into the blocking sampler. A greedy request
/// whose logits are rewritten first (penalties, grammar, logit_bias) has no
/// device equivalent, so it always pays the D2H and reaches the host sampler.
#[cfg(feature = "metal")]
fn sample_inflight(
    slot: usize,
    logits: &mlx::MlxArray,
    params: &infer_plan::SamplingParams,
    history: Option<infer_plan::PenaltyHistory<'_>>,
    position: u64,
) -> MetalInflight {
    let downgrade = !params.is_greedy() && !crate::runtime_flags::host_sampling();
    if params.is_raw_argmax() || downgrade {
        if downgrade {
            warn_host_sampling_downgrade();
        }
        let sampled = mlx::argmax(logits);
        mlx::async_eval(&[&sampled]);
        return MetalInflight::Sampled { slot, sampled };
    }
    let logits_f32 = mlx::as_dtype(logits, mlx::Dtype::Float32);
    mlx::eval(&[&logits_f32]);
    let logits_f32 = logits_f32.as_slice_f32();
    let token = match history {
        Some(history) => infer_plan::sample_token_penalized(logits_f32, params, position, history),
        None => infer_plan::sample_token(logits_f32, params, position),
    };
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
fn materialize_inflight_now(inflight: MetalInflight) -> anyhow::Result<StepOutput> {
    match inflight {
        MetalInflight::Ready(output) => Ok(output),
        MetalInflight::Sampled { slot, sampled } => {
            mlx::eval(&[&sampled]);
            Ok(StepOutput {
                tokens: vec![SlotToken {
                    slot,
                    token: sampled.item_i32() as u32,
                    logprob: None,
                    finish: None,
                }],
            })
        }
    }
}

#[cfg(feature = "metal")]
#[cfg(feature = "metal")]
fn warn_host_sampling_downgrade() {
    use std::sync::atomic::{AtomicBool, Ordering};
    static WARNED: AtomicBool = AtomicBool::new(false);
    if !WARNED.swap(true, Ordering::Relaxed) {
        log::warn!(
            "Metal non-greedy sampling requested, but host logits sampling is disabled; \
             using device greedy argmax. Set --metal-host-sampling to opt into \
             the blocking D2H sampler."
        );
    }
}

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
    #[must_use]
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "metal")]
            real: None,
        }
    }

    #[cfg(feature = "metal")]
    pub fn from_model_path(model_path: impl AsRef<Path>) -> anyhow::Result<Self> {
        Self::from_model_path_with_kv_cache_dtype(model_path, MetalKvCacheDtype::default())
    }

    #[cfg(feature = "metal")]
    pub fn from_model_path_with_kv_cache_dtype(
        model_path: impl AsRef<Path>,
        kv_cache_dtype: MetalKvCacheDtype,
    ) -> anyhow::Result<Self> {
        let model_source = model_path.as_ref().to_string_lossy();
        let resolved = model_source::resolve_model_path(&model_source)?;
        let resource_plan = crate::resource::plan_resource_budget(
            &resolved,
            crate::resource::MetalResourceRequest {
                kv_cache_dtype,
                num_slots: 1,
                total_pages: 8192,
                page_size: 16,
                low_impact: true,
                memory_budget_bytes: None,
                system_reserve_bytes: None,
                allow_swap: false,
                mem_fraction_static: 0.9,
            },
        )?;
        Self::from_resolved_model_path_with_plan(&resolved, kv_cache_dtype, Some(resource_plan))
    }

    #[cfg(feature = "metal")]
    pub fn from_model_path_with_kv_cache_dtype_and_resource_plan(
        model_path: impl AsRef<Path>,
        kv_cache_dtype: MetalKvCacheDtype,
        resource_plan: crate::resource::MetalResourcePlan,
    ) -> anyhow::Result<Self> {
        let model_source = model_path.as_ref().to_string_lossy();
        let resolved = model_source::resolve_model_path(&model_source)?;
        Self::from_resolved_model_path_with_plan(&resolved, kv_cache_dtype, Some(resource_plan))
    }

    #[cfg(feature = "metal")]
    fn from_resolved_model_path_with_plan(
        resolved: &Path,
        kv_cache_dtype: MetalKvCacheDtype,
        resource_plan: Option<crate::resource::MetalResourcePlan>,
    ) -> anyhow::Result<Self> {
        let _guard = mlx_sys::mlx_guard();
        crate::resource::apply_startup_mlx_limits(resolved, resource_plan.as_ref(), None, true);
        let config = config::load_metal_config(resolved)?;
        if kv_cache_dtype == MetalKvCacheDtype::Int8 {
            validate_int8_kv_config(&config)?;
        }
        eprintln!("[infer-metal] kv cache dtype = {}", kv_cache_dtype.label());
        let weights = qwen35::load_qwen35_metal_weights(resolved, &config)?;
        let dflash = resolve_dflash(resolved, &config)?;
        Ok(Self {
            real: Some(RealMetalExecutor {
                config,
                kv_cache_dtype,
                weights,
                slots: HashMap::new(),
                page_store: MetalPageStore::default(),
                active_session_slot: None,
                pending: None,
                dflash,
                kv_recall: false,
                recall_cfg: default_recall_config(),
                recall_int8_warned: false,
            }),
        })
    }

    /// Opt into session KV-recall (the `--kv-recall` flag, default off). Mirrors
    /// `set_kv_tier_disk`: a post-construction setter so the executor builder
    /// signatures stay stable. With recall off the decode hot path is unchanged.
    #[cfg(feature = "metal")]
    pub fn set_kv_recall(&mut self, enabled: bool) {
        if let Some(real) = self.real.as_mut() {
            real.kv_recall = enabled;
        }
    }

    #[cfg(feature = "metal")]
    pub fn set_kv_tier_disk(
        &mut self,
        root: PathBuf,
        budget_bytes: usize,
        page_size: usize,
    ) -> bool {
        let Some(real) = self.real.as_mut() else {
            return false;
        };
        let bytes_per_page =
            estimated_metal_kv_page_bytes(&real.config, real.kv_cache_dtype, page_size.max(1));
        real.page_store.set_ssd(root, budget_bytes, bytes_per_page)
    }

    /// Feature-free placeholder forward: one deterministic token per scheduled
    /// row, so the submit/poll seam is exercisable on CPU without MLX.
    fn placeholder_forward(plan: &ForwardPlan) -> StepOutput {
        let tokens = plan
            .decode_rows
            .iter()
            .map(|row| SlotToken {
                slot: row.slot,
                token: row.last_token.wrapping_add(1),
                logprob: None,
                finish: None,
            })
            .chain(plan.prefill_rows.iter().map(|row| SlotToken {
                slot: row.slot,
                token: row.tokens.last().copied().unwrap_or(0).wrapping_add(1),
                logprob: None,
                finish: None,
            }))
            .collect();
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

    fn step_limits(&self) -> infer_seam::StepLimits {
        #[cfg(feature = "metal")]
        if let Some(real) = self.real.as_ref() {
            return infer_seam::StepLimits {
                max_rows_per_step: real.max_rows_per_step(),
                max_live_requests: real.max_live_requests(),
                ..infer_seam::StepLimits::default()
            };
        }
        infer_seam::StepLimits {
            max_rows_per_step: 1,
            max_live_requests: 1,
            ..infer_seam::StepLimits::default()
        }
    }

    fn prefix_reuse(&mut self) -> Option<&mut dyn infer_seam::PrefixReuse> {
        Some(self)
    }

    fn kv_page_tier(&mut self) -> Option<&mut dyn infer_seam::KvPageTier> {
        Some(self)
    }

    fn warmup(&mut self) -> anyhow::Result<()> {
        #[cfg(feature = "metal")]
        if let Some(real) = self.real.as_mut() {
            return real.warmup();
        }
        Ok(())
    }
}

impl infer_seam::PrefixReuse for MetalExecutor {
    fn reusable_prefix_blocks(&self, blocks: &[PrefixBlock]) -> usize {
        #[cfg(feature = "metal")]
        if let Some(real) = self.real.as_ref() {
            if real.dflash.is_some() {
                // Prefix reuse is a restore-boundary promise. DFlash needs
                // target-hidden features and draft KV in addition to target
                // K/V plus recurrent state; the current snapshot publishes only
                // the target restore image, so no DFlash boundary is complete.
                return 0;
            }
            return real.page_store.reusable_prefix_blocks(blocks);
        }
        blocks.len()
    }

    fn reusable_prefix_blocks_for_prompt(&self, blocks: &[PrefixBlock], _tokens: &[u32]) -> usize {
        // No content-verified tail on Metal: defer to the strict block count.
        self.reusable_prefix_blocks(blocks)
    }

    fn release_prefix_pages(&mut self, _pages: &[u32]) {
        #[cfg(feature = "metal")]
        if let Some(real) = self.real.as_mut() {
            real.page_store.release_pages(_pages);
        }
    }

    fn release_provisional_prefix_pages(&mut self, _pages: &[u32]) {}

    fn cached_prefix_match_len(&self, _tokens: &[u32]) -> anyhow::Result<usize> {
        Ok(0)
    }

    fn restore_cached_prefix(
        &mut self,
        _slot: usize,
        _tokens: &[u32],
        _matched_len: usize,
        _slot_pages: &[u32],
    ) -> anyhow::Result<()> {
        anyhow::bail!("Metal backend has no position-0 prefix store")
    }

    fn restore_prefix_sidecar(
        &mut self,
        _slot: usize,
        _tokens: &[u32],
        matched_len: usize,
        _prefix_pages: &[u32],
    ) -> anyhow::Result<usize> {
        // Full-attention restore: pages carry the whole state.
        Ok(matched_len)
    }

    fn capture_finish_frontier(
        &mut self,
        _slot: usize,
        _tokens: &[u32],
        _slot_pages: &[u32],
    ) -> anyhow::Result<()> {
        Ok(())
    }

    fn save_prefix_sidecar(
        &mut self,
        _slot: usize,
        _tokens: &[u32],
        _matched_len: usize,
        _prefix_pages: &[u32],
        _slot_pages: &[u32],
        _newly_cached: &[u32],
    ) -> anyhow::Result<()> {
        Ok(())
    }
}

impl infer_seam::KvPageTier for MetalExecutor {
    fn kv_tier_capacity_pages(&self) -> usize {
        #[cfg(feature = "metal")]
        if let Some(real) = self.real.as_ref() {
            return real.page_store.kv_tier_capacity_pages();
        }
        0
    }

    fn kv_tier_page_bytes(&self) -> usize {
        #[cfg(feature = "metal")]
        if let Some(real) = self.real.as_ref() {
            return real.page_store.kv_tier_page_bytes();
        }
        0
    }

    fn kv_tier_host_demoted_pages(&self) -> usize {
        // Metal's tier store is disk-backed only; nothing sits host-demoted.
        0
    }

    fn kv_tier_read_hits(&self) -> infer_seam::KvTierReadHits {
        infer_seam::KvTierReadHits::default()
    }

    fn kv_tier_transfer_is_zero_copy(&self) -> bool {
        // KvTierStore disk reads are Cow::Borrowed mmap slices; the decode into
        // MLX arrays is the payload materialization, not a staging copy.
        true
    }

    fn kv_tier_disk_pages(&self) -> usize {
        #[cfg(feature = "metal")]
        if let Some(real) = self.real.as_ref() {
            return real.page_store.kv_tier_disk_pages();
        }
        0
    }

    fn kv_tier_io_stats(&self) -> infer_seam::KvTierIoStats {
        #[cfg(feature = "metal")]
        if let Some(real) = self.real.as_ref() {
            return real.page_store.kv_tier_io_stats();
        }
        infer_seam::KvTierIoStats::default()
    }

    fn kv_tier_location(&self, key: u64) -> Option<infer_seam::KvTierLocation> {
        #[cfg(feature = "metal")]
        if let Some(real) = self.real.as_ref() {
            return real.page_store.kv_tier_location(key);
        }
        let _ = key;
        None
    }

    fn demote_prefix_pages(&mut self, entries: &[(u32, u64)]) -> anyhow::Result<usize> {
        #[cfg(feature = "metal")]
        if let Some(real) = self.real.as_mut() {
            return real.page_store.demote_prefix_pages(entries);
        }
        let _ = entries;
        Ok(0)
    }

    fn promote_prefix_pages(&mut self, entries: &[(u64, u32)]) -> anyhow::Result<()> {
        #[cfg(feature = "metal")]
        if let Some(real) = self.real.as_mut() {
            return real.page_store.promote_prefix_pages(entries);
        }
        let _ = entries;
        anyhow::bail!("Metal placeholder backend has no KV tier store")
    }

    fn drop_kv_tier_entries(&mut self, keys: &[u64]) {
        #[cfg(feature = "metal")]
        if let Some(real) = self.real.as_mut() {
            real.page_store.drop_kv_tier_entries(keys);
        }
        #[cfg(not(feature = "metal"))]
        let _ = keys;
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

/// Validated session KV-recall budget (per the offline Qwen3.6 quality gate +
/// `wins/2026-06-23-kv-recall-arle-core-e2e.md`): sink 32, local 256, block 32,
/// top-k 8 → working set 32 + 256 + 8·32 = 544 tokens (9.6% KV in the e2e).
/// Recall only restricts attention once `cache_len` exceeds this budget; below
/// it `plan_recall` returns the full contiguous range (no-op).
#[cfg(feature = "metal")]
fn default_recall_config() -> infer_core::RecallConfig {
    infer_core::RecallConfig {
        n_init: 32,
        n_local: 256,
        l_bs: 32,
        top_k: 8,
    }
}

#[cfg(feature = "metal")]
struct RealMetalExecutor {
    config: config::MetalModelConfig,
    kv_cache_dtype: MetalKvCacheDtype,
    weights: qwen35::Qwen35MetalWeights,
    slots: HashMap<usize, MetalSlotState>,
    page_store: MetalPageStore,
    active_session_slot: Option<usize>,
    pending: Option<PendingStep>,
    /// Opt-in single-request DFlash side path. When present, decode must route
    /// through DFlash or fail; it must never silently fall back to target-only.
    dflash: Option<dflash::MetalDflashRuntime>,
    /// Session KV-recall opt-in (`--kv-recall`). When off (default) the decode
    /// hot path does no scoring and `recall_ranges` stays `None` → baseline
    /// byte-identical. Recall is bf16-only; int8 KV falls back to full attention
    /// (logged once).
    kv_recall: bool,
    /// Recall budget regions (validated defaults). Carved into sink + recalled
    /// top-k + local; the working set is bounded regardless of history length.
    recall_cfg: infer_core::RecallConfig,
    recall_int8_warned: bool,
}

#[cfg(feature = "metal")]
impl RealMetalExecutor {
    fn max_rows_per_step(&self) -> usize {
        self.dflash
            .as_ref()
            .map_or(1, |runtime| runtime.max_rows().max(1))
    }

    fn max_live_requests(&self) -> usize {
        self.max_rows_per_step()
    }

    /// Pre-build (JIT-compile) the prefill + decode MLX graphs at load so turn-0
    /// is not cold. After the steady-decode pipeline recovery
    /// (`wins/2026-06-04-metal-rewrite-decode-pipeline-recovery`), the residual
    /// turn-wall gap is turn-0's lazy graph build + first MoE encode landing on
    /// the first real request. A tiny throwaway forward on a reserved warmup slot
    /// (never published to the kv pool) pre-pays that JIT at load instead. Opt
    /// out with `--metal-warmup false`.
    fn warmup(&mut self) -> anyhow::Result<()> {
        let on = crate::runtime_flags::warmup();
        eprintln!("[infer-metal] warmup (--metal-warmup) = {on}");
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
        let prefill = mlx::MlxArray::from_slice_i32(&[0, 0], &[2]);
        let logits = model.prefill_session(&prefill, 2, 0)?;
        mlx::async_eval(&[&logits]);
        state.cache_len = 2;
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
        anyhow::ensure!(row_count > 0, "R3a MetalExecutor received an idle plan");
        if !plan.prefill_rows.is_empty() && !plan.decode_rows.is_empty() {
            if self.dflash.is_some() {
                return self.submit_dflash_mixed_rows(&plan.prefill_rows, &plan.decode_rows, kv);
            }
            anyhow::bail!("R3a MetalExecutor does not support mixed prefill/decode plans");
        }

        if !plan.prefill_rows.is_empty() {
            if self.dflash.is_some() && plan.prefill_rows.len() > 1 {
                return self.submit_dflash_prefill_rows(&plan.prefill_rows, kv);
            }
            anyhow::ensure!(
                plan.prefill_rows.len() == 1,
                "R3a MetalExecutor supports exactly one prefill row, got {}",
                plan.prefill_rows.len()
            );
            let row = &plan.prefill_rows[0];
            return self.submit_prefill(row, kv);
        }

        if !plan.decode_rows.is_empty() {
            if self.dflash.is_some() {
                return self.submit_dflash_decode_rows(&plan.decode_rows, kv);
            }
            anyhow::ensure!(
                plan.decode_rows.len() == 1,
                "R3a MetalExecutor supports exactly one target-only decode row, got {}",
                plan.decode_rows.len()
            );
            let row = &plan.decode_rows[0];
            return self.submit_decode(row, kv);
        }

        anyhow::bail!("R3a MetalExecutor received a non-idle plan with no rows")
    }

    fn submit_dflash_prefill_rows(
        &mut self,
        rows: &[infer_plan::PrefillRow],
        kv: &mut dyn KvPool,
    ) -> anyhow::Result<MetalInflight> {
        self.preflight_dflash_prefill_rows(rows)?;
        Ok(MetalInflight::Ready(
            self.run_dflash_prefill_rows(rows, kv)?,
        ))
    }

    fn submit_dflash_mixed_rows(
        &mut self,
        prefill_rows: &[infer_plan::PrefillRow],
        decode_rows: &[infer_plan::DecodeRow],
        kv: &mut dyn KvPool,
    ) -> anyhow::Result<MetalInflight> {
        let runtime = self
            .dflash
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("DFlash mixed plan requested without a runtime"))?;
        let row_count = prefill_rows.len() + decode_rows.len();
        anyhow::ensure!(
            row_count <= runtime.max_rows(),
            "DFlash mixed plan received {row_count} rows but INFER_METAL_DFLASH_MAX_ROWS allows {}",
            runtime.max_rows()
        );
        self.preflight_dflash_prefill_rows(prefill_rows)?;
        self.preflight_dflash_decode_rows(decode_rows, kv)?;

        log::info!(
            "Metal DFlash scheduler-mixed lane live: prefill_rows={}, decode_rows={}",
            prefill_rows.len(),
            decode_rows.len()
        );
        // Decode first: minimise TTFT/ITL for active decode requests before
        // running the more expensive prefill sub-steps.
        let mut tokens = self.run_dflash_decode_rows(decode_rows)?.tokens;
        tokens.extend(self.run_dflash_prefill_rows(prefill_rows, kv)?.tokens);
        Ok(MetalInflight::Ready(StepOutput { tokens }))
    }

    fn preflight_dflash_prefill_rows(&self, rows: &[infer_plan::PrefillRow]) -> anyhow::Result<()> {
        let runtime = self
            .dflash
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("DFlash prefill requested without a runtime"))?;
        anyhow::ensure!(
            rows.len() <= runtime.max_rows(),
            "DFlash prefill received {} rows but INFER_METAL_DFLASH_MAX_ROWS allows {}",
            rows.len(),
            runtime.max_rows()
        );
        anyhow::ensure!(
            self.active_session_slot.is_none(),
            "DFlash batched prefill requires no active scalar session"
        );
        let mut seen = BTreeSet::new();
        for row in rows {
            anyhow::ensure!(
                seen.insert(row.slot),
                "DFlash prefill received duplicate slot {} in one scheduler step",
                row.slot
            );
            anyhow::ensure!(
                !row.tokens.is_empty(),
                "DFlash prefill row for slot {} must contain at least one token",
                row.slot
            );
            anyhow::ensure!(
                row.params.is_greedy(),
                "Metal DFlash currently supports greedy sampling only; refusing prefill slot {}",
                row.slot
            );
        }
        Ok(())
    }

    fn run_dflash_prefill_rows(
        &mut self,
        rows: &[infer_plan::PrefillRow],
        kv: &mut dyn KvPool,
    ) -> anyhow::Result<StepOutput> {
        log::info!(
            "Metal DFlash scheduler-prefill lane live: rows={} (serial prefill)",
            rows.len()
        );
        let mut tokens = Vec::new();
        for row in rows {
            let output = materialize_inflight_now(self.submit_prefill(row, kv)?)?;
            tokens.extend(output.tokens);
        }
        Ok(StepOutput { tokens })
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
        let capture_dflash_hidden = self.dflash.is_some();
        if let Some(runtime) = self.dflash.as_ref() {
            model.set_capture_layers(runtime.target_layer_ids())?;
        }
        let logits = match model.prefill_session(
            &token_arr,
            token_values.len() as i32,
            row.start_pos as i32,
        ) {
            Ok(logits) => logits,
            Err(err) => {
                if capture_dflash_hidden {
                    model.clear_capture_layers();
                }
                return Err(err);
            }
        };
        let dflash_hidden = if let Some(runtime) = self.dflash.as_ref() {
            let captured = match model.drain_captured_hidden() {
                Ok(captured) => captured,
                Err(err) => {
                    model.clear_capture_layers();
                    return Err(err);
                }
            };
            model.clear_capture_layers();
            Some(dflash::build_target_hidden_from_captures(
                &captured,
                runtime.target_layer_ids().len(),
                token_values.len() as i32,
            )?)
        } else {
            None
        };
        mlx::async_eval(&[&logits]);
        slot.cache_len = row.start_pos + row.tokens.len();
        slot.committed_len = slot.cache_len;
        slot.last_sampled = None;
        if let Some(hidden) = dflash_hidden {
            if row.start_pos == 0 {
                slot.dflash_target_hidden = Some(hidden);
                if let Some(runtime) = self.dflash.as_ref() {
                    slot.dflash_draft_state = Some(
                        runtime.new_draft_state(row.total_tokens + runtime.block_size() + 512),
                    );
                }
            } else {
                slot.dflash_target_hidden = Some(match slot.dflash_target_hidden.take() {
                    Some(prev) => mlx::concatenate_axis(&[prev, hidden], 0),
                    None => hidden,
                });
            }
        }
        let position = slot.cache_len as u64;
        slot.drain_session(model)?;
        self.active_session_slot = None;
        // Publish is prefill-only by design: engine-core's radix cache only ever
        // offers PROMPT pages for attach (`infer-core` prefix.rs:
        // `publishable_tokens = request.prompt_len().min(self.kv.seq_len(slot))`),
        // so pages/snapshots covering generated tokens are unreachable. The old
        // decode-time publishes were a per-token O(full_pages) re-slice plus an
        // unbounded restore-snapshot leak (`prefixes` is never evicted).
        self.page_store.publish_slot(slot, kv)?;
        // A new prefill restarts this slot's token stream; any decode prequeue
        // from a prior turn is stale.
        self.pending = None;

        let history = row
            .penalty_history
            .as_deref()
            .map(|tokens| infer_plan::PenaltyHistory {
                tokens,
                prompt_len: row.penalty_prompt_len,
            });
        Ok(sample_inflight(
            row.slot,
            &logits,
            &row.params,
            history,
            position,
        ))
    }

    fn submit_dflash_decode_rows(
        &mut self,
        rows: &[infer_plan::DecodeRow],
        kv: &mut dyn KvPool,
    ) -> anyhow::Result<MetalInflight> {
        self.pending = None;
        self.preflight_dflash_decode_rows(rows, kv)?;
        Ok(MetalInflight::Ready(self.run_dflash_decode_rows(rows)?))
    }

    fn run_dflash_decode_rows(
        &mut self,
        rows: &[infer_plan::DecodeRow],
    ) -> anyhow::Result<StepOutput> {
        let model = self.weights.cpp_model()?;
        let block_size = self
            .dflash
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("DFlash decode requested without a runtime"))?
            .block_size();
        for row in rows {
            let slot = self
                .slots
                .get_mut(&row.slot)
                .ok_or_else(|| anyhow::anyhow!("DFlash decode missing slot {}", row.slot))?;
            slot.ensure_kv_capacity(model, block_size)?;
            if slot.session_active {
                slot.drain_session(model)?;
            }
        }

        if rows.len() > 1 {
            log::info!(
                "Metal DFlash scheduler-row lane live: rows={} (serial verified blocks)",
                rows.len()
            );
        }

        let mut tokens = Vec::new();
        for row in rows {
            let output = self.run_dflash_decode_row(row)?;
            tokens.extend(output.tokens);
        }
        Ok(StepOutput { tokens })
    }

    fn preflight_dflash_decode_rows(
        &mut self,
        rows: &[infer_plan::DecodeRow],
        kv: &dyn KvPool,
    ) -> anyhow::Result<()> {
        let runtime = self
            .dflash
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("DFlash decode requested without a runtime"))?;
        anyhow::ensure!(
            !rows.is_empty(),
            "DFlash decode requires at least one decode row"
        );
        anyhow::ensure!(
            rows.len() <= runtime.max_rows(),
            "DFlash decode received {} rows but INFER_METAL_DFLASH_MAX_ROWS allows {}",
            rows.len(),
            runtime.max_rows()
        );
        anyhow::ensure!(
            self.active_session_slot.is_none(),
            "DFlash batched decode requires no active scalar session"
        );
        let mut seen = BTreeSet::new();
        for row in rows {
            anyhow::ensure!(
                seen.insert(row.slot),
                "DFlash decode received duplicate slot {} in one scheduler step",
                row.slot
            );
            anyhow::ensure!(
                row.params.is_greedy(),
                "Metal DFlash currently supports greedy sampling only; refusing slot {}",
                row.slot
            );
            self.reset_slot_if_epoch_changed(row.slot, kv)?;
        }
        for row in rows {
            let slot = self.slots.get(&row.slot).ok_or_else(|| {
                anyhow::anyhow!(
                    "DFlash decode for slot {} has no resident slot; prefix-cache-only DFlash is not wired",
                    row.slot
                )
            })?;
            anyhow::ensure!(
                row.kv_seq_len == slot.committed_len && slot.committed_len == slot.cache_len,
                "DFlash decode kv_seq_len mismatch for slot {}: plan={}, committed={}, metal_state={}",
                row.slot,
                row.kv_seq_len,
                slot.committed_len,
                slot.cache_len
            );
            anyhow::ensure!(
                slot.dflash_target_hidden.is_some(),
                "DFlash decode for slot {} has no target hidden feature store; prefix-cache-only DFlash is not wired",
                row.slot
            );
            anyhow::ensure!(
                slot.dflash_draft_state.is_some(),
                "DFlash decode for slot {} has no draft cache state",
                row.slot
            );
        }
        Ok(())
    }

    fn run_dflash_decode_row(&mut self, row: &infer_plan::DecodeRow) -> anyhow::Result<StepOutput> {
        let runtime = self
            .dflash
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("DFlash decode requested without a runtime"))?;
        let qwen35::Qwen35Embedding::Dense(embed_tokens) = &self.weights.embedding;
        let model = self.weights.cpp_model()?;
        let slot = self
            .slots
            .get_mut(&row.slot)
            .ok_or_else(|| anyhow::anyhow!("DFlash decode missing slot {}", row.slot))?;
        let target_hidden = slot
            .dflash_target_hidden
            .as_ref()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "DFlash decode for slot {} has no target hidden feature store; prefix-cache-only DFlash is not wired",
                    row.slot
                )
            })?;
        let draft_state = slot.dflash_draft_state.as_mut().ok_or_else(|| {
            anyhow::anyhow!(
                "DFlash decode for slot {} has no draft cache state",
                row.slot
            )
        })?;
        let result = dflash::qwen35_speculative_block(
            runtime,
            row.slot,
            row.last_token,
            target_hidden,
            embed_tokens,
            &self.weights.lm_head,
            &self.config,
            model,
            &row.params,
            &mut slot.kv_flat,
            &mut slot.gdr_flat,
            &mut slot.cache_len,
            draft_state,
        )?;
        let output = result.output;
        slot.committed_len = slot.cache_len;
        slot.dflash_target_hidden = Some(result.updated_target_hidden);
        slot.last_sampled = None;
        self.active_session_slot = None;
        Ok(output)
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
            && row.params.is_raw_argmax()
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
        let kv_cache_dtype = self.kv_cache_dtype;
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
        // Session KV-recall: emit the layer-0 query from this step's forward so
        // the post-drain scoring (`maybe_recompute_recall`) can plan the next
        // step's ranges. Touched only when recall is opted in on a bf16 cache —
        // the default path makes no recall FFI call and stays byte-identical.
        if self.kv_recall && kv_cache_dtype == MetalKvCacheDtype::Bf16 {
            model.set_recall_emit_query(true);
        }
        self.active_session_slot = Some(row.slot);
        let token_arr = mlx::MlxArray::from_slice_i32(&[row.last_token as i32], &[1]);
        let logits = step_session_decode(model, slot, kv_cache_dtype, &token_arr)?;
        mlx::async_eval(&[&logits]);
        slot.cache_len = slot.cache_len.saturating_add(1);
        slot.committed_len = slot.cache_len;
        let position = slot.cache_len as u64;
        slot.drain_session(model)?;
        self.active_session_slot = None;
        self.maybe_recompute_recall(row.slot)?;

        let history = row
            .penalty_history
            .as_deref()
            .map(|tokens| infer_plan::PenaltyHistory {
                tokens,
                prompt_len: row.penalty_prompt_len,
            });
        let inflight = sample_inflight(row.slot, &logits, &row.params, history, position);

        // Cold start of a greedy decode run: seed the pipeline. Record this
        // step's sampled token and issue the next step's forward so subsequent
        // ticks take the fast path and overlap.
        if pipeline_decode_enabled()
            && row.params.is_raw_argmax()
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
        self.maybe_recompute_recall(row.slot)?;
        self.prequeue_decode(row.slot, kv)
    }

    /// Session KV-recall (#2/#3/#5): after a step drains (so `slot.kv_flat` holds
    /// the up-to-date K cache) score the resident block reps against this step's
    /// layer-0 query and set `recall_ranges` for the next step. No-op unless
    /// `--kv-recall` is on; recall is bf16-only (int8 falls back to full
    /// attention, logged once). Off → the decode hot path is byte-identical.
    fn maybe_recompute_recall(&mut self, slot_idx: usize) -> anyhow::Result<()> {
        if !self.kv_recall {
            return Ok(());
        }
        if self.kv_cache_dtype != MetalKvCacheDtype::Bf16 {
            if !self.recall_int8_warned {
                log::warn!(
                    "--kv-recall requested with int8 KV cache; recall is bf16-only — \
                     falling back to full attention (use --kv-cache-dtype bf16 to enable recall)"
                );
                self.recall_int8_warned = true;
            }
            return Ok(());
        }
        let model = self.weights.cpp_model()?;
        let cfg = self.recall_cfg;
        if let Some(slot) = self.slots.get_mut(&slot_idx) {
            slot.recompute_recall_plan(model, &cfg)?;
        }
        Ok(())
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
        let kv_cache_dtype = self.kv_cache_dtype;
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
        // Session KV-recall: keep the layer-0 query emit on for the prequeued
        // step; this step's drain (in `commit_pending_then_prequeue`) scores it.
        // Untouched on the default path → byte-identical.
        if self.kv_recall && kv_cache_dtype == MetalKvCacheDtype::Bf16 {
            model.set_recall_emit_query(true);
        }
        self.active_session_slot = Some(slot_idx);
        let logits = step_session_decode(model, slot, kv_cache_dtype, &token_arr)?;
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
            if self.pending.as_ref().is_some_and(|p| p.slot == slot) {
                self.pending = None;
            }
        }
        Ok(())
    }
}

#[cfg(feature = "metal")]
#[path = "kv_ssd.rs"]
mod kv_ssd;
#[cfg(feature = "metal")]
use kv_ssd::*;

#[cfg(feature = "metal")]
#[path = "slot.rs"]
mod slot;
#[cfg(feature = "metal")]
use slot::*;

#[cfg(feature = "metal")]
fn step_session_decode(
    model: &qwen35::CppQwen35Model,
    slot: &MetalSlotState,
    kv_cache_dtype: MetalKvCacheDtype,
    token: &mlx::MlxArray,
) -> anyhow::Result<mlx::MlxArray> {
    let cache_pos = usize_to_i32(slot.cache_len)?;
    if paged_kv_read_enabled() {
        if slot.cache_len > 0 {
            let logits = match kv_cache_dtype {
                MetalKvCacheDtype::Bf16 => {
                    // Recall plan set → attend only sink ∪ recalled ∪ local (#5);
                    // otherwise the full contiguous read (byte-identical default).
                    let (k_full, v_full) = match &slot.recall_ranges {
                        Some(ranges) => slot.bf16_recall_read_inputs(ranges)?,
                        None => slot.bf16_prefix_read_inputs(slot.cache_len)?,
                    };
                    model.step_session_paged_bf16(token, cache_pos, &k_full, &v_full)
                }
                MetalKvCacheDtype::Int8 => {
                    let (k_full, v_full) = slot.int8_prefix_read_inputs(slot.cache_len)?;
                    model.step_session_paged_int8(token, cache_pos, &k_full, &v_full)
                }
            }
            .map_err(|err| anyhow::anyhow!("paged KV read step_session failed: {err}"))?;
            probe_paged_kv_read_hit();
            return Ok(logits);
        }
        probe_paged_kv_read_fallback();
    }
    model.step_session(token, cache_pos)
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

/// Gather a token-axis subset of a rank-4 KV array by slicing each contiguous
/// token range and concatenating along the token axis (axis 2). The building
/// block of session KV-recall reads (sink ∪ recalled blocks ∪ local). Dtype-
/// agnostic — reuses `slice_kv_tokens` + `concatenate_or_single`.
#[cfg(feature = "metal")]
fn gather_kv_ranges(
    array: &mlx::MlxArray,
    ranges: &[(usize, usize)],
) -> anyhow::Result<mlx::MlxArray> {
    let parts: anyhow::Result<Vec<_>> = ranges
        .iter()
        .map(|&(s, e)| slice_kv_tokens(array, s, e))
        .collect();
    Ok(concatenate_or_single(parts?))
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

/// NextN/MTP draft head auto-resolved for the Qwen3.6-27B family when the user
/// gives no explicit draft and does not opt out. Auto-downloads through the same
/// `model_source::resolve_model_path` hf_hub path as `--model-path`.
#[cfg(feature = "metal")]
const QWEN36_27B_DEFAULT_DRAFT_MODEL: &str = "mlx-community/Qwen3.6-27B-MTP-4bit";

/// Default speculative draft depth (measured optimum on Qwen3.6-27B: ~18.1 tok/s
/// vs ~12.3 no-spec). Matches the `arle serve --speculative-tokens` default.
#[cfg(feature = "metal")]
const METAL_DEFAULT_SPECULATIVE_TOKENS: usize = 2;

/// True when the served model is Qwen3.6-27B-family, so its NextN/MTP draft head
/// should be auto-enabled. Matches on the resolved id/path containing
/// `Qwen3.6-27B` — OptiQ (`...-27B-OptiQ-4bit`) and MTP variants match; Qwen3.5
/// and the 35B-A3B MoE do not. The cache snapshot dir name
/// (`models--mlx-community--Qwen3.6-27B-OptiQ-4bit`) carries the substring, as do
/// local dirs (`models/Qwen3.6-27B-...`).
#[cfg(feature = "metal")]
fn is_qwen36_27b_model(resolved: &Path) -> bool {
    resolved.to_string_lossy().contains("Qwen3.6-27B")
}

/// Resolve the Metal DFlash speculative-decode runtime for the served model,
/// from the applied runtime flags (CLI `--no-speculative` / `--draft-model` /
/// `--speculative-tokens` / `--spec-accept-topk`).
///
/// Precedence:
///   1. `--no-speculative` → disabled (also suppresses the Qwen3.6-27B auto-enable).
///   2. `--draft-model` non-empty → explicit draft head.
///   3. Qwen3.6-27B-family served model → auto-enable the NextN-MTP draft head at
///      the default depth, downloading it on first use.
///   4. otherwise → no speculative decode.
#[cfg(feature = "metal")]
fn resolve_dflash(
    resolved: &Path,
    config: &config::MetalModelConfig,
) -> anyhow::Result<Option<dflash::MetalDflashRuntime>> {
    let flags = crate::runtime_flags::spec_flags();
    if !flags.speculative {
        return Ok(None);
    }

    let flag_draft = flags
        .draft_model
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string);

    let (draft_model, default_tokens) = match flag_draft {
        Some(draft) => (draft, None),
        None if is_qwen36_27b_model(resolved) => {
            eprintln!(
                "[infer-metal] speculative decode auto-enabled: NextN-MTP head {QWEN36_27B_DEFAULT_DRAFT_MODEL}, depth {METAL_DEFAULT_SPECULATIVE_TOKENS}"
            );
            (
                QWEN36_27B_DEFAULT_DRAFT_MODEL.to_string(),
                Some(METAL_DEFAULT_SPECULATIVE_TOKENS),
            )
        }
        None => return Ok(None),
    };

    let max_rows = match std::env::var("INFER_METAL_DFLASH_MAX_ROWS") {
        Ok(value) if !value.trim().is_empty() => value.trim().parse::<usize>().map_err(|err| {
            anyhow::anyhow!("invalid INFER_METAL_DFLASH_MAX_ROWS='{value}': {err}")
        })?,
        _ => 4,
    };
    let options = dflash::MetalDflashOptions {
        draft_model,
        speculative_tokens: flags.speculative_tokens.or(default_tokens),
        max_rows,
        accept_topk: flags.spec_accept_topk.max(1),
    };
    dflash::MetalDflashRuntime::load(&options, config).map(Some)
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
pub(crate) fn int8_kv_group_size(head_dim: usize) -> anyhow::Result<usize> {
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
fn estimated_metal_kv_page_bytes(
    config: &config::MetalModelConfig,
    kv_cache_dtype: MetalKvCacheDtype,
    page_size: usize,
) -> usize {
    let layers = config.arch.num_full_attention_layers();
    let nkv = config.num_key_value_heads;
    let hd = config.head_dim;
    match kv_cache_dtype {
        MetalKvCacheDtype::Bf16 => layers
            .saturating_mul(2)
            .saturating_mul(nkv)
            .saturating_mul(page_size)
            .saturating_mul(hd)
            .saturating_mul(dtype_size(mlx::Dtype::Bfloat16)),
        MetalKvCacheDtype::Int8 => {
            let group_size = int8_kv_group_size(config.head_dim).unwrap_or(128);
            let packed = nkv
                .saturating_mul(page_size)
                .saturating_mul(hd / 4)
                .saturating_mul(dtype_size(mlx::Dtype::Uint32));
            let scale_or_bias = nkv
                .saturating_mul(page_size)
                .saturating_mul(hd / group_size)
                .saturating_mul(dtype_size(mlx::Dtype::Bfloat16));
            layers.saturating_mul(
                packed
                    .saturating_mul(2)
                    .saturating_add(scale_or_bias.saturating_mul(4)),
            )
        }
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
            (0..full_layers)
                .flat_map(|_| {
                    [
                        mlx::zeros(&cache_shape, mlx::Dtype::Bfloat16),
                        mlx::zeros(&cache_shape, mlx::Dtype::Bfloat16),
                    ]
                })
                .collect()
        }
        MetalKvCacheDtype::Int8 => {
            let group_size = int8_kv_group_size(config.head_dim)
                .expect("validated before slot allocation") as i32;
            let packed_shape = [1, nkv, capacity, hd / 4];
            let scale_shape = [1, nkv, capacity, hd / group_size];
            // K: packed uint32 data + bf16 scale/bias, then V with the same
            // layout. C++ session code interprets n_kv=6*full_layers as
            // quantized KV.
            (0..full_layers)
                .flat_map(|_| {
                    [
                        mlx::zeros(&packed_shape, mlx::Dtype::Uint32),
                        mlx::zeros(&scale_shape, mlx::Dtype::Bfloat16),
                        mlx::zeros(&scale_shape, mlx::Dtype::Bfloat16),
                        mlx::zeros(&packed_shape, mlx::Dtype::Uint32),
                        mlx::zeros(&scale_shape, mlx::Dtype::Bfloat16),
                        mlx::zeros(&scale_shape, mlx::Dtype::Bfloat16),
                    ]
                })
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_pool::MetalKvPool;
    use infer_plan::{DecodeRow, ForwardMode, PrefillRow};

    // The whole point of seam-level dispatch (#68): a backend resolves the
    // requested dtype against its OWN support matrix and fails loud on an
    // unsupported request rather than silently downgrading. Metal supports
    // bf16/int8 (Auto → int8); the CUDA-only fp8/tq4 modes must error here.
    #[test]
    fn resolve_metal_support_matrix() {
        use infer_seam::KvCacheDtype;
        assert_eq!(
            MetalKvCacheDtype::resolve(KvCacheDtype::Auto).unwrap(),
            MetalKvCacheDtype::Int8
        );
        assert_eq!(
            MetalKvCacheDtype::resolve(KvCacheDtype::Int8).unwrap(),
            MetalKvCacheDtype::Int8
        );
        assert_eq!(
            MetalKvCacheDtype::resolve(KvCacheDtype::Bf16).unwrap(),
            MetalKvCacheDtype::Bf16
        );
        assert!(MetalKvCacheDtype::resolve(KvCacheDtype::Fp8).is_err());
        assert!(MetalKvCacheDtype::resolve(KvCacheDtype::Tq4).is_err());
    }

    // Session KV-recall page-gather (#4): slicing selected contiguous token ranges
    // (sink ∪ recalled blocks ∪ local) and concatenating along the token axis must
    // produce a working set whose token count is the sum of the range sizes, in
    // order — and reject out-of-range ends. Dtype-agnostic, so an i32 array suffices.
    #[cfg(feature = "metal")]
    #[test]
    fn gather_kv_ranges_concats_selected_token_ranges() {
        // rank-4 [1, 1, 6, 2] (batch, kv-heads, tokens=6, head_dim=2).
        let data: Vec<i32> = (0..12).collect();
        let arr = mlx::MlxArray::from_slice_i32(&data, &[1, 1, 6, 2]);
        // sink [0,2) + local [4,6) -> 4 tokens kept (the recall working-set shape).
        let g = gather_kv_ranges(&arr, &[(0, 2), (4, 6)]).unwrap();
        mlx::eval(&[&g]);
        assert_eq!(g.shape()[2], 4, "gathered token count = sum of range sizes");
        // a single full range round-trips the token count.
        let full = gather_kv_ranges(&arr, &[(0, 6)]).unwrap();
        assert_eq!(full.shape()[2], 6);
        // an end past the token axis is rejected, not silently clamped.
        assert!(gather_kv_ranges(&arr, &[(0, 7)]).is_err());
    }

    // Session KV-recall reps (#2): `update_block_reps` mean-pools layer-0 K over
    // each frozen middle block into a resident `[nkv, hd]` rep. With n_init=0,
    // n_local=2, l_bs=2 over a 6-token cache, the middle is [0,4) = 2 frozen
    // blocks; their reps are the per-block token means. Per-token-constant values
    // (bf16-exact) make the expected mean exact.
    #[cfg(feature = "metal")]
    #[test]
    fn update_block_reps_mean_pools_frozen_middle_blocks() {
        let _guard = mlx_sys::mlx_guard();
        // [B=1, nkv=1, seq=6, hd=2]; token t holds (t, t) so block [0,2) mean is
        // (0.5,0.5) and block [2,4) mean is (2.5,2.5).
        let k: Vec<i32> = vec![0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5];
        let k = mlx::MlxArray::from_slice_i32(&k, &[1, 1, 6, 2]);
        let k = mlx::as_dtype(&k, mlx::Dtype::Bfloat16);
        // V is unused by reps; share the K shape.
        let v = mlx::zeros(&[1, 1, 6, 2], mlx::Dtype::Bfloat16);
        let mut slot = MetalSlotState::from_arrays(0, 0, 6, vec![k, v], Vec::new());
        let cfg = infer_core::RecallConfig {
            n_init: 0,
            n_local: 2,
            l_bs: 2,
            top_k: 1,
        };
        slot.update_block_reps(&cfg).unwrap();
        assert_eq!(slot.block_reps.len(), 2, "two frozen middle blocks");
        assert_eq!(slot.block_reps[0], vec![0.5, 0.5], "block [0,2) token mean");
        assert_eq!(slot.block_reps[1], vec![2.5, 2.5], "block [2,4) token mean");
        // Idempotent: re-running adds no blocks (only newly-completed recomputed).
        slot.update_block_reps(&cfg).unwrap();
        assert_eq!(slot.block_reps.len(), 2);
    }

    // Qwen3.6-27B-family auto-enable detection must fire for OptiQ + MTP variants
    // (whose cache-dir / id carries `Qwen3.6-27B`) and must NOT misfire on
    // Qwen3.5 or the 35B-A3B MoE — otherwise a non-MTP model would try to load a
    // mismatched draft head.
    #[cfg(feature = "metal")]
    #[test]
    fn qwen36_27b_detection_matches_family_and_rejects_others() {
        use std::path::Path;
        // Positive: OptiQ + MTP, both HF cache-dir and local-dir shapes.
        for p in [
            "/u/.cache/huggingface/hub/models--mlx-community--Qwen3.6-27B-OptiQ-4bit/snapshots/abc",
            "/u/.cache/huggingface/hub/models--mlx-community--Qwen3.6-27B-MTP-4bit/snapshots/abc",
            "models/Qwen3.6-27B-OptiQ-4bit",
        ] {
            assert!(
                is_qwen36_27b_model(Path::new(p)),
                "expected Qwen3.6-27B detection for {p}"
            );
        }
        // Negative: Qwen3.5 and the 35B-A3B MoE must not auto-enable.
        for p in [
            "models/Qwen3.5-0.8B-MLX-4bit",
            "models/Qwen3.5-4B",
            "/u/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/abc",
        ] {
            assert!(
                !is_qwen36_27b_model(Path::new(p)),
                "expected NO Qwen3.6-27B detection for {p}"
            );
        }
    }

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

    #[cfg(feature = "metal")]
    #[test]
    fn bf16_prefix_read_inputs_slices_live_prefix_pairs() {
        let _guard = mlx_sys::mlx_guard();
        let state = MetalSlotState::from_arrays(
            0,
            0,
            3,
            vec![
                kv_bf16_array(5, 10),
                kv_bf16_array(5, 20),
                kv_bf16_array(5, 30),
                kv_bf16_array(5, 40),
            ],
            vec![],
        );

        let (k_full, v_full) = state.bf16_prefix_read_inputs(3).unwrap();
        assert_eq!(k_full.len(), 2);
        assert_eq!(v_full.len(), 2);
        assert_eq!(k_full[0].shape(), &[1, 1, 3, 2]);
        assert_eq!(v_full[1].shape(), &[1, 1, 3, 2]);
        assert_eq!(k_full[0].dtype(), mlx::Dtype::Bfloat16);

        let k0 = mlx::as_dtype(&k_full[0], mlx::Dtype::Float32);
        mlx::eval(&[&k0]);
        assert_eq!(k0.as_slice_f32(), &[10.0, 11.0, 20.0, 21.0, 30.0, 31.0]);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn int8_prefix_read_inputs_slices_q_scale_bias_triples() {
        let _guard = mlx_sys::mlx_guard();
        let state = MetalSlotState::from_arrays(
            0,
            0,
            3,
            vec![
                kv_u32_array(5, 10),
                kv_bf16_array(5, 20),
                kv_bf16_array(5, 30),
                kv_u32_array(5, 40),
                kv_bf16_array(5, 50),
                kv_bf16_array(5, 60),
            ],
            vec![],
        );

        let (k_full, v_full) = state.int8_prefix_read_inputs(3).unwrap();
        assert_eq!(k_full.len(), 3);
        assert_eq!(v_full.len(), 3);
        assert_eq!(k_full[0].shape(), &[1, 1, 3, 2]);
        assert_eq!(k_full[0].dtype(), mlx::Dtype::Uint32);
        assert_eq!(k_full[1].dtype(), mlx::Dtype::Bfloat16);
        assert_eq!(v_full[0].dtype(), mlx::Dtype::Uint32);

        let k_scale = mlx::as_dtype(&k_full[1], mlx::Dtype::Float32);
        mlx::eval(&[&k_scale]);
        assert_eq!(
            k_scale.as_slice_f32(),
            &[20.0, 21.0, 30.0, 31.0, 40.0, 41.0]
        );
    }

    /// Rank-4 `[1, 1, seq, 2]` f32 K/V array filled with `fill` — the minimal
    /// shape `slice_kv_tokens` accepts, no model load needed.
    #[cfg(feature = "metal")]
    fn kv_array(seq: usize, fill: i32) -> mlx::MlxArray {
        let vals = vec![fill; seq * 2];
        let arr = mlx::MlxArray::from_slice_i32(&vals, &[1, 1, seq as i32, 2]);
        mlx::as_dtype(&arr, mlx::Dtype::Float32)
    }

    /// Rank-4 `[1, 1, seq, 2]` bf16 K/V array with per-token values:
    /// `base + token*10 + dim`. Small integers round-trip exactly through bf16.
    #[cfg(feature = "metal")]
    fn kv_bf16_array(seq: usize, base: i32) -> mlx::MlxArray {
        let vals: Vec<i32> = (0..seq)
            .flat_map(|token| [base + token as i32 * 10, base + token as i32 * 10 + 1])
            .collect();
        let arr = mlx::MlxArray::from_slice_i32(&vals, &[1, 1, seq as i32, 2]);
        mlx::as_dtype(&arr, mlx::Dtype::Bfloat16)
    }

    #[cfg(feature = "metal")]
    fn kv_u32_array(seq: usize, base: i32) -> mlx::MlxArray {
        let vals: Vec<i32> = (0..seq)
            .flat_map(|token| [base + token as i32 * 10, base + token as i32 * 10 + 1])
            .collect();
        let arr = mlx::MlxArray::from_slice_i32(&vals, &[1, 1, seq as i32, 2]);
        mlx::as_dtype(&arr, mlx::Dtype::Uint32)
    }

    /// Tiny stand-in restore-state array carrying a distinguishable `fill`
    /// value so a test can tell WHICH occupant's snapshot survives in
    /// `prefixes`.
    #[cfg(feature = "metal")]
    fn gdr_array(fill: i32) -> mlx::MlxArray {
        let arr = mlx::MlxArray::from_slice_i32(&[fill], &[1]);
        mlx::as_dtype(&arr, mlx::Dtype::Float32)
    }

    #[cfg(feature = "metal")]
    fn resident_prefix_blocks(pages: &[u32]) -> Vec<PrefixBlock> {
        pages
            .iter()
            .copied()
            .map(PrefixBlock::ResidentPage)
            .collect()
    }

    #[cfg(feature = "metal")]
    fn temp_ssd_root(label: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |duration| duration.as_nanos());
        let root = std::env::temp_dir().join(format!(
            "arle-metal-ssd-{label}-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    // Defect-2 regression guard (stale prefix snapshot aliasing): host page ids
    // are recycled LIFO after radix eviction, but `prefixes` keys used to live
    // forever. A later radix match colliding with a stale key would serve the
    // NEW occupant's K/V pages with the OLD occupant's restore state. Publishing
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
        // -> publishes both page blocks and a restore snapshot.
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
        let first_key = store
            .logical_key_for_pages(&first_pages)
            .expect("first occupant logical key");
        assert_eq!(
            store.reusable_prefix_blocks(&resident_prefix_blocks(&first_pages)),
            2
        );

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
            !store.prefixes.contains_key(&first_key),
            "stale prefix key {first_key:?} must be pruned on page reuse"
        );
        assert_eq!(
            store.reusable_prefix_blocks(&resident_prefix_blocks(&first_pages)),
            0
        );

        // The new occupant's own boundary snapshot survives and carries ITS
        // restore state, not the first occupant's.
        assert_eq!(
            store.reusable_prefix_blocks(&resident_prefix_blocks(&second_pages)),
            2
        );
        let second_key = store
            .logical_key_for_pages(&second_pages)
            .expect("second occupant logical key");
        let snap = store
            .prefixes
            .get(&second_key)
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
        let one_key = store
            .logical_key_for_pages(&one_page)
            .expect("one-page logical key");
        assert_eq!(
            store.reusable_prefix_blocks(&resident_prefix_blocks(&one_page)),
            1
        );

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
        let two_key = store
            .logical_key_for_pages(&two_pages)
            .expect("two-page logical key");

        assert!(store.prefixes.contains_key(&one_key));
        assert!(store.prefixes.contains_key(&two_key));
        assert_eq!(
            store.reusable_prefix_blocks(&resident_prefix_blocks(&one_page)),
            1
        );
        assert_eq!(
            store.reusable_prefix_blocks(&resident_prefix_blocks(&two_pages)),
            2
        );
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
        let key = store
            .logical_key_for_pages(&pages)
            .expect("published logical key");
        assert_eq!(store.pages.len(), 2);
        assert_eq!(
            store.reusable_prefix_blocks(&resident_prefix_blocks(&pages)),
            2
        );

        store.release_pages(&[pages[0]]);

        assert!(
            !store.pages.contains_key(&pages[0]),
            "evicted page mirror must be dropped"
        );
        assert!(
            !store.prefixes.contains_key(&key),
            "prefix snapshot containing the evicted page must be pruned"
        );
        assert_eq!(
            store.reusable_prefix_blocks(&resident_prefix_blocks(&pages)),
            0
        );
    }

    #[cfg(feature = "metal")]
    #[test]
    fn ssd_write_through_promotes_released_pages_and_prefix_snapshot() {
        use infer_seam::{KvAllocator, KvQuery};
        let _guard = mlx_sys::mlx_guard();
        let root = temp_ssd_root("promote");
        let mut store = MetalPageStore::default();
        assert!(store.set_ssd(root.clone(), 8 * 1024 * 1024, 1024));
        let mut pool = MetalKvPool::new(1, 8, 4);

        pool.alloc(0, 8).unwrap();
        let pages: Vec<u32> = pool.page_indices(0).to_vec();
        let state = MetalSlotState::from_arrays(
            0,
            pool.slot_epoch(0),
            8,
            vec![kv_bf16_array(8, 10)],
            vec![gdr_array(7)],
        );
        store.publish_slot(&state, &pool).unwrap();
        let logical_key = store
            .logical_key_for_pages(&pages)
            .expect("published logical key");
        assert!(
            store.has_disk_prefix(&logical_key),
            "write-through must persist the prefix snapshot"
        );
        assert_eq!(
            store
                .demote_prefix_pages(&[(pages[0], 10), (pages[1], 11)])
                .unwrap(),
            2
        );

        store.release_pages(&pages);
        assert!(store.pages.is_empty(), "release must drop RAM page mirrors");
        assert!(
            store.prefixes.is_empty(),
            "release must drop RAM prefix snapshots"
        );
        assert_eq!(
            store.reusable_prefix_blocks(&resident_prefix_blocks(&pages)),
            0
        );

        let demoted = [PrefixBlock::DemotedKey(10), PrefixBlock::DemotedKey(11)];
        assert_eq!(
            store.reusable_prefix_blocks(&demoted),
            2,
            "T2 mcheck must prove the prefix before mget/promote"
        );
        store
            .promote_prefix_pages(&[(10, pages[0]), (11, pages[1])])
            .unwrap();
        assert_eq!(
            store.reusable_prefix_blocks(&resident_prefix_blocks(&pages)),
            2
        );
        let restored = store
            .materialize_slot_from_prefix(0, pool.slot_epoch(0), &pool, 8, 8)
            .unwrap();
        assert_eq!(restored.cache_len, 8);
        let kv = mlx::as_dtype(&restored.kv_flat[0], mlx::Dtype::Float32);
        mlx::eval(&[&kv, &restored.gdr_flat[0]]);
        assert_eq!(
            &kv.as_slice_f32()[..16],
            &[
                10.0, 11.0, 20.0, 21.0, 30.0, 31.0, 40.0, 41.0, 50.0, 51.0, 60.0, 61.0, 70.0, 71.0,
                80.0, 81.0
            ]
        );
        assert_eq!(restored.gdr_flat[0].as_slice_f32(), &[7.0]);
        let _ = std::fs::remove_dir_all(root);
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
                    penalty_history: None,
                    penalty_prompt_len: 0,
                },
                DecodeRow {
                    slot: 1,
                    last_token: 20,
                    kv_seq_len: 7,
                    params: infer_plan::SamplingParams::default(),
                    penalty_history: None,
                    penalty_prompt_len: 0,
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
                penalty_history: None,
                penalty_prompt_len: 0,
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
