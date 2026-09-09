//! `LoadedInferenceEngine` — the public engine handle.
//!
//! A backend-neutral struct over a [`ServeInferenceEngine`] plus the runtime
//! backend name. Backend selection is a runtime decision: the leaf binary
//! registers one [`BackendBuilderFn`] per compiled-in backend via
//! [`register_backend`], and [`LoadedInferenceEngine::load_with_config`]
//! dispatches by [`EngineLoadConfig::backend`] (or the first registered
//! backend when unset). [`EngineLoadConfig`] is always available; the CUDA
//! OPD/multiproc helpers under [`cuda`] require the `cuda` feature.

use std::sync::Arc;

use anyhow::Result;
use tokio::sync::mpsc::UnboundedSender;

use crate::serve_engine::ServeInferenceEngine;
use crate::types::{
    ChatPromptMessage, CompletionOutput, CompletionRequest, CompletionStreamDelta, EngineTelemetry,
    InferenceEngine, MultimodalChatRequest,
};

/// Requested KV-cache dtype — re-exported from the device-neutral seam so the
/// service/scheduler layers stay backend-agnostic. Backends resolve it against
/// their own support matrix at construction (Metal → `MetalKvCacheDtype`).
pub use infer_seam::KvCacheDtype;
/// Requested KV tier budget (bytes | fraction | off) — re-exported from the
/// seam like [`KvCacheDtype`]. Deployment-total; the engine constructor
/// divides by the TP world size.
pub use infer_seam::KvTierBudget;

/// Slot / page configuration for [`LoadedInferenceEngine::load_with_config`].
///
/// Serde: the multiproc coordinator serializes its resolved config into
/// `ARLE_WORKER_ENGINE_CONFIG` so worker ranks build their engines from the
/// SAME values — any divergence (slots, budgets, chunk size) diverges the
/// deterministic planner across ranks and deadlocks the NCCL lockstep.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EngineLoadConfig {
    /// Executor hot-workspace slots. Serving CLI leaves this at the default
    /// unless an internal caller deliberately budgets executor capacity.
    pub num_slots: usize,
    /// Requested active-request cap. Builders size hot workspace to cover it;
    /// the scheduler enforces it after backend budget clamps.
    #[serde(default)]
    pub max_running_requests: Option<usize>,
    pub total_pages: usize,
    pub page_size: usize,
    pub max_prompt_tokens: usize,
    pub max_total_tokens: usize,
    /// Per-request prefill chunk size; `None` = backend/model-kind default.
    #[serde(default)]
    pub chunked_prefill_size: Option<usize>,
    /// `Some(n)` = MTP spec decode on with draft depth `n`; `None` = off.
    pub mtp_draft_tokens: Option<usize>,
    /// `Some(k)` = D2 MTP root-branch top-k width; verifier rows are root + candidates.
    #[serde(default)]
    pub mtp_draft_topk: Option<usize>,
    /// Requested KV-cache dtype. Backends resolve `Auto` inside their own
    /// builder so the service/scheduler layers stay device-neutral.
    #[serde(default)]
    pub kv_cache_dtype: KvCacheDtype,
    /// Whole-process memory budget for unified-memory backends. Metal maps this
    /// to MLX memory/cache/wired limits before loading weights and clamps KV
    /// capacity to fit. `None` lets the backend derive a budget from physical
    /// and currently available memory.
    #[serde(default)]
    pub memory_budget_bytes: Option<usize>,
    /// Physical memory to leave for macOS and foreground apps on unified-memory
    /// backends. `None` uses the backend's anti-swap default.
    #[serde(default)]
    pub system_reserve_bytes: Option<usize>,
    /// Allow startup when macOS swap is already materially active. Default is
    /// fail-closed because swap uses SSD and can stall the whole system.
    #[serde(default)]
    pub allow_swap: bool,
    /// Low-impact local serving mode: keep work chunks cooperative for desktop
    /// responsiveness. Backend builders may install a resource governor when
    /// this is set; server-style defaults leave it off.
    #[serde(default)]
    pub low_impact: bool,
    /// Generation-token bound for chat requests that enable thinking
    /// (`chat_template_kwargs.enable_thinking=true`). `0` = off (unbounded),
    /// keeping the chat path byte-identical to before this knob. The OpenAI
    /// facade clamps `max_tokens` to this when thinking is on.
    #[serde(default)]
    pub max_thinking_tokens: usize,
    /// Fraction of total VRAM the static KV pool may claim, profiled from
    /// MEASURED free VRAM after weights load (SGLang's `mem_fraction_static`).
    /// `reserve = total × (1 − frac)` is left for activations/scratch; the rest
    /// of free VRAM becomes the KV token pool. Clamped to `[0.05, 0.97]` by the
    /// sizer. Wired for the Qwen3.5/3.6 full-attention pool
    /// (`profile_kv_pool_tokens`); DSv4 keeps its per-slot sizing.
    #[serde(default = "default_mem_fraction_static")]
    pub mem_fraction_static: f64,
    /// L2 (host DRAM) KV tier budget, deployment-total. Default: half of
    /// MemAvailable. `Off` disables the level.
    #[serde(default)]
    pub kv_dram: KvTierBudget,
    /// Opt-in L3 (NVMe) KV spill root (`--kv-disk`). Lives in the engine
    /// config — not the serve-layer options — so the multiproc coordinator's
    /// `ARLE_WORKER_ENGINE_CONFIG` carries it and every worker rank attaches
    /// the tier at build (each process namespaces its own store under this
    /// root, so ranks never collide).
    #[serde(default)]
    pub kv_ssd_root: Option<std::path::PathBuf>,
    /// L3 (NVMe) cap under `kv_ssd_root`, deployment-total. `None` derives
    /// half of free disk at the root; `Some` without a root fails closed.
    #[serde(default)]
    pub kv_disk_limit: Option<KvTierBudget>,
    /// Serve-wide sampling defaults for fields the request omits, applied over
    /// the checkpoint's `generation_config.json`. A checkpoint that ships no
    /// sampling config (LFM2.5-MLX) otherwise serves greedy with no penalty,
    /// and a client that sends no sampling fields (eli) cannot change that.
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    /// Opt-in running-cap oversubscription: rotate waiters in by parking the
    /// longest-running decode's whole-slot image (requires a whole-slot tier
    /// backend). Default false → byte-identical.
    #[serde(default)]
    pub slot_oversubscription: bool,
    /// Minimum decode tokens before an oversubscribed request may be parked again.
    #[serde(default = "default_oversubscription_min_slice")]
    pub oversubscription_min_slice: usize,
    /// `--lora-adapters`: trained student LoRA safetensors (train
    /// `--save-lora-adapters` output) re-merged into the resident projection
    /// weights once at engine build. Rides the engine config so multiproc
    /// worker ranks apply it too. CUDA Qwen3.5/3.6 only.
    #[serde(default)]
    pub student_lora_adapters: Option<std::path::PathBuf>,
    /// LoRA alpha for the `--lora-adapters` re-merge (`scale = alpha / rank`;
    /// rank is read from the adapter tensor shapes).
    #[serde(default = "default_student_lora_alpha")]
    pub student_lora_alpha: f32,
    /// `--spec-type dspark`: DSpark/DFlash block-drafter checkpoint dir for the
    /// CUDA Qwen3.5/3.6 executor. Rides the engine config so multiproc worker
    /// ranks load it too. `None` = spec off (baseline byte-identical).
    #[serde(default)]
    pub dspark_draft_model: Option<std::path::PathBuf>,
    /// DSpark verify-step cost model `step_ms = bias + row · verify_rows`
    /// driving the goodput budget (checkpoints without a confidence head
    /// ignore it). Defaults are the H20 ThinkingCap-27B c=16 measurement.
    #[serde(default = "default_dspark_sps_bias_ms")]
    pub dspark_sps_bias_ms: f32,
    #[serde(default = "default_dspark_sps_row_ms")]
    pub dspark_sps_row_ms: f32,
    /// Materialize an empty Markov head slot of this rank when the draft
    /// checkpoint ships without one (DFlash backbones do). Set only by
    /// `--dspark-markov-init`, which has nothing to install over otherwise; a
    /// plain serve leaves the drafter head-less rather than pay a vocab-wide
    /// gemm to add zero.
    #[serde(default)]
    pub markov_head_rank: Option<usize>,
    /// Cap the draft block length. A block longer than the accepted prefix costs a
    /// draft forward and a verify row per position and can never commit them.
    #[serde(default)]
    pub dspark_block_size: Option<usize>,
    /// CUDA runtime toggles (CLI flags → `infer_cuda::apply_runtime_flags`
    /// before executor construction; multiproc workers included).
    #[serde(default)]
    pub cuda: infer_seam::CudaRuntimeFlags,
    /// Metal runtime toggles (CLI flags → `infer_metal::apply_runtime_flags`).
    #[serde(default)]
    pub metal: infer_seam::MetalRuntimeFlags,
    /// `--diffusion-max-denoising-steps`: cap block-diffusion denoising steps
    /// per row. `None` = checkpoint default.
    #[serde(default)]
    pub diffusion_max_denoising_steps: Option<usize>,
    /// `--vulkan-submit-cap`: max compute dispatches per Vulkan command buffer
    /// (TDR/latency safety valve). `None` = whole token in one submit.
    #[serde(default)]
    pub vulkan_submit_cap: Option<usize>,
    /// Explicit world size (total GPU ranks) for this engine. `None` = resolve
    /// from `INFER_TP_SIZE` / `INFER_CUDA_DEVICES` env (the legacy path).
    /// Set from `--tensor-parallel-size × --context-parallel-size`.
    #[serde(default)]
    pub world_size: Option<usize>,
    /// Context-parallel degree (sequence sharded across N GPUs). `None` = 1.
    /// Set from `--context-parallel-size`; threaded to the executor's
    /// `MultiAxisConfig` via `TpEnvGuard`.
    #[serde(default)]
    pub context_parallel_size: Option<usize>,
    /// Runtime backend selection. `None` = the first registered backend (the
    /// leaf binary's compiled-in default). Set from `--backend`; the registry
    /// dispatches by name at load time.
    #[serde(default)]
    pub backend: Option<String>,
}

fn default_dspark_sps_bias_ms() -> f32 {
    211.0
}

fn default_dspark_sps_row_ms() -> f32 {
    0.53
}

/// `--lora-alpha` default (the common rank-32 PEFT convention); a free function
/// so `#[serde(default = ...)]` can name it.
fn default_student_lora_alpha() -> f32 {
    32.0
}

/// SGLang's default static-memory fraction (0.9): 90% of VRAM for weights+KV,
/// 10% headroom for activations/scratch/fragmentation. A free function (not an
/// inline literal) so `#[serde(default = ...)]` can name it.
fn default_mem_fraction_static() -> f64 {
    0.9
}

fn default_oversubscription_min_slice() -> usize {
    infer_core::DEFAULT_OVERSUBSCRIPTION_MIN_SLICE
}

impl Default for EngineLoadConfig {
    fn default() -> Self {
        // Conservative local-serving defaults shared by every backend builder.
        Self {
            // Auto-budget ceiling, NOT a concurrency cap: the executor clamps
            // this to what post-weights VRAM affords (each backend's
            // `kv_budget_plan`), and `max_running_requests` is the
            // user-facing concurrency knob — when set it replaces this ceiling
            // as the executor slot budget (`hot_workspace_slots`; post-#154-3b
            // slots trade against comp-pool tokens, so "VRAM budget always
            // binds first" no longer holds). `--num-slots` was removed; the
            // old default of 4 lingered as a hard 4-slot cap that starved
            // concurrency regardless of VRAM.
            num_slots: 256,
            max_running_requests: None,
            total_pages: 8192,
            page_size: 16,
            // Sentinel: unset → bound by KV capacity in `scheduler_config` (#145).
            max_prompt_tokens: usize::MAX,
            max_total_tokens: 65_536,
            chunked_prefill_size: None,
            mtp_draft_tokens: None,
            mtp_draft_topk: None,
            kv_cache_dtype: KvCacheDtype::Auto,
            memory_budget_bytes: None,
            system_reserve_bytes: None,
            allow_swap: false,
            low_impact: false,
            max_thinking_tokens: 0,
            mem_fraction_static: default_mem_fraction_static(),
            kv_dram: KvTierBudget::default(),
            kv_ssd_root: None,
            kv_disk_limit: None,
            temperature: None,
            repetition_penalty: None,
            slot_oversubscription: false,
            oversubscription_min_slice: default_oversubscription_min_slice(),
            student_lora_adapters: None,
            student_lora_alpha: default_student_lora_alpha(),
            dspark_draft_model: None,
            dspark_sps_bias_ms: default_dspark_sps_bias_ms(),
            dspark_sps_row_ms: default_dspark_sps_row_ms(),
            markov_head_rank: None,
            dspark_block_size: None,
            cuda: infer_seam::CudaRuntimeFlags::default(),
            metal: infer_seam::MetalRuntimeFlags::default(),
            diffusion_max_denoising_steps: None,
            vulkan_submit_cap: None,
            world_size: None,
            context_parallel_size: None,
            backend: None,
        }
    }
}

impl EngineLoadConfig {
    // A set `--max-running-requests` IS the executor slot budget: the scheduler
    // runs at most `cap` requests, and post-#154-3b DSv4 slots TRADE against
    // shared comp-pool tokens (each ~338MB), so provisioning `num_slots` slots
    // for a capped scheduler reserves VRAM no request can ever use. Unset, the
    // `num_slots` auto-ceiling applies and the VRAM budget binds.
    pub fn hot_workspace_slots(&self) -> usize {
        self.max_running_requests.unwrap_or(self.num_slots).max(1)
    }

    pub fn mtp_enabled(&self) -> bool {
        self.mtp_draft_tokens.is_some() || self.mtp_draft_topk.is_some()
    }

    /// Single-slot, full-context teacher-forcing load: one sequence, no batching,
    /// page_size 16, static KV reservation sized to `seq`. Shared by the OPD
    /// teacher/student loaders and the PPL harness. Struct-update over this for
    /// the few sites that also carry dspark draft fields.
    pub fn single_sequence(seq: usize) -> Self {
        Self {
            num_slots: 1,
            page_size: 16,
            total_pages: seq.div_ceil(16),
            max_prompt_tokens: seq,
            max_total_tokens: seq,
            chunked_prefill_size: Some(seq),
            ..Self::default()
        }
    }

    pub fn kv_ssd_requested(&self) -> bool {
        self.kv_ssd_root.is_some() || self.kv_disk_limit.is_some()
    }

    /// `default_budget(root, fraction)` probes free disk; `world` divides the
    /// deployment-total cap into per-rank shares.
    pub fn kv_ssd_spill(
        &self,
        world: usize,
        default_budget: impl FnOnce(&std::path::Path, f64) -> usize,
    ) -> anyhow::Result<Option<(std::path::PathBuf, usize)>> {
        let world = world.max(1);
        match (&self.kv_ssd_root, self.kv_disk_limit) {
            (Some(root), limit) => {
                let total = match limit {
                    // #158: derived 0 (disk under the reserve floor) degrades to
                    // no-tier; an explicit --kv-disk-limit still fails loudly.
                    None => {
                        let budget = default_budget(root, 0.5);
                        if budget == 0 {
                            log::warn!(
                                "--kv-disk {}: derived budget is 0 (no free disk \
                                 space) — disabling the KV disk tier; pass \
                                 --kv-disk-limit to force a budget",
                                root.display()
                            );
                            return Ok(None);
                        }
                        budget
                    }
                    Some(KvTierBudget::Fraction(f)) => {
                        anyhow::ensure!(
                            f > 0.0 && f <= 1.0,
                            "--kv-disk-limit fraction must be in (0, 1]"
                        );
                        default_budget(root, f)
                    }
                    Some(KvTierBudget::Bytes(b)) => {
                        anyhow::ensure!(b > 0, "--kv-disk-limit must be positive");
                        b
                    }
                    Some(KvTierBudget::Off) => {
                        anyhow::bail!("--kv-disk-limit off is meaningless; omit --kv-disk instead")
                    }
                };
                Ok(Some((root.clone(), total / world)))
            }
            (None, Some(_)) => anyhow::bail!("--kv-disk-limit requires --kv-disk"),
            (None, None) => Ok(None),
        }
    }

    pub fn scheduler_config(&self) -> infer_core::SchedulerConfig {
        let mut config = infer_core::SchedulerConfig::for_slots(self.hot_workspace_slots());
        // Prompt cap = min(requested, KV capacity − gen reserve). For shared
        // KV pools (Qwen3.6/DSv4) the device pool is profiled
        // from free VRAM after load, so `total_pages` here is just the
        // advisory default (8192) — using it would cap prompts at 114k even
        // though the profiled pool holds 770k+. Bind to `max_total_tokens`
        // instead; the post-load profiled-capacity clamp (M2) binds it down
        // to the real device pool if needed.
        let per_req_cap = self
            .max_total_tokens
            .max(self.total_pages.saturating_mul(self.page_size));
        let gen_reserve = per_req_cap / 8;
        config.max_prompt_tokens = self
            .max_prompt_tokens
            .min(per_req_cap.saturating_sub(gen_reserve));
        config.max_total_tokens = self.max_total_tokens;
        // Unset → 64: the Metal-interactivity default (small ticks keep the
        // single-threaded MLX encode loop responsive between decode steps).
        // The CUDA load path re-resolves per model kind before use.
        config.chunked_prefill_size = self.chunked_prefill_size.unwrap_or(64);
        config.max_running_requests = self.max_running_requests;
        config.slot_oversubscription = self.slot_oversubscription;
        config.oversubscription_min_slice = self.oversubscription_min_slice.max(1);
        // Diagnostic-only escape hatch (not a shipped feature) for the
        // concurrent-decode digit-corruption investigation — see
        // docs/experience/errors/2026-07-06-dsv4-concurrent-decode-digit-corruption-unresolved.md.
        if std::env::var("ARLE_DISABLE_PREFIX_CACHE").is_ok() {
            config.enable_prefix_cache = false;
        }
        config
    }
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CudaModelKind {
    /// Dense Qwen3 (`model_type=qwen3`, `Qwen3ForCausalLM`): no CUDA path.
    DenseQwen3Unsupported,
    Qwen35,
    /// DeepSeek-V4-Flash (multi-GPU only).
    Dsv4,
    /// Vanilla public Qwen3-MoE (`model_type=qwen3_moe`,
    /// `Qwen3MoeForCausalLM`) — NOT ARLE's Qwen3.5/3.6 (`qwen3_5*`). The
    /// qwen35 CUDA loader is hardwired for gated-attn + `model.language_model`
    /// prefix + shared-expert, none of which vanilla Qwen3-MoE has, so it can
    /// never load on CUDA today. Classified distinctly so the load path
    /// fails fast with an actionable message instead of an opaque serde error.
    Qwen3MoeUnsupported,
}

/// Pure classification of a parsed `config.json`: DeepSeek-V4 by
/// `model_type`/`architectures`, else MoE if an expert count or `*Moe*`
/// architecture is present, else dense Qwen3 (unsupported). Kept
/// dependency-light + testable.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn classify_cuda_model(v: &serde_json::Value) -> CudaModelKind {
    let model_type = v.get("model_type").and_then(|x| x.as_str()).unwrap_or("");
    let arch_contains = |needle: &str| {
        v.get("architectures")
            .and_then(|a| a.as_array())
            .is_some_and(|a| {
                a.iter()
                    .any(|s| s.as_str().is_some_and(|s| s.contains(needle)))
            })
    };
    if model_type == "deepseek_v4" || arch_contains("DeepseekV4") {
        return CudaModelKind::Dsv4;
    }
    // GLM-5.2 (glm_moe_dsa) is the DeepSeek-V3.2-DSA family — it rides the DSv4
    // V32 path (adapter in `dsv4.rs` resolves the config dialect). Must precede
    // the expert-count→Qwen35 branch below, else GLM's 256 experts misroute.
    if model_type == "glm_moe_dsa" || arch_contains("GlmMoeDsa") {
        return CudaModelKind::Dsv4;
    }
    // Vanilla public Qwen3-MoE (`model_type=qwen3_moe`) is a different schema
    // and a different forward from ARLE's Qwen3.5/3.6 (`qwen3_5*`): no gated
    // attn, no `model.language_model` prefix, no shared expert. The qwen35
    // CUDA loader cannot load it, so fail fast here rather than fall through to
    // the MoE→Qwen35 branch (which then errors with an opaque serde mismatch).
    // Key on `model_type`, NOT the architecture string: Qwen3.6 ships
    // `model_type=qwen3_5` but a bare `Qwen3MoeForCausalLM` architecture, so
    // an arch-based guard would misroute it.
    if model_type == "qwen3_moe" {
        return CudaModelKind::Qwen3MoeUnsupported;
    }
    let expert_count = |key: &str| v.get(key).and_then(|x| x.as_u64()).unwrap_or(0);
    let is_qwen35 = matches!(model_type, "qwen3_5" | "qwen3_5_moe") || arch_contains("Qwen3_5");
    let is_moe = arch_contains("Moe")
        || expert_count("num_experts") > 0
        || expert_count("n_routed_experts") > 0;
    if is_qwen35 || is_moe {
        CudaModelKind::Qwen35
    } else {
        CudaModelKind::DenseQwen3Unsupported
    }
}

/// Read and parse `<model_path>/config.json`. Callers probing for optional keys
/// treat `Err` as absent; callers that classify the checkpoint (the CUDA
/// model-kind path) keep the error context.
pub(crate) fn read_config_json(model_path: &str) -> anyhow::Result<serde_json::Value> {
    use anyhow::Context;
    let cfg_path = std::path::Path::new(model_path).join("config.json");
    let raw = std::fs::read_to_string(&cfg_path)
        .with_context(|| format!("read {}", cfg_path.display()))?;
    serde_json::from_str(&raw).context("parse config.json")
}

// OPD API-teacher raw-logits HTTP surface (CUDA-only; merged into the cuda router).
#[cfg(feature = "cuda")]
#[path = "loaded/raw_logits_route.rs"]
mod raw_logits_route;

/// Serve-handle builder, registered by the leaf binary for each compiled-in
/// backend. The builder resolves the model path, constructs the backend
/// executor + KV pool, and spawns the serve handle. Backend selection is a
/// runtime decision: the leaf binary registers one builder per backend, and
/// [`LoadedInferenceEngine::load_with_config`] dispatches by name.
pub type BackendBuilderFn = fn(
    model_path: &str,
    config: &EngineLoadConfig,
    shutdown: infer_server::ServeShutdown,
) -> anyhow::Result<(infer_server::ServeHandle, infer_server::OpenAiTokenizer, String)>;

struct BackendEntry {
    name: &'static str,
    build: BackendBuilderFn,
}

static REGISTRY: std::sync::OnceLock<std::sync::RwLock<Vec<BackendEntry>>> =
    std::sync::OnceLock::new();

/// Register a backend builder. Called by the leaf binary at startup, before
/// any engine load. Duplicate names are ignored (first registration wins).
pub fn register_backend(name: &'static str, build: BackendBuilderFn) {
    let registry = REGISTRY.get_or_init(|| std::sync::RwLock::new(Vec::new()));
    let mut guard = registry.write().unwrap();
    if !guard.iter().any(|e| e.name == name) {
        guard.push(BackendEntry { name, build });
    }
}

pub(crate) fn lookup_backend(name: &str) -> Option<BackendBuilderFn> {
    let registry = REGISTRY.get()?;
    let guard = registry.read().unwrap();
    guard.iter().find(|e| e.name == name).map(|e| e.build)
}

/// The first registered backend name, or `"cpu"` when the registry is empty.
/// Callers resolving a `--backend auto` / unset [`EngineLoadConfig::backend`]
/// use this so CLI-level backend gates see the effective backend.
#[must_use]
pub fn default_backend() -> String {
    let registry = REGISTRY.get_or_init(|| std::sync::RwLock::new(Vec::new()));
    let guard = registry.read().unwrap();
    guard
        .first()
        .map(|e| e.name.to_string())
        .unwrap_or_else(|| "cpu".to_string())
}

/// Whether a backend with this name is registered. CLI `--backend` validation
/// uses this to fail early before loading the model.
#[must_use]
pub fn is_backend_registered(name: &str) -> bool {
    lookup_backend(name).is_some()
}

/// The public engine handle: one loaded [`ServeInferenceEngine`] plus the
/// runtime backend name. Backend-neutral; the CUDA-only OPD methods below
/// downcast the engine-thread executor via the serve control seam.
pub struct LoadedInferenceEngine {
    engine: ServeInferenceEngine,
    backend: String,
}

impl LoadedInferenceEngine {
    /// Single-user load (REPL, OCR): caps slots at 1 so the GDR recurrent
    /// state doesn't reserve `num_slots`× per-slot bytes (12 GiB for a 9B
    /// model at the default 256 slots). Multi-request serving uses
    /// `load_with_config` with the serve-derived slot budget.
    pub fn load(model_path: &str) -> Result<Self> {
        let config = EngineLoadConfig {
            max_running_requests: Some(1),
            ..EngineLoadConfig::default()
        };
        Self::load_with_config(model_path, config)
    }

    pub fn load_with_config(model_path: &str, config: EngineLoadConfig) -> Result<Self> {
        // Model-driven serve defaults for omitted sampling fields (nucleus +
        // temperature). The cc rollout lane overrides `.temperature` after this.
        infer_server::set_sampling_defaults(
            infer_server::SamplingDefaults::from_generation_config(model_path)
                .with_overrides(config.temperature, config.repetition_penalty),
        );
        let backend = config.backend.clone().unwrap_or_else(default_backend);
        let build = lookup_backend(&backend).ok_or_else(|| {
            anyhow::anyhow!("backend '{backend}' not registered; compiled-in backends only")
        })?;
        let (serve, tokenizer, model_id) =
            build(model_path, &config, infer_server::ServeShutdown::new())?;
        Ok(Self {
            engine: ServeInferenceEngine::new(model_id, tokenizer, serve),
            backend,
        })
    }

    #[must_use]
    pub fn backend_name(&self) -> &str {
        &self.backend
    }

    /// OPD-teacher raw-logits forward: run the full `[seq_len, vocab]` teacher
    /// forward over `(input_ids, positions)` (no sampling) and return the
    /// device logits. CUDA-only; the downcast inside `ServeInferenceEngine`
    /// rejects any other executor.
    #[cfg(feature = "cuda")]
    pub fn forward_token_logits(
        &self,
        input_ids: &[u32],
        positions: &[u32],
    ) -> Result<crate::types::RawLogits> {
        self.engine.forward_token_logits(input_ids, positions)
    }

    /// Trunk taps at `target_layer_ids` (`[seq, taps·hidden]`) and the
    /// final-normed hidden states (`[seq, hidden]`), host f32 — what
    /// `spec_train::trainer::Target` needs per sample. CUDA-only.
    #[cfg(feature = "cuda")]
    pub fn forward_training_taps(
        &self,
        input_ids: &[u32],
        target_layer_ids: &[i64],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        self.engine
            .forward_training_taps(input_ids, target_layer_ids)
    }

    /// Hot-swap the DSpark Markov head weights from a host f32 snapshot.
    /// Called by the train sidecar after each acceptance-weighted step.
    #[cfg(feature = "cuda")]
    pub fn update_dspark_markov_weights(&self, w1: &[f32], w2: &[f32]) -> Result<()> {
        self.engine
            .update_dspark_markov_weights(w1.to_vec(), w2.to_vec())
    }

    /// Programmatic token-id generation over the serving scheduler/KV path.
    /// OPD uses this for student rollout: one submitted request owns one KV
    /// slot and decodes incrementally until `max_tokens` is reached.
    #[cfg(feature = "cuda")]
    pub fn generate_token_ids(
        &self,
        prompt_token_ids: &[u32],
        max_tokens: usize,
        sampling: infer_plan::SamplingParams,
    ) -> Result<Vec<u32>> {
        self.engine
            .generate_token_ids(prompt_token_ids, max_tokens, sampling)
    }

    /// Batched [`generate_token_ids`]: submit all `(prompt, sampling)`
    /// requests to the continuous-batching engine at once, then collect each.
    /// Used by rubric-OPD eval (16 prompts) and rollout (N samples) to keep
    /// the batcher busy instead of decoding one request at a time.
    #[cfg(feature = "cuda")]
    pub fn generate_token_ids_batch(
        &self,
        requests: &[(Vec<u32>, infer_plan::SamplingParams)],
        max_tokens: usize,
    ) -> Result<Vec<Vec<u32>>> {
        self.engine.generate_token_ids_batch(requests, max_tokens)
    }

    /// Batched text completion: submit all `CompletionRequest`s to the
    /// continuous-batching engine at once, then collect each. Used by
    /// rubric-OPD judging to decode N rollout verdicts of the same problem
    /// concurrently instead of one verdict at a time.
    #[cfg(feature = "cuda")]
    pub fn complete_batch(&self, reqs: Vec<CompletionRequest>) -> Result<Vec<CompletionOutput>> {
        self.engine.complete_batch(reqs)
    }

    /// Offload the engine's device weights to host RAM (OPD teacher weight
    /// time-share), returning the device bytes freed. Threads to the backend
    /// executor on the engine thread; backends without device-weight offload
    /// reject it there.
    pub fn offload_engine_weights(&self) -> Result<usize> {
        self.engine.offload_engine_weights()
    }

    /// Quiesce engine admission (the serve loop defers new admission) and
    /// cancel every in-flight (waiting + active) request, returning how many
    /// were cancelled. The OPD round-loop writeback bracket; pairs with
    /// [`Self::ensure_kv_pool_and_resume_admissions`] after the KV pool is re-acquired.
    pub fn quiesce_admissions(&self) -> Result<usize> {
        self.engine.quiesce_admissions()
    }

    /// Re-acquire the KV pool, then resume admission only after success.
    pub fn ensure_kv_pool_and_resume_admissions(&self) -> Result<()> {
        self.engine.ensure_kv_pool_and_resume_admissions()
    }

    /// Reload the engine's device weights from the host snapshot (OPD teacher
    /// weight time-share).
    pub fn reload_engine_weights(&self) -> Result<()> {
        self.engine.reload_engine_weights()
    }

    /// Release the engine's inference forward scratch WITHOUT offloading weights
    /// or evicting KV (OPD rollout->writeback VRAM reclaim). A no-op on
    /// backends with no forward scratch to release.
    pub fn release_inference_scratch(&self) -> Result<()> {
        self.engine.release_inference_scratch()
    }

    /// Drop the engine's KV pool WITHOUT offloading weights (OPD writeback
    /// headroom: the writeback's fresh autograd forward never reads this
    /// engine's KV). A no-op on backends without a droppable pool.
    pub fn release_kv_pool(&self) -> Result<()> {
        self.engine.release_kv_pool()
    }

    /// Re-acquire the KV pool dropped by [`Self::release_kv_pool`] before the
    /// next rollout. A no-op on backends without a droppable pool.
    pub fn ensure_kv_pool(&self) -> Result<()> {
        self.engine.ensure_kv_pool()
    }

    /// Fold a fresh student LoRA update into the resident Qwen3.5/3.6
    /// projection weights (OPD per-step re-merge). CUDA-only: the downcast
    /// inside `ServeInferenceEngine` rejects any other executor.
    ///
    /// The CUDA forward path implements the merge (see
    /// [`infer_cuda::CudaExecutor::remerge_student_lora`] +
    /// `infer_cuda::qwen35::Qwen35Model::remerge_student_lora`): resident
    /// `DeviceMatrix` weights are re-merged in place from a pristine
    /// base-weight cache, and the next forward picks them up. The executor
    /// lives on the [`infer_server::ServeHandle`] engine thread; this routes
    /// the merge through the out-of-band `run_on_executor` control seam (the
    /// same seam the raw-logits forward + weight offload/reload use), so it
    /// runs between scheduler steps with exclusive `&mut E` access. Takes
    /// `&self` (interior mutability via the control channel) so the train OPD
    /// loop can call it on a shared `MutexGuard` binding.
    #[cfg(feature = "cuda")]
    pub fn remerge_student_lora(&self, update: infer_cuda::StudentLoraUpdate) -> Result<()> {
        self.engine.remerge_student_lora(update)
    }

    /// Read-only borrow of resident FP8 block-scaled base projection
    /// pointers for train-infer weight sharing (`--share-frozen-base`).
    /// CUDA-only: only the Qwen3.5/3.6 hybrid student carries shareable FP8
    /// base weights. Returns the pointer table (raw device `u64`s + dims);
    /// the train loader imports a NON-OWNING view over these instead of
    /// allocating its own copy of the shared frozen base.
    #[cfg(feature = "cuda")]
    pub fn frozen_base_fp4_pointers(&self) -> Result<Vec<infer_cuda::SharedFp4BaseProjection>> {
        self.engine.frozen_base_fp4_pointers()
    }

    #[cfg(feature = "cuda")]
    pub fn frozen_base_fp8_pointers(&self) -> Result<Vec<infer_cuda::SharedFp8BaseProjection>> {
        self.engine.frozen_base_fp8_pointers()
    }

    /// Non-owning views of every resident dense-BF16 base projection's
    /// device pointer, for refreshing the train student's frozen base AFTER
    /// a LoRA re-merge.
    #[cfg(feature = "cuda")]
    pub fn frozen_base_bf16_pointers(
        &self,
    ) -> Result<Vec<infer_cuda::SharedBf16BaseProjection>> {
        self.engine.frozen_base_bf16_pointers()
    }

    /// OpenAI-compat HTTP router over this ALREADY-loaded engine's
    /// `ServeHandle` (same engine thread, same KV pool) — unlike
    /// `router_for_backend`, which spawns a second engine. Serve it with
    /// [`crate::serve_router_on_thread`].
    #[cfg(feature = "cuda")]
    pub fn local_router(&self, max_thinking_tokens: usize) -> Result<axum::Router> {
        Ok(infer_server::coordinator_local_router(
            self.engine.serve_arc(),
            self.engine.tokenizer().clone(),
            self.engine.model_id().to_string(),
            max_thinking_tokens,
            None,
        ))
    }
}

/// Mirrors [`LoadedInferenceEngine::load_with_config`] but returns the bare
/// [`axum::Router`] the in-process [`crate::serve_http`] loop binds, rather
/// than the [`InferenceEngine`] adapter the agent/OPD callers use. Spawns the
/// same [`infer_server::ServeHandle`] the matching builder spawns, then hands
/// it to the backend-neutral [`infer_server::coordinator_local_router`].
pub(crate) fn router_for_backend(
    model_path: &str,
    config: EngineLoadConfig,
    shutdown: infer_server::ServeShutdown,
) -> Result<(axum::Router, Option<Arc<LoadedInferenceEngine>>)> {
    // Model-driven serve defaults for omitted sampling fields (nucleus +
    // temperature) — the `arle serve` router lane. CLI flags win over the
    // checkpoint.
    infer_server::set_sampling_defaults(
        infer_server::SamplingDefaults::from_generation_config(model_path)
            .with_overrides(config.temperature, config.repetition_penalty),
    );
    let backend = config.backend.clone().unwrap_or_else(default_backend);
    let build = lookup_backend(&backend).ok_or_else(|| {
        anyhow::anyhow!("backend '{backend}' not registered; compiled-in backends only")
    })?;
    let (serve, tokenizer, model_id) = build(model_path, &config, shutdown)?;
    let serve_engine = ServeInferenceEngine::new(model_id.clone(), tokenizer.clone(), serve);
    let serve_arc = serve_engine.serve_arc();
    // VLM models (DeepSeek-OCR) need the multimodal kind to wire the HTTP
    // endpoint; probe the executor once rather than threading model-specific
    // detection through the backend-neutral builder.
    let multimodal_kind = serve_arc
        .run_on_executor(|executor| executor.multimodal().and_then(|mm| mm.multimodal_kind()))
        .unwrap_or(None);
    let engine = Arc::new(LoadedInferenceEngine {
        engine: serve_engine,
        backend: backend.clone(),
    });
    let router = infer_server::coordinator_local_router(
        serve_arc,
        tokenizer,
        model_id,
        config.max_thinking_tokens,
        multimodal_kind,
    );
    // The OPD API-teacher raw-logits route is CUDA-only; merge it only there.
    #[cfg(feature = "cuda")]
    let router = router.merge(raw_logits_route::raw_logits_router(engine.clone()));
    Ok((router, Some(engine)))
}

impl InferenceEngine for LoadedInferenceEngine {
    fn model_id(&self) -> &str {
        self.engine.model_id()
    }

    fn complete(&mut self, req: CompletionRequest) -> Result<CompletionOutput> {
        self.engine.complete(req)
    }

    fn complete_multimodal_chat(
        &mut self,
        req: MultimodalChatRequest,
    ) -> Result<CompletionOutput> {
        self.engine.complete_multimodal_chat(req)
    }

    fn complete_stream(
        &mut self,
        req: CompletionRequest,
        tx: UnboundedSender<CompletionStreamDelta>,
    ) -> Result<()> {
        self.engine.complete_stream(req, tx)
    }

    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        self.engine.tokenize(text)
    }

    fn render_chat_prompt(&self, messages: &[ChatPromptMessage]) -> Result<String> {
        self.engine.render_chat_prompt(messages)
    }

    fn telemetry(&self) -> EngineTelemetry {
        self.engine.telemetry()
    }

    fn supports_multimodal_chat(&self) -> bool {
        self.engine.supports_multimodal_chat()
    }
}

/// CUDA-only engine construction: the single-GPU serve builder, the multiproc
/// worker engine, and the model-kind classifier + TP env helpers they share.
/// Everything above stays backend-neutral; this module is the one place
/// infer-api names CUDA executor types.
#[cfg(feature = "cuda")]
mod cuda {
    use anyhow::Result;

    use super::EngineLoadConfig;
    use super::{classify_cuda_model, read_config_json, CudaModelKind};
    use infer_cuda::{CudaExecutor, CudaKvPool};
    use infer_seam::BackendExecutor;

    fn detect_cuda_model_kind(model_path: &str) -> Result<CudaModelKind> {
        Ok(classify_cuda_model(&read_config_json(model_path)?))
    }

    /// Whether `model_path`'s checkpoint takes the multiproc TP serve path when
    /// the env world size > 1: DSv4 (multi-GPU by design) and the Qwen3.5/3.6
    /// MoE hybrid (a `TpRuntime` consumer since the TP port; its executor
    /// resolves rank/world + the NCCL communicator from env at load, exactly
    /// like DSv4). Dense Qwen3 stays single-process. `false` on any
    /// config-read/parse failure (the single-process path then errors with its
    /// normal message).
    #[must_use]
    pub fn cuda_model_takes_multiproc_serve(model_path: &str) -> bool {
        matches!(
            detect_cuda_model_kind(model_path),
            Ok(CudaModelKind::Dsv4 | CudaModelKind::Qwen35)
        )
    }

    /// Admission page-pool capacity, derived uniformly for every model — one rule,
    /// each backend declares its KV token-capacity. The scheduler gates admission on
    /// `pages_needed = (prompt_len + max_tokens) / page_size` (a full-attention
    /// estimate, infer-core `prefix.rs`), so the host `CudaKvPool` must cover the
    /// backend's actual KV capacity or a long prompt is falsely rejected → it sits
    /// in `waiting` → `is_idle()` is never true → the engine spins `while !is_idle()`
    /// (100% CPU, GPU 0%). Three regimes:
    ///   - Qwen3-Dense: SHARED paged pool — the executor allocates one device pool
    ///     and host page ids mirror it 1:1, so admission MUST equal the device pool's
    ///     ACTUAL page count (host total == device total is load-bearing: the device
    ///     pool consumes host page ids directly). With the §3 profiler the device
    ///     pool is sized from MEASURED free VRAM, NOT `config.total_pages`, so the
    ///     executor reports its effective page count via `effective_total_pages()`
    ///     and this passes it in as `dense_total_pages` (the hardcoded
    ///     `config.total_pages` is the request, not the truth).
    ///   - Qwen3.6-MoE: SHARED paged pool (since the shared-paged migration) —
    ///     the executor allocates ONE profile-sized full-attn `PagedKVPool` and
    ///     host page ids mirror it 1:1, exactly like dense. Admission MUST equal
    ///     the device pool's ACTUAL page count (`effective_total_pages()`), not
    ///     `num_slots × total_pages`. The linear-attn recurrent state stays
    ///     per-slot but is not page-addressable, so it never touches the host
    ///     KV pool.
    ///   - DSv4: SHARED MLA latent pool — the executor profiles the pool TOTAL
    ///     from measured free VRAM (`profile_kv_pool_tokens`) and derives per-slot
    ///     length as total/num_slots, exactly like dense. Admission MUST equal
    ///     the device pool's ACTUAL page count (`effective_total_pages()`), not
    ///     the old `num_slots × 32768`. The host pool ALSO mirrors the device page
    ///     SIZE (64-tok `page_block_size`, via `effective_page_size()`), not
    ///     `config.page_size` (16) — H3: page_size mismatch gated host admission
    ///     at 1/4 device token capacity → early-OOM with no tier to evict.
    ///
    /// `CudaKvPool::new` allocates NO HBM (just a `Vec<u32>` of page ids).
    ///
    /// `paged_pool_pages` is the paged-pool executor's ACTUAL device pool page
    /// count (profiled from free VRAM; dense + Qwen3.6 + DSv4). For those branches the
    /// host pool mirrors it exactly — never floored back up to the requested
    /// `config.total_pages`, because a profiled pool may legitimately be SMALLER
    /// (big weights / small card) and a host pool larger than the device pool
    /// hands out page ids the device pool has no HBM for. Other kinds ignore it.
    fn cuda_admission_total_pages(
        kind: CudaModelKind,
        config: &EngineLoadConfig,
        page_size: usize,
        paged_pool_pages: usize,
    ) -> usize {
        let ps = page_size.max(1);
        // Paged-pool models (Qwen3.6 + DSv4 MLA latent pool): the host
        // admission pool is exactly the device pool — the profiled page count,
        // NOT a token-derived re-ceiling, and NOT floored at the requested
        // config value. DSv4's MLA pool is now free-VRAM-sized like the others.
        if matches!(kind, CudaModelKind::Qwen35 | CudaModelKind::Dsv4) {
            return paged_pool_pages.max(1);
        }
        // Qwen3MoeUnsupported | DenseQwen3Unsupported
        let capacity_tokens = config.total_pages.saturating_mul(ps);
        capacity_tokens.div_ceil(ps).max(config.total_pages)
    }

    /// TP world size fallback when `config.world_size` is `None`:
    /// `INFER_TP_SIZE`, else the `INFER_CUDA_DEVICES` count, else 1. Every rank
    /// sees identical env, so budget division is rank-invariant.
    fn tp_world_size() -> usize {
        if let Some(n) = std::env::var("INFER_TP_SIZE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
        {
            return n;
        }
        std::env::var("INFER_CUDA_DEVICES")
            .ok()
            .map(|list| list.split(',').filter(|s| !s.trim().is_empty()).count())
            .filter(|&n| n > 0)
            .unwrap_or(1)
    }

    /// RAII guard that temporarily overrides `INFER_TP_SIZE` and
    /// `INFER_ATTN_CP_SIZE` for the executor's `TpRuntime` / `MultiAxisConfig`
    /// resolution. Restores the previous values on drop. `None` fields leave
    /// the corresponding env var untouched.
    struct TpEnvGuard {
        saved_tp: Option<Option<String>>,
        saved_cp: Option<Option<String>>,
    }

    impl TpEnvGuard {
        fn new(world_size: Option<usize>, cp_size: Option<usize>) -> Self {
            let saved_tp = world_size.map(|ws| {
                let saved = std::env::var("INFER_TP_SIZE").ok();
                // SAFETY: single-threaded during engine build, no other readers.
                unsafe { std::env::set_var("INFER_TP_SIZE", ws.to_string()) };
                saved
            });
            let saved_cp = cp_size.map(|cp| {
                let saved = std::env::var("INFER_ATTN_CP_SIZE").ok();
                // SAFETY: single-threaded during engine build, no other readers.
                unsafe { std::env::set_var("INFER_ATTN_CP_SIZE", cp.to_string()) };
                saved
            });
            Self { saved_tp, saved_cp }
        }
    }

    impl Drop for TpEnvGuard {
        fn drop(&mut self) {
            if let Some(saved) = self.saved_tp.take() {
                // SAFETY: single-threaded during engine build, no other readers.
                unsafe {
                    match saved {
                        Some(v) => std::env::set_var("INFER_TP_SIZE", v),
                        None => std::env::remove_var("INFER_TP_SIZE"),
                    }
                }
            }
            if let Some(saved) = self.saved_cp.take() {
                // SAFETY: single-threaded during engine build, no other readers.
                unsafe {
                    match saved {
                        Some(v) => std::env::set_var("INFER_ATTN_CP_SIZE", v),
                        None => std::env::remove_var("INFER_ATTN_CP_SIZE"),
                    }
                }
            }
        }
    }

    /// Build the CUDA `Engine` (executor + admission KV pool + scheduler) for
    /// `model_path` — the ONE engine constructor every rank uses. rank 0 runs it
    /// inside [`infer_server::ServeHandle::spawn_with_engine_builder`]; multiproc
    /// worker ranks run it directly on their driver thread (`CudaWorkerEngine`).
    /// All ranks building through this same helper with the same
    /// [`EngineLoadConfig`] is a lockstep invariant: any per-rank divergence in
    /// scheduler knobs diverges the deterministic planner and deadlocks NCCL.
    pub fn build_cuda_engine(
        model_path: &str,
        config: &EngineLoadConfig,
    ) -> Result<infer_core::Engine> {
        // Single funnel for single-proc serve AND multiproc workers — flags
        // land in the statics before any CUDA context/executor exists.
        infer_cuda::apply_runtime_flags(&config.cuda);
        let kind = detect_cuda_model_kind(model_path)?;
        if matches!(kind, CudaModelKind::Qwen3MoeUnsupported) {
            anyhow::bail!(
                "vanilla Qwen3-MoE (Qwen3MoeForCausalLM) is not supported on the \
                 CUDA backend; use --backend metal"
            );
        }
        if matches!(kind, CudaModelKind::DenseQwen3Unsupported) {
            anyhow::bail!("Qwen3 dense is not supported on CUDA; use a Qwen3.5-family checkpoint");
        }
        // Resolve the requested KV dtype against the CUDA support matrix at the
        // engine boundary, mirroring the Metal path's `MetalKvCacheDtype::resolve`
        // (#68 T2). Admits BF16/INT8/FP8 (tq4 fails loud — see resolve); the
        // resolved dtype threads into the Qwen35 constructor below (#68 T3).
        let kv_dtype = infer_cuda::CudaKvCacheDtype::resolve(config.kv_cache_dtype)?;
        if kv_dtype != infer_cuda::CudaKvCacheDtype::Bf16 && !matches!(kind, CudaModelKind::Qwen35)
        {
            anyhow::bail!(
                "--kv-cache-dtype {} is not supported for {kind:?}. The flag selects the \
                 Qwen3.5/3.6 two-plane paged quant pool (separate 1-byte K and V planes \
                 with per-(token, kv_head) f32 scales). DSv4 MLA KV is already FP8-packed \
                 unconditionally at 584 B/token, a layout the vendored FlashMLA decode \
                 kernel fixes by reinterpreting the KV pointer as fp8*, so `fp8` would be \
                 a no-op and `int8` would need that kernel forked for no byte saving. \
                 Drop the flag rather than adding it.",
                kv_dtype.label()
            );
        }
        let mut scheduler = config.scheduler_config();
        // Default-not-override chunk resolution. Unset → per-kind default: on
        // CUDA a chunk is an entire engine tick plus a full launch round, so
        // the 64-token Metal-interactivity base default would pay ~32x the
        // tick/launch overhead on a 2048-token prompt (KV bytes read are
        // chunk-invariant) — Qwen kinds default 2048 (audit QW-KV-07); DSv4
        // defaults to its prefill scratch bound (`DSV4_PREFILL_QUERY_CHUNK` =
        // 4096; the forward asserts each call passes <= that many query
        // tokens, so long prompts MUST chunk — single-chunk max_seq_len both
        // trips that assert at >4096 and OOMs the M×K scratch at 900K), while
        // the planner still caps each chunk to the executor's
        // `max_prefill_chunk()` capability. An explicit value is honored,
        // clamped into the executor-safe [128, 4096] and rounded down to a
        // 128 multiple (KV page × restore-alignment grain).
        scheduler.chunked_prefill_size = match config.chunked_prefill_size {
            None if matches!(kind, CudaModelKind::Dsv4) => 4096,
            None => 2048,
            Some(v) => {
                let clamped = (v.clamp(128, 4096) / 128) * 128;
                if clamped != v {
                    log::warn!(
                        "--chunked-prefill-size {v} clamped to {clamped} \
                         ([128, 4096], rounded down to a 128 multiple)"
                    );
                }
                clamped
            }
        };
        let num_slots = config.hot_workspace_slots();
        // Before any weight loads: `num_slots` is the decode row ceiling, the
        // min-chain the most rows one prefill chunk of this config can carry.
        infer_cuda::apply_dense_gemm_row_envelope(
            num_slots,
            scheduler
                .chunked_prefill_size
                .min(scheduler.max_num_batched_tokens)
                .min(scheduler.max_prefill_tokens),
        );
        let page_size = config.page_size;
        let mtp_requested = config.mtp_enabled();
        if mtp_requested && config.dspark_draft_model.is_some() {
            anyhow::bail!(
                "--spec-type mtp and --spec-type dspark are mutually exclusive; select one drafter"
            );
        }
        if mtp_requested && !matches!(kind, CudaModelKind::Dsv4 | CudaModelKind::Qwen35) {
            anyhow::bail!(
                "--spec-type mtp / --mtp-draft-* is only wired for CUDA DSv4 and Qwen3.5/3.6 checkpoints; \
                 model kind {kind:?} would otherwise ignore the request"
            );
        }
        if config.dspark_draft_model.is_some()
            && !matches!(kind, CudaModelKind::Qwen35 | CudaModelKind::Dsv4)
        {
            anyhow::bail!(
                "--spec-type dspark is only wired for CUDA Qwen3.5/3.6 and DSv4 checkpoints; \
                 model kind {kind:?} would otherwise ignore the request"
            );
        }
        // Executors receive the CONFIGURED `total_pages` (Dense: shared device
        // pool size; Qwen3.5/3.6: per-slot token budget / page_size). The host
        // admission pool capacity is derived separately below — after the
        // executor reports its EFFECTIVE slot count (post KV-budget clamp).
        //
        // When config.tp_size is set, temporarily override INFER_TP_SIZE so the
        // executor's TpRuntime resolves to it. Restored on drop — scoped to
        // this synchronous constructor call, no global side effects.
        let _tp_guard = TpEnvGuard::new(config.world_size, config.context_parallel_size);
        let executor = match kind {
            // Qwen35 clamps `num_slots` to free HBM inside the constructor
            // (`Qwen35Model::kv_budget_plan`, the DSv4 joint solve via the
            // infer-seam budget kernel) — no longer the #60 OOM risk.
            CudaModelKind::Qwen35 => CudaExecutor::from_qwen35_safetensors(
                model_path,
                num_slots,
                config.total_pages,
                config.max_total_tokens,
                kv_dtype,
                config.mem_fraction_static,
                config.dspark_draft_model.as_deref(),
                config.dspark_sps_bias_ms,
                config.dspark_sps_row_ms,
                config.markov_head_rank,
                config.dspark_block_size,
                config.mtp_draft_tokens,
                config.memory_budget_bytes,
            )?,
            // DSv4 multi-rank serve. The DSv4 executor resolves its TP
            // rank/world-size + EP expert split + NCCL communicator from the
            // environment during construction (`INFER_TP_RANK` /
            // `INFER_TP_SIZE` / `INFER_CUDA_DEVICES`, plus
            // `INFER_NCCL_UNIQUE_ID` / `INFER_NCCL_ID_FILE` rendezvous) — set
            // by the multiproc coordinator/launcher before this runs. On a
            // single GPU (world_size==1) it loads as one rank. DSv4 owns its
            // MLA KV state inside the forward, so the host `CudaKvPool` is
            // only present to satisfy the `submit(.., &KvBatchDescriptor, &mut dyn KvSlotAccounting)`
            // signature; `max_seq_len` is `config.max_total_tokens` — the same
            // global cap `--max-total-tokens` sets for every backend (DSv4
            // multiproc auto-resolves it from the checkpoint's
            // `max_position_embeddings` when unset, `serve.rs`). No separate
            // DSv4-only knob: a slot's arena must hold prompt+generated tokens
            // up to exactly that cap, so there is nothing left to reconcile
            // between the scheduler's admission cap and the executor's arena.
            CudaModelKind::Dsv4 => CudaExecutor::from_dsv4_fp8_safetensors(
                model_path,
                num_slots,
                config.max_total_tokens,
                config.mtp_draft_tokens,
                config.mtp_draft_topk,
                config.dspark_draft_model.as_deref(),
                config.dspark_sps_bias_ms,
                config.dspark_sps_row_ms,
            )?,
            CudaModelKind::Qwen3MoeUnsupported | CudaModelKind::DenseQwen3Unsupported => {
                unreachable!("checked before CUDA executor build")
            }
        };
        let mut executor = executor;
        // Fail loud before any budget setter runs: a CUDA executor without the
        // prefix-reuse capability would silently no-op decode reuse.
        anyhow::ensure!(
            executor.prefix_reuse().is_some(),
            "the {kind:?} CUDA executor exposes no prefix-reuse capability"
        );
        // L2 budget: deployment-total → per-rank share, resolved at the ONE
        // constructor every rank runs (world size is env-identical per rank, so
        // the division is lockstep-deterministic).
        let world = config.world_size.unwrap_or_else(tp_world_size);
        let dram_rank_bytes = infer_cuda::resolve_dram_budget_bytes(config.kv_dram, world);
        executor.set_kv_tier_budget_bytes(dram_rank_bytes);
        // Opt-in L3 NVMe spill (`--kv-disk`): attached HERE so single-proc
        // rank 0 and multiproc worker ranks agree (the old post-spawn serve-layer
        // hook never reached multiproc workers → workers served with a zero-page
        // tier and every demote fell back to recompute). Must follow the budget
        // setters above: the tier-store arms rebuild their store on re-budget,
        // which would drop an earlier disk attach.
        let kv_disk = config.kv_ssd_spill(world, infer_cuda::default_t2_budget_bytes)?;
        if let Some((root, budget)) = &kv_disk {
            anyhow::ensure!(
                executor.set_kv_tier_disk(root.clone(), *budget),
                "--kv-disk: the loaded model has no KV tier store to spill \
                 (Qwen3.6/DSv4 slot tier; a budget \
                 below one page also lands here — raise --kv-disk-limit)"
            );
        }
        log::info!(
            "KV tiers: dtype={} | L1 mem_fraction_static={} | L2 {dram_rank_bytes}B/rank \
             (deployment {:?}, world {world}) | L3 {} | features: prefix{}",
            kv_dtype.label(),
            config.mem_fraction_static,
            config.kv_dram,
            match &kv_disk {
                Some((root, budget)) => format!("root={} cap {budget}B/rank", root.display()),
                None => "off".to_string(),
            },
            if config.slot_oversubscription {
                ",park"
            } else {
                ""
            },
        );
        // The DSv4 constructor may clamp slots below the request (dynamic KV
        // mem budget, NCCL min-reduced ⇒ identical on every rank). Scheduler +
        // admission pool MUST follow the effective count: admitting to a slot
        // the executor has no arena for fails at submit, and (lockstep) a
        // scheduler-visible capacity that diverged from the executor's would
        // diverge the deterministic planner.
        let num_slots = executor.effective_num_slots().unwrap_or(num_slots);
        // Paged-pool models (Qwen3.6 + DSv4 MLA pool) profile their
        // device KV pool from measured free VRAM (`profile_kv_pool_tokens`), so
        // the host admission pool must mirror that ACTUAL page count, not the
        // requested `config.total_pages`.
        let paged_pool_pages = executor
            .effective_total_pages()
            .unwrap_or(config.total_pages);
        // H3: the host pool's page_size must match the device pool's page
        // granularity. DSv4's MLA pool pages at 64 (`page_block_size`), not
        // `config.page_size` (16) — using 16 would gate host admission at 1/4 the
        // device token capacity and early-OOM
        let page_size = executor.effective_page_size().unwrap_or(page_size);
        let total_pages = cuda_admission_total_pages(kind, config, page_size, paged_pool_pages);
        if matches!(kind, CudaModelKind::Qwen35) && paged_pool_pages != config.total_pages {
            log::info!(
                "CUDA {kind:?}: profiled full-attn KV pool {paged_pool_pages} pages from measured \
                 free VRAM (requested total_pages={}, mem_fraction_static={}); host admission \
                 follows the device pool",
                config.total_pages,
                config.mem_fraction_static
            );
        }
        // M2: scheduler_config() raised max_prompt_tokens from the REQUESTED
        // total_pages; on the Qwen3.6 arm the device pool is profiled
        // from free VRAM and may be SMALLER, so bind the ingress caps DOWN to the
        // profiled pool capacity and to max_total_tokens — else a long prompt
        // clears ingress, can't draw enough pages, and silently completes empty.
        // Mirrors DSv4 (max_seq clamp) and Metal (resource-guard clamp).
        if matches!(kind, CudaModelKind::Qwen35) {
            let profiled_capacity = total_pages.saturating_mul(page_size).max(1);
            scheduler.max_total_tokens = scheduler.max_total_tokens.min(profiled_capacity);
            scheduler.max_prompt_tokens =
                scheduler.max_prompt_tokens.min(scheduler.max_total_tokens);
        } else if matches!(kind, CudaModelKind::Dsv4) {
            // scheduler_config() derived max_prompt_tokens from the DEFAULT
            // total_pages (8192·16), unrelated to DSv4's MLA arena whose real
            // per-slot capacity is max_total_tokens (the executor's max_seq_len).
            // Re-bind ingress to the arena from the requested config value so a
            // long prompt in (default-derived-cap, max_total_tokens] is not
            // wrongly rejected; still bounded by the arena.
            scheduler.max_prompt_tokens = config.max_prompt_tokens.min(scheduler.max_total_tokens);
        }
        if num_slots != scheduler.num_slots {
            log::warn!(
                "CUDA engine: executor clamped slots {} -> {num_slots}; scheduler follows",
                scheduler.num_slots
            );
            scheduler.num_slots = num_slots;
        }
        // `--lora-adapters`: fold the trained student LoRA into the resident
        // base once, pre-serving. Applied at the ONE engine constructor every
        // rank runs, so single-GPU and multiproc TP ranks agree; the engine is
        // not built yet, so no prefix cache exists to invalidate.
        if let Some(path) = &config.student_lora_adapters {
            let update =
                crate::student_lora::load_student_lora_update(path, config.student_lora_alpha)?;
            log::info!(
                "student LoRA re-merge: {} layers, rank={} alpha={} from {}",
                update.layers.len(),
                update.rank,
                update.alpha,
                path.display()
            );
            executor.remerge_student_lora(update)?;
        }
        let mut kv = CudaKvPool::new(num_slots, total_pages, page_size);
        if let Some(pages) = executor.effective_fixed_pages_per_slot() {
            kv.set_fixed_pages_per_slot(pages);
        }
        // 2D (attn_tp × cp): the host pool filters alloc/attach to this rank's
        // block-cyclic shard (logical page `i` on shard `i % cp`).
        if let Some((rank, size)) = executor.kv_shard_spec() {
            kv.set_shard(rank, size);
        }
        infer_core::Engine::with_config(Box::new(executor), Box::new(kv), scheduler)
    }

    type PendingTokens = std::rc::Rc<
        std::cell::RefCell<
            std::collections::HashMap<
                infer_core::RequestHandle,
                Vec<(u32, Option<f32>, Vec<(u32, f32)>)>,
            >,
        >,
    >;

    /// One multiproc worker rank's engine. Steps SYNCHRONOUSLY per relayed
    /// `TickAdmissions` so every rank admits at the same step index (lockstep).
    /// Rank 0 (`owns_output`) also tracks + emits completions; followers
    /// discard their TP-replicated tokens.
    pub struct CudaWorkerEngine {
        engine: infer_core::Engine,
        /// Rank 0 owns the visible output; followers skip all output bookkeeping.
        owns_output: bool,
        /// engine handle -> coordinator request_id (output owner only); removed
        /// once its terminal delta is emitted.
        tracked: std::collections::HashMap<infer_core::RequestHandle, u64>,
        /// Fed by the token observer inside `engine.step()`, drained right after
        /// by `drain_completions()` (same thread, never concurrent).
        pending: PendingTokens,
    }

    impl CudaWorkerEngine {
        /// Build the rank-R engine from rank 0's resolved config
        /// (`ARLE_WORKER_ENGINE_CONFIG`); NCCL rank/world come from env.
        pub fn load(
            model_path: &str,
            config: &EngineLoadConfig,
            owns_output: bool,
        ) -> Result<Self> {
            let mut engine = build_cuda_engine(model_path, config)?;
            let pending = PendingTokens::default();
            if owns_output {
                let pending = std::rc::Rc::clone(&pending);
                engine.set_token_observer(Box::new(move |handle, token| {
                    pending.borrow_mut().entry(handle).or_default().push((
                        token.token,
                        token.logprob,
                        token.top_logprobs.clone(),
                    ));
                }));
            }
            Ok(Self {
                engine,
                owns_output,
                tracked: std::collections::HashMap::new(),
                pending,
            })
        }

        /// Inject one relayed request. Every rank tracks `request_id` -> handle
        /// (not just the output owner) so [`Self::cancel`] can find it too.
        pub fn inject(
            &mut self,
            request_id: u64,
            prompt_tokens: Vec<u32>,
            max_tokens: usize,
            sampling: infer_plan::SamplingParams,
        ) {
            let handle = self.engine.submit_request_with_options(
                prompt_tokens,
                max_tokens,
                infer_core::RequestOptions {
                    sampling,
                    ..infer_core::RequestOptions::default()
                },
            );
            self.tracked.insert(handle, request_id);
        }

        /// Cancel a relayed request (client disconnected/timed out); no-op if
        /// unknown. Must be called with the same `request_id`, at the same
        /// lockstep tick, on every rank.
        pub fn cancel(&mut self, request_id: u64) {
            let Some(&handle) = self
                .tracked
                .iter()
                .find(|(_, rid)| **rid == request_id)
                .map(|(h, _)| h)
            else {
                return;
            };
            self.tracked.remove(&handle);
            self.engine.cancel_request(handle);
        }

        /// Followers never call `drain_completions` (which prunes for free), so
        /// without this their `tracked` map grows for the process lifetime.
        pub fn prune_finished(&mut self) {
            if self.owns_output || self.tracked.is_empty() {
                return;
            }
            let engine_idle = self.engine.is_idle();
            self.tracked
                .retain(|&handle, _| self.engine.completed(handle).is_none() && !engine_idle);
        }

        #[must_use]
        pub fn is_idle(&self) -> bool {
            self.engine.is_idle()
        }

        pub fn step(&mut self) -> Result<()> {
            self.engine.step()
        }

        pub fn prefix_cache_stats(&self) -> infer_core::PrefixCacheStats {
            self.engine.prefix_cache_stats()
        }

        pub fn throughput_stats(&self) -> infer_core::ThroughputStats {
            self.engine.throughput_stats()
        }

        pub fn kv_tier_stats(&self) -> infer_core::KvTierStats {
            self.engine.kv_tier_stats()
        }

        pub fn kv_system_metrics(&self) -> infer_core::KvSystemMetrics {
            self.engine.kv_system_metrics()
        }

        pub fn backend_stats(&self) -> infer_seam::BackendStats {
            self.engine.backend_stats()
        }

        pub fn active_count(&self) -> usize {
            self.engine.active_count()
        }

        pub fn waiting_count(&self) -> usize {
            self.engine.waiting_count()
        }

        pub fn kv_free_pages(&self) -> usize {
            self.engine.kv_free_pages()
        }

        /// Drain this tick's new tokens per tracked request (output owner only),
        /// plus a terminal delta for any that just finished or were dropped.
        /// No-op on followers.
        pub fn drain_completions(&mut self) -> Vec<(u64, infer_server::RelayCompletionDelta)> {
            if !self.owns_output || self.tracked.is_empty() {
                return Vec::new();
            }
            // Idle engine: a tracked-but-not-completed handle was dropped.
            let engine_idle = self.engine.is_idle();
            let mut out = Vec::new();
            let mut finished = Vec::new();
            for (&handle, &request_id) in &self.tracked {
                let new_tokens = self
                    .pending
                    .borrow_mut()
                    .remove(&handle)
                    .unwrap_or_default();
                let (finish, finish_reason, error) = match self.engine.completed(handle) {
                    Some(completed) => (true, completed.finish.clone(), None),
                    None if engine_idle => (
                        true,
                        None,
                        Some(format!(
                            "request dropped by engine without completing (handle={handle:?})"
                        )),
                    ),
                    None => (false, None, None),
                };
                if new_tokens.is_empty() && !finish {
                    continue; // nothing new to report this tick
                }
                // logprobs is all-or-nothing per delta (a partial vector would
                // misalign the sidecar's token↔logprob pairing downstream).
                let logprobs = new_tokens
                    .iter()
                    .map(|&(_, lp, _)| lp)
                    .collect::<Option<Vec<f32>>>()
                    .unwrap_or_default();
                // Same all-or-nothing rule for the OpenAI logprobs capture.
                let top_logprobs = if !new_tokens.is_empty()
                    && new_tokens.iter().all(|(_, _, top)| !top.is_empty())
                {
                    new_tokens.iter().map(|(_, _, top)| top.clone()).collect()
                } else {
                    Vec::new()
                };
                out.push((
                    request_id,
                    infer_server::RelayCompletionDelta {
                        text_delta: String::new(),
                        token_ids: new_tokens.into_iter().map(|(t, _, _)| t).collect(),
                        logprobs,
                        top_logprobs,
                        finish,
                        finish_reason,
                        error,
                    },
                ));
                if finish {
                    finished.push(handle);
                }
            }
            for handle in finished {
                self.tracked.remove(&handle);
                self.pending.borrow_mut().remove(&handle);
            }
            out
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda::{build_cuda_engine, cuda_model_takes_multiproc_serve, CudaWorkerEngine};
