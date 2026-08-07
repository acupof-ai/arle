//! `LoadedInferenceEngine` — the backend-dispatching public engine.
//!
//! A feature-gated enum over the available backends (`metal`/`cuda`/`hip`/`vulkan`/`cpu`,
//! selected at compile time) with a `load(model_path, enable_cuda_graph)`
//! constructor dispatching to the active variant. [`EngineLoadConfig`] is always
//! available; the enum + impls require a backend feature.

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
    /// Physical KV pages.
    pub total_pages: usize,
    /// Tokens per KV page.
    pub page_size: usize,
    /// Max prompt tokens accepted at ingress.
    pub max_prompt_tokens: usize,
    /// Max prompt+generated tokens for one request.
    pub max_total_tokens: usize,
    /// Per-request prefill chunk size; `None` = backend/model-kind default.
    #[serde(default)]
    pub chunked_prefill_size: Option<usize>,
    /// Token budget for one scheduler tick — the M dimension of every GEMM in
    /// the step, so it is the throughput/latency dial: above the roofline ridge
    /// more tokens buy no throughput and cost step latency linearly, and any
    /// decode row sharing the tick waits that long. `None` = the shipped
    /// constant.
    #[serde(default)]
    pub max_num_batched_tokens: Option<usize>,
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
    /// Opt into session KV-recall ("infinite memory"). Metal-only, bf16-KV only;
    /// default off → baseline byte-identical. The Metal builder turns it on via
    /// `MetalExecutor::set_kv_recall`; other backends ignore it.
    #[serde(default)]
    pub kv_recall: bool,
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
    /// sizer. Wired for the dense Qwen3 CUDA pool (`profile_kv_pool_tokens`);
    /// Qwen3.5/3.6 and DSv4 keep their per-slot sizing this phase.
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
    /// Opt-in running-cap oversubscription: rotate waiters in by parking the
    /// longest-running decode's whole-slot image (requires a whole-slot tier
    /// backend). Default false → byte-identical.
    #[serde(default)]
    pub slot_oversubscription: bool,
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

impl Default for EngineLoadConfig {
    fn default() -> Self {
        // Conservative local-serving defaults shared by every backend builder.
        Self {
            // Auto-budget ceiling, NOT a concurrency cap: the executor clamps
            // this to what post-weights VRAM affords (`kv_budget_plan` /
            // `kv_budget_num_slots`), and `max_running_requests` is the
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
            max_num_batched_tokens: None,
            mtp_draft_tokens: None,
            mtp_draft_topk: None,
            kv_cache_dtype: KvCacheDtype::Auto,
            memory_budget_bytes: None,
            system_reserve_bytes: None,
            allow_swap: false,
            low_impact: false,
            kv_recall: false,
            max_thinking_tokens: 0,
            mem_fraction_static: default_mem_fraction_static(),
            kv_dram: KvTierBudget::default(),
            kv_ssd_root: None,
            kv_disk_limit: None,
            slot_oversubscription: false,
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
        }
    }
}

impl EngineLoadConfig {
    // All callsites are under cfg(feature = "cuda"/"metal"/"hip"/"vulkan");
    // the cpu-only CI surface compiles none of them.
    // A set `--max-running-requests` IS the executor slot budget: the scheduler
    // runs at most `cap` requests, and post-#154-3b DSv4 slots TRADE against
    // shared comp-pool tokens (each ~338MB), so provisioning `num_slots` slots
    // for a capped scheduler reserves VRAM no request can ever use. Unset, the
    // `num_slots` auto-ceiling applies and the VRAM budget binds.
    #[allow(dead_code)]
    fn hot_workspace_slots(&self) -> usize {
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

    /// Whether an L3 (NVMe) KV spill was requested at all.
    #[allow(dead_code)]
    fn kv_ssd_requested(&self) -> bool {
        self.kv_ssd_root.is_some() || self.kv_disk_limit.is_some()
    }

    /// The resolved per-rank L3 spill request: `Some((root, budget_bytes))`.
    /// `default_budget(root, fraction)` probes free disk; `world` divides the
    /// deployment-total cap into per-rank shares.
    #[allow(dead_code)]
    fn kv_ssd_spill(
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
                                "--kv-disk {}: derived budget is 0 (free space is \
                                 below the reserve, max(50 GiB, 10% of disk)) — \
                                 disabling the KV disk tier; pass --kv-disk-limit \
                                 to force a budget",
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
}

/// Which CUDA forward a checkpoint needs, classified from its `config.json`.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CudaModelKind {
    /// Dense Qwen3 (BF16).
    Qwen3Dense,
    /// Qwen3.5 / 3.6 hybrid dense-or-MoE (BF16).
    Qwen35,
    /// DeepSeek-V4-Flash (multi-GPU only).
    Dsv4,
    /// DiffusionGemma/Gemma4 block-diffusion checkpoint. Not a CUDA AR path.
    DiffusionGemma,
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
/// architecture is present, else dense Qwen3. Kept dependency-light + testable.
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
    if model_type == "diffusion_gemma"
        || model_type == "gemma4"
        || arch_contains("DiffusionGemma")
        || arch_contains("Gemma4")
    {
        return CudaModelKind::DiffusionGemma;
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
        CudaModelKind::Qwen3Dense
    }
}

// OPD API-teacher raw-logits HTTP surface (CUDA-only; merged into router_cuda).
#[cfg(feature = "cuda")]
#[path = "loaded/raw_logits_route.rs"]
mod raw_logits_route;

#[cfg(test)]
mod hot_workspace_tests {
    use super::EngineLoadConfig;

    #[test]
    fn max_running_requests_caps_executor_slot_budget() {
        let mut config = EngineLoadConfig::default();
        assert_eq!(config.hot_workspace_slots(), config.num_slots);
        config.max_running_requests = Some(32);
        assert_eq!(config.hot_workspace_slots(), 32);
        // Cap above the auto-ceiling still grows the workspace: the knob is
        // the intended concurrency, not a clamp.
        config.max_running_requests = Some(config.num_slots + 1);
        assert_eq!(config.hot_workspace_slots(), config.num_slots + 1);
        config.max_running_requests = Some(0);
        assert_eq!(config.hot_workspace_slots(), 1);
    }
}

#[cfg(test)]
mod classify_tests {
    use super::{CudaModelKind, classify_cuda_model};
    use serde_json::json;

    #[test]
    fn classifies_cuda_checkpoints_from_config() {
        assert_eq!(
            classify_cuda_model(
                &json!({"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"})
            ),
            CudaModelKind::Qwen3Dense
        );
        // Bare `Qwen3MoeForCausalLM` architecture with NO `model_type` is how
        // Qwen3.6 ships its arch field; the qwen35 loader handles it, so it must
        // stay Qwen35. The vanilla-Qwen3-MoE guard keys on `model_type`, not the
        // architecture string, precisely so this case is unaffected.
        assert_eq!(
            classify_cuda_model(&json!({"architectures": ["Qwen3MoeForCausalLM"]})),
            CudaModelKind::Qwen35
        );
        // Vanilla public Qwen3-MoE: same architecture string BUT
        // `model_type=qwen3_moe`. Must classify as the unsupported kind (the
        // load path turns this into a clear `--backend metal` bail), NOT Qwen35.
        assert_eq!(
            classify_cuda_model(
                &json!({"architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe", "num_experts": 128})
            ),
            CudaModelKind::Qwen3MoeUnsupported
        );
        assert_eq!(
            classify_cuda_model(
                &json!({"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5"})
            ),
            CudaModelKind::Qwen35
        );
        assert_eq!(
            classify_cuda_model(
                &json!({"architectures": ["Qwen3_5MoeForConditionalGeneration"], "model_type": "qwen3_5_moe"})
            ),
            CudaModelKind::Qwen35
        );
        assert_eq!(
            classify_cuda_model(&json!({"model_type": "qwen3", "num_experts": 128})),
            CudaModelKind::Qwen35
        );
        assert_eq!(
            classify_cuda_model(
                &json!({"architectures": ["DeepseekV4ForCausalLM"], "model_type": "deepseek_v4"})
            ),
            CudaModelKind::Dsv4
        );
        // GLM-5.2 (glm_moe_dsa) rides the DSv4 path; its 256 experts must not
        // misroute to Qwen35.
        assert_eq!(
            classify_cuda_model(
                &json!({"architectures": ["GlmMoeDsaForCausalLM"], "model_type": "glm_moe_dsa", "n_routed_experts": 256})
            ),
            CudaModelKind::Dsv4
        );
        assert_eq!(
            classify_cuda_model(
                &json!({"architectures": ["DiffusionGemmaForBlockDiffusion"], "model_type": "diffusion_gemma"})
            ),
            CudaModelKind::DiffusionGemma
        );
        assert_eq!(
            classify_cuda_model(
                &json!({"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"})
            ),
            CudaModelKind::DiffusionGemma
        );
        // Unknown / minimal config falls back to dense Qwen3.
        assert_eq!(classify_cuda_model(&json!({})), CudaModelKind::Qwen3Dense);
    }
}

#[cfg(any(
    feature = "metal",
    feature = "cuda",
    feature = "hip",
    feature = "vulkan",
    feature = "cpu"
))]
mod backend {
    use std::sync::Arc;

    use anyhow::Result;
    use infer_core::SchedulerConfig;
    use infer_server::ServeHandle;
    use tokio::sync::mpsc::UnboundedSender;

    #[cfg(feature = "cuda")]
    use super::CudaModelKind;
    use super::EngineLoadConfig;
    use crate::serve_engine::ServeInferenceEngine;
    use crate::types::{
        ChatPromptMessage, CompletionOutput, CompletionRequest, CompletionStreamDelta,
        EngineTelemetry, InferenceEngine, MultimodalChatRequest,
    };

    #[cfg(feature = "cuda")]
    use infer_cuda::{CudaExecutor, CudaKvPool};
    // For the `--kv-oversubscription` whole-slot-tier capability probe.
    #[cfg(feature = "hip")]
    use infer_hip::{HipDsv4Executor, HipKvPool};
    #[cfg(feature = "metal")]
    use infer_metal::{
        MetalDeepseekOcrModel, MetalDiffusionGemmaModel, MetalExecutor, MetalGemma4Model,
        MetalKvPool,
    };
    #[cfg(feature = "cuda")]
    use infer_seam::BackendExecutor;
    #[cfg(feature = "metal")]
    use infer_seam::{BufferedDiffusionExecutor, HostPagedKvPool};
    #[cfg(feature = "vulkan")]
    use infer_vulkan::{VulkanExecutor, VulkanKvPool};
    // The CPU path reuses infer-metal's feature-free placeholder executor over
    // the backend-neutral host paged KV pool.
    #[cfg(all(feature = "cpu", not(feature = "metal")))]
    use infer_metal::MetalExecutor;
    #[cfg(all(feature = "cpu", not(feature = "metal")))]
    use infer_seam::HostPagedKvPool;

    impl EngineLoadConfig {
        pub(super) fn scheduler_config(&self) -> SchedulerConfig {
            let mut config = SchedulerConfig::for_slots(self.hot_workspace_slots());
            // Prompt cap = min(requested, KV capacity − gen reserve). Capacity is
            // a hard ceiling, not a floor: over-length writes past the fixed DSv4
            // bands (#145). usize::MAX sentinel = unset → capacity-bound.
            let per_req_cap = self.total_pages.saturating_mul(self.page_size);
            let gen_reserve = per_req_cap / 8;
            config.max_prompt_tokens = self
                .max_prompt_tokens
                .min(per_req_cap.saturating_sub(gen_reserve));
            config.max_total_tokens = self.max_total_tokens;
            // Unset → 64: the Metal-interactivity default (small ticks keep the
            // single-threaded MLX encode loop responsive between decode steps).
            // The CUDA load path re-resolves per model kind before use.
            config.chunked_prefill_size = self.chunked_prefill_size.unwrap_or(64);
            if let Some(v) = self.max_num_batched_tokens {
                config.max_num_batched_tokens = v.max(1);
            }
            config.max_running_requests = self.max_running_requests;
            config.slot_oversubscription = self.slot_oversubscription;
            // Diagnostic-only escape hatch (not a shipped feature) for the
            // concurrent-decode digit-corruption investigation — see
            // docs/experience/errors/2026-07-06-dsv4-concurrent-decode-digit-corruption-unresolved.md.
            if std::env::var("ARLE_DISABLE_PREFIX_CACHE").is_ok() {
                config.enable_prefix_cache = false;
            }
            config
        }
    }

    /// Backend-dispatching public engine; one variant per compiled backend.
    pub enum LoadedInferenceEngine {
        /// Metal backend (Apple Silicon, MLX). Fully wired and runnable.
        #[cfg(feature = "metal")]
        Metal(ServeInferenceEngine<MetalExecutor, MetalKvPool>),
        /// Metal DiffusionGemma backend. The block-diffusion model is adapted
        /// to the shared autoregressive engine by a buffered executor.
        #[cfg(feature = "metal")]
        MetalDiffusionGemma(
            ServeInferenceEngine<
                BufferedDiffusionExecutor<MetalDiffusionGemmaModel>,
                HostPagedKvPool,
            >,
        ),
        /// Metal Gemma4 backend. The Gemma4 MLX bridge owns generation and is
        /// adapted to the shared autoregressive engine by a buffered executor.
        #[cfg(feature = "metal")]
        MetalGemma4(
            ServeInferenceEngine<BufferedDiffusionExecutor<MetalGemma4Model>, HostPagedKvPool>,
        ),
        /// Metal DeepSeek-OCR VLM backend. The DeepEncoder + DeepSeek-MoE MLX
        /// bridge owns generation and is adapted to the shared autoregressive
        /// engine by a buffered executor (single image, 1024x1024 base view).
        #[cfg(feature = "metal")]
        MetalDeepseekOcr(
            ServeInferenceEngine<BufferedDiffusionExecutor<MetalDeepseekOcrModel>, HostPagedKvPool>,
        ),
        /// CUDA backend (Linux + NVIDIA). Structurally wired (typechecks); the
        /// real forward is lead-owned and not yet runnable.
        #[cfg(feature = "cuda")]
        Cuda(ServeInferenceEngine<CudaExecutor, CudaKvPool>),
        /// HIP backend (AMD ROCm, DSv4 GGUF). Host-side wiring typechecks
        /// everywhere; the device forward runs on a ROCm box (pending-remote).
        #[cfg(feature = "hip")]
        Hip(ServeInferenceEngine<HipDsv4Executor, HipKvPool>),
        /// Vulkan backend (cross-vendor, GGUF). Host-side wiring typechecks
        /// everywhere; numeric device forward is pending AIPC on-box bring-up.
        #[cfg(feature = "vulkan")]
        Vulkan(ServeInferenceEngine<VulkanExecutor, VulkanKvPool>),
        /// Portable CPU backend: the placeholder `MetalExecutor` over the
        /// backend-neutral host paged KV pool (no MLX, no CUDA). Smoke / CI.
        #[cfg(all(feature = "cpu", not(feature = "metal")))]
        Cpu(ServeInferenceEngine<MetalExecutor, HostPagedKvPool>),
    }

    impl LoadedInferenceEngine {
        /// Load the inference engine for the compiled backend.
        /// `enable_cuda_graph` is honored by the CUDA path only.
        ///
        /// Single-user load (REPL, OCR): caps slots at 1 so the GDR recurrent
        /// state doesn't reserve `num_slots`× per-slot bytes (12 GiB for a 9B
        /// model at the default 256 slots). Multi-request serving uses
        /// `load_with_config` with the serve-derived slot budget.
        pub fn load(model_path: &str, enable_cuda_graph: bool) -> Result<Self> {
            let config = EngineLoadConfig {
                max_running_requests: Some(1),
                ..EngineLoadConfig::default()
            };
            Self::load_with_config(model_path, enable_cuda_graph, config)
        }

        /// Load with explicit slot / page configuration.
        // Each arm is a feature-gated `return` (the tail arm varies by feature
        // set), so a bare expression would not compile in single-backend builds.
        #[allow(clippy::needless_return)]
        pub fn load_with_config(
            model_path: &str,
            enable_cuda_graph: bool,
            config: EngineLoadConfig,
        ) -> Result<Self> {
            // Model-driven serve defaults for omitted sampling fields (nucleus +
            // temperature). The cc rollout lane overrides `.temperature` after this.
            infer_server::set_sampling_defaults(
                infer_server::SamplingDefaults::from_generation_config(model_path),
            );

            #[cfg(feature = "metal")]
            {
                let _ = enable_cuda_graph;
                return Self::load_metal(model_path, &config);
            }

            #[cfg(all(not(feature = "metal"), feature = "cuda"))]
            {
                return Self::load_cuda(model_path, enable_cuda_graph, &config);
            }

            #[cfg(all(not(feature = "metal"), not(feature = "cuda"), feature = "hip"))]
            {
                let _ = enable_cuda_graph;
                return Self::load_hip(model_path, &config);
            }

            #[cfg(all(
                not(feature = "metal"),
                not(feature = "cuda"),
                not(feature = "hip"),
                feature = "vulkan"
            ))]
            {
                let _ = enable_cuda_graph;
                return Self::load_vulkan(model_path, &config);
            }

            #[cfg(all(
                not(feature = "metal"),
                not(feature = "cuda"),
                not(feature = "hip"),
                not(feature = "vulkan"),
                feature = "cpu"
            ))]
            {
                let _ = enable_cuda_graph;
                return Self::load_cpu(model_path, &config);
            }
        }

        /// Name of the active backend variant.
        #[must_use]
        pub fn backend_name(&self) -> &'static str {
            match self {
                #[cfg(feature = "metal")]
                Self::Metal(_) => "metal",
                #[cfg(feature = "metal")]
                Self::MetalDiffusionGemma(_) => "metal-diffusion-gemma",
                #[cfg(feature = "metal")]
                Self::MetalGemma4(_) => "metal-gemma4",
                #[cfg(feature = "metal")]
                Self::MetalDeepseekOcr(_) => "metal-deepseek-ocr",
                #[cfg(feature = "cuda")]
                Self::Cuda(_) => "cuda",
                #[cfg(feature = "hip")]
                Self::Hip(_) => "hip",
                #[cfg(feature = "vulkan")]
                Self::Vulkan(_) => "vulkan",
                #[cfg(all(feature = "cpu", not(feature = "metal")))]
                Self::Cpu(_) => "cpu",
            }
        }

        /// OPD-teacher raw-logits forward: run the full `[seq_len, vocab]` teacher
        /// forward over `(input_ids, positions)` (no sampling) and return the
        /// device logits. CUDA-only; Metal/CPU bail. The `train` OPD path couples
        /// to this method on the runtime-led engine.
        #[cfg(feature = "cuda")]
        pub fn forward_token_logits(
            &self,
            input_ids: &[u32],
            positions: &[u32],
        ) -> Result<crate::types::RawLogits> {
            match self {
                Self::Cuda(engine) => engine.forward_token_logits(input_ids, positions),
                #[cfg(feature = "metal")]
                Self::Metal(_) => {
                    anyhow::bail!("forward_token_logits is CUDA-only (OPD teacher raw logits)")
                }
                #[cfg(feature = "metal")]
                Self::MetalDiffusionGemma(_) => {
                    anyhow::bail!("forward_token_logits is CUDA-only (OPD teacher raw logits)")
                }
                #[cfg(feature = "metal")]
                Self::MetalGemma4(_) => {
                    anyhow::bail!("forward_token_logits is CUDA-only (OPD teacher raw logits)")
                }
                #[cfg(feature = "metal")]
                Self::MetalDeepseekOcr(_) => {
                    anyhow::bail!("forward_token_logits is CUDA-only (OPD teacher raw logits)")
                }
                #[cfg(feature = "hip")]
                Self::Hip(_) => {
                    anyhow::bail!("forward_token_logits is CUDA-only (OPD teacher raw logits)")
                }
                #[cfg(feature = "vulkan")]
                Self::Vulkan(_) => {
                    anyhow::bail!("forward_token_logits is CUDA-only (OPD teacher raw logits)")
                }
                #[cfg(all(feature = "cpu", not(feature = "metal")))]
                Self::Cpu(_) => {
                    anyhow::bail!("forward_token_logits is CUDA-only (OPD teacher raw logits)")
                }
            }
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
            match self {
                Self::Cuda(engine) => engine.forward_training_taps(input_ids, target_layer_ids),
                #[cfg(feature = "metal")]
                Self::Metal(_)
                | Self::MetalDiffusionGemma(_)
                | Self::MetalGemma4(_)
                | Self::MetalDeepseekOcr(_) => {
                    anyhow::bail!("forward_training_taps is CUDA-only")
                }
                #[cfg(feature = "hip")]
                Self::Hip(_) => anyhow::bail!("forward_training_taps is CUDA-only"),
                #[cfg(feature = "vulkan")]
                Self::Vulkan(_) => anyhow::bail!("forward_training_taps is CUDA-only"),
                #[cfg(all(feature = "cpu", not(feature = "metal")))]
                Self::Cpu(_) => anyhow::bail!("forward_training_taps is CUDA-only"),
            }
        }

        /// Hot-swap the DSpark Markov head weights from a host f32 snapshot.
        /// Called by the train sidecar after each acceptance-weighted step.
        #[cfg(feature = "cuda")]
        pub fn update_dspark_markov_weights(&self, w1: &[f32], w2: &[f32]) -> Result<()> {
            match self {
                Self::Cuda(engine) => engine.update_dspark_markov_weights(w1.to_vec(), w2.to_vec()),
                #[cfg(feature = "metal")]
                Self::Metal(_)
                | Self::MetalDiffusionGemma(_)
                | Self::MetalGemma4(_)
                | Self::MetalDeepseekOcr(_) => {
                    anyhow::bail!("update_dspark_markov_weights is CUDA-only")
                }
                #[cfg(feature = "hip")]
                Self::Hip(_) => anyhow::bail!("update_dspark_markov_weights is CUDA-only"),
                #[cfg(feature = "vulkan")]
                Self::Vulkan(_) => anyhow::bail!("update_dspark_markov_weights is CUDA-only"),
                #[cfg(all(feature = "cpu", not(feature = "metal")))]
                Self::Cpu(_) => anyhow::bail!("update_dspark_markov_weights is CUDA-only"),
            }
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
            match self {
                Self::Cuda(engine) => {
                    engine.generate_token_ids(prompt_token_ids, max_tokens, sampling)
                }
                #[cfg(feature = "metal")]
                Self::Metal(_) => {
                    anyhow::bail!("generate_token_ids is CUDA-only for OPD student rollout")
                }
                #[cfg(feature = "metal")]
                Self::MetalDiffusionGemma(_) => {
                    anyhow::bail!("generate_token_ids is CUDA-only for OPD student rollout")
                }
                #[cfg(feature = "metal")]
                Self::MetalGemma4(_) => {
                    anyhow::bail!("generate_token_ids is CUDA-only for OPD student rollout")
                }
                #[cfg(feature = "metal")]
                Self::MetalDeepseekOcr(_) => {
                    anyhow::bail!("generate_token_ids is CUDA-only for OPD student rollout")
                }
                #[cfg(feature = "hip")]
                Self::Hip(_) => {
                    anyhow::bail!("generate_token_ids is CUDA-only for OPD student rollout")
                }
                #[cfg(feature = "vulkan")]
                Self::Vulkan(_) => {
                    anyhow::bail!("generate_token_ids is CUDA-only for OPD student rollout")
                }
                #[cfg(all(feature = "cpu", not(feature = "metal")))]
                Self::Cpu(_) => {
                    anyhow::bail!("generate_token_ids is CUDA-only for OPD student rollout")
                }
            }
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
            match self {
                Self::Cuda(engine) => engine.generate_token_ids_batch(requests, max_tokens),
                #[cfg(feature = "metal")]
                Self::Metal(_) => {
                    anyhow::bail!("generate_token_ids_batch is CUDA-only for OPD")
                }
                #[cfg(feature = "metal")]
                Self::MetalDiffusionGemma(_) => {
                    anyhow::bail!("generate_token_ids_batch is CUDA-only for OPD")
                }
                #[cfg(feature = "metal")]
                Self::MetalGemma4(_) => {
                    anyhow::bail!("generate_token_ids_batch is CUDA-only for OPD")
                }
                #[cfg(feature = "metal")]
                Self::MetalDeepseekOcr(_) => {
                    anyhow::bail!("generate_token_ids_batch is CUDA-only for OPD")
                }
                #[cfg(feature = "hip")]
                Self::Hip(_) => {
                    anyhow::bail!("generate_token_ids_batch is CUDA-only for OPD")
                }
                #[cfg(feature = "vulkan")]
                Self::Vulkan(_) => {
                    anyhow::bail!("generate_token_ids_batch is CUDA-only for OPD")
                }
                #[cfg(all(feature = "cpu", not(feature = "metal")))]
                Self::Cpu(_) => {
                    anyhow::bail!("generate_token_ids_batch is CUDA-only for OPD")
                }
            }
        }

        /// Batched text completion: submit all `CompletionRequest`s to the
        /// continuous-batching engine at once, then collect each. Used by
        /// rubric-OPD judging to decode N rollout verdicts of the same problem
        /// concurrently instead of one verdict at a time.
        #[cfg(feature = "cuda")]
        pub fn complete_batch(
            &self,
            reqs: Vec<CompletionRequest>,
        ) -> Result<Vec<CompletionOutput>> {
            match self {
                Self::Cuda(engine) => engine.complete_batch(reqs),
                #[cfg(feature = "metal")]
                Self::Metal(_) => {
                    anyhow::bail!("complete_batch is CUDA-only for OPD")
                }
                #[cfg(feature = "metal")]
                Self::MetalDiffusionGemma(_) => {
                    anyhow::bail!("complete_batch is CUDA-only for OPD")
                }
                #[cfg(feature = "metal")]
                Self::MetalGemma4(_) => {
                    anyhow::bail!("complete_batch is CUDA-only for OPD")
                }
                #[cfg(feature = "metal")]
                Self::MetalDeepseekOcr(_) => {
                    anyhow::bail!("complete_batch is CUDA-only for OPD")
                }
                #[cfg(feature = "hip")]
                Self::Hip(_) => {
                    anyhow::bail!("complete_batch is CUDA-only for OPD")
                }
                #[cfg(feature = "vulkan")]
                Self::Vulkan(_) => {
                    anyhow::bail!("complete_batch is CUDA-only for OPD")
                }
                #[cfg(all(feature = "cpu", not(feature = "metal")))]
                Self::Cpu(_) => {
                    anyhow::bail!("complete_batch is CUDA-only for OPD")
                }
            }
        }

        /// Offload the engine's device weights to host RAM (OPD teacher weight
        /// time-share), returning the device bytes freed. CUDA-only: the
        /// Qwen3.5/3.6 hybrid OPD teacher path moves its weights off-device so a
        /// co-resident student backward reuses the VRAM. Metal/CPU have no
        /// device-weight offload path and bail.
        pub fn offload_engine_weights(&self) -> Result<usize> {
            match self {
                #[cfg(feature = "cuda")]
                Self::Cuda(engine) => engine.offload_engine_weights(),
                #[cfg(feature = "metal")]
                Self::Metal(_) => anyhow::bail!("offload_engine_weights is only available on CUDA"),
                #[cfg(feature = "metal")]
                Self::MetalDiffusionGemma(_) => {
                    anyhow::bail!("offload_engine_weights is only available on CUDA")
                }
                #[cfg(feature = "metal")]
                Self::MetalGemma4(_) => {
                    anyhow::bail!("offload_engine_weights is only available on CUDA")
                }
                #[cfg(feature = "metal")]
                Self::MetalDeepseekOcr(_) => {
                    anyhow::bail!("offload_engine_weights is only available on CUDA")
                }
                #[cfg(feature = "hip")]
                Self::Hip(_) => anyhow::bail!("offload_engine_weights is only available on CUDA"),
                #[cfg(feature = "vulkan")]
                Self::Vulkan(_) => {
                    anyhow::bail!("offload_engine_weights is only available on CUDA")
                }
                #[cfg(all(feature = "cpu", not(feature = "metal")))]
                Self::Cpu(_) => anyhow::bail!("offload_engine_weights is only available on CUDA"),
            }
        }

        /// Quiesce engine admission (the serve loop defers new admission) and
        /// cancel every in-flight (waiting + active) request, returning how many
        /// were cancelled. The OPD round-loop writeback bracket; pairs with
        /// [`Self::resume_admissions`] after the KV pool is re-acquired.
        pub fn quiesce_admissions(&self) -> Result<usize> {
            match self {
                #[cfg(feature = "metal")]
                Self::Metal(engine) => engine.quiesce_admissions(),
                #[cfg(feature = "metal")]
                Self::MetalDiffusionGemma(engine) => engine.quiesce_admissions(),
                #[cfg(feature = "metal")]
                Self::MetalGemma4(engine) => engine.quiesce_admissions(),
                #[cfg(feature = "metal")]
                Self::MetalDeepseekOcr(engine) => engine.quiesce_admissions(),
                #[cfg(feature = "cuda")]
                Self::Cuda(engine) => engine.quiesce_admissions(),
                #[cfg(feature = "hip")]
                Self::Hip(engine) => engine.quiesce_admissions(),
                #[cfg(feature = "vulkan")]
                Self::Vulkan(engine) => engine.quiesce_admissions(),
                #[cfg(all(feature = "cpu", not(feature = "metal")))]
                Self::Cpu(engine) => engine.quiesce_admissions(),
            }
        }

        /// Re-arm admission after the OPD writeback bracket (KV pool re-acquired).
        pub fn resume_admissions(&self) -> Result<()> {
            match self {
                #[cfg(feature = "metal")]
                Self::Metal(engine) => engine.resume_admissions(),
                #[cfg(feature = "metal")]
                Self::MetalDiffusionGemma(engine) => engine.resume_admissions(),
                #[cfg(feature = "metal")]
                Self::MetalGemma4(engine) => engine.resume_admissions(),
                #[cfg(feature = "metal")]
                Self::MetalDeepseekOcr(engine) => engine.resume_admissions(),
                #[cfg(feature = "cuda")]
                Self::Cuda(engine) => engine.resume_admissions(),
                #[cfg(feature = "hip")]
                Self::Hip(engine) => engine.resume_admissions(),
                #[cfg(feature = "vulkan")]
                Self::Vulkan(engine) => engine.resume_admissions(),
                #[cfg(all(feature = "cpu", not(feature = "metal")))]
                Self::Cpu(engine) => engine.resume_admissions(),
            }
        }

        /// Re-acquire the KV pool, then resume admission only after success.
        pub fn ensure_kv_pool_and_resume_admissions(&self) -> Result<()> {
            match self {
                #[cfg(feature = "metal")]
                Self::Metal(engine) => engine.ensure_kv_pool_and_resume_admissions(),
                #[cfg(feature = "metal")]
                Self::MetalDiffusionGemma(engine) => engine.ensure_kv_pool_and_resume_admissions(),
                #[cfg(feature = "metal")]
                Self::MetalGemma4(engine) => engine.ensure_kv_pool_and_resume_admissions(),
                #[cfg(feature = "metal")]
                Self::MetalDeepseekOcr(engine) => engine.ensure_kv_pool_and_resume_admissions(),
                #[cfg(feature = "cuda")]
                Self::Cuda(engine) => engine.ensure_kv_pool_and_resume_admissions(),
                #[cfg(feature = "hip")]
                Self::Hip(engine) => engine.ensure_kv_pool_and_resume_admissions(),
                #[cfg(feature = "vulkan")]
                Self::Vulkan(engine) => engine.ensure_kv_pool_and_resume_admissions(),
                #[cfg(all(feature = "cpu", not(feature = "metal")))]
                Self::Cpu(engine) => engine.ensure_kv_pool_and_resume_admissions(),
            }
        }

        /// Reload the engine's device weights from the host snapshot (OPD teacher
        /// weight time-share). CUDA-only; Metal/CPU bail.
        pub fn reload_engine_weights(&self) -> Result<()> {
            match self {
                #[cfg(feature = "cuda")]
                Self::Cuda(engine) => engine.reload_engine_weights(),
                #[cfg(feature = "metal")]
                Self::Metal(_) => anyhow::bail!("reload_engine_weights is only available on CUDA"),
                #[cfg(feature = "metal")]
                Self::MetalDiffusionGemma(_) => {
                    anyhow::bail!("reload_engine_weights is only available on CUDA")
                }
                #[cfg(feature = "metal")]
                Self::MetalGemma4(_) => {
                    anyhow::bail!("reload_engine_weights is only available on CUDA")
                }
                #[cfg(feature = "metal")]
                Self::MetalDeepseekOcr(_) => {
                    anyhow::bail!("reload_engine_weights is only available on CUDA")
                }
                #[cfg(feature = "hip")]
                Self::Hip(_) => anyhow::bail!("reload_engine_weights is only available on CUDA"),
                #[cfg(feature = "vulkan")]
                Self::Vulkan(_) => {
                    anyhow::bail!("reload_engine_weights is only available on CUDA")
                }
                #[cfg(all(feature = "cpu", not(feature = "metal")))]
                Self::Cpu(_) => anyhow::bail!("reload_engine_weights is only available on CUDA"),
            }
        }

        /// Release the engine's inference forward scratch WITHOUT offloading weights
        /// or evicting KV (OPD rollout->writeback VRAM reclaim). CUDA-only behavior;
        /// Metal/CPU/other arms are no-ops (no `Qwen35Workspace`-style scratch to
        /// release) so this is safe to call unconditionally on the writeback path.
        pub fn release_inference_scratch(&self) -> Result<()> {
            match self {
                #[cfg(feature = "cuda")]
                Self::Cuda(engine) => engine.release_inference_scratch(),
                #[cfg(feature = "metal")]
                Self::Metal(_) => Ok(()),
                #[cfg(feature = "metal")]
                Self::MetalDiffusionGemma(_) => Ok(()),
                #[cfg(feature = "metal")]
                Self::MetalGemma4(_) => Ok(()),
                #[cfg(feature = "metal")]
                Self::MetalDeepseekOcr(_) => Ok(()),
                #[cfg(feature = "hip")]
                Self::Hip(_) => Ok(()),
                #[cfg(feature = "vulkan")]
                Self::Vulkan(_) => Ok(()),
                #[cfg(all(feature = "cpu", not(feature = "metal")))]
                Self::Cpu(_) => Ok(()),
            }
        }

        /// Drop the engine's KV pool WITHOUT offloading weights (OPD writeback
        /// headroom: the writeback's fresh autograd forward never reads this
        /// engine's KV). CUDA Qwen3.5/3.6 only; other arms are no-ops, so this is
        /// safe to call unconditionally on the agent-OPD writeback path.
        pub fn release_kv_pool(&self) -> Result<()> {
            match self {
                #[cfg(feature = "cuda")]
                Self::Cuda(engine) => engine.release_kv_pool(),
                #[cfg(feature = "metal")]
                Self::Metal(_) => Ok(()),
                #[cfg(feature = "metal")]
                Self::MetalDiffusionGemma(_) => Ok(()),
                #[cfg(feature = "metal")]
                Self::MetalGemma4(_) => Ok(()),
                #[cfg(feature = "metal")]
                Self::MetalDeepseekOcr(_) => Ok(()),
                #[cfg(feature = "hip")]
                Self::Hip(_) => Ok(()),
                #[cfg(feature = "vulkan")]
                Self::Vulkan(_) => Ok(()),
                #[cfg(all(feature = "cpu", not(feature = "metal")))]
                Self::Cpu(_) => Ok(()),
            }
        }

        /// Re-acquire the KV pool dropped by [`Self::release_kv_pool`] before the
        /// next rollout. CUDA Qwen3.5/3.6 only; other arms are no-ops.
        pub fn ensure_kv_pool(&self) -> Result<()> {
            match self {
                #[cfg(feature = "cuda")]
                Self::Cuda(engine) => engine.ensure_kv_pool(),
                #[cfg(feature = "metal")]
                Self::Metal(_) => Ok(()),
                #[cfg(feature = "metal")]
                Self::MetalDiffusionGemma(_) => Ok(()),
                #[cfg(feature = "metal")]
                Self::MetalGemma4(_) => Ok(()),
                #[cfg(feature = "metal")]
                Self::MetalDeepseekOcr(_) => Ok(()),
                #[cfg(feature = "hip")]
                Self::Hip(_) => Ok(()),
                #[cfg(feature = "vulkan")]
                Self::Vulkan(_) => Ok(()),
                #[cfg(all(feature = "cpu", not(feature = "metal")))]
                Self::Cpu(_) => Ok(()),
            }
        }

        /// Fold a fresh student LoRA update into the resident Qwen3.5/3.6
        /// projection weights (OPD per-step re-merge). CUDA-only: the Metal /
        /// CPU arms reject it.
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
            match self {
                Self::Cuda(engine) => engine.remerge_student_lora(update),
                #[cfg(feature = "metal")]
                Self::Metal(_) => {
                    let _ = update;
                    anyhow::bail!("student LoRA re-merge is CUDA-only; active backend is Metal")
                }
                #[cfg(feature = "metal")]
                Self::MetalDiffusionGemma(_) => {
                    let _ = update;
                    anyhow::bail!(
                        "student LoRA re-merge is CUDA-only; active backend is Metal DiffusionGemma"
                    )
                }
                #[cfg(feature = "metal")]
                Self::MetalGemma4(_) => {
                    let _ = update;
                    anyhow::bail!(
                        "student LoRA re-merge is CUDA-only; active backend is Metal Gemma4"
                    )
                }
                #[cfg(feature = "metal")]
                Self::MetalDeepseekOcr(_) => {
                    let _ = update;
                    anyhow::bail!(
                        "student LoRA re-merge is CUDA-only; active backend is Metal DeepSeek-OCR"
                    )
                }
                #[cfg(feature = "hip")]
                Self::Hip(_) => {
                    let _ = update;
                    anyhow::bail!("student LoRA re-merge is CUDA-only; active backend is HIP")
                }
                #[cfg(feature = "vulkan")]
                Self::Vulkan(_) => {
                    let _ = update;
                    anyhow::bail!("student LoRA re-merge is CUDA-only; active backend is Vulkan")
                }
                #[cfg(all(feature = "cpu", not(feature = "metal")))]
                Self::Cpu(_) => {
                    let _ = update;
                    anyhow::bail!("student LoRA re-merge is CUDA-only; active backend is CPU")
                }
            }
        }

        /// Read-only borrow of resident FP8 block-scaled base projection
        /// pointers for train-infer weight sharing (`--share-frozen-base`).
        /// CUDA-only: only the Qwen3.5/3.6 hybrid student carries shareable FP8
        /// base weights. Returns the pointer table (raw device `u64`s + dims);
        /// the train loader imports a NON-OWNING view over these instead of
        /// allocating its own copy of the shared frozen base.
        #[cfg(feature = "cuda")]
        pub fn frozen_base_fp8_pointers(&self) -> Result<Vec<infer_cuda::SharedFp8BaseProjection>> {
            match self {
                Self::Cuda(engine) => engine.frozen_base_fp8_pointers(),
                #[cfg(feature = "metal")]
                Self::Metal(_) => {
                    anyhow::bail!("frozen-base FP8 sharing is CUDA-only; active backend is Metal")
                }
                #[cfg(feature = "metal")]
                Self::MetalDiffusionGemma(_) => anyhow::bail!(
                    "frozen-base FP8 sharing is CUDA-only; active backend is Metal DiffusionGemma"
                ),
                #[cfg(feature = "metal")]
                Self::MetalGemma4(_) => anyhow::bail!(
                    "frozen-base FP8 sharing is CUDA-only; active backend is Metal Gemma4"
                ),
                #[cfg(feature = "metal")]
                Self::MetalDeepseekOcr(_) => anyhow::bail!(
                    "frozen-base FP8 sharing is CUDA-only; active backend is Metal DeepSeek-OCR"
                ),
                #[cfg(feature = "hip")]
                Self::Hip(_) => {
                    anyhow::bail!("frozen-base FP8 sharing is CUDA-only; active backend is HIP")
                }
                #[cfg(feature = "vulkan")]
                Self::Vulkan(_) => {
                    anyhow::bail!("frozen-base FP8 sharing is CUDA-only; active backend is Vulkan")
                }
                #[cfg(all(feature = "cpu", not(feature = "metal")))]
                Self::Cpu(_) => {
                    anyhow::bail!("frozen-base FP8 sharing is CUDA-only; active backend is CPU")
                }
            }
        }

        /// OpenAI-compat HTTP router over this ALREADY-loaded engine's
        /// `ServeHandle` (same engine thread, same KV pool) — unlike
        /// `router_for_backend`, which spawns a second engine. Serve it with
        /// [`crate::serve_router_on_thread`].
        #[cfg(feature = "cuda")]
        pub fn local_router(&self, max_thinking_tokens: usize) -> Result<axum::Router> {
            match self {
                Self::Cuda(engine) => Ok(infer_server::coordinator_local_router(
                    engine.serve_arc(),
                    engine.tokenizer().clone(),
                    engine.model_id().to_string(),
                    max_thinking_tokens,
                    None,
                )),
                #[cfg(feature = "metal")]
                Self::Metal(_) => {
                    anyhow::bail!("local router is CUDA-only; active backend is Metal")
                }
                #[cfg(feature = "metal")]
                Self::MetalDiffusionGemma(_) => anyhow::bail!(
                    "local router is CUDA-only; active backend is Metal DiffusionGemma"
                ),
                #[cfg(feature = "metal")]
                Self::MetalGemma4(_) => {
                    anyhow::bail!("local router is CUDA-only; active backend is Metal Gemma4")
                }
                #[cfg(feature = "metal")]
                Self::MetalDeepseekOcr(_) => {
                    anyhow::bail!("local router is CUDA-only; active backend is Metal DeepSeek-OCR")
                }
                #[cfg(feature = "hip")]
                Self::Hip(_) => {
                    anyhow::bail!("local router is CUDA-only; active backend is HIP")
                }
                #[cfg(feature = "vulkan")]
                Self::Vulkan(_) => {
                    anyhow::bail!("local router is CUDA-only; active backend is Vulkan")
                }
                #[cfg(all(feature = "cpu", not(feature = "metal")))]
                Self::Cpu(_) => {
                    anyhow::bail!("local router is CUDA-only; active backend is CPU")
                }
            }
        }

        #[cfg(feature = "metal")]
        fn load_metal(model_path: &str, config: &EngineLoadConfig) -> Result<Self> {
            let resolved = infer_metal::resolve_model_path(model_path)?;
            if infer_metal::model_dir_is_diffusion_gemma(&resolved) {
                let (serve, tokenizer, model_id) = metal_diffusion_gemma_serve_handle(
                    model_path,
                    &resolved,
                    config,
                    infer_server::ServeShutdown::new(),
                )?;
                return Ok(Self::MetalDiffusionGemma(ServeInferenceEngine::new(
                    model_id, tokenizer, serve,
                )));
            }
            if infer_metal::model_dir_is_gemma4(&resolved) {
                let (serve, tokenizer, model_id) = metal_gemma4_serve_handle(
                    model_path,
                    &resolved,
                    config,
                    infer_server::ServeShutdown::new(),
                )?;
                return Ok(Self::MetalGemma4(ServeInferenceEngine::new(
                    model_id, tokenizer, serve,
                )));
            }
            if infer_metal::model_dir_is_deepseek_ocr(&resolved) {
                let (serve, tokenizer, model_id) = metal_deepseek_ocr_serve_handle(
                    model_path,
                    &resolved,
                    config,
                    infer_server::ServeShutdown::new(),
                )?;
                return Ok(Self::MetalDeepseekOcr(ServeInferenceEngine::new(
                    model_id, tokenizer, serve,
                )));
            }
            let (serve, tokenizer, model_id) =
                metal_serve_handle(model_path, config, infer_server::ServeShutdown::new())?;
            Ok(Self::Metal(ServeInferenceEngine::new(
                model_id, tokenizer, serve,
            )))
        }

        #[cfg(feature = "cuda")]
        fn load_cuda(
            model_path: &str,
            enable_cuda_graph: bool,
            config: &EngineLoadConfig,
        ) -> Result<Self> {
            // Single-GPU CUDA load: dispatch by checkpoint kind (Qwen3 dense +
            // Qwen3.5/3.6 MoE; DSv4 is multi-GPU only and errors). `enable_cuda_graph`
            // (CLI --cuda-graph, default on) sets the decode-graph default;
            // `--no-cuda-graph` controls it, `warmup` gates it off under TP/MoE.
            // Shares the engine builder with `router_cuda` via `cuda_serve_handle`.
            let (serve, tokenizer, model_id) = cuda_serve_handle(
                model_path,
                enable_cuda_graph,
                config,
                infer_server::ServeShutdown::new(),
            )?;
            Ok(Self::Cuda(ServeInferenceEngine::new(
                model_id, tokenizer, serve,
            )))
        }

        #[cfg(feature = "hip")]
        fn load_hip(model_path: &str, config: &EngineLoadConfig) -> Result<Self> {
            // HIP DSv4 GGUF load. Shares the engine builder with `router_hip`
            // via `hip_serve_handle`, mirroring the CUDA `cuda_serve_handle`
            // split (Metal instead reuses an infer-server facade; infer-server
            // has no HIP code, so the handle is built here).
            let (serve, tokenizer, model_id) =
                hip_serve_handle(model_path, config, infer_server::ServeShutdown::new())?;
            Ok(Self::Hip(ServeInferenceEngine::new(
                model_id, tokenizer, serve,
            )))
        }

        #[cfg(feature = "vulkan")]
        fn load_vulkan(model_path: &str, config: &EngineLoadConfig) -> Result<Self> {
            // Vulkan GGUF load. Shares the engine builder with `router_vulkan`
            // via `vulkan_serve_handle`, mirroring the HIP path while keeping
            // all Vulkan types below the seam.
            let (serve, tokenizer, model_id) =
                vulkan_serve_handle(model_path, config, infer_server::ServeShutdown::new())?;
            Ok(Self::Vulkan(ServeInferenceEngine::new(
                model_id, tokenizer, serve,
            )))
        }

        #[cfg(all(
            feature = "cpu",
            not(feature = "metal"),
            not(feature = "cuda"),
            not(feature = "hip"),
            not(feature = "vulkan")
        ))]
        fn load_cpu(model_path: &str, config: &EngineLoadConfig) -> Result<Self> {
            use infer_server::OpenAiTokenizer;

            if config.mtp_enabled() {
                anyhow::bail!("MTP speculative decode is only supported by the CUDA backend");
            }
            anyhow::ensure!(
                !config.kv_ssd_requested(),
                "--kv-disk: the CPU backend has no KV tier store"
            );
            // CPU smoke: placeholder executor over a real host KV pool; still
            // needs a tokenizer dir for encode/decode.
            let tokenizer = OpenAiTokenizer::from_model_dir(model_path)?;
            let model_id = crate::serve_engine::model_id_from_path(model_path);
            let executor = MetalExecutor::new();
            let kv = HostPagedKvPool::new(
                config.hot_workspace_slots(),
                config.total_pages,
                config.page_size,
            );
            let serve = ServeHandle::spawn(executor, kv, config.scheduler_config());
            Ok(Self::Cpu(ServeInferenceEngine::new(
                model_id, tokenizer, serve,
            )))
        }
    }

    /// Build the OpenAI v1 axum router for the compiled backend.
    ///
    /// Mirrors [`LoadedInferenceEngine::load_with_config`] but returns the bare
    /// [`axum::Router`] the in-process [`crate::serve_http`] loop binds, rather
    /// than the [`InferenceEngine`] adapter the agent/OPD callers use. Each arm
    /// spawns the same [`ServeHandle`] the matching `load_*` method spawns, then
    /// hands it to the backend-neutral [`infer_server::openai_router`].
    // Each arm is a feature-gated `return` (the tail arm varies by feature set),
    // so a bare expression would not compile in single-backend builds.
    #[allow(clippy::needless_return)]
    pub(crate) fn router_for_backend(
        model_path: &str,
        enable_cuda_graph: bool,
        config: EngineLoadConfig,
        shutdown: infer_server::ServeShutdown,
    ) -> Result<(axum::Router, Option<Arc<LoadedInferenceEngine>>)> {
        // Model-driven serve defaults for omitted sampling fields (nucleus +
        // temperature) — the `arle serve` router lane.
        infer_server::set_sampling_defaults(
            infer_server::SamplingDefaults::from_generation_config(model_path),
        );

        // The L3 disk tier is consumed by CUDA and Metal; every other
        // backend fails closed on an explicit request instead of silently
        // serving without it.
        #[cfg(all(not(feature = "metal"), not(feature = "cuda")))]
        anyhow::ensure!(
            !config.kv_ssd_requested(),
            "--kv-disk: the L3 KV tier is only supported by CUDA and Metal today"
        );

        #[cfg(feature = "metal")]
        {
            let _ = enable_cuda_graph;
            let router = router_metal(model_path, &config, shutdown)?;
            return Ok((router, None));
        }

        #[cfg(all(not(feature = "metal"), feature = "cuda"))]
        {
            return router_cuda(model_path, enable_cuda_graph, &config, shutdown)
                .map(|(r, e)| (r, Some(e)));
        }

        #[cfg(all(not(feature = "metal"), not(feature = "cuda"), feature = "hip"))]
        {
            let _ = enable_cuda_graph;
            let router = router_hip(model_path, &config, shutdown)?;
            return Ok((router, None));
        }

        #[cfg(all(
            not(feature = "metal"),
            not(feature = "cuda"),
            not(feature = "hip"),
            feature = "vulkan"
        ))]
        {
            let _ = enable_cuda_graph;
            let router = router_vulkan(model_path, &config, shutdown)?;
            return Ok((router, None));
        }

        #[cfg(all(
            not(feature = "metal"),
            not(feature = "cuda"),
            not(feature = "hip"),
            not(feature = "vulkan"),
            feature = "cpu"
        ))]
        {
            let _ = enable_cuda_graph;
            let router = router_cpu(model_path, &config, shutdown)?;
            return Ok((router, None));
        }
    }

    /// Shared Metal engine builder for [`LoadedInferenceEngine::load_metal`] and
    /// [`router_metal`]. The service layer stays backend-neutral: Metal-specific
    /// model resolution, executor construction, and KV-pool sizing happen here.
    #[cfg(feature = "metal")]
    fn metal_serve_handle(
        model_path: &str,
        config: &EngineLoadConfig,
        shutdown: infer_server::ServeShutdown,
    ) -> Result<(
        ServeHandle<MetalExecutor, MetalKvPool>,
        infer_server::OpenAiTokenizer,
        String,
    )> {
        use infer_server::OpenAiTokenizer;

        if config.mtp_enabled() {
            anyhow::bail!("MTP speculative decode is only supported by the CUDA backend");
        }
        // Flags land in the statics before executor construction (spec-decode
        // resolver + pipeline/warmup/paged-read/sampling gates).
        infer_metal::apply_runtime_flags(&config.metal);
        let metal_kv_dtype = infer_metal::MetalKvCacheDtype::resolve(config.kv_cache_dtype)?;
        let resolved = infer_metal::resolve_model_path(model_path)?;
        let tokenizer = OpenAiTokenizer::from_model_dir(&resolved)?;
        let model_id = crate::serve_engine::model_id_from_path(model_path);

        let model_source = resolved.to_string_lossy().to_string();
        let mut scheduler = config.scheduler_config();
        let num_slots = config.hot_workspace_slots();
        let page_size = config.page_size;
        let low_impact = config.low_impact;
        let kv_recall = config.kv_recall;
        let resource_plan = infer_metal::plan_resource_budget(
            &resolved,
            infer_metal::MetalResourceRequest {
                kv_cache_dtype: metal_kv_dtype,
                num_slots,
                total_pages: config.total_pages,
                page_size,
                low_impact,
                memory_budget_bytes: config.memory_budget_bytes,
                system_reserve_bytes: config.system_reserve_bytes,
                allow_swap: config.allow_swap,
                mem_fraction_static: config.mem_fraction_static,
            },
        )?;
        let total_pages = resource_plan.planned_total_pages;
        let planned_capacity_tokens = resource_plan.capacity_tokens;
        if planned_capacity_tokens < scheduler.max_total_tokens {
            log::warn!(
                "Metal resource guard clamps max_total_tokens {} -> {}",
                scheduler.max_total_tokens,
                planned_capacity_tokens
            );
            scheduler.max_total_tokens = planned_capacity_tokens.max(1);
        }
        if scheduler.max_prompt_tokens > scheduler.max_total_tokens {
            log::warn!(
                "Metal resource guard clamps max_prompt_tokens {} -> {}",
                scheduler.max_prompt_tokens,
                scheduler.max_total_tokens
            );
            scheduler.max_prompt_tokens = scheduler.max_total_tokens;
        }
        // Opt-in L3 NVMe spill (`--kv-disk`): attached inside the builder —
        // at construction, like every other tier knob — never post-spawn.
        // Metal serves single-process, so the deployment-total cap is the
        // per-rank cap (world = 1).
        let kv_ssd = config.kv_ssd_spill(1, infer_metal::default_t2_budget_bytes)?;
        let serve = ServeHandle::spawn_with_engine_builder_and_shutdown(
            move || {
                let mut executor =
                    MetalExecutor::from_model_path_with_kv_cache_dtype_and_resource_plan(
                        &model_source,
                        metal_kv_dtype,
                        resource_plan,
                    )?;
                executor.set_kv_recall(kv_recall);
                if let Some((root, budget)) = kv_ssd {
                    anyhow::ensure!(
                        executor.set_kv_tier_disk(root, budget, page_size),
                        "--kv-disk: the loaded Metal model has no usable \
                         page-addressable KV tier store (a budget below one \
                         page also lands here; raise --kv-disk-limit)"
                    );
                }
                let kv = MetalKvPool::new(num_slots, total_pages, page_size);
                if low_impact {
                    let governor = infer_seam::CooperativeGovernor::new(infer_seam::StepBudget {
                        max_tokens: scheduler.chunked_prefill_size.max(1),
                        max_micros: 20_000,
                    })
                    .with_yield_every_ticks(8);
                    Ok(infer_core::Engine::with_config_and_governor(
                        executor,
                        kv,
                        scheduler,
                        Box::new(governor),
                    ))
                } else {
                    Ok(infer_core::Engine::with_config(executor, kv, scheduler))
                }
            },
            shutdown,
        )?;
        Ok((serve, tokenizer, model_id))
    }

    /// Metal serve router. Builds the same `ServeHandle` as
    /// [`LoadedInferenceEngine::load_metal`] and wraps it in the unified OpenAI
    /// facade.
    #[cfg(feature = "metal")]
    fn router_metal(
        model_path: &str,
        config: &EngineLoadConfig,
        shutdown: infer_server::ServeShutdown,
    ) -> Result<axum::Router> {
        let resolved = infer_metal::resolve_model_path(model_path)?;
        if infer_metal::model_dir_is_diffusion_gemma(&resolved) {
            let (serve, tokenizer, model_id) =
                metal_diffusion_gemma_serve_handle(model_path, &resolved, config, shutdown)?;
            return Ok(infer_server::coordinator_local_router(
                Arc::new(serve),
                tokenizer,
                model_id,
                config.max_thinking_tokens,
                Some(infer_plan::MultimodalKind::Gemma4),
            ));
        }
        if infer_metal::model_dir_is_gemma4(&resolved) {
            let (serve, tokenizer, model_id) =
                metal_gemma4_serve_handle(model_path, &resolved, config, shutdown)?;
            return Ok(infer_server::coordinator_local_router(
                Arc::new(serve),
                tokenizer,
                model_id,
                config.max_thinking_tokens,
                Some(infer_plan::MultimodalKind::Gemma4),
            ));
        }
        if infer_metal::model_dir_is_deepseek_ocr(&resolved) {
            let (serve, tokenizer, model_id) =
                metal_deepseek_ocr_serve_handle(model_path, &resolved, config, shutdown)?;
            return Ok(infer_server::coordinator_local_router(
                Arc::new(serve),
                tokenizer,
                model_id,
                config.max_thinking_tokens,
                Some(infer_plan::MultimodalKind::DeepseekOcr),
            ));
        }
        let (serve, tokenizer, model_id) = metal_serve_handle(model_path, config, shutdown)?;
        Ok(infer_server::coordinator_local_router(
            Arc::new(serve),
            tokenizer,
            model_id,
            config.max_thinking_tokens,
            None,
        ))
    }

    #[cfg(feature = "metal")]
    fn metal_diffusion_gemma_serve_handle(
        model_path: &str,
        resolved: &std::path::Path,
        config: &EngineLoadConfig,
        shutdown: infer_server::ServeShutdown,
    ) -> Result<(
        ServeHandle<BufferedDiffusionExecutor<MetalDiffusionGemmaModel>, HostPagedKvPool>,
        infer_server::OpenAiTokenizer,
        String,
    )> {
        use infer_server::OpenAiTokenizer;

        if config.mtp_enabled() {
            anyhow::bail!("MTP speculative decode is only supported by the CUDA backend");
        }
        anyhow::ensure!(
            !config.kv_ssd_requested(),
            "--kv-disk: DiffusionGemma Metal owns no page-addressable KV tier store"
        );

        let tokenizer = OpenAiTokenizer::from_model_dir(resolved)?;
        let model_id = crate::serve_engine::model_id_from_path(model_path);
        let model_source = resolved.to_string_lossy().to_string();
        let mut scheduler = config.scheduler_config();
        scheduler.num_slots = 1;
        scheduler.max_prompt_tokens = scheduler.max_prompt_tokens.min(scheduler.max_total_tokens);
        let page_size = config.page_size.max(1);
        let total_pages = config.total_pages.max(1);
        let low_impact = config.low_impact;
        let resource_plan = infer_metal::plan_weight_only_resource_budget(
            resolved,
            infer_metal::MetalWeightOnlyResourceRequest {
                low_impact,
                memory_budget_bytes: config.memory_budget_bytes,
                system_reserve_bytes: config.system_reserve_bytes,
                allow_swap: config.allow_swap,
            },
        )?;
        let cancel = shutdown.cancel_flag();
        let max_denoising_steps = config.diffusion_max_denoising_steps.filter(|&s| s > 0);

        let serve = ServeHandle::spawn_with_engine_builder_and_shutdown(
            move || {
                let loaded = infer_metal::MetalDiffusionGemmaModel::load_with_resource_plan(
                    std::path::Path::new(&model_source),
                    Some(resource_plan),
                )?;
                let mut generation = loaded.generation;
                if let Some(steps) = max_denoising_steps {
                    generation.max_denoising_steps = steps;
                }
                let executor =
                    BufferedDiffusionExecutor::new_with_cancel(loaded.model, generation, cancel);
                let kv = HostPagedKvPool::new(1, total_pages, page_size);
                if low_impact {
                    let governor = infer_seam::CooperativeGovernor::new(infer_seam::StepBudget {
                        max_tokens: scheduler.chunked_prefill_size.max(1),
                        max_micros: 20_000,
                    })
                    .with_yield_every_ticks(8);
                    Ok(infer_core::Engine::with_config_and_governor(
                        executor,
                        kv,
                        scheduler,
                        Box::new(governor),
                    ))
                } else {
                    Ok(infer_core::Engine::with_config(executor, kv, scheduler))
                }
            },
            shutdown,
        )?;
        Ok((serve, tokenizer, model_id))
    }

    #[cfg(feature = "metal")]
    fn metal_gemma4_serve_handle(
        model_path: &str,
        resolved: &std::path::Path,
        config: &EngineLoadConfig,
        shutdown: infer_server::ServeShutdown,
    ) -> Result<(
        ServeHandle<BufferedDiffusionExecutor<MetalGemma4Model>, HostPagedKvPool>,
        infer_server::OpenAiTokenizer,
        String,
    )> {
        use infer_server::OpenAiTokenizer;

        if config.mtp_enabled() {
            anyhow::bail!("MTP speculative decode is only supported by the CUDA backend");
        }
        anyhow::ensure!(
            !config.kv_ssd_requested(),
            "--kv-disk: Gemma4 Metal owns no page-addressable KV tier store"
        );

        let tokenizer = OpenAiTokenizer::from_model_dir(resolved)?;
        let model_id = crate::serve_engine::model_id_from_path(model_path);
        let model_source = resolved.to_string_lossy().to_string();
        let mut scheduler = config.scheduler_config();
        scheduler.num_slots = 1;
        scheduler.max_prompt_tokens = scheduler.max_prompt_tokens.min(scheduler.max_total_tokens);
        let page_size = config.page_size.max(1);
        let total_pages = config.total_pages.max(1);
        let low_impact = config.low_impact;
        let resource_plan = infer_metal::plan_weight_only_resource_budget(
            resolved,
            infer_metal::MetalWeightOnlyResourceRequest {
                low_impact,
                memory_budget_bytes: config.memory_budget_bytes,
                system_reserve_bytes: config.system_reserve_bytes,
                allow_swap: config.allow_swap,
            },
        )?;
        let cancel = shutdown.cancel_flag();
        let max_denoising_steps = config.diffusion_max_denoising_steps.filter(|&s| s > 0);

        let serve = ServeHandle::spawn_with_engine_builder_and_shutdown(
            move || {
                let loaded = infer_metal::MetalGemma4Model::load_with_resource_plan(
                    std::path::Path::new(&model_source),
                    Some(resource_plan),
                )?;
                if loaded.image_token_id.is_some() {
                    log::info!(
                        "Gemma4 VLM ids detected: image_token_id={:?}, vision_soft_tokens_per_image={:?}; Metal image soft-token bridge enabled",
                        loaded.image_token_id,
                        loaded.vision_soft_tokens_per_image
                    );
                }
                let mut generation = loaded.generation;
                if let Some(steps) = max_denoising_steps {
                    generation.max_denoising_steps = steps;
                }
                let executor =
                    BufferedDiffusionExecutor::new_with_cancel(loaded.model, generation, cancel);
                let kv = HostPagedKvPool::new(1, total_pages, page_size);
                if low_impact {
                    let governor = infer_seam::CooperativeGovernor::new(infer_seam::StepBudget {
                        max_tokens: scheduler.chunked_prefill_size.max(1),
                        max_micros: 20_000,
                    })
                    .with_yield_every_ticks(8);
                    Ok(infer_core::Engine::with_config_and_governor(
                        executor,
                        kv,
                        scheduler,
                        Box::new(governor),
                    ))
                } else {
                    Ok(infer_core::Engine::with_config(executor, kv, scheduler))
                }
            },
            shutdown,
        )?;
        Ok((serve, tokenizer, model_id))
    }

    #[cfg(feature = "metal")]
    fn metal_deepseek_ocr_serve_handle(
        model_path: &str,
        resolved: &std::path::Path,
        config: &EngineLoadConfig,
        shutdown: infer_server::ServeShutdown,
    ) -> Result<(
        ServeHandle<BufferedDiffusionExecutor<MetalDeepseekOcrModel>, HostPagedKvPool>,
        infer_server::OpenAiTokenizer,
        String,
    )> {
        use infer_server::OpenAiTokenizer;

        if config.mtp_enabled() {
            anyhow::bail!("MTP speculative decode is only supported by the CUDA backend");
        }
        anyhow::ensure!(
            !config.kv_ssd_requested(),
            "--kv-disk: DeepSeek-OCR Metal owns no page-addressable KV tier store"
        );

        let mut tokenizer = OpenAiTokenizer::from_model_dir(resolved)?;
        // DeepSeek-OCR's tokenizer.json ships a byte-level BPE vocab but a
        // mismatched decoder, leaking `Ġ`/`Ċ` glyphs into the OCR text. Force a
        // byte-level decoder so the output is real UTF-8.
        tokenizer.force_byte_level_decoder();
        let model_id = crate::serve_engine::model_id_from_path(model_path);
        let model_source = resolved.to_string_lossy().to_string();
        let mut scheduler = config.scheduler_config();
        scheduler.num_slots = 1;
        scheduler.max_prompt_tokens = scheduler.max_prompt_tokens.min(scheduler.max_total_tokens);
        let page_size = config.page_size.max(1);
        let total_pages = config.total_pages.max(1);
        let low_impact = config.low_impact;
        let resource_plan = infer_metal::plan_weight_only_resource_budget(
            resolved,
            infer_metal::MetalWeightOnlyResourceRequest {
                low_impact,
                memory_budget_bytes: config.memory_budget_bytes,
                system_reserve_bytes: config.system_reserve_bytes,
                allow_swap: config.allow_swap,
            },
        )?;
        let cancel = shutdown.cancel_flag();
        let max_denoising_steps = config.diffusion_max_denoising_steps.filter(|&s| s > 0);

        let serve = ServeHandle::spawn_with_engine_builder_and_shutdown(
            move || {
                let loaded = infer_metal::MetalDeepseekOcrModel::load_with_resource_plan(
                    std::path::Path::new(&model_source),
                    Some(resource_plan),
                )?;
                log::info!(
                    "DeepSeek-OCR VLM loaded: image_token_id={}; Metal DeepEncoder soft-token bridge enabled",
                    loaded.image_token_id
                );
                let mut generation = loaded.generation;
                if let Some(steps) = max_denoising_steps {
                    generation.max_denoising_steps = steps;
                }
                let executor =
                    BufferedDiffusionExecutor::new_with_cancel(loaded.model, generation, cancel);
                let kv = HostPagedKvPool::new(1, total_pages, page_size);
                if low_impact {
                    let governor = infer_seam::CooperativeGovernor::new(infer_seam::StepBudget {
                        max_tokens: scheduler.chunked_prefill_size.max(1),
                        max_micros: 20_000,
                    })
                    .with_yield_every_ticks(8);
                    Ok(infer_core::Engine::with_config_and_governor(
                        executor,
                        kv,
                        scheduler,
                        Box::new(governor),
                    ))
                } else {
                    Ok(infer_core::Engine::with_config(executor, kv, scheduler))
                }
            },
            shutdown,
        )?;
        Ok((serve, tokenizer, model_id))
    }

    /// Read a CUDA checkpoint's `config.json` and classify it for `load_cuda`.
    #[cfg(feature = "cuda")]
    fn detect_cuda_model_kind(model_path: &str) -> Result<super::CudaModelKind> {
        use anyhow::Context;
        let cfg_path = std::path::Path::new(model_path).join("config.json");
        let raw = std::fs::read_to_string(&cfg_path)
            .with_context(|| format!("read {}", cfg_path.display()))?;
        let v: serde_json::Value = serde_json::from_str(&raw).context("parse config.json")?;
        Ok(super::classify_cuda_model(&v))
    }

    /// Whether `model_path`'s checkpoint takes the multiproc TP serve path when
    /// the env world size > 1: DSv4 (multi-GPU by design) and the Qwen3.5/3.6
    /// MoE hybrid (a `TpRuntime` consumer since the TP port; its executor
    /// resolves rank/world + the NCCL communicator from env at load, exactly
    /// like DSv4). Dense Qwen3 stays single-process. `false` on any
    /// config-read/parse failure (the single-process path then errors with its
    /// normal message).
    #[cfg(feature = "cuda")]
    #[must_use]
    pub fn cuda_model_takes_multiproc_serve(model_path: &str) -> bool {
        matches!(
            detect_cuda_model_kind(model_path),
            Ok(CudaModelKind::Dsv4 | CudaModelKind::Qwen35)
        )
    }

    /// DSv4's FlashMLA pool sizing (`kv_layout.rs`) isn't reconciled with the
    /// KV-budget affordability gate (`dsv4.rs`) for very large `max_seq_len` —
    /// a single slot's fixed band can outgrow the whole pool's page budget and
    /// crash the coordinator instead of degrading `num_slots` (pod-verified
    /// 2026-07-06: DeepSeek-V4-Flash-FP8's native 1,048,576-token context
    /// crashes on boot with no explicit `--max-total-tokens`). Cap the
    /// checkpoint-auto-resolved default at this known-safe ceiling; an
    /// explicit `--max-total-tokens` (e.g. the C4 budget-reject regression
    /// test's deliberate 2,000,000) still bypasses it untouched.
    #[cfg(feature = "cuda")]
    pub const DSV4_AUTO_CONTEXT_CEILING: usize = 32768;

    #[cfg(feature = "cuda")]
    #[must_use]
    pub fn cuda_model_is_dsv4(model_path: &str) -> bool {
        matches!(detect_cuda_model_kind(model_path), Ok(CudaModelKind::Dsv4))
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
    ///     length as total/num_slots, exactly like dense. Admission MUST equal the
    ///     device pool's ACTUAL page count (`effective_total_pages()`), not the old
    ///     `num_slots × 32768`. The host pool ALSO mirrors the device page SIZE
    ///     (64-tok `page_block_size`, via `effective_page_size()`), not
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
    #[cfg(feature = "cuda")]
    fn cuda_admission_total_pages(
        kind: CudaModelKind,
        config: &EngineLoadConfig,
        page_size: usize,
        paged_pool_pages: usize,
    ) -> usize {
        let ps = page_size.max(1);
        // Paged-pool models (dense Qwen3 + Qwen3.6 + DSv4 MLA latent pool): the
        // host admission pool is exactly the device pool — the profiled page
        // count, NOT a token-derived re-ceiling, and NOT floored at the requested
        // config value. DSv4's MLA pool is now free-VRAM-sized like the others.
        if matches!(
            kind,
            CudaModelKind::Qwen3Dense | CudaModelKind::Qwen35 | CudaModelKind::Dsv4
        ) {
            return paged_pool_pages.max(1);
        }
        let capacity_tokens = match kind {
            CudaModelKind::Qwen3Dense | CudaModelKind::Qwen35 | CudaModelKind::Dsv4 => {
                unreachable!("handled above")
            }
            CudaModelKind::DiffusionGemma | CudaModelKind::Qwen3MoeUnsupported => {
                config.total_pages.saturating_mul(ps)
            }
        };
        capacity_tokens.div_ceil(ps).max(config.total_pages)
    }

    /// Shared single-GPU CUDA engine builder for
    /// [`LoadedInferenceEngine::load_cuda`] and [`router_cuda`]. Sets the
    /// decode-graph default, resolves the tokenizer + model id, classifies the
    /// checkpoint, and spawns the `ServeHandle` (dispatching by kind; DSv4 is
    /// multi-GPU only and errors). Callers wrap the returned handle in either
    /// [`ServeInferenceEngine`] (the agent/OPD adapter) or
    /// [`infer_server::openai_router`] (the in-process serve loop).
    #[cfg(feature = "cuda")]
    fn cuda_serve_handle(
        model_path: &str,
        enable_cuda_graph: bool,
        config: &EngineLoadConfig,
        shutdown: infer_server::ServeShutdown,
    ) -> Result<(
        ServeHandle<CudaExecutor, CudaKvPool>,
        infer_server::OpenAiTokenizer,
        String,
    )> {
        use infer_server::OpenAiTokenizer;

        infer_cuda::set_decode_graph_default(enable_cuda_graph);
        // Resolve HF id → local cache dir, downloading if absent. Mirrors the
        // Metal path's `infer_metal::resolve_model_path` so `arle serve
        // --model-path Qwen/Qwen3.5-4B` works on CUDA without a pre-download.
        let resolved = infer_util::hf_hub::resolve_model_path(model_path)?;
        let resolved_str = resolved.to_string_lossy().to_string();
        let tokenizer = OpenAiTokenizer::from_model_dir(&resolved)?;
        let model_id = crate::serve_engine::model_id_from_path(&resolved_str);

        let model_source = resolved_str;
        let engine_config = config.clone();
        let serve = ServeHandle::spawn_with_engine_builder_and_shutdown(
            move || build_cuda_engine(&model_source, &engine_config),
            shutdown,
        )?;
        Ok((serve, tokenizer, model_id))
    }

    /// TP world size, mirroring cli::serve_multiproc::world_size_from_env —
    /// INFER_TP_SIZE, else the INFER_CUDA_DEVICES count, else 1. Every rank
    /// sees identical env, so budget division is rank-invariant.
    #[cfg(feature = "cuda")]
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

    /// Build the CUDA `Engine` (executor + admission KV pool + scheduler) for
    /// `model_path` — the ONE engine constructor every rank uses. rank 0 runs it
    /// inside [`ServeHandle::spawn_with_engine_builder`]; multiproc worker ranks
    /// run it directly on their driver thread ([`super::CudaWorkerEngine`]). All
    /// ranks building through this same helper with the same
    /// [`EngineLoadConfig`] is a lockstep invariant: any per-rank divergence in
    /// scheduler knobs diverges the deterministic planner and deadlocks NCCL.
    #[cfg(feature = "cuda")]
    pub(super) fn build_cuda_engine(
        model_path: &str,
        config: &EngineLoadConfig,
    ) -> Result<infer_core::Engine<CudaExecutor, CudaKvPool>> {
        // Single funnel for single-proc serve AND multiproc workers — flags
        // land in the statics before any CUDA context/executor exists.
        infer_cuda::apply_runtime_flags(&config.cuda);
        let kind = detect_cuda_model_kind(model_path)?;
        if matches!(kind, CudaModelKind::DiffusionGemma) {
            anyhow::bail!(
                "DiffusionGemma CUDA loading is not wired: the repository has the \
                 backend-neutral block-diffusion generate loop, but no CUDA Gemma4/\
                 DiffusionGemma forward path or weight mapping"
            );
        }
        if matches!(kind, CudaModelKind::Qwen3MoeUnsupported) {
            anyhow::bail!(
                "vanilla Qwen3-MoE (Qwen3MoeForCausalLM) is not supported on the \
                 CUDA backend; use --backend metal"
            );
        }
        // Resolve the requested KV dtype against the CUDA support matrix at the
        // engine boundary, mirroring the Metal path's `MetalKvCacheDtype::resolve`
        // (#68 T2). Admits BF16/INT8/FP8 (tq4 fails loud — see resolve); the
        // resolved dtype threads into the dense-Qwen3 constructor below (#68 T3).
        let kv_dtype = infer_cuda::CudaKvCacheDtype::resolve(config.kv_cache_dtype)?;
        if kv_dtype != infer_cuda::CudaKvCacheDtype::Bf16
            && !matches!(kind, CudaModelKind::Qwen3Dense | CudaModelKind::Qwen35)
        {
            anyhow::bail!(
                "--kv-cache-dtype {} is not supported for {kind:?}; \
                 only Qwen3Dense and Qwen35 support quantized paged KV",
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
        let executor = match kind {
            CudaModelKind::Qwen3Dense => CudaExecutor::from_qwen3_bf16_safetensors(
                model_path,
                num_slots,
                config.total_pages,
                kv_dtype,
                config.mem_fraction_static,
            )?,
            // Qwen35 clamps `num_slots` to free HBM inside the constructor
            // (`Qwen35Model::kv_budget_num_slots`, unified with DSv4 via the
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
            )?,
            // DSv4 multi-rank serve. The DSv4 executor resolves its TP
            // rank/world-size + EP expert split + NCCL communicator from the
            // environment during construction (`INFER_TP_RANK` /
            // `INFER_TP_SIZE` / `INFER_CUDA_DEVICES`, plus
            // `INFER_NCCL_UNIQUE_ID` / `INFER_NCCL_ID_FILE` rendezvous) — set
            // by the multiproc coordinator/launcher before this runs. On a
            // single GPU (world_size==1) it loads as one rank. DSv4 owns its
            // MLA KV state inside the forward, so the host `CudaKvPool` is
            // only present to satisfy the `submit(.., &mut dyn KvPool)`
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
            CudaModelKind::DiffusionGemma | CudaModelKind::Qwen3MoeUnsupported => {
                unreachable!("checked before CUDA executor build")
            }
        };
        let mut executor = executor;
        // Fail loud on flag+model combos the executor would otherwise silently
        // ignore, BEFORE any budget setter runs.
        if config.kv_recall {
            anyhow::ensure!(
                matches!(kind, CudaModelKind::Qwen3Dense | CudaModelKind::Qwen35),
                "--kv-recall is not wired for {kind:?}; it would be silently ignored"
            );
        }
        if config.slot_oversubscription {
            anyhow::ensure!(
                executor.kv_slot_tier_enabled(),
                "--kv-oversubscription: {kind:?} has no whole-slot tier \
                 (dense Qwen3 preempts via its page tier; DSv4 preempts via \
                 KV-overflow requeue + prefix-state pool restore, #154 Phase 2b)"
            );
        }
        // L2 budget: deployment-total → per-rank share, resolved at the ONE
        // constructor every rank runs (world size is env-identical per rank, so
        // the division is lockstep-deterministic).
        let world = tp_world_size();
        let dram_rank_bytes = infer_cuda::resolve_dram_budget_bytes(config.kv_dram, world);
        executor.set_kv_tier_budget_bytes(dram_rank_bytes);
        // Opt-in L3 NVMe spill (`--kv-disk`): attached HERE so single-proc
        // rank 0 and multiproc worker ranks agree (the old post-spawn serve-layer
        // hook never reached multiproc workers → workers served with a zero-page
        // tier and every demote fell back to recompute). Must follow the budget
        // setters above: the tier-store arms rebuild their store on re-budget,
        // which would drop an earlier disk attach. Must PRECEDE `set_kv_recall`:
        // Qwen3.6 lazily builds its recall tier on first enable, and only a disk
        // root stashed beforehand lets it take the durable prior-session reload
        // (`tier.load`) instead of a fresh create.
        let kv_disk = config.kv_ssd_spill(world, infer_cuda::default_t2_budget_bytes)?;
        if let Some((root, budget)) = &kv_disk {
            anyhow::ensure!(
                executor.set_kv_tier_disk(root.clone(), *budget),
                "--kv-disk: the loaded model has no KV tier store to spill \
                 (Qwen3-dense page tier + Qwen3.6/DSv4 slot tier; a budget \
                 below one page also lands here — raise --kv-disk-limit)"
            );
        }
        // Session KV-recall ("infinite memory", `--kv-recall`, default off). Off →
        // the decode hot path is byte-identical (CUDA is the Stable backend). Set
        // here (the ONE engine constructor every rank uses) so single-GPU + TP
        // agree, and after the disk attach so the recall tier sees the stash.
        executor.set_kv_recall(config.kv_recall)?;
        log::info!(
            "KV tiers: dtype={} | L1 mem_fraction_static={} | L2 {dram_rank_bytes}B/rank \
             (deployment {:?}, world {world}) | L3 {} | features: prefix{}{}",
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
            if config.kv_recall { ",recall" } else { "" },
        );
        // The DSv4 constructor may clamp slots below the request (dynamic KV
        // mem budget, NCCL min-reduced ⇒ identical on every rank). Scheduler +
        // admission pool MUST follow the effective count: admitting to a slot
        // the executor has no arena for fails at submit, and (lockstep) a
        // scheduler-visible capacity that diverged from the executor's would
        // diverge the deterministic planner.
        let num_slots = executor.effective_num_slots().unwrap_or(num_slots);
        // Paged-pool models (dense Qwen3 + Qwen3.6 + DSv4 MLA pool) profile their
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
        if matches!(kind, CudaModelKind::Qwen3Dense | CudaModelKind::Qwen35)
            && paged_pool_pages != config.total_pages
        {
            log::info!(
                "CUDA {kind:?}: profiled full-attn KV pool {paged_pool_pages} pages from measured \
                 free VRAM (requested total_pages={}, mem_fraction_static={}); host admission \
                 follows the device pool",
                config.total_pages,
                config.mem_fraction_static
            );
        }
        // M2: scheduler_config() raised max_prompt_tokens from the REQUESTED
        // total_pages; on the dense-Qwen3/Qwen3.6 arm the device pool is profiled
        // from free VRAM and may be SMALLER, so bind the ingress caps DOWN to the
        // profiled pool capacity and to max_total_tokens — else a long prompt
        // clears ingress, can't draw enough pages, and silently completes empty.
        // Mirrors DSv4 (max_seq clamp) and Metal (resource-guard clamp).
        if matches!(kind, CudaModelKind::Qwen3Dense | CudaModelKind::Qwen35) {
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
        Ok(infer_core::Engine::with_config(executor, kv, scheduler))
    }

    /// New tokens (+ optional behavior logprob) since the last drain, per handle.
    #[cfg(feature = "cuda")]
    type PendingTokens = std::rc::Rc<
        std::cell::RefCell<
            std::collections::HashMap<infer_core::RequestHandle, Vec<(u32, Option<f32>)>>,
        >,
    >;

    /// One multiproc worker rank's engine. Steps SYNCHRONOUSLY per relayed
    /// `TickAdmissions` so every rank admits at the same step index (lockstep).
    /// Rank 0 (`owns_output`) also tracks + emits completions; followers
    /// discard their TP-replicated tokens.
    #[cfg(feature = "cuda")]
    pub struct CudaWorkerEngine {
        engine: infer_core::Engine<CudaExecutor, CudaKvPool>,
        /// Rank 0 owns the visible output; followers skip all output bookkeeping.
        owns_output: bool,
        /// engine handle -> coordinator request_id (output owner only); removed
        /// once its terminal delta is emitted.
        tracked: std::collections::HashMap<infer_core::RequestHandle, u64>,
        /// Fed by the token observer inside `engine.step()`, drained right after
        /// by `drain_completions()` (same thread, never concurrent).
        pending: PendingTokens,
    }

    #[cfg(feature = "cuda")]
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
                    pending
                        .borrow_mut()
                        .entry(handle)
                        .or_default()
                        .push((token.token, token.logprob));
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

        /// Whether the engine has no queued/active/in-flight work — same state
        /// every rank evaluates, so all sides step or skip symmetrically per tick.
        #[must_use]
        pub fn is_idle(&self) -> bool {
            self.engine.is_idle()
        }

        /// Run exactly one scheduler tick (apply previous output → admit
        /// waiting → build plan → submit forward).
        pub fn step(&mut self) -> Result<()> {
            self.engine.step()
        }

        /// Snapshot prefix-cache stats for the coordinator `/v1/stats` relay.
        pub fn prefix_cache_stats(&self) -> infer_core::PrefixCacheStats {
            self.engine.prefix_cache_stats()
        }

        /// Snapshot throughput counters for the coordinator `/v1/stats` relay.
        pub fn throughput_stats(&self) -> infer_core::ThroughputStats {
            self.engine.throughput_stats()
        }

        /// Snapshot KV tier counters for the coordinator `/v1/stats` relay.
        pub fn kv_tier_stats(&self) -> infer_core::KvTierStats {
            self.engine.kv_tier_stats()
        }

        /// Snapshot KV system counters for the coordinator `/v1/stats` relay.
        pub fn kv_system_metrics(&self) -> infer_core::KvSystemMetrics {
            self.engine.kv_system_metrics()
        }

        /// Snapshot spec-decode counters for the coordinator `/v1/stats` relay.
        pub fn spec_decode_stats(&self) -> infer_seam::SpecDecodeStats {
            self.engine.spec_decode_stats()
        }

        /// Snapshot operator-dispatch counters for the coordinator `/v1/stats` relay.
        pub fn operator_dispatch_stats(&self) -> infer_seam::OperatorDispatchStats {
            self.engine.operator_dispatch_stats()
        }

        /// Backend artifact identity for the coordinator `/v1/stats` relay.
        pub fn artifact_identity(&self) -> infer_seam::BackendArtifactIdentity {
            self.engine.artifact_identity()
        }

        /// Active (running) request count.
        pub fn active_count(&self) -> usize {
            self.engine.active_count()
        }

        /// Waiting (queued) request count.
        pub fn waiting_count(&self) -> usize {
            self.engine.waiting_count()
        }

        /// Free KV pages remaining.
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
                    .map(|&(_, lp)| lp)
                    .collect::<Option<Vec<f32>>>()
                    .unwrap_or_default();
                out.push((
                    request_id,
                    infer_server::RelayCompletionDelta {
                        text_delta: String::new(),
                        token_ids: new_tokens.into_iter().map(|(t, _)| t).collect(),
                        logprobs,
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

    /// Single-GPU CUDA serve router. Builds the same `ServeHandle` as
    /// [`LoadedInferenceEngine::load_cuda`] via [`cuda_serve_handle`], then wraps
    /// it in [`infer_server::coordinator_local_router`].
    #[cfg(feature = "cuda")]
    fn router_cuda(
        model_path: &str,
        enable_cuda_graph: bool,
        config: &EngineLoadConfig,
        shutdown: infer_server::ServeShutdown,
    ) -> Result<(axum::Router, Arc<LoadedInferenceEngine>)> {
        let (serve, tokenizer, model_id) =
            cuda_serve_handle(model_path, enable_cuda_graph, config, shutdown)?;
        let serve_engine = ServeInferenceEngine::new(model_id.clone(), tokenizer.clone(), serve);
        let serve_arc = serve_engine.serve_arc();
        let engine = Arc::new(LoadedInferenceEngine::Cuda(serve_engine));
        let router = infer_server::coordinator_local_router(
            serve_arc,
            tokenizer,
            model_id,
            config.max_thinking_tokens,
            None,
        )
        .merge(super::raw_logits_route::raw_logits_router(engine.clone()));
        Ok((router, engine))
    }

    /// Resolve `model_path` to a `.gguf` checkpoint: either the file itself or a
    /// directory containing exactly one `*.gguf`. No HF-repo resolution (that
    /// is the Metal facade's surface); a plain file-path check with a clear
    /// error is the MVP contract for GGUF-only backends.
    #[cfg(any(feature = "hip", feature = "vulkan"))]
    fn resolve_gguf_path(model_path: &str, backend_label: &str) -> Result<std::path::PathBuf> {
        use anyhow::{Context, bail, ensure};

        let path = std::path::Path::new(model_path);
        if path.is_file() {
            ensure!(
                path.extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf")),
                "{backend_label} backend serves GGUF checkpoints only; {model_path} is not a .gguf file"
            );
            return Ok(path.to_path_buf());
        }
        if path.is_dir() {
            let mut ggufs: Vec<std::path::PathBuf> = std::fs::read_dir(path)
                .with_context(|| format!("read model dir {model_path}"))?
                .filter_map(|entry| entry.ok().map(|e| e.path()))
                .filter(|p| {
                    p.is_file()
                        && p.extension()
                            .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
                })
                .collect();
            return match ggufs.len() {
                1 => Ok(ggufs.remove(0)),
                0 => bail!(
                    "no .gguf file in {model_path}; the {backend_label} backend serves GGUF checkpoints only"
                ),
                n => bail!("{n} .gguf files in {model_path}; pass the .gguf file path explicitly"),
            };
        }
        bail!(
            "{backend_label} model path {model_path} not found \
             (expected a .gguf file or a directory containing exactly one)"
        )
    }

    /// Shared HIP engine builder for [`LoadedInferenceEngine::load_hip`] and
    /// [`router_hip`], mirroring [`cuda_serve_handle`]. Resolves the `.gguf`
    /// checkpoint + sibling `tokenizer.json` (the GGUF's directory), then
    /// spawns the `ServeHandle` over [`infer_hip::load_dsv4_gguf`], which
    /// returns the matched executor + host KV pool pair. DSv4 is a slot-arena
    /// model: per-slot depth is the per-request `max_total_tokens` budget.
    #[cfg(feature = "hip")]
    fn hip_serve_handle(
        model_path: &str,
        config: &EngineLoadConfig,
        shutdown: infer_server::ServeShutdown,
    ) -> Result<(
        ServeHandle<HipDsv4Executor, HipKvPool>,
        infer_server::OpenAiTokenizer,
        String,
    )> {
        use infer_server::OpenAiTokenizer;

        if config.mtp_enabled() {
            anyhow::bail!("MTP speculative decode is only supported by the CUDA backend");
        }
        anyhow::ensure!(
            !config.kv_ssd_requested(),
            "--kv-disk: the HIP backend has no KV tier store"
        );
        let gguf_path = resolve_gguf_path(model_path, "HIP")?;
        let tokenizer_dir = gguf_path
            .parent()
            .filter(|dir| !dir.as_os_str().is_empty())
            .map_or_else(
                || std::path::PathBuf::from("."),
                std::path::Path::to_path_buf,
            );
        let tokenizer = OpenAiTokenizer::from_model_dir(&tokenizer_dir)?;
        let model_id = crate::serve_engine::model_id_from_path(model_path);

        let mut scheduler = config.scheduler_config();
        scheduler.num_slots = 1;
        let num_slots = 1;
        let max_seq_len = config.max_total_tokens;
        let serve = ServeHandle::spawn_with_engine_builder_and_shutdown(
            move || {
                let (executor, kv) = infer_hip::load_dsv4_gguf(&gguf_path, num_slots, max_seq_len)?;
                Ok(infer_core::Engine::with_config(executor, kv, scheduler))
            },
            shutdown,
        )?;
        Ok((serve, tokenizer, model_id))
    }

    /// Shared Vulkan engine builder for [`LoadedInferenceEngine::load_vulkan`]
    /// and [`router_vulkan`]. Resolves the `.gguf` checkpoint + sibling
    /// `tokenizer.json`, then spawns the `ServeHandle` over
    /// [`infer_vulkan::load_qwen3_gguf`]. P7 wires the CLI endpoint; the
    /// numeric Vulkan forward remains pending AIPC/on-box validation and fails
    /// loud inside the backend.
    #[cfg(feature = "vulkan")]
    fn vulkan_serve_handle(
        model_path: &str,
        config: &EngineLoadConfig,
        shutdown: infer_server::ServeShutdown,
    ) -> Result<(
        ServeHandle<VulkanExecutor, VulkanKvPool>,
        infer_server::OpenAiTokenizer,
        String,
    )> {
        use infer_server::OpenAiTokenizer;

        if config.mtp_enabled() {
            anyhow::bail!("MTP speculative decode is only supported by the CUDA backend");
        }
        if let Some(cap) = config.vulkan_submit_cap {
            infer_vulkan::forward::set_submit_cap(cap);
        }
        anyhow::ensure!(
            !config.kv_ssd_requested(),
            "--kv-disk: the Vulkan backend has no KV tier store"
        );
        let gguf_path = resolve_gguf_path(model_path, "Vulkan")?;
        let tokenizer_dir = gguf_path
            .parent()
            .filter(|dir| !dir.as_os_str().is_empty())
            .map_or_else(
                || std::path::PathBuf::from("."),
                std::path::Path::to_path_buf,
            );
        let tokenizer = OpenAiTokenizer::from_model_dir(&tokenizer_dir)?;
        let model_id = crate::serve_engine::model_id_from_path(model_path);

        let mut scheduler = config.scheduler_config();
        scheduler.num_slots = 1;
        let num_slots = 1;
        let max_seq_len = config.max_total_tokens;
        let serve = ServeHandle::spawn_with_engine_builder_and_shutdown(
            move || {
                let (executor, kv) =
                    infer_vulkan::load_qwen3_gguf(&gguf_path, num_slots, max_seq_len)?;
                Ok(infer_core::Engine::with_config(executor, kv, scheduler))
            },
            shutdown,
        )?;
        Ok((serve, tokenizer, model_id))
    }

    /// HIP serve router. Builds the same `ServeHandle` as
    /// [`LoadedInferenceEngine::load_hip`] via [`hip_serve_handle`], then wraps
    /// it in [`infer_server::coordinator_local_router`]. Mirrors [`router_cuda`].
    #[cfg(feature = "hip")]
    fn router_hip(
        model_path: &str,
        config: &EngineLoadConfig,
        shutdown: infer_server::ServeShutdown,
    ) -> Result<axum::Router> {
        let (serve, tokenizer, model_id) = hip_serve_handle(model_path, config, shutdown)?;
        Ok(infer_server::coordinator_local_router(
            Arc::new(serve),
            tokenizer,
            model_id,
            config.max_thinking_tokens,
            None,
        ))
    }

    /// Vulkan serve router. Builds the same `ServeHandle` as
    /// [`LoadedInferenceEngine::load_vulkan`] via [`vulkan_serve_handle`], then
    /// wraps it in [`infer_server::coordinator_local_router`]. Mirrors [`router_hip`].
    #[cfg(feature = "vulkan")]
    fn router_vulkan(
        model_path: &str,
        config: &EngineLoadConfig,
        shutdown: infer_server::ServeShutdown,
    ) -> Result<axum::Router> {
        let (serve, tokenizer, model_id) = vulkan_serve_handle(model_path, config, shutdown)?;
        Ok(infer_server::coordinator_local_router(
            Arc::new(serve),
            tokenizer,
            model_id,
            config.max_thinking_tokens,
            None,
        ))
    }

    /// Portable CPU serve router: the placeholder `MetalExecutor` over the real
    /// backend-neutral host paged KV pool (no MLX, no CUDA), wrapped in
    /// [`infer_server::coordinator_local_router`]. Mirrors
    /// [`LoadedInferenceEngine::load_cpu`].
    #[cfg(all(
        feature = "cpu",
        not(feature = "metal"),
        not(feature = "cuda"),
        not(feature = "hip"),
        not(feature = "vulkan")
    ))]
    fn router_cpu(
        model_path: &str,
        config: &EngineLoadConfig,
        shutdown: infer_server::ServeShutdown,
    ) -> Result<axum::Router> {
        use infer_server::OpenAiTokenizer;

        if config.mtp_enabled() {
            anyhow::bail!("MTP speculative decode is only supported by the CUDA backend");
        }
        let tokenizer = OpenAiTokenizer::from_model_dir(model_path)?;
        let model_id = crate::serve_engine::model_id_from_path(model_path);
        let executor = MetalExecutor::new();
        let kv = HostPagedKvPool::new(
            config.hot_workspace_slots(),
            config.total_pages,
            config.page_size,
        );
        let serve =
            ServeHandle::spawn_with_shutdown(executor, kv, config.scheduler_config(), shutdown);
        Ok(infer_server::coordinator_local_router(
            Arc::new(serve),
            tokenizer,
            model_id,
            config.max_thinking_tokens,
            None,
        ))
    }

    impl InferenceEngine for LoadedInferenceEngine {
        fn model_id(&self) -> &str {
            match self {
                #[cfg(feature = "metal")]
                Self::Metal(engine) => engine.model_id(),
                #[cfg(feature = "metal")]
                Self::MetalDiffusionGemma(engine) => engine.model_id(),
                #[cfg(feature = "metal")]
                Self::MetalGemma4(engine) => engine.model_id(),
                #[cfg(feature = "metal")]
                Self::MetalDeepseekOcr(engine) => engine.model_id(),
                #[cfg(feature = "cuda")]
                Self::Cuda(engine) => engine.model_id(),
                #[cfg(feature = "hip")]
                Self::Hip(engine) => engine.model_id(),
                #[cfg(feature = "vulkan")]
                Self::Vulkan(engine) => engine.model_id(),
                #[cfg(all(feature = "cpu", not(feature = "metal")))]
                Self::Cpu(engine) => engine.model_id(),
            }
        }

        fn complete(&mut self, req: CompletionRequest) -> Result<CompletionOutput> {
            match self {
                #[cfg(feature = "metal")]
                Self::Metal(engine) => engine.complete(req),
                #[cfg(feature = "metal")]
                Self::MetalDiffusionGemma(engine) => engine.complete(req),
                #[cfg(feature = "metal")]
                Self::MetalGemma4(engine) => engine.complete(req),
                #[cfg(feature = "metal")]
                Self::MetalDeepseekOcr(engine) => engine.complete(req),
                #[cfg(feature = "cuda")]
                Self::Cuda(engine) => engine.complete(req),
                #[cfg(feature = "hip")]
                Self::Hip(engine) => engine.complete(req),
                #[cfg(feature = "vulkan")]
                Self::Vulkan(engine) => engine.complete(req),
                #[cfg(all(feature = "cpu", not(feature = "metal")))]
                Self::Cpu(engine) => engine.complete(req),
            }
        }

        fn complete_multimodal_chat(
            &mut self,
            req: MultimodalChatRequest,
        ) -> Result<CompletionOutput> {
            match self {
                #[cfg(feature = "metal")]
                Self::Metal(engine) => engine.complete_multimodal_chat(req),
                #[cfg(feature = "metal")]
                Self::MetalDiffusionGemma(engine) => engine.complete_multimodal_chat(req),
                #[cfg(feature = "metal")]
                Self::MetalGemma4(engine) => engine.complete_multimodal_chat(req),
                #[cfg(feature = "metal")]
                Self::MetalDeepseekOcr(engine) => engine.complete_multimodal_chat(req),
                #[cfg(feature = "cuda")]
                Self::Cuda(engine) => engine.complete_multimodal_chat(req),
                #[cfg(feature = "hip")]
                Self::Hip(engine) => engine.complete_multimodal_chat(req),
                #[cfg(feature = "vulkan")]
                Self::Vulkan(engine) => engine.complete_multimodal_chat(req),
                #[cfg(all(feature = "cpu", not(feature = "metal")))]
                Self::Cpu(engine) => engine.complete_multimodal_chat(req),
            }
        }

        fn complete_stream(
            &mut self,
            req: CompletionRequest,
            tx: UnboundedSender<CompletionStreamDelta>,
        ) -> Result<()> {
            match self {
                #[cfg(feature = "metal")]
                Self::Metal(engine) => engine.complete_stream(req, tx),
                #[cfg(feature = "metal")]
                Self::MetalDiffusionGemma(engine) => engine.complete_stream(req, tx),
                #[cfg(feature = "metal")]
                Self::MetalGemma4(engine) => engine.complete_stream(req, tx),
                #[cfg(feature = "metal")]
                Self::MetalDeepseekOcr(engine) => engine.complete_stream(req, tx),
                #[cfg(feature = "cuda")]
                Self::Cuda(engine) => engine.complete_stream(req, tx),
                #[cfg(feature = "hip")]
                Self::Hip(engine) => engine.complete_stream(req, tx),
                #[cfg(feature = "vulkan")]
                Self::Vulkan(engine) => engine.complete_stream(req, tx),
                #[cfg(all(feature = "cpu", not(feature = "metal")))]
                Self::Cpu(engine) => engine.complete_stream(req, tx),
            }
        }

        fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
            match self {
                #[cfg(feature = "metal")]
                Self::Metal(engine) => engine.tokenize(text),
                #[cfg(feature = "metal")]
                Self::MetalDiffusionGemma(engine) => engine.tokenize(text),
                #[cfg(feature = "metal")]
                Self::MetalGemma4(engine) => engine.tokenize(text),
                #[cfg(feature = "metal")]
                Self::MetalDeepseekOcr(engine) => engine.tokenize(text),
                #[cfg(feature = "cuda")]
                Self::Cuda(engine) => engine.tokenize(text),
                #[cfg(feature = "hip")]
                Self::Hip(engine) => engine.tokenize(text),
                #[cfg(feature = "vulkan")]
                Self::Vulkan(engine) => engine.tokenize(text),
                #[cfg(all(feature = "cpu", not(feature = "metal")))]
                Self::Cpu(engine) => engine.tokenize(text),
            }
        }

        fn render_chat_prompt(&self, messages: &[ChatPromptMessage]) -> Result<String> {
            match self {
                #[cfg(feature = "metal")]
                Self::Metal(engine) => engine.render_chat_prompt(messages),
                #[cfg(feature = "metal")]
                Self::MetalDiffusionGemma(engine) => engine.render_chat_prompt(messages),
                #[cfg(feature = "metal")]
                Self::MetalGemma4(engine) => engine.render_chat_prompt(messages),
                #[cfg(feature = "metal")]
                Self::MetalDeepseekOcr(engine) => engine.render_chat_prompt(messages),
                #[cfg(feature = "cuda")]
                Self::Cuda(engine) => engine.render_chat_prompt(messages),
                #[cfg(feature = "hip")]
                Self::Hip(engine) => engine.render_chat_prompt(messages),
                #[cfg(feature = "vulkan")]
                Self::Vulkan(engine) => engine.render_chat_prompt(messages),
                #[cfg(all(feature = "cpu", not(feature = "metal")))]
                Self::Cpu(engine) => engine.render_chat_prompt(messages),
            }
        }

        fn telemetry(&self) -> EngineTelemetry {
            match self {
                #[cfg(feature = "metal")]
                Self::Metal(engine) => engine.telemetry(),
                #[cfg(feature = "metal")]
                Self::MetalDiffusionGemma(engine) => engine.telemetry(),
                #[cfg(feature = "metal")]
                Self::MetalGemma4(engine) => engine.telemetry(),
                #[cfg(feature = "metal")]
                Self::MetalDeepseekOcr(engine) => engine.telemetry(),
                #[cfg(feature = "cuda")]
                Self::Cuda(engine) => engine.telemetry(),
                #[cfg(feature = "hip")]
                Self::Hip(engine) => engine.telemetry(),
                #[cfg(feature = "vulkan")]
                Self::Vulkan(engine) => engine.telemetry(),
                #[cfg(all(feature = "cpu", not(feature = "metal")))]
                Self::Cpu(engine) => engine.telemetry(),
            }
        }
    }
}

#[cfg(feature = "cuda")]
pub use backend::CudaWorkerEngine;
#[cfg(any(
    feature = "metal",
    feature = "cuda",
    feature = "hip",
    feature = "vulkan",
    feature = "cpu"
))]
pub use backend::LoadedInferenceEngine;
#[cfg(feature = "cuda")]
pub use backend::cuda_model_takes_multiproc_serve;
#[cfg(any(
    feature = "metal",
    feature = "cuda",
    feature = "hip",
    feature = "vulkan",
    feature = "cpu"
))]
pub(crate) use backend::router_for_backend;
#[cfg(feature = "cuda")]
pub use backend::{DSV4_AUTO_CONTEXT_CEILING, cuda_model_is_dsv4};

#[cfg(test)]
mod kv_disk_budget_tests {
    use super::EngineLoadConfig;

    // #158: a derived budget of 0 (disk under the reserve floor) must degrade
    // to no-tier instead of fail-closing the engine build.
    #[test]
    fn zero_derived_budget_degrades_to_no_tier() {
        let cfg = EngineLoadConfig {
            kv_ssd_root: Some("/tmp/kv".into()),
            ..EngineLoadConfig::default()
        };
        assert_eq!(cfg.kv_ssd_spill(1, |_, _| 0).unwrap(), None);
        assert_eq!(
            cfg.kv_ssd_spill(2, |_, _| 4096).unwrap(),
            Some(("/tmp/kv".into(), 2048))
        );
    }
}
