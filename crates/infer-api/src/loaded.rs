//! `LoadedInferenceEngine` — the backend-dispatching public engine.
//!
//! A feature-gated enum over the available backends (`metal`/`cuda`/`hip`/`vulkan`/`cpu`,
//! selected at compile time) with a `load(model_path, enable_cuda_graph)`
//! constructor dispatching to the active variant. [`EngineLoadConfig`] is always
//! available; the enum + impls require a backend feature.

/// Slot / page configuration for [`LoadedInferenceEngine::load_with_config`].
///
/// Serde: the multiproc coordinator serializes its resolved config into
/// `ARLE_WORKER_ENGINE_CONFIG` so worker ranks build their engines from the
/// SAME values — any divergence (slots, budgets, chunk size) diverges the
/// deterministic planner across ranks and deadlocks the NCCL lockstep.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvCacheDtype {
    /// Backend default. Metal resolves this to INT8 after the Metal int8 gate;
    /// other backends keep their established default.
    #[default]
    Auto,
    /// Native BF16 / model-dtype KV cache.
    Bf16,
    /// INT8 KV cache. Metal uses MLX affine 8-bit groups; CUDA support is a
    /// separate backend implementation detail and must not be silently assumed.
    Int8,
}

/// Slot / page configuration for [`LoadedInferenceEngine::load_with_config`].
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct EngineLoadConfig {
    /// Logical request slots.
    pub num_slots: usize,
    /// Physical KV pages.
    pub total_pages: usize,
    /// Tokens per KV page.
    pub page_size: usize,
    /// Max prompt tokens accepted at ingress.
    pub max_prompt_tokens: usize,
    /// Max prompt+generated tokens for one request.
    pub max_total_tokens: usize,
    /// Per-request prefill chunk size.
    pub chunked_prefill_size: usize,
    /// `Some(n)` = MTP spec decode on with draft depth `n`; `None` = off.
    pub mtp_draft_tokens: Option<usize>,
    /// Requested KV-cache dtype. Backends resolve `Auto` inside their own
    /// builder so the service/scheduler layers stay device-neutral.
    #[serde(default)]
    pub kv_cache_dtype: KvCacheDtype,
    /// Host-RAM budget for the T1 prefix-KV tier in bytes. `None` keeps the
    /// backend default (CUDA dense: 4 GiB, default-on); `Some(0)` disables
    /// the tier. Backends without a tier store ignore it.
    #[serde(default)]
    pub kv_t1_budget_bytes: Option<usize>,
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
}

impl Default for EngineLoadConfig {
    fn default() -> Self {
        // Conservative local-serving defaults shared by every backend builder.
        Self {
            num_slots: 4,
            total_pages: 8192,
            page_size: 16,
            max_prompt_tokens: 32_768,
            max_total_tokens: 65_536,
            chunked_prefill_size: 64,
            mtp_draft_tokens: None,
            kv_cache_dtype: KvCacheDtype::Auto,
            kv_t1_budget_bytes: None,
            memory_budget_bytes: None,
            system_reserve_bytes: None,
            allow_swap: false,
            low_impact: false,
        }
    }
}

/// Which CUDA forward a checkpoint needs, classified from its `config.json`.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CudaModelKind {
    /// Dense Qwen3 (BF16).
    Qwen3Dense,
    /// Qwen3.5 / 3.6 MoE (BF16).
    Qwen3Moe,
    /// DeepSeek-V4-Flash (multi-GPU only).
    Dsv4,
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
    let expert_count = |key: &str| v.get(key).and_then(|x| x.as_u64()).unwrap_or(0);
    let is_moe = arch_contains("Moe")
        || expert_count("num_experts") > 0
        || expert_count("n_routed_experts") > 0;
    if is_moe {
        CudaModelKind::Qwen3Moe
    } else {
        CudaModelKind::Qwen3Dense
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
        assert_eq!(
            classify_cuda_model(&json!({"architectures": ["Qwen3MoeForCausalLM"]})),
            CudaModelKind::Qwen3Moe
        );
        assert_eq!(
            classify_cuda_model(&json!({"model_type": "qwen3", "num_experts": 128})),
            CudaModelKind::Qwen3Moe
        );
        assert_eq!(
            classify_cuda_model(
                &json!({"architectures": ["DeepseekV4ForCausalLM"], "model_type": "deepseek_v4"})
            ),
            CudaModelKind::Dsv4
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
    use anyhow::Result;
    use infer_core::SchedulerConfig;
    use infer_server::ServeHandle;
    use tokio::sync::mpsc::UnboundedSender;

    #[cfg(feature = "cuda")]
    use super::CudaModelKind;
    use super::EngineLoadConfig;
    #[cfg(feature = "metal")]
    use super::KvCacheDtype;
    use crate::serve_engine::ServeInferenceEngine;
    use crate::types::{
        CompletionOutput, CompletionRequest, CompletionStreamDelta, EngineTelemetry,
        InferenceEngine,
    };

    #[cfg(feature = "cuda")]
    use infer_cuda::{CudaExecutor, CudaKvPool};
    #[cfg(feature = "hip")]
    use infer_hip::{HipDsv4Executor, HipKvPool};
    #[cfg(feature = "metal")]
    use infer_metal::{MetalExecutor, MetalKvPool};
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
            let mut config = SchedulerConfig::for_slots(self.num_slots);
            config.max_prompt_tokens = self.max_prompt_tokens;
            config.max_total_tokens = self.max_total_tokens;
            config.chunked_prefill_size = self.chunked_prefill_size;
            config
        }
    }

    /// Backend-dispatching public engine; one variant per compiled backend.
    pub enum LoadedInferenceEngine {
        /// Metal backend (Apple Silicon, MLX). Fully wired and runnable.
        #[cfg(feature = "metal")]
        Metal(ServeInferenceEngine<MetalExecutor, MetalKvPool>),
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
        pub fn load(model_path: &str, enable_cuda_graph: bool) -> Result<Self> {
            Self::load_with_config(model_path, enable_cuda_graph, EngineLoadConfig::default())
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

        /// Reload the engine's device weights from the host snapshot (OPD teacher
        /// weight time-share). CUDA-only; Metal/CPU bail.
        pub fn reload_engine_weights(&self) -> Result<()> {
            match self {
                #[cfg(feature = "cuda")]
                Self::Cuda(engine) => engine.reload_engine_weights(),
                #[cfg(feature = "metal")]
                Self::Metal(_) => anyhow::bail!("reload_engine_weights is only available on CUDA"),
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

        /// Fold a fresh student LoRA update into the resident Qwen3.5/3.6 q/v
        /// projection weights (OPD per-step re-merge). CUDA-only: the Metal /
        /// CPU arms reject it.
        ///
        /// The CUDA forward path implements the merge (see
        /// [`infer_cuda::CudaExecutor::remerge_student_lora`] +
        /// `infer_cuda::qwen35::Qwen35Model::remerge_student_lora`): the resident
        /// q/v `DeviceMatrix` weights are re-merged in place from a pristine
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

        #[cfg(feature = "metal")]
        fn load_metal(model_path: &str, config: &EngineLoadConfig) -> Result<Self> {
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
            // `INFER_CUDA_DECODE_GRAPH` overrides, `warmup` gates it off under TP/MoE.
            // Shares the engine builder with `router_cuda` via `cuda_serve_handle`.
            let (serve, tokenizer, model_id) = cuda_serve_handle(
                model_path,
                enable_cuda_graph,
                config,
                &crate::serve::ServeKvSsdOptions::default(),
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

            if config.mtp_draft_tokens.is_some() {
                anyhow::bail!("MTP speculative decode is only supported by the CUDA backend");
            }
            // CPU smoke: placeholder executor over a real host KV pool; still
            // needs a tokenizer dir for encode/decode.
            let tokenizer = OpenAiTokenizer::from_model_dir(model_path)?;
            let model_id = crate::serve_engine::model_id_from_path(model_path);
            let executor = MetalExecutor::new();
            let kv = HostPagedKvPool::new(config.num_slots, config.total_pages, config.page_size);
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
        kv_ssd: &crate::serve::ServeKvSsdOptions,
        shutdown: infer_server::ServeShutdown,
    ) -> Result<axum::Router> {
        // The T2 disk tier is consumed by the CUDA arm only; every other
        // backend fails closed on an explicit request instead of silently
        // serving without it.
        #[cfg(not(all(not(feature = "metal"), feature = "cuda")))]
        anyhow::ensure!(
            !kv_ssd.requested(),
            "--kv-ssd-path: the T2 KV tier is CUDA-only today (Metal pending #74/#83)"
        );

        #[cfg(feature = "metal")]
        {
            let _ = enable_cuda_graph;
            return router_metal(model_path, &config, shutdown);
        }

        #[cfg(all(not(feature = "metal"), feature = "cuda"))]
        {
            return router_cuda(model_path, enable_cuda_graph, &config, kv_ssd, shutdown);
        }

        #[cfg(all(not(feature = "metal"), not(feature = "cuda"), feature = "hip"))]
        {
            let _ = enable_cuda_graph;
            return router_hip(model_path, &config, shutdown);
        }

        #[cfg(all(
            not(feature = "metal"),
            not(feature = "cuda"),
            not(feature = "hip"),
            feature = "vulkan"
        ))]
        {
            let _ = enable_cuda_graph;
            return router_vulkan(model_path, &config, shutdown);
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
            return router_cpu(model_path, &config, shutdown);
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

        if config.mtp_draft_tokens.is_some() {
            anyhow::bail!("MTP speculative decode is only supported by the CUDA backend");
        }
        let metal_kv_dtype = match config.kv_cache_dtype {
            KvCacheDtype::Auto | KvCacheDtype::Int8 => infer_metal::MetalKvCacheDtype::Int8,
            KvCacheDtype::Bf16 => infer_metal::MetalKvCacheDtype::Bf16,
        };
        let resolved = infer_metal::resolve_model_path(model_path)?;
        let tokenizer = OpenAiTokenizer::from_model_dir(&resolved)?;
        let model_id = crate::serve_engine::model_id_from_path(model_path);

        let model_source = resolved.to_string_lossy().to_string();
        let mut scheduler = config.scheduler_config();
        let num_slots = config.num_slots;
        let page_size = config.page_size;
        let low_impact = config.low_impact;
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
        let serve = ServeHandle::spawn_with_engine_builder_and_shutdown(
            move || {
                let executor =
                    MetalExecutor::from_model_path_with_kv_cache_dtype_and_resource_plan(
                        &model_source,
                        metal_kv_dtype,
                        resource_plan,
                    )?;
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
        let (serve, tokenizer, model_id) = metal_serve_handle(model_path, config, shutdown)?;
        Ok(infer_server::openai_router(serve, tokenizer, model_id))
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
            Ok(CudaModelKind::Dsv4 | CudaModelKind::Qwen3Moe)
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
    ///     of `total_pages` and host page ids mirror it 1:1, so admission MUST be
    ///     exactly `config.total_pages` (host total == device total is load-bearing:
    ///     the device pool consumes host page ids directly).
    ///   - Qwen3.5/3.6-MoE: SLOT ARENA — `Qwen35SlotState::new` eagerly allocates a
    ///     contiguous full-attn K/V cache of `total_pages × page_size` tokens per
    ///     layer PER SLOT at load (true VRAM = num_slots ×), so admission is
    ///     `num_slots × per-slot tokens` — the page gate never binds before the
    ///     slot gate.
    ///   - DSv4: SLOT ARENA (SW ring + compressed, each slot covers max context)
    ///     with its own dynamic mem-budget slot clamp; admission is
    ///     `num_slots × per-slot tokens`. Sizing a slot-arena model for a single
    ///     max-context request under-admits at c>1 (a second long request waits
    ///     for fictional pages while a real slot arena sits free).
    ///
    /// `CudaKvPool::new` allocates NO HBM (just a `Vec<u32>` of page ids).
    ///
    /// `num_slots` must be the EFFECTIVE slot count (post KV-budget clamp), not
    /// the requested one.
    #[cfg(feature = "cuda")]
    fn cuda_admission_total_pages(
        kind: CudaModelKind,
        config: &EngineLoadConfig,
        page_size: usize,
        num_slots: usize,
    ) -> usize {
        let ps = page_size.max(1);
        let capacity_tokens = match kind {
            CudaModelKind::Qwen3Dense => config.total_pages.saturating_mul(ps),
            CudaModelKind::Qwen3Moe => config
                .total_pages
                .saturating_mul(ps)
                .saturating_mul(num_slots.max(1)),
            CudaModelKind::Dsv4 => infer_cuda::dsv4_max_seq_len()
                .saturating_add(4096)
                .saturating_mul(num_slots.max(1)),
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
        kv_ssd: &crate::serve::ServeKvSsdOptions,
        shutdown: infer_server::ServeShutdown,
    ) -> Result<(
        ServeHandle<CudaExecutor, CudaKvPool>,
        infer_server::OpenAiTokenizer,
        String,
    )> {
        use infer_server::OpenAiTokenizer;

        infer_cuda::set_decode_graph_default(enable_cuda_graph);
        let tokenizer = OpenAiTokenizer::from_model_dir(model_path)?;
        let model_id = crate::serve_engine::model_id_from_path(model_path);

        let model_source = model_path.to_string();
        let engine_config = *config;
        let serve = ServeHandle::spawn_with_engine_builder_and_shutdown(
            move || build_cuda_engine(&model_source, &engine_config),
            shutdown,
        )?;
        // Opt-in T2 disk spill (`--kv-ssd-path`): attach pre-traffic via the
        // engine-thread control seam; fail closed instead of silently
        // serving without the requested tier.
        if kv_ssd.requested() {
            let root = kv_ssd
                .root
                .clone()
                .ok_or_else(|| anyhow::anyhow!("--kv-ssd-max-bytes requires --kv-ssd-path"))?;
            let budget = kv_ssd
                .max_bytes
                .unwrap_or(infer_cuda::DEFAULT_KV_SSD_BUDGET_BYTES);
            let consumed = serve.run_on_executor(move |e| e.set_kv_tier_disk(root, budget))?;
            anyhow::ensure!(
                consumed,
                "--kv-ssd-path: the loaded model has no page-addressable KV tier store \
                 (Qwen3-dense only today; DSv4/hybrid pending #85)"
            );
        }
        Ok((serve, tokenizer, model_id))
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
        let kind = detect_cuda_model_kind(model_path)?;
        let mut scheduler = config.scheduler_config();
        // Cross-request prompt-prefix reuse (the host radix cache) is only sound
        // when a cached prefix's KV can be re-attached to a new slot and read
        // back unchanged — i.e. a page-addressable full-attention pool. The
        // hybrid / recurrent-KV models advance per-slot state in place and assert
        // contiguous appends from a reset (`seq_len == start_pos`):
        //   - DSv4: sliding-window ring + compressor/indexer running state,
        //   - Qwen3.5/3.6 MoE: gated-delta linear-attention recurrent state +
        //     conv ring (the majority of its layers).
        // Neither can honor a prefix-cache `start_pos > 0`, so prefix reuse is
        // disabled for them and every request resets at `start_pos == 0`. Only
        // pure full-attention Qwen3-dense keeps it. (Default stays on — see
        // `SchedulerConfig::enable_prefix_cache`.)
        if matches!(kind, CudaModelKind::Dsv4 | CudaModelKind::Qwen3Moe) {
            scheduler.enable_prefix_cache = false;
        }
        // The 64-token `chunked_prefill_size` default is a Metal-interactivity
        // tune (small ticks keep the single-threaded MLX encode loop responsive
        // between decode steps); on CUDA the per-chunk cost is an entire engine
        // tick plus a full launch round (GDR/conv/MoE kernels per layer), so a
        // 2048-token prompt at 64-token ticks pays ~32x the tick/launch overhead
        // for the same KV-read volume (KV bytes read are chunk-invariant).
        // Floor the CUDA Qwen kinds at 2048, mirroring the DSv4 override below;
        // an explicitly larger configured chunk is preserved. (audit QW-KV-07)
        if matches!(kind, CudaModelKind::Qwen3Dense | CudaModelKind::Qwen3Moe) {
            scheduler.chunked_prefill_size = scheduler.chunked_prefill_size.max(2048);
        }
        // DSv4 prefill activation scratch is bounded by the query-chunk size
        // (`DSV4_PREFILL_QUERY_CHUNK` = 4096): the chunked-prefill forward asserts
        // each call passes <= that many query tokens, so long prompts MUST chunk
        // (single-chunk max_seq_len both trips that assert at >4096 and OOMs the
        // M×K scratch at 900K). Contiguous chunks are recurrent-KV-safe now that
        // cross-request prefix reuse is disabled above (each request still resets
        // at start_pos==0; chunks advance start_pos contiguously). Cap at 4096.
        if matches!(kind, CudaModelKind::Dsv4) {
            scheduler.chunked_prefill_size = 4096;
            // Long-context: lift the prompt/total token caps to the model's
            // configured max_seq_len so a 900K-token needle isn't rejected with an
            // empty completion. The 32768/65536 defaults are a short-context DoS
            // guard, not a model limit; DSv4's KV budget (dsv4.rs) separately
            // clamps slot count to what HBM affords at this length.
            let max_seq = infer_cuda::dsv4_max_seq_len();
            scheduler.max_prompt_tokens = scheduler.max_prompt_tokens.max(max_seq);
            scheduler.max_total_tokens = scheduler.max_total_tokens.max(max_seq + 4096);
        }
        let num_slots = config.num_slots;
        let page_size = config.page_size;
        // Executors receive the CONFIGURED `total_pages` (Dense: shared device
        // pool size; Qwen3.5/3.6: per-slot token budget / page_size). The host
        // admission pool capacity is derived separately below — after the
        // executor reports its EFFECTIVE slot count (post KV-budget clamp).
        let executor = match kind {
            CudaModelKind::Qwen3Dense => CudaExecutor::from_qwen3_bf16_safetensors(
                model_path,
                num_slots,
                config.total_pages,
            )?,
            // NOTE: Qwen3Moe has no DSv4-style `kv_budget_num_slots` mem clamp yet
            // (load OOM risk if total_pages × num_slots exceeds free HBM) — deferred.
            CudaModelKind::Qwen3Moe => CudaExecutor::from_qwen35_moe_safetensors(
                model_path,
                num_slots,
                config.total_pages,
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
            // signature; `max_seq_len` threads from `dsv4_max_seq_len()`
            // (`INFER_DSV4_MAX_SEQ_LEN`).
            CudaModelKind::Dsv4 => CudaExecutor::from_dsv4_fp8_safetensors(
                model_path,
                num_slots,
                infer_cuda::dsv4_max_seq_len(),
                config.mtp_draft_tokens,
            )?,
        };
        let mut executor = executor;
        if let Some(bytes) = config.kv_t1_budget_bytes {
            // Pre-serve re-budget of the T1 prefix tier (0 disables); None
            // keeps the executor's default-on budget.
            executor.set_kv_tier_budget_bytes(bytes);
        }
        // The DSv4 constructor may clamp slots below the request (dynamic KV
        // mem budget, NCCL min-reduced ⇒ identical on every rank). Scheduler +
        // admission pool MUST follow the effective count: admitting to a slot
        // the executor has no arena for fails at submit, and (lockstep) a
        // scheduler-visible capacity that diverged from the executor's would
        // diverge the deterministic planner.
        let num_slots = executor.effective_num_slots().unwrap_or(num_slots);
        let total_pages = cuda_admission_total_pages(kind, config, page_size, num_slots);
        if num_slots != scheduler.num_slots {
            log::warn!(
                "CUDA engine: executor clamped slots {} -> {num_slots}; scheduler follows",
                scheduler.num_slots
            );
            scheduler.num_slots = num_slots;
        }
        let kv = CudaKvPool::new(num_slots, total_pages, page_size);
        Ok(infer_core::Engine::with_config(executor, kv, scheduler))
    }

    /// Multiproc worker rank's directly-driven engine (rank 1..N-1).
    ///
    /// Unlike rank 0 (whose engine lives on a free-running [`ServeHandle`]
    /// thread), a worker steps its engine SYNCHRONOUSLY — exactly once per
    /// relayed `TickAdmissions` envelope — so admission lands at the same step
    /// index on every rank (the lockstep contract; see
    /// `infer_server::set_tick_broadcaster`). Built and driven on one thread.
    ///
    /// Known growth, mirroring rank 0: the engine's completed-request map is
    /// never drained (no collector on workers), one entry per finished request.
    #[cfg(feature = "cuda")]
    pub struct CudaWorkerEngine(infer_core::Engine<CudaExecutor, CudaKvPool>);

    #[cfg(feature = "cuda")]
    impl CudaWorkerEngine {
        /// Build the rank-R engine from the SAME config rank 0 resolved
        /// (`ARLE_WORKER_ENGINE_CONFIG`); NCCL rank/world come from env during
        /// executor construction.
        pub fn load(model_path: &str, config: &EngineLoadConfig) -> Result<Self> {
            Ok(Self(build_cuda_engine(model_path, config)?))
        }

        /// Inject one relayed request, mirroring rank 0's admission options
        /// (`RequestOptions { sampling, ..default() }` — `admit_submission`).
        pub fn inject(
            &mut self,
            prompt_tokens: Vec<u32>,
            max_tokens: usize,
            sampling: infer_plan::SamplingParams,
        ) {
            let _handle = self.0.submit_request_with_options(
                prompt_tokens,
                max_tokens,
                infer_core::RequestOptions {
                    sampling,
                    ..infer_core::RequestOptions::default()
                },
            );
        }

        /// Whether the engine has no queued, active, or in-flight work —
        /// evaluated on the same state rank 0 evaluates, so both sides step (or
        /// skip) symmetrically per tick.
        #[must_use]
        pub fn is_idle(&self) -> bool {
            self.0.is_idle()
        }

        /// Run exactly one scheduler tick (apply previous output → admit
        /// waiting → build plan → submit forward).
        pub fn step(&mut self) -> Result<()> {
            self.0.step()
        }
    }

    /// Single-GPU CUDA serve router. Builds the same `ServeHandle` as
    /// [`LoadedInferenceEngine::load_cuda`] via [`cuda_serve_handle`], then wraps
    /// it in [`infer_server::openai_router`].
    #[cfg(feature = "cuda")]
    fn router_cuda(
        model_path: &str,
        enable_cuda_graph: bool,
        config: &EngineLoadConfig,
        kv_ssd: &crate::serve::ServeKvSsdOptions,
        shutdown: infer_server::ServeShutdown,
    ) -> Result<axum::Router> {
        let (serve, tokenizer, model_id) =
            cuda_serve_handle(model_path, enable_cuda_graph, config, kv_ssd, shutdown)?;
        Ok(infer_server::openai_router(serve, tokenizer, model_id))
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

        if config.mtp_draft_tokens.is_some() {
            anyhow::bail!("MTP speculative decode is only supported by the CUDA backend");
        }
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
        // HIP serves DSv4 only: sliding-window ring + compressor running state
        // cannot honor a prefix-cache `start_pos > 0`, exactly like the CUDA
        // DSv4 arm (`build_cuda_engine`) — disable cross-request prefix reuse.
        scheduler.enable_prefix_cache = false;
        let num_slots = config.num_slots;
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

        if config.mtp_draft_tokens.is_some() {
            anyhow::bail!("MTP speculative decode is only supported by the CUDA backend");
        }
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

        let scheduler = config.scheduler_config();
        let num_slots = config.num_slots;
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
    /// it in [`infer_server::openai_router`]. Mirrors [`router_cuda`].
    #[cfg(feature = "hip")]
    fn router_hip(
        model_path: &str,
        config: &EngineLoadConfig,
        shutdown: infer_server::ServeShutdown,
    ) -> Result<axum::Router> {
        let (serve, tokenizer, model_id) = hip_serve_handle(model_path, config, shutdown)?;
        Ok(infer_server::openai_router(serve, tokenizer, model_id))
    }

    /// Vulkan serve router. Builds the same `ServeHandle` as
    /// [`LoadedInferenceEngine::load_vulkan`] via [`vulkan_serve_handle`], then
    /// wraps it in [`infer_server::openai_router`]. Mirrors [`router_hip`].
    #[cfg(feature = "vulkan")]
    fn router_vulkan(
        model_path: &str,
        config: &EngineLoadConfig,
        shutdown: infer_server::ServeShutdown,
    ) -> Result<axum::Router> {
        let (serve, tokenizer, model_id) = vulkan_serve_handle(model_path, config, shutdown)?;
        Ok(infer_server::openai_router(serve, tokenizer, model_id))
    }

    /// Portable CPU serve router: the placeholder `MetalExecutor` over the real
    /// backend-neutral host paged KV pool (no MLX, no CUDA), wrapped in
    /// [`infer_server::openai_router`]. Mirrors
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
        use infer_server::{OpenAiTokenizer, openai_router};

        if config.mtp_draft_tokens.is_some() {
            anyhow::bail!("MTP speculative decode is only supported by the CUDA backend");
        }
        let tokenizer = OpenAiTokenizer::from_model_dir(model_path)?;
        let model_id = crate::serve_engine::model_id_from_path(model_path);
        let executor = MetalExecutor::new();
        let kv = HostPagedKvPool::new(config.num_slots, config.total_pages, config.page_size);
        let serve =
            ServeHandle::spawn_with_shutdown(executor, kv, config.scheduler_config(), shutdown);
        Ok(openai_router(serve, tokenizer, model_id))
    }

    impl InferenceEngine for LoadedInferenceEngine {
        fn model_id(&self) -> &str {
            match self {
                #[cfg(feature = "metal")]
                Self::Metal(engine) => engine.model_id(),
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

        fn complete_stream(
            &mut self,
            req: CompletionRequest,
            tx: UnboundedSender<CompletionStreamDelta>,
        ) -> Result<()> {
            match self {
                #[cfg(feature = "metal")]
                Self::Metal(engine) => engine.complete_stream(req, tx),
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

        fn telemetry(&self) -> EngineTelemetry {
            match self {
                #[cfg(feature = "metal")]
                Self::Metal(engine) => engine.telemetry(),
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
