//! `LoadedInferenceEngine` — the backend-dispatching public engine.
//!
//! A feature-gated enum over the available backends (`metal`/`cuda`/`cpu`,
//! selected at compile time) with a `load(model_path, enable_cuda_graph)`
//! constructor dispatching to the active variant. [`EngineLoadConfig`] is always
//! available; the enum + impls require a backend feature.

/// Slot / page configuration for [`LoadedInferenceEngine::load_with_config`].
#[derive(Debug, Clone, Copy)]
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
}

impl Default for EngineLoadConfig {
    fn default() -> Self {
        // Matches `metal_openai_router_from_model_path` defaults.
        Self {
            num_slots: 4,
            total_pages: 8192,
            page_size: 16,
            max_prompt_tokens: 32_768,
            max_total_tokens: 65_536,
            chunked_prefill_size: 64,
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

#[cfg(any(feature = "metal", feature = "cuda", feature = "cpu"))]
mod backend {
    use anyhow::Result;
    use infer_core::SchedulerConfig;
    use infer_server::ServeHandle;
    use tokio::sync::mpsc::UnboundedSender;

    #[cfg(feature = "cuda")]
    use super::CudaModelKind;
    use super::EngineLoadConfig;
    use crate::serve_engine::ServeInferenceEngine;
    use crate::types::{
        CompletionOutput, CompletionRequest, CompletionStreamDelta, EngineTelemetry,
        InferenceEngine,
    };

    #[cfg(feature = "cuda")]
    use infer_cuda::{CudaExecutor, CudaKvPool};
    #[cfg(feature = "metal")]
    use infer_metal::{MetalExecutor, MetalKvPool};
    // The CPU path reuses infer-metal's feature-free placeholder executor + pool.
    #[cfg(all(feature = "cpu", not(feature = "metal")))]
    use infer_metal::{MetalExecutor, MetalKvPool};

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
        /// Portable CPU backend: the placeholder `MetalExecutor` over the real
        /// host `MetalKvPool` (no MLX, no CUDA). Smoke / CI.
        #[cfg(all(feature = "cpu", not(feature = "metal")))]
        Cpu(ServeInferenceEngine<MetalExecutor, MetalKvPool>),
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

            #[cfg(all(not(feature = "metal"), not(feature = "cuda"), feature = "cpu"))]
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
                #[cfg(all(feature = "cpu", not(feature = "metal")))]
                Self::Cpu(_) => {
                    let _ = update;
                    anyhow::bail!("student LoRA re-merge is CUDA-only; active backend is CPU")
                }
            }
        }

        #[cfg(feature = "metal")]
        fn load_metal(model_path: &str, config: &EngineLoadConfig) -> Result<Self> {
            use infer_server::OpenAiTokenizer;

            let resolved = infer_metal::resolve_model_path(model_path)?;
            let tokenizer = OpenAiTokenizer::from_model_dir(&resolved)?;
            let model_id = crate::serve_engine::model_id_from_path(model_path);

            let model_source = resolved.to_string_lossy().to_string();
            let scheduler = config.scheduler_config();
            let num_slots = config.num_slots;
            let total_pages = config.total_pages;
            let page_size = config.page_size;
            let serve = ServeHandle::spawn_with_engine_builder(move || {
                let executor = MetalExecutor::from_model_path(&model_source)?;
                let kv = MetalKvPool::new(num_slots, total_pages, page_size);
                Ok(infer_core::Engine::with_config(executor, kv, scheduler))
            })?;
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
            use infer_server::OpenAiTokenizer;

            // Single-GPU CUDA load: dispatch by checkpoint kind from config.json.
            // Qwen3 dense + Qwen3.5/3.6 MoE run here; DSv4 is multi-GPU only and
            // errors with a pointer to the launcher. Wire `enable_cuda_graph` (CLI
            // `--cuda-graph`/`--no-cuda-graph`, default on) into the decode-graph
            // default; `INFER_CUDA_DECODE_GRAPH` still overrides at runtime, and
            // `warmup` gates the graph off under TP (NCCL) and MoE (host routing).
            infer_cuda::set_decode_graph_default(enable_cuda_graph);
            let tokenizer = OpenAiTokenizer::from_model_dir(model_path)?;
            let model_id = crate::serve_engine::model_id_from_path(model_path);
            let kind = detect_cuda_model_kind(model_path)?;

            let model_source = model_path.to_string();
            let scheduler = config.scheduler_config();
            let num_slots = config.num_slots;
            let total_pages = config.total_pages;
            let page_size = config.page_size;
            let serve = ServeHandle::spawn_with_engine_builder(move || {
                let executor = match kind {
                    CudaModelKind::Qwen3Dense => CudaExecutor::from_qwen3_bf16_safetensors(
                        &model_source,
                        num_slots,
                        total_pages,
                    )?,
                    CudaModelKind::Qwen3Moe => CudaExecutor::from_qwen35_moe_safetensors(
                        &model_source,
                        num_slots,
                        total_pages,
                    )?,
                    CudaModelKind::Dsv4 => anyhow::bail!(
                        "DSv4 is multi-GPU only (TP=8/EP=8); launch via \
                         scripts/dsv4_multigpu_parity.sh, not the single-process loader"
                    ),
                };
                let kv = CudaKvPool::new(num_slots, total_pages, page_size);
                Ok(infer_core::Engine::with_config(executor, kv, scheduler))
            })?;
            Ok(Self::Cuda(ServeInferenceEngine::new(
                model_id, tokenizer, serve,
            )))
        }

        #[cfg(all(feature = "cpu", not(feature = "metal")))]
        fn load_cpu(model_path: &str, config: &EngineLoadConfig) -> Result<Self> {
            use infer_server::OpenAiTokenizer;

            // CPU smoke: placeholder executor over a real host KV pool; still
            // needs a tokenizer dir for encode/decode.
            let tokenizer = OpenAiTokenizer::from_model_dir(model_path)?;
            let model_id = crate::serve_engine::model_id_from_path(model_path);
            let executor = MetalExecutor::new();
            let kv = MetalKvPool::new(config.num_slots, config.total_pages, config.page_size);
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
    /// hands it to [`infer_server::openai_router`] (the Metal arm reuses the
    /// existing [`infer_server::metal_openai_router_from_model_path`] facade, which
    /// also resolves the tokenizer via the HF cache).
    // Each arm is a feature-gated `return` (the tail arm varies by feature set),
    // so a bare expression would not compile in single-backend builds.
    #[allow(clippy::needless_return)]
    pub(crate) fn router_for_backend(
        model_path: &str,
        enable_cuda_graph: bool,
        config: EngineLoadConfig,
    ) -> Result<axum::Router> {
        #[cfg(feature = "metal")]
        {
            let _ = (enable_cuda_graph, &config);
            return infer_server::metal_openai_router_from_model_path(model_path);
        }

        #[cfg(all(not(feature = "metal"), feature = "cuda"))]
        {
            return router_cuda(model_path, enable_cuda_graph, &config);
        }

        #[cfg(all(not(feature = "metal"), not(feature = "cuda"), feature = "cpu"))]
        {
            let _ = enable_cuda_graph;
            return router_cpu(model_path, &config);
        }
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

    /// Single-GPU CUDA serve router. Spawns the same `ServeHandle` as
    /// [`LoadedInferenceEngine::load_cuda`] (dispatching by checkpoint kind;
    /// DSv4 is multi-GPU only and errors), then wraps it in
    /// [`infer_server::openai_router`].
    #[cfg(feature = "cuda")]
    fn router_cuda(
        model_path: &str,
        enable_cuda_graph: bool,
        config: &EngineLoadConfig,
    ) -> Result<axum::Router> {
        use infer_server::{OpenAiTokenizer, openai_router};

        infer_cuda::set_decode_graph_default(enable_cuda_graph);
        let tokenizer = OpenAiTokenizer::from_model_dir(model_path)?;
        let model_id = crate::serve_engine::model_id_from_path(model_path);
        let kind = detect_cuda_model_kind(model_path)?;

        let model_source = model_path.to_string();
        let scheduler = config.scheduler_config();
        let num_slots = config.num_slots;
        let total_pages = config.total_pages;
        let page_size = config.page_size;
        let serve = ServeHandle::spawn_with_engine_builder(move || {
            let executor = match kind {
                CudaModelKind::Qwen3Dense => CudaExecutor::from_qwen3_bf16_safetensors(
                    &model_source,
                    num_slots,
                    total_pages,
                )?,
                CudaModelKind::Qwen3Moe => CudaExecutor::from_qwen35_moe_safetensors(
                    &model_source,
                    num_slots,
                    total_pages,
                )?,
                CudaModelKind::Dsv4 => anyhow::bail!(
                    "DSv4 is multi-GPU only (TP=8/EP=8); launch via \
                     scripts/dsv4_multigpu_parity.sh, not the single-process loader"
                ),
            };
            let kv = CudaKvPool::new(num_slots, total_pages, page_size);
            Ok(infer_core::Engine::with_config(executor, kv, scheduler))
        })?;
        Ok(openai_router(serve, tokenizer, model_id))
    }

    /// Portable CPU serve router: the placeholder `MetalExecutor` over the real
    /// host `MetalKvPool` (no MLX, no CUDA), wrapped in
    /// [`infer_server::openai_router`]. Mirrors
    /// [`LoadedInferenceEngine::load_cpu`].
    #[cfg(all(feature = "cpu", not(feature = "metal")))]
    fn router_cpu(model_path: &str, config: &EngineLoadConfig) -> Result<axum::Router> {
        use infer_server::{OpenAiTokenizer, openai_router};

        let tokenizer = OpenAiTokenizer::from_model_dir(model_path)?;
        let model_id = crate::serve_engine::model_id_from_path(model_path);
        let executor = MetalExecutor::new();
        let kv = MetalKvPool::new(config.num_slots, config.total_pages, config.page_size);
        let serve = ServeHandle::spawn(executor, kv, config.scheduler_config());
        Ok(openai_router(serve, tokenizer, model_id))
    }

    impl InferenceEngine for LoadedInferenceEngine {
        fn model_id(&self) -> &str {
            match self {
                #[cfg(feature = "metal")]
                Self::Metal(engine) => engine.model_id(),
                #[cfg(feature = "cuda")]
                Self::Cuda(engine) => engine.model_id(),
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
                #[cfg(all(feature = "cpu", not(feature = "metal")))]
                Self::Cpu(engine) => engine.telemetry(),
            }
        }
    }
}

#[cfg(any(feature = "metal", feature = "cuda", feature = "cpu"))]
pub use backend::LoadedInferenceEngine;
#[cfg(any(feature = "metal", feature = "cuda", feature = "cpu"))]
pub(crate) use backend::router_for_backend;
