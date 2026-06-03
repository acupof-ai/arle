//! `LoadedInferenceEngine` — the backend-dispatching public engine.
//!
//! Mirrors `infer::server_engine::LoadedInferenceEngine`: a feature-gated enum
//! over the available backends with a `load(model_path, enable_cuda_graph)`
//! constructor and an [`InferenceEngine`] impl that dispatches to the active
//! variant. The backend is selected by compiled feature flags (`metal` / `cuda`
//! / `cpu`), exactly as in the legacy crate — there is no runtime backend arg.
//!
//! [`EngineLoadConfig`] (pure slot/page data) is always available; the
//! [`LoadedInferenceEngine`] enum and its impls exist only when a backend
//! feature is enabled, matching the legacy crate's
//! `#[cfg(any(cuda, metal, cpu))]` gate on the type.

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
        // Matches `infer_server::metal_openai_router_from_model_path` defaults.
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

#[cfg(any(feature = "metal", feature = "cuda", feature = "cpu"))]
mod backend {
    use anyhow::Result;
    use infer_core::SchedulerConfig;
    use infer_server::ServeHandle;
    use tokio::sync::mpsc::UnboundedSender;

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

    /// Backend-dispatching public engine.
    ///
    /// One variant per compiled backend. Construct with [`Self::load`]; drive
    /// through the [`InferenceEngine`] trait.
    pub enum LoadedInferenceEngine {
        /// Metal backend (Apple Silicon, MLX). Fully wired and runnable today.
        #[cfg(feature = "metal")]
        Metal(ServeInferenceEngine<MetalExecutor, MetalKvPool>),
        /// CUDA backend (Linux + NVIDIA). Structurally wired here; the real CUDA
        /// forward is Phase-0 / lead-owned and not yet runnable, but this path
        /// typechecks via the Mac CUDA-Rust recipe.
        #[cfg(feature = "cuda")]
        Cuda(ServeInferenceEngine<CudaExecutor, CudaKvPool>),
        /// Portable CPU backend: the feature-free placeholder `MetalExecutor`
        /// over the real host `MetalKvPool` (no MLX, no CUDA). For smoke / CI.
        #[cfg(all(feature = "cpu", not(feature = "metal")))]
        Cpu(ServeInferenceEngine<MetalExecutor, MetalKvPool>),
    }

    impl LoadedInferenceEngine {
        /// Load the inference engine for the compiled backend.
        ///
        /// Signature matches the legacy
        /// `infer::server_engine::LoadedInferenceEngine::load` so the `infer` ->
        /// `infer-api` swap needs no caller change. `enable_cuda_graph` is
        /// honored by the CUDA path only; other backends ignore it.
        pub fn load(model_path: &str, enable_cuda_graph: bool) -> Result<Self> {
            Self::load_with_config(model_path, enable_cuda_graph, EngineLoadConfig::default())
        }

        /// Load with explicit slot / page configuration.
        // Each backend arm is a feature-gated `return`; which one is the
        // function tail depends on the enabled features, so a bare expression
        // would not compile across all single-backend builds.
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

            // GAP (documented): the real CUDA forward + model build is Phase-0 /
            // lead-owned (the CUDA model.rs/executor.rs are being fixed
            // separately, and this crate may not edit them). The path below is
            // wired structurally — same tokenize/submit/collect/detokenize shape
            // as Metal — so the surface typechecks under the Mac CUDA-Rust
            // recipe and the engine thread is built exactly as it will be when
            // the real forward lands. It is NOT runnable today: the executor
            // builder requires a device and the Phase-0 forward, so
            // `spawn_with_engine_builder` surfaces that error at load. The CUDA
            // tokenizer is resolved from the local model dir (no HF fetch
            // helper exists in infer-cuda yet); `enable_cuda_graph` is reserved
            // for the real executor builder.
            let _ = enable_cuda_graph;
            let tokenizer = OpenAiTokenizer::from_model_dir(model_path)?;
            let model_id = crate::serve_engine::model_id_from_path(model_path);

            let model_source = model_path.to_string();
            let scheduler = config.scheduler_config();
            let num_slots = config.num_slots;
            let total_pages = config.total_pages;
            let page_size = config.page_size;
            let serve = ServeHandle::spawn_with_engine_builder(move || {
                let executor = CudaExecutor::from_qwen3_bf16_safetensors(
                    &model_source,
                    num_slots,
                    total_pages,
                )?;
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

            // The CPU smoke path drives the feature-free placeholder executor
            // over a real host KV pool. It still needs a tokenizer dir for
            // encode/decode.
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
