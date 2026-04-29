use anyhow::Result;
use tokio::sync::mpsc::UnboundedSender;

#[cfg(feature = "cpu")]
use crate::backend::cpu::CpuBackend;
#[cfg(feature = "cuda")]
use crate::backend::cuda::bootstrap::InferenceEngineOptions;
#[cfg(feature = "metal")]
use crate::backend::metal::{MetalSchedulerHandle, spawn_metal_scheduler_handle_from_path};
use crate::session_persistence::SessionPersistence;

#[cfg(feature = "cpu")]
use super::BackendInferenceEngine;
#[cfg(any(feature = "cuda", feature = "metal"))]
use super::RequestHandleInferenceEngine;
#[cfg(feature = "metal")]
use super::stream::model_id_from_path;
use super::{CompletionOutput, CompletionRequest, CompletionStreamDelta, InferenceEngine};

#[cfg(feature = "metal")]
impl RequestHandleInferenceEngine<MetalSchedulerHandle> {
    pub(super) fn load(model_path: &str) -> Result<Self> {
        let handle = spawn_metal_scheduler_handle_from_path(model_path, 0)?;
        Ok(Self {
            model_id: model_id_from_path(model_path),
            handle,
        })
    }
}

pub enum LoadedInferenceEngine {
    /// Unified CUDA variant: drives the multi-request scheduler runtime
    /// through the same `RequestHandle` contract as Metal. The held
    /// `SchedulerRuntimeGuard` keeps the scheduler thread joined on drop.
    #[cfg(feature = "cuda")]
    Cuda {
        engine: RequestHandleInferenceEngine<crate::scheduler::SchedulerHandle>,
        _guard: crate::backend::cuda::bootstrap::SchedulerRuntimeGuard,
    },
    #[cfg(feature = "metal")]
    Metal(RequestHandleInferenceEngine<MetalSchedulerHandle>),
    #[cfg(feature = "cpu")]
    Cpu(BackendInferenceEngine<CpuBackend>),
}

impl LoadedInferenceEngine {
    pub fn load(model_path: &str, enable_cuda_graph: bool) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            return Self::load_with_options(
                model_path,
                42,
                InferenceEngineOptions { enable_cuda_graph },
            );
        }

        #[cfg(all(not(feature = "cuda"), feature = "metal"))]
        {
            let _ = enable_cuda_graph;
            return Ok(Self::Metal(RequestHandleInferenceEngine::load(model_path)?));
        }

        #[cfg(all(not(feature = "cuda"), not(feature = "metal"), feature = "cpu"))]
        {
            let _ = enable_cuda_graph;
            return Ok(Self::Cpu(BackendInferenceEngine::load(model_path)?));
        }

        #[allow(unreachable_code)]
        {
            let _ = (model_path, enable_cuda_graph);
            anyhow::bail!("no inference backend enabled")
        }
    }

    #[cfg(feature = "cuda")]
    pub fn load_with_options(
        model_path: &str,
        seed: u64,
        options: InferenceEngineOptions,
    ) -> Result<Self> {
        let runtime = crate::backend::cuda::bootstrap::ServerRuntimeConfig {
            engine: options,
            seed,
            ..Default::default()
        };
        let metrics = crate::metrics::ServerMetrics::new("");
        let (handle, guard) = crate::backend::cuda::bootstrap::spawn_scheduler_handle_from_path(
            model_path, runtime, metrics,
        )?;
        let model_id = handle.model_id().to_string();
        Ok(Self::Cuda {
            engine: RequestHandleInferenceEngine::from_handle(model_id, handle),
            _guard: guard,
        })
    }

    pub fn backend_name(&self) -> &'static str {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda { .. } => "cuda",
            #[cfg(feature = "metal")]
            Self::Metal(_) => "metal",
            #[cfg(feature = "cpu")]
            Self::Cpu(_) => "cpu",
        }
    }
}

impl InferenceEngine for LoadedInferenceEngine {
    fn model_id(&self) -> &str {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda { engine, .. } => engine.model_id(),
            #[cfg(feature = "metal")]
            Self::Metal(engine) => engine.model_id(),
            #[cfg(feature = "cpu")]
            Self::Cpu(engine) => engine.model_id(),
        }
    }

    fn complete(&mut self, req: CompletionRequest) -> Result<CompletionOutput> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda { engine, .. } => engine.complete(req),
            #[cfg(feature = "metal")]
            Self::Metal(engine) => engine.complete(req),
            #[cfg(feature = "cpu")]
            Self::Cpu(engine) => engine.complete(req),
        }
    }

    fn complete_stream(
        &mut self,
        req: CompletionRequest,
        tx: UnboundedSender<CompletionStreamDelta>,
    ) -> Result<()> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda { engine, .. } => engine.complete_stream(req, tx),
            #[cfg(feature = "metal")]
            Self::Metal(engine) => engine.complete_stream(req, tx),
            #[cfg(feature = "cpu")]
            Self::Cpu(engine) => engine.complete_stream(req, tx),
        }
    }

    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda { engine, .. } => engine.tokenize(text),
            #[cfg(feature = "metal")]
            Self::Metal(engine) => engine.tokenize(text),
            #[cfg(feature = "cpu")]
            Self::Cpu(engine) => engine.tokenize(text),
        }
    }
}

impl SessionPersistence for LoadedInferenceEngine {}
