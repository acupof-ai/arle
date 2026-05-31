use anyhow::Result;
use tokio::sync::mpsc::UnboundedSender;

#[cfg(feature = "cpu")]
use crate::backend::cpu::CpuBackend;
#[cfg(feature = "cuda")]
use crate::backend::cuda::bootstrap::InferenceEngineOptions;
#[cfg(feature = "metal")]
use crate::backend::metal::{
    MetalBackendOptions, MetalSchedulerHandle, auto_wired_limit_bytes,
    spawn_metal_scheduler_handle_from_path_with_options,
};

#[cfg(feature = "cpu")]
use super::BackendInferenceEngine;
#[cfg(any(feature = "cuda", feature = "metal"))]
use super::RequestHandleInferenceEngine;
#[cfg(feature = "metal")]
use super::stream::model_id_from_path;
use super::{
    CompletionOutput, CompletionRequest, CompletionStreamDelta, EngineTelemetry, InferenceEngine,
};

#[cfg(feature = "metal")]
impl RequestHandleInferenceEngine<MetalSchedulerHandle> {
    pub(super) fn load(model_path: &str) -> Result<Self> {
        // Warm-load parity with `metal_serve`: pin model weights in RAM via
        // `mlx::set_wired_limit` (applied by `MetalRuntimeLimits::apply` during
        // backend load) so macOS doesn't page out cold expert weights under
        // pressure. Drops c=1 p99 from 86 ms to 15 ms on Qwen3.6
        // (docs/experience/wins/2026-05-07-bench-qwen36-mle-perf.md). The CLI
        // load path previously left this unset; metal_serve already auto-pins.
        let options = MetalBackendOptions {
            runtime_limits: crate::backend::metal::MetalRuntimeLimits {
                wired_limit_bytes: auto_wired_limit_bytes(model_path),
                ..Default::default()
            },
            ..Default::default()
        };
        let handle = spawn_metal_scheduler_handle_from_path_with_options(model_path, options, 0)?;
        let mut engine = Self {
            model_id: model_id_from_path(model_path),
            handle,
        };
        engine.warmup();
        Ok(engine)
    }

    /// Run a short generation at load so the model weights fault into RAM and
    /// wire (under `wired_limit_bytes`) *before* the first interactive turn —
    /// otherwise that turn is disk-I/O-bound (~9 tok/s cold vs ~75 warm on
    /// Qwen3.6; RSS climbs ~0.1 GB → ~18.5 GB as MoE experts fault in). This is
    /// the in-process analogue of `metal_serve`'s startup warmup. Best-effort:
    /// a warmup failure must never block CLI startup.
    fn warmup(&mut self) {
        use std::io::Write;
        // Visible status: the first load runs a warmup so the FIRST real turn is
        // fast instead of disk-I/O-bound. Goes to stderr so it never pollutes
        // piped stdout.
        eprint!("  warming up model weights (first run, ~a few seconds)… ");
        let _ = std::io::stderr().flush();
        // A long, lexically-diverse prompt prefills many tokens in ONE batched
        // pass — each token routes to its own MoE experts, so this faults+wires
        // a broad expert set cheaply (prefill ~200 tok/s) instead of waiting on a
        // slow token-by-token decode. A short decode confirms the decode path is
        // hot too.
        let prompt = "Across mountains, oceans, deserts and frozen tundra, scientists, \
            engineers, musicians, farmers, doctors and philosophers debate history, \
            biology, quantum mechanics, economics, poetry, law, medicine, architecture \
            and cuisine. Rivers carved canyons; satellites mapped distant galaxies; \
            algorithms sorted petabytes of data; orchestras tuned violins, cellos, \
            trumpets and drums. Translate, summarize, compute, imagine, remember, \
            predict and explain — numbers like 3, 17, 256, 1024 and 65536 mingle with \
            verbs, adjectives, nouns and punctuation across many languages and domains.";
        let req = CompletionRequest {
            prompt: prompt.to_string(),
            max_tokens: 32,
            sampling: crate::sampler::SamplingParams::default(),
            stop: None,
            logprobs: false,
            session_id: None,
            trace_context: None,
            cancel: None,
        };
        let _ = self.complete(req);
        eprintln!("done");
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

    #[cfg(feature = "cuda")]
    pub fn load_with_runtime_config(
        model_path: &str,
        runtime: crate::backend::cuda::bootstrap::ServerRuntimeConfig,
    ) -> Result<Self> {
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

    #[cfg(feature = "cuda")]
    pub fn forward_token_logits(
        &self,
        input_ids: &[u32],
        positions: &[u32],
    ) -> Result<super::RawLogits> {
        match self {
            Self::Cuda { engine, .. } => engine.forward_token_logits(input_ids, positions),
            #[cfg(feature = "metal")]
            Self::Metal(_) => anyhow::bail!("forward_token_logits is only available on CUDA"),
            #[cfg(feature = "cpu")]
            Self::Cpu(_) => anyhow::bail!("forward_token_logits is only available on CUDA"),
        }
    }

    /// Push a fresh student LoRA adapter into the running CUDA engine for the
    /// per-step OPD rollout sync (P2). The re-merge runs on the scheduler
    /// thread that owns the model.
    #[cfg(feature = "cuda")]
    pub fn remerge_student_lora(&self, update: crate::model::StudentLoraUpdate) -> Result<()> {
        match self {
            Self::Cuda { engine, .. } => engine.remerge_student_lora(update),
            #[cfg(feature = "metal")]
            Self::Metal(_) => anyhow::bail!("remerge_student_lora is only available on CUDA"),
            #[cfg(feature = "cpu")]
            Self::Cpu(_) => anyhow::bail!("remerge_student_lora is only available on CUDA"),
        }
    }

    /// Offload the engine's device weights to host RAM, freeing VRAM for the
    /// OPD time-share. Returns the device bytes freed.
    #[cfg(feature = "cuda")]
    pub fn offload_engine_weights(&self) -> Result<usize> {
        match self {
            Self::Cuda { engine, .. } => engine.offload_engine_weights(),
            #[cfg(feature = "metal")]
            Self::Metal(_) => anyhow::bail!("offload_engine_weights is only available on CUDA"),
            #[cfg(feature = "cpu")]
            Self::Cpu(_) => anyhow::bail!("offload_engine_weights is only available on CUDA"),
        }
    }

    /// Reload the engine's device weights from the host snapshot.
    #[cfg(feature = "cuda")]
    pub fn reload_engine_weights(&self) -> Result<()> {
        match self {
            Self::Cuda { engine, .. } => engine.reload_engine_weights(),
            #[cfg(feature = "metal")]
            Self::Metal(_) => anyhow::bail!("reload_engine_weights is only available on CUDA"),
            #[cfg(feature = "cpu")]
            Self::Cpu(_) => anyhow::bail!("reload_engine_weights is only available on CUDA"),
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

    fn telemetry(&self) -> EngineTelemetry {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda { engine, .. } => engine.telemetry(),
            #[cfg(feature = "metal")]
            Self::Metal(engine) => engine.telemetry(),
            #[cfg(feature = "cpu")]
            Self::Cpu(engine) => engine.telemetry(),
        }
    }
}
