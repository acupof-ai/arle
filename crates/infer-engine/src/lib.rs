#[cfg(any(feature = "metal", feature = "cpu"))]
use std::panic::{AssertUnwindSafe, catch_unwind};
#[cfg(any(feature = "metal", feature = "cpu", test))]
use std::path::Path;

use anyhow::Result;
#[cfg(any(feature = "metal", feature = "cpu"))]
use anyhow::anyhow;
#[cfg(feature = "cuda")]
use infer::server_engine::FinishReason;

#[cfg(feature = "cuda")]
use infer::backend::cuda::bootstrap::EngineOptions;
#[cfg(feature = "cuda")]
use infer::server_engine::CompleteRequest;
#[cfg(feature = "cuda")]
use infer::server_engine::LoadedServerEngine;

#[cfg(any(feature = "metal", feature = "cpu"))]
use infer::backend::InferenceBackend;
#[cfg(feature = "cpu")]
use infer::backend::cpu::CpuBackend;
#[cfg(feature = "metal")]
use infer::backend::metal::MetalBackend;
#[cfg(any(feature = "metal", feature = "cpu"))]
use infer::sampler::SamplingParams;

#[derive(Clone, Debug, PartialEq)]
pub struct AgentCompleteRequest {
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub stop: Option<Vec<String>>,
    pub logprobs: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AgentFinishReason {
    Length,
    Stop,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AgentUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AgentCompleteOutput {
    pub text: String,
    pub finish_reason: AgentFinishReason,
    pub usage: AgentUsage,
    pub token_logprobs: Vec<f32>,
}

pub trait AgentEngine {
    fn model_id(&self) -> &str;
    fn complete(&mut self, req: AgentCompleteRequest) -> Result<AgentCompleteOutput>;
}

#[cfg(any(feature = "metal", feature = "cpu"))]
fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    match payload.downcast::<String>() {
        Ok(msg) => *msg,
        Err(payload) => match payload.downcast::<&'static str>() {
            Ok(msg) => (*msg).to_string(),
            Err(_) => "unknown panic payload".to_string(),
        },
    }
}

#[cfg(feature = "cuda")]
struct CudaAgentEngine {
    inner: LoadedServerEngine,
}

#[cfg(feature = "cuda")]
impl CudaAgentEngine {
    fn load(model_path: &str, enable_cuda_graph: bool) -> Result<Self> {
        Ok(Self {
            inner: LoadedServerEngine::load_with_options(
                model_path,
                42,
                EngineOptions { enable_cuda_graph },
            )?,
        })
    }

    fn set_max_gpu_kv(&mut self, max_tokens: usize) {
        self.inner.set_max_gpu_kv(max_tokens);
    }
}

#[cfg(feature = "cuda")]
impl AgentEngine for CudaAgentEngine {
    fn model_id(&self) -> &str {
        infer::server_engine::ServerEngine::model_id(&self.inner)
    }

    fn complete(&mut self, req: AgentCompleteRequest) -> Result<AgentCompleteOutput> {
        let output = infer::server_engine::ServerEngine::complete(
            &mut self.inner,
            CompleteRequest {
                prompt: req.prompt,
                max_tokens: req.max_tokens,
                sampling: infer::sampler::SamplingParams {
                    temperature: req.temperature,
                    ..infer::sampler::SamplingParams::default()
                },
                stop: req.stop,
                logprobs: req.logprobs,
            },
        )?;
        Ok(AgentCompleteOutput {
            text: output.text,
            finish_reason: finish_reason_from_infer(output.finish_reason),
            usage: AgentUsage {
                prompt_tokens: output.usage.prompt_tokens,
                completion_tokens: output.usage.completion_tokens,
                total_tokens: output.usage.total_tokens,
            },
            token_logprobs: output.token_logprobs,
        })
    }
}

#[cfg(any(feature = "metal", feature = "cpu"))]
struct BackendAgentEngine<B: InferenceBackend> {
    model_id: String,
    backend: B,
}

#[cfg(feature = "metal")]
impl BackendAgentEngine<MetalBackend> {
    pub fn load(model_path: &str) -> Result<Self> {
        let mut backend = MetalBackend::new();
        backend.load(Path::new(model_path))?;
        Ok(Self {
            model_id: model_id_from_path(model_path),
            backend,
        })
    }
}

#[cfg(feature = "cpu")]
impl BackendAgentEngine<CpuBackend> {
    pub fn load(model_path: &str) -> Result<Self> {
        let mut backend = CpuBackend::new();
        backend.load(Path::new(model_path))?;
        Ok(Self {
            model_id: model_id_from_path(model_path),
            backend,
        })
    }
}

#[cfg(any(feature = "metal", feature = "cpu"))]
impl<B: InferenceBackend> AgentEngine for BackendAgentEngine<B> {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn complete(&mut self, req: AgentCompleteRequest) -> Result<AgentCompleteOutput> {
        let mut sampling = SamplingParams {
            temperature: req.temperature,
            ..SamplingParams::default()
        };
        sampling.max_new_tokens = Some(req.max_tokens);

        let generated = catch_unwind(AssertUnwindSafe(|| {
            self.backend.generate(&req.prompt, &sampling)
        }))
        .map_err(|panic| {
            anyhow!(
                "{} backend panicked during completion: {}",
                self.backend.name(),
                panic_message(panic)
            )
        })??;
        let mut text = generated.text;
        let mut finish_reason = parse_finish_reason(&generated.finish_reason);

        if let Some(stops) = req.stop
            && let Some(truncated) = truncate_at_first_stop(&text, &stops)
        {
            text = truncated;
            finish_reason = AgentFinishReason::Stop;
        }

        Ok(AgentCompleteOutput {
            text,
            finish_reason,
            usage: AgentUsage {
                prompt_tokens: generated.prompt_tokens,
                completion_tokens: generated.completion_tokens,
                total_tokens: generated.prompt_tokens + generated.completion_tokens,
            },
            token_logprobs: Vec::new(),
        })
    }
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
enum LoadedAgentEngineInner {
    #[cfg(feature = "cuda")]
    Cuda(CudaAgentEngine),
    #[cfg(feature = "metal")]
    Metal(BackendAgentEngine<MetalBackend>),
    #[cfg(feature = "cpu")]
    Cpu(BackendAgentEngine<CpuBackend>),
}

#[cfg(feature = "cuda")]
impl LoadedAgentEngineInner {
    fn load(model_path: &str, enable_cuda_graph: bool) -> Result<Self> {
        Ok(Self::Cuda(CudaAgentEngine::load(
            model_path,
            enable_cuda_graph,
        )?))
    }
}

#[cfg(all(not(feature = "cuda"), feature = "metal"))]
impl LoadedAgentEngineInner {
    fn load(model_path: &str, _enable_cuda_graph: bool) -> Result<Self> {
        Ok(Self::Metal(BackendAgentEngine::load(model_path)?))
    }
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal"), feature = "cpu"))]
impl LoadedAgentEngineInner {
    fn load(model_path: &str, _enable_cuda_graph: bool) -> Result<Self> {
        Ok(Self::Cpu(BackendAgentEngine::load(model_path)?))
    }
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
impl LoadedAgentEngineInner {
    fn backend_name(&self) -> &'static str {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => "cuda",
            #[cfg(feature = "metal")]
            Self::Metal(_) => "metal",
            #[cfg(feature = "cpu")]
            Self::Cpu(_) => "cpu",
        }
    }

    fn set_max_gpu_kv(&mut self, max_tokens: usize) {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(engine) => engine.set_max_gpu_kv(max_tokens),
            #[cfg(feature = "metal")]
            Self::Metal(_) => {
                let _ = max_tokens;
            }
            #[cfg(feature = "cpu")]
            Self::Cpu(_) => {
                let _ = max_tokens;
            }
        }
    }
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
impl AgentEngine for LoadedAgentEngineInner {
    fn model_id(&self) -> &str {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(engine) => engine.model_id(),
            #[cfg(feature = "metal")]
            Self::Metal(engine) => engine.model_id(),
            #[cfg(feature = "cpu")]
            Self::Cpu(engine) => engine.model_id(),
        }
    }

    fn complete(&mut self, req: AgentCompleteRequest) -> Result<AgentCompleteOutput> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(engine) => engine.complete(req),
            #[cfg(feature = "metal")]
            Self::Metal(engine) => engine.complete(req),
            #[cfg(feature = "cpu")]
            Self::Cpu(engine) => engine.complete(req),
        }
    }
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
pub struct LoadedAgentEngine {
    inner: LoadedAgentEngineInner,
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
impl LoadedAgentEngine {
    pub fn load(model_path: &str, enable_cuda_graph: bool) -> Result<Self> {
        Ok(Self {
            inner: LoadedAgentEngineInner::load(model_path, enable_cuda_graph)?,
        })
    }

    pub fn backend_name(&self) -> &'static str {
        self.inner.backend_name()
    }

    pub fn set_max_gpu_kv(&mut self, max_tokens: usize) {
        self.inner.set_max_gpu_kv(max_tokens);
    }
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
impl AgentEngine for LoadedAgentEngine {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    fn complete(&mut self, req: AgentCompleteRequest) -> Result<AgentCompleteOutput> {
        self.inner.complete(req)
    }
}

#[cfg(any(feature = "metal", feature = "cpu", test))]
fn model_id_from_path(model_path: &str) -> String {
    Path::new(model_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(model_path)
        .to_string()
}

#[cfg(any(feature = "metal", feature = "cpu", test))]
fn parse_finish_reason(finish_reason: &str) -> AgentFinishReason {
    match finish_reason {
        "length" => AgentFinishReason::Length,
        _ => AgentFinishReason::Stop,
    }
}

#[cfg(any(feature = "metal", feature = "cpu", test))]
fn truncate_at_first_stop(text: &str, stops: &[String]) -> Option<String> {
    let mut earliest = None::<usize>;
    for stop in stops {
        let stop = stop.as_str();
        if stop.is_empty() {
            continue;
        }
        if let Some(pos) = text.find(stop) {
            earliest = Some(match earliest {
                None => pos,
                Some(existing) => existing.min(pos),
            });
        }
    }
    earliest.map(|pos| text[..pos].to_string())
}

#[cfg(feature = "cuda")]
fn finish_reason_from_infer(reason: FinishReason) -> AgentFinishReason {
    match reason {
        FinishReason::Length => AgentFinishReason::Length,
        FinishReason::Stop => AgentFinishReason::Stop,
    }
}

pub fn init_default_logging() {
    infer::logging::init_default();
}

pub fn resolve_model_source(explicit_model_path: Option<&str>) -> Result<String> {
    if let Some(model_path) = explicit_model_path
        && !model_path.trim().is_empty()
    {
        return Ok(model_path.to_string());
    }

    if let Ok(model) = std::env::var("AGENT_INFER_MODEL")
        && !model.trim().is_empty()
    {
        return Ok(model);
    }

    if let Some((candidate, local_path)) = infer::hf_hub::discover_local_model() {
        log::info!(
            "Auto-detected local model '{}' at {}",
            candidate,
            local_path.display()
        );
        return Ok(candidate);
    }

    anyhow::bail!(
        "No model specified and no local model was auto-detected. Pass --model-path or set AGENT_INFER_MODEL."
    )
}

#[cfg(test)]
mod tests {
    use super::AgentFinishReason;
    use super::{model_id_from_path, parse_finish_reason, truncate_at_first_stop};

    #[test]
    fn model_id_uses_final_path_segment() {
        assert_eq!(
            model_id_from_path("mlx-community/Qwen3-0.6B-4bit"),
            "Qwen3-0.6B-4bit"
        );
        assert_eq!(model_id_from_path("/tmp/models/Qwen3-4B"), "Qwen3-4B");
    }

    #[test]
    fn parse_finish_reason_defaults_to_stop() {
        assert_eq!(parse_finish_reason("length"), AgentFinishReason::Length);
        assert_eq!(parse_finish_reason("stop"), AgentFinishReason::Stop);
        assert_eq!(parse_finish_reason("tool_calls"), AgentFinishReason::Stop);
    }

    #[test]
    fn truncate_at_first_stop_picks_earliest_match() {
        let stops = vec!["END".to_string(), "\n\n".to_string()];
        assert_eq!(
            truncate_at_first_stop("hello\n\nworldEND", &stops),
            Some("hello".to_string())
        );
        assert_eq!(truncate_at_first_stop("hello world", &stops), None);
    }
}
