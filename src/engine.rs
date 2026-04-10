#[cfg(feature = "metal")]
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;

use anyhow::Result;
#[cfg(feature = "metal")]
use anyhow::anyhow;

use infer::server_engine::{CompleteOutput, CompleteRequest, FinishReason};

#[cfg(feature = "cuda")]
use infer::bootstrap::EngineOptions;
#[cfg(feature = "cuda")]
use infer::server_engine::LoadedServerEngine;

#[cfg(feature = "metal")]
use infer::backend::InferenceBackend;
#[cfg(feature = "metal")]
use infer::metal_backend::MetalBackend;
#[cfg(feature = "metal")]
use infer::sampler::SamplingParams;
#[cfg(feature = "metal")]
use infer::server_engine::Usage;

#[cfg(feature = "metal")]
fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    match payload.downcast::<String>() {
        Ok(msg) => *msg,
        Err(payload) => match payload.downcast::<&'static str>() {
            Ok(msg) => (*msg).to_string(),
            Err(_) => "unknown panic payload".to_string(),
        },
    }
}

#[cfg_attr(not(any(feature = "cuda", feature = "metal")), allow(dead_code))]
pub trait AgentEngine {
    fn model_id(&self) -> &str;
    fn complete(&mut self, req: CompleteRequest) -> Result<CompleteOutput>;
}

#[cfg(feature = "cuda")]
impl AgentEngine for LoadedServerEngine {
    fn model_id(&self) -> &str {
        infer::server_engine::ServerEngine::model_id(self)
    }

    fn complete(&mut self, req: CompleteRequest) -> Result<CompleteOutput> {
        infer::server_engine::ServerEngine::complete(self, req)
    }
}

#[cfg(feature = "metal")]
pub struct BackendAgentEngine<B: InferenceBackend> {
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

#[cfg(feature = "metal")]
impl<B: InferenceBackend> AgentEngine for BackendAgentEngine<B> {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn complete(&mut self, req: CompleteRequest) -> Result<CompleteOutput> {
        let mut sampling: SamplingParams = req.sampling.clone();
        sampling.max_new_tokens = Some(req.max_tokens);

        let generated = catch_unwind(AssertUnwindSafe(|| {
            self.backend.generate(&req.prompt, &sampling)
        }))
        .map_err(|panic| {
            anyhow!(
                "metal backend panicked during completion: {}",
                panic_message(panic)
            )
        })??;
        let mut text = generated.text;
        let mut finish_reason = parse_finish_reason(&generated.finish_reason);

        if let Some(stops) = req.stop
            && let Some(truncated) = truncate_at_first_stop(&text, &stops)
        {
            text = truncated;
            finish_reason = FinishReason::Stop;
        }

        Ok(CompleteOutput {
            text,
            finish_reason,
            usage: Usage {
                prompt_tokens: generated.prompt_tokens,
                completion_tokens: generated.completion_tokens,
                total_tokens: generated.prompt_tokens + generated.completion_tokens,
            },
            token_logprobs: Vec::new(),
        })
    }
}

#[cfg(any(feature = "cuda", feature = "metal"))]
pub enum LoadedAgentEngine {
    #[cfg(feature = "cuda")]
    Cuda(LoadedServerEngine),
    #[cfg(feature = "metal")]
    Metal(BackendAgentEngine<MetalBackend>),
}

#[cfg(feature = "cuda")]
impl LoadedAgentEngine {
    pub fn load(model_path: &str, enable_cuda_graph: bool) -> Result<Self> {
        Ok(Self::Cuda(LoadedServerEngine::load_with_options(
            model_path,
            42,
            EngineOptions { enable_cuda_graph },
        )?))
    }
}

#[cfg(all(not(feature = "cuda"), feature = "metal"))]
impl LoadedAgentEngine {
    pub fn load(model_path: &str, _enable_cuda_graph: bool) -> Result<Self> {
        Ok(Self::Metal(BackendAgentEngine::load(model_path)?))
    }
}

#[cfg(any(feature = "cuda", feature = "metal"))]
impl LoadedAgentEngine {
    pub fn backend_name(&self) -> &'static str {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => "cuda",
            #[cfg(feature = "metal")]
            Self::Metal(_) => "metal",
        }
    }

    pub fn set_max_gpu_kv(&mut self, max_tokens: usize) {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(engine) => engine.set_max_gpu_kv(max_tokens),
            #[cfg(feature = "metal")]
            Self::Metal(_) => {
                let _ = max_tokens;
            }
        }
    }
}

#[cfg(any(feature = "cuda", feature = "metal"))]
impl AgentEngine for LoadedAgentEngine {
    fn model_id(&self) -> &str {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(engine) => engine.model_id(),
            #[cfg(feature = "metal")]
            Self::Metal(engine) => engine.model_id(),
        }
    }

    fn complete(&mut self, req: CompleteRequest) -> Result<CompleteOutput> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(engine) => engine.complete(req),
            #[cfg(feature = "metal")]
            Self::Metal(engine) => engine.complete(req),
        }
    }
}

fn model_id_from_path(model_path: &str) -> String {
    Path::new(model_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(model_path)
        .to_string()
}

fn parse_finish_reason(finish_reason: &str) -> FinishReason {
    match finish_reason {
        "length" => FinishReason::Length,
        _ => FinishReason::Stop,
    }
}

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

#[cfg(test)]
mod tests {
    use infer::server_engine::FinishReason;

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
        assert_eq!(parse_finish_reason("length"), FinishReason::Length);
        assert_eq!(parse_finish_reason("stop"), FinishReason::Stop);
        assert_eq!(parse_finish_reason("tool_calls"), FinishReason::Stop);
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
