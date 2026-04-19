//! Backend abstraction for inference execution.
//!
//! Both the CUDA path and the Metal (MLX) path implement [`InferenceBackend`],
//! allowing the scheduler and HTTP server to be backend-agnostic.
//!
//! Concrete backend implementations live under this module:
//! - `cpu`   — development-only CPU backend (feature `cpu`).
//! - [`cuda`]  — NVIDIA/cudarc backend (feature `cuda`).
//! - [`metal`] — Apple Silicon MLX-backed backend (feature `metal`).
//! - [`runtime`] — cross-backend serial runtime handle used by the Metal
//!   and CPU paths.

#[cfg(feature = "cpu")]
#[path = "backend/cpu.rs"]
pub mod cpu;
#[path = "backend/cuda.rs"]
pub mod cuda;
#[path = "backend/metal.rs"]
pub mod metal;
#[path = "backend/runtime.rs"]
pub mod runtime;

use std::path::Path;

use anyhow::Result;

use crate::sampler::SamplingParams;

/// A single-request, synchronous inference backend.
///
/// Implementors load model weights once, then answer repeated `generate` calls.
/// For multi-request batching use the CUDA-path `Scheduler` (under
/// `crate::scheduler`) on top of the CUDA path; the Metal backend currently
/// supports one request at a time.
///
/// Backends must be movable across threads (`Send`) because runtimes may hand
/// ownership to a dedicated worker thread. They are not required to be `Sync`:
/// concurrency, if any, should be introduced by the runtime on top, not by
/// sharing one backend instance across threads.
pub trait InferenceBackend: Send {
    /// Load model weights and tokenizer from `model_path` (a local directory
    /// containing `config.json`, `tokenizer.json`, and `.safetensors` files).
    ///
    /// Called once at startup. Must be called before [`Self::generate`].
    fn load(&mut self, model_path: &Path) -> Result<()>;

    /// Run a single completion and return the generated text.
    ///
    /// `prompt` is a raw text string (not token IDs).  The backend is
    /// responsible for tokenisation, sampling, and detokenisation.
    fn generate(&self, prompt: &str, params: &SamplingParams) -> Result<GenerateResult>;

    /// Backend name for logging / metrics.
    fn name(&self) -> &'static str;
}

/// Streaming-capable inference backend.
///
/// Backends can override this for token/chunk-level streaming. The default
/// implementation falls back to [`InferenceBackend::generate`] and emits the
/// full text as a single chunk.
pub trait StreamingInferenceBackend: InferenceBackend {
    fn generate_stream<F>(
        &self,
        prompt: &str,
        params: &SamplingParams,
        mut on_chunk: F,
    ) -> Result<GenerateResult>
    where
        F: FnMut(&str) -> Result<()>,
    {
        let generated = self.generate(prompt, params)?;
        if !generated.text.is_empty() {
            on_chunk(&generated.text)?;
        }
        Ok(generated)
    }
}

/// Output from a single [`InferenceBackend::generate`] call.
#[derive(Debug, Clone)]
pub struct GenerateResult {
    /// Generated text (not including the prompt).
    pub text: String,
    /// Number of tokens in the prompt.
    pub prompt_tokens: usize,
    /// Number of generated tokens.
    pub completion_tokens: usize,
    /// Why generation stopped (`"stop"`, `"length"`, …).
    pub finish_reason: String,
    /// Time-to-first-token in milliseconds (0.0 if unknown).
    pub ttft_ms: f64,
    /// Prompt throughput in tokens/s (0.0 if unknown).
    pub prompt_tps: f64,
    /// Generation throughput in tokens/s, excluding prompt/prefill time.
    pub generation_tps: f64,
    /// End-to-end wall-clock time in milliseconds.
    pub total_time_ms: f64,
}
