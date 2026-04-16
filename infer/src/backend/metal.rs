//! Metal inference backend for Apple Silicon.
//!
//! Uses `crate::mlx` (thin mlx-sys wrappers) for Metal-accelerated tensor
//! operations.  No mlx-rs dependency.
//!
//! # Architecture
//!
//! Implements a full Qwen3/Qwen3.5 transformer forward pass:
//! - safetensors weight loading into MLX unified memory
//! - RMSNorm, RoPE, GQA attention, SwiGLU MLP
//! - Simple append-based KV cache (one buffer per generate call)
//! - Greedy / temperature sampling with `top_k=1`
//!
//! # Feature flag
//!
//! Compile with `--no-default-features --features metal,no-cuda`.
//!
//! # Example
//! ```no_run
//! use infer::backend::metal::MetalBackend;
//! use infer::backend::InferenceBackend;
//! use std::path::Path;
//!
//! let mut backend = MetalBackend::new();
//! backend.load(Path::new("mlx-community/Qwen3-0.6B-4bit")).unwrap();
//! ```

#[cfg(feature = "metal")]
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
#[cfg(feature = "metal")]
use std::time::Instant;

#[cfg(feature = "metal")]
use anyhow::anyhow;
#[cfg(feature = "metal")]
use anyhow::bail;
use anyhow::{Context, Result};

#[cfg(feature = "metal")]
use crate::backend::StreamingInferenceBackend;
use crate::{
    backend::{GenerateResult, InferenceBackend},
    hf_hub,
    sampler::SamplingParams,
    tokenizer::Tokenizer,
};

#[path = "metal/config.rs"]
mod config;
#[cfg(feature = "metal")]
#[path = "metal/dflash.rs"]
mod dflash;
#[cfg(feature = "metal")]
#[path = "metal/forward.rs"]
pub mod forward;
#[cfg(feature = "metal")]
#[path = "metal/generate.rs"]
pub mod generate;
#[cfg(feature = "metal")]
#[path = "metal/loader.rs"]
mod loader;
#[cfg(feature = "metal")]
#[path = "metal/ops.rs"]
pub mod ops;
#[cfg(feature = "metal")]
#[path = "metal/qwen35.rs"]
mod qwen35;
#[cfg(feature = "metal")]
#[path = "metal/request_state.rs"]
pub mod request_state;
#[cfg(feature = "metal")]
#[path = "metal/runtime.rs"]
pub mod runtime;
#[cfg(feature = "metal")]
#[path = "metal/sampling.rs"]
pub mod sampling;
#[cfg(feature = "metal")]
#[path = "metal/weights.rs"]
pub mod weights;

// Submodules that used to live at the crate root as `metal_*.rs` — moved
// under `metal_backend/` so that all Metal-specific code lives in one place
// (see `docs/plans/` for the reorganisation plan).
#[path = "metal/gdr.rs"]
pub mod gdr;
#[path = "metal/kv_pool.rs"]
pub mod kv_pool;
#[cfg(feature = "metal")]
#[path = "metal/mlx.rs"]
pub mod mlx;
#[path = "metal/prefix_cache.rs"]
pub mod prefix_cache;
#[path = "metal/scheduler.rs"]
pub mod scheduler;

// ── mlx types (Metal GPU required) ───────────────────────────────────────────
#[cfg(feature = "metal")]
use self::generate::{KV_CACHE_CHUNK, MetalGenerateOutput};
#[cfg(feature = "metal")]
use self::ops::{clear_metal_cache, extend_kv_cache, linear};
#[cfg(feature = "metal")]
use self::sampling::gpu_sample_token;
#[cfg(feature = "metal")]
use self::weights::{
    MetalWeights, MlpInputProjection, WeightTensor, merge_quantized_projection_rows,
};
use config::{MetalModelArch, MetalModelConfig, load_metal_config};
#[cfg(feature = "metal")]
use config::{MetalQwen35ArchConfig, MetalQwen35LayerType, QuantConfig};
#[cfg(feature = "metal")]
pub use dflash::MetalDflashOptions;
#[cfg(feature = "metal")]
use loader::{
    TensorMap, load_embed_tokens_from_tensors, load_proj_from_tensors, load_tensor_map, tensor_get,
    tie_lm_head_from_embed_tokens,
};
#[cfg(feature = "metal")]
use qwen35::{load_qwen35_metal_weights, metal_generate_qwen35};
#[cfg(feature = "metal")]
pub use runtime::{
    spawn_metal_scheduler_handle_from_path, spawn_metal_scheduler_handle_from_path_with_options,
    spawn_metal_scheduler_handle_from_path_with_options_and_metrics,
};

// NOTE: The legacy fused-ops FFI modules (`metal_ffi`, `metal_capi_ffi`) were
// removed during the mlx-sys migration. Qwen3 now runs on the maintained
// Rust/MLX path; Qwen3.5 optionally uses the dedicated C++ step model in
// `qwen35.rs`.

/// Apple Silicon Metal inference backend.
///
/// One instance per process; weights are loaded once and kept resident in
/// unified memory (Metal).
pub struct MetalBackend {
    /// Local directory containing config.json, tokenizer.json, and weight shards.
    model_dir: Option<PathBuf>,
    /// Tokenizer loaded from tokenizer.json.
    tokenizer: Option<Tokenizer>,
    /// Loaded model configuration.
    config: Option<MetalModelConfig>,
    /// Weight tensors — resident in Metal unified memory.
    // GPU required: populated in `load()` via mlx-sys-backed MLX tensors.
    #[cfg(feature = "metal")]
    weights: Option<MetalWeights>,
    #[cfg(feature = "metal")]
    dflash_options: Option<MetalDflashOptions>,
    #[cfg(feature = "metal")]
    dflash: Option<dflash::MetalDflashRuntime>,
    #[cfg(feature = "metal")]
    kv_pool_enabled: bool,
    #[cfg(feature = "metal")]
    runtime_limits: MetalRuntimeLimits,
    #[cfg(not(feature = "metal"))]
    _weights: (),
}

impl MetalBackend {
    pub fn new() -> Self {
        Self::with_options(MetalBackendOptions::default())
    }

    // The `metal` feature branch destructures `options` to consume
    // its fields; the no-cuda branch only needs the value to exist
    // so the `with_options` API stays uniform across feature gates.
    // Clippy's needless_pass_by_value fires on the no-cuda branch
    // alone — suppress at the function level since the by-value
    // signature is intentional for the metal branch.
    #[allow(clippy::needless_pass_by_value)]
    pub fn with_options(options: MetalBackendOptions) -> Self {
        #[cfg(not(feature = "metal"))]
        let _ = options;
        #[cfg(feature = "metal")]
        let MetalBackendOptions {
            dflash,
            kv_pool,
            runtime_limits,
        } = options;

        Self {
            model_dir: None,
            tokenizer: None,
            config: None,
            #[cfg(feature = "metal")]
            weights: None,
            #[cfg(feature = "metal")]
            dflash_options: dflash,
            #[cfg(feature = "metal")]
            dflash: None,
            #[cfg(feature = "metal")]
            kv_pool_enabled: self::generate::resolve_metal_kv_pool_enabled(kv_pool),
            #[cfg(feature = "metal")]
            runtime_limits,
            #[cfg(not(feature = "metal"))]
            _weights: (),
        }
    }

    pub fn generate_stream<F>(
        &self,
        prompt: &str,
        params: &SamplingParams,
        mut on_chunk: F,
    ) -> Result<GenerateResult>
    where
        F: FnMut(&str) -> Result<()>,
    {
        run_with_metal_panic_boundary("stream generation", || {
            let tokenizer = self
                .tokenizer
                .as_ref()
                .context("model not loaded — call load() first")?;
            let input_ids = tokenizer.encode(prompt)?;
            let mut decoder = tokenizer.incremental_decoder();

            let result =
                self.generate_from_token_ids_with_callback(&input_ids, params, |token_id| {
                    if let Some(chunk) = decoder.step(token_id)? {
                        on_chunk(&chunk)?;
                    }
                    Ok(())
                })?;

            if let Some(tail) = decoder.finish()? {
                on_chunk(&tail)?;
            }

            Ok(result)
        })
    }

    pub fn generate_from_token_ids(
        &self,
        input_ids: &[u32],
        params: &SamplingParams,
    ) -> Result<GenerateResult> {
        run_with_metal_panic_boundary("token-id generation", || {
            self.generate_from_token_ids_with_callback(input_ids, params, |_token_id| Ok(()))
        })
    }

    /// Return the DFlash runtime ref with `'static` lifetime.
    ///
    /// # Safety
    /// Only valid when called on a leaked `&'static MetalBackend` (scheduler
    /// runtime thread — `Box::leak` at `runtime.rs:591`). The borrow checker
    /// can't prove this at the call site, so we use pointer-cast.
    #[cfg(feature = "metal")]
    pub(crate) unsafe fn dflash_runtime_static(
        &self,
    ) -> Option<(
        &'static dflash::MetalDflashRuntime,
        &'static MetalModelConfig,
    )> {
        let rt: &dflash::MetalDflashRuntime = self.dflash.as_ref()?;
        let cfg: &MetalModelConfig = self.config.as_ref()?;
        // SAFETY: the scheduler runtime leaks the backend to 'static
        // (`Box::leak(Box::new(backend))` at runtime.rs:591) before any
        // request state is created, so these references outlive all requests.
        unsafe {
            Some((
                &*(rt as *const dflash::MetalDflashRuntime),
                &*(cfg as *const MetalModelConfig),
            ))
        }
    }

    #[cfg(feature = "metal")]
    pub fn create_request_state(
        &self,
        input_ids: &[u32],
        params: &SamplingParams,
    ) -> Result<request_state::MetalRequestState<'_>> {
        // DFlash runtime is not threaded here — only the scheduler runtime
        // (which has a 'static backend ref) can pass it. The legacy serial
        // runtime never calls create_request_state for DFlash.
        self.create_request_state_with_dflash(input_ids, params, None)
    }

    /// Like `create_request_state` but accepts an explicit DFlash runtime
    /// reference with `'static` lifetime. Called from the scheduler runtime
    /// where the backend is leaked to `'static`.
    #[cfg(feature = "metal")]
    pub fn create_request_state_with_dflash(
        &self,
        input_ids: &[u32],
        params: &SamplingParams,
        dflash_runtime: Option<(
            &'static dflash::MetalDflashRuntime,
            &'static MetalModelConfig,
        )>,
    ) -> Result<request_state::MetalRequestState<'_>> {
        let config = self.config.as_ref().context("model not loaded")?;
        let weights = self.weights.as_ref().context("weights not loaded")?;
        let max_new_tokens = params.max_new_tokens.unwrap_or(512);
        request_state::MetalRequestState::new(
            weights,
            config,
            input_ids.to_vec(),
            params,
            self.kv_pool_enabled,
            max_new_tokens,
            dflash_runtime,
        )
    }

    #[allow(unused_mut)]
    fn generate_from_token_ids_with_callback<F>(
        &self,
        input_ids: &[u32],
        params: &SamplingParams,
        mut on_token: F,
    ) -> Result<GenerateResult>
    where
        F: FnMut(u32) -> Result<()>,
    {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .context("model not loaded — call load() first")?;
        let config = self.config.as_ref().unwrap();
        let prompt_tokens = input_ids.len();

        #[cfg(feature = "metal")]
        {
            let weights = self.weights.as_ref().context("weights not loaded")?;
            let max_new_tokens = params.max_new_tokens.unwrap_or(512);
            let t0 = Instant::now();

            let generated = if let Some(runtime) = self.dflash.as_ref() {
                let weights = match weights {
                    MetalWeights::Qwen3(weights) => weights,
                    MetalWeights::Qwen35(_) => {
                        bail!(
                            "Metal DFlash draft '{}' is currently available for Qwen3 targets only",
                            runtime.draft_model_id()
                        );
                    }
                };
                dflash::metal_generate_dflash_qwen3(
                    runtime,
                    input_ids,
                    weights,
                    config,
                    params,
                    max_new_tokens,
                    t0,
                    &mut on_token,
                )?
            } else {
                match weights {
                    MetalWeights::Qwen3(weights) => self::generate::metal_generate(
                        input_ids,
                        weights,
                        config,
                        params,
                        self.kv_pool_enabled,
                        max_new_tokens,
                        t0,
                        &mut on_token,
                    )?,
                    MetalWeights::Qwen35(weights) => metal_generate_qwen35(
                        {
                            if self.kv_pool_enabled {
                                log::warn!(
                                    "MetalKVPool is currently wired for the Qwen3 fallback path only; \
                                     Qwen3.5 continues on the existing Metal route"
                                );
                            }
                            input_ids
                        },
                        weights,
                        config,
                        params,
                        max_new_tokens,
                        t0,
                        &mut on_token,
                    )?,
                }
            };

            let ttft_s = generated.ttft_ms / 1000.0;
            let total_s = generated.total_time_ms / 1000.0;
            let prompt_tps = if generated.ttft_ms > 0.0 && prompt_tokens > 0 {
                prompt_tokens as f64 / ttft_s.max(1e-9)
            } else {
                0.0
            };
            let generation_tps = if generated.tokens.is_empty() {
                0.0
            } else {
                generated.tokens.len() as f64 / (total_s - ttft_s).max(1e-9)
            };

            let text = tokenizer.decode(&generated.tokens)?;
            Ok(GenerateResult {
                text,
                prompt_tokens,
                completion_tokens: generated.tokens.len(),
                finish_reason: generated.finish_reason.to_string(),
                ttft_ms: generated.ttft_ms,
                prompt_tps,
                generation_tps,
                total_time_ms: generated.total_time_ms,
            })
        }

        #[cfg(not(feature = "metal"))]
        {
            let _ = (tokenizer, config, params, prompt_tokens, on_token);
            anyhow::bail!(
                "Metal GPU required: rebuild with --no-default-features \
                 --features metal,no-cuda to enable Metal inference"
            )
        }
    }

    /// Build a deterministic benchmark prompt with an exact token count.
    pub fn benchmark_prompt_ids(&self, prompt_tokens: usize) -> Result<Vec<u32>> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .context("model not loaded — call load() first")?;

        if prompt_tokens == 0 {
            return Ok(Vec::new());
        }

        let mut prompt = String::from(BENCHMARK_PROMPT_CHUNK);
        let mut input_ids = tokenizer.encode(&prompt)?;
        while input_ids.len() < prompt_tokens {
            prompt.push_str(BENCHMARK_PROMPT_CHUNK);
            input_ids = tokenizer.encode(&prompt)?;
        }
        input_ids.truncate(prompt_tokens);
        Ok(input_ids)
    }
}

#[cfg(feature = "metal")]
fn run_with_metal_panic_boundary<T>(op: &str, f: impl FnOnce() -> Result<T>) -> Result<T> {
    catch_unwind(AssertUnwindSafe(f)).map_err(|panic| {
        anyhow!(
            "metal backend panicked during {op}: {}",
            panic_message(panic)
        )
    })?
}

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

#[cfg(not(feature = "metal"))]
fn run_with_metal_panic_boundary<T>(_op: &str, f: impl FnOnce() -> Result<T>) -> Result<T> {
    f()
}

#[derive(Clone, Debug, Default)]
pub struct MetalBackendOptions {
    #[cfg(feature = "metal")]
    pub dflash: Option<MetalDflashOptions>,
    #[cfg(feature = "metal")]
    pub kv_pool: Option<bool>,
    #[cfg(feature = "metal")]
    pub runtime_limits: MetalRuntimeLimits,
}

#[cfg(feature = "metal")]
#[derive(Clone, Debug, Default)]
pub struct MetalRuntimeLimits {
    pub memory_limit_bytes: Option<usize>,
    pub cache_limit_bytes: Option<usize>,
    pub wired_limit_bytes: Option<usize>,
}

#[cfg(feature = "metal")]
impl MetalRuntimeLimits {
    fn apply(&self) {
        if let Some(limit) = self.memory_limit_bytes {
            let previous = mlx::set_memory_limit_bytes(limit as u64);
            log::info!(
                "Metal runtime memory limit set to {} bytes (previous {})",
                limit,
                previous
            );
        }
        if let Some(limit) = self.cache_limit_bytes {
            let previous = mlx::set_cache_limit_bytes(limit as u64);
            log::info!(
                "Metal runtime cache limit set to {} bytes (previous {})",
                limit,
                previous
            );
        }
        if let Some(limit) = self.wired_limit_bytes {
            let previous = mlx::set_wired_limit_bytes(limit as u64);
            log::info!(
                "Metal runtime wired limit set to {} bytes (previous {})",
                limit,
                previous
            );
        }
    }
}

impl Default for MetalBackend {
    fn default() -> Self {
        Self::new()
    }
}

// MlxArray wraps a *mut c_void (Metal buffer handle). MLX manages its own
// internal locking, so it is safe to send the backend across threads.
// SAFETY: MetalBackend is used from a single inference thread at a time (the
// scheduler ensures exclusive access). No concurrent mutation occurs.
#[cfg(feature = "metal")]
unsafe impl Send for MetalBackend {}

impl InferenceBackend for MetalBackend {
    fn name(&self) -> &'static str {
        "metal"
    }

    /// Load model from `model_path`.
    ///
    /// `model_path` may be:
    /// - An existing local directory (e.g. `/path/to/Qwen3-0.6B-4bit`)
    /// - A HuggingFace model ID (e.g. `"mlx-community/Qwen3-0.6B-4bit"`)
    fn load(&mut self, model_path: &Path) -> Result<()> {
        #[cfg(feature = "metal")]
        self.runtime_limits.apply();

        // ── 1. Resolve model path ────────────────────────────────────────────
        let path_str = model_path.to_string_lossy();
        let local_dir = hf_hub::resolve_model_path(&path_str)
            .with_context(|| format!("failed to resolve model '{path_str}'"))?;

        log::info!("MetalBackend: loading model from {}", local_dir.display());

        // ── 2. Load tokenizer ────────────────────────────────────────────────
        let tokenizer = Tokenizer::from_file(local_dir.to_str().unwrap_or("."))
            .with_context(|| format!("failed to load tokenizer from {}", local_dir.display()))?;

        // ── 3. Parse config.json ─────────────────────────────────────────────
        let config = load_metal_config(&local_dir)?;

        match &config.arch {
            MetalModelArch::Qwen3 => log::info!(
                "  arch: Qwen3 {} layers, hidden={}, heads={}/{}(kv), vocab={}, eos={}",
                config.num_hidden_layers,
                config.hidden_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.vocab_size,
                config.eos_token_id,
            ),
            MetalModelArch::Qwen35(arch) => log::info!(
                "  arch: Qwen3.5 {} layers (full={}, linear={}), hidden={}, heads={}/{}(kv), vocab={}, eos={}",
                config.num_hidden_layers,
                arch.num_full_attention_layers(),
                arch.num_linear_attention_layers(),
                config.hidden_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.vocab_size,
                config.eos_token_id,
            ),
        }

        // ── 4. Load weights into Metal memory ───────────────────────────────
        #[cfg(feature = "metal")]
        {
            let weights = match &config.arch {
                MetalModelArch::Qwen3 => MetalWeights::Qwen3(
                    self::weights::load_qwen3_metal_weights(&local_dir, &config)
                        .with_context(|| "failed to load Qwen3 weights into Metal memory")?,
                ),
                MetalModelArch::Qwen35(_) => MetalWeights::Qwen35(
                    load_qwen35_metal_weights(&local_dir, &config)
                        .with_context(|| "failed to load Qwen3.5 weights into Metal memory")?,
                ),
            };
            self.weights = Some(weights);
            log::info!("  weights loaded into Metal unified memory");

            let dflash_options = self.dflash_options.clone();
            self.dflash = if let Some(ref options) = dflash_options {
                Some(dflash::MetalDflashRuntime::load(options, &config)?)
            } else {
                None
            };
            self.dflash_options = dflash_options;
        }
        #[cfg(not(feature = "metal"))]
        {
            log::warn!(
                "MetalBackend: compiled without 'metal' feature — \
                 weights not loaded into GPU. Rebuild with --features metal,no-cuda."
            );
        }

        self.tokenizer = Some(tokenizer);
        self.config = Some(config);
        self.model_dir = Some(local_dir);
        Ok(())
    }

    fn generate(&self, prompt: &str, params: &SamplingParams) -> Result<GenerateResult> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .context("model not loaded — call load() first")?;
        let input_ids = tokenizer.encode(prompt)?;
        self.generate_from_token_ids(&input_ids, params)
    }
}

#[cfg(feature = "metal")]
impl StreamingInferenceBackend for MetalBackend {
    fn generate_stream<F>(
        &self,
        prompt: &str,
        params: &SamplingParams,
        on_chunk: F,
    ) -> Result<GenerateResult>
    where
        F: FnMut(&str) -> Result<()>,
    {
        Self::generate_stream(self, prompt, params, on_chunk)
    }
}

const BENCHMARK_PROMPT_CHUNK: &str = " benchmark throughput";
// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(all(test, feature = "metal"))]
mod tests {
    use super::*;
    use crate::backend::InferenceBackend;
    use crate::sampler::SamplingParams;

    fn run_bench(model_dir: &std::path::Path, label: &str) {
        if !model_dir.exists() {
            eprintln!("SKIP: model not found at {}", model_dir.display());
            return;
        }

        let mut backend = MetalBackend::new();
        backend.load(model_dir).expect("load model");

        let prompt = "<|im_start|>user\nWrite a short poem about systems programming.\
                      <|im_end|>\n<|im_start|>assistant\n";
        let params = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };

        // Warm-up pass (fills KV cache, JIT compiles Metal kernels)
        let _ = backend.generate(prompt, &params).expect("warmup");

        // Timed pass
        let result = backend.generate(prompt, &params).expect("generate");

        println!("\n=== Metal Benchmark: {label} ===");
        println!("Model:      {}", model_dir.display());
        println!("Prompt TPS: {:.1} tok/s", result.prompt_tps);
        println!("Gen TPS:    {:.1} tok/s", result.generation_tps);
        println!("TTFT:       {:.1}ms", result.ttft_ms);
        println!("Tokens out: {}", result.completion_tokens);
        println!("Total wall: {:.2}s", result.total_time_ms / 1000.0);
        println!("Finished:   {}", result.finish_reason);
        println!(
            "Output:     {:?}",
            &result.text[..result.text.len().min(200)]
        );
        println!("=================={}===\n", "=".repeat(label.len()));

        assert!(
            result.completion_tokens > 0,
            "should generate at least one token"
        );
    }

    /// Throughput benchmark for a 4-bit quantized MLX model.
    ///
    /// ```bash
    /// cargo test --no-default-features --features metal,no-cuda --release \
    ///     --lib -- bench_4bit --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "requires Metal GPU + model weights at models/Qwen3-0.6B-4bit"]
    fn bench_4bit() {
        let model_dir =
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/Qwen3-0.6B-4bit");
        run_bench(&model_dir, "Qwen3-0.6B 4-bit");
    }

    /// Throughput benchmark for a BF16 MLX model (comparison baseline).
    ///
    /// ```bash
    /// cargo test --no-default-features --features metal,no-cuda --release \
    ///     --lib -- bench_bf16 --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "requires Metal GPU + model weights at models/Qwen3-0.6B-bf16"]
    fn bench_bf16() {
        let model_dir =
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/Qwen3-0.6B-bf16");
        run_bench(&model_dir, "Qwen3-0.6B BF16");
    }
}
