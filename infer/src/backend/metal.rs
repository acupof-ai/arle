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
use anyhow::{Context, Result};

#[cfg(feature = "metal")]
use crate::backend::StreamingInferenceBackend;
use crate::{
    backend::{GenerateResult, InferenceBackend},
    hf_hub,
    sampler::SamplingParams,
    tokenizer::Tokenizer,
};

mod config;
#[cfg(feature = "metal")]
pub mod forward;
#[cfg(feature = "metal")]
mod loader;
#[cfg(feature = "metal")]
pub mod ops;
#[cfg(feature = "metal")]
mod qwen35;
#[cfg(feature = "metal")]
pub mod sampling;
#[cfg(feature = "metal")]
pub mod weights;

// Submodules that used to live at the crate root as `metal_*.rs` — moved
// under `metal_backend/` so that all Metal-specific code lives in one place
// (see `docs/plans/` for the reorganisation plan).
pub mod gdr;
pub mod kv_pool;
#[cfg(feature = "metal")]
pub mod mlx;
pub mod prefix_cache;
pub mod scheduler;

// ── mlx types (Metal GPU required) ───────────────────────────────────────────
#[cfg(feature = "metal")]
use self::kv_pool::MetalKVPool;
#[cfg(feature = "metal")]
use self::mlx::MlxArray;
#[cfg(feature = "metal")]
use self::ops::{clear_metal_cache, extend_kv_cache, linear};
#[cfg(feature = "metal")]
use self::sampling::gpu_sample_token;
#[cfg(feature = "metal")]
use self::weights::{
    MetalWeights, MlpInputProjection, StandardMetalWeights, WeightTensor,
    merge_quantized_projection_rows,
};
use config::{
    MetalModelArch, MetalModelConfig, MetalQwen35ArchConfig, MetalQwen35LayerType, QuantConfig,
    load_metal_config,
};
#[cfg(feature = "metal")]
use loader::{
    TensorMap, load_embed_tokens_from_tensors, load_proj_from_tensors, load_tensor_map, tensor_get,
    tie_lm_head_from_embed_tokens,
};
#[cfg(feature = "metal")]
use qwen35::{load_qwen35_metal_weights, metal_generate_qwen35};

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
    #[cfg(not(feature = "metal"))]
    _weights: (),
}

impl MetalBackend {
    pub fn new() -> Self {
        Self {
            model_dir: None,
            tokenizer: None,
            config: None,
            #[cfg(feature = "metal")]
            weights: None,
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

            let generated = match weights {
                MetalWeights::Qwen3(weights) => metal_generate(
                    input_ids,
                    weights,
                    config,
                    params,
                    max_new_tokens,
                    t0,
                    &mut on_token,
                )?,
                MetalWeights::Qwen35(weights) => metal_generate_qwen35(
                    {
                        if metal_kv_pool_enabled() {
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

// ── Metal forward pass (GPU required) ────────────────────────────────────────

/// Autoregressive generation using MLX Metal kernels.
///
/// Forward pass per step:
/// ```text
/// input_ids → embed → [transformer layer × N] → norm → lm_head → logits → sample
/// ```
///
/// Transformer layer (Qwen3 / Qwen3.5):
/// 1. `residual = x`
/// 2. `x = rms_norm(x, input_layernorm)`
/// 3. `q = rms_norm(x @ q_proj.T, q_norm)`, `k = rms_norm(x @ k_proj.T, k_norm)`
/// 4. `v = x @ v_proj.T`
/// 5. Reshape, apply RoPE to q/k, append to KV cache
/// 6. GQA via `fast::scaled_dot_product_attention`
/// 7. `x = residual + attn_out @ o_proj.T`
/// 8. `x = x + silu(x_norm @ gate.T) * (x_norm @ up.T) @ down.T`
///
/// Optimisations applied:
/// - **P1**: Dense weights are pre-transposed at load time (no per-step transpose).
/// - **P3/P6**: async_eval double-buffering: GPU computes step N while CPU processes
///   step N-1's token. Graph construction (CPU-only) overlaps GPU execution.
/// - **P4**: argmax / categorical sampling stays on GPU (no 152 K float transfer).
/// - **P5**: KV cache grows in 256-token chunks; `metal_clear_cache()` every 256 steps.
// P5: KV cache grows in this many token increments (aligned with mlx-lm convention).
#[cfg(feature = "metal")]
const KV_CACHE_CHUNK: i32 = 256;
#[cfg(feature = "metal")]
const METAL_KV_POOL_REQUEST_ID: usize = 0;

const BENCHMARK_PROMPT_CHUNK: &str = " benchmark throughput";

#[cfg(feature = "metal")]
struct MetalGenerateOutput {
    tokens: Vec<u32>,
    finish_reason: &'static str,
    ttft_ms: f64,
    total_time_ms: f64,
}

// TODO: dead code for current use cases — MetalKVPool is only activated via
// the `AGENT_INFER_METAL_KV_POOL=1` env var and only wired into the Qwen3
// fallback path. Qwen3.5 bypasses it entirely (see the warning at the call
// site). Consider removing if the KV pool experiment is abandoned.
#[cfg(feature = "metal")]
fn metal_kv_pool_enabled() -> bool {
    std::env::var("AGENT_INFER_METAL_KV_POOL")
        .ok()
        .is_some_and(|value| metal_kv_pool_flag_is_truthy(&value))
}

#[cfg(feature = "metal")]
struct MetalKvPoolRequestCleanup {
    pool: *mut MetalKVPool,
    request_id: usize,
}

#[cfg(feature = "metal")]
impl MetalKvPoolRequestCleanup {
    fn new(pool: &mut MetalKVPool, request_id: usize) -> Self {
        Self {
            pool: pool as *mut MetalKVPool,
            request_id,
        }
    }
}

#[cfg(feature = "metal")]
impl Drop for MetalKvPoolRequestCleanup {
    fn drop(&mut self) {
        if self.pool.is_null() {
            return;
        }

        // SAFETY: the guard is created inside `metal_generate` after the pool
        // and is dropped before the pool goes out of scope.
        unsafe {
            (&mut *self.pool).free_request(self.request_id);
        }
    }
}

#[cfg(feature = "metal")]
fn metal_kv_pool_flag_is_truthy(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

// GPU required: all tensor operations use MlxArray on Metal unified memory.
#[cfg(feature = "metal")]
fn metal_generate(
    input_ids: &[u32],
    weights: &StandardMetalWeights,
    config: &MetalModelConfig,
    params: &SamplingParams,
    max_new_tokens: usize,
    t0: Instant,
    on_token: &mut impl FnMut(u32) -> Result<()>,
) -> Result<MetalGenerateOutput> {
    use self::mlx::zeros;

    if max_new_tokens == 0 {
        return Ok(MetalGenerateOutput {
            tokens: Vec::new(),
            finish_reason: "length",
            ttft_ms: 0.0,
            total_time_ms: 0.0,
        });
    }

    // C++ full generate path (same as Qwen3.5 — batch prefill + double-buffered decode)
    if let Some(ref cpp_model) = weights.cpp_model {
        log::info!("Metal Qwen3: C++ full generate");
        let mut stop_ids: Vec<u32> = params.stop_token_ids.clone();
        if !params.ignore_eos {
            stop_ids.push(config.eos_token_id);
        }
        let (tokens, prefill_ms, decode_ms) = cpp_model.generate(
            input_ids,
            max_new_tokens,
            params.temperature,
            &stop_ids,
            on_token,
        )?;
        let total_time_ms = prefill_ms + decode_ms;
        let finish_reason = if tokens.last().is_some_and(|t| stop_ids.contains(t)) {
            "stop"
        } else {
            "length"
        };
        return Ok(MetalGenerateOutput {
            tokens,
            finish_reason,
            ttft_ms: prefill_ms,
            total_time_ms,
        });
    }

    let n_layers = config.num_hidden_layers;
    let n_heads = config.num_attention_heads as i32;
    let n_kv_heads = config.num_key_value_heads as i32;
    let head_dim = config.head_dim as i32;
    let eps = config.rms_norm_eps as f32;
    let rope_base = config.rope_theta as f32;
    let attn_scale = 1.0f32 / (head_dim as f32).sqrt();
    let use_kv_pool = metal_kv_pool_enabled();
    self::sampling::validate_metal_sampling_params(params)?;

    log::info!("Metal transformer path: Rust/MLX");
    if use_kv_pool {
        log::info!("MetalKVPool enabled via AGENT_INFER_METAL_KV_POOL=1");
    }

    // P5: KV cache starts at the next 256-token boundary above the prefill length,
    // plus one chunk for initial decode steps.  Grown lazily in KV_CACHE_CHUNK steps.
    let prefill_len = input_ids.len() as i32;
    let initial_cap = ((prefill_len + KV_CACHE_CHUNK - 1) / KV_CACHE_CHUNK + 1) * KV_CACHE_CHUNK;
    let mut kv_capacity = initial_cap;

    let kv_dtype = weights.layers[0].attention_inputs.kv_dtype();
    let cache_shape = [1i32, n_kv_heads, initial_cap, head_dim];
    let mut k_caches: Vec<MlxArray> = (0..n_layers)
        .map(|_| zeros(&cache_shape, kv_dtype))
        .collect();
    let mut v_caches: Vec<MlxArray> = (0..n_layers)
        .map(|_| zeros(&cache_shape, kv_dtype))
        .collect();
    let mut kv_pool = if use_kv_pool {
        let pool_tokens = std::cmp::max(
            initial_cap as usize,
            input_ids.len().saturating_add(max_new_tokens),
        );
        Some(
            MetalKVPool::new(
                n_layers,
                n_kv_heads as usize,
                head_dim as usize,
                pool_tokens,
                kv_dtype,
            )
            .context("pre-alloc MetalKVPool")?,
        )
    } else {
        None
    };
    let _kv_pool_cleanup = kv_pool
        .as_mut()
        .map(|pool| MetalKvPoolRequestCleanup::new(pool, METAL_KV_POOL_REQUEST_ID));

    let mut generated: Vec<u32> = Vec::new();
    let mut cache_len: i32 = 0;
    let mut ttft_ms = 0.0;

    // ── Phase 1: Prefill — build lazy graph, schedule GPU asynchronously ─────
    let prefill_token = self::forward::build_forward_graph(
        input_ids,
        weights,
        &mut k_caches,
        &mut v_caches,
        cache_len,
        n_heads,
        n_kv_heads,
        head_dim,
        attn_scale,
        rope_base,
        eps,
        kv_pool.as_mut(),
        METAL_KV_POOL_REQUEST_ID,
        params,
    )?;
    // P6: schedule GPU execution without blocking CPU.
    self::ops::metal_async_eval(&prefill_token)?;
    cache_len += prefill_len;

    // ── Phase 2: Decode loop (double-buffered — P3/P6) ────────────────────────
    //
    // Each iteration: sync *previous* token via item() (GPU likely done since
    // async_eval), then build *next* graph and async_eval it before looping.
    // CPU graph-build overlaps with GPU execution of the current step.
    let mut pending = prefill_token;
    let mut decode_step: usize = 0;

    let finish_reason = loop {
        let next_token = pending.item_i32() as u32;

        if decode_step == 0 {
            ttft_ms = t0.elapsed().as_secs_f64() * 1000.0;
            log::info!(
                "  TTFT: {ttft_ms:.1}ms (prefill {} tokens)",
                input_ids.len()
            );
        }

        let stop = (!params.ignore_eos && config.is_stop_token(next_token))
            || params.stop_token_ids.contains(&next_token);
        generated.push(next_token);
        on_token(next_token)?;

        if stop {
            break "stop";
        }
        if generated.len() >= max_new_tokens {
            break "length";
        }

        // P5: grow KV cache in 256-token chunks when capacity is about to overflow.
        if !use_kv_pool && cache_len + 1 > kv_capacity {
            let new_cap = kv_capacity + KV_CACHE_CHUNK;
            for li in 0..n_layers {
                self::ops::extend_kv_cache(&mut k_caches[li], n_kv_heads, head_dim, new_cap)?;
                self::ops::extend_kv_cache(&mut v_caches[li], n_kv_heads, head_dim, new_cap)?;
            }
            kv_capacity = new_cap;
        }

        // P5: release accumulated temporary Metal allocations every 256 steps.
        if decode_step > 0 && decode_step.is_multiple_of(256) {
            self::ops::clear_metal_cache();
        }

        let new_pending = self::forward::build_forward_graph(
            &[next_token],
            weights,
            &mut k_caches,
            &mut v_caches,
            cache_len,
            n_heads,
            n_kv_heads,
            head_dim,
            attn_scale,
            rope_base,
            eps,
            kv_pool.as_mut(),
            METAL_KV_POOL_REQUEST_ID,
            params,
        )?;
        cache_len += 1;
        decode_step += 1;

        // P6: kick off GPU — CPU syncs at top of next iteration via item().
        self::ops::metal_async_eval(&new_pending)?;

        pending = new_pending;
    };

    let elapsed = t0.elapsed().as_secs_f64();
    let total_time_ms = elapsed * 1000.0;
    let decode_elapsed = (elapsed - ttft_ms / 1000.0).max(1e-9);
    let tps = generated.len() as f64 / decode_elapsed;
    log::info!("  generated {} tokens  ({tps:.1} tok/s)", generated.len());

    Ok(MetalGenerateOutput {
        tokens: generated,
        finish_reason,
        ttft_ms,
        total_time_ms,
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(all(test, feature = "metal"))]
mod tests {
    use super::*;
    use crate::backend::InferenceBackend;
    use crate::sampler::SamplingParams;

    #[test]
    fn kv_pool_flag_parser_accepts_common_truthy_values() {
        for value in ["1", "true", "TRUE", "yes", "on", " 1 "] {
            assert!(
                metal_kv_pool_flag_is_truthy(value),
                "{value:?} should be truthy"
            );
        }
    }

    #[test]
    fn kv_pool_flag_parser_rejects_falsey_values() {
        for value in ["", "0", "false", "off", "no", "maybe"] {
            assert!(
                !metal_kv_pool_flag_is_truthy(value),
                "{value:?} should be falsey"
            );
        }
    }

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
