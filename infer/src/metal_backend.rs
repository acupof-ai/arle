//! Metal inference backend for Apple Silicon.
//!
//! Uses `crate::mlx` (thin mlx-sys wrappers) for Metal-accelerated tensor
//! operations.  No mlx-rs dependency.
//!
//! # Architecture
//!
//! Implements a full Qwen2.5 transformer forward pass:
//! - safetensors weight loading into MLX unified memory
//! - RMSNorm, RoPE, GQA attention, SwiGLU MLP
//! - Simple append-based KV cache (one buffer per generate call)
//! - Greedy / temperature + top-p sampling
//!
//! # Feature flag
//!
//! Compile with `--no-default-features --features metal,no-cuda`.
//!
//! # Example
//! ```no_run
//! use infer::metal_backend::MetalBackend;
//! use infer::backend::InferenceBackend;
//! use std::path::Path;
//!
//! let mut backend = MetalBackend::new();
//! backend.load(Path::new("Qwen/Qwen2.5-0.5B-Instruct")).unwrap();
//! ```

use std::path::{Path, PathBuf};
#[cfg(feature = "metal")]
use std::time::Instant;

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
mod loader;
#[cfg(feature = "metal")]
mod qwen35;

// ── mlx types (Metal GPU required) ───────────────────────────────────────────
#[cfg(feature = "metal")]
use crate::metal_kv_pool::MetalKVPool;
#[cfg(feature = "metal")]
use crate::mlx::{Dtype, MlxArray};
use config::{MetalModelArch, MetalModelConfig, load_metal_config};
#[cfg(feature = "metal")]
use config::{MetalQwen35ArchConfig, MetalQwen35LayerType, QuantConfig};
#[cfg(feature = "metal")]
use loader::{
    TensorMap, load_embed_tokens_from_tensors, load_proj_from_tensors, load_tensor_map, tensor_get,
    tie_lm_head_from_embed_tokens,
};
#[cfg(feature = "metal")]
use qwen35::{Qwen35MetalWeights, load_qwen35_metal_weights, metal_generate_qwen35};

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

/// A weight matrix that is either full-precision BF16 or MLX affine 4-bit quantized.
///
/// MLX quantization format (affine per-group):
/// - `w`: packed uint32 — shape `[out, in / (32/bits)]`
/// - `scales`: f16/bf16 — shape `[out, in / group_size]`
/// - `biases`: f16/bf16 — shape `[out, in / group_size]`
///
/// `Dense(w_t)` stores the **transposed** weight `w.T` — shape `[in, out]`.
/// Pre-transposed at load time so `linear()` calls `matmul(x, w_t)` directly,
/// eliminating one `transpose()` view creation per forward-pass call.
// GPU required: MlxArray is backed by Metal buffers.
#[cfg(feature = "metal")]
pub enum WeightTensor {
    /// Pre-transposed weight: shape [in, out] = w.T. Ready for `x @ w_t`.
    Dense(MlxArray),
    Quantized {
        w: MlxArray,
        scales: MlxArray,
        biases: MlxArray,
        group_size: i32,
        bits: i32,
    },
}

#[cfg(feature = "metal")]
impl WeightTensor {
    /// Returns the element dtype: for Dense the array dtype; for Quantized the scales dtype
    /// (scales hold the dequantization type, e.g. bfloat16).
    pub fn dtype(&self) -> Dtype {
        match self {
            WeightTensor::Dense(a) => a.dtype(),
            WeightTensor::Quantized { scales, .. } => scales.dtype(),
        }
    }

    /// Logical output width of the projection.
    ///
    /// Dense tensors are stored transposed as `[in, out]`; quantized tensors
    /// keep MLX packed layout `[out, packed_in]`.
    pub fn output_dim(&self) -> Result<i32> {
        let shape = match self {
            WeightTensor::Dense(w_t) => w_t.shape(),
            WeightTensor::Quantized { w, .. } => w.shape(),
        };

        match self {
            WeightTensor::Dense(_) => shape
                .get(1)
                .copied()
                .context("dense projection missing output dimension"),
            WeightTensor::Quantized { .. } => shape
                .first()
                .copied()
                .context("quantized projection missing output dimension"),
        }
    }
}

/// Attention input projections for one transformer layer.
#[cfg(feature = "metal")]
pub enum AttentionInputProjection {
    /// Dense/non-merged path.
    Split {
        q_proj: WeightTensor,
        k_proj: WeightTensor,
        v_proj: WeightTensor,
    },
    /// Quantized fallback path with a single `qkv` matmul.
    MergedQuantized {
        qkv_proj: WeightTensor,
        q_dim: i32,
        k_dim: i32,
        v_dim: i32,
    },
}

#[cfg(feature = "metal")]
impl AttentionInputProjection {
    fn kv_dtype(&self) -> Dtype {
        match self {
            Self::Split { k_proj, .. } => k_proj.dtype(),
            Self::MergedQuantized { qkv_proj, .. } => qkv_proj.dtype(),
        }
    }

    fn project(&self, x: &MlxArray) -> (MlxArray, MlxArray, MlxArray) {
        use crate::mlx::slice;

        match self {
            Self::Split {
                q_proj,
                k_proj,
                v_proj,
            } => (linear(x, q_proj), linear(x, k_proj), linear(x, v_proj)),
            Self::MergedQuantized {
                qkv_proj,
                q_dim,
                k_dim,
                v_dim,
            } => {
                let qkv = linear(x, qkv_proj);
                let seq = qkv.shape()[0];
                let q_raw = slice(&qkv, &[0, 0], &[seq, *q_dim], &[1, 1]);
                let k_raw = slice(&qkv, &[0, *q_dim], &[seq, *q_dim + *k_dim], &[1, 1]);
                let v_raw = slice(
                    &qkv,
                    &[0, *q_dim + *k_dim],
                    &[seq, *q_dim + *k_dim + *v_dim],
                    &[1, 1],
                );
                (q_raw, k_raw, v_raw)
            }
        }
    }
}

/// MLP input projections for one transformer layer.
#[cfg(feature = "metal")]
pub enum MlpInputProjection {
    /// Dense/non-merged path.
    Split {
        gate_proj: WeightTensor,
        up_proj: WeightTensor,
    },
    /// Quantized fallback path with a single `gate+up` matmul.
    MergedQuantized {
        gate_up_proj: WeightTensor,
        gate_dim: i32,
        up_dim: i32,
    },
}

#[cfg(feature = "metal")]
impl MlpInputProjection {
    fn project(&self, x: &MlxArray) -> (MlxArray, MlxArray) {
        use crate::mlx::slice;

        match self {
            Self::Split { gate_proj, up_proj } => (linear(x, gate_proj), linear(x, up_proj)),
            Self::MergedQuantized {
                gate_up_proj,
                gate_dim,
                up_dim,
            } => {
                let gate_up = linear(x, gate_up_proj);
                let seq = gate_up.shape()[0];
                let gate_raw = slice(&gate_up, &[0, 0], &[seq, *gate_dim], &[1, 1]);
                let up_raw = slice(
                    &gate_up,
                    &[0, *gate_dim],
                    &[seq, *gate_dim + *up_dim],
                    &[1, 1],
                );
                (gate_raw, up_raw)
            }
        }
    }
}

/// Weight tensors loaded from safetensors shards into Metal unified memory.
// GPU required: all fields are MlxArrays backed by Metal buffers.
#[cfg(feature = "metal")]
struct StandardMetalWeights {
    /// Token embedding table — shape [vocab_size, hidden_size] (dequantized to float at load).
    pub embed_tokens: MlxArray,
    /// Per-layer attention + MLP weights.
    pub layers: Vec<StandardMetalLayerWeights>,
    /// Final layer-norm scale — shape [hidden_size].
    pub norm: MlxArray,
    /// Output projection (lm_head) — dense or quantized.
    pub lm_head: WeightTensor,
    /// Quantized embed weights for as_linear lm_head.
    pub embed_quantized: Option<WeightTensor>,
    /// C++ compiled model (reuses Qwen3.5 model struct with all full-attn layers).
    pub cpp_model: Option<qwen35::CppQwen35Model>,
}

#[cfg(feature = "metal")]
enum MetalWeights {
    Qwen3(StandardMetalWeights),
    Qwen35(Qwen35MetalWeights),
}

/// Weights for a single transformer layer.
// GPU required: all fields are MlxArrays or WeightTensors.
#[cfg(feature = "metal")]
struct StandardMetalLayerWeights {
    pub attention_inputs: AttentionInputProjection,
    pub o_proj: WeightTensor,
    pub mlp_inputs: MlpInputProjection,
    pub down_proj: WeightTensor,
    pub q_norm: MlxArray,
    pub k_norm: MlxArray,
    pub input_layernorm: MlxArray,
    pub post_attention_layernorm: MlxArray,
    /// Individual q/k/v projections for C++ model (separate from merged attention_inputs).
    pub q_proj_individual: Option<WeightTensor>,
    pub k_proj_individual: Option<WeightTensor>,
    pub v_proj_individual: Option<WeightTensor>,
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
    }

    pub fn generate_from_token_ids(
        &self,
        input_ids: &[u32],
        params: &SamplingParams,
    ) -> Result<GenerateResult> {
        self.generate_from_token_ids_with_callback(input_ids, params, |_token_id| Ok(()))
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
    /// - An existing local directory (e.g. `/path/to/Qwen2.5-0.5B-Instruct`)
    /// - A HuggingFace model ID (e.g. `"Qwen/Qwen2.5-0.5B-Instruct"`)
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
                    load_qwen3_metal_weights(&local_dir, &config)
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
/// Transformer layer (Qwen2.5):
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
    use crate::mlx::zeros;

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
    validate_metal_sampling_params(params)?;

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
    let prefill_token = build_forward_graph(
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
    metal_async_eval(&prefill_token)?;
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
                extend_kv_cache(&mut k_caches[li], n_kv_heads, head_dim, new_cap)?;
                extend_kv_cache(&mut v_caches[li], n_kv_heads, head_dim, new_cap)?;
            }
            kv_capacity = new_cap;
        }

        // P5: release accumulated temporary Metal allocations every 256 steps.
        if decode_step > 0 && decode_step.is_multiple_of(256) {
            clear_metal_cache();
        }

        let new_pending = build_forward_graph(
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
        metal_async_eval(&new_pending)?;

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

#[cfg(feature = "metal")]
fn metal_async_eval(arr: &MlxArray) -> Result<()> {
    crate::mlx::async_eval(&[arr]);
    Ok(())
}

#[cfg(feature = "metal")]
fn clear_metal_cache() {
    crate::mlx::clear_cache();
}

#[cfg(feature = "metal")]
fn extend_kv_cache(
    cache: &mut MlxArray,
    n_kv_heads: i32,
    head_dim: i32,
    new_cap: i32,
) -> Result<()> {
    use crate::mlx::{concatenate_axis, zeros};

    let current_cap = cache.shape().get(2).copied().unwrap_or_default();
    if new_cap <= current_cap {
        return Ok(());
    }

    let extra = zeros(
        &[1, n_kv_heads, new_cap - current_cap, head_dim],
        cache.dtype(),
    );
    *cache = concatenate_axis(&[cache.clone(), extra], 2);
    Ok(())
}

/// Build one forward-pass compute graph (lazy — no GPU work until eval/async_eval).
///
/// Returns an unsynchronised token `MlxArray` (greedy argmax or categorical sample).
/// The caller must `async_eval` or `eval` then call `.item_i32()` to materialise the token.
// GPU required: all ops register Metal-backed lazy computation nodes.
#[cfg(feature = "metal")]
#[allow(clippy::too_many_arguments)]
fn build_forward_graph(
    current_ids: &[u32],
    weights: &StandardMetalWeights,
    k_caches: &mut [MlxArray],
    v_caches: &mut [MlxArray],
    cache_len: i32,
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    attn_scale: f32,
    rope_base: f32,
    eps: f32,
    mut metal_kv_pool: Option<&mut MetalKVPool>,
    request_id: usize,
    params: &SamplingParams,
) -> Result<MlxArray> {
    use crate::mlx::{rms_norm, take_axis};

    let seq = current_ids.len() as i32;
    if let Some(pool) = metal_kv_pool.as_deref_mut() {
        pool.alloc_tokens(request_id, seq as usize)
            .context("alloc MetalKVPool slots")?;
    }

    // ── Embedding lookup ─────────────────────────────────────────────────────
    let idx_i32: Vec<i32> = current_ids.iter().map(|&t| t as i32).collect();
    let idx_arr = MlxArray::from_slice_i32(&idx_i32, &[seq]);
    let mut x = take_axis(&weights.embed_tokens, &idx_arr, 0);

    // ── Transformer layers ────────────────────────────────────────────────────
    for (li, layer) in weights.layers.iter().enumerate() {
        x = rust_transformer_layer(
            x,
            layer,
            li,
            k_caches,
            v_caches,
            seq,
            cache_len,
            n_heads,
            n_kv_heads,
            head_dim,
            attn_scale,
            rope_base,
            eps,
            metal_kv_pool.as_deref_mut(),
            request_id,
        )?;
    }

    // ── Final norm + lm_head ─────────────────────────────────────────────────
    let last_idx = MlxArray::from_slice_i32(&[seq - 1], &[1]);
    let last_x = take_axis(&x, &last_idx, 0);
    let last_x = rms_norm(&last_x, &weights.norm, eps);
    let logits = linear(&last_x, &weights.lm_head); // [1, vocab]

    // P4: GPU-side sampling — stays on GPU, only scalar crosses on .item() later.
    gpu_sample_token(&logits, params)
}

/// Single transformer layer for the maintained Rust/MLX Qwen3 path.
// GPU required
#[cfg(feature = "metal")]
#[allow(clippy::too_many_arguments, clippy::needless_pass_by_value)]
fn rust_transformer_layer(
    x: MlxArray,
    layer: &StandardMetalLayerWeights,
    li: usize,
    k_caches: &mut [MlxArray],
    v_caches: &mut [MlxArray],
    seq: i32,
    cache_len: i32,
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    attn_scale: f32,
    rope_base: f32,
    eps: f32,
    mut metal_kv_pool: Option<&mut MetalKVPool>,
    request_id: usize,
) -> Result<MlxArray> {
    use crate::mlx::{
        add, multiply, reshape, rms_norm, rope, scaled_dot_product_attention, silu, slice,
        slice_update, transpose_axes,
    };

    // 1. Input norm + residual
    let residual = x.clone();
    let x = rms_norm(&x, &layer.input_layernorm, eps);
    // 2. QKV projections
    let (q_raw, k_raw, v_raw) = layer.attention_inputs.project(&x);

    // 3+4+5+6. Reshape → per-head norm → transpose → RoPE.
    //
    // rope expects (B, *, T, D) where T is the second-to-last dim (sequence axis).
    // Transpose to [1, n_heads, seq, d] BEFORE rope so T = seq (correct positions).
    let q = reshape(&q_raw, &[1, seq, n_heads, head_dim]);
    let q = rms_norm(&q, &layer.q_norm, eps);
    let q = transpose_axes(&q, &[0, 2, 1, 3]); // [1, n_heads, seq, d]
    let q = rope(&q, head_dim, false, rope_base, 1.0f32, cache_len);

    let k = reshape(&k_raw, &[1, seq, n_kv_heads, head_dim]);
    let k = rms_norm(&k, &layer.k_norm, eps);
    let k = transpose_axes(&k, &[0, 2, 1, 3]); // [1, n_kv, seq, d]
    let k = rope(&k, head_dim, false, rope_base, 1.0f32, cache_len);

    let v = reshape(&v_raw, &[1, seq, n_kv_heads, head_dim]);
    let v = transpose_axes(&v, &[0, 2, 1, 3]); // [1, n_kv, seq, d]

    // 7. KV cache update
    let (k_full, v_full) = if let Some(pool) = metal_kv_pool.as_deref_mut() {
        let k_rows = transpose_axes(&k, &[0, 2, 1, 3]);
        let k_rows = reshape(&k_rows, &[seq, n_kv_heads * head_dim]);
        let v_rows = transpose_axes(&v, &[0, 2, 1, 3]);
        let v_rows = reshape(&v_rows, &[seq, n_kv_heads * head_dim]);
        pool.write_kv(li, request_id, &k_rows, &v_rows)
            .context("write MetalKVPool")?;
        pool.gather_kv(li, request_id)
            .context("gather MetalKVPool")?
    } else {
        let end_pos = cache_len + seq;
        k_caches[li] = slice_update(
            &mut k_caches[li],
            &k,
            &[0, 0, cache_len, 0],
            &[1, n_kv_heads, end_pos, head_dim],
        );
        v_caches[li] = slice_update(
            &mut v_caches[li],
            &v,
            &[0, 0, cache_len, 0],
            &[1, n_kv_heads, end_pos, head_dim],
        );
        // Read back the full KV sequence
        let k_full = slice(
            &k_caches[li],
            &[0, 0, 0, 0],
            &[1, n_kv_heads, end_pos, head_dim],
            &[1, 1, 1, 1],
        );
        let v_full = slice(
            &v_caches[li],
            &[0, 0, 0, 0],
            &[1, n_kv_heads, end_pos, head_dim],
            &[1, 1, 1, 1],
        );
        (k_full, v_full)
    };

    // 8. Attention
    let use_causal = cache_len == 0 && seq > 1;
    let mask_arg = if use_causal { Some("causal") } else { None };
    let attn_out = scaled_dot_product_attention(&q, &k_full, &v_full, attn_scale, mask_arg);

    // 9. Reshape + output proj + residual
    let attn_out = transpose_axes(&attn_out, &[0, 2, 1, 3]);
    let attn_out = reshape(&attn_out, &[seq, n_heads * head_dim]);
    let attn_out = linear(&attn_out, &layer.o_proj);
    let x = add(&residual, &attn_out);

    // 10. MLP
    let residual2 = x.clone();
    let xn = rms_norm(&x, &layer.post_attention_layernorm, eps);
    let (gate_raw, up) = layer.mlp_inputs.project(&xn);
    let gate = silu(&gate_raw);
    let mlp = linear(&multiply(&gate, &up), &layer.down_proj);
    Ok(add(&residual2, &mlp))
}

#[cfg(feature = "metal")]
fn validate_metal_sampling_params(params: &SamplingParams) -> Result<()> {
    let mut unsupported = Vec::new();

    if params.top_k != -1 && params.top_k != 1 {
        unsupported.push(format!("top_k={}", params.top_k));
    }
    if params.top_p < 1.0 {
        unsupported.push(format!("top_p={}", params.top_p));
    }
    if params.min_p > 0.0 {
        unsupported.push(format!("min_p={}", params.min_p));
    }
    if params.repetition_penalty != 1.0 {
        unsupported.push(format!("repetition_penalty={}", params.repetition_penalty));
    }
    if params.frequency_penalty != 0.0 {
        unsupported.push(format!("frequency_penalty={}", params.frequency_penalty));
    }
    if params.presence_penalty != 0.0 {
        unsupported.push(format!("presence_penalty={}", params.presence_penalty));
    }
    if let Some(seed) = params.seed {
        unsupported.push(format!("seed={seed}"));
    }

    anyhow::ensure!(
        unsupported.is_empty(),
        "Metal backend currently supports only temperature sampling and greedy decoding (top_k=1); unsupported sampling params: {}",
        unsupported.join(", ")
    );

    Ok(())
}

/// P4 — GPU-side sampling: argmax or categorical, stays on GPU until `.item()`.
#[cfg(feature = "metal")]
fn gpu_sample_token(logits: &MlxArray, params: &SamplingParams) -> Result<MlxArray> {
    if params.temperature <= 1e-6 || params.top_k == 1 {
        return Ok(greedy_sample_token(logits));
    }

    // Temperature scaling then GPU categorical sample.
    let inv_t = MlxArray::scalar_f32(1.0f32 / params.temperature);
    let scaled = crate::mlx::multiply(logits, &inv_t);
    Ok(categorical_sample_token(&scaled))
}

#[cfg(feature = "metal")]
fn greedy_sample_token(logits: &MlxArray) -> MlxArray {
    crate::mlx::argmax(logits)
}

#[cfg(feature = "metal")]
fn categorical_sample_token(logits: &MlxArray) -> MlxArray {
    crate::mlx::categorical(logits)
}

/// `x @ weight.T` — no bias, dispatches to dense matmul or quantized matmul.
///
/// For `Dense(w_t)`, `w_t` is already transposed at load time (shape `[in, out]`),
/// so this is just `matmul(x, w_t)` without an extra transpose.
#[cfg(feature = "metal")]
#[inline]
fn linear(x: &MlxArray, weight: &WeightTensor) -> MlxArray {
    match weight {
        WeightTensor::Dense(w_t) => {
            // w_t is pre-transposed [in, out]; direct matmul, no per-call transpose.
            crate::mlx::matmul(x, w_t)
        }
        WeightTensor::Quantized {
            w,
            scales,
            biases,
            group_size,
            bits,
        } => {
            // w stored as [out, in] packed uint32; transpose=true → x @ w.T
            crate::mlx::quantized_matmul(x, w, scales, biases, true, *group_size, *bits)
        }
    }
}

#[cfg(feature = "metal")]
fn merge_quantized_projection_rows(weights: &[&WeightTensor]) -> Result<Option<WeightTensor>> {
    use crate::mlx::{concatenate_axis, eval};

    if weights.is_empty() {
        return Ok(None);
    }

    let mut ws = Vec::with_capacity(weights.len());
    let mut scales = Vec::with_capacity(weights.len());
    let mut biases = Vec::with_capacity(weights.len());
    let mut expected_w_dtype = None;
    let mut expected_scales_dtype = None;
    let mut expected_biases_dtype = None;
    let mut expected_group_size = None;
    let mut expected_bits = None;
    let mut expected_packed_in = None;
    let mut expected_group_cols = None;

    for weight in weights {
        let WeightTensor::Quantized {
            w,
            scales: scale,
            biases: bias,
            group_size,
            bits,
        } = weight
        else {
            return Ok(None);
        };

        let packed_in = w
            .shape()
            .get(1)
            .copied()
            .context("quantized projection missing packed input dimension")?;
        let group_cols = scale
            .shape()
            .get(1)
            .copied()
            .context("quantized projection missing scale group dimension")?;

        if let Some(expected) = expected_group_size {
            if *group_size != expected {
                return Ok(None);
            }
        } else {
            expected_group_size = Some(*group_size);
        }

        if let Some(expected) = expected_bits {
            if *bits != expected {
                return Ok(None);
            }
        } else {
            expected_bits = Some(*bits);
        }

        if let Some(expected) = expected_packed_in {
            if packed_in != expected {
                return Ok(None);
            }
        } else {
            expected_packed_in = Some(packed_in);
        }

        if let Some(expected) = expected_group_cols {
            if group_cols != expected {
                return Ok(None);
            }
        } else {
            expected_group_cols = Some(group_cols);
        }

        if let Some(expected) = expected_w_dtype {
            if w.dtype() != expected {
                return Ok(None);
            }
        } else {
            expected_w_dtype = Some(w.dtype());
        }

        if let Some(expected) = expected_scales_dtype {
            if scale.dtype() != expected {
                return Ok(None);
            }
        } else {
            expected_scales_dtype = Some(scale.dtype());
        }

        if let Some(expected) = expected_biases_dtype {
            if bias.dtype() != expected {
                return Ok(None);
            }
        } else {
            expected_biases_dtype = Some(bias.dtype());
        }

        ws.push(w.clone());
        scales.push(scale.clone());
        biases.push(bias.clone());
    }

    let merged_w = concatenate_axis(&ws, 0);
    let merged_scales = concatenate_axis(&scales, 0);
    let merged_biases = concatenate_axis(&biases, 0);
    eval(&[&merged_w, &merged_scales, &merged_biases]);

    Ok(Some(WeightTensor::Quantized {
        w: merged_w,
        scales: merged_scales,
        biases: merged_biases,
        group_size: expected_group_size.unwrap_or_default(),
        bits: expected_bits.unwrap_or_default(),
    }))
}

// ── Weight loading (Metal GPU required) ───────────────────────────────────────

/// Load safetensors shards into Metal unified memory.
// GPU required: MlxArray is backed by Metal buffers.
#[cfg(feature = "metal")]
fn load_qwen3_metal_weights(
    model_dir: &Path,
    config: &MetalModelConfig,
) -> Result<StandardMetalWeights> {
    let tensors = load_tensor_map(model_dir)?;
    let get = |name: &str| tensor_get(&tensors, name);
    let load_proj = |base: &str| load_proj_from_tensors(&tensors, base, config.quantization);

    // embed_tokens: dequantize at load time if quantized (needed for embedding lookup via take_axis).
    let embed_tokens =
        load_embed_tokens_from_tensors(&tensors, "model.embed_tokens", config.quantization)?;

    let norm = get("model.norm.weight")?;

    // lm_head may be weight-tied to embed_tokens; handle both dense and quantized.
    // When tied, use quantized_matmul on the original packed weights (as_linear pattern)
    // instead of dense matmul on dequantized bf16 — saves ~7.5ms/step for large vocabs.
    let lm_head =
        if tensors.contains_key("lm_head.weight") || tensors.contains_key("lm_head.scales") {
            load_proj("lm_head")?
        } else if config.quantization.is_some() {
            // Tied + quantized: load original packed embed weights for quantized_matmul
            load_proj("model.embed_tokens")?
        } else {
            tie_lm_head_from_embed_tokens(&embed_tokens)?
        };

    let n = config.num_hidden_layers;
    let mut layers = Vec::with_capacity(n);
    for i in 0..n {
        let p = |s: &str| format!("model.layers.{i}.{s}");
        let q_proj = load_proj(&p("self_attn.q_proj"))?;
        let k_proj = load_proj(&p("self_attn.k_proj"))?;
        let v_proj = load_proj(&p("self_attn.v_proj"))?;
        let q_dim = q_proj.output_dim()?;
        let k_dim = k_proj.output_dim()?;
        let v_dim = v_proj.output_dim()?;
        let attention_inputs = if let Some(qkv_proj) =
            merge_quantized_projection_rows(&[&q_proj, &k_proj, &v_proj])?
        {
            AttentionInputProjection::MergedQuantized {
                qkv_proj,
                q_dim,
                k_dim,
                v_dim,
            }
        } else {
            AttentionInputProjection::Split {
                q_proj,
                k_proj,
                v_proj,
            }
        };

        let gate_proj = load_proj(&p("mlp.gate_proj"))?;
        let up_proj = load_proj(&p("mlp.up_proj"))?;
        let gate_dim = gate_proj.output_dim()?;
        let up_dim = up_proj.output_dim()?;
        let mlp_inputs =
            if let Some(gate_up_proj) = merge_quantized_projection_rows(&[&gate_proj, &up_proj])? {
                MlpInputProjection::MergedQuantized {
                    gate_up_proj,
                    gate_dim,
                    up_dim,
                }
            } else {
                MlpInputProjection::Split { gate_proj, up_proj }
            };

        // Reload individual q/k/v for C++ model (separate from merged)
        let q_ind = load_proj(&p("self_attn.q_proj")).ok();
        let k_ind = load_proj(&p("self_attn.k_proj")).ok();
        let v_ind = load_proj(&p("self_attn.v_proj")).ok();

        layers.push(StandardMetalLayerWeights {
            attention_inputs,
            o_proj: load_proj(&p("self_attn.o_proj"))?,
            mlp_inputs,
            down_proj: load_proj(&p("mlp.down_proj"))?,
            q_norm: get(&p("self_attn.q_norm.weight"))?,
            k_norm: get(&p("self_attn.k_norm.weight"))?,
            input_layernorm: get(&p("input_layernorm.weight"))?,
            post_attention_layernorm: get(&p("post_attention_layernorm.weight"))?,
            q_proj_individual: q_ind,
            k_proj_individual: k_ind,
            v_proj_individual: v_ind,
        });
    }

    // Load quantized embed for as_linear lm_head
    let embed_quantized = if config.quantization.is_some() {
        load_proj("model.embed_tokens").ok()
    } else {
        None
    };

    // Build C++ model (all full-attn layers — reuses Qwen3.5 model struct)
    let cpp_model = build_qwen3_cpp_model(
        &embed_tokens,
        &norm,
        &lm_head,
        embed_quantized.as_ref(),
        &layers,
        config,
    );

    Ok(StandardMetalWeights {
        embed_tokens,
        layers,
        norm,
        lm_head,
        embed_quantized,
        cpp_model,
    })
}

/// Build C++ compiled model for Qwen3 (pure transformer, all full-attn layers).
/// Reuses the Qwen3.5 C++ model struct with zero GDR layers.
#[cfg(feature = "metal")]
fn build_qwen3_cpp_model(
    embed_tokens: &MlxArray,
    norm: &MlxArray,
    lm_head: &WeightTensor,
    embed_quantized: Option<&WeightTensor>,
    layers: &[StandardMetalLayerWeights],
    config: &MetalModelConfig,
) -> Option<qwen35::CppQwen35Model> {
    use qwen35::CppQwen35Model;

    let model = unsafe { mlx_sys::qwen35_compiled_new() };
    if model.is_null() {
        return None;
    }

    unsafe {
        mlx_sys::qwen35_compiled_set_config(
            model,
            config.rope_theta as f32,
            config.rms_norm_eps as f32,
            config.num_attention_heads as i32,
            config.num_key_value_heads as i32,
            config.head_dim as i32,
            config.head_dim as i32, // rotary_dim = head_dim for Qwen3
            config.hidden_size as i32,
        );
    }

    // Embed + norm + lm_head
    let (lm_w, lm_s, lm_b, lm_gs, lm_bits) = match lm_head {
        WeightTensor::Quantized {
            w,
            scales,
            biases,
            group_size,
            bits,
        } => (
            w.as_raw(),
            scales.as_raw(),
            biases.as_raw(),
            *group_size,
            *bits,
        ),
        WeightTensor::Dense(w_t) => (
            w_t.as_raw(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            0,
        ),
    };
    unsafe {
        mlx_sys::qwen35_compiled_set_embed(
            model,
            embed_tokens.as_raw(),
            norm.as_raw(),
            lm_w,
            lm_s,
            lm_b,
            lm_gs,
            lm_bits,
        );
        if let Some(WeightTensor::Quantized {
            w,
            scales,
            biases,
            group_size,
            bits,
        }) = embed_quantized
        {
            mlx_sys::qwen35_compiled_set_embed_as_linear(
                model,
                w.as_raw(),
                scales.as_raw(),
                biases.as_raw(),
                *group_size,
                *bits,
            );
        }
    }

    // All layers are full attention
    for layer in layers {
        let extract = |wt: &WeightTensor| -> Option<(
            *mut mlx_sys::mlx_array,
            *mut mlx_sys::mlx_array,
            *mut mlx_sys::mlx_array,
            i32,
            i32,
        )> {
            match wt {
                WeightTensor::Quantized {
                    w,
                    scales,
                    biases,
                    group_size,
                    bits,
                } => Some((
                    w.as_raw(),
                    scales.as_raw(),
                    biases.as_raw(),
                    *group_size,
                    *bits,
                )),
                WeightTensor::Dense(_) => None,
            }
        };

        // Extract attention projections from AttentionInputProjection
        let (q, k, v) = match &layer.attention_inputs {
            AttentionInputProjection::Split {
                q_proj,
                k_proj,
                v_proj,
            } => match (extract(q_proj), extract(k_proj), extract(v_proj)) {
                (Some(q), Some(k), Some(v)) => (q, k, v),
                _ => {
                    unsafe { mlx_sys::qwen35_compiled_free(model) };
                    return None;
                }
            },
            AttentionInputProjection::MergedQuantized { .. } => {
                // Use individual weights stored alongside merged
                match (
                    layer.q_proj_individual.as_ref().and_then(|w| extract(w)),
                    layer.k_proj_individual.as_ref().and_then(|w| extract(w)),
                    layer.v_proj_individual.as_ref().and_then(|w| extract(w)),
                ) {
                    (Some(q), Some(k), Some(v)) => (q, k, v),
                    _ => {
                        unsafe { mlx_sys::qwen35_compiled_free(model) };
                        return None;
                    }
                }
            }
        };
        let o = match extract(&layer.o_proj) {
            Some(o) => o,
            None => {
                unsafe { mlx_sys::qwen35_compiled_free(model) };
                return None;
            }
        };

        // MLP
        let (gu_w, gu_s, gu_b, gu_gs, gu_bits, gate_dim) = match &layer.mlp_inputs {
            MlpInputProjection::MergedQuantized {
                gate_up_proj:
                    WeightTensor::Quantized {
                        w,
                        scales,
                        biases,
                        group_size,
                        bits,
                    },
                gate_dim,
                ..
            } => (
                w.as_raw(),
                scales.as_raw(),
                biases.as_raw(),
                *group_size,
                *bits,
                *gate_dim,
            ),
            _ => {
                unsafe { mlx_sys::qwen35_compiled_free(model) };
                return None;
            }
        };
        let (dw_w, dw_s, dw_b) = match extract(&layer.down_proj) {
            Some((w, s, b, _, _)) => (w, s, b),
            None => {
                unsafe { mlx_sys::qwen35_compiled_free(model) };
                return None;
            }
        };

        unsafe {
            mlx_sys::qwen35_compiled_push_full_attn(
                model,
                layer.input_layernorm.as_raw(),
                layer.post_attention_layernorm.as_raw(),
                q.0,
                q.1,
                q.2,
                q.3,
                q.4,
                k.0,
                k.1,
                k.2,
                v.0,
                v.1,
                v.2,
                o.0,
                o.1,
                o.2,
                layer.q_norm.as_raw(),
                layer.k_norm.as_raw(),
                gu_w,
                gu_s,
                gu_b,
                gu_gs,
                gu_bits,
                gate_dim,
                dw_w,
                dw_s,
                dw_b,
            );
        }
    }

    let rc = unsafe { mlx_sys::qwen35_compiled_finalize(model) };
    if rc != 0 {
        unsafe { mlx_sys::qwen35_compiled_free(model) };
        return None;
    }

    log::info!(
        "  C++ Qwen3 model ready ({} full-attn layers)",
        layers.len()
    );
    Some(CppQwen35Model::from_raw(model))
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
