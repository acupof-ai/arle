use std::{
    collections::HashSet,
    path::Path,
    sync::{Mutex, OnceLock},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow, ensure};
use serde::Deserialize;

use super::{
    config::{MetalModelArch, MetalModelConfig, QuantConfig},
    forward::rust_transformer_layer,
    generate::{KV_CACHE_CHUNK, MetalGenerateOutput},
    loader::{load_proj_from_tensors, load_tensor_map, tensor_get},
    mlx::{MlxArray, concatenate_axis, eval, rms_norm, slice, take_axis, zeros},
    ops::{extend_kv_cache, linear},
    sampling::{gpu_sample_token, validate_metal_sampling_params},
    weights::{MlpInputProjection, StandardMetalWeights, WeightTensor},
};
use crate::{hf_hub, sampler::SamplingParams};

/// Draft KV cache sink size (attention-sink tokens kept at the start).
const DRAFT_CACHE_SINK_SIZE: i32 = 64;
/// Draft KV cache window size (recent tokens kept at the end).
const DRAFT_CACHE_WINDOW_SIZE: i32 = 1024;

/// Rolling aggregate profile over N blocks. Captures the full phase
/// breakdown + K-histogram so we can read the real bottleneck instead of
/// guessing from single-block samples.
#[derive(Default)]
struct Qwen35BlockProfileWindow {
    blocks: usize,
    block_size: usize,
    draft: Vec<Duration>,
    verify: Vec<Duration>,
    sample: Vec<Duration>,
    rollback: Vec<Duration>,
    eval: Vec<Duration>,
    total: Vec<Duration>,
    k_hist: Vec<usize>, // k_hist[k] = #blocks that accepted exactly k
    k_total: usize,     // sum of all accepted K (for mean)
    /// Per-position agreement: pos_match[i] = #blocks where draft[i+1] == posterior[i]
    /// computed over ALL block_size-1 draft positions, NOT short-circuited at
    /// first mismatch. High K=0 with non-trivial pos_match[5..10] means draft
    /// recovers after early mismatch (sticky-drift bug). Low pos_match[0]
    /// means the very first draft step is off (draft forward / rope / cache bug).
    pos_match: Vec<usize>,
}

impl Qwen35BlockProfileWindow {
    fn reset(&mut self, block_size: usize) {
        self.blocks = 0;
        self.block_size = block_size;
        self.draft.clear();
        self.verify.clear();
        self.sample.clear();
        self.rollback.clear();
        self.eval.clear();
        self.total.clear();
        self.k_hist.clear();
        self.k_hist.resize(block_size + 1, 0);
        self.k_total = 0;
        self.pos_match.clear();
        self.pos_match.resize(block_size.saturating_sub(1), 0);
    }
}

#[allow(clippy::too_many_arguments)]
fn record_qwen35_block_profile(
    block_size: usize,
    accepted_k: usize,
    draft: Duration,
    verify: Duration,
    sample: Duration,
    rollback: Duration,
    eval: Duration,
    total: Duration,
    per_pos_match: &[bool],
) {
    static WINDOW: OnceLock<Mutex<Qwen35BlockProfileWindow>> = OnceLock::new();
    let window = WINDOW.get_or_init(|| Mutex::new(Qwen35BlockProfileWindow::default()));
    let mut state = window.lock().expect("Qwen35 block profile window poisoned");
    if state.blocks == 0 {
        state.reset(block_size);
    }
    state.blocks += 1;
    state.draft.push(draft);
    state.verify.push(verify);
    state.sample.push(sample);
    state.rollback.push(rollback);
    state.eval.push(eval);
    state.total.push(total);
    if accepted_k < state.k_hist.len() {
        state.k_hist[accepted_k] += 1;
    }
    state.k_total += accepted_k;
    for (i, &hit) in per_pos_match.iter().take(state.pos_match.len()).enumerate() {
        if hit {
            state.pos_match[i] += 1;
        }
    }

    const WINDOW_BLOCKS: usize = 50;
    if state.blocks >= WINDOW_BLOCKS {
        let mean = |v: &[Duration]| -> f64 {
            v.iter().map(|d| d.as_secs_f64()).sum::<f64>() / v.len() as f64 * 1000.0
        };
        let quantile = |v: &mut Vec<Duration>, q: f64| -> f64 {
            v.sort();
            let idx = ((v.len() - 1) as f64 * q) as usize;
            v[idx].as_secs_f64() * 1000.0
        };
        let mut draft_v = state.draft.clone();
        let mut verify_v = state.verify.clone();
        let mut total_v = state.total.clone();
        let mean_k = state.k_total as f64 / state.blocks as f64;
        let mean_total_ms = mean(&state.total);
        let effective_tok_s = 1000.0 * mean_k / mean_total_ms;

        let mut hist_s = String::new();
        for (k, count) in state.k_hist.iter().enumerate() {
            if *count > 0 {
                let pct = 100.0 * *count as f64 / state.blocks as f64;
                hist_s.push_str(&format!(" K{k}:{count}({pct:.0}%)"));
            }
        }

        log::info!(
            "qwen35_dflash[agg {} blocks]: draft μ={:.1}ms p90={:.1}ms | verify μ={:.1}ms p90={:.1}ms | sample μ={:.1}ms | rollback μ={:.1}ms | eval μ={:.1}ms | total μ={:.1}ms p90={:.1}ms | K̄={:.2}/{} | eff={:.1} tok/s",
            state.blocks,
            mean(&state.draft),
            quantile(&mut draft_v, 0.90),
            mean(&state.verify),
            quantile(&mut verify_v, 0.90),
            mean(&state.sample),
            mean(&state.rollback),
            mean(&state.eval),
            mean_total_ms,
            quantile(&mut total_v, 0.90),
            mean_k,
            state.block_size,
            effective_tok_s,
        );
        log::info!("qwen35_dflash[agg K-hist]:{hist_s}");
        // Per-position draft↔target agreement (not short-circuited at first
        // mismatch). Reads the shape of the acceptance curve: flat-low means
        // the very first draft step is off; high-then-cliff means drift accrues.
        let mut pos_s = String::new();
        for (i, hits) in state.pos_match.iter().enumerate() {
            let pct = 100.0 * *hits as f64 / state.blocks as f64;
            pos_s.push_str(&format!(" p{}:{:.0}%", i + 1, pct));
        }
        log::info!("qwen35_dflash[agg pos-agree]:{pos_s}");
        state.reset(block_size);
    }
}

#[derive(Clone, Debug)]
pub struct MetalDflashOptions {
    pub draft_model: String,
    pub speculative_tokens: Option<usize>,
}

impl MetalDflashOptions {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.draft_model.trim().is_empty(),
            "Metal DFlash draft model must not be empty"
        );
        if let Some(tokens) = self.speculative_tokens {
            ensure!(
                tokens > 0,
                "Metal DFlash speculative token override must be >= 1 when set"
            );
        }
        Ok(())
    }
}

pub(crate) struct MetalDflashRuntime {
    block_size: usize,
    mask_token_id: u32,
    target_layer_ids: Vec<usize>,
    draft_model_id: String,
    draft_config: DFlashDraftConfig,
    draft_weights: DFlashDraftWeights,
    draft_cpp_model: Option<DFlashDraftCppModel>,
    /// Attention mask mode inside the draft block's self-attention ("causal" or "none").
    /// Matches dflash-mlx api.py auto-select: causal for Qwen3.5 hybrid, none for
    /// Qwen3. `DFLASH_DRAFT_MASK=causal|none` overrides. Reference benches show
    /// causal beats none on Qwen3.5 (K̄ 6.10 vs 5.37).
    draft_attention_mask: String,
}

impl MetalDflashRuntime {
    pub(crate) fn load(
        options: &MetalDflashOptions,
        target_config: &MetalModelConfig,
    ) -> Result<Self> {
        options.validate()?;
        ensure!(
            matches!(
                target_config.arch,
                MetalModelArch::Qwen3 | MetalModelArch::Qwen35(_)
            ),
            "Metal DFlash requires Qwen3 or Qwen3.5 target model"
        );

        let draft_model_dir = hf_hub::resolve_weighted_model_path(&options.draft_model)
            .with_context(|| {
                format!(
                    "failed to resolve DFlash draft model '{}'",
                    options.draft_model
                )
            })?;
        let draft_config = DFlashDraftConfig::load(&draft_model_dir)?;
        ensure!(
            !draft_config.target_layer_ids.is_empty(),
            "DFlash draft config must declare at least one target layer id"
        );
        ensure!(
            draft_config.hidden_size == target_config.hidden_size,
            "DFlash draft hidden_size {} != target hidden_size {}",
            draft_config.hidden_size,
            target_config.hidden_size
        );

        let draft_weights = DFlashDraftWeights::load(&draft_model_dir, &draft_config)
            .with_context(|| {
                format!(
                    "failed to load DFlash draft weights from {}",
                    draft_model_dir.display()
                )
            })?;
        let draft_cpp_model = DFlashDraftCppModel::build(&draft_weights, &draft_config);
        let default_block_size = draft_config.block_size.max(1);
        let requested_block_size = options
            .speculative_tokens
            .unwrap_or(default_block_size)
            .max(1);
        if let Some(requested) = options.speculative_tokens {
            if requested < default_block_size {
                log::warn!(
                    "Metal DFlash speculative block override {} is below the draft default {}; this can reduce acceptance and throughput",
                    requested,
                    default_block_size
                );
            } else if requested > default_block_size {
                log::warn!(
                    "Metal DFlash speculative block override {} exceeds the draft default {}; clamping to {}",
                    requested,
                    default_block_size,
                    default_block_size
                );
            }
        }
        let block_size = requested_block_size.min(default_block_size);

        // Reference dflash-mlx picks "causal" for Qwen3.5 (K̄=6.10 vs
        // none=5.37). Our measured K̄ inverts that (none=4.54 > causal=3.60),
        // likely an MLX-C / fast_sdpa causal-with-q_len<k_len discrepancy
        // still to root-cause. Default "none" until fixed;
        // DFLASH_DRAFT_MASK=causal forces causal for investigation.
        let auto_mask = "none";
        let draft_attention_mask = std::env::var("DFLASH_DRAFT_MASK")
            .ok()
            .map(|v| v.to_lowercase())
            .filter(|v| v == "causal" || v == "none")
            .unwrap_or_else(|| auto_mask.to_string());

        log::info!(
            "Metal DFlash enabled: draft='{}', block_size={}, draft_attention_mask={}, target_layers={:?}",
            options.draft_model,
            block_size,
            draft_attention_mask,
            draft_config.target_layer_ids
        );

        Ok(Self {
            block_size,
            mask_token_id: draft_config.mask_token_id,
            target_layer_ids: draft_config.target_layer_ids.clone(),
            draft_model_id: options.draft_model.clone(),
            draft_config,
            draft_weights,
            draft_cpp_model,
            draft_attention_mask,
        })
    }

    pub(crate) fn draft_model_id(&self) -> &str {
        &self.draft_model_id
    }

    pub(crate) fn target_layer_ids(&self) -> &[usize] {
        &self.target_layer_ids
    }

    pub(crate) fn draft_num_hidden_layers(&self) -> usize {
        self.draft_config.num_hidden_layers
    }

    pub(crate) fn draft_n_kv_heads(&self) -> i32 {
        self.draft_config.num_key_value_heads as i32
    }

    pub(crate) fn draft_head_dim(&self) -> i32 {
        self.draft_config.head_dim as i32
    }
}

impl MetalDflashRuntime {
    pub(crate) fn block_size(&self) -> usize {
        self.block_size
    }
}

#[derive(Clone, Debug)]
pub(crate) struct DFlashDraftConfig {
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    block_size: usize,
    mask_token_id: u32,
    target_layer_ids: Vec<usize>,
    quantization: Option<QuantConfig>,
}

#[derive(Debug, Deserialize)]
struct RawDraftQuantConfig {
    group_size: Option<i32>,
    bits: Option<i32>,
}

#[derive(Debug, Deserialize)]
struct RawDflashConfig {
    target_layer_ids: Vec<usize>,
    mask_token_id: u32,
}

#[derive(Debug, Deserialize)]
struct RawDraftConfig {
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rms_norm_eps: f64,
    rope_theta: f64,
    block_size: Option<usize>,
    dflash_config: RawDflashConfig,
    quantization: Option<RawDraftQuantConfig>,
    quantization_config: Option<RawDraftQuantConfig>,
}

impl DFlashDraftConfig {
    fn load(model_dir: &Path) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let raw: RawDraftConfig = serde_json::from_str(
            &std::fs::read_to_string(&config_path)
                .with_context(|| format!("cannot read {}", config_path.display()))?,
        )
        .with_context(|| format!("cannot parse {}", config_path.display()))?;

        let quant_source = raw.quantization.or(raw.quantization_config);
        let quantization = quant_source.map(|q| QuantConfig {
            group_size: q.group_size.unwrap_or(64),
            bits: q.bits.unwrap_or(4),
        });

        Ok(Self {
            hidden_size: raw.hidden_size,
            num_hidden_layers: raw.num_hidden_layers,
            num_attention_heads: raw.num_attention_heads,
            num_key_value_heads: raw.num_key_value_heads,
            head_dim: raw.head_dim,
            rms_norm_eps: raw.rms_norm_eps as f32,
            rope_theta: raw.rope_theta as f32,
            block_size: raw.block_size.unwrap_or(16),
            mask_token_id: raw.dflash_config.mask_token_id,
            target_layer_ids: raw.dflash_config.target_layer_ids,
            quantization,
        })
    }
}

struct DFlashDraftLayerWeights {
    q_proj: WeightTensor,
    k_proj: WeightTensor,
    v_proj: WeightTensor,
    o_proj: WeightTensor,
    input_layernorm: MlxArray,
    post_attention_layernorm: MlxArray,
    q_norm: MlxArray,
    k_norm: MlxArray,
    gate_proj: WeightTensor,
    up_proj: WeightTensor,
    mlp_inputs: MlpInputProjection,
    down_proj: WeightTensor,
}

struct DFlashDraftWeights {
    layers: Vec<DFlashDraftLayerWeights>,
    fc: WeightTensor,
    hidden_norm: MlxArray,
    norm: MlxArray,
}

impl DFlashDraftWeights {
    fn load(model_dir: &Path, config: &DFlashDraftConfig) -> Result<Self> {
        let tensors = load_tensor_map(model_dir)?;
        let get = |name: &str| tensor_get(&tensors, name);
        let load_proj = |base: &str| load_proj_from_tensors(&tensors, base, config.quantization);

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let p = |suffix: &str| format!("layers.{i}.{suffix}");

            let gate_proj = load_proj(&p("mlp.gate_proj"))?;
            let up_proj = load_proj(&p("mlp.up_proj"))?;
            let gate_dim = gate_proj.output_dim()?;
            let up_dim = up_proj.output_dim()?;
            let mlp_inputs = if let Some(gate_up_proj) =
                super::weights::merge_quantized_projection_rows(&[&gate_proj, &up_proj])?
            {
                MlpInputProjection::MergedQuantized {
                    gate_up_proj,
                    gate_dim,
                    up_dim,
                }
            } else {
                MlpInputProjection::Split {
                    gate_proj: clone_weight_tensor(&gate_proj),
                    up_proj: clone_weight_tensor(&up_proj),
                }
            };

            layers.push(DFlashDraftLayerWeights {
                q_proj: load_proj(&p("self_attn.q_proj"))?,
                k_proj: load_proj(&p("self_attn.k_proj"))?,
                v_proj: load_proj(&p("self_attn.v_proj"))?,
                o_proj: load_proj(&p("self_attn.o_proj"))?,
                input_layernorm: get(&p("input_layernorm.weight"))?,
                post_attention_layernorm: get(&p("post_attention_layernorm.weight"))?,
                q_norm: get(&p("self_attn.q_norm.weight"))?,
                k_norm: get(&p("self_attn.k_norm.weight"))?,
                gate_proj,
                up_proj,
                mlp_inputs,
                down_proj: load_proj(&p("mlp.down_proj"))?,
            });
        }

        Ok(Self {
            layers,
            fc: load_proj("fc")?,
            hidden_norm: get("hidden_norm.weight")?,
            norm: get("norm.weight")?,
        })
    }
}

fn clone_weight_tensor(weight: &WeightTensor) -> WeightTensor {
    match weight {
        WeightTensor::Dense(w) => WeightTensor::Dense(w.clone()),
        WeightTensor::Quantized {
            w,
            scales,
            biases,
            group_size,
            bits,
        } => WeightTensor::Quantized {
            w: w.clone(),
            scales: scales.clone(),
            biases: biases.clone(),
            group_size: *group_size,
            bits: *bits,
        },
    }
}

fn extract_dflash_weight(
    weight: &WeightTensor,
) -> (
    *mut mlx_sys::mlx_array,
    *mut mlx_sys::mlx_array,
    *mut mlx_sys::mlx_array,
    i32,
    i32,
) {
    match weight {
        WeightTensor::Dense(w) => (w.as_raw(), std::ptr::null_mut(), std::ptr::null_mut(), 0, 0),
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
    }
}

fn use_dflash_draft_cpp() -> bool {
    matches!(std::env::var("DFLASH_DRAFT_CPP").as_deref(), Ok("1"))
}

struct DFlashDraftCppModel(*mut std::ffi::c_void);

impl Drop for DFlashDraftCppModel {
    fn drop(&mut self) {
        unsafe { mlx_sys::dflash_draft_free(self.0) }
    }
}

unsafe impl Send for DFlashDraftCppModel {}

impl DFlashDraftCppModel {
    fn build(weights: &DFlashDraftWeights, config: &DFlashDraftConfig) -> Option<Self> {
        let model = unsafe { mlx_sys::dflash_draft_new() };
        if model.is_null() {
            log::warn!("DFlash draft C++ model init failed; falling back to Rust path");
            return None;
        }

        unsafe {
            mlx_sys::dflash_draft_set_config(
                model,
                config.hidden_size as i32,
                config.num_attention_heads as i32,
                config.num_key_value_heads as i32,
                config.head_dim as i32,
                config.num_hidden_layers as i32,
                config.rope_theta,
                config.rms_norm_eps,
            );
        }

        for layer in &weights.layers {
            let q = extract_dflash_weight(&layer.q_proj);
            let k = extract_dflash_weight(&layer.k_proj);
            let v = extract_dflash_weight(&layer.v_proj);
            let o = extract_dflash_weight(&layer.o_proj);
            let gate = extract_dflash_weight(&layer.gate_proj);
            let up = extract_dflash_weight(&layer.up_proj);
            let down = extract_dflash_weight(&layer.down_proj);
            unsafe {
                mlx_sys::dflash_draft_push_layer(
                    model,
                    q.0,
                    q.1,
                    q.2,
                    q.3,
                    q.4,
                    k.0,
                    k.1,
                    k.2,
                    k.3,
                    k.4,
                    v.0,
                    v.1,
                    v.2,
                    v.3,
                    v.4,
                    o.0,
                    o.1,
                    o.2,
                    o.3,
                    o.4,
                    gate.0,
                    gate.1,
                    gate.2,
                    gate.3,
                    gate.4,
                    up.0,
                    up.1,
                    up.2,
                    up.3,
                    up.4,
                    down.0,
                    down.1,
                    down.2,
                    down.3,
                    down.4,
                    layer.input_layernorm.as_raw(),
                    layer.post_attention_layernorm.as_raw(),
                    layer.q_norm.as_raw(),
                    layer.k_norm.as_raw(),
                );
            }
        }

        let fc = extract_dflash_weight(&weights.fc);
        unsafe {
            mlx_sys::dflash_draft_set_fc_norms(
                model,
                fc.0,
                fc.1,
                fc.2,
                fc.3,
                fc.4,
                weights.hidden_norm.as_raw(),
                weights.norm.as_raw(),
            );
        }

        let rc = unsafe { mlx_sys::dflash_draft_finalize(model) };
        if rc != 0 {
            log::warn!("DFlash draft C++ model finalize failed; falling back to Rust path");
            unsafe { mlx_sys::dflash_draft_free(model) };
            return None;
        }

        log::info!(
            "Metal DFlash draft C++ model ready ({} layers compiled as one forward graph)",
            config.num_hidden_layers
        );
        Some(Self(model))
    }

    fn forward(
        &self,
        noise_embedding: &MlxArray,
        target_hidden: &MlxArray,
        rope_offset: i32,
        kv_flat: &mut [MlxArray],
    ) -> Result<MlxArray> {
        let n_kv = kv_flat.len() as i32;
        let mut kv_ptrs: Vec<*mut mlx_sys::mlx_array> =
            kv_flat.iter().map(MlxArray::as_raw).collect();
        let mut out_hidden: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let mut out_kv: Vec<*mut mlx_sys::mlx_array> = vec![std::ptr::null_mut(); kv_flat.len()];
        let rc = unsafe {
            mlx_sys::dflash_draft_forward(
                self.0,
                noise_embedding.as_raw(),
                target_hidden.as_raw(),
                kv_ptrs.as_mut_ptr(),
                n_kv,
                rope_offset,
                &raw mut out_hidden,
                out_kv.as_mut_ptr(),
            )
        };
        if rc != 0 {
            return Err(super::mlx::check_mlx_error().unwrap_err());
        }
        for (slot, ptr) in kv_flat.iter_mut().zip(out_kv.into_iter()) {
            let old = std::mem::replace(slot, unsafe { MlxArray::from_raw(ptr) });
            drop(old);
        }
        Ok(unsafe { MlxArray::from_raw(out_hidden) })
    }
}

pub(crate) struct ContiguousKvState {
    k_caches: Vec<MlxArray>,
    v_caches: Vec<MlxArray>,
    len: i32,
    capacity: i32,
    n_kv_heads: i32,
    head_dim: i32,
    /// Cumulative context position for RoPE (may diverge from `len` after
    /// sink+window eviction compacts the physical cache).
    rope_offset: i32,
}

impl ContiguousKvState {
    pub(crate) fn new(
        num_layers: usize,
        n_kv_heads: i32,
        head_dim: i32,
        initial_tokens: usize,
    ) -> Self {
        let initial_cap = ((i32::try_from(initial_tokens).unwrap_or_default() + KV_CACHE_CHUNK
            - 1)
            / KV_CACHE_CHUNK
            + 1)
            * KV_CACHE_CHUNK;
        let cache_shape = [1i32, n_kv_heads, initial_cap.max(KV_CACHE_CHUNK), head_dim];
        let mut k_caches = Vec::with_capacity(num_layers);
        let mut v_caches = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            k_caches.push(zeros(&cache_shape, super::mlx::Dtype::Bfloat16));
            v_caches.push(zeros(&cache_shape, super::mlx::Dtype::Bfloat16));
        }
        Self {
            k_caches,
            v_caches,
            len: 0,
            capacity: cache_shape[2],
            n_kv_heads,
            head_dim,
            rope_offset: 0,
        }
    }

    pub(crate) fn from_dtype(
        num_layers: usize,
        n_kv_heads: i32,
        head_dim: i32,
        initial_tokens: usize,
        dtype: super::mlx::Dtype,
    ) -> Self {
        let initial_cap = ((i32::try_from(initial_tokens).unwrap_or_default() + KV_CACHE_CHUNK
            - 1)
            / KV_CACHE_CHUNK
            + 1)
            * KV_CACHE_CHUNK;
        let cache_shape = [1i32, n_kv_heads, initial_cap.max(KV_CACHE_CHUNK), head_dim];
        let mut k_caches = Vec::with_capacity(num_layers);
        let mut v_caches = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            k_caches.push(zeros(&cache_shape, dtype));
            v_caches.push(zeros(&cache_shape, dtype));
        }
        Self {
            k_caches,
            v_caches,
            len: 0,
            capacity: cache_shape[2],
            n_kv_heads,
            head_dim,
            rope_offset: 0,
        }
    }

    fn ensure_capacity(&mut self, required_len: i32) {
        if required_len <= self.capacity {
            return;
        }
        let new_capacity = ((required_len + KV_CACHE_CHUNK - 1) / KV_CACHE_CHUNK) * KV_CACHE_CHUNK;
        for cache in &mut self.k_caches {
            extend_kv_cache(cache, self.n_kv_heads, self.head_dim, new_capacity);
        }
        for cache in &mut self.v_caches {
            extend_kv_cache(cache, self.n_kv_heads, self.head_dim, new_capacity);
        }
        self.capacity = new_capacity;
    }

    fn active_kv_flat(&self) -> Vec<MlxArray> {
        let mut flat = Vec::with_capacity(self.k_caches.len() * 2);
        for layer_idx in 0..self.k_caches.len() {
            flat.push(slice(
                &self.k_caches[layer_idx],
                &[0, 0, 0, 0],
                &[1, self.n_kv_heads, self.len, self.head_dim],
                &[1, 1, 1, 1],
            ));
            flat.push(slice(
                &self.v_caches[layer_idx],
                &[0, 0, 0, 0],
                &[1, self.n_kv_heads, self.len, self.head_dim],
                &[1, 1, 1, 1],
            ));
        }
        flat
    }

    fn replace_active_kv_flat(&mut self, flat: Vec<MlxArray>) -> Result<()> {
        ensure!(
            flat.len() == self.k_caches.len() * 2,
            "DFlash active KV replacement count mismatch: expected {}, got {}",
            self.k_caches.len() * 2,
            flat.len()
        );
        let mut iter = flat.into_iter();
        let mut new_capacity = 0;
        for layer_idx in 0..self.k_caches.len() {
            let new_k = iter
                .next()
                .ok_or_else(|| anyhow!("missing DFlash K cache for layer {layer_idx}"))?;
            let new_v = iter
                .next()
                .ok_or_else(|| anyhow!("missing DFlash V cache for layer {layer_idx}"))?;
            let k_shape = new_k.shape();
            let v_shape = new_v.shape();
            ensure!(
                k_shape.len() == 4
                    && v_shape.len() == 4
                    && k_shape[0] == 1
                    && v_shape[0] == 1
                    && k_shape[1] == self.n_kv_heads
                    && v_shape[1] == self.n_kv_heads
                    && k_shape[3] == self.head_dim
                    && v_shape[3] == self.head_dim
                    && k_shape[2] == v_shape[2],
                "invalid DFlash KV cache shapes for layer {layer_idx}: k={k_shape:?}, v={v_shape:?}"
            );
            new_capacity = k_shape[2];
            self.k_caches[layer_idx] = new_k;
            self.v_caches[layer_idx] = new_v;
        }
        self.capacity = new_capacity;
        Ok(())
    }

    fn trim(&mut self, num_tokens: usize) {
        let delta = i32::try_from(num_tokens).unwrap_or_default();
        self.len = self.len.saturating_sub(delta);
        self.rope_offset = self.rope_offset.saturating_sub(delta);
    }

    /// Sink+window eviction for the draft cache. Keeps the first `sink_size`
    /// entries and the last `window_size` entries, discarding the middle.
    /// `rope_offset` is NOT changed — cached K/V retain their original RoPE.
    fn apply_window(&mut self, sink_size: i32, window_size: i32) {
        let max_len = sink_size + window_size;
        if self.len <= max_len || max_len <= 0 {
            return;
        }
        let window_start = self.len - window_size;
        for layer in 0..self.k_caches.len() {
            let k_win = slice(
                &self.k_caches[layer],
                &[0, 0, window_start, 0],
                &[1, self.n_kv_heads, self.len, self.head_dim],
                &[1, 1, 1, 1],
            );
            self.k_caches[layer] = super::mlx::slice_update(
                &mut self.k_caches[layer],
                &k_win,
                &[0, 0, sink_size, 0],
                &[1, self.n_kv_heads, max_len, self.head_dim],
            );
            let v_win = slice(
                &self.v_caches[layer],
                &[0, 0, window_start, 0],
                &[1, self.n_kv_heads, self.len, self.head_dim],
                &[1, 1, 1, 1],
            );
            self.v_caches[layer] = super::mlx::slice_update(
                &mut self.v_caches[layer],
                &v_win,
                &[0, 0, sink_size, 0],
                &[1, self.n_kv_heads, max_len, self.head_dim],
            );
        }
        self.len = max_len;
    }
}

pub(crate) fn metal_generate_dflash_qwen3(
    runtime: &MetalDflashRuntime,
    input_ids: &[u32],
    weights: &StandardMetalWeights,
    config: &MetalModelConfig,
    params: &SamplingParams,
    max_new_tokens: usize,
    t0: Instant,
    on_token: &mut impl FnMut(u32) -> Result<()>,
) -> Result<MetalGenerateOutput> {
    ensure!(
        !input_ids.is_empty(),
        "Metal DFlash requires at least one prompt token"
    );
    validate_metal_sampling_params(params)?;

    if max_new_tokens == 0 {
        return Ok(MetalGenerateOutput {
            tokens: Vec::new(),
            finish_reason: "length",
            ttft_ms: 0.0,
            total_time_ms: 0.0,
        });
    }

    let dtype = weights.layers[0].attention_inputs.kv_dtype();
    let mut target_state = ContiguousKvState::from_dtype(
        config.num_hidden_layers,
        config.num_key_value_heads as i32,
        config.head_dim as i32,
        input_ids.len() + max_new_tokens,
        dtype,
    );
    let mut draft_state = ContiguousKvState::new(
        runtime.draft_config.num_hidden_layers,
        runtime.draft_config.num_key_value_heads as i32,
        runtime.draft_config.head_dim as i32,
        input_ids.len() + max_new_tokens,
    );

    let (prompt_norm_hidden, mut target_hidden) = qwen3_forward_with_hidden_states(
        input_ids,
        weights,
        config,
        &runtime.target_layer_ids,
        &mut target_state,
    )?;
    let prompt_logits = linear(&prompt_norm_hidden, &weights.lm_head);
    let first_token = sample_last_token(&prompt_logits, params)?;
    let ttft_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let mut generated = vec![first_token];
    on_token(first_token)?;
    if is_stop_token(config, params, first_token) || generated.len() >= max_new_tokens {
        let total_time_ms = t0.elapsed().as_secs_f64() * 1000.0;
        return Ok(MetalGenerateOutput {
            tokens: generated,
            finish_reason: if is_stop_token(config, params, first_token) {
                "stop"
            } else {
                "length"
            },
            ttft_ms,
            total_time_ms,
        });
    }

    let mut current_token = first_token;
    let mut acceptance_lengths = Vec::new();
    let finish_reason = loop {
        let mut block_tokens = vec![runtime.mask_token_id; runtime.block_size];
        block_tokens[0] = current_token;

        let noise_embedding = embed_tokens(&weights.embed_tokens, &block_tokens);
        let draft_hidden =
            dflash_draft_forward(runtime, &noise_embedding, &target_hidden, &mut draft_state)?;
        let block_hidden = slice(
            &draft_hidden,
            &[1, 0],
            &[
                i32::try_from(runtime.block_size).unwrap_or_default(),
                i32::try_from(config.hidden_size).unwrap_or_default(),
            ],
            &[1, 1],
        );
        let draft_logits = linear(&block_hidden, &weights.lm_head);
        let drafted_suffix = sample_rows(&draft_logits, params)?;
        draft_state.trim(runtime.block_size);
        draft_state.apply_window(DRAFT_CACHE_SINK_SIZE, DRAFT_CACHE_WINDOW_SIZE);
        for (dst, src) in block_tokens.iter_mut().skip(1).zip(drafted_suffix.iter()) {
            *dst = *src;
        }

        let (verifier_norm_hidden, verifier_hidden) = qwen3_forward_with_hidden_states(
            &block_tokens,
            weights,
            config,
            &runtime.target_layer_ids,
            &mut target_state,
        )?;
        let verifier_logits = linear(&verifier_norm_hidden, &weights.lm_head);
        let posterior = sample_rows(&verifier_logits, params)?;
        let matched = block_tokens
            .iter()
            .skip(1)
            .zip(posterior.iter())
            .take(runtime.block_size.saturating_sub(1))
            .take_while(|(draft, target)| draft == target)
            .count();
        let accepted_inputs = matched + 1;
        let posterior_token = *posterior
            .get(matched)
            .ok_or_else(|| anyhow!("DFlash verifier produced too few tokens"))?;
        if accepted_inputs < runtime.block_size {
            target_state.trim(runtime.block_size - accepted_inputs);
        }

        acceptance_lengths.push(accepted_inputs);
        target_hidden = slice(
            &verifier_hidden,
            &[0, 0],
            &[
                i32::try_from(accepted_inputs).unwrap_or_default(),
                i32::try_from(runtime.target_layer_ids.len() * config.hidden_size)
                    .unwrap_or_default(),
            ],
            &[1, 1],
        );

        let mut accepted_finish_reason = None;
        for token in block_tokens
            .iter()
            .skip(1)
            .take(accepted_inputs.saturating_sub(1))
        {
            generated.push(*token);
            on_token(*token)?;
            if is_stop_token(config, params, *token) {
                log::info!(
                    "Metal DFlash: accepted {:?} (avg {:.2}) before stop",
                    acceptance_lengths,
                    average_acceptance(&acceptance_lengths)
                );
                accepted_finish_reason = Some("stop");
                break;
            }
            if generated.len() >= max_new_tokens {
                log::info!(
                    "Metal DFlash: accepted {:?} (avg {:.2}) before length stop",
                    acceptance_lengths,
                    average_acceptance(&acceptance_lengths)
                );
                accepted_finish_reason = Some("length");
                break;
            }
        }
        if let Some(reason) = accepted_finish_reason {
            break reason;
        }

        generated.push(posterior_token);
        on_token(posterior_token)?;
        current_token = posterior_token;
        if is_stop_token(config, params, posterior_token) {
            log::info!(
                "Metal DFlash: accepted {:?} (avg {:.2})",
                acceptance_lengths,
                average_acceptance(&acceptance_lengths)
            );
            break "stop";
        }
        if generated.len() >= max_new_tokens {
            log::info!(
                "Metal DFlash: accepted {:?} (avg {:.2})",
                acceptance_lengths,
                average_acceptance(&acceptance_lengths)
            );
            break "length";
        }
    };

    let total_time_ms = t0.elapsed().as_secs_f64() * 1000.0;
    Ok(MetalGenerateOutput {
        tokens: generated,
        finish_reason,
        ttft_ms,
        total_time_ms,
    })
}

/// Result of one DFlash speculative block (draft → verify → accept/reject).
pub(crate) struct DFlashBlockResult {
    pub accepted_tokens: Vec<u32>,
    pub updated_target_hidden: MlxArray,
    pub accepted_inputs: usize,
}

/// Run one DFlash speculative block: draft N tokens, verify against target
/// model, accept the longest matching prefix, trim rejected KV.
pub(crate) fn dflash_speculative_block(
    runtime: &MetalDflashRuntime,
    current_token: u32,
    target_hidden: &MlxArray,
    weights: &StandardMetalWeights,
    config: &MetalModelConfig,
    params: &SamplingParams,
    target_state: &mut ContiguousKvState,
    draft_state: &mut ContiguousKvState,
) -> Result<DFlashBlockResult> {
    let mut block_tokens = vec![runtime.mask_token_id; runtime.block_size];
    block_tokens[0] = current_token;
    let noise_embedding = embed_tokens(&weights.embed_tokens, &block_tokens);
    let draft_hidden = dflash_draft_forward(runtime, &noise_embedding, target_hidden, draft_state)?;
    let block_hidden = slice(
        &draft_hidden,
        &[1, 0],
        &[
            i32::try_from(runtime.block_size).unwrap_or_default(),
            i32::try_from(config.hidden_size).unwrap_or_default(),
        ],
        &[1, 1],
    );
    let draft_logits = linear(&block_hidden, &weights.lm_head);
    let drafted_suffix = sample_rows(&draft_logits, params)?;
    draft_state.trim(runtime.block_size);
    draft_state.apply_window(DRAFT_CACHE_SINK_SIZE, DRAFT_CACHE_WINDOW_SIZE);
    for (dst, src) in block_tokens.iter_mut().skip(1).zip(drafted_suffix.iter()) {
        *dst = *src;
    }
    let (verifier_norm_hidden, verifier_hidden) = qwen3_forward_with_hidden_states(
        &block_tokens,
        weights,
        config,
        &runtime.target_layer_ids,
        target_state,
    )?;
    let verifier_logits = linear(&verifier_norm_hidden, &weights.lm_head);
    let posterior = sample_rows(&verifier_logits, params)?;
    let matched = block_tokens
        .iter()
        .skip(1)
        .zip(posterior.iter())
        .take(runtime.block_size.saturating_sub(1))
        .take_while(|(draft, target)| draft == target)
        .count();
    let accepted_inputs = matched + 1;
    let posterior_token = *posterior
        .get(matched)
        .ok_or_else(|| anyhow!("DFlash verifier produced too few tokens"))?;
    if accepted_inputs < runtime.block_size {
        target_state.trim(runtime.block_size - accepted_inputs);
    }
    let updated_target_hidden = slice(
        &verifier_hidden,
        &[0, 0],
        &[
            i32::try_from(accepted_inputs).unwrap_or_default(),
            i32::try_from(runtime.target_layer_ids.len() * config.hidden_size).unwrap_or_default(),
        ],
        &[1, 1],
    );
    let mut accepted_tokens = Vec::with_capacity(accepted_inputs);
    for &token in block_tokens
        .iter()
        .skip(1)
        .take(accepted_inputs.saturating_sub(1))
    {
        accepted_tokens.push(token);
    }
    accepted_tokens.push(posterior_token);
    Ok(DFlashBlockResult {
        accepted_tokens,
        updated_target_hidden,
        accepted_inputs,
    })
}

/// Public wrapper for the scheduler path — runs the full Qwen3 forward
/// on `ContiguousKvState` and captures hidden states at target layers.
pub(crate) fn qwen3_forward_with_hidden_states_on_state(
    input_ids: &[u32],
    weights: &StandardMetalWeights,
    config: &MetalModelConfig,
    target_layer_ids: &[usize],
    state: &mut ContiguousKvState,
) -> Result<(MlxArray, MlxArray)> {
    qwen3_forward_with_hidden_states(input_ids, weights, config, target_layer_ids, state)
}

fn qwen3_forward_with_hidden_states(
    input_ids: &[u32],
    weights: &StandardMetalWeights,
    config: &MetalModelConfig,
    target_layer_ids: &[usize],
    state: &mut ContiguousKvState,
) -> Result<(MlxArray, MlxArray)> {
    let seq = i32::try_from(input_ids.len()).context("input length exceeds i32")?;
    state.ensure_capacity(state.len + seq);
    let n_heads = i32::try_from(config.num_attention_heads).unwrap_or_default();
    let n_kv_heads = i32::try_from(config.num_key_value_heads).unwrap_or_default();
    let head_dim = i32::try_from(config.head_dim).unwrap_or_default();
    let attn_scale = 1.0f32 / (head_dim as f32).sqrt();
    let rope_base = config.rope_theta as f32;
    let eps = config.rms_norm_eps as f32;
    let selected: HashSet<_> = target_layer_ids.iter().copied().collect();
    let mut selected_hidden = Vec::with_capacity(target_layer_ids.len());

    let mut x = embed_tokens(&weights.embed_tokens, input_ids);
    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        x = rust_transformer_layer(
            x,
            layer,
            layer_idx,
            &mut state.k_caches,
            &mut state.v_caches,
            seq,
            state.len,
            n_heads,
            n_kv_heads,
            head_dim,
            attn_scale,
            rope_base,
            eps,
            None,
            0,
        )?;
        if selected.contains(&layer_idx) {
            selected_hidden.push(x.clone());
        }
    }
    state.len += seq;

    ensure!(
        !selected_hidden.is_empty(),
        "DFlash target_layer_ids selected no hidden states"
    );

    let norm_hidden = rms_norm(&x, &weights.norm, eps);
    Ok((norm_hidden, concatenate_axis(&selected_hidden, 1)))
}

fn dflash_draft_forward(
    runtime: &MetalDflashRuntime,
    noise_embedding: &MlxArray,
    target_hidden: &MlxArray,
    state: &mut ContiguousKvState,
) -> Result<MlxArray> {
    if use_dflash_draft_cpp()
        && let Some(cpp_model) = runtime.draft_cpp_model.as_ref()
    {
        return dflash_draft_forward_cpp(cpp_model, noise_embedding, target_hidden, state);
    }
    dflash_draft_forward_rust(runtime, noise_embedding, target_hidden, state)
}

fn dflash_draft_forward_cpp(
    cpp_model: &DFlashDraftCppModel,
    noise_embedding: &MlxArray,
    target_hidden: &MlxArray,
    state: &mut ContiguousKvState,
) -> Result<MlxArray> {
    let context_len = *target_hidden
        .shape()
        .first()
        .ok_or_else(|| anyhow!("target_hidden must be rank-2"))?;
    let seq = *noise_embedding
        .shape()
        .first()
        .ok_or_else(|| anyhow!("noise_embedding must be rank-2"))?;
    let mut kv_flat = state.active_kv_flat();
    let hidden = cpp_model.forward(
        noise_embedding,
        target_hidden,
        state.rope_offset,
        &mut kv_flat,
    )?;
    state.replace_active_kv_flat(kv_flat)?;
    state.len += context_len + seq;
    state.rope_offset += context_len + seq;
    Ok(hidden)
}

fn dflash_draft_forward_rust(
    runtime: &MetalDflashRuntime,
    noise_embedding: &MlxArray,
    target_hidden: &MlxArray,
    state: &mut ContiguousKvState,
) -> Result<MlxArray> {
    let context_len = *target_hidden
        .shape()
        .first()
        .ok_or_else(|| anyhow!("target_hidden must be rank-2"))?;
    let seq = *noise_embedding
        .shape()
        .first()
        .ok_or_else(|| anyhow!("noise_embedding must be rank-2"))?;
    state.ensure_capacity(state.len + context_len + seq);

    let n_heads = i32::try_from(runtime.draft_config.num_attention_heads).unwrap_or_default();
    let n_kv_heads = i32::try_from(runtime.draft_config.num_key_value_heads).unwrap_or_default();
    let head_dim = i32::try_from(runtime.draft_config.head_dim).unwrap_or_default();
    let attn_scale = 1.0f32 / (head_dim as f32).sqrt();
    let rope_base = runtime.draft_config.rope_theta;
    let eps = runtime.draft_config.rms_norm_eps;

    let target_hidden = linear(target_hidden, &runtime.draft_weights.fc);
    let target_hidden = rms_norm(&target_hidden, &runtime.draft_weights.hidden_norm, eps);
    let mut hidden_states = noise_embedding.clone();

    let mask_mode = runtime.draft_attention_mask.as_str();
    for (layer_idx, layer) in runtime.draft_weights.layers.iter().enumerate() {
        hidden_states = dflash_draft_layer_forward(
            &hidden_states,
            &target_hidden,
            layer,
            layer_idx,
            state,
            n_heads,
            n_kv_heads,
            head_dim,
            attn_scale,
            rope_base,
            eps,
            mask_mode,
        );
    }

    state.len += context_len + seq;
    state.rope_offset += context_len + seq;
    Ok(rms_norm(&hidden_states, &runtime.draft_weights.norm, eps))
}

#[allow(clippy::too_many_arguments)]
fn dflash_draft_layer_forward(
    hidden_states: &MlxArray,
    target_hidden: &MlxArray,
    layer: &DFlashDraftLayerWeights,
    layer_idx: usize,
    state: &mut ContiguousKvState,
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    attn_scale: f32,
    rope_base: f32,
    eps: f32,
    mask_mode: &str,
) -> MlxArray {
    use super::mlx::{
        add, multiply, reshape, rope, scaled_dot_product_attention, silu, slice_update,
        transpose_axes,
    };

    let seq = hidden_states.shape()[0];
    let context_len = target_hidden.shape()[0];
    let residual = hidden_states.clone();
    let normed_hidden_states = rms_norm(hidden_states, &layer.input_layernorm, eps);

    let q_raw = linear(&normed_hidden_states, &layer.q_proj);
    let kv_states = concatenate_axis(&[target_hidden.clone(), normed_hidden_states], 0);
    let k_raw = linear(&kv_states, &layer.k_proj);
    let v_raw = linear(&kv_states, &layer.v_proj);

    let q = reshape(&q_raw, &[1, seq, n_heads, head_dim]);
    let q = rms_norm(&q, &layer.q_norm, eps);
    let q = transpose_axes(&q, &[0, 2, 1, 3]);
    let q = rope(
        &q,
        head_dim,
        false,
        rope_base,
        1.0,
        state.rope_offset + context_len,
    );

    let total_len = context_len + seq;
    let k = reshape(&k_raw, &[1, total_len, n_kv_heads, head_dim]);
    let k = rms_norm(&k, &layer.k_norm, eps);
    let k = transpose_axes(&k, &[0, 2, 1, 3]);
    let k = rope(&k, head_dim, false, rope_base, 1.0, state.rope_offset);

    let v = reshape(&v_raw, &[1, total_len, n_kv_heads, head_dim]);
    let v = transpose_axes(&v, &[0, 2, 1, 3]);

    let end_pos = state.len + total_len;
    state.k_caches[layer_idx] = slice_update(
        &mut state.k_caches[layer_idx],
        &k,
        &[0, 0, state.len, 0],
        &[1, n_kv_heads, end_pos, head_dim],
    );
    state.v_caches[layer_idx] = slice_update(
        &mut state.v_caches[layer_idx],
        &v,
        &[0, 0, state.len, 0],
        &[1, n_kv_heads, end_pos, head_dim],
    );

    let k_full = slice(
        &state.k_caches[layer_idx],
        &[0, 0, 0, 0],
        &[1, n_kv_heads, end_pos, head_dim],
        &[1, 1, 1, 1],
    );
    let v_full = slice(
        &state.v_caches[layer_idx],
        &[0, 0, 0, 0],
        &[1, n_kv_heads, end_pos, head_dim],
        &[1, 1, 1, 1],
    );

    // Reference dflash-mlx (draft.py:149): causal when `mask_mode == "causal"`
    // and query_len > 1. Our draft block always has query_len = seq = block_size
    // which is >1 in practice. Pass `None` for "none" so the SDPA wrapper sees
    // an empty mask string (= no mask, full bidirectional within the Q range
    // against all K positions).
    let sdpa_mask = if mask_mode == "causal" {
        Some("causal")
    } else {
        None
    };
    let attn = scaled_dot_product_attention(&q, &k_full, &v_full, attn_scale, sdpa_mask);
    let attn = transpose_axes(&attn, &[0, 2, 1, 3]);
    let attn = reshape(&attn, &[seq, n_heads * head_dim]);
    let attn = linear(&attn, &layer.o_proj);
    let hidden_states = add(&residual, &attn);

    let residual = hidden_states.clone();
    let hidden_states = rms_norm(&hidden_states, &layer.post_attention_layernorm, eps);
    let (gate_raw, up) = layer.mlp_inputs.project(&hidden_states);
    let mlp = linear(&multiply(&silu(&gate_raw), &up), &layer.down_proj);
    add(&residual, &mlp)
}

fn embed_tokens(embed_table: &MlxArray, input_ids: &[u32]) -> MlxArray {
    let ids: Vec<i32> = input_ids.iter().map(|&token| token as i32).collect();
    let indices = MlxArray::from_slice_i32(&ids, &[i32::try_from(ids.len()).unwrap_or_default()]);
    take_axis(embed_table, &indices, 0)
}

fn sample_last_token(logits: &MlxArray, params: &SamplingParams) -> Result<u32> {
    let shape = logits.shape();
    ensure!(
        shape.len() == 2,
        "expected rank-2 logits, got shape {shape:?}"
    );
    let last_row = slice(logits, &[shape[0] - 1, 0], &[shape[0], shape[1]], &[1, 1]);
    let token = gpu_sample_token(&last_row, params);
    eval(&[&token]);
    Ok(token.item_i32() as u32)
}

fn sample_rows(logits: &MlxArray, params: &SamplingParams) -> Result<Vec<u32>> {
    let shape = logits.shape();
    ensure!(
        shape.len() == 2,
        "expected rank-2 logits, got shape {shape:?}"
    );
    // Fast path: greedy (temperature ≤ ε) — single batched argmax, one eval,
    // one GPU→CPU transfer via .tolist() instead of N separate .item() calls.
    if params.temperature <= 1e-6 || params.top_k == 1 {
        let tokens = super::mlx::argmax_axis(logits, -1);
        eval(&[&tokens]);
        let flat: Vec<i32> = tokens.as_slice_i32();
        return Ok(flat.iter().map(|&t| t as u32).collect());
    }
    // Slow path: temperature sampling (per-row, unavoidable for different random draws).
    let mut row_tokens = Vec::with_capacity(shape[0] as usize);
    for row in 0..shape[0] {
        let row_logits = slice(logits, &[row, 0], &[row + 1, shape[1]], &[1, 1]);
        row_tokens.push(gpu_sample_token(&row_logits, params));
    }
    let refs: Vec<_> = row_tokens.iter().collect();
    eval(&refs);
    Ok(row_tokens
        .iter()
        .map(|token| token.item_i32() as u32)
        .collect())
}

#[derive(Clone)]
struct Qwen35GdrTape {
    innovation_tape: MlxArray,
    k: MlxArray,
    g: MlxArray,
    qkv: MlxArray,
}

struct Qwen35VerifyStateGuard {
    raw: *mut std::ffi::c_void,
}

impl Drop for Qwen35VerifyStateGuard {
    fn drop(&mut self) {
        unsafe {
            mlx_sys::qwen35_set_tape_mode(self.raw, false);
            mlx_sys::qwen35_set_capture_layers(self.raw, std::ptr::null(), 0);
        }
    }
}

fn drain_current_qwen35_gdr_tapes(
    cpp_model: &super::qwen35::CppQwen35Model,
    expected_tapes: usize,
) -> Result<Vec<Qwen35GdrTape>> {
    let tape_count = unsafe { mlx_sys::qwen35_get_tape_count(cpp_model.as_raw()) };
    ensure!(
        tape_count >= 0,
        "Qwen3.5 DFlash returned negative tape count: {tape_count}"
    );
    ensure!(
        tape_count as usize == expected_tapes,
        "Qwen3.5 DFlash tape count mismatch: expected {expected_tapes}, got {tape_count}"
    );

    if tape_count == 0 {
        return Ok(Vec::new());
    }

    let tape_count_usize = tape_count as usize;
    let mut tape_ptrs = vec![std::ptr::null_mut(); tape_count_usize];
    let mut k_ptrs = vec![std::ptr::null_mut(); tape_count_usize];
    let mut g_ptrs = vec![std::ptr::null_mut(); tape_count_usize];
    let mut qkv_ptrs = vec![std::ptr::null_mut(); tape_count_usize];
    let drained_count = unsafe {
        mlx_sys::qwen35_read_and_clear_gdr_tapes(
            cpp_model.as_raw(),
            tape_ptrs.as_mut_ptr(),
            k_ptrs.as_mut_ptr(),
            g_ptrs.as_mut_ptr(),
            qkv_ptrs.as_mut_ptr(),
            tape_count,
        )
    };
    ensure!(
        drained_count == tape_count,
        "Qwen3.5 DFlash drained tape count mismatch: expected {tape_count}, got {drained_count}"
    );

    let mut tapes = Vec::with_capacity(expected_tapes);
    for tape_idx in 0..tape_count_usize {
        let tape_ptr = tape_ptrs[tape_idx];
        let k_ptr = k_ptrs[tape_idx];
        let g_ptr = g_ptrs[tape_idx];
        let qkv_ptr = qkv_ptrs[tape_idx];
        ensure!(
            !tape_ptr.is_null() && !k_ptr.is_null() && !g_ptr.is_null() && !qkv_ptr.is_null(),
            "Qwen3.5 DFlash failed to capture tape {tape_idx}"
        );
        tapes.push(Qwen35GdrTape {
            innovation_tape: unsafe { MlxArray::from_raw(tape_ptr) },
            k: unsafe { MlxArray::from_raw(k_ptr) },
            g: unsafe { MlxArray::from_raw(g_ptr) },
            qkv: unsafe { MlxArray::from_raw(qkv_ptr) },
        });
    }

    Ok(tapes)
}

// ── Qwen3.5 DFlash speculative block ─────────────────────────────────────

/// Qwen3.5 verify: run N tokens through the target C++ model with tape mode,
/// then match against drafted tokens and rollback GDR state on partial rejection.
///
/// Returns the same `DFlashBlockResult` as the Qwen3 variant.
pub(crate) fn qwen35_dflash_speculative_block(
    runtime: &MetalDflashRuntime,
    current_token: u32,
    target_hidden: &MlxArray,
    embed_table: &MlxArray,
    lm_head: &super::weights::WeightTensor,
    target_config: &super::config::MetalModelConfig,
    cpp_model: &super::qwen35::CppQwen35Model,
    params: &crate::sampler::SamplingParams,
    // Target model state (C++ flat arrays)
    target_kv_flat: &mut [MlxArray],
    target_gdr_flat: &mut [MlxArray],
    target_cache_len: &mut i32,
    // Draft model state
    draft_state: &mut ContiguousKvState,
) -> Result<DFlashBlockResult> {
    use super::mlx::{MlxArray as Arr, eval};

    let profile = std::env::var("QWEN35_DFLASH_PROFILE").is_ok();
    let t_start = std::time::Instant::now();

    // ── 1. Draft forward (same as Qwen3 — draft model is pure transformer) ──
    let block_size_i32 =
        i32::try_from(runtime.block_size).context("Qwen3.5 DFlash block_size does not fit i32")?;
    let mut block_tokens = vec![runtime.mask_token_id; runtime.block_size];
    block_tokens[0] = current_token;
    let noise_embedding = embed_tokens(embed_table, &block_tokens);
    let draft_hidden = dflash_draft_forward(runtime, &noise_embedding, target_hidden, draft_state)?;
    let draft_block_hidden = slice(
        &draft_hidden,
        &[1, 0],
        &[
            block_size_i32,
            i32::try_from(target_config.hidden_size).unwrap_or_default(),
        ],
        &[1, 1],
    );
    let draft_logits = linear(&draft_block_hidden, lm_head);
    let drafted_suffix = sample_rows(&draft_logits, params)?;
    draft_state.trim(runtime.block_size);
    draft_state.apply_window(DRAFT_CACHE_SINK_SIZE, DRAFT_CACHE_WINDOW_SIZE);
    for (dst, src) in block_tokens.iter_mut().skip(1).zip(drafted_suffix.iter()) {
        *dst = *src;
    }
    let t_draft = t_start.elapsed();

    // ── 2. Snapshot GDR states before verify ──
    let gdr_snapshot: Vec<Arr> = target_gdr_flat.to_vec();
    let t_snapshot = t_start.elapsed();

    // ── 3. Enable tape mode + hidden capture, verify via C++ model ──
    unsafe { mlx_sys::qwen35_set_tape_mode(cpp_model.as_raw(), true) };
    let layer_ids_i32: Vec<i32> = runtime
        .target_layer_ids
        .iter()
        .map(|&id| id as i32)
        .collect();
    unsafe {
        mlx_sys::qwen35_set_capture_layers(
            cpp_model.as_raw(),
            layer_ids_i32.as_ptr(),
            layer_ids_i32.len() as i32,
        );
    };
    let _verify_state_guard = Qwen35VerifyStateGuard {
        raw: cpp_model.as_raw(),
    };

    let n_capture_layers = runtime.target_layer_ids.len();
    let expected_tape_count = target_gdr_flat.len() / 2;
    let block_tokens_i32: Vec<i32> = block_tokens.iter().map(|&tok| tok as i32).collect();
    let tokens_arr = MlxArray::from_slice_i32(&block_tokens_i32, &[block_size_i32]);
    let logits = cpp_model.step_block(
        &tokens_arr,
        block_size_i32,
        *target_cache_len,
        target_kv_flat,
        target_gdr_flat,
    )?;
    if profile {
        eval(&[&logits]);
    }
    let t_verify = t_start.elapsed();
    let block_tapes = drain_current_qwen35_gdr_tapes(cpp_model, expected_tape_count)?;
    let n_cap = unsafe { mlx_sys::qwen35_get_captured_hidden_count(cpp_model.as_raw()) };
    let mut block_hidden: Vec<MlxArray> = Vec::with_capacity(n_cap.max(0) as usize);
    for ci in 0..n_cap {
        let mut h_ptr: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let rc =
            unsafe { mlx_sys::qwen35_get_captured_hidden(cpp_model.as_raw(), ci, &raw mut h_ptr) };
        if rc == 0 && !h_ptr.is_null() {
            block_hidden.push(unsafe { MlxArray::from_raw(h_ptr) });
        }
    }
    let logits_shape = logits.shape();
    ensure!(
        logits_shape.len() == 3 && logits_shape[0] == 1 && logits_shape[1] == block_size_i32,
        "Qwen3.5 DFlash verifier logits shape mismatch: expected [1, {block_size_i32}, vocab], got {logits_shape:?}"
    );
    let vocab = *logits_shape
        .get(2)
        .ok_or_else(|| anyhow!("Qwen3.5 DFlash verifier logits missing vocab dim"))?;
    // Batched posterior sample: one argmax over [block, vocab] + one eval +
    // one GPU→CPU transfer. Avoids 16 per-row slice + item sync points.
    let logits_rows = super::mlx::reshape(&logits, &[block_size_i32, vocab]);
    let posterior_tokens = sample_rows(&logits_rows, params)?;
    let t_sample = t_start.elapsed();

    // ── 4. Token matching ──
    // Full per-position agreement (no early-stop) — feeds the diagnostic
    // window so we can see whether rejections are concentrated at the first
    // draft step or spread across the block.
    let per_pos_match: Vec<bool> = block_tokens
        .iter()
        .skip(1)
        .zip(posterior_tokens.iter())
        .take(runtime.block_size.saturating_sub(1))
        .map(|(draft, target)| draft == target)
        .collect();
    let matched = per_pos_match.iter().take_while(|hit| **hit).count();
    let accepted_inputs = matched + 1;
    let posterior_token = *posterior_tokens
        .get(matched)
        .ok_or_else(|| anyhow::anyhow!("Qwen3.5 DFlash verifier produced too few tokens"))?;
    let accepted_i32 = i32::try_from(accepted_inputs)
        .context("Qwen3.5 DFlash accepted_inputs does not fit i32")?;
    let take_prefix = |arr: &Arr| {
        let shape = arr.shape();
        let mut stop = shape.to_vec();
        if let Some(seq_dim) = stop.get_mut(1) {
            *seq_dim = accepted_i32;
        }
        let start = vec![0; shape.len()];
        let strides = vec![1; shape.len()];
        slice(arr, &start, &stop, &strides)
    };
    log::debug!(
        "qwen35_dflash: accepted={}/{} draft={:?} posterior={:?}",
        accepted_inputs,
        runtime.block_size,
        &block_tokens[1..(matched + 2).min(block_tokens.len())],
        &posterior_tokens[..(matched + 1).min(posterior_tokens.len())]
    );

    // ── 5. Rollback rejected tokens ──
    let rejected = runtime.block_size - accepted_inputs;
    if rejected > 0 {
        // 5a. KV caches: the C++ model advanced cache_pos by block_size steps.
        //     We need to "un-advance" the rejected steps. For full-attention layers,
        //     the KV entries at rejected positions are just stale data that will be
        //     overwritten. The cache_pos pointer is what matters.
        //     (KV arrays don't need physical trimming — next step writes at accepted pos.)

        // 5b. GDR states: restore snapshot + tape_replay for accepted steps.
        for (i, snapshot_arr) in gdr_snapshot.iter().enumerate() {
            // gdr_flat layout: [gdr_state_0, conv_state_0, gdr_state_1, conv_state_1, ...]
            // Even indices = gdr_state, odd indices = conv_state
            if i % 2 == 0 {
                // Restore gdr_state from snapshot
                target_gdr_flat[i] = snapshot_arr.clone();
                // BUG4 fix: also restore conv_state from snapshot before rebuild
                let conv_idx = i + 1;
                if conv_idx < target_gdr_flat.len() && conv_idx < gdr_snapshot.len() {
                    target_gdr_flat[conv_idx] = gdr_snapshot[conv_idx].clone();
                }
                // Replay accepted steps using the block tapes captured during verify.
                let tape_idx = i / 2;
                let block_tape = block_tapes.get(tape_idx).cloned().ok_or_else(|| {
                    anyhow!("Qwen3.5 DFlash missing tape for GDR layer {tape_idx} during rollback")
                })?;
                let tape_arr = take_prefix(&block_tape.innovation_tape);
                let k_arr = take_prefix(&block_tape.k);
                let g_arr = take_prefix(&block_tape.g);
                let replayed = unsafe {
                    Arr::from_raw_checked(mlx_sys::mlx_tape_replay(
                        tape_arr.as_raw(),
                        k_arr.as_raw(),
                        g_arr.as_raw(),
                        target_gdr_flat[i].as_raw(),
                        accepted_i32,
                    ))
                }?;
                target_gdr_flat[i] = replayed;

                // Conv state rebuild: take last (conv_kernel-1) accepted qkv frames.
                let conv_idx = i + 1; // conv_state follows gdr_state
                if conv_idx < target_gdr_flat.len() {
                    let conv_state = &target_gdr_flat[conv_idx];
                    let conv_kernel_minus_1 = conv_state.shape().get(1).copied().unwrap_or(3);
                    let qkv_arr = take_prefix(&block_tape.qkv);
                    let combined = concatenate_axis(&[conv_state.clone(), qkv_arr], 1);
                    let combined_len = combined.shape()[1];
                    if combined_len > conv_kernel_minus_1 {
                        let start = combined_len - conv_kernel_minus_1;
                        target_gdr_flat[conv_idx] = slice(
                            &combined,
                            &[0, start, 0],
                            &[combined.shape()[0], combined_len, combined.shape()[2]],
                            &[1, 1, 1],
                        );
                    } else {
                        target_gdr_flat[conv_idx] = combined;
                    }
                }
            }
            // Odd indices (conv_state) are handled above alongside their gdr_state
        }
    }

    // Update target cache position to accepted count
    *target_cache_len += accepted_inputs as i32;

    // ── 6. Build target_hidden from per-step captured hidden states ──
    // Each captured hidden array is [1, block_size, hidden_size].
    // Slice the accepted prefix per layer, reshape to [accepted, hidden_size],
    // then concatenate layers along hidden dim.
    let updated_target_hidden = if n_capture_layers > 0 && block_hidden.len() == n_capture_layers {
        let accepted_hidden: Vec<Arr> = block_hidden
            .iter()
            .map(|hidden| {
                let accepted_hidden = take_prefix(hidden);
                let hidden_size = accepted_hidden.shape().last().copied().unwrap_or(1);
                super::mlx::reshape(&accepted_hidden, &[accepted_i32, hidden_size])
            })
            .collect();
        concatenate_axis(&accepted_hidden, 1)
    } else {
        target_hidden.clone()
    };

    let mut accepted_tokens_out = Vec::with_capacity(accepted_inputs);
    for &tok in block_tokens
        .iter()
        .skip(1)
        .take(accepted_inputs.saturating_sub(1))
    {
        accepted_tokens_out.push(tok);
    }
    accepted_tokens_out.push(posterior_token);

    let t_rollback = t_start.elapsed();

    // Eval all modified state to materialize
    let mut to_eval: Vec<&Arr> = target_gdr_flat.iter().collect();
    to_eval.extend(target_kv_flat.iter());
    eval(&to_eval);
    let t_total = t_start.elapsed();

    if profile {
        log::debug!(
            "qwen35_dflash: accept={}/{} draft={:.1}ms snapshot={:.1}ms verify={:.1}ms sample={:.1}ms rollback={:.1}ms eval={:.1}ms total={:.1}ms",
            accepted_inputs,
            runtime.block_size,
            t_draft.as_secs_f32() * 1000.0,
            (t_snapshot - t_draft).as_secs_f32() * 1000.0,
            (t_verify - t_snapshot).as_secs_f32() * 1000.0,
            (t_sample - t_verify).as_secs_f32() * 1000.0,
            (t_rollback - t_sample).as_secs_f32() * 1000.0,
            (t_total - t_rollback).as_secs_f32() * 1000.0,
            t_total.as_secs_f32() * 1000.0,
        );
        record_qwen35_block_profile(
            runtime.block_size,
            accepted_inputs,
            t_draft,
            t_verify - t_snapshot,
            t_sample - t_verify,
            t_rollback - t_sample,
            t_total - t_rollback,
            t_total,
            &per_pos_match,
        );
    }

    Ok(DFlashBlockResult {
        accepted_tokens: accepted_tokens_out,
        updated_target_hidden,
        accepted_inputs,
    })
}

fn is_stop_token(config: &MetalModelConfig, params: &SamplingParams, token: u32) -> bool {
    (!params.ignore_eos && config.is_stop_token(token)) || params.stop_token_ids.contains(&token)
}

fn average_acceptance(lengths: &[usize]) -> f64 {
    if lengths.is_empty() {
        0.0
    } else {
        lengths.iter().sum::<usize>() as f64 / lengths.len() as f64
    }
}
