use std::{
    collections::HashSet,
    fmt::Write as _,
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
use crate::backend::is_stream_stop_matched;
use crate::{hf_hub, sampler::SamplingParams};

/// Draft KV cache sink size (attention-sink tokens kept at the start).
const DRAFT_CACHE_SINK_SIZE: i32 = 64;
/// Draft KV cache window size (recent tokens kept at the end).
const DRAFT_CACHE_WINDOW_SIZE: i32 = 1024;
const QWEN35_BLOCK_PROFILE_WINDOW_BLOCKS: usize = 50;

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

    if state.blocks >= QWEN35_BLOCK_PROFILE_WINDOW_BLOCKS {
        let mean = |v: &[Duration]| -> f64 {
            v.iter().map(Duration::as_secs_f64).sum::<f64>() / v.len() as f64 * 1000.0
        };
        let quantile = |v: &mut Vec<Duration>, q: f64| -> f64 {
            v.sort();
            let idx = ((v.len() - 1) as f64 * q) as usize;
            v[idx].as_secs_f64() * 1000.0
        };
        let mut draft_v = state.draft.clone();
        let mut verify_v = state.verify.clone();
        let mut total_v = state.total.clone();
        // mean_k = mean matched draft prefix (0..block_size-1). Effective
        // tokens produced per block = matched + 1 posterior token.
        let mean_k = state.k_total as f64 / state.blocks as f64;
        let mean_total_ms = mean(&state.total);
        let mean_tokens_per_block = mean_k + 1.0;
        let effective_tok_s = 1000.0 * mean_tokens_per_block / mean_total_ms;

        let mut hist_s = String::new();
        for (k, count) in state.k_hist.iter().enumerate() {
            if *count > 0 {
                let pct = 100.0 * *count as f64 / state.blocks as f64;
                let _ = write!(hist_s, " K{k}:{count}({pct:.0}%)");
            }
        }

        log::info!(
            "qwen35_dflash[agg {} blocks]: draft μ={:.1}ms p90={:.1}ms | verify μ={:.1}ms p90={:.1}ms | sample μ={:.1}ms | rollback μ={:.1}ms | eval μ={:.1}ms | total μ={:.1}ms p90={:.1}ms | matched K̄={:.2}/{} | tok/block={:.2} | eff={:.1} tok/s",
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
            state.block_size.saturating_sub(1),
            mean_tokens_per_block,
            effective_tok_s,
        );
        log::info!("qwen35_dflash[agg K-hist]:{hist_s}");
        // Per-position draft↔target agreement (not short-circuited at first
        // mismatch). Reads the shape of the acceptance curve: flat-low means
        // the very first draft step is off; high-then-cliff means drift accrues.
        let mut pos_s = String::new();
        for (i, hits) in state.pos_match.iter().enumerate() {
            let pct = 100.0 * *hits as f64 / state.blocks as f64;
            let _ = write!(pos_s, " p{}:{pct:.0}%", i + 1);
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
    /// Debug-only SDPA mask mode for the draft block self-attention
    /// ("none" or "causal"). Reference dflash-mlx always passes mask=None
    /// (see dflash_mlx/model.py `DFlashAttention.__call__`); "none" is the
    /// production setting. `DFLASH_DRAFT_MASK=causal` forces the Rust
    /// draft forward (the compiled C++ graph has no causal branch) so the
    /// empirical causal-vs-none gap can be reproduced on demand.
    draft_attention_mask: String,
}

/// Internal error variant for the DFlash load routine. `Fatal` is an
/// unrecoverable anyhow::Error (missing file, parse failure, FFI panic);
/// `Compat` is a user-fixable shape/config mismatch that triggers a warn +
/// fallback to the standard Metal path.
enum LoadError {
    Fatal(anyhow::Error),
    Compat(DflashCompatError),
}

/// Reasons DFlash load can be disabled gracefully rather than crashing the
/// backend. Anything the user can fix by swapping the draft model belongs
/// here; config parse errors / missing files / FFI panics stay as hard errors.
#[derive(Debug)]
pub(crate) enum DflashCompatError {
    /// Specific named field mismatch (`field`, `target_value`, `draft_value`).
    FieldMismatch {
        field: &'static str,
        target: String,
        draft: String,
        suggestion: String,
    },
    /// Target architecture family isn't supported by any DFlash draft.
    Architecture { detail: String, suggestion: String },
}

impl std::fmt::Display for DflashCompatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FieldMismatch {
                field,
                target,
                draft,
                suggestion,
            } => write!(
                f,
                "DFlash draft/target {field} mismatch (target={target}, draft={draft}). Fix: {suggestion}"
            ),
            Self::Architecture { detail, suggestion } => {
                write!(
                    f,
                    "DFlash architecture mismatch: {detail}. Fix: {suggestion}"
                )
            }
        }
    }
}

/// Pure-logic compatibility check between a target model config and a loaded
/// draft config. Returns `Err(DflashCompatError)` for user-fixable mismatches
/// (swap the draft model); returns `Ok(())` when the draft is safe to run.
pub(crate) fn check_compatibility(
    target: &MetalModelConfig,
    draft: &DFlashDraftConfig,
    draft_model_id: &str,
) -> std::result::Result<(), DflashCompatError> {
    let target_q_width = target.num_attention_heads.saturating_mul(target.head_dim);
    let draft_q_width = draft.num_attention_heads.saturating_mul(draft.head_dim);
    let target_kv_width = target.num_key_value_heads.saturating_mul(target.head_dim);
    let draft_kv_width = draft.num_key_value_heads.saturating_mul(draft.head_dim);

    if !matches!(
        target.arch,
        MetalModelArch::Qwen3 | MetalModelArch::Qwen35(_)
    ) {
        return Err(DflashCompatError::Architecture {
            detail: "target is not Qwen3 or Qwen3.5".to_string(),
            suggestion: "disable DFlash or switch to a Qwen3/Qwen3.5 target model".to_string(),
        });
    }
    if draft.target_layer_ids.is_empty() {
        return Err(DflashCompatError::Architecture {
            detail: format!("draft '{draft_model_id}' has empty target_layer_ids"),
            suggestion: "rebuild the draft with a valid dflash_config.target_layer_ids list"
                .to_string(),
        });
    }
    if let Some(&max_layer) = draft.target_layer_ids.iter().max()
        && max_layer >= target.num_hidden_layers
    {
        return Err(DflashCompatError::FieldMismatch {
            field: "target_layer_ids",
            target: format!("num_hidden_layers={}", target.num_hidden_layers),
            draft: format!("max target_layer_id={max_layer}"),
            suggestion: "use a draft whose target layer indices are within the target's layer \
                         range, or rebuild the draft against this target"
                .to_string(),
        });
    }
    if draft.hidden_size != target.hidden_size {
        return Err(DflashCompatError::FieldMismatch {
            field: "hidden_size",
            target: target.hidden_size.to_string(),
            draft: draft.hidden_size.to_string(),
            suggestion: format!(
                "pick a draft trained for hidden_size={} (e.g. the DFlash pair shipped alongside \
                 this target)",
                target.hidden_size
            ),
        });
    }
    if draft_q_width != target_q_width {
        return Err(DflashCompatError::FieldMismatch {
            field: "q_proj_width",
            target: format!(
                "{}x{}={}",
                target.num_attention_heads, target.head_dim, target_q_width
            ),
            draft: format!(
                "{}x{}={}",
                draft.num_attention_heads, draft.head_dim, draft_q_width
            ),
            suggestion: format!(
                "use a draft whose num_attention_heads*head_dim equals {}",
                target_q_width
            ),
        });
    }
    if draft_kv_width != target_kv_width {
        return Err(DflashCompatError::FieldMismatch {
            field: "kv_proj_width",
            target: format!(
                "{}x{}={}",
                target.num_key_value_heads, target.head_dim, target_kv_width
            ),
            draft: format!(
                "{}x{}={}",
                draft.num_key_value_heads, draft.head_dim, draft_kv_width
            ),
            suggestion: format!(
                "use a draft whose num_key_value_heads*head_dim equals {}",
                target_kv_width
            ),
        });
    }
    Ok(())
}

impl MetalDflashRuntime {
    /// Load the DFlash draft, validating compatibility with the target.
    ///
    /// Hard errors (missing config.json, weight load failure, FFI panic) still
    /// propagate — those mean the draft itself is broken. User-fixable
    /// mismatches (hidden_size / head count / target layer ids / unsupported
    /// target arch) return `Ok(None)` with a `log::warn!` that names the
    /// field and suggests a fix, so the server can fall back to standard
    /// Metal without crashing.
    pub(crate) fn load_or_fallback(
        options: &MetalDflashOptions,
        target_config: &MetalModelConfig,
    ) -> Result<Option<Self>> {
        match Self::load_validated(options, target_config) {
            Ok(rt) => Ok(Some(rt)),
            Err(LoadError::Compat(reason)) => {
                log::warn!(
                    "DFlash disabled: {reason}. Falling back to standard Metal path. (draft='{}')",
                    options.draft_model
                );
                Ok(None)
            }
            Err(LoadError::Fatal(err)) => Err(err),
        }
    }

    /// Private wrapper that splits recoverable compat errors from fatal ones.
    fn load_validated(
        options: &MetalDflashOptions,
        target_config: &MetalModelConfig,
    ) -> std::result::Result<Self, LoadError> {
        options.validate().map_err(LoadError::Fatal)?;

        let draft_model_dir = hf_hub::resolve_weighted_model_path(&options.draft_model)
            .with_context(|| {
                format!(
                    "failed to resolve DFlash draft model '{}'",
                    options.draft_model
                )
            })
            .map_err(LoadError::Fatal)?;
        let draft_config = DFlashDraftConfig::load(&draft_model_dir).map_err(LoadError::Fatal)?;

        if let Err(compat) = check_compatibility(target_config, &draft_config, &options.draft_model)
        {
            return Err(LoadError::Compat(compat));
        }

        let draft_weights = DFlashDraftWeights::load(&draft_model_dir, &draft_config)
            .with_context(|| {
                format!(
                    "failed to load DFlash draft weights from {}",
                    draft_model_dir.display()
                )
            })
            .map_err(LoadError::Fatal)?;
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

        // Reference dflash-mlx draft SDPA always passes mask=None
        // (dflash_mlx/model.py DFlashAttention); causal is not part of the
        // published config. `DFLASH_DRAFT_MASK=causal` exists for debug
        // only and matches the empirical K̄=3.60 vs none=4.54 gap.
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

    /// Legacy test-only wrapper. Flattens `Ok(None)` (fallback) to an error
    /// so existing ignored-tests that expected `?` to either produce a
    /// runtime or bail continue to compile.
    #[cfg(test)]
    pub(crate) fn load(
        options: &MetalDflashOptions,
        target_config: &MetalModelConfig,
    ) -> Result<Self> {
        Self::load_or_fallback(options, target_config)?
            .ok_or_else(|| anyhow!("DFlash load disabled by compatibility fallback"))
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

    /// Whether the Phase 2B batched DFlash speculative path can legitimately
    /// run. Mirrors the scalar routing predicate in `dflash_draft_forward`
    /// (see `dflash.rs` ~line 1253): the batched C++ graph assumes
    /// `DFLASH_DRAFT_CPP=1` is set AND the operator did not request
    /// `DFLASH_DRAFT_MASK=causal` (the compiled graph has no causal branch).
    /// Rows that fail this predicate MUST fall back to per-row scalar decode;
    /// silently ignoring the override would produce different numerics than
    /// the user-selected scalar path.
    pub(crate) fn batched_draft_path_eligible(&self) -> bool {
        use_dflash_draft_cpp()
            && self.draft_cpp_model.is_some()
            && self.draft_attention_mask != "causal"
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

    fn forward_batched(
        &self,
        noise_embedding: &MlxArray,
        target_hidden: &MlxArray,
        batch_size: i32,
        q_offsets: &MlxArray,
        k_offsets: &MlxArray,
        kv_caches: &[MlxArray],
        attn_mask: Option<&MlxArray>,
    ) -> Result<(MlxArray, Vec<MlxArray>)> {
        let n_kv = kv_caches.len() as i32;
        let mut kv_ptrs: Vec<*mut mlx_sys::mlx_array> =
            kv_caches.iter().map(MlxArray::as_raw).collect();
        let attn_mask_ptr = attn_mask.map_or(std::ptr::null_mut(), MlxArray::as_raw);
        let mut out_hidden: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let mut out_kv: Vec<*mut mlx_sys::mlx_array> = vec![std::ptr::null_mut(); kv_caches.len()];
        let rc = unsafe {
            mlx_sys::dflash_draft_forward_batched(
                self.0,
                noise_embedding.as_raw(),
                target_hidden.as_raw(),
                batch_size,
                q_offsets.as_raw(),
                k_offsets.as_raw(),
                kv_ptrs.as_mut_ptr(),
                n_kv,
                attn_mask_ptr,
                &raw mut out_hidden,
                out_kv.as_mut_ptr(),
            )
        };
        if rc != 0 {
            return Err(super::mlx::check_mlx_error().unwrap_err());
        }
        Ok((
            unsafe { MlxArray::from_raw(out_hidden) },
            out_kv
                .into_iter()
                .map(|ptr| unsafe { MlxArray::from_raw(ptr) })
                .collect(),
        ))
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

    /// Active prefix length (positions `[0..len)` carry real K/V; the tail
    /// up to `capacity` is zero-padded inactive space). The scalar DFlash
    /// draft forward operates on `active_kv_flat()` which slices to `[..len]`;
    /// the batched path uses this accessor to gate + slice per-row before
    /// stacking so SDPA never attends over the zero tail.
    pub(super) fn active_len(&self) -> i32 {
        self.len
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
    if let Err(err) = on_token(first_token) {
        if is_stream_stop_matched(&err) {
            let total_time_ms = t0.elapsed().as_secs_f64() * 1000.0;
            return Ok(MetalGenerateOutput {
                tokens: generated,
                finish_reason: "stop",
                ttft_ms,
                total_time_ms,
            });
        }
        return Err(err);
    }
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
            if let Err(err) = on_token(*token) {
                if is_stream_stop_matched(&err) {
                    log::info!(
                        "Metal DFlash: accepted {:?} (avg {:.2}) before stream stop",
                        acceptance_lengths,
                        average_acceptance(&acceptance_lengths)
                    );
                    accepted_finish_reason = Some("stop");
                    break;
                }
                return Err(err);
            }
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
        if let Err(err) = on_token(posterior_token) {
            if is_stream_stop_matched(&err) {
                log::info!(
                    "Metal DFlash: accepted {:?} (avg {:.2}) before stream stop",
                    acceptance_lengths,
                    average_acceptance(&acceptance_lengths)
                );
                break "stop";
            }
            return Err(err);
        }
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
    // Compiled MLX graph is built with mask=None (matches reference
    // dflash-mlx: draft SDPA always mask=None). If the operator explicitly
    // requested `causal`, route to the Rust path — the C++ graph has no
    // causal-mask branch and silently dropping the override would be a lie.
    if use_dflash_draft_cpp()
        && let Some(cpp_model) = runtime.draft_cpp_model.as_ref()
        && runtime.draft_attention_mask != "causal"
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
    let last_row = match shape {
        [rows, vocab] => slice(logits, &[rows - 1, 0], &[*rows, *vocab], &[1, 1]),
        [1, rows, vocab] => {
            let squeezed = super::mlx::reshape(logits, &[*rows, *vocab]);
            slice(&squeezed, &[rows - 1, 0], &[*rows, *vocab], &[1, 1])
        }
        _ => anyhow::bail!("expected rank-2 logits or [1, T, vocab], got shape {shape:?}"),
    };
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

/// Slice a rank-3 or rank-4 array along axis 1 to keep the first `count` entries.
/// Used to narrow per-step tape / hidden captures to the accepted-prefix window.
fn slice_prefix_axis1(arr: &MlxArray, count: i32) -> MlxArray {
    let shape = arr.shape();
    debug_assert!(
        shape.len() >= 2,
        "slice_prefix_axis1 expects rank >= 2, got {shape:?}"
    );
    let start: Vec<i32> = vec![0; shape.len()];
    let mut stop: Vec<i32> = shape.to_vec();
    stop[1] = count;
    let strides: Vec<i32> = vec![1; shape.len()];
    slice(arr, &start, &stop, &strides)
}

/// Drain the capture_layer_ids hidden states recorded by the latest C++ target
/// step / verify forward. Captured arrays are rank-3 `[B, T, hidden_size]`,
/// where `T` is `1` for scalar prefix verify and `block_size` for packed verify.
fn drain_captured_hidden(cpp_model: &super::qwen35::CppQwen35Model) -> Result<Vec<MlxArray>> {
    let n_cap = unsafe { mlx_sys::qwen35_get_captured_hidden_count(cpp_model.as_raw()) };
    ensure!(
        n_cap >= 0,
        "Qwen3.5 DFlash returned negative captured-hidden count: {n_cap}"
    );
    let mut out = Vec::with_capacity(n_cap.max(0) as usize);
    for ci in 0..n_cap {
        let mut h_ptr: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let rc =
            unsafe { mlx_sys::qwen35_get_captured_hidden(cpp_model.as_raw(), ci, &raw mut h_ptr) };
        ensure!(
            rc == 0 && !h_ptr.is_null(),
            "Qwen3.5 DFlash failed to fetch captured hidden #{ci}"
        );
        out.push(unsafe { MlxArray::from_raw(h_ptr) });
    }
    Ok(out)
}

/// Restore per-GDR-layer state to the pre-verify snapshot and replay only the
/// accepted prefix. Called on partial rejection. Mutates `gdr_flat` in place.
///
/// Layout of `gdr_flat`: `[gdr_state_0, conv_state_0, gdr_state_1, conv_state_1, …]`.
/// Each `tapes[i]` corresponds to gdr layer `i` (pair index `2i` / `2i+1` in `gdr_flat`).
#[cfg(test)]
fn qwen35_rollback_to_accepted(
    gdr_flat: &mut [MlxArray],
    gdr_snapshot: &[MlxArray],
    tapes: &[Qwen35GdrTape],
    accepted_inputs: usize,
) -> Result<()> {
    let accepted_i32 = accepted_inputs as i32;
    for (pair_idx, tape_entry) in tapes.iter().enumerate() {
        let state_idx = 2 * pair_idx;
        let conv_idx = state_idx + 1;
        ensure!(
            conv_idx < gdr_flat.len() && conv_idx < gdr_snapshot.len(),
            "Qwen3.5 DFlash gdr_flat/snapshot shorter than tape count"
        );

        gdr_flat[state_idx] = gdr_snapshot[state_idx].clone();
        gdr_flat[conv_idx] = gdr_snapshot[conv_idx].clone();

        let tape_sliced = slice_prefix_axis1(&tape_entry.innovation_tape, accepted_i32);
        let k_sliced = slice_prefix_axis1(&tape_entry.k, accepted_i32);
        let g_sliced = slice_prefix_axis1(&tape_entry.g, accepted_i32);
        let qkv_sliced = slice_prefix_axis1(&tape_entry.qkv, accepted_i32);

        let replayed = unsafe {
            MlxArray::from_raw_checked(mlx_sys::mlx_tape_replay(
                tape_sliced.as_raw(),
                k_sliced.as_raw(),
                g_sliced.as_raw(),
                gdr_flat[state_idx].as_raw(),
                accepted_i32,
            ))
        }?;
        gdr_flat[state_idx] = replayed;

        let conv_state = &gdr_flat[conv_idx];
        let conv_kernel_minus_1 = conv_state.shape().get(1).copied().unwrap_or(3);
        let combined = concatenate_axis(&[conv_state.clone(), qkv_sliced], 1);
        let combined_len = combined.shape()[1];
        gdr_flat[conv_idx] = if combined_len > conv_kernel_minus_1 {
            let start = combined_len - conv_kernel_minus_1;
            slice(
                &combined,
                &[0, start, 0],
                &[combined.shape()[0], combined_len, combined.shape()[2]],
                &[1, 1, 1],
            )
        } else {
            combined
        };
    }
    Ok(())
}

/// Varlen counterpart of `qwen35_rollback_to_accepted`: restores per-GDR-layer
/// state from `gdr_snapshot` and replays each row's accepted prefix through a
/// single batched `mlx_tape_replay_varlen` call, where `accepted_inputs[b]` is
/// the prefix length for row `b` (0 ≤ accepted_inputs[b] ≤ T_padded).
///
/// Each tape in `tapes` is assumed to be `[B, T_padded, Hv, Dv/Dk]` with
/// `T_padded == accepted_inputs.iter().max().unwrap()`. Per-row independence
/// means every row may consume a different suffix of the tape; the conv state
/// is rebuilt row-by-row (slice on axis 0, concat with the row's accepted
/// qkv prefix, trim to `conv_kernel-1`) and re-stacked.
///
/// Layer 2c.4 will call this from the packed verify path. Until then the
/// only caller is the bit-ident test below; a `dead_code` warning in release
/// is expected and will retire together with the other 2c staging functions.
fn qwen35_rollback_to_accepted_varlen(
    gdr_flat: &mut [MlxArray],
    gdr_snapshot: &[MlxArray],
    tapes: &[Qwen35GdrTape],
    accepted_inputs: &[i32],
) -> Result<()> {
    let b = accepted_inputs.len();
    ensure!(
        b > 0,
        "qwen35_rollback_to_accepted_varlen: accepted_inputs empty"
    );
    ensure!(
        accepted_inputs.iter().all(|&v| v >= 0),
        "qwen35_rollback_to_accepted_varlen: accepted_inputs must be non-negative"
    );
    let t_padded = *accepted_inputs.iter().max().unwrap();
    let b_i32 = i32::try_from(b)
        .context("qwen35_rollback_to_accepted_varlen: batch dimension does not fit i32")?;
    let steps_arr = MlxArray::from_slice_i32(accepted_inputs, &[b_i32]);

    for (pair_idx, tape_entry) in tapes.iter().enumerate() {
        let state_idx = 2 * pair_idx;
        let conv_idx = state_idx + 1;
        ensure!(
            conv_idx < gdr_flat.len() && conv_idx < gdr_snapshot.len(),
            "Qwen3.5 DFlash gdr_flat/snapshot shorter than tape count"
        );

        // Restore pre-verify state, then replay accepted prefix.
        gdr_flat[state_idx] = gdr_snapshot[state_idx].clone();
        gdr_flat[conv_idx] = gdr_snapshot[conv_idx].clone();

        if t_padded > 0 {
            // Pre-slice tapes to T_padded on axis 1 (kernel requires uniform T).
            let tape_sliced = slice_prefix_axis1(&tape_entry.innovation_tape, t_padded);
            let k_sliced = slice_prefix_axis1(&tape_entry.k, t_padded);
            let g_sliced = slice_prefix_axis1(&tape_entry.g, t_padded);

            let replayed = unsafe {
                MlxArray::from_raw_checked(mlx_sys::mlx_tape_replay_varlen(
                    tape_sliced.as_raw(),
                    k_sliced.as_raw(),
                    g_sliced.as_raw(),
                    gdr_flat[state_idx].as_raw(),
                    steps_arr.as_raw(),
                ))
            }?;
            gdr_flat[state_idx] = replayed;
        }

        // Per-row conv update: slice-to-accepted, concat with prior conv tail,
        // trim to conv_kernel-1. Rows re-stacked along axis 0.
        let conv_state = gdr_flat[conv_idx].clone();
        let conv_shape = conv_state.shape().to_vec();
        let conv_kernel_minus_1 = conv_shape.get(1).copied().unwrap_or(3);
        let qkv_tape = &tape_entry.qkv;
        let qkv_shape = qkv_tape.shape().to_vec();
        let qkv_cols = qkv_shape.get(2).copied().unwrap_or(0);

        let mut per_row: Vec<MlxArray> = Vec::with_capacity(b);
        for row in 0..b_i32 {
            let conv_row = slice(
                &conv_state,
                &[row, 0, 0],
                &[row + 1, conv_kernel_minus_1, qkv_cols],
                &[1, 1, 1],
            );
            let accepted = accepted_inputs[row as usize];
            let qkv_row = slice(
                qkv_tape,
                &[row, 0, 0],
                &[row + 1, accepted, qkv_cols],
                &[1, 1, 1],
            );
            let combined = concatenate_axis(&[conv_row, qkv_row], 1);
            let combined_len = combined.shape()[1];
            let trimmed = if combined_len > conv_kernel_minus_1 {
                let start = combined_len - conv_kernel_minus_1;
                slice(
                    &combined,
                    &[0, start, 0],
                    &[1, combined_len, qkv_cols],
                    &[1, 1, 1],
                )
            } else {
                combined
            };
            per_row.push(trimmed);
        }
        gdr_flat[conv_idx] = concatenate_axis(&per_row, 0);
    }
    Ok(())
}

/// Build per-row `updated_target_hidden` tensors expected by the scheduler
/// from the per-capture-layer hidden states emitted by a single verify forward.
///
/// Each `captured_hiddens[li]` has shape `[B, block_size, hidden_size]`. For each
/// row `b` we slice to `[1, accepted_inputs[b], hidden_size]`, reshape to
/// `[accepted_inputs[b], hidden_size]`, and concatenate all capture layers along
/// the hidden dimension (axis 1) to produce one
/// `[accepted_inputs[b], n_capture_layers * hidden_size]` tensor per row.
///
/// Falls back to `B` clones of `fallback` if the capture count does not match
/// the expected layer count.
///
/// Single-row (B=1) callers pass `accepted_inputs = &[k as i32]` and take
/// `.into_iter().next().unwrap()` from the returned vector.
fn qwen35_build_updated_target_hidden(
    captured_hiddens: &[MlxArray],
    n_capture_layers: usize,
    accepted_inputs: &[i32],
    fallbacks: &[MlxArray],
) -> Vec<MlxArray> {
    let batch = accepted_inputs.len();
    debug_assert!(
        fallbacks.len() >= batch,
        "qwen35_build_updated_target_hidden: need {} fallbacks, got {}",
        batch,
        fallbacks.len()
    );
    if captured_hiddens.len() != n_capture_layers || n_capture_layers == 0 {
        return (0..batch).map(|b| fallbacks[b].clone()).collect();
    }
    let mut out = Vec::with_capacity(batch);
    for (b, &accepted) in accepted_inputs.iter().enumerate() {
        let b_i32 = b as i32;
        let per_layer: Vec<MlxArray> = captured_hiddens
            .iter()
            .map(|h| {
                let shape = h.shape();
                debug_assert!(
                    shape.len() == 3,
                    "captured hidden expected rank-3, got {shape:?}"
                );
                let block = shape.get(1).copied().unwrap_or(1);
                let hdim = *shape.last().unwrap_or(&1);
                // Slice axis 0 to row `b`, then axis 1 to `accepted`.
                let row = slice(h, &[b_i32, 0, 0], &[b_i32 + 1, block, hdim], &[1, 1, 1]);
                let row = slice(&row, &[0, 0, 0], &[1, accepted, hdim], &[1, 1, 1]);
                super::mlx::reshape(&row, &[accepted, hdim])
            })
            .collect();
        out.push(concatenate_axis(&per_layer, 1));
    }
    out
}

// ── Qwen3.5 DFlash speculative block ─────────────────────────────────────

/// Qwen3.5 single-row verify: run target decode steps until the first
/// mismatch-inclusive position, accepting every executed step and reusing the
/// captured hidden states to seed the next draft block.
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

    let n_capture_layers = runtime.target_layer_ids.len();
    let mut matched = 0usize;
    let mut accepted_tokens_out = Vec::with_capacity(runtime.block_size);
    let mut captured_per_layer: Vec<Vec<Arr>> = vec![Vec::new(); n_capture_layers];
    let mut per_pos_match = vec![false; runtime.block_size.saturating_sub(1)];
    let t_snapshot = t_start.elapsed();

    // ── 2. Prefix verify via scalar decode steps ──
    //
    // On Apple/MLX the stock M=16 verify path is still expensive. Most Qwen3.6
    // blocks on this workload reject at the second draft position, so paying a
    // full block verify wastes the target pass. Verify only until the first
    // mismatch (inclusive): every executed step is accepted, so we avoid both
    // over-verification and GDR rollback.
    super::qwen35::with_qwen35_capture_layers(
        cpp_model.as_raw(),
        &runtime.target_layer_ids,
        || {
            for step_idx in 0..runtime.block_size {
                let token = MlxArray::from_slice_i32(&[block_tokens[step_idx] as i32], &[1]);
                let logits =
                    cpp_model.step(&token, *target_cache_len, target_kv_flat, target_gdr_flat)?;
                let posterior = sample_last_token(&logits, params)?;
                let captured = drain_captured_hidden(cpp_model)?;
                ensure!(
                    captured.len() == n_capture_layers || n_capture_layers == 0,
                    "Qwen3.5 DFlash captured hidden count mismatch: expected {n_capture_layers}, got {}",
                    captured.len()
                );
                for (layer_idx, hidden) in captured.into_iter().enumerate() {
                    captured_per_layer[layer_idx].push(hidden);
                }

                *target_cache_len += 1;
                accepted_tokens_out.push(posterior);

                let Some(&draft_token) = block_tokens.get(step_idx + 1) else {
                    break;
                };
                let is_match = posterior == draft_token;
                per_pos_match[step_idx] = is_match;
                if is_match {
                    matched += 1;
                    continue;
                }
                break;
            }
            Ok(())
        },
    )?;
    let t_verify = t_start.elapsed();
    let t_sample = t_verify;

    let accepted_inputs = accepted_tokens_out.len();
    ensure!(
        accepted_inputs > 0,
        "Qwen3.5 DFlash prefix verify produced no accepted tokens"
    );

    log::debug!(
        "qwen35_dflash: accepted={}/{} draft={:?} posterior={:?}",
        accepted_inputs,
        runtime.block_size,
        &block_tokens[1..(matched + 2).min(block_tokens.len())],
        &accepted_tokens_out[..accepted_inputs.min(accepted_tokens_out.len())]
    );

    // ── 3. Build updated_target_hidden from accepted per-step captures ──
    let captured_hiddens: Vec<Arr> = captured_per_layer
        .into_iter()
        .map(|mut per_layer| {
            ensure!(
                !per_layer.is_empty(),
                "Qwen3.5 DFlash missing captured hidden for an accepted layer"
            );
            Ok(if per_layer.len() == 1 {
                per_layer.pop().expect("checked non-empty")
            } else {
                concatenate_axis(&per_layer, 1)
            })
        })
        .collect::<Result<_>>()?;
    let updated_target_hidden = qwen35_build_updated_target_hidden(
        &captured_hiddens,
        n_capture_layers,
        &[accepted_inputs as i32],
        std::slice::from_ref(target_hidden),
    )
    .into_iter()
    .next()
    .ok_or_else(|| anyhow!("qwen35_build_updated_target_hidden returned empty Vec"))?;

    let t_rollback = t_start.elapsed();

    // Eval all modified state to materialize
    let mut to_eval: Vec<&Arr> = target_gdr_flat.iter().collect();
    to_eval.extend(target_kv_flat.iter());
    eval(&to_eval);
    let t_total = t_start.elapsed();

    if profile {
        let snapshot_ms = t_snapshot.saturating_sub(t_draft).as_secs_f32() * 1000.0;
        let verify_ms = t_verify.saturating_sub(t_snapshot).as_secs_f32() * 1000.0;
        let sample_ms = t_sample.saturating_sub(t_verify).as_secs_f32() * 1000.0;
        let rollback_ms = t_rollback.saturating_sub(t_sample).as_secs_f32() * 1000.0;
        let eval_ms = t_total.saturating_sub(t_rollback).as_secs_f32() * 1000.0;

        log::debug!(
            "qwen35_dflash: accept={}/{} draft={:.1}ms snapshot={:.1}ms verify={:.1}ms sample={:.1}ms rollback={:.1}ms eval={:.1}ms total={:.1}ms",
            accepted_inputs,
            runtime.block_size,
            t_draft.as_secs_f32() * 1000.0,
            snapshot_ms,
            verify_ms,
            sample_ms,
            rollback_ms,
            eval_ms,
            t_total.as_secs_f32() * 1000.0,
        );
        // `matched` = number of draft positions that agreed with the
        // posterior (0..block_size-1). `accepted_inputs = matched + 1`
        // includes the mandatory posterior token. The aggregate K-histogram
        // tracks matched so K=0 buckets are reachable and K̄ is directly
        // comparable to the reference's "Accepted Length − 1" metric.
        record_qwen35_block_profile(
            runtime.block_size,
            matched,
            t_draft,
            t_verify.saturating_sub(t_snapshot),
            t_sample.saturating_sub(t_verify),
            t_rollback.saturating_sub(t_sample),
            t_total.saturating_sub(t_rollback),
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

// ── Qwen3.5 DFlash speculative block — batched ──────────────────────────
//
// Bit-identical analogue of `qwen35_dflash_speculative_block` for `B` rows
// in a single MLX subgraph. Prod caller lives in `request_state.rs`
// (`try_decode_qwen35_dflash_speculative_batch`), routed from the scheduler
// runtime's `execute_qwen35_dflash_packed_batch` when ≥2 DFlash-enabled
// rows are open in the same tick.
//
// API contract:
// - `target_hidden_per_row[b]` — rank-2 `[ctx_b, n_capture_layers * hidden]`.
//   All rows MUST share `ctx_b == ctx`, otherwise stacking on axis 0 fails;
//   Phase 2 will pad / equalize at the scheduler boundary.
// - `packed_target_kv_flat[l]` — `[B, n_kv_heads, kv_cap, head_dim]` already
//   sized for `kv_cap >= batch_cache_len + block_size`. Caller owns capacity.
// - `packed_target_gdr_flat[2*g]` (state) and `[2*g + 1]` (conv) — `[B, ...]`.
// - `target_cache_lens[b]` — physical write cursor per row pre-verify;
//   advanced in place by `accepted_inputs[b]`.
// - `left_padding[b]` — `batch_cache_len - target_cache_lens[b]`. Must be
//   in `0..=batch_cache_len`. Used for the additive verify mask.
// - `batch_cache_len` — shared cursor; equals `max(target_cache_lens)`.
// - `draft_states[b]` — per-row `ContiguousKvState`. We stack their
//   active KV slices, run `forward_batched`, then unstack back per row.
//   Each row's `len` and `rope_offset` advance + `trim` + `apply_window`
//   happen in scalar form (cheap for B ≤ 16).
#[allow(clippy::too_many_arguments)]
pub(super) fn qwen35_dflash_speculative_block_batched(
    runtime: &MetalDflashRuntime,
    embed_table: &MlxArray,
    lm_head: &super::weights::WeightTensor,
    target_config: &super::config::MetalModelConfig,
    cpp_model: &super::qwen35::CppQwen35Model,
    params_per_row: &[SamplingParams],
    current_tokens: &[u32],
    target_hidden_per_row: &[MlxArray],
    packed_target_kv_flat: &mut [MlxArray],
    packed_target_gdr_flat: &mut [MlxArray],
    target_cache_lens: &mut [i32],
    left_padding: &[i32],
    batch_cache_len: i32,
    draft_states: &mut [ContiguousKvState],
) -> Result<Vec<DFlashBlockResult>> {
    use super::mlx::{MlxArray as Arr, async_eval, expand_dims, reshape};

    let batch = current_tokens.len();
    ensure!(batch > 0, "Qwen3.5 DFlash batched block: empty batch");
    ensure!(
        batch == target_hidden_per_row.len()
            && batch == params_per_row.len()
            && batch == target_cache_lens.len()
            && batch == left_padding.len()
            && batch == draft_states.len(),
        "Qwen3.5 DFlash batched block: per-row slice length mismatch (B={batch})"
    );
    if let Some((first, rest)) = params_per_row.split_first() {
        ensure!(
            rest.iter().all(|params| {
                params.temperature == first.temperature
                    && params.top_k == first.top_k
                    && params.top_p == first.top_p
                    && params.min_p == first.min_p
                    && params.repetition_penalty == first.repetition_penalty
                    && params.frequency_penalty == first.frequency_penalty
                    && params.presence_penalty == first.presence_penalty
                    && params.seed == first.seed
            }),
            "Qwen3.5 DFlash batched block requires identical sampling params per row"
        );
    }
    let batch_i32 = i32::try_from(batch).context("Qwen3.5 DFlash batch does not fit i32")?;

    let block_size_i32 =
        i32::try_from(runtime.block_size).context("Qwen3.5 DFlash block_size does not fit i32")?;
    let hidden_size_i32 = i32::try_from(target_config.hidden_size)
        .context("Qwen3.5 DFlash hidden_size does not fit i32")?;

    // ── 1. Pack block tokens ─ [B, block_size] int32. ──
    let mut packed_block_tokens: Vec<i32> = Vec::with_capacity(batch * runtime.block_size);
    let mut per_row_block_tokens: Vec<Vec<u32>> = Vec::with_capacity(batch);
    for &cur in current_tokens {
        let mut row = vec![runtime.mask_token_id; runtime.block_size];
        row[0] = cur;
        for &tok in &row {
            packed_block_tokens.push(tok as i32);
        }
        per_row_block_tokens.push(row);
    }

    // ── 2. Pack noise embeddings + target hiddens. ──
    //
    // `embed_tokens` of the flat [B*block_size] token list, reshape to
    // [B, block_size, hidden]. Equivalent to per-row embed + axis-0 stack
    // but cheaper.
    let flat_tokens_u32: Vec<u32> = per_row_block_tokens.iter().flatten().copied().collect();
    let noise_flat = embed_tokens(embed_table, &flat_tokens_u32);
    let noise_packed = reshape(&noise_flat, &[batch_i32, block_size_i32, hidden_size_i32]);

    // Stack target_hidden along axis 0 — requires equal context length per row.
    let ctx_len = *target_hidden_per_row[0]
        .shape()
        .first()
        .ok_or_else(|| anyhow!("target_hidden_per_row[0] must be rank-2"))?;
    for (b, h) in target_hidden_per_row.iter().enumerate() {
        let s = h.shape();
        ensure!(
            s.len() == 2 && s[0] == ctx_len,
            "Qwen3.5 DFlash batched block: target_hidden_per_row[{b}] has shape {s:?}, expected [{ctx_len}, *]"
        );
    }
    let target_hidden_rows: Vec<Arr> = target_hidden_per_row
        .iter()
        .map(|h| expand_dims(h, 0))
        .collect();
    let target_hidden_packed = concatenate_axis(&target_hidden_rows, 0);

    // ── 3. Stack draft KV state across rows + run batched draft forward. ──
    //
    // `forward_batched` requires per-layer caches with shape
    // `[B, n_kv_heads, key_len, head_dim]`. We slice each row's cache to its
    // active prefix `[..ds.len]` (mirroring the scalar path's
    // `active_kv_flat()` — see Finding 2). All rows must share `ds.len` so
    // the per-row slices stack along axis 0 without padding; the caller
    // eligibility gate in `try_decode_qwen35_dflash_speculative_batch`
    // enforces this. Using the full physical capacity would attend over
    // zero-padded inactive slots and produce different numerics than the
    // scalar path.
    let draft_n_layers = draft_states[0].k_caches.len();
    for ds in draft_states.iter() {
        ensure!(
            ds.k_caches.len() == draft_n_layers && ds.v_caches.len() == draft_n_layers,
            "Qwen3.5 DFlash batched block: draft layer count mismatch"
        );
    }
    let draft_len = draft_states[0].len;
    for ds in draft_states.iter() {
        ensure!(
            ds.len == draft_len,
            "Qwen3.5 DFlash batched block: draft len mismatch (row len={}, expected {})",
            ds.len,
            draft_len
        );
    }
    let draft_n_kv_heads = draft_states[0].n_kv_heads;
    let draft_head_dim = draft_states[0].head_dim;
    for ds in draft_states.iter() {
        ensure!(
            ds.n_kv_heads == draft_n_kv_heads && ds.head_dim == draft_head_dim,
            "Qwen3.5 DFlash batched block: draft KV head/dim mismatch"
        );
    }

    // Stack `[k0_b0, v0_b0, k0_b1, v0_b1, ...]` per layer along axis 0 →
    // `[k0_packed, v0_packed, k1_packed, ...]` with each entry `[B, n_kv, len, head_dim]`.
    let mut packed_draft_kv: Vec<Arr> = Vec::with_capacity(draft_n_layers * 2);
    let slice_active = |cache: &Arr| -> Arr {
        slice(
            cache,
            &[0, 0, 0, 0],
            &[1, draft_n_kv_heads, draft_len, draft_head_dim],
            &[1, 1, 1, 1],
        )
    };
    for layer_idx in 0..draft_n_layers {
        let k_rows: Vec<Arr> = draft_states
            .iter()
            .map(|ds| slice_active(&ds.k_caches[layer_idx]))
            .collect();
        packed_draft_kv.push(concatenate_axis(&k_rows, 0));
        let v_rows: Vec<Arr> = draft_states
            .iter()
            .map(|ds| slice_active(&ds.v_caches[layer_idx]))
            .collect();
        packed_draft_kv.push(concatenate_axis(&v_rows, 0));
    }

    // Per-row q_offsets / k_offsets for varlen draft forward.
    let q_offsets_data: Vec<i32> = draft_states
        .iter()
        .map(|ds| ds.rope_offset + ctx_len)
        .collect();
    let k_offsets_data: Vec<i32> = draft_states.iter().map(|ds| ds.rope_offset).collect();
    let q_offsets = MlxArray::from_slice_i32(&q_offsets_data, &[batch_i32]);
    let k_offsets = MlxArray::from_slice_i32(&k_offsets_data, &[batch_i32]);

    let draft_cpp_model = runtime
        .draft_cpp_model
        .as_ref()
        .context("Qwen3.5 DFlash batched block requires the C++ draft model")?;

    let (draft_hidden_packed, draft_kv_out) = draft_cpp_model.forward_batched(
        &noise_packed,
        &target_hidden_packed,
        batch_i32,
        &q_offsets,
        &k_offsets,
        &packed_draft_kv,
        None, // attn_mask: None — equal-length blocks, draft uses no mask.
    )?;

    // Unstack draft KV back into per-row scalar states.
    ensure!(
        draft_kv_out.len() == draft_n_layers * 2,
        "Qwen3.5 DFlash batched draft forward returned {} kv slabs, expected {}",
        draft_kv_out.len(),
        draft_n_layers * 2
    );
    for (layer_idx, kv_pair) in draft_kv_out.chunks_exact(2).enumerate() {
        let k_packed = &kv_pair[0];
        let v_packed = &kv_pair[1];
        let k_shape = k_packed.shape().to_vec();
        let v_shape = v_packed.shape().to_vec();
        ensure!(
            k_shape.len() == 4 && v_shape.len() == 4 && k_shape[0] == batch_i32,
            "Qwen3.5 DFlash batched draft kv shape unexpected: k={k_shape:?}, v={v_shape:?}"
        );
        for b in 0..batch_i32 {
            let k_row = slice(
                k_packed,
                &[b, 0, 0, 0],
                &[b + 1, k_shape[1], k_shape[2], k_shape[3]],
                &[1, 1, 1, 1],
            );
            let v_row = slice(
                v_packed,
                &[b, 0, 0, 0],
                &[b + 1, v_shape[1], v_shape[2], v_shape[3]],
                &[1, 1, 1, 1],
            );
            draft_states[b as usize].k_caches[layer_idx] = k_row;
            draft_states[b as usize].v_caches[layer_idx] = v_row;
        }
    }
    // Sliced-input stacking (Finding 2) made the input key_len = `draft_len`,
    // so the C++ concat returns `[..draft_len + ctx_len + block_size]`. The
    // physical `capacity` field on each state must track axis-2 of the stored
    // `k_caches` (mirroring the scalar path's `replace_active_kv_flat` at
    // line 853), otherwise later `ensure_capacity` calls skip extension and
    // `trim/apply_window` see stale bounds.
    let new_draft_len = draft_len + ctx_len + block_size_i32;
    for ds in draft_states.iter_mut() {
        ds.len = new_draft_len;
        ds.rope_offset += ctx_len + block_size_i32;
        ds.capacity = new_draft_len;
    }

    // ── 4. Per-row draft sampling. ──
    //
    // Slice draft_hidden_packed `[B, block_size, hidden]` per row,
    // skip position 0 (which sees current_token), keep positions
    // `1..block_size` → block_size-1 draft positions.
    let draft_suffix_len = block_size_i32 - 1;
    debug_assert!(draft_suffix_len >= 0);
    for b in 0..batch_i32 {
        let row_hidden = slice(
            &draft_hidden_packed,
            &[b, 1, 0],
            &[b + 1, block_size_i32, hidden_size_i32],
            &[1, 1, 1],
        );
        let row_hidden_2d = reshape(&row_hidden, &[draft_suffix_len, hidden_size_i32]);
        let row_logits = linear(&row_hidden_2d, lm_head);
        let drafted_suffix = sample_rows(&row_logits, &params_per_row[b as usize])?;
        let row_block = &mut per_row_block_tokens[b as usize];
        for (dst, src) in row_block.iter_mut().skip(1).zip(drafted_suffix.iter()) {
            *dst = *src;
        }
    }

    // Per-row trim + window for the draft cache (mirrors scalar block).
    for ds in draft_states.iter_mut() {
        ds.trim(runtime.block_size);
        ds.apply_window(DRAFT_CACHE_SINK_SIZE, DRAFT_CACHE_WINDOW_SIZE);
    }

    // ── 5. Snapshot packed GDR state before verify (for rollback). ──
    let gdr_snapshot: Vec<Arr> = packed_target_gdr_flat.to_vec();

    // ── 6. Enable tape mode + capture layers (single guard for the batch). ──
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

    // ── 7. Build packed verify inputs and call verify_block_batched. ──
    //
    // After the per-row token rewrites in step 4 we need to repack the
    // [B, block_size] int32 token tensor — block_tokens may now contain
    // the drafted suffix.
    packed_block_tokens.clear();
    for row in &per_row_block_tokens {
        for &tok in row {
            packed_block_tokens.push(tok as i32);
        }
    }
    let tokens_arr = MlxArray::from_slice_i32(&packed_block_tokens, &[batch_i32, block_size_i32]);
    let cache_pos_arr = MlxArray::from_slice_i32(target_cache_lens, &[batch_i32]);
    let rope_offsets = MlxArray::from_slice_i32(target_cache_lens, &[batch_i32]);
    let attn_mask =
        super::mlx::build_varlen_verify_mask(left_padding, block_size_i32, batch_cache_len);

    let n_capture_layers = runtime.target_layer_ids.len();
    let expected_tape_count = packed_target_gdr_flat.len() / 2;

    let posterior_tokens = cpp_model.verify_block_batched_sampled(
        &tokens_arr,
        batch_i32,
        block_size_i32,
        &cache_pos_arr,
        packed_target_kv_flat,
        packed_target_gdr_flat,
        Some(&attn_mask),
        &rope_offsets,
        &params_per_row[0],
    )?;

    // ── 8. Drain tapes + captured hidden, per-row matching. ──
    let tapes = drain_current_qwen35_gdr_tapes(cpp_model, expected_tape_count)?;
    let captured_hiddens = drain_captured_hidden(cpp_model)?;

    let posterior_shape = posterior_tokens.shape();
    ensure!(
        posterior_shape.len() == 2
            && posterior_shape[0] == batch_i32
            && posterior_shape[1] == block_size_i32,
        "Qwen3.5 DFlash batched sampled verify tokens unexpected shape {posterior_shape:?}"
    );
    eval(&[&posterior_tokens]);
    let posterior_tokens = posterior_tokens.as_slice_i32();

    let mut accepted_inputs: Vec<i32> = Vec::with_capacity(batch);
    let mut posterior_token_per_row: Vec<u32> = Vec::with_capacity(batch);
    for b in 0..batch_i32 {
        let row_offset = (b as usize) * runtime.block_size;
        let posterior: Vec<u32> = posterior_tokens[row_offset..row_offset + runtime.block_size]
            .iter()
            .map(|&tok| tok as u32)
            .collect();

        let row_block = &per_row_block_tokens[b as usize];
        let matched = row_block
            .iter()
            .skip(1)
            .zip(posterior.iter())
            .take(runtime.block_size.saturating_sub(1))
            .take_while(|(draft, target)| draft == target)
            .count();
        let accepted = (matched + 1) as i32;
        let posterior_token = *posterior.get(matched).ok_or_else(|| {
            anyhow!("Qwen3.5 DFlash batched verifier produced too few tokens for row {b}")
        })?;
        accepted_inputs.push(accepted);
        posterior_token_per_row.push(posterior_token);
    }

    // ── 9. Rollback packed GDR state on partial accept. ──
    if accepted_inputs.iter().any(|&k| k < block_size_i32) {
        qwen35_rollback_to_accepted_varlen(
            packed_target_gdr_flat,
            &gdr_snapshot,
            &tapes,
            &accepted_inputs,
        )?;
    }

    // ── 10. Per-row updated_target_hidden via the generalized helper. ──
    // Pass per-row fallbacks so that on a capture-count mismatch each row
    // preserves its own pre-verify hidden state (P2 codex fix — row 0's
    // fallback must not propagate to rows 1..N).
    let updated_per_row = qwen35_build_updated_target_hidden(
        &captured_hiddens,
        n_capture_layers,
        &accepted_inputs,
        target_hidden_per_row,
    );
    ensure!(
        updated_per_row.len() == batch,
        "Qwen3.5 DFlash batched: updated_target_hidden returned {} rows, expected {}",
        updated_per_row.len(),
        batch
    );

    // ── 11. Per-row cache_len advance. ──
    for (b, &k) in accepted_inputs.iter().enumerate() {
        target_cache_lens[b] += k;
    }

    // ── 12. Queue packed state for async materialization. ──
    //
    // Previously this was a blocking `eval` — a full CPU↔GPU fence right
    // before returning. Defer via `async_eval` so the GPU can continue
    // draining the queued work while the caller builds the *next* DFlash
    // block's graph (mirrors the scalar Qwen3.5 step-driver double-buffer
    // landed in commit f6be5f6: queue step N+1 before materializing step N).
    //
    // Correctness: callers of this function only consume the returned
    // `updated_target_hidden` as an opaque MlxArray handle (stashed back
    // into `dflash.target_hidden` at request_state.rs:1991 and fed as an
    // input to the next block at request_state.rs:2702 / 3435). The packed
    // KV/GDR arrays are sliced per-row (lazy view ops) and re-stashed. No
    // caller issues a host-side read (`.item()` / `.as_slice()`) on these
    // handles before the next DFlash block's own sync, so deferring is
    // safe. The next block's prefix-match scan in `sample_rows()`
    // (`dflash.rs:1505` → `.item_i32()`) or any subsequent `eval` call
    // will flush this queue.
    {
        let mut to_eval: Vec<&Arr> = Vec::new();
        to_eval.extend(packed_target_kv_flat.iter());
        to_eval.extend(packed_target_gdr_flat.iter());
        to_eval.extend(updated_per_row.iter());
        async_eval(&to_eval);
    }

    // ── Assemble per-row DFlashBlockResult. ──
    let mut out = Vec::with_capacity(batch);
    for (b, updated_target_hidden) in updated_per_row.into_iter().enumerate() {
        let accepted = accepted_inputs[b] as usize;
        let row_block = &per_row_block_tokens[b];
        let mut accepted_tokens = Vec::with_capacity(accepted);
        for &tok in row_block.iter().skip(1).take(accepted.saturating_sub(1)) {
            accepted_tokens.push(tok);
        }
        accepted_tokens.push(posterior_token_per_row[b]);
        out.push(DFlashBlockResult {
            accepted_tokens,
            updated_target_hidden,
            accepted_inputs: accepted,
        });
    }
    Ok(out)
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

#[cfg(test)]
mod tests {
    use std::{env, path::PathBuf};

    use super::*;
    use crate::backend::metal::{
        config::{MetalModelArch, load_metal_config},
        mlx::{Dtype, as_dtype, eval, expand_dims, reshape, slice},
    };
    use crate::test_support::metal_test_guard;

    fn dflash_fc_input_dim(weight: &WeightTensor) -> i32 {
        match weight {
            WeightTensor::Dense(w) => w.shape()[0],
            WeightTensor::Quantized {
                scales, group_size, ..
            } => scales.shape()[1] * *group_size,
        }
    }

    #[test]
    fn draft_forward_batched_matches_forward_for_b1() -> Result<()> {
        let Some(model_path) = env::var_os("QWEN35_MODEL_PATH").map(PathBuf::from) else {
            eprintln!(
                "QWEN35_MODEL_PATH unset; skipping DFlash draft batched B=1 equivalence test"
            );
            return Ok(());
        };
        eprintln!("draft_forward_batched_matches_forward_for_b1 env_ready");
        let _guard = metal_test_guard();
        eprintln!("draft_forward_batched_matches_forward_for_b1 guard_ready");

        let target_config = load_metal_config(&model_path)?;
        let MetalModelArch::Qwen35(_) = &target_config.arch else {
            anyhow::bail!("QWEN35_MODEL_PATH must point to a Qwen3.5 model");
        };
        eprintln!("draft_forward_batched_matches_forward_for_b1 config_ready");

        let runtime = match MetalDflashRuntime::load(
            &MetalDflashOptions {
                draft_model: "z-lab/Qwen3.5-4B-DFlash".to_string(),
                speculative_tokens: None,
            },
            &target_config,
        ) {
            Ok(rt) => rt,
            Err(err) => {
                eprintln!(
                    "DFlash draft model unavailable ({err:#}); skipping draft_forward_batched_matches_forward_for_b1. Set `QWEN35_DFLASH_DRAFT_PATH` to a local checkpoint to enable."
                );
                return Ok(());
            }
        };
        eprintln!("draft_forward_batched_matches_forward_for_b1 runtime_loaded");
        let cpp_model = runtime
            .draft_cpp_model
            .as_ref()
            .context("DFlash draft C++ model unavailable")?;

        let seq = 3_i32;
        let context_len = 2_i32;
        let hidden_size = runtime.draft_config.hidden_size as i32;
        let target_hidden_width = dflash_fc_input_dim(&runtime.draft_weights.fc);

        let noise_data: Vec<f32> = (0..(seq * hidden_size))
            .map(|idx| idx as f32 / 128.0)
            .collect();
        let target_data: Vec<f32> = (0..(context_len * target_hidden_width))
            .map(|idx| (idx as f32 - 17.0) / 256.0)
            .collect();
        let noise_embedding = MlxArray::from_slice_f32(&noise_data, &[seq, hidden_size]);
        let target_hidden =
            MlxArray::from_slice_f32(&target_data, &[context_len, target_hidden_width]);

        let initial_tokens = usize::try_from(context_len + seq + 4).unwrap_or_default();
        let mut scalar_state = ContiguousKvState::new(
            runtime.draft_config.num_hidden_layers,
            runtime.draft_config.num_key_value_heads as i32,
            runtime.draft_config.head_dim as i32,
            initial_tokens,
        );
        let scalar_hidden = dflash_draft_forward_cpp(
            cpp_model,
            &noise_embedding,
            &target_hidden,
            &mut scalar_state,
        )?;
        eprintln!("draft_forward_batched_matches_forward_for_b1 scalar_done");

        let noise_embedding_batched = expand_dims(&noise_embedding, 0);
        let target_hidden_batched = expand_dims(&target_hidden, 0);
        let q_offsets = MlxArray::from_slice_i32(&[context_len], &[1]);
        let k_offsets = MlxArray::from_slice_i32(&[0], &[1]);

        let batched_state = ContiguousKvState::new(
            runtime.draft_config.num_hidden_layers,
            runtime.draft_config.num_key_value_heads as i32,
            runtime.draft_config.head_dim as i32,
            initial_tokens,
        );
        let batched_kv = batched_state.active_kv_flat();
        let (batched_hidden, _) = cpp_model.forward_batched(
            &noise_embedding_batched,
            &target_hidden_batched,
            1,
            &q_offsets,
            &k_offsets,
            &batched_kv,
            None,
        )?;
        eprintln!("draft_forward_batched_matches_forward_for_b1 batched_done");

        let batched_row0 = slice(
            &batched_hidden,
            &[0, 0, 0],
            &[1, seq, hidden_size],
            &[1, 1, 1],
        );
        let batched_row0 = reshape(&batched_row0, &[seq, hidden_size]);

        let scalar_hidden_f32 = as_dtype(&scalar_hidden, Dtype::Float32);
        let batched_hidden_f32 = as_dtype(&batched_row0, Dtype::Float32);
        eval(&[&scalar_hidden_f32, &batched_hidden_f32]);

        assert_eq!(scalar_hidden_f32.shape(), batched_hidden_f32.shape());
        let mut max_abs_delta = 0.0_f32;
        for (idx, (lhs, rhs)) in scalar_hidden_f32
            .as_slice_f32()
            .iter()
            .zip(batched_hidden_f32.as_slice_f32().iter())
            .enumerate()
        {
            let delta = (lhs - rhs).abs();
            max_abs_delta = max_abs_delta.max(delta);
            assert!(
                delta < 1e-3,
                "hidden[{idx}] mismatch: {lhs} vs {rhs} (|delta|={delta})"
            );
        }
        eprintln!("draft_forward_batched_matches_forward_for_b1 max_abs_delta={max_abs_delta}");

        Ok(())
    }

    #[test]
    fn qwen35_rollback_to_accepted_varlen_matches_scalar_for_b1() -> Result<()> {
        use super::super::gdr::MetalRecurrentState;
        use super::super::mlx::zeros;
        use super::super::qwen35::load_qwen35_metal_weights;

        let Some(model_path) = env::var_os("QWEN35_MODEL_PATH").map(PathBuf::from) else {
            eprintln!(
                "QWEN35_MODEL_PATH unset; skipping DFlash rollback varlen B=1 equivalence test"
            );
            return Ok(());
        };
        eprintln!("qwen35_rollback_to_accepted_varlen_matches_scalar_for_b1 env_ready");
        let _guard = metal_test_guard();

        let config = load_metal_config(&model_path)?;
        let MetalModelArch::Qwen35(arch) = &config.arch else {
            anyhow::bail!("QWEN35_MODEL_PATH must point to a Qwen3.5 model");
        };
        let weights = load_qwen35_metal_weights(&model_path, &config)?;
        let cpp_model = weights
            .cpp_model
            .as_ref()
            .context("Qwen3.5 compiled C++ model unavailable")?;

        let prompt_tokens = [1_i32, 2, 3, 4];
        let block_tokens = [5_i32, 6, 7, 8];
        let prompt_len = prompt_tokens.len() as i32;
        let block_size = block_tokens.len() as i32;
        let kv_capacity = prompt_len + block_size + 4;
        let cache_shape = [
            1_i32,
            config.num_key_value_heads as i32,
            kv_capacity,
            config.head_dim as i32,
        ];

        let num_full_layers = arch.num_full_attention_layers();
        let mut kv_flat: Vec<MlxArray> = (0..num_full_layers)
            .flat_map(|_| {
                [
                    zeros(&cache_shape, Dtype::Bfloat16),
                    zeros(&cache_shape, Dtype::Bfloat16),
                ]
            })
            .collect();
        let recurrent = MetalRecurrentState::new(arch.num_linear_attention_layers(), &arch.linear);
        let mut gdr_flat: Vec<MlxArray> = recurrent
            .states
            .iter()
            .zip(recurrent.conv_states.iter())
            .flat_map(|(state, conv)| [state.clone(), conv.clone()])
            .collect();

        // Prefill to warm KV and GDR state.
        let prompt_arr = MlxArray::from_slice_i32(&prompt_tokens, &[prompt_len]);
        let prompt_logits =
            cpp_model.prefill(&prompt_arr, prompt_len, 0, &mut kv_flat, &mut gdr_flat)?;
        let mut prompt_refs: Vec<&MlxArray> =
            Vec::with_capacity(1 + kv_flat.len() + gdr_flat.len());
        prompt_refs.push(&prompt_logits);
        prompt_refs.extend(kv_flat.iter());
        prompt_refs.extend(gdr_flat.iter());
        eval(&prompt_refs);
        eprintln!("qwen35_rollback_to_accepted_varlen_matches_scalar_for_b1 prefill_done");

        // Snapshot GDR state *before* verify — rollback paths will restore
        // from this and replay the accepted prefix through the captured tapes.
        let gdr_snapshot: Vec<MlxArray> = gdr_flat.to_vec();

        // Enable tape mode + run verify_block_batched with B=1.
        unsafe { mlx_sys::qwen35_set_tape_mode(cpp_model.as_raw(), true) };
        let _tape_guard = Qwen35VerifyStateGuard {
            raw: cpp_model.as_raw(),
        };

        let cache_pos = prompt_len;
        let mut post_verify_kv = kv_flat.clone();
        let mut post_verify_gdr = gdr_flat.clone();
        let batched_tokens = MlxArray::from_slice_i32(&block_tokens, &[1, block_size]);
        let cache_pos_arr = MlxArray::from_slice_i32(&[cache_pos], &[1]);
        let rope_offsets = MlxArray::from_slice_i32(&[cache_pos], &[1]);
        let verify_logits = cpp_model.verify_block_batched(
            &batched_tokens,
            1,
            block_size,
            &cache_pos_arr,
            &mut post_verify_kv,
            &mut post_verify_gdr,
            None,
            &rope_offsets,
        )?;
        eval(&[&verify_logits]);
        eprintln!("qwen35_rollback_to_accepted_varlen_matches_scalar_for_b1 verify_done");

        let expected_tape_count = gdr_snapshot.len() / 2;
        let tapes = drain_current_qwen35_gdr_tapes(cpp_model, expected_tape_count)?;
        ensure!(
            !tapes.is_empty(),
            "expected at least one GDR tape from verify_block_batched"
        );
        eprintln!(
            "qwen35_rollback_to_accepted_varlen_matches_scalar_for_b1 tapes={} T={:?}",
            tapes.len(),
            tapes[0].innovation_tape.shape()
        );

        // Vary the accepted prefix length to exercise several T_padded values,
        // including full acceptance and full rejection.
        let ks: [i32; 4] = [0, 1, 2, block_size];
        let mut max_abs_delta_overall = 0.0_f32;
        for &k in &ks {
            let mut scalar_gdr = gdr_snapshot.clone();
            let mut varlen_gdr = gdr_snapshot.clone();

            qwen35_rollback_to_accepted(&mut scalar_gdr, &gdr_snapshot, &tapes, k as usize)?;
            qwen35_rollback_to_accepted_varlen(&mut varlen_gdr, &gdr_snapshot, &tapes, &[k])?;

            let mut eval_refs: Vec<&MlxArray> =
                Vec::with_capacity(scalar_gdr.len() + varlen_gdr.len());
            eval_refs.extend(scalar_gdr.iter());
            eval_refs.extend(varlen_gdr.iter());
            eval(&eval_refs);

            assert_eq!(
                scalar_gdr.len(),
                varlen_gdr.len(),
                "rollback output count mismatch at k={k}"
            );
            for (idx, (lhs, rhs)) in scalar_gdr.iter().zip(varlen_gdr.iter()).enumerate() {
                assert_eq!(
                    lhs.shape(),
                    rhs.shape(),
                    "rollback[{idx}] shape mismatch at k={k}: {:?} vs {:?}",
                    lhs.shape(),
                    rhs.shape()
                );
                let lhs_f32 = as_dtype(lhs, Dtype::Float32);
                let rhs_f32 = as_dtype(rhs, Dtype::Float32);
                eval(&[&lhs_f32, &rhs_f32]);
                let mut max_abs_delta = 0.0_f32;
                for (lv, rv) in lhs_f32
                    .as_slice_f32()
                    .iter()
                    .zip(rhs_f32.as_slice_f32().iter())
                {
                    let delta = (lv - rv).abs();
                    max_abs_delta = max_abs_delta.max(delta);
                }
                assert!(
                    max_abs_delta < 1e-3,
                    "rollback[{idx}] mismatch at k={k}: max_abs_delta={max_abs_delta}"
                );
                max_abs_delta_overall = max_abs_delta_overall.max(max_abs_delta);
            }
            eprintln!(
                "qwen35_rollback_to_accepted_varlen_matches_scalar_for_b1 k={k} max_abs_delta={max_abs_delta_overall}"
            );
        }
        eprintln!(
            "qwen35_rollback_to_accepted_varlen_matches_scalar_for_b1 overall_max_abs_delta={max_abs_delta_overall}"
        );

        Ok(())
    }

    /// Phase-1 bit-ident test for `qwen35_dflash_speculative_block_batched`.
    ///
    /// Build two synthetic DFlash rows with distinct prompts. Run two
    /// sequential `qwen35_dflash_speculative_block` calls to capture the
    /// scalar baseline (per-row `accepted_inputs`, `accepted_tokens`,
    /// `updated_target_hidden`). Then re-run the same inputs through
    /// `qwen35_dflash_speculative_block_batched` (B=2, equal cache_lens →
    /// `left_padding = [0, 0]`, `batch_cache_len = prompt_len`) and assert
    /// per-row identity.
    #[test]
    fn dflash_qwen35_verify_batched_matches_two_single_row_runs() -> Result<()> {
        use super::super::gdr::MetalRecurrentState;
        use super::super::mlx::zeros;
        use super::super::qwen35::load_qwen35_metal_weights;

        let Some(model_path) = env::var_os("QWEN35_MODEL_PATH").map(PathBuf::from) else {
            eprintln!(
                "QWEN35_MODEL_PATH unset; skipping DFlash batched verify B=2 equivalence test"
            );
            return Ok(());
        };
        eprintln!("dflash_qwen35_verify_batched_matches_two_single_row_runs env_ready");
        let _guard = metal_test_guard();

        let target_config = load_metal_config(&model_path)?;
        let MetalModelArch::Qwen35(arch) = &target_config.arch else {
            anyhow::bail!("QWEN35_MODEL_PATH must point to a Qwen3.5 model");
        };
        let weights = load_qwen35_metal_weights(&model_path, &target_config)?;
        let cpp_model = weights
            .cpp_model
            .as_ref()
            .context("Qwen3.5 compiled C++ model unavailable")?;

        let runtime = match MetalDflashRuntime::load(
            &MetalDflashOptions {
                draft_model: "z-lab/Qwen3.5-4B-DFlash".to_string(),
                speculative_tokens: None,
            },
            &target_config,
        ) {
            Ok(rt) => rt,
            Err(err) => {
                eprintln!(
                    "DFlash draft model unavailable ({err:#}); skipping dflash_qwen35_verify_batched_matches_two_single_row_runs. Set `QWEN35_DFLASH_DRAFT_PATH` to a local checkpoint to enable."
                );
                return Ok(());
            }
        };
        eprintln!("dflash_qwen35_verify_batched_matches_two_single_row_runs runtime_loaded");

        // Two distinct prompts of equal length so both rows share the same
        // pre-verify cache_len (no left-padding plumbing is exercised at the
        // Phase-1 layer; that's Phase 2's job at the scheduler boundary).
        let row_prompts: [[u32; 4]; 2] = [[1, 2, 3, 4], [5, 6, 7, 8]];
        let row_currents: [u32; 2] = [11, 13]; // distinct first-block tokens
        let prompt_len = row_prompts[0].len() as i32;
        let block_size = runtime.block_size as i32;
        let kv_capacity = prompt_len + block_size + KV_CACHE_CHUNK;
        let cache_shape = [
            1_i32,
            target_config.num_key_value_heads as i32,
            kv_capacity,
            target_config.head_dim as i32,
        ];

        let num_full_layers = arch.num_full_attention_layers();
        let n_capture_layers = runtime.target_layer_ids.len();
        let target_hidden_width = i32::try_from(n_capture_layers * target_config.hidden_size)
            .context("target_hidden width does not fit i32")?;
        let ctx_len = 1_i32; // synthetic single-row warmup hidden

        let params = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };

        // Synthesize a deterministic, distinct target_hidden per row so the
        // draft branches actually diverge. Shape `[ctx_len, n_cap*hidden]`.
        let make_target_hidden = |row_idx: usize| -> MlxArray {
            let n = (ctx_len as usize) * (target_hidden_width as usize);
            let data: Vec<f32> = (0..n)
                .map(|i| ((row_idx as f32) + 1.0) * (i as f32) / (n as f32 * 64.0))
                .collect();
            MlxArray::from_slice_f32(&data, &[ctx_len, target_hidden_width])
        };

        // Helper: build a fresh per-row target state by running prefill on
        // the row's prompt — populates `kv_flat` + `gdr_flat` consistent with
        // the C++ verify path. Returns post-prefill state.
        let build_row_state = |prompt: &[u32]| -> Result<(Vec<MlxArray>, Vec<MlxArray>)> {
            let mut kv_flat: Vec<MlxArray> = (0..num_full_layers)
                .flat_map(|_| {
                    [
                        zeros(&cache_shape, super::super::mlx::Dtype::Bfloat16),
                        zeros(&cache_shape, super::super::mlx::Dtype::Bfloat16),
                    ]
                })
                .collect();
            let recurrent =
                MetalRecurrentState::new(arch.num_linear_attention_layers(), &arch.linear);
            let mut gdr_flat: Vec<MlxArray> = recurrent
                .states
                .iter()
                .zip(recurrent.conv_states.iter())
                .flat_map(|(state, conv)| [state.clone(), conv.clone()])
                .collect();

            let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
            let prompt_arr = MlxArray::from_slice_i32(&prompt_i32, &[prompt_len]);
            let prompt_logits =
                cpp_model.prefill(&prompt_arr, prompt_len, 0, &mut kv_flat, &mut gdr_flat)?;
            let mut refs: Vec<&MlxArray> = Vec::with_capacity(1 + kv_flat.len() + gdr_flat.len());
            refs.push(&prompt_logits);
            refs.extend(kv_flat.iter());
            refs.extend(gdr_flat.iter());
            eval(&refs);
            Ok((kv_flat, gdr_flat))
        };

        let (row0_kv, row0_gdr) = build_row_state(&row_prompts[0])?;
        let (row1_kv, row1_gdr) = build_row_state(&row_prompts[1])?;

        let target_hidden_per_row: Vec<MlxArray> = (0..2).map(make_target_hidden).collect();

        // ── Scalar baseline: run two sequential single-row blocks. ──
        eprintln!("dflash_qwen35_verify_batched_matches_two_single_row_runs scalar_start");
        let mut scalar_results: Vec<DFlashBlockResult> = Vec::with_capacity(2);
        let mut scalar_post_kv: Vec<Vec<MlxArray>> = Vec::with_capacity(2);
        let mut scalar_post_gdr: Vec<Vec<MlxArray>> = Vec::with_capacity(2);
        let mut scalar_post_cache_lens: Vec<i32> = Vec::with_capacity(2);
        for (row_idx, (kv_in, gdr_in)) in [
            (row0_kv.clone(), row0_gdr.clone()),
            (row1_kv.clone(), row1_gdr.clone()),
        ]
        .into_iter()
        .enumerate()
        {
            let mut kv_flat = kv_in;
            let mut gdr_flat = gdr_in;
            let mut cache_len = prompt_len;
            let mut draft_state = ContiguousKvState::new(
                runtime.draft_config.num_hidden_layers,
                runtime.draft_config.num_key_value_heads as i32,
                runtime.draft_config.head_dim as i32,
                64,
            );
            let block = qwen35_dflash_speculative_block(
                &runtime,
                row_currents[row_idx],
                &target_hidden_per_row[row_idx],
                &weights.embed_tokens,
                &weights.lm_head,
                &target_config,
                cpp_model,
                &params,
                &mut kv_flat,
                &mut gdr_flat,
                &mut cache_len,
                &mut draft_state,
            )?;
            // Materialize the result tensor so its values are independent of
            // any post-call mutation (defensive — should already be evaluated
            // by the function's terminal `eval`).
            eval(&[&block.updated_target_hidden]);
            scalar_results.push(block);
            scalar_post_kv.push(kv_flat);
            scalar_post_gdr.push(gdr_flat);
            scalar_post_cache_lens.push(cache_len);
        }
        eprintln!(
            "dflash_qwen35_verify_batched_matches_two_single_row_runs scalar_done accepted_inputs=[{}, {}]",
            scalar_results[0].accepted_inputs, scalar_results[1].accepted_inputs
        );

        // ── Batched run: stack pre-verify per-row state + one batched call. ──
        let n_kv = row0_kv.len();
        let n_gdr = row0_gdr.len();
        let mut packed_kv: Vec<MlxArray> = Vec::with_capacity(n_kv);
        for kv_idx in 0..n_kv {
            let stacked = vec![row0_kv[kv_idx].clone(), row1_kv[kv_idx].clone()];
            packed_kv.push(concatenate_axis(&stacked, 0));
        }
        let mut packed_gdr: Vec<MlxArray> = Vec::with_capacity(n_gdr);
        for gdr_idx in 0..n_gdr {
            let stacked = vec![row0_gdr[gdr_idx].clone(), row1_gdr[gdr_idx].clone()];
            packed_gdr.push(concatenate_axis(&stacked, 0));
        }

        let mut target_cache_lens: [i32; 2] = [prompt_len, prompt_len];
        let left_padding: [i32; 2] = [0, 0];
        let batch_cache_len = prompt_len;

        let mut draft_states: Vec<ContiguousKvState> = (0..2)
            .map(|_| {
                ContiguousKvState::new(
                    runtime.draft_config.num_hidden_layers,
                    runtime.draft_config.num_key_value_heads as i32,
                    runtime.draft_config.head_dim as i32,
                    64,
                )
            })
            .collect();

        let params_per_row = vec![params.clone(), params.clone()];
        let current_tokens = row_currents.to_vec();

        eprintln!("dflash_qwen35_verify_batched_matches_two_single_row_runs batched_start");
        let batched_results = qwen35_dflash_speculative_block_batched(
            &runtime,
            &weights.embed_tokens,
            &weights.lm_head,
            &target_config,
            cpp_model,
            &params_per_row,
            &current_tokens,
            &target_hidden_per_row,
            &mut packed_kv,
            &mut packed_gdr,
            &mut target_cache_lens,
            &left_padding,
            batch_cache_len,
            &mut draft_states,
        )?;
        eprintln!(
            "dflash_qwen35_verify_batched_matches_two_single_row_runs batched_done accepted_inputs=[{}, {}]",
            batched_results[0].accepted_inputs, batched_results[1].accepted_inputs
        );

        assert_eq!(batched_results.len(), 2, "batched returned wrong row count");

        // ── Predicates: per-row equality of accepted_inputs + accepted_tokens
        //                + updated_target_hidden (max_abs_delta == 0.0). ──
        let mut overall_max_abs_delta = 0.0_f32;
        for b in 0..2 {
            let scalar = &scalar_results[b];
            let batched = &batched_results[b];
            assert_eq!(
                batched.accepted_inputs, scalar.accepted_inputs,
                "row {b}: accepted_inputs mismatch (batched={}, scalar={})",
                batched.accepted_inputs, scalar.accepted_inputs
            );
            assert_eq!(
                batched.accepted_tokens, scalar.accepted_tokens,
                "row {b}: accepted_tokens mismatch (batched={:?}, scalar={:?})",
                batched.accepted_tokens, scalar.accepted_tokens
            );

            let scalar_f32 = as_dtype(&scalar.updated_target_hidden, Dtype::Float32);
            let batched_f32 = as_dtype(&batched.updated_target_hidden, Dtype::Float32);
            eval(&[&scalar_f32, &batched_f32]);
            assert_eq!(
                scalar_f32.shape(),
                batched_f32.shape(),
                "row {b}: updated_target_hidden shape mismatch (scalar={:?}, batched={:?})",
                scalar_f32.shape(),
                batched_f32.shape()
            );

            let mut max_abs_delta = 0.0_f32;
            for (idx, (lhs, rhs)) in scalar_f32
                .as_slice_f32()
                .iter()
                .zip(batched_f32.as_slice_f32().iter())
                .enumerate()
            {
                let delta = (lhs - rhs).abs();
                if delta > 0.0 {
                    panic!(
                        "row {b}: updated_target_hidden[{idx}] mismatch: scalar={lhs} batched={rhs} (|delta|={delta})"
                    );
                }
                max_abs_delta = max_abs_delta.max(delta);
            }
            overall_max_abs_delta = overall_max_abs_delta.max(max_abs_delta);
            eprintln!(
                "dflash_qwen35_verify_batched_matches_two_single_row_runs row {b} max_abs_delta={max_abs_delta}"
            );
        }
        // Suppress unused warnings for state we set up but don't compare here
        // (Phase 2 covers KV/GDR per-row unstacking + cache_len consistency).
        let _ = (
            scalar_post_kv,
            scalar_post_gdr,
            scalar_post_cache_lens,
            target_cache_lens,
        );

        eprintln!(
            "dflash_qwen35_verify_batched_matches_two_single_row_runs overall_max_abs_delta={overall_max_abs_delta}"
        );
        assert_eq!(overall_max_abs_delta, 0.0);

        Ok(())
    }

    /// Phase-2B bit-ident test for `MetalRequestState::try_decode_qwen35_dflash_speculative_batch`.
    ///
    /// Constructs three Qwen3.5 DFlash `MetalRequestState` instances that share
    /// the same prompt, sampling params (greedy, temperature=0.0) and model
    /// handles. Drives each through prefill to the Decode phase (which
    /// captures `target_hidden` + a committed `last_token` per the Qwen3.5
    /// prefill path), then:
    ///
    ///   * Runs a single scalar `decode_step` on state C to snapshot the
    ///     expected first token — the scalar DFlash `decode_token` routes
    ///     through `qwen35_dflash_speculative_block`.
    ///   * Runs `MetalRequestState::try_decode_qwen35_dflash_speculative_batch`
    ///     on `[&mut A, &mut B]` — exercises stacking, the batched kernel,
    ///     unstacking, draft-state reinstallation, and `record_sampled_token`.
    ///
    /// Because all three rows start from identical state with greedy sampling,
    /// the returned first tokens must match bit-identically
    /// (`sampled[0] == sampled[1] == scalar_first_token`). This proves the
    /// wrapper's stack/unstack + scheduler-state glue preserves scalar
    /// semantics; the kernel-level numerics (updated_target_hidden,
    /// accepted_tokens) are already covered by
    /// `dflash_qwen35_verify_batched_matches_two_single_row_runs`.
    #[test]
    fn qwen35_dflash_packed_batch_b2_matches_scalar_runs() -> Result<()> {
        use super::super::qwen35::load_qwen35_metal_weights;
        use super::super::request_state::MetalRequestState;
        use super::super::weights::MetalWeights;

        let Some(model_path) = env::var_os("QWEN35_MODEL_PATH").map(PathBuf::from) else {
            eprintln!(
                "QWEN35_MODEL_PATH unset; skipping DFlash packed batch B=2 wrapper equivalence test"
            );
            return Ok(());
        };
        eprintln!("qwen35_dflash_packed_batch_b2_matches_scalar_runs env_ready");
        let _guard = metal_test_guard();

        // Force both the scalar and batched paths to use the C++ draft forward
        // so their numerics are apples-to-apples. The batched path gates on
        // `DFLASH_DRAFT_CPP=1` (see `MetalDflashRuntime::batched_draft_path_eligible`);
        // the scalar path gates on the same env var in `dflash_draft_forward`
        // (see ~line 1253). Without this, scalar would use Rust and batched
        // would be rejected at the eligibility gate — not a bit-ident test.
        // SAFETY: set_var in tests is a thread-unsafe stdlib API; metal_test_guard
        // holds a global lock that serializes Metal tests, so this is sound here.
        unsafe {
            env::set_var("DFLASH_DRAFT_CPP", "1");
        }

        let config = load_metal_config(&model_path)?;
        let MetalModelArch::Qwen35(_) = &config.arch else {
            anyhow::bail!("QWEN35_MODEL_PATH must point to a Qwen3.5 model");
        };

        // Runtime + target_config must outlive every request state; the
        // `MetalRequestState::new` dflash arg wants `&'static` handles. Leak
        // owned values (the test process exits immediately after, so this
        // is a bounded ≤1MB leak per run). Weights + config similarly leak
        // for the `&'a` constructor bounds.
        let runtime = match MetalDflashRuntime::load(
            &MetalDflashOptions {
                draft_model: "z-lab/Qwen3.5-4B-DFlash".to_string(),
                speculative_tokens: None,
            },
            &config,
        ) {
            Ok(rt) => rt,
            Err(err) => {
                eprintln!(
                    "DFlash draft model unavailable ({err:#}); skipping qwen35_dflash_packed_batch_b2_matches_scalar_runs. Set `QWEN35_DFLASH_DRAFT_PATH` to a local checkpoint to enable."
                );
                return Ok(());
            }
        };
        eprintln!("qwen35_dflash_packed_batch_b2_matches_scalar_runs runtime_loaded");

        let runtime_static: &'static MetalDflashRuntime = Box::leak(Box::new(runtime));

        // Load weights once and share the same `&MetalWeights` across all three
        // states. We Box::leak the MetalWeights enum since MetalRequestState is
        // parameterized by `'a` tied to weights/config references.
        let weights_inner = load_qwen35_metal_weights(&model_path, &config)?;
        let weights_leaked: &'static MetalWeights =
            Box::leak(Box::new(MetalWeights::Qwen35(weights_inner)));
        let config_leaked: &'static MetalModelConfig = Box::leak(Box::new(config));

        // DFlash target_config is the same MetalModelConfig we already leaked.
        let dflash_tuple: Option<(&'static MetalDflashRuntime, &'static MetalModelConfig)> =
            Some((runtime_static, config_leaked));

        let params = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        // Short prompt that's trivially tokenized via raw IDs (no tokenizer
        // dependency in this test). Any valid token sequence with ≥2 tokens
        // exercises the prefill → Decode transition.
        let prompt: Vec<u32> = vec![1, 2, 3, 4, 5];
        let max_new_tokens = 4_usize;

        let build_state = || -> Result<MetalRequestState<'static>> {
            MetalRequestState::new(
                weights_leaked,
                config_leaked,
                prompt.clone(),
                &params,
                false, // use_kv_pool=false (DFlash disables pool anyway)
                max_new_tokens,
                dflash_tuple,
            )
        };

        // Prefill a state to reach Decode phase with target_hidden captured
        // and last_token committed. Budget ≥ prompt_len guarantees terminal
        // prefill step in one call.
        let prefill_to_decode = |state: &mut MetalRequestState<'_>| -> Result<()> {
            let budget = prompt.len() + 1;
            let result = state.prefill_chunk(budget)?;
            ensure!(
                result.emitted_token.is_some(),
                "terminal prefill did not emit a token"
            );
            ensure!(
                state.phase() == super::super::request_state::MetalRequestPhase::Decode,
                "expected Decode phase after prefill, got {:?}",
                state.phase()
            );
            Ok(())
        };

        // ── Scalar baseline: state C drives one decode_step through the
        //     single-row DFlash path. ──
        let mut state_c = build_state()?;
        prefill_to_decode(&mut state_c)?;
        eprintln!("qwen35_dflash_packed_batch_b2_matches_scalar_runs scalar_prefill_done");
        let scalar_first_token = state_c
            .decode_step()?
            .context("scalar decode_step returned None in Decode phase")?;
        eprintln!(
            "qwen35_dflash_packed_batch_b2_matches_scalar_runs scalar_first_token={scalar_first_token}"
        );

        // ── Batched run: states A and B exercise the wrapper. ──
        let mut state_a = build_state()?;
        let mut state_b = build_state()?;
        prefill_to_decode(&mut state_a)?;
        prefill_to_decode(&mut state_b)?;
        eprintln!("qwen35_dflash_packed_batch_b2_matches_scalar_runs batched_prefill_done");

        let mut states: Vec<&mut MetalRequestState<'static>> = vec![&mut state_a, &mut state_b];
        let outcome = MetalRequestState::try_decode_qwen35_dflash_speculative_batch(&mut states)?
            .context(
            "wrapper returned Ok(None) despite satisfying eligibility preconditions",
        )?;
        ensure!(
            outcome.tokens.len() == 2,
            "wrapper returned {} first tokens, expected 2",
            outcome.tokens.len()
        );
        ensure!(
            outcome.ready_indices == vec![0, 1],
            "wrapper routed rows {:?}, expected [0, 1]",
            outcome.ready_indices
        );
        let sampled = outcome.tokens;
        eprintln!(
            "qwen35_dflash_packed_batch_b2_matches_scalar_runs batched_first_tokens=[{}, {}]",
            sampled[0], sampled[1]
        );

        assert_eq!(
            sampled[0], scalar_first_token,
            "row 0 first token mismatch: batched={} scalar={}",
            sampled[0], scalar_first_token
        );
        assert_eq!(
            sampled[1], scalar_first_token,
            "row 1 first token mismatch: batched={} scalar={}",
            sampled[1], scalar_first_token
        );

        Ok(())
    }

    // ── Item 2: compatibility-fallback unit tests ────────────────────────────
    //
    // These exercise the pure-logic `check_compatibility` helper with
    // synthetic configs — no GPU, no weights, no network. They pin the
    // contract that shape/arch mismatches produce a named `FieldMismatch`
    // with both values and a "Fix:" suggestion instead of an opaque
    // FFI crash at weight-load time.

    fn synthetic_target_config() -> super::super::config::MetalModelConfig {
        use super::super::config::{MetalModelArch, MetalModelConfig, MetalNormWeightMode};
        MetalModelConfig {
            hidden_size: 2048,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            num_hidden_layers: 36,
            vocab_size: 151_936,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            head_dim: 128,
            eos_token_id: 151_643,
            stop_token_ids: vec![151_643],
            quantization: None,
            norm_weight_mode: MetalNormWeightMode::AddUnitOffset,
            arch: MetalModelArch::Qwen3,
        }
    }

    fn synthetic_draft_config() -> super::DFlashDraftConfig {
        super::DFlashDraftConfig {
            hidden_size: 2048,
            num_hidden_layers: 1,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            block_size: 4,
            mask_token_id: 0,
            target_layer_ids: vec![35],
            quantization: None,
        }
    }

    #[test]
    fn compat_check_matching_configs_pass() {
        let target = synthetic_target_config();
        let draft = synthetic_draft_config();
        super::check_compatibility(&target, &draft, "synthetic/draft").expect("should accept");
    }

    #[test]
    fn compat_check_hidden_size_mismatch_names_field_and_values() {
        let target = synthetic_target_config();
        let mut draft = synthetic_draft_config();
        draft.hidden_size = 4096;
        let err = super::check_compatibility(&target, &draft, "synthetic/draft")
            .expect_err("should reject mismatched hidden_size");
        let msg = err.to_string();
        assert!(
            msg.contains("hidden_size"),
            "error should name the field: {msg}"
        );
        assert!(msg.contains("2048"), "error should include target: {msg}");
        assert!(msg.contains("4096"), "error should include draft: {msg}");
        assert!(msg.contains("Fix:"), "error should suggest a fix: {msg}");
    }

    #[test]
    fn compat_check_kv_projection_width_mismatch_named() {
        let target = synthetic_target_config();
        let mut draft = synthetic_draft_config();
        draft.num_key_value_heads = 4;
        let err = super::check_compatibility(&target, &draft, "synthetic/draft")
            .expect_err("should reject mismatched kv projection width");
        assert!(err.to_string().contains("kv_proj_width"));
    }

    #[test]
    fn compat_check_rebucketed_heads_are_accepted_when_widths_match() {
        let target = synthetic_target_config();
        let mut draft = synthetic_draft_config();
        draft.num_attention_heads = 32;
        draft.num_key_value_heads = 16;
        draft.head_dim = 64;
        super::check_compatibility(&target, &draft, "synthetic/draft")
            .expect("same q/kv projection widths should be accepted");
    }

    #[test]
    fn compat_check_target_layer_oob_named() {
        let target = synthetic_target_config();
        let mut draft = synthetic_draft_config();
        draft.target_layer_ids = vec![99]; // target has 36 layers
        let err = super::check_compatibility(&target, &draft, "synthetic/draft")
            .expect_err("should reject out-of-range target_layer_ids");
        assert!(err.to_string().contains("target_layer_ids"));
    }

    #[test]
    fn metal_dflash_options_rejects_zero_speculative_tokens() {
        let options = super::MetalDflashOptions {
            draft_model: "z-lab/Qwen3.5-4B-DFlash".to_string(),
            speculative_tokens: Some(0),
        };
        let err = options
            .validate()
            .expect_err("validate() must reject speculative_tokens=0");
        let msg = err.to_string();
        assert!(
            msg.contains(">= 1") || msg.contains("must be"),
            "error should explain the >= 1 requirement: {msg}"
        );
    }

    #[test]
    fn metal_dflash_options_accepts_unset_and_positive_speculative_tokens() {
        let unset = super::MetalDflashOptions {
            draft_model: "z-lab/Qwen3.5-4B-DFlash".to_string(),
            speculative_tokens: None,
        };
        unset
            .validate()
            .expect("unset speculative_tokens must fall through to draft default");
        let positive = super::MetalDflashOptions {
            draft_model: "z-lab/Qwen3.5-4B-DFlash".to_string(),
            speculative_tokens: Some(4),
        };
        positive
            .validate()
            .expect("positive speculative_tokens must validate");
    }

    #[test]
    fn metal_dflash_options_rejects_empty_draft_model() {
        let options = super::MetalDflashOptions {
            draft_model: "   ".to_string(),
            speculative_tokens: Some(4),
        };
        options
            .validate()
            .expect_err("empty draft model must be rejected before load");
    }
}
