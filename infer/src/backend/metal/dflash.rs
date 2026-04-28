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
    mlx::{
        Dtype, MlxArray, as_dtype, async_eval, concatenate_axis, eval, gather_axis1_i32,
        prefix_match_len_i32_batched, rms_norm, slice, take_axis, zeros,
    },
    ops::{extend_kv_cache, linear},
    sampling::{
        gpu_sample_token_batched_masked, gpu_sample_token_masked, validate_metal_sampling_params,
    },
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

    pub(crate) fn mask_token_id(&self) -> u32 {
        self.mask_token_id
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
        WeightTensor::GgufPacked {
            w,
            format,
            rows,
            cols,
        } => WeightTensor::GgufPacked {
            w: w.clone(),
            format: *format,
            rows: *rows,
            cols: *cols,
        },
        WeightTensor::GgufPackedInputReordered {
            w,
            format,
            rows,
            cols,
            num_key_heads,
            num_value_heads_per_key,
            head_dim,
        } => WeightTensor::GgufPackedInputReordered {
            w: w.clone(),
            format: *format,
            rows: *rows,
            cols: *cols,
            num_key_heads: *num_key_heads,
            num_value_heads_per_key: *num_value_heads_per_key,
            head_dim: *head_dim,
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
        WeightTensor::GgufPacked { .. } | WeightTensor::GgufPackedInputReordered { .. } => {
            panic!("DFlash draft model does not support packed GGUF weights")
        }
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
        for (slot, ptr) in kv_flat.iter_mut().zip(out_kv) {
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

#[derive(Clone)]
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
    let first_token =
        sample_last_token_suppress(&prompt_logits, params, Some(runtime.mask_token_id()))?;
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
        let drafted_suffix =
            sample_rows_array_suppress(&draft_logits, params, Some(runtime.mask_token_id()))?;
        let drafted_suffix = materialize_token_array(&drafted_suffix);
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
        let posterior =
            sample_rows_array_suppress(&verifier_logits, params, Some(runtime.mask_token_id()))?;
        let posterior = materialize_token_array(&posterior);
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
    pub prefetched_next_draft: Option<Qwen35PrefetchedDraft>,
}

pub(crate) struct Qwen35PrefetchedDraft {
    seed_token: u32,
    block_tokens: MlxArray,
}

fn qwen35_build_block_tokens(current_token: u32, drafted_suffix: &MlxArray) -> MlxArray {
    let current_token_arr = MlxArray::from_slice_i32(&[current_token as i32], &[1]);
    concatenate_axis(&[current_token_arr, drafted_suffix.clone()], 0)
}

fn qwen35_prepare_draft_block(
    runtime: &MetalDflashRuntime,
    current_token: u32,
    target_hidden: &MlxArray,
    embed_table: &MlxArray,
    lm_head: &super::weights::WeightTensor,
    target_config: &super::config::MetalModelConfig,
    params: &crate::sampler::SamplingParams,
    draft_state: &mut ContiguousKvState,
) -> Result<MlxArray> {
    let block_size_i32 =
        i32::try_from(runtime.block_size).context("Qwen3.5 DFlash block_size does not fit i32")?;
    let mut draft_input_tokens = vec![runtime.mask_token_id; runtime.block_size];
    draft_input_tokens[0] = current_token;
    let noise_embedding = embed_tokens(embed_table, &draft_input_tokens);
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
    let drafted_suffix =
        sample_rows_array_suppress(&draft_logits, params, Some(runtime.mask_token_id()))?;
    draft_state.trim(runtime.block_size);
    draft_state.apply_window(DRAFT_CACHE_SINK_SIZE, DRAFT_CACHE_WINDOW_SIZE);
    Ok(qwen35_build_block_tokens(current_token, &drafted_suffix))
}

pub(crate) fn qwen35_prefetch_next_draft(
    runtime: &MetalDflashRuntime,
    current_token: u32,
    target_hidden: &MlxArray,
    embed_table: &MlxArray,
    lm_head: &super::weights::WeightTensor,
    target_config: &super::config::MetalModelConfig,
    params: &crate::sampler::SamplingParams,
    draft_state: &mut ContiguousKvState,
) -> Result<Qwen35PrefetchedDraft> {
    let block_tokens = qwen35_prepare_draft_block(
        runtime,
        current_token,
        target_hidden,
        embed_table,
        lm_head,
        target_config,
        params,
        draft_state,
    )?;

    let mut eval_refs: Vec<&MlxArray> =
        Vec::with_capacity(1 + draft_state.k_caches.len() + draft_state.v_caches.len());
    eval_refs.push(&block_tokens);
    eval_refs.extend(draft_state.k_caches.iter());
    eval_refs.extend(draft_state.v_caches.iter());
    async_eval(&eval_refs);

    Ok(Qwen35PrefetchedDraft {
        seed_token: current_token,
        block_tokens,
    })
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
    let drafted_suffix =
        sample_rows_array_suppress(&draft_logits, params, Some(runtime.mask_token_id()))?;
    let drafted_suffix = materialize_token_array(&drafted_suffix);
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
    let posterior =
        sample_rows_array_suppress(&verifier_logits, params, Some(runtime.mask_token_id()))?;
    let posterior = materialize_token_array(&posterior);
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
        prefetched_next_draft: None,
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

pub(crate) fn sample_last_token_suppress(
    logits: &MlxArray,
    params: &SamplingParams,
    suppress_token_id: Option<u32>,
) -> Result<u32> {
    let shape = logits.shape();
    let last_row = match shape {
        [rows, vocab] => slice(logits, &[rows - 1, 0], &[*rows, *vocab], &[1, 1]),
        [1, rows, vocab] => {
            let squeezed = super::mlx::reshape(logits, &[*rows, *vocab]);
            slice(&squeezed, &[rows - 1, 0], &[*rows, *vocab], &[1, 1])
        }
        _ => anyhow::bail!("expected rank-2 logits or [1, T, vocab], got shape {shape:?}"),
    };
    let token = gpu_sample_token_masked(&last_row, params, suppress_token_id);
    eval(&[&token]);
    Ok(token.item_i32() as u32)
}

fn sample_rows_array_suppress(
    logits: &MlxArray,
    params: &SamplingParams,
    suppress_token_id: Option<u32>,
) -> Result<MlxArray> {
    let shape = logits.shape();
    ensure!(
        shape.len() == 2,
        "expected rank-2 logits, got shape {shape:?}"
    );
    Ok(as_dtype(
        &gpu_sample_token_batched_masked(logits, params, suppress_token_id),
        Dtype::Int32,
    ))
}

fn materialize_token_array(tokens: &MlxArray) -> Vec<u32> {
    let tokens_i32 = if tokens.dtype() == Dtype::Int32 {
        tokens.clone()
    } else {
        as_dtype(tokens, Dtype::Int32)
    };
    eval(&[&tokens_i32]);
    tokens_i32
        .as_slice_i32()
        .into_iter()
        .map(|token| token as u32)
        .collect()
}

fn prefix_match_len_tokens_batched(lhs: &MlxArray, rhs: &MlxArray) -> MlxArray {
    let lhs_i32 = if lhs.dtype() == Dtype::Int32 {
        lhs.clone()
    } else {
        as_dtype(lhs, Dtype::Int32)
    };
    let rhs_i32 = if rhs.dtype() == Dtype::Int32 {
        rhs.clone()
    } else {
        as_dtype(rhs, Dtype::Int32)
    };
    prefix_match_len_i32_batched(&lhs_i32, &rhs_i32)
}

fn packed_verify_needs_attn_mask(left_padding: &[i32]) -> bool {
    left_padding.iter().any(|&padding| padding != 0)
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

#[allow(clippy::float_cmp)]
fn qwen35_same_sampling_params(a: &SamplingParams, b: &SamplingParams) -> bool {
    // Batched DFlash shares one sampled verify call across rows, so the
    // sampling contract must be bit-identical rather than approximately equal.
    a.temperature == b.temperature
        && a.top_k == b.top_k
        && a.top_p == b.top_p
        && a.min_p == b.min_p
        && a.repetition_penalty == b.repetition_penalty
        && a.frequency_penalty == b.frequency_penalty
        && a.presence_penalty == b.presence_penalty
        && a.seed == b.seed
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
    prefetched_draft: Option<Qwen35PrefetchedDraft>,
) -> Result<DFlashBlockResult> {
    use super::mlx::MlxArray as Arr;

    let profile = std::env::var("QWEN35_DFLASH_PROFILE").is_ok();
    let t_start = std::time::Instant::now();

    // ── 1. Draft forward (same as Qwen3 — draft model is pure transformer) ──
    let block_size_i32 =
        i32::try_from(runtime.block_size).context("Qwen3.5 DFlash block_size does not fit i32")?;
    let block_tokens = if let Some(staged) = prefetched_draft {
        ensure!(
            staged.seed_token == current_token,
            "Qwen3.5 DFlash prefetched seed {} != current token {}",
            staged.seed_token,
            current_token
        );
        staged.block_tokens
    } else {
        qwen35_prepare_draft_block(
            runtime,
            current_token,
            target_hidden,
            embed_table,
            lm_head,
            target_config,
            params,
            draft_state,
        )?
    };
    let t_draft = t_start.elapsed();

    let n_capture_layers = runtime.target_layer_ids.len();
    let expected_tape_count = target_gdr_flat.len() / 2;
    let mut per_pos_match = vec![false; runtime.block_size.saturating_sub(1)];
    let t_snapshot = t_start.elapsed();

    // ── 2. Sampled full-block verify ──
    //
    // Single-row Qwen3.5/Qwen3.6 verify now uses the native scalar-cache C++
    // entrypoint rather than routing through the packed `cache_pos_arr`
    // verifier with `B=1`. That keeps the hot path aligned with the single-row
    // decode/session contract while still sampling the whole posterior block in
    // one target forward.
    let gdr_snapshot: Vec<Arr> = target_gdr_flat.to_vec();
    let layer_ids_i32: Vec<i32> = runtime
        .target_layer_ids
        .iter()
        .map(|&id| id as i32)
        .collect();
    unsafe { mlx_sys::qwen35_set_tape_mode(cpp_model.as_raw(), true) };
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

    let verify_summary = cpp_model.verify_block_summary(
        &block_tokens,
        block_size_i32,
        *target_cache_len,
        target_kv_flat,
        target_gdr_flat,
        params,
        Some(runtime.mask_token_id()),
    )?;
    let tapes = drain_current_qwen35_gdr_tapes(cpp_model, expected_tape_count)?;
    let captured_hiddens = drain_captured_hidden(cpp_model)?;
    let matched = verify_summary.matched_prefix_len;
    ensure!(
        matched < runtime.block_size,
        "Qwen3.5 DFlash verify summary returned matched_prefix_len={} for block_size={}",
        matched,
        runtime.block_size
    );
    if profile {
        for hit in per_pos_match.iter_mut().take(matched) {
            *hit = true;
        }
    }
    let accepted_inputs = matched + 1;
    if accepted_inputs < runtime.block_size {
        qwen35_rollback_to_accepted_varlen(
            target_gdr_flat,
            &gdr_snapshot,
            &tapes,
            &[accepted_inputs as i32],
        )?;
    }
    *target_cache_len += accepted_inputs as i32;
    let t_verify = t_start.elapsed();
    let t_sample = t_verify;

    let mut accepted_tokens_out = if matched == 0 {
        Vec::new()
    } else {
        let accepted_prefix = slice(&block_tokens, &[1], &[1 + matched as i32], &[1]);
        materialize_token_array(&accepted_prefix)
    };
    accepted_tokens_out.push(verify_summary.next_token);

    log::debug!(
        "qwen35_dflash: accepted={}/{} matched_prefix={}",
        accepted_inputs,
        runtime.block_size,
        matched
    );

    // ── 3. Build updated_target_hidden from accepted verify positions. ──
    let updated_target_hidden = qwen35_build_updated_target_hidden(
        &captured_hiddens,
        n_capture_layers,
        &[accepted_inputs as i32],
        std::slice::from_ref(target_hidden),
    )
    .into_iter()
    .next()
    .ok_or_else(|| anyhow!("qwen35_build_updated_target_hidden returned empty Vec"))?;
    let next_prefetched_draft = if profile {
        None
    } else if let Some(next_seed_token) = accepted_tokens_out.last().copied() {
        if is_stop_token(target_config, params, next_seed_token) {
            None
        } else {
            match qwen35_prefetch_next_draft(
                runtime,
                next_seed_token,
                &updated_target_hidden,
                embed_table,
                lm_head,
                target_config,
                params,
                draft_state,
            ) {
                Ok(prefetched) => Some(prefetched),
                Err(err) => {
                    log::warn!(
                        "qwen35_dflash: next-block prefetch failed after accepted_inputs={} next_seed_token={}: {err:#}",
                        accepted_inputs,
                        next_seed_token
                    );
                    None
                }
            }
        }
    } else {
        None
    };

    let t_rollback = t_start.elapsed();

    // Queue all modified state for materialization. The caller drains
    // `token_buffer` for accepted tokens before the next GPU block, so a
    // blocking fence here is unnecessary.
    let mut to_eval: Vec<&Arr> = vec![&updated_target_hidden];
    to_eval.extend(target_gdr_flat.iter());
    to_eval.extend(target_kv_flat.iter());
    async_eval(&to_eval);
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
        prefetched_next_draft: next_prefetched_draft,
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
            rest.iter()
                .all(|params| qwen35_same_sampling_params(params, first)),
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
    for &cur in current_tokens {
        packed_block_tokens.push(cur as i32);
        packed_block_tokens.extend(std::iter::repeat_n(
            runtime.mask_token_id as i32,
            runtime.block_size - 1,
        ));
    }

    // ── 2. Pack noise embeddings + target hiddens. ──
    //
    // `embed_tokens` of the flat [B*block_size] token list, reshape to
    // [B, block_size, hidden]. Equivalent to per-row embed + axis-0 stack
    // but cheaper.
    let flat_tokens_u32: Vec<u32> = packed_block_tokens
        .iter()
        .map(|&token| token as u32)
        .collect();
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

    // ── 4. Packed draft sampling. ──
    //
    // Slice draft_hidden_packed `[B, block_size, hidden]`, skip position 0
    // for every row, flatten to `[B * (block_size - 1), hidden]`, and sample
    // the whole packed suffix in one linear + batched-sampling pass.
    let draft_suffix_len = block_size_i32 - 1;
    debug_assert!(draft_suffix_len >= 0);
    let suffix_hidden = slice(
        &draft_hidden_packed,
        &[0, 1, 0],
        &[batch_i32, block_size_i32, hidden_size_i32],
        &[1, 1, 1],
    );
    let suffix_hidden_2d = reshape(
        &suffix_hidden,
        &[batch_i32 * draft_suffix_len, hidden_size_i32],
    );
    let suffix_logits = linear(&suffix_hidden_2d, lm_head);
    let drafted_suffix = sample_rows_array_suppress(
        &suffix_logits,
        &params_per_row[0],
        Some(runtime.mask_token_id()),
    )?;
    let drafted_suffix = materialize_token_array(&drafted_suffix);
    ensure!(
        drafted_suffix.len() == batch * runtime.block_size.saturating_sub(1),
        "Qwen3.5 DFlash batched draft sampled {} suffix tokens, expected {}",
        drafted_suffix.len(),
        batch * runtime.block_size.saturating_sub(1)
    );
    for (row_idx, drafted_row) in drafted_suffix
        .chunks_exact(runtime.block_size.saturating_sub(1))
        .enumerate()
    {
        let row_start = row_idx * runtime.block_size + 1;
        let row_end = row_start + drafted_row.len();
        for (dst, src) in packed_block_tokens[row_start..row_end]
            .iter_mut()
            .zip(drafted_row.iter())
        {
            *dst = *src as i32;
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
    let tokens_arr = MlxArray::from_slice_i32(&packed_block_tokens, &[batch_i32, block_size_i32]);
    let rope_offsets = MlxArray::from_slice_i32(target_cache_lens, &[batch_i32]);
    // When every row shares the same physical cache cursor there is no
    // left-padding to hide, so we can skip the additive mask and let the C++
    // verify path take its exact 2-pass SDPA fast path.
    let attn_mask = if packed_verify_needs_attn_mask(left_padding) {
        Some(super::mlx::build_varlen_verify_mask(
            left_padding,
            block_size_i32,
            batch_cache_len,
        ))
    } else {
        None
    };

    let n_capture_layers = runtime.target_layer_ids.len();
    let expected_tape_count = packed_target_gdr_flat.len() / 2;

    let posterior_tokens = cpp_model.verify_block_batched_sampled(
        &tokens_arr,
        batch_i32,
        block_size_i32,
        target_cache_lens,
        packed_target_kv_flat,
        packed_target_gdr_flat,
        attn_mask.as_ref(),
        &rope_offsets,
        &params_per_row[0],
        Some(runtime.mask_token_id()),
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
    let posterior_tokens_i32 = if posterior_tokens.dtype() == Dtype::Int32 {
        posterior_tokens.clone()
    } else {
        as_dtype(&posterior_tokens, Dtype::Int32)
    };

    let drafted_prefix = slice(&tokens_arr, &[0, 1], &[batch_i32, block_size_i32], &[1, 1]);
    let posterior_prefix = slice(
        &posterior_tokens_i32,
        &[0, 0],
        &[batch_i32, draft_suffix_len],
        &[1, 1],
    );
    let matched_per_row = prefix_match_len_tokens_batched(&drafted_prefix, &posterior_prefix);
    let posterior_token_per_row = gather_axis1_i32(&posterior_tokens_i32, &matched_per_row);
    eval(&[&matched_per_row, &posterior_token_per_row]);

    let matched_per_row = matched_per_row.as_slice_i32();
    ensure!(
        matched_per_row.iter().all(|&value| value >= 0),
        "mlx_prefix_match_len_i32_batched returned a negative prefix length: {matched_per_row:?}"
    );
    let accepted_inputs: Vec<i32> = matched_per_row.iter().map(|&matched| matched + 1).collect();
    let posterior_token_per_row: Vec<u32> = posterior_token_per_row
        .as_slice_i32()
        .into_iter()
        .map(|token| token as u32)
        .collect();
    ensure!(
        matched_per_row.len() == batch && posterior_token_per_row.len() == batch,
        "Qwen3.5 DFlash batched gather returned matched={} posterior={} rows, expected {batch}",
        matched_per_row.len(),
        posterior_token_per_row.len()
    );

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
    // safe. The next block's prefix-match scan in the sampled-row helper
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
        let row_offset = b * runtime.block_size;
        let row_block = &packed_block_tokens[row_offset..row_offset + runtime.block_size];
        let mut accepted_tokens = Vec::with_capacity(accepted);
        for &tok in row_block.iter().skip(1).take(accepted.saturating_sub(1)) {
            accepted_tokens.push(tok as u32);
        }
        accepted_tokens.push(posterior_token_per_row[b]);
        out.push(DFlashBlockResult {
            accepted_tokens,
            updated_target_hidden,
            accepted_inputs: accepted,
            prefetched_next_draft: None,
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
#[path = "dflash/tests.rs"]
mod tests;
