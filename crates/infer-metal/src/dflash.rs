//! Fail-closed single-request DFlash substrate for the Metal Qwen3.5/Qwen3.6 path.

use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};
use infer_plan::{SamplingParams, SlotToken, StepOutput};
use serde::Deserialize;

use crate::{
    config::{MetalModelConfig, QuantConfig},
    loader::{load_proj_from_tensors, load_tensor_map, tensor_get},
    mlx::{self, MlxArray},
    model_source,
    qwen35::{
        CppQwen35Model, Qwen35GdrTape, qwen35_norm_needs_offset_correction,
        qwen35_normalize_direct_norm_weight,
    },
    weights::WeightTensor,
};

const DRAFT_CACHE_SINK_SIZE: i32 = 64;
const DRAFT_CACHE_WINDOW_SIZE: i32 = 1024;

#[derive(Clone, Debug)]
pub(crate) struct MetalDflashOptions {
    pub(crate) draft_model: String,
    pub(crate) speculative_tokens: Option<usize>,
    pub(crate) max_rows: usize,
    /// Spec-decode acceptance width: accept a draft token if it lies in the
    /// target's top-k for that position (1 = exact greedy, bit-identical).
    pub(crate) accept_topk: i32,
}

impl MetalDflashOptions {
    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(
            !self.draft_model.trim().is_empty(),
            "Metal DFlash draft model must not be empty"
        );
        if let Some(tokens) = self.speculative_tokens {
            ensure!(
                tokens >= 2,
                "Metal DFlash speculative token override must be >= 2 when set"
            );
        }
        ensure!(
            self.max_rows >= 1,
            "Metal DFlash max rows must be at least 1"
        );
        Ok(())
    }
}

pub(crate) struct MetalDflashRuntime {
    block_size: usize,
    accept_topk: i32,
    mask_token_id: u32,
    max_rows: usize,
    target_layer_ids: Vec<usize>,
    draft_config: DFlashDraftConfig,
    _draft_weights: DFlashDraftWeights,
    draft_cpp_model: DFlashDraftCppModel,
    markov: Option<MarkovHead>,
}

impl MetalDflashRuntime {
    pub(crate) fn load(
        options: &MetalDflashOptions,
        target_config: &MetalModelConfig,
    ) -> Result<Self> {
        options.validate()?;
        let draft_dir =
            model_source::resolve_model_path(&options.draft_model).with_context(|| {
                format!(
                    "failed to resolve Metal DFlash draft model '{}'",
                    options.draft_model
                )
            })?;
        let draft_config = DFlashDraftConfig::load(&draft_dir, target_config.num_hidden_layers)?;
        check_compatibility(target_config, &draft_config, &options.draft_model)?;
        let mut draft_weights =
            DFlashDraftWeights::load(&draft_dir, &draft_config).with_context(|| {
                format!(
                    "failed to load Metal DFlash draft weights from {}",
                    draft_dir.display()
                )
            })?;
        let draft_cpp_model = DFlashDraftCppModel::build(&draft_weights, &draft_config)?;
        let markov = draft_weights.markov.take();
        let default_block_size = draft_config.block_size.max(2);
        let block_size = options
            .speculative_tokens
            .unwrap_or(default_block_size)
            .clamp(2, default_block_size);
        log::info!(
            "Metal DFlash loaded: draft='{}', block_size={}, max_rows={}, target_layers={:?}",
            options.draft_model,
            block_size,
            options.max_rows,
            draft_config.target_layer_ids
        );
        Ok(Self {
            block_size,
            accept_topk: options.accept_topk.max(1),
            mask_token_id: draft_config.mask_token_id,
            max_rows: options.max_rows,
            target_layer_ids: draft_config.target_layer_ids.clone(),
            draft_config,
            _draft_weights: draft_weights,
            draft_cpp_model,
            markov,
        })
    }

    pub(crate) fn accept_topk(&self) -> i32 {
        self.accept_topk
    }

    pub(crate) fn block_size(&self) -> usize {
        self.block_size
    }

    pub(crate) fn max_rows(&self) -> usize {
        self.max_rows
    }

    pub(crate) fn target_layer_ids(&self) -> &[usize] {
        &self.target_layer_ids
    }

    pub(crate) fn new_draft_state(&self, initial_tokens: usize) -> DFlashDraftState {
        DFlashDraftState::new(
            self.draft_config.num_hidden_layers,
            self.draft_config.num_key_value_heads as i32,
            self.draft_config.head_dim as i32,
            initial_tokens,
        )
    }

    fn mask_token_id(&self) -> u32 {
        self.mask_token_id
    }

    fn forward_draft(
        &self,
        noise_embedding: &MlxArray,
        target_hidden: &MlxArray,
        state: &mut DFlashDraftState,
    ) -> Result<MlxArray> {
        let context_len = *target_hidden
            .shape()
            .first()
            .context("DFlash target_hidden must be rank-2")?;
        let seq = *noise_embedding
            .shape()
            .first()
            .context("DFlash noise_embedding must be rank-2")?;
        let mut kv_flat = state.active_kv_flat();
        let hidden = self.draft_cpp_model.forward(
            noise_embedding,
            target_hidden,
            state.rope_offset,
            &mut kv_flat,
        )?;
        state.replace_active_kv_flat(kv_flat)?;
        // The z-lab DFlash forward writes the fc-projected target context AND the
        // `seq` draft rows into its KV (the dual-stream concat). The Qwen35Mtp head
        // has no context prepend — only the `seq` draft rows enter the head's KV.
        let written = match self.draft_config.draft_kind {
            DraftKind::DFlashEagle => context_len + seq,
            DraftKind::Qwen35Mtp => seq,
        };
        state.len += written;
        state.rope_offset += written;
        Ok(hidden)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct DflashCompatError(String);

impl std::fmt::Display for DflashCompatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for DflashCompatError {}

fn check_compatibility(
    target: &MetalModelConfig,
    draft: &DFlashDraftConfig,
    draft_model_id: &str,
) -> std::result::Result<(), DflashCompatError> {
    if draft.target_layer_ids.is_empty() {
        return Err(DflashCompatError(format!(
            "DFlash draft '{draft_model_id}' has empty target_layer_ids"
        )));
    }
    if let Some(&max_layer) = draft.target_layer_ids.iter().max()
        && max_layer >= target.num_hidden_layers
    {
        return Err(DflashCompatError(format!(
            "DFlash draft/target target_layer_ids mismatch: target num_hidden_layers={}, draft max target_layer_id={max_layer}",
            target.num_hidden_layers
        )));
    }
    // The only hard cross-model contract is the hidden width: the head reads the
    // base's hidden (and, for Qwen35Mtp, the base token embedding) at this dim.
    if draft.hidden_size != target.hidden_size {
        return Err(DflashCompatError(format!(
            "DFlash draft/target hidden_size mismatch: target={}, draft={}",
            target.hidden_size, draft.hidden_size
        )));
    }
    // The Qwen35Mtp head has its OWN attention (q/k/v/o proj sized to the head's
    // own num_heads/head_dim, independent of the base). Only DFlashEagle shares
    // the base attention projection widths and must match them.
    if draft.draft_kind == DraftKind::DFlashEagle {
        let target_q_width = target.num_attention_heads.saturating_mul(target.head_dim);
        let draft_q_width = draft.num_attention_heads.saturating_mul(draft.head_dim);
        let target_kv_width = target.num_key_value_heads.saturating_mul(target.head_dim);
        let draft_kv_width = draft.num_key_value_heads.saturating_mul(draft.head_dim);
        if draft_q_width != target_q_width {
            return Err(DflashCompatError(format!(
                "DFlash draft/target q_proj width mismatch: target={target_q_width}, draft={draft_q_width}"
            )));
        }
        if draft_kv_width != target_kv_width {
            return Err(DflashCompatError(format!(
                "DFlash draft/target kv_proj width mismatch: target={target_kv_width}, draft={draft_kv_width}"
            )));
        }
    }
    Ok(())
}

/// One EAGLE draft head, two configs.
///
/// `DFlashEagle` is the z-lab DFlash head: `fc` consumes the base hidden only, a
/// single `hidden_norm`, a non-gated layer with full rotary, and the draft block
/// is produced in one parallel forward. `Qwen35Mtp` is the Qwen3.5/3.6 NextN/MTP
/// head: `fc` consumes `concat[pre_fc_norm_embedding(embed), pre_fc_norm_hidden(
/// hidden)]`, the single layer is a gated full-attention layer (partial rotary),
/// and the draft block is produced by sequential autoregressive steps chaining the
/// head's own hidden output. The two branch once for the fc-input/norm + layer
/// step and share the KV/rollback/verify machinery.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DraftKind {
    DFlashEagle,
    Qwen35Mtp,
}

/// DSpark Markov bigram head. `w1` is the [vocab, rank] embedding table;
/// `w2` is pre-transposed to [rank, vocab] for `matmul(latent, w2)`.
/// The bias for a given previous token is `w2(w1(prev))` — a rank-256
/// bigram table that refines the draft backbone's base logits per step.
struct MarkovHead {
    w1: MlxArray,
    w2: MlxArray,
}

impl MarkovHead {
    /// Bigram bias `[1, vocab]` for the given previous token.
    fn bias(&self, prev_token: u32) -> MlxArray {
        let idx = MlxArray::from_slice_i32(&[prev_token as i32], &[1]);
        let latent = mlx::take_axis(&self.w1, &idx, 0); // [1, rank]
        mlx::matmul(&latent, &self.w2) // [1, vocab]
    }
}

#[derive(Clone, Debug)]
struct DFlashDraftConfig {
    draft_kind: DraftKind,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    attn_output_gate: bool,
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

// Qwen3.5/3.6 NextN/MTP head config (model_type == "qwen3_5_mtp"). The real
// attention dims live under `text_config`; rope parameters carry the partial
// rotary factor + theta. There is no `dflash_config`/`mask_token_id` — the MTP
// draft is autoregressive, not masked.
#[derive(Debug, Deserialize)]
struct RawMtpRopeParams {
    rope_theta: f64,
    #[serde(default = "default_partial_rotary")]
    partial_rotary_factor: f64,
}

fn default_partial_rotary() -> f64 {
    1.0
}

#[derive(Debug, Deserialize)]
struct RawMtpTextConfig {
    hidden_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rms_norm_eps: f64,
    #[serde(default)]
    attn_output_gate: bool,
    rope_parameters: RawMtpRopeParams,
}

#[derive(Debug, Deserialize)]
struct RawMtpConfig {
    block_size: Option<usize>,
    text_config: RawMtpTextConfig,
    quantization: Option<RawDraftQuantConfig>,
    quantization_config: Option<RawDraftQuantConfig>,
}

// Probe the model_type / tensor inventory without committing to one schema.
#[derive(Debug, Deserialize)]
struct RawConfigProbe {
    model_type: Option<String>,
}

// Fallback head-kind detector: the Qwen3.5/3.6 MTP head carries
// `pre_fc_norm_embedding.weight`; the z-lab DFlash head does not. We look in the
// safetensors index (sharded) or the single-shard header without loading weights.
fn draft_head_has_pre_fc_norm(model_dir: &Path) -> bool {
    const NEEDLE: &str = "pre_fc_norm_embedding.weight";
    let index = model_dir.join("model.safetensors.index.json");
    if let Ok(text) = std::fs::read_to_string(&index) {
        return text.contains(NEEDLE);
    }
    let single = model_dir.join("model.safetensors");
    if let Ok(bytes) = std::fs::read(&single)
        && bytes.len() >= 8
    {
        let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap_or([0; 8])) as usize;
        let end = (8 + header_len).min(bytes.len());
        if let Ok(header) = std::str::from_utf8(&bytes[8..end]) {
            return header.contains(NEEDLE);
        }
    }
    false
}

fn raw_quant(
    quantization: Option<RawDraftQuantConfig>,
    quantization_config: Option<RawDraftQuantConfig>,
) -> Option<QuantConfig> {
    quantization.or(quantization_config).map(|q| QuantConfig {
        group_size: q.group_size.unwrap_or(64),
        bits: q.bits.unwrap_or(4),
        // DFlash draft checkpoints ship affine 4-bit; mxfp4 drafts are out of scope.
        mode: crate::config::QuantMode::Affine,
        per_weight: std::sync::Arc::new(std::collections::HashMap::new()),
    })
}

impl DFlashDraftConfig {
    fn load(model_dir: &Path, target_num_layers: usize) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let config_text = std::fs::read_to_string(&config_path)
            .with_context(|| format!("cannot read {}", config_path.display()))?;
        let probe: RawConfigProbe = serde_json::from_str(&config_text)
            .with_context(|| format!("cannot parse {}", config_path.display()))?;
        // Auto-detect the Qwen3.5/3.6 NextN/MTP head: its model_type is
        // `qwen3_5_mtp` and it carries the two pre_fc norms that the z-lab DFlash
        // head does not (see DraftKind docs).
        let is_mtp = probe
            .model_type
            .as_deref()
            .is_some_and(|t| t.contains("qwen3_5_mtp"))
            || draft_head_has_pre_fc_norm(model_dir);
        if is_mtp {
            Self::load_qwen35_mtp(&config_text, &config_path, target_num_layers)
        } else {
            Self::load_dflash_eagle(&config_text, &config_path)
        }
    }

    fn load_dflash_eagle(config_text: &str, config_path: &Path) -> Result<Self> {
        let raw: RawDraftConfig = serde_json::from_str(config_text)
            .with_context(|| format!("cannot parse {}", config_path.display()))?;
        let quantization = raw_quant(raw.quantization, raw.quantization_config);
        Ok(Self {
            draft_kind: DraftKind::DFlashEagle,
            hidden_size: raw.hidden_size,
            num_hidden_layers: raw.num_hidden_layers,
            num_attention_heads: raw.num_attention_heads,
            num_key_value_heads: raw.num_key_value_heads,
            head_dim: raw.head_dim,
            rotary_dim: raw.head_dim,
            attn_output_gate: false,
            rms_norm_eps: raw.rms_norm_eps as f32,
            rope_theta: raw.rope_theta as f32,
            block_size: raw.block_size.unwrap_or(16),
            mask_token_id: raw.dflash_config.mask_token_id,
            target_layer_ids: raw.dflash_config.target_layer_ids,
            quantization,
        })
    }

    fn load_qwen35_mtp(
        config_text: &str,
        config_path: &Path,
        target_num_layers: usize,
    ) -> Result<Self> {
        let raw: RawMtpConfig = serde_json::from_str(config_text)
            .with_context(|| format!("cannot parse {}", config_path.display()))?;
        let t = raw.text_config;
        let quantization = raw_quant(raw.quantization, raw.quantization_config);
        let rotary_dim = (t.head_dim as f64 * t.rope_parameters.partial_rotary_factor) as usize;
        ensure!(
            target_num_layers >= 1,
            "Qwen3.5 MTP head requires target with at least one layer"
        );
        Ok(Self {
            draft_kind: DraftKind::Qwen35Mtp,
            hidden_size: t.hidden_size,
            // The MTP head ships exactly one decoder layer (mtp_num_hidden_layers=1).
            num_hidden_layers: 1,
            num_attention_heads: t.num_attention_heads,
            num_key_value_heads: t.num_key_value_heads,
            head_dim: t.head_dim,
            rotary_dim,
            attn_output_gate: t.attn_output_gate,
            rms_norm_eps: t.rms_norm_eps as f32,
            rope_theta: t.rope_parameters.rope_theta as f32,
            block_size: raw.block_size.unwrap_or(3),
            // No mask token — the MTP draft is autoregressive. Use a sentinel that
            // the draft loop never emits/checks.
            mask_token_id: u32::MAX,
            // The head consumes the BASE residual stream after the final decoder
            // layer (its own pre_fc_norm_hidden re-norms it), which is exactly the
            // capture at the last target layer.
            target_layer_ids: vec![target_num_layers - 1],
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
    down_proj: WeightTensor,
}

struct DFlashDraftWeights {
    layers: Vec<DFlashDraftLayerWeights>,
    fc: WeightTensor,
    // DFlashEagle: the single `hidden_norm`. Qwen35Mtp: the two pre_fc norms.
    hidden_norm: MlxArray,
    pre_fc_norm_embedding: MlxArray,
    pre_fc_norm_hidden: MlxArray,
    norm: MlxArray,
    // DSpark only: the Markov bigram head (absent for plain DFlashEagle/MTP).
    markov: Option<MarkovHead>,
}

impl DFlashDraftWeights {
    fn load(model_dir: &Path, config: &DFlashDraftConfig) -> Result<Self> {
        let tensors = load_tensor_map(model_dir)?;
        let get = |name: &str| tensor_get(&tensors, name);
        let load_proj =
            |base: &str| load_proj_from_tensors(&tensors, base, config.quantization.clone());

        let is_mtp = config.draft_kind == DraftKind::Qwen35Mtp;
        // The Qwen3.5/3.6 head's RMSNorms are GemmaRMSNorm `(1+w)`; reuse the base
        // model's per-conversion offset detection (sampled on input_layernorm and
        // applied uniformly, exactly as qwen35.rs does). DFlashEagle norms are
        // loaded raw (unchanged from prior behavior).
        let mtp_offset = if is_mtp {
            qwen35_norm_needs_offset_correction(&get("layers.0.input_layernorm.weight")?)
        } else {
            false
        };
        let load_norm = |name: &str| -> Result<MlxArray> {
            let w = get(name)?;
            Ok(if is_mtp {
                qwen35_normalize_direct_norm_weight(&w, mtp_offset)
            } else {
                w
            })
        };

        let layers = (0..config.num_hidden_layers)
            .map(|i| -> anyhow::Result<_> {
                let p = |suffix: &str| format!("layers.{i}.{suffix}");
                Ok(DFlashDraftLayerWeights {
                    q_proj: load_proj(&p("self_attn.q_proj"))?,
                    k_proj: load_proj(&p("self_attn.k_proj"))?,
                    v_proj: load_proj(&p("self_attn.v_proj"))?,
                    o_proj: load_proj(&p("self_attn.o_proj"))?,
                    input_layernorm: load_norm(&p("input_layernorm.weight"))?,
                    post_attention_layernorm: load_norm(&p("post_attention_layernorm.weight"))?,
                    q_norm: load_norm(&p("self_attn.q_norm.weight"))?,
                    k_norm: load_norm(&p("self_attn.k_norm.weight"))?,
                    gate_proj: load_proj(&p("mlp.gate_proj"))?,
                    up_proj: load_proj(&p("mlp.up_proj"))?,
                    down_proj: load_proj(&p("mlp.down_proj"))?,
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let zero = || mlx::zeros(&[1], mlx::Dtype::Bfloat16);
        let (hidden_norm, pre_fc_norm_embedding, pre_fc_norm_hidden) = if is_mtp {
            (
                zero(),
                load_norm("pre_fc_norm_embedding.weight")?,
                load_norm("pre_fc_norm_hidden.weight")?,
            )
        } else {
            (get("hidden_norm.weight")?, zero(), zero())
        };

        // DSpark Markov head (optional — absent in plain DFlashEagle checkpoints).
        let markov = match (
            get("markov_head.markov_w1.weight"),
            get("markov_head.markov_w2.weight"),
        ) {
            (Ok(w1), Ok(w2)) => {
                // w2 ships as [vocab, rank] (PyTorch Linear [out, in]); transpose
                // to [rank, vocab] so `matmul(latent, w2)` yields the bias.
                let w2 = mlx::transpose_all(&w2);
                Some(MarkovHead { w1, w2 })
            }
            _ => None,
        };

        Ok(Self {
            layers,
            fc: load_proj("fc")?,
            hidden_norm,
            pre_fc_norm_embedding,
            pre_fc_norm_hidden,
            norm: load_norm("norm.weight")?,
            markov,
        })
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
            ..
        } => (
            w.as_raw(),
            scales.as_raw(),
            MlxArray::as_raw_opt(biases.as_ref()),
            *group_size,
            *bits,
        ),
    }
}

struct DFlashDraftCppModel(*mut std::ffi::c_void);

impl Drop for DFlashDraftCppModel {
    fn drop(&mut self) {
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        unsafe {
            mlx_sys::dflash_draft_free(self.0);
        }
    }
}

// SAFETY: the wrapper solely owns its C++ model handle and all bridge/MLX access is serialized, so it may cross threads.
unsafe impl Send for DFlashDraftCppModel {}

impl DFlashDraftCppModel {
    fn build(weights: &DFlashDraftWeights, config: &DFlashDraftConfig) -> Result<Self> {
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        let model = unsafe { mlx_sys::dflash_draft_new() };
        ensure!(
            !model.is_null(),
            "DFlash draft C++ model init returned null"
        );
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        unsafe {
            mlx_sys::dflash_draft_set_config(
                model,
                config.hidden_size as i32,
                config.num_attention_heads as i32,
                config.num_key_value_heads as i32,
                config.head_dim as i32,
                config.num_hidden_layers as i32,
                config.rotary_dim as i32,
                i32::from(config.attn_output_gate),
                match config.draft_kind {
                    DraftKind::DFlashEagle => 0,
                    DraftKind::Qwen35Mtp => 1,
                },
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
            // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
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
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
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
        if config.draft_kind == DraftKind::Qwen35Mtp {
            // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
            unsafe {
                mlx_sys::dflash_draft_set_qwen35_mtp_norms(
                    model,
                    weights.pre_fc_norm_embedding.as_raw(),
                    weights.pre_fc_norm_hidden.as_raw(),
                );
            }
        }
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        let rc = unsafe { mlx_sys::dflash_draft_finalize(model) };
        if rc != 0 {
            let err = mlx::check_mlx_error()
                .err()
                .map_or_else(|| anyhow::anyhow!("unknown MLX error"), Into::into);
            // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
            unsafe {
                mlx_sys::dflash_draft_free(model);
            }
            return Err(err).context("DFlash draft C++ model finalize failed");
        }
        Ok(Self(model))
    }

    fn forward(
        &self,
        noise_embedding: &MlxArray,
        target_hidden: &MlxArray,
        rope_offset: i32,
        kv_flat: &mut [MlxArray],
    ) -> Result<MlxArray> {
        let mut kv_ptrs: Vec<*mut mlx_sys::mlx_array> =
            kv_flat.iter().map(MlxArray::as_raw).collect();
        let mut out_hidden: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let mut out_kv: Vec<*mut mlx_sys::mlx_array> = vec![std::ptr::null_mut(); kv_flat.len()];
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        let rc = unsafe {
            mlx_sys::dflash_draft_forward(
                self.0,
                noise_embedding.as_raw(),
                target_hidden.as_raw(),
                kv_ptrs.as_mut_ptr(),
                kv_ptrs.len() as i32,
                rope_offset,
                &raw mut out_hidden,
                out_kv.as_mut_ptr(),
            )
        };
        if rc != 0 {
            return Err(mlx::check_mlx_error().unwrap_err());
        }
        for (slot, ptr) in kv_flat.iter_mut().zip(out_kv) {
            // SAFETY: the bridge wrote a valid owned MLX handle to this out-param on success.
            *slot = unsafe { MlxArray::from_raw(ptr) };
        }
        // SAFETY: the bridge wrote a valid owned MLX handle to this out-param on success.
        Ok(unsafe { MlxArray::from_raw(out_hidden) })
    }
}

pub(crate) struct DFlashDraftState {
    k_caches: Vec<MlxArray>,
    v_caches: Vec<MlxArray>,
    len: i32,
    capacity: i32,
    n_kv_heads: i32,
    head_dim: i32,
    rope_offset: i32,
}

impl DFlashDraftState {
    fn new(num_layers: usize, n_kv_heads: i32, head_dim: i32, initial_tokens: usize) -> Self {
        let initial = i32::try_from(initial_tokens).unwrap_or(i32::MAX.saturating_sub(256));
        let capacity = ((initial + 255) / 256 + 1) * 256;
        let shape = [1, n_kv_heads, capacity.max(256), head_dim];
        let (k_caches, v_caches): (Vec<_>, Vec<_>) = (0..num_layers)
            .map(|_| {
                (
                    mlx::zeros(&shape, mlx::Dtype::Bfloat16),
                    mlx::zeros(&shape, mlx::Dtype::Bfloat16),
                )
            })
            .unzip();
        Self {
            k_caches,
            v_caches,
            len: 0,
            capacity: shape[2],
            n_kv_heads,
            head_dim,
            rope_offset: 0,
        }
    }

    /// Drop all written KV: the next `active_kv_flat` returns empty slices so the
    /// head's KV is rebuilt from scratch. Used by the Qwen35Mtp draft, which seeds
    /// a fresh head KV each block (see prepare_draft_block_mtp).
    fn reset(&mut self) {
        self.len = 0;
        self.rope_offset = 0;
    }

    fn active_kv_flat(&self) -> Vec<MlxArray> {
        let lo = [0, 0, 0, 0];
        let hi = [1, self.n_kv_heads, self.len, self.head_dim];
        let st = [1, 1, 1, 1];
        self.k_caches
            .iter()
            .zip(&self.v_caches)
            .flat_map(|(k, v)| [mlx::slice(k, &lo, &hi, &st), mlx::slice(v, &lo, &hi, &st)])
            .collect()
    }

    fn replace_active_kv_flat(&mut self, flat: Vec<MlxArray>) -> Result<()> {
        ensure!(
            flat.len() == self.k_caches.len() * 2,
            "DFlash active KV replacement count mismatch: expected {}, got {}",
            self.k_caches.len() * 2,
            flat.len()
        );
        let mut iter = flat.into_iter();
        for layer in 0..self.k_caches.len() {
            let k = iter
                .next()
                .ok_or_else(|| anyhow::anyhow!("missing DFlash K cache for layer {layer}"))?;
            let v = iter
                .next()
                .ok_or_else(|| anyhow::anyhow!("missing DFlash V cache for layer {layer}"))?;
            let k_shape = k.shape();
            let v_shape = v.shape();
            ensure!(
                k_shape.len() == 4
                    && v_shape.len() == 4
                    && k_shape[0] == 1
                    && v_shape[0] == 1
                    && k_shape[1] == self.n_kv_heads
                    && v_shape[1] == self.n_kv_heads
                    && k_shape[2] == v_shape[2]
                    && k_shape[3] == self.head_dim
                    && v_shape[3] == self.head_dim,
                "invalid DFlash KV cache shapes for layer {layer}: k={k_shape:?}, v={v_shape:?}"
            );
            self.capacity = k_shape[2];
            self.k_caches[layer] = k;
            self.v_caches[layer] = v;
        }
        Ok(())
    }

    fn trim(&mut self, num_tokens: usize) {
        let delta = i32::try_from(num_tokens).unwrap_or(i32::MAX);
        self.len = self.len.saturating_sub(delta);
        self.rope_offset = self.rope_offset.saturating_sub(delta);
    }

    fn apply_window(&mut self, sink_size: i32, window_size: i32) {
        let max_len = sink_size + window_size;
        if self.len <= max_len || max_len <= 0 {
            return;
        }
        let window_start = self.len - window_size;
        for layer in 0..self.k_caches.len() {
            let k_win = mlx::slice(
                &self.k_caches[layer],
                &[0, 0, window_start, 0],
                &[1, self.n_kv_heads, self.len, self.head_dim],
                &[1, 1, 1, 1],
            );
            self.k_caches[layer] = mlx::slice_update(
                &self.k_caches[layer],
                &k_win,
                &[0, 0, sink_size, 0],
                &[1, self.n_kv_heads, max_len, self.head_dim],
            );
            let v_win = mlx::slice(
                &self.v_caches[layer],
                &[0, 0, window_start, 0],
                &[1, self.n_kv_heads, self.len, self.head_dim],
                &[1, 1, 1, 1],
            );
            self.v_caches[layer] = mlx::slice_update(
                &self.v_caches[layer],
                &v_win,
                &[0, 0, sink_size, 0],
                &[1, self.n_kv_heads, max_len, self.head_dim],
            );
        }
        self.len = max_len;
    }
}

pub(crate) struct DFlashBlockResult {
    pub(crate) output: StepOutput,
    pub(crate) updated_target_hidden: MlxArray,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen35_speculative_block(
    runtime: &MetalDflashRuntime,
    slot: usize,
    current_token: u32,
    target_hidden: &MlxArray,
    embed_tokens: &MlxArray,
    lm_head: &WeightTensor,
    target_config: &MetalModelConfig,
    cpp_model: &CppQwen35Model,
    params: &SamplingParams,
    target_kv_flat: &mut [MlxArray],
    target_gdr_flat: &mut [MlxArray],
    target_cache_len: &mut usize,
    draft_state: &mut DFlashDraftState,
) -> Result<DFlashBlockResult> {
    // Draft and verify both take a raw device argmax, so anything that rewrites
    // the logits first (penalties, grammar, logit_bias) has no DFlash equivalent.
    ensure!(
        params.is_raw_argmax(),
        "Metal DFlash currently supports raw-argmax greedy sampling only; refusing request with temperature / penalties / grammar / logit_bias"
    );
    let block_size_i32 =
        i32::try_from(runtime.block_size).context("DFlash block_size does not fit in i32")?;
    let trace = dflash_trace_enabled();
    let t0 = Instant::now();
    let block_tokens = prepare_draft_block(
        runtime,
        current_token,
        target_hidden,
        embed_tokens,
        lm_head,
        target_config,
        params,
        draft_state,
    )
    .map_err(|err| anyhow::anyhow!("DFlash draft block preparation failed: {err:#}"))?;
    let t_draft = Instant::now();
    let old_cache_len = *target_cache_len;
    let kv_snapshot: Vec<MlxArray> = target_kv_flat.to_vec();
    let gdr_snapshot: Vec<MlxArray> = target_gdr_flat.to_vec();
    let expected_tape_count = target_gdr_flat.len() / 2;
    let layer_ids = runtime.target_layer_ids();

    cpp_model.set_tape_mode(true);
    cpp_model.set_capture_layers(layer_ids)?;
    let _guard = Qwen35DflashStateGuard { model: cpp_model };
    let verify_summary = cpp_model
        .verify_block_summary(
            &block_tokens,
            block_size_i32,
            i32::try_from(old_cache_len)
                .context("DFlash target cache length does not fit in i32")?,
            target_kv_flat,
            target_gdr_flat,
            params,
            Some(runtime.mask_token_id()),
            runtime.accept_topk(),
        )
        .map_err(|err| anyhow::anyhow!("DFlash target verify summary failed: {err:#}"))?;
    let t_verify = Instant::now();
    let tapes = cpp_model
        .drain_current_gdr_tapes(expected_tape_count)
        .map_err(|err| anyhow::anyhow!("DFlash GDR tape drain failed: {err:#}"))?;
    let captured_hiddens = cpp_model
        .drain_captured_hidden()
        .map_err(|err| anyhow::anyhow!("DFlash captured hidden drain failed: {err:#}"))?;
    let t_drain = Instant::now();

    let matched = verify_summary.matched_prefix_len;
    ensure!(
        matched < runtime.block_size,
        "DFlash verify summary returned matched_prefix_len={} for block_size={}",
        matched,
        runtime.block_size
    );
    let accepted_inputs = matched + 1;
    // Posterior staged-first: the first rejected/unchecked position is always
    // committed from the target posterior (`next_token`) before the next draft
    // block is allowed to extend from it. Rollback trims target state to the
    // exact accepted input span, then the hidden store is rebuilt for that span.
    if accepted_inputs < runtime.block_size {
        rollback_kv_to_accepted(target_kv_flat, &kv_snapshot, old_cache_len, accepted_inputs)
            .map_err(|err| anyhow::anyhow!("DFlash target KV rollback failed: {err:#}"))?;
        rollback_gdr_to_accepted(
            target_gdr_flat,
            &gdr_snapshot,
            &tapes,
            accepted_inputs as i32,
        )
        .map_err(|err| anyhow::anyhow!("DFlash target GDR rollback failed: {err:#}"))?;
    }
    let t_rollback = Instant::now();
    *target_cache_len = target_cache_len.saturating_add(accepted_inputs);

    let mut accepted_tokens = Vec::with_capacity(accepted_inputs);
    if matched > 0 {
        let prefix = mlx::slice(&block_tokens, &[1], &[1 + matched as i32], &[1]);
        accepted_tokens.extend(
            materialize_i32_tokens(&prefix)?
                .into_iter()
                .map(|t| t as u32),
        );
    }
    accepted_tokens.push(verify_summary.next_token);

    let updated_target_hidden = build_target_hidden_from_captures(
        &captured_hiddens,
        layer_ids.len(),
        accepted_inputs as i32,
    )?;
    let t_commit = Instant::now();
    let mut eval_refs: Vec<&MlxArray> = vec![&updated_target_hidden];
    eval_refs.extend(target_kv_flat.iter());
    eval_refs.extend(target_gdr_flat.iter());
    if trace {
        mlx::eval(&eval_refs);
    } else {
        mlx::async_eval(&eval_refs);
    }
    let t_eval = Instant::now();

    if trace {
        log::info!(
            concat!(
                "Metal DFlash trace slot={} cache_len={} block={} matched={} accepted={} ",
                "draft_ms={:.3} verify_ms={:.3} drain_ms={:.3} rollback_ms={:.3} ",
                "commit_ms={:.3} eval_ms={:.3} total_ms={:.3}"
            ),
            slot,
            old_cache_len,
            runtime.block_size,
            matched,
            accepted_inputs,
            millis(t_draft - t0),
            millis(t_verify - t_draft),
            millis(t_drain - t_verify),
            millis(t_rollback - t_drain),
            millis(t_commit - t_rollback),
            millis(t_eval - t_commit),
            millis(t_eval - t0)
        );
    }

    Ok(DFlashBlockResult {
        output: StepOutput {
            tokens: accepted_tokens
                .into_iter()
                .map(|token| SlotToken {
                    slot,
                    token,
                    logprob: None,
                    top_logprobs: Vec::new(),
                    finish: None,
                })
                .collect(),
        },
        updated_target_hidden,
    })
}

fn dflash_trace_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("INFER_METAL_DFLASH_TRACE")
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

pub(crate) fn dflash_draft_trace_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("INFER_METAL_DFLASH_DRAFT_TRACE")
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

struct Qwen35DflashStateGuard<'a> {
    model: &'a CppQwen35Model,
}

impl Drop for Qwen35DflashStateGuard<'_> {
    fn drop(&mut self) {
        self.model.set_tape_mode(false);
        self.model.clear_capture_layers();
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_draft_block(
    runtime: &MetalDflashRuntime,
    current_token: u32,
    target_hidden: &MlxArray,
    embed_table: &MlxArray,
    lm_head: &WeightTensor,
    target_config: &MetalModelConfig,
    params: &SamplingParams,
    draft_state: &mut DFlashDraftState,
) -> Result<MlxArray> {
    if runtime.draft_config.draft_kind == DraftKind::Qwen35Mtp {
        return prepare_draft_block_mtp(
            runtime,
            current_token,
            target_hidden,
            embed_table,
            lm_head,
            target_config,
            params,
            draft_state,
        );
    }
    let block_size_i32 =
        i32::try_from(runtime.block_size).context("DFlash block_size does not fit in i32")?;
    let draft_trace = dflash_draft_trace_enabled();
    let t0 = Instant::now();
    let mut input_tokens = vec![runtime.mask_token_id(); runtime.block_size];
    input_tokens[0] = current_token;
    let noise_embedding = embed_token_rows(embed_table, &input_tokens);
    if draft_trace {
        mlx::eval(&[&noise_embedding]);
    }
    let t_embed = Instant::now();
    let draft_hidden = runtime.forward_draft(&noise_embedding, target_hidden, draft_state)?;
    if draft_trace {
        mlx::eval(&[&draft_hidden]);
    }
    let t_forward = Instant::now();
    let hidden_i32 =
        i32::try_from(target_config.hidden_size).context("hidden_size does not fit in i32")?;
    let draft_logits = if runtime.markov.is_some() {
        // DSpark: the backbone runs once over [real, mask, …] and the Markov
        // head refines each position sequentially — use ALL rows (no slice).
        apply_weight(&draft_hidden, lm_head)
    } else {
        let draft_block_hidden = mlx::slice(
            &draft_hidden,
            &[1, 0],
            &[block_size_i32, hidden_i32],
            &[1, 1],
        );
        apply_weight(&draft_block_hidden, lm_head)
    };
    if draft_trace {
        mlx::eval(&[&draft_logits]);
    }
    let t_lm_head = Instant::now();
    if let Some(markov) = &runtime.markov {
        // DSpark: the backbone runs once over [real, mask, …] (block_size rows)
        // and the Markov head produces block_size draft tokens — the verify block
        // is [real, d1..d_block_size] (block_size+1 tokens). Reference:
        // DeepSpec sample_block_tokens loops range(proposal_len) = block_size.
        let vocab = *draft_logits
            .shape()
            .get(1)
            .context("DSpark draft logits missing vocab dim")?;
        let mut prev = current_token;
        let mut draft_tokens = Vec::with_capacity(runtime.block_size);
        for step in 0..runtime.block_size {
            let s = i32::try_from(step).context("draft step does not fit in i32")?;
            let step_logits = mlx::slice(&draft_logits, &[s, 0], &[s + 1, vocab], &[1, 1]);
            let bias = markov.bias(prev);
            let corrected = mlx::add(&step_logits, &bias);
            let next = mlx::argmax(&corrected).item_i32();
            ensure!(
                next >= 0,
                "DSpark draft emitted a negative token id: {next}"
            );
            ensure!(
                next as u32 != runtime.mask_token_id(),
                "DSpark draft emitted mask_token_id {}; refusing to verify a masked draft token",
                runtime.mask_token_id()
            );
            draft_tokens.push(next as u32);
            prev = next as u32;
        }
        // The verify block is [real, d1..d_block_size] — block_size+1 tokens.
        let mut block = Vec::with_capacity(runtime.block_size + 1);
        block.push(current_token);
        block.extend(draft_tokens);
        draft_state.trim(runtime.block_size);
        draft_state.apply_window(DRAFT_CACHE_SINK_SIZE, DRAFT_CACHE_WINDOW_SIZE);
        let block_arr = tokens_to_array(&block);
        let t_argmax = Instant::now();
        if draft_trace {
            log::info!(
                concat!(
                    "Metal DFlash draft trace context={} block={} ",
                    "embed_ms={:.3} forward_ms={:.3} logits_ms={:.3} ",
                    "argmax_ms={:.3} tokens_ms={:.3} total_ms={:.3}"
                ),
                target_hidden.shape().first().copied().unwrap_or_default(),
                runtime.block_size,
                millis(t_embed - t0),
                millis(t_forward - t_embed),
                millis(t_lm_head - t_forward),
                millis(t_argmax - t_lm_head),
                0.0,
                millis(t_argmax - t0)
            );
        }
        return Ok(block_arr);
    }
    let suffix = mlx::argmax_axis(&draft_logits, -1);
    let suffix_tokens = materialize_i32_tokens(&suffix)?;
    for (dst, token) in input_tokens.iter_mut().skip(1).zip(suffix_tokens) {
        ensure!(
            token >= 0,
            "DFlash draft emitted a negative token id: {token}"
        );
        ensure!(
            token as u32 != runtime.mask_token_id(),
            "DFlash draft emitted mask_token_id {}; refusing to verify a masked draft token",
            runtime.mask_token_id()
        );
        *dst = token as u32;
    }
    let t_argmax = Instant::now();
    draft_state.trim(runtime.block_size);
    draft_state.apply_window(DRAFT_CACHE_SINK_SIZE, DRAFT_CACHE_WINDOW_SIZE);
    let tokens = tokens_to_array(&input_tokens);
    let t_tokens = Instant::now();
    if draft_trace {
        log::info!(
            concat!(
                "Metal DFlash draft trace context={} block={} ",
                "embed_ms={:.3} forward_ms={:.3} logits_ms={:.3} ",
                "argmax_ms={:.3} tokens_ms={:.3} total_ms={:.3}"
            ),
            target_hidden.shape().first().copied().unwrap_or_default(),
            runtime.block_size,
            millis(t_embed - t0),
            millis(t_forward - t_embed),
            millis(t_lm_head - t_forward),
            millis(t_argmax - t_lm_head),
            millis(t_tokens - t_argmax),
            millis(t_tokens - t0)
        );
    }
    Ok(tokens)
}

/// Qwen3.5/3.6 NextN/MTP draft block.
///
/// The MTP head is autoregressive (one next-token per forward), so a depth-`d`
/// block is `d-1` sequential head forwards chaining the head's own hidden output
/// (the SGLang EAGLE draft loop, `draft_forward`). Unlike the z-lab DFlash head
/// there is no parallel masked block and no re-fed KV context: the head KV is
/// seeded fresh each block from the single last-accepted position
/// (`current_token` + its base hidden), and each step extends it by one. The verify
/// guarantees target-correct accepted tokens regardless, so a shorter head context
/// only trades acceptance rate, never correctness.
#[allow(clippy::too_many_arguments)]
fn prepare_draft_block_mtp(
    runtime: &MetalDflashRuntime,
    current_token: u32,
    target_hidden: &MlxArray,
    embed_table: &MlxArray,
    lm_head: &WeightTensor,
    _target_config: &MetalModelConfig,
    _params: &SamplingParams,
    draft_state: &mut DFlashDraftState,
) -> Result<MlxArray> {
    let draft_trace = dflash_draft_trace_enabled();
    let t0 = Instant::now();

    // Fresh head KV per block: drop everything written by the prior block so the
    // head attends only over this block's draft chain seeded by the current token.
    draft_state.reset();

    let hidden_size = target_hidden.shape().get(1).copied().unwrap_or(0);
    let ctx_rows = target_hidden.shape().first().copied().unwrap_or(0);
    ensure!(
        ctx_rows >= 1 && hidden_size > 0,
        "DFlash MTP target_hidden must have at least one row, got shape {:?}",
        target_hidden.shape()
    );
    // Seed from the LAST accepted position — the hidden that produced current_token.
    let mut step_hidden = mlx::slice(
        target_hidden,
        &[ctx_rows - 1, 0],
        &[ctx_rows, hidden_size],
        &[1, 1],
    );

    let mut input_tokens = vec![0u32; runtime.block_size];
    input_tokens[0] = current_token;
    let mut prev_token = current_token;

    for slot in input_tokens.iter_mut().skip(1) {
        let noise_embedding = embed_token_rows(embed_table, &[prev_token]);
        let draft_hidden = runtime.forward_draft(&noise_embedding, &step_hidden, draft_state)?;
        // The head returns [1, H] (rank-2) for a single position; keep that shape.
        let flat = mlx::reshape(&draft_hidden, &[1, hidden_size]);
        let logits = apply_weight(&flat, lm_head);
        let argmax = mlx::argmax_axis(&logits, -1);
        let next = materialize_i32_tokens(&argmax)?
            .first()
            .copied()
            .context("DFlash MTP argmax produced no token")?;
        ensure!(
            next >= 0,
            "DFlash MTP draft emitted negative token id: {next}"
        );
        let next = next as u32;
        *slot = next;
        prev_token = next;
        step_hidden = flat;
        if draft_trace {
            mlx::eval(&[&step_hidden]);
        }
    }

    let tokens = tokens_to_array(&input_tokens);
    if draft_trace {
        log::info!(
            "Metal DFlash MTP draft trace ctx_rows={} block={} total_ms={:.3}",
            ctx_rows,
            runtime.block_size,
            millis(Instant::now() - t0)
        );
    }
    Ok(tokens)
}

pub(crate) fn build_target_hidden_from_captures(
    captured_hiddens: &[MlxArray],
    expected_layers: usize,
    accepted_inputs: i32,
) -> Result<MlxArray> {
    ensure!(
        expected_layers > 0,
        "DFlash target hidden capture requires at least one target layer"
    );
    ensure!(
        captured_hiddens.len() == expected_layers,
        "DFlash captured hidden count mismatch: expected {}, got {}",
        expected_layers,
        captured_hiddens.len()
    );
    let per_layer = captured_hiddens
        .iter()
        .enumerate()
        .map(|(idx, hidden)| {
            let shape = hidden.shape();
            ensure!(
                shape.len() == 3 && shape[0] == 1 && shape[1] >= accepted_inputs,
                "DFlash captured hidden #{idx} expected shape [1, >= {accepted_inputs}, H], got {shape:?}"
            );
            let hidden_size = shape[2];
            let row = mlx::slice(hidden, &[0, 0, 0], &[1, accepted_inputs, hidden_size], &[1, 1, 1]);
            Ok(mlx::reshape(&row, &[accepted_inputs, hidden_size]))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(mlx::concatenate_axis(&per_layer, 1))
}

fn rollback_kv_to_accepted(
    kv_flat: &mut [MlxArray],
    snapshot: &[MlxArray],
    prefix_len: usize,
    accepted_inputs: usize,
) -> Result<()> {
    ensure!(
        kv_flat.len() == snapshot.len(),
        "DFlash KV snapshot count mismatch: live={}, snapshot={}",
        kv_flat.len(),
        snapshot.len()
    );
    let start = i32::try_from(prefix_len).context("DFlash KV prefix length does not fit in i32")?;
    let accepted = i32::try_from(accepted_inputs)
        .context("DFlash accepted input count does not fit in i32")?;
    let end = start + accepted;
    for (idx, (live, snap)) in kv_flat.iter_mut().zip(snapshot).enumerate() {
        let shape = live.shape();
        ensure!(
            shape.len() == 4,
            "DFlash KV rollback expects rank-4 KV arrays, got #{idx} shape {shape:?}"
        );
        let accepted_slice = mlx::slice(
            live,
            &[0, 0, start, 0],
            &[shape[0], shape[1], end, shape[3]],
            &[1, 1, 1, 1],
        );
        *live = mlx::slice_update(
            snap,
            &accepted_slice,
            &[0, 0, start, 0],
            &[shape[0], shape[1], end, shape[3]],
        );
    }
    Ok(())
}

fn rollback_gdr_to_accepted(
    gdr_flat: &mut [MlxArray],
    snapshot: &[MlxArray],
    tapes: &[Qwen35GdrTape],
    accepted_inputs: i32,
) -> Result<()> {
    for (pair_idx, tape) in tapes.iter().enumerate() {
        let state_idx = 2 * pair_idx;
        let conv_idx = state_idx + 1;
        ensure!(
            conv_idx < gdr_flat.len() && conv_idx < snapshot.len(),
            "DFlash GDR rollback snapshot shorter than tape count"
        );
        gdr_flat[state_idx] = snapshot[state_idx].clone();
        gdr_flat[conv_idx] = snapshot[conv_idx].clone();
        if accepted_inputs > 0 {
            let tape_sliced = slice_prefix_axis1(&tape.innovation_tape, accepted_inputs);
            let k_sliced = slice_prefix_axis1(&tape.k, accepted_inputs);
            let g_sliced = slice_prefix_axis1(&tape.g, accepted_inputs);
            // SAFETY: the bridge wrote a valid owned MLX handle to this out-param on success.
            let replayed = unsafe {
                MlxArray::from_raw(mlx_sys::mlx_tape_replay(
                    tape_sliced.as_raw(),
                    k_sliced.as_raw(),
                    g_sliced.as_raw(),
                    gdr_flat[state_idx].as_raw(),
                    accepted_inputs,
                ))
            };
            gdr_flat[state_idx] = replayed;
        }

        let conv_state = gdr_flat[conv_idx].clone();
        let conv_shape = conv_state.shape().to_vec();
        let conv_kernel_minus_1 = conv_shape.get(1).copied().unwrap_or(3);
        let qkv_shape = tape.qkv.shape().to_vec();
        let qkv_cols = qkv_shape.get(2).copied().unwrap_or(0);
        let qkv_sliced = mlx::slice(
            &tape.qkv,
            &[0, 0, 0],
            &[1, accepted_inputs, qkv_cols],
            &[1, 1, 1],
        );
        let combined = mlx::concatenate_axis(&[conv_state, qkv_sliced], 1);
        let combined_len = combined.shape()[1];
        gdr_flat[conv_idx] = if combined_len > conv_kernel_minus_1 {
            let start = combined_len - conv_kernel_minus_1;
            mlx::slice(
                &combined,
                &[0, start, 0],
                &[1, combined_len, qkv_cols],
                &[1, 1, 1],
            )
        } else {
            combined
        };
    }
    Ok(())
}

fn slice_prefix_axis1(array: &MlxArray, len: i32) -> MlxArray {
    let shape = array.shape();
    match shape {
        [b, _t, d0] => mlx::slice(array, &[0, 0, 0], &[*b, len, *d0], &[1, 1, 1]),
        [b, _t, d0, d1] => mlx::slice(array, &[0, 0, 0, 0], &[*b, len, *d0, *d1], &[1, 1, 1, 1]),
        _ => panic!("DFlash tape expected rank-3 or rank-4, got {shape:?}"),
    }
}

fn embed_token_rows(embed_table: &MlxArray, input_ids: &[u32]) -> MlxArray {
    let ids: Vec<i32> = input_ids.iter().map(|&token| token as i32).collect();
    let indices = MlxArray::from_slice_i32(&ids, &[ids.len() as i32]);
    mlx::take_axis(embed_table, &indices, 0)
}

fn tokens_to_array(tokens: &[u32]) -> MlxArray {
    let ids: Vec<i32> = tokens.iter().map(|&token| token as i32).collect();
    MlxArray::from_slice_i32(&ids, &[ids.len() as i32])
}

pub(crate) fn materialize_i32_tokens(tokens: &MlxArray) -> Result<Vec<i32>> {
    let tokens_i32 = if tokens.dtype() == mlx::Dtype::Int32 {
        tokens.clone()
    } else {
        mlx::as_dtype(tokens, mlx::Dtype::Int32)
    };
    mlx::eval(&[&tokens_i32]);
    Ok(tokens_i32.as_slice_i32().to_vec())
}

fn apply_weight(x: &MlxArray, weight: &WeightTensor) -> MlxArray {
    match weight {
        WeightTensor::Dense(w) => mlx::matmul(x, w),
        WeightTensor::Quantized {
            w,
            scales,
            biases,
            group_size,
            bits,
            mode,
        } => mlx::quantized_matmul(
            x,
            w,
            scales,
            biases.as_ref(),
            true,
            *group_size,
            *bits,
            *mode,
        ),
    }
}
