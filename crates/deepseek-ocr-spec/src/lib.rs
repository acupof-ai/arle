//! DeepSeek-OCR (`UnlimitedOCRForCausalLM` / `model_type=deepseekocr`) config.
//!
//! The checkpoint is an MLX/MXFP8 vision-language model: a DeepEncoder
//! (SAM-base + CLIP-large + 16x conv compressor + linear projector) feeding a
//! DeepSeek-MoE text decoder. This crate parses `config.json` into typed config
//! the Metal builder consumes; it carries no MLX/runtime dependency, mirroring
//! `gemma-spec`.

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DeepseekOcrConfigError {
    #[error("invalid deepseek-ocr config: {0}")]
    InvalidConfig(&'static str),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, DeepseekOcrConfigError>;

fn default_rms_norm_eps() -> f32 {
    1e-6
}

fn default_rope_theta() -> f32 {
    10000.0
}

/// DeepSeek-MoE text decoder config (the `language_config` block, with root
/// fallbacks). Attention is plain MHA (`use_mla=false`, `qk_*_head_dim=0`),
/// layer 0 is dense, layers `>= first_k_dense_replace` are MoE.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct DeepseekOcrTextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub n_shared_experts: usize,
    pub n_routed_experts: usize,
    #[serde(default = "default_routed_scaling")]
    pub routed_scaling_factor: f32,
    pub num_experts_per_tok: usize,
    #[serde(default)]
    pub first_k_dense_replace: usize,
    #[serde(default = "default_moe_layer_freq")]
    pub moe_layer_freq: usize,
    pub v_head_dim: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_scoring_func")]
    pub scoring_func: String,
    #[serde(default = "default_topk_method")]
    pub topk_method: String,
    #[serde(default = "default_one")]
    pub n_group: usize,
    #[serde(default = "default_one")]
    pub topk_group: usize,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
    #[serde(default)]
    pub bos_token_id: Option<u32>,
}

fn default_routed_scaling() -> f32 {
    1.0
}

fn default_moe_layer_freq() -> usize {
    1
}

fn default_scoring_func() -> String {
    "softmax".to_string()
}

fn default_topk_method() -> String {
    "greedy".to_string()
}

fn default_one() -> usize {
    1
}

impl DeepseekOcrTextConfig {
    /// Per-layer head dim for plain MHA (`hidden_size / num_attention_heads`).
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads.max(1)
    }

    /// True for a MoE layer: idx >= first_k_dense_replace and on the MoE
    /// frequency. Layer 0 (with `first_k_dense_replace=1`) is dense.
    #[must_use]
    pub fn is_moe_layer(&self, idx: usize) -> bool {
        self.n_routed_experts > 0
            && idx >= self.first_k_dense_replace
            && idx.is_multiple_of(self.moe_layer_freq.max(1))
    }

    /// Fused shared-expert intermediate size (`moe_intermediate_size *
    /// n_shared_experts`); the checkpoint ships the shared experts pre-fused into
    /// one wide SwiGLU.
    #[must_use]
    pub fn shared_expert_intermediate_size(&self) -> usize {
        self.moe_intermediate_size * self.n_shared_experts
    }

    pub fn validate(&self) -> Result<()> {
        if self.num_hidden_layers == 0 {
            return Err(DeepseekOcrConfigError::InvalidConfig(
                "num_hidden_layers must be non-zero",
            ));
        }
        if self.num_attention_heads == 0 || self.hidden_size == 0 {
            return Err(DeepseekOcrConfigError::InvalidConfig(
                "hidden_size and num_attention_heads must be non-zero",
            ));
        }
        if self.n_routed_experts == 0 || self.num_experts_per_tok == 0 {
            return Err(DeepseekOcrConfigError::InvalidConfig(
                "DeepSeek-OCR decoder must be MoE (n_routed_experts and num_experts_per_tok > 0)",
            ));
        }
        Ok(())
    }
}

/// CLIP-large encoder config (the root `vision_config`, `clip-l-14-224` width
/// block). `image_size`/`patch_size` describe the pretrain resolution; at
/// inference the patch grid follows the SAM compressor output.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct DeepseekOcrVisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub image_size: usize,
    pub patch_size: usize,
    #[serde(default = "default_clip_eps")]
    pub layer_norm_eps: f32,
}

fn default_clip_eps() -> f32 {
    1e-6
}

/// SAM-base encoder config (`sam_vit_b` width block). Fixed-shape SAM ViT-B with
/// windowed local attention and dense global attention at `global_attn_indexes`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DeepseekOcrSamConfig {
    pub image_size: usize,
    pub width: usize,
    pub layers: usize,
    pub heads: usize,
    pub patch_size: usize,
    pub window_size: usize,
    pub global_attn_indexes: Vec<usize>,
    pub downsample_channels: Vec<usize>,
    pub mlp_ratio: f32,
}

impl Default for DeepseekOcrSamConfig {
    fn default() -> Self {
        Self {
            image_size: 1024,
            width: 768,
            layers: 12,
            heads: 12,
            patch_size: 16,
            window_size: 14,
            global_attn_indexes: vec![2, 5, 8, 11],
            downsample_channels: vec![512, 1024],
            mlp_ratio: 4.0,
        }
    }
}

/// Linear projector config (`projector_config`): maps concatenated SAM+CLIP
/// features (`input_dim`) to the decoder embedding width (`n_embed`).
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct DeepseekOcrProjectorConfig {
    #[serde(default = "default_projector_type")]
    pub projector_type: String,
    pub input_dim: usize,
    pub n_embed: usize,
}

fn default_projector_type() -> String {
    "linear".to_string()
}

/// MLX quantization descriptor. The checkpoint ships MXFP8 (microscaling FP8):
/// `mode="mxfp8"`, `group_size=32`, `bits=8`.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct DeepseekOcrQuantization {
    #[serde(default = "default_group_size")]
    pub group_size: i32,
    #[serde(default = "default_bits")]
    pub bits: i32,
    #[serde(default = "default_quant_mode")]
    pub mode: String,
}

fn default_group_size() -> i32 {
    32
}

fn default_bits() -> i32 {
    8
}

fn default_quant_mode() -> String {
    "affine".to_string()
}

/// Parsed DeepSeek-OCR config: text decoder + vision encoders + projector +
/// quant + the 2D-tiling layout fields.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DeepseekOcrConfig {
    pub text: DeepseekOcrTextConfig,
    pub vision: DeepseekOcrVisionConfig,
    pub sam: DeepseekOcrSamConfig,
    pub projector: DeepseekOcrProjectorConfig,
    pub quantization: Option<DeepseekOcrQuantization>,
    pub tile_tag: String,
    pub global_view_pos: String,
}

impl DeepseekOcrConfig {
    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        Self::from_json_str(&content)
    }

    pub fn from_json_str(content: &str) -> Result<Self> {
        let value: serde_json::Value = serde_json::from_str(content)?;
        Self::from_json_value(&value)
    }

    pub fn from_json_value(value: &serde_json::Value) -> Result<Self> {
        let root = value
            .as_object()
            .ok_or(DeepseekOcrConfigError::InvalidConfig(
                "config.json root must be a JSON object",
            ))?;

        // Text decoder: prefer the nested `language_config`, fall back to root.
        let text_value = root.get("language_config").unwrap_or(value);
        let text: DeepseekOcrTextConfig = serde_json::from_value(text_value.clone())?;
        text.validate()?;

        // CLIP-large width block lives under vision_config.width["clip-l-14-224"].
        let vision = parse_vision(root)?;
        let sam = parse_sam(root);

        let projector_value =
            root.get("projector_config")
                .ok_or(DeepseekOcrConfigError::InvalidConfig(
                    "config.json missing projector_config",
                ))?;
        let projector: DeepseekOcrProjectorConfig =
            serde_json::from_value(projector_value.clone())?;

        let quantization = root
            .get("quantization")
            .or_else(|| root.get("quantization_config"))
            .map(|q| serde_json::from_value::<DeepseekOcrQuantization>(q.clone()))
            .transpose()?;

        let tile_tag = root
            .get("tile_tag")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("2D")
            .to_string();
        let global_view_pos = root
            .get("global_view_pos")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("head")
            .to_string();

        Ok(Self {
            text,
            vision,
            sam,
            projector,
            quantization,
            tile_tag,
            global_view_pos,
        })
    }

    /// True when the checkpoint is MXFP8-quantized.
    #[must_use]
    pub fn is_mxfp8(&self) -> bool {
        self.quantization
            .as_ref()
            .is_some_and(|q| q.mode == "mxfp8")
    }
}

fn parse_vision(
    root: &serde_json::Map<String, serde_json::Value>,
) -> Result<DeepseekOcrVisionConfig> {
    let vision = root
        .get("vision_config")
        .and_then(serde_json::Value::as_object)
        .ok_or(DeepseekOcrConfigError::InvalidConfig(
            "config.json missing vision_config",
        ))?;
    let clip = vision
        .get("width")
        .and_then(serde_json::Value::as_object)
        .and_then(|w| w.get("clip-l-14-224"))
        .and_then(serde_json::Value::as_object)
        .ok_or(DeepseekOcrConfigError::InvalidConfig(
            "vision_config missing width.clip-l-14-224 block",
        ))?;
    let get = |obj: &serde_json::Map<String, serde_json::Value>, key: &str, default: usize| {
        obj.get(key)
            .and_then(serde_json::Value::as_u64)
            .map_or(default, |n| n as usize)
    };
    let width = get(clip, "width", 1024);
    Ok(DeepseekOcrVisionConfig {
        hidden_size: width,
        // CLIP MLP intermediate = 4x width (fc1 4096 for width 1024).
        intermediate_size: get(clip, "intermediate_size", width * 4),
        num_hidden_layers: get(clip, "layers", 24),
        num_attention_heads: get(clip, "heads", 16),
        image_size: get(clip, "image_size", 224),
        patch_size: get(clip, "patch_size", 14),
        layer_norm_eps: 1e-6,
    })
}

fn parse_sam(root: &serde_json::Map<String, serde_json::Value>) -> DeepseekOcrSamConfig {
    let mut sam = DeepseekOcrSamConfig::default();
    let Some(block) = root
        .get("vision_config")
        .and_then(serde_json::Value::as_object)
        .and_then(|v| v.get("width"))
        .and_then(serde_json::Value::as_object)
        .and_then(|w| w.get("sam_vit_b"))
        .and_then(serde_json::Value::as_object)
    else {
        return sam;
    };
    if let Some(width) = block.get("width").and_then(serde_json::Value::as_u64) {
        sam.width = width as usize;
    }
    if let Some(layers) = block.get("layers").and_then(serde_json::Value::as_u64) {
        sam.layers = layers as usize;
    }
    if let Some(heads) = block.get("heads").and_then(serde_json::Value::as_u64) {
        sam.heads = heads as usize;
    }
    if let Some(idxs) = block
        .get("global_attn_indexes")
        .and_then(serde_json::Value::as_array)
    {
        sam.global_attn_indexes = idxs
            .iter()
            .filter_map(|v| v.as_u64().map(|n| n as usize))
            .collect();
    }
    if let Some(ch) = block
        .get("downsample_channels")
        .and_then(serde_json::Value::as_array)
    {
        sam.downsample_channels = ch
            .iter()
            .filter_map(|v| v.as_u64().map(|n| n as usize))
            .collect();
    }
    sam
}
