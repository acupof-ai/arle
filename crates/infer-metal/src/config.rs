//! Qwen3.5/Qwen3.6 config parsing for the Metal executor.

use std::path::Path;

use anyhow::{Context, Result};
use infer_plan::DiffusionGenerationConfig;

/// MLX quantization mode. `Affine` is the classic MLX 4/8-bit format (per-group
/// scale + bias). `Mxfp4` is OCP MX FP4: E2M1 weights with one E8M0 scale per
/// 32-element group and no bias.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum QuantMode {
    Affine,
    Mxfp4,
}

impl QuantMode {
    /// FFI code understood by `quant_mode_str` in mlx_common.h.
    pub(crate) fn ffi(self) -> i32 {
        match self {
            QuantMode::Affine => 0,
            QuantMode::Mxfp4 => 1,
        }
    }
}

/// MLX quantization settings.
///
/// `group_size`/`bits` are the *global* (default) quant params from
/// `config.json`'s `quantization` dict. `per_weight` carries the per-tensor
/// overrides some MLX checkpoints (e.g. OptiQ) ship: the override key is the
/// full tensor name (the loader's `base`), mapping to `(bits, group_size)`.
#[derive(Debug, Clone)]
pub(crate) struct QuantConfig {
    pub(crate) group_size: i32,
    pub(crate) bits: i32,
    pub(crate) mode: QuantMode,
    /// name -> (bits, group_size) overrides; empty when the checkpoint uses a
    /// single global quant config for every weight.
    pub(crate) per_weight: std::sync::Arc<std::collections::HashMap<String, (i32, i32)>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MetalQwen35LayerType {
    FullAttention,
    LinearAttention,
}

#[derive(Debug, Clone)]
pub(crate) struct MetalGdrConfig {
    pub(crate) num_key_heads: usize,
    pub(crate) key_dim: usize,
    pub(crate) num_value_heads: usize,
    pub(crate) value_dim: usize,
    pub(crate) conv_kernel: usize,
    pub(crate) rms_norm_eps: f32,
}

impl MetalGdrConfig {
    pub(crate) fn qkv_dim(&self) -> usize {
        let qk_dim = self.num_key_heads * self.key_dim;
        let v_dim = self.num_value_heads * self.value_dim;
        qk_dim * 2 + v_dim
    }
}

#[derive(Debug, Clone)]
pub(crate) struct MetalQwen35MoeConfig {
    pub(crate) num_experts: usize,
    pub(crate) num_experts_per_tok: usize,
    pub(crate) decoder_sparse_step: usize,
    pub(crate) norm_topk_prob: bool,
    pub(crate) mlp_only_layers: Vec<usize>,
    pub(crate) router_bits: i32,
    pub(crate) router_group_size: i32,
    pub(crate) expert_bits: i32,
    pub(crate) expert_group_size: i32,
}

impl MetalQwen35MoeConfig {
    pub(crate) fn is_moe_layer(&self, idx: usize) -> bool {
        !self.mlp_only_layers.contains(&idx)
            && (idx + 1).is_multiple_of(self.decoder_sparse_step.max(1))
    }
}

/// Qwen3.5/Qwen3.6 architecture parameters used by the C++ compiled builder.
#[derive(Debug, Clone)]
pub(crate) struct MetalQwen35ArchConfig {
    pub(crate) layer_types: Vec<MetalQwen35LayerType>,
    pub(crate) rotary_dim: usize,
    pub(crate) linear: MetalGdrConfig,
    pub(crate) moe: Option<MetalQwen35MoeConfig>,
}

impl MetalQwen35ArchConfig {
    pub(crate) fn num_full_attention_layers(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|&&layer| layer == MetalQwen35LayerType::FullAttention)
            .count()
    }

    pub(crate) fn num_linear_attention_layers(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|&&layer| layer == MetalQwen35LayerType::LinearAttention)
            .count()
    }
}

/// Qwen3.5/Qwen3.6 model config for the Metal executor. `stop_token_ids` is the
/// model-default stop set the executor exposes via `model_stop_token_ids`.
#[derive(Debug, Clone)]
pub(crate) struct MetalModelConfig {
    pub(crate) hidden_size: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) rms_norm_eps: f64,
    pub(crate) rope_theta: f64,
    pub(crate) head_dim: usize,
    pub(crate) stop_token_ids: Vec<u32>,
    pub(crate) quantization: Option<QuantConfig>,
    pub(crate) arch: MetalQwen35ArchConfig,
}

#[derive(Debug, Clone)]
pub(crate) struct MetalDiffusionGemmaConfig {
    pub(crate) config: gemma_spec::DiffusionGemmaConfig,
    pub(crate) generation: DiffusionGenerationConfig,
}

#[derive(Debug, Clone)]
pub(crate) struct MetalGemma4Config {
    pub(crate) text: gemma_spec::Gemma4TextConfig,
    pub(crate) generation: DiffusionGenerationConfig,
    pub(crate) image_token_id: Option<u32>,
    pub(crate) vision_soft_tokens_per_image: Option<usize>,
    pub(crate) vision: Option<MetalGemma4VisionConfig>,
}

#[derive(Debug, Clone)]
pub(crate) struct MetalGemma4VisionConfig {
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) patch_size: usize,
    pub(crate) pooling_kernel_size: usize,
    pub(crate) default_output_length: usize,
    pub(crate) position_embedding_size: usize,
    pub(crate) rope_theta: f32,
    pub(crate) rms_norm_eps: f32,
    pub(crate) use_clipped_linears: bool,
}

/// DeepSeek-OCR (`deepseekocr` / `UnlimitedOCRForCausalLM`) Metal config.
///
/// The text decoder + DeepEncoder + projector are parsed by `deepseek-ocr-spec`.
/// `generation` flattens the diffusion plumbing to autoregressive
/// (`canvas_length=1`, `max_denoising_steps=1`) the same way Gemma4 does, and
/// `image_token_id` is the `<image>` placeholder id resolved from the tokenizer.
#[derive(Debug, Clone)]
pub(crate) struct MetalDeepseekOcrConfig {
    pub(crate) spec: deepseek_ocr_spec::DeepseekOcrConfig,
    pub(crate) generation: DiffusionGenerationConfig,
    pub(crate) image_token_id: u32,
}

pub fn model_dir_is_deepseek_ocr(model_dir: &Path) -> bool {
    try_read_config_json(model_dir)
        .as_ref()
        .and_then(serde_json::Value::as_object)
        .is_some_and(is_deepseek_ocr_config)
}

fn is_deepseek_ocr_config(root: &serde_json::Map<String, serde_json::Value>) -> bool {
    model_type_is(root, "deepseekocr") || architectures_contain(root, "UnlimitedOCR")
}

/// Resolve the `<image>` placeholder token id from `tokenizer_config.json`'s
/// added-tokens table (the processor uses `image_token="<image>"`).
fn resolve_image_token_id(model_dir: &Path) -> Result<u32> {
    let path = model_dir.join("tokenizer_config.json");
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("cannot read {}", path.display()))?;
    let value: serde_json::Value =
        serde_json::from_str(&raw).context("tokenizer_config.json parse")?;
    if let Some(table) = value
        .get("added_tokens_decoder")
        .and_then(serde_json::Value::as_object)
    {
        for (id, entry) in table {
            if entry.get("content").and_then(serde_json::Value::as_str) == Some("<image>")
                && let Ok(parsed) = id.parse::<u32>()
            {
                return Ok(parsed);
            }
        }
    }
    // Fallback: the DeepSeek-OCR mlx-vlm config pins image_token_index=128815.
    Ok(128_815)
}

pub(crate) fn load_deepseek_ocr_config(model_dir: &Path) -> Result<MetalDeepseekOcrConfig> {
    let path = model_dir.join("config.json");
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("cannot read {}", path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&raw).context("config.json parse")?;
    let root = value
        .as_object()
        .context("config.json root must be a JSON object")?;
    anyhow::ensure!(
        is_deepseek_ocr_config(root),
        "config.json is not a DeepSeek-OCR (deepseekocr) checkpoint"
    );
    let spec = deepseek_ocr_spec::DeepseekOcrConfig::from_json_value(&value)
        .map_err(|err| anyhow::anyhow!("{err}"))?;
    anyhow::ensure!(
        spec.is_mxfp8(),
        "DeepSeek-OCR Metal path currently requires an MXFP8 checkpoint (quantization.mode=mxfp8)"
    );
    anyhow::ensure!(
        spec.tile_tag == "2D",
        "DeepSeek-OCR Metal path only supports tile_tag=2D"
    );
    let image_token_id = resolve_image_token_id(model_dir)?;
    let generation = deepseek_ocr_generation_from_config(model_dir, root, &spec)?;
    Ok(MetalDeepseekOcrConfig {
        spec,
        generation,
        image_token_id,
    })
}

fn deepseek_ocr_generation_from_config(
    model_dir: &Path,
    root: &serde_json::Map<String, serde_json::Value>,
    spec: &deepseek_ocr_spec::DeepseekOcrConfig,
) -> Result<DiffusionGenerationConfig> {
    let vocab_size = u32::try_from(spec.text.vocab_size)
        .context("DeepSeek-OCR vocab_size does not fit in u32")?;
    let generation_json = match std::fs::read_to_string(model_dir.join("generation_config.json")) {
        Ok(content) => Some(
            serde_json::from_str::<serde_json::Value>(&content)
                .context("generation_config.json parse")?,
        ),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => None,
        Err(err) => return Err(err.into()),
    };
    let max_new_tokens = generation_json
        .as_ref()
        .and_then(|value| value.get("max_new_tokens"))
        .and_then(serde_json::Value::as_u64)
        .map_or(8192, |n| n as usize);
    let mut generation = DiffusionGenerationConfig::diffusion_gemma(max_new_tokens, vocab_size);
    generation.canvas_length = 1;
    generation.max_denoising_steps = 1;
    generation.pad_token_id = generation_json
        .as_ref()
        .and_then(|value| value.get("pad_token_id"))
        .and_then(serde_json::Value::as_u64)
        .map_or(0, |n| n as u32);
    generation.stop_token_ids = resolve_stop_token_ids(model_dir, root, root)?;
    Ok(generation)
}

pub(crate) fn load_diffusion_gemma_config(model_dir: &Path) -> Result<MetalDiffusionGemmaConfig> {
    let path = model_dir.join("config.json");
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("cannot read {}", path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&raw).context("config.json parse")?;
    diffusion_gemma_config_from_value(&value)
}

pub(crate) fn load_gemma4_config(model_dir: &Path) -> Result<MetalGemma4Config> {
    let path = model_dir.join("config.json");
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("cannot read {}", path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&raw).context("config.json parse")?;
    let root = value
        .as_object()
        .context("config.json root must be a JSON object")?;
    let text_config = root
        .get("text_config")
        .and_then(serde_json::Value::as_object);
    let model = text_config.unwrap_or(root);
    anyhow::ensure!(
        is_gemma4_config(root, model) && !is_diffusion_gemma_config(root, model),
        "config.json is not a normal Gemma4 autoregressive checkpoint"
    );
    let text = gemma_spec::Gemma4TextConfig::from_json_value(&value)
        .map_err(|err| anyhow::anyhow!("{err}"))?;
    let generation = gemma4_generation_from_config(model_dir, root, model, &text)?;
    let image_token_id = root
        .get("image_token_id")
        .and_then(serde_json::Value::as_u64)
        .map(|id| id as u32);
    let vision_soft_tokens_per_image = root
        .get("vision_soft_tokens_per_image")
        .and_then(serde_json::Value::as_u64)
        .map(|n| n as usize);
    let vision = gemma4_vision_config_from_root(root, vision_soft_tokens_per_image)?;
    Ok(MetalGemma4Config {
        text,
        generation,
        image_token_id,
        vision_soft_tokens_per_image,
        vision,
    })
}

fn gemma4_vision_config_from_root(
    root: &serde_json::Map<String, serde_json::Value>,
    root_soft_tokens: Option<usize>,
) -> Result<Option<MetalGemma4VisionConfig>> {
    let Some(vision) = root
        .get("vision_config")
        .and_then(serde_json::Value::as_object)
    else {
        return Ok(None);
    };
    let get_usize = |key: &str, default: usize| -> usize {
        vision
            .get(key)
            .and_then(serde_json::Value::as_u64)
            .map_or(default, |n| n as usize)
    };
    let get_f32 = |key: &str, default: f32| -> f32 {
        vision
            .get(key)
            .and_then(serde_json::Value::as_f64)
            .map_or(default, |n| n as f32)
    };
    let rope_theta = vision
        .get("rope_parameters")
        .and_then(|value| value.get("rope_theta"))
        .and_then(serde_json::Value::as_f64)
        .map_or(100.0, |n| n as f32);
    let hidden_size = get_usize("hidden_size", 768);
    let num_attention_heads = get_usize("num_attention_heads", 12);
    anyhow::ensure!(
        num_attention_heads > 0,
        "Gemma4 vision num_attention_heads must be positive"
    );
    Ok(Some(MetalGemma4VisionConfig {
        hidden_size,
        intermediate_size: get_usize("intermediate_size", hidden_size * 4),
        num_hidden_layers: get_usize("num_hidden_layers", 0),
        num_attention_heads,
        num_key_value_heads: get_usize("num_key_value_heads", num_attention_heads),
        head_dim: get_usize("head_dim", hidden_size / num_attention_heads),
        patch_size: get_usize("patch_size", 16),
        pooling_kernel_size: get_usize("pooling_kernel_size", 3),
        default_output_length: get_usize("default_output_length", root_soft_tokens.unwrap_or(280)),
        position_embedding_size: get_usize("position_embedding_size", 10240),
        rope_theta,
        rms_norm_eps: get_f32("rms_norm_eps", 1e-6),
        use_clipped_linears: vision
            .get("use_clipped_linears")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
    }))
}

pub fn model_dir_is_diffusion_gemma(model_dir: &Path) -> bool {
    let Some(value) = try_read_config_json(model_dir) else {
        return false;
    };
    let Some(root) = value.as_object() else {
        return false;
    };
    let model = root
        .get("text_config")
        .and_then(serde_json::Value::as_object)
        .unwrap_or(root);
    is_diffusion_gemma_config(root, model)
}

pub fn model_dir_is_gemma4(model_dir: &Path) -> bool {
    let Some(value) = try_read_config_json(model_dir) else {
        return false;
    };
    let Some(root) = value.as_object() else {
        return false;
    };
    let model = root
        .get("text_config")
        .and_then(serde_json::Value::as_object)
        .unwrap_or(root);
    is_gemma4_config(root, model) && !is_diffusion_gemma_config(root, model)
}

fn try_read_config_json(model_dir: &Path) -> Option<serde_json::Value> {
    let raw = std::fs::read_to_string(model_dir.join("config.json")).ok()?;
    serde_json::from_str(&raw).ok()
}

fn parse_quant_config(q: &serde_json::Value) -> Result<QuantConfig> {
    let group_size = q
        .get("group_size")
        .and_then(serde_json::Value::as_i64)
        .map_or(64, |n| n as i32);
    let bits = q
        .get("bits")
        .and_then(serde_json::Value::as_i64)
        .map_or(4, |n| n as i32);
    let mode = match q.get("mode").and_then(serde_json::Value::as_str) {
        Some("mxfp4") => QuantMode::Mxfp4,
        Some("affine") | None => QuantMode::Affine,
        Some(other) => anyhow::bail!("unsupported quantization mode '{other}'"),
    };
    // Per-weight overrides: the `quantization` dict's object-valued entries are
    // keyed by the full tensor name and carry their own `bits`/`group_size`.
    // Scalar/string keys are skipped because they are not JSON objects.
    let mut per_weight = std::collections::HashMap::new();
    if let Some(obj) = q.as_object() {
        for (name, value) in obj {
            let Some(entry) = value.as_object() else {
                continue;
            };
            let w_bits = entry
                .get("bits")
                .and_then(serde_json::Value::as_i64)
                .map_or(bits, |n| n as i32);
            let w_gs = entry
                .get("group_size")
                .and_then(serde_json::Value::as_i64)
                .map_or(group_size, |n| n as i32);
            per_weight.insert(name.clone(), (w_bits, w_gs));
        }
    }
    Ok(QuantConfig {
        group_size,
        bits,
        mode,
        per_weight: std::sync::Arc::new(per_weight),
    })
}

pub(crate) fn load_metal_config(model_dir: &Path) -> Result<MetalModelConfig> {
    let path = model_dir.join("config.json");
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("cannot read {}", path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&raw).context("config.json parse")?;
    let root = value
        .as_object()
        .context("config.json root must be a JSON object")?;
    let text_config = root
        .get("text_config")
        .and_then(serde_json::Value::as_object);
    let model = text_config.unwrap_or(root);
    if is_diffusion_gemma_config(root, model) {
        let diffusion = diffusion_gemma_config_from_value(&value)?;
        anyhow::bail!(
            "DiffusionGemma cannot be loaded by the autoregressive Qwen Metal executor: \
             parsed text hidden_size={} layers={} canvas={} vocab={}; use the \
             infer-api MetalDiffusionGemma route, which owns the block-diffusion \
             generate loop and dedicated MLX Gemma4/DiffusionGemma bridge",
            diffusion.config.text_config.hidden_size,
            diffusion.config.text_config.num_hidden_layers,
            diffusion.generation.canvas_length,
            diffusion.generation.vocab_size,
        );
    }
    if is_gemma4_config(root, model) {
        anyhow::bail!(
            "Gemma4 cannot be loaded by the Qwen Metal executor: use the \
             infer-api MetalGemma4 route, which owns the Gemma4 MLX forward path \
             and weight mapping"
        );
    }

    let get_usize =
        |obj: &serde_json::Map<String, serde_json::Value>, key: &str, default: usize| -> usize {
            obj.get(key)
                .and_then(serde_json::Value::as_u64)
                .map_or(default, |x| x as usize)
        };
    let get_f64 =
        |obj: &serde_json::Map<String, serde_json::Value>, key: &str, default: f64| -> f64 {
            obj.get(key)
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(default)
        };

    let layer_types = model
        .get("layer_types")
        .and_then(serde_json::Value::as_array)
        .context("R3a Metal executor requires Qwen3.5 `layer_types`")?
        .iter()
        .map(|value| match value.as_str() {
            Some("full_attention") => Ok(MetalQwen35LayerType::FullAttention),
            Some("linear_attention") => Ok(MetalQwen35LayerType::LinearAttention),
            Some(other) => anyhow::bail!("unsupported Qwen3.5 layer type '{other}'"),
            None => anyhow::bail!("Qwen3.5 layer_types entries must be strings"),
        })
        .collect::<Result<Vec<_>>>()?;

    let hidden_size = get_usize(model, "hidden_size", 2048);
    let num_attention_heads = get_usize(model, "num_attention_heads", 16);
    let head_dim = get_usize(model, "head_dim", hidden_size / num_attention_heads.max(1));
    let num_hidden_layers = get_usize(model, "num_hidden_layers", layer_types.len());
    anyhow::ensure!(
        layer_types.len() == num_hidden_layers,
        "Qwen3.5 layer_types length {} != num_hidden_layers {}",
        layer_types.len(),
        num_hidden_layers
    );

    let rms_norm_eps = get_f64(model, "rms_norm_eps", 1e-6);
    let rope_parameters = model
        .get("rope_parameters")
        .and_then(serde_json::Value::as_object);
    let partial_rotary_factor = rope_parameters
        .and_then(|rope| rope.get("partial_rotary_factor"))
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(1.0);
    let rope_theta = rope_parameters
        .and_then(|rope| rope.get("rope_theta"))
        .and_then(serde_json::Value::as_f64)
        .unwrap_or_else(|| get_f64(model, "rope_theta", 1_000_000.0));

    let quantization = root
        .get("quantization")
        .or_else(|| root.get("quantization_config"))
        .map(parse_quant_config)
        .transpose()?;

    let moe = {
        let nested_moe = model
            .get("moe_config")
            .and_then(serde_json::Value::as_object);
        let mut raw_num_experts = get_usize(model, "num_experts", 0);
        let mut raw_top_k = get_usize(model, "num_experts_per_tok", 0);
        let mut decoder_sparse_step = get_usize(model, "decoder_sparse_step", 1).max(1);
        let mut norm_topk_prob = model
            .get("norm_topk_prob")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(true);
        let mut mlp_only_layers = parse_usize_array(model, "mlp_only_layers").unwrap_or_default();

        if let Some(nested) = nested_moe {
            let nested_num_experts = get_usize(nested, "num_experts", 0);
            if nested_num_experts > 0 {
                raw_num_experts = nested_num_experts;
            }
            let nested_top_k = get_usize(nested, "num_experts_per_tok", 0);
            if nested_top_k > 0 {
                raw_top_k = nested_top_k;
            }
            let nested_sparse_step = get_usize(nested, "decoder_sparse_step", 1);
            if nested_sparse_step > 1 {
                decoder_sparse_step = nested_sparse_step;
            }
            if nested
                .get("norm_topk_prob")
                .and_then(serde_json::Value::as_bool)
                .is_some_and(|value| !value)
            {
                norm_topk_prob = false;
            }
            if let Some(nested_layers) =
                parse_usize_array(nested, "mlp_only_layers").filter(|layers| !layers.is_empty())
            {
                mlp_only_layers = nested_layers;
            }
        }

        if raw_num_experts > 0 {
            let (group_size_default, bits_default) = quantization
                .as_ref()
                .map_or((64, 4), |qc| (qc.group_size, qc.bits));
            Some(MetalQwen35MoeConfig {
                num_experts: raw_num_experts,
                num_experts_per_tok: raw_top_k,
                decoder_sparse_step,
                norm_topk_prob,
                mlp_only_layers,
                router_bits: 8.max(bits_default),
                router_group_size: group_size_default,
                expert_bits: bits_default,
                expert_group_size: group_size_default,
            })
        } else {
            None
        }
    };

    let stop_token_ids = resolve_stop_token_ids(model_dir, root, model)?;

    Ok(MetalModelConfig {
        hidden_size,
        num_attention_heads,
        num_key_value_heads: get_usize(model, "num_key_value_heads", 8),
        num_hidden_layers,
        rms_norm_eps,
        rope_theta,
        head_dim,
        stop_token_ids,
        quantization,
        arch: MetalQwen35ArchConfig {
            layer_types,
            rotary_dim: (head_dim as f64 * partial_rotary_factor) as usize,
            linear: MetalGdrConfig {
                num_key_heads: get_usize(model, "linear_num_key_heads", 0),
                key_dim: get_usize(model, "linear_key_head_dim", 0),
                num_value_heads: get_usize(model, "linear_num_value_heads", 0),
                value_dim: get_usize(model, "linear_value_head_dim", 0),
                conv_kernel: get_usize(model, "linear_conv_kernel_dim", 4),
                rms_norm_eps: rms_norm_eps as f32,
            },
            moe,
        },
    })
}

fn parse_eos_field(value: &serde_json::Value) -> Vec<u32> {
    match value {
        serde_json::Value::Number(n) => n.as_u64().map(|id| vec![id as u32]).unwrap_or_default(),
        serde_json::Value::Array(arr) => arr
            .iter()
            .filter_map(|item| item.as_u64().map(|id| id as u32))
            .collect(),
        _ => Vec::new(),
    }
}

fn resolve_stop_token_ids(
    model_dir: &Path,
    root: &serde_json::Map<String, serde_json::Value>,
    text_config: &serde_json::Map<String, serde_json::Value>,
) -> Result<Vec<u32>> {
    let from_generation_config =
        match std::fs::read_to_string(model_dir.join("generation_config.json")) {
            Ok(content) => {
                let value: serde_json::Value =
                    serde_json::from_str(&content).context("generation_config.json parse")?;
                value
                    .get("eos_token_id")
                    .map(parse_eos_field)
                    .unwrap_or_default()
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(err) => return Err(err.into()),
        };

    let mut ids = from_generation_config;
    extend_unique(
        &mut ids,
        root.get("eos_token_id")
            .map(parse_eos_field)
            .unwrap_or_default(),
    );
    extend_unique(
        &mut ids,
        text_config
            .get("eos_token_id")
            .map(parse_eos_field)
            .unwrap_or_default(),
    );
    if ids.is_empty() {
        ids.push(151_645);
    }
    Ok(ids)
}

fn extend_unique(target: &mut Vec<u32>, src: Vec<u32>) {
    for id in src {
        if !target.contains(&id) {
            target.push(id);
        }
    }
}

pub(crate) fn diffusion_gemma_config_from_value(
    value: &serde_json::Value,
) -> Result<MetalDiffusionGemmaConfig> {
    let config = gemma_spec::DiffusionGemmaConfig::from_json_value(value)
        .map_err(|err| anyhow::anyhow!("{err}"))?;
    let generation = diffusion_generation_from_config(&config)?;
    Ok(MetalDiffusionGemmaConfig { config, generation })
}

fn diffusion_generation_from_config(
    config: &gemma_spec::DiffusionGemmaConfig,
) -> Result<DiffusionGenerationConfig> {
    let text = &config.text_config;
    let vocab_size =
        u32::try_from(text.vocab_size).context("DiffusionGemma vocab_size does not fit in u32")?;
    let generation_config = &config.generation_config;
    let max_new_tokens = generation_config
        .max_new_tokens
        .unwrap_or(config.canvas_length);
    let mut generation = DiffusionGenerationConfig::diffusion_gemma(max_new_tokens, vocab_size);
    generation.canvas_length = config.canvas_length;
    generation.max_denoising_steps = generation_config
        .max_denoising_steps
        .unwrap_or(generation.max_denoising_steps);
    generation.pad_token_id = generation_config
        .pad_token_id
        .unwrap_or(generation.pad_token_id);
    generation.confidence_threshold = generation_config
        .confidence_threshold
        .unwrap_or(generation.confidence_threshold);
    generation.entropy_bound = generation_config
        .sampler_config
        .as_ref()
        .and_then(|sampler| sampler.entropy_bound)
        .unwrap_or(generation.entropy_bound);
    generation.stability_threshold = generation_config
        .stability_threshold
        .unwrap_or(generation.stability_threshold);
    generation.t_min = generation_config.t_min.unwrap_or(generation.t_min);
    generation.t_max = generation_config.t_max.unwrap_or(generation.t_max);
    generation.stop_token_ids = generation_config
        .eos_token_id
        .as_ref()
        .or(config.eos_token_id.as_ref())
        .map(parse_eos_field)
        .filter(|ids| !ids.is_empty())
        // Diffusion-Gemma default stop/EOS token-id set when the config carries
        // none: 1 = `<eos>`, 106 = `<end_of_turn>`, 50 = the model's extra stop
        // id (mirrors the same fallback in `infer-plan`'s
        // `DiffusionGenerationConfig::diffusion_gemma`).
        .unwrap_or_else(|| vec![1, 106, 50]);
    Ok(generation)
}

fn gemma4_generation_from_config(
    model_dir: &Path,
    root: &serde_json::Map<String, serde_json::Value>,
    text_config: &serde_json::Map<String, serde_json::Value>,
    text: &gemma_spec::Gemma4TextConfig,
) -> Result<DiffusionGenerationConfig> {
    let vocab_size =
        u32::try_from(text.vocab_size).context("Gemma4 vocab_size does not fit in u32")?;
    let generation_json = match std::fs::read_to_string(model_dir.join("generation_config.json")) {
        Ok(content) => Some(
            serde_json::from_str::<serde_json::Value>(&content)
                .context("generation_config.json parse")?,
        ),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => None,
        Err(err) => return Err(err.into()),
    };
    let max_new_tokens = generation_json
        .as_ref()
        .and_then(|value| value.get("max_new_tokens"))
        .and_then(serde_json::Value::as_u64)
        .map_or(512, |n| n as usize);
    let mut generation = DiffusionGenerationConfig::diffusion_gemma(max_new_tokens, vocab_size);
    generation.canvas_length = 1;
    generation.max_denoising_steps = 1;
    generation.pad_token_id = generation_json
        .as_ref()
        .and_then(|value| value.get("pad_token_id"))
        .and_then(serde_json::Value::as_u64)
        .map_or(0, |n| n as u32);
    generation.stop_token_ids = resolve_stop_token_ids(model_dir, root, text_config)?;
    Ok(generation)
}

fn is_diffusion_gemma_config(
    root: &serde_json::Map<String, serde_json::Value>,
    model: &serde_json::Map<String, serde_json::Value>,
) -> bool {
    model_type_is(root, "diffusion_gemma")
        || model_type_is(model, "diffusion_gemma")
        || architectures_contain(root, "DiffusionGemma")
        || architectures_contain(model, "DiffusionGemma")
}

fn is_gemma4_config(
    root: &serde_json::Map<String, serde_json::Value>,
    model: &serde_json::Map<String, serde_json::Value>,
) -> bool {
    model_type_is(root, "gemma4")
        || model_type_is(model, "gemma4")
        || architectures_contain(root, "Gemma4")
        || architectures_contain(model, "Gemma4")
}

fn model_type_is(obj: &serde_json::Map<String, serde_json::Value>, expected: &str) -> bool {
    obj.get("model_type")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|model_type| model_type == expected)
}

fn architectures_contain(obj: &serde_json::Map<String, serde_json::Value>, needle: &str) -> bool {
    obj.get("architectures")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|archs| {
            archs
                .iter()
                .filter_map(serde_json::Value::as_str)
                .any(|arch| arch.contains(needle))
        })
}

/// Parse a JSON array field into `Vec<usize>`, dropping non-integer elements.
/// Returns `None` when the key is absent or not an array.
fn parse_usize_array(
    obj: &serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Option<Vec<usize>> {
    obj.get(key)
        .and_then(serde_json::Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_u64().map(|n| n as usize))
                .collect::<Vec<_>>()
        })
}
