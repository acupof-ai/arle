//! Qwen35 Metal config parsing for the R3a executor.

use std::path::Path;

use anyhow::{Context, Result};

/// MLX affine quantization settings.
#[derive(Debug, Clone, Copy)]
pub(crate) struct QuantConfig {
    pub(crate) group_size: i32,
    pub(crate) bits: i32,
}

/// Qwen3.5 layer type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MetalQwen35LayerType {
    FullAttention,
    LinearAttention,
}

/// Configuration for Qwen3.5 linear-attention/GDR layers.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct MetalGdrConfig {
    pub(crate) num_key_heads: usize,
    pub(crate) key_dim: usize,
    pub(crate) num_value_heads: usize,
    pub(crate) value_dim: usize,
    pub(crate) conv_kernel: usize,
    pub(crate) hidden_size: usize,
    pub(crate) rms_norm_eps: f32,
}

impl MetalGdrConfig {
    pub(crate) fn qkv_dim(&self) -> usize {
        let q_dim = self.num_key_heads * self.key_dim;
        let k_dim = q_dim;
        let v_dim = self.num_value_heads * self.value_dim;
        q_dim + k_dim + v_dim
    }
}

/// Qwen3.5 architecture parameters used by the C++ compiled builder.
#[derive(Debug, Clone)]
pub(crate) struct MetalQwen35ArchConfig {
    pub(crate) layer_types: Vec<MetalQwen35LayerType>,
    pub(crate) rotary_dim: usize,
    pub(crate) linear: MetalGdrConfig,
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

/// Qwen35 model config needed by R3a.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct MetalModelConfig {
    pub(crate) hidden_size: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) vocab_size: usize,
    pub(crate) rms_norm_eps: f64,
    pub(crate) rope_theta: f64,
    pub(crate) head_dim: usize,
    pub(crate) eos_token_id: u32,
    pub(crate) stop_token_ids: Vec<u32>,
    pub(crate) quantization: Option<QuantConfig>,
    pub(crate) arch: MetalQwen35ArchConfig,
}

/// Load a safetensors Qwen3.5 config. Qwen3/Qwen3.6 MoE/GGUF stay deferred.
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

    let num_experts = get_usize(model, "num_experts", 0).max(
        model
            .get("moe_config")
            .and_then(serde_json::Value::as_object)
            .map_or(0, |moe| get_usize(moe, "num_experts", 0)),
    );
    anyhow::ensure!(
        num_experts == 0,
        "R3a Metal executor targets dense Qwen3.5 only; Qwen3.6 MoE is deferred"
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
        .map(|q| QuantConfig {
            group_size: q
                .get("group_size")
                .and_then(serde_json::Value::as_i64)
                .map_or(64, |n| n as i32),
            bits: q
                .get("bits")
                .and_then(serde_json::Value::as_i64)
                .map_or(4, |n| n as i32),
        });

    let stop_token_ids = resolve_stop_token_ids(model_dir, root, model)?;
    let eos_token_id = stop_token_ids.first().copied().unwrap_or(151_645);

    Ok(MetalModelConfig {
        hidden_size,
        num_attention_heads,
        num_key_value_heads: get_usize(model, "num_key_value_heads", 8),
        num_hidden_layers,
        vocab_size: get_usize(model, "vocab_size", 151_936),
        rms_norm_eps,
        rope_theta,
        head_dim,
        eos_token_id,
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
                hidden_size,
                rms_norm_eps: rms_norm_eps as f32,
            },
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
