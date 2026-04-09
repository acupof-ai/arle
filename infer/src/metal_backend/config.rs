use std::path::Path;

use anyhow::{Context, Result};

use crate::model_registry::{ModelArch, detect_arch_from_json};

#[derive(Debug, Clone, Copy)]
pub(super) struct QuantConfig {
    pub(super) group_size: i32,
    pub(super) bits: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MetalNormWeightMode {
    Direct,
    AddUnitOffset,
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
impl MetalNormWeightMode {
    pub(super) fn uses_offset(self) -> bool {
        matches!(self, Self::AddUnitOffset)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MetalQwen35LayerType {
    FullAttention,
    LinearAttention,
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
#[derive(Debug, Clone)]
pub(super) struct MetalQwen35ArchConfig {
    pub(super) layer_types: Vec<MetalQwen35LayerType>,
    pub(super) rotary_dim: usize,
    #[cfg(feature = "metal")]
    pub(super) linear: crate::metal_gdr::MetalGdrConfig,
}

impl MetalQwen35ArchConfig {
    pub(super) fn num_full_attention_layers(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|&&layer| layer == MetalQwen35LayerType::FullAttention)
            .count()
    }

    pub(super) fn num_linear_attention_layers(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|&&layer| layer == MetalQwen35LayerType::LinearAttention)
            .count()
    }
}

#[derive(Debug, Clone)]
pub(super) enum MetalModelArch {
    Qwen3,
    Qwen35(MetalQwen35ArchConfig),
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
#[derive(Debug, Clone)]
pub(super) struct MetalModelConfig {
    pub(super) hidden_size: usize,
    pub(super) num_attention_heads: usize,
    pub(super) num_key_value_heads: usize,
    pub(super) num_hidden_layers: usize,
    pub(super) vocab_size: usize,
    pub(super) rms_norm_eps: f64,
    pub(super) rope_theta: f64,
    pub(super) head_dim: usize,
    pub(super) eos_token_id: u32,
    pub(super) stop_token_ids: Vec<u32>,
    pub(super) quantization: Option<QuantConfig>,
    pub(super) norm_weight_mode: MetalNormWeightMode,
    pub(super) arch: MetalModelArch,
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
impl MetalModelConfig {
    pub(super) fn is_stop_token(&self, token_id: u32) -> bool {
        self.stop_token_ids.contains(&token_id)
    }
}

pub(super) fn load_metal_config(model_dir: &Path) -> Result<MetalModelConfig> {
    let path = model_dir.join("config.json");
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("cannot read {}", path.display()))?;
    let declared_arch = detect_arch_from_json(&raw)
        .context("Metal backend could not determine model architecture")?;
    let v: serde_json::Value = serde_json::from_str(&raw).context("config.json parse")?;
    let root = v
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
    let get_eos = |obj: &serde_json::Map<String, serde_json::Value>, fallback: u32| -> u32 {
        obj.get("eos_token_id")
            .and_then(|e| {
                e.as_u64().map(|n| n as u32).or_else(|| {
                    e.as_array()
                        .and_then(|a| a.first())
                        .and_then(serde_json::Value::as_u64)
                        .map(|n| n as u32)
                })
            })
            .unwrap_or(fallback)
    };

    let hidden_size = get_usize(model, "hidden_size", 2048);
    let num_attention_heads = get_usize(model, "num_attention_heads", 16);
    let head_dim = get_usize(model, "head_dim", hidden_size / num_attention_heads.max(1));
    let eos_token_id = get_eos(model, get_eos(root, 151_645));
    let rms_norm_eps = get_f64(model, "rms_norm_eps", 1e-6);
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

    if let Some(qc) = quantization {
        log::info!(
            "  quantization: {} bits, group_size={}",
            qc.bits,
            qc.group_size
        );
    }

    // Qwen3.5 is identified by `layer_types` containing "full_attention"/"linear_attention"
    // entries. `text_config` alone is NOT sufficient — many multimodal models wrap their
    // text config without being Qwen3.5.
    let has_layer_types = model
        .get("layer_types")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|arr| !arr.is_empty());

    let arch = match declared_arch {
        ModelArch::Qwen35 => {
            anyhow::ensure!(
                has_layer_types,
                "Qwen3.5 Metal config requires non-empty `layer_types`"
            );
            let layer_types = model
                .get("layer_types")
                .and_then(serde_json::Value::as_array)
                .context("Qwen3.5 layer_types missing")?
                .iter()
                .map(|value| match value.as_str() {
                    Some("full_attention") => Ok(MetalQwen35LayerType::FullAttention),
                    Some("linear_attention") => Ok(MetalQwen35LayerType::LinearAttention),
                    Some(other) => Err(anyhow::anyhow!("unsupported Qwen3.5 layer type '{other}'")),
                    None => Err(anyhow::anyhow!(
                        "Qwen3.5 layer_types entries must be strings"
                    )),
                })
                .collect::<Result<Vec<_>>>()?;
            anyhow::ensure!(
                layer_types.len() == get_usize(model, "num_hidden_layers", layer_types.len()),
                "Qwen3.5 layer_types length {} != num_hidden_layers {}",
                layer_types.len(),
                get_usize(model, "num_hidden_layers", layer_types.len())
            );

            let rope_parameters = model
                .get("rope_parameters")
                .and_then(serde_json::Value::as_object);
            let partial_rotary_factor = rope_parameters
                .and_then(|rope| rope.get("partial_rotary_factor"))
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(1.0);
            MetalModelArch::Qwen35(MetalQwen35ArchConfig {
                rotary_dim: (head_dim as f64 * partial_rotary_factor) as usize,
                layer_types,
                #[cfg(feature = "metal")]
                linear: crate::metal_gdr::MetalGdrConfig {
                    num_key_heads: get_usize(model, "linear_num_key_heads", 0),
                    key_dim: get_usize(model, "linear_key_head_dim", 0),
                    num_value_heads: get_usize(model, "linear_num_value_heads", 0),
                    value_dim: get_usize(model, "linear_value_head_dim", 0),
                    conv_kernel: get_usize(model, "linear_conv_kernel_dim", 4),
                    hidden_size,
                    rms_norm_eps: rms_norm_eps as f32,
                },
            })
        }
        ModelArch::Qwen3 => MetalModelArch::Qwen3,
        other => anyhow::bail!(
            "Metal backend currently supports Qwen3/Qwen3.5 only; got {}",
            other.display_name()
        ),
    };

    let rope_theta = model
        .get("rope_parameters")
        .and_then(serde_json::Value::as_object)
        .and_then(|rope| rope.get("rope_theta"))
        .and_then(serde_json::Value::as_f64)
        .unwrap_or_else(|| get_f64(model, "rope_theta", 1_000_000.0));

    let norm_weight_mode = match declared_arch {
        // MLX-converted Qwen3.5 checkpoints already run sanitize(), which shifts
        // the offset-style RMSNorm weights during conversion. The Metal path
        // must consume those weights directly instead of applying a second `+1`.
        ModelArch::Qwen35 => MetalNormWeightMode::Direct,
        ModelArch::Qwen3 => MetalNormWeightMode::AddUnitOffset,
        _ => unreachable!("unsupported architectures return earlier"),
    };

    Ok(MetalModelConfig {
        hidden_size,
        num_attention_heads,
        num_key_value_heads: get_usize(model, "num_key_value_heads", 8),
        num_hidden_layers: get_usize(model, "num_hidden_layers", 24),
        vocab_size: get_usize(model, "vocab_size", 151_936),
        rms_norm_eps,
        rope_theta,
        head_dim,
        eos_token_id,
        stop_token_ids: load_stop_token_ids(model_dir, eos_token_id)?,
        quantization,
        norm_weight_mode,
        arch,
    })
}

fn load_stop_token_ids(model_dir: &Path, fallback_eos_token_id: u32) -> Result<Vec<u32>> {
    let generation_config_path = model_dir.join("generation_config.json");
    match std::fs::read_to_string(&generation_config_path) {
        Ok(content) => {
            let v: serde_json::Value =
                serde_json::from_str(&content).context("generation_config.json parse")?;
            let mut ids = Vec::new();
            if let Some(eos) = v.get("eos_token_id") {
                match eos {
                    serde_json::Value::Number(n) => {
                        if let Some(id) = n.as_u64() {
                            ids.push(id as u32);
                        }
                    }
                    serde_json::Value::Array(arr) => {
                        for item in arr {
                            if let Some(id) = item.as_u64() {
                                ids.push(id as u32);
                            }
                        }
                    }
                    _ => {}
                }
            }
            if ids.is_empty() {
                ids.push(fallback_eos_token_id);
            }
            ids.sort_unstable();
            ids.dedup();
            Ok(ids)
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(vec![fallback_eos_token_id]),
        Err(err) => Err(err.into()),
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::tempdir;

    use super::{MetalModelArch, MetalNormWeightMode, load_metal_config};

    fn write_config_file(contents: &str) -> tempfile::TempDir {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("config.json"), contents).unwrap();
        dir
    }

    #[test]
    fn loads_qwen35_config_without_treating_text_config_as_generic_qwen() {
        let dir = write_config_file(
            r#"{
                "architectures": ["Qwen2ForCausalLM"],
                "text_config": {
                    "hidden_size": 2560,
                    "num_attention_heads": 32,
                    "num_key_value_heads": 8,
                    "num_hidden_layers": 2,
                    "head_dim": 80,
                    "layer_types": ["full_attention", "linear_attention"]
                }
            }"#,
        );

        let config = load_metal_config(dir.path()).unwrap();
        assert_eq!(config.norm_weight_mode, MetalNormWeightMode::Direct);
        match config.arch {
            MetalModelArch::Qwen35(arch) => {
                assert_eq!(arch.num_full_attention_layers(), 1);
                assert_eq!(arch.num_linear_attention_layers(), 1);
            }
            other => panic!("expected Qwen3.5 config, got {other:?}"),
        }
    }

    #[test]
    fn rejects_non_qwen_metal_config_instead_of_silently_falling_back() {
        let dir = write_config_file(
            r#"{
                "architectures": ["ChatGLMModel"],
                "hidden_size": 4096,
                "num_attention_heads": 32
            }"#,
        );

        let err = load_metal_config(dir.path()).unwrap_err().to_string();
        assert!(err.contains("supports Qwen3/Qwen3.5 only"), "err={err}");
        assert!(err.contains("GLM-4"), "err={err}");
    }
}
