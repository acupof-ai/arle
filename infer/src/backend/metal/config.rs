use std::path::Path;

use anyhow::{Context, Result};

use crate::{
    gguf::GgufFile,
    model_registry::{ModelArch, detect_arch_from_json},
};

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

/// Mixture-of-Experts architectural parameters for Qwen3.5/3.6.
///
/// Populated only when the checkpoint declares a non-zero `num_experts`
/// (Qwen3.6-35B-A3B and future MoE variants). Dense Qwen3.5 leaves this
/// `None` — the old SwiGLU path stays intact.
#[cfg_attr(not(feature = "metal"), allow(dead_code))]
#[derive(Debug, Clone)]
pub(super) struct MetalQwen35MoeConfig {
    pub(super) num_experts: usize,
    pub(super) num_experts_per_tok: usize,
    pub(super) decoder_sparse_step: usize,
    pub(super) norm_topk_prob: bool,
    pub(super) mlp_only_layers: Vec<usize>,
    /// Router quantization: `mlp.gate` and `mlp.shared_expert_gate` — 8-bit
    /// on MLX-community A3B-4bit, group_size 64.
    pub(super) router_bits: i32,
    pub(super) router_group_size: i32,
    /// Expert quantization: `mlp.switch_mlp.*` and `mlp.shared_expert.*` —
    /// 4-bit on MLX-community A3B-4bit, group_size 64.
    pub(super) expert_bits: i32,
    pub(super) expert_group_size: i32,
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
impl MetalQwen35MoeConfig {
    /// Whether the given layer index uses a MoE block (mirrors
    /// `Qwen35Config::is_moe_layer` — kept local so the Metal config doesn't
    /// need to depend on `qwen35-spec`).
    pub(super) fn is_moe_layer(&self, idx: usize) -> bool {
        !self.mlp_only_layers.contains(&idx)
            && (idx + 1).is_multiple_of(self.decoder_sparse_step.max(1))
    }
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
#[derive(Debug, Clone)]
pub(super) struct MetalQwen35ArchConfig {
    pub(super) layer_types: Vec<MetalQwen35LayerType>,
    pub(super) rotary_dim: usize,
    #[cfg(feature = "metal")]
    pub(super) linear: super::gdr::MetalGdrConfig,
    /// `Some` for Qwen3.6 / Qwen3_5_Moe checkpoints; `None` for dense Qwen3.5.
    pub(super) moe: Option<MetalQwen35MoeConfig>,
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
pub(crate) struct MetalModelConfig {
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
    let hidden_size = get_usize(model, "hidden_size", 2048);
    let num_attention_heads = get_usize(model, "num_attention_heads", 16);
    let head_dim = get_usize(model, "head_dim", hidden_size / num_attention_heads.max(1));

    // HuggingFace precedence: generation_config.json beats config.json; eos_token_id
    // can be a scalar or an array, and the entire array is the stop set. Multimodal
    // configs may put a base-LM EOS in text_config and the chat-end EOS in the root
    // array — taking only the text_config scalar would let the model generate past
    // <|im_end|> and leak fake role markers (Qwen3.6 multimodal MoE).
    let stop_token_ids = resolve_stop_token_ids(model_dir, root, model)?;
    let eos_token_id = stop_token_ids.first().copied().unwrap_or(151_645);
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
        ModelArch::Qwen35 | ModelArch::Qwen3_5_Moe => {
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
            // Optional MoE sub-block — Qwen3.6 only (Qwen3_5_Moe). Absence of
            // `num_experts` (or 0) = dense Qwen3.5 path, preserved unchanged.
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
                let mut mlp_only_layers = model
                    .get("mlp_only_layers")
                    .and_then(serde_json::Value::as_array)
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_u64().map(|n| n as usize))
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();

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
                    if let Some(nested_layers) = nested
                        .get("mlp_only_layers")
                        .and_then(serde_json::Value::as_array)
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_u64().map(|n| n as usize))
                                .collect::<Vec<_>>()
                        })
                        .filter(|layers| !layers.is_empty())
                    {
                        mlp_only_layers = nested_layers;
                    }
                }

                if raw_num_experts > 0 {
                    // Quantization: defaults from `quantization` at the root.
                    // Router and shared_expert_gate are 8-bit in MLX-community
                    // A3B-4bit; experts and shared_expert SwiGLU are 4-bit.
                    // We encode the expected pattern here — a future checkpoint
                    // with different overrides would need per-layer bit reads.
                    let (group_size_default, bits_default) =
                        quantization.map_or((64, 4), |qc| (qc.group_size, qc.bits));
                    // If base quantization is 8-bit, router and experts
                    // collapse to the same bit width — safe to still drive the
                    // MoE block; the FFI accepts independent bits per side.
                    let router_bits = 8.max(bits_default);
                    let expert_bits = bits_default;
                    Some(MetalQwen35MoeConfig {
                        num_experts: raw_num_experts,
                        num_experts_per_tok: raw_top_k,
                        decoder_sparse_step,
                        norm_topk_prob,
                        mlp_only_layers,
                        router_bits,
                        router_group_size: group_size_default,
                        expert_bits,
                        expert_group_size: group_size_default,
                    })
                } else {
                    None
                }
            };

            MetalModelArch::Qwen35(MetalQwen35ArchConfig {
                rotary_dim: (head_dim as f64 * partial_rotary_factor) as usize,
                layer_types,
                #[cfg(feature = "metal")]
                linear: super::gdr::MetalGdrConfig {
                    num_key_heads: get_usize(model, "linear_num_key_heads", 0),
                    key_dim: get_usize(model, "linear_key_head_dim", 0),
                    num_value_heads: get_usize(model, "linear_num_value_heads", 0),
                    value_dim: get_usize(model, "linear_value_head_dim", 0),
                    conv_kernel: get_usize(model, "linear_conv_kernel_dim", 4),
                    hidden_size,
                    rms_norm_eps: rms_norm_eps as f32,
                },
                moe,
            })
        }
        ModelArch::Qwen3 => MetalModelArch::Qwen3,
        other => anyhow::bail!(
            "Metal backend currently supports Qwen3/Qwen3.5/Qwen3.6 only; got {}",
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
        // Metal normalizes Qwen3.5 norm tensors to direct form at load time:
        // GGUF ships `1 + w`, official HF safetensors ship raw offset weights,
        // and the loader canonicalizes both before the runtime sees them.
        ModelArch::Qwen35 | ModelArch::Qwen3_5_Moe => MetalNormWeightMode::Direct,
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
        stop_token_ids,
        quantization,
        norm_weight_mode,
        arch,
    })
}

pub(super) fn load_metal_config_from_gguf(gguf: &GgufFile) -> Result<MetalModelConfig> {
    let arch = gguf
        .architecture()
        .context("GGUF missing general.architecture")?;
    match arch {
        "qwen35" => load_qwen35_config_from_gguf(gguf),
        other => anyhow::bail!(
            "Metal GGUF metadata fallback currently supports qwen35 only; got '{other}'"
        ),
    }
}

pub(super) fn apply_gguf_metadata_overrides(config: &mut MetalModelConfig, gguf: &GgufFile) {
    let Some(arch) = gguf.architecture() else {
        return;
    };
    let p = |field: &str| format!("{arch}.{field}");
    if let Some(theta) = gguf.meta_f32(&p("rope.freq_base")) {
        config.rope_theta = theta as f64;
    }
    if let Some(eos_token_id) = gguf.meta_u32("tokenizer.ggml.eos_token_id") {
        config.eos_token_id = eos_token_id;
        config.stop_token_ids = vec![eos_token_id];
    }
}

fn load_qwen35_config_from_gguf(gguf: &GgufFile) -> Result<MetalModelConfig> {
    let qwen35 = gguf.extract_qwen35_config()?;
    let layer_types = qwen35
        .layer_types
        .iter()
        .map(|layer| match layer {
            qwen35_spec::LayerType::FullAttention => MetalQwen35LayerType::FullAttention,
            qwen35_spec::LayerType::LinearAttention => MetalQwen35LayerType::LinearAttention,
        })
        .collect();

    Ok(MetalModelConfig {
        hidden_size: qwen35.hidden_size,
        num_attention_heads: qwen35.num_attention_heads,
        num_key_value_heads: qwen35.num_key_value_heads,
        num_hidden_layers: qwen35.num_hidden_layers,
        vocab_size: qwen35.vocab_size,
        rms_norm_eps: qwen35.rms_norm_eps as f64,
        rope_theta: qwen35.rope_theta as f64,
        head_dim: qwen35.head_dim,
        eos_token_id: qwen35.eos_token_id,
        stop_token_ids: qwen35.stop_token_ids.clone(),
        quantization: None,
        norm_weight_mode: MetalNormWeightMode::Direct,
        arch: MetalModelArch::Qwen35(MetalQwen35ArchConfig {
            layer_types,
            rotary_dim: qwen35.rotary_dim,
            #[cfg(feature = "metal")]
            linear: super::gdr::MetalGdrConfig {
                num_key_heads: qwen35.linear_num_key_heads,
                key_dim: qwen35.linear_key_head_dim,
                num_value_heads: qwen35.linear_num_value_heads,
                value_dim: qwen35.linear_value_head_dim,
                conv_kernel: qwen35.linear_conv_kernel_dim,
                hidden_size: qwen35.hidden_size,
                rms_norm_eps: qwen35.rms_norm_eps,
            },
            moe: None,
        }),
    })
}

/// Parse a HuggingFace `eos_token_id` value (scalar or array) into a Vec.
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

/// Resolve the stop-token list following HuggingFace precedence:
/// `generation_config.json::eos_token_id` (preferred, used by `model.generate()`)
/// then `config.json::eos_token_id` (root, then `text_config`). All values from
/// the chosen source are stop tokens; ordering preserves the source order
/// (HF uses the first id as the "primary" EOS).
fn resolve_stop_token_ids(
    model_dir: &Path,
    root: &serde_json::Map<String, serde_json::Value>,
    text_config: &serde_json::Map<String, serde_json::Value>,
) -> Result<Vec<u32>> {
    let from_generation_config =
        match std::fs::read_to_string(model_dir.join("generation_config.json")) {
            Ok(content) => {
                let v: serde_json::Value =
                    serde_json::from_str(&content).context("generation_config.json parse")?;
                v.get("eos_token_id")
                    .map(parse_eos_field)
                    .unwrap_or_default()
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(err) => return Err(err.into()),
        };

    let mut ids: Vec<u32> = from_generation_config;
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
            MetalModelArch::Qwen3 => panic!("expected Qwen3.5 config, got Qwen3"),
        }
    }

    #[test]
    fn rejects_non_qwen_metal_config_instead_of_silently_falling_back() {
        let dir = write_config_file(
            r#"{
                "architectures": ["LlamaForCausalLM"],
                "hidden_size": 4096,
                "num_attention_heads": 32
            }"#,
        );

        let err = load_metal_config(dir.path()).unwrap_err().to_string();
        assert!(
            err.contains("supports Qwen3/Qwen3.5/Qwen3.6 only"),
            "err={err}"
        );
        assert!(err.contains("Llama"), "err={err}");
    }

    #[test]
    fn loads_qwen36_config_with_nested_moe_block() {
        let dir = write_config_file(
            r#"{
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                "text_config": {
                    "hidden_size": 2048,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 2,
                    "num_hidden_layers": 4,
                    "head_dim": 128,
                    "layer_types": [
                        "linear_attention",
                        "full_attention",
                        "linear_attention",
                        "full_attention"
                    ],
                    "linear_num_key_heads": 8,
                    "linear_key_head_dim": 128,
                    "linear_num_value_heads": 16,
                    "linear_value_head_dim": 128,
                    "linear_conv_kernel_dim": 4,
                    "rope_parameters": {
                        "partial_rotary_factor": 0.25,
                        "rope_theta": 1000000.0
                    },
                    "moe_config": {
                        "num_experts": 128,
                        "num_experts_per_tok": 4,
                        "decoder_sparse_step": 2,
                        "norm_topk_prob": false,
                        "mlp_only_layers": [1]
                    }
                },
                "quantization": {
                    "bits": 4,
                    "group_size": 64
                }
            }"#,
        );

        let config = load_metal_config(dir.path()).unwrap();
        match config.arch {
            MetalModelArch::Qwen35(arch) => {
                let moe = arch.moe.expect("expected nested moe_config to be loaded");
                assert_eq!(moe.num_experts, 128);
                assert_eq!(moe.num_experts_per_tok, 4);
                assert_eq!(moe.decoder_sparse_step, 2);
                assert!(!moe.norm_topk_prob);
                assert_eq!(moe.mlp_only_layers, vec![1]);
                assert_eq!(moe.router_bits, 8);
                assert_eq!(moe.expert_bits, 4);
                assert!(!moe.is_moe_layer(0));
                assert!(!moe.is_moe_layer(1));
                assert!(!moe.is_moe_layer(2));
                assert!(moe.is_moe_layer(3));
            }
            MetalModelArch::Qwen3 => panic!("expected Qwen3.6 config, got Qwen3"),
        }
    }

    #[test]
    fn resolves_stop_tokens_following_hf_precedence() {
        use super::resolve_stop_token_ids;
        use serde_json::Value;

        // generation_config wins over config.json; both root and text_config
        // contributions get merged (dedup, generation_config first).
        let dir = tempdir().unwrap();
        fs::write(
            dir.path().join("generation_config.json"),
            r#"{"eos_token_id": [9001, 9000]}"#,
        )
        .unwrap();

        let cfg: Value = serde_json::from_str(
            r#"{"eos_token_id": [9000, 9002], "text_config": {"eos_token_id": 9000}}"#,
        )
        .unwrap();
        let root = cfg.as_object().unwrap();
        let text_config = root.get("text_config").and_then(Value::as_object).unwrap();

        let ids = resolve_stop_token_ids(dir.path(), root, text_config).unwrap();
        assert_eq!(ids, vec![9001, 9000, 9002]);
    }

    #[test]
    fn resolves_stop_tokens_handles_scalar_and_missing_generation_config() {
        use super::resolve_stop_token_ids;
        use serde_json::Value;

        // No generation_config.json, scalar eos in root, no text_config eos.
        let dir = tempdir().unwrap();
        let cfg: Value = serde_json::from_str(r#"{"eos_token_id": 7}"#).unwrap();
        let root = cfg.as_object().unwrap();
        let ids = resolve_stop_token_ids(dir.path(), root, root).unwrap();
        assert_eq!(ids, vec![7]);
    }

    #[test]
    fn resolves_stop_tokens_falls_back_when_nothing_specified() {
        use super::resolve_stop_token_ids;
        use serde_json::Value;

        let dir = tempdir().unwrap();
        let cfg: Value = serde_json::from_str("{}").unwrap();
        let root = cfg.as_object().unwrap();
        let ids = resolve_stop_token_ids(dir.path(), root, root).unwrap();
        assert_eq!(ids, vec![151_645]);
    }
}
