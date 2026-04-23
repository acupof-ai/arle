//! Model architecture registry — pure Rust, no GPU dependency.
//!
//! Reads `config.json` from a model directory and maps the `architectures`
//! field to a known [`ModelArch`] variant.  Provides metadata (family,
//! attention variant) useful for routing to the correct CUDA implementation.
//!
//! # Supported architectures
//!
//! | `architectures` value (config.json)   | `ModelArch`              |
//! |---------------------------------------|--------------------------|
//! | `Qwen2ForCausalLM`                    | `Qwen3`                  |
//! | `Qwen2_5_VLForCausalLM` / Qwen text wrapper | `Qwen35`            |
//! | `LlamaForCausalLM`                    | `Llama`                  |
//! | `MistralForCausalLM`                  | `Mistral`                |
//! | `MixtralForCausalLM`                  | `Mixtral`                |
//! | `DeepseekV2ForCausalLM`               | `DeepSeekV2`             |
//! | `DeepseekV3ForCausalLM`               | `DeepSeekV3`             |
//! | `GemmaForCausalLM` / `Gemma2ForCausalLM` | `Gemma`               |
//! | `PhiForCausalLM` / `Phi3ForCausalLM`  | `Phi`                    |

use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;

use anyhow::{Context, Result, bail};

// ============================================================================
// ModelArch enum
// ============================================================================

/// Known model architectures.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ModelArch {
    Qwen3,
    Qwen35,
    /// Qwen3.5 Mixture-of-Experts variant (Qwen3.6-35B-A3B and friends).
    /// Shares Qwen3.5's hybrid linear+full attention; only the MLP block
    /// changes to a SparseMoeBlock + shared expert.
    ///
    /// Name intentionally mirrors the HuggingFace `Qwen3_5MoeForCausalLM`
    /// architecture string to make the mapping grep-visible.
    #[allow(non_camel_case_types)]
    Qwen3_5_Moe,
    Llama,
    Mistral,
    Mixtral,
    DeepSeekV2,
    DeepSeekV3,
    Gemma,
    Phi,
}

impl ModelArch {
    /// Human-readable display name.
    pub fn display_name(self) -> &'static str {
        match self {
            Self::Qwen3 => "Qwen3",
            Self::Qwen35 => "Qwen3.5",
            Self::Qwen3_5_Moe => "Qwen3.5-MoE",
            Self::Llama => "Llama",
            Self::Mistral => "Mistral",
            Self::Mixtral => "Mixtral",
            Self::DeepSeekV2 => "DeepSeek-V2",
            Self::DeepSeekV3 => "DeepSeek-V3",
            Self::Gemma => "Gemma",
            Self::Phi => "Phi",
        }
    }

    /// Attention variant used by this architecture.
    pub fn attention_variant(self) -> AttentionVariant {
        match self {
            Self::Qwen35 | Self::Qwen3_5_Moe => AttentionVariant::HybridGqa,
            Self::DeepSeekV2 | Self::DeepSeekV3 => AttentionVariant::Mla,
            Self::Gemma => AttentionVariant::Mha,
            Self::Qwen3 | Self::Llama | Self::Mistral | Self::Mixtral | Self::Phi => {
                AttentionVariant::Gqa
            }
        }
    }

    /// Whether an implementation is available in this build.
    ///
    /// `Qwen3_5_Moe` is Metal-only for now; the CUDA path is a `todo!` stub
    /// until the CUDA MoE kernel lands (see `docs/plans/qwen36-moe-metal.md`).
    pub fn is_implemented(self) -> bool {
        match self {
            Self::Qwen3 | Self::Qwen35 => true,
            Self::Qwen3_5_Moe => cfg!(feature = "metal"),
            Self::Llama
            | Self::Mistral
            | Self::Mixtral
            | Self::DeepSeekV2
            | Self::DeepSeekV3
            | Self::Gemma
            | Self::Phi => false,
        }
    }
}

impl std::fmt::Display for ModelArch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.display_name())
    }
}

// ============================================================================
// AttentionVariant
// ============================================================================

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttentionVariant {
    /// Multi-head attention (standard).
    Mha,
    /// Grouped-query attention (GQA / MQA).
    Gqa,
    /// Multi-head latent attention (DeepSeek MLA).
    Mla,
    /// Hybrid: alternates linear recurrent layers with full attention (Qwen3.5).
    HybridGqa,
}

// ============================================================================
// Static registry
// ============================================================================

/// Maps the `architectures` string from `config.json` to a `ModelArch`.
fn architecture_map() -> &'static HashMap<&'static str, ModelArch> {
    static MAP: OnceLock<HashMap<&'static str, ModelArch>> = OnceLock::new();
    MAP.get_or_init(|| {
        let mut m = HashMap::new();
        // Qwen
        m.insert("Qwen2ForCausalLM", ModelArch::Qwen3);
        m.insert("Qwen3ForCausalLM", ModelArch::Qwen3);
        // Qwen3.5 often appears as a Qwen text wrapper; keep direct keys for
        // checkpoints that already declare the dedicated architecture string.
        m.insert("Qwen2_5_VLForCausalLM", ModelArch::Qwen35);
        m.insert("Qwen3_5ForCausalLM", ModelArch::Qwen35);
        m.insert("Qwen3_5ForConditionalGeneration", ModelArch::Qwen35);
        // Qwen3.5 / Qwen3.6 Mixture-of-Experts variants.
        m.insert("Qwen3_5MoeForCausalLM", ModelArch::Qwen3_5_Moe);
        m.insert("Qwen3_5MoeForConditionalGeneration", ModelArch::Qwen3_5_Moe);
        // Llama
        m.insert("LlamaForCausalLM", ModelArch::Llama);
        m.insert("Llama3ForCausalLM", ModelArch::Llama);
        m.insert("MistralForCausalLM", ModelArch::Mistral);
        m.insert("MixtralForCausalLM", ModelArch::Mixtral);
        // DeepSeek
        m.insert("DeepseekV2ForCausalLM", ModelArch::DeepSeekV2);
        m.insert("DeepseekV3ForCausalLM", ModelArch::DeepSeekV3);
        // Gemma
        m.insert("GemmaForCausalLM", ModelArch::Gemma);
        m.insert("Gemma2ForCausalLM", ModelArch::Gemma);
        m.insert("Gemma3ForCausalLM", ModelArch::Gemma);
        m.insert("Gemma3ForConditionalGeneration", ModelArch::Gemma);
        m.insert("Gemma4ForCausalLM", ModelArch::Gemma);
        m.insert("Gemma4ForConditionalGeneration", ModelArch::Gemma);
        // Phi
        m.insert("PhiForCausalLM", ModelArch::Phi);
        m.insert("Phi3ForCausalLM", ModelArch::Phi);
        m.insert("Phi3SmallForCausalLM", ModelArch::Phi);
        m
    })
}

// ============================================================================
// Public API
// ============================================================================

/// Detect the model architecture by reading `<model_path>/config.json`.
///
/// Uses the `architectures` array first; only falls back to a Qwen3.5-specific
/// heuristic when the config shape is otherwise ambiguous.
pub fn detect_arch(model_path: &str) -> Result<ModelArch> {
    let config_path = Path::new(model_path).join("config.json");
    let content = std::fs::read_to_string(&config_path)
        .with_context(|| format!("reading {}", config_path.display()))?;
    detect_arch_from_json(&content)
}

/// Detect architecture from a `config.json` string (testable without disk I/O).
pub fn detect_arch_from_json(json_str: &str) -> Result<ModelArch> {
    let v: serde_json::Value = serde_json::from_str(json_str).context("parsing config.json")?;
    let has_text_config = v.get("text_config").is_some();

    // Qwen3.5-family checkpoints occasionally advertise a generic arch string
    // but expose MoE fields under `text_config.num_experts`. Treat any such
    // checkpoint as Qwen3_5_Moe regardless of what the arch string promises.
    let has_moe_experts = v
        .get("text_config")
        .and_then(|tc| tc.get("num_experts"))
        .and_then(serde_json::Value::as_u64)
        .is_some_and(|n| n > 0)
        || v.get("num_experts")
            .and_then(serde_json::Value::as_u64)
            .is_some_and(|n| n > 0);

    // Primary: `architectures` array.
    if let Some(archs) = v.get("architectures").and_then(|a| a.as_array()) {
        let map = architecture_map();
        for arch_val in archs {
            if let Some(arch_str) = arch_val.as_str() {
                if let Some(&arch) = map.get(arch_str) {
                    // Qwen3.5 checkpoints are frequently wrapped as top-level
                    // `Qwen2ForCausalLM` plus a nested `text_config`.
                    if arch == ModelArch::Qwen3 && has_text_config {
                        if has_moe_experts {
                            return Ok(ModelArch::Qwen3_5_Moe);
                        }
                        return Ok(ModelArch::Qwen35);
                    }
                    // Promote Qwen3.5 → Qwen3_5_Moe if the config actually
                    // carries expert fields.
                    if arch == ModelArch::Qwen35 && has_moe_experts {
                        return Ok(ModelArch::Qwen3_5_Moe);
                    }
                    return Ok(arch);
                }
            }
        }
        // Found architectures array but none matched.
        let names: Vec<&str> = archs.iter().filter_map(|v| v.as_str()).collect();
        bail!("unknown architectures: {:?}", names);
    }

    // Legacy fallback for configs without `architectures`: only treat them as
    // Qwen3.5 if the embedded text config exposes Qwen3.5-specific layer types.
    let has_qwen35_layer_types = v
        .get("text_config")
        .and_then(|tc| tc.get("layer_types"))
        .or_else(|| v.get("layer_types"))
        .and_then(|lt| lt.as_array())
        .is_some_and(|arr| !arr.is_empty());
    if has_qwen35_layer_types {
        return Ok(ModelArch::Qwen35);
    }

    bail!("config.json has no `architectures` field")
}

/// Return a human-readable summary for a model directory.
pub fn model_info_string(model_path: &str) -> String {
    match detect_arch(model_path) {
        Ok(arch) => format!(
            "{} ({:?} attention, implemented={})",
            arch.display_name(),
            arch.attention_variant(),
            arch.is_implemented(),
        ),
        Err(e) => format!("unknown ({e})"),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen3_config() -> &'static str {
        r#"{"architectures":["Qwen2ForCausalLM"],"hidden_size":2048}"#
    }

    fn qwen35_config() -> &'static str {
        r#"{"architectures":["Qwen2ForCausalLM"],"text_config":{"hidden_size":2048,"layer_types":["full_attention","linear_attention"]}}"#
    }

    fn llama_config() -> &'static str {
        r#"{"architectures":["LlamaForCausalLM"],"hidden_size":4096}"#
    }

    fn deepseek_v3_config() -> &'static str {
        r#"{"architectures":["DeepseekV3ForCausalLM"],"hidden_size":7168}"#
    }

    fn gemma_config() -> &'static str {
        r#"{"architectures":["Gemma2ForCausalLM"],"hidden_size":3584}"#
    }

    fn gemma4_multimodal_config() -> &'static str {
        r#"{"architectures":["Gemma4ForConditionalGeneration"],"text_config":{"hidden_size":3584}}"#
    }

    fn phi_config() -> &'static str {
        r#"{"architectures":["Phi3ForCausalLM"],"hidden_size":3072}"#
    }

    fn qwen35_moe_explicit_arch_config() -> &'static str {
        r#"{"architectures":["Qwen3_5MoeForConditionalGeneration"],"text_config":{"hidden_size":2048,"num_experts":256}}"#
    }

    fn qwen35_moe_via_text_config_experts() -> &'static str {
        r#"{"architectures":["Qwen2ForCausalLM"],"text_config":{"hidden_size":2048,"num_experts":256,"layer_types":["full_attention","linear_attention"]}}"#
    }

    fn unknown_config() -> &'static str {
        r#"{"architectures":["SomeNewModelForCausalLM"]}"#
    }

    fn no_arch_config() -> &'static str {
        r#"{"hidden_size":2048}"#
    }

    #[test]
    fn detects_qwen3() {
        assert_eq!(
            detect_arch_from_json(qwen3_config()).unwrap(),
            ModelArch::Qwen3
        );
    }

    #[test]
    fn detects_qwen35_via_text_config() {
        assert_eq!(
            detect_arch_from_json(qwen35_config()).unwrap(),
            ModelArch::Qwen35
        );
    }

    #[test]
    fn detects_llama() {
        assert_eq!(
            detect_arch_from_json(llama_config()).unwrap(),
            ModelArch::Llama
        );
    }

    #[test]
    fn detects_deepseek_v3() {
        assert_eq!(
            detect_arch_from_json(deepseek_v3_config()).unwrap(),
            ModelArch::DeepSeekV3
        );
    }

    #[test]
    fn detects_gemma() {
        assert_eq!(
            detect_arch_from_json(gemma_config()).unwrap(),
            ModelArch::Gemma
        );
    }

    #[test]
    fn detects_gemma4_multimodal_without_misclassifying_as_qwen35() {
        assert_eq!(
            detect_arch_from_json(gemma4_multimodal_config()).unwrap(),
            ModelArch::Gemma
        );
    }

    #[test]
    fn detects_phi() {
        assert_eq!(detect_arch_from_json(phi_config()).unwrap(), ModelArch::Phi);
    }

    #[test]
    fn detects_qwen35_moe_via_explicit_arch() {
        assert_eq!(
            detect_arch_from_json(qwen35_moe_explicit_arch_config()).unwrap(),
            ModelArch::Qwen3_5_Moe
        );
    }

    #[test]
    fn detects_qwen35_moe_via_text_config_experts() {
        assert_eq!(
            detect_arch_from_json(qwen35_moe_via_text_config_experts()).unwrap(),
            ModelArch::Qwen3_5_Moe
        );
    }

    #[test]
    fn unknown_arch_returns_err() {
        assert!(detect_arch_from_json(unknown_config()).is_err());
    }

    #[test]
    fn no_arch_field_returns_err() {
        assert!(detect_arch_from_json(no_arch_config()).is_err());
    }

    #[test]
    fn implemented_models() {
        assert!(ModelArch::Qwen3.is_implemented());
        assert!(ModelArch::Qwen35.is_implemented());
    }

    #[test]
    fn llama_not_yet_implemented() {
        assert!(!ModelArch::Llama.is_implemented());
        assert!(!ModelArch::DeepSeekV3.is_implemented());
    }

    #[test]
    fn attention_variants_correct() {
        assert_eq!(
            ModelArch::DeepSeekV2.attention_variant(),
            AttentionVariant::Mla
        );
        assert_eq!(
            ModelArch::Qwen35.attention_variant(),
            AttentionVariant::HybridGqa
        );
        assert_eq!(
            ModelArch::Qwen3_5_Moe.attention_variant(),
            AttentionVariant::HybridGqa
        );
        assert_eq!(ModelArch::Gemma.attention_variant(), AttentionVariant::Mha);
        assert_eq!(ModelArch::Llama.attention_variant(), AttentionVariant::Gqa);
    }

    #[test]
    fn display_name_non_empty() {
        for arch in [
            ModelArch::Qwen3,
            ModelArch::Qwen35,
            ModelArch::Qwen3_5_Moe,
            ModelArch::Llama,
            ModelArch::Mistral,
            ModelArch::Mixtral,
            ModelArch::DeepSeekV2,
            ModelArch::DeepSeekV3,
            ModelArch::Gemma,
            ModelArch::Phi,
        ] {
            assert!(!arch.display_name().is_empty(), "arch={arch:?}");
        }
    }
}
