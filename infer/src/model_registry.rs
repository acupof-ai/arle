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
//! | `Qwen2_5_VLForCausalLM` (text_config) | `Qwen35`                 |
//! | `LlamaForCausalLM`                    | `Llama`                  |
//! | `MistralForCausalLM`                  | `Mistral`                |
//! | `MixtralForCausalLM`                  | `Mixtral`                |
//! | `DeepseekV2ForCausalLM`               | `DeepSeekV2`             |
//! | `DeepseekV3ForCausalLM`               | `DeepSeekV3`             |
//! | `GemmaForCausalLM` / `Gemma2ForCausalLM` | `Gemma`               |
//! | `PhiForCausalLM` / `Phi3ForCausalLM`  | `Phi`                    |
//! | `ChatGLMModel` / `Glm4ForCausalLM`   | `GLM4`                   |

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
    Llama,
    Mistral,
    Mixtral,
    DeepSeekV2,
    DeepSeekV3,
    Gemma,
    Phi,
    GLM4,
}

impl ModelArch {
    /// Human-readable display name.
    pub fn display_name(self) -> &'static str {
        match self {
            Self::Qwen3 => "Qwen3",
            Self::Qwen35 => "Qwen3.5",
            Self::Llama => "Llama",
            Self::Mistral => "Mistral",
            Self::Mixtral => "Mixtral",
            Self::DeepSeekV2 => "DeepSeek-V2",
            Self::DeepSeekV3 => "DeepSeek-V3",
            Self::Gemma => "Gemma",
            Self::Phi => "Phi",
            Self::GLM4 => "GLM-4",
        }
    }

    /// Attention variant used by this architecture.
    pub fn attention_variant(self) -> AttentionVariant {
        match self {
            Self::Qwen35 => AttentionVariant::HybridGqa,
            Self::DeepSeekV2 | Self::DeepSeekV3 => AttentionVariant::Mla,
            Self::Gemma => AttentionVariant::Mha,
            Self::Qwen3 | Self::Llama | Self::Mistral | Self::Mixtral | Self::Phi | Self::GLM4 => {
                AttentionVariant::Gqa
            }
        }
    }

    /// Whether a CUDA implementation is available in this build.
    pub fn is_implemented(self) -> bool {
        matches!(self, Self::Qwen3 | Self::Qwen35 | Self::GLM4)
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
        // Qwen3.5 is identified by the presence of text_config (handled below),
        // but register a direct key too in case future checkpoints use it.
        m.insert("Qwen2_5_VLForCausalLM", ModelArch::Qwen35);
        m.insert("Qwen3_5ForCausalLM", ModelArch::Qwen35);
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
        // Phi
        m.insert("PhiForCausalLM", ModelArch::Phi);
        m.insert("Phi3ForCausalLM", ModelArch::Phi);
        m.insert("Phi3SmallForCausalLM", ModelArch::Phi);
        // GLM-4
        m.insert("ChatGLMModel", ModelArch::GLM4);
        m.insert("Glm4ForCausalLM", ModelArch::GLM4);
        m
    })
}

// ============================================================================
// Public API
// ============================================================================

/// Detect the model architecture by reading `<model_path>/config.json`.
///
/// Uses the `architectures` array first; falls back to heuristics for
/// multi-modal configs that embed a `text_config` sub-object (Qwen3.5).
pub fn detect_arch(model_path: &str) -> Result<ModelArch> {
    let config_path = Path::new(model_path).join("config.json");
    let content = std::fs::read_to_string(&config_path)
        .with_context(|| format!("reading {}", config_path.display()))?;
    detect_arch_from_json(&content)
}

/// Detect architecture from a `config.json` string (testable without disk I/O).
pub fn detect_arch_from_json(json_str: &str) -> Result<ModelArch> {
    let v: serde_json::Value = serde_json::from_str(json_str).context("parsing config.json")?;

    // Heuristic: Qwen3.5 embeds a `text_config` block at the top level.
    if v.get("text_config").is_some() {
        return Ok(ModelArch::Qwen35);
    }

    // Primary: `architectures` array.
    if let Some(archs) = v.get("architectures").and_then(|a| a.as_array()) {
        let map = architecture_map();
        for arch_val in archs {
            if let Some(arch_str) = arch_val.as_str() {
                if let Some(&arch) = map.get(arch_str) {
                    return Ok(arch);
                }
            }
        }
        // Found architectures array but none matched.
        let names: Vec<&str> = archs.iter().filter_map(|v| v.as_str()).collect();
        bail!("unknown architectures: {:?}", names);
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
        r#"{"architectures":["Qwen2ForCausalLM"],"text_config":{"hidden_size":2048}}"#
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

    fn phi_config() -> &'static str {
        r#"{"architectures":["Phi3ForCausalLM"],"hidden_size":3072}"#
    }

    fn glm4_config() -> &'static str {
        r#"{"architectures":["ChatGLMModel"],"hidden_size":4096}"#
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
    fn detects_phi() {
        assert_eq!(detect_arch_from_json(phi_config()).unwrap(), ModelArch::Phi);
    }

    #[test]
    fn detects_glm4() {
        assert_eq!(
            detect_arch_from_json(glm4_config()).unwrap(),
            ModelArch::GLM4
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
        assert!(ModelArch::GLM4.is_implemented());
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
        assert_eq!(ModelArch::Gemma.attention_variant(), AttentionVariant::Mha);
        assert_eq!(ModelArch::Llama.attention_variant(), AttentionVariant::Gqa);
        assert_eq!(ModelArch::GLM4.attention_variant(), AttentionVariant::Gqa);
    }

    #[test]
    fn display_name_non_empty() {
        for arch in [
            ModelArch::Qwen3,
            ModelArch::Qwen35,
            ModelArch::Llama,
            ModelArch::Mistral,
            ModelArch::Mixtral,
            ModelArch::DeepSeekV2,
            ModelArch::DeepSeekV3,
            ModelArch::Gemma,
            ModelArch::Phi,
            ModelArch::GLM4,
        ] {
            assert!(!arch.display_name().is_empty(), "arch={arch:?}");
        }
    }
}
