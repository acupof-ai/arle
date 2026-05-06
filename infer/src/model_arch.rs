//! Backend-neutral model architecture metadata.
//!
//! This module stays available in CUDA, Metal, and no-GPU builds. It describes
//! model shape only; backend-specific weights, buffers, and execution handles
//! belong in the backend/model modules that implement the contract.

use serde::Serialize;
use serde::ser::{SerializeStruct, Serializer};

use crate::model_registry::ModelArch;

/// Serializable snapshot of the architecture metadata shared by all backends.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelArchSummary {
    pub arch: ModelArch,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_kv_layers: usize,
    pub num_kv_heads: usize,
    pub num_q_heads: usize,
    pub head_dim: usize,
    pub kv_cache_bytes_per_token: usize,
}

impl ModelArchSummary {
    /// Stable label used for telemetry and JSON serialization.
    pub fn arch_label(&self) -> &'static str {
        self.arch.display_name()
    }
}

impl Serialize for ModelArchSummary {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("ModelArchSummary", 9)?;
        state.serialize_field("arch", self.arch_label())?;
        state.serialize_field("hidden_size", &self.hidden_size)?;
        state.serialize_field("vocab_size", &self.vocab_size)?;
        state.serialize_field("num_hidden_layers", &self.num_hidden_layers)?;
        state.serialize_field("num_kv_layers", &self.num_kv_layers)?;
        state.serialize_field("num_kv_heads", &self.num_kv_heads)?;
        state.serialize_field("num_q_heads", &self.num_q_heads)?;
        state.serialize_field("head_dim", &self.head_dim)?;
        state.serialize_field("kv_cache_bytes_per_token", &self.kv_cache_bytes_per_token)?;
        state.end()
    }
}

/// Backend-neutral architecture shape contract.
///
/// Implementors expose static model metadata needed by scheduler telemetry,
/// KV accounting, and cross-backend capability reporting. The trait must not
/// expose CUDA, Metal, or model-weight handle types.
pub trait ModelArchInfo {
    fn arch_kind(&self) -> ModelArch;
    fn hidden_size(&self) -> usize;
    fn vocab_size(&self) -> usize;
    fn num_hidden_layers(&self) -> usize;
    fn num_kv_layers(&self) -> usize;
    fn num_kv_heads(&self) -> usize;
    fn num_q_heads(&self) -> usize;
    fn head_dim(&self) -> usize;
    fn kv_cache_bytes_per_token(&self) -> usize;

    fn arch_summary(&self) -> ModelArchSummary {
        ModelArchSummary {
            arch: self.arch_kind(),
            hidden_size: self.hidden_size(),
            vocab_size: self.vocab_size(),
            num_hidden_layers: self.num_hidden_layers(),
            num_kv_layers: self.num_kv_layers(),
            num_kv_heads: self.num_kv_heads(),
            num_q_heads: self.num_q_heads(),
            head_dim: self.head_dim(),
            kv_cache_bytes_per_token: self.kv_cache_bytes_per_token(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy)]
    struct MockArch {
        arch: ModelArch,
        hidden_size: usize,
        vocab_size: usize,
        num_hidden_layers: usize,
        num_kv_layers: usize,
        num_kv_heads: usize,
        num_q_heads: usize,
        head_dim: usize,
        kv_cache_bytes_per_token: usize,
    }

    impl ModelArchInfo for MockArch {
        fn arch_kind(&self) -> ModelArch {
            self.arch
        }

        fn hidden_size(&self) -> usize {
            self.hidden_size
        }

        fn vocab_size(&self) -> usize {
            self.vocab_size
        }

        fn num_hidden_layers(&self) -> usize {
            self.num_hidden_layers
        }

        fn num_kv_layers(&self) -> usize {
            self.num_kv_layers
        }

        fn num_kv_heads(&self) -> usize {
            self.num_kv_heads
        }

        fn num_q_heads(&self) -> usize {
            self.num_q_heads
        }

        fn head_dim(&self) -> usize {
            self.head_dim
        }

        fn kv_cache_bytes_per_token(&self) -> usize {
            self.kv_cache_bytes_per_token
        }
    }

    fn qwen3_shape() -> MockArch {
        MockArch {
            arch: ModelArch::Qwen3,
            hidden_size: 2560,
            vocab_size: 151_936,
            num_hidden_layers: 36,
            num_kv_layers: 36,
            num_kv_heads: 8,
            num_q_heads: 32,
            head_dim: 128,
            kv_cache_bytes_per_token: 36 * 8 * 128 * 2 * 2,
        }
    }

    #[test]
    fn arch_summary_copies_backend_neutral_shape() {
        let summary = qwen3_shape().arch_summary();

        assert_eq!(
            summary,
            ModelArchSummary {
                arch: ModelArch::Qwen3,
                hidden_size: 2560,
                vocab_size: 151_936,
                num_hidden_layers: 36,
                num_kv_layers: 36,
                num_kv_heads: 8,
                num_q_heads: 32,
                head_dim: 128,
                kv_cache_bytes_per_token: 147_456,
            }
        );
    }

    #[test]
    fn arch_summary_serializes_arch_as_display_name() {
        let summary = MockArch {
            arch: ModelArch::Qwen35,
            hidden_size: 3072,
            vocab_size: 151_936,
            num_hidden_layers: 36,
            num_kv_layers: 18,
            num_kv_heads: 8,
            num_q_heads: 32,
            head_dim: 128,
            kv_cache_bytes_per_token: 73_728,
        }
        .arch_summary();

        let value = serde_json::to_value(&summary).expect("summary should serialize");
        assert_eq!(value["arch"].as_str(), Some("Qwen3.5"));
        assert_eq!(value["hidden_size"].as_u64(), Some(3072));
        assert_eq!(value["num_kv_layers"].as_u64(), Some(18));
        assert_eq!(value["kv_cache_bytes_per_token"].as_u64(), Some(73_728));
    }
}
