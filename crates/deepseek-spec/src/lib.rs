use thiserror::Error;

pub mod v4;

pub use v4::{
    DeepSeekV4AttentionLayerPlan, DeepSeekV4AttentionMode, DeepSeekV4AttentionOperatorSummary,
    DeepSeekV4AttentionTensorNames, DeepSeekV4CompressorShape, DeepSeekV4CompressorTensorNames,
    DeepSeekV4Config, DeepSeekV4ExpertTensorNames, DeepSeekV4HyperConnectionTensorNames,
    DeepSeekV4IndexerShape, DeepSeekV4IndexerTensorNames, DeepSeekV4LayerTensorNames,
    DeepSeekV4MoeRoute, DeepSeekV4MoeRoutingKind, DeepSeekV4MoeTensorNames,
    DeepSeekV4MtpTensorNames, DeepSeekV4OutputProjectionShape, DeepSeekV4RopeParameters,
    DeepSeekV4TensorNames,
};

#[derive(Debug, Error)]
pub enum DeepSeekConfigError {
    #[error("invalid deepseek config: {0}")]
    InvalidConfig(&'static str),
    #[error("invalid deepseek V4 forward batch: {0}")]
    InvalidForwardBatch(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, DeepSeekConfigError>;

/// How a DeepSeek V4 tensor should be partitioned across distributed ranks.
///
/// `dim` follows the HF safetensors `nn.Linear` layout: dim 0 is output
/// features, dim 1 is input features. `ExpertParallel` means the expert axis is
/// owned by EP/MoE-EP placement rather than tensor-parallel slicing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shard {
    Replicated,
    Column { dim: usize },
    Row { dim: usize },
    MergedColumn { dim: usize },
    VocabParallel { dim: usize },
    ExpertParallel { dim: usize },
}
