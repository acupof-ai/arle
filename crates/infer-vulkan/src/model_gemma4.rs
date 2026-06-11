//! Gemma 4 Vulkan forward contract.
//!
//! Source facts checked against Hugging Face Transformers Gemma4 docs and the
//! Google model card on 2026-06-11: Gemma 4 text config uses alternating
//! sliding/global attention, PLE, QK norm, p-RoPE/global KV sharing fields,
//! and optional MoE fields. P6 pins the text forward contract; multimodal
//! encoders are out of this backend phase.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gemma4Op {
    TokenEmbedding,
    PerLayerEmbedding,
    PreAttentionRmsNorm,
    QProj,
    KProj,
    VProj,
    QNorm,
    KNorm,
    StandardRope,
    ProportionalRope,
    SlidingWindowAttention,
    GlobalAttention,
    OProj,
    AttentionResidual,
    PreMlpRmsNorm,
    GateProj,
    UpProj,
    GeGlu,
    DownProj,
    MlpResidual,
    Router,
    RoutedExperts,
    SharedExpert,
    FinalRmsNorm,
    LmHead,
}

pub const GEMMA4_RMS_NORM_ADDS_ONE: bool = true;

pub fn gemma4_forward_ops(config: &gemma_spec::Gemma4TextConfig) -> Vec<Gemma4Op> {
    let mut ops = Vec::new();
    ops.push(Gemma4Op::TokenEmbedding);
    for (layer_idx, layer_type) in config.layer_types.iter().enumerate() {
        if config.has_per_layer_embeddings() {
            ops.push(Gemma4Op::PerLayerEmbedding);
        }
        ops.extend_from_slice(&[
            Gemma4Op::PreAttentionRmsNorm,
            Gemma4Op::QProj,
            Gemma4Op::KProj,
            Gemma4Op::VProj,
            Gemma4Op::QNorm,
            Gemma4Op::KNorm,
        ]);
        match layer_type {
            gemma_spec::Gemma4LayerType::SlidingAttention => {
                ops.push(Gemma4Op::StandardRope);
                ops.push(Gemma4Op::SlidingWindowAttention);
            }
            gemma_spec::Gemma4LayerType::FullAttention => {
                ops.push(Gemma4Op::ProportionalRope);
                ops.push(Gemma4Op::GlobalAttention);
            }
        }
        ops.extend_from_slice(&[
            Gemma4Op::OProj,
            Gemma4Op::AttentionResidual,
            Gemma4Op::PreMlpRmsNorm,
        ]);
        if config.enable_moe_block && layer_idx + 1 != config.num_hidden_layers {
            ops.extend_from_slice(&[
                Gemma4Op::Router,
                Gemma4Op::RoutedExperts,
                Gemma4Op::SharedExpert,
                Gemma4Op::MlpResidual,
            ]);
        } else {
            ops.extend_from_slice(&[
                Gemma4Op::GateProj,
                Gemma4Op::UpProj,
                Gemma4Op::GeGlu,
                Gemma4Op::DownProj,
                Gemma4Op::MlpResidual,
            ]);
        }
    }
    ops.push(Gemma4Op::FinalRmsNorm);
    ops.push(Gemma4Op::LmHead);
    ops
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gemma4KvShape {
    pub local_window: usize,
    pub local_kv_heads: usize,
    pub global_kv_heads: usize,
    pub local_head_dim: usize,
    pub global_head_dim: usize,
    pub shared_global_layers: usize,
}

impl Gemma4KvShape {
    pub fn from_config(config: &gemma_spec::Gemma4TextConfig) -> Self {
        Self {
            local_window: config.sliding_window,
            local_kv_heads: config.num_key_value_heads,
            global_kv_heads: config.global_kv_heads(),
            local_head_dim: config.head_dim,
            global_head_dim: config.global_attention_head_dim(),
            shared_global_layers: config.num_kv_shared_layers,
        }
    }
}

#[cfg(feature = "vulkan")]
pub struct VulkanGemma4Model {
    pub config: gemma_spec::Gemma4TextConfig,
}

#[cfg(feature = "vulkan")]
impl VulkanGemma4Model {
    pub fn forward_token(
        &mut self,
        _slot: usize,
        _epoch: u64,
        _token: u32,
        _start_pos: usize,
    ) -> anyhow::Result<Vec<f32>> {
        anyhow::bail!(
            "Vulkan Gemma4 numeric forward is blocked: p-RoPE/global-KV sharing and Gemma \
             RMSNorm(+1) kernels are not validated yet"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> gemma_spec::Gemma4TextConfig {
        gemma_spec::Gemma4TextConfig::from_json_str(
            r#"{
                "vocab_size": 262144,
                "hidden_size": 2304,
                "intermediate_size": 9216,
                "num_hidden_layers": 4,
                "num_attention_heads": 8,
                "num_key_value_heads": 4,
                "head_dim": 256,
                "hidden_activation": "gelu_pytorch_tanh",
                "max_position_embeddings": 131072,
                "initializer_range": 0.02,
                "rms_norm_eps": 1e-6,
                "sliding_window": 512,
                "layer_types": [
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "full_attention"
                ],
                "vocab_size_per_layer_input": 262144,
                "hidden_size_per_layer_input": 256,
                "num_global_key_value_heads": 1,
                "global_head_dim": 512,
                "attention_k_eq_v": true,
                "num_kv_shared_layers": 3
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn gemma4_ops_have_sliding_then_global_final() {
        let cfg = config();
        let ops = gemma4_forward_ops(&cfg);
        assert_eq!(ops.first(), Some(&Gemma4Op::TokenEmbedding));
        assert!(ops.contains(&Gemma4Op::PerLayerEmbedding));
        assert!(ops.contains(&Gemma4Op::QNorm));
        assert!(ops.contains(&Gemma4Op::KNorm));
        assert!(ops.contains(&Gemma4Op::SlidingWindowAttention));
        assert!(ops.contains(&Gemma4Op::GlobalAttention));
        assert!(ops.contains(&Gemma4Op::ProportionalRope));
        assert!(ops.contains(&Gemma4Op::GeGlu));
        assert_eq!(ops[ops.len() - 2], Gemma4Op::FinalRmsNorm);
        assert_eq!(ops[ops.len() - 1], Gemma4Op::LmHead);
        assert!(GEMMA4_RMS_NORM_ADDS_ONE);
    }

    #[test]
    fn kv_shape_separates_local_and_global_attention() {
        let shape = Gemma4KvShape::from_config(&config());
        assert_eq!(shape.local_window, 512);
        assert_eq!(shape.local_kv_heads, 4);
        assert_eq!(shape.global_kv_heads, 1);
        assert_eq!(shape.local_head_dim, 256);
        assert_eq!(shape.global_head_dim, 512);
        assert_eq!(shape.shared_global_layers, 3);
    }
}
