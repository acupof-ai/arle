//! Qwen3.5 hybrid Vulkan forward contract.
//!
//! Authoritative source: `crates/infer-cuda/src/qwen35.rs`. Qwen3.5
//! alternates gated-delta linear attention with periodic full attention
//! (production pattern is 3 linear layers then 1 full-attention layer).
//! P4 pins the order and recurrent state contract; numeric Vulkan kernels for
//! conv4 + recurrent gated delta are not present in the vendored GLSL corpus.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35Op {
    TokenEmbedding,
    InputRmsNorm,
    LinearInProjQkv,
    LinearInProjZ,
    LinearInProjB,
    LinearInProjA,
    DepthwiseConv4,
    GatedDeltaRecurrent,
    GatedOutputRmsNorm,
    LinearOutProj,
    FullQProjWithGate,
    FullKProj,
    FullVProj,
    FullQNorm,
    FullKNorm,
    FullRope,
    FullAttentionHd256,
    FullAttentionGate,
    FullOProj,
    AttentionResidual,
    PostAttentionRmsNorm,
    MlpGateProj,
    MlpUpProj,
    SwiGlu,
    MlpDownProj,
    MlpResidual,
    FinalRmsNorm,
    LmHead,
}

pub const QWEN35_LINEAR_LAYER_OPS: &[Qwen35Op] = &[
    Qwen35Op::InputRmsNorm,
    Qwen35Op::LinearInProjQkv,
    Qwen35Op::LinearInProjZ,
    Qwen35Op::LinearInProjB,
    Qwen35Op::LinearInProjA,
    Qwen35Op::DepthwiseConv4,
    Qwen35Op::GatedDeltaRecurrent,
    Qwen35Op::GatedOutputRmsNorm,
    Qwen35Op::LinearOutProj,
    Qwen35Op::AttentionResidual,
    Qwen35Op::PostAttentionRmsNorm,
    Qwen35Op::MlpGateProj,
    Qwen35Op::MlpUpProj,
    Qwen35Op::SwiGlu,
    Qwen35Op::MlpDownProj,
    Qwen35Op::MlpResidual,
];

pub const QWEN35_FULL_LAYER_OPS: &[Qwen35Op] = &[
    Qwen35Op::InputRmsNorm,
    Qwen35Op::FullQProjWithGate,
    Qwen35Op::FullKProj,
    Qwen35Op::FullVProj,
    Qwen35Op::FullQNorm,
    Qwen35Op::FullKNorm,
    Qwen35Op::FullRope,
    Qwen35Op::FullAttentionHd256,
    Qwen35Op::FullAttentionGate,
    Qwen35Op::FullOProj,
    Qwen35Op::AttentionResidual,
    Qwen35Op::PostAttentionRmsNorm,
    Qwen35Op::MlpGateProj,
    Qwen35Op::MlpUpProj,
    Qwen35Op::SwiGlu,
    Qwen35Op::MlpDownProj,
    Qwen35Op::MlpResidual,
];

pub fn qwen35_forward_ops(layer_types: &[qwen35_spec::LayerType]) -> Vec<Qwen35Op> {
    let mut ops = Vec::with_capacity(
        3 + layer_types.len()
            * QWEN35_FULL_LAYER_OPS
                .len()
                .max(QWEN35_LINEAR_LAYER_OPS.len()),
    );
    ops.push(Qwen35Op::TokenEmbedding);
    for layer_type in layer_types {
        match layer_type {
            qwen35_spec::LayerType::LinearAttention => {
                ops.extend_from_slice(QWEN35_LINEAR_LAYER_OPS);
            }
            qwen35_spec::LayerType::FullAttention => {
                ops.extend_from_slice(QWEN35_FULL_LAYER_OPS);
            }
        }
    }
    ops.push(Qwen35Op::FinalRmsNorm);
    ops.push(Qwen35Op::LmHead);
    ops
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen35HybridCounts {
    pub linear_layers: usize,
    pub full_layers: usize,
}

pub fn hybrid_counts(layer_types: &[qwen35_spec::LayerType]) -> Qwen35HybridCounts {
    let full_layers = layer_types
        .iter()
        .filter(|&&layer| layer == qwen35_spec::LayerType::FullAttention)
        .count();
    Qwen35HybridCounts {
        linear_layers: layer_types.len() - full_layers,
        full_layers,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen35LinearStateShape {
    pub recurrent_f32: usize,
    pub conv_bf16: usize,
}

pub fn linear_state_shape(config: &qwen35_spec::Qwen35Config) -> Qwen35LinearStateShape {
    Qwen35LinearStateShape {
        recurrent_f32: config.linear_num_value_heads
            * config.linear_key_head_dim
            * config.linear_value_head_dim,
        conv_bf16: (2 * config.linear_num_key_heads * config.linear_key_head_dim
            + config.linear_num_value_heads * config.linear_value_head_dim)
            * config.linear_conv_kernel_dim.saturating_sub(1),
    }
}

pub const QWEN35_MUTATED_RECURRENT_BUFFERS: &[&str] = &[
    "slot.linear.gated_delta_state_f32",
    "slot.linear.conv4_ring_bf16",
    "slot.full.k_cache_bf16",
    "slot.full.v_cache_bf16",
];

#[cfg(feature = "vulkan")]
pub struct VulkanQwen35Model {
    pub config: qwen35_spec::Qwen35Config,
}

#[cfg(feature = "vulkan")]
impl VulkanQwen35Model {
    pub fn forward_token(
        &mut self,
        _slot: usize,
        _epoch: u64,
        _token: u32,
        _start_pos: usize,
    ) -> anyhow::Result<Vec<f32>> {
        anyhow::bail!(
            "Vulkan Qwen3.5 numeric forward is blocked: conv4 and recurrent gated-delta \
             shaders are not ported yet"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qwen35_spec::LayerType;

    #[test]
    fn qwen35_three_linear_one_full_pattern_is_pinned() {
        let layers = [
            LayerType::LinearAttention,
            LayerType::LinearAttention,
            LayerType::LinearAttention,
            LayerType::FullAttention,
            LayerType::LinearAttention,
        ];
        let counts = hybrid_counts(&layers);
        assert_eq!(counts.linear_layers, 4);
        assert_eq!(counts.full_layers, 1);

        let ops = qwen35_forward_ops(&layers);
        assert_eq!(ops.first(), Some(&Qwen35Op::TokenEmbedding));
        assert_eq!(ops.last(), Some(&Qwen35Op::LmHead));
        assert!(ops.contains(&Qwen35Op::DepthwiseConv4));
        assert!(ops.contains(&Qwen35Op::GatedDeltaRecurrent));
        assert!(ops.contains(&Qwen35Op::FullAttentionHd256));
        assert!(ops.contains(&Qwen35Op::FullAttentionGate));
        assert!(ops.contains(&Qwen35Op::SwiGlu));
    }

    #[test]
    fn qwen35_recurrent_buffers_are_explicit() {
        assert_eq!(
            QWEN35_MUTATED_RECURRENT_BUFFERS,
            &[
                "slot.linear.gated_delta_state_f32",
                "slot.linear.conv4_ring_bf16",
                "slot.full.k_cache_bf16",
                "slot.full.v_cache_bf16"
            ]
        );
    }
}
