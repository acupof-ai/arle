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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35LauncherKind {
    TokenEmbedding,
    QuantizedGemv,
    RmsNorm,
    Rope,
    DepthwiseConv4,
    GatedDeltaRecurrent,
    FullAttentionHd256,
    SwiGlu,
    Add,
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

pub fn qwen35_launcher_kind(op: Qwen35Op) -> Qwen35LauncherKind {
    match op {
        Qwen35Op::TokenEmbedding => Qwen35LauncherKind::TokenEmbedding,
        Qwen35Op::LinearInProjQkv
        | Qwen35Op::LinearInProjZ
        | Qwen35Op::LinearInProjB
        | Qwen35Op::LinearInProjA
        | Qwen35Op::LinearOutProj
        | Qwen35Op::FullQProjWithGate
        | Qwen35Op::FullKProj
        | Qwen35Op::FullVProj
        | Qwen35Op::FullOProj
        | Qwen35Op::MlpGateProj
        | Qwen35Op::MlpUpProj
        | Qwen35Op::MlpDownProj
        | Qwen35Op::LmHead => Qwen35LauncherKind::QuantizedGemv,
        Qwen35Op::InputRmsNorm
        | Qwen35Op::GatedOutputRmsNorm
        | Qwen35Op::FullQNorm
        | Qwen35Op::FullKNorm
        | Qwen35Op::PostAttentionRmsNorm
        | Qwen35Op::FinalRmsNorm => Qwen35LauncherKind::RmsNorm,
        Qwen35Op::DepthwiseConv4 => Qwen35LauncherKind::DepthwiseConv4,
        Qwen35Op::GatedDeltaRecurrent => Qwen35LauncherKind::GatedDeltaRecurrent,
        Qwen35Op::FullRope => Qwen35LauncherKind::Rope,
        Qwen35Op::FullAttentionHd256 | Qwen35Op::FullAttentionGate => {
            Qwen35LauncherKind::FullAttentionHd256
        }
        Qwen35Op::SwiGlu => Qwen35LauncherKind::SwiGlu,
        Qwen35Op::AttentionResidual | Qwen35Op::MlpResidual => Qwen35LauncherKind::Add,
    }
}

pub fn qwen35_launcher_sequence(layer_types: &[qwen35_spec::LayerType]) -> Vec<Qwen35LauncherKind> {
    qwen35_forward_ops(layer_types)
        .into_iter()
        .map(qwen35_launcher_kind)
        .collect()
}

#[cfg(feature = "vulkan")]
pub fn qwen35_kernel_for_launcher(kind: Qwen35LauncherKind) -> vulkan_kernels::Kernel {
    match kind {
        Qwen35LauncherKind::TokenEmbedding => vulkan_kernels::Kernel::GetRows,
        Qwen35LauncherKind::QuantizedGemv => vulkan_kernels::Kernel::GemvQ4K,
        Qwen35LauncherKind::RmsNorm => vulkan_kernels::Kernel::RmsNorm,
        Qwen35LauncherKind::Rope => vulkan_kernels::Kernel::RopeNeox,
        Qwen35LauncherKind::DepthwiseConv4 => vulkan_kernels::Kernel::Qwen35SsmConv,
        Qwen35LauncherKind::GatedDeltaRecurrent => vulkan_kernels::Kernel::Qwen35GatedDeltaNet,
        Qwen35LauncherKind::FullAttentionHd256 => vulkan_kernels::Kernel::FlashAttn,
        Qwen35LauncherKind::SwiGlu => vulkan_kernels::Kernel::SwiGlu,
        Qwen35LauncherKind::Add => vulkan_kernels::Kernel::Add,
    }
}

#[cfg(feature = "vulkan")]
pub fn qwen35_forward_kernel_sequence(
    layer_types: &[qwen35_spec::LayerType],
) -> Vec<vulkan_kernels::Kernel> {
    qwen35_launcher_sequence(layer_types)
        .into_iter()
        .map(qwen35_kernel_for_launcher)
        .collect()
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
        let _ = qwen35_forward_kernel_sequence(&self.config.layer_types);
        anyhow::bail!(
            "Vulkan Qwen3.5 numeric forward requires GGUF/safetensor residency binding for the launcher sequence"
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

    #[test]
    fn qwen35_launcher_sequence_covers_linear_and_full_attention() {
        let layers = [
            LayerType::LinearAttention,
            LayerType::LinearAttention,
            LayerType::LinearAttention,
            LayerType::FullAttention,
        ];
        let kinds = qwen35_launcher_sequence(&layers);
        assert_eq!(kinds.first(), Some(&Qwen35LauncherKind::TokenEmbedding));
        assert!(kinds.contains(&Qwen35LauncherKind::DepthwiseConv4));
        assert!(kinds.contains(&Qwen35LauncherKind::GatedDeltaRecurrent));
        assert!(kinds.contains(&Qwen35LauncherKind::FullAttentionHd256));
        assert!(kinds.contains(&Qwen35LauncherKind::Rope));
        assert!(kinds.contains(&Qwen35LauncherKind::SwiGlu));
        assert_eq!(kinds.len(), qwen35_forward_ops(&layers).len());
    }

    #[test]
    fn qwen35_linear_state_shape_uses_conv_kernel_minus_one() {
        let cfg = qwen35_spec::Qwen35Config::from_json_str(
            r#"{
                "architectures": ["Qwen3MoeForCausalLM"],
                "model_type": "qwen3_moe",
                "vocab_size": 1000,
                "hidden_size": 512,
                "intermediate_size": 1024,
                "num_hidden_layers": 4,
                "num_attention_heads": 16,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "hidden_act": "silu",
                "rms_norm_eps": 1e-6,
                "max_position_embeddings": 32768,
                "rope_theta": 1000000.0,
                "tie_word_embeddings": false,
                "num_experts": 8,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 256,
                "shared_expert_intermediate_size": 512,
                "decoder_sparse_step": 1,
                "first_k_dense_replace": 0,
                "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 32,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4
            }"#,
        )
        .unwrap();
        let shape = linear_state_shape(&cfg);
        assert_eq!(shape.recurrent_f32, 32 * 128 * 128);
        assert_eq!(shape.conv_bf16, (2 * 16 * 128 + 32 * 128) * 3);
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn qwen35_fused_launcher_classes_map_to_kernels() {
        assert_eq!(
            qwen35_kernel_for_launcher(Qwen35LauncherKind::DepthwiseConv4),
            vulkan_kernels::Kernel::Qwen35SsmConv
        );
        assert_eq!(
            qwen35_kernel_for_launcher(Qwen35LauncherKind::GatedDeltaRecurrent),
            vulkan_kernels::Kernel::Qwen35GatedDeltaNet
        );
        assert_eq!(
            qwen35_kernel_for_launcher(Qwen35LauncherKind::FullAttentionHd256),
            vulkan_kernels::Kernel::FlashAttn
        );
    }
}
