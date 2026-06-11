//! Qwen3.6 MoE Vulkan forward contract.
//!
//! Qwen3.6 shares the Qwen3.5 hybrid attention substrate and adds sparse MoE
//! layers. This module pins the routing/expert-mix sequence so the Vulkan lane
//! does not collapse MoE into the dense Qwen3.5 MLP path.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen36MoeOp {
    RouterGemv,
    TopK,
    NormalizeTopK,
    RoutedExpertGate,
    RoutedExpertUp,
    SwiGlu,
    RoutedExpertDown,
    SharedExpertGate,
    SharedExpertUp,
    SharedExpertDown,
    ExpertWeightedSum,
}

pub const QWEN36_MOE_OPS: &[Qwen36MoeOp] = &[
    Qwen36MoeOp::RouterGemv,
    Qwen36MoeOp::TopK,
    Qwen36MoeOp::NormalizeTopK,
    Qwen36MoeOp::RoutedExpertGate,
    Qwen36MoeOp::RoutedExpertUp,
    Qwen36MoeOp::SwiGlu,
    Qwen36MoeOp::RoutedExpertDown,
    Qwen36MoeOp::SharedExpertGate,
    Qwen36MoeOp::SharedExpertUp,
    Qwen36MoeOp::SharedExpertDown,
    Qwen36MoeOp::ExpertWeightedSum,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen36MoeShape {
    pub num_experts: usize,
    pub experts_per_token: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
}

impl Qwen36MoeShape {
    pub fn from_config(config: &qwen35_spec::Qwen35Config) -> Option<Self> {
        (config.num_experts > 0).then_some(Self {
            num_experts: config.num_experts,
            experts_per_token: config.num_experts_per_tok,
            moe_intermediate_size: config.moe_intermediate_size,
            shared_expert_intermediate_size: config.shared_expert_intermediate_size,
        })
    }
}

pub fn qwen36_sparse_layers(config: &qwen35_spec::Qwen35Config) -> Vec<usize> {
    (0..config.num_hidden_layers)
        .filter(|&layer_idx| config.is_moe_layer(layer_idx))
        .collect()
}

pub const QWEN36_MUTATED_MOE_BUFFERS: &[&str] = &[
    "scratch.router_logits",
    "scratch.topk_ids",
    "scratch.topk_weights",
    "scratch.expert_gate",
    "scratch.expert_up",
    "scratch.expert_act",
    "scratch.expert_down",
    "scratch.shared_expert",
    "scratch.expert_mix",
];

#[cfg(feature = "vulkan")]
pub struct VulkanQwen36Model {
    pub config: qwen35_spec::Qwen35Config,
}

#[cfg(feature = "vulkan")]
impl VulkanQwen36Model {
    pub fn forward_token(
        &mut self,
        _slot: usize,
        _epoch: u64,
        _token: u32,
        _start_pos: usize,
    ) -> anyhow::Result<Vec<f32>> {
        anyhow::bail!(
            "Vulkan Qwen3.6 numeric forward is blocked: MoE router/expert-mix kernels are not \
             integrated yet"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moe_ops_include_router_routed_and_shared_experts() {
        assert_eq!(QWEN36_MOE_OPS[0], Qwen36MoeOp::RouterGemv);
        assert!(QWEN36_MOE_OPS.contains(&Qwen36MoeOp::TopK));
        assert!(QWEN36_MOE_OPS.contains(&Qwen36MoeOp::RoutedExpertGate));
        assert!(QWEN36_MOE_OPS.contains(&Qwen36MoeOp::RoutedExpertUp));
        assert!(QWEN36_MOE_OPS.contains(&Qwen36MoeOp::RoutedExpertDown));
        assert!(QWEN36_MOE_OPS.contains(&Qwen36MoeOp::SharedExpertGate));
        assert_eq!(QWEN36_MOE_OPS.last(), Some(&Qwen36MoeOp::ExpertWeightedSum));
    }

    #[test]
    fn mutated_moe_scratch_is_enumerated() {
        assert_eq!(QWEN36_MUTATED_MOE_BUFFERS.len(), 9);
        assert!(QWEN36_MUTATED_MOE_BUFFERS.contains(&"scratch.router_logits"));
        assert!(QWEN36_MUTATED_MOE_BUFFERS.contains(&"scratch.expert_mix"));
    }
}
