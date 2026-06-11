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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen36MoeLauncherKind {
    RouterGemv,
    HostTopK,
    QuantizedExpertGemv,
    SwiGlu,
    ExpertWeightedAdd,
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

pub fn qwen36_moe_launcher_kind(op: Qwen36MoeOp) -> Qwen36MoeLauncherKind {
    match op {
        Qwen36MoeOp::RouterGemv => Qwen36MoeLauncherKind::RouterGemv,
        Qwen36MoeOp::TopK | Qwen36MoeOp::NormalizeTopK => Qwen36MoeLauncherKind::HostTopK,
        Qwen36MoeOp::RoutedExpertGate
        | Qwen36MoeOp::RoutedExpertUp
        | Qwen36MoeOp::RoutedExpertDown
        | Qwen36MoeOp::SharedExpertGate
        | Qwen36MoeOp::SharedExpertUp
        | Qwen36MoeOp::SharedExpertDown => Qwen36MoeLauncherKind::QuantizedExpertGemv,
        Qwen36MoeOp::SwiGlu => Qwen36MoeLauncherKind::SwiGlu,
        Qwen36MoeOp::ExpertWeightedSum => Qwen36MoeLauncherKind::ExpertWeightedAdd,
    }
}

pub fn qwen36_moe_launcher_sequence() -> Vec<Qwen36MoeLauncherKind> {
    QWEN36_MOE_OPS
        .iter()
        .copied()
        .map(qwen36_moe_launcher_kind)
        .collect()
}

#[cfg(feature = "vulkan")]
pub fn qwen36_kernel_for_launcher(kind: Qwen36MoeLauncherKind) -> Option<vulkan_kernels::Kernel> {
    Some(match kind {
        Qwen36MoeLauncherKind::RouterGemv | Qwen36MoeLauncherKind::QuantizedExpertGemv => {
            vulkan_kernels::Kernel::GemvQ4K
        }
        Qwen36MoeLauncherKind::SwiGlu => vulkan_kernels::Kernel::SwiGlu,
        Qwen36MoeLauncherKind::ExpertWeightedAdd => vulkan_kernels::Kernel::Add,
        Qwen36MoeLauncherKind::HostTopK => return None,
    })
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Qwen36Route {
    pub expert: usize,
    pub weight: f32,
}

pub fn qwen36_topk_routes(logits: &[f32], top_k: usize) -> Vec<Qwen36Route> {
    if top_k == 0 || logits.is_empty() {
        return Vec::new();
    }
    let mut scored: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    scored.sort_by(|(a_idx, a), (b_idx, b)| b.total_cmp(a).then_with(|| a_idx.cmp(b_idx)));
    scored.truncate(top_k.min(scored.len()));
    let max = scored
        .iter()
        .map(|(_, score)| *score)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut denom = 0.0f32;
    for (_, score) in &mut scored {
        *score = (*score - max).exp();
        denom += *score;
    }
    scored
        .into_iter()
        .map(|(expert, score)| Qwen36Route {
            expert,
            weight: if denom > 0.0 { score / denom } else { 0.0 },
        })
        .collect()
}

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
        let _ = qwen36_moe_launcher_sequence();
        anyhow::bail!(
            "Vulkan Qwen3.6 numeric forward requires MoE weight residency binding for the launcher sequence"
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

    #[test]
    fn moe_launcher_sequence_marks_host_topk_and_device_experts() {
        let kinds = qwen36_moe_launcher_sequence();
        assert_eq!(kinds[0], Qwen36MoeLauncherKind::RouterGemv);
        assert!(kinds.contains(&Qwen36MoeLauncherKind::HostTopK));
        assert!(kinds.contains(&Qwen36MoeLauncherKind::QuantizedExpertGemv));
        assert!(kinds.contains(&Qwen36MoeLauncherKind::SwiGlu));
        assert_eq!(
            kinds.last(),
            Some(&Qwen36MoeLauncherKind::ExpertWeightedAdd)
        );
    }

    #[test]
    fn topk_routes_are_sorted_and_normalized() {
        let routes = qwen36_topk_routes(&[0.0, 3.0, 1.0, 3.0], 2);
        assert_eq!(routes.len(), 2);
        assert_eq!(routes[0].expert, 1);
        assert_eq!(routes[1].expert, 3);
        let sum: f32 = routes.iter().map(|r| r.weight).sum();
        assert!((sum - 1.0).abs() < 1.0e-6);
        assert!((routes[0].weight - 0.5).abs() < 1.0e-6);
        assert!((routes[1].weight - 0.5).abs() < 1.0e-6);
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn moe_launcher_classes_map_to_kernels() {
        assert_eq!(
            qwen36_kernel_for_launcher(Qwen36MoeLauncherKind::RouterGemv),
            Some(vulkan_kernels::Kernel::GemvQ4K)
        );
        assert_eq!(
            qwen36_kernel_for_launcher(Qwen36MoeLauncherKind::SwiGlu),
            Some(vulkan_kernels::Kernel::SwiGlu)
        );
        assert_eq!(
            qwen36_kernel_for_launcher(Qwen36MoeLauncherKind::ExpertWeightedAdd),
            Some(vulkan_kernels::Kernel::Add)
        );
        assert_eq!(
            qwen36_kernel_for_launcher(Qwen36MoeLauncherKind::HostTopK),
            None
        );
    }
}
