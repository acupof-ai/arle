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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gemma4LauncherKind {
    TokenEmbedding,
    PerLayerEmbedding,
    GemmaRmsNorm,
    QuantizedGemv,
    QkNorm,
    RopeStandard,
    RopeProportional,
    SlidingWindowAttention,
    GlobalAttention,
    GeGlu,
    Add,
    MoeRouter,
    MoeExperts,
}

pub const GEMMA4_RMS_NORM_ADDS_ONE: bool = false;

pub fn gemma4_forward_ops(config: &gemma_spec::Gemma4TextConfig) -> Vec<Gemma4Op> {
    let mut ops = Vec::new();
    ops.push(Gemma4Op::TokenEmbedding);
    for layer_type in &config.layer_types {
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
        ops.extend_from_slice(&[
            Gemma4Op::GateProj,
            Gemma4Op::UpProj,
            Gemma4Op::GeGlu,
            Gemma4Op::DownProj,
        ]);
        if config.uses_moe_block() {
            ops.extend_from_slice(&[Gemma4Op::Router, Gemma4Op::RoutedExperts]);
        }
        ops.push(Gemma4Op::MlpResidual);
    }
    ops.push(Gemma4Op::FinalRmsNorm);
    ops.push(Gemma4Op::LmHead);
    ops
}

pub fn gemma4_launcher_kind(op: Gemma4Op) -> Gemma4LauncherKind {
    match op {
        Gemma4Op::TokenEmbedding => Gemma4LauncherKind::TokenEmbedding,
        Gemma4Op::PerLayerEmbedding => Gemma4LauncherKind::PerLayerEmbedding,
        Gemma4Op::PreAttentionRmsNorm | Gemma4Op::PreMlpRmsNorm | Gemma4Op::FinalRmsNorm => {
            Gemma4LauncherKind::GemmaRmsNorm
        }
        Gemma4Op::QProj
        | Gemma4Op::KProj
        | Gemma4Op::VProj
        | Gemma4Op::OProj
        | Gemma4Op::GateProj
        | Gemma4Op::UpProj
        | Gemma4Op::DownProj
        | Gemma4Op::LmHead => Gemma4LauncherKind::QuantizedGemv,
        Gemma4Op::QNorm | Gemma4Op::KNorm => Gemma4LauncherKind::QkNorm,
        Gemma4Op::StandardRope => Gemma4LauncherKind::RopeStandard,
        Gemma4Op::ProportionalRope => Gemma4LauncherKind::RopeProportional,
        Gemma4Op::SlidingWindowAttention => Gemma4LauncherKind::SlidingWindowAttention,
        Gemma4Op::GlobalAttention => Gemma4LauncherKind::GlobalAttention,
        Gemma4Op::GeGlu => Gemma4LauncherKind::GeGlu,
        Gemma4Op::AttentionResidual | Gemma4Op::MlpResidual => Gemma4LauncherKind::Add,
        Gemma4Op::Router => Gemma4LauncherKind::MoeRouter,
        Gemma4Op::RoutedExperts | Gemma4Op::SharedExpert => Gemma4LauncherKind::MoeExperts,
    }
}

pub fn gemma4_launcher_sequence(config: &gemma_spec::Gemma4TextConfig) -> Vec<Gemma4LauncherKind> {
    gemma4_forward_ops(config)
        .into_iter()
        .map(gemma4_launcher_kind)
        .collect()
}

pub fn gemma4_effective_norm_weight(raw_weight: f32) -> f32 {
    if GEMMA4_RMS_NORM_ADDS_ONE {
        raw_weight + 1.0
    } else {
        raw_weight
    }
}

#[cfg(feature = "vulkan")]
pub fn gemma4_kernel_for_launcher(kind: Gemma4LauncherKind) -> vulkan_kernels::Kernel {
    match kind {
        Gemma4LauncherKind::TokenEmbedding | Gemma4LauncherKind::PerLayerEmbedding => {
            vulkan_kernels::Kernel::GetRows
        }
        Gemma4LauncherKind::GemmaRmsNorm | Gemma4LauncherKind::QkNorm => {
            vulkan_kernels::Kernel::RmsNorm
        }
        Gemma4LauncherKind::QuantizedGemv
        | Gemma4LauncherKind::MoeRouter
        | Gemma4LauncherKind::MoeExperts => vulkan_kernels::Kernel::GemvQ4K,
        Gemma4LauncherKind::RopeStandard | Gemma4LauncherKind::RopeProportional => {
            vulkan_kernels::Kernel::RopeNeox
        }
        Gemma4LauncherKind::SlidingWindowAttention | Gemma4LauncherKind::GlobalAttention => {
            vulkan_kernels::Kernel::FlashAttn
        }
        Gemma4LauncherKind::GeGlu => vulkan_kernels::Kernel::GeGlu,
        Gemma4LauncherKind::Add => vulkan_kernels::Kernel::Add,
    }
}

#[cfg(feature = "vulkan")]
pub fn gemma4_attention_spec(
    config: &gemma_spec::Gemma4TextConfig,
    kind: Gemma4LauncherKind,
) -> Option<vulkan_kernels::FlashAttentionSpec> {
    match kind {
        Gemma4LauncherKind::SlidingWindowAttention => Some(
            vulkan_kernels::FlashAttentionSpec::f32_f16(config.head_dim as u32),
        ),
        Gemma4LauncherKind::GlobalAttention => Some(vulkan_kernels::FlashAttentionSpec::f32_f16(
            config.global_attention_head_dim() as u32,
        )),
        _ => None,
    }
}

#[cfg(feature = "vulkan")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gemma4KernelLaunch {
    pub kind: Gemma4LauncherKind,
    pub kernel: vulkan_kernels::Kernel,
    pub flash_attention_spec: Option<vulkan_kernels::FlashAttentionSpec>,
}

#[cfg(feature = "vulkan")]
pub fn gemma4_forward_launch_sequence(
    config: &gemma_spec::Gemma4TextConfig,
) -> Vec<Gemma4KernelLaunch> {
    gemma4_launcher_sequence(config)
        .into_iter()
        .map(|kind| Gemma4KernelLaunch {
            kind,
            kernel: gemma4_kernel_for_launcher(kind),
            flash_attention_spec: gemma4_attention_spec(config, kind),
        })
        .collect()
}

#[cfg(feature = "vulkan")]
pub fn gemma4_forward_kernel_sequence(
    config: &gemma_spec::Gemma4TextConfig,
) -> Vec<vulkan_kernels::Kernel> {
    gemma4_forward_launch_sequence(config)
        .into_iter()
        .map(|launch| launch.kernel)
        .collect()
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
        let _ = gemma4_forward_launch_sequence(&self.config);
        anyhow::bail!(
            "Vulkan Gemma4 numeric forward requires weight/KV residency binding for the launcher sequence"
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
        const {
            assert!(!GEMMA4_RMS_NORM_ADDS_ONE);
        }
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

    #[test]
    fn gemma4_launcher_sequence_tracks_attention_and_norm_variants() {
        let cfg = config();
        let kinds = gemma4_launcher_sequence(&cfg);
        assert_eq!(kinds.first(), Some(&Gemma4LauncherKind::TokenEmbedding));
        assert!(kinds.contains(&Gemma4LauncherKind::PerLayerEmbedding));
        assert!(kinds.contains(&Gemma4LauncherKind::GemmaRmsNorm));
        assert!(kinds.contains(&Gemma4LauncherKind::QkNorm));
        assert!(kinds.contains(&Gemma4LauncherKind::SlidingWindowAttention));
        assert!(kinds.contains(&Gemma4LauncherKind::GlobalAttention));
        assert!(kinds.contains(&Gemma4LauncherKind::RopeProportional));
        assert!(kinds.contains(&Gemma4LauncherKind::GeGlu));
        assert_eq!(kinds.len(), gemma4_forward_ops(&cfg).len());
    }

    #[test]
    fn gemma4_norm_weight_is_plain_scale() {
        assert_eq!(gemma4_effective_norm_weight(0.0), 0.0);
        assert_eq!(gemma4_effective_norm_weight(-0.25), -0.25);
        const {
            assert!(!GEMMA4_RMS_NORM_ADDS_ONE);
        }
    }

    #[test]
    fn gemma4_moe_layers_keep_dense_mlp_and_router() {
        let mut cfg = config();
        cfg.num_experts = Some(128);
        cfg.top_k_experts = Some(8);
        cfg.moe_intermediate_size = Some(512);
        assert!(cfg.uses_moe_block());

        let ops = gemma4_forward_ops(&cfg);
        assert!(ops.contains(&Gemma4Op::GateProj));
        assert!(ops.contains(&Gemma4Op::UpProj));
        assert!(ops.contains(&Gemma4Op::DownProj));
        assert!(ops.contains(&Gemma4Op::Router));
        assert!(ops.contains(&Gemma4Op::RoutedExperts));
        assert!(!ops.contains(&Gemma4Op::SharedExpert));
        assert_eq!(
            ops.iter().filter(|op| **op == Gemma4Op::Router).count(),
            cfg.num_hidden_layers
        );
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn gemma4_launcher_classes_map_to_kernels() {
        assert_eq!(
            gemma4_kernel_for_launcher(Gemma4LauncherKind::GeGlu),
            vulkan_kernels::Kernel::GeGlu
        );
        assert_eq!(
            gemma4_kernel_for_launcher(Gemma4LauncherKind::SlidingWindowAttention),
            vulkan_kernels::Kernel::FlashAttn
        );
        assert_eq!(
            gemma4_kernel_for_launcher(Gemma4LauncherKind::GemmaRmsNorm),
            vulkan_kernels::Kernel::RmsNorm
        );
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn gemma4_attention_spec_separates_local_and_global_head_dims() {
        let cfg = config();
        let sliding =
            gemma4_attention_spec(&cfg, Gemma4LauncherKind::SlidingWindowAttention).unwrap();
        let global = gemma4_attention_spec(&cfg, Gemma4LauncherKind::GlobalAttention).unwrap();
        assert_eq!(sliding.specialization_u32()[3], (3, 256));
        assert_eq!(sliding.specialization_u32()[4], (4, 256));
        assert_eq!(global.specialization_u32()[3], (3, 512));
        assert_eq!(global.specialization_u32()[4], (4, 512));
        assert_eq!(global.specialization_u32()[12], (12, 1));
        assert_eq!(global.specialization_u32()[14], (14, 2));
        assert!(gemma4_attention_spec(&cfg, Gemma4LauncherKind::GeGlu).is_none());
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn gemma4_forward_launch_sequence_preserves_attention_specs() {
        let cfg = config();
        let launches = gemma4_forward_launch_sequence(&cfg);
        assert_eq!(launches.len(), gemma4_forward_ops(&cfg).len());

        let sliding = launches
            .iter()
            .find(|launch| launch.kind == Gemma4LauncherKind::SlidingWindowAttention)
            .unwrap();
        assert_eq!(sliding.kernel, vulkan_kernels::Kernel::FlashAttn);
        let sliding_spec = sliding.flash_attention_spec.unwrap();
        assert_eq!(sliding_spec.specialization_u32()[3], (3, 256));
        assert_eq!(sliding_spec.specialization_u32()[4], (4, 256));

        let global = launches
            .iter()
            .find(|launch| launch.kind == Gemma4LauncherKind::GlobalAttention)
            .unwrap();
        assert_eq!(global.kernel, vulkan_kernels::Kernel::FlashAttn);
        let global_spec = global.flash_attention_spec.unwrap();
        assert_eq!(global_spec.specialization_u32()[3], (3, 512));
        assert_eq!(global_spec.specialization_u32()[4], (4, 512));

        let geglu = launches
            .iter()
            .find(|launch| launch.kind == Gemma4LauncherKind::GeGlu)
            .unwrap();
        assert_eq!(geglu.kernel, vulkan_kernels::Kernel::GeGlu);
        assert!(geglu.flash_attention_spec.is_none());
    }
}
