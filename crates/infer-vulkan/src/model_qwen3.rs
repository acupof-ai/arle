//! Dense Qwen3 Vulkan forward order.
//!
//! Authoritative source: `crates/infer-cuda/src/model.rs` eager
//! `forward_tokens`. P2 encodes the exact operator order and buffer families;
//! numeric execution waits for the Vulkan shader ABI and Qwen3 GGUF/safetensor
//! residency loader.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen3Op {
    Embedding,
    InputRmsNorm,
    QProj,
    KProj,
    VProj,
    QNorm,
    KNorm,
    Rope,
    Attention,
    OProj,
    AttentionResidual,
    PostAttentionRmsNorm,
    GateProj,
    UpProj,
    SwiGlu,
    DownProj,
    MlpResidual,
    FinalRmsNorm,
    LmHead,
}

pub const DENSE_QWEN3_LAYER_OPS: &[Qwen3Op] = &[
    Qwen3Op::InputRmsNorm,
    Qwen3Op::QProj,
    Qwen3Op::KProj,
    Qwen3Op::VProj,
    Qwen3Op::QNorm,
    Qwen3Op::KNorm,
    Qwen3Op::Rope,
    Qwen3Op::Attention,
    Qwen3Op::OProj,
    Qwen3Op::AttentionResidual,
    Qwen3Op::PostAttentionRmsNorm,
    Qwen3Op::GateProj,
    Qwen3Op::UpProj,
    Qwen3Op::SwiGlu,
    Qwen3Op::DownProj,
    Qwen3Op::MlpResidual,
];

pub fn dense_qwen3_forward_ops(num_layers: usize) -> Vec<Qwen3Op> {
    std::iter::once(Qwen3Op::Embedding)
        .chain((0..num_layers).flat_map(|_| DENSE_QWEN3_LAYER_OPS.iter().copied()))
        .chain([Qwen3Op::FinalRmsNorm, Qwen3Op::LmHead])
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen3ResidencyPlan {
    pub num_layers: usize,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub head_dim: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
}

impl Qwen3ResidencyPlan {
    pub fn from_config(config: &qwen3_spec::Qwen3Config) -> Self {
        Self {
            num_layers: config.num_hidden_layers,
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            head_dim: config.head_dim,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
        }
    }
}

#[cfg(feature = "vulkan")]
pub struct VulkanQwen3Model {
    pub config: qwen3_spec::Qwen3Config,
    _ctx: vulkan_sys::VulkanContext,
}

#[cfg(feature = "vulkan")]
impl VulkanQwen3Model {
    pub fn new(config: qwen3_spec::Qwen3Config) -> anyhow::Result<Self> {
        let ctx = vulkan_sys::VulkanContext::create()
            .map_err(|e| anyhow::anyhow!("create Vulkan context: {e}"))?;
        Ok(Self { config, _ctx: ctx })
    }

    pub fn forward_token(
        &mut self,
        _slot: usize,
        _epoch: u64,
        _token: u32,
        _start_pos: usize,
    ) -> anyhow::Result<Vec<f32>> {
        anyhow::bail!(
            "Vulkan Qwen3 numeric forward is pending shader specialization and residency upload"
        )
    }
}
