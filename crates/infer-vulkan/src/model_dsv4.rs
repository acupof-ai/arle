//! DSv4-Flash Vulkan forward contract.
//!
//! Authoritative source: `crates/infer-cuda/src/dsv4.rs` fallback decode path
//! plus DSv4 attention fallback launch order in `attention.rs`. This module
//! pins the order and state-mutation contract before numeric Vulkan kernels
//! land. FlashMLA, official DSA, DeepGEMM, and DeepEP remain datacenter-only.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dsv4Op {
    TokenEmbedding,
    MhcExpand,
    HcAttnFn,
    MhcParams,
    MhcPreRmsNormAttn,
    WqA,
    QNorm,
    WqB,
    WKv,
    KvNorm,
    PrepareQk,
    SlidingWindowAttention,
    CompressorUpdate,
    IndexerCompressorUpdate,
    IndexerQB,
    IndexerProj,
    CsaSelect,
    HybridAttention,
    WoA,
    WoB,
    MhcPostAttention,
    HcFfnFn,
    MhcPreRmsNormFfn,
    RouterGemv,
    HostSqrtSoftplusRouting,
    RoutedExpertGateUp,
    SwigluClamped,
    RoutedExpertDown,
    SharedExpert,
    ExpertMixResidual,
    MhcPostFfn,
    MhcHeadPre,
    FinalRmsNorm,
    LmHead,
}

pub const DSV4_PREFIX_CACHE_ENABLED: bool = false;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dsv4LauncherKind {
    TokenEmbedding,
    QuantizedGemv,
    RmsNorm,
    Dsv4PrepareQk,
    Dsv4SwaAttention,
    Dsv4CompressorUpdate,
    Dsv4CsaSelect,
    Dsv4HybridAttention,
    Dsv4Mhc,
    SwigluClamped,
    Add,
    HostRouting,
}

pub const DSV4_FALLBACK_LAYER_OPS: &[Dsv4Op] = &[
    Dsv4Op::HcAttnFn,
    Dsv4Op::MhcParams,
    Dsv4Op::MhcPreRmsNormAttn,
    Dsv4Op::WqA,
    Dsv4Op::QNorm,
    Dsv4Op::WqB,
    Dsv4Op::WKv,
    Dsv4Op::KvNorm,
    Dsv4Op::PrepareQk,
    Dsv4Op::SlidingWindowAttention,
    Dsv4Op::CompressorUpdate,
    Dsv4Op::IndexerCompressorUpdate,
    Dsv4Op::IndexerQB,
    Dsv4Op::IndexerProj,
    Dsv4Op::CsaSelect,
    Dsv4Op::HybridAttention,
    Dsv4Op::WoA,
    Dsv4Op::WoB,
    Dsv4Op::MhcPostAttention,
    Dsv4Op::HcFfnFn,
    Dsv4Op::MhcParams,
    Dsv4Op::MhcPreRmsNormFfn,
    Dsv4Op::RouterGemv,
    Dsv4Op::HostSqrtSoftplusRouting,
    Dsv4Op::RoutedExpertGateUp,
    Dsv4Op::SwigluClamped,
    Dsv4Op::RoutedExpertDown,
    Dsv4Op::SharedExpert,
    Dsv4Op::ExpertMixResidual,
    Dsv4Op::MhcPostFfn,
];

pub fn dsv4_fallback_forward_ops(num_layers: usize) -> Vec<Dsv4Op> {
    [Dsv4Op::TokenEmbedding, Dsv4Op::MhcExpand]
        .into_iter()
        .chain((0..num_layers).flat_map(|_| DSV4_FALLBACK_LAYER_OPS.iter().copied()))
        .chain([Dsv4Op::MhcHeadPre, Dsv4Op::FinalRmsNorm, Dsv4Op::LmHead])
        .collect()
}

pub fn dsv4_launcher_kind(op: Dsv4Op) -> Dsv4LauncherKind {
    match op {
        Dsv4Op::TokenEmbedding => Dsv4LauncherKind::TokenEmbedding,
        Dsv4Op::HcAttnFn
        | Dsv4Op::WqA
        | Dsv4Op::WqB
        | Dsv4Op::WKv
        | Dsv4Op::IndexerQB
        | Dsv4Op::IndexerProj
        | Dsv4Op::WoA
        | Dsv4Op::WoB
        | Dsv4Op::HcFfnFn
        | Dsv4Op::RouterGemv
        | Dsv4Op::RoutedExpertGateUp
        | Dsv4Op::RoutedExpertDown
        | Dsv4Op::SharedExpert
        | Dsv4Op::LmHead => Dsv4LauncherKind::QuantizedGemv,
        Dsv4Op::QNorm | Dsv4Op::KvNorm | Dsv4Op::FinalRmsNorm => Dsv4LauncherKind::RmsNorm,
        Dsv4Op::PrepareQk => Dsv4LauncherKind::Dsv4PrepareQk,
        Dsv4Op::SlidingWindowAttention => Dsv4LauncherKind::Dsv4SwaAttention,
        Dsv4Op::CompressorUpdate | Dsv4Op::IndexerCompressorUpdate => {
            Dsv4LauncherKind::Dsv4CompressorUpdate
        }
        Dsv4Op::CsaSelect => Dsv4LauncherKind::Dsv4CsaSelect,
        Dsv4Op::HybridAttention => Dsv4LauncherKind::Dsv4HybridAttention,
        Dsv4Op::MhcExpand
        | Dsv4Op::MhcParams
        | Dsv4Op::MhcPreRmsNormAttn
        | Dsv4Op::MhcPostAttention
        | Dsv4Op::MhcPreRmsNormFfn
        | Dsv4Op::MhcPostFfn
        | Dsv4Op::MhcHeadPre => Dsv4LauncherKind::Dsv4Mhc,
        Dsv4Op::SwigluClamped => Dsv4LauncherKind::SwigluClamped,
        Dsv4Op::ExpertMixResidual => Dsv4LauncherKind::Add,
        Dsv4Op::HostSqrtSoftplusRouting => Dsv4LauncherKind::HostRouting,
    }
}

pub fn dsv4_launcher_sequence(num_layers: usize) -> Vec<Dsv4LauncherKind> {
    dsv4_fallback_forward_ops(num_layers)
        .into_iter()
        .map(dsv4_launcher_kind)
        .collect()
}

#[cfg(feature = "vulkan")]
pub fn dsv4_kernel_for_launcher(kind: Dsv4LauncherKind) -> Option<vulkan_kernels::Kernel> {
    Some(match kind {
        Dsv4LauncherKind::TokenEmbedding => vulkan_kernels::Kernel::GetRows,
        Dsv4LauncherKind::QuantizedGemv => vulkan_kernels::Kernel::GemvQ4K,
        Dsv4LauncherKind::RmsNorm => vulkan_kernels::Kernel::RmsNorm,
        Dsv4LauncherKind::Dsv4PrepareQk => vulkan_kernels::Kernel::Dsv4PrepareQk,
        Dsv4LauncherKind::Dsv4SwaAttention => vulkan_kernels::Kernel::Dsv4SwaAttention,
        Dsv4LauncherKind::Dsv4CompressorUpdate => vulkan_kernels::Kernel::Dsv4CompressorUpdate,
        Dsv4LauncherKind::Dsv4CsaSelect => vulkan_kernels::Kernel::Dsv4CsaSelect,
        Dsv4LauncherKind::Dsv4HybridAttention => vulkan_kernels::Kernel::Dsv4HybridAttention,
        Dsv4LauncherKind::Dsv4Mhc => vulkan_kernels::Kernel::Dsv4Mhc,
        Dsv4LauncherKind::SwigluClamped => vulkan_kernels::Kernel::SwigluClamped,
        Dsv4LauncherKind::Add => vulkan_kernels::Kernel::Add,
        Dsv4LauncherKind::HostRouting => return None,
    })
}

#[cfg(feature = "vulkan")]
pub fn dsv4_forward_kernel_sequence(num_layers: usize) -> Vec<Option<vulkan_kernels::Kernel>> {
    dsv4_launcher_sequence(num_layers)
        .into_iter()
        .map(dsv4_kernel_for_launcher)
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dsv4LayerRope {
    pub layer_idx: usize,
    pub compress_ratio: usize,
    pub rope_theta: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LauncherWrites {
    pub launcher: &'static str,
    pub writes: &'static [&'static str],
}

pub const DSV4_MUTATED_SLOT_BUFFERS: &[&str] = &[
    "slot.sw_window_ring",
    "slot.compressor.pending_kv",
    "slot.compressor.pending_score",
    "slot.compressor.prev_overlap_kv",
    "slot.compressor.prev_overlap_score",
    "slot.compressor.compressed",
    "slot.indexer.pending_kv",
    "slot.indexer.pending_score",
    "slot.indexer.prev_overlap_kv",
    "slot.indexer.prev_overlap_score",
    "slot.indexer.compressed",
];

pub const DSV4_ATTENTION_LAUNCHER_WRITES: &[LauncherWrites] = &[
    LauncherWrites {
        launcher: "dsv4_prepare_qk",
        writes: &["scratch.q_prepared", "scratch.k_prepared"],
    },
    LauncherWrites {
        launcher: "dsv4_swa_attention(write_window_cache=1)",
        writes: &["scratch.attn_local", "slot.sw_window_ring"],
    },
    LauncherWrites {
        launcher: "dsv4_compressor_update(compressor)",
        writes: &[
            "slot.compressor.pending_kv",
            "slot.compressor.pending_score",
            "slot.compressor.prev_overlap_kv",
            "slot.compressor.prev_overlap_score",
            "slot.compressor.compressed",
        ],
    },
    LauncherWrites {
        launcher: "dsv4_compressor_update(indexer)",
        writes: &[
            "slot.indexer.pending_kv",
            "slot.indexer.pending_score",
            "slot.indexer.prev_overlap_kv",
            "slot.indexer.prev_overlap_score",
            "slot.indexer.compressed",
        ],
    },
    LauncherWrites {
        launcher: "dsv4_csa_select",
        writes: &["scratch.selected"],
    },
    LauncherWrites {
        launcher: "dsv4_hybrid_attention(write_window_cache=1)",
        writes: &["scratch.attn_local", "slot.sw_window_ring"],
    },
];

#[cfg(feature = "vulkan")]
pub struct VulkanDsv4Model {
    pub config: deepseek_spec::v4::DeepSeekV4Config,
}

#[cfg(feature = "vulkan")]
impl VulkanDsv4Model {
    pub fn forward_token(
        &mut self,
        _slot: usize,
        _epoch: u64,
        _token: u32,
        _start_pos: usize,
    ) -> anyhow::Result<Vec<f32>> {
        let _ = dsv4_forward_kernel_sequence(self.config.num_hidden_layers);
        anyhow::bail!(
            "Vulkan DSv4 numeric forward requires GGUF residency binding for the launcher sequence"
        )
    }
}
