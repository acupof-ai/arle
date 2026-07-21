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

pub fn layer_rope_thetas(config: &deepseek_spec::v4::DeepSeekV4Config) -> Vec<Dsv4LayerRope> {
    config
        .compress_ratios
        .iter()
        .copied()
        .take(config.num_hidden_layers)
        .enumerate()
        .map(|(layer_idx, compress_ratio)| {
            let rope_theta = if compress_ratio > 0 {
                config.compress_rope_theta
            } else {
                config.rope_theta
            };
            Dsv4LayerRope {
                layer_idx,
                compress_ratio,
                rope_theta,
            }
        })
        .collect()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dsv4_fallback_order_is_pinned() {
        let ops = dsv4_fallback_forward_ops(1);
        assert_eq!(ops[0], Dsv4Op::TokenEmbedding);
        assert_eq!(ops[1], Dsv4Op::MhcExpand);
        assert_eq!(
            &ops[2..2 + DSV4_FALLBACK_LAYER_OPS.len()],
            DSV4_FALLBACK_LAYER_OPS
        );
        assert!(ops.contains(&Dsv4Op::PrepareQk));
        assert!(ops.contains(&Dsv4Op::CsaSelect));
        assert!(ops.contains(&Dsv4Op::HybridAttention));
        assert!(ops.contains(&Dsv4Op::HostSqrtSoftplusRouting));
        assert_eq!(ops[ops.len() - 3], Dsv4Op::MhcHeadPre);
        assert_eq!(ops[ops.len() - 2], Dsv4Op::FinalRmsNorm);
        assert_eq!(ops[ops.len() - 1], Dsv4Op::LmHead);
        const {
            assert!(!DSV4_PREFIX_CACHE_ENABLED);
        }
    }

    #[test]
    fn mutated_slot_buffer_enumeration_is_exact() {
        let mut found: Vec<&'static str> = DSV4_ATTENTION_LAUNCHER_WRITES
            .iter()
            .flat_map(|entry| entry.writes.iter().copied())
            .filter(|name| name.starts_with("slot."))
            .collect();
        found.sort_unstable();
        found.dedup();
        let mut expected = DSV4_MUTATED_SLOT_BUFFERS.to_vec();
        expected.sort_unstable();
        assert_eq!(found, expected);
    }

    #[test]
    fn every_dsv4_op_has_a_launcher_class() {
        let kinds = dsv4_launcher_sequence(1);
        assert_eq!(kinds.first(), Some(&Dsv4LauncherKind::TokenEmbedding));
        assert!(kinds.contains(&Dsv4LauncherKind::Dsv4PrepareQk));
        assert!(kinds.contains(&Dsv4LauncherKind::Dsv4CompressorUpdate));
        assert!(kinds.contains(&Dsv4LauncherKind::Dsv4CsaSelect));
        assert!(kinds.contains(&Dsv4LauncherKind::Dsv4HybridAttention));
        assert!(kinds.contains(&Dsv4LauncherKind::Dsv4SwaAttention));
        assert!(kinds.contains(&Dsv4LauncherKind::Dsv4Mhc));
        assert!(kinds.contains(&Dsv4LauncherKind::SwigluClamped));
        assert!(kinds.contains(&Dsv4LauncherKind::HostRouting));
        assert_eq!(kinds.len(), dsv4_fallback_forward_ops(1).len());
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn dsv4_fused_launcher_classes_map_to_kernels() {
        assert_eq!(
            dsv4_kernel_for_launcher(Dsv4LauncherKind::Dsv4PrepareQk),
            Some(vulkan_kernels::Kernel::Dsv4PrepareQk)
        );
        assert_eq!(
            dsv4_kernel_for_launcher(Dsv4LauncherKind::Dsv4CompressorUpdate),
            Some(vulkan_kernels::Kernel::Dsv4CompressorUpdate)
        );
        assert_eq!(
            dsv4_kernel_for_launcher(Dsv4LauncherKind::Dsv4CsaSelect),
            Some(vulkan_kernels::Kernel::Dsv4CsaSelect)
        );
        assert_eq!(
            dsv4_kernel_for_launcher(Dsv4LauncherKind::Dsv4HybridAttention),
            Some(vulkan_kernels::Kernel::Dsv4HybridAttention)
        );
        assert_eq!(
            dsv4_kernel_for_launcher(Dsv4LauncherKind::Dsv4SwaAttention),
            Some(vulkan_kernels::Kernel::Dsv4SwaAttention)
        );
        assert_eq!(
            dsv4_kernel_for_launcher(Dsv4LauncherKind::SwigluClamped),
            Some(vulkan_kernels::Kernel::SwigluClamped)
        );
        assert_eq!(
            dsv4_kernel_for_launcher(Dsv4LauncherKind::HostRouting),
            None
        );
    }

    #[test]
    fn rope_theta_switch_follows_compression() {
        let config = deepseek_spec::v4::DeepSeekV4Config::from_json_str(
            r#"{
                "architectures": ["DeepseekV4ForCausalLM"],
                "model_type": "deepseek_v4",
                "dtype": "bfloat16",
                "vocab_size": 1000,
                "hidden_size": 512,
                "num_hidden_layers": 3,
                "num_attention_heads": 8,
                "num_key_value_heads": 1,
                "head_dim": 64,
                "hidden_act": "silu",
                "swiglu_limit": 10.0,
                "q_lora_rank": 256,
                "o_lora_rank": 32,
                "o_groups": 8,
                "qk_rope_head_dim": 32,
                "n_routed_experts": 16,
                "n_shared_experts": 1,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 256,
                "routed_scaling_factor": 1.5,
                "norm_topk_prob": true,
                "scoring_func": "sqrtsoftplus",
                "topk_method": "noaux_tc",
                "index_n_heads": 4,
                "index_head_dim": 64,
                "index_topk": 128,
                "num_hash_layers": 1,
                "sliding_window": 32,
                "compress_ratios": [0, 4, 32],
                "compress_rope_theta": 160000.0,
                "hc_mult": 4,
                "hc_sinkhorn_iters": 20,
                "hc_eps": 1.0e-6,
                "num_nextn_predict_layers": 0,
                "max_position_embeddings": 4096,
                "rope_theta": 10000.0,
                "rope_scaling": {
                    "type": "yarn",
                    "factor": 16.0,
                    "original_max_position_embeddings": 2048,
                    "beta_fast": 32.0,
                    "beta_slow": 1.0
                },
                "rms_norm_eps": 1.0e-6,
                "initializer_range": 0.02,
                "tie_word_embeddings": true,
                "attention_bias": false,
                "attention_dropout": 0.0,
                "bos_token_id": 0,
                "eos_token_id": 1,
                "pad_token_id": null
            }"#,
        )
        .unwrap();
        let thetas = layer_rope_thetas(&config);
        assert_eq!(thetas[0].rope_theta, 10000.0);
        assert_eq!(thetas[1].rope_theta, 160000.0);
        assert_eq!(thetas[2].rope_theta, 160000.0);
    }
}
