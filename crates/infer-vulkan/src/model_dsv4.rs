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
    let mut ops = Vec::with_capacity(4 + num_layers * DSV4_FALLBACK_LAYER_OPS.len());
    ops.push(Dsv4Op::TokenEmbedding);
    ops.push(Dsv4Op::MhcExpand);
    for _ in 0..num_layers {
        ops.extend_from_slice(DSV4_FALLBACK_LAYER_OPS);
    }
    ops.push(Dsv4Op::MhcHeadPre);
    ops.push(Dsv4Op::FinalRmsNorm);
    ops.push(Dsv4Op::LmHead);
    ops
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
        anyhow::bail!(
            "Vulkan DSv4 numeric forward is blocked: CSA/compressor/hybrid-attention shaders \
             are not ported yet"
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
        assert!(!DSV4_PREFIX_CACHE_ENABLED);
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
