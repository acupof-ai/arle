//! GGUF (`deepseek4` arch) → [`DeepSeekV4Config`] + tensor-name → role map.
//!
//! Key strings and tensor names extracted from the nisparks DSv4 branch
//! (`pr/01-deepseek-v4-arch` @ 9cb6ae64): arch name "deepseek4"
//! (llama-arch.cpp:78), KV key table llama-arch.cpp:142-263, tensor-name
//! table llama-arch.cpp:345-571, deepseek4 hparam loads
//! llama-model.cpp:2038-2060, tensor creation llama-model.cpp:5441-5567.
//! Per-layer compress ratio is NOT a metadata key — it is
//! `blk.%d.attn_compress_ape` ne[1], 0 when the tensor is absent
//! (llama-memory-deepseek4.cpp:61-63). Hash routing is per-layer
//! `blk.%d.ffn_gate_tid2eid` presence; scoring is hardcoded
//! sqrt(softplus) (models/deepseek4.cpp:486-487).

use anyhow::{Context, Result, anyhow, bail};
use deepseek_spec::v4::{DeepSeekV4Config, DeepSeekV4RopeParameters};

use crate::gguf::GgufFile;

const ARCH: &str = "deepseek4";

fn key(suffix: &str) -> String {
    format!("{ARCH}.{suffix}")
}

fn req_usize(g: &GgufFile, suffix: &str) -> Result<usize> {
    g.get_usize(&key(suffix))
        .ok_or_else(|| anyhow!("GGUF missing required key {}", key(suffix)))
}

fn req_f32(g: &GgufFile, suffix: &str) -> Result<f32> {
    g.get_f32(&key(suffix))
        .ok_or_else(|| anyhow!("GGUF missing required key {}", key(suffix)))
}

/// Per-layer compress ratios from `blk.%d.attn_compress_ape` ne[1]
/// (llama-memory-deepseek4.cpp:61-63); 0 = sliding-window layer.
pub fn compress_ratios_from_tensors(g: &GgufFile, num_layers: usize) -> Vec<usize> {
    (0..num_layers)
        .map(|i| {
            g.tensor(&format!("blk.{i}.attn_compress_ape"))
                .and_then(|t| t.dims.get(1))
                .map_or(0, |&r| r as usize)
        })
        .collect()
}

/// Leading run of hash-routed layers = `blk.%d.ffn_gate_tid2eid` presence.
pub fn num_hash_layers_from_tensors(g: &GgufFile, num_layers: usize) -> usize {
    (0..num_layers)
        .take_while(|i| g.tensor(&format!("blk.{i}.ffn_gate_tid2eid")).is_some())
        .count()
}

pub fn dsv4_config_from_gguf(g: &GgufFile) -> Result<DeepSeekV4Config> {
    let arch = g
        .get_str("general.architecture")
        .ok_or_else(|| anyhow!("GGUF missing general.architecture"))?;
    if arch != ARCH {
        bail!("GGUF architecture {arch:?} is not {ARCH:?}");
    }

    let num_hidden_layers = req_usize(g, "block_count")?;
    let hidden_size = req_usize(g, "embedding_length")?;
    let num_attention_heads = req_usize(g, "attention.head_count")?;
    let head_dim = req_usize(g, "attention.key_length")?;

    let vocab_size = g
        .get_usize(&key("vocab_size"))
        .or_else(|| {
            g.tensor("token_embd.weight")
                .and_then(|t| t.dims.get(1))
                .map(|&v| v as usize)
        })
        .ok_or_else(|| {
            anyhow!(
                "GGUF has neither {} nor token_embd.weight",
                key("vocab_size")
            )
        })?;

    // hc_mult = hc_head_base ne[0] (llama-model.cpp:5462), default 4.
    let hc_mult = g
        .tensor("hc_head_base")
        .and_then(|t| t.dims.first())
        .map_or(4, |&v| v as usize);

    // o_groups/o_lora_rank derived from attn_output_a (llama-model.cpp:
    // 5483-5495): ggml ne = {head_dim, n_groups * o_rank} with
    // group_dim = head_dim → n_groups = n_head; default o_rank = head_dim/2.
    let (o_groups, o_lora_rank) = match g.tensor("blk.0.attn_output_a.weight") {
        Some(t) if t.dims.len() >= 2 && num_attention_heads > 0 => (
            num_attention_heads,
            (t.dims[1] as usize / num_attention_heads).max(1),
        ),
        _ => (num_attention_heads, (head_dim / 2).max(1)),
    };

    let rope_theta = req_f32(g, "rope.freq_base")?;
    let rope_parameters = DeepSeekV4RopeParameters {
        rope_type: g
            .get_str(&key("rope.scaling.type"))
            .unwrap_or("yarn")
            .to_string(),
        factor: g.get_f32(&key("rope.scaling.factor")).unwrap_or(1.0),
        original_max_position_embeddings: g
            .get_usize(&key("rope.scaling.original_context_length"))
            .unwrap_or_else(|| g.get_usize(&key("context_length")).unwrap_or(0)),
        beta_fast: g
            .get_f32(&key("rope.scaling.yarn_beta_fast"))
            .unwrap_or(32.0),
        beta_slow: g
            .get_f32(&key("rope.scaling.yarn_beta_slow"))
            .unwrap_or(1.0),
        rope_theta: Some(rope_theta),
    };

    let config = DeepSeekV4Config {
        architectures: vec!["DeepseekV4ForCausalLM".to_string()],
        model_type: "deepseek_v4".to_string(),
        dtype: "bfloat16".to_string(),
        vocab_size,
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads: g.get_usize(&key("attention.head_count_kv")).unwrap_or(1),
        head_dim,
        hidden_act: "silu".to_string(),
        // llama-model.cpp:2054-2055 defaults the clamp to 10.0.
        swiglu_limit: g
            .get_f32_scalar_or_first(&key("swiglu_clamp_exp"))
            .unwrap_or(10.0),
        q_lora_rank: req_usize(g, "attention.q_lora_rank")?,
        o_lora_rank,
        o_groups,
        qk_rope_head_dim: req_usize(g, "rope.dimension_count")?,
        n_routed_experts: req_usize(g, "expert_count")?,
        n_shared_experts: g.get_usize(&key("expert_shared_count")).unwrap_or(0),
        num_experts_per_tok: req_usize(g, "expert_used_count")?,
        moe_intermediate_size: req_usize(g, "expert_feed_forward_length")?,
        routed_scaling_factor: g.get_f32(&key("expert_weights_scale")).unwrap_or(1.0),
        norm_topk_prob: g.get_bool(&key("expert_weights_norm")).unwrap_or(true),
        scoring_func: "sqrtsoftplus".to_string(),
        topk_method: "noaux_tc".to_string(),
        index_n_heads: g
            .get_usize(&key("attention.indexer.head_count"))
            .unwrap_or(1),
        index_head_dim: g
            .get_usize(&key("attention.indexer.key_length"))
            .unwrap_or(head_dim),
        index_topk: g.get_usize(&key("attention.indexer.top_k")).unwrap_or(2048),
        num_hash_layers: num_hash_layers_from_tensors(g, num_hidden_layers),
        sliding_window: req_usize(g, "attention.sliding_window")?,
        compress_ratios: compress_ratios_from_tensors(g, num_hidden_layers),
        compress_rope_theta: g.get_f32(&key("rope.freq_base_swa")).unwrap_or(rope_theta),
        hc_mult,
        hc_sinkhorn_iters: 20,
        hc_eps: 1.0e-6,
        num_nextn_predict_layers: g.get_usize(&key("nextn_predict_layers")).unwrap_or(0),
        max_position_embeddings: req_usize(g, "context_length")?,
        rope_theta,
        rope_parameters,
        rms_norm_eps: req_f32(g, "attention.layer_norm_rms_epsilon")?,
        initializer_range: 0.02,
        tie_word_embeddings: g.tensor("output.weight").is_none(),
        attention_bias: false,
        attention_dropout: 0.0,
        bos_token_id: g
            .get_u64("tokenizer.ggml.bos_token_id")
            .and_then(|v| u32::try_from(v).ok()),
        eos_token_id: g
            .get_u64("tokenizer.ggml.eos_token_id")
            .and_then(|v| u32::try_from(v).ok()),
        pad_token_id: g
            .get_u64("tokenizer.ggml.padding_token_id")
            .and_then(|v| u32::try_from(v).ok()),
        // GLM-5.2 dialect extensions: unused on the DSv4 GGUF path.
        kv_lora_rank: 0,
        qk_nope_head_dim: 0,
        v_head_dim: 0,
        plain_o_proj: false,
        per_layer_attention_mode: None,
        per_layer_dense_mlp: None,
        per_layer_full_indexer: None,
        tensor_dialect: deepseek_spec::TensorDialect::Dsv4,
        // GGUF DSv4 path is native MTP / non-spec: DSpark disabled (0 = off).
        dspark_block_size: 0,
        dspark_target_layer_ids: Vec::new(),
        dspark_markov_rank: 0,
        dspark_noise_token_id: 0,
        dspark_num_stages: 0,
    };
    config
        .validate()
        .context("GGUF-derived DSv4 config invalid")?;
    Ok(config)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dsv4TensorKind {
    TokenEmbedding,
    OutputNorm,
    OutputHead,
    HeadHyperConnection,
    AttnNorm,
    FfnNorm,
    AttnQNorm,
    AttnKvNorm,
    AttnQA,
    AttnQB,
    AttnKvLatent,
    AttnOutA,
    AttnOutB,
    AttnSink,
    CompressorApe,
    CompressorNorm,
    CompressorKv,
    CompressorGate,
    IndexerProj,
    IndexerQB,
    IndexerCompressApe,
    IndexerCompressNorm,
    IndexerCompressKv,
    IndexerCompressGate,
    HyperConnection,
    RouterGate,
    RouterBias,
    RouterHashTable,
    RoutedExpertGate,
    RoutedExpertUp,
    RoutedExpertGateUp,
    RoutedExpertDown,
    SharedExpertGate,
    SharedExpertUp,
    SharedExpertDown,
}

impl Dsv4TensorKind {
    pub fn is_routed_expert(self) -> bool {
        matches!(
            self,
            Self::RoutedExpertGate
                | Self::RoutedExpertUp
                | Self::RoutedExpertGateUp
                | Self::RoutedExpertDown
        )
    }

    pub fn is_norm(self) -> bool {
        matches!(
            self,
            Self::OutputNorm
                | Self::AttnNorm
                | Self::FfnNorm
                | Self::AttnQNorm
                | Self::AttnKvNorm
                | Self::CompressorNorm
                | Self::IndexerCompressNorm
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dsv4TensorRole {
    pub layer: Option<usize>,
    pub kind: Dsv4TensorKind,
}

/// Map a GGUF tensor name to its DSv4 role; unknown names are an error so
/// loaders never silently skip weights.
pub fn classify_tensor(name: &str) -> Result<Dsv4TensorRole> {
    use Dsv4TensorKind::*;
    let global = |kind| Ok(Dsv4TensorRole { layer: None, kind });
    match name {
        "token_embd.weight" => return global(TokenEmbedding),
        "output_norm.weight" => return global(OutputNorm),
        "output.weight" => return global(OutputHead),
        "hc_head_base" | "hc_head_fn" | "hc_head_scale" => return global(HeadHyperConnection),
        _ => {}
    }
    let rest = name
        .strip_prefix("blk.")
        .ok_or_else(|| anyhow!("unknown DSv4 tensor name {name:?}"))?;
    let dot = rest
        .find('.')
        .ok_or_else(|| anyhow!("unknown DSv4 tensor name {name:?}"))?;
    let layer: usize = rest[..dot]
        .parse()
        .map_err(|_| anyhow!("bad layer index in tensor name {name:?}"))?;
    let kind = match &rest[dot + 1..] {
        "attn_norm.weight" => AttnNorm,
        "ffn_norm.weight" => FfnNorm,
        "attn_q_a_norm.weight" => AttnQNorm,
        "attn_kv_a_norm.weight" => AttnKvNorm,
        "attn_q_a.weight" => AttnQA,
        "attn_q_b.weight" => AttnQB,
        "attn_kv_latent.weight" => AttnKvLatent,
        "attn_output_a.weight" => AttnOutA,
        "attn_output_b.weight" => AttnOutB,
        "attn_sinks" => AttnSink,
        "attn_compress_ape" => CompressorApe,
        "attn_compress_norm.weight" => CompressorNorm,
        "attn_compress_kv.weight" => CompressorKv,
        "attn_compress_gate.weight" => CompressorGate,
        "indexer.proj.weight" => IndexerProj,
        "indexer.attn_q_b.weight" => IndexerQB,
        "indexer.compress_ape" => IndexerCompressApe,
        "indexer.compress_norm.weight" => IndexerCompressNorm,
        "indexer.compress_kv.weight" => IndexerCompressKv,
        "indexer.compress_gate.weight" => IndexerCompressGate,
        "hc_attn_base" | "hc_attn_fn" | "hc_attn_scale" | "hc_ffn_base" | "hc_ffn_fn"
        | "hc_ffn_scale" => HyperConnection,
        "ffn_gate_inp.weight" => RouterGate,
        "exp_probs_b" => RouterBias,
        "ffn_gate_tid2eid" => RouterHashTable,
        "ffn_gate_exps.weight" => RoutedExpertGate,
        "ffn_up_exps.weight" => RoutedExpertUp,
        "ffn_gate_up_exps.weight" => RoutedExpertGateUp,
        "ffn_down_exps.weight" => RoutedExpertDown,
        "ffn_gate_shexp.weight" => SharedExpertGate,
        "ffn_up_shexp.weight" => SharedExpertUp,
        "ffn_down_shexp.weight" => SharedExpertDown,
        _ => bail!("unknown DSv4 tensor name {name:?}"),
    };
    Ok(Dsv4TensorRole {
        layer: Some(layer),
        kind,
    })
}
