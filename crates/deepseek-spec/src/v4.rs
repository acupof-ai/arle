use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{DeepSeekConfigError, Result, Shard};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeepSeekV4RopeParameters {
    pub rope_type: String,
    #[serde(default, rename = "type")]
    pub type_alias: Option<String>,
    pub factor: f32,
    pub original_max_position_embeddings: usize,
    pub beta_fast: f32,
    pub beta_slow: f32,
    #[serde(default)]
    pub rope_theta: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeepSeekV4Config {
    pub architectures: Vec<String>,
    pub model_type: String,
    pub dtype: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub hidden_act: String,
    pub swiglu_limit: f32,
    pub q_lora_rank: usize,
    pub o_lora_rank: usize,
    pub o_groups: usize,
    pub qk_rope_head_dim: usize,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub routed_scaling_factor: f32,
    pub norm_topk_prob: bool,
    pub scoring_func: String,
    pub topk_method: String,
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    pub num_hash_layers: usize,
    pub sliding_window: usize,
    pub compress_ratios: Vec<usize>,
    pub compress_rope_theta: f32,
    pub hc_mult: usize,
    pub hc_sinkhorn_iters: usize,
    pub hc_eps: f32,
    pub num_nextn_predict_layers: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
    pub rope_parameters: DeepSeekV4RopeParameters,
    pub rms_norm_eps: f32,
    pub initializer_range: f32,
    pub tie_word_embeddings: bool,
    pub attention_bias: bool,
    pub attention_dropout: f32,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,
}

impl DeepSeekV4Config {
    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        Self::from_json_str(&content)
    }

    pub fn from_json_str(content: &str) -> Result<Self> {
        let value: serde_json::Value = serde_json::from_str(content)?;
        Self::from_json_value(&value)
    }

    pub fn from_json_value(value: &serde_json::Value) -> Result<Self> {
        let config: Self = serde_json::from_value(value.clone())?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if self.model_type != "deepseek_v4" {
            return Err(DeepSeekConfigError::InvalidConfig(
                "model_type must be deepseek_v4",
            ));
        }
        if !self
            .architectures
            .iter()
            .any(|arch| arch == "DeepseekV4ForCausalLM")
        {
            return Err(DeepSeekConfigError::InvalidConfig(
                "architectures must contain DeepseekV4ForCausalLM",
            ));
        }
        if self.hidden_size == 0
            || self.num_hidden_layers == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || self.head_dim == 0
        {
            return Err(DeepSeekConfigError::InvalidConfig(
                "hidden size, layers, and attention heads must be non-zero",
            ));
        }
        if self.num_key_value_heads != 1 {
            return Err(DeepSeekConfigError::InvalidConfig(
                "DSV4 replica expects num_key_value_heads=1",
            ));
        }
        if !self.num_attention_heads.is_multiple_of(self.o_groups) {
            return Err(DeepSeekConfigError::InvalidConfig(
                "num_attention_heads must be divisible by o_groups",
            ));
        }
        if self.compress_ratios.len() != self.num_hidden_layers {
            return Err(DeepSeekConfigError::InvalidConfig(
                "compress_ratios length must match num_hidden_layers",
            ));
        }
        if self.q_lora_rank == 0
            || self.o_lora_rank == 0
            || self.qk_rope_head_dim == 0
            || self.index_n_heads == 0
            || self.index_head_dim == 0
            || self.hc_mult == 0
        {
            return Err(DeepSeekConfigError::InvalidConfig(
                "DSV4 low-rank, indexer, and mHC dimensions must be non-zero",
            ));
        }
        Ok(())
    }

    pub fn tensor_names(&self) -> DeepSeekV4TensorNames {
        DeepSeekV4TensorNames
    }

    pub fn layer_tensor_names(&self, layer_idx: usize) -> DeepSeekV4LayerTensorNames {
        let compress_ratio = self.compress_ratios[layer_idx];
        self.tensor_names().layer(
            layer_idx,
            compress_ratio,
            layer_idx < self.num_hash_layers,
            self.n_shared_experts > 0,
        )
    }

    pub fn mtp_tensor_names(&self, mtp_idx: usize) -> DeepSeekV4MtpTensorNames {
        DeepSeekV4MtpTensorNames::new(format!("mtp.{mtp_idx}"), self.n_shared_experts > 0)
    }

    pub fn shard_for_global_tensor(&self, name: &str) -> Option<Shard> {
        match name {
            "embed.weight" | "head.weight" => Some(Shard::VocabParallel { dim: 0 }),
            "norm.weight" | "hc_head_base" | "hc_head_fn" | "hc_head_scale" => {
                Some(Shard::Replicated)
            }
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DeepSeekV4TensorNames;

impl DeepSeekV4TensorNames {
    pub fn embed_tokens(&self) -> &'static str {
        "embed.weight"
    }

    pub fn norm(&self) -> &'static str {
        "norm.weight"
    }

    pub fn lm_head(&self) -> &'static str {
        "head.weight"
    }

    pub fn head_hc(&self) -> DeepSeekV4HyperConnectionTensorNames {
        DeepSeekV4HyperConnectionTensorNames::new("hc_head")
    }

    pub fn layer(
        &self,
        layer_idx: usize,
        compress_ratio: usize,
        hash_routing: bool,
        include_shared_experts: bool,
    ) -> DeepSeekV4LayerTensorNames {
        DeepSeekV4LayerTensorNames::new(
            format!("layers.{layer_idx}"),
            compress_ratio,
            hash_routing,
            include_shared_experts,
        )
    }

    pub fn mtp(&self, mtp_idx: usize) -> DeepSeekV4MtpTensorNames {
        DeepSeekV4MtpTensorNames::new(format!("mtp.{mtp_idx}"), true)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4HyperConnectionTensorNames {
    pub base: String,
    pub mix_fn: String,
    pub scale: String,
}

impl DeepSeekV4HyperConnectionTensorNames {
    fn new(prefix: &str) -> Self {
        Self {
            base: format!("{prefix}_base"),
            mix_fn: format!("{prefix}_fn"),
            scale: format!("{prefix}_scale"),
        }
    }

    pub fn shard_for(&self, name: &str) -> Option<Shard> {
        (name == self.base || name == self.mix_fn || name == self.scale)
            .then_some(Shard::Replicated)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4CompressorTensorNames {
    pub prefix: String,
    pub wkv: String,
    pub wgate: String,
    pub ape: String,
    pub norm: String,
}

impl DeepSeekV4CompressorTensorNames {
    fn new(prefix: String) -> Self {
        Self {
            wkv: format!("{prefix}.wkv.weight"),
            wgate: format!("{prefix}.wgate.weight"),
            ape: format!("{prefix}.ape"),
            norm: format!("{prefix}.norm.weight"),
            prefix,
        }
    }

    pub fn shard_for(&self, name: &str) -> Option<Shard> {
        match name {
            n if n == self.wkv || n == self.wgate => Some(Shard::Column { dim: 0 }),
            n if n == self.ape || n == self.norm => Some(Shard::Replicated),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4IndexerTensorNames {
    pub prefix: String,
    pub wq_b: String,
    pub weights_proj: String,
    pub compressor: DeepSeekV4CompressorTensorNames,
}

impl DeepSeekV4IndexerTensorNames {
    fn new(prefix: String) -> Self {
        Self {
            wq_b: format!("{prefix}.wq_b.weight"),
            weights_proj: format!("{prefix}.weights_proj.weight"),
            compressor: DeepSeekV4CompressorTensorNames::new(format!("{prefix}.compressor")),
            prefix,
        }
    }

    pub fn shard_for(
        &self,
        config: &DeepSeekV4Config,
        name: &str,
        tensor_parallel_size: usize,
    ) -> Option<Shard> {
        if name == self.wq_b || name == self.weights_proj {
            return Some(
                if config.index_n_heads.is_multiple_of(tensor_parallel_size) {
                    Shard::Column { dim: 0 }
                } else {
                    Shard::Replicated
                },
            );
        }
        self.compressor.shard_for(name)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4AttentionTensorNames {
    pub prefix: String,
    pub wq_a: String,
    pub q_norm: String,
    pub wq_b: String,
    pub wkv: String,
    pub kv_norm: String,
    pub wo_a: String,
    pub wo_b: String,
    pub attn_sink: String,
    pub compressor: Option<DeepSeekV4CompressorTensorNames>,
    pub indexer: Option<DeepSeekV4IndexerTensorNames>,
}

impl DeepSeekV4AttentionTensorNames {
    fn new(prefix: String, compress_ratio: usize) -> Self {
        let compressor = (compress_ratio > 0)
            .then(|| DeepSeekV4CompressorTensorNames::new(format!("{prefix}.compressor")));
        let indexer = (compress_ratio > 0 && compress_ratio < 16)
            .then(|| DeepSeekV4IndexerTensorNames::new(format!("{prefix}.indexer")));
        Self {
            wq_a: format!("{prefix}.wq_a.weight"),
            q_norm: format!("{prefix}.q_norm.weight"),
            wq_b: format!("{prefix}.wq_b.weight"),
            wkv: format!("{prefix}.wkv.weight"),
            kv_norm: format!("{prefix}.kv_norm.weight"),
            wo_a: format!("{prefix}.wo_a.weight"),
            wo_b: format!("{prefix}.wo_b.weight"),
            attn_sink: format!("{prefix}.attn_sink"),
            compressor,
            indexer,
            prefix,
        }
    }

    pub fn shard_for(
        &self,
        config: &DeepSeekV4Config,
        name: &str,
        tensor_parallel_size: usize,
    ) -> Option<Shard> {
        if name == self.wq_a || name == self.q_norm || name == self.wkv || name == self.kv_norm {
            return Some(Shard::Replicated);
        }
        if name == self.wq_b {
            return Some(Shard::Column { dim: 0 });
        }
        if name == self.wo_a {
            return Some(if config.o_groups.is_multiple_of(tensor_parallel_size) {
                Shard::Column { dim: 0 }
            } else {
                Shard::Replicated
            });
        }
        if name == self.wo_b {
            return Some(Shard::Row { dim: 1 });
        }
        if name == self.attn_sink {
            return Some(Shard::Replicated);
        }
        self.compressor
            .as_ref()
            .and_then(|compressor| compressor.shard_for(name))
            .or_else(|| {
                self.indexer
                    .as_ref()
                    .and_then(|indexer| indexer.shard_for(config, name, tensor_parallel_size))
            })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4ExpertTensorNames {
    pub prefix: String,
    pub w1: String,
    pub w2: String,
    pub w3: String,
}

impl DeepSeekV4ExpertTensorNames {
    fn new(prefix: String) -> Self {
        Self {
            w1: format!("{prefix}.w1.weight"),
            w2: format!("{prefix}.w2.weight"),
            w3: format!("{prefix}.w3.weight"),
            prefix,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4MoeTensorNames {
    pub prefix: String,
    pub gate_weight: String,
    pub gate_bias: Option<String>,
    pub gate_tid2eid: Option<String>,
    pub experts_prefix: String,
    pub shared_experts: Option<DeepSeekV4ExpertTensorNames>,
}

impl DeepSeekV4MoeTensorNames {
    fn new(prefix: String, hash_routing: bool, include_shared_experts: bool) -> Self {
        Self {
            gate_weight: format!("{prefix}.gate.weight"),
            gate_bias: (!hash_routing).then(|| format!("{prefix}.gate.bias")),
            gate_tid2eid: hash_routing.then(|| format!("{prefix}.gate.tid2eid")),
            experts_prefix: format!("{prefix}.experts"),
            shared_experts: include_shared_experts
                .then(|| DeepSeekV4ExpertTensorNames::new(format!("{prefix}.shared_experts"))),
            prefix,
        }
    }

    pub fn expert(&self, expert_idx: usize) -> DeepSeekV4ExpertTensorNames {
        DeepSeekV4ExpertTensorNames::new(format!("{}.{}", self.experts_prefix, expert_idx))
    }

    pub fn shard_for(&self, name: &str) -> Option<Shard> {
        if name == self.gate_weight
            || self.gate_bias.as_ref().is_some_and(|bias| name == bias)
            || self
                .gate_tid2eid
                .as_ref()
                .is_some_and(|table| name == table)
        {
            return Some(Shard::Replicated);
        }
        if name.starts_with(&self.experts_prefix) {
            return Some(Shard::ExpertParallel { dim: 0 });
        }
        if let Some(shared) = &self.shared_experts {
            if name == shared.w1 || name == shared.w3 {
                return Some(Shard::Column { dim: 0 });
            }
            if name == shared.w2 {
                return Some(Shard::Row { dim: 1 });
            }
        }
        None
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4LayerTensorNames {
    pub prefix: String,
    pub attn_norm: String,
    pub ffn_norm: String,
    pub hc_attn: DeepSeekV4HyperConnectionTensorNames,
    pub hc_ffn: DeepSeekV4HyperConnectionTensorNames,
    pub attn: DeepSeekV4AttentionTensorNames,
    pub ffn: DeepSeekV4MoeTensorNames,
}

impl DeepSeekV4LayerTensorNames {
    fn new(
        prefix: String,
        compress_ratio: usize,
        hash_routing: bool,
        include_shared_experts: bool,
    ) -> Self {
        Self {
            attn_norm: format!("{prefix}.attn_norm.weight"),
            ffn_norm: format!("{prefix}.ffn_norm.weight"),
            hc_attn: DeepSeekV4HyperConnectionTensorNames::new(&format!("{prefix}.hc_attn")),
            hc_ffn: DeepSeekV4HyperConnectionTensorNames::new(&format!("{prefix}.hc_ffn")),
            attn: DeepSeekV4AttentionTensorNames::new(format!("{prefix}.attn"), compress_ratio),
            ffn: DeepSeekV4MoeTensorNames::new(
                format!("{prefix}.ffn"),
                hash_routing,
                include_shared_experts,
            ),
            prefix,
        }
    }

    pub fn shard_for(
        &self,
        config: &DeepSeekV4Config,
        name: &str,
        tensor_parallel_size: usize,
    ) -> Option<Shard> {
        if name == self.attn_norm || name == self.ffn_norm {
            return Some(Shard::Replicated);
        }
        self.hc_attn
            .shard_for(name)
            .or_else(|| self.hc_ffn.shard_for(name))
            .or_else(|| self.attn.shard_for(config, name, tensor_parallel_size))
            .or_else(|| self.ffn.shard_for(name))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4MtpTensorNames {
    pub prefix: String,
    pub enorm: String,
    pub hnorm: String,
    pub e_proj: String,
    pub h_proj: String,
    pub attn_norm: String,
    pub ffn_norm: String,
    pub norm: String,
    pub hc_attn: DeepSeekV4HyperConnectionTensorNames,
    pub hc_ffn: DeepSeekV4HyperConnectionTensorNames,
    pub hc_head: DeepSeekV4HyperConnectionTensorNames,
    pub attn: DeepSeekV4AttentionTensorNames,
    pub ffn: DeepSeekV4MoeTensorNames,
}

impl DeepSeekV4MtpTensorNames {
    fn new(prefix: String, include_shared_experts: bool) -> Self {
        Self {
            enorm: format!("{prefix}.enorm.weight"),
            hnorm: format!("{prefix}.hnorm.weight"),
            e_proj: format!("{prefix}.e_proj.weight"),
            h_proj: format!("{prefix}.h_proj.weight"),
            attn_norm: format!("{prefix}.attn_norm.weight"),
            ffn_norm: format!("{prefix}.ffn_norm.weight"),
            norm: format!("{prefix}.norm.weight"),
            hc_attn: DeepSeekV4HyperConnectionTensorNames::new(&format!("{prefix}.hc_attn")),
            hc_ffn: DeepSeekV4HyperConnectionTensorNames::new(&format!("{prefix}.hc_ffn")),
            hc_head: DeepSeekV4HyperConnectionTensorNames::new(&format!("{prefix}.hc_head")),
            attn: DeepSeekV4AttentionTensorNames::new(format!("{prefix}.attn"), 0),
            ffn: DeepSeekV4MoeTensorNames::new(
                format!("{prefix}.ffn"),
                false,
                include_shared_experts,
            ),
            prefix,
        }
    }

    pub fn shard_for(
        &self,
        config: &DeepSeekV4Config,
        name: &str,
        tensor_parallel_size: usize,
    ) -> Option<Shard> {
        if name == self.enorm
            || name == self.hnorm
            || name == self.attn_norm
            || name == self.ffn_norm
            || name == self.norm
        {
            return Some(Shard::Replicated);
        }
        if name == self.e_proj || name == self.h_proj {
            return Some(Shard::Column { dim: 0 });
        }
        self.hc_attn
            .shard_for(name)
            .or_else(|| self.hc_ffn.shard_for(name))
            .or_else(|| self.hc_head.shard_for(name))
            .or_else(|| self.attn.shard_for(config, name, tensor_parallel_size))
            .or_else(|| self.ffn.shard_for(name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn replica_config_path() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../infer/models/dsv4-mini-1B-init/config.json")
    }

    fn replica_config() -> DeepSeekV4Config {
        DeepSeekV4Config::from_json_file(replica_config_path()).unwrap()
    }

    #[test]
    fn parses_hf_replica_config() {
        let cfg = replica_config();
        assert_eq!(cfg.model_type, "deepseek_v4");
        assert_eq!(cfg.dtype, "bfloat16");
        assert_eq!(cfg.vocab_size, 129_280);
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.num_hidden_layers, 24);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.num_key_value_heads, 1);
        assert_eq!(cfg.head_dim, 64);
        assert_eq!(cfg.swiglu_limit, 10.0);
        assert_eq!(cfg.q_lora_rank, 384);
        assert_eq!(cfg.o_lora_rank, 384);
        assert_eq!(cfg.o_groups, 4);
        assert_eq!(cfg.scoring_func, "sqrtsoftplus");
        assert_eq!(cfg.topk_method, "noaux_tc");
        assert_eq!(cfg.index_n_heads, 8);
        assert_eq!(cfg.index_head_dim, 64);
        assert_eq!(cfg.index_topk, 128);
        assert_eq!(cfg.num_hash_layers, 2);
        assert_eq!(cfg.sliding_window, 64);
        assert_eq!(cfg.compress_ratios.len(), 24);
        assert_eq!(cfg.compress_ratios[2], 4);
        assert_eq!(cfg.compress_rope_theta, 160_000.0);
        assert_eq!(cfg.hc_mult, 4);
        assert_eq!(cfg.hc_sinkhorn_iters, 20);
        assert_eq!(cfg.hc_eps, 1.0e-6);
        assert_eq!(cfg.num_nextn_predict_layers, 1);
        assert_eq!(cfg.rope_parameters.rope_type, "yarn");
        assert_eq!(cfg.rope_parameters.factor, 16.0);
        assert!(!cfg.attention_bias);
        assert_eq!(cfg.attention_dropout, 0.0);
        assert_eq!(cfg.pad_token_id, None);
    }

    #[test]
    fn tensor_names_match_hf_replica_layout() {
        let cfg = replica_config();
        let top = cfg.tensor_names();
        assert_eq!(top.embed_tokens(), "embed.weight");
        assert_eq!(top.lm_head(), "head.weight");
        assert_eq!(top.head_hc().mix_fn, "hc_head_fn");

        let sw_hash = cfg.layer_tensor_names(0);
        assert_eq!(sw_hash.attn.wq_a, "layers.0.attn.wq_a.weight");
        assert_eq!(sw_hash.attn.wkv, "layers.0.attn.wkv.weight");
        assert_eq!(sw_hash.hc_attn.mix_fn, "layers.0.hc_attn_fn");
        assert!(sw_hash.attn.compressor.is_none());
        assert_eq!(
            sw_hash.ffn.gate_tid2eid.as_deref(),
            Some("layers.0.ffn.gate.tid2eid")
        );
        assert!(sw_hash.ffn.gate_bias.is_none());

        let csa = cfg.layer_tensor_names(2);
        assert_eq!(
            csa.attn.compressor.as_ref().unwrap().wgate,
            "layers.2.attn.compressor.wgate.weight"
        );
        assert_eq!(
            csa.attn.indexer.as_ref().unwrap().compressor.ape,
            "layers.2.attn.indexer.compressor.ape"
        );
        assert_eq!(csa.ffn.gate_bias.as_deref(), Some("layers.2.ffn.gate.bias"));
        assert_eq!(csa.ffn.expert(7).w2, "layers.2.ffn.experts.7.w2.weight");

        let hca = cfg.layer_tensor_names(3);
        assert!(hca.attn.compressor.is_some());
        assert!(hca.attn.indexer.is_none());

        let mtp = cfg.mtp_tensor_names(0);
        assert_eq!(mtp.enorm, "mtp.0.enorm.weight");
        assert_eq!(mtp.e_proj, "mtp.0.e_proj.weight");
        assert_eq!(mtp.attn.attn_sink, "mtp.0.attn.attn_sink");
        assert_eq!(mtp.hc_head.scale, "mtp.0.hc_head_scale");
        assert_eq!(
            mtp.ffn.shared_experts.as_ref().unwrap().w3,
            "mtp.0.ffn.shared_experts.w3.weight"
        );
    }

    #[test]
    fn shard_policy_handles_dsv4_shapes() {
        let cfg = replica_config();
        let csa = cfg.layer_tensor_names(2);
        assert_eq!(
            csa.shard_for(&cfg, &csa.attn.wkv, 4),
            Some(Shard::Replicated)
        );
        assert_eq!(
            csa.shard_for(&cfg, &csa.attn.wq_b, 4),
            Some(Shard::Column { dim: 0 })
        );
        assert_eq!(
            csa.shard_for(&cfg, &csa.attn.wo_a, 4),
            Some(Shard::Column { dim: 0 })
        );
        assert_eq!(
            csa.shard_for(&cfg, &csa.attn.wo_b, 4),
            Some(Shard::Row { dim: 1 })
        );
        assert_eq!(
            csa.shard_for(&cfg, &csa.hc_ffn.mix_fn, 4),
            Some(Shard::Replicated)
        );
        assert_eq!(
            csa.shard_for(&cfg, csa.attn.indexer.as_ref().unwrap().wq_b.as_str(), 4),
            Some(Shard::Column { dim: 0 })
        );
        assert_eq!(
            csa.shard_for(&cfg, csa.attn.indexer.as_ref().unwrap().wq_b.as_str(), 3),
            Some(Shard::Replicated)
        );
        assert_eq!(
            csa.shard_for(&cfg, &csa.ffn.gate_weight, 4),
            Some(Shard::Replicated)
        );
        assert_eq!(
            csa.shard_for(&cfg, &csa.ffn.expert(0).w1, 4),
            Some(Shard::ExpertParallel { dim: 0 })
        );
        assert_eq!(
            cfg.shard_for_global_tensor("head.weight"),
            Some(Shard::VocabParallel { dim: 0 })
        );
    }
}
