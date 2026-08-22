use super::*;

/// Applies the model-agnostic `TpContext` (`crate::tensor_parallel`) to this
/// model's dimensions; only `Qwen35Config`-reading shard sizes live here.
pub(super) trait Qwen35TpDims {
    fn validate(self, cfg: &Qwen35Config) -> Result<()>;
    fn local_attention_heads(self, cfg: &Qwen35Config) -> Result<usize>;
    fn local_key_value_heads(self, cfg: &Qwen35Config) -> Result<usize>;
    fn local_intermediate_size(self, cfg: &Qwen35Config) -> Result<usize>;
    fn local_moe_intermediate_size(self, cfg: &Qwen35Config) -> Result<usize>;
    fn local_shared_expert_intermediate_size(self, cfg: &Qwen35Config) -> Result<usize>;
    fn full_attn_q_proj_dim(self, cfg: &Qwen35Config) -> Result<usize>;
    fn full_attn_q_dim(self, cfg: &Qwen35Config) -> Result<usize>;
    fn full_attn_kv_dim(self, cfg: &Qwen35Config) -> Result<usize>;
}

impl Qwen35TpDims for TpContext {
    fn validate(self, cfg: &Qwen35Config) -> Result<()> {
        if self.world_size == 0 {
            return Err(Qwen35Error::InvalidConfig(
                "tensor-parallel world size must be non-zero",
            ));
        }
        if self.rank >= self.world_size {
            return Err(Qwen35Error::InvalidConfig(
                "tensor-parallel rank must be smaller than world size",
            ));
        }
        if !self.is_enabled() {
            return Ok(());
        }
        if !cfg
            .layer_types
            .iter()
            .all(|&layer| layer == LayerType::FullAttention)
        {
            return Err(Qwen35Error::InvalidConfig(
                "tensor-parallel train path currently supports full-attention layers only",
            ));
        }
        let _ = self.local_attention_heads(cfg)?;
        let _ = self.local_key_value_heads(cfg)?;
        let _ = self.local_intermediate_size(cfg)?;
        Ok(())
    }

    fn local_attention_heads(self, cfg: &Qwen35Config) -> Result<usize> {
        self.divide(cfg.num_attention_heads)
            .ok_or(Qwen35Error::InvalidConfig(
                "num_attention_heads must divide tensor-parallel world size",
            ))
    }

    fn local_key_value_heads(self, cfg: &Qwen35Config) -> Result<usize> {
        self.divide(cfg.num_key_value_heads)
            .ok_or(Qwen35Error::InvalidConfig(
                "num_key_value_heads must divide tensor-parallel world size",
            ))
    }

    fn local_intermediate_size(self, cfg: &Qwen35Config) -> Result<usize> {
        self.divide(cfg.intermediate_size)
            .ok_or(Qwen35Error::InvalidConfig(
                "intermediate_size must divide tensor-parallel world size",
            ))
    }

    fn local_moe_intermediate_size(self, cfg: &Qwen35Config) -> Result<usize> {
        self.divide(cfg.moe_intermediate_size)
            .ok_or(Qwen35Error::InvalidConfig(
                "moe_intermediate_size must divide tensor-parallel world size",
            ))
    }

    fn local_shared_expert_intermediate_size(self, cfg: &Qwen35Config) -> Result<usize> {
        self.divide(cfg.shared_expert_intermediate_size)
            .ok_or(Qwen35Error::InvalidConfig(
                "shared_expert_intermediate_size must divide tensor-parallel world size",
            ))
    }

    fn full_attn_q_proj_dim(self, cfg: &Qwen35Config) -> Result<usize> {
        let local_heads = self.local_attention_heads(cfg)?;
        Ok(if cfg.full_attn_gated {
            local_heads * cfg.head_dim * 2
        } else {
            local_heads * cfg.head_dim
        })
    }

    fn full_attn_q_dim(self, cfg: &Qwen35Config) -> Result<usize> {
        Ok(self.local_attention_heads(cfg)? * cfg.head_dim)
    }

    fn full_attn_kv_dim(self, cfg: &Qwen35Config) -> Result<usize> {
        Ok(self.local_key_value_heads(cfg)? * cfg.head_dim)
    }
}
