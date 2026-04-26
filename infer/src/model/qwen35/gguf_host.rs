use anyhow::{Result, ensure};

#[cfg(feature = "cuda")]
use qwen35_spec::Qwen35Config as Config35;

#[derive(Debug, Clone, Copy)]
pub(crate) struct Qwen35LinearGgufLayout {
    pub(crate) num_key_heads: usize,
    pub(crate) num_value_heads: usize,
    pub(crate) key_head_dim: usize,
    pub(crate) value_head_dim: usize,
    pub(crate) conv_kernel_dim: usize,
}

impl Qwen35LinearGgufLayout {
    pub(crate) fn new(
        num_key_heads: usize,
        num_value_heads: usize,
        key_head_dim: usize,
        value_head_dim: usize,
        conv_kernel_dim: usize,
    ) -> Result<Self> {
        ensure!(
            num_key_heads > 0 && num_value_heads.is_multiple_of(num_key_heads),
            "invalid Qwen3.5 linear-attention dimensions: num_key_heads={num_key_heads}, num_value_heads={num_value_heads}"
        );
        ensure!(
            key_head_dim > 0 && value_head_dim > 0 && conv_kernel_dim > 0,
            "invalid Qwen3.5 linear-attention head/kernel dims: key_head_dim={key_head_dim}, value_head_dim={value_head_dim}, conv_kernel_dim={conv_kernel_dim}"
        );
        Ok(Self {
            num_key_heads,
            num_value_heads,
            key_head_dim,
            value_head_dim,
            conv_kernel_dim,
        })
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn from_config(config: &Config35) -> Result<Self> {
        Self::new(
            config.linear_num_key_heads,
            config.linear_num_value_heads,
            config.linear_key_head_dim,
            config.linear_value_head_dim,
            config.linear_conv_kernel_dim,
        )
    }

    pub(crate) fn num_value_heads_per_key(&self) -> usize {
        self.num_value_heads / self.num_key_heads
    }
}
