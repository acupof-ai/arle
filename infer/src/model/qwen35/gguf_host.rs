use anyhow::{Result, ensure};
use half::bf16;

use crate::gguf::{
    GgufFile, HostTensor, load_matrix_v_reorder_cols_bf16_host,
    load_matrix_v_reorder_rows_bf16_host, load_qwen35_a_log_f32_host, load_qwen35_conv1d_bf16_host,
    load_qwen35_qkv_matrix_bf16_host, load_vector_f32_host, load_vector_v_reorder_bf16_host,
};
#[cfg(feature = "metal")]
use crate::gguf::{find_tensor_name, load_matrix_bf16_host, load_vector_bf16_host};
#[cfg(feature = "metal")]
use qwen35_spec::LayerType;
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

#[cfg(feature = "metal")]
#[derive(Debug, Clone)]
pub(crate) struct Qwen35GgufFullAttentionHost {
    pub(crate) q_proj: HostTensor<bf16>,
    pub(crate) k_proj: HostTensor<bf16>,
    pub(crate) v_proj: HostTensor<bf16>,
    pub(crate) o_proj: HostTensor<bf16>,
    pub(crate) q_norm: HostTensor<bf16>,
    pub(crate) k_norm: HostTensor<bf16>,
}

#[derive(Debug, Clone)]
pub(crate) struct Qwen35GgufLinearAttentionHost {
    pub(crate) in_proj_qkv: HostTensor<bf16>,
    pub(crate) in_proj_z: HostTensor<bf16>,
    pub(crate) in_proj_b: HostTensor<bf16>,
    pub(crate) in_proj_a: HostTensor<bf16>,
    pub(crate) conv1d_weight: HostTensor<bf16>,
    pub(crate) dt_bias: HostTensor<bf16>,
    pub(crate) a_log: HostTensor<f32>,
    pub(crate) norm_weight: HostTensor<f32>,
    pub(crate) out_proj: HostTensor<bf16>,
}

#[cfg(feature = "metal")]
#[derive(Debug, Clone)]
pub(crate) enum Qwen35GgufAttentionHost {
    Full(Qwen35GgufFullAttentionHost),
    Linear(Qwen35GgufLinearAttentionHost),
}

#[cfg(feature = "metal")]
#[derive(Debug, Clone)]
pub(crate) struct Qwen35GgufDenseMlpHost {
    pub(crate) gate: HostTensor<bf16>,
    pub(crate) up: HostTensor<bf16>,
    pub(crate) down: HostTensor<bf16>,
}

#[cfg(feature = "metal")]
#[derive(Debug, Clone)]
pub(crate) struct Qwen35GgufBlockHost {
    pub(crate) input_layernorm: HostTensor<bf16>,
    pub(crate) attention: Qwen35GgufAttentionHost,
    pub(crate) post_attention_layernorm: HostTensor<bf16>,
    pub(crate) mlp: Qwen35GgufDenseMlpHost,
}

#[cfg(feature = "metal")]
#[derive(Debug, Clone)]
pub(crate) struct Qwen35GgufHostWeights {
    pub(crate) embed_tokens: HostTensor<bf16>,
    pub(crate) norm: HostTensor<bf16>,
    pub(crate) lm_head: Option<HostTensor<bf16>>,
    pub(crate) layers: Vec<Qwen35GgufBlockHost>,
}

pub(crate) fn load_qwen35_linear_attention_host(
    gguf: &GgufFile,
    attn_prefix: &str,
    layout: Qwen35LinearGgufLayout,
) -> Result<Qwen35GgufLinearAttentionHost> {
    let num_v_per_k = layout.num_value_heads_per_key();
    Ok(Qwen35GgufLinearAttentionHost {
        in_proj_qkv: load_qwen35_qkv_matrix_bf16_host(
            gguf,
            &format!("{attn_prefix}.in_proj_qkv.weight"),
            layout.num_key_heads,
            num_v_per_k,
            layout.key_head_dim,
            layout.value_head_dim,
        )?,
        in_proj_z: load_matrix_v_reorder_rows_bf16_host(
            gguf,
            &format!("{attn_prefix}.in_proj_z.weight"),
            layout.num_key_heads,
            num_v_per_k,
            layout.value_head_dim,
        )?,
        in_proj_b: load_matrix_v_reorder_rows_bf16_host(
            gguf,
            &format!("{attn_prefix}.in_proj_b.weight"),
            layout.num_key_heads,
            num_v_per_k,
            1,
        )?,
        in_proj_a: load_matrix_v_reorder_rows_bf16_host(
            gguf,
            &format!("{attn_prefix}.in_proj_a.weight"),
            layout.num_key_heads,
            num_v_per_k,
            1,
        )?,
        conv1d_weight: load_qwen35_conv1d_bf16_host(
            gguf,
            &format!("{attn_prefix}.conv1d.weight"),
            layout.num_key_heads,
            layout.key_head_dim,
            layout.num_value_heads,
            layout.value_head_dim,
            layout.conv_kernel_dim,
        )?,
        dt_bias: load_vector_v_reorder_bf16_host(
            gguf,
            &format!("{attn_prefix}.dt_bias"),
            layout.num_key_heads,
            num_v_per_k,
            1,
        )?,
        a_log: load_qwen35_a_log_f32_host(
            gguf,
            &format!("{attn_prefix}.a_log"),
            layout.num_key_heads,
            num_v_per_k,
        )?,
        norm_weight: load_vector_f32_host(gguf, &format!("{attn_prefix}.norm.weight"))?,
        out_proj: load_matrix_v_reorder_cols_bf16_host(
            gguf,
            &format!("{attn_prefix}.out_proj.weight"),
            layout.num_key_heads,
            num_v_per_k,
            layout.value_head_dim,
        )?,
    })
}

#[cfg(feature = "metal")]
pub(crate) fn load_qwen35_gguf_host_weights_with_layout(
    gguf: &GgufFile,
    layer_types: &[LayerType],
    num_hidden_layers: usize,
    layout: Qwen35LinearGgufLayout,
) -> Result<Qwen35GgufHostWeights> {
    ensure!(
        layer_types.len() == num_hidden_layers,
        "Qwen3.5 GGUF host loading got {} layer types for {num_hidden_layers} hidden layers",
        layer_types.len()
    );
    let prefix = "model";
    let mut layers = Vec::with_capacity(num_hidden_layers);

    for (i, layer_type) in layer_types.iter().copied().enumerate() {
        let layer_prefix = format!("{prefix}.layers.{i}");
        let attention = match layer_type {
            LayerType::FullAttention => {
                Qwen35GgufAttentionHost::Full(Qwen35GgufFullAttentionHost {
                    q_proj: load_matrix_bf16_host(
                        gguf,
                        &format!("{layer_prefix}.self_attn.q_proj.weight"),
                    )?,
                    k_proj: load_matrix_bf16_host(
                        gguf,
                        &format!("{layer_prefix}.self_attn.k_proj.weight"),
                    )?,
                    v_proj: load_matrix_bf16_host(
                        gguf,
                        &format!("{layer_prefix}.self_attn.v_proj.weight"),
                    )?,
                    o_proj: load_matrix_bf16_host(
                        gguf,
                        &format!("{layer_prefix}.self_attn.o_proj.weight"),
                    )?,
                    q_norm: load_vector_bf16_host(
                        gguf,
                        &format!("{layer_prefix}.self_attn.q_norm.weight"),
                    )?,
                    k_norm: load_vector_bf16_host(
                        gguf,
                        &format!("{layer_prefix}.self_attn.k_norm.weight"),
                    )?,
                })
            }
            LayerType::LinearAttention => {
                let attn_prefix = format!("{layer_prefix}.linear_attn");
                Qwen35GgufAttentionHost::Linear(load_qwen35_linear_attention_host(
                    gguf,
                    &attn_prefix,
                    layout,
                )?)
            }
        };

        layers.push(Qwen35GgufBlockHost {
            input_layernorm: load_vector_bf16_host(
                gguf,
                &format!("{layer_prefix}.input_layernorm.weight"),
            )?,
            attention,
            post_attention_layernorm: load_vector_bf16_host(
                gguf,
                &format!("{layer_prefix}.post_attention_layernorm.weight"),
            )?,
            mlp: Qwen35GgufDenseMlpHost {
                gate: load_matrix_bf16_host(gguf, &format!("{layer_prefix}.mlp.gate_proj.weight"))?,
                up: load_matrix_bf16_host(gguf, &format!("{layer_prefix}.mlp.up_proj.weight"))?,
                down: load_matrix_bf16_host(gguf, &format!("{layer_prefix}.mlp.down_proj.weight"))?,
            },
        });
    }

    Ok(Qwen35GgufHostWeights {
        embed_tokens: load_matrix_bf16_host(gguf, &format!("{prefix}.embed_tokens.weight"))?,
        norm: load_vector_bf16_host(gguf, &format!("{prefix}.norm.weight"))?,
        lm_head: load_optional_lm_head_host(gguf)?,
        layers,
    })
}

#[cfg(feature = "metal")]
fn load_optional_lm_head_host(gguf: &GgufFile) -> Result<Option<HostTensor<bf16>>> {
    if find_tensor_name(gguf, "lm_head.weight").is_err() {
        return Ok(None);
    }
    Ok(Some(load_matrix_bf16_host(gguf, "lm_head.weight")?))
}
