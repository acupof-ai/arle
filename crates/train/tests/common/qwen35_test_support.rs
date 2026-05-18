#![allow(dead_code)]

use std::error::Error;

use train::qwen35::{LayerType, Qwen35Config};

pub type TestResult<T = ()> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

pub const TEST_LR: f32 = 5.0e-3;

#[allow(dead_code)]
pub fn dense_qwen35_config() -> Qwen35Config {
    let cfg = base_qwen35_config();
    cfg.validate_train_dense_full_attention_contract()
        .expect("dense helper config should satisfy scratch contract");
    cfg
}

#[allow(dead_code)]
pub fn hybrid_qwen35_config() -> Qwen35Config {
    let mut cfg = base_qwen35_config();
    cfg.rotary_dim = cfg.head_dim / 2;
    cfg.partial_rotary_factor = 0.5;
    cfg.layer_types = vec![LayerType::FullAttention, LayerType::LinearAttention];
    cfg.validate_train_lora_or_frozen_contract()
        .expect("hybrid helper config should satisfy LoRA/eval contract");
    cfg
}

#[allow(dead_code)]
pub fn tiny_qwen35_scratch_config(max_seq_len: usize) -> Qwen35Config {
    tiny_qwen35_scratch_config_with_vocab(max_seq_len, 16)
}

#[allow(dead_code)]
pub fn tiny_hybrid_qwen35_scratch_config(max_seq_len: usize) -> Qwen35Config {
    tiny_hybrid_qwen35_scratch_config_with_vocab(max_seq_len, 16)
}

#[allow(dead_code)]
pub fn tiny_qwen35_scratch_config_with_vocab(
    max_seq_len: usize,
    vocab_size: usize,
) -> Qwen35Config {
    let cfg = tiny_base_qwen35_config(max_seq_len, vocab_size);
    cfg.validate_train_scratch_contract()
        .expect("tiny dense helper config should satisfy scratch contract");
    cfg
}

#[allow(dead_code)]
pub fn tiny_hybrid_qwen35_scratch_config_with_vocab(
    max_seq_len: usize,
    vocab_size: usize,
) -> Qwen35Config {
    let mut cfg = tiny_base_qwen35_config(max_seq_len, vocab_size);
    cfg.rotary_dim = cfg.head_dim / 2;
    cfg.partial_rotary_factor = 0.5;
    cfg.linear_key_head_dim = cfg.rotary_dim;
    cfg.linear_value_head_dim = cfg.rotary_dim;
    cfg.layer_types = vec![LayerType::FullAttention, LayerType::LinearAttention];
    cfg.validate_train_scratch_contract()
        .expect("tiny hybrid helper config should satisfy scratch contract");
    cfg
}

fn base_qwen35_config() -> Qwen35Config {
    Qwen35Config {
        hidden_size: 64,
        intermediate_size: 128,
        num_hidden_layers: 2,
        vocab_size: 256,
        rms_norm_eps: 1.0e-6,
        stop_token_ids: vec![2],
        bos_token_id: Some(1),
        eos_token_id: 2,
        tie_word_embeddings: false,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 16,
        linear_num_key_heads: 4,
        linear_key_head_dim: 8,
        linear_num_value_heads: 4,
        linear_value_head_dim: 8,
        linear_conv_kernel_dim: 4,
        rope_theta: 10_000.0,

        rope_scaling: None,
        partial_rotary_factor: 1.0,
        rotary_dim: 16,
        rope_cache_len_hint: Some(32),
        layer_types: vec![LayerType::FullAttention; 2],
        num_experts: 0,
        num_experts_per_tok: 0,
        decoder_sparse_step: 1,
        moe_intermediate_size: 0,
        shared_expert_intermediate_size: 0,
        norm_topk_prob: true,
        mlp_only_layers: Vec::new(),
    }
}

fn tiny_base_qwen35_config(max_seq_len: usize, vocab_size: usize) -> Qwen35Config {
    let eos_token_id = u32::try_from(vocab_size.saturating_sub(1)).expect("tiny vocab fits in u32");
    Qwen35Config {
        hidden_size: 16,
        intermediate_size: 32,
        num_hidden_layers: 2,
        vocab_size,
        rms_norm_eps: 1.0e-6,
        stop_token_ids: vec![eos_token_id],
        bos_token_id: Some(1),
        eos_token_id,
        tie_word_embeddings: false,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 8,
        linear_num_key_heads: 2,
        linear_key_head_dim: 8,
        linear_num_value_heads: 2,
        linear_value_head_dim: 8,
        linear_conv_kernel_dim: 4,
        rope_theta: 10_000.0,

        rope_scaling: None,
        partial_rotary_factor: 1.0,
        rotary_dim: 8,
        rope_cache_len_hint: Some(max_seq_len),
        layer_types: vec![LayerType::FullAttention; 2],
        num_experts: 0,
        num_experts_per_tok: 0,
        decoder_sparse_step: 1,
        moe_intermediate_size: 0,
        shared_expert_intermediate_size: 0,
        norm_topk_prob: true,
        mlp_only_layers: Vec::new(),
    }
}
