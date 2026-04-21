#![allow(dead_code)]

use std::error::Error;

use autograd::{
    Tape, TensorId, TensorStore,
    ops::{gather_last_dim, log_softmax, mul, mul_scalar, sum},
};
use train::{
    qwen35::{LayerType, Qwen35Config, Qwen35Model},
    sft_data::TokenizedSft,
};

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

pub fn tiny_sft_examples() -> Vec<TokenizedSft> {
    vec![
        TokenizedSft {
            input_ids: vec![11, 12, 13, 14, 21, 22, 23, 24],
            labels: vec![-100, -100, -100, -100, 21, 22, 23, 24],
        },
        TokenizedSft {
            input_ids: vec![31, 32, 33, 34, 41, 42, 43, 44],
            labels: vec![-100, -100, -100, -100, 41, 42, 43, 44],
        },
        TokenizedSft {
            input_ids: vec![51, 52, 53, 54, 61, 62, 63, 64],
            labels: vec![-100, -100, -100, -100, 61, 62, 63, 64],
        },
    ]
}

pub fn train_on_example(
    model: &Qwen35Model,
    example: &TokenizedSft,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> TestResult<f32> {
    let input_len = example.input_ids.len() - 1;
    let position_ids = (0..input_len).map(|index| index as u32).collect::<Vec<_>>();
    let logits = model.forward(store, tape, &example.input_ids[..input_len], &position_ids)?;
    let loss_id = assistant_masked_loss(logits, &example.labels[1..], store, tape)?;
    let loss = store.to_host(loss_id)?[0];
    tape.backward(loss_id, store)?;
    Ok(loss)
}

pub fn assistant_masked_loss(
    logits: TensorId,
    labels: &[i32],
    store: &mut TensorStore,
    tape: &mut Tape,
) -> TestResult<TensorId> {
    let logits_shape = store
        .get(logits)
        .expect("logits tensor exists")
        .shape
        .clone();
    let prefix_elems = logits_shape[..logits_shape.len() - 1]
        .iter()
        .product::<usize>();
    assert_eq!(labels.len(), prefix_elems);

    let gather_indices = labels
        .iter()
        .map(|&label| usize::try_from(label).unwrap_or(0))
        .collect::<Vec<_>>();
    let mask_values = labels
        .iter()
        .map(|&label| if label >= 0 { 1.0 } else { 0.0 })
        .collect::<Vec<_>>();
    let valid_count = mask_values.iter().filter(|&&v| v > 0.0).count();
    assert!(valid_count > 0);

    let log_probs = log_softmax(logits, store, tape)?;
    let gathered = gather_last_dim(log_probs, &gather_indices, store, tape)?;
    let gathered_shape = store
        .get(gathered)
        .expect("gathered tensor exists")
        .shape
        .clone();
    let mask = store.from_slice(&mask_values, &gathered_shape)?;
    let masked = mul(gathered, mask, store, tape)?;
    let total = sum(masked, store, tape)?;
    Ok(mul_scalar(total, -1.0 / valid_count as f32, store, tape)?)
}

pub fn mean(values: &[f32]) -> f32 {
    values.iter().sum::<f32>() / values.len() as f32
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
