use autograd::{Tape, TensorStore, optim::AdamW};
use train::{
    dataset::{CopyDataset, Dataset},
    qwen35::{LayerType, Qwen35Config, Qwen35Model},
    trainer::{clip_grad_norm, cross_entropy_loss, retained_param_and_grad_ids},
};

fn tiny_qwen35_config() -> Qwen35Config {
    Qwen35Config {
        hidden_size: 16,
        intermediate_size: 32,
        num_hidden_layers: 2,
        vocab_size: 16,
        rms_norm_eps: 1.0e-6,
        stop_token_ids: vec![15],
        bos_token_id: Some(1),
        eos_token_id: 15,
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
        rope_cache_len_hint: Some(8),
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

#[test]
fn lm_copy_loss_drops_over_three_steps() {
    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let model = Qwen35Model::new(&tiny_qwen35_config(), &mut store).expect("build qwen3.5 model");
    let params = model.all_parameter_ids();
    let mut optimizer = AdamW::new(1.0e-2, (0.9, 0.999), 1.0e-8, 0.0);
    let mut losses = Vec::with_capacity(3);

    for _ in 0..3 {
        let mut dataset = CopyDataset::with_vocab(1, 4, 7, 15, 15);
        let (inputs, targets) = dataset.sample();
        let (batch, seq_len) = dataset.batch_shape();

        tape.entries.clear();
        tape.set_enabled(true);

        let logits = model
            .forward_batch_tokens(&inputs, batch, seq_len, &mut store, &mut tape)
            .expect("forward");
        let loss = cross_entropy_loss(logits, &targets, &mut store, &mut tape).expect("loss");
        losses.push(store.to_host(loss).expect("loss value")[0]);

        optimizer.zero_grad(&params, &mut store);
        tape.backward(loss, &mut store).expect("backward");
        clip_grad_norm(&params, 1.0, &mut store);
        optimizer.step(&params, &mut store);

        tape.entries.clear();
        tape.set_enabled(true);
        let keep = retained_param_and_grad_ids(&params, &store);
        store.retain_ids(&keep);
    }

    assert!(
        losses[2] < losses[0],
        "expected loss to decrease, got {losses:?}"
    );
}
