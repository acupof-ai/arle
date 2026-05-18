use autograd::{Tape, Tensor, TensorStore, optim::AdamW};
use train::{
    loss::kl_distill_loss,
    qwen35::{LayerType, Qwen35Config, Qwen35Model},
    trainer::{clip_grad_norm, retained_param_and_grad_ids},
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
        rope_scaling: None,
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

/// Smoke for `kl_distill_loss`: a fixed soft target acts as the "teacher
/// logits" with mass concentrated on a single vocab token; the student
/// should learn to match it (KL ↓ across steps). We use a fixed target
/// tensor with `requires_grad = false` rather than a second
/// `Qwen35Model` here so the assertion only depends on `kl_distill_loss`
/// + Qwen35Model backward; the runtime-coupled rollout path is exercised
/// separately by `crates/train/src/opd.rs::opd_step`.
#[test]
fn kl_distill_loss_drops_over_three_steps() {
    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let cfg = tiny_qwen35_config();
    let model = Qwen35Model::new(&cfg, &mut store).expect("build student");
    let params = model.all_parameter_ids();
    let mut optimizer = AdamW::new(1.0e-2, (0.9, 0.999), 1.0e-8, 0.0);

    let inputs: Vec<usize> = vec![3, 8, 15, 3];
    let batch = 1;
    let seq_len = inputs.len();
    let vocab = cfg.vocab_size;

    // Soft target with mass concentrated on token 5 across all positions.
    let teacher_logits_data: Vec<f32> = (0..seq_len * vocab)
        .map(|i| if i % vocab == 5 { 5.0 } else { 0.0 })
        .collect();

    let mut losses = Vec::with_capacity(3);

    for _ in 0..3 {
        tape.entries.clear();
        tape.set_enabled(true);

        let teacher_logits = store.alloc(
            Tensor::new(
                teacher_logits_data.clone(),
                vec![batch, seq_len, vocab],
                false,
            )
            .expect("teacher logits"),
        );

        let student_logits = model
            .forward_batch_tokens(&inputs, batch, seq_len, &mut store, &mut tape)
            .expect("student forward");
        let loss = kl_distill_loss(
            student_logits,
            teacher_logits,
            seq_len,
            &mut store,
            &mut tape,
        )
        .expect("kl loss");
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
        "expected KL distill loss to decrease, got {losses:?}"
    );
}
