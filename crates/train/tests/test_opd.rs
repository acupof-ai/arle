use autograd::{Tape, Tensor, TensorStore, optim::AdamW};
use train::{
    loss::{KlDirection, kl_distill_loss},
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
        full_attn_gated: true,
            output_gate_type: "sigmoid".to_string(),
    }
}

/// Smoke for `kl_distill_loss`: a fixed soft target acts as the "teacher
/// logits" with mass concentrated on a single vocab token; the student
/// should learn to match it (KL ↓ across steps).
///
/// We use a fixed target tensor with `requires_grad = false` rather than
/// a second `Qwen35Model` here so the assertion only depends on
/// `kl_distill_loss` + Qwen35Model backward; the runtime-coupled rollout
/// path is exercised separately by `crates/train/src/opd.rs::opd_step`.
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
            1.0,
            KlDirection::Forward,
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

/// Regression guard for `errors/2026-06-16-opd-kl-vocab-reduction-lr-collapse.md`.
/// The forward-KL student-logit gradient is `(s_j - t_j)/positions` under
/// `batchmean` vs `.../(positions*vocab)` under the buggy reduction — so this
/// fails by exactly `vocab×` if `kl_batchmean_scale` is dropped.
#[test]
fn kl_distill_gradient_is_batchmean_scaled_not_vocab_collapsed() {
    let positions = 4usize;
    let vocab = 16usize;

    // Trainable student logits (the leaf we read grads from) + frozen teacher.
    let student_data: Vec<f32> = (0..positions * vocab)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
        .collect();
    let teacher_data: Vec<f32> = (0..positions * vocab)
        .map(|i| if i % vocab == 5 { 3.0 } else { 0.0 })
        .collect();

    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    tape.set_enabled(true);

    let student = store.alloc(
        Tensor::new(student_data.clone(), vec![positions, vocab], true).expect("student logits"),
    );
    let teacher = store
        .alloc(Tensor::new(teacher_data, vec![positions, vocab], false).expect("teacher logits"));

    let loss = kl_distill_loss(
        student,
        teacher,
        positions,
        1.0,
        KlDirection::Forward,
        &mut store,
        &mut tape,
    )
    .expect("kl loss");
    tape.backward(loss, &mut store).expect("backward");

    let grad_id = store
        .get(student)
        .and_then(|t| t.grad)
        .expect("student grad");
    let grads = store.to_host(grad_id).expect("grad host");

    // Analytic batchmean gradient: d/ds_j = (softmax(s)_j - t_j)/positions.
    let mut expected = vec![0.0f32; positions * vocab];
    for p in 0..positions {
        let s = &student_data[p * vocab..(p + 1) * vocab];
        let max = s.iter().cloned().fold(f32::MIN, f32::max);
        let exps: Vec<f32> = s.iter().map(|v| (v - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let t_logits: Vec<f32> = (0..vocab).map(|j| if j == 5 { 3.0 } else { 0.0 }).collect();
        let t_max = t_logits.iter().cloned().fold(f32::MIN, f32::max);
        let t_exps: Vec<f32> = t_logits.iter().map(|v| (v - t_max).exp()).collect();
        let t_sum: f32 = t_exps.iter().sum();
        for j in 0..vocab {
            let s_prob = exps[j] / sum;
            let t_prob = t_exps[j] / t_sum;
            expected[p * vocab + j] = (s_prob - t_prob) / positions as f32;
        }
    }

    let max_abs_grad = grads.iter().cloned().fold(0.0f32, |a, g| a.max(g.abs()));
    // Collapsed reduction would be ~1/vocab of this — below any grad-clip.
    assert!(
        max_abs_grad > 1.0e-3,
        "grad magnitude {max_abs_grad:.3e} looks vocab-collapsed (batchmean scale dropped?)"
    );
    for (got, want) in grads.iter().zip(expected.iter()) {
        let tol = 1.0e-4 + want.abs() * 1.0e-3;
        assert!(
            (got - want).abs() <= tol,
            "grad {got:.6e} != analytic batchmean grad {want:.6e} (tol {tol:.2e}); \
             a vocab× mismatch means the kl_batchmean_scale correction regressed"
        );
    }
}
