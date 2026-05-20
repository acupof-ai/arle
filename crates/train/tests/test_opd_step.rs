use autograd::{Tape, TensorStore, optim::AdamW};
use train::{
    opd::{OpdError, OpdStepConfig, opd_step},
    qwen35::{LayerType, Qwen35Config, Qwen35Model},
};

fn live_tensor_count(store: &TensorStore) -> usize {
    store.tensors.iter().filter(|slot| slot.is_some()).count()
}

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
    }
}

/// End-to-end opd_step smoke: rollout → teacher forward → student forward
/// → KL → backward → AdamW step. Teacher is built with `new_for_eval`, so
/// its deterministic scratch weights match the student initializer while
/// carrying `requires_grad = false`. The test runs `opd_step` and verifies
/// the loss is finite + the call returns the expected rollout length. The
/// full loss-decrease curve is exercised on real GPU smoke (see
/// `arle train opd --self-distill-smoke` once wired).
#[test]
fn opd_step_runs_end_to_end() {
    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let cfg = tiny_qwen35_config();

    let teacher = Qwen35Model::new_for_eval(&cfg, &mut store).expect("build teacher");
    let student = Qwen35Model::new(&cfg, &mut store).expect("build student");
    let student_params = student.all_parameter_ids();

    let mut optimizer = AdamW::new(1.0e-3, (0.9, 0.999), 1.0e-8, 0.0);
    let prompt_ids: Vec<u32> = vec![1, 3, 8]; // arbitrary chat-style prompt prefix

    let outcome = opd_step(
        &student,
        &teacher,
        &prompt_ids,
        OpdStepConfig {
            rollout_len: 2,
            grad_clip: 1.0,
        },
        &student_params,
        &mut optimizer,
        &mut store,
        &mut tape,
    )
    .expect("opd_step runs without panic");

    assert_eq!(outcome.rollout_len, prompt_ids.len() + 2);
    assert!(
        outcome.loss.is_finite(),
        "opd_step loss should be finite, got {}",
        outcome.loss
    );
}

#[test]
fn opd_step_prunes_ephemeral_tensors_between_steps() {
    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let cfg = tiny_qwen35_config();

    let teacher = Qwen35Model::new_for_eval(&cfg, &mut store).expect("build teacher");
    let student = Qwen35Model::new(&cfg, &mut store).expect("build student");
    let student_params = student.all_parameter_ids();

    let mut optimizer = AdamW::new(1.0e-3, (0.9, 0.999), 1.0e-8, 0.0);
    let prompt_ids: Vec<u32> = vec![1, 3, 8];
    let step_cfg = OpdStepConfig {
        rollout_len: 2,
        grad_clip: 1.0,
    };

    let mut live_counts = Vec::with_capacity(3);
    for _ in 0..3 {
        let outcome = opd_step(
            &student,
            &teacher,
            &prompt_ids,
            step_cfg,
            &student_params,
            &mut optimizer,
            &mut store,
            &mut tape,
        )
        .expect("opd_step runs without panic");
        assert!(outcome.loss.is_finite());
        live_counts.push(live_tensor_count(&store));
    }

    assert_eq!(
        live_counts[1], live_counts[0],
        "second step should reuse the same retained tensor set, got {live_counts:?}"
    );
    assert_eq!(
        live_counts[2], live_counts[0],
        "third step should not accumulate rollout/forward temporaries, got {live_counts:?}"
    );
}

#[test]
fn opd_step_rejects_empty_prompt_with_actionable_error() {
    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let cfg = tiny_qwen35_config();

    let teacher = Qwen35Model::new_for_eval(&cfg, &mut store).expect("build teacher");
    let student = Qwen35Model::new(&cfg, &mut store).expect("build student");
    let student_params = student.all_parameter_ids();
    let mut optimizer = AdamW::new(1.0e-3, (0.9, 0.999), 1.0e-8, 0.0);

    let err = opd_step(
        &student,
        &teacher,
        &[],
        OpdStepConfig {
            rollout_len: 2,
            grad_clip: 1.0,
        },
        &student_params,
        &mut optimizer,
        &mut store,
        &mut tape,
    )
    .expect_err("empty prompt should be rejected before rollout");

    let OpdError::InvalidInput(message) = err else {
        panic!("expected InvalidInput, got {err:?}");
    };
    assert!(message.contains("non-empty prompt_ids"));
    assert!(message.contains("2026-05-18-opd-only-pivot.md"));
}

#[test]
fn opd_step_rejects_prompt_token_outside_student_vocab() {
    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let cfg = tiny_qwen35_config();

    let teacher = Qwen35Model::new_for_eval(&cfg, &mut store).expect("build teacher");
    let student = Qwen35Model::new(&cfg, &mut store).expect("build student");
    let student_params = student.all_parameter_ids();
    let mut optimizer = AdamW::new(1.0e-3, (0.9, 0.999), 1.0e-8, 0.0);

    let err = opd_step(
        &student,
        &teacher,
        &[1, 16],
        OpdStepConfig {
            rollout_len: 2,
            grad_clip: 1.0,
        },
        &student_params,
        &mut optimizer,
        &mut store,
        &mut tape,
    )
    .expect_err("out-of-vocab prompt token should be rejected before rollout");

    let OpdError::InvalidInput(message) = err else {
        panic!("expected InvalidInput, got {err:?}");
    };
    assert!(message.contains("prompt_ids[1]"));
    assert!(message.contains("vocab_size=16"));
    assert!(message.contains("tokenizer"));
    assert!(message.contains("2026-05-18-opd-only-pivot.md"));
}

#[test]
fn opd_step_rejects_teacher_student_vocab_mismatch() {
    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let student_cfg = tiny_qwen35_config();
    let mut teacher_cfg = tiny_qwen35_config();
    teacher_cfg.vocab_size = 12;
    teacher_cfg.stop_token_ids = vec![11];
    teacher_cfg.eos_token_id = 11;

    let teacher = Qwen35Model::new_for_eval(&teacher_cfg, &mut store).expect("build teacher");
    let student = Qwen35Model::new(&student_cfg, &mut store).expect("build student");
    let student_params = student.all_parameter_ids();
    let mut optimizer = AdamW::new(1.0e-3, (0.9, 0.999), 1.0e-8, 0.0);

    let err = opd_step(
        &student,
        &teacher,
        &[1, 3, 8],
        OpdStepConfig {
            rollout_len: 2,
            grad_clip: 1.0,
        },
        &student_params,
        &mut optimizer,
        &mut store,
        &mut tape,
    )
    .expect_err("vocab mismatch should be rejected before rollout");

    let OpdError::InvalidInput(message) = err else {
        panic!("expected InvalidInput, got {err:?}");
    };
    assert!(message.contains("teacher.config().vocab_size=12"));
    assert!(message.contains("student.config().vocab_size=16"));
    assert!(message.contains("tokenizer"));
    assert!(message.contains("2026-05-18-opd-only-pivot.md"));
}
