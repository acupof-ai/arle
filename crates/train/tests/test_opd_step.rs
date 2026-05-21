#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
use std::sync::Arc;

use autograd::{Tape, TensorId, TensorStore, optim::AdamW};
#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
use autograd::{backend::Backend, backend_cuda::CudaBackend};
#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
use train::lora::LoraTargetSet;
use train::{
    lora::LoraConfig,
    opd::{OpdError, OpdStepConfig, opd_step},
    qwen35::{LayerType, Qwen35Config, Qwen35Model},
};

fn live_tensor_count(store: &TensorStore) -> usize {
    store.tensors.iter().filter(|slot| slot.is_some()).count()
}

fn perturb_student_params(store: &mut TensorStore, params: &[TensorId]) {
    for (param_index, &param_id) in params.iter().enumerate() {
        let Some(tensor) = store.get_mut(param_id) else {
            continue;
        };
        if !tensor.requires_grad {
            continue;
        }
        for (value_index, value) in tensor.data.iter_mut().enumerate() {
            let offset = ((param_index + value_index) % 7) as f32 - 3.0;
            *value += offset * 1.0e-4;
        }
    }
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
fn opd_step_updates_student_without_mutating_teacher() {
    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let cfg = tiny_qwen35_config();

    let teacher = Qwen35Model::new_for_eval(&cfg, &mut store).expect("build teacher");
    let student = Qwen35Model::new(&cfg, &mut store).expect("build student");
    let student_params = student.all_parameter_ids();
    perturb_student_params(&mut store, &student_params);

    let teacher_before = teacher
        .all_parameter_ids()
        .into_iter()
        .map(|id| {
            let tensor = store.get(id).expect("teacher tensor");
            (id, tensor.data.clone())
        })
        .collect::<Vec<_>>();
    let student_before = student_params
        .iter()
        .copied()
        .filter_map(|id| {
            let tensor = store.get(id)?;
            tensor.requires_grad.then(|| (id, tensor.data.clone()))
        })
        .collect::<Vec<_>>();
    assert!(
        !student_before.is_empty(),
        "scratch OPD student must expose trainable parameters"
    );

    let mut optimizer = AdamW::new(1.0e-2, (0.9, 0.999), 1.0e-8, 0.0);
    let outcome = opd_step(
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
    .expect("opd_step should update a perturbed student against a frozen teacher");
    assert!(outcome.loss.is_finite());

    for (id, before) in teacher_before {
        let tensor = store.get(id).expect("teacher tensor survives cleanup");
        assert_eq!(
            tensor.data, before,
            "opd_step must not mutate frozen teacher tensor {id}"
        );
        assert!(
            !tensor.requires_grad,
            "teacher tensor {id} must remain frozen after opd_step"
        );
        assert!(
            tensor.grad.is_none(),
            "teacher tensor {id} must not receive gradients"
        );
    }

    let student_changed = student_before.iter().any(|(id, before)| {
        store
            .get(*id)
            .expect("student tensor survives cleanup")
            .data
            != *before
    });
    assert!(
        student_changed,
        "opd_step should update at least one trainable student tensor"
    );
}

#[test]
fn opd_step_error_after_rollout_cleans_tape_and_temporaries() {
    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let mut cfg = tiny_qwen35_config();
    cfg.rope_cache_len_hint = Some(2);

    let teacher = Qwen35Model::new_for_eval(&cfg, &mut store).expect("build teacher");
    let student = Qwen35Model::new(&cfg, &mut store).expect("build student");
    let student_params = student.all_parameter_ids();
    let live_before = live_tensor_count(&store);
    let mut optimizer = AdamW::new(1.0e-3, (0.9, 0.999), 1.0e-8, 0.0);

    let err = opd_step(
        &student,
        &teacher,
        &[1],
        OpdStepConfig {
            rollout_len: 2,
            grad_clip: 1.0,
        },
        &student_params,
        &mut optimizer,
        &mut store,
        &mut tape,
    )
    .expect_err("teacher scoring should fail after student rollout grows past rope cache");

    let OpdError::InvalidInput(message) = err else {
        panic!("expected InvalidInput, got {err:?}");
    };
    assert!(message.contains("OPD teacher scoring"));
    assert!(
        tape.enabled,
        "opd_step failure cleanup must re-enable tape for the next call"
    );
    assert!(
        tape.entries.is_empty(),
        "opd_step failure cleanup must clear stale tape entries"
    );
    assert_eq!(
        live_tensor_count(&store),
        live_before,
        "opd_step failure cleanup must prune rollout/forward temporaries"
    );
}

#[test]
fn opd_step_with_lora_adapter_params_retains_frozen_student_base() {
    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let cfg = tiny_qwen35_config();

    let teacher = Qwen35Model::new_for_eval(&cfg, &mut store).expect("build teacher");
    let student = Qwen35Model::new_with_lora(
        &cfg,
        Some(LoraConfig {
            rank: 2,
            alpha: 4.0,
        }),
        &mut store,
    )
    .expect("build lora student");
    let adapter_params = student
        .adapter_name_map()
        .values()
        .copied()
        .collect::<Vec<_>>();
    assert!(
        !adapter_params.is_empty(),
        "LoRA student should expose adapter tensors"
    );
    let frozen_base_params = student
        .param_name_map()
        .values()
        .copied()
        .filter(|id| !store.get(*id).expect("student base param").requires_grad)
        .collect::<Vec<_>>();
    assert!(
        !frozen_base_params.is_empty(),
        "LoRA student should keep frozen base tensors"
    );

    let mut optimizer = AdamW::new(1.0e-3, (0.9, 0.999), 1.0e-8, 0.0);
    let prompt_ids: Vec<u32> = vec![1, 3, 8];
    let step_cfg = OpdStepConfig {
        rollout_len: 1,
        grad_clip: 1.0,
    };

    for _ in 0..2 {
        let outcome = opd_step(
            &student,
            &teacher,
            &prompt_ids,
            step_cfg,
            &adapter_params,
            &mut optimizer,
            &mut store,
            &mut tape,
        )
        .expect("LoRA adapter-only opd_step should retain base weights");
        assert!(outcome.loss.is_finite());
        for &param_id in &frozen_base_params {
            assert!(
                store.get(param_id).is_some(),
                "cleanup must retain frozen student base tensor {param_id} \
                 even when only adapter ids are optimized"
            );
        }
    }
}

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
#[test]
fn lora_opd_step_cuda_matches_cpu_loss() {
    fn run_once(cuda: bool) -> f32 {
        let mut tape = Tape::new();
        let cfg = tiny_qwen35_config();
        let (mut store, mut optimizer) = if cuda {
            let backend: Arc<dyn Backend + Send + Sync> =
                Arc::new(CudaBackend::new(0).expect("create CUDA backend"));
            (
                TensorStore::with_backend(backend.clone()),
                AdamW::new_with_device(1.0e-4, (0.9, 0.999), 1.0e-8, 0.0, backend),
            )
        } else {
            (
                TensorStore::default(),
                AdamW::new(1.0e-4, (0.9, 0.999), 1.0e-8, 0.0),
            )
        };

        let teacher = Qwen35Model::new_for_eval(&cfg, &mut store).expect("build teacher");
        let student = Qwen35Model::new_lora_from_base(
            &teacher,
            LoraConfig {
                rank: 2,
                alpha: 4.0,
            },
            LoraTargetSet::AttentionQv,
            &mut store,
        )
        .expect("build shared-base LoRA student");
        let adapter_params = student
            .adapter_name_map()
            .values()
            .copied()
            .collect::<Vec<_>>();
        perturb_student_params(&mut store, &adapter_params);

        opd_step(
            &student,
            &teacher,
            &[1, 3, 8],
            OpdStepConfig {
                rollout_len: 2,
                grad_clip: 1.0,
            },
            &adapter_params,
            &mut optimizer,
            &mut store,
            &mut tape,
        )
        .expect("LoRA OPD step should run")
        .loss
    }

    let cpu = run_once(false);
    let cuda = run_once(true);
    let denom = cpu.abs().max(cuda.abs()).max(1.0e-12);
    let relerr = (cpu - cuda).abs() / denom;
    eprintln!("lora_opd_step_cuda_matches_cpu_loss cpu={cpu:e} cuda={cuda:e} relerr={relerr:e}");
    assert!(
        relerr <= 1.0e-4,
        "LoRA OPD CPU/CUDA loss relerr {relerr:e} exceeds 1e-4 (cpu={cpu:e}, cuda={cuda:e})"
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

#[test]
fn opd_step_rejects_trainable_teacher_model() {
    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let cfg = tiny_qwen35_config();

    let teacher = Qwen35Model::new(&cfg, &mut store).expect("build trainable teacher");
    let student = Qwen35Model::new(&cfg, &mut store).expect("build student");
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
    .expect_err("trainable teacher should be rejected before rollout");

    let OpdError::InvalidInput(message) = err else {
        panic!("expected InvalidInput, got {err:?}");
    };
    assert!(message.contains("teacher_params"));
    assert!(message.contains("requires_grad=true"));
    assert!(message.contains("new_for_eval"));
}

#[test]
fn opd_step_rejects_teacher_param_mixed_into_student_params() {
    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let cfg = tiny_qwen35_config();

    let teacher = Qwen35Model::new_for_eval(&cfg, &mut store).expect("build teacher");
    let student = Qwen35Model::new(&cfg, &mut store).expect("build student");
    let mut mixed_params = student.all_parameter_ids();
    mixed_params.push(
        teacher
            .all_parameter_ids()
            .into_iter()
            .next()
            .expect("teacher param"),
    );
    let mut optimizer = AdamW::new(1.0e-3, (0.9, 0.999), 1.0e-8, 0.0);

    let err = opd_step(
        &student,
        &teacher,
        &[1, 3, 8],
        OpdStepConfig {
            rollout_len: 2,
            grad_clip: 1.0,
        },
        &mixed_params,
        &mut optimizer,
        &mut store,
        &mut tape,
    )
    .expect_err("teacher parameter ids must not be accepted as student params");

    let OpdError::InvalidInput(message) = err else {
        panic!("expected InvalidInput, got {err:?}");
    };
    assert!(message.contains("belongs to the frozen"));
    assert!(message.contains("teacher weights must not be optimized"));
}

#[test]
fn opd_step_rejects_short_rope_cache_with_actionable_error() {
    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let mut cfg = tiny_qwen35_config();
    cfg.rope_cache_len_hint = Some(2);

    let teacher = Qwen35Model::new_for_eval(&cfg, &mut store).expect("build teacher");
    let student = Qwen35Model::new(&cfg, &mut store).expect("build student");
    let student_params = student.all_parameter_ids();
    let mut optimizer = AdamW::new(1.0e-3, (0.9, 0.999), 1.0e-8, 0.0);

    let err = opd_step(
        &student,
        &teacher,
        &[1, 3, 8],
        OpdStepConfig {
            rollout_len: 0,
            grad_clip: 1.0,
        },
        &student_params,
        &mut optimizer,
        &mut store,
        &mut tape,
    )
    .expect_err("rope cache shorter than prompt must fail with OPD context");

    let OpdError::InvalidInput(message) = err else {
        panic!("expected InvalidInput, got {err:?}");
    };
    assert!(message.contains("OPD teacher scoring"));
    assert!(message.contains("rope_cache_len_hint"));
    assert!(message.contains("prompt length plus rollout length"));
}
