#![cfg(feature = "metal")]

use std::sync::Arc;

use autograd::{
    Backend, Tape, TensorStore, adamw_state::AdamWState, backend_metal::MetalBackend, optim::AdamW,
};
use tempfile::tempdir;
use train::{
    causal_lm::{
        build_registry, live_tensor_ids, retained_ids, save_materialized_registry,
        trainable_param_name_map, trainable_params,
    },
    grpo::{GrpoConfig, group_advantages, grpo_loss},
    lora::LoraConfig,
    loss::cross_entropy_loss,
    qwen35::Qwen35Model,
    rollout::Trajectory,
};

mod common;

use common::qwen35_test_support::{
    TEST_LR, TestResult, hybrid_qwen35_config, tiny_hybrid_qwen35_scratch_config_with_vocab,
};

const ABS_TOL: f32 = 5.0e-4;
const REL_TOL: f32 = 5.0e-4;

#[derive(Debug)]
struct StepSnapshot {
    losses: Vec<f32>,
    reload_logits: Vec<f32>,
    named_params: Vec<(String, Vec<f32>)>,
    optim_state: AdamWState,
}

#[test]
fn qwen35_hybrid_scratch_step_matches_cpu_on_metal() -> TestResult {
    let cfg = tiny_hybrid_qwen35_scratch_config_with_vocab(8, 32);
    let cpu = run_hybrid_scratch_steps(None, &cfg, &[vec![1, 2, 3, 4]], 1)?;
    let metal =
        run_hybrid_scratch_steps(Some(Arc::new(MetalBackend)), &cfg, &[vec![1, 2, 3, 4]], 1)?;
    assert_snapshot_close("hybrid scratch", &cpu, &metal);
    Ok(())
}

#[test]
fn qwen35_hybrid_grpo_step_matches_cpu_on_metal() -> TestResult {
    let cfg = hybrid_qwen35_config();
    let lora = LoraConfig {
        rank: 2,
        alpha: 4.0,
    };
    let trajectories = fixed_grpo_trajectories();
    let cpu = run_hybrid_grpo_steps(None, &cfg, lora, &trajectories, 1)?;
    let metal = run_hybrid_grpo_steps(Some(Arc::new(MetalBackend)), &cfg, lora, &trajectories, 1)?;
    assert_snapshot_close("hybrid grpo", &cpu, &metal);
    Ok(())
}

#[test]
fn qwen35_hybrid_scratch_longer_sequence_steps_match_cpu_on_metal() -> TestResult {
    let cfg = tiny_hybrid_qwen35_scratch_config_with_vocab(16, 64);
    let examples = vec![
        vec![1, 2, 3, 4, 5, 6, 7, 8],
        vec![9, 10, 11, 12, 13, 14, 15, 16],
        vec![17, 18, 19, 20, 21, 22, 23, 24],
    ];
    let cpu = run_hybrid_scratch_steps(None, &cfg, &examples, 4)?;
    let metal = run_hybrid_scratch_steps(Some(Arc::new(MetalBackend)), &cfg, &examples, 4)?;
    assert_snapshot_close("hybrid scratch long", &cpu, &metal);
    Ok(())
}

#[test]
fn qwen35_hybrid_grpo_longer_sequence_steps_match_cpu_on_metal() -> TestResult {
    let mut cfg = hybrid_qwen35_config();
    cfg.rope_cache_len_hint = Some(16);
    let lora = LoraConfig {
        rank: 2,
        alpha: 4.0,
    };
    let trajectories = fixed_grpo_trajectories_longer();
    let cpu = run_hybrid_grpo_steps(None, &cfg, lora, &trajectories, 3)?;
    let metal = run_hybrid_grpo_steps(Some(Arc::new(MetalBackend)), &cfg, lora, &trajectories, 3)?;
    assert_snapshot_close("hybrid grpo long", &cpu, &metal);
    Ok(())
}

fn run_hybrid_scratch_steps(
    backend: Option<Arc<dyn Backend>>,
    cfg: &train::qwen35::Qwen35Config,
    examples: &[Vec<u32>],
    steps: usize,
) -> TestResult<StepSnapshot> {
    let mut store = match backend {
        Some(backend) => TensorStore::with_backend(backend),
        None => TensorStore::default(),
    };
    let model = Qwen35Model::new(cfg, &mut store)?;
    let params = trainable_params(&model, &store);
    let param_names = trainable_param_name_map(&model, &store);
    let model_ids = live_tensor_ids(&store);
    let mut optimizer = AdamW::new(TEST_LR, (0.9, 0.999), 1.0e-8, 0.0);
    let mut tape = Tape::new();
    let mut losses = Vec::with_capacity(steps);

    for step in 0..steps {
        let example = &examples[step % examples.len()];
        let input_ids = &example[..example.len() - 1];
        let targets = example[1..]
            .iter()
            .map(|&token| token as usize)
            .collect::<Vec<_>>();
        let position_ids = (0..input_ids.len())
            .map(|idx| idx as u32)
            .collect::<Vec<_>>();

        optimizer.zero_grad(&params, &mut store);
        let logits = model.forward(&mut store, &mut tape, input_ids, &position_ids)?;
        let loss_id = cross_entropy_loss(logits, &targets, &mut store, &mut tape)?;
        losses.push(store.to_host(loss_id)?[0]);
        tape.backward(loss_id, &mut store)?;
        optimizer.step(&params, &mut store);

        tape.entries.clear();
        tape.set_enabled(true);
        let keep = retained_ids(&model_ids, &params, &store);
        store.retain_ids(&keep);
    }

    Ok(StepSnapshot {
        losses,
        reload_logits: export_reload_logits(&model, &mut store, &mut tape, cfg)?,
        named_params: export_named_params(&mut store, &param_names)?,
        optim_state: optimizer.export_state(&param_names),
    })
}

fn run_hybrid_grpo_steps(
    backend: Option<Arc<dyn Backend>>,
    cfg: &train::qwen35::Qwen35Config,
    lora: LoraConfig,
    trajectories: &[Trajectory],
    steps: usize,
) -> TestResult<StepSnapshot> {
    let mut store = match backend {
        Some(backend) => TensorStore::with_backend(backend),
        None => TensorStore::default(),
    };
    let model = Qwen35Model::new_with_lora(cfg, Some(lora), &mut store)?;
    let params = trainable_params(&model, &store);
    let param_names = trainable_param_name_map(&model, &store);
    let model_ids = live_tensor_ids(&store);
    let mut optimizer = AdamW::new(TEST_LR, (0.9, 0.999), 1.0e-8, 0.0);
    let mut tape = Tape::new();
    let rewards = trajectories
        .iter()
        .map(|trajectory| trajectory.reward)
        .collect::<Vec<_>>();
    let advantages = group_advantages(&rewards, 2);
    let mut losses = Vec::with_capacity(steps);

    for _ in 0..steps {
        optimizer.zero_grad(&params, &mut store);
        let loss_id = grpo_loss(
            &model,
            trajectories,
            &advantages,
            &GrpoConfig {
                clip_eps: 0.2,
                kl_coef: 0.02,
                group_size: 2,
            },
            cfg,
            &mut store,
            &mut tape,
        )?;
        losses.push(store.to_host(loss_id)?[0]);
        tape.backward(loss_id, &mut store)?;
        optimizer.step(&params, &mut store);

        tape.entries.clear();
        tape.set_enabled(true);
        let keep = retained_ids(&model_ids, &params, &store);
        store.retain_ids(&keep);
    }

    Ok(StepSnapshot {
        losses,
        reload_logits: export_reload_logits(&model, &mut store, &mut tape, cfg)?,
        named_params: export_named_params(&mut store, &param_names)?,
        optim_state: optimizer.export_state(&param_names),
    })
}

fn export_named_params(
    store: &mut TensorStore,
    param_names: &[(autograd::TensorId, String)],
) -> TestResult<Vec<(String, Vec<f32>)>> {
    let mut named = Vec::with_capacity(param_names.len());
    for (tensor_id, name) in param_names {
        named.push((name.clone(), store.to_host(*tensor_id)?));
    }
    Ok(named)
}

fn export_reload_logits(
    model: &Qwen35Model,
    store: &mut TensorStore,
    tape: &mut Tape,
    cfg: &train::qwen35::Qwen35Config,
) -> TestResult<Vec<f32>> {
    let dir = tempdir()?;
    let path = dir.path().join("model.safetensors");
    save_materialized_registry(model, store, tape, &path, false)?;
    tape.entries.clear();
    tape.set_enabled(true);

    let mut loaded_store = TensorStore::default();
    let loaded_model = Qwen35Model::new_for_eval(cfg, &mut loaded_store)?;
    let mut loaded_registry = build_registry(&loaded_model);
    loaded_registry.load_into(&mut loaded_store, &path)?;

    let input_ids = [3, 4, 5, 6, 7];
    let position_ids = [0, 1, 2, 3, 4];
    let mut loaded_tape = Tape::new();
    let logits = loaded_model.forward(
        &mut loaded_store,
        &mut loaded_tape,
        &input_ids,
        &position_ids,
    )?;
    Ok(loaded_store.to_host(logits)?)
}

fn fixed_grpo_trajectories() -> Vec<Trajectory> {
    vec![
        Trajectory {
            prompt_ids: vec![1, 2, 3, 4, 5, 0, 0, 0],
            response_mask: vec![false, false, false, false, false, true, true, true],
            full_ids: vec![1, 2, 3, 4, 5, 6, 7, 8],
            old_log_probs: vec![0.0, 0.0, 0.0, 0.0, 0.0, -1.2, -1.1, -1.0],
            ref_log_probs: vec![0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -0.9, -0.8],
            reward: 1.0,
        },
        Trajectory {
            prompt_ids: vec![2, 3, 4, 5, 6, 0, 0, 0],
            response_mask: vec![false, false, false, false, false, true, true, true],
            full_ids: vec![2, 3, 4, 5, 6, 7, 8, 9],
            old_log_probs: vec![0.0, 0.0, 0.0, 0.0, 0.0, -1.3, -1.2, -1.1],
            ref_log_probs: vec![0.0, 0.0, 0.0, 0.0, 0.0, -1.1, -1.0, -0.9],
            reward: 0.5,
        },
    ]
}

fn fixed_grpo_trajectories_longer() -> Vec<Trajectory> {
    vec![
        Trajectory {
            prompt_ids: vec![1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0],
            response_mask: vec![
                false, false, false, false, false, false, false, false, true, true, true, true,
            ],
            full_ids: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            old_log_probs: vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.3, -1.2, -1.1, -1.0,
            ],
            ref_log_probs: vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.1, -1.0, -0.9, -0.8,
            ],
            reward: 1.0,
        },
        Trajectory {
            prompt_ids: vec![2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0],
            response_mask: vec![
                false, false, false, false, false, false, false, false, true, true, true, true,
            ],
            full_ids: vec![2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            old_log_probs: vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.4, -1.3, -1.2, -1.1,
            ],
            ref_log_probs: vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.2, -1.1, -1.0, -0.9,
            ],
            reward: 0.5,
        },
    ]
}

fn assert_snapshot_close(label: &str, cpu: &StepSnapshot, metal: &StepSnapshot) {
    assert_eq!(cpu.losses.len(), metal.losses.len(), "{label} loss count");
    for (index, (&cpu_loss, &metal_loss)) in cpu.losses.iter().zip(metal.losses.iter()).enumerate()
    {
        assert_close_scalar(&format!("{label} loss[{index}]"), cpu_loss, metal_loss);
    }
    assert_close_slice(
        &format!("{label} reload_logits"),
        &cpu.reload_logits,
        &metal.reload_logits,
    );
    assert_eq!(cpu.named_params.len(), metal.named_params.len());
    for ((cpu_name, cpu_values), (metal_name, metal_values)) in
        cpu.named_params.iter().zip(metal.named_params.iter())
    {
        assert_eq!(cpu_name, metal_name, "{label} param name mismatch");
        assert_close_slice(
            &format!("{label} param {cpu_name}"),
            cpu_values,
            metal_values,
        );
    }
    assert_adamw_state_close(label, &cpu.optim_state, &metal.optim_state);
}

fn assert_adamw_state_close(label: &str, cpu: &AdamWState, metal: &AdamWState) {
    assert_eq!(cpu.step, metal.step, "{label} optimizer step");
    assert_eq!(
        cpu.skipped_export, metal.skipped_export,
        "{label} optimizer skipped_export"
    );
    assert_eq!(
        cpu.params.len(),
        metal.params.len(),
        "{label} optimizer param count"
    );
    for (cpu_param, metal_param) in cpu.params.iter().zip(metal.params.iter()) {
        assert_eq!(
            cpu_param.name, metal_param.name,
            "{label} optimizer param name"
        );
        assert_eq!(
            cpu_param.shape, metal_param.shape,
            "{label} optimizer param shape"
        );
        assert_close_slice(
            &format!("{label} optimizer {}.m", cpu_param.name),
            &cpu_param.m,
            &metal_param.m,
        );
        assert_close_slice(
            &format!("{label} optimizer {}.v", cpu_param.name),
            &cpu_param.v,
            &metal_param.v,
        );
    }
}

fn assert_close_slice(label: &str, cpu: &[f32], metal: &[f32]) {
    assert_eq!(cpu.len(), metal.len(), "{label} length");
    for (index, (&cpu_value, &metal_value)) in cpu.iter().zip(metal.iter()).enumerate() {
        let abs_err = (cpu_value - metal_value).abs();
        let rel_err = abs_err / cpu_value.abs().max(1.0e-5);
        assert!(
            abs_err <= ABS_TOL || rel_err <= REL_TOL,
            "{label}[{index}] drift: cpu={cpu_value} metal={metal_value} abs={abs_err} rel={rel_err}"
        );
    }
}

fn assert_close_scalar(label: &str, cpu: f32, metal: f32) {
    let abs_err = (cpu - metal).abs();
    let rel_err = abs_err / cpu.abs().max(1.0e-5);
    assert!(
        abs_err <= ABS_TOL || rel_err <= REL_TOL,
        "{label} drift: cpu={cpu} metal={metal} abs={abs_err} rel={rel_err}"
    );
}
