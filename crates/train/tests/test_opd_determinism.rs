mod common;

use autograd::{Tape, TensorId, TensorStore, optim::AdamW};
use common::qwen35_test_support::{TestResult, tiny_qwen35_scratch_config};
use train::{
    opd::{OpdStepConfig, opd_step},
    qwen35::Qwen35Model,
};

fn run_opd_loss_bits(seed: u64, lr: f32, prompt_ids: &[u32]) -> TestResult<Vec<u32>> {
    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let cfg = tiny_qwen35_scratch_config(8);

    let teacher = Qwen35Model::new(&cfg, &mut store)?;
    let student = Qwen35Model::new(&cfg, &mut store)?;
    let student_params = student.all_parameter_ids();
    perturb_params_from_seed(&mut store, &student_params, seed);

    let mut optimizer = AdamW::new(lr, (0.9, 0.999), 1.0e-8, 0.0);
    let step_cfg = OpdStepConfig {
        rollout_len: 2,
        grad_clip: 1.0,
    };
    let mut loss_bits = Vec::with_capacity(3);
    for _ in 0..3 {
        let outcome = opd_step(
            &student,
            &teacher,
            prompt_ids,
            step_cfg,
            &student_params,
            &mut optimizer,
            &mut store,
            &mut tape,
        )?;
        assert!(outcome.loss.is_finite(), "loss must be finite");
        loss_bits.push(outcome.loss.to_bits());
    }

    Ok(loss_bits)
}

fn perturb_params_from_seed(store: &mut TensorStore, params: &[TensorId], seed: u64) {
    let mut state = seed;
    for &param in params {
        let tensor = store.get_mut(param).expect("student param exists");
        for value in &mut tensor.data {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = ((state >> 32) as f32) / (u32::MAX as f32);
            *value += (unit - 0.5) * 1.0e-3;
        }
    }
}

#[test]
fn opd_step_same_prompt_seed_and_lr_is_bit_identical() -> TestResult {
    let prompt_ids = [1, 3, 8];
    let seed = 0x0A11_CE5E_ED5E_ED5E;
    let lr = 1.0e-3;

    let first = run_opd_loss_bits(seed, lr, &prompt_ids)?;
    let second = run_opd_loss_bits(seed, lr, &prompt_ids)?;

    assert_eq!(
        first, second,
        "OPD losses must be bit-identical for the same prompt, seed, and lr"
    );
    Ok(())
}
