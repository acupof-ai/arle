use std::time::Instant;

use autograd::{Tape, TensorId, TensorStore, optim::AdamW};
use qwen35_spec::{LayerType, Qwen35Config};
use train::{
    opd::{OpdStepConfig, opd_step},
    qwen35::Qwen35Model,
};

const WARMUP_RUNS: usize = 1;
const MEASURED_RUNS: usize = 5;
const STEPS_PER_RUN: usize = 100;
const SEED: u64 = 0xB100_0D15_71A0_2026;
const LR: f32 = 1.0e-3;
const ROLLOUT_LEN: usize = 2;
const PROMPT_IDS: &[u32] = &[1, 3, 8];

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

fn run_once() -> (f64, f32, f32) {
    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let cfg = tiny_qwen35_config();
    let teacher = Qwen35Model::new(&cfg, &mut store).expect("teacher");
    let student = Qwen35Model::new(&cfg, &mut store).expect("student");
    let student_params = student.all_parameter_ids();
    perturb_params_from_seed(&mut store, &student_params, SEED);
    let mut optimizer = AdamW::new(LR, (0.9, 0.999), 1.0e-8, 0.0);
    let step_cfg = OpdStepConfig {
        rollout_len: ROLLOUT_LEN,
        grad_clip: 1.0,
    };

    let started = Instant::now();
    let mut first_loss = None;
    let mut last_loss = 0.0_f32;
    for _ in 0..STEPS_PER_RUN {
        let outcome = opd_step(
            &student,
            &teacher,
            PROMPT_IDS,
            step_cfg,
            &student_params,
            &mut optimizer,
            &mut store,
            &mut tape,
        )
        .expect("opd_step");
        first_loss.get_or_insert(outcome.loss);
        last_loss = outcome.loss;
    }
    (
        started.elapsed().as_secs_f64(),
        first_loss.unwrap(),
        last_loss,
    )
}

fn main() {
    println!(
        "config backend=cpu hidden=16 layers=2 vocab=16 prompt={PROMPT_IDS:?} rollout_len={ROLLOUT_LEN} lr={LR} steps_per_run={STEPS_PER_RUN} warmup_runs={WARMUP_RUNS} measured_runs={MEASURED_RUNS}"
    );

    for _ in 0..WARMUP_RUNS {
        let _ = run_once();
    }

    let mut rates = Vec::with_capacity(MEASURED_RUNS);
    for run in 1..=MEASURED_RUNS {
        let (secs, first_loss, last_loss) = run_once();
        let steps_per_sec = STEPS_PER_RUN as f64 / secs;
        rates.push(steps_per_sec);
        println!(
            "run={run} seconds={secs:.6} steps_per_sec={steps_per_sec:.6} first_loss={first_loss:.9} last_loss={last_loss:.9}"
        );
    }

    let mean = rates.iter().sum::<f64>() / rates.len() as f64;
    let variance = rates
        .iter()
        .map(|rate| {
            let delta = rate - mean;
            delta * delta
        })
        .sum::<f64>()
        / rates.len() as f64;
    let sigma = variance.sqrt();
    let sigma_pct = if mean == 0.0 {
        0.0
    } else {
        sigma / mean * 100.0
    };
    let mut sorted = rates.clone();
    sorted.sort_by(f64::total_cmp);
    let median = sorted[sorted.len() / 2];
    println!(
        "summary mean_steps_per_sec={mean:.6} median_steps_per_sec={median:.6} sigma_steps_per_sec={sigma:.6} sigma_pct={sigma_pct:.3}"
    );
}
