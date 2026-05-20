//! End-to-end OPD step CPU bench at a **moderate** Qwen3.5-like shape:
//! hidden=512, intermediate=1536, layers=12, vocab=32768, num_heads=8,
//! head_dim=64, num_kv_heads=4. ~100M params total. The Qwen3-0.6B shape
//! (hidden=1024, layers=28, vocab=151_936) would not fit our dev box's
//! 31 GB / 21 GB-already-used RAM (codex B1b probe aborted), so this
//! middle shape is the SOLID wall-clock ground truth to cross-check the
//! `cpu_matmul_microbench` projections.
//!
//! Per CLAUDE.md §0, microbench projections (3 forwards ≈ 0.83 s, 1
//! backward ≈ 0.97 s on Qwen3-0.6B) are a *hypothesis* about end-to-end
//! step cost; only an actual step bench reports the wall-clock ground
//! truth and the residual share that is **not** matmul.
//!
//! Run:
//!   cargo run -p train --example opd_step_cpu_moderate_bench --release

use std::time::Instant;

use autograd::{Tape, TensorId, TensorStore, optim::AdamW};
use qwen35_spec::{LayerType, Qwen35Config};
use train::{
    opd::{OpdStepConfig, opd_step},
    qwen35::Qwen35Model,
};

const WARMUP_RUNS: usize = 1;
const MEASURED_RUNS: usize = 3;
const STEPS_PER_RUN: usize = 10;
const SEED: u64 = 0xB100_0D15_71A0_2026;
const LR: f32 = 1.0e-3;
const ROLLOUT_LEN: usize = 2;
const PROMPT_IDS: &[u32] = &[1, 3, 8];

fn moderate_qwen35_config() -> Qwen35Config {
    Qwen35Config {
        hidden_size: 512,
        intermediate_size: 1536,
        num_hidden_layers: 12,
        vocab_size: 32_768,
        rms_norm_eps: 1.0e-6,
        stop_token_ids: vec![32_767],
        bos_token_id: Some(1),
        eos_token_id: 32_767,
        tie_word_embeddings: false,
        num_attention_heads: 8,
        num_key_value_heads: 4,
        head_dim: 64,
        linear_num_key_heads: 8,
        linear_key_head_dim: 64,
        linear_num_value_heads: 8,
        linear_value_head_dim: 64,
        linear_conv_kernel_dim: 4,
        rope_theta: 10_000.0,
        rope_scaling: None,
        partial_rotary_factor: 1.0,
        rotary_dim: 64,
        rope_cache_len_hint: Some(64),
        layer_types: vec![LayerType::FullAttention; 12],
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
    let cfg = moderate_qwen35_config();
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
        "config backend=cpu hidden=512 intermediate=1536 layers=12 vocab=32768 num_heads=8 head_dim=64 num_kv_heads=4 prompt={PROMPT_IDS:?} rollout_len={ROLLOUT_LEN} lr={LR} steps_per_run={STEPS_PER_RUN} warmup_runs={WARMUP_RUNS} measured_runs={MEASURED_RUNS}"
    );

    for _ in 0..WARMUP_RUNS {
        let _ = run_once();
    }

    let mut rates = Vec::with_capacity(MEASURED_RUNS);
    let mut step_seconds_runs = Vec::with_capacity(MEASURED_RUNS);
    for run in 1..=MEASURED_RUNS {
        let (secs, first_loss, last_loss) = run_once();
        let per_step = secs / STEPS_PER_RUN as f64;
        let steps_per_sec = STEPS_PER_RUN as f64 / secs;
        rates.push(steps_per_sec);
        step_seconds_runs.push(per_step);
        println!(
            "run={run} wall_seconds={secs:.6} per_step_seconds={per_step:.6} steps_per_sec={steps_per_sec:.6} first_loss={first_loss:.9} last_loss={last_loss:.9}"
        );
    }

    let mean = rates.iter().sum::<f64>() / rates.len() as f64;
    let mean_step_secs = step_seconds_runs.iter().sum::<f64>() / step_seconds_runs.len() as f64;
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
    let mut sorted_step = step_seconds_runs.clone();
    sorted_step.sort_by(f64::total_cmp);
    let median_step_secs = sorted_step[sorted_step.len() / 2];
    println!(
        "summary mean_steps_per_sec={mean:.6} median_steps_per_sec={median:.6} sigma_steps_per_sec={sigma:.6} sigma_pct={sigma_pct:.3} mean_step_seconds={mean_step_secs:.6} median_step_seconds={median_step_secs:.6}"
    );
}
