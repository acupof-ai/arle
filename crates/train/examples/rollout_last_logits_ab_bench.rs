//! Single-variable A/B: rollout student forward — full lm_head (all positions)
//! vs. last-row-only lm_head (`forward_last_logits`).
//!
//! Hypothesis: under OPD rollout, only the last position's logits are read by
//! `greedy_argmax_last_row`. Computing lm_head over the earlier (S-1)
//! positions is dead work. At Qwen3-0.6B shape (vocab=151_936) the wasted
//! FMAs dominate rollout cost.
//!
//! Method: build one student, run `cfg.rollout_len` greedy-rollout iterations
//! using each variant, repeat WARMUP_RUNS + MEASURED_RUNS times. Both variants
//! share embedding + transformer + final_norm; only the lm_head extent differs.
//! Greedy-argmax determinism: both variants must select the same next token at
//! each step.
//!
//! Run:
//!   cargo run -p train --example rollout_last_logits_ab_bench --release

use std::{collections::HashSet, time::Instant};

use autograd::{Tape, TensorStore};
use qwen35_spec::{LayerType, Qwen35Config};
use train::qwen35::Qwen35Model;

const WARMUP_RUNS: usize = 1;
const MEASURED_RUNS: usize = 3;
const ROLLOUT_LEN: usize = 2;
const PROMPT_IDS: &[u32] = &[1, 3, 8];

/// Qwen3-0.6B-vocab shape to make lm_head dominate. Production vocab is
/// 151 936; `hidden` and `intermediate` are reduced and `num_layers` cut
/// to 4 so the transformer body cannot crowd out the lm_head signal we
/// want to isolate. The lm_head weight alone at this shape is 1024 × 151 936
/// × 4 B = 623 MB; total RAM well under 2 GB.
fn moderate_config() -> Qwen35Config {
    Qwen35Config {
        hidden_size: 1024,
        intermediate_size: 3072,
        num_hidden_layers: 4,
        vocab_size: 151_936,
        rms_norm_eps: 1.0e-6,
        stop_token_ids: vec![32_767],
        bos_token_id: Some(1),
        eos_token_id: 32_767,
        tie_word_embeddings: false,
        num_attention_heads: 16,
        num_key_value_heads: 8,
        head_dim: 64,
        linear_num_key_heads: 16,
        linear_key_head_dim: 64,
        linear_num_value_heads: 16,
        linear_value_head_dim: 64,
        linear_conv_kernel_dim: 4,
        rope_theta: 10_000.0,
        rope_scaling: None,
        partial_rotary_factor: 1.0,
        rotary_dim: 64,
        rope_cache_len_hint: Some(64),
        layer_types: vec![LayerType::FullAttention; 4],
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

fn greedy_argmax_full_seq(host_logits: &[f32], seq_len: usize, vocab: usize) -> u32 {
    let last_row_start = (seq_len - 1) * vocab;
    let row = &host_logits[last_row_start..last_row_start + vocab];
    let mut best_idx: usize = 0;
    let mut best_val: f32 = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx as u32
}

fn greedy_argmax_single(host_logits: &[f32]) -> u32 {
    let mut best_idx: usize = 0;
    let mut best_val: f32 = f32::NEG_INFINITY;
    for (i, &v) in host_logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx as u32
}

fn run_rollout_full(
    student: &Qwen35Model,
    store: &mut TensorStore,
    tape: &mut Tape,
    prompt: &[u32],
    vocab: usize,
) -> (f64, Vec<u32>) {
    tape.entries.clear();
    tape.set_enabled(false);
    let mut rollout: Vec<u32> = prompt.to_vec();
    let mut tokens = Vec::with_capacity(ROLLOUT_LEN);
    let started = Instant::now();
    for _ in 0..ROLLOUT_LEN {
        let positions: Vec<u32> = (0..rollout.len() as u32).collect();
        let logits = student
            .forward(store, tape, &rollout, &positions)
            .expect("forward");
        let host = store.to_host(logits).expect("to_host");
        let next = greedy_argmax_full_seq(&host, rollout.len(), vocab);
        rollout.push(next);
        tokens.push(next);
    }
    (started.elapsed().as_secs_f64(), tokens)
}

fn run_rollout_last(
    student: &Qwen35Model,
    store: &mut TensorStore,
    tape: &mut Tape,
    prompt: &[u32],
) -> (f64, Vec<u32>) {
    tape.entries.clear();
    tape.set_enabled(false);
    let mut rollout: Vec<u32> = prompt.to_vec();
    let mut tokens = Vec::with_capacity(ROLLOUT_LEN);
    let started = Instant::now();
    for _ in 0..ROLLOUT_LEN {
        let positions: Vec<u32> = (0..rollout.len() as u32).collect();
        let logits = student
            .forward_last_logits(store, tape, &rollout, &positions)
            .expect("forward_last_logits");
        let host = store.to_host(logits).expect("to_host");
        let next = greedy_argmax_single(&host);
        rollout.push(next);
        tokens.push(next);
    }
    (started.elapsed().as_secs_f64(), tokens)
}

fn retain_model_tensors(store: &mut TensorStore, tape: &mut Tape, keep: &HashSet<usize>) {
    tape.entries.clear();
    tape.set_enabled(false);
    store.retain_ids(keep);
}

fn report(label: &str, times: &[f64]) {
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let var = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / times.len() as f64;
    let sigma = var.sqrt();
    let pct = if mean > 0.0 {
        sigma / mean * 100.0
    } else {
        0.0
    };
    let mut sorted = times.to_vec();
    sorted.sort_by(f64::total_cmp);
    let median = sorted[sorted.len() / 2];
    println!(
        "{label:<24} mean={mean:.6}s median={median:.6}s sigma={sigma:.6}s sigma_pct={pct:.3}%"
    );
}

fn main() {
    let cfg = moderate_config();
    println!(
        "config hidden={} intermediate={} layers={} vocab={} prompt={:?} rollout_len={ROLLOUT_LEN} warmup={WARMUP_RUNS} measured={MEASURED_RUNS}",
        cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers, cfg.vocab_size, PROMPT_IDS,
    );

    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let student = Qwen35Model::new(&cfg, &mut store).expect("student");
    let vocab = cfg.vocab_size;
    let keep: HashSet<usize> = store
        .tensors
        .iter()
        .enumerate()
        .filter_map(|(id, slot)| slot.as_ref().map(|_| id))
        .collect();

    // Equivalence check: tokens produced must match across variants.
    let (_, tokens_full) = run_rollout_full(&student, &mut store, &mut tape, PROMPT_IDS, vocab);
    retain_model_tensors(&mut store, &mut tape, &keep);
    let (_, tokens_last) = run_rollout_last(&student, &mut store, &mut tape, PROMPT_IDS);
    retain_model_tensors(&mut store, &mut tape, &keep);
    assert_eq!(
        tokens_full, tokens_last,
        "BUG: last-logits rollout produced different tokens than full-lm_head rollout"
    );
    println!("equivalence_ok rollout_tokens_full=last={tokens_full:?}");

    let mut full_times = Vec::with_capacity(MEASURED_RUNS);
    let mut last_times = Vec::with_capacity(MEASURED_RUNS);

    for _ in 0..WARMUP_RUNS {
        let _ = run_rollout_full(&student, &mut store, &mut tape, PROMPT_IDS, vocab);
        retain_model_tensors(&mut store, &mut tape, &keep);
        let _ = run_rollout_last(&student, &mut store, &mut tape, PROMPT_IDS);
        retain_model_tensors(&mut store, &mut tape, &keep);
    }

    for run in 1..=MEASURED_RUNS {
        let (t_full, _) = run_rollout_full(&student, &mut store, &mut tape, PROMPT_IDS, vocab);
        retain_model_tensors(&mut store, &mut tape, &keep);
        let (t_last, _) = run_rollout_last(&student, &mut store, &mut tape, PROMPT_IDS);
        retain_model_tensors(&mut store, &mut tape, &keep);
        println!(
            "run={run} full={t_full:.6}s last={t_last:.6}s speedup={:.3}x",
            t_full / t_last
        );
        full_times.push(t_full);
        last_times.push(t_last);
    }

    println!();
    report("full lm_head (baseline)", &full_times);
    report("last-row lm_head", &last_times);
    let mean_full = full_times.iter().sum::<f64>() / full_times.len() as f64;
    let mean_last = last_times.iter().sum::<f64>() / last_times.len() as f64;
    println!(
        "\nrollout speedup mean_full/mean_last = {:.3}x",
        mean_full / mean_last
    );
    println!(
        "rollout saved per iteration: mean {:.6}s ({} iterations)",
        (mean_full - mean_last) / ROLLOUT_LEN as f64,
        ROLLOUT_LEN
    );
}
