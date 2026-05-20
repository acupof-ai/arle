//! OPD CPU phase profile at the same moderate shape used by
//! `opd_step_cpu_moderate_bench`.
//!
//! This harness mirrors `train::opd::opd_step`, including teacher keep-set
//! construction and post-backward TensorStore cleanup, so the phase attribution
//! stays comparable to production OPD behavior without scaling to a full
//! Qwen3-0.6B shape on the 31 GiB dev host.

use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};

use autograd::{
    BackwardProfile, Tape, TensorId, TensorStore,
    optim::{AdamW, Optimizer},
};
use qwen35_spec::{LayerType, Qwen35Config};
use train::{
    grad_clip::clip_grad_norm,
    loss::kl_distill_loss,
    opd::{OpdStepConfig, OpdStepOutcome},
    qwen35::Qwen35Model,
    trainer::{cleanup_after_backward, retained_param_and_grad_ids},
};

const WARMUP_RUNS: usize = 2;
const MEASURED_RUNS: usize = 3;
const STEPS_PER_RUN: usize = 5;
const SEED: u64 = 0xB300_0D15_71A0_2026;
const LR: f32 = 1.0e-3;
const ROLLOUT_LEN: usize = 2;
const PROMPT_IDS: &[u32] = &[1, 3, 8];

type AnyResult<T> = Result<T, Box<dyn std::error::Error>>;

#[derive(Debug, Default, Clone)]
struct PhaseTotals {
    durations: BTreeMap<&'static str, Duration>,
}

impl PhaseTotals {
    fn add(&mut self, phase: &'static str, duration: Duration) {
        *self.durations.entry(phase).or_default() += duration;
    }

    fn merge(&mut self, other: &PhaseTotals) {
        for (&phase, &duration) in &other.durations {
            self.add(phase, duration);
        }
    }

    fn seconds(&self, phase: &'static str) -> f64 {
        self.durations
            .get(phase)
            .copied()
            .unwrap_or_default()
            .as_secs_f64()
    }
}

fn timed<T>(
    totals: &mut PhaseTotals,
    phase: &'static str,
    f: impl FnOnce() -> AnyResult<T>,
) -> AnyResult<T> {
    let started = Instant::now();
    let value = f()?;
    totals.add(phase, started.elapsed());
    Ok(value)
}

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

fn greedy_next_token(
    logits_id: TensorId,
    seq_len: usize,
    vocab: usize,
    store: &mut TensorStore,
) -> AnyResult<u32> {
    let host = store.to_host(logits_id)?;
    let last_row_start = (seq_len - 1) * vocab;
    let row = &host[last_row_start..last_row_start + vocab];
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    Ok(best_idx as u32)
}

fn profiled_opd_step<O: Optimizer>(
    student: &Qwen35Model,
    teacher: &Qwen35Model,
    prompt_ids: &[u32],
    cfg: OpdStepConfig,
    student_params: &[TensorId],
    optimizer: &mut O,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> AnyResult<(OpdStepOutcome, PhaseTotals, BackwardProfile)> {
    let total_started = Instant::now();
    let mut totals = PhaseTotals::default();
    let vocab = student.config().vocab_size;

    let keep_extra = timed(&mut totals, "keep_extra_build", || {
        let teacher_params = teacher.all_parameter_ids();
        Ok(retained_param_and_grad_ids(&teacher_params, store))
    })?;

    timed(&mut totals, "rollout_tape_disable", || {
        tape.entries.clear();
        tape.set_enabled(false);
        Ok(())
    })?;

    let mut rollout = prompt_ids.to_vec();
    for _ in 0..cfg.rollout_len {
        let positions = timed(&mut totals, "rollout_positions", || {
            Ok((0..rollout.len() as u32).collect::<Vec<_>>())
        })?;
        let logits = timed(&mut totals, "rollout_student_forward", || {
            Ok(student.forward(store, tape, &rollout, &positions)?)
        })?;
        let next = timed(&mut totals, "rollout_argmax_readback", || {
            greedy_next_token(logits, rollout.len(), vocab, store)
        })?;
        rollout.push(next);
    }

    let positions = timed(&mut totals, "full_positions", || {
        Ok((0..rollout.len() as u32).collect::<Vec<_>>())
    })?;

    let teacher_logits = timed(&mut totals, "teacher_forward", || {
        Ok(teacher.forward(store, tape, &rollout, &positions)?)
    })?;

    timed(&mut totals, "student_tape_enable", || {
        tape.set_enabled(true);
        Ok(())
    })?;

    let student_logits = timed(&mut totals, "student_forward", || {
        Ok(student.forward(store, tape, &rollout, &positions)?)
    })?;

    let loss = timed(&mut totals, "kl_distill_loss", || {
        Ok(kl_distill_loss(
            student_logits,
            teacher_logits,
            rollout.len(),
            store,
            tape,
        )?)
    })?;
    let loss_value = timed(&mut totals, "loss_readback", || Ok(store.to_host(loss)?[0]))?;

    timed(&mut totals, "optimizer_zero_grad", || {
        optimizer.zero_grad(store, student_params);
        Ok(())
    })?;
    let backward_profile = timed(&mut totals, "backward", || {
        let (_, profile) = tape.backward_profiled(loss, store)?;
        Ok(profile)
    })?;
    timed(&mut totals, "grad_clip", || {
        clip_grad_norm(student_params, cfg.grad_clip, store);
        Ok(())
    })?;
    timed(&mut totals, "optimizer_step", || {
        optimizer.step(store, student_params)?;
        Ok(())
    })?;
    timed(&mut totals, "post_step_cleanup", || {
        cleanup_after_backward(store, tape, student_params, &keep_extra);
        Ok(())
    })?;

    totals.add("total_step", total_started.elapsed());

    Ok((
        OpdStepOutcome {
            loss: loss_value,
            rollout_len: rollout.len(),
        },
        totals,
        backward_profile,
    ))
}

fn run_once() -> AnyResult<(f64, f32, f32, PhaseTotals, BackwardProfile)> {
    let mut store = TensorStore::default();
    let mut tape = Tape::new();
    let cfg = moderate_qwen35_config();
    let teacher = Qwen35Model::new_for_eval(&cfg, &mut store)?;
    let student = Qwen35Model::new(&cfg, &mut store)?;
    let student_params = student.all_parameter_ids();
    perturb_params_from_seed(&mut store, &student_params, SEED);
    let mut optimizer = AdamW::new(LR, (0.9, 0.999), 1.0e-8, 0.0);
    let step_cfg = OpdStepConfig {
        rollout_len: ROLLOUT_LEN,
        grad_clip: 1.0,
    };

    let started = Instant::now();
    let mut totals = PhaseTotals::default();
    let mut backward_profile = BackwardProfile::default();
    let mut first_loss = None;
    let mut last_loss = 0.0f32;
    for _ in 0..STEPS_PER_RUN {
        let (outcome, step_totals, step_backward_profile) = profiled_opd_step(
            &student,
            &teacher,
            PROMPT_IDS,
            step_cfg,
            &student_params,
            &mut optimizer,
            &mut store,
            &mut tape,
        )?;
        first_loss.get_or_insert(outcome.loss);
        last_loss = outcome.loss;
        totals.merge(&step_totals);
        backward_profile.merge(&step_backward_profile);
    }

    Ok((
        started.elapsed().as_secs_f64(),
        first_loss.expect("at least one OPD step"),
        last_loss,
        totals,
        backward_profile,
    ))
}

fn main() -> AnyResult<()> {
    println!(
        "config backend=cpu hidden=512 intermediate=1536 layers=12 vocab=32768 num_heads=8 head_dim=64 num_kv_heads=4 prompt={PROMPT_IDS:?} rollout_len={ROLLOUT_LEN} lr={LR} steps_per_run={STEPS_PER_RUN} warmup_runs={WARMUP_RUNS} measured_runs={MEASURED_RUNS}"
    );

    for _ in 0..WARMUP_RUNS {
        let _ = run_once()?;
    }

    let mut aggregate = PhaseTotals::default();
    let mut aggregate_backward = BackwardProfile::default();
    let mut rates = Vec::with_capacity(MEASURED_RUNS);
    for run in 1..=MEASURED_RUNS {
        let (secs, first_loss, last_loss, totals, backward_profile) = run_once()?;
        let steps_per_sec = STEPS_PER_RUN as f64 / secs;
        let total_step_secs = totals.seconds("total_step");
        rates.push(steps_per_sec);
        aggregate.merge(&totals);
        aggregate_backward.merge(&backward_profile);
        println!(
            "run={run} wall_seconds={secs:.6} summed_step_seconds={total_step_secs:.6} steps_per_sec={steps_per_sec:.6} first_loss={first_loss:.9} last_loss={last_loss:.9}"
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
    let mut sorted_rates = rates.clone();
    sorted_rates.sort_by(f64::total_cmp);
    let median = sorted_rates[sorted_rates.len() / 2];
    let total_step_secs = aggregate.seconds("total_step");
    println!(
        "summary mean_steps_per_sec={mean:.6} median_steps_per_sec={median:.6} sigma_steps_per_sec={sigma:.6} sigma_pct={sigma_pct:.3} total_step_seconds={total_step_secs:.6}"
    );

    let mut phase_rows: Vec<(&'static str, f64)> = aggregate
        .durations
        .iter()
        .filter_map(|(&phase, duration)| {
            (phase != "total_step").then_some((phase, duration.as_secs_f64()))
        })
        .collect();
    phase_rows.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));

    for (rank, (phase, seconds)) in phase_rows.iter().enumerate() {
        let pct_total = if total_step_secs == 0.0 {
            0.0
        } else {
            seconds / total_step_secs * 100.0
        };
        println!(
            "phase_summary rank={} phase={} seconds={:.6} pct_total={:.3}",
            rank + 1,
            phase,
            seconds,
            pct_total
        );
    }

    let backward_total_secs = aggregate_backward.total_duration.as_secs_f64();
    let backward_op_secs = aggregate_backward.total_op_duration().as_secs_f64();
    let backward_merge_secs = aggregate_backward.merge_grad_duration.as_secs_f64();
    let backward_prelude_secs = aggregate_backward.prelude_duration.as_secs_f64();
    let backward_unattributed_secs =
        (backward_total_secs - backward_op_secs - backward_merge_secs - backward_prelude_secs)
            .max(0.0);
    println!(
        "backward_profile_summary total_seconds={backward_total_secs:.6} op_seconds={backward_op_secs:.6} merge_grad_seconds={backward_merge_secs:.6} prelude_seconds={backward_prelude_secs:.6} unattributed_seconds={backward_unattributed_secs:.6}"
    );

    let mut backward_rows = aggregate_backward
        .op_totals
        .iter()
        .map(|(&op, stats)| (op, stats.count, stats.duration.as_secs_f64()))
        .collect::<Vec<_>>();
    backward_rows.sort_by(|a, b| b.2.total_cmp(&a.2).then_with(|| a.0.cmp(&b.0)));
    for (rank, (op, count, seconds)) in backward_rows.iter().enumerate() {
        let pct_backward = if backward_total_secs == 0.0 {
            0.0
        } else {
            seconds / backward_total_secs * 100.0
        };
        println!(
            "backward_op_summary rank={} op={:?} count={} seconds={:.6} pct_backward={:.3}",
            rank + 1,
            op,
            count,
            seconds,
            pct_backward
        );
    }

    Ok(())
}
