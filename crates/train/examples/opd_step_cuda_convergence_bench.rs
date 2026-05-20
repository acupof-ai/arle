#![cfg_attr(
    not(all(feature = "cuda", not(feature = "no-cuda"))),
    allow(dead_code, unused_imports)
)]

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
mod app {
    use std::{
        collections::{HashMap, HashSet},
        env,
        path::PathBuf,
        sync::Arc,
        time::Instant,
    };

    use autograd::{
        backend_cuda::CudaBackend,
        optim::AdamW,
        tensor::{TensorId, TensorStore},
        Tape,
    };
    use qwen35_spec::{LayerType, Qwen35Config};
    use train::{
        opd::{opd_step, OpdStepConfig},
        qwen35::Qwen35Model,
    };

    const TRAIN_STEPS: usize = 500;
    const REPEAT_STEPS: usize = 10;
    const CROSS_BACKEND_STEPS: usize = 10;
    const ROLLOUT_LEN: usize = 2;
    const DECODE_LEN: usize = 8;
    const PERTURB_SCALE: f32 = 0.05;
    const LEARNING_RATE: f32 = 1.0e-3;
    const GRAD_CLIP: f32 = 1.0;

    const PROMPTS: &[&[u32]] = &[&[1, 3, 8], &[2, 5, 13], &[7, 11, 19]];
    const EVAL_STEPS: &[usize] = &[0, 50, 100, 500];

    #[derive(Clone, Copy, Debug)]
    enum BackendKind {
        Cpu,
        Cuda,
    }

    #[derive(Debug)]
    struct EvalSnapshot {
        step: usize,
        prompt_index: usize,
        teacher_tokens: Vec<u32>,
        student_tokens: Vec<u32>,
        overlap_pct: f64,
    }

    #[derive(Debug)]
    struct TrainingRun {
        backend: BackendKind,
        losses: Vec<f64>,
        observed_rollouts: Vec<Vec<u32>>,
        eval_snapshots: Vec<EvalSnapshot>,
        wall_seconds: f64,
    }

    #[derive(Debug)]
    struct Lcg {
        state: u64,
    }

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_f32(&mut self) -> f32 {
            self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let bits = ((self.state >> 32) as u32) as f32 / (u32::MAX as f32);
            bits * 2.0 - 1.0
        }
    }

    pub fn main() -> Result<(), Box<dyn std::error::Error>> {
        println!(
            "config hidden=512 intermediate=1536 layers=12 vocab=32768 heads=8 kv_heads=4 head_dim=64 train_steps={TRAIN_STEPS} repeat_steps={REPEAT_STEPS} cross_backend_steps={CROSS_BACKEND_STEPS} prompt={:?} rollout_len={ROLLOUT_LEN} decode_len={DECODE_LEN} lr={LEARNING_RATE} perturb_scale={PERTURB_SCALE}",
            PROMPTS[0]
        );
        report_real_checkpoint_probe();

        let cuda_repeat_a = run_training(BackendKind::Cuda, REPEAT_STEPS, &[], true)?;
        let cuda_repeat_b = run_training(BackendKind::Cuda, REPEAT_STEPS, &[], true)?;
        report_repeat("cuda_repeat", &cuda_repeat_a, &cuda_repeat_b);

        let cpu_cross = run_training(BackendKind::Cpu, CROSS_BACKEND_STEPS, &[], true)?;
        let cuda_cross = run_training(BackendKind::Cuda, CROSS_BACKEND_STEPS, &[], true)?;
        report_cross_backend(&cpu_cross, &cuda_cross);

        let cuda_main = run_training(BackendKind::Cuda, TRAIN_STEPS, EVAL_STEPS, true)?;
        report_main_run(&cuda_main);

        Ok(())
    }

    fn run_training(
        backend: BackendKind,
        steps: usize,
        eval_steps: &[usize],
        collect_rollouts: bool,
    ) -> Result<TrainingRun, Box<dyn std::error::Error>> {
        let cuda_backend = match backend {
            BackendKind::Cpu => None,
            BackendKind::Cuda => Some(Arc::new(CudaBackend::new(0)?)),
        };
        let mut store = match &cuda_backend {
            Some(backend) => TensorStore::with_backend(backend.clone()),
            None => TensorStore::default(),
        };
        let mut tape = Tape::new();
        tape.set_enabled(true);

        let config = moderate_config();
        let teacher = Qwen35Model::new_for_eval(&config, &mut store)?;
        let student = Qwen35Model::new(&config, &mut store)?;
        perturb_student(&student, &mut store, 0xa11c_e55d, PERTURB_SCALE);

        let student_params = student.all_parameter_ids();
        let teacher_params = teacher.all_parameter_ids();
        let mut optimizer = match cuda_backend {
            Some(backend) => {
                AdamW::new_with_device(LEARNING_RATE, (0.9, 0.999), 1.0e-8, 0.0, backend)
            }
            None => AdamW::new(LEARNING_RATE, (0.9, 0.999), 1.0e-8, 0.0),
        };
        let step_config = OpdStepConfig {
            rollout_len: ROLLOUT_LEN,
            grad_clip: GRAD_CLIP,
        };

        let mut eval_snapshots = Vec::new();
        if eval_steps.contains(&0) {
            eval_snapshots.extend(evaluate_decode_overlap(
                0,
                &teacher,
                &student,
                &teacher_params,
                &student_params,
                &mut store,
                &mut tape,
            )?);
        }

        let mut losses = Vec::with_capacity(steps);
        let mut observed_rollouts = Vec::with_capacity(if collect_rollouts { steps } else { 0 });
        let started = Instant::now();

        for step_idx in 0..steps {
            if collect_rollouts {
                let rollout = greedy_decode_suffix(
                    &student,
                    PROMPTS[0],
                    ROLLOUT_LEN,
                    &teacher_params,
                    &student_params,
                    &mut store,
                    &mut tape,
                )?;
                observed_rollouts.push(rollout);
            }

            let outcome = opd_step(
                &student,
                &teacher,
                PROMPTS[0],
                step_config,
                &student_params,
                &mut optimizer,
                &mut store,
                &mut tape,
            )?;
            losses.push(outcome.loss as f64);

            let completed_step = step_idx + 1;
            if eval_steps.contains(&completed_step) {
                eval_snapshots.extend(evaluate_decode_overlap(
                    completed_step,
                    &teacher,
                    &student,
                    &teacher_params,
                    &student_params,
                    &mut store,
                    &mut tape,
                )?);
            }
        }

        Ok(TrainingRun {
            backend,
            losses,
            observed_rollouts,
            eval_snapshots,
            wall_seconds: started.elapsed().as_secs_f64(),
        })
    }

    fn report_real_checkpoint_probe() {
        let explicit = env::var_os("ARLE_OPD_REAL_MODEL_DIR").map(PathBuf::from);
        let default = PathBuf::from("/home/ckl/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B");
        let path = explicit.unwrap_or(default);
        let present = path.join("config.json").exists() && path.join("model.safetensors").exists();
        let mode = if env::var_os("ARLE_OPD_REAL_MODEL_DIR").is_some() {
            "explicit"
        } else {
            "auto_probe"
        };
        println!(
            "real_checkpoint_probe mode={mode} path={} present={} run=false reason=\"convergence bench keeps the exercised substrate at the moderate shape; full Qwen3-0.6B OPD eval is recorded as follow-up unless explicitly promoted to a separate memory-budgeted run\"",
            path.display(),
            present
        );
    }

    fn report_repeat(label: &str, a: &TrainingRun, b: &TrainingRun) {
        let loss_bit_identical = a
            .losses
            .iter()
            .zip(&b.losses)
            .all(|(left, right)| left.to_bits() == right.to_bits());
        let rollout_identical = a.observed_rollouts == b.observed_rollouts;
        let max_abs_loss_diff = a
            .losses
            .iter()
            .zip(&b.losses)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f64, f64::max);

        println!(
            "{label} backend={:?} steps={} loss_bit_identical={} rollout_identical={} max_abs_loss_diff={:.12e}",
            a.backend,
            a.losses.len().min(b.losses.len()),
            loss_bit_identical,
            rollout_identical,
            max_abs_loss_diff
        );
    }

    fn report_cross_backend(cpu: &TrainingRun, cuda: &TrainingRun) {
        let mut rollout_match_steps = 0usize;
        let mut first_divergence = None;
        for (idx, (cpu_rollout, cuda_rollout)) in cpu
            .observed_rollouts
            .iter()
            .zip(&cuda.observed_rollouts)
            .enumerate()
        {
            if cpu_rollout == cuda_rollout {
                rollout_match_steps += 1;
            } else if first_divergence.is_none() {
                first_divergence = Some((idx + 1, cpu_rollout.clone(), cuda_rollout.clone()));
            }
        }

        let max_loss_relerr = cpu
            .losses
            .iter()
            .zip(&cuda.losses)
            .map(|(left, right)| relerr(*left, *right))
            .fold(0.0_f64, f64::max);

        match first_divergence {
            Some((step, cpu_tokens, cuda_tokens)) => println!(
                "cross_backend steps={} rollout_match_steps={}/{} max_loss_relerr={:.12e} first_divergence_step={} cpu={:?} cuda={:?}",
                cpu.losses.len().min(cuda.losses.len()),
                rollout_match_steps,
                cpu.observed_rollouts.len().min(cuda.observed_rollouts.len()),
                max_loss_relerr,
                step,
                cpu_tokens,
                cuda_tokens
            ),
            None => println!(
                "cross_backend steps={} rollout_match_steps={}/{} max_loss_relerr={:.12e} first_divergence_step=none",
                cpu.losses.len().min(cuda.losses.len()),
                rollout_match_steps,
                cpu.observed_rollouts.len().min(cuda.observed_rollouts.len()),
                max_loss_relerr
            ),
        }
    }

    fn report_main_run(run: &TrainingRun) {
        let first = run.losses.first().copied().unwrap_or(0.0);
        let last = run.losses.last().copied().unwrap_or(0.0);
        let min = run.losses.iter().copied().fold(f64::INFINITY, f64::min);
        let delta_pct = if first == 0.0 {
            0.0
        } else {
            ((last - first) / first) * 100.0
        };
        let per_step_seconds = run.wall_seconds / run.losses.len().max(1) as f64;

        println!(
            "main_summary backend={:?} steps={} wall_seconds={:.6} step_seconds={:.9} first_loss={:.12e} last_loss={:.12e} min_loss={:.12e} delta_pct={:.6}",
            run.backend,
            run.losses.len(),
            run.wall_seconds,
            per_step_seconds,
            first,
            last,
            min,
            delta_pct
        );

        for (idx, loss) in run.losses.iter().enumerate() {
            let step = idx + 1;
            if step == 1 || step % 10 == 0 || step == run.losses.len() {
                println!("loss_trajectory step={} loss={:.12e}", step, loss);
            }
        }

        let mut overlap_by_step: HashMap<usize, Vec<f64>> = HashMap::new();
        for snapshot in &run.eval_snapshots {
            overlap_by_step
                .entry(snapshot.step)
                .or_default()
                .push(snapshot.overlap_pct);
            println!(
                "decode_overlap step={} prompt_index={} overlap_pct={:.3} teacher={:?} student={:?}",
                snapshot.step,
                snapshot.prompt_index,
                snapshot.overlap_pct,
                snapshot.teacher_tokens,
                snapshot.student_tokens
            );
        }
        let mut steps: Vec<_> = overlap_by_step.keys().copied().collect();
        steps.sort_unstable();
        for step in steps {
            let values = &overlap_by_step[&step];
            let mean = values.iter().sum::<f64>() / values.len().max(1) as f64;
            println!(
                "decode_overlap_summary step={} mean_overlap_pct={:.3}",
                step, mean
            );
        }
    }

    fn evaluate_decode_overlap(
        step: usize,
        teacher: &Qwen35Model,
        student: &Qwen35Model,
        teacher_params: &[TensorId],
        student_params: &[TensorId],
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<Vec<EvalSnapshot>, Box<dyn std::error::Error>> {
        let mut snapshots = Vec::with_capacity(PROMPTS.len());
        for (prompt_index, prompt) in PROMPTS.iter().enumerate() {
            let teacher_tokens = greedy_decode_suffix(
                teacher,
                prompt,
                DECODE_LEN,
                teacher_params,
                student_params,
                store,
                tape,
            )?;
            let student_tokens = greedy_decode_suffix(
                student,
                prompt,
                DECODE_LEN,
                teacher_params,
                student_params,
                store,
                tape,
            )?;
            let matches = teacher_tokens
                .iter()
                .zip(&student_tokens)
                .filter(|(left, right)| left == right)
                .count();
            let overlap_pct = (matches as f64 / DECODE_LEN as f64) * 100.0;
            snapshots.push(EvalSnapshot {
                step,
                prompt_index,
                teacher_tokens,
                student_tokens,
                overlap_pct,
            });
        }
        Ok(snapshots)
    }

    fn greedy_decode_suffix(
        model: &Qwen35Model,
        prompt: &[u32],
        generated_len: usize,
        teacher_params: &[TensorId],
        student_params: &[TensorId],
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        tape.entries.clear();
        tape.set_enabled(false);

        let mut tokens = prompt.to_vec();
        for _ in 0..generated_len {
            let positions: Vec<u32> = (0..tokens.len() as u32).collect();
            let logits = model.forward(store, tape, &tokens, &positions)?;
            let next = argmax_last_token(store, logits, tokens.len(), model.config().vocab_size)?;
            tokens.push(next);
        }

        tape.entries.clear();
        tape.set_enabled(true);
        retain_model_state(store, teacher_params, student_params);
        Ok(tokens[prompt.len()..].to_vec())
    }

    fn argmax_last_token(
        store: &mut TensorStore,
        logits: TensorId,
        seq_len: usize,
        vocab_size: usize,
    ) -> Result<u32, Box<dyn std::error::Error>> {
        let logits_host = store.to_host(logits)?;
        let start = (seq_len - 1) * vocab_size;
        let row = &logits_host[start..start + vocab_size];
        let mut best_idx = 0usize;
        let mut best_value = f32::NEG_INFINITY;
        for (idx, value) in row.iter().enumerate() {
            if *value > best_value {
                best_value = *value;
                best_idx = idx;
            }
        }
        Ok(best_idx as u32)
    }

    fn retain_model_state(
        store: &mut TensorStore,
        teacher_params: &[TensorId],
        student_params: &[TensorId],
    ) {
        let mut keep = HashSet::new();
        for param_id in teacher_params.iter().chain(student_params) {
            keep.insert(*param_id);
            if let Some(tensor) = store.get(*param_id) {
                if let Some(grad_id) = tensor.grad {
                    keep.insert(grad_id);
                }
            }
        }
        store.retain_ids(&keep);
    }

    fn perturb_student(model: &Qwen35Model, store: &mut TensorStore, seed: u64, scale: f32) {
        let mut rng = Lcg::new(seed);
        for id in model.all_parameter_ids() {
            if let Some(tensor) = store.get_mut(id) {
                if !tensor.requires_grad {
                    continue;
                }
                for value in &mut tensor.data {
                    *value += rng.next_f32() * scale;
                }
            }
        }
    }

    fn moderate_config() -> Qwen35Config {
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

    fn relerr(left: f64, right: f64) -> f64 {
        let denom = left.abs().max(right.abs()).max(1.0e-12);
        (left - right).abs() / denom
    }
}

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    app::main()
}

#[cfg(not(all(feature = "cuda", not(feature = "no-cuda"))))]
fn main() {
    eprintln!(
        "opd_step_cuda_convergence_bench requires --features cuda without the no-cuda feature"
    );
}
