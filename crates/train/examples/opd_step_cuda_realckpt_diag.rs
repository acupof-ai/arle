#![cfg_attr(
    not(all(feature = "cuda", not(feature = "no-cuda"))),
    allow(dead_code, unused_imports)
)]

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
mod app {
    use std::{
        collections::HashSet,
        env,
        path::{Path, PathBuf},
        sync::Arc,
        time::Instant,
    };

    use autograd::{backend_cuda::CudaBackend, optim::AdamW, Tape, TensorId, TensorStore};
    use train::{
        opd::{opd_step, OpdStepConfig},
        qwen35::{forward_rollout_cached, Qwen35KvCache, Qwen35Model},
        qwen35_loader::{load_qwen35_from_hf_dir, load_qwen35_trainable_from_hf_dir},
        trainer::extend_keep_with_params_and_grads,
    };

    const DEFAULT_MODEL_DIR: &str = "/home/ckl/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B";
    const STEPS_PER_CONFIG: usize = 100;
    const ROLLOUT_LEN: usize = 8;
    const DECODE_LEN: usize = 16;
    const GRAD_CLIP: f32 = 1.0;
    const PERTURB_SEED: u64 = 0x0f0d_cafe_2026_0521;
    const EVAL_STEPS: &[usize] = &[0, 25, 50, 100];

    const TRAIN_PROMPTS: &[&[u32]] = &[
        &[1, 872, 198, 3456],
        &[1, 198, 1512, 429],
        &[1, 770, 3186, 25, 220],
        &[1, 644, 374, 279, 1887],
        &[1, 3838, 374, 264, 2077, 13],
        &[1, 785, 594, 287, 374, 1690],
        &[1, 3347, 11, 358, 1052, 429],
        &[1, 2610, 527, 1139, 304, 279, 1670],
    ];

    const HELDOUT_PROMPTS: &[&[u32]] = &[
        &[1, 4438, 374, 279, 2768],
        &[1, 1516, 374, 264, 1296, 4339],
        &[1, 785, 1401, 315, 279, 1967],
        &[1, 3198, 279, 1296, 25, 220],
    ];

    const CONFIGS: &[DiagConfig] = &[
        DiagConfig {
            label: "A_min_perturb_same_lr",
            perturb_scale: 1.0e-5,
            learning_rate: 5.0e-5,
        },
        DiagConfig {
            label: "B_same_perturb_min_lr",
            perturb_scale: 1.0e-3,
            learning_rate: 1.0e-7,
        },
        DiagConfig {
            label: "C_min_perturb_min_lr",
            perturb_scale: 1.0e-5,
            learning_rate: 1.0e-7,
        },
    ];

    type AnyResult<T> = Result<T, Box<dyn std::error::Error>>;

    #[derive(Debug, Clone, Copy)]
    struct DiagConfig {
        label: &'static str,
        perturb_scale: f32,
        learning_rate: f32,
    }

    #[derive(Debug, Clone)]
    struct EvalSummary {
        step: usize,
        train_overlap_pct: f64,
        heldout_overlap_pct: f64,
        train_kl: f64,
        heldout_kl: f64,
    }

    #[derive(Debug)]
    struct DecodeEval {
        overlap_pct: f64,
        kl: f64,
    }

    #[derive(Debug)]
    struct RunSummary {
        config: DiagConfig,
        evals: Vec<EvalSummary>,
        first_loss: f64,
        final_loss: f64,
        mean_step_seconds: f64,
        median_step_seconds: f64,
        total_wall_seconds: f64,
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
            self.state = self
                .state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = ((self.state >> 32) as u32) as f32 / (u32::MAX as f32);
            unit * 2.0 - 1.0
        }
    }

    pub fn main() -> AnyResult<()> {
        let model_dir = resolve_model_dir()?;
        print_global_config(&model_dir);

        let cuda_backend = Arc::new(CudaBackend::new(0)?);
        let mut summaries = Vec::with_capacity(CONFIGS.len());
        for config in CONFIGS {
            summaries.push(run_config(*config, &model_dir, cuda_backend.clone())?);
        }
        print_diagnosis_matrix(&summaries);
        Ok(())
    }

    fn run_config(
        config: DiagConfig,
        model_dir: &Path,
        cuda_backend: Arc<CudaBackend>,
    ) -> AnyResult<RunSummary> {
        let config_started = Instant::now();
        println!(
            "config_start label={} perturb_scale={:.9e} lr={:.9e} rollout_len={ROLLOUT_LEN} steps={STEPS_PER_CONFIG}",
            config.label, config.perturb_scale, config.learning_rate
        );

        let mut store = TensorStore::with_backend(cuda_backend.clone());
        let mut tape = Tape::new();

        let teacher_load_started = Instant::now();
        let teacher = load_qwen35_from_hf_dir(model_dir, &mut store)?;
        let teacher_load_seconds = teacher_load_started.elapsed().as_secs_f64();
        let student_load_started = Instant::now();
        let student = load_qwen35_trainable_from_hf_dir(model_dir, &mut store)?;
        let student_load_seconds = student_load_started.elapsed().as_secs_f64();

        let teacher_params = teacher.all_parameter_ids();
        let student_model_params = student.all_parameter_ids();
        let student_trainable_params = trainable_params(&student, &store);
        perturb_params(
            &student_trainable_params,
            &mut store,
            PERTURB_SEED,
            config.perturb_scale,
        );
        let mut optimizer = AdamW::new_with_device(
            config.learning_rate,
            (0.9, 0.999),
            1.0e-8,
            0.0,
            cuda_backend,
        );
        let step_config = OpdStepConfig {
            rollout_len: ROLLOUT_LEN,
            grad_clip: GRAD_CLIP,
        };

        println!(
            "model_summary label={} hidden={} intermediate={} layers={} vocab={} num_heads={} num_kv_heads={} head_dim={} teacher_params={} student_model_params={} student_trainable_params={} teacher_load_seconds={teacher_load_seconds:.6} student_load_seconds={student_load_seconds:.6}",
            config.label,
            student.config().hidden_size,
            student.config().intermediate_size,
            student.config().num_hidden_layers,
            student.config().vocab_size,
            student.config().num_attention_heads,
            student.config().num_key_value_heads,
            student.config().head_dim,
            param_element_count(&teacher_params, &store),
            param_element_count(&student_model_params, &store),
            param_element_count(&student_trainable_params, &store)
        );

        let mut evals = Vec::with_capacity(EVAL_STEPS.len());
        evals.push(evaluate_snapshot(
            config.label,
            0,
            &teacher,
            &student,
            &teacher_params,
            &student_model_params,
            &mut store,
            &mut tape,
        )?);

        let mut losses = Vec::with_capacity(STEPS_PER_CONFIG);
        let mut step_seconds = Vec::with_capacity(STEPS_PER_CONFIG);
        for step in 1..=STEPS_PER_CONFIG {
            let prompt_index = (step - 1) % TRAIN_PROMPTS.len();
            let prompt = TRAIN_PROMPTS[prompt_index];
            let step_started = Instant::now();
            let outcome = opd_step(
                &student,
                &teacher,
                prompt,
                step_config,
                &student_trainable_params,
                &mut optimizer,
                &mut store,
                &mut tape,
            )?;
            let elapsed = step_started.elapsed().as_secs_f64();
            losses.push(outcome.loss as f64);
            step_seconds.push(elapsed);
            if step == 1 || step % 10 == 0 || EVAL_STEPS.contains(&step) {
                println!(
                    "train_step label={} step={step} prompt_index={prompt_index} loss={:.12e} rollout_len={} step_seconds={elapsed:.6}",
                    config.label, outcome.loss, outcome.rollout_len
                );
            }
            if EVAL_STEPS.contains(&step) {
                evals.push(evaluate_snapshot(
                    config.label,
                    step,
                    &teacher,
                    &student,
                    &teacher_params,
                    &student_model_params,
                    &mut store,
                    &mut tape,
                )?);
            }
        }

        let mut sorted_step_seconds = step_seconds.clone();
        sorted_step_seconds.sort_by(f64::total_cmp);
        let mean_step_seconds = mean(step_seconds.iter().copied());
        let median_step_seconds = sorted_step_seconds[sorted_step_seconds.len() / 2];
        let first_loss = losses.first().copied().unwrap_or(0.0);
        let final_loss = losses.last().copied().unwrap_or(0.0);
        let total_wall_seconds = config_started.elapsed().as_secs_f64();
        let summary = RunSummary {
            config,
            evals,
            first_loss,
            final_loss,
            mean_step_seconds,
            median_step_seconds,
            total_wall_seconds,
        };
        print_config_summary(&summary);
        Ok(summary)
    }

    fn evaluate_snapshot(
        label: &'static str,
        step: usize,
        teacher: &Qwen35Model,
        student: &Qwen35Model,
        teacher_params: &[TensorId],
        student_params: &[TensorId],
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> AnyResult<EvalSummary> {
        let started = Instant::now();
        let train = evaluate_split(
            label,
            "train",
            step,
            TRAIN_PROMPTS,
            teacher,
            student,
            teacher_params,
            student_params,
            store,
            tape,
        )?;
        let heldout = evaluate_split(
            label,
            "heldout",
            step,
            HELDOUT_PROMPTS,
            teacher,
            student,
            teacher_params,
            student_params,
            store,
            tape,
        )?;
        let summary = EvalSummary {
            step,
            train_overlap_pct: mean(train.iter().map(|eval| eval.overlap_pct)),
            heldout_overlap_pct: mean(heldout.iter().map(|eval| eval.overlap_pct)),
            train_kl: mean(train.iter().map(|eval| eval.kl)),
            heldout_kl: mean(heldout.iter().map(|eval| eval.kl)),
        };
        println!(
            "eval_summary label={label} step={step} train_overlap_pct={:.6} heldout_overlap_pct={:.6} train_kl={:.12e} heldout_kl={:.12e} eval_seconds={:.6}",
            summary.train_overlap_pct,
            summary.heldout_overlap_pct,
            summary.train_kl,
            summary.heldout_kl,
            started.elapsed().as_secs_f64()
        );
        Ok(summary)
    }

    fn evaluate_split(
        label: &'static str,
        split: &'static str,
        step: usize,
        prompts: &[&[u32]],
        teacher: &Qwen35Model,
        student: &Qwen35Model,
        teacher_params: &[TensorId],
        student_params: &[TensorId],
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> AnyResult<Vec<DecodeEval>> {
        let mut rows = Vec::with_capacity(prompts.len());
        for (index, prompt) in prompts.iter().enumerate() {
            let teacher_suffix = greedy_decode_suffix_cached(
                teacher,
                prompt,
                DECODE_LEN,
                teacher_params,
                student_params,
                store,
                tape,
            )?;
            let student_suffix = greedy_decode_suffix_cached(
                student,
                prompt,
                DECODE_LEN,
                teacher_params,
                student_params,
                store,
                tape,
            )?;
            let overlap_pct = exact_overlap_pct(&student_suffix, &teacher_suffix);
            let kl = mean_per_token_forward_kl(
                teacher,
                student,
                prompt,
                &teacher_suffix,
                teacher_params,
                student_params,
                store,
                tape,
            )?;
            println!(
                "eval_detail label={label} step={step} split={split} prompt_index={index} overlap_pct={overlap_pct:.6} kl={kl:.12e}"
            );
            rows.push(DecodeEval { overlap_pct, kl });
        }
        Ok(rows)
    }

    fn greedy_decode_suffix_cached(
        model: &Qwen35Model,
        prompt: &[u32],
        decode_len: usize,
        teacher_params: &[TensorId],
        student_params: &[TensorId],
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> AnyResult<Vec<u32>> {
        if prompt.is_empty() {
            return Err("greedy decode requires a non-empty prompt".into());
        }
        tape.entries.clear();
        tape.set_enabled(false);
        let mut cache = Qwen35KvCache::new(model);
        let mut tokens = prompt.to_vec();
        let mut suffix = Vec::with_capacity(decode_len);
        let vocab = model.config().vocab_size;
        for decode_step in 0..decode_len {
            let (input_ids, positions) = if decode_step == 0 {
                (tokens.clone(), (0..tokens.len() as u32).collect::<Vec<_>>())
            } else {
                let last = *tokens
                    .last()
                    .ok_or("greedy decode cache cannot decode from empty token list")?;
                (vec![last], vec![(tokens.len() - 1) as u32])
            };
            let logits =
                forward_rollout_cached(model, store, tape, &input_ids, &positions, &mut cache)?;
            let next = greedy_next_token(logits, 1, vocab, store)?;
            tokens.push(next);
            suffix.push(next);
        }
        retain_model_state(store, tape, teacher_params, student_params);
        Ok(suffix)
    }

    fn mean_per_token_forward_kl(
        teacher: &Qwen35Model,
        student: &Qwen35Model,
        prompt: &[u32],
        teacher_suffix: &[u32],
        teacher_params: &[TensorId],
        student_params: &[TensorId],
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> AnyResult<f64> {
        if prompt.is_empty() || teacher_suffix.len() < DECODE_LEN {
            return Err("KL eval requires a non-empty prompt and full teacher suffix".into());
        }
        let mut sequence = prompt.to_vec();
        sequence.extend_from_slice(teacher_suffix);
        let positions = (0..sequence.len() as u32).collect::<Vec<_>>();

        tape.entries.clear();
        tape.set_enabled(false);
        let teacher_logits = teacher.forward(store, tape, &sequence, &positions)?;
        let student_logits = student.forward(store, tape, &sequence, &positions)?;
        let teacher_host = store.to_host(teacher_logits)?;
        let student_host = store.to_host(student_logits)?;
        let vocab = teacher.config().vocab_size;
        let first_row = prompt.len() - 1;
        let kl = mean_forward_kl_rows(&teacher_host, &student_host, first_row, DECODE_LEN, vocab)?;
        retain_model_state(store, tape, teacher_params, student_params);
        Ok(kl)
    }

    fn mean_forward_kl_rows(
        teacher_logits: &[f32],
        student_logits: &[f32],
        first_row: usize,
        rows: usize,
        vocab: usize,
    ) -> AnyResult<f64> {
        let required = (first_row + rows) * vocab;
        if teacher_logits.len() < required || student_logits.len() < required {
            return Err(format!(
                "KL logits are too short: required {required}, teacher={}, student={}",
                teacher_logits.len(),
                student_logits.len()
            )
            .into());
        }
        let mut total = 0.0f64;
        for row_idx in first_row..first_row + rows {
            let offset = row_idx * vocab;
            total += forward_kl_row(
                &teacher_logits[offset..offset + vocab],
                &student_logits[offset..offset + vocab],
            );
        }
        Ok(total / rows as f64)
    }

    fn forward_kl_row(teacher: &[f32], student: &[f32]) -> f64 {
        let teacher_max = teacher.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let student_max = student.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let teacher_sum_exp = teacher
            .iter()
            .map(|&value| ((value as f64) - teacher_max).exp())
            .sum::<f64>();
        let student_sum_exp = student
            .iter()
            .map(|&value| ((value as f64) - student_max).exp())
            .sum::<f64>();
        let teacher_log_z = teacher_max + teacher_sum_exp.ln();
        let student_log_z = student_max + student_sum_exp.ln();
        teacher
            .iter()
            .zip(student)
            .map(|(&teacher_value, &student_value)| {
                let teacher_log_prob = teacher_value as f64 - teacher_log_z;
                let student_log_prob = student_value as f64 - student_log_z;
                teacher_log_prob.exp() * (teacher_log_prob - student_log_prob)
            })
            .sum()
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
        for (idx, &value) in row.iter().enumerate() {
            if value > best_val {
                best_val = value;
                best_idx = idx;
            }
        }
        Ok(best_idx as u32)
    }

    fn print_global_config(model_dir: &Path) {
        println!(
            "diag_config backend=cuda model_dir={} steps_per_config={STEPS_PER_CONFIG} rollout_len={ROLLOUT_LEN} decode_len={DECODE_LEN} grad_clip={GRAD_CLIP} perturb_seed=0x{PERTURB_SEED:016x}",
            model_dir.display()
        );
        for config in CONFIGS {
            println!(
                "diag_case label={} perturb_scale={:.9e} lr={:.9e}",
                config.label, config.perturb_scale, config.learning_rate
            );
        }
        for (idx, prompt) in TRAIN_PROMPTS.iter().enumerate() {
            println!("prompt split=train index={idx} ids={prompt:?}");
        }
        for (idx, prompt) in HELDOUT_PROMPTS.iter().enumerate() {
            println!("prompt split=heldout index={idx} ids={prompt:?}");
        }
    }

    fn print_config_summary(summary: &RunSummary) {
        let start = summary
            .evals
            .iter()
            .find(|eval| eval.step == 0)
            .expect("step 0 eval exists");
        let end = summary
            .evals
            .iter()
            .find(|eval| eval.step == STEPS_PER_CONFIG)
            .expect("final eval exists");
        println!(
            "config_summary label={} first_loss={:.12e} final_loss={:.12e} sampled_loss_reduction_pct={:.6} train_kl_start={:.12e} train_kl_final={:.12e} train_kl_reduction_pct={:.6} train_overlap_start={:.6} train_overlap_final={:.6} heldout_kl_start={:.12e} heldout_kl_final={:.12e} heldout_overlap_start={:.6} heldout_overlap_final={:.6} mean_step_seconds={:.6} median_step_seconds={:.6} total_wall_seconds={:.6}",
            summary.config.label,
            summary.first_loss,
            summary.final_loss,
            pct_reduction(summary.first_loss, summary.final_loss),
            start.train_kl,
            end.train_kl,
            pct_reduction(start.train_kl, end.train_kl),
            start.train_overlap_pct,
            end.train_overlap_pct,
            start.heldout_kl,
            end.heldout_kl,
            start.heldout_overlap_pct,
            end.heldout_overlap_pct,
            summary.mean_step_seconds,
            summary.median_step_seconds,
            summary.total_wall_seconds
        );
        for eval in &summary.evals {
            println!(
                "summary_eval_row label={} step={} train_overlap_pct={:.6} heldout_overlap_pct={:.6} train_kl={:.12e} heldout_kl={:.12e}",
                summary.config.label,
                eval.step,
                eval.train_overlap_pct,
                eval.heldout_overlap_pct,
                eval.train_kl,
                eval.heldout_kl
            );
        }
    }

    fn print_diagnosis_matrix(summaries: &[RunSummary]) {
        for summary in summaries {
            let start = summary
                .evals
                .iter()
                .find(|eval| eval.step == 0)
                .expect("step 0 eval exists");
            let end = summary
                .evals
                .iter()
                .find(|eval| eval.step == STEPS_PER_CONFIG)
                .expect("final eval exists");
            let train_kl_ratio = if start.train_kl == 0.0 {
                f64::INFINITY
            } else {
                end.train_kl / start.train_kl
            };
            let stable =
                train_kl_ratio <= 1.25 && end.train_overlap_pct >= start.train_overlap_pct - 5.0;
            println!(
                "diagnosis_matrix label={} perturb_scale={:.9e} lr={:.9e} stable={} train_kl_ratio={:.6} train_overlap_delta_pct={:.6} heldout_kl_ratio={:.6}",
                summary.config.label,
                summary.config.perturb_scale,
                summary.config.learning_rate,
                stable,
                train_kl_ratio,
                end.train_overlap_pct - start.train_overlap_pct,
                if start.heldout_kl == 0.0 {
                    f64::INFINITY
                } else {
                    end.heldout_kl / start.heldout_kl
                }
            );
        }
    }

    fn retain_model_state(
        store: &mut TensorStore,
        tape: &mut Tape,
        teacher_params: &[TensorId],
        student_params: &[TensorId],
    ) {
        tape.entries.clear();
        tape.set_enabled(true);
        let mut keep = HashSet::with_capacity((teacher_params.len() + student_params.len()) * 2);
        extend_keep_with_params_and_grads(&mut keep, teacher_params.iter().copied(), store);
        extend_keep_with_params_and_grads(&mut keep, student_params.iter().copied(), store);
        store.retain_ids(&keep);
    }

    fn resolve_model_dir() -> AnyResult<PathBuf> {
        let path = env::var_os("ARLE_OPD_REALCKPT_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_DIR));
        if !path.join("config.json").is_file() || !path.join("model.safetensors").is_file() {
            return Err(format!(
                "{} is not a complete Qwen3-0.6B ModelScope checkpoint directory",
                path.display()
            )
            .into());
        }
        Ok(path)
    }

    fn trainable_params(model: &Qwen35Model, store: &TensorStore) -> Vec<TensorId> {
        model
            .all_parameter_ids()
            .into_iter()
            .filter(|id| store.get(*id).is_some_and(|tensor| tensor.requires_grad))
            .collect()
    }

    fn param_element_count(params: &[TensorId], store: &TensorStore) -> usize {
        params
            .iter()
            .filter_map(|id| store.get(*id).map(|tensor| tensor.size))
            .sum()
    }

    fn perturb_params(params: &[TensorId], store: &mut TensorStore, seed: u64, scale: f32) {
        let mut rng = Lcg::new(seed);
        for &id in params {
            if let Some(tensor) = store.get_mut(id) {
                for value in &mut tensor.data {
                    *value += rng.next_f32() * scale;
                }
            }
        }
    }

    fn exact_overlap_pct(student: &[u32], teacher: &[u32]) -> f64 {
        let compared = student.len().min(teacher.len());
        if compared == 0 {
            return 0.0;
        }
        let matches = student
            .iter()
            .zip(teacher)
            .take(compared)
            .filter(|(student, teacher)| student == teacher)
            .count();
        matches as f64 / compared as f64 * 100.0
    }

    fn mean(values: impl Iterator<Item = f64>) -> f64 {
        let mut total = 0.0;
        let mut count = 0usize;
        for value in values {
            total += value;
            count += 1;
        }
        if count == 0 {
            0.0
        } else {
            total / count as f64
        }
    }

    fn pct_reduction(start: f64, end: f64) -> f64 {
        if start == 0.0 {
            0.0
        } else {
            (start - end) / start * 100.0
        }
    }
}

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    app::main()
}

#[cfg(not(all(feature = "cuda", not(feature = "no-cuda"))))]
fn main() {
    eprintln!(
        "opd_step_cuda_realckpt_diag requires CUDA. Run with: \
         cargo run -p train --example opd_step_cuda_realckpt_diag --release --features cuda"
    );
    std::process::exit(1);
}
