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
    const TRAIN_STEPS: usize = 500;
    const ROLLOUT_LEN: usize = 8;
    const DECODE_LEN: usize = 16;
    const LEARNING_RATE: f32 = 5.0e-5;
    const GRAD_CLIP: f32 = 1.0;
    const PERTURB_SCALE: f32 = 1.0e-3;
    const PERTURB_SEED: u64 = 0x0f0d_cafe_2026_0521;
    const SAFETY_FIRST_STEP_MAX_SECONDS: f64 = 0.5;
    const EVAL_STEPS: &[usize] = &[0, 50, 100, 200, 500];

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

    type AnyResult<T> = Result<T, Box<dyn std::error::Error>>;

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
        print_config(&model_dir);

        let cuda_backend = Arc::new(CudaBackend::new(0)?);
        let mut store = TensorStore::with_backend(cuda_backend.clone());
        let mut tape = Tape::new();

        let teacher_load_started = Instant::now();
        let teacher = load_qwen35_from_hf_dir(&model_dir, &mut store)?;
        let teacher_load_seconds = teacher_load_started.elapsed().as_secs_f64();
        let student_load_started = Instant::now();
        let student = load_qwen35_trainable_from_hf_dir(&model_dir, &mut store)?;
        let student_load_seconds = student_load_started.elapsed().as_secs_f64();

        let teacher_params = teacher.all_parameter_ids();
        let student_model_params = student.all_parameter_ids();
        let student_trainable_params = trainable_params(&student, &store);
        let teacher_param_elements = param_element_count(&teacher_params, &store);
        let student_model_elements = param_element_count(&student_model_params, &store);
        let student_trainable_elements = param_element_count(&student_trainable_params, &store);

        perturb_params(
            &student_trainable_params,
            &mut store,
            PERTURB_SEED,
            PERTURB_SCALE,
        );
        let mut optimizer =
            AdamW::new_with_device(LEARNING_RATE, (0.9, 0.999), 1.0e-8, 0.0, cuda_backend);
        let step_config = OpdStepConfig {
            rollout_len: ROLLOUT_LEN,
            grad_clip: GRAD_CLIP,
        };

        println!(
            "model_summary hidden={} intermediate={} layers={} vocab={} num_heads={} num_kv_heads={} head_dim={} tie_word_embeddings={} rope_theta={} teacher_param_elements={} student_model_elements={} student_trainable_elements={} teacher_load_seconds={teacher_load_seconds:.6} student_load_seconds={student_load_seconds:.6}",
            student.config().hidden_size,
            student.config().intermediate_size,
            student.config().num_hidden_layers,
            student.config().vocab_size,
            student.config().num_attention_heads,
            student.config().num_key_value_heads,
            student.config().head_dim,
            student.config().tie_word_embeddings,
            student.config().rope_theta,
            teacher_param_elements,
            student_model_elements,
            student_trainable_elements
        );

        let mut eval_summaries = Vec::new();
        eval_summaries.push(evaluate_snapshot(
            0,
            &teacher,
            &student,
            &teacher_params,
            &student_model_params,
            &mut store,
            &mut tape,
        )?);

        let mut step_losses = Vec::with_capacity(TRAIN_STEPS);
        let mut step_seconds = Vec::with_capacity(TRAIN_STEPS);
        let total_started = Instant::now();
        for step in 1..=TRAIN_STEPS {
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
            if step == 1 && elapsed > SAFETY_FIRST_STEP_MAX_SECONDS {
                println!(
                    "safety_stop first_step_seconds={elapsed:.6} max_allowed_seconds={SAFETY_FIRST_STEP_MAX_SECONDS:.6}"
                );
                return Err(format!(
                    "first OPD step took {elapsed:.6}s, exceeding the {SAFETY_FIRST_STEP_MAX_SECONDS:.6}s safety ceiling"
                )
                .into());
            }
            step_losses.push(outcome.loss as f64);
            step_seconds.push(elapsed);
            if step == 1 || step % 10 == 0 || EVAL_STEPS.contains(&step) {
                println!(
                    "train_step step={step} prompt_index={prompt_index} prompt={prompt:?} loss={:.12e} rollout_len={} step_seconds={elapsed:.6}",
                    outcome.loss, outcome.rollout_len
                );
            }
            if EVAL_STEPS.contains(&step) {
                eval_summaries.push(evaluate_snapshot(
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
        let total_seconds = total_started.elapsed().as_secs_f64();

        print_training_summary(&step_losses, &step_seconds, total_seconds, &eval_summaries);
        Ok(())
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

    fn print_config(model_dir: &Path) {
        println!(
            "config backend=cuda model_dir={} train_steps={TRAIN_STEPS} rollout_len={ROLLOUT_LEN} decode_len={DECODE_LEN} lr={LEARNING_RATE} grad_clip={GRAD_CLIP} perturb_scale={PERTURB_SCALE} perturb_seed=0x{PERTURB_SEED:016x} safety_first_step_max_seconds={SAFETY_FIRST_STEP_MAX_SECONDS}",
            model_dir.display()
        );
        for (idx, prompt) in TRAIN_PROMPTS.iter().enumerate() {
            println!("prompt split=train index={idx} ids={prompt:?}");
        }
        for (idx, prompt) in HELDOUT_PROMPTS.iter().enumerate() {
            println!("prompt split=heldout index={idx} ids={prompt:?}");
        }
    }

    fn evaluate_snapshot(
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
            "eval_summary step={step} train_overlap_pct={:.6} heldout_overlap_pct={:.6} train_kl={:.12e} heldout_kl={:.12e} eval_seconds={:.6}",
            summary.train_overlap_pct,
            summary.heldout_overlap_pct,
            summary.train_kl,
            summary.heldout_kl,
            started.elapsed().as_secs_f64()
        );
        Ok(summary)
    }

    fn evaluate_split(
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
                "eval_detail step={step} split={split} prompt_index={index} prompt={prompt:?} teacher_suffix={teacher_suffix:?} student_suffix={student_suffix:?} overlap_pct={overlap_pct:.6} kl={kl:.12e}"
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

    fn print_training_summary(
        losses: &[f64],
        step_seconds: &[f64],
        total_seconds: f64,
        eval_summaries: &[EvalSummary],
    ) {
        let mean_step_seconds = mean(step_seconds.iter().copied());
        let mut sorted_step_seconds = step_seconds.to_vec();
        sorted_step_seconds.sort_by(f64::total_cmp);
        let median_step_seconds = sorted_step_seconds[sorted_step_seconds.len() / 2];
        let mean_loss = mean(losses.iter().copied());
        let first_loss = losses.first().copied().unwrap_or(0.0);
        let step_200_loss = losses.get(199).copied().unwrap_or(0.0);
        let last_loss = losses.last().copied().unwrap_or(0.0);
        let sampled_loss_reduction_200_pct = pct_reduction(first_loss, step_200_loss);
        let sampled_loss_reduction_final_pct = pct_reduction(first_loss, last_loss);
        let eval_0 = eval_summaries.iter().find(|summary| summary.step == 0);
        let eval_200 = eval_summaries.iter().find(|summary| summary.step == 200);
        let eval_500 = eval_summaries.iter().find(|summary| summary.step == 500);
        let train_kl_reduction_200_pct = match (eval_0, eval_200) {
            (Some(start), Some(end)) => pct_reduction(start.train_kl, end.train_kl),
            _ => 0.0,
        };
        let train_kl_reduction_final_pct = match (eval_0, eval_500) {
            (Some(start), Some(end)) => pct_reduction(start.train_kl, end.train_kl),
            _ => 0.0,
        };
        println!(
            "training_summary total_steps={} total_wall_seconds={total_seconds:.6} mean_step_seconds={mean_step_seconds:.6} median_step_seconds={median_step_seconds:.6} mean_sampled_loss={mean_loss:.12e} first_sampled_loss={first_loss:.12e} step200_sampled_loss={step_200_loss:.12e} final_sampled_loss={last_loss:.12e} sampled_loss_reduction_200_pct={sampled_loss_reduction_200_pct:.6} sampled_loss_reduction_final_pct={sampled_loss_reduction_final_pct:.6} train_kl_reduction_200_pct={train_kl_reduction_200_pct:.6} train_kl_reduction_final_pct={train_kl_reduction_final_pct:.6}",
            losses.len()
        );
        for summary in eval_summaries {
            println!(
                "summary_eval_row step={} train_overlap_pct={:.6} heldout_overlap_pct={:.6} train_kl={:.12e} heldout_kl={:.12e}",
                summary.step,
                summary.train_overlap_pct,
                summary.heldout_overlap_pct,
                summary.train_kl,
                summary.heldout_kl
            );
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
        "opd_step_cuda_realckpt_train requires CUDA. Run with: \
         cargo run -p train --example opd_step_cuda_realckpt_train --release --features cuda"
    );
    std::process::exit(1);
}
