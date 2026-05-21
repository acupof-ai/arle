#![cfg_attr(
    not(all(feature = "cuda", not(feature = "no-cuda"))),
    allow(dead_code, unused_imports)
)]

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
mod app {
    use std::{
        collections::HashSet,
        path::{Path, PathBuf},
        sync::{Arc, Mutex},
        time::Instant,
    };

    use autograd::{Backend, Tape, TensorId, TensorStore, backend_cuda::CudaBackend, optim::AdamW};
    use infer::server_engine::{
        InferenceEngineOptions, LoadedInferenceEngine, ServerRuntimeConfig,
    };
    use train::{
        LoraConfig, LoraTargetSet,
        loss::kl_distill_loss,
        opd::{OpdStepConfig, OpdStepProfile, opd_step_with_teacher_forward_profiled},
        prompts::load_jsonl_prompt_sets,
        qwen35::Qwen35Model,
        qwen35_loader::load_qwen35_lora_from_hf_dir,
        teacher_infer::{InferTeacher, TeacherForward},
        trainer::extend_keep_with_params_and_grads,
    };

    const DEFAULT_QWEN35_08B_DIR: &str =
        "/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base";
    const DEFAULT_STEPS: usize = 1;
    const DEFAULT_ROLLOUT_LEN: usize = 8;
    const DEFAULT_LR: f32 = 1.0e-5;
    const DEFAULT_PROMPT_MAX_TOKENS: usize = 16;
    const DEFAULT_HELDOUT_PROMPTS: usize = 4;
    const LORA_RANK: usize = 16;
    const LORA_ALPHA: f32 = 32.0;
    const LORA_TARGET_SET: LoraTargetSet = LoraTargetSet::AttentionQv;
    const GRAD_CLIP: f32 = 1.0;

    #[derive(Debug)]
    struct Args {
        teacher_model: PathBuf,
        student_model: PathBuf,
        prompts_file: Option<PathBuf>,
        steps: usize,
        rollout_len: usize,
        lr: f32,
        eval_steps: Vec<usize>,
        prompt_max_tokens: usize,
        max_step_seconds: Option<f64>,
        enable_cuda_graph: bool,
    }

    #[derive(Debug)]
    struct PromptSets {
        train: Vec<Vec<u32>>,
        heldout: Vec<Vec<u32>>,
        source: String,
    }

    pub fn main() -> Result<(), Box<dyn std::error::Error>> {
        let args = parse_args()?;
        let prompts = load_prompts(&args)?;
        println!(
            "config backend=cuda teacher_model={} student_model={} student_mode=lora \
             lora_rank={LORA_RANK} lora_alpha={LORA_ALPHA:.6} lora_target_set={} \
             steps={} rollout_len={} lr={:.9e} grad_clip={GRAD_CLIP} \
             prompt_source={} train_prompt_count={} heldout_prompt_count={} \
             eval_steps={:?} cuda_graph={}",
            args.teacher_model.display(),
            args.student_model.display(),
            LORA_TARGET_SET.label(),
            args.steps,
            args.rollout_len,
            args.lr,
            prompts.source,
            prompts.train.len(),
            prompts.heldout.len(),
            args.eval_steps,
            args.enable_cuda_graph
        );
        for (idx, prompt) in prompts.train.iter().enumerate() {
            println!("prompt split=train index={idx} ids={prompt:?}");
        }
        for (idx, prompt) in prompts.heldout.iter().enumerate() {
            println!("prompt split=heldout index={idx} ids={prompt:?}");
        }

        let cuda_backend = Arc::new(CudaBackend::new(0)?);
        let teacher_backend: Arc<dyn Backend> = cuda_backend.clone();
        let mut store = TensorStore::with_backend(cuda_backend.clone());
        let mut tape = Tape::new();

        let student_load_started = Instant::now();
        let student = load_qwen35_lora_from_hf_dir(
            &args.student_model,
            LoraConfig {
                rank: LORA_RANK,
                alpha: LORA_ALPHA,
            },
            LORA_TARGET_SET,
            &mut store,
        )?;
        let student_load_seconds = student_load_started.elapsed().as_secs_f64();
        let student_model_params = student.all_parameter_ids();
        let student_trainable_params = trainable_params(&student, &store);

        let infer_load_started = Instant::now();
        let infer_engine = load_infer_engine(
            &args.teacher_model,
            args.prompt_max_tokens + args.rollout_len + 32,
            args.enable_cuda_graph,
        )?;
        let infer_load_seconds = infer_load_started.elapsed().as_secs_f64();
        let teacher = InferTeacher::new(
            Arc::new(Mutex::new(infer_engine)),
            teacher_backend,
            student.config().vocab_size,
        );
        let mut optimizer =
            AdamW::new_with_device(args.lr, (0.9, 0.999), 1.0e-8, 0.0, cuda_backend);
        println!(
            "model_summary student_hidden={} student_layers={} student_vocab={} \
             student_model_elements={} student_trainable_elements={} \
             student_load_seconds={student_load_seconds:.6} infer_load_seconds={infer_load_seconds:.6}",
            student.config().hidden_size,
            student.config().num_hidden_layers,
            student.config().vocab_size,
            param_element_count(&student_model_params, &store),
            param_element_count(&student_trainable_params, &store)
        );

        maybe_eval(
            0,
            &args,
            &prompts,
            &teacher,
            &student,
            &student_model_params,
            &mut store,
            &mut tape,
        )?;

        let mut step_losses = Vec::with_capacity(args.steps);
        let mut step_seconds = Vec::with_capacity(args.steps);
        let total_started = Instant::now();
        for step in 1..=args.steps {
            let prompt_index = (step - 1) % prompts.train.len();
            let prompt = prompts.train[prompt_index].as_slice();
            let mut profile = OpdStepProfile::default();
            let step_started = Instant::now();
            let outcome = opd_step_with_teacher_forward_profiled(
                &student,
                &teacher,
                prompt,
                OpdStepConfig {
                    rollout_len: args.rollout_len,
                    grad_clip: GRAD_CLIP,
                },
                &student_trainable_params,
                &mut optimizer,
                &mut store,
                &mut tape,
                Some(&mut profile),
            )?;
            let elapsed = step_started.elapsed().as_secs_f64();
            if let Some(max_step_seconds) = args.max_step_seconds {
                if step == 1 && elapsed > max_step_seconds {
                    return Err(format!(
                        "first InferTeacher OPD step took {elapsed:.6}s, above configured ceiling {max_step_seconds:.6}s"
                    )
                    .into());
                }
            }
            let teacher_profile = teacher.last_profile();
            step_losses.push(outcome.loss as f64);
            step_seconds.push(elapsed);
            println!(
                "train_step step={step} prompt_index={prompt_index} loss={:.12e} \
                 rollout_len={} step_seconds={elapsed:.6}",
                outcome.loss, outcome.rollout_len
            );
            println!(
                "phase_summary step={step} total={:.6} student_rollout={:.6} \
                 infer_forward_token_logits={:.6} infer_sync={:.6} d2d_bridge_import={:.6} \
                 teacher_forward_total={:.6} student_forward={:.6} kl_loss={:.6} \
                 optimizer_zero_grad={:.6} backward={:.6} grad_clip={:.6} \
                 optimizer_step={:.6} post_step_cleanup={:.6} teacher_seq_len={} teacher_vocab={}",
                profile.total_seconds,
                profile.student_rollout_seconds,
                teacher_profile.raw_forward_seconds,
                teacher_profile.sync_seconds,
                teacher_profile.d2d_bridge_import_seconds,
                profile.teacher_forward_seconds,
                profile.student_forward_seconds,
                profile.kl_loss_seconds,
                profile.optimizer_zero_grad_seconds,
                profile.backward_seconds,
                profile.grad_clip_seconds,
                profile.optimizer_step_seconds,
                profile.post_step_cleanup_seconds,
                teacher_profile.seq_len,
                teacher_profile.vocab_size
            );
            maybe_eval(
                step,
                &args,
                &prompts,
                &teacher,
                &student,
                &student_model_params,
                &mut store,
                &mut tape,
            )?;
        }
        println!(
            "training_summary total_steps={} total_wall_seconds={:.6} mean_step_seconds={:.6} \
             median_step_seconds={:.6} first_loss={:.12e} final_loss={:.12e} \
             sampled_loss_reduction_pct={:.6}",
            args.steps,
            total_started.elapsed().as_secs_f64(),
            mean(&step_seconds),
            median(&step_seconds),
            step_losses.first().copied().unwrap_or(f64::NAN),
            step_losses.last().copied().unwrap_or(f64::NAN),
            reduction_pct(
                step_losses.first().copied().unwrap_or(f64::NAN),
                step_losses.last().copied().unwrap_or(f64::NAN)
            )
        );

        Ok(())
    }

    fn parse_args() -> Result<Args, Box<dyn std::error::Error>> {
        let mut teacher_model = PathBuf::from(DEFAULT_QWEN35_08B_DIR);
        let mut student_model = PathBuf::from(DEFAULT_QWEN35_08B_DIR);
        let mut prompts_file = None;
        let mut steps = DEFAULT_STEPS;
        let mut rollout_len = DEFAULT_ROLLOUT_LEN;
        let mut lr = DEFAULT_LR;
        let mut eval_steps = Vec::new();
        let mut prompt_max_tokens = DEFAULT_PROMPT_MAX_TOKENS;
        let mut max_step_seconds = None;
        let mut enable_cuda_graph = true;

        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--teacher-model" => teacher_model = PathBuf::from(next_arg(&mut args, &arg)?),
                "--student-model" => student_model = PathBuf::from(next_arg(&mut args, &arg)?),
                "--prompts-file" => prompts_file = Some(PathBuf::from(next_arg(&mut args, &arg)?)),
                "--steps" => steps = parse_positive_usize(&arg, &next_arg(&mut args, &arg)?)?,
                "--rollout-len" => {
                    rollout_len = parse_positive_usize(&arg, &next_arg(&mut args, &arg)?)?
                }
                "--lr" => lr = next_arg(&mut args, &arg)?.parse::<f32>()?,
                "--eval-steps" => eval_steps = parse_step_csv(&next_arg(&mut args, &arg)?)?,
                "--prompt-max-tokens" => {
                    prompt_max_tokens = parse_positive_usize(&arg, &next_arg(&mut args, &arg)?)?
                }
                "--max-step-seconds" => {
                    max_step_seconds = Some(next_arg(&mut args, &arg)?.parse::<f64>()?)
                }
                "--no-cuda-graph" => enable_cuda_graph = false,
                "--help" | "-h" => {
                    println!(
                        "usage: cargo run -p train --example opd_step_cuda_infer_teacher_train \
                         --release --features cuda -- [--teacher-model DIR] [--student-model DIR] \
                         [--prompts-file JSONL] [--steps N] [--rollout-len N] [--lr LR] \
                         [--eval-steps CSV] [--prompt-max-tokens N] [--max-step-seconds SEC] \
                         [--no-cuda-graph]"
                    );
                    std::process::exit(0);
                }
                _ => return Err(format!("unknown argument `{arg}`").into()),
            }
        }
        if eval_steps.is_empty() {
            eval_steps = if steps == 1 {
                vec![0]
            } else {
                vec![0, steps / 4, steps / 2, steps]
            };
            eval_steps.sort_unstable();
            eval_steps.dedup();
        }
        Ok(Args {
            teacher_model,
            student_model,
            prompts_file,
            steps,
            rollout_len,
            lr,
            eval_steps,
            prompt_max_tokens,
            max_step_seconds,
            enable_cuda_graph,
        })
    }

    fn next_arg(
        args: &mut impl Iterator<Item = String>,
        flag: &str,
    ) -> Result<String, Box<dyn std::error::Error>> {
        args.next()
            .ok_or_else(|| format!("{flag} requires a value").into())
    }

    fn parse_positive_usize(flag: &str, raw: &str) -> Result<usize, Box<dyn std::error::Error>> {
        let value = raw.parse::<usize>()?;
        if value == 0 {
            return Err(format!("{flag} must be positive").into());
        }
        Ok(value)
    }

    fn parse_step_csv(raw: &str) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
        let mut out = Vec::new();
        for item in raw.split(',') {
            let item = item.trim();
            if item.is_empty() {
                continue;
            }
            out.push(item.parse::<usize>()?);
        }
        out.sort_unstable();
        out.dedup();
        Ok(out)
    }

    fn load_prompts(args: &Args) -> Result<PromptSets, Box<dyn std::error::Error>> {
        if let Some(path) = args.prompts_file.as_ref() {
            let loaded = load_jsonl_prompt_sets(
                &args.student_model,
                path,
                args.prompt_max_tokens,
                DEFAULT_HELDOUT_PROMPTS,
            )?;
            return Ok(PromptSets {
                train: loaded.train,
                heldout: loaded.heldout,
                source: format!(
                    "jsonl:{} rows={} tokenizer={} truncated_rows={}",
                    loaded.prompt_file.display(),
                    loaded.jsonl_rows,
                    loaded.tokenizer_path.display(),
                    loaded.truncated_rows
                ),
            });
        }
        Ok(PromptSets {
            train: vec![vec![9419]],
            heldout: Vec::new(),
            source: "single-token-hello".to_string(),
        })
    }

    fn load_infer_engine(
        model_dir: &Path,
        max_seq_len: usize,
        enable_cuda_graph: bool,
    ) -> anyhow::Result<LoadedInferenceEngine> {
        let max_seq_len = max_seq_len.max(128);
        let mut runtime = ServerRuntimeConfig {
            engine: InferenceEngineOptions { enable_cuda_graph },
            max_seq_len: Some(max_seq_len),
            ..ServerRuntimeConfig::default()
        };
        runtime.scheduler.max_slots = 1;
        runtime.scheduler.chunked_prefill_size = max_seq_len;
        runtime.scheduler.max_num_batched_tokens = max_seq_len;
        runtime.scheduler.max_prefill_tokens = max_seq_len;
        runtime.scheduler.long_prefill_token_threshold = max_seq_len;
        runtime.scheduler.prefill_max_requests = Some(1);
        runtime.scheduler.mem_fraction_static = 0.05;
        runtime.scheduler.kv_pool_fallback_bytes = 128 * 1024 * 1024;
        LoadedInferenceEngine::load_with_runtime_config(
            model_dir
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("model path is not valid UTF-8"))?,
            runtime,
        )
    }

    fn maybe_eval(
        step: usize,
        args: &Args,
        prompts: &PromptSets,
        teacher: &InferTeacher,
        student: &Qwen35Model,
        student_model_params: &[TensorId],
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !args.eval_steps.contains(&step) {
            return Ok(());
        }
        let started = Instant::now();
        let train_kl = mean_prompt_kl(
            &prompts.train,
            teacher,
            student,
            student_model_params,
            store,
            tape,
        )?;
        let heldout_kl = mean_prompt_kl(
            &prompts.heldout,
            teacher,
            student,
            student_model_params,
            store,
            tape,
        )?;
        println!(
            "eval_summary step={step} train_kl={train_kl:.12e} heldout_kl={heldout_kl:.12e} \
             eval_seconds={:.6}",
            started.elapsed().as_secs_f64()
        );
        Ok(())
    }

    fn mean_prompt_kl(
        prompts: &[Vec<u32>],
        teacher: &InferTeacher,
        student: &Qwen35Model,
        student_model_params: &[TensorId],
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        if prompts.is_empty() {
            return Ok(f64::NAN);
        }
        let mut total = 0.0f64;
        for prompt in prompts {
            tape.entries.clear();
            tape.set_enabled(false);
            let positions = (0..prompt.len() as u32).collect::<Vec<_>>();
            let teacher_logits = teacher.forward_logits_device(prompt, &positions, store, tape)?;
            let student_logits = student.forward(store, tape, prompt, &positions)?;
            let loss = kl_distill_loss(
                student_logits,
                teacher_logits.tensor_id,
                prompt.len(),
                store,
                tape,
            )?;
            total += store.to_host(loss)?[0] as f64;
            retain_student_state(store, tape, student_model_params);
        }
        Ok(total / prompts.len() as f64)
    }

    fn retain_student_state(store: &mut TensorStore, tape: &mut Tape, params: &[TensorId]) {
        tape.entries.clear();
        let mut keep = HashSet::with_capacity(params.len() * 2);
        extend_keep_with_params_and_grads(&mut keep, params.iter().copied(), store);
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
            .filter_map(|&id| store.get(id).map(|tensor| tensor.size))
            .sum()
    }

    fn mean(values: &[f64]) -> f64 {
        if values.is_empty() {
            return f64::NAN;
        }
        values.iter().sum::<f64>() / values.len() as f64
    }

    fn median(values: &[f64]) -> f64 {
        if values.is_empty() {
            return f64::NAN;
        }
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        sorted[sorted.len() / 2]
    }

    fn reduction_pct(first: f64, last: f64) -> f64 {
        if !first.is_finite() || first.abs() < f64::EPSILON {
            return f64::NAN;
        }
        (first - last) / first * 100.0
    }
}

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    app::main()
}

#[cfg(not(all(feature = "cuda", not(feature = "no-cuda"))))]
fn main() {
    eprintln!(
        "opd_step_cuda_infer_teacher_train requires CUDA. Run with: \
         cargo run -p train --example opd_step_cuda_infer_teacher_train --release --features cuda"
    );
}
