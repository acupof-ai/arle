#![cfg_attr(
    not(all(feature = "cuda", not(feature = "no-cuda"))),
    allow(dead_code, unused_imports)
)]

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
pub mod app {
    use std::{
        collections::HashSet,
        env,
        path::{Path, PathBuf},
        sync::Arc,
        time::Instant,
    };

    use autograd::{Tape, TensorId, TensorStore, backend_cuda::CudaBackend, optim::AdamW};
    use train::{
        LoraConfig, LoraTargetSet,
        opd::{OpdStepConfig, opd_step},
        prompts::load_jsonl_prompt_sets,
        qwen35::{Qwen35KvCache, Qwen35Model, forward_rollout_cached},
        qwen35_loader::{load_qwen35_from_hf_dir, load_qwen35_trainable_from_hf_dir},
        trainer::extend_keep_with_params_and_grads,
    };

    const DEFAULT_MODEL_DIR: &str = "/home/ckl/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B";
    const DEFAULT_TRAIN_STEPS: usize = 500;
    const DEFAULT_ROLLOUT_LEN: usize = 8;
    const DEFAULT_PROMPT_MAX_TOKENS: usize = 16;
    const DEFAULT_JSONL_HELDOUT_PROMPTS: usize = 4;
    const DECODE_LEN: usize = 16;
    const DEFAULT_LEARNING_RATE: f32 = 5.0e-5;
    const DEFAULT_LORA_LEARNING_RATE: f32 = 1.0e-5;
    const DEFAULT_LORA_RANK: usize = 16;
    const DEFAULT_LORA_ALPHA: f32 = 32.0;
    const GRAD_CLIP: f32 = 1.0;
    const PERTURB_SCALE: f32 = 1.0e-3;
    const PERTURB_SEED: u64 = 0x0f0d_cafe_2026_0521;
    const SAFETY_FIRST_STEP_MAX_SECONDS: f64 = 0.5;
    const LORA_SAFETY_FIRST_STEP_MAX_SECONDS: f64 = 0.3;
    const EVAL_STEPS: &[usize] = &[0, 100, 250, 500, 1000, 2000];

    const TRAIN_PROMPTS_8: &[&[u32]] = &[
        &[1, 872, 198, 3456],
        &[1, 198, 1512, 429],
        &[1, 770, 3186, 25, 220],
        &[1, 644, 374, 279, 1887],
        &[1, 3838, 374, 264, 2077, 13],
        &[1, 785, 594, 287, 374, 1690],
        &[1, 3347, 11, 358, 1052, 429],
        &[1, 2610, 527, 1139, 304, 279, 1670],
    ];

    const TRAIN_PROMPTS_32: &[&[u32]] = &[
        &[1, 872, 198, 3456],
        &[1, 198, 1512, 429],
        &[1, 770, 3186, 25, 220],
        &[1, 644, 374, 279, 1887],
        &[1, 3838, 374, 264, 2077, 13],
        &[1, 785, 594, 287, 374, 1690],
        &[1, 3347, 11, 358, 1052, 429],
        &[1, 2610, 527, 1139, 304, 279, 1670],
        &[1, 888, 536, 4697, 972],
        &[1, 374, 11, 279, 1372, 315],
        &[1, 2874, 369, 279, 31559],
        &[1, 7521, 481, 362, 5714],
        &[1, 43059, 21938, 315, 7148],
        &[1, 358, 646, 944, 1490, 432],
        &[1, 477, 11, 323, 279, 62],
        &[1, 576, 1102, 315, 264, 729],
        &[1, 291, 504, 279, 1467, 11],
        &[1, 702, 1012, 1483, 311, 7512],
        &[1, 264, 11245, 2168, 429, 702],
        &[1, 3555, 374, 264, 5714, 30],
        &[1, 19257, 311, 279, 1251, 315],
        &[1, 1156, 3019, 304, 279, 1882],
        &[1, 2701, 1467, 25, 4710, 785],
        &[1, 315, 279, 3364, 13, 576],
        &[1, 279, 897, 5927, 553, 279],
        &[1, 2055, 11, 369, 279, 1140],
        &[1, 28469, 9363, 525, 279],
        &[1, 1012, 13570, 14975, 304, 279],
        &[1, 1887, 2242, 1294, 2827, 8],
        &[1, 62, 716, 477, 11, 323],
        &[1, 1512, 429, 374, 11, 279],
        &[1, 74595, 11, 714, 279, 1467],
    ];

    const HELDOUT_PROMPTS: &[&[u32]] = &[
        &[1, 4438, 374, 279, 2768],
        &[1, 1516, 374, 264, 1296, 4339],
        &[1, 785, 1401, 315, 279, 1967],
        &[1, 3198, 279, 1296, 25, 220],
    ];

    pub type AnyResult<T> = Result<T, Box<dyn std::error::Error>>;

    #[derive(Debug, Clone, Copy)]
    enum StudentMode {
        FullFineTune,
        Lora {
            rank: usize,
            alpha: f32,
            target_set: LoraTargetSet,
            default_learning_rate: f32,
            safety_first_step_max_seconds: f64,
        },
    }

    impl StudentMode {
        fn full_finetune() -> Self {
            Self::FullFineTune
        }

        fn lora_rank16() -> Self {
            Self::Lora {
                rank: DEFAULT_LORA_RANK,
                alpha: DEFAULT_LORA_ALPHA,
                target_set: LoraTargetSet::AttentionQv,
                default_learning_rate: DEFAULT_LORA_LEARNING_RATE,
                safety_first_step_max_seconds: LORA_SAFETY_FIRST_STEP_MAX_SECONDS,
            }
        }

        fn label(self) -> &'static str {
            match self {
                Self::FullFineTune => "full-finetune",
                Self::Lora { .. } => "lora",
            }
        }

        fn default_learning_rate(self) -> f32 {
            match self {
                Self::FullFineTune => DEFAULT_LEARNING_RATE,
                Self::Lora {
                    default_learning_rate,
                    ..
                } => default_learning_rate,
            }
        }

        fn safety_first_step_max_seconds(self) -> f64 {
            match self {
                Self::FullFineTune => SAFETY_FIRST_STEP_MAX_SECONDS,
                Self::Lora {
                    safety_first_step_max_seconds,
                    ..
                } => safety_first_step_max_seconds,
            }
        }

        fn lora_config(self) -> Option<LoraConfig> {
            match self {
                Self::FullFineTune => None,
                Self::Lora { rank, alpha, .. } => Some(LoraConfig { rank, alpha }),
            }
        }

        fn lora_target_set(self) -> Option<LoraTargetSet> {
            match self {
                Self::FullFineTune => None,
                Self::Lora { target_set, .. } => Some(target_set),
            }
        }

        fn usage_example(self) -> &'static str {
            match self {
                Self::FullFineTune => "opd_step_cuda_realckpt_train",
                Self::Lora { .. } => "opd_step_cuda_realckpt_lora_bench",
            }
        }
    }

    #[derive(Debug, Clone)]
    struct EvalSummary {
        step: usize,
        train_overlap_pct: f64,
        heldout_overlap_pct: f64,
        train_kl: f64,
        heldout_kl: f64,
        train_teacher_nll: f64,
        heldout_teacher_nll: f64,
        train_top3_overlap_pct: f64,
        heldout_top3_overlap_pct: f64,
    }

    #[derive(Debug, Clone)]
    struct TrainingArgs {
        learning_rate: f32,
        train_steps: usize,
        rollout_len: usize,
        eval_steps: Vec<usize>,
        prompt_source: PromptSourceArg,
        prompt_max_tokens: usize,
    }

    #[derive(Debug, Clone, Copy)]
    enum PromptSetArg {
        Eight,
        ThirtyTwo,
    }

    impl PromptSetArg {
        fn parse(raw: &str) -> AnyResult<Self> {
            match raw {
                "8" => Ok(Self::Eight),
                "32" => Ok(Self::ThirtyTwo),
                _ => Err(format!("--prompt-set must be `8` or `32`, got `{raw}`").into()),
            }
        }

        fn label(self) -> &'static str {
            match self {
                Self::Eight => "8",
                Self::ThirtyTwo => "32",
            }
        }

        fn prompts(self) -> &'static [&'static [u32]] {
            match self {
                Self::Eight => TRAIN_PROMPTS_8,
                Self::ThirtyTwo => TRAIN_PROMPTS_32,
            }
        }
    }

    #[derive(Debug, Clone)]
    enum PromptSourceArg {
        BuiltIn(PromptSetArg),
        Jsonl { path: PathBuf, label: &'static str },
    }

    impl PromptSourceArg {
        fn label(&self) -> String {
            match self {
                Self::BuiltIn(prompt_set) => format!("built-in-{}", prompt_set.label()),
                Self::Jsonl { path, label } => format!("{label}:{}", path.display()),
            }
        }
    }

    #[derive(Debug, Clone)]
    struct PromptData {
        train: Vec<Vec<u32>>,
        heldout: Vec<Vec<u32>>,
        source_label: String,
        prompt_file: Option<PathBuf>,
        tokenizer_path: Option<PathBuf>,
        jsonl_rows: Option<usize>,
        default_max_tokens: Option<usize>,
        truncated_rows: Option<usize>,
    }

    #[derive(Debug)]
    struct DecodeEval {
        overlap_pct: f64,
        kl: f64,
        teacher_nll: f64,
        top3_overlap_pct: f64,
    }

    #[derive(Debug)]
    struct TeacherForcedMetrics {
        kl: f64,
        teacher_nll: f64,
        top3_overlap_pct: f64,
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
        run(StudentMode::full_finetune())
    }

    pub fn main_lora_rank16() -> Result<(), Box<dyn std::error::Error>> {
        run(StudentMode::lora_rank16())
    }

    fn run(student_mode: StudentMode) -> AnyResult<()> {
        let model_dir = resolve_model_dir()?;
        let Some(args) = resolve_training_args(student_mode)? else {
            return Ok(());
        };
        let prompts = resolve_prompts(&model_dir, &args)?;
        let eval_steps = eval_steps_for(args.train_steps, &args.eval_steps);
        print_config(&model_dir, student_mode, &args, &prompts, &eval_steps);

        let cuda_backend = Arc::new(CudaBackend::new(0)?);
        let mut store = TensorStore::with_backend(cuda_backend.clone());
        let mut tape = Tape::new();

        let teacher_load_started = Instant::now();
        let teacher = load_qwen35_from_hf_dir(&model_dir, &mut store)?;
        let teacher_load_seconds = teacher_load_started.elapsed().as_secs_f64();
        let student_load_started = Instant::now();
        let student = match student_mode.lora_config() {
            Some(lora) => Qwen35Model::new_lora_from_base(
                &teacher,
                lora,
                student_mode
                    .lora_target_set()
                    .unwrap_or(LoraTargetSet::AllLinear),
                &mut store,
            )?,
            None => load_qwen35_trainable_from_hf_dir(&model_dir, &mut store)?,
        };
        let student_load_seconds = student_load_started.elapsed().as_secs_f64();

        let teacher_params = teacher.all_parameter_ids();
        let student_model_params = student.all_parameter_ids();
        let student_trainable_params = trainable_params(&student, &store);
        let teacher_param_elements = param_element_count(&teacher_params, &store);
        let student_model_elements = param_element_count(&student_model_params, &store);
        let student_trainable_elements = param_element_count(&student_trainable_params, &store);
        let student_base_shared_with_teacher = student_mode.lora_config().is_some();

        perturb_params(
            &student_trainable_params,
            &mut store,
            PERTURB_SEED,
            PERTURB_SCALE,
        );
        let mut optimizer =
            AdamW::new_with_device(args.learning_rate, (0.9, 0.999), 1.0e-8, 0.0, cuda_backend);
        let step_config = OpdStepConfig {
            rollout_len: args.rollout_len,
            grad_clip: GRAD_CLIP,
        };

        println!(
            "model_summary student_mode={} lora_rank={} lora_alpha={:.6} lora_target_set={} student_base_shared_with_teacher={} hidden={} intermediate={} layers={} vocab={} num_heads={} num_kv_heads={} head_dim={} tie_word_embeddings={} rope_theta={} teacher_param_elements={} student_model_elements={} student_trainable_elements={} teacher_load_seconds={teacher_load_seconds:.6} student_load_seconds={student_load_seconds:.6}",
            student_mode.label(),
            student_mode.lora_config().map(|cfg| cfg.rank).unwrap_or(0),
            student_mode
                .lora_config()
                .map(|cfg| cfg.alpha)
                .unwrap_or(0.0),
            student_mode
                .lora_target_set()
                .map(LoraTargetSet::label)
                .unwrap_or("none"),
            student_base_shared_with_teacher,
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
            &prompts.train,
            &prompts.heldout,
            &teacher,
            &student,
            &teacher_params,
            &student_model_params,
            &mut store,
            &mut tape,
        )?);

        let mut step_losses = Vec::with_capacity(args.train_steps);
        let mut step_seconds = Vec::with_capacity(args.train_steps);
        let total_started = Instant::now();
        for step in 1..=args.train_steps {
            let prompt_index = (step - 1) % prompts.train.len();
            let prompt = prompts.train[prompt_index].as_slice();
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
            let safety_first_step_max_seconds = student_mode.safety_first_step_max_seconds();
            if step == 1 && elapsed > safety_first_step_max_seconds {
                println!(
                    "safety_stop first_step_seconds={elapsed:.6} max_allowed_seconds={safety_first_step_max_seconds:.6}"
                );
                return Err(format!(
                    "first OPD step took {elapsed:.6}s, exceeding the {safety_first_step_max_seconds:.6}s safety ceiling"
                )
                .into());
            }
            step_losses.push(outcome.loss as f64);
            step_seconds.push(elapsed);
            if step <= 5 || step % 10 == 0 || eval_steps.contains(&step) {
                println!(
                    "train_step step={step} prompt_index={prompt_index} prompt={prompt:?} loss={:.12e} rollout_len={} step_seconds={elapsed:.6}",
                    outcome.loss, outcome.rollout_len
                );
            }
            if eval_steps.contains(&step) {
                eval_summaries.push(evaluate_snapshot(
                    step,
                    &prompts.train,
                    &prompts.heldout,
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

    fn resolve_training_args(student_mode: StudentMode) -> AnyResult<Option<TrainingArgs>> {
        let mut learning_rate = student_mode.default_learning_rate();
        let mut train_steps = DEFAULT_TRAIN_STEPS;
        let mut rollout_len = DEFAULT_ROLLOUT_LEN;
        let mut eval_steps = EVAL_STEPS.to_vec();
        let mut prompt_source = PromptSourceArg::BuiltIn(PromptSetArg::Eight);
        let mut prompt_source_explicit = false;
        let mut prompt_max_tokens = DEFAULT_PROMPT_MAX_TOKENS;
        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--lr" => {
                    let Some(raw) = args.next() else {
                        return Err("--lr requires a positive finite f32 value".into());
                    };
                    learning_rate = parse_positive_f32("--lr", &raw)?;
                }
                "--steps" => {
                    let Some(raw) = args.next() else {
                        return Err("--steps requires a positive usize value".into());
                    };
                    train_steps = parse_positive_usize("--steps", &raw)?;
                }
                "--rollout-len" => {
                    let Some(raw) = args.next() else {
                        return Err("--rollout-len requires a positive usize value".into());
                    };
                    rollout_len = parse_positive_usize("--rollout-len", &raw)?;
                }
                "--eval-steps" => {
                    let Some(raw) = args.next() else {
                        return Err("--eval-steps requires a comma-separated step list".into());
                    };
                    eval_steps = parse_eval_steps(&raw)?;
                }
                "--prompt-set" => {
                    let Some(raw) = args.next() else {
                        return Err("--prompt-set requires `8` or `32`".into());
                    };
                    if prompt_source_explicit {
                        return Err("--prompt-set cannot be combined with --prompts-file or --example-prompts-file".into());
                    }
                    prompt_source = PromptSourceArg::BuiltIn(PromptSetArg::parse(&raw)?);
                    prompt_source_explicit = true;
                }
                "--prompts-file" => {
                    let Some(raw) = args.next() else {
                        return Err("--prompts-file requires a JSONL path".into());
                    };
                    if prompt_source_explicit {
                        return Err("--prompts-file cannot be combined with --prompt-set or --example-prompts-file".into());
                    }
                    prompt_source = PromptSourceArg::Jsonl {
                        path: PathBuf::from(raw),
                        label: "jsonl",
                    };
                    prompt_source_explicit = true;
                }
                "--example-prompts-file" => {
                    let Some(raw) = args.next() else {
                        return Err("--example-prompts-file requires a JSONL path".into());
                    };
                    if prompt_source_explicit {
                        return Err("--example-prompts-file cannot be combined with --prompt-set or --prompts-file".into());
                    }
                    prompt_source = PromptSourceArg::Jsonl {
                        path: PathBuf::from(raw),
                        label: "example-jsonl",
                    };
                    prompt_source_explicit = true;
                }
                "--prompt-max-tokens" => {
                    let Some(raw) = args.next() else {
                        return Err("--prompt-max-tokens requires a positive usize value".into());
                    };
                    prompt_max_tokens = parse_positive_usize("--prompt-max-tokens", &raw)?;
                }
                "-h" | "--help" => {
                    println!(
                        "usage: cargo run -p train --example {} --release --features cuda -- [--lr VALUE] [--steps VALUE] [--rollout-len VALUE] [--eval-steps CSV] [--prompt-set 8|32] [--prompts-file PATH | --example-prompts-file PATH] [--prompt-max-tokens VALUE]\n\nJSONL prompt rows use: {{\"text\":\"...\",\"max_tokens\":16}}. The final 4 rows are held out; earlier rows train.",
                        student_mode.usage_example()
                    );
                    return Ok(None);
                }
                unknown => {
                    return Err(format!(
                        "unknown argument `{unknown}`. Supported arguments: --lr VALUE, --steps VALUE, --rollout-len VALUE, --eval-steps CSV, --prompt-set 8|32, --prompts-file PATH, --example-prompts-file PATH, --prompt-max-tokens VALUE"
                    )
                    .into());
                }
            }
        }
        Ok(Some(TrainingArgs {
            learning_rate,
            train_steps,
            rollout_len,
            eval_steps,
            prompt_source,
            prompt_max_tokens,
        }))
    }

    fn parse_positive_f32(name: &str, raw: &str) -> AnyResult<f32> {
        let value = raw
            .parse::<f32>()
            .map_err(|err| format!("invalid {name} value `{raw}`: {err}"))?;
        if value.is_finite() && value > 0.0 {
            Ok(value)
        } else {
            Err(format!("{name} must be positive and finite, got `{raw}`").into())
        }
    }

    fn parse_positive_usize(name: &str, raw: &str) -> AnyResult<usize> {
        let value = raw
            .parse::<usize>()
            .map_err(|err| format!("invalid {name} value `{raw}`: {err}"))?;
        if value > 0 {
            Ok(value)
        } else {
            Err(format!("{name} must be positive, got `{raw}`").into())
        }
    }

    fn parse_eval_steps(raw: &str) -> AnyResult<Vec<usize>> {
        let mut steps = Vec::new();
        for part in raw.split(',') {
            let trimmed = part.trim();
            if trimmed.is_empty() {
                return Err(format!("--eval-steps contains an empty item in `{raw}`").into());
            }
            steps.push(
                trimmed
                    .parse::<usize>()
                    .map_err(|err| format!("invalid --eval-steps item `{trimmed}`: {err}"))?,
            );
        }
        if steps.is_empty() {
            return Err("--eval-steps requires at least one step".into());
        }
        steps.sort_unstable();
        steps.dedup();
        Ok(steps)
    }

    fn eval_steps_for(train_steps: usize, configured_steps: &[usize]) -> Vec<usize> {
        let mut steps = configured_steps
            .iter()
            .copied()
            .filter(|step| *step <= train_steps)
            .collect::<Vec<_>>();
        if !steps.contains(&train_steps) {
            steps.push(train_steps);
        }
        steps
    }

    fn resolve_prompts(model_dir: &Path, args: &TrainingArgs) -> AnyResult<PromptData> {
        match &args.prompt_source {
            PromptSourceArg::BuiltIn(prompt_set) => Ok(PromptData {
                train: prompt_set
                    .prompts()
                    .iter()
                    .map(|prompt| prompt.to_vec())
                    .collect(),
                heldout: HELDOUT_PROMPTS
                    .iter()
                    .map(|prompt| prompt.to_vec())
                    .collect(),
                source_label: args.prompt_source.label(),
                prompt_file: None,
                tokenizer_path: None,
                jsonl_rows: None,
                default_max_tokens: None,
                truncated_rows: None,
            }),
            PromptSourceArg::Jsonl { path, .. } => {
                let loaded = load_jsonl_prompt_sets(
                    model_dir,
                    path,
                    args.prompt_max_tokens,
                    DEFAULT_JSONL_HELDOUT_PROMPTS,
                )?;
                Ok(PromptData {
                    train: loaded.train,
                    heldout: loaded.heldout,
                    source_label: args.prompt_source.label(),
                    prompt_file: Some(loaded.prompt_file),
                    tokenizer_path: Some(loaded.tokenizer_path),
                    jsonl_rows: Some(loaded.jsonl_rows),
                    default_max_tokens: Some(loaded.default_max_tokens),
                    truncated_rows: Some(loaded.truncated_rows),
                })
            }
        }
    }

    fn print_config(
        model_dir: &Path,
        student_mode: StudentMode,
        args: &TrainingArgs,
        prompts: &PromptData,
        eval_steps: &[usize],
    ) {
        println!(
            "config backend=cuda model_dir={} student_mode={} lora_rank={} lora_alpha={:.6} lora_target_set={} train_steps={} rollout_len={} default_rollout_len={DEFAULT_ROLLOUT_LEN} decode_len={DECODE_LEN} lr={:.9e} default_lr={:.9e} grad_clip={GRAD_CLIP} perturb_scale={PERTURB_SCALE} perturb_seed=0x{PERTURB_SEED:016x} safety_first_step_max_seconds={:.6} prompt_source={} train_prompt_count={} heldout_prompt_count={} eval_steps={eval_steps:?}",
            model_dir.display(),
            student_mode.label(),
            student_mode.lora_config().map(|cfg| cfg.rank).unwrap_or(0),
            student_mode
                .lora_config()
                .map(|cfg| cfg.alpha)
                .unwrap_or(0.0),
            student_mode
                .lora_target_set()
                .map(LoraTargetSet::label)
                .unwrap_or("none"),
            args.train_steps,
            args.rollout_len,
            args.learning_rate,
            student_mode.default_learning_rate(),
            student_mode.safety_first_step_max_seconds(),
            prompts.source_label,
            prompts.train.len(),
            prompts.heldout.len()
        );
        if let Some(path) = &prompts.prompt_file {
            println!(
                "prompt_file path={} jsonl_rows={} tokenizer_path={} default_max_tokens={} truncated_rows={}",
                path.display(),
                prompts.jsonl_rows.unwrap_or(0),
                prompts
                    .tokenizer_path
                    .as_ref()
                    .map(|path| path.display().to_string())
                    .unwrap_or_else(|| "<none>".to_string()),
                prompts.default_max_tokens.unwrap_or(0),
                prompts.truncated_rows.unwrap_or(0)
            );
        }
        for (idx, prompt) in prompts.train.iter().enumerate() {
            println!("prompt split=train index={idx} ids={prompt:?}");
        }
        for (idx, prompt) in prompts.heldout.iter().enumerate() {
            println!("prompt split=heldout index={idx} ids={prompt:?}");
        }
    }

    fn evaluate_snapshot(
        step: usize,
        train_prompts: &[Vec<u32>],
        heldout_prompts: &[Vec<u32>],
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
            train_prompts,
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
            heldout_prompts,
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
            train_teacher_nll: mean(train.iter().map(|eval| eval.teacher_nll)),
            heldout_teacher_nll: mean(heldout.iter().map(|eval| eval.teacher_nll)),
            train_top3_overlap_pct: mean(train.iter().map(|eval| eval.top3_overlap_pct)),
            heldout_top3_overlap_pct: mean(heldout.iter().map(|eval| eval.top3_overlap_pct)),
        };
        println!(
            "eval_summary step={step} train_overlap_pct={:.6} heldout_overlap_pct={:.6} train_kl={:.12e} heldout_kl={:.12e} train_teacher_nll={:.12e} heldout_teacher_nll={:.12e} train_top3_overlap_pct={:.6} heldout_top3_overlap_pct={:.6} eval_seconds={:.6}",
            summary.train_overlap_pct,
            summary.heldout_overlap_pct,
            summary.train_kl,
            summary.heldout_kl,
            summary.train_teacher_nll,
            summary.heldout_teacher_nll,
            summary.train_top3_overlap_pct,
            summary.heldout_top3_overlap_pct,
            started.elapsed().as_secs_f64()
        );
        Ok(summary)
    }

    fn evaluate_split(
        split: &'static str,
        step: usize,
        prompts: &[Vec<u32>],
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
            let metrics = teacher_forced_metrics(
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
                "eval_detail step={step} split={split} prompt_index={index} prompt={prompt:?} teacher_suffix={teacher_suffix:?} student_suffix={student_suffix:?} overlap_pct={overlap_pct:.6} kl={:.12e} teacher_nll={:.12e} top3_overlap_pct={:.6}",
                metrics.kl, metrics.teacher_nll, metrics.top3_overlap_pct
            );
            rows.push(DecodeEval {
                overlap_pct,
                kl: metrics.kl,
                teacher_nll: metrics.teacher_nll,
                top3_overlap_pct: metrics.top3_overlap_pct,
            });
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

    fn teacher_forced_metrics(
        teacher: &Qwen35Model,
        student: &Qwen35Model,
        prompt: &[u32],
        teacher_suffix: &[u32],
        teacher_params: &[TensorId],
        student_params: &[TensorId],
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> AnyResult<TeacherForcedMetrics> {
        if prompt.is_empty() || teacher_suffix.len() < DECODE_LEN {
            return Err(
                "teacher-forced eval requires a non-empty prompt and full teacher suffix".into(),
            );
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
        let metrics = mean_teacher_forced_metrics_rows(
            &teacher_host,
            &student_host,
            first_row,
            &teacher_suffix[..DECODE_LEN],
            vocab,
        )?;
        retain_model_state(store, tape, teacher_params, student_params);
        Ok(metrics)
    }

    fn mean_teacher_forced_metrics_rows(
        teacher_logits: &[f32],
        student_logits: &[f32],
        first_row: usize,
        targets: &[u32],
        vocab: usize,
    ) -> AnyResult<TeacherForcedMetrics> {
        let rows = targets.len();
        let required = (first_row + rows) * vocab;
        if teacher_logits.len() < required || student_logits.len() < required {
            return Err(format!(
                "teacher-forced logits are too short: required {required}, teacher={}, student={}",
                teacher_logits.len(),
                student_logits.len()
            )
            .into());
        }
        let mut total_kl = 0.0f64;
        let mut total_teacher_nll = 0.0f64;
        let mut top3_hits = 0usize;
        for (target_idx, row_idx) in (first_row..first_row + rows).enumerate() {
            let target = targets[target_idx] as usize;
            if target >= vocab {
                return Err(format!("teacher target token {target} exceeds vocab {vocab}").into());
            }
            let offset = row_idx * vocab;
            let teacher_row = &teacher_logits[offset..offset + vocab];
            let student_row = &student_logits[offset..offset + vocab];
            let student_log_z = log_sum_exp_row(student_row);
            total_kl += forward_kl_row_with_student_log_z(teacher_row, student_row, student_log_z);
            total_teacher_nll += student_log_z - student_row[target] as f64;
            if top_k_contains(student_row, target, 3) {
                top3_hits += 1;
            }
        }
        Ok(TeacherForcedMetrics {
            kl: total_kl / rows as f64,
            teacher_nll: total_teacher_nll / rows as f64,
            top3_overlap_pct: top3_hits as f64 / rows as f64 * 100.0,
        })
    }

    fn forward_kl_row_with_student_log_z(
        teacher: &[f32],
        student: &[f32],
        student_log_z: f64,
    ) -> f64 {
        let teacher_max = teacher.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let teacher_sum_exp = teacher
            .iter()
            .map(|&value| ((value as f64) - teacher_max).exp())
            .sum::<f64>();
        let teacher_log_z = teacher_max + teacher_sum_exp.ln();
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

    fn log_sum_exp_row(row: &[f32]) -> f64 {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let sum_exp = row
            .iter()
            .map(|&value| ((value as f64) - max).exp())
            .sum::<f64>();
        max + sum_exp.ln()
    }

    fn top_k_contains(row: &[f32], target: usize, k: usize) -> bool {
        let mut top = vec![(f32::NEG_INFINITY, usize::MAX); k];
        for (idx, &value) in row.iter().enumerate() {
            for slot in 0..k {
                let (slot_value, slot_idx) = top[slot];
                if value > slot_value || (value == slot_value && idx < slot_idx) {
                    for shift in (slot + 1..k).rev() {
                        top[shift] = top[shift - 1];
                    }
                    top[slot] = (value, idx);
                    break;
                }
            }
        }
        top.iter().any(|&(_, idx)| idx == target)
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
        let step_250_loss = losses.get(249).copied();
        let last_loss = losses.last().copied().unwrap_or(0.0);
        let sampled_loss_reduction_250_pct = step_250_loss
            .map(|loss| pct_reduction(first_loss, loss))
            .unwrap_or(f64::NAN);
        let sampled_loss_reduction_final_pct = pct_reduction(first_loss, last_loss);
        let eval_0 = eval_summaries.iter().find(|summary| summary.step == 0);
        let eval_250 = eval_summaries.iter().find(|summary| summary.step == 250);
        let eval_final = eval_summaries.last();
        let train_kl_reduction_250_pct = match (eval_0, eval_250) {
            (Some(start), Some(end)) => pct_reduction(start.train_kl, end.train_kl),
            _ => f64::NAN,
        };
        let train_kl_reduction_final_pct = match (eval_0, eval_final) {
            (Some(start), Some(end)) => pct_reduction(start.train_kl, end.train_kl),
            _ => f64::NAN,
        };
        println!(
            "training_summary total_steps={} total_wall_seconds={total_seconds:.6} mean_step_seconds={mean_step_seconds:.6} median_step_seconds={median_step_seconds:.6} mean_sampled_loss={mean_loss:.12e} first_sampled_loss={first_loss:.12e} step250_sampled_loss={step_250_loss:.12e} final_sampled_loss={last_loss:.12e} sampled_loss_reduction_250_pct={sampled_loss_reduction_250_pct:.6} sampled_loss_reduction_final_pct={sampled_loss_reduction_final_pct:.6} train_kl_reduction_250_pct={train_kl_reduction_250_pct:.6} train_kl_reduction_final_pct={train_kl_reduction_final_pct:.6}",
            losses.len(),
            step_250_loss = step_250_loss.unwrap_or(f64::NAN)
        );
        for summary in eval_summaries {
            println!(
                "summary_eval_row step={} train_overlap_pct={:.6} heldout_overlap_pct={:.6} train_kl={:.12e} heldout_kl={:.12e} train_teacher_nll={:.12e} heldout_teacher_nll={:.12e} train_top3_overlap_pct={:.6} heldout_top3_overlap_pct={:.6}",
                summary.step,
                summary.train_overlap_pct,
                summary.heldout_overlap_pct,
                summary.train_kl,
                summary.heldout_kl,
                summary.train_teacher_nll,
                summary.heldout_teacher_nll,
                summary.train_top3_overlap_pct,
                summary.heldout_top3_overlap_pct
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
