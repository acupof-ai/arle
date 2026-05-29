#![cfg_attr(
    not(all(feature = "cuda", not(feature = "no-cuda"))),
    allow(dead_code, unused_imports)
)]

const DEFAULT_EVAL_TRAIN_PROMPT_LIMIT: usize = 1;
const STATIC_PARAM_EVICT_MIN_ELEMENTS: usize = 1_000_000;

fn eval_prompt_limit_len(total: usize, limit: Option<usize>) -> usize {
    limit.map_or(total, |limit| limit.min(total))
}

#[cfg(test)]
mod tests {
    use super::eval_prompt_limit_len;

    #[test]
    fn eval_prompt_limit_defaults_to_bounded_prefix() {
        assert_eq!(eval_prompt_limit_len(52, Some(1)), 1);
        assert_eq!(eval_prompt_limit_len(2, Some(8)), 2);
        assert_eq!(eval_prompt_limit_len(52, Some(0)), 0);
    }

    #[test]
    fn eval_prompt_limit_none_keeps_full_split() {
        assert_eq!(eval_prompt_limit_len(52, None), 52);
    }
}

#[cfg(all(feature = "cuda", not(feature = "no-cuda")))]
mod app {
    use super::{DEFAULT_EVAL_TRAIN_PROMPT_LIMIT, STATIC_PARAM_EVICT_MIN_ELEMENTS};
    use std::{
        collections::HashSet,
        fs,
        path::{Path, PathBuf},
        sync::{Arc, Mutex},
        time::{Duration, Instant},
    };

    use autograd::{Backend, Tape, TensorId, TensorStore, backend_cuda::CudaBackend, optim::AdamW};
    use infer::server_engine::{
        InferenceEngineOptions, LoadedInferenceEngine, ServerRuntimeConfig,
    };
    use train::{
        LoraAdapterConfig, LoraConfig, LoraTargetSet,
        infer_student::InferStudent,
        loss::{DEFAULT_KL_CHUNK_SIZE, kl_distill_loss, kl_distill_loss_chunked},
        opd::{
            GkdLossConfig, GkdSftAnchor, InferRolloutCtx, OpdKlMask, OpdStepConfig, OpdStepProfile,
            infer_rollout_flag_enabled, opd_step_with_teacher_forward_profiled_gkd_anchor,
        },
        prompts::load_jsonl_prompt_sets,
        qwen35::{Qwen35Model, SequenceWindow},
        qwen35_checkpoint::{
            ConfigJsonSource, GenerationConfigSource, Qwen35NamedCheckpoint, Qwen35StepCheckpoint,
            Qwen35StudentWeights, save_named_qwen35_student_checkpoint,
            save_qwen35_student_checkpoint,
        },
        qwen35_loader::{load_qwen35_from_hf_dir, load_qwen35_lora_from_hf_dir},
        teacher_infer::{
            ApiTeacher, InProcessTeacher, InferTeacher, MultiTeacher, TeacherEntry, TeacherForward,
            TeacherRoute,
        },
        trainer::extend_keep_with_params_and_grads,
    };

    const DEFAULT_QWEN35_08B_DIR: &str =
        "/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base";
    const DEFAULT_STEPS: usize = 1;
    const DEFAULT_ROLLOUT_LEN: usize = 8;
    const DEFAULT_LR: f32 = 1.0e-5;
    const DEFAULT_PROMPT_MAX_TOKENS: usize = 16;
    const DEFAULT_HELDOUT_PROMPTS: usize = 4;
    const DEFAULT_LORA_RANK: usize = 16;
    const DEFAULT_LORA_ALPHA: f32 = 32.0;
    const DEFAULT_LORA_TARGET_SET: LoraTargetSet = LoraTargetSet::AttentionQv;
    const GRAD_CLIP: f32 = 1.0;

    #[derive(Debug)]
    struct Args {
        teacher_model: PathBuf,
        student_model: PathBuf,
        teacher_api_url: Option<String>,
        teacher_api_key_env: Option<String>,
        teacher_api_dtype: String,
        teacher_config: Option<PathBuf>,
        prompts_file: Option<PathBuf>,
        steps: usize,
        rollout_len: usize,
        lr: f32,
        eval_steps: Vec<usize>,
        prompt_max_tokens: usize,
        max_step_seconds: Option<f64>,
        enable_cuda_graph: bool,
        save_student_checkpoint: Option<PathBuf>,
        save_every: usize,
        gkd_lambda: f32,
        sft_anchor: GkdSftAnchor,
        kl_chunk_size: Option<usize>,
        logits_window_size: Option<usize>,
        opd_kl_mask: OpdKlMask,
        eval_train_prompt_limit: Option<usize>,
        lora_rank: usize,
        lora_alpha: f32,
        lora_target_set: LoraTargetSet,
    }

    #[derive(Debug)]
    struct PromptSets {
        train: Vec<Vec<u32>>,
        heldout: Vec<Vec<u32>>,
        train_completions: Vec<Option<Vec<u32>>>,
        source: String,
    }

    #[derive(Debug, serde::Deserialize)]
    struct TeacherConfigFile {
        default_teacher: String,
        teachers: Vec<ApiTeacherConfig>,
        #[serde(default)]
        routes: Vec<TeacherRouteConfig>,
    }

    #[derive(Debug, serde::Deserialize)]
    struct ApiTeacherConfig {
        id: String,
        url: String,
        #[serde(default)]
        vocab_size: Option<usize>,
        #[serde(default)]
        dtype: Option<String>,
        #[serde(default)]
        api_key_env: Option<String>,
        #[serde(default)]
        timeout_seconds: Option<u64>,
    }

    #[derive(Debug, serde::Deserialize)]
    struct TeacherRouteConfig {
        teacher_id: String,
        token_prefix: Vec<u32>,
    }

    struct NamedApiTeacher {
        id: String,
        teacher: ApiTeacher,
    }

    struct NamedApiTeachers {
        default_teacher: String,
        teachers: Vec<NamedApiTeacher>,
        routes: Vec<TeacherRoute>,
    }

    #[derive(Debug, Default, Clone, Copy)]
    struct RuntimeTeacherProfile {
        infer_forward_seconds: f64,
        infer_sync_seconds: f64,
        d2d_bridge_import_seconds: f64,
        api_http_seconds: f64,
        api_decode_seconds: f64,
        api_upload_seconds: f64,
        seq_len: usize,
        vocab_size: usize,
    }

    pub fn main() -> Result<(), Box<dyn std::error::Error>> {
        let args = parse_args()?;
        let prompts = load_prompts(&args)?;
        validate_prompt_sft_anchors(&args, &prompts)?;
        println!(
            "config backend=cuda teacher_model={} teacher_api_url={} teacher_config={} \
             student_model={} student_mode=lora \
             lora_rank={} lora_alpha={:.6} lora_target_set={} \
             steps={} rollout_len={} lr={:.9e} grad_clip={GRAD_CLIP} \
             prompt_source={} train_prompt_count={} heldout_prompt_count={} \
             eval_steps={:?} cuda_graph={} save_student_checkpoint={} save_every={} \
             gkd_lambda={:.6} sft_anchor={} kl_chunk_size={} logits_window_size={} \
             opd_kl_mask={} eval_train_prompt_limit={}",
            args.teacher_model.display(),
            args.teacher_api_url.as_deref().unwrap_or("none"),
            args.teacher_config
                .as_ref()
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| "none".to_owned()),
            args.student_model.display(),
            args.lora_rank,
            args.lora_alpha,
            args.lora_target_set.label(),
            args.steps,
            args.rollout_len,
            args.lr,
            prompts.source,
            prompts.train.len(),
            prompts.heldout.len(),
            args.eval_steps,
            args.enable_cuda_graph,
            args.save_student_checkpoint
                .as_ref()
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| "none".to_owned()),
            args.save_every,
            args.gkd_lambda,
            sft_anchor_label(args.sft_anchor),
            args.kl_chunk_size
                .map(|value| value.to_string())
                .unwrap_or_else(|| "none".to_owned()),
            args.logits_window_size
                .map(|value| value.to_string())
                .unwrap_or_else(|| "none".to_owned()),
            opd_kl_mask_label(args.opd_kl_mask),
            args.eval_train_prompt_limit
                .map(|value| value.to_string())
                .unwrap_or_else(|| "all".to_owned())
        );
        for (idx, prompt) in prompts.train.iter().enumerate() {
            println!("prompt split=train index={idx} ids={prompt:?}");
            if let Some(completion) = prompts.train_completions.get(idx).and_then(Option::as_ref) {
                println!("prompt split=train index={idx} completion_ids={completion:?}");
            }
        }
        for (idx, prompt) in prompts.heldout.iter().enumerate() {
            println!("prompt split=heldout index={idx} ids={prompt:?}");
        }

        let cuda_backend = Arc::new(CudaBackend::new(0)?);
        let teacher_backend: Arc<dyn Backend> = cuda_backend.clone();
        let mut store = TensorStore::with_backend(cuda_backend.clone());
        let mut tape = Tape::new();
        log_device_vram("00_after_backend_init", &cuda_backend);

        let student_load_started = Instant::now();
        let student = load_qwen35_lora_from_hf_dir(
            &args.student_model,
            LoraConfig {
                rank: args.lora_rank,
                alpha: args.lora_alpha,
            },
            args.lora_target_set,
            &mut store,
        )?;
        let student_load_seconds = student_load_started.elapsed().as_secs_f64();
        log_device_vram("01_after_train_student_base_load", &cuda_backend);
        let student_model_params = student.all_parameter_ids();
        let student_trainable_params = trainable_params(&student, &store);
        let student_host_evict_params = host_evict_param_ids(&student);

        // P4: load the in-process infer student engine (now the default
        // rollout path). Opt out with `ARLE_OPD_INFER_ROLLOUT=0`. The engine
        // loads from the *student* model dir with **no** disk adapter: the
        // pristine BF16 base is snapshotted lazily on the first
        // `remerge_lora` (driven by the per-step `sync_lora_from_store`), so
        // the prior temp-PEFT-dir + `INFER_LORA_PATH` juggling is gone (P4
        // infra hardening — see `infer/src/model/qwen35/weights.rs`).
        let infer_student = if infer_rollout_flag_enabled() {
            if args.lora_target_set != LoraTargetSet::AttentionQv {
                return Err(
                    "the infer rollout path requires --lora-target-set attention-qv (the infer \
                     merge path only carries q/v adapters on full-attention layers); pass \
                     ARLE_OPD_INFER_ROLLOUT=0 to fall back to the train-crate rollout"
                        .into(),
                );
            }
            let infer_student_load_started = Instant::now();
            let engine = load_infer_engine(
                &args.student_model,
                args.prompt_max_tokens + args.rollout_len + 32,
                args.enable_cuda_graph,
            )?;
            let infer_student = InferStudent::new(
                Arc::new(Mutex::new(engine)),
                cuda_backend.clone() as Arc<dyn Backend>,
                student.config().vocab_size,
            );
            println!(
                "infer_student_loaded seconds={:.6} student_model={}",
                infer_student_load_started.elapsed().as_secs_f64(),
                args.student_model.display()
            );
            log_memory_summary("after_infer_student_load", &store);
            log_device_vram("02_after_infer_student_load", &cuda_backend);
            Some(infer_student)
        } else {
            None
        };

        if let Some(config_path) = args.teacher_config.as_ref() {
            let named_teachers = load_api_teacher_config(config_path, student.config().vocab_size)?;
            let entries = named_teachers
                .teachers
                .iter()
                .map(|teacher| {
                    TeacherEntry::new(teacher.id.clone(), &teacher.teacher as &dyn TeacherForward)
                })
                .collect::<Vec<_>>();
            let multi_teacher = MultiTeacher::with_routes(
                entries,
                &named_teachers.default_teacher,
                named_teachers.routes,
            )?;
            return run_training(
                &args,
                &prompts,
                &student,
                &student_model_params,
                &student_trainable_params,
                &student_host_evict_params,
                &mut store,
                &mut tape,
                cuda_backend,
                &multi_teacher,
                "api-multi",
                student_load_seconds,
                0.0,
                infer_student.as_ref(),
                || RuntimeTeacherProfile::default(),
                |prompt| multi_teacher.selected_teacher_id(prompt).to_owned(),
            );
        }

        if let Some(endpoint) = args.teacher_api_url.as_ref() {
            let api_teacher = build_api_teacher(
                endpoint,
                student.config().vocab_size,
                args.teacher_api_key_env.as_deref(),
                &args.teacher_api_dtype,
                None,
            )?;
            return run_training(
                &args,
                &prompts,
                &student,
                &student_model_params,
                &student_trainable_params,
                &student_host_evict_params,
                &mut store,
                &mut tape,
                cuda_backend,
                &api_teacher,
                "api",
                student_load_seconds,
                0.0,
                infer_student.as_ref(),
                || profile_from_api(&api_teacher),
                |_| "api".to_owned(),
            );
        }

        if args.logits_window_size.is_some() {
            let teacher_load_started = Instant::now();
            let teacher_model = load_qwen35_from_hf_dir(&args.teacher_model, &mut store)?;
            let teacher_load_seconds = teacher_load_started.elapsed().as_secs_f64();
            let in_process_teacher = InProcessTeacher::new(&teacher_model);
            let mut host_evict_params = student_host_evict_params.clone();
            host_evict_params.extend(host_evict_param_ids(&teacher_model));
            return run_training(
                &args,
                &prompts,
                &student,
                &student_model_params,
                &student_trainable_params,
                &host_evict_params,
                &mut store,
                &mut tape,
                cuda_backend,
                &in_process_teacher,
                "in-process",
                student_load_seconds,
                teacher_load_seconds,
                infer_student.as_ref(),
                || RuntimeTeacherProfile::default(),
                |_| "in-process".to_owned(),
            );
        }

        let infer_load_started = Instant::now();
        let infer_engine = load_infer_engine(
            &args.teacher_model,
            args.prompt_max_tokens + args.rollout_len + 32,
            args.enable_cuda_graph,
        )?;
        let infer_load_seconds = infer_load_started.elapsed().as_secs_f64();
        log_device_vram("03_after_teacher_infer_load", &cuda_backend);
        let infer_teacher = InferTeacher::new(
            Arc::new(Mutex::new(infer_engine)),
            teacher_backend,
            student.config().vocab_size,
        );
        run_training(
            &args,
            &prompts,
            &student,
            &student_model_params,
            &student_trainable_params,
            &student_host_evict_params,
            &mut store,
            &mut tape,
            cuda_backend,
            &infer_teacher,
            "infer",
            student_load_seconds,
            infer_load_seconds,
            infer_student.as_ref(),
            || profile_from_infer(&infer_teacher),
            |_| "infer".to_owned(),
        )
    }

    fn parse_args() -> Result<Args, Box<dyn std::error::Error>> {
        let mut teacher_model = PathBuf::from(DEFAULT_QWEN35_08B_DIR);
        let mut student_model = PathBuf::from(DEFAULT_QWEN35_08B_DIR);
        let mut teacher_api_url = None;
        let mut teacher_api_key_env = None;
        let mut teacher_api_dtype = "bf16".to_owned();
        let mut teacher_config = None;
        let mut prompts_file = None;
        let mut steps = DEFAULT_STEPS;
        let mut rollout_len = DEFAULT_ROLLOUT_LEN;
        let mut lr = DEFAULT_LR;
        let mut eval_steps = Vec::new();
        let mut prompt_max_tokens = DEFAULT_PROMPT_MAX_TOKENS;
        let mut max_step_seconds = None;
        let mut enable_cuda_graph = true;
        let mut save_student_checkpoint = None;
        let mut save_every = 0usize;
        let mut gkd_lambda = 0.0f32;
        let mut sft_anchor = GkdSftAnchor::StudentRollout;
        let mut kl_chunk_size = Some(DEFAULT_KL_CHUNK_SIZE);
        let mut logits_window_size = None;
        let mut opd_kl_mask = OpdKlMask::CompletionOnly;
        let mut eval_train_prompt_limit = Some(DEFAULT_EVAL_TRAIN_PROMPT_LIMIT);
        let mut lora_rank = DEFAULT_LORA_RANK;
        let mut lora_alpha = DEFAULT_LORA_ALPHA;
        let mut lora_target_set = DEFAULT_LORA_TARGET_SET;

        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--teacher-model" => teacher_model = PathBuf::from(next_arg(&mut args, &arg)?),
                "--student-model" => student_model = PathBuf::from(next_arg(&mut args, &arg)?),
                "--teacher-api-url" => teacher_api_url = Some(next_arg(&mut args, &arg)?),
                "--teacher-api-key-env" => teacher_api_key_env = Some(next_arg(&mut args, &arg)?),
                "--teacher-api-dtype" => teacher_api_dtype = next_arg(&mut args, &arg)?,
                "--teacher-config" => {
                    teacher_config = Some(PathBuf::from(next_arg(&mut args, &arg)?))
                }
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
                "--save-student-checkpoint" => {
                    save_student_checkpoint = Some(PathBuf::from(next_arg(&mut args, &arg)?))
                }
                "--save-every" => save_every = next_arg(&mut args, &arg)?.parse::<usize>()?,
                "--gkd-lambda" => gkd_lambda = parse_gkd_lambda(&next_arg(&mut args, &arg)?)?,
                "--sft-anchor" => sft_anchor = parse_sft_anchor(&next_arg(&mut args, &arg)?)?,
                "--kl-chunk-size" => {
                    kl_chunk_size = Some(parse_positive_usize(&arg, &next_arg(&mut args, &arg)?)?)
                }
                "--logits-window-size" => {
                    logits_window_size =
                        Some(parse_positive_usize(&arg, &next_arg(&mut args, &arg)?)?)
                }
                "--opd-kl-mask" => opd_kl_mask = parse_opd_kl_mask(&next_arg(&mut args, &arg)?)?,
                "--eval-train-prompt-limit" => {
                    eval_train_prompt_limit =
                        parse_optional_usize_or_all(&arg, &next_arg(&mut args, &arg)?)?
                }
                "--lora-rank" => {
                    lora_rank = parse_positive_usize(&arg, &next_arg(&mut args, &arg)?)?
                }
                "--lora-alpha" => lora_alpha = next_arg(&mut args, &arg)?.parse::<f32>()?,
                "--lora-target-set" => {
                    lora_target_set = parse_lora_target_set(&next_arg(&mut args, &arg)?)?
                }
                "--no-cuda-graph" => enable_cuda_graph = false,
                "--help" | "-h" => {
                    println!(
                        "usage: cargo run -p train --example opd_step_cuda_infer_teacher_train \
                         --release --features cuda -- [--teacher-model DIR] [--student-model DIR] \
                         [--teacher-api-url URL] [--teacher-config JSON] [--prompts-file JSONL] \
                         [--steps N] [--rollout-len N] [--lr LR] \
                         [--eval-steps CSV] [--prompt-max-tokens N] [--max-step-seconds SEC] \
                         [--save-student-checkpoint DIR] [--save-every N] \
                         [--gkd-lambda LAMBDA] [--sft-anchor student-rollout|corpus-truth] \
                         [--kl-chunk-size N(default 32)] [--logits-window-size N] \
                         [--opd-kl-mask full|completion-only(default)] \
                         [--eval-train-prompt-limit N|all(default {DEFAULT_EVAL_TRAIN_PROMPT_LIMIT})] \
                         [--lora-rank N(default {DEFAULT_LORA_RANK})] \
                         [--lora-alpha F(default {DEFAULT_LORA_ALPHA})] \
                         [--lora-target-set attention-qv(default)|all-linear] \
                         [--no-cuda-graph]"
                    );
                    std::process::exit(0);
                }
                _ => return Err(format!("unknown argument `{arg}`").into()),
            }
        }
        if teacher_api_url.is_some() && teacher_config.is_some() {
            return Err("--teacher-api-url and --teacher-config are mutually exclusive".into());
        }
        if save_every > 0 && save_student_checkpoint.is_none() {
            return Err("--save-every requires --save-student-checkpoint".into());
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
            teacher_api_url,
            teacher_api_key_env,
            teacher_api_dtype,
            teacher_config,
            prompts_file,
            steps,
            rollout_len,
            lr,
            eval_steps,
            prompt_max_tokens,
            max_step_seconds,
            enable_cuda_graph,
            save_student_checkpoint,
            save_every,
            gkd_lambda,
            sft_anchor,
            kl_chunk_size,
            logits_window_size,
            opd_kl_mask,
            eval_train_prompt_limit,
            lora_rank,
            lora_alpha,
            lora_target_set,
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

    fn parse_optional_usize_or_all(
        flag: &str,
        raw: &str,
    ) -> Result<Option<usize>, Box<dyn std::error::Error>> {
        if raw.eq_ignore_ascii_case("all") {
            return Ok(None);
        }
        raw.parse::<usize>()
            .map(Some)
            .map_err(|err| format!("{flag} must be a non-negative integer or `all`: {err}").into())
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

    fn parse_gkd_lambda(raw: &str) -> Result<f32, Box<dyn std::error::Error>> {
        let value = raw.parse::<f32>()?;
        if !(0.0..=1.0).contains(&value) || !value.is_finite() {
            return Err("--gkd-lambda must be finite and in [0.0, 1.0]".into());
        }
        Ok(value)
    }

    fn parse_sft_anchor(raw: &str) -> Result<GkdSftAnchor, Box<dyn std::error::Error>> {
        match raw {
            "student-rollout" => Ok(GkdSftAnchor::StudentRollout),
            "corpus-truth" => Ok(GkdSftAnchor::CorpusTruth),
            _ => Err(format!(
                "--sft-anchor must be one of student-rollout|corpus-truth, got `{raw}`"
            )
            .into()),
        }
    }

    fn parse_opd_kl_mask(raw: &str) -> Result<OpdKlMask, Box<dyn std::error::Error>> {
        match raw {
            "full" => Ok(OpdKlMask::Full),
            "completion-only" => Ok(OpdKlMask::CompletionOnly),
            _ => Err(
                format!("--opd-kl-mask must be one of full|completion-only, got `{raw}`").into(),
            ),
        }
    }

    fn parse_lora_target_set(raw: &str) -> Result<LoraTargetSet, Box<dyn std::error::Error>> {
        match raw {
            "attention-qv" | "attention_qv" | "qv" => Ok(LoraTargetSet::AttentionQv),
            "all-linear" | "all_linear" | "all" => Ok(LoraTargetSet::AllLinear),
            _ => Err(format!(
                "--lora-target-set must be one of attention-qv|all-linear, got `{raw}`"
            )
            .into()),
        }
    }

    fn sft_anchor_label(anchor: GkdSftAnchor) -> &'static str {
        match anchor {
            GkdSftAnchor::StudentRollout => "student-rollout",
            GkdSftAnchor::CorpusTruth => "corpus-truth",
        }
    }

    fn opd_kl_mask_label(mask: OpdKlMask) -> &'static str {
        match mask {
            OpdKlMask::Full => "full",
            OpdKlMask::CompletionOnly => "completion-only",
        }
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
                train_completions: loaded.train_completions,
                source: format!(
                    "jsonl:{} rows={} tokenizer={} truncated_rows={} completion_rows={} truncated_completion_rows={}",
                    loaded.prompt_file.display(),
                    loaded.jsonl_rows,
                    loaded.tokenizer_path.display(),
                    loaded.truncated_rows,
                    loaded.completion_rows,
                    loaded.truncated_completion_rows
                ),
            });
        }
        Ok(PromptSets {
            train: vec![vec![9419]],
            heldout: Vec::new(),
            train_completions: vec![None],
            source: "single-token-hello".to_string(),
        })
    }

    fn validate_prompt_sft_anchors(
        args: &Args,
        prompts: &PromptSets,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if args.gkd_lambda == 0.0 || args.sft_anchor != GkdSftAnchor::CorpusTruth {
            return Ok(());
        }
        if args.prompts_file.is_none() {
            return Err("--sft-anchor corpus-truth requires --prompts-file with completion or target fields".into());
        }
        if prompts.train_completions.len() != prompts.train.len() {
            return Err(format!(
                "internal prompt/completion split mismatch: train_prompts={} train_completions={}",
                prompts.train.len(),
                prompts.train_completions.len()
            )
            .into());
        }
        if let Some((index, _)) = prompts
            .train_completions
            .iter()
            .enumerate()
            .find(|(_, completion)| completion.as_ref().map_or(true, |tokens| tokens.is_empty()))
        {
            return Err(format!(
                "--sft-anchor corpus-truth requires non-empty completion/target tokens for every training row; missing train index {index}"
            )
            .into());
        }
        Ok(())
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

    fn run_training<T, ProfileFn, RouteFn>(
        args: &Args,
        prompts: &PromptSets,
        student: &Qwen35Model,
        student_model_params: &[TensorId],
        student_trainable_params: &[TensorId],
        host_evict_params: &[TensorId],
        store: &mut TensorStore,
        tape: &mut Tape,
        cuda_backend: Arc<CudaBackend>,
        teacher: &T,
        teacher_source: &str,
        student_load_seconds: f64,
        teacher_load_seconds: f64,
        infer_student: Option<&InferStudent>,
        mut teacher_profile: ProfileFn,
        route_teacher_id: RouteFn,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        T: TeacherForward + ?Sized,
        ProfileFn: FnMut() -> RuntimeTeacherProfile,
        RouteFn: Fn(&[u32]) -> String,
    {
        let vram_backend = cuda_backend.clone();
        log_device_vram("04_before_optimizer_init", &vram_backend);
        let mut optimizer =
            AdamW::new_with_device(args.lr, (0.9, 0.999), 1.0e-8, 0.0, cuda_backend);
        println!(
            "model_summary teacher_source={teacher_source} student_hidden={} student_layers={} student_vocab={} \
             student_model_elements={} student_trainable_elements={} \
             student_load_seconds={student_load_seconds:.6} teacher_load_seconds={teacher_load_seconds:.6}",
            student.config().hidden_size,
            student.config().num_hidden_layers,
            student.config().vocab_size,
            param_element_count(student_model_params, store),
            param_element_count(student_trainable_params, store)
        );
        log_memory_summary("after_model_summary", store);
        let evict_started = Instant::now();
        let before_host_bytes = tensor_host_bytes(store);
        let evicted_bytes = evict_static_param_host_mirrors(store, host_evict_params)?;
        println!(
            "model_host_evict_summary evicted_bytes={evicted_bytes} \
             host_tensor_bytes_before={before_host_bytes} \
             host_tensor_bytes_after={} seconds={:.6}",
            tensor_host_bytes(store),
            evict_started.elapsed().as_secs_f64()
        );
        log_memory_summary("after_model_host_evict", store);

        maybe_eval(
            0,
            args,
            prompts,
            teacher,
            student,
            student_model_params,
            store,
            tape,
        )?;

        let mut step_losses = Vec::with_capacity(args.steps);
        let mut step_seconds = Vec::with_capacity(args.steps);
        let total_started = Instant::now();
        for step in 1..=args.steps {
            let prompt_index = (step - 1) % prompts.train.len();
            let prompt = prompts.train[prompt_index].as_slice();
            let corpus_tokens = if args.sft_anchor == GkdSftAnchor::CorpusTruth {
                prompts
                    .train_completions
                    .get(prompt_index)
                    .and_then(Option::as_deref)
            } else {
                None
            };
            let selected_teacher = route_teacher_id(prompt);
            let mut profile = OpdStepProfile::default();
            log_memory_summary("before_train_step", store);
            log_device_vram(&format!("05_before_train_step_{step}"), &vram_backend);
            let infer_rollout = infer_student.map(|student| InferRolloutCtx {
                student,
                lora_config: LoraConfig {
                    rank: args.lora_rank,
                    alpha: args.lora_alpha,
                },
            });
            let step_started = Instant::now();
            let outcome = opd_step_with_teacher_forward_profiled_gkd_anchor(
                student,
                teacher,
                prompt,
                OpdStepConfig {
                    rollout_len: args.rollout_len,
                    grad_clip: GRAD_CLIP,
                },
                student_trainable_params,
                &mut optimizer,
                store,
                tape,
                GkdLossConfig {
                    lambda: args.gkd_lambda,
                    sft_anchor: args.sft_anchor,
                    corpus_tokens,
                    kl_chunk_size: args.kl_chunk_size,
                    logits_window_size: args.logits_window_size,
                    kl_mask: args.opd_kl_mask,
                },
                infer_rollout,
                Some(&mut profile),
            )?;
            let elapsed = step_started.elapsed().as_secs_f64();
            log_memory_summary("after_train_step", store);
            log_device_vram(&format!("06_after_train_step_{step}"), &vram_backend);
            if let Some(max_step_seconds) = args.max_step_seconds {
                if step == 1 && elapsed > max_step_seconds {
                    return Err(format!(
                        "first {teacher_source} TeacherForward OPD step took {elapsed:.6}s, above configured ceiling {max_step_seconds:.6}s"
                    )
                    .into());
                }
            }
            let runtime_profile = teacher_profile();
            step_losses.push(outcome.loss as f64);
            step_seconds.push(elapsed);
            println!(
                "train_step step={step} prompt_index={prompt_index} teacher_id={selected_teacher} \
                 loss={:.12e} rollout_len={} step_seconds={elapsed:.6} sft_anchor={}",
                outcome.loss,
                outcome.rollout_len,
                sft_anchor_label(args.sft_anchor)
            );
            println!(
                "phase_summary step={step} total={:.6} student_rollout={:.6} \
                 infer_forward_token_logits={:.6} infer_sync={:.6} d2d_bridge_import={:.6} \
                 api_http={:.6} api_decode={:.6} api_upload={:.6} \
                 teacher_forward_total={:.6} student_forward={:.6} kl_loss={:.6} \
                 optimizer_zero_grad={:.6} backward={:.6} grad_clip={:.6} \
                 optimizer_step={:.6} post_step_cleanup={:.6} teacher_seq_len={} teacher_vocab={}",
                profile.total_seconds,
                profile.student_rollout_seconds,
                runtime_profile.infer_forward_seconds,
                runtime_profile.infer_sync_seconds,
                runtime_profile.d2d_bridge_import_seconds,
                runtime_profile.api_http_seconds,
                runtime_profile.api_decode_seconds,
                runtime_profile.api_upload_seconds,
                profile.teacher_forward_seconds,
                profile.student_forward_seconds,
                profile.kl_loss_seconds,
                profile.optimizer_zero_grad_seconds,
                profile.backward_seconds,
                profile.grad_clip_seconds,
                profile.optimizer_step_seconds,
                profile.post_step_cleanup_seconds,
                runtime_profile.seq_len,
                runtime_profile.vocab_size
            );
            maybe_eval(
                step,
                args,
                prompts,
                teacher,
                student,
                student_model_params,
                store,
                tape,
            )?;
            maybe_save_student_checkpoint(
                args,
                CheckpointTarget::Step(step),
                student,
                store,
                tape,
            )?;
        }
        maybe_save_student_checkpoint(args, CheckpointTarget::Final, student, store, tape)?;
        println!(
            "training_summary teacher_source={teacher_source} total_steps={} total_wall_seconds={:.6} mean_step_seconds={:.6} \
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

    #[derive(Clone, Copy)]
    enum CheckpointTarget {
        Step(usize),
        Final,
    }

    fn maybe_save_student_checkpoint(
        args: &Args,
        target: CheckpointTarget,
        student: &Qwen35Model,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let Some(out_dir) = args.save_student_checkpoint.as_ref() else {
            return Ok(());
        };
        match target {
            CheckpointTarget::Step(step) => {
                if args.save_every == 0 || step % args.save_every != 0 {
                    return Ok(());
                }
                save_student_checkpoint(
                    args,
                    out_dir,
                    CheckpointTarget::Step(step),
                    student,
                    store,
                    tape,
                )
            }
            CheckpointTarget::Final => save_student_checkpoint(
                args,
                out_dir,
                CheckpointTarget::Final,
                student,
                store,
                tape,
            ),
        }
    }

    fn save_student_checkpoint(
        args: &Args,
        out_dir: &Path,
        target: CheckpointTarget,
        student: &Qwen35Model,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<(), Box<dyn std::error::Error>> {
        fs::create_dir_all(out_dir)?;
        let started = Instant::now();
        let adapter_config = lora_adapter_config(
            &args.student_model,
            args.lora_rank,
            args.lora_alpha,
            args.lora_target_set,
        );
        let sources = checkpoint_sources(args)?;
        let tokenizer_path = sources.tokenizer_path.as_deref();
        let saved_dir = match target {
            CheckpointTarget::Step(step) => save_qwen35_student_checkpoint(
                Qwen35StepCheckpoint {
                    out_dir,
                    step,
                    tokenizer_path,
                    config_json: ConfigJsonSource::CopyFrom(&sources.config_path),
                    generation_config: GenerationConfigSource::CopyOrSynthesize {
                        source_path: &sources.generation_config_path,
                        fallback_config_path: &sources.config_path,
                    },
                },
                student,
                store,
                tape,
                Qwen35StudentWeights::AdapterOnly {
                    bf16: true,
                    adapter_config: &adapter_config,
                },
            )?,
            CheckpointTarget::Final => save_named_qwen35_student_checkpoint(
                Qwen35NamedCheckpoint {
                    out_dir,
                    dirname: "final",
                    tokenizer_path,
                    config_json: ConfigJsonSource::CopyFrom(&sources.config_path),
                    generation_config: GenerationConfigSource::CopyOrSynthesize {
                        source_path: &sources.generation_config_path,
                        fallback_config_path: &sources.config_path,
                    },
                },
                student,
                store,
                tape,
                Qwen35StudentWeights::AdapterOnly {
                    bf16: true,
                    adapter_config: &adapter_config,
                },
            )?,
        };
        println!(
            "checkpoint_saved kind=lora_adapter target={} dir={} seconds={:.6}",
            match target {
                CheckpointTarget::Step(step) => format!("step_{step:06}"),
                CheckpointTarget::Final => "final".to_owned(),
            },
            saved_dir.display(),
            started.elapsed().as_secs_f64()
        );
        Ok(())
    }

    struct CheckpointSources {
        tokenizer_path: Option<PathBuf>,
        config_path: PathBuf,
        generation_config_path: PathBuf,
    }

    fn checkpoint_sources(args: &Args) -> Result<CheckpointSources, Box<dyn std::error::Error>> {
        let config_path = args.student_model.join("config.json");
        if !config_path.is_file() {
            return Err(format!(
                "student config.json not found at {}; adapter checkpoint save needs the base HF config",
                config_path.display()
            )
            .into());
        }
        let tokenizer_path = args.student_model.join("tokenizer.json");
        let generation_config_path = args.student_model.join("generation_config.json");
        Ok(CheckpointSources {
            tokenizer_path: tokenizer_path.is_file().then_some(tokenizer_path),
            config_path,
            generation_config_path,
        })
    }

    fn lora_adapter_config(
        student_model: &Path,
        lora_rank: usize,
        lora_alpha: f32,
        lora_target_set: LoraTargetSet,
    ) -> LoraAdapterConfig {
        let mut config = LoraAdapterConfig::new(
            student_model.display().to_string(),
            "qwen35",
            LoraConfig {
                rank: lora_rank,
                alpha: lora_alpha,
            },
        );
        config.target_modules = match lora_target_set {
            LoraTargetSet::AttentionQv => vec!["q_proj".to_owned(), "v_proj".to_owned()],
            LoraTargetSet::AllLinear => vec!["all-linear".to_owned()],
        };
        config
    }

    fn build_api_teacher(
        endpoint: &str,
        vocab_size: usize,
        api_key_env: Option<&str>,
        dtype: &str,
        timeout_seconds: Option<u64>,
    ) -> Result<ApiTeacher, Box<dyn std::error::Error>> {
        let timeout = Duration::from_secs(timeout_seconds.unwrap_or(30));
        let mut teacher = ApiTeacher::with_timeout(endpoint.to_owned(), vocab_size, timeout)
            .with_request_dtype(dtype.to_owned());
        if let Some(env_name) = api_key_env {
            let api_key = std::env::var(env_name)
                .map_err(|_| format!("--teacher-api-key-env {env_name} is not set"))?;
            teacher = teacher.with_api_key(api_key);
        }
        Ok(teacher)
    }

    fn load_api_teacher_config(
        path: &Path,
        default_vocab_size: usize,
    ) -> Result<NamedApiTeachers, Box<dyn std::error::Error>> {
        let raw = fs::read_to_string(path)
            .map_err(|err| format!("failed to read teacher config {}: {err}", path.display()))?;
        let config: TeacherConfigFile = serde_json::from_str(&raw)
            .map_err(|err| format!("invalid teacher config {}: {err}", path.display()))?;
        if config.teachers.is_empty() {
            return Err("teacher config requires at least one teacher".into());
        }
        let mut teachers = Vec::with_capacity(config.teachers.len());
        for teacher in config.teachers {
            if teacher.id.trim().is_empty() {
                return Err("teacher config teacher id must be non-empty".into());
            }
            let vocab_size = teacher.vocab_size.unwrap_or(default_vocab_size);
            let dtype = teacher.dtype.as_deref().unwrap_or("bf16");
            let api_teacher = build_api_teacher(
                &teacher.url,
                vocab_size,
                teacher.api_key_env.as_deref(),
                dtype,
                teacher.timeout_seconds,
            )?;
            teachers.push(NamedApiTeacher {
                id: teacher.id,
                teacher: api_teacher,
            });
        }
        let routes = config
            .routes
            .into_iter()
            .map(|route| TeacherRoute::new(route.teacher_id, route.token_prefix))
            .collect();
        Ok(NamedApiTeachers {
            default_teacher: config.default_teacher,
            teachers,
            routes,
        })
    }

    fn profile_from_infer(teacher: &InferTeacher) -> RuntimeTeacherProfile {
        let profile = teacher.last_profile();
        RuntimeTeacherProfile {
            infer_forward_seconds: profile.raw_forward_seconds,
            infer_sync_seconds: profile.sync_seconds,
            d2d_bridge_import_seconds: profile.d2d_bridge_import_seconds,
            seq_len: profile.seq_len,
            vocab_size: profile.vocab_size,
            ..RuntimeTeacherProfile::default()
        }
    }

    fn profile_from_api(teacher: &ApiTeacher) -> RuntimeTeacherProfile {
        let profile = teacher.last_profile();
        RuntimeTeacherProfile {
            api_http_seconds: profile.http_seconds,
            api_decode_seconds: profile.decode_seconds,
            api_upload_seconds: profile.upload_seconds,
            seq_len: profile.seq_len,
            vocab_size: profile.vocab_size,
            ..RuntimeTeacherProfile::default()
        }
    }

    fn maybe_eval<T: TeacherForward + ?Sized>(
        step: usize,
        args: &Args,
        prompts: &PromptSets,
        teacher: &T,
        student: &Qwen35Model,
        student_model_params: &[TensorId],
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !args.eval_steps.contains(&step) {
            return Ok(());
        }
        let started = Instant::now();
        let train_prompt_count =
            super::eval_prompt_limit_len(prompts.train.len(), args.eval_train_prompt_limit);
        let train_kl = mean_prompt_kl(
            "train",
            &prompts.train[..train_prompt_count],
            teacher,
            student,
            student_model_params,
            store,
            tape,
            args.kl_chunk_size,
            args.logits_window_size,
        )?;
        let heldout_kl = mean_prompt_kl(
            "heldout",
            &prompts.heldout,
            teacher,
            student,
            student_model_params,
            store,
            tape,
            args.kl_chunk_size,
            args.logits_window_size,
        )?;
        println!(
            "eval_summary step={step} train_kl={train_kl:.12e} heldout_kl={heldout_kl:.12e} \
             eval_seconds={:.6} train_eval_count={} heldout_eval_count={}",
            started.elapsed().as_secs_f64(),
            train_prompt_count,
            prompts.heldout.len()
        );
        Ok(())
    }

    fn mean_prompt_kl<T: TeacherForward + ?Sized>(
        split: &str,
        prompts: &[Vec<u32>],
        teacher: &T,
        student: &Qwen35Model,
        student_model_params: &[TensorId],
        store: &mut TensorStore,
        tape: &mut Tape,
        kl_chunk_size: Option<usize>,
        logits_window_size: Option<usize>,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        if prompts.is_empty() {
            return Ok(f64::NAN);
        }
        let mut total = 0.0f64;
        for (prompt_index, prompt) in prompts.iter().enumerate() {
            let prompt_started = Instant::now();
            tape.entries.clear();
            tape.set_enabled(false);
            let positions = (0..prompt.len() as u32).collect::<Vec<_>>();
            if let Some(window_size) = logits_window_size {
                let mut prompt_kl = 0.0f64;
                let mut start = 0usize;
                let mut window_index = 0usize;
                while start < prompt.len() {
                    let window_started = Instant::now();
                    let end = start.saturating_add(window_size).min(prompt.len());
                    let window = SequenceWindow { start, end };
                    let teacher_logits = teacher
                        .forward_logits_window_device(prompt, &positions, window, store, tape)?;
                    let student_logits =
                        student.forward_logits_window(store, tape, prompt, &positions, window)?;
                    let loss = match kl_chunk_size {
                        Some(chunk_size) => kl_distill_loss_chunked(
                            student_logits,
                            teacher_logits.tensor_id,
                            window.len(),
                            chunk_size,
                            store,
                            tape,
                        )?,
                        None => kl_distill_loss(
                            student_logits,
                            teacher_logits.tensor_id,
                            window.len(),
                            store,
                            tape,
                        )?,
                    };
                    prompt_kl += store.to_host(loss)?[0] as f64
                        * (window.len() as f64 / prompt.len() as f64);
                    retain_eval_state(store, tape, student_model_params, teacher.parameter_ids());
                    println!(
                        "eval_window_summary split={split} prompt_index={prompt_index} \
                         window_index={window_index} start={start} end={end} \
                         window_seconds={:.6} live_tensors={} tape_entries={}",
                        window_started.elapsed().as_secs_f64(),
                        live_tensor_count(store),
                        tape.entries.len()
                    );
                    tape.set_enabled(false);
                    start = end;
                    window_index += 1;
                }
                total += prompt_kl;
                println!(
                    "eval_prompt_summary split={split} prompt_index={prompt_index} \
                     prompt_len={} windows={window_index} kl={prompt_kl:.12e} \
                     prompt_seconds={:.6} live_tensors={} tape_entries={}",
                    prompt.len(),
                    prompt_started.elapsed().as_secs_f64(),
                    live_tensor_count(store),
                    tape.entries.len()
                );
            } else {
                let teacher_logits =
                    teacher.forward_logits_device(prompt, &positions, store, tape)?;
                let student_logits = student.forward(store, tape, prompt, &positions)?;
                let loss = match kl_chunk_size {
                    Some(chunk_size) => kl_distill_loss_chunked(
                        student_logits,
                        teacher_logits.tensor_id,
                        prompt.len(),
                        chunk_size,
                        store,
                        tape,
                    )?,
                    None => kl_distill_loss(
                        student_logits,
                        teacher_logits.tensor_id,
                        prompt.len(),
                        store,
                        tape,
                    )?,
                };
                let prompt_kl = store.to_host(loss)?[0] as f64;
                total += prompt_kl;
                retain_eval_state(store, tape, student_model_params, teacher.parameter_ids());
                println!(
                    "eval_prompt_summary split={split} prompt_index={prompt_index} \
                     prompt_len={} windows=1 kl={prompt_kl:.12e} prompt_seconds={:.6} \
                     live_tensors={} tape_entries={}",
                    prompt.len(),
                    prompt_started.elapsed().as_secs_f64(),
                    live_tensor_count(store),
                    tape.entries.len()
                );
            }
        }
        Ok(total / prompts.len() as f64)
    }

    fn live_tensor_count(store: &TensorStore) -> usize {
        store.tensors.iter().filter(|slot| slot.is_some()).count()
    }

    fn tensor_host_bytes(store: &TensorStore) -> usize {
        store
            .tensors
            .iter()
            .filter_map(|slot| slot.as_ref())
            .map(|tensor| tensor.data.capacity() * std::mem::size_of::<f32>())
            .sum()
    }

    fn process_rss_kb() -> Option<u64> {
        let status = fs::read_to_string("/proc/self/status").ok()?;
        status.lines().find_map(|line| {
            let rest = line.strip_prefix("VmRSS:")?;
            rest.split_whitespace().next()?.parse::<u64>().ok()
        })
    }

    fn log_memory_summary(label: &str, store: &TensorStore) {
        let rss_kb = process_rss_kb()
            .map(|value| value.to_string())
            .unwrap_or_else(|| "na".to_owned());
        println!(
            "memory_summary label={label} rss_kb={rss_kb} \
             host_tensor_bytes={} live_tensors={} device_only_tensors={}",
            tensor_host_bytes(store),
            live_tensor_count(store),
            store
                .tensors
                .iter()
                .filter_map(|slot| slot.as_ref())
                .filter(|tensor| tensor.data.is_empty() && tensor.device_handle.is_some())
                .count()
        );
    }

    /// Print device VRAM `(used, free, total)` MiB for the given phase label.
    /// Used by the rollout-128 VRAM-fit attribution (free = total - free is
    /// the cudarc `mem_get_info` semantics: it returns `(free, total)`).
    fn log_device_vram(label: &str, backend: &CudaBackend) {
        match backend.mem_get_info() {
            Ok((free, total)) => {
                let used = total.saturating_sub(free);
                println!(
                    "device_vram label={label} used_mib={} free_mib={} total_mib={}",
                    used / (1024 * 1024),
                    free / (1024 * 1024),
                    total / (1024 * 1024)
                );
            }
            Err(err) => println!("device_vram label={label} error={err:?}"),
        }
    }

    fn evict_static_param_host_mirrors(
        store: &mut TensorStore,
        params: &[TensorId],
    ) -> autograd::Result<usize> {
        let mut seen = HashSet::new();
        let mut evicted = 0usize;
        for id in params.iter().copied() {
            if !seen.insert(id) {
                continue;
            }
            let should_evict = store.get(id).is_some_and(|tensor| {
                tensor.shape.len() == 2
                    && tensor.size >= STATIC_PARAM_EVICT_MIN_ELEMENTS
                    && tensor.data.capacity() > 0
            });
            if !should_evict {
                continue;
            }
            evicted += store.evict_host_mirror(id)?;
        }
        Ok(evicted)
    }

    fn host_evict_param_ids(model: &Qwen35Model) -> Vec<TensorId> {
        model.param_name_map().values().copied().collect()
    }

    fn retain_eval_state(
        store: &mut TensorStore,
        tape: &mut Tape,
        student_params: &[TensorId],
        teacher_params: &[TensorId],
    ) {
        tape.entries.clear();
        let mut keep = HashSet::with_capacity((student_params.len() + teacher_params.len()) * 2);
        extend_keep_with_params_and_grads(&mut keep, student_params.iter().copied(), store);
        extend_keep_with_params_and_grads(&mut keep, teacher_params.iter().copied(), store);
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
