use std::{
    fs,
    path::{Path, PathBuf},
    process::ExitCode,
    time::Instant,
};

use anyhow::{Context, Result, anyhow, bail};
use autograd::{Tape, TensorId, TensorStore};
use deepseek_spec::{DeepSeekV4AttentionMode, DeepSeekV4Config};
use indicatif::{ProgressBar, ProgressStyle};
use infer_plan::SamplingParams;
use qwen35_spec::{LayerType, Qwen35Config};
use serde::Serialize;
use train::{
    model_family::{ModelFamily, resolve_model_family},
    tokenizer::ChatTokenizer,
};

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
use crate::args::{ModelArgs, ModelCommand, ModelDownloadArgs, ModelSourceArg};
use crate::{
    args::{
        KlDirectionArg, LrScheduleArg, ModelFamilyArg, OpdBackendArg, OpdKlMaskArg,
        OpdSftAnchorArg, OpdTeacherRuntimeArg, PretrainPresetArg, SaveDtypeArg, TapeDtypeArg,
        TrainAgentOpdArgs, TrainArgs, TrainCcConvertArgs, TrainCommand, TrainEnvArgs,
        TrainEstimateMemoryArgs, TrainOpdArgs, TrainPplArgs, TrainRubricOpdArgs, TrainSelfOpdArgs,
        TrainW2sArgs,
    },
    hardware, hub_discovery,
};

const TRAIN_ENV_COMMANDS: &[&str] = &["train env", "train estimate-memory", "train opd"];

fn opd_step_profile_enabled() -> bool {
    std::env::var_os("ARLE_OPD_STEP_PROFILE").is_some()
}

fn opd_logits_window_size_arg(value: usize) -> Option<usize> {
    (value > 0).then_some(value)
}

fn print_opd_step_profile(step: usize, profile: &train::opd::OpdStepProfile) {
    println!(
        "opd_step_profile step={step} total_seconds={:.6} student_rollout_seconds={:.6} \
         teacher_forward_seconds={:.6} student_forward_seconds={:.6} kl_loss_seconds={:.6} \
         backward_seconds={:.6} optimizer_zero_grad_seconds={:.6} grad_clip_seconds={:.6} \
         optimizer_step_seconds={:.6} post_step_cleanup_seconds={:.6}",
        profile.total_seconds,
        profile.student_rollout_seconds,
        profile.teacher_forward_seconds,
        profile.student_forward_seconds,
        profile.kl_loss_seconds,
        profile.backward_seconds,
        profile.optimizer_zero_grad_seconds,
        profile.grad_clip_seconds,
        profile.optimizer_step_seconds,
        profile.post_step_cleanup_seconds,
    );
}

#[derive(Debug, Clone, Serialize)]
struct OpdStepMetric {
    step: usize,
    loss: f32,
    lr: f32,
    grad_norm: f32,
    rollout_len: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OpdSummary {
    step_count: usize,
    final_loss: Option<f32>,
    mean_loss: Option<f32>,
    min_loss: Option<f32>,
    max_loss: Option<f32>,
}

pub(crate) fn run_train(train: TrainArgs) -> ExitCode {
    // OPD runtime toggles land in the train/autograd statics once, before any
    // model load or step (runtime config = CLI flags, not env).
    match train.command {
        TrainCommand::Env(args) => exit_from_result(run_train_env(args)),
        TrainCommand::EstimateMemory(args) => exit_from_result(run_train_estimate_memory(args)),
        TrainCommand::Opd(args) => {
            train::apply_runtime_flags(&args.runtime.to_flags());
            run_opd(args)
        }
        TrainCommand::SelfOpd(args) => {
            train::apply_runtime_flags(&args.runtime.to_flags());
            run_self_opd(args)
        }
        TrainCommand::RubricOpd(args) => {
            train::apply_runtime_flags(&args.runtime.to_flags());
            exit_from_result(run_rubric_opd_impl(args))
        }
        TrainCommand::AgentOpd(args) => {
            train::apply_runtime_flags(&args.runtime.to_flags());
            exit_from_result(run_agent_opd_impl(*args))
        }
        TrainCommand::CcConvert(args) => exit_from_result(run_cc_convert_impl(args)),
        TrainCommand::Ppl(args) => exit_from_result(run_ppl(args)),
        TrainCommand::SpecDraft(args) => {
            train::apply_runtime_flags(&args.runtime.to_flags());
            exit_from_result(run_spec_draft(args))
        }
        TrainCommand::W2s(args) => {
            train::apply_runtime_flags(&args.runtime.to_flags());
            exit_from_result(run_w2s(args))
        }
    }
}

/// `arle train ppl` — perplexity over a text corpus via teacher-forced logits,
/// to calibrate FP8 / quant checkpoint quality. CUDA-only (the forward path is
/// `LoadedInferenceEngine::forward_token_logits`, which the OPD lane also uses).
#[cfg(feature = "cuda")]
fn run_ppl(args: TrainPplArgs) -> Result<()> {
    use infer_api::{EngineLoadConfig, LoadedInferenceEngine};

    let ctx = args.ctx;
    let model_path = args
        .model_path
        .to_str()
        .ok_or_else(|| anyhow!("model path is not valid UTF-8"))?;

    // Tokenize the corpus into one stream with the model's own tokenizer.
    let tokenizer_path = resolve_local_tokenizer_path(&args.model_path)?;
    let tokenizer = ChatTokenizer::from_file(&tokenizer_path)
        .with_context(|| format!("load tokenizer {}", tokenizer_path.display()))?;
    let corpus = fs::read_to_string(&args.corpus)
        .with_context(|| format!("read corpus {}", args.corpus.display()))?;
    let tokens = tokenizer
        .encode(&corpus, false)
        .context("tokenize corpus")?;
    if tokens.len() < 2 {
        bail!(
            "corpus tokenizes to {} tokens; need at least 2 for a next-token NLL",
            tokens.len()
        );
    }

    let engine = LoadedInferenceEngine::load_with_config(
        model_path,
        /*cuda_graph=*/ true,
        EngineLoadConfig::single_sequence(ctx),
    )
    .with_context(|| format!("load engine from {model_path}"))?;

    // Non-overlapping windows of `ctx`; each contributes len-1 next-token NLLs.
    let mut sum_nll = 0.0f64;
    let mut count = 0usize;
    let mut windows = 0usize;
    for chunk in tokens
        .chunks(ctx)
        .take(args.max_windows.unwrap_or(usize::MAX))
    {
        if chunk.len() < 2 {
            break; // a trailing 1-token window has no next-token target
        }
        let positions: Vec<u32> = (0..chunk.len() as u32).collect();
        let raw = engine
            .forward_token_logits(chunk, &positions)
            .with_context(|| format!("forward window {windows}"))?;
        if raw.seq_len() != chunk.len() {
            bail!(
                "ppl window seq_len mismatch: logits={}, tokens={}",
                raw.seq_len(),
                chunk.len()
            );
        }
        let vocab = raw.vocab_size();
        let host = raw.to_host_f32()?;
        // chunks_exact yields chunk.len() rows; windows(2) yields len-1 pairs →
        // zip drops the final row (no next token), scoring each token once.
        for (row, pair) in host.chunks_exact(vocab).zip(chunk.windows(2)) {
            sum_nll += row_nll(row, pair[1] as usize)?;
            count += 1;
        }
        windows += 1;
    }

    if count == 0 {
        bail!("no scorable positions (corpus shorter than 2 tokens per window)");
    }
    let ppl = (sum_nll / count as f64).exp();

    if args.json {
        println!(
            "{}",
            serde_json::json!({
                "ppl": ppl,
                "tokens": count,
                "windows": windows,
                "ctx": ctx,
                "model_path": model_path,
            })
        );
    } else {
        println!("ppl={ppl} tokens={count} windows={windows} ctx={ctx}");
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn run_ppl(_args: TrainPplArgs) -> Result<()> {
    bail!("arle train ppl requires the CUDA backend (forward_token_logits is CUDA-only)")
}

#[cfg(feature = "cuda")]
use crate::spec_train_target::run_spec_draft;

#[cfg(not(feature = "cuda"))]
fn run_spec_draft(_args: crate::args::TrainSpecDraftArgs) -> Result<()> {
    bail!("arle train spec-draft requires the CUDA backend (the trunk forward is CUDA-only)")
}

/// Numerically stable next-token NLL: `-log_softmax(row)[target]`, subtracting
/// the row max before exp.
fn row_nll(row: &[f32], target: usize) -> Result<f64> {
    let target_logit = *row
        .get(target)
        .ok_or_else(|| anyhow!("target token {target} out of vocab {}", row.len()))?
        as f64;
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum_exp: f64 = row.iter().map(|&l| (l as f64 - max).exp()).sum();
    // NLL = -(logit - max - ln(sum_exp)) = ln(sum_exp) + max - logit
    Ok(sum_exp.ln() + max - target_logit)
}

/// `arle train cc-convert` — backend-independent (no CUDA): dumps → verl-style
/// token records via `train::cc_convert`.
fn run_cc_convert_impl(args: TrainCcConvertArgs) -> Result<()> {
    use train::cc_convert::CcWindow;

    // Windows from --windows (JSONL) plus repeated --window START:END[:LABEL].
    let mut windows: Vec<CcWindow> = Vec::new();
    if let Some(path) = args.windows.as_deref() {
        let raw = fs::read_to_string(path)
            .with_context(|| format!("read --windows {}", path.display()))?;
        for line in raw.lines().filter(|line| !line.trim().is_empty()) {
            windows.push(
                serde_json::from_str(line)
                    .with_context(|| format!("parse --windows row: {line}"))?,
            );
        }
    }
    for (idx, spec) in args.window.iter().enumerate() {
        let mut parts = spec.splitn(3, ':');
        let (Some(start), Some(end)) = (parts.next(), parts.next()) else {
            bail!("--window {spec}: expected <t_start_ms>:<t_end_ms>[:<label>]");
        };
        windows.push(CcWindow {
            t_start_ms: start
                .parse()
                .with_context(|| format!("--window {spec}: bad t_start_ms"))?,
            t_end_ms: end
                .parse()
                .with_context(|| format!("--window {spec}: bad t_end_ms"))?,
            label: parts
                .next()
                .map_or_else(|| format!("w{idx}"), str::to_owned),
            // Manual --window has no attempt reward → passing default (CE flow).
            reward: 1.0,
            errored: false,
            model: None,
        });
    }

    let records =
        train::cc_convert::run_cc_convert(&args.dump_dir, &args.tokenizer, &args.out, &windows)?;
    for record in &records {
        eprintln!(
            "[arle train cc-convert] {}: prompt={} response={} masked={}/{} tokens",
            record.label,
            record.prompt_ids.len(),
            record.response_ids.len(),
            record.masked_tokens,
            record.total_tokens
        );
    }
    eprintln!(
        "[arle train cc-convert] wrote {} record(s) to {}",
        records.len(),
        args.out.display()
    );
    Ok(())
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
pub(crate) fn run_model(model: ModelArgs) -> ExitCode {
    match model.command {
        ModelCommand::Download(args) => run_model_download(args),
    }
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "cpu"))]
fn run_model_download(args: ModelDownloadArgs) -> ExitCode {
    let source_label = match args.source {
        ModelSourceArg::Hf => "hf",
        ModelSourceArg::Modelscope => "modelscope",
    };
    if args.render.dry_run {
        if args.render.json {
            println!(
                "{}",
                serde_json::json!({
                    "command": "model download",
                    "argv": [args.model_id],
                    "source": source_label,
                })
            );
        } else {
            println!("command model download");
            println!("argv {}", args.model_id);
            println!("source {source_label}");
        }
        return ExitCode::SUCCESS;
    }
    let result = match args.source {
        ModelSourceArg::Hf => crate::download::download_model_with_progress(&args.model_id),
        ModelSourceArg::Modelscope => {
            crate::modelscope::download_model_from_modelscope_with_progress(&args.model_id)
        }
    };
    match result {
        Ok(path) => {
            eprintln!(
                "[ARLE model download] downloaded ({source_label}) to: {}",
                path.display()
            );
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("[ARLE model download] error: {err:#}");
            ExitCode::FAILURE
        }
    }
}

fn run_train_env(args: TrainEnvArgs) -> Result<()> {
    let info = hardware::detect_system();
    let report = TrainEnvReport {
        version: env!("CARGO_PKG_VERSION"),
        train_default_backend: default_train_backend(),
        compiled_infer_backend: info.compiled_backend.name(),
        supports_inference: info.compiled_backend.supports_inference(),
        cpu: info.cpu_name,
        cpu_cores: info.cpu_cores,
        total_ram_gb: info.total_ram_gb,
        available_ram_gb: info.available_ram_gb,
        gpu: gpu_label(&info.gpu),
        hf_cache_root: hub_discovery::hub_cache_root()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "<unavailable>".to_string()),
        cwd: std::env::current_dir()
            .ok()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "<unknown>".to_string()),
        commands: TRAIN_ENV_COMMANDS,
    };
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    println!("ARLE train env");
    println!("version {}", report.version);
    println!("train default backend {}", report.train_default_backend);
    println!("compiled infer backend {}", report.compiled_infer_backend);
    println!("cpu {} · {} cores", report.cpu, report.cpu_cores);
    println!(
        "ram {:.1} GB total · {:.1} GB free",
        report.total_ram_gb, report.available_ram_gb
    );
    println!("gpu {}", report.gpu);
    println!("hf cache {}", report.hf_cache_root);
    println!("cwd {}", report.cwd);
    println!("commands {}", report.commands.join(", "));
    Ok(())
}

fn run_train_estimate_memory(args: TrainEstimateMemoryArgs) -> Result<()> {
    let report = if let Some(model_source) = args.model.as_deref() {
        estimate_from_model_dir(model_source, &args)?
    } else {
        estimate_from_scratch(&args)?
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    println!("ARLE train estimate-memory");
    println!("mode {}", report.mode);
    println!("family {}", report.family);
    if let Some(model_dir) = &report.model_dir {
        println!("model {}", model_dir);
    }
    if let Some(tokenizer_path) = &report.tokenizer_path {
        println!("tokenizer {}", tokenizer_path);
    }
    println!("params {}", format_count(report.param_count));
    println!(
        "trainable params {}",
        format_count(report.trainable_param_count)
    );
    println!("weights fp32 {}", format_bytes(report.weight_bytes_fp32));
    println!("grads fp32 {}", format_bytes(report.gradient_bytes_fp32));
    println!(
        "adam states fp32 {}",
        format_bytes(report.adam_state_bytes_fp32)
    );
    println!(
        "checkpoint {} {}",
        report.save_dtype,
        format_bytes(report.checkpoint_bytes)
    );
    if let Some(adapter_bytes) = report.adapter_checkpoint_bytes {
        println!("adapter checkpoint {}", format_bytes(adapter_bytes));
    }
    println!(
        "activation floor (batch={} seq={}) {}",
        report.batch,
        report.seq,
        format_bytes(report.activation_floor_bytes)
    );
    if let Some(vocab_size) = report.vocab_size {
        println!("vocab {}", vocab_size);
    }
    Ok(())
}

fn run_opd(args: TrainOpdArgs) -> ExitCode {
    if args.smoke {
        return exit_from_result(run_opd_smoke(args));
    }
    if args.student_model.is_some() {
        return exit_from_result(run_opd_from_dirs(args));
    }
    eprintln!(
        "[arle train opd] error: either `--student-model <dir>` or `--smoke` is required.\n\
         See `arle train opd --help` for the full surface."
    );
    ExitCode::FAILURE
}

/// `arle train w2s` — weak-to-strong online distillation.
///
/// Loads the student (LoRA), two auxiliary models (each with pre-RL + post-RL
/// checkpoints), and runs the w2s step: ΔT → proxy teacher → reverse KL +
/// local/global KL regularization.
fn run_w2s(args: TrainW2sArgs) -> Result<()> {
    use crate::args::W2sAuxBackendArg;
    use autograd::{Tape, TensorId, optim::AdamW};
    use train::{
        lora::LoraConfig,
        qwen35::Qwen35Model,
        qwen35_loader::load_qwen35_from_hf_dir,
        w2s::{W2sAuxModel, W2sConfig, sync_lora_adapters, w2s_step},
    };

    let target_set = parse_lora_target_set(&args.lora_target_set)?;
    let lora = LoraConfig {
        rank: args.lora_rank,
        alpha: args.lora_alpha,
    };

    let (mut store, train_backend, backend_label) = build_opd_store(args.backend)?;
    #[cfg(not(feature = "cuda"))]
    let _ = train_backend;
    let mut tape = Tape::new();

    eprintln!("[arle train w2s] backend={backend_label}");

    fn vram(store: &TensorStore, phase: &str) {
        if let Some((free, total)) = store.backend().device_mem_info() {
            let gb = |b: usize| b as f64 / (1u64 << 30) as f64;
            eprintln!(
                "[arle train w2s] vram after {phase}: used={:.1} GB free={:.1} GB total={:.1} GB",
                gb(total - free),
                gb(free),
                gb(total)
            );
        }
    }

    macro_rules! load_model {
        ($label:expr, $path:expr, $store:expr) => {{
            eprintln!(
                "[arle train w2s] loading {} from {}",
                $label,
                $path.display()
            );
            load_qwen35_from_hf_dir($path, $store)
                .with_context(|| format!("load {} from {}", $label, $path.display()))?
        }};
    }

    // π_base for the global KL regularizer.
    let base = load_model!("base", &args.student_model, &mut store);
    #[cfg(feature = "cuda")]
    let vocab_size = base.config().vocab_size;

    // Student (shadow adapter) — the model being trained.
    eprintln!("[arle train w2s] creating student (shadow adapter)");
    let student = Qwen35Model::new_lora_from_base(&base, lora, target_set, &mut store)
        .context("create student shadow adapter")?;
    vram(&store, "base+student load");

    // Serving adapter — π_old for the local KL regularizer. Created from the
    // student (not the base) so `new_lora_from_base`'s retain_ids keeps the
    // student's adapter params alive alongside the serving adapter's.
    eprintln!("[arle train w2s] creating serving adapter");
    let serving = Qwen35Model::new_lora_from_base(&student, lora, target_set, &mut store)
        .context("create serving adapter")?;

    // Initialize serving = shadow (both start from the same LoRA init).
    sync_lora_adapters(&student, &serving, &mut store)?;

    let aux1 = match args.aux_backend {
        W2sAuxBackendArg::InProcess => {
            // Omitted --aux1-pre reuses the base: ΔT = post_RL − base measures
            // how far the teacher sits above the student's starting point.
            let aux1_pre = match &args.aux1_pre {
                Some(path) => load_model!("aux1 pre-RL", path, &mut store),
                None => {
                    eprintln!("[arle train w2s] aux1 pre-RL = student base model (reused)");
                    base.clone()
                }
            };
            let aux1_post = load_model!("aux1 post-RL", &args.aux1_post, &mut store);
            W2sAuxModel::new_in_process(aux1_pre, aux1_post)
        }
        #[cfg(feature = "cuda")]
        W2sAuxBackendArg::Infer => {
            use infer_api::{EngineLoadConfig, LoadedInferenceEngine};
            use std::sync::{Arc, Mutex};
            use train::teacher_infer::{InferTeacher, TeacherForward};

            // Aux engines run forward passes over the full prompt+completion
            // (chain-of-thought KL), so the sequence budget must cover GSM8K
            // solutions (~500 tokens). `mem_fraction_static` keeps the KV pool
            // small so the student's activations + aux weights fit on one GPU.
            let aux_cfg = EngineLoadConfig {
                mem_fraction_static: 0.1,
                ..EngineLoadConfig::single_sequence(2048)
            };
            // Each engine offloads right after load so only one aux is resident;
            // the w2s step reloads it just-in-time for the delta forward.
            let load_teacher = |label: &str, path: &std::path::Path| -> Result<InferTeacher> {
                eprintln!(
                    "[arle train w2s] loading aux1 {label} (infer) from {}",
                    path.display()
                );
                let engine = LoadedInferenceEngine::load_with_config(
                    path.to_str()
                        .ok_or_else(|| anyhow!("aux1 {} path not UTF-8", label))?,
                    true,
                    aux_cfg.clone(),
                )
                .with_context(|| format!("load aux1 {label} infer from {}", path.display()))?;
                let teacher = InferTeacher::new(
                    Arc::new(Mutex::new(engine)),
                    train_backend.clone(),
                    vocab_size,
                );
                let freed = teacher
                    .offload_engine_weights()
                    .with_context(|| format!("offload aux1 {label} engine weights"))?;
                eprintln!("[arle train w2s] aux1 {label} offload freed {freed} bytes");
                Ok(teacher)
            };
            let pre_teacher = load_teacher(
                "pre-RL",
                args.aux1_pre.as_ref().unwrap_or(&args.student_model),
            )?;
            let post_teacher = load_teacher("post-RL", &args.aux1_post)?;
            W2sAuxModel::new_infer(pre_teacher, post_teacher)
        }
        #[cfg(not(feature = "cuda"))]
        W2sAuxBackendArg::Infer => {
            bail!("--aux-backend infer requires the cuda feature");
        }
    };

    let aux2 = match args.aux_backend {
        W2sAuxBackendArg::InProcess => {
            match (&args.aux2_pre, &args.aux2_post) {
                (Some(pre_path), Some(post_path)) => {
                    let aux2_pre = load_model!("aux2 pre-RL", pre_path, &mut store);
                    let aux2_post = load_model!("aux2 post-RL", post_path, &mut store);
                    (W2sAuxModel::new_in_process(aux2_pre, aux2_post), false)
                }
                _ => {
                    // Sharing aux1 makes the consistency gate a no-op; the rest
                    // of the pipeline still runs.
                    eprintln!("[arle train w2s] no aux2 checkpoints — sharing aux1");
                    (aux1.clone(), true)
                }
            }
        }
        #[cfg(feature = "cuda")]
        W2sAuxBackendArg::Infer => {
            // Independent aux2 engines are not wired for the infer backend.
            eprintln!("[arle train w2s] sharing aux1 engines for aux2");
            (aux1.clone(), true)
        }
        #[cfg(not(feature = "cuda"))]
        W2sAuxBackendArg::Infer => {
            bail!("--aux-backend infer requires the cuda feature");
        }
    };
    let (aux2, share_aux) = aux2;

    vram(&store, "aux load");

    let trainable_params: Vec<TensorId> = student
        .all_parameter_ids()
        .into_iter()
        .filter(|id| store.get(*id).is_some_and(|t| t.requires_grad))
        .collect();
    eprintln!(
        "[arle train w2s] trainable params: {}",
        trainable_params.len()
    );

    let cfg = W2sConfig {
        alpha: args.alpha,
        temperature: args.temperature,
        confidence_threshold: args.confidence_threshold,
        consistency_threshold: args.consistency_threshold,
        beta_local: args.beta_local,
        beta_global: args.beta_global,
        grad_clip: args.grad_clip,
    };

    // Build the list of training prompts. If `--train-data` is set, load the
    // JSONL and tokenize each row; otherwise fall back to the single
    // `--prompt-ids` sequence (or the default BOS smoke-test prompt).
    let prompts: Vec<Vec<u32>> = match &args.train_data {
        Some(path) => {
            use std::io::{BufRead, BufReader};
            let file = std::fs::File::open(path)
                .with_context(|| format!("open train data {}", path.display()))?;
            let reader = BufReader::new(file);
            let mut out = Vec::new();
            for (i, line) in reader.lines().enumerate() {
                let line = line.with_context(|| format!("read train data line {i}"))?;
                if line.trim().is_empty() {
                    continue;
                }
                let row: serde_json::Value = serde_json::from_str(&line)
                    .with_context(|| format!("parse train data line {i}"))?;
                if let Some(ids) = row.get("prompt_ids").and_then(|v| v.as_array()) {
                    let ids: Vec<u32> = ids
                        .iter()
                        .filter_map(|v| v.as_u64().map(|n| n as u32))
                        .collect();
                    if !ids.is_empty() {
                        out.push(ids);
                    }
                } else if let Some(text) = row.get("text").and_then(|v| v.as_str()) {
                    // Tokenize with the student's tokenizer. If a `completion`
                    // field is present, append it so the KL covers the full
                    // chain-of-thought, not just the first answer token.
                    let mut full_text = text.to_string();
                    if let Some(completion) = row.get("completion").and_then(|v| v.as_str()) {
                        full_text.push_str(completion);
                    }
                    let ids = tokenize_text(&full_text, &args.student_model)?;
                    if !ids.is_empty() {
                        out.push(ids);
                    }
                }
            }
            if out.is_empty() {
                bail!("no valid prompts found in {}", path.display());
            }
            eprintln!(
                "[arle train w2s] loaded {} training prompts from {}",
                out.len(),
                path.display()
            );
            out
        }
        None => {
            let ids = match &args.prompt_ids {
                Some(s) => s
                    .split(',')
                    .map(|t| t.trim().parse::<u32>().context("parse prompt token id"))
                    .collect::<Result<Vec<_>>>()?,
                None => vec![1, 3, 8],
            };
            vec![ids]
        }
    };

    let mut optimizer = AdamW::new(args.lr, (0.9, 0.999), 1e-8, 0.0);

    for step in 0..args.steps {
        let prompt_ids = &prompts[step % prompts.len()];
        let outcome = w2s_step(
            &student,
            &serving,
            &base,
            &aux1,
            &aux2,
            prompt_ids,
            &cfg,
            &trainable_params,
            &mut optimizer,
            &mut store,
            &mut tape,
            share_aux,
        )?;

        if outcome.skipped {
            eprintln!(
                "[arle train w2s] step={step} skipped reason={:?} max_prob={:.4} consistency={:.4}",
                outcome.skip_reason, outcome.max_prob, outcome.consistency
            );
        } else {
            eprintln!(
                "[arle train w2s] step={step} loss={:.6} max_prob={:.4} consistency={:.4}",
                outcome.loss, outcome.max_prob, outcome.consistency
            );
            // Shadow → serving: the local KL regularizer now anchors against
            // the just-updated adapter. In the full online flow this only
            // happens after a validation-set eval passes; here we sync every
            // step so the local KL tracks the latest shadow state.
            sync_lora_adapters(&student, &serving, &mut store)?;
        }
    }

    // Save the trained LoRA adapter if requested.
    if let Some(adapter_dir) = &args.save_adapter {
        save_w2s_adapter(
            &student,
            &mut store,
            adapter_dir,
            &args.student_model,
            &lora,
            target_set,
        )?;
        eprintln!(
            "[arle train w2s] adapter saved to {}",
            adapter_dir.display()
        );
    }

    Ok(())
}

/// Tokenize a text string using the model's tokenizer.json.
fn tokenize_text(text: &str, model_dir: &std::path::Path) -> Result<Vec<u32>> {
    use tokenizers::Tokenizer;
    let tok_path = model_dir.join("tokenizer.json");
    let tok = Tokenizer::from_file(&tok_path)
        .map_err(|e| anyhow!("load tokenizer from {}: {e}", tok_path.display()))?;
    let ids = tok
        .encode(text, false)
        .map_err(|e| anyhow!("tokenize: {e}"))?;
    Ok(ids.get_ids().to_vec())
}

/// Save the student's LoRA adapter as a PEFT-style directory (adapter_model.safetensors
/// + adapter_config.json + tokenizer/config copies), loadable by `arle serve --lora`.
fn save_w2s_adapter(
    student: &train::qwen35::Qwen35Model,
    store: &mut TensorStore,
    adapter_dir: &std::path::Path,
    student_model: &std::path::Path,
    lora: &train::lora::LoraConfig,
    target_set: train::lora::LoraTargetSet,
) -> Result<()> {
    use train::lora::{LoraAdapterConfig, LoraTargetSet};
    use train::qwen35_checkpoint::{
        ConfigJsonSource, GenerationConfigSource, Qwen35NamedCheckpoint, Qwen35StudentWeights,
        save_named_qwen35_student_checkpoint,
    };

    // The adapter dir may have been created (empty) by an earlier setup step;
    // the checkpoint writer refuses to merge into an existing directory.
    if adapter_dir.exists() {
        fs::remove_dir_all(adapter_dir)
            .with_context(|| format!("remove existing adapter dir {}", adapter_dir.display()))?;
    }

    let sources = checkpoint_sources(student_model)?;
    let mut adapter_config =
        LoraAdapterConfig::new(student_model.display().to_string(), "qwen35", *lora);
    adapter_config.target_modules = match target_set {
        LoraTargetSet::AttentionQv => vec!["q_proj".to_owned(), "v_proj".to_owned()],
        LoraTargetSet::AttentionFull => vec![
            "q_proj".to_owned(),
            "k_proj".to_owned(),
            "v_proj".to_owned(),
            "o_proj".to_owned(),
            "in_proj_qkv".to_owned(),
            "out_proj".to_owned(),
        ],
        LoraTargetSet::AllLinear => vec!["all-linear".to_owned()],
    };
    let mut adapter_tape = Tape::new();
    save_named_qwen35_student_checkpoint(
        Qwen35NamedCheckpoint {
            out_dir: adapter_dir.parent().unwrap_or(std::path::Path::new(".")),
            dirname: adapter_dir
                .file_name()
                .unwrap_or_default()
                .to_str()
                .unwrap_or("adapter"),
            tokenizer_path: Some(&sources.tokenizer_path),
            config_json: ConfigJsonSource::CopyFrom(&sources.config_path),
            generation_config: GenerationConfigSource::CopyOrSynthesize {
                source_path: &sources.generation_config_path,
                fallback_config_path: &sources.config_path,
            },
        },
        student,
        store,
        &mut adapter_tape,
        Qwen35StudentWeights::AdapterOnly {
            bf16: true,
            adapter_config: &adapter_config,
        },
    )
    .with_context(|| format!("save w2s adapter to {}", adapter_dir.display()))?;
    Ok(())
}

/// Reject advertised-but-unimplemented distillation objectives before any
/// model/store is loaded, so a no-op flag fails fast instead of silently
/// running a different objective mid-training.
fn reject_unimplemented_gkd_objectives(
    gkd_entropy_weight: f32,
    teacher_topk: Option<usize>,
) -> Result<()> {
    if gkd_entropy_weight != 0.0 {
        bail!(
            "--gkd-entropy-weight {gkd_entropy_weight} is not implemented: the per-position \
             entropy-weighted objective (AEPO) does not exist yet. Omit the flag (0.0) to run \
             unweighted KL."
        );
    }
    if teacher_topk.is_some() {
        bail!(
            "--teacher-topk requires an engine-side top-k teacher-logprob producer (H20 Piece A) \
             that is not wired. Omit the flag to run dense KL."
        );
    }
    Ok(())
}

fn validate_train_opd_gkd_args(gkd_lambda: f32, sft_anchor: OpdSftAnchorArg) -> Result<()> {
    if !(0.0..=1.0).contains(&gkd_lambda) || !gkd_lambda.is_finite() {
        bail!("--gkd-lambda must be finite and in [0.0, 1.0], got {gkd_lambda}");
    }
    match sft_anchor {
        OpdSftAnchorArg::StudentRollout => {
            if gkd_lambda == 0.0 {
                Ok(())
            } else {
                bail!(
                    "`arle train opd` keeps the student-rollout anchor pure-KL; \
                     --gkd-lambda must be 0.0 unless --sft-anchor corpus-truth is used, \
                     got {gkd_lambda}"
                )
            }
        }
        OpdSftAnchorArg::CorpusTruth => {
            if gkd_lambda > 0.0 {
                Ok(())
            } else {
                bail!(
                    "--sft-anchor corpus-truth requires --gkd-lambda > 0.0; \
                     use --gkd-lambda 1.0 for off-policy SFT on teacher completions"
                )
            }
        }
    }
}

fn opd_sft_anchor_arg(arg: OpdSftAnchorArg) -> train::opd::GkdSftAnchor {
    match arg {
        OpdSftAnchorArg::StudentRollout => train::opd::GkdSftAnchor::StudentRollout,
        OpdSftAnchorArg::CorpusTruth => train::opd::GkdSftAnchor::CorpusTruth,
    }
}

struct OpdCheckpointSources {
    config_path: PathBuf,
    tokenizer_path: PathBuf,
    generation_config_path: PathBuf,
}

fn checkpoint_sources(model_dir: &Path) -> Result<OpdCheckpointSources> {
    let config_path = model_dir.join("config.json");
    if !config_path.is_file() {
        bail!(
            "cannot save OPD checkpoint: source config.json is missing at {}",
            config_path.display()
        );
    }
    let tokenizer_path = model_dir.join("tokenizer.json");
    if !tokenizer_path.is_file() {
        bail!(
            "cannot save OPD checkpoint: source tokenizer.json is missing at {}",
            tokenizer_path.display()
        );
    }
    Ok(OpdCheckpointSources {
        config_path,
        tokenizer_path,
        generation_config_path: model_dir.join("generation_config.json"),
    })
}

fn should_save_step_checkpoint(step: usize, total_steps: usize, save_every: usize) -> bool {
    step == total_steps || (save_every > 0 && step.is_multiple_of(save_every))
}

fn maybe_save_full_student_checkpoint(
    label: &str,
    save_checkpoint: Option<&Path>,
    save_every: usize,
    step: usize,
    total_steps: usize,
    model_dir: &Path,
    student: &train::qwen35::Qwen35Model,
    store: &mut TensorStore,
    tape: &mut autograd::Tape,
) -> Result<Option<PathBuf>> {
    let Some(out_dir) = save_checkpoint else {
        return Ok(None);
    };
    if !should_save_step_checkpoint(step, total_steps, save_every) {
        return Ok(None);
    }

    use train::qwen35_checkpoint::{
        ConfigJsonSource, GenerationConfigSource, Qwen35StepCheckpoint, Qwen35StudentWeights,
        save_qwen35_student_checkpoint,
    };

    fs::create_dir_all(out_dir)
        .with_context(|| format!("create OPD checkpoint root {}", out_dir.display()))?;
    let started = Instant::now();
    let sources = checkpoint_sources(model_dir)?;
    let saved_dir = save_qwen35_student_checkpoint(
        Qwen35StepCheckpoint {
            out_dir,
            step,
            tokenizer_path: Some(&sources.tokenizer_path),
            config_json: ConfigJsonSource::CopyFrom(&sources.config_path),
            generation_config: GenerationConfigSource::CopyOrSynthesize {
                source_path: &sources.generation_config_path,
                fallback_config_path: &sources.config_path,
            },
        },
        student,
        store,
        tape,
        Qwen35StudentWeights::FullMaterialized { bf16: true },
    )
    .with_context(|| format!("save {label} full-materialized checkpoint at step {step}"))?;
    println!(
        "checkpoint_saved kind=full_materialized mode={label} step={step} dir={} seconds={:.6}",
        saved_dir.display(),
        started.elapsed().as_secs_f64()
    );
    Ok(Some(saved_dir))
}

struct OpdPromptSource {
    train_prompts: Vec<Vec<u32>>,
    train_completions: Vec<Option<Vec<u32>>>,
    eval_ids: Vec<u32>,
    report_prompt_ids: Vec<u32>,
    completion_rows: usize,
}

#[derive(Debug, Clone)]
struct PromptSampler {
    state: u64,
}

impl PromptSampler {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9e37_79b9_7f4a_7c15,
        }
    }

    fn next_index(&mut self, len: usize) -> usize {
        debug_assert!(len > 0);
        if len <= 1 {
            return 0;
        }
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.state >> 32) as usize) % len
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    fn next_unit(&mut self) -> f64 {
        self.next_index(1 << 24) as f64 / (1 << 24) as f64
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct TaskStats {
    ema_pass: Option<f32>,
    hot_rounds: u32,
    retired: bool,
}

/// Pass-rate task selection: variance-weighted sampling — concentrate rollout
/// on the p≈0.5 max-variance band where reward-bearing R(p,k)=1−p^k−(1−p)^k peaks
/// — plus EMA-pass retirement of mastered tasks, with a 0.1 exploration floor.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
struct TaskSelection {
    rng: PromptSampler,
    stats: Vec<TaskStats>,
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
impl TaskSelection {
    fn new(n_tasks: usize) -> Self {
        Self {
            // Fixed seed → runs are reproducible.
            rng: PromptSampler::new(0x5EED),
            stats: vec![TaskStats::default(); n_tasks],
        }
    }

    /// Predictive keep-probability from the online pass-estimate: run a task in
    /// proportion to its reward-bearing variance v = p(1−p) (∈[0,0.25], max at
    /// p=0.5), normalized so p≈0.5 always runs and a 0.1 floor keeps the tails
    /// explored (a skipped task's EMA can only refresh on a round it runs).
    fn keep_prob(ema_pass: Option<f32>) -> f64 {
        let Some(p) = ema_pass else {
            return 1.0; // unseen → run: need an estimate before we can weight
        };
        let v = f64::from(p) * f64::from(1.0 - p);
        (v / 0.25).max(0.1)
    }

    /// Update from a completed group; skipped rounds freeze the estimate.
    fn record(&mut self, task: usize, pass_rate: f32) {
        let s = &mut self.stats[task];
        let ema = s.ema_pass.map_or(pass_rate, |e| 0.3 * pass_rate + 0.7 * e);
        s.ema_pass = Some(ema);
        s.hot_rounds = if ema >= 0.9 { s.hot_rounds + 1 } else { 0 };
        s.retired = s.retired || s.hot_rounds >= 3;
    }

    /// A mastered task (EMA pass ≥ 0.9 for 3 rounds) — never a DAPO refill target.
    fn is_retired(&self, task: usize) -> bool {
        self.stats[task].retired
    }

    /// Task indices to run this round + (skipped, retired) counts.
    /// Round 0 runs ALL tasks: no history yet, and the baseline needs it.
    fn select(&mut self, round: usize) -> (Vec<usize>, usize, usize) {
        let (mut run, mut skipped, mut retired) = (Vec::new(), 0, 0);
        for i in 0..self.stats.len() {
            let s = self.stats[i];
            if s.retired {
                retired += 1;
            } else if round > 0 && self.rng.next_unit() >= Self::keep_prob(s.ema_pass) {
                skipped += 1;
            } else {
                run.push(i);
            }
        }
        (run, skipped, retired)
    }
}

/// Experience replay: age-bounded, |A|-prioritized buffer of trained groups.
/// Entries retain the generation-time behavior sidecars, so every reuse stays
/// corrected against the policy that sampled the trajectories.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
#[derive(Clone)]
struct ReplayEntry {
    batch: Vec<train::update_strategy::ScoredTrajectory>,
    task_id: String,
    round: usize,
    priority: f32,
}

/// Survey-grounded staleness bound: evict entries older than 10 rounds.
const REPLAY_MAX_AGE: usize = 10;

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
#[derive(Default)]
struct ReplayBuffer {
    entries: Vec<ReplayEntry>,
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
impl ReplayBuffer {
    /// Priority = mean |reward − group mean|; 0 ⇔ zero-variance (nothing to learn).
    fn priority(rewards: &[f32]) -> f32 {
        if rewards.is_empty() {
            return 0.0;
        }
        let mean = rewards.iter().sum::<f32>() / rewards.len() as f32;
        rewards.iter().map(|r| (r - mean).abs()).sum::<f32>() / rewards.len() as f32
    }

    /// Zero-variance groups never enter the buffer.
    fn push(
        &mut self,
        round: usize,
        task_id: String,
        batch: Vec<train::update_strategy::ScoredTrajectory>,
    ) {
        let rewards: Vec<f32> = batch.iter().map(|t| t.reward).collect();
        let priority = Self::priority(&rewards);
        if priority > 0.0 {
            self.entries.push(ReplayEntry {
                batch,
                task_id,
                round,
                priority,
            });
        }
    }

    /// Evict age > [`REPLAY_MAX_AGE`], then draw up to `n` entries without
    /// replacement, P(entry) ∝ priority — deterministic given the fixed-seed rng.
    fn draw(&mut self, round: usize, n: usize, rng: &mut PromptSampler) -> Vec<ReplayEntry> {
        self.entries.retain(|e| round - e.round <= REPLAY_MAX_AGE);
        let mut pool: Vec<usize> = (0..self.entries.len()).collect();
        let mut out = Vec::new();
        while out.len() < n && !pool.is_empty() {
            let total: f64 = pool
                .iter()
                .map(|&i| f64::from(self.entries[i].priority))
                .sum();
            let mut x = rng.next_unit() * total;
            let pick = pool
                .iter()
                .position(|&i| {
                    x -= f64::from(self.entries[i].priority);
                    x <= 0.0
                })
                .unwrap_or(pool.len() - 1);
            out.push(self.entries[pool.remove(pick)].clone());
        }
        out
    }
}

#[derive(Debug, Clone, Copy)]
struct OpdLrSchedule {
    mode: LrScheduleArg,
    base_lr: f32,
    min_lr: f32,
    warmup_steps: u64,
    total_steps: u64,
}

impl OpdLrSchedule {
    fn new(
        mode: LrScheduleArg,
        base_lr: f32,
        warmup_steps: Option<usize>,
        total_steps: usize,
    ) -> Result<Self> {
        if mode == LrScheduleArg::Cosine && (!base_lr.is_finite() || base_lr < 0.0) {
            bail!("--lr-schedule cosine requires a finite non-negative --lr, got {base_lr}");
        }
        let warmup_steps = warmup_steps.unwrap_or_else(|| default_cosine_warmup_steps(total_steps));
        Ok(Self {
            mode,
            base_lr,
            min_lr: base_lr * 0.1,
            warmup_steps: warmup_steps as u64,
            total_steps: total_steps as u64,
        })
    }

    fn lr_at_step(self, step: u64) -> f32 {
        match self.mode {
            LrScheduleArg::Fixed => self.base_lr,
            LrScheduleArg::Cosine => self.cosine_lr_at_step(step),
        }
    }

    fn apply_to_optimizer(self, optimizer: &mut autograd::AdamW, step: u64) -> f32 {
        let lr = self.lr_at_step(step);
        if self.mode == LrScheduleArg::Cosine {
            autograd::Optimizer::set_lr(optimizer, lr);
        }
        lr
    }

    fn cosine_lr_at_step(self, step: u64) -> f32 {
        if self.warmup_steps > 0 && step < self.warmup_steps {
            return self.base_lr * (step as f32 / self.warmup_steps as f32);
        }
        if self.total_steps <= self.warmup_steps {
            return self.base_lr;
        }
        if self.total_steps > 0 && step >= self.total_steps - 1 {
            return self.min_lr;
        }
        let decay_span = self.total_steps - self.warmup_steps - 1;
        if decay_span == 0 {
            return self.min_lr;
        }
        let progress = (step - self.warmup_steps) as f32 / decay_span as f32;
        let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
        self.min_lr + (self.base_lr - self.min_lr) * cosine
    }
}

fn default_cosine_warmup_steps(total_steps: usize) -> usize {
    if total_steps == 0 {
        0
    } else {
        total_steps.saturating_mul(3).div_ceil(100).max(1)
    }
}

fn validate_token_ids(label: &str, ids: &[u32], vocab_size: usize) -> Result<()> {
    if ids.is_empty() {
        bail!("{label} must contain at least one token id");
    }
    if ids.iter().any(|&id| (id as usize) >= vocab_size) {
        bail!("{label} token ids must be < {vocab_size} (student vocab size); got {ids:?}");
    }
    Ok(())
}

fn validate_prompt_collection(label: &str, prompts: &[Vec<u32>], vocab_size: usize) -> Result<()> {
    if prompts.is_empty() {
        bail!("{label} must contain at least one prompt");
    }
    for (idx, prompt) in prompts.iter().enumerate() {
        validate_token_ids(&format!("{label}[{idx}]"), prompt, vocab_size)?;
    }
    Ok(())
}

fn validate_completion_collection(
    label: &str,
    completions: &[Option<Vec<u32>>],
    prompts_len: usize,
    vocab_size: usize,
) -> Result<()> {
    if completions.len() != prompts_len {
        bail!(
            "{label} length {} must match prompt count {prompts_len}",
            completions.len()
        );
    }
    for (idx, completion) in completions.iter().enumerate() {
        if let Some(tokens) = completion {
            validate_token_ids(&format!("{label}[{idx}]"), tokens, vocab_size)?;
        }
    }
    Ok(())
}

fn load_opd_prompt_source(
    args: &TrainOpdArgs,
    student_dir: &Path,
    vocab_size: usize,
) -> Result<OpdPromptSource> {
    if let Some(prompt_file) = args.prompts_file.as_deref() {
        let loaded = train::prompts::load_jsonl_prompt_sets(
            student_dir,
            prompt_file,
            args.prompt_max_tokens,
            1,
        )
        .with_context(|| format!("load OPD prompt corpus from {}", prompt_file.display()))?;
        validate_prompt_collection("--prompts-file train prompt", &loaded.train, vocab_size)?;
        validate_prompt_collection("--prompts-file heldout prompt", &loaded.heldout, vocab_size)?;
        validate_completion_collection(
            "--prompts-file train completion",
            &loaded.train_completions,
            loaded.train.len(),
            vocab_size,
        )?;
        validate_completion_collection(
            "--prompts-file heldout completion",
            &loaded.heldout_completions,
            loaded.heldout.len(),
            vocab_size,
        )?;
        let eval_ids = match args.eval_ids.as_deref() {
            Some(raw) => {
                let ids = parse_prompt_ids(Some(raw))?;
                validate_token_ids("--eval-ids", &ids, vocab_size)?;
                ids
            }
            None => loaded
                .heldout
                .first()
                .cloned()
                .unwrap_or_else(|| loaded.train[0].clone()),
        };
        eprintln!(
            "[arle train opd] loaded prompt corpus {}: train={} heldout={} rows={} truncated={} completion_rows={} truncated_completion_rows={}",
            prompt_file.display(),
            loaded.train.len(),
            loaded.heldout.len(),
            loaded.jsonl_rows,
            loaded.truncated_rows,
            loaded.completion_rows,
            loaded.truncated_completion_rows,
        );
        let report_prompt_ids = loaded.train[0].clone();
        return Ok(OpdPromptSource {
            train_prompts: loaded.train,
            train_completions: loaded.train_completions,
            eval_ids,
            report_prompt_ids,
            completion_rows: loaded.completion_rows,
        });
    }

    let prompt_ids = parse_prompt_ids(args.prompt_ids.as_deref())?;
    validate_token_ids("--prompt-ids", &prompt_ids, vocab_size)?;
    let eval_ids = match args.eval_ids.as_deref() {
        Some(raw) => {
            let ids = parse_prompt_ids(Some(raw))?;
            validate_token_ids("--eval-ids", &ids, vocab_size)?;
            ids
        }
        None => prompt_ids.clone(),
    };
    Ok(OpdPromptSource {
        train_prompts: vec![prompt_ids.clone()],
        train_completions: vec![None],
        eval_ids,
        report_prompt_ids: prompt_ids,
        completion_rows: 0,
    })
}

enum OpdCliTeacher<'a> {
    InProcess(train::teacher_infer::InProcessTeacher<'a>),
    #[cfg(feature = "cuda")]
    Infer(train::teacher_infer::InferTeacher),
    Api(train::teacher_infer::ApiTeacher),
    CorpusSftOnly {
        vocab_size: usize,
    },
}

impl train::teacher_infer::TeacherForward for OpdCliTeacher<'_> {
    fn forward_logits_device(
        &self,
        input_ids: &[u32],
        positions: &[u32],
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> std::result::Result<
        train::teacher_infer::DeviceLogits,
        train::teacher_infer::TeacherForwardError,
    > {
        match self {
            Self::InProcess(teacher) => {
                train::teacher_infer::TeacherForward::forward_logits_device(
                    teacher, input_ids, positions, store, tape,
                )
            }
            #[cfg(feature = "cuda")]
            Self::Infer(teacher) => train::teacher_infer::TeacherForward::forward_logits_device(
                teacher, input_ids, positions, store, tape,
            ),
            Self::Api(teacher) => train::teacher_infer::TeacherForward::forward_logits_device(
                teacher, input_ids, positions, store, tape,
            ),
            Self::CorpusSftOnly { .. } => {
                Err(train::teacher_infer::TeacherForwardError::InvalidInput(
                    "corpus-truth SFT-only teacher was asked to score KL logits; \
                     this path is only valid with --sft-anchor corpus-truth --gkd-lambda 1.0"
                        .to_owned(),
                ))
            }
        }
    }

    fn forward_logits_window_device(
        &self,
        input_ids: &[u32],
        positions: &[u32],
        window: train::qwen35::SequenceWindow,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> std::result::Result<
        train::teacher_infer::DeviceLogits,
        train::teacher_infer::TeacherForwardError,
    > {
        match self {
            Self::InProcess(teacher) => {
                train::teacher_infer::TeacherForward::forward_logits_window_device(
                    teacher, input_ids, positions, window, store, tape,
                )
            }
            #[cfg(feature = "cuda")]
            Self::Infer(teacher) => {
                train::teacher_infer::TeacherForward::forward_logits_window_device(
                    teacher, input_ids, positions, window, store, tape,
                )
            }
            Self::Api(teacher) => {
                train::teacher_infer::TeacherForward::forward_logits_window_device(
                    teacher, input_ids, positions, window, store, tape,
                )
            }
            Self::CorpusSftOnly { .. } => {
                Err(train::teacher_infer::TeacherForwardError::InvalidInput(
                    "corpus-truth SFT-only teacher was asked to score windowed KL logits; \
                     this path is only valid with --sft-anchor corpus-truth --gkd-lambda 1.0"
                        .to_owned(),
                ))
            }
        }
    }

    fn vocab_size(&self) -> usize {
        match self {
            Self::InProcess(teacher) => train::teacher_infer::TeacherForward::vocab_size(teacher),
            #[cfg(feature = "cuda")]
            Self::Infer(teacher) => train::teacher_infer::TeacherForward::vocab_size(teacher),
            Self::Api(teacher) => train::teacher_infer::TeacherForward::vocab_size(teacher),
            Self::CorpusSftOnly { vocab_size } => *vocab_size,
        }
    }

    fn parameter_ids(&self) -> &[TensorId] {
        match self {
            Self::InProcess(teacher) => {
                train::teacher_infer::TeacherForward::parameter_ids(teacher)
            }
            #[cfg(feature = "cuda")]
            Self::Infer(teacher) => train::teacher_infer::TeacherForward::parameter_ids(teacher),
            Self::Api(teacher) => train::teacher_infer::TeacherForward::parameter_ids(teacher),
            Self::CorpusSftOnly { .. } => &[],
        }
    }

    fn offload_engine_weights(
        &self,
    ) -> std::result::Result<usize, train::teacher_infer::TeacherForwardError> {
        match self {
            Self::InProcess(teacher) => {
                train::teacher_infer::TeacherForward::offload_engine_weights(teacher)
            }
            #[cfg(feature = "cuda")]
            Self::Infer(teacher) => {
                train::teacher_infer::TeacherForward::offload_engine_weights(teacher)
            }
            Self::Api(teacher) => {
                train::teacher_infer::TeacherForward::offload_engine_weights(teacher)
            }
            Self::CorpusSftOnly { .. } => Ok(0),
        }
    }

    fn reload_engine_weights(
        &self,
    ) -> std::result::Result<(), train::teacher_infer::TeacherForwardError> {
        match self {
            Self::InProcess(teacher) => {
                train::teacher_infer::TeacherForward::reload_engine_weights(teacher)
            }
            #[cfg(feature = "cuda")]
            Self::Infer(teacher) => {
                train::teacher_infer::TeacherForward::reload_engine_weights(teacher)
            }
            Self::Api(teacher) => {
                train::teacher_infer::TeacherForward::reload_engine_weights(teacher)
            }
            Self::CorpusSftOnly { .. } => Ok(()),
        }
    }
}

fn run_opd_from_dirs(args: TrainOpdArgs) -> Result<()> {
    use autograd::{Tape, optim::AdamW};
    use train::{
        lora::LoraConfig,
        opd::{GkdLossConfig, OpdStepConfig, opd_step_with_teacher_forward_profiled_gkd_anchor},
        qwen35_loader::{load_qwen35_from_hf_dir, load_qwen35_lora_from_hf_dir_with_layer_start},
    };

    let student_dir = args
        .student_model
        .as_deref()
        .ok_or_else(|| anyhow!("--student-model <dir> is required for non-smoke runs"))?;
    validate_train_opd_gkd_args(args.gkd_lambda, args.sft_anchor)?;
    reject_unimplemented_gkd_objectives(0.0, args.teacher_topk)?;
    let sft_anchor = opd_sft_anchor_arg(args.sft_anchor);
    let corpus_sft_only = args.sft_anchor == OpdSftAnchorArg::CorpusTruth && args.gkd_lambda == 1.0;
    if corpus_sft_only && args.logits_window_size == 0 {
        bail!(
            "--sft-anchor corpus-truth --gkd-lambda 1.0 requires the windowed logits path; \
             omit --logits-window-size or pass a positive value"
        );
    }
    let teacher_dir = args.teacher_model.as_deref().unwrap_or(student_dir);
    apply_opd_rollout_engine(args.rollout_engine);
    let target_set = parse_lora_target_set(&args.lora_target_set)?;
    let lora = LoraConfig {
        rank: args.lora_rank,
        alpha: args.lora_alpha,
    };

    let (mut store, train_backend, backend_label) = build_opd_store(args.backend)?;
    apply_tape_dtype(&mut store, args.tape_dtype)?;
    let mut tape = Tape::new();

    let teacher_model = if corpus_sft_only {
        None
    } else if matches!(args.teacher_runtime, OpdTeacherRuntimeArg::InProcess) {
        eprintln!(
            "[arle train opd] loading in-process teacher from {}",
            teacher_dir.display()
        );
        Some(
            load_qwen35_from_hf_dir(teacher_dir, &mut store)
                .with_context(|| format!("load teacher from {}", teacher_dir.display()))?,
        )
    } else {
        None
    };
    eprintln!(
        "[arle train opd] loading student from {}",
        student_dir.display()
    );
    let student = load_qwen35_lora_from_hf_dir_with_layer_start(
        student_dir,
        lora,
        target_set,
        args.lora_layer_start,
        &mut store,
    )
    .with_context(|| format!("load LoRA student from {}", student_dir.display()))?;
    let student_params: Vec<TensorId> = student
        .all_parameter_ids()
        .into_iter()
        .filter(|id| store.get(*id).is_some_and(|tensor| tensor.requires_grad))
        .collect();
    if std::env::var_os("ARLE_OPD_LOG_TRAINABLE_PARAMS").is_some() {
        let mut names = std::collections::HashMap::new();
        for (name, id) in student.param_name_map() {
            names.insert(id, name.to_owned());
        }
        for (name, id) in student.adapter_name_map() {
            names.insert(id, name.to_owned());
        }
        let mut rows = student_params
            .iter()
            .filter_map(|&id| {
                store.get(id).map(|tensor| {
                    let elems = tensor.shape.iter().product::<usize>();
                    let alloc_elems = tensor.data.len().max(tensor.size);
                    (
                        alloc_elems,
                        elems,
                        id,
                        tensor.data.len(),
                        tensor.size,
                        tensor.shape.clone(),
                    )
                })
            })
            .collect::<Vec<_>>();
        rows.sort_by_key(|(alloc_elems, _, _, _, _, _)| std::cmp::Reverse(*alloc_elems));
        let total = rows
            .iter()
            .map(|(_, elems, _, _, _, _)| *elems)
            .sum::<usize>();
        let total_alloc = rows
            .iter()
            .map(|(alloc_elems, _, _, _, _, _)| *alloc_elems)
            .sum::<usize>();
        eprintln!(
            "[arle train opd] trainable_params count={} total_elems={} total_alloc_elems={} top_shapes:",
            rows.len(),
            total,
            total_alloc
        );
        for (alloc_elems, elems, id, data_len, size, shape) in rows.iter().take(12) {
            let name = names.get(id).map(String::as_str).unwrap_or("<unnamed>");
            eprintln!(
                "[arle train opd] trainable_param id={} elems={} alloc_elems={} data_len={} size={} shape={:?} name={}",
                id, elems, alloc_elems, data_len, size, shape, name
            );
        }
    }
    let cfg = student.config().clone();
    #[cfg(feature = "cuda")]
    let infer_student = if corpus_sft_only {
        None
    } else {
        // The rollout student re-merges LoRA into experts each step only when the
        // target set covers them (all-linear); tell the loader to keep routed FP8
        // experts as per-expert BF16 so that merge has a mutable matrix to fold
        // into. Attention-only training keeps the cheaper grouped-FP8 experts.
        if matches!(target_set, train::lora::LoraTargetSet::AllLinear) {
            infer_api::set_qwen35_moe_experts_bf16_resident(true);
        }
        load_opd_infer_student(
            student_dir,
            args.prompt_max_tokens + args.rollout_len + 32,
            train_backend.clone(),
            cfg.vocab_size,
            &args.runtime,
        )?
    };
    #[cfg(not(feature = "cuda"))]
    let _ = &train_backend;

    let teacher_forward = match args.teacher_runtime {
        OpdTeacherRuntimeArg::InProcess if corpus_sft_only => OpdCliTeacher::CorpusSftOnly {
            vocab_size: cfg.vocab_size,
        },
        OpdTeacherRuntimeArg::Infer if corpus_sft_only => OpdCliTeacher::CorpusSftOnly {
            vocab_size: cfg.vocab_size,
        },
        OpdTeacherRuntimeArg::Api if corpus_sft_only => OpdCliTeacher::CorpusSftOnly {
            vocab_size: cfg.vocab_size,
        },
        OpdTeacherRuntimeArg::InProcess => {
            let teacher = teacher_model
                .as_ref()
                .expect("in-process teacher was loaded before student");
            OpdCliTeacher::InProcess(train::teacher_infer::InProcessTeacher::new(teacher))
        }
        OpdTeacherRuntimeArg::Infer => {
            #[cfg(not(feature = "cuda"))]
            {
                bail!(
                    "--teacher-runtime infer requires a CUDA build because infer raw-logits \
                     teacher scoring is CUDA-only"
                );
            }
            #[cfg(feature = "cuda")]
            {
                maybe_preoffload_infer_student_before_teacher(&infer_student, &train_backend)?;
                let teacher = OpdCliTeacher::Infer(load_opd_infer_teacher(
                    teacher_dir,
                    args.prompt_max_tokens + args.rollout_len + 32,
                    train_backend.clone(),
                    cfg.vocab_size,
                )?);
                maybe_preoffload_infer_teacher_before_steps(&teacher, &train_backend)?;
                teacher
            }
        }
        OpdTeacherRuntimeArg::Api => {
            // Teacher lives in a separate `arle serve` (own GPU); we only POST tokens
            // and receive raw logits. No teacher weights on the training device — the
            // only layout that fits the full FP8 teacher beside an FP8 student.
            let url = args.teacher_url.as_deref().ok_or_else(|| {
                anyhow!("--teacher-runtime api requires --teacher-url (the raw_logits endpoint)")
            })?;
            OpdCliTeacher::Api(
                train::teacher_infer::ApiTeacher::new(url, cfg.vocab_size)
                    .with_request_dtype("bf16"),
            )
        }
    };

    let prompt_source = load_opd_prompt_source(&args, student_dir, cfg.vocab_size)?;
    if args.sft_anchor == OpdSftAnchorArg::CorpusTruth {
        if args.prompts_file.is_none() {
            bail!("--sft-anchor corpus-truth requires --prompts-file with completion/target rows");
        }
        if prompt_source
            .train_completions
            .iter()
            .any(|completion| completion.as_deref().is_none_or(<[u32]>::is_empty))
        {
            bail!(
                "--sft-anchor corpus-truth requires every train row in --prompts-file \
                 to have non-empty completion, target, or completion_ids"
            );
        }
    }

    let mut optimizer = AdamW::new(args.lr, (0.9, 0.999), 1.0e-8, 0.0);
    let lr_schedule =
        OpdLrSchedule::new(args.lr_schedule, args.lr, args.lr_warmup_steps, args.steps)?;
    let rollout_sampling = rollout_sampling_params(
        args.rollout_temperature,
        args.rollout_top_p,
        args.rollout_top_k,
        args.rollout_seed,
    );
    let step_cfg = OpdStepConfig {
        rollout_len: args.rollout_len,
        rollout_sampling,
        grad_clip: args.grad_clip,
    };
    let kl_direction = kl_direction_arg(args.kl_direction);
    let kl_mask = kl_mask_arg(args.kl_mask);

    let gate_on = args.gate_every_n > 0;
    let mut nll_baseline = if gate_on {
        heldout_nll(
            &student,
            &prompt_source.eval_ids,
            cfg.vocab_size,
            &mut store,
        )?
    } else {
        f32::INFINITY
    };
    if gate_on && !nll_baseline.is_finite() {
        bail!(
            "initial held-out NLL is non-finite ({nll_baseline}); cannot establish \
             an OPD evaluation baseline"
        );
    }
    if gate_on && !args.json {
        println!("gate baseline_nll {nll_baseline:.6}");
    }

    let mut losses: Vec<f32> = Vec::with_capacity(args.steps);
    let mut gate_nlls: Vec<(usize, f32)> = Vec::new();
    let mut prompt_sampler = PromptSampler::new(args.prompt_seed);
    for step in 1..=args.steps {
        let _step_lr = lr_schedule.apply_to_optimizer(&mut optimizer, (step - 1) as u64);
        let prompt_index = prompt_sampler.next_index(prompt_source.train_prompts.len());
        let prompt_ids = prompt_source.train_prompts[prompt_index].as_slice();
        let corpus_tokens = prompt_source.train_completions[prompt_index].as_deref();
        let mut step_profile = opd_step_profile_enabled().then(train::opd::OpdStepProfile::default);
        #[cfg(feature = "cuda")]
        let infer_rollout = infer_student
            .as_ref()
            .map(|student| train::opd::InferRolloutCtx {
                student,
                lora_config: lora,
            });
        let outcome = opd_step_with_teacher_forward_profiled_gkd_anchor(
            &student,
            &teacher_forward,
            prompt_ids,
            step_cfg.clone(),
            &student_params,
            &mut optimizer,
            &mut store,
            &mut tape,
            GkdLossConfig {
                lambda: args.gkd_lambda,
                sft_anchor,
                corpus_tokens,
                kl_chunk_size: GkdLossConfig::default().kl_chunk_size,
                kl_direction,
                kl_temperature: args.kl_temperature,
                kl_beta: args.kl_beta,
                teacher_topk: args.teacher_topk,
                fused_distill: args.fused_distill && !args.no_fused_distill,
                logits_window_size: opd_logits_window_size_arg(args.logits_window_size),
                kl_mask,
            },
            None,
            #[cfg(feature = "cuda")]
            infer_rollout,
            step_profile.as_mut(),
        )
        .with_context(|| format!("opd step {step} failed"))?;
        losses.push(outcome.loss);
        let mut gate_nll = f32::NAN;
        if gate_on && step % args.gate_every_n == 0 {
            gate_nll = heldout_nll(
                &student,
                &prompt_source.eval_ids,
                cfg.vocab_size,
                &mut store,
            )?;
            nll_baseline = gate_nll;
            gate_nlls.push((step, gate_nll));
        }
        if !args.json {
            if gate_on && step % args.gate_every_n == 0 {
                println!(
                    "step {step}/{total} loss {loss:.6} rollout_len {rl} gate observe nll \
                     {gate_nll:.6} baseline {nll_baseline:.6}",
                    total = args.steps,
                    loss = outcome.loss,
                    rl = outcome.rollout_len,
                );
            } else {
                println!(
                    "step {step}/{total} loss {loss:.6} rollout_len {rl}",
                    total = args.steps,
                    loss = outcome.loss,
                    rl = outcome.rollout_len,
                );
            }
            if let Some(profile) = step_profile.as_ref() {
                print_opd_step_profile(step, profile);
            }
        }
        maybe_save_full_student_checkpoint(
            "opd",
            args.save_checkpoint.as_deref(),
            args.save_every,
            step,
            args.steps,
            student_dir,
            &student,
            &mut store,
            &mut tape,
        )?;
    }

    if args.steps == 0 {
        maybe_save_full_student_checkpoint(
            "opd",
            args.save_checkpoint.as_deref(),
            args.save_every,
            0,
            0,
            student_dir,
            &student,
            &mut store,
            &mut tape,
        )?;
    }

    if args.json {
        let report = serde_json::json!({
            "mode": "from-dirs",
            "backend": backend_label,
            "student_model": student_dir.display().to_string(),
            "teacher_model": teacher_dir.display().to_string(),
            "teacher_runtime": format!("{:?}", args.teacher_runtime),
            "steps": args.steps,
            "rollout_len": args.rollout_len,
            "lr": args.lr,
            "gkd_lambda": args.gkd_lambda,
            "sft_anchor": format!("{:?}", args.sft_anchor),
            "kl_temperature": args.kl_temperature,
            "kl_beta": args.kl_beta,
            "teacher_topk": args.teacher_topk,
            "fused_distill": args.fused_distill && !args.no_fused_distill,
            "gate_every_n": args.gate_every_n,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_target_set": args.lora_target_set,
            "lora_layer_start": args.lora_layer_start,
            "save_checkpoint": args.save_checkpoint.as_ref().map(|path| path.display().to_string()),
            "save_every": args.save_every,
            "final_nll_baseline": nll_baseline,
            "losses": losses,
            "gate_nlls": gate_nlls,
            "prompt_ids": prompt_source.report_prompt_ids,
            "completion_rows": prompt_source.completion_rows,
            "eval_ids": prompt_source.eval_ids,
            "vocab_size": cfg.vocab_size,
            "hidden_size": cfg.hidden_size,
            "num_hidden_layers": cfg.num_hidden_layers,
            "full_attn_gated": cfg.full_attn_gated,
        });
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!(
            "ARLE train opd: ran {} step(s) on Qwen3.x (vocab={}, hidden={}, layers={}, full_attn_gated={}, backend={})",
            args.steps,
            cfg.vocab_size,
            cfg.hidden_size,
            cfg.num_hidden_layers,
            cfg.full_attn_gated,
            backend_label,
        );
    }
    Ok(())
}

fn run_opd_smoke(args: TrainOpdArgs) -> Result<()> {
    use autograd::{Tape, optim::AdamW};
    use train::{
        opd::{
            GkdLossConfig, GkdSftAnchor, OpdStepConfig,
            opd_step_with_teacher_forward_profiled_gkd_anchor,
        },
        qwen35::Qwen35Model,
    };

    validate_train_opd_gkd_args(args.gkd_lambda, args.sft_anchor)?;
    reject_unimplemented_gkd_objectives(0.0, args.teacher_topk)?;
    if args.sft_anchor != OpdSftAnchorArg::StudentRollout {
        bail!("train opd --smoke does not support --sft-anchor corpus-truth");
    }
    if args.prompts_file.is_some() {
        bail!("--prompts-file requires --student-model; smoke mode uses --prompt-ids");
    }
    let cfg = embedded_tiny_qwen35_config();
    let prompt_ids = parse_prompt_ids(args.prompt_ids.as_deref())?;
    if prompt_ids.iter().any(|&id| (id as usize) >= cfg.vocab_size) {
        bail!(
            "smoke prompt token ids must be < {} (the embedded tiny vocab size)",
            cfg.vocab_size
        );
    }

    let (mut store, _train_backend, backend_label) = build_opd_store(args.backend)?;
    apply_tape_dtype(&mut store, args.tape_dtype)?;
    let mut tape = Tape::new();
    let teacher = Qwen35Model::new_for_eval(&cfg, &mut store).context("build smoke teacher")?;
    let teacher_forward = train::teacher_infer::InProcessTeacher::new(&teacher);
    let student = Qwen35Model::new(&cfg, &mut store).context("build smoke student")?;
    let student_params = student.all_parameter_ids();

    let mut optimizer = AdamW::new(args.lr, (0.9, 0.999), 1.0e-8, 0.0);
    let lr_schedule =
        OpdLrSchedule::new(args.lr_schedule, args.lr, args.lr_warmup_steps, args.steps)?;
    let rollout_sampling = rollout_sampling_params(
        args.rollout_temperature,
        args.rollout_top_p,
        args.rollout_top_k,
        args.rollout_seed,
    );
    let step_cfg = OpdStepConfig {
        rollout_len: args.rollout_len,
        rollout_sampling,
        grad_clip: args.grad_clip,
    };
    let kl_direction = kl_direction_arg(args.kl_direction);
    let kl_mask = kl_mask_arg(args.kl_mask);

    let mut losses: Vec<f32> = Vec::with_capacity(args.steps);
    let mut step_metrics: Vec<OpdStepMetric> = Vec::with_capacity(args.steps);
    let progress = if args.json || args.steps == 0 {
        None
    } else {
        let progress = ProgressBar::new(args.steps as u64);
        progress.set_style(opd_progress_style()?);
        progress.set_message("avg_loss=pending");
        Some(progress)
    };
    let mut loss_sum = 0.0_f32;
    for step in 1..=args.steps {
        let step_lr = lr_schedule.apply_to_optimizer(&mut optimizer, (step - 1) as u64);
        let outcome = opd_step_with_teacher_forward_profiled_gkd_anchor(
            &student,
            &teacher_forward,
            &prompt_ids,
            step_cfg.clone(),
            &student_params,
            &mut optimizer,
            &mut store,
            &mut tape,
            GkdLossConfig {
                lambda: 0.0,
                sft_anchor: GkdSftAnchor::StudentRollout,
                corpus_tokens: None,
                kl_chunk_size: GkdLossConfig::default().kl_chunk_size,
                kl_direction,
                kl_temperature: args.kl_temperature,
                kl_beta: args.kl_beta,
                teacher_topk: args.teacher_topk,
                fused_distill: args.fused_distill && !args.no_fused_distill,
                logits_window_size: opd_logits_window_size_arg(args.logits_window_size),
                kl_mask,
            },
            None,
            #[cfg(feature = "cuda")]
            None,
            None,
        )
        .with_context(|| format!("opd step {step} failed"))?;
        let grad_norm = current_grad_norm(&student_params, &store);
        loss_sum += outcome.loss;
        let avg_loss = loss_sum / step as f32;
        losses.push(outcome.loss);
        step_metrics.push(OpdStepMetric {
            step,
            loss: outcome.loss,
            lr: step_lr,
            grad_norm,
            rollout_len: outcome.rollout_len,
        });
        if let Some(progress) = &progress {
            progress.set_message(format!("{avg_loss:.6}"));
            progress.inc(1);
        }
    }
    if let Some(progress) = progress {
        let final_loss = losses
            .last()
            .map(|loss| format!("{loss:.6}"))
            .unwrap_or_else(|| "n/a".to_string());
        progress.finish_with_message(format!("final_loss={final_loss}"));
    }

    if args.json {
        let report = serde_json::json!({
            "mode": "smoke",
            "backend": backend_label,
            "steps": args.steps,
            "rollout_len": args.rollout_len,
            "lr": args.lr,
            "kl_temperature": args.kl_temperature,
            "kl_beta": args.kl_beta,
            "teacher_topk": args.teacher_topk,
            "fused_distill": args.fused_distill && !args.no_fused_distill,
            "losses": losses,
            "step_metrics": step_metrics,
            "summary": opd_summary(&step_metrics),
            "prompt_ids": prompt_ids,
        });
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!(
            "ARLE train opd smoke: ran {} step(s) on tiny Qwen3.5 (vocab={}, hidden={}, layers={}, backend={})",
            args.steps, cfg.vocab_size, cfg.hidden_size, cfg.num_hidden_layers, backend_label,
        );
    }
    Ok(())
}

fn parse_lora_target_set(raw: &str) -> Result<train::lora::LoraTargetSet> {
    use train::lora::LoraTargetSet;
    match raw {
        "attention-qv" | "attention_qv" | "qv" => Ok(LoraTargetSet::AttentionQv),
        "attention-full" | "attention_full" | "full-attn" => Ok(LoraTargetSet::AttentionFull),
        "all-linear" | "all_linear" | "all" => Ok(LoraTargetSet::AllLinear),
        other => bail!(
            "unknown --lora-target-set {other:?} (expected attention-qv, attention-full, or all-linear)"
        ),
    }
}

fn rollout_sampling_params(
    temperature: f32,
    top_p: f32,
    top_k: i32,
    seed: Option<u64>,
) -> Option<SamplingParams> {
    if temperature == 0.0 {
        None
    } else {
        Some(SamplingParams {
            temperature,
            top_p,
            top_k,
            seed,
            ..SamplingParams::default()
        })
    }
}

fn kl_direction_arg(arg: KlDirectionArg) -> train::loss::KlDirection {
    match arg {
        KlDirectionArg::Forward => train::loss::KlDirection::Forward,
        KlDirectionArg::Reverse => train::loss::KlDirection::Reverse,
    }
}

fn kl_mask_arg(arg: OpdKlMaskArg) -> train::opd::OpdKlMask {
    match arg {
        OpdKlMaskArg::Completion => train::opd::OpdKlMask::CompletionOnly,
        OpdKlMaskArg::Full => train::opd::OpdKlMask::Full,
    }
}

fn run_self_opd(args: TrainSelfOpdArgs) -> ExitCode {
    // Reject 0.0, negatives, AND NaN (`NaN <= 0.0` is false, so the is_nan arm).
    if args.gkd_lambda <= 0.0 || args.gkd_lambda.is_nan() {
        eprintln!(
            "[arle train self-opd] error: --gkd-lambda must be > 0 ({} given).\n\
             SOPD cold-starts with the EMA teacher == an exact copy of the student\n\
             (lora_b=0 ⇒ student == EMA == base), so the pure KL term has zero gradient\n\
             and the run is a silent no-op. The bootstrap gradient comes from the λ>0\n\
             GKD CE self-anchor on the rollouts; default is 0.5.",
            args.gkd_lambda
        );
        return ExitCode::FAILURE;
    }
    if args.smoke {
        return exit_from_result(run_self_opd_smoke(args));
    }
    if args.student_model.is_some() {
        return exit_from_result(run_self_opd_from_dir(args));
    }
    eprintln!(
        "[arle train self-opd] error: either `--student-model <dir>` or `--smoke` is required.\n\
         See `arle train self-opd --help` for the full surface."
    );
    ExitCode::FAILURE
}

#[cfg(not(feature = "cuda"))]
fn run_rubric_opd_impl(_args: TrainRubricOpdArgs) -> Result<()> {
    bail!(
        "rubric-opd requires the cuda feature (the Flash judge + rollout engines are \
         CUDA-only). Build with --features cuda,nccl."
    )
}

#[cfg(not(feature = "cuda"))]
fn run_agent_opd_impl(_args: TrainAgentOpdArgs) -> Result<()> {
    bail!(
        "agent-opd requires the cuda feature (the in-process rollout engine + CE \
         writeback are CUDA-only). Build with --features cuda,nccl."
    )
}

/// Load the in-process eval set (jsonl `{problem, answer}`), head-`n`, rendered with
/// the same math prompt prefix the training corpus uses, tokenized. Returns
/// `(prompt_ids, problem_text, gold_answer)`.
#[cfg(feature = "cuda")]
fn load_rubric_eval_items(
    path: &Path,
    n: usize,
    tokenizer: &train::tokenizer::ChatTokenizer,
    vocab: usize,
) -> Result<Vec<(Vec<u32>, String, String)>> {
    let raw =
        fs::read_to_string(path).with_context(|| format!("read eval file {}", path.display()))?;
    let mut items = Vec::new();
    for line in raw.lines() {
        if items.len() >= n {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let value: serde_json::Value =
            serde_json::from_str(line).context("parse --eval-prompts-file JSONL line")?;
        let problem = value.get("problem").and_then(|p| p.as_str()).unwrap_or("");
        let gold = value.get("answer").and_then(|a| a.as_str()).unwrap_or("");
        if problem.is_empty() {
            continue;
        }
        let rendered = format!(
            "Solve the following math problem. Show your reasoning, and put only the final \
             answer in \\boxed{{}}.\n\nProblem:\n{problem}"
        );
        let ids = tokenizer
            .encode(&rendered, true)
            .map_err(|err| anyhow!("encode eval prompt: {err}"))?;
        if ids.is_empty() || ids.iter().any(|&id| (id as usize) >= vocab) {
            continue;
        }
        items.push((ids, problem.to_string(), gold.to_string()));
    }
    Ok(items)
}

/// Generate one greedy answer per eval item via the rollout engine (the trained
/// student after `sync_lora`), and dump `{problem, gold, answer}` jsonl for scoring.
#[cfg(feature = "cuda")]
fn rubric_eval_pass(
    infer_student: &train::infer_student::InferStudent,
    eval_items: &[(Vec<u32>, String, String)],
    tokenizer: &train::tokenizer::ChatTokenizer,
    max_new: usize,
    out_path: &Path,
) -> Result<()> {
    use std::io::Write;

    let greedy = SamplingParams {
        temperature: 0.0,
        ..SamplingParams::default()
    };
    // Batch all eval prompts through the continuous-batching engine instead of
    // decoding one at a time (per-request B=1 decode is memory-bandwidth-bound;
    // the scheduler admits as many as KV fits and queues the rest).
    let requests: Vec<(Vec<u32>, SamplingParams)> = eval_items
        .iter()
        .map(|(prompt_ids, _, _)| (prompt_ids.clone(), greedy.clone()))
        .collect();
    let generated = infer_student.generate_batch(&requests, max_new)?;

    let mut file = fs::File::create(out_path)
        .with_context(|| format!("create eval out {}", out_path.display()))?;
    for ((_, problem, gold), ids) in eval_items.iter().zip(generated.iter()) {
        let answer = tokenizer
            .decode(ids, true)
            .map_err(|err| anyhow!("decode eval answer: {err}"))?;
        let line = serde_json::json!({"problem": problem, "gold": gold, "answer": answer});
        writeln!(file, "{line}")?;
    }
    file.flush()?;
    eprintln!(
        "[rubric eval] wrote {} answers -> {}",
        eval_items.len(),
        out_path.display()
    );
    Ok(())
}

/// Rubric-OPD RFT loop: the student samples N rollouts/prompt, DeepSeek-V4-Flash
/// judges each against a text-level rubric (vocab-agnostic), and the accepted
/// rollouts are written back as CE targets. Mode A (select-only) for now.
#[cfg(feature = "cuda")]
fn run_rubric_opd_impl(args: TrainRubricOpdArgs) -> Result<()> {
    use std::sync::{Arc, Mutex};

    use autograd::optim::AdamW;
    use infer_api::{EngineLoadConfig, LoadedInferenceEngine};
    use train::{
        infer_student::{InferStudent, save_lora_adapters},
        lora::LoraConfig,
        opd::rubric_writeback_ce_step_batched,
        qwen35_loader::{SharedFrozenBaseEntry, load_qwen35_lora_from_hf_dir_with_shared_base},
        rubric::{bfcl_agentic_rubric, math_rubric},
        rubric_opd::{FlashJudge, RubricOpdConfig, run_rubric_rounds},
        tokenizer::ChatTokenizer,
    };

    use crate::args::{RubricTaskArg, RubricWritebackArg};

    if matches!(args.writeback, RubricWritebackArg::Correction) {
        bail!(
            "--writeback correction (Mode B: teacher rewrites rejected rollouts) is not yet \
             wired; use --writeback accepted (Mode A, RFT) for now."
        );
    }

    let student_dir = args.student_model.as_path();
    let teacher_dir = args.teacher_model.as_path();
    let target_set = parse_lora_target_set(&args.lora_target_set)?;
    let lora = LoraConfig {
        rank: args.lora_rank,
        alpha: args.lora_alpha,
    };

    let (mut store, train_backend, _backend_label) = build_opd_store(args.backend)?;
    apply_tape_dtype(&mut store, args.tape_dtype)?;

    // Vocab is read from the checkpoint config (not the autograd student) so the
    // rollout engine can load BEFORE the autograd student when `--share-frozen-base`
    // is set — the student must then import the engine's resident FP8 base ptrs.
    // Same value the autograd `Qwen35Model::config()` exposes (both built from the
    // same config.json), so the default path is unchanged.
    let hf_config = Qwen35Config::from_json_file(student_dir.join("config.json"))
        .with_context(|| format!("read config.json from {}", student_dir.display()))?;
    let vocab = hf_config.vocab_size;

    // Student tokenizer: encodes problems, decodes rollouts for the judge.
    let tokenizer_path = resolve_local_tokenizer_path(student_dir)?;
    let tokenizer = ChatTokenizer::from_file(&tokenizer_path)
        .with_context(|| format!("load tokenizer from {}", tokenizer_path.display()))?;

    // Corpus: each JSONL line is {"text": "<problem>"}.
    let raw = fs::read_to_string(&args.prompts_file)
        .with_context(|| format!("read prompts file {}", args.prompts_file.display()))?;
    let mut prompts: Vec<(String, Vec<u32>)> = Vec::new();
    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let value: serde_json::Value =
            serde_json::from_str(line).context("parse --prompts-file JSONL line")?;
        let text = value
            .get("text")
            .and_then(|t| t.as_str())
            .ok_or_else(|| anyhow!("--prompts-file line missing a string `text` field"))?;
        let ids = tokenizer
            .encode(text, true)
            .map_err(|err| anyhow!("encode prompt: {err}"))?;
        if ids.iter().any(|&id| (id as usize) >= vocab) {
            bail!("prompt token id >= vocab {vocab} in --prompts-file");
        }
        if !ids.is_empty() {
            prompts.push((text.to_string(), ids));
        }
    }
    if prompts.is_empty() {
        bail!("no usable prompts in {}", args.prompts_file.display());
    }
    eprintln!(
        "[arle train rubric-opd] loaded {} prompts; rubric={:?} writeback={:?}",
        prompts.len(),
        args.rubric_task,
        args.writeback
    );

    // Rollout engine (student) — own KV/scheduler path for N-sample generation AND
    // in-process eval. Size for the LARGER of rollout vs eval token budgets (eval
    // generates up to --eval-max-new-tokens to reach the final \boxed{}).
    //
    // Load order: when `--share-frozen-base`, the rollout engine loads FIRST so the
    // autograd student can import (zero-copy) its resident FP8 base. In the default
    // path the relative order of the two loads is immaterial (each allocates its
    // own copy); building the engine first is byte-identical for the default.
    let max_prompt_len = prompts.iter().map(|(_, ids)| ids.len()).max().unwrap_or(0);
    let gen_budget = args.max_new_tokens.max(args.eval_max_new_tokens);
    let student_seq = (max_prompt_len + gen_budget + 32).max(128);

    // Load order matters for the rollout engine's num_slots clamp (computed from
    // post-weights free VRAM). Default: load the autograd student FIRST so the
    // engine then sees post-student free VRAM — byte-identical to the
    // pre-weight-share order. --share-frozen-base: the engine must load FIRST so
    // the student can import (zero-copy) its resident FP8 base, so the student
    // load is deferred to the `match` below.
    let prebuilt_student = if !args.no_share_frozen_base {
        None
    } else {
        eprintln!(
            "[arle train rubric-opd] loading student from {}",
            student_dir.display()
        );
        Some(
            load_qwen35_lora_from_hf_dir_with_shared_base(
                student_dir,
                lora,
                target_set,
                args.lora_layer_start,
                false,
                None,
                &mut store,
            )
            .with_context(|| format!("load LoRA student from {}", student_dir.display()))?,
        )
    };

    eprintln!(
        "[arle train rubric-opd] loading rollout engine from {} (max_seq_len={student_seq})",
        student_dir.display()
    );
    let student_engine = LoadedInferenceEngine::load_with_config(
        student_dir
            .to_str()
            .ok_or_else(|| anyhow!("student path is not valid UTF-8"))?,
        true,
        EngineLoadConfig {
            // num_slots>1 lets batched eval (generate_batch) decode multiple
            // prompts concurrently — B=1 decode is memory-bandwidth-bound, so
            // concurrency amortizes the 27B weight reads. KV pool is allocated at
            // load, and three models are co-resident (~92/96 GB): num_slots=4
            // OOM'd the 35B-judge weight upload, so 2 is the headroom-safe max
            // (eval scheduler runs 2 at a time + queues the rest).
            num_slots: args.rollout_num_slots,
            page_size: 16,
            total_pages: args.rollout_num_slots.max(1) * student_seq.div_ceil(16),
            max_prompt_tokens: student_seq,
            max_total_tokens: student_seq,
            chunked_prefill_size: Some(student_seq),
            // Not `--rollout-mem-fraction`: this path loads a third ~35B judge at
            // 0.9 AFTER this engine, and the envelope above is already the
            // tightest in the file. Unmeasured at 0.5.
            mem_fraction_static: 0.2,
            dspark_draft_model: args.runtime.dspark_draft_model.clone(),
            dspark_sps_bias_ms: args.runtime.dspark_sps_bias_ms,
            dspark_sps_row_ms: args.runtime.dspark_sps_row_ms,
            ..EngineLoadConfig::default()
        },
    )
    .with_context(|| format!("load rollout engine from {}", student_dir.display()))?;

    // Train-infer FP8 weight sharing (`--share-frozen-base`): borrow the rollout
    // engine's resident FP8 base pointers, map to the loader's backend-agnostic
    // table, and pass it into the autograd student load so its frozen FP8 base
    // projections import a NON-OWNING view instead of allocating ~27 GB.
    let shared_base_entries: Vec<SharedFrozenBaseEntry> = if !args.no_share_frozen_base {
        let table = student_engine
            .frozen_base_fp8_pointers()
            .context("borrow rollout-engine FP8 base pointers for --share-frozen-base")?;
        eprintln!(
            "[arle train rubric-opd] --share-frozen-base: borrowing {} resident FP8 base projections from the rollout engine (zero-copy)",
            table.len()
        );
        table
            .into_iter()
            .map(|p| SharedFrozenBaseEntry {
                layer_idx: p.layer_idx,
                proj_suffix: p.proj_suffix,
                weight_ptr: p.weight_ptr,
                scale_ptr: p.scale_ptr,
                rows: p.rows,
                cols: p.cols,
                block_m: p.block_m,
                block_k: p.block_k,
            })
            .collect()
    } else {
        Vec::new()
    };
    let shared_base = if !args.no_share_frozen_base {
        Some(shared_base_entries.as_slice())
    } else {
        None
    };

    let student = match prebuilt_student {
        // Default path: the student was already loaded first (above), engine second.
        Some(s) => s,
        // --share-frozen-base: load the student now, AFTER the engine, importing
        // its resident FP8 base pointers (zero-copy) for the frozen projections.
        None => {
            eprintln!(
                "[arle train rubric-opd] loading student from {}",
                student_dir.display()
            );
            load_qwen35_lora_from_hf_dir_with_shared_base(
                student_dir,
                lora,
                target_set,
                args.lora_layer_start,
                false,
                shared_base,
                &mut store,
            )
            .with_context(|| format!("load LoRA student from {}", student_dir.display()))?
        }
    };

    // When sharing, the autograd student's frozen base now ALIASES the engine's
    // resident FP8 bytes. Drain the autograd backend's OWN upload stream (the
    // student's trainable-suffix uploads) before the first autograd forward so the
    // shared bytes are guaranteed valid across the (separate) train/infer streams
    // on the shared primary context (cross-stream handoff fence). MUST be
    // stream-scoped, NOT `cuCtxSynchronize`: the co-resident engine's streams on
    // this shared primary context run event-tracking-disabled + idle-parked, so a
    // context-wide drain deadlocks (see the agent-OPD share-frozen-base fix). The
    // engine's resident FP8 base was already written by its own load+warmup.
    if !args.no_share_frozen_base {
        train_backend
            .stream_synchronize()
            .context("stream sync before first shared-base autograd forward")?;
    }

    let all_params: Vec<TensorId> = student.all_parameter_ids();
    let trainable: Vec<TensorId> = all_params
        .iter()
        .copied()
        .filter(|id| store.get(*id).is_some_and(|tensor| tensor.requires_grad))
        .collect();
    if trainable.is_empty() {
        bail!("rubric-opd student has no trainable (LoRA) parameters; check --lora-target-set");
    }

    let infer_student = InferStudent::new(
        Arc::new(Mutex::new(student_engine)),
        train_backend.clone(),
        vocab,
    );

    // Judge engine (DeepSeek-V4-Flash) — text in, verdict out (own tokenizer).
    // Self-consistency mode loads NO judge (the student majority-votes on its own
    // \boxed answer), freeing the ~35GB judge VRAM; teacher-model is ignored.
    let judge = if args.self_consistency {
        eprintln!(
            "[arle train rubric-opd] self-consistency mode: no judge engine loaded (majority-vote on \\boxed)"
        );
        None
    } else {
        let judge_prompt_cap = (student_seq * 2 + 1024).max(2048);
        let judge_total = judge_prompt_cap + args.max_verdict_tokens;
        eprintln!(
            "[arle train rubric-opd] loading Flash judge from {} (max_total={judge_total})",
            teacher_dir.display()
        );
        let judge_engine = LoadedInferenceEngine::load_with_config(
            teacher_dir
                .to_str()
                .ok_or_else(|| anyhow!("teacher path is not valid UTF-8"))?,
            true,
            EngineLoadConfig {
                num_slots: args.judge_num_slots,
                page_size: 16,
                total_pages: args.judge_num_slots.max(1) * judge_total.div_ceil(16),
                max_prompt_tokens: judge_prompt_cap,
                max_total_tokens: judge_total,
                chunked_prefill_size: Some(judge_prompt_cap),
                ..EngineLoadConfig::default()
            },
        )
        .with_context(|| format!("load Flash judge from {}", teacher_dir.display()))?;
        Some(FlashJudge::new(
            Arc::new(Mutex::new(judge_engine)),
            args.max_verdict_tokens,
        ))
    };

    let mut optimizer = AdamW::new(args.lr, (0.9, 0.999), 1.0e-8, 0.0);
    let rubric = match args.rubric_task {
        RubricTaskArg::Math => math_rubric(),
        RubricTaskArg::Agentic => bfcl_agentic_rubric(),
    };
    let sampling = rollout_sampling_params(
        args.rollout_temperature,
        args.rollout_top_p,
        args.rollout_top_k,
        args.rollout_seed,
    );
    let round_cfg = RubricOpdConfig {
        rounds: 1,
        samples_per_prompt: args.samples_per_prompt,
        max_new_tokens: args.max_new_tokens,
        writeback_cap: args.writeback_cap,
        writeback_batch: args.writeback_batch,
        correction_cap: args.correction_cap,
        correction_max_tokens: args.correction_max_tokens,
        share_frozen_base: !args.no_share_frozen_base,
        distill_shortest: args.distill_shortest,
    };

    // In-process eval (base + per-round) via the rollout engine — no checkpoint save.
    let eval_items = match args.eval_prompts_file.as_deref() {
        Some(path) => load_rubric_eval_items(path, args.eval_n, &tokenizer, vocab)?,
        None => Vec::new(),
    };
    let eval_dir = args.eval_out_dir.clone();
    if let Some(dir) = eval_dir.as_deref()
        && !eval_items.is_empty()
    {
        fs::create_dir_all(dir)
            .with_context(|| format!("create eval out dir {}", dir.display()))?;
        rubric_eval_pass(
            &infer_student,
            &eval_items,
            &tokenizer,
            args.eval_max_new_tokens,
            &dir.join("eval_round_base.jsonl"),
        )?;
    }

    for round in 0..args.rounds {
        // Fresh closures each round so the &mut store / &mut optimizer borrows
        // are released before the per-round checkpoint reuses the store.
        let reports = {
            let student_ref = &student;
            let all_ref = all_params.as_slice();
            let trainable_ref = trainable.as_slice();
            let store_ref = &mut store;
            let opt_ref = &mut optimizer;
            let tok_ref = &tokenizer;
            run_rubric_rounds(
                &infer_student,
                judge.as_ref(),
                &rubric,
                &prompts,
                &round_cfg,
                sampling.as_ref(),
                |ids| {
                    tok_ref
                        .decode(ids, true)
                        .map_err(|err| anyhow!("decode rollout: {err}"))
                },
                |chunk: &[(Vec<u32>, Vec<u32>)]| {
                    rubric_writeback_ce_step_batched(
                        student_ref,
                        all_ref,
                        trainable_ref,
                        opt_ref,
                        chunk,
                        vocab,
                        args.writeback_window,
                        store_ref,
                    )
                    .map_err(anyhow::Error::from)
                },
                |text: &str| {
                    tok_ref
                        .encode(text, false)
                        .map_err(|err| anyhow!("encode correction: {err}"))
                },
            )?
        };
        let rep = &reports[0];
        eprintln!(
            "[arle train rubric-opd] round {round}: accepted={} distinct={} parse_err={} trained={} corrected={} mean_loss={:.4}",
            rep.accepted,
            rep.distinct_accepted,
            rep.parse_errors,
            rep.trained,
            rep.corrected,
            rep.mean_train_loss
        );

        // Sync the trained LoRA into the rollout engine (always): the eval below
        // and the next round both need this round's improved student. Round 0
        // sampled from base, as intended.
        // The writeback's autograd forward+backward leaves activation/gradient
        // tensors resident in the store; drop everything except the model
        // parameters (and let the optimizer re-create its states next step) so
        // the engine-thread LoRA re-merge has headroom.
        let keep: std::collections::HashSet<_> = all_params.iter().copied().collect();
        eprintln!("[rubric-opd] before retain_ids: keep={} params", keep.len());
        store.retain_ids(&keep);
        // DEBUG: account for device memory held by the store's live parameters.
        {
            let mut fp8_bytes = 0usize;
            let mut bf16_bytes = 0usize;
            let mut f32_bytes = 0usize;
            let mut other_bytes = 0usize;
            for &id in all_params.iter() {
                if let Some(t) = store.get(id) {
                    match &t.device_handle {
                        Some(autograd::DeviceHandle::CudaFp8BlockScaled(_)) => {
                            fp8_bytes += t.size;
                        }
                        Some(autograd::DeviceHandle::CudaBf16(_)) => {
                            bf16_bytes += t.size * 2;
                        }
                        Some(autograd::DeviceHandle::Cuda(_)) => {
                            f32_bytes += t.size * 4;
                        }
                        Some(_) => other_bytes += t.size * 4,
                        None => {}
                    }
                }
            }
            eprintln!(
                "[rubric-opd] store params device bytes: fp8={}MiB bf16={}MiB f32={}MiB other={}MiB",
                fp8_bytes >> 20,
                bf16_bytes >> 20,
                f32_bytes >> 20,
                other_bytes >> 20
            );
        }
        // The optimizer's AdamW moments (m, v) live as DeviceHandles outside the
        // store, so retain_ids does not free them. Drop them here — the next
        // writeback step re-creates them from zero — or the engine-thread LoRA
        // re-merge OOMs on the ~2×param bytes they hold.
        optimizer.clear_param_state(&all_params);
        log_opd_vram("rubric post-retain_ids", &train_backend);
        if let Err(err) = store.backend().trim_memory_pool() {
            eprintln!("[rubric-opd] device-pool trim before LoRA sync failed: {err}");
        }
        log_opd_vram("rubric post-trim", &train_backend);
        // The rollout KV pool was re-acquired in run_rubric_rounds' Phase D, but
        // the LoRA re-merge needs headroom for the per-layer BF16 promotion
        // (FP8 base stays resident as keepalive for the share-frozen-base
        // student alias, so peak = FP8 + BF16 ≈ 3× base bytes). The KV pool is
        // dead during the merge — release it and re-acquire after, symmetric
        // with the agent-OPD sync_and_restore_engines pattern.
        if let Err(err) = infer_student.release_kv_pool() {
            eprintln!("[rubric-opd] release KV pool before LoRA sync failed: {err}");
        }
        infer_student
            .sync_lora_from_store(
                &mut store,
                &student.adapter_name_map(),
                &student.param_name_map(),
                lora,
            )
            .context("sync trained LoRA into rollout engine")?;
        if let Err(err) = infer_student.ensure_kv_pool() {
            eprintln!("[rubric-opd] re-acquire KV pool after LoRA sync failed: {err}");
        }

        // Eval this round's student in-process (rollout engine now holds round-N LoRA).
        if let Some(dir) = eval_dir.as_deref()
            && !eval_items.is_empty()
        {
            rubric_eval_pass(
                &infer_student,
                &eval_items,
                &tokenizer,
                args.eval_max_new_tokens,
                &dir.join(format!("eval_round{round}.jsonl")),
            )?;
        }

        // Fast adapter-only (LoRA) save — avoids the full-materialize host-loop hang.
        if let Some(adapter_dir) = args.save_lora_adapters.as_deref()
            && should_save_step_checkpoint(round + 1, args.rounds, args.save_every)
        {
            fs::create_dir_all(adapter_dir)
                .with_context(|| format!("create LoRA adapter dir {}", adapter_dir.display()))?;
            let out = adapter_dir.join(format!("adapters_round{}.safetensors", round + 1));
            let started = Instant::now();
            save_lora_adapters(&mut store, &student.adapter_name_map(), &out)
                .with_context(|| format!("save LoRA adapters at round {}", round + 1))?;
            println!(
                "checkpoint_saved kind=lora_adapters mode=rubric-opd step={} dir={} seconds={:.6}",
                round + 1,
                out.display(),
                started.elapsed().as_secs_f64()
            );
        }

        let mut ckpt_tape = Tape::new();
        maybe_save_full_student_checkpoint(
            "rubric-opd",
            args.save_checkpoint.as_deref(),
            args.save_every,
            round + 1,
            args.rounds,
            student_dir,
            &student,
            &mut store,
            &mut ckpt_tape,
        )?;
    }

    eprintln!("[arle train rubric-opd] done ({} rounds)", args.rounds);
    Ok(())
}

/// Log device VRAM `(used / free / total MiB)` at an agent-OPD milestone when
/// `ARLE_OPD_VRAM_TRACE` is set (default off — zero overhead on the hot path).
/// Attributes the resident-floor + writeback transient breakdown without
/// `nvidia-smi`. Uses the train backend's bound CUDA context (the single-GPU
/// agent-OPD device), so `device_mem_info` reflects the whole-process resident.
#[cfg(feature = "cuda")]
fn log_opd_vram(label: &str, backend: &std::sync::Arc<dyn autograd::Backend>) {
    if std::env::var("ARLE_OPD_VRAM_TRACE").is_err() {
        return;
    }
    match backend.device_mem_info() {
        Some((free, total)) => {
            let used = total.saturating_sub(free);
            eprintln!(
                "[opd-vram] {label}: used={}MiB free={}MiB total={}MiB hoarded={}",
                used >> 20,
                free >> 20,
                total >> 20,
                train::opd::fmt_hoarded(backend.hoarded_mib()),
            );
        }
        None => eprintln!("[opd-vram] {label}: device_mem_info unavailable"),
    }
}

/// Held-out eval via the cc harness: ONE sample per task (K=1 — best-of-N would
/// inflate the pass-rate vs the single-shot production setting), no training.
/// Writes the same `eval_round_{label}.jsonl` shape as the in-house eval pass
/// and returns the pass-rate. Sampling params are CC's own (the harness has no
/// temperature knob — cc drives the API request).
#[cfg(feature = "cuda")]
fn run_cc_eval(
    harness: &train::cc_harness::CcHarness,
    eval_tasks: &[(std::sync::Arc<train::swe_dataset::SweTask>, PathBuf)],
    out_dir: &Path,
    label: &str,
    concurrency: usize,
) -> Result<f32> {
    use std::io::Write;
    use std::sync::atomic::{AtomicUsize, Ordering};

    if eval_tasks.is_empty() {
        return Ok(0.0);
    }
    fs::create_dir_all(out_dir)
        .with_context(|| format!("create eval out dir {}", out_dir.display()))?;
    let out_path = out_dir.join(format!("eval_round_{label}.jsonl"));
    let mut file = fs::File::create(&out_path)
        .with_context(|| format!("create eval out {}", out_path.display()))?;

    // Bounded task-level concurrency: N workers steal task indices and send each
    // result back tagged with its index, so the output order matches the serial
    // path exactly. thread::scope joins all workers before the channel is drained.
    let concurrency = concurrency.clamp(1, eval_tasks.len());
    let next = AtomicUsize::new(0);
    let next = &next; // shared by ref so each `move` worker copies the ref, not the counter
    type EvalOut = Result<(serde_json::Value, bool, bool, f32)>;
    let (tx, rx) = std::sync::mpsc::channel::<(usize, EvalOut)>();
    std::thread::scope(|scope| {
        for _ in 0..concurrency {
            let tx = tx.clone();
            scope.spawn(move || {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some((task, staged)) = eval_tasks.get(i) else {
                        break;
                    };
                    let out = harness
                        .run_group(harness.boot_group(task, staged, 1))
                        .map(|g| {
                            let s = &g.samples[0];
                            let line = serde_json::json!({
                                "instance_id": s.task_id,
                                "passed": s.passed(),
                                "edited": s.edited,
                                "note": s.note,
                                "reward": s.reward,
                            });
                            (line, s.passed(), s.edited, s.reward)
                        });
                    let _ = tx.send((i, out));
                }
            });
        }
    });
    drop(tx); // workers (and their clones) are joined; drop ours so rx ends
    let mut results: Vec<Option<EvalOut>> = (0..eval_tasks.len()).map(|_| None).collect();
    for (i, out) in rx {
        results[i] = Some(out);
    }
    let (mut passed, mut edited, mut dense_sum) = (0usize, 0usize, 0.0f32);
    for out in results {
        let (line, p, e, r) = out.expect("every eval task produced a result")?;
        passed += usize::from(p);
        edited += usize::from(e);
        dense_sum += r;
        writeln!(file, "{line}")?;
    }
    let pass_rate = passed as f32 / eval_tasks.len() as f32;
    let mean_dense = dense_sum / eval_tasks.len() as f32;
    let agg = serde_json::json!({
        "aggregate": true,
        "label": label,
        "pass_rate": pass_rate,
        "mean_dense": mean_dense,
        "passed": passed,
        "edited": edited,
        "tasks": eval_tasks.len(),
    });
    writeln!(file, "{agg}")?;
    file.flush()?;
    eprintln!(
        "[arle train agent-opd] eval[{label}]: held-out pass_rate={pass_rate:.4} mean_dense={mean_dense:.4} ({passed}/{} tasks) -> {}",
        eval_tasks.len(),
        out_path.display(),
    );
    Ok(pass_rate)
}

/// Drain the serve engine before the weight re-merge / KV-pool drop. The
/// group's cc children have exited by the time this runs, so any request still
/// in flight is an orphan by definition (its client is dead) — cancel them all,
/// then REQUIRE active_requests == 0: a live request past this point reads
/// stale or freed engine state (an engine-thread panic).
#[cfg(feature = "cuda")]
fn quiesce_serve(
    engine: &std::sync::Arc<std::sync::Mutex<infer_api::LoadedInferenceEngine>>,
) -> Result<()> {
    use infer_api::InferenceEngine as _;

    let lock = || {
        engine
            .lock()
            .map_err(|err| anyhow!("LoadedInferenceEngine lock poisoned: {err}"))
    };
    let cancelled = lock()?.quiesce_admissions()?;
    if cancelled > 0 {
        eprintln!("[agent-opd] quiesce: cancelled {cancelled} orphaned request(s)");
    }
    let started = Instant::now();
    let mut next_warn = 10u64;
    loop {
        let active = lock()?.telemetry().active_requests;
        if active == 0 {
            return Ok(());
        }
        let elapsed = started.elapsed().as_secs();
        anyhow::ensure!(
            elapsed < 60,
            "quiesce: {active} request(s) still active {elapsed}s after cancellation — \
             refusing to proceed (a live request during weight re-merge / KV drop \
             corrupts the engine)"
        );
        if elapsed >= next_warn {
            eprintln!("[agent-opd] quiesce: {active} request(s) still active after {elapsed}s");
            next_warn += 10;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}

/// One task group's pending rollout (`--staleness` dial). Staleness 0 keeps
/// today's boot-ahead: sandboxes build in the background, the rollout itself
/// runs inline at collect (strictly on-policy). Staleness 1 runs the WHOLE
/// rollout on a background thread launched before the previous group's
/// train+merge, tagged with the policy version it launched under.
#[cfg(feature = "cuda")]
enum PendingGroup {
    Booted(train::cc_harness::BootedGroup),
    Rolling {
        behavior_version: u64,
        handle: std::thread::JoinHandle<Result<train::cc_harness::CcGroup>>,
    },
}

/// Append-only JSONL metrics sink (`--metrics-out`); write failures log and
/// never abort training.
#[cfg(feature = "cuda")]
struct JsonlSink {
    path: PathBuf,
}

#[cfg(feature = "cuda")]
impl JsonlSink {
    fn new(path: PathBuf) -> Self {
        Self { path }
    }

    fn append(&self, row: &serde_json::Value) {
        let write = || -> Result<()> {
            use std::io::Write;
            if let Some(parent) = self.path.parent() {
                fs::create_dir_all(parent)?;
            }
            let mut file = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.path)?;
            writeln!(file, "{row}")?;
            Ok(())
        };
        if let Err(err) = write() {
            eprintln!(
                "[agent-opd] metrics write to {} failed: {err}",
                self.path.display()
            );
        }
    }
}

/// PEFT adapter config for `--save-lora-adapters` (mainstream HF PEFT dir:
/// adapter_config.json + adapter_model.safetensors).
#[cfg(feature = "cuda")]
fn agent_opd_adapter_config(
    student_dir: &Path,
    target_set: train::lora::LoraTargetSet,
    lora: train::lora::LoraConfig,
) -> train::lora::LoraAdapterConfig {
    use train::lora::{LoraAdapterConfig, LoraTargetSet};

    let mut config = LoraAdapterConfig::new(student_dir.display().to_string(), "qwen35", lora);
    config.target_modules = match target_set {
        LoraTargetSet::AttentionQv => vec!["q_proj".to_owned(), "v_proj".to_owned()],
        LoraTargetSet::AttentionFull => vec![
            "q_proj".to_owned(),
            "k_proj".to_owned(),
            "v_proj".to_owned(),
            "o_proj".to_owned(),
            "in_proj_qkv".to_owned(),
            "out_proj".to_owned(),
        ],
        LoraTargetSet::AllLinear => vec!["all-linear".to_owned()],
    };
    config
}

/// Fast adapter-only (LoRA) save as a mainstream HF PEFT adapter dir — shared
/// by the per-round agent-OPD save and the replay-mode final save. Avoids the
/// full-materialize host-loop hang; loadable by HF PEFT / vLLM / SGLang.
#[cfg(feature = "cuda")]
fn save_agent_opd_adapters(
    adapter_dir: &Path,
    dirname: &str,
    step: usize,
    student_dir: &Path,
    student: &train::qwen35::Qwen35Model,
    store: &mut TensorStore,
    adapter_config: &train::lora::LoraAdapterConfig,
) -> Result<()> {
    use train::qwen35_checkpoint::{
        ConfigJsonSource, GenerationConfigSource, Qwen35NamedCheckpoint, Qwen35StudentWeights,
        save_named_qwen35_student_checkpoint,
    };

    fs::create_dir_all(adapter_dir)
        .with_context(|| format!("create LoRA adapter dir {}", adapter_dir.display()))?;
    let sources = checkpoint_sources(student_dir)?;
    let started = Instant::now();
    let mut adapter_tape = Tape::new();
    let saved_dir = save_named_qwen35_student_checkpoint(
        Qwen35NamedCheckpoint {
            out_dir: adapter_dir,
            dirname,
            tokenizer_path: Some(&sources.tokenizer_path),
            config_json: ConfigJsonSource::CopyFrom(&sources.config_path),
            generation_config: GenerationConfigSource::CopyOrSynthesize {
                source_path: &sources.generation_config_path,
                fallback_config_path: &sources.config_path,
            },
        },
        student,
        store,
        &mut adapter_tape,
        Qwen35StudentWeights::AdapterOnly {
            bf16: true,
            adapter_config,
        },
    )
    .with_context(|| format!("save LoRA PEFT adapter dir {dirname}"))?;
    println!(
        "checkpoint_saved kind=peft_adapter mode=agent-opd step={step} dir={} seconds={:.6}",
        saved_dir.display(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

/// One `--replay-records` JSONL row (`arle train cc-convert` output).
#[cfg(feature = "cuda")]
#[derive(Clone, serde::Deserialize)]
struct ReplayRecord {
    label: Option<String>,
    prompt_ids: Vec<u32>,
    response_ids: Vec<u32>,
    response_mask: Vec<u8>,
    /// Generation-time behavior logprobs (one per masked token).
    gen_logprobs: Option<Vec<f32>>,
    /// Attempt reward for SAO advantage; pre-reward records (CE-only flow)
    /// default to 1.0 = a passing trajectory, keeping rejection-CE unchanged.
    #[serde(default = "replay_default_reward")]
    reward: f32,
    /// Budget/timeout/error artifact — excluded by the DAPO `DropTruncated`
    /// filter. Older dumps without the field default to false.
    #[serde(default)]
    truncated: bool,
}

#[cfg(feature = "cuda")]
fn replay_default_reward() -> f32 {
    1.0
}

#[cfg(feature = "cuda")]
fn replay_groups(
    preset: train::update_strategy::UpdatePreset,
    records: &[ReplayRecord],
    per_epoch_cap: usize,
) -> Vec<Vec<train::update_strategy::ScoredTrajectory>> {
    use train::update_strategy::ScoredTrajectory;

    let mut all_groups: Vec<(&str, Vec<ScoredTrajectory>)> = Vec::new();
    for record in records {
        let key = task_key(record.label.as_deref());
        let group_id = all_groups
            .iter()
            .position(|(group_key, _)| *group_key == key)
            .unwrap_or_else(|| {
                all_groups.push((key, Vec::new()));
                all_groups.len() - 1
            });
        all_groups[group_id].1.push(ScoredTrajectory {
            prompt_ids: record.prompt_ids.clone(),
            response_ids: record.response_ids.clone(),
            response_mask: record.response_mask.clone(),
            reward: record.reward,
            behavior_logprobs: record.gen_logprobs.clone(),
            group_id,
            truncated: record.truncated,
        });
    }

    let mut used = 0usize;
    let mut selected = Vec::new();
    for (_, group) in all_groups {
        let trainable = preset.planned_training_count(&group);
        if trainable == 0 {
            continue;
        }
        if trainable > per_epoch_cap.saturating_sub(used) {
            break;
        }
        used += trainable;
        selected.push(group);
    }
    selected
}

/// Task key for SAO grouping: the label prefix before `#` (`iid#sample` → `iid`).
#[cfg(feature = "cuda")]
fn task_key(label: Option<&str>) -> &str {
    label.unwrap_or("").split('#').next().unwrap_or("")
}

/// A replay record is CE-trainable iff its reward cleared the bar and it has at
/// least one masked target token. Imitating a failed fix (reward < 1.0) or a
/// mask-less record is actively harmful, so both the preflight batch and the CE
/// loop select on this identical predicate.
#[cfg(feature = "cuda")]
fn ce_trainable(record: &ReplayRecord) -> bool {
    record.reward >= 1.0 && record.response_mask.iter().any(|&mask| mask == 1)
}

/// PG-preset replay: group cc records by task and apply the same
/// [`train::update_strategy::UpdatePreset::update`] the online path uses. The
/// generation-time sidecar is the sole behavior denominator.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn replay_pg(
    preset: train::update_strategy::UpdatePreset,
    args: &TrainAgentOpdArgs,
    groups: &[Vec<train::update_strategy::ScoredTrajectory>],
    student: &train::qwen35::Qwen35Model,
    all_params: &[autograd::TensorId],
    trainable: &[autograd::TensorId],
    optimizer: &mut autograd::optim::AdamW,
    vocab: usize,
    epochs: usize,
    store: &mut autograd::TensorStore,
) -> Result<()> {
    let mut value_critic =
        build_value_critic(&preset, student.config().hidden_size, args.value_lr, store)?;

    eprintln!(
        "[agent-opd] replay PG: preset={preset:?} groups={}",
        groups.len()
    );

    for epoch in 0..epochs {
        let mut losses = Vec::new();
        for batch in groups {
            let report = preset
                .update(
                    batch,
                    student,
                    all_params,
                    trainable,
                    optimizer,
                    value_critic.as_mut(),
                    vocab,
                    args.writeback_window,
                    store,
                )
                .map_err(anyhow::Error::from)?;
            losses.push(report.loss);
            // #45: freed writeback blocks hoarded in the device pool starve
            // the next group's capture forward (38–87 GB oscillation, dies
            // entering group 3); return them per group.
            if let Err(err) = store.backend().trim_memory_pool() {
                eprintln!("[agent-opd] replay device-pool trim failed: {err}");
            }
        }
        let mean_loss = if losses.is_empty() {
            0.0
        } else {
            losses.iter().sum::<f32>() / losses.len() as f32
        };
        eprintln!(
            "[agent-opd] replay PG epoch={epoch} groups={} mean_loss={mean_loss:.4}",
            losses.len()
        );
    }
    Ok(())
}

/// Agent-OPD replay mode: run the SAME masked-CE writeback the round loop
/// passes as `train_on_accepted` over pre-converted token records — no rollout
/// engine, sandboxes, or datasets (no engine VRAM, no staged trees).
#[cfg(feature = "cuda")]
fn run_agent_opd_replay(
    args: &TrainAgentOpdArgs,
    records_path: &Path,
    lora: train::lora::LoraConfig,
    target_set: train::lora::LoraTargetSet,
) -> Result<()> {
    use autograd::optim::AdamW;
    use train::{
        ema_self_teacher::EmaSelfTeacher,
        opd::{gkd_writeback_step, masked_writeback_ce_step_dispatch},
        qwen35_checkpoint::load_qwen35_lora_adapters,
        qwen35_loader::load_qwen35_lora_from_hf_dir_with_shared_base,
    };

    use crate::args::GkdTeacherArg;

    let student_dir = args.student_model.as_path();
    let raw = fs::read_to_string(records_path)
        .with_context(|| format!("read --replay-records {}", records_path.display()))?;
    let records: Vec<ReplayRecord> = raw
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            serde_json::from_str(line).with_context(|| format!("parse replay record: {line}"))
        })
        .collect::<Result<_>>()?;
    if records.is_empty() {
        bail!("no records in {}", records_path.display());
    }
    let preset = args.update_preset();
    let per_epoch_cap = args.writeback_cap.unwrap_or(usize::MAX);

    let hf_config = Qwen35Config::from_json_file(student_dir.join("config.json"))
        .with_context(|| format!("read config.json from {}", student_dir.display()))?;
    let vocab = hf_config.vocab_size;
    let replay_groups = if preset.needs().behavior_logprobs {
        let groups = replay_groups(preset, &records, per_epoch_cap);
        for batch in &groups {
            preset
                .preflight(batch, vocab, args.writeback_window)
                .map_err(anyhow::Error::from)
                .with_context(|| {
                    format!("validate replay records from {}", records_path.display())
                })?;
        }
        groups
    } else {
        let batch: Vec<_> = records
            .iter()
            .filter(|record| ce_trainable(record))
            .take(per_epoch_cap)
            .enumerate()
            .map(
                |(group_id, record)| train::update_strategy::ScoredTrajectory {
                    prompt_ids: record.prompt_ids.clone(),
                    response_ids: record.response_ids.clone(),
                    response_mask: record.response_mask.clone(),
                    reward: record.reward,
                    behavior_logprobs: record.gen_logprobs.clone(),
                    group_id,
                    truncated: record.truncated,
                },
            )
            .collect();
        preset
            .preflight(&batch, vocab, args.writeback_window)
            .map_err(anyhow::Error::from)
            .with_context(|| format!("validate replay records from {}", records_path.display()))?;
        Vec::new()
    };
    let (mut store, train_backend, backend_label) = build_opd_store(args.backend)?;
    apply_tape_dtype(&mut store, args.tape_dtype)?;

    eprintln!(
        "[arle train agent-opd] replay: {} record(s) from {} on {backend_label} (no rollout engine)",
        records.len(),
        records_path.display()
    );
    let student = load_qwen35_lora_from_hf_dir_with_shared_base(
        student_dir,
        lora,
        target_set,
        args.lora_layer_start,
        args.lora_skip_experts,
        // No rollout engine in replay mode, so no FP8 base to share.
        None,
        &mut store,
    )
    .with_context(|| format!("load LoRA student from {}", student_dir.display()))?;
    // Resume: overlay a saved adapter BEFORE the GKD self-teacher snapshots the
    // student, so the frozen `self` teacher captures the resumed adapter, not zeros.
    if let Some(dir) = args.lora_adapters.as_deref() {
        load_qwen35_lora_adapters(&student, &mut store, dir)
            .with_context(|| format!("resume LoRA adapter from {}", dir.display()))?;
        eprintln!("[agent-opd] resumed adapter from {}", dir.display());
    }
    // GKD teacher (built here — immediately after the student and BEFORE any
    // other store scratch, per EmaSelfTeacher::from_student's retain_ids contract).
    // `ema` EMA-updates each step; `self` stays frozen at the student's initial
    // adapter+base snapshot (never updated). Both reuse the EmaSelfTeacher
    // machinery — the only difference is whether `update()` runs.
    // GKD distils a teacher distribution; SAO trains on reward advantage — the
    // replay dispatch runs one OR the other, so reject the silent-no-op combo
    // (would build + hold the teacher's VRAM, then ignore it).
    if args.gkd && args.update_strategy != crate::args::UpdateStrategyArg::RejectionCe {
        bail!(
            "--gkd is incompatible with --update-strategy {:?}: pick GKD (teacher distill) or SAO (reward advantage)",
            args.update_strategy
        );
    }
    let mut gkd_teacher = if args.gkd {
        Some(
            EmaSelfTeacher::from_student(&student, lora, target_set, &mut store)
                .context("build GKD EMA self-teacher")?,
        )
    } else {
        None
    };
    let all_params: Vec<TensorId> = student.all_parameter_ids();
    let trainable: Vec<TensorId> = all_params
        .iter()
        .copied()
        .filter(|id| store.get(*id).is_some_and(|tensor| tensor.requires_grad))
        .collect();
    if trainable.is_empty() {
        bail!("agent-opd student has no trainable (LoRA) parameters; check --lora-target-set");
    }
    let mut optimizer = AdamW::new(args.lr, (0.9, 0.999), 1.0e-8, 0.0);
    if args.gkd {
        eprintln!(
            "[arle train agent-opd] replay GKD mode: teacher={:?} temperature={} \
             entropy_weight={} ema_alpha={}",
            args.gkd_teacher, args.gkd_temperature, args.gkd_entropy_weight, args.gkd_ema_alpha
        );
    }
    log_opd_vram("replay: after student load", &train_backend);

    // Default rejection-CE (incl. GKD) is byte-identical to before; ratio-
    // weighted presets dispatch to the same `UpdatePreset::update` the online
    // path uses.
    let epochs = args.replay_epochs.max(1);
    // The flat CE arm keeps the pre-preset record-order + filter-then-cap
    // semantics byte-identical and hosts the GKD teacher fork (a distribution-
    // level KL, not a per-token weight — it dies in T5 with the cc harness).
    if !preset.needs().behavior_logprobs {
        // Rejection sampling: imitate only SOLVED trajectories. Now that cc
        // collects failing attempts too (for SAO's 0-reward arm), CE must reject
        // them — imitating a failed fix is actively harmful. reward == 1.0 iff all
        // fail_to_pass tests pass; pre-reward records default to 1.0 (unchanged).
        for epoch in 0..epochs {
            let mut losses = Vec::new();
            for record in records
                .iter()
                .filter(|r| ce_trainable(r))
                .take(per_epoch_cap)
            {
                let loss = if let Some(ema) = gkd_teacher.as_mut() {
                    // GKD: distil the teacher's per-position distribution on the same
                    // masked trajectory tokens (forward-KL), then (ema only) nudge the
                    // EMA teacher toward the just-updated student.
                    let step_loss = {
                        let teacher = ema.as_teacher();
                        gkd_writeback_step(
                            &student,
                            &teacher,
                            all_params.as_slice(),
                            trainable.as_slice(),
                            &mut optimizer,
                            &record.prompt_ids,
                            &record.response_ids,
                            &record.response_mask,
                            vocab,
                            args.writeback_window,
                            args.gkd_temperature,
                            args.gkd_entropy_weight,
                            &mut store,
                        )
                        .with_context(|| format!("replay GKD writeback ({:?})", record.label))?
                    };
                    if matches!(args.gkd_teacher, GkdTeacherArg::Ema) {
                        ema.update(&student, &mut store, args.gkd_ema_alpha)
                            .context("GKD EMA teacher update")?;
                    }
                    step_loss
                } else {
                    masked_writeback_ce_step_dispatch(
                        &student,
                        all_params.as_slice(),
                        trainable.as_slice(),
                        &mut optimizer,
                        &record.prompt_ids,
                        &record.response_ids,
                        &record.response_mask,
                        vocab,
                        args.writeback_window,
                        train::context_parallel::CpContext::from_env(),
                        train::context_parallel::DpContext::from_env(),
                        &mut store,
                    )
                    .with_context(|| format!("replay writeback ({:?})", record.label))?
                };
                losses.push(loss);
            }
            let mean_loss = if losses.is_empty() {
                0.0
            } else {
                losses.iter().sum::<f32>() / losses.len() as f32
            };
            eprintln!(
                "[agent-opd] replay epoch={epoch} trained_pairs={} mean_loss={mean_loss:.4}",
                losses.len()
            );
        }
    } else {
        replay_pg(
            preset,
            args,
            &replay_groups,
            &student,
            all_params.as_slice(),
            trainable.as_slice(),
            &mut optimizer,
            vocab,
            epochs,
            &mut store,
        )?;
    }
    log_opd_vram("replay: after writeback", &train_backend);

    if let Some(adapter_dir) = args.save_lora_adapters.as_deref() {
        let adapter_config = agent_opd_adapter_config(student_dir, target_set, lora);
        save_agent_opd_adapters(
            adapter_dir,
            "adapters_replay",
            epochs,
            student_dir,
            &student,
            &mut store,
            &adapter_config,
        )?;
    }
    eprintln!("[arle train agent-opd] replay done ({epochs} epoch(s))");
    Ok(())
}

#[cfg(feature = "cuda")]
fn validate_online_rollout_temperature(
    preset: train::update_strategy::UpdatePreset,
    strategy: crate::args::UpdateStrategyArg,
    temperature: f32,
) -> Result<()> {
    if preset.needs().behavior_logprobs && temperature <= 0.0 {
        bail!(
            "--rollout-temperature must be > 0 for ratio-weighted --update-strategy {strategy:?}; \
             greedy rollout does not produce behavior logprobs (rejection-ce permits greedy)"
        );
    }
    Ok(())
}

/// Agent-OPD RFT loop: the student drives the read/write/replace/bash tool loop
/// against a per-task repo sandbox (SWE-bench-Pro); the reward is EXECUTION (the
/// hidden tests are run on `git diff`), no text judge is loaded; passing
/// trajectories are written back as CE targets via the same path as rubric-OPD.
#[cfg(feature = "cuda")]
/// Directional finite difference on one param, stepping along its own analytic
/// gradient so ΔL = 2ε‖g‖ — a single-scalar probe drowns in bf16 loss noise.
/// `fd/‖g‖` near 1 means that arm's gradient is right; a global norm can only say
/// cp=1 and cp=2 disagree, not which one is wrong (#85).
#[allow(clippy::too_many_arguments)]
fn synthetic_writeback_fd_probe<O: autograd::Optimizer>(
    target: &str,
    student: &train::qwen35::Qwen35Model,
    all_params: &[autograd::TensorId],
    trainable: &[autograd::TensorId],
    optimizer: &mut O,
    prompt_ids: &[u32],
    response_ids: &[u32],
    response_mask: &[u8],
    vocab: usize,
    window_size: usize,
    cp: train::context_parallel::CpContext,
    dp: train::context_parallel::DpContext,
    store: &mut autograd::TensorStore,
) -> Result<()> {
    use train::opd::{WritebackLoss, masked_writeback_step};

    // The caller ran with step_optimizer=false to keep the weights put, and that is
    // the same branch that skips the CP/DP reduce (opd.rs) — so without this the
    // probe would read this rank's un-reduced shard, not the global gradient.
    if cp.is_enabled() || dp.is_enabled() {
        train::grad_clip::all_reduce_cp_grads(trainable, store)?;
    }
    let param = student
        .param_name_map()
        .into_iter()
        .chain(student.adapter_name_map())
        .find(|(name, _)| *name == target)
        .map(|(_, id)| id)
        .with_context(|| format!("ARLE_OPD_FD_PARAM: no param named {target}"))?;
    let grad = store
        .get(param)
        .and_then(|tensor| tensor.grad)
        .with_context(|| format!("{target} has no gradient"))?;
    let direction = store.to_host(grad)?;
    let norm = direction
        .iter()
        .map(|&v| f64::from(v) * f64::from(v))
        .sum::<f64>()
        .sqrt();
    if norm == 0.0 {
        anyhow::bail!("{target} gradient is exactly zero — pick a param with signal");
    }
    let base = store.to_host(param)?;
    let shape = store
        .get(param)
        .with_context(|| format!("{target} is not in the store"))?
        .shape
        .clone();
    let eps: f32 = std::env::var("ARLE_OPD_FD_EPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1e-2);

    let mut probes = [0.0f32; 2];
    for (probe, sign) in probes.iter_mut().zip([1.0f32, -1.0]) {
        let step = sign * eps / norm as f32;
        let perturbed: Vec<f32> = base
            .iter()
            .zip(&direction)
            .map(|(&w, &g)| w + step * g)
            .collect();
        let handle = store.backend().upload(&perturbed, &shape)?;
        store.replace_device_handle(param, handle)?;
        *probe = masked_writeback_step(
            WritebackLoss::Ce,
            student,
            all_params,
            trainable,
            optimizer,
            false,
            prompt_ids,
            response_ids,
            response_mask,
            vocab,
            window_size,
            cp,
            dp,
            store,
        )?
        .0;
        optimizer.zero_grad(store, all_params);
    }
    let handle = store.backend().upload(&base, &shape)?;
    store.replace_device_handle(param, handle)?;

    let fd = f64::from(probes[0] - probes[1]) / (2.0 * f64::from(eps));
    eprintln!(
        "[param-fd] {target} analytic={norm:.6e} fd={fd:.6e} ratio={:.4} eps={eps:.1e} \
         loss_plus={:.6} loss_minus={:.6}",
        fd / norm,
        probes[0],
        probes[1]
    );
    Ok(())
}

/// Rollout engine + cc serve + autograd student for one agent-opd rank. Every
/// mesh rank loads this (the cp fleet serves one group's samples in parallel);
/// rank 0 additionally owns the harness, filtering, and saves.
#[cfg(feature = "cuda")]
struct AgentOpdServeStudent {
    store: autograd::TensorStore,
    train_backend: std::sync::Arc<dyn autograd::Backend>,
    vocab: usize,
    infer_student: train::infer_student::InferStudent,
    student: train::qwen35::Qwen35Model,
    serve_thread: infer_api::ServeThread,
    dump_dir: PathBuf,
    cc_model_id: String,
    all_params: Vec<TensorId>,
    trainable: Vec<TensorId>,
}

#[cfg(feature = "cuda")]
fn load_agent_opd_serve_student(
    args: &TrainAgentOpdArgs,
    lora: train::lora::LoraConfig,
    target_set: train::lora::LoraTargetSet,
    serve_port: u16,
) -> Result<AgentOpdServeStudent> {
    use std::sync::{Arc, Mutex};

    use infer_api::{EngineLoadConfig, LoadedInferenceEngine};
    use train::{
        infer_student::InferStudent,
        qwen35_checkpoint::load_qwen35_lora_adapters,
        qwen35_loader::{SharedFrozenBaseEntry, load_qwen35_lora_from_hf_dir_with_shared_base},
    };

    let student_dir = args.student_model.as_path();
    let (mut store, train_backend, _backend_label) = build_opd_store(args.backend)?;
    apply_tape_dtype(&mut store, args.tape_dtype)?;
    // Vocab from the checkpoint config (not the autograd student) so the rollout
    // engine can load BEFORE the autograd student when `--share-frozen-base` is
    // set. Same value `Qwen35Model::config()` exposes.
    let hf_config = Qwen35Config::from_json_file(student_dir.join("config.json"))
        .with_context(|| format!("read config.json from {}", student_dir.display()))?;
    let vocab = hf_config.vocab_size;
    let eval_out_dir = agent_opd_eval_out_dir(args);
    // Rollout engine (student) doubles as the cc serve. KV budget for cc
    // traffic: K concurrent cc streams × the measured session ceiling, +25%
    // headroom, page_size 16.
    use train::cc_harness::{CC_MAX_SESSION_TOKENS, CC_SESSION_TOKENS};
    // Sessions THIS rank serves at once. Round-robin restarts per group
    // (`sample % base_urls.len()`), so every concurrent group hands this rank
    // its own ceil(K/cp) — G × ceil(K/cp), not ceil(G×K/cp).
    let width = args.prompts_per_update.max(1)
        * args
            .samples_per_prompt
            .max(1)
            .div_ceil(train::context_parallel::CpContext::from_env().size.max(1));
    // Pool = K typical streams, but never smaller than one full long-horizon
    // session (+25% headroom each) so a 200K session stays schedulable.
    let cc_pages = (width * CC_SESSION_TOKENS)
        .max(CC_MAX_SESSION_TOKENS)
        .div_ceil(16);
    let cc_total_pages = cc_pages + cc_pages / 4;

    // Load order: default loads the autograd student FIRST (engine then sees
    // post-student free VRAM — byte-identical). --share-frozen-base loads the
    // engine FIRST so the student can import (zero-copy) its resident FP8 base.
    let prebuilt_student = if !args.no_share_frozen_base {
        None
    } else {
        eprintln!(
            "[arle train agent-opd] loading student from {}",
            student_dir.display()
        );
        Some(
            load_qwen35_lora_from_hf_dir_with_shared_base(
                student_dir,
                lora,
                target_set,
                args.lora_layer_start,
                args.lora_skip_experts,
                None,
                &mut store,
            )
            .with_context(|| format!("load LoRA student from {}", student_dir.display()))?,
        )
    };

    eprintln!(
        "[arle train agent-opd] loading rollout engine from {} (slots={width} pages={cc_total_pages})",
        student_dir.display()
    );
    let student_engine = LoadedInferenceEngine::load_with_config(
        student_dir
            .to_str()
            .ok_or_else(|| anyhow!("student path is not valid UTF-8"))?,
        // agent-OPD: decode CUDA-graph default-OFF. Its captured workspace (~30 GB
        // on the 27B MoE, captured during the rollout's decode) would co-reside
        // with the masked-CE writeback and OOM it (post-rollout engine ~87 GB vs
        // ~55 GB no-graph). A measured ~26 GB headroom made it worth exposing as
        // --qwen35-decode-graph; the default flip waits for the co-residency license.
        args.runtime.qwen35_decode_graph,
        EngineLoadConfig {
            num_slots: width,
            page_size: 16,
            total_pages: cc_total_pages,
            // Request caps are POOL-derived, not per-session: capping at
            // CC_SESSION_TOKENS silently aborted cc mid-conversation (a
            // tripwire — sidecars showed prompt>22K → gen 0). The pool bounds
            // memory and the engine preempts under KV pressure (#162).
            max_prompt_tokens: cc_total_pages * 16 - 256,
            max_total_tokens: cc_total_pages * 16,
            chunked_prefill_size: Some(CC_SESSION_TOKENS),
            mem_fraction_static: args.runtime.rollout_mem_fraction,
            dspark_draft_model: args.runtime.dspark_draft_model.clone(),
            dspark_sps_bias_ms: args.runtime.dspark_sps_bias_ms,
            dspark_sps_row_ms: args.runtime.dspark_sps_row_ms,
            // MTP GPU gate passed 2026-07-17 (1.21×); default-on waits for the
            // depth sweep + an in-loop A/B.
            mtp_draft_tokens: args.runtime.mtp_draft_tokens,
            cuda: args.runtime.cuda_runtime_flags(),
            ..EngineLoadConfig::default()
        },
    )
    .with_context(|| format!("load rollout engine from {}", student_dir.display()))?;
    log_opd_vram(
        "after rollout engine load (KV pool alloc'd)",
        &train_backend,
    );

    // cc serve over THIS engine (same engine thread, same KV pool): install the
    // dump sink BEFORE any traffic, then serve the router on a background
    // thread. Token-only shutdown (`serve_thread.shutdown()` at run end).
    // Every fleet rank shares ONE dump dir (filenames are pid-tagged).
    let dump_dir = eval_out_dir.join("dumps");
    infer_api::set_messages_dump_dir(&dump_dir)
        .with_context(|| format!("create cc dump dir {}", dump_dir.display()))?;
    let cc_model_id = infer_api::InferenceEngine::model_id(&student_engine).to_owned();
    // Rollout = flag temperature (>0 keeps behavior logprobs non-empty) +
    // model nucleus from generation_config (truncates the tail — no salad).
    let mut sampling_defaults = infer_api::SamplingDefaults::from_generation_config(student_dir);
    sampling_defaults.temperature = args.rollout_temperature;
    infer_api::set_sampling_defaults(sampling_defaults);
    let serve_thread = infer_api::serve_router_on_thread(
        student_engine.local_router(0)?, // thinking unbounded — the serve default
        "127.0.0.1",
        serve_port,
    )?;
    eprintln!(
        "[arle train agent-opd] cc serve on http://127.0.0.1:{} (model={cc_model_id}, dumps={})",
        serve_port,
        dump_dir.display()
    );

    // Train-infer FP8 weight sharing (`--share-frozen-base`): borrow the rollout
    // engine's resident FP8 base pointers and pass them into the autograd student
    // load so its frozen FP8 base projections import a NON-OWNING view.
    let shared_base_entries: Vec<SharedFrozenBaseEntry> = if !args.no_share_frozen_base {
        let table = student_engine
            .frozen_base_fp8_pointers()
            .context("borrow rollout-engine FP8 base pointers for --share-frozen-base")?;
        eprintln!(
            "[arle train agent-opd] --share-frozen-base: borrowing {} resident FP8 base projections from the rollout engine (zero-copy)",
            table.len()
        );
        table
            .into_iter()
            .map(|p| SharedFrozenBaseEntry {
                layer_idx: p.layer_idx,
                proj_suffix: p.proj_suffix,
                weight_ptr: p.weight_ptr,
                scale_ptr: p.scale_ptr,
                rows: p.rows,
                cols: p.cols,
                block_m: p.block_m,
                block_k: p.block_k,
            })
            .collect()
    } else {
        Vec::new()
    };
    let shared_base = if !args.no_share_frozen_base {
        Some(shared_base_entries.as_slice())
    } else {
        None
    };

    let student = match prebuilt_student {
        Some(s) => s,
        None => {
            eprintln!(
                "[arle train agent-opd] loading student from {}",
                student_dir.display()
            );
            load_qwen35_lora_from_hf_dir_with_shared_base(
                student_dir,
                lora,
                target_set,
                args.lora_layer_start,
                args.lora_skip_experts,
                shared_base,
                &mut store,
            )
            .with_context(|| format!("load LoRA student from {}", student_dir.display()))?
        }
    };

    // Resume: overlay a saved adapter onto the fresh student (both load branches
    // merge here) BEFORE the handoff fence, so its A/B uploads drain with the base.
    if let Some(dir) = args.lora_adapters.as_deref() {
        load_qwen35_lora_adapters(&student, &mut store, dir)
            .with_context(|| format!("resume LoRA adapter from {}", dir.display()))?;
        eprintln!("[agent-opd] resumed adapter from {}", dir.display());
    }

    // Shared-base bytes alias the engine's resident FP8 — drain the autograd
    // backend's OWN in-flight uploads before the first autograd forward
    // (cross-stream handoff fence). MUST be a stream-scoped sync, NOT a
    // context-wide `cuCtxSynchronize`: the co-resident rollout engine shares this
    // device primary context but runs its streams with cudarc event-tracking
    // DISABLED and idle-parked between scheduler steps, so a `cuCtxSynchronize`
    // here blocks forever in poll() draining the engine's never-host-progressed
    // streams (measured deadlock — main thread poll(nfds=2,-1), GPU flat, all
    // 851 student tensors already materialized). The engine's resident FP8 base
    // weights are fully written by its own load+warmup before this point, so the
    // borrow only needs the student's own upload stream drained.
    if !args.no_share_frozen_base {
        let opd_load_trace = std::env::var("ARLE_OPD_LOAD_TRACE").is_ok();
        if opd_load_trace {
            eprintln!("[opd-load-trace] pre stream-sync (share-frozen-base handoff fence)");
        }
        train_backend
            .stream_synchronize()
            .context("stream sync before first shared-base autograd forward")?;
        if opd_load_trace {
            eprintln!("[opd-load-trace] post stream-sync OK; building InferStudent");
        }
    }

    let all_params: Vec<TensorId> = student.all_parameter_ids();
    let trainable: Vec<TensorId> = all_params
        .iter()
        .copied()
        .filter(|id| store.get(*id).is_some_and(|tensor| tensor.requires_grad))
        .collect();
    if trainable.is_empty() {
        bail!("agent-opd student has no trainable (LoRA) parameters; check --lora-target-set");
    }

    let infer_student = InferStudent::new(
        Arc::new(Mutex::new(student_engine)),
        train_backend.clone(),
        vocab,
    );
    log_opd_vram(
        "after autograd student load (resident floor)",
        &train_backend,
    );

    Ok(AgentOpdServeStudent {
        store,
        train_backend,
        vocab,
        infer_student,
        student,
        serve_thread,
        dump_dir,
        cc_model_id,
        all_params,
        trainable,
    })
}

/// Rank-0 → follower stream for cp>1 agent-opd. Stochastic cc rollouts diverge
/// across mesh ranks, so rank 0 owns the harness + filtering and publishes
/// every update's batch plus the engine-lifecycle decisions around it;
/// followers serve rollouts and mirror the calls, keeping the writeback's cp
/// collectives on identical call sequences. Files live under the
/// coordinator-minted `ARLE_TRAIN_MESH_DIR`; publish is write-then-rename so a
/// reader never sees a partial file.
#[cfg(feature = "cuda")]
#[derive(serde::Serialize, serde::Deserialize)]
enum MeshMsg {
    Update {
        batch: Vec<train::update_strategy::ScoredTrajectory>,
        /// Mirror of the leader's quiesce + scratch/KV release before this
        /// update (skipped under staleness>0 with a group in flight).
        release_engines: bool,
    },
    /// End of one group's updates: re-merge LoRA when the leader did, then
    /// re-acquire the KV pool for the next rollout.
    GroupEnd { synced: bool },
}

/// Borrow twin of [`MeshMsg`] for the publish side — same serde shape, no
/// batch clone.
#[cfg(feature = "cuda")]
#[derive(serde::Serialize)]
enum MeshMsgRef<'a> {
    Update {
        batch: &'a [train::update_strategy::ScoredTrajectory],
        release_engines: bool,
    },
    GroupEnd {
        synced: bool,
    },
}

#[cfg(feature = "cuda")]
struct MeshUpdateChannel {
    dir: PathBuf,
    seq: u64,
    /// Next consumed file the GC may delete (last cp rank only).
    gc_next: u64,
}

#[cfg(feature = "cuda")]
impl MeshUpdateChannel {
    fn from_env() -> Result<Self> {
        let dir = std::env::var_os("ARLE_TRAIN_MESH_DIR").ok_or_else(|| {
            anyhow!("cp>1 agent-opd needs the mesh coordinator (ARLE_TRAIN_MESH_DIR unset)")
        })?;
        Ok(Self {
            dir: dir.into(),
            seq: 0,
            gc_next: 0,
        })
    }

    fn upd_path(&self, seq: u64) -> PathBuf {
        self.dir.join(format!("upd-{seq:08}.json"))
    }

    fn publish<M: serde::Serialize>(&mut self, msg: &M) -> Result<()> {
        let tmp = self.dir.join(format!("upd-{:08}.tmp", self.seq));
        std::fs::write(&tmp, serde_json::to_vec(msg)?)?;
        std::fs::rename(&tmp, self.upd_path(self.seq))?;
        self.seq += 1;
        Ok(())
    }

    /// Delete every consumed file. Call ONLY right after an `Update` completed:
    /// its collective proves every rank has consumed all files below `seq`.
    /// (`GroupEnd` carries no collective, so it proves nothing — an eager
    /// delete there deadlocks a slower rank at cp>=3.)
    fn gc_consumed(&mut self) {
        while self.gc_next < self.seq {
            let _ = std::fs::remove_file(self.upd_path(self.gc_next));
            self.gc_next += 1;
        }
    }

    fn finish(&self) -> Result<()> {
        std::fs::write(self.dir.join("end"), b"").map_err(Into::into)
    }

    /// Blocks until the next message (`Some`) or the end marker (`None`).
    fn recv(&mut self) -> Result<Option<MeshMsg>> {
        loop {
            match std::fs::read(self.upd_path(self.seq)) {
                Ok(bytes) => {
                    self.seq += 1;
                    return Ok(Some(serde_json::from_slice(&bytes)?));
                }
                Err(_) => {
                    if self.dir.join("end").exists() {
                        return Ok(None);
                    }
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
            }
        }
    }
}

/// Eval dumps land in --eval-out-dir, else next to the LoRA adapters / full
/// checkpoint, else the current dir. Pure function of args so every fleet rank
/// derives the same shared dump dir.
#[cfg(feature = "cuda")]
fn agent_opd_eval_out_dir(args: &TrainAgentOpdArgs) -> PathBuf {
    args.eval_out_dir
        .clone()
        .or_else(|| args.save_lora_adapters.clone())
        .or_else(|| args.save_checkpoint.clone())
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Value critic (ValueGae presets): built after the `trainable` filter and not
/// a student param, so the policy optimizer / LoRA sync / adapter save never
/// touch it — fully isolated, own AdamW.
#[cfg(feature = "cuda")]
fn build_value_critic(
    preset: &train::update_strategy::UpdatePreset,
    hidden_size: usize,
    value_lr: f32,
    store: &mut autograd::TensorStore,
) -> Result<Option<train::opd::ValueCritic>> {
    match preset.advantage {
        train::update_strategy::Advantage::ValueGae { gamma, lam } => Ok(Some(
            train::opd::ValueCritic::new(hidden_size, value_lr, gamma, lam, store)
                .map_err(anyhow::Error::from)?,
        )),
        _ => Ok(None),
    }
}

/// Every mesh rank runs this exact sequence before a mirrored update — the cp
/// collectives stay aligned by construction, not by mirrored edits. Quiesce
/// first (the group's cc children exited with run_group, so this only drains a
/// straggler request); then drop the scratch + DEAD rollout KV pool — the two
/// headroom levers (scratch otherwise OOMs the logits alloc; the writeback's
/// fresh autograd forward never reads the engine KV).
#[cfg(feature = "cuda")]
fn quiesce_and_release_engines(infer_student: &train::infer_student::InferStudent) -> Result<()> {
    train::aopd_profile::time_try("quiesce", train::aopd_profile::WALL, || {
        quiesce_serve(infer_student.engine())
    })?;
    if let Err(err) = infer_student.release_inference_scratch() {
        eprintln!("[agent-opd] release inference scratch failed: {err}");
    }
    if let Err(err) = infer_student.release_kv_pool() {
        eprintln!("[agent-opd] release KV pool failed: {err}");
    }
    Ok(())
}

/// Group-end twin of [`quiesce_and_release_engines`]: re-merge the trained
/// LoRA into this rank's engine when the leader synced, then re-acquire the KV
/// pool. Returns the sync wall (0.0 when not synced).
#[cfg(feature = "cuda")]
fn sync_and_restore_engines(
    infer_student: &train::infer_student::InferStudent,
    store: &mut autograd::TensorStore,
    adapter_map: &std::collections::HashMap<&'static str, TensorId>,
    param_name_map: &std::collections::HashMap<&'static str, TensorId>,
    lora: train::lora::LoraConfig,
    synced: bool,
) -> Result<f64> {
    let mut sync_secs = 0.0;
    if synced {
        let started = Instant::now();
        infer_student
            .sync_lora_from_store(store, adapter_map, param_name_map, lora)
            .context("sync trained LoRA into rollout engine")?;
        sync_secs = started.elapsed().as_secs_f64();
        train::aopd_profile::record("sync_lora", train::aopd_profile::GPU, sync_secs);
    }
    infer_student
        .ensure_kv_pool_and_resume_admissions()
        .context("restore rollout engine after group")?;
    Ok(sync_secs)
}

/// cp rank > 0 of the cc-rollout lane: serve rollouts for rank 0's harness
/// (fleet endpoint) and mirror rank 0's update stream — including its engine
/// quiesce/release and LoRA re-merge decisions — until end-of-stream.
#[cfg(feature = "cuda")]
fn run_agent_opd_cp_follower(
    args: &TrainAgentOpdArgs,
    lora: train::lora::LoraConfig,
    target_set: train::lora::LoraTargetSet,
    serve_port: u16,
) -> Result<()> {
    use autograd::optim::AdamW;

    let cp = train::context_parallel::CpContext::from_env();
    let update_preset = args.update_preset();

    let mut fleet = load_agent_opd_serve_student(args, lora, target_set, serve_port)?;
    let vocab = fleet.vocab;
    let mut optimizer = AdamW::new(args.lr, (0.9, 0.999), 1.0e-8, 0.0);
    let mut value_critic = build_value_critic(
        &update_preset,
        fleet.student.config().hidden_size,
        args.value_lr,
        &mut fleet.store,
    )?;
    let adapter_map = fleet.student.adapter_name_map();

    let mut rx = MeshUpdateChannel::from_env()?;
    // Rank 0 waits for this before its first session can hit our endpoint.
    std::fs::write(rx.dir.join(format!("serve-r{}.ready", cp.rank)), b"")?;
    let gc_owner = cp.rank + 1 == cp.size;
    while let Some(msg) = rx.recv()? {
        match msg {
            MeshMsg::Update {
                batch,
                release_engines,
            } => {
                if release_engines {
                    quiesce_and_release_engines(&fleet.infer_student)?;
                }
                let report = update_preset
                    .update(
                        &batch,
                        &fleet.student,
                        fleet.all_params.as_slice(),
                        fleet.trainable.as_slice(),
                        &mut optimizer,
                        value_critic.as_mut(),
                        vocab,
                        args.writeback_window,
                        &mut fleet.store,
                    )
                    .map_err(anyhow::Error::from)?;
                eprintln!(
                    "[agent-opd] follower update {}: trained={} loss={:.6}",
                    rx.seq - 1,
                    report.trained,
                    report.loss
                );
                if let Err(err) = fleet.store.backend().trim_memory_pool() {
                    eprintln!("[agent-opd] follower device-pool trim failed: {err}");
                }
                if gc_owner {
                    rx.gc_consumed();
                }
            }
            MeshMsg::GroupEnd { synced } => {
                sync_and_restore_engines(
                    &fleet.infer_student,
                    &mut fleet.store,
                    &adapter_map,
                    &fleet.student.param_name_map(),
                    lora,
                    synced,
                )?;
            }
        }
    }
    eprintln!("[agent-opd] cp follower rank {}: end of stream", cp.rank);
    fleet.serve_thread.shutdown()?;
    Ok(())
}

#[cfg(feature = "cuda")]
fn run_agent_opd_impl(args: TrainAgentOpdArgs) -> Result<()> {
    use std::sync::Arc;

    use autograd::optim::AdamW;
    use std::collections::VecDeque;

    use train::{
        lora::LoraConfig,
        opd::{WritebackLoss, masked_writeback_ce_step_dispatch, masked_writeback_step},
        swe_dataset::SweTask,
    };

    let student_dir = args.student_model.as_path();
    reject_unimplemented_gkd_objectives(args.gkd_entropy_weight, None)?;
    let target_set = parse_lora_target_set(&args.lora_target_set)?;
    let lora = LoraConfig {
        rank: args.lora_rank,
        alpha: args.lora_alpha,
    };

    // Replay mode: no rollout engine / sandboxes / datasets — just the trainer
    // student + the same masked-CE writeback over pre-converted records.
    if let Some(records_path) = args.replay_records.clone() {
        return run_agent_opd_replay(&args, &records_path, lora, target_set);
    }
    let update_preset = args.update_preset();
    // Per-rank serve port: mesh ranks re-exec identical argv, so they'd all bind
    // the same --serve-port. Offset by WORLD rank (cp rank alone collides across
    // dp replicas); single card keeps the exact requested port.
    let serve_port = {
        let cp = train::context_parallel::CpContext::from_env();
        let dp = train::context_parallel::DpContext::from_env();
        args.serve_port + train::context_parallel::world_rank(cp, dp) as u16
    };
    validate_online_rollout_temperature(
        update_preset,
        args.update_strategy,
        args.rollout_temperature,
    )?;
    let dataset = args
        .dataset
        .as_deref()
        .ok_or_else(|| anyhow!("--dataset is required without --replay-records"))?;
    let staged_root = args
        .staged_root
        .as_deref()
        .ok_or_else(|| anyhow!("--staged-root is required without --replay-records"))?;

    // Stochastic cc rollouts diverge across mesh ranks (different accepted sets
    // → mismatched collectives → deadlock), so dp>1 is unsupported and cp
    // followers branch off below to mirror rank 0's update stream. The
    // synthetic probe is deterministic and exempt.
    if args.synthetic_writeback_seq == 0 && args.dp_size.max(1) > 1 {
        bail!(
            "agent-opd cc rollout does not support --dp-size > 1: per-replica rollouts \
             diverge at the gradient all-reduce; shard the sequence with --cp-size instead"
        );
    }

    // MUST run before the sandbox-spawner fork and the first CUDA context —
    // the coordinator owns neither.
    #[cfg(all(unix, feature = "cuda"))]
    if crate::train_multiproc::maybe_spawn_mesh_and_wait(
        args.cp_size,
        args.dp_size,
        &args.mesh_devices(),
    )? {
        return Ok(());
    }

    if args.synthetic_writeback_seq == 0 && train::context_parallel::CpContext::from_env().rank > 0
    {
        return run_agent_opd_cp_follower(&args, lora, target_set, serve_port);
    }

    // Pre-CUDA sandbox-spawner: fork ONE non-CUDA helper to own all rollout
    // subprocess spawns (bash/cp/git/pytest) BEFORE the first CUDA context below.
    // The parent is still non-CUDA-resident here, so this single fork is
    // ELKEID-safe; the helper then does the forking on a process that never
    // touches CUDA. `_spawner`'s Drop reaps the helper at function exit. Sets
    // `ARLE_SPAWNER_SOCKET`, routing `crate::sandbox` through the helper; when
    // unset (normal serve/CLI) sandbox spawns directly, byte-identical default.
    let _spawner = train::spawner::SpawnerHandle::launch()
        .context("launch pre-CUDA sandbox-spawner helper")?;

    // SWE-bench-Pro tasks; each task's staged tree is `<staged_root>/<instance_id>/`.
    let mut tasks_raw = train::swe_dataset::load_swe_tasks(dataset)?;
    if let Some(n) = args.task_limit {
        tasks_raw.truncate(n);
    }
    let tasks: Vec<(Arc<SweTask>, PathBuf)> = tasks_raw
        .into_iter()
        .map(|t| {
            let tree = staged_root.join(&t.instance_id);
            (Arc::new(t), tree)
        })
        .collect();
    if tasks.is_empty() {
        bail!("no usable tasks in {}", dataset.display());
    }
    eprintln!(
        "[arle train agent-opd] loaded {} tasks from {}",
        tasks.len(),
        dataset.display()
    );
    // Pass-rate selection state: in-memory, this run only (metrics.jsonl has the history).
    let mut selection = args.task_selection.then(|| TaskSelection::new(tasks.len()));

    // HELD-OUT eval tasks (separate from --dataset). Staged under
    // --eval-staged-root (falls back to --staged-root). Loaded once up front so
    // the round-0 baseline + per-round eval reuse the same set.
    let eval_tasks: Vec<(Arc<SweTask>, PathBuf)> = match args.eval_dataset.as_deref() {
        Some(eval_path) => {
            let eval_staged_root = args.eval_staged_root.as_deref().unwrap_or(staged_root);
            let mut eval_raw = train::swe_dataset::load_swe_tasks(eval_path)?;
            if let Some(n) = args.eval_n {
                eval_raw.truncate(n);
            }
            // Guard the held-out invariant: an overlap with the train set would
            // turn the pass-rate into a train-set memorization metric.
            let train_ids: std::collections::HashSet<&str> =
                tasks.iter().map(|(t, _)| t.instance_id.as_str()).collect();
            let overlap: Vec<&str> = eval_raw
                .iter()
                .map(|t| t.instance_id.as_str())
                .filter(|id| train_ids.contains(id))
                .collect();
            if !overlap.is_empty() {
                bail!(
                    "--eval-dataset overlaps --dataset on {} task(s) (e.g. {}); the eval set MUST be \
                     held-out so the pass-rate measures generalization, not memorization",
                    overlap.len(),
                    overlap.first().copied().unwrap_or("")
                );
            }
            let eval_tasks: Vec<(Arc<SweTask>, PathBuf)> = eval_raw
                .into_iter()
                .map(|t| {
                    let tree = eval_staged_root.join(&t.instance_id);
                    (Arc::new(t), tree)
                })
                .collect();
            eprintln!(
                "[arle train agent-opd] loaded {} HELD-OUT eval tasks from {} (staged under {})",
                eval_tasks.len(),
                eval_path.display(),
                eval_staged_root.display()
            );
            eval_tasks
        }
        None => Vec::new(),
    };
    let eval_out_dir: PathBuf = agent_opd_eval_out_dir(&args);
    // Structured per-round metrics sink (one JSON line/round). Machine-readable
    // replacement for stderr regex-scraping; defaults beside the eval dumps.
    let metrics_path: PathBuf = args
        .metrics_out
        .clone()
        .unwrap_or_else(|| eval_out_dir.join("metrics.jsonl"));

    let width = args.samples_per_prompt.max(1);
    let AgentOpdServeStudent {
        mut store,
        train_backend,
        vocab,
        infer_student,
        student,
        serve_thread,
        dump_dir,
        cc_model_id,
        all_params,
        trainable,
    } = load_agent_opd_serve_student(&args, lora, target_set, serve_port)?;

    let mut optimizer = AdamW::new(args.lr, (0.9, 0.999), 1.0e-8, 0.0);
    let mut value_critic = build_value_critic(
        &update_preset,
        student.config().hidden_size,
        args.value_lr,
        &mut store,
    )?;

    // Diagnostic: skip the (slow, stochastic) agent rollout and drive ONE masked-CE
    // writeback on a synthetic trajectory of length N, so the writeback's OOM
    // instrumentation fires deterministically. Same call the rollout closure below
    // makes (masked_writeback_ce_step with the trained student + optimizer + store).
    if args.synthetic_writeback_seq > 0 {
        let n = args.synthetic_writeback_seq;
        let prompt_len = 256.min(n / 2);
        let prompt_ids: Vec<u32> = (0..prompt_len as u32).map(|i| (i % 1000) + 1).collect();
        // response = the rest; all masked (=1) so every position is a loss target (worst case).
        let response_ids: Vec<u32> = (0..(n - prompt_len) as u32)
            .map(|i| (i % 30000) + 1)
            .collect();
        let response_mask: Vec<u8> = vec![1u8; response_ids.len()];
        eprintln!(
            "[synthetic-writeback] seq={n} prompt_len={prompt_len} response_len={} (all masked)",
            response_ids.len()
        );
        log_opd_vram("synthetic-writeback pre-release", &train_backend);
        // Mirror the real round closure: release BOTH the inference scratch and
        // the (dead) rollout KV pool before the masked-CE writeback, so the
        // diagnostic reproduces the real writeback's resident floor.
        if let Err(err) = infer_student.release_inference_scratch() {
            eprintln!("[synthetic-writeback] release inference scratch failed: {err}");
        }
        if let Err(err) = infer_student.release_kv_pool() {
            eprintln!("[synthetic-writeback] release KV pool failed: {err}");
        }
        log_opd_vram("synthetic-writeback pre-writeback", &train_backend);
        let started = std::time::Instant::now();
        let cp = train::context_parallel::CpContext::from_env();
        let dp = train::context_parallel::DpContext::from_env();
        let fd_target = std::env::var("ARLE_OPD_FD_PARAM").ok();
        let loss = if fd_target.is_some() {
            masked_writeback_step(
                WritebackLoss::Ce,
                &student,
                all_params.as_slice(),
                trainable.as_slice(),
                &mut optimizer,
                false,
                &prompt_ids,
                &response_ids,
                &response_mask,
                vocab,
                args.writeback_window,
                cp,
                dp,
                &mut store,
            )?
            .0
        } else {
            masked_writeback_ce_step_dispatch(
                &student,
                all_params.as_slice(),
                trainable.as_slice(),
                &mut optimizer,
                &prompt_ids,
                &response_ids,
                &response_mask,
                vocab,
                args.writeback_window,
                cp,
                dp,
                &mut store,
            )?
        };
        log_opd_vram("synthetic-writeback post-writeback", &train_backend);
        if let Some(target) = fd_target {
            synthetic_writeback_fd_probe(
                &target,
                &student,
                all_params.as_slice(),
                trainable.as_slice(),
                &mut optimizer,
                &prompt_ids,
                &response_ids,
                &response_mask,
                vocab,
                args.writeback_window,
                cp,
                dp,
                &mut store,
            )?;
        }
        eprintln!(
            "[synthetic-writeback] DONE loss={loss:.6} elapsed={:?}",
            started.elapsed()
        );
        return Ok(());
    }

    // cc rollout harness over the serve. --staleness 0 (default): task groups
    // run sequentially — in-flight concurrency = the K samples of one group.
    // --staleness 1: the next group rolls on a background thread while this
    // group trains+merges (Arc'd for that thread's 'static bound).
    let cp = train::context_parallel::CpContext::from_env();
    let harness = Arc::new(train::cc_harness::CcHarness {
        work_root: args.work_root.clone(),
        dump_dir,
        // Fleet endpoints: rank r serves on base port + world rank r (== cp
        // rank — dp>1 is rejected above), the same offset the follower's own
        // serve_port derives; samples spread round-robin across them.
        base_urls: (0..cp.size.max(1))
            .map(|r| format!("http://127.0.0.1:{}", args.serve_port + r as u16))
            .collect(),
        model_id: cc_model_id,
        cc_timeout_secs: args.cc_timeout,
        test_timeout_secs: args.test_timeout_secs,
        pythonpath: args.pythonpath.clone(),
        reward_shape: args.reward_shape.into(),
        tokenizer: train::cc_harness::load_tokenizer(&student_dir.join("tokenizer.json"))?,
    });

    // cp>1: this process is rank 0 (followers branched off above). Stream every
    // update's batch + engine-lifecycle decisions so follower ranks mirror the
    // collective sequence; wait for every follower serve before any cc traffic.
    let mut mesh_tx = cp
        .is_enabled()
        .then(MeshUpdateChannel::from_env)
        .transpose()?;
    if let Some(tx) = mesh_tx.as_ref() {
        for r in 1..cp.size {
            let marker = tx.dir.join(format!("serve-r{r}.ready"));
            while !marker.exists() {
                std::thread::sleep(std::time::Duration::from_millis(500));
            }
        }
        eprintln!(
            "[agent-opd] rollout fleet ready: {} serve endpoints",
            cp.size
        );
    }

    // One deep tokenizer clone for the post-merge warm-up threads.
    let warmup_tokenizer = Arc::new(harness.tokenizer.clone());
    // Last warm-up's handle: joined before serve teardown so it never races
    // engine shutdown (earlier ones finish inside the next group's rollout).
    let mut warmup_join: Option<std::thread::JoinHandle<()>> = None;

    // PEFT adapter config for `--save-lora-adapters` (mainstream HF PEFT dir:
    // adapter_config.json + adapter_model.safetensors). Built once; borrowed by
    // every per-round adapter save.
    let lora_adapter_config = agent_opd_adapter_config(student_dir, target_set, lora);
    let metrics = JsonlSink::new(metrics_path);

    // Round-0 BASELINE held-out eval BEFORE any training: the un-tuned student's
    // pass-rate every per-round eval is read against.
    let baseline_pass_rate: Option<f32> = (!eval_tasks.is_empty())
        .then(|| {
            run_cc_eval(
                &harness,
                &eval_tasks,
                &eval_out_dir,
                "base",
                args.eval_concurrency,
            )
        })
        .transpose()?;

    let needs = update_preset.needs();
    let sync_every_group = matches!(args.sync, crate::args::SyncArg::EveryGroup);
    let preset_name = clap::ValueEnum::to_possible_value(&args.update_strategy)
        .map_or_else(String::new, |v| v.get_name().to_owned());
    if args.replay_reuse > 0 && !needs.behavior_logprobs {
        bail!(
            "--replay-reuse {} with --update-strategy {preset_name}: the preset has no IS ratio \
             (ratio == None), so replayed groups would be uncorrected off-policy. Use a \
             ratio-weighted preset (grpo | dapo | dr-grpo | gspo | cispo | sao-*).",
            args.replay_reuse
        );
    }
    if args.staleness > 0 && !needs.behavior_logprobs {
        bail!(
            "--staleness {} with --update-strategy {preset_name}: the preset has no IS ratio \
             (ratio == None), so stale groups would be uncorrected off-policy. Use a \
             ratio-weighted preset (grpo | dapo | dr-grpo | gspo | cispo | sao-*).",
            args.staleness
        );
    }
    let mut replay =
        (args.replay_reuse > 0).then(|| (ReplayBuffer::default(), PromptSampler::new(0x05EE_D1A7)));
    // LoRA-merge counter. A group is tagged with the version its rollouts
    // LAUNCHED under (behavior_version) and trains at the current version;
    // staleness = current − behavior (0 today; 1 for overlapped groups).
    let mut policy_version = 0u64;
    // Previous eval round's gap between train reward and held-out pass rate. When
    // train reward climbs but held-out doesn't, the two diverge — a sign the model
    // is gaming the reward. We warn only when the gap is large AND still widening.
    let mut prev_eval_gap: Option<f64> = None;
    for round in 0..args.rounds {
        // Anchor the per-stage profiler (human table opt-in via ARLE_AOPD_PROFILE).
        train::aopd_profile::begin_round();
        // Re-acquire the rollout KV pool the previous writeback dropped
        // (no-op when resident).
        if let Err(err) =
            train::aopd_profile::time_try("kv_pool_ensure", train::aopd_profile::GPU, || {
                infer_student.ensure_kv_pool()
            })
        {
            eprintln!("[agent-opd] ensure KV pool (round {round}) failed: {err}");
        }

        let mut losses: Vec<f32> = Vec::new();
        let (mut rollouts, mut passed, mut tasks_passed, mut zero_variance_groups) =
            (0usize, 0usize, 0usize, 0usize);
        let mut replayed_groups = 0usize;
        let mut reward_sum = 0.0f64;
        // How far the generation-time token probabilities drift from the
        // training-time ones for the same weights; near 0 means our rollout and
        // training math agree. PgStats already sums this per masked token from the
        // same Δ = train_logp − rollout_logp the IS ratio uses (no extra forward),
        // so we just carry the sum/count across the round. Ratio-free presets
        // (rejection-ce) contribute 0 tokens → the field is omitted below.
        let (mut round_k3_sum, mut round_k3_tokens) = (0.0f64, 0usize);
        let (mut prompt_tokens, mut completion_tokens) = (0u64, 0u64);
        let mut sync_lora_secs = 0.0f64;
        // Round-scoped budget of TRAINABLE trajectories (parity with the
        // pre-cc loop, where the cap bounded accepted pairs per round).
        let mut cap_left = args.writeback_cap.unwrap_or(usize::MAX);

        let (round_tasks, tasks_skipped, tasks_retired) = match selection.as_mut() {
            Some(sel) => sel.select(round),
            None => ((0..tasks.len()).collect(), 0, 0),
        };
        if tasks_skipped + tasks_retired > 0 {
            eprintln!(
                "[arle train agent-opd] round {round}: task-selection ran={}/{} skipped={tasks_skipped} retired={tasks_retired}",
                round_tasks.len(),
                tasks.len()
            );
        }

        let launch = |&i: &usize, version: u64| -> PendingGroup {
            let (task, staged) = &tasks[i];
            let booted = harness.boot_group(task, staged, width);
            if args.staleness == 0 {
                return PendingGroup::Booted(booted);
            }
            let harness = Arc::clone(&harness);
            PendingGroup::Rolling {
                behavior_version: version,
                handle: std::thread::spawn(move || harness.run_group(booted)),
            }
        };
        // DAPO dynamic sampling: `work` grows as dead groups get replaced (below);
        // `reserve` draws this round's unscheduled tasks, so the launch cap AND
        // reserve exhaustion both terminate an all-dead corpus. Off ⇒
        // max_launches == target ⇒ refill is inert (one path, static behavior).
        let target = round_tasks.len();
        let max_launches = if args.dynamic_sampling {
            ((target as f32) * args.dynamic_sampling_max_factor).ceil() as usize
        } else {
            target
        };
        let mut refill = train::cc_harness::RefillBudget::new(
            target,
            max_launches,
            args.dynamic_sampling_token_budget,
        );
        let scheduled: std::collections::HashSet<usize> = round_tasks.iter().copied().collect();
        // Refill reserve: this round's unscheduled tasks, MINUS retired ones — a
        // mastered task must not be resurrected by a dead group's replacement.
        let mut reserve = (0..tasks.len())
            .filter(|i| {
                !scheduled.contains(i) && selection.as_ref().is_none_or(|sel| !sel.is_retired(*i))
            })
            .collect::<Vec<_>>()
            .into_iter();
        let mut work = round_tasks;
        // Prompts per update (verl shape): G groups roll concurrently under ONE
        // policy version, then a single update trains their merged batch. Keeps
        // G × width sessions in flight so the fleet's slots stay fed and one
        // group's straggler no longer idles the others; staleness stays 0 at any
        // G. G=1 is the per-group loop.
        let g = args.prompts_per_update.max(1);
        let mut pending: VecDeque<(usize, PendingGroup)> = VecDeque::new();
        let mut launched = 0usize;
        let top_up = |pending: &mut VecDeque<(usize, PendingGroup)>,
                      launched: &mut usize,
                      work: &[usize],
                      version: u64| {
            while pending.len() < g && *launched < work.len() {
                let i = work[*launched];
                pending.push_back((i, launch(&i, version)));
                *launched += 1;
            }
        };
        top_up(&mut pending, &mut launched, &work, policy_version);
        let mut pos = 0;
        let mut groups_done = 0usize;
        while pos < work.len() {
            let window: Vec<(usize, PendingGroup)> = pending.drain(..).collect();
            let n = window.len();
            // Boot-ahead (staleness 0): the next window's sandboxes build during
            // this window's rollout + train (CPU only — staleness-free).
            if args.staleness == 0 {
                top_up(&mut pending, &mut launched, &work, policy_version);
            }
            let gpu_busy_before = infer_api::engine_forward_busy_micros();
            let rolled = train::aopd_profile::time_try(
                "cc_rollout",
                train::aopd_profile::WALL,
                || -> Result<Vec<(usize, train::cc_harness::CcGroup, u64)>> {
                    // Booted groups start here so the whole window rolls at once;
                    // Rolling ones (staleness>0) are already in flight under the
                    // version they launched with.
                    let started: Vec<_> = window
                        .into_iter()
                        .map(|(idx, pending)| match pending {
                            PendingGroup::Booted(booted) => {
                                let harness = Arc::clone(&harness);
                                (
                                    idx,
                                    policy_version,
                                    std::thread::spawn(move || harness.run_group(booted)),
                                )
                            }
                            PendingGroup::Rolling {
                                behavior_version,
                                handle,
                            } => (idx, behavior_version, handle),
                        })
                        .collect();
                    started
                        .into_iter()
                        .map(|(idx, version, handle)| {
                            let group = handle
                                .join()
                                .map_err(|_| anyhow!("rollout thread panicked"))??;
                            Ok((idx, group, version))
                        })
                        .collect()
                },
            )?;
            // Staleness 1: admit the NEXT window's rollout now, BEFORE this
            // window's train + merge — cc rolls while the GPU trains; the
            // version tag + sidecar IS ratio below correct the one-step drift.
            if args.staleness > 0 {
                top_up(&mut pending, &mut launched, &work, policy_version);
            }
            // Window-level: the window's groups roll concurrently, so engine
            // forward time cannot be split between them.
            let gpu_busy_secs = infer_api::engine_forward_busy_micros()
                .saturating_sub(gpu_busy_before) as f64
                / 1e6;

            // The window shares one launch version, so its staleness is a single
            // number (0 at --staleness 0; 1 for a window admitted before the
            // previous update).
            let window_behavior = rolled.first().map_or(policy_version, |(_, _, v)| *v);
            let mut window_groups: Vec<(usize, train::cc_harness::CcGroup, u64, u64)> =
                Vec::with_capacity(n);
            for (group_idx, group, behavior_version) in rolled {
                let group_staleness = policy_version - behavior_version;
                let group_passed = group.samples.iter().filter(|s| s.passed()).count();
                let group_zero_variance = train::cc_harness::zero_variance(&group.samples);
                rollouts += group.samples.len();
                passed += group_passed;
                tasks_passed += usize::from(group_passed > 0);
                zero_variance_groups += usize::from(group_zero_variance);
                if let Some(sel) = selection.as_mut() {
                    sel.record(
                        group_idx,
                        group_passed as f32 / group.samples.len().max(1) as f32,
                    );
                }
                let rewards: Vec<f32> = group.samples.iter().map(|s| s.reward).collect();
                let reward_mean = rewards.iter().sum::<f32>() / rewards.len().max(1) as f32;
                let reward_std = (rewards
                    .iter()
                    .map(|r| (r - reward_mean).powi(2))
                    .sum::<f32>()
                    / rewards.len().max(1) as f32)
                    .sqrt();
                reward_sum += f64::from(rewards.iter().sum::<f32>());
                let g_prompt: u64 = group.samples.iter().filter_map(|s| s.cc_input_tokens).sum();
                let g_completion: u64 = group
                    .samples
                    .iter()
                    .filter_map(|s| s.cc_output_tokens)
                    .sum();
                prompt_tokens += g_prompt;
                completion_tokens += g_completion;
                // Never fatal: `--save-every 0` saves only after the final round, and
                // a failed sandbox spawn also lands here.
                if g_completion == 0 {
                    eprintln!(
                        "[agent-opd] group {group_idx} ({}) generated 0 completion tokens across {} \
                         sample(s) — check the engine log for a closed engine thread or a KV pool \
                         collapsed to the token floor",
                        group.task_id,
                        group.samples.len(),
                    );
                }
                // Group rollout wall = first cc start → last cc end.
                let rollout_secs = group
                    .samples
                    .iter()
                    .map(|s| s.t_end_ms)
                    .max()
                    .unwrap_or(0)
                    .saturating_sub(
                        group
                            .samples
                            .iter()
                            .map(|s| s.t_start_ms)
                            .min()
                            .unwrap_or(0),
                    ) as f64
                    / 1000.0;
                // GPU-busy fraction of the rollout wall: engine forward wall
                // (submit→ready) over this group's window; the remainder is
                // agent-latency idle (tool-exec / pytest / HTTP between turns).
                // Only separable when the window IS this group. LEADER-ONLY under
                // cp>1: the counter is process-local, so follower-endpoint forward
                // time is not counted — read gpu_busy_frac as a floor.
                let solo_busy = (n == 1).then_some(gpu_busy_secs);
                metrics.append(&serde_json::json!({
                    "kind": "group",
                    "round": round,
                    "task_id": group.task_id,
                    "rewards": rewards,
                    "reward_mean": reward_mean,
                    "reward_std": reward_std,
                    "zero_variance": group_zero_variance,
                    "behavior_version": behavior_version,
                    "staleness": group_staleness,
                    "passed": group_passed,
                    "edited": group.samples.iter().filter(|s| s.edited).count(),
                    "prompt_tokens": g_prompt,
                    "completion_tokens": g_completion,
                    "rollout_secs": rollout_secs,
                    "rollout_tok_per_sec": g_completion as f64 / rollout_secs.max(1e-9),
                    "gpu_busy_secs": solo_busy,
                    "gpu_busy_frac": solo_busy.map(|b| b / rollout_secs.max(1e-9)),
                }));
                window_groups.push((group_idx, group, behavior_version, g_completion));
            }

            // Staleness 1 skips quiesce AND the releases ONLY while a next
            // window is actually Rolling: its requests are legitimately in
            // flight — cancel_all would kill them, and a dropped KV pool /
            // scratch would be read by live decodes. The re-merge below is
            // atomic engine-side; in-flight requests keep their pinned pages
            // and finish on mixed KV (#92 caveat) — exactly the drift the
            // version tag + sidecar IS ratio correct. A current-window
            // straggler (its cc child exited) is harmless without the pool
            // drop and drains on its own. With NOTHING in flight, release
            // exactly as at staleness 0 — a resident pool+scratch atop the
            // capture forward OOMed at 97.4 GB.
            let released_engines = args.staleness == 0 || pending.is_empty();
            if released_engines {
                quiesce_and_release_engines(&infer_student)?;
            }
            log_opd_vram("agent-opd pre-writeback", &train_backend);

            // One batch per group (replay stores them per task), concatenated
            // into the window's single update.
            let mut per_group: Vec<(String, Vec<train::update_strategy::ScoredTrajectory>)> =
                Vec::with_capacity(n);
            for (group_idx, group, _, g_completion) in window_groups {
                let mut batch: Vec<train::update_strategy::ScoredTrajectory> = group
                    .records
                    .into_iter()
                    .map(|r| train::update_strategy::ScoredTrajectory {
                        prompt_ids: r.prompt_ids,
                        response_ids: r.response_ids,
                        response_mask: r.response_mask,
                        reward: r.reward,
                        behavior_logprobs: (!r.gen_logprobs.is_empty()).then_some(r.gen_logprobs),
                        group_id: group_idx,
                        // Timeout/harness-error attempts: drop them from the update
                        // (a budget artifact, not a learnable fail) via DAPO's filter.
                        truncated: r.truncated,
                    })
                    .collect();
                // VRAM-wall length guard before writeback. Skipped here == skipped
                // by update() anyway, while keeping an overlong record out of the
                // batch-wide sidecar validation and any subsequent model forward.
                if args.runtime.max_update_seq != 0 {
                    batch.retain(|t| {
                        let seq = t.prompt_ids.len() + t.response_ids.len();
                        let keep = seq <= args.runtime.max_update_seq;
                        if !keep {
                            // reward: a dropped pass is lost signal, a dropped fail is not.
                            eprintln!(
                                "[agent-opd] SKIP trajectory pre-capture: seq {seq} > max_update_seq {} (VRAM wall), prompt {} reward {:.2} supervised {}",
                                args.runtime.max_update_seq,
                                t.prompt_ids.len(),
                                t.reward,
                                t.response_mask.iter().filter(|&&m| m == 1).count()
                            );
                        }
                        keep
                    });
                }
                // Deadness for DAPO refill = "no learning signal", judged on the
                // batch BEFORE the writeback-cap truncation — an exhausted cap is a
                // budget artifact, not a zero-variance group, and must not classify a
                // live group as dead (which would refill forever once cap_left == 0).
                // Matches update's own filter, so no double forward.
                let group_trained = update_preset.planned_training_count(&batch) > 0;
                if args.writeback_cap.is_some() {
                    if !needs.keep_failing {
                        // Don't let failures (rejected inside `update`) burn budget.
                        batch.retain(|t| t.reward >= 1.0);
                    }
                    batch.truncate(cap_left);
                    cap_left -= batch.len();
                }
                // Append a replacement for a dead group (grow `work` now so the
                // last-window sync check stays correct; launch is deferred to
                // end-of-body so a refill thread never races the VRAM release).
                // Always count the group in `refill`; only skip the append once the
                // writeback cap is spent — a replacement could not train anyway.
                let cap_open = args.writeback_cap.is_none() || cap_left > 0;
                let want_refill = refill.complete(work.len(), group_trained, g_completion);
                if cap_open
                    && want_refill
                    && let Some(idx) = reserve.next()
                {
                    work.push(idx);
                }
                per_group.push((group.task_id, batch));
            }

            // The update validates the entire ratio-weighted batch before any
            // critic/student forward. Ratio-free CE/GKD ignores absent sidecars.
            // One update + its metrics row; `extra` carries the arm-specific
            // fields (fresh: window; replay: group/task_id/replayed/age).
            let mut run_update = |batch: &[train::update_strategy::ScoredTrajectory],
                                  extra: serde_json::Value,
                                  release_engines: bool|
             -> Result<()> {
                if let Some(tx) = mesh_tx.as_mut() {
                    tx.publish(&MeshMsgRef::Update {
                        batch,
                        release_engines,
                    })?;
                }
                let update_started = Instant::now();
                let report =
                    train::aopd_profile::time_try("update", train::aopd_profile::GPU, || {
                        update_preset.update(
                            batch,
                            &student,
                            all_params.as_slice(),
                            trainable.as_slice(),
                            &mut optimizer,
                            value_critic.as_mut(),
                            vocab,
                            args.writeback_window,
                            &mut store,
                        )
                    })
                    .map_err(anyhow::Error::from)?;
                losses.push(report.loss);
                round_k3_sum += report.stats.kl_sum;
                round_k3_tokens += report.stats.tokens;
                let mut row = serde_json::json!({
                    "kind": "update",
                    "round": round,
                    "preset": preset_name,
                    "trajectories": report.trained,
                    "tokens_trained": report.tokens,
                    "policy_loss": report.loss,
                    "critic_mse": report.critic_mse,
                    "kl_rollout": report.stats.kl_mean(),
                    "is_ratio_mean": report.stats.ratio_mean(),
                    "is_ratio_max": report.stats.ratio_max,
                    "clip_frac": report.stats.clip_frac(),
                    "adv_mean": report.adv_mean,
                    "adv_std": report.adv_std,
                    "update_secs": update_started.elapsed().as_secs_f64(),
                });
                if let serde_json::Value::Object(extra) = extra {
                    row.as_object_mut()
                        .expect("update row is an object")
                        .extend(extra);
                }
                metrics.append(&row);
                Ok(())
            };
            let merged: Vec<train::update_strategy::ScoredTrajectory> = per_group
                .iter()
                .flat_map(|(_, batch)| batch.iter().cloned())
                .collect();
            run_update(
                &merged,
                serde_json::json!({
                    "groups": per_group.len(),
                    "behavior_version": window_behavior,
                    "staleness": policy_version - window_behavior,
                    "window_gpu_busy_secs": gpu_busy_secs,
                }),
                released_engines,
            )?;

            // Experience replay (fresh-anchored: only after the fresh window
            // trained). Stored batches retain the immutable generation-time
            // behavior sidecars. Push after drawing: first drawable next window.
            if let Some((buffer, rng)) = replay.as_mut() {
                // Per FRESH group, as at G=1 — one draw per window would thin
                // replay by G.
                for entry in (0..per_group.len())
                    .flat_map(|_| buffer.draw(round, args.replay_reuse, rng))
                    .collect::<Vec<_>>()
                {
                    run_update(
                        &entry.batch,
                        serde_json::json!({
                            "group": entry.batch[0].group_id,
                            "task_id": entry.task_id,
                            "replayed": true,
                            "age": round - entry.round,
                        }),
                        false,
                    )?;
                    replayed_groups += 1;
                }
                for (task_id, batch) in per_group {
                    buffer.push(round, task_id, batch);
                }
            }

            // Writeback transients leave freed blocks hoarded in the autograd
            // device pool; return them to the driver here or the engine-thread
            // LoRA re-merge below OOMs even with the KV pool released.
            if let Err(err) = store.backend().trim_memory_pool() {
                eprintln!("[agent-opd] device-pool trim after writeback failed: {err}");
            }
            // Sync the trained LoRA into the serve engine (atomic re-merge +
            // prefix-cache drop in one engine-thread closure); every-round
            // syncs once, at the last window. Then re-acquire the KV pool for
            // the next rollout.
            let synced = sync_every_group || pos + n == work.len();
            // Followers re-merge into their own engines and re-acquire their KV
            // pools in parallel with ours.
            if let Some(tx) = mesh_tx.as_mut() {
                tx.publish(&MeshMsgRef::GroupEnd { synced })?;
            }
            sync_lora_secs += sync_and_restore_engines(
                &infer_student,
                &mut store,
                &student.adapter_name_map(),
                &student.param_name_map(),
                lora,
                synced,
            )?;
            if synced {
                policy_version += 1;
            }
            // Post-merge prefix warm-up: the re-merge flushed the radix cache,
            // so re-prefill the shared cc prompt (newest dump's prompt portion,
            // max_tokens=1) on a background thread — overlapped with the next
            // window's boot; a queued real request just lands behind it in the
            // same engine FIFO. Log-and-continue: never kills training.
            if synced {
                let engine = Arc::clone(infer_student.engine());
                let tokenizer = Arc::clone(&warmup_tokenizer);
                let dump_dir = harness.dump_dir.clone();
                warmup_join = Some(std::thread::spawn(move || {
                    let t0 = Instant::now();
                    let outcome = train::cc_convert::newest_dump_prompt_ids(&dump_dir, &tokenizer)
                        .and_then(|prompt| match prompt {
                            // No dump yet (round 0 window 0): skip silently.
                            None => Ok(false),
                            Some(ids) => engine
                                .lock()
                                .map_err(|err| anyhow!("engine lock poisoned: {err}"))?
                                .generate_token_ids(&ids, 1, SamplingParams::default())
                                .map(|_| true),
                        });
                    match outcome {
                        Ok(true) => train::aopd_profile::record(
                            "prefix_warmup",
                            train::aopd_profile::GPU,
                            t0.elapsed().as_secs_f64(),
                        ),
                        Ok(false) => {}
                        Err(err) => {
                            eprintln!("[agent-opd] prefix warm-up failed (continuing): {err:#}");
                        }
                    }
                }));
            }
            // Launch a deferred DAPO refill (appended above) only now that the KV
            // pool is restored. Static path: `work` didn't grow, no-op.
            pos += n;
            groups_done += n;
            top_up(&mut pending, &mut launched, &work, policy_version);
        }
        // Windowing must consume every scheduled task exactly once (`work` grows
        // under DAPO refill, so this is the one place the arithmetic is checked).
        if groups_done != work.len() {
            bail!(
                "round {round}: windowed scheduler ran {groups_done} groups for {} scheduled",
                work.len()
            );
        }

        let mean_loss = if losses.is_empty() {
            0.0
        } else {
            losses.iter().sum::<f32>() / losses.len() as f32
        };
        eprintln!(
            "[arle train agent-opd] round {round}: tasks={} groups={} effective={} discarded={} rollouts={rollouts} passed={passed} tasks_passed={tasks_passed} zero_variance_groups={zero_variance_groups} mean_loss={mean_loss:.4}",
            target,
            work.len(),
            refill.effective,
            refill.discarded,
        );
        log_opd_vram(&format!("after round {round} writeback"), &train_backend);

        // HELD-OUT eval of THIS round's student (serve engine now holds the
        // round-N LoRA under every-group AND every-round sync). Runs every
        // --eval-every rounds (0 = off) and always on the final round.
        let is_final_round = round + 1 == args.rounds;
        let do_eval = !eval_tasks.is_empty()
            && (is_final_round || (args.eval_every > 0 && (round + 1) % args.eval_every == 0));
        let mut held_out_pass_rate: Option<f32> = None;
        if do_eval {
            let pass_rate =
                train::aopd_profile::time_try("eval", train::aopd_profile::GPU, || {
                    run_cc_eval(
                        &harness,
                        &eval_tasks,
                        &eval_out_dir,
                        &(round + 1).to_string(),
                        args.eval_concurrency,
                    )
                })?;
            held_out_pass_rate = Some(pass_rate);
            match baseline_pass_rate {
                Some(base) => eprintln!(
                    "[arle train agent-opd] round {round}: held-out pass_rate={pass_rate:.4} (baseline={base:.4}, Δ={:+.4}) train_mean_loss={mean_loss:.4}",
                    pass_rate - base,
                ),
                None => eprintln!(
                    "[arle train agent-opd] round {round}: held-out pass_rate={pass_rate:.4} train_mean_loss={mean_loss:.4}",
                ),
            }
        }

        // Fast adapter-only (LoRA) save as a mainstream HF PEFT adapter dir
        // (adapter_config.json + adapter_model.safetensors) — loadable by HF PEFT
        // / vLLM / SGLang. Avoids the full-materialize host-loop hang.
        if let Some(adapter_dir) = args.save_lora_adapters.as_deref()
            && should_save_step_checkpoint(round + 1, args.rounds, args.save_every)
        {
            train::aopd_profile::time_try("save_adapters", train::aopd_profile::DISK, || {
                save_agent_opd_adapters(
                    adapter_dir,
                    &format!("adapters_round{}", round + 1),
                    round + 1,
                    student_dir,
                    &student,
                    &mut store,
                    &lora_adapter_config,
                )
            })?;
        }

        let save_started = Instant::now();
        let mut ckpt_tape = Tape::new();
        maybe_save_full_student_checkpoint(
            "agent-opd",
            args.save_checkpoint.as_deref(),
            args.save_every,
            round + 1,
            args.rounds,
            student_dir,
            &student,
            &mut store,
            &mut ckpt_tape,
        )?;
        let save_secs = save_started.elapsed().as_secs_f64();
        train::aopd_profile::record("save_checkpoint", train::aopd_profile::DISK, save_secs);
        eprintln!("[agent-opd] phase=round_tail_save seconds={save_secs:.3}");

        // Per-round metrics row; the update/group rows land per group above.
        let groups = work.len().max(1);
        let reward_mean = reward_sum / rollouts.max(1) as f64;
        // Mean drift over the round's masked tokens; None on ratio-free presets.
        let rollout_train_k3_kl =
            (round_k3_tokens > 0).then(|| round_k3_sum / round_k3_tokens as f64);
        // Only meaningful on eval rounds (held_out is None otherwise). A large,
        // still-widening gap is our reward-hacking alarm.
        const REWARD_HELDOUT_GAP_WARN: f64 = 0.2;
        let reward_heldout_gap = held_out_pass_rate.map(|pass| reward_mean - f64::from(pass));
        if let Some(gap) = reward_heldout_gap {
            if let Some(prev) = prev_eval_gap
                && gap > REWARD_HELDOUT_GAP_WARN
                && gap > prev
            {
                eprintln!(
                    "[arle train agent-opd] WARNING round {round}: reward↔held-out gap \
                     {gap:+.4} exceeds {REWARD_HELDOUT_GAP_WARN} and is rising (prev \
                     {prev:+.4}) — possible reward hacking",
                );
            }
            prev_eval_gap = Some(gap);
        }
        metrics.append(&serde_json::json!({
            "kind": "round",
            "round": round,
            "tasks": target,
            "tasks_skipped": tasks_skipped,
            "tasks_retired": tasks_retired,
            "groups_launched": work.len(),
            "groups_effective": refill.effective,
            "groups_discarded": refill.discarded,
            "rollouts": rollouts,
            "passed": passed,
            "pass_at_k": tasks_passed as f32 / groups as f32,
            "zero_variance_group_frac": zero_variance_groups as f32 / groups as f32,
            "replayed_groups": replayed_groups,
            "reward_mean": reward_mean,
            "rollout_train_k3_kl": rollout_train_k3_kl,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "mean_train_loss": mean_loss,
            "sync_lora_secs": sync_lora_secs,
            "save_secs": save_secs,
            "phase_secs": serde_json::Map::from_iter(
                train::aopd_profile::phase_secs()
                    .into_iter()
                    .map(|(stage, secs)| (stage.to_owned(), secs.into())),
            ),
            "held_out_pass_rate": held_out_pass_rate,
            "baseline_pass_rate": baseline_pass_rate,
            "delta": held_out_pass_rate.zip(baseline_pass_rate).map(|(p, b)| p - b),
            "reward_heldout_gap": reward_heldout_gap,
        }));

        // Per-stage ms + %-of-round breakdown (opt-in ARLE_AOPD_PROFILE; no-op off).
        train::aopd_profile::print_round(round);
    }

    // Release the followers now; the coordinator keeps the group alive until
    // rank 0 (this process) finishes eval/save below.
    if let Some(tx) = mesh_tx.as_ref() {
        tx.finish()?;
    }

    if let Some(join) = warmup_join.take() {
        let _ = join.join();
    }
    serve_thread.shutdown()?;
    eprintln!("[arle train agent-opd] done ({} rounds)", args.rounds);
    Ok(())
}

/// Forward-only held-out NLL: mean next-token cross-entropy of predicting
/// `eval_ids[t+1]` from `eval_ids[..=t]` over a FIXED reference sequence.
/// Tape-off (scratch tape dropped after the forward); non-circular — the
/// reference text is fixed, independent of the moving EMA teacher — so it is a
/// valid SOPD no-regression signal (KL-vs-EMA would be circular).
fn heldout_nll(
    student: &train::qwen35::Qwen35Model,
    eval_ids: &[u32],
    vocab: usize,
    store: &mut TensorStore,
) -> Result<f32> {
    use autograd::Tape;
    if eval_ids.len() < 2 {
        bail!("--eval-ids needs at least 2 token ids for a next-token NLL");
    }
    let input: Vec<usize> = eval_ids[..eval_ids.len() - 1]
        .iter()
        .map(|&id| id as usize)
        .collect();
    let seq_len = input.len();
    let mut eval_tape = Tape::new();
    let logits_id = student
        .forward_tokens(&input, store, &mut eval_tape)
        .context("held-out NLL forward")?;
    let host = store.to_host(logits_id)?;
    let expected = seq_len
        .checked_mul(vocab)
        .ok_or_else(|| anyhow!("held-out NLL logits shape overflow"))?;
    if host.len() != expected {
        bail!(
            "held-out NLL logits len {} != seq_len*vocab {} (seq_len={seq_len}, vocab={vocab})",
            host.len(),
            expected
        );
    }
    let mut nll_sum = 0.0_f64;
    for t in 0..seq_len {
        let row = &host[t * vocab..(t + 1) * vocab];
        nll_sum += row_nll(row, eval_ids[t + 1] as usize)?;
    }
    Ok((nll_sum / seq_len as f64) as f32)
}

fn run_self_opd_from_dir(args: TrainSelfOpdArgs) -> Result<()> {
    use autograd::{Tape, optim::AdamW};
    use train::{
        ema_self_teacher::EmaSelfTeacher,
        lora::LoraConfig,
        opd::{
            GkdLossConfig, GkdSftAnchor, OpdKlMask, OpdStepConfig,
            opd_step_with_teacher_forward_profiled_gkd_anchor,
        },
        qwen35_loader::load_qwen35_lora_from_hf_dir,
    };

    let student_dir = args
        .student_model
        .as_deref()
        .ok_or_else(|| anyhow!("--student-model <dir> is required for non-smoke runs"))?;
    reject_unimplemented_gkd_objectives(0.0, args.teacher_topk)?;
    let target_set = parse_lora_target_set(&args.lora_target_set)?;
    let lora = LoraConfig {
        rank: args.lora_rank,
        alpha: args.lora_alpha,
    };

    let (mut store, train_backend, backend_label) = build_opd_store(args.backend)?;
    apply_tape_dtype(&mut store, args.tape_dtype)?;
    let mut tape = Tape::new();

    eprintln!(
        "[arle train self-opd] loading student from {}",
        student_dir.display()
    );
    let student = load_qwen35_lora_from_hf_dir(student_dir, lora, target_set, &mut store)
        .with_context(|| format!("load LoRA student from {}", student_dir.display()))?;
    // Build the EMA self-teacher IMMEDIATELY after the student and BEFORE any
    // other scratch alloc: from_student calls retain_ids which frees the rest.
    let mut ema = EmaSelfTeacher::from_student(&student, lora, target_set, &mut store)
        .context("build EMA self-teacher")?;

    let student_trainable: Vec<TensorId> = student
        .all_parameter_ids()
        .into_iter()
        .filter(|id| store.get(*id).is_some_and(|tensor| tensor.requires_grad))
        .collect();
    let cfg = student.config().clone();
    let vocab = cfg.vocab_size;

    let prompt_ids = parse_prompt_ids(args.prompt_ids.as_deref())?;
    if prompt_ids.iter().any(|&id| (id as usize) >= vocab) {
        bail!("prompt token ids must be < {vocab} (student vocab size); got {prompt_ids:?}");
    }
    let eval_ids = match args.eval_ids.as_deref() {
        Some(raw) => parse_prompt_ids(Some(raw))?,
        None => prompt_ids.clone(),
    };
    if eval_ids.iter().any(|&id| (id as usize) >= vocab) {
        bail!("eval token ids must be < {vocab} (student vocab size); got {eval_ids:?}");
    }
    #[cfg(feature = "cuda")]
    let infer_student = load_opd_infer_student(
        student_dir,
        prompt_ids.len() + args.rollout_len + 32,
        train_backend.clone(),
        vocab,
        &args.runtime,
    )?;
    #[cfg(not(feature = "cuda"))]
    let _ = &train_backend;

    let mut optimizer = AdamW::new(args.lr, (0.9, 0.999), 1.0e-8, 0.0);
    let lr_schedule =
        OpdLrSchedule::new(args.lr_schedule, args.lr, args.lr_warmup_steps, args.steps)?;
    let rollout_sampling = rollout_sampling_params(
        args.rollout_temperature,
        args.rollout_top_p,
        args.rollout_top_k,
        args.rollout_seed,
    );
    let step_cfg = OpdStepConfig {
        rollout_len: args.rollout_len,
        rollout_sampling,
        grad_clip: args.grad_clip,
    };
    let kl_direction = kl_direction_arg(args.kl_direction);

    let gate_on = args.gate_every_n > 0;
    let mut snap = ema
        .snapshot(&student, &optimizer, &mut store)
        .context("initial EMA snapshot")?;
    let mut nll_baseline = if gate_on {
        heldout_nll(&student, &eval_ids, vocab, &mut store)?
    } else {
        f32::INFINITY
    };
    if gate_on && !nll_baseline.is_finite() {
        bail!(
            "initial held-out NLL is non-finite ({nll_baseline}); the student is degenerate \
             — cannot establish a no-regression gate baseline. Check the loaded adapter/weights."
        );
    }
    if gate_on && !args.json {
        println!("gate baseline_nll {nll_baseline:.6}");
    }

    let mut losses: Vec<f32> = Vec::with_capacity(args.steps);
    let mut reverts = 0usize;
    for step in 1..=args.steps {
        let _step_lr = lr_schedule.apply_to_optimizer(&mut optimizer, (step - 1) as u64);
        let mut step_profile = opd_step_profile_enabled().then(train::opd::OpdStepProfile::default);
        #[cfg(feature = "cuda")]
        let infer_rollout = infer_student
            .as_ref()
            .map(|student| train::opd::InferRolloutCtx {
                student,
                lora_config: lora,
            });
        let outcome = {
            let teacher = ema.as_teacher();
            opd_step_with_teacher_forward_profiled_gkd_anchor(
                &student,
                &teacher,
                &prompt_ids,
                step_cfg.clone(),
                &student_trainable,
                &mut optimizer,
                &mut store,
                &mut tape,
                GkdLossConfig {
                    lambda: args.gkd_lambda,
                    sft_anchor: GkdSftAnchor::StudentRollout,
                    corpus_tokens: None,
                    kl_chunk_size: GkdLossConfig::default().kl_chunk_size,
                    kl_direction,
                    kl_temperature: args.kl_temperature,
                    kl_beta: args.kl_beta,
                    teacher_topk: args.teacher_topk,
                    fused_distill: args.fused_distill,
                    logits_window_size: None,
                    kl_mask: OpdKlMask::CompletionOnly,
                },
                None,
                #[cfg(feature = "cuda")]
                infer_rollout,
                step_profile.as_mut(),
            )
            .with_context(|| format!("self-opd step {step} failed"))?
        };
        ema.update(&student, &mut store, args.ema_alpha)
            .with_context(|| format!("EMA update at step {step}"))?;
        losses.push(outcome.loss);

        let mut gate_action = "none";
        let mut gate_nll = f32::NAN;
        if gate_on && step % args.gate_every_n == 0 {
            gate_nll = heldout_nll(&student, &eval_ids, vocab, &mut store)?;
            // A non-finite gate NLL means the update diverged — `NaN > x` is false,
            // so without this guard the accept branch would store NaN as the baseline
            // and permanently disable the gate. Treat it as a regression → revert.
            if !gate_nll.is_finite() || gate_nll > nll_baseline * (1.0 + args.gate_regress_tol) {
                ema.restore(&snap, &student, &mut optimizer, &mut store)
                    .with_context(|| format!("EMA revert at step {step}"))?;
                reverts += 1;
                gate_action = "revert";
            } else {
                nll_baseline = gate_nll;
                gate_action = "accept";
            }
            // Re-snapshot from the post-gate (restored-or-accepted) good state.
            snap = ema
                .snapshot(&student, &optimizer, &mut store)
                .with_context(|| format!("EMA re-snapshot at step {step}"))?;
        }

        if !args.json {
            let grad_norm = current_grad_norm(&student_trainable, &store);
            if gate_on && step % args.gate_every_n == 0 {
                println!(
                    "step {step}/{total} loss {loss:.6} grad_norm {grad_norm:.6} rollout_len {rl} \
                     gate {gate_action} nll {gate_nll:.6} baseline {nll_baseline:.6}",
                    total = args.steps,
                    loss = outcome.loss,
                    rl = outcome.rollout_len,
                );
            } else {
                println!(
                    "step {step}/{total} loss {loss:.6} grad_norm {grad_norm:.6} rollout_len {rl}",
                    total = args.steps,
                    loss = outcome.loss,
                    rl = outcome.rollout_len,
                );
            }
            if let Some(profile) = step_profile.as_ref() {
                print_opd_step_profile(step, profile);
            }
        }
        maybe_save_full_student_checkpoint(
            "self-opd",
            args.save_checkpoint.as_deref(),
            args.save_every,
            step,
            args.steps,
            student_dir,
            &student,
            &mut store,
            &mut tape,
        )?;
    }

    if args.steps == 0 {
        maybe_save_full_student_checkpoint(
            "self-opd",
            args.save_checkpoint.as_deref(),
            args.save_every,
            0,
            0,
            student_dir,
            &student,
            &mut store,
            &mut tape,
        )?;
    }

    if args.json {
        let report = serde_json::json!({
            "mode": "self-opd",
            "backend": backend_label,
            "student_model": student_dir.display().to_string(),
            "steps": args.steps,
            "rollout_len": args.rollout_len,
            "lr": args.lr,
            "ema_alpha": args.ema_alpha,
            "gkd_lambda": args.gkd_lambda,
            "kl_temperature": args.kl_temperature,
            "kl_beta": args.kl_beta,
            "teacher_topk": args.teacher_topk,
            "gate_every_n": args.gate_every_n,
            "gate_regress_tol": args.gate_regress_tol,
            "gate_reverts": reverts,
            "final_nll_baseline": nll_baseline,
            "losses": losses,
            "prompt_ids": prompt_ids,
            "eval_ids": eval_ids,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_target_set": args.lora_target_set,
            "save_checkpoint": args.save_checkpoint.as_ref().map(|path| path.display().to_string()),
            "save_every": args.save_every,
            "vocab_size": vocab,
            "hidden_size": cfg.hidden_size,
            "num_hidden_layers": cfg.num_hidden_layers,
        });
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!(
            "ARLE train self-opd: ran {} step(s) on Qwen3.x (vocab={}, hidden={}, layers={}, \
             ema_alpha={}, gkd_lambda={}, gate_every_n={}, reverts={}, backend={})",
            args.steps,
            vocab,
            cfg.hidden_size,
            cfg.num_hidden_layers,
            args.ema_alpha,
            args.gkd_lambda,
            args.gate_every_n,
            reverts,
            backend_label,
        );
    }
    Ok(())
}

fn run_self_opd_smoke(args: TrainSelfOpdArgs) -> Result<()> {
    use autograd::{Tape, optim::AdamW};
    use train::{
        ema_self_teacher::EmaSelfTeacher,
        lora::LoraConfig,
        opd::{
            GkdLossConfig, GkdSftAnchor, OpdKlMask, OpdStepConfig,
            opd_step_with_teacher_forward_profiled_gkd_anchor,
        },
        qwen35::Qwen35Model,
    };

    let cfg = embedded_tiny_qwen35_config();
    reject_unimplemented_gkd_objectives(0.0, args.teacher_topk)?;
    let target_set = parse_lora_target_set(&args.lora_target_set)?;
    let lora = LoraConfig {
        rank: args.lora_rank.min(4),
        alpha: args.lora_alpha,
    };
    let prompt_ids = parse_prompt_ids(args.prompt_ids.as_deref())?;
    if prompt_ids.iter().any(|&id| (id as usize) >= cfg.vocab_size) {
        bail!(
            "smoke prompt token ids must be < {} (embedded tiny vocab size)",
            cfg.vocab_size
        );
    }

    let (mut store, _train_backend, backend_label) = build_opd_store(args.backend)?;
    apply_tape_dtype(&mut store, args.tape_dtype)?;
    let mut tape = Tape::new();
    let student = Qwen35Model::new_with_lora_targets(&cfg, lora, target_set, &mut store)
        .context("build smoke LoRA student")?;
    let mut ema = EmaSelfTeacher::from_student(&student, lora, target_set, &mut store)
        .context("build smoke EMA self-teacher")?;
    let student_trainable: Vec<TensorId> = student
        .all_parameter_ids()
        .into_iter()
        .filter(|id| store.get(*id).is_some_and(|tensor| tensor.requires_grad))
        .collect();

    let mut optimizer = AdamW::new(args.lr, (0.9, 0.999), 1.0e-8, 0.0);
    let lr_schedule =
        OpdLrSchedule::new(args.lr_schedule, args.lr, args.lr_warmup_steps, args.steps)?;
    let rollout_sampling = rollout_sampling_params(
        args.rollout_temperature,
        args.rollout_top_p,
        args.rollout_top_k,
        args.rollout_seed,
    );
    let step_cfg = OpdStepConfig {
        rollout_len: args.rollout_len,
        rollout_sampling,
        grad_clip: args.grad_clip,
    };
    let kl_direction = kl_direction_arg(args.kl_direction);

    let mut losses: Vec<f32> = Vec::with_capacity(args.steps);
    for step in 1..=args.steps {
        let _step_lr = lr_schedule.apply_to_optimizer(&mut optimizer, (step - 1) as u64);
        let outcome = {
            let teacher = ema.as_teacher();
            opd_step_with_teacher_forward_profiled_gkd_anchor(
                &student,
                &teacher,
                &prompt_ids,
                step_cfg.clone(),
                &student_trainable,
                &mut optimizer,
                &mut store,
                &mut tape,
                GkdLossConfig {
                    lambda: args.gkd_lambda,
                    sft_anchor: GkdSftAnchor::StudentRollout,
                    corpus_tokens: None,
                    kl_chunk_size: GkdLossConfig::default().kl_chunk_size,
                    kl_direction,
                    kl_temperature: args.kl_temperature,
                    kl_beta: args.kl_beta,
                    teacher_topk: args.teacher_topk,
                    fused_distill: args.fused_distill,
                    logits_window_size: None,
                    kl_mask: OpdKlMask::CompletionOnly,
                },
                None,
                #[cfg(feature = "cuda")]
                None,
                None,
            )
            .with_context(|| format!("self-opd smoke step {step} failed"))?
        };
        ema.update(&student, &mut store, args.ema_alpha)
            .with_context(|| format!("EMA update at smoke step {step}"))?;
        losses.push(outcome.loss);
        if !args.json {
            println!(
                "step {step}/{total} loss {loss:.6} rollout_len {rl}",
                total = args.steps,
                loss = outcome.loss,
                rl = outcome.rollout_len,
            );
        }
    }

    if args.json {
        let report = serde_json::json!({
            "mode": "self-opd-smoke",
            "backend": backend_label,
            "steps": args.steps,
            "rollout_len": args.rollout_len,
            "lr": args.lr,
            "ema_alpha": args.ema_alpha,
            "gkd_lambda": args.gkd_lambda,
            "kl_temperature": args.kl_temperature,
            "kl_beta": args.kl_beta,
            "teacher_topk": args.teacher_topk,
            "losses": losses,
            "prompt_ids": prompt_ids,
        });
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!(
            "ARLE train self-opd smoke: ran {} step(s) on tiny Qwen3.5 (vocab={}, hidden={}, \
             layers={}, ema_alpha={}, gkd_lambda={}, backend={})",
            args.steps,
            cfg.vocab_size,
            cfg.hidden_size,
            cfg.num_hidden_layers,
            args.ema_alpha,
            args.gkd_lambda,
            backend_label,
        );
    }
    Ok(())
}

fn opd_progress_style() -> Result<ProgressStyle> {
    ProgressStyle::with_template(
        "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} \
         avg_loss={msg} eta={eta_precise}",
    )
    .map(|style| style.progress_chars("=>-"))
    .context("build OPD progress style")
}

fn current_grad_norm(params: &[TensorId], store: &TensorStore) -> f32 {
    let mut total_sq_norm = 0.0_f64;
    for &param_id in params {
        let Some(grad_id) = store.get(param_id).and_then(|tensor| tensor.grad) else {
            continue;
        };
        let Some(grad) = store.get(grad_id) else {
            continue;
        };
        total_sq_norm += grad
            .data
            .iter()
            .map(|&value| {
                let value = f64::from(value);
                value * value
            })
            .sum::<f64>();
    }
    total_sq_norm.sqrt() as f32
}

fn opd_summary(step_metrics: &[OpdStepMetric]) -> OpdSummary {
    let final_loss = step_metrics.last().map(|metric| metric.loss);
    let mean_loss = if step_metrics.is_empty() {
        None
    } else {
        Some(step_metrics.iter().map(|metric| metric.loss).sum::<f32>() / step_metrics.len() as f32)
    };
    let min_loss = step_metrics
        .iter()
        .map(|metric| metric.loss)
        .min_by(f32::total_cmp);
    let max_loss = step_metrics
        .iter()
        .map(|metric| metric.loss)
        .max_by(f32::total_cmp);
    OpdSummary {
        step_count: step_metrics.len(),
        final_loss,
        mean_loss,
        min_loss,
        max_loss,
    }
}

#[cfg(feature = "cuda")]
fn load_opd_infer_teacher(
    teacher_dir: &Path,
    max_seq_len: usize,
    train_backend: std::sync::Arc<dyn autograd::Backend>,
    vocab_size: usize,
) -> Result<train::teacher_infer::InferTeacher> {
    use std::sync::{Arc, Mutex};

    use infer_api::{EngineLoadConfig, LoadedInferenceEngine};

    let max_seq_len = max_seq_len.max(128);
    eprintln!(
        "[arle train opd] loading infer teacher from {} (max_seq_len={max_seq_len})",
        teacher_dir.display()
    );
    let engine = LoadedInferenceEngine::load_with_config(
        teacher_dir
            .to_str()
            .ok_or_else(|| anyhow!("teacher model path is not valid UTF-8"))?,
        true,
        EngineLoadConfig::single_sequence(max_seq_len),
    )
    .with_context(|| format!("load infer teacher from {}", teacher_dir.display()))?;

    Ok(train::teacher_infer::InferTeacher::new(
        Arc::new(Mutex::new(engine)),
        train_backend,
        vocab_size,
    ))
}

/// Install the `--rollout-engine` selection (CUDA only; inert on CPU, which has
/// no infer engine). Unset → `infer` default.
#[cfg(feature = "cuda")]
fn apply_opd_rollout_engine(engine: Option<crate::args::OpdRolloutEngineArg>) {
    if let Some(engine) = engine {
        train::opd::set_infer_rollout_override(engine == crate::args::OpdRolloutEngineArg::Infer);
    }
}

#[cfg(not(feature = "cuda"))]
fn apply_opd_rollout_engine(_engine: Option<crate::args::OpdRolloutEngineArg>) {}

#[cfg(feature = "cuda")]
fn load_opd_infer_student(
    student_dir: &Path,
    max_seq_len: usize,
    train_backend: std::sync::Arc<dyn autograd::Backend>,
    vocab_size: usize,
    runtime: &crate::args::OpdRuntimeArgs,
) -> Result<Option<train::infer_student::InferStudent>> {
    if !train::opd::infer_rollout_flag_enabled() {
        return Ok(None);
    }

    use std::sync::{Arc, Mutex};

    use infer_api::{EngineLoadConfig, LoadedInferenceEngine};

    let max_seq_len = max_seq_len.max(128);
    eprintln!(
        "[arle train opd] loading infer rollout student from {} (max_seq_len={max_seq_len})",
        student_dir.display()
    );
    let engine = LoadedInferenceEngine::load_with_config(
        student_dir
            .to_str()
            .ok_or_else(|| anyhow!("student model path is not valid UTF-8"))?,
        true,
        EngineLoadConfig {
            dspark_draft_model: runtime.dspark_draft_model.clone(),
            dspark_sps_bias_ms: runtime.dspark_sps_bias_ms,
            dspark_sps_row_ms: runtime.dspark_sps_row_ms,
            mem_fraction_static: runtime.rollout_mem_fraction,
            // Whole-step decode graph for the rollout: eager per-token decode is
            // host-launch-bound (~156 ms/token), the OPD step's dominant cost.
            cuda: runtime.cuda_runtime_flags(),
            ..EngineLoadConfig::single_sequence(max_seq_len)
        },
    )
    .with_context(|| format!("load infer rollout student from {}", student_dir.display()))?;

    Ok(Some(train::infer_student::InferStudent::new(
        Arc::new(Mutex::new(engine)),
        train_backend,
        vocab_size,
    )))
}

#[cfg(feature = "cuda")]
fn maybe_preoffload_infer_student_before_teacher(
    infer_student: &Option<train::infer_student::InferStudent>,
    train_backend: &std::sync::Arc<dyn autograd::Backend>,
) -> Result<()> {
    if !train::opd::engine_offload_mode().offloads_student() {
        return Ok(());
    }
    let Some(student) = infer_student.as_ref() else {
        return Ok(());
    };

    train_backend
        .device_synchronize()
        .context("synchronize train backend before infer rollout student pre-teacher offload")?;
    let freed = student
        .offload_engine_weights()
        .context("offload infer rollout student before infer teacher load")?;
    eprintln!(
        "opd_engine_offload student_pre_teacher_offloaded freed_bytes={freed} freed_mib={:.1}",
        freed as f64 / (1024.0 * 1024.0)
    );
    Ok(())
}

#[cfg(feature = "cuda")]
fn maybe_preoffload_infer_teacher_before_steps(
    teacher: &OpdCliTeacher<'_>,
    train_backend: &std::sync::Arc<dyn autograd::Backend>,
) -> Result<()> {
    if !train::opd::engine_offload_mode().offloads_teacher() {
        return Ok(());
    }

    train_backend
        .device_synchronize()
        .context("synchronize train backend before infer teacher pre-step offload")?;
    let freed = train::teacher_infer::TeacherForward::offload_engine_weights(teacher)
        .map_err(|err| anyhow!("offload infer teacher before first OPD step failed: {err}"))?;
    eprintln!(
        "opd_engine_offload teacher_pre_step_offloaded freed_bytes={freed} freed_mib={:.1}",
        freed as f64 / (1024.0 * 1024.0)
    );
    Ok(())
}

/// Bf16 tape is CUDA-only; bail instead of silently training on an f32 tape.
fn apply_tape_dtype(store: &mut TensorStore, requested: TapeDtypeArg) -> Result<()> {
    store.set_tape_dtype(requested.as_tape_dtype());
    if requested == TapeDtypeArg::Bf16 && store.backend().tape_dtype() != autograd::TapeDtype::Bf16
    {
        bail!(
            "--tape-dtype bf16 is a no-op on the {:?} backend; use --backend cuda",
            store.backend().device()
        );
    }
    Ok(())
}

#[allow(unused_variables)]
fn build_opd_store(
    arg: OpdBackendArg,
) -> Result<(
    autograd::TensorStore,
    std::sync::Arc<dyn autograd::Backend>,
    &'static str,
)> {
    #[cfg(feature = "cuda")]
    {
        use std::sync::Arc;
        let want_cuda = matches!(arg, OpdBackendArg::Cuda | OpdBackendArg::Auto);
        if want_cuda {
            // Mesh env comes from the launcher; both axes size<=1 keeps the
            // single-card new(0) byte-identical.
            let cp = train::context_parallel::CpContext::from_env();
            let dp = train::context_parallel::DpContext::from_env();
            let backend = if cp.is_enabled() || dp.is_enabled() {
                #[cfg(feature = "nccl")]
                {
                    let ordinal = std::env::var("INFER_CUDA_DEVICE")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0);
                    let uid = infer_api::nccl_unique_id_from_env()
                        .context("CP/DP: read INFER_NCCL_UNIQUE_ID")?;
                    let world_rank = train::context_parallel::world_rank(cp, dp);
                    let seq_group =
                        (cp.is_enabled() && dp.is_enabled()).then_some((dp.rank, cp.size, cp.rank));
                    let backend = autograd::backend_cuda::CudaBackend::new_with_mesh(
                        ordinal,
                        uid,
                        cp.size * dp.size,
                        world_rank,
                        seq_group,
                    )
                    .context("init CUDA+NCCL backend for the CP×DP mesh")?;
                    if !backend.has_collective() {
                        bail!(
                            "multi-rank mesh (cp={}×dp={}) got a CUDA backend without an NCCL \
                             communicator; refusing the silent host-transport fallback",
                            cp.size,
                            dp.size
                        );
                    }
                    backend
                }
                #[cfg(not(feature = "nccl"))]
                {
                    bail!(
                        "context/data parallelism (ARLE_TRAIN_CP_SIZE>1 or ARLE_TRAIN_DP_SIZE>1) \
                         requires the nccl feature; rebuild with --features cuda,nccl"
                    );
                }
            } else {
                autograd::backend_cuda::CudaBackend::new(0).context("init CUDA backend (GPU 0)")?
            };
            let backend: Arc<dyn autograd::Backend> = Arc::new(backend);
            let label = match (cp.is_enabled(), dp.is_enabled()) {
                (true, true) => "cuda:cp×dp",
                (true, false) => "cuda:cp",
                (false, true) => "cuda:dp",
                (false, false) => "cuda:0",
            };
            return Ok((
                autograd::TensorStore::with_backend(backend.clone()),
                backend,
                label,
            ));
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        if matches!(arg, OpdBackendArg::Cuda) {
            bail!(
                "arle was built without the cuda feature; rebuild with \
                 `cargo build --release --features cuda` to use --backend cuda"
            );
        }
    }
    let backend: std::sync::Arc<dyn autograd::Backend> = std::sync::Arc::new(autograd::CpuBackend);
    Ok((
        autograd::TensorStore::with_backend(backend.clone()),
        backend,
        "cpu",
    ))
}

fn parse_prompt_ids(raw: Option<&str>) -> Result<Vec<u32>> {
    let raw = raw.unwrap_or("1,3,8");
    raw.split(',')
        .map(|piece| {
            piece
                .trim()
                .parse::<u32>()
                .with_context(|| format!("invalid prompt id `{piece}` (expected u32)"))
        })
        .collect()
}

fn embedded_tiny_qwen35_config() -> Qwen35Config {
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
        rope_cache_len_hint: Some(64),
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

fn estimate_from_model_dir(
    model_source: &Path,
    args: &TrainEstimateMemoryArgs,
) -> Result<EstimateMemoryReport> {
    let model = inspect_model_source(model_source, false)?;
    let local_dir = model.local_dir_path().ok_or_else(|| {
        anyhow!("estimate-memory requires a local model dir or cached HF model id")
    })?;
    let summary = inspect_resolved_model_dir(&local_dir)?;
    let trainable_params = lora_param_count(&summary.config, args.lora_rank);
    let checkpoint_bytes = bytes_for_params(summary.param_count, args.save_dtype.bytes_per_param());
    let adapter_checkpoint_bytes =
        bytes_for_params(trainable_params, args.save_dtype.bytes_per_param());
    Ok(EstimateMemoryReport {
        mode: "sft-lora".to_string(),
        family: summary.family.clone(),
        model_dir: Some(local_dir.display().to_string()),
        tokenizer_path: summary.tokenizer_path.clone(),
        vocab_size: Some(summary.vocab_size),
        batch: args.batch,
        seq: args.seq,
        param_count: summary.param_count,
        trainable_param_count: trainable_params,
        weight_bytes_fp32: bytes_for_params(summary.param_count, 4),
        gradient_bytes_fp32: bytes_for_params(trainable_params, 4),
        adam_state_bytes_fp32: bytes_for_params(trainable_params, 8),
        checkpoint_bytes,
        adapter_checkpoint_bytes: Some(adapter_checkpoint_bytes),
        activation_floor_bytes: activation_floor_bytes(summary.hidden_size, args.batch, args.seq),
        save_dtype: args.save_dtype.as_train_dtype().to_string(),
    })
}

fn estimate_from_scratch(args: &TrainEstimateMemoryArgs) -> Result<EstimateMemoryReport> {
    let tokenizer_source = args
        .tokenizer
        .as_deref()
        .ok_or_else(|| anyhow!("estimate-memory requires either --model or --tokenizer"))?;
    let tokenizer_path = resolve_local_tokenizer_path(tokenizer_source)?;
    let tokenizer = ChatTokenizer::from_file(&tokenizer_path)?;
    let mut shape = ScratchShape::default();
    if let Some(preset) = args.preset {
        shape.apply_preset(preset);
    }
    shape.apply_overrides(
        args.hidden,
        args.layers,
        args.heads,
        args.kv_heads,
        args.head_dim,
        args.intermediate,
        args.max_pos,
        args.linear_attn_every,
    );
    let vocab_size = args.vocab_size.unwrap_or_else(|| tokenizer.vocab_size());
    let family = args
        .model_family
        .unwrap_or(ModelFamilyArg::Qwen35)
        .as_train_family()
        .to_string();
    let param_count = qwen35_param_count(&shape.qwen35_config(vocab_size));
    let hidden_size = shape.hidden;
    Ok(EstimateMemoryReport {
        mode: "scratch-pretrain".to_string(),
        family,
        model_dir: None,
        tokenizer_path: Some(tokenizer_path.display().to_string()),
        vocab_size: Some(vocab_size),
        batch: args.batch,
        seq: args.seq,
        param_count,
        trainable_param_count: param_count,
        weight_bytes_fp32: bytes_for_params(param_count, 4),
        gradient_bytes_fp32: bytes_for_params(param_count, 4),
        adam_state_bytes_fp32: bytes_for_params(param_count, 8),
        checkpoint_bytes: bytes_for_params(param_count, args.save_dtype.bytes_per_param()),
        adapter_checkpoint_bytes: None,
        activation_floor_bytes: activation_floor_bytes(hidden_size, args.batch, args.seq),
        save_dtype: args.save_dtype.as_train_dtype().to_string(),
    })
}

fn inspect_model_source(source: &Path, allow_download: bool) -> Result<ModelInspection> {
    let raw_source = source.display().to_string();
    let resolved_dir = if allow_download {
        Some(resolve_model_dir_allow_download(source)?)
    } else {
        resolve_model_dir_local_only(source)
    };
    let notes: Vec<String> = (!allow_download && resolved_dir.is_none())
        .then(|| "model source is not local/cached; dry-run skipped remote resolution".to_string())
        .into_iter()
        .collect();
    let summary = resolved_dir
        .as_deref()
        .map(inspect_resolved_model_dir)
        .transpose()?;

    Ok(ModelInspection {
        source: raw_source,
        resolved_dir: resolved_dir.as_ref().map(|path| path.display().to_string()),
        config_path: summary.as_ref().map(|s| s.config_path.clone()),
        tokenizer_path: summary.as_ref().and_then(|s| s.tokenizer_path.clone()),
        generation_config_path: summary
            .as_ref()
            .and_then(|s| s.generation_config_path.clone()),
        family: summary.as_ref().map(|s| s.family.clone()),
        notes,
    })
}

fn inspect_resolved_model_dir(model_dir: &Path) -> Result<ModelDirSummary> {
    let config_path = model_dir.join("config.json");
    let config_value: serde_json::Value = serde_json::from_str(&fs::read_to_string(&config_path)?)
        .with_context(|| {
            format!(
                "reading model inspection config from {}",
                config_path.display()
            )
        })?;
    let is_deepseek_v4 = config_value
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|model_type| model_type == "deepseek_v4")
        || config_value
            .get("architectures")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|architectures| {
                architectures
                    .iter()
                    .any(|arch| arch.as_str() == Some("DeepseekV4ForCausalLM"))
            });
    if is_deepseek_v4 {
        let cfg = DeepSeekV4Config::from_json_value(&config_value)?;
        return Ok(ModelDirSummary {
            family: "deepseek-v4".to_string(),
            config: ResolvedModelConfig::DeepSeekV4,
            config_path: config_path.display().to_string(),
            tokenizer_path: existing_display_path(model_dir.join("tokenizer.json")),
            generation_config_path: existing_display_path(model_dir.join("generation_config.json")),
            vocab_size: cfg.vocab_size,
            hidden_size: cfg.hidden_size,
            param_count: deepseek_v4_param_count(&cfg),
        });
    }

    let family = match resolve_model_family(&config_path, ModelFamily::Auto)? {
        ModelFamily::Qwen35 => "qwen35",
        ModelFamily::Auto => unreachable!("auto must resolve to a concrete family"),
    };
    match family {
        "qwen35" => {
            let cfg = Qwen35Config::from_json_file(&config_path)?;
            Ok(ModelDirSummary {
                family: "qwen35".to_string(),
                config: ResolvedModelConfig::Qwen35(Box::new(cfg.clone())),
                config_path: config_path.display().to_string(),
                tokenizer_path: existing_display_path(model_dir.join("tokenizer.json")),
                generation_config_path: existing_display_path(
                    model_dir.join("generation_config.json"),
                ),
                vocab_size: cfg.vocab_size,
                hidden_size: cfg.hidden_size,
                param_count: qwen35_param_count(&cfg),
            })
        }
        _ => unreachable!("family resolver returned an unknown family"),
    }
}

fn resolve_model_dir_allow_download(source: &Path) -> Result<PathBuf> {
    let source_text = source.display().to_string();
    infer_util::hf_hub::resolve_model_path(&source_text)
        .with_context(|| format!("resolving model source {source_text}"))
}

fn resolve_model_dir_local_only(source: &Path) -> Option<PathBuf> {
    let source_text = source.display().to_string();
    infer_util::hf_hub::resolve_local_model_path(&source_text)
}

pub(crate) fn resolve_local_tokenizer_path(source: &Path) -> Result<PathBuf> {
    if source.is_file() {
        return Ok(source.to_path_buf());
    }
    if source.is_dir() {
        let candidate = source.join("tokenizer.json");
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    let source_text = source.display().to_string();
    if let Some(model_dir) = infer_util::hf_hub::resolve_local_model_path(&source_text) {
        let candidate = model_dir.join("tokenizer.json");
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    bail!(
        "tokenizer source {} must be tokenizer.json or a local model dir containing tokenizer.json",
        source.display()
    );
}

fn qwen35_param_count(cfg: &Qwen35Config) -> u64 {
    let embed = mul_u64(cfg.vocab_size, cfg.hidden_size);
    let lm_head = if cfg.tie_word_embeddings { 0 } else { embed };
    let common = mul_u64(2, cfg.hidden_size)
        + mul_u64(cfg.hidden_size, cfg.intermediate_size) * 2
        + mul_u64(cfg.intermediate_size, cfg.hidden_size);
    let attention = cfg
        .layer_types
        .iter()
        .map(|layer_type| match layer_type {
            LayerType::FullAttention => {
                mul_u64(cfg.hidden_size, cfg.full_attn_q_proj_dim())
                    + mul_u64(cfg.hidden_size, cfg.full_attn_kv_dim()) * 2
                    + mul_u64(cfg.full_attn_q_dim(), cfg.hidden_size)
                    + mul_u64(2, cfg.head_dim)
            }
            LayerType::LinearAttention => {
                mul_u64(cfg.hidden_size, cfg.linear_attn_qkv_dim())
                    + mul_u64(cfg.hidden_size, cfg.linear_attn_z_dim())
                    + mul_u64(cfg.hidden_size, cfg.linear_num_value_heads) * 2
                    + mul_u64(cfg.linear_attn_qkv_dim(), cfg.linear_conv_kernel_dim)
                    + mul_u64(2, cfg.linear_num_value_heads)
                    + cfg.linear_value_head_dim as u64
                    + mul_u64(cfg.linear_attn_z_dim(), cfg.hidden_size)
            }
        })
        .sum::<u64>();
    embed
        + lm_head
        + (cfg.num_hidden_layers as u64).saturating_mul(common)
        + attention
        + cfg.hidden_size as u64
}

fn deepseek_v4_param_count(cfg: &DeepSeekV4Config) -> u64 {
    let embed = mul_u64(cfg.vocab_size, cfg.hidden_size);
    let lm_head = if cfg.tie_word_embeddings { 0 } else { embed };
    let hc_mix = (2 + cfg.hc_mult) * cfg.hc_mult;
    let hc_flat = cfg.hc_mult * cfg.hidden_size;
    let head_hc = mul_u64(cfg.hc_mult, hc_flat) + cfg.hc_mult as u64 + 1;
    let per_hc = mul_u64(hc_mix, hc_flat) + hc_mix as u64 + 3;
    let heads_per_group = cfg.num_attention_heads / cfg.o_groups;
    let base_attn = mul_u64(cfg.q_lora_rank, cfg.hidden_size)
        + cfg.q_lora_rank as u64
        + mul_u64(cfg.num_attention_heads * cfg.head_dim, cfg.q_lora_rank)
        + mul_u64(cfg.head_dim, cfg.hidden_size)
        + cfg.head_dim as u64
        + mul_u64(
            cfg.o_groups * cfg.o_lora_rank,
            heads_per_group * cfg.head_dim,
        )
        + mul_u64(cfg.hidden_size, cfg.o_groups * cfg.o_lora_rank)
        + cfg.num_attention_heads as u64;
    let expert = mul_u64(cfg.moe_intermediate_size, cfg.hidden_size) * 2
        + mul_u64(cfg.hidden_size, cfg.moe_intermediate_size);
    let routed_experts = (cfg.n_routed_experts as u64).saturating_mul(expert);
    let shared_experts = if cfg.n_shared_experts == 0 {
        0
    } else {
        let shared_intermediate = cfg.moe_intermediate_size * cfg.n_shared_experts;
        mul_u64(shared_intermediate, cfg.hidden_size) * 2
            + mul_u64(cfg.hidden_size, shared_intermediate)
    };
    let gate_bias_or_hash = cfg
        .n_routed_experts
        .max(cfg.vocab_size * cfg.num_experts_per_tok);
    let moe = mul_u64(cfg.n_routed_experts, cfg.hidden_size)
        + gate_bias_or_hash as u64
        + routed_experts
        + shared_experts;

    let layers = cfg
        .compress_ratios
        .iter()
        .copied()
        .map(|compress_ratio| {
            let compressor = cfg
                .compressor_shape(compress_ratio)
                .map(|shape| {
                    mul_u64(shape.wkv_rows, shape.wkv_cols)
                        + mul_u64(shape.wgate_rows, shape.wgate_cols)
                        + mul_u64(shape.ape_rows, shape.ape_cols)
                        + shape.norm_len as u64
                })
                .unwrap_or(0);
            let indexer = if cfg.attention_mode_for_compress_ratio(compress_ratio)
                == DeepSeekV4AttentionMode::CompressedSparse
            {
                let shape = cfg
                    .indexer_shape(compress_ratio)
                    .expect("CSA layer has indexer shape");
                let compressor = shape
                    .compressor
                    .as_ref()
                    .expect("CSA indexer always has a compressor");
                mul_u64(shape.wq_b_rows, shape.wq_b_cols)
                    + mul_u64(shape.weights_proj_rows, shape.weights_proj_cols)
                    + mul_u64(compressor.wkv_rows, compressor.wkv_cols)
                    + mul_u64(compressor.wgate_rows, compressor.wgate_cols)
                    + mul_u64(compressor.ape_rows, compressor.ape_cols)
                    + compressor.norm_len as u64
            } else {
                0
            };
            mul_u64(2, cfg.hidden_size) + per_hc * 2 + base_attn + compressor + indexer + moe
        })
        .sum::<u64>();

    let mtp = (cfg.num_nextn_predict_layers as u64).saturating_mul(
        mul_u64(7, cfg.hidden_size)
            + mul_u64(2, cfg.hidden_size * cfg.hidden_size)
            + per_hc * 2
            + head_hc
            + base_attn
            + moe,
    );

    embed + lm_head + cfg.hidden_size as u64 + head_hc + layers + mtp
}

fn lora_param_count(config: &ResolvedModelConfig, rank: usize) -> u64 {
    match config {
        ResolvedModelConfig::Qwen35(cfg) => {
            let common = lora_linear(cfg.hidden_size, cfg.intermediate_size, rank) * 2
                + lora_linear(cfg.intermediate_size, cfg.hidden_size, rank);
            let attention = cfg
                .layer_types
                .iter()
                .map(|layer_type| match layer_type {
                    LayerType::FullAttention => {
                        lora_linear(cfg.hidden_size, cfg.full_attn_q_proj_dim(), rank)
                            + lora_linear(cfg.hidden_size, cfg.full_attn_kv_dim(), rank) * 2
                            + lora_linear(cfg.full_attn_q_dim(), cfg.hidden_size, rank)
                    }
                    LayerType::LinearAttention => {
                        lora_linear(cfg.hidden_size, cfg.linear_attn_qkv_dim(), rank)
                            + lora_linear(cfg.hidden_size, cfg.linear_attn_z_dim(), rank)
                            + lora_linear(cfg.hidden_size, cfg.linear_num_value_heads, rank) * 2
                            + lora_linear(cfg.linear_attn_z_dim(), cfg.hidden_size, rank)
                    }
                })
                .sum::<u64>();
            (cfg.num_hidden_layers as u64).saturating_mul(common) + attention
        }
        ResolvedModelConfig::DeepSeekV4 => 0,
    }
}

fn activation_floor_bytes(hidden_size: usize, batch: usize, seq: usize) -> u64 {
    mul_u64(hidden_size, batch * seq * 4)
}

fn bytes_for_params(param_count: u64, bytes_per_param: u64) -> u64 {
    param_count.saturating_mul(bytes_per_param)
}

fn lora_linear(in_features: usize, out_features: usize, rank: usize) -> u64 {
    mul_u64(rank, in_features + out_features)
}

fn mul_u64(lhs: usize, rhs: usize) -> u64 {
    (lhs as u64).saturating_mul(rhs as u64)
}

fn existing_display_path(path: PathBuf) -> Option<String> {
    path.is_file().then(|| path.display().to_string())
}

fn format_count(count: u64) -> String {
    if count >= 1_000_000_000 {
        format!("{:.2}B", count as f64 / 1_000_000_000.0)
    } else if count >= 1_000_000 {
        format!("{:.2}M", count as f64 / 1_000_000.0)
    } else if count >= 1_000 {
        format!("{:.2}K", count as f64 / 1_000.0)
    } else {
        count.to_string()
    }
}

fn format_bytes(bytes: u64) -> String {
    let kib = 1024.0;
    let mib = kib * 1024.0;
    let gib = mib * 1024.0;
    let bytes = bytes as f64;
    if bytes >= gib {
        format!("{:.2} GiB", bytes / gib)
    } else if bytes >= mib {
        format!("{:.2} MiB", bytes / mib)
    } else if bytes >= kib {
        format!("{:.2} KiB", bytes / kib)
    } else {
        format!("{} B", bytes as u64)
    }
}

fn exit_from_result(result: Result<()>) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("[ARLE train] error: {err:#}");
            ExitCode::FAILURE
        }
    }
}

fn gpu_label(info: &hardware::GpuInfo) -> String {
    match info {
        hardware::GpuInfo::Cuda { name, vram_gb } => format!("{name} ({vram_gb:.1} GB VRAM)"),
        hardware::GpuInfo::Metal {
            chip,
            unified_memory_gb,
            recommended_working_set_gb,
        } => {
            if let Some(working_set) = recommended_working_set_gb {
                format!(
                    "{chip} ({unified_memory_gb:.1} GB unified, {working_set:.1} GB working set)"
                )
            } else {
                format!("{chip} ({unified_memory_gb:.1} GB unified)")
            }
        }
        hardware::GpuInfo::None => "none".to_string(),
    }
}

// cfg arms are additive: in single-backend builds only one `return` is live, so
// clippy sees it as needless; the `return`s are required so the multi-backend
// (`cuda` + `metal`) build still compiles. Matches `CompiledBackend::detect`.
#[allow(clippy::needless_return)]
fn default_train_backend() -> &'static str {
    #[cfg(feature = "cuda")]
    {
        return "cuda";
    }
    #[cfg(all(not(feature = "cuda"), feature = "metal"))]
    {
        return "metal";
    }
    #[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
    {
        "cpu"
    }
}

#[derive(Debug, Clone)]
struct ScratchShape {
    hidden: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    intermediate: usize,
    max_pos: usize,
    linear_attn_every: usize,
}

impl Default for ScratchShape {
    fn default() -> Self {
        Self {
            hidden: 256,
            layers: 4,
            heads: 4,
            kv_heads: 2,
            head_dim: 64,
            intermediate: 512,
            max_pos: 512,
            linear_attn_every: 0,
        }
    }
}

impl ScratchShape {
    fn apply_preset(&mut self, preset: PretrainPresetArg) {
        match preset {
            PretrainPresetArg::Tiny3m => {
                self.hidden = 96;
                self.layers = 2;
                self.heads = 3;
                self.kv_heads = 3;
                self.head_dim = 32;
                self.intermediate = 192;
                self.max_pos = 256;
                self.linear_attn_every = 0;
            }
            PretrainPresetArg::Small25m => {
                self.hidden = 160;
                self.layers = 2;
                self.heads = 5;
                self.kv_heads = 5;
                self.head_dim = 32;
                self.intermediate = 320;
                self.max_pos = 512;
                self.linear_attn_every = 0;
            }
            PretrainPresetArg::Small30m => {
                self.hidden = 192;
                self.layers = 2;
                self.heads = 6;
                self.kv_heads = 3;
                self.head_dim = 32;
                self.intermediate = 384;
                self.max_pos = 512;
                self.linear_attn_every = 0;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_overrides(
        &mut self,
        hidden: Option<usize>,
        layers: Option<usize>,
        heads: Option<usize>,
        kv_heads: Option<usize>,
        head_dim: Option<usize>,
        intermediate: Option<usize>,
        max_pos: Option<usize>,
        linear_attn_every: Option<usize>,
    ) {
        if let Some(hidden) = hidden {
            self.hidden = hidden;
        }
        if let Some(layers) = layers {
            self.layers = layers;
        }
        if let Some(heads) = heads {
            self.heads = heads;
        }
        if let Some(kv_heads) = kv_heads {
            self.kv_heads = kv_heads;
        }
        if let Some(head_dim) = head_dim {
            self.head_dim = head_dim;
        }
        if let Some(intermediate) = intermediate {
            self.intermediate = intermediate;
        }
        if let Some(max_pos) = max_pos {
            self.max_pos = max_pos;
        }
        if let Some(linear_attn_every) = linear_attn_every {
            self.linear_attn_every = linear_attn_every;
        }
    }

    fn qwen35_config(&self, vocab_size: usize) -> Qwen35Config {
        let mut layer_types = vec![LayerType::FullAttention; self.layers];
        if self.linear_attn_every > 0 {
            for (layer_idx, layer_type) in layer_types.iter_mut().enumerate().take(self.layers) {
                if (layer_idx + 1) % self.linear_attn_every == 0 {
                    *layer_type = LayerType::LinearAttention;
                }
            }
        }
        Qwen35Config {
            hidden_size: self.hidden,
            intermediate_size: self.intermediate,
            num_hidden_layers: self.layers,
            vocab_size,
            rms_norm_eps: 1.0e-6,
            stop_token_ids: vec![vocab_size.saturating_sub(1) as u32],
            bos_token_id: Some(1),
            eos_token_id: vocab_size.saturating_sub(1) as u32,
            tie_word_embeddings: true,
            num_attention_heads: self.heads,
            num_key_value_heads: self.kv_heads,
            head_dim: self.head_dim,
            linear_num_key_heads: self.heads,
            linear_key_head_dim: self.head_dim,
            linear_num_value_heads: self.heads,
            linear_value_head_dim: self.head_dim,
            linear_conv_kernel_dim: 4,
            rope_theta: 1_000_000.0,
            rope_scaling: None,
            partial_rotary_factor: 1.0,
            rotary_dim: self.head_dim,
            rope_cache_len_hint: Some(self.max_pos),
            layer_types,
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
}

#[derive(Debug, Serialize)]
struct TrainEnvReport {
    version: &'static str,
    train_default_backend: &'static str,
    compiled_infer_backend: &'static str,
    supports_inference: bool,
    cpu: String,
    cpu_cores: usize,
    total_ram_gb: f64,
    available_ram_gb: f64,
    gpu: String,
    hf_cache_root: String,
    cwd: String,
    commands: &'static [&'static str],
}

#[derive(Debug, Clone, Serialize)]
struct ModelInspection {
    source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    resolved_dir: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    config_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tokenizer_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    family: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    notes: Vec<String>,
}

impl ModelInspection {
    fn local_dir_path(&self) -> Option<PathBuf> {
        self.resolved_dir.as_ref().map(PathBuf::from)
    }
}

#[derive(Debug)]
struct ModelDirSummary {
    family: String,
    config: ResolvedModelConfig,
    config_path: String,
    tokenizer_path: Option<String>,
    generation_config_path: Option<String>,
    vocab_size: usize,
    hidden_size: usize,
    param_count: u64,
}

#[derive(Debug)]
enum ResolvedModelConfig {
    Qwen35(Box<Qwen35Config>),
    DeepSeekV4,
}

#[derive(Debug, Serialize)]
struct EstimateMemoryReport {
    mode: String,
    family: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_dir: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tokenizer_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    vocab_size: Option<usize>,
    batch: usize,
    seq: usize,
    param_count: u64,
    trainable_param_count: u64,
    weight_bytes_fp32: u64,
    gradient_bytes_fp32: u64,
    adam_state_bytes_fp32: u64,
    checkpoint_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    adapter_checkpoint_bytes: Option<u64>,
    activation_floor_bytes: u64,
    save_dtype: String,
}

impl SaveDtypeArg {
    fn bytes_per_param(self) -> u64 {
        match self {
            SaveDtypeArg::F32 => 4,
            SaveDtypeArg::Bf16 => 2,
        }
    }
}
