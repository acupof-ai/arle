use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, ExitCode, Output},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow, bail};
use qwen3_spec::Qwen3Config;
use qwen35_spec::{LayerType, Qwen35Config};
use serde::Serialize;
use train::{
    model_family::{ModelFamily, resolve_model_family},
    tokenizer::{ChatTokenizer, write_chatml_wordlevel_tokenizer},
};

use crate::{
    args::{
        BackendArg, DataArgs, DataCommand, DataConvertArgs, DataDownloadArgs, DatasetFormatArg,
        ModelFamilyArg, PretrainPresetArg, RenderArgs, SaveDtypeArg, TrainArgs, TrainCommand,
        TrainEnvArgs, TrainEstimateMemoryArgs, TrainEvalArgs, TrainGrpoArgs, TrainMultiTurnArgs,
        TrainPretrainArgs, TrainSftArgs, TrainTestArgs,
    },
    hardware, hub_discovery,
};

#[allow(dead_code)]
#[path = "../../train/src/bin/eval_lm.rs"]
mod eval_lm_entry;
#[allow(dead_code)]
#[path = "../../train/src/bin/pretrain.rs"]
mod pretrain_entry;
#[allow(dead_code)]
#[path = "../../train/src/bin/train_grpo.rs"]
mod train_grpo_entry;
#[allow(dead_code)]
#[path = "../../train/src/bin/train_multi_turn.rs"]
mod train_multi_turn_entry;
#[allow(dead_code)]
#[path = "../../train/src/bin/train_sft.rs"]
mod train_sft_entry;

const TRAIN_ENV_COMMANDS: &[&str] = &[
    "train env",
    "train test",
    "train estimate-memory",
    "train pretrain",
    "train sft",
    "train grpo",
    "train multi-turn",
    "train eval",
    "data download",
    "data convert",
];

pub(crate) fn run_train(train: TrainArgs) -> ExitCode {
    match train.command {
        TrainCommand::Env(args) => exit_from_result(run_train_env(args)),
        TrainCommand::Test(args) => run_train_test(args),
        TrainCommand::EstimateMemory(args) => exit_from_result(run_train_estimate_memory(args)),
        TrainCommand::Pretrain(args) => run_pretrain(args),
        TrainCommand::Sft(args) => run_sft(args),
        TrainCommand::Grpo(args) => run_grpo(args),
        TrainCommand::MultiTurn(args) => run_multi_turn(args),
        TrainCommand::Eval(args) => run_eval(args),
    }
}

pub(crate) fn run_data(data: DataArgs) -> ExitCode {
    match data.command {
        DataCommand::Download(args) => run_data_download(args),
        DataCommand::Convert(args) => run_data_convert(args),
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

fn run_train_test(args: TrainTestArgs) -> ExitCode {
    let root_dir = args
        .out_dir
        .clone()
        .unwrap_or_else(|| unique_temp_dir("arle-train-test"));
    let cleanup_path = if args.keep_artifacts || args.out_dir.is_some() {
        None
    } else {
        Some(root_dir.clone())
    };
    let result = train_test_inner(&args, &root_dir);
    if let Some(path) = cleanup_path.as_ref() {
        let _ = fs::remove_dir_all(path);
    }
    match result {
        Ok(report) => {
            if args.json {
                println!("{}", serde_json::to_string_pretty(&report).unwrap());
            } else {
                println!("ARLE train test");
                println!("backend {}", report.backend);
                println!("root {}", report.root_dir);
                if let Some(model_dir) = &report.servable_model_dir {
                    println!("model {}", model_dir);
                } else {
                    println!(
                        "note pass --keep-artifacts or --out-dir to keep the final checkpoint"
                    );
                }
                for step in &report.steps {
                    println!("{} {}", step.name, step.status);
                }
                if let Some(eval_summary) = &report.eval_summary {
                    println!(
                        "eval metrics loss={:.6} ppl={:.6} tokens={}",
                        eval_summary.loss, eval_summary.ppl, eval_summary.tokens
                    );
                }
                println!("wall {:.2}s", report.wall_secs);
                if report.kept_artifacts {
                    println!("artifacts kept {}", report.root_dir);
                }
            }
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("[ARLE train test] error: {err:#}");
            ExitCode::FAILURE
        }
    }
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

fn run_pretrain(args: TrainPretrainArgs) -> ExitCode {
    run_train_command(
        "train pretrain",
        resolve_pretrain_invocation(&args),
        &args.render,
        pretrain_entry::dispatch_from_args,
    )
}

fn run_sft(args: TrainSftArgs) -> ExitCode {
    run_train_command(
        "train sft",
        resolve_sft_invocation(&args),
        &args.render,
        train_sft_entry::dispatch_from_args,
    )
}

fn run_eval(args: TrainEvalArgs) -> ExitCode {
    run_train_command(
        "train eval",
        resolve_eval_invocation(&args),
        &args.render,
        eval_lm_entry::dispatch_from_args,
    )
}

fn run_grpo(args: TrainGrpoArgs) -> ExitCode {
    run_train_command(
        "train grpo",
        Ok(resolve_grpo_invocation(&args)),
        &args.render,
        train_grpo_entry::dispatch_from_args,
    )
}

fn run_multi_turn(args: TrainMultiTurnArgs) -> ExitCode {
    run_train_command(
        "train multi-turn",
        Ok(resolve_multi_turn_invocation(&args)),
        &args.render,
        train_multi_turn_entry::dispatch_from_args,
    )
}

fn run_data_convert(args: DataConvertArgs) -> ExitCode {
    let used_default_output = args.output.is_none();
    let output = args
        .output
        .unwrap_or_else(|| default_chat_output_path(&args.input));
    let invocation = ResolvedInvocation {
        command: "data convert",
        argv: vec![
            "--input".to_string(),
            args.input.display().to_string(),
            "--format".to_string(),
            args.format.as_train_format().to_string(),
            "--output".to_string(),
            output.display().to_string(),
        ],
        backend: None,
        output_dir: None,
        model: None,
        notes: if used_default_output {
            vec!["output omitted; defaulted from input path".to_string()]
        } else {
            Vec::new()
        },
    };

    run_passthrough_invocation(
        invocation,
        &args.render,
        train::commands::convert_dataset::dispatch_from_args,
    )
}

fn run_data_download(args: DataDownloadArgs) -> ExitCode {
    let invocation = ResolvedInvocation {
        command: "data download",
        argv: vec![
            "--repo".to_string(),
            args.repo,
            "--file".to_string(),
            args.file,
        ],
        backend: None,
        output_dir: None,
        model: None,
        notes: Vec::new(),
    };

    run_passthrough_invocation(
        invocation,
        &args.render,
        train::commands::download_dataset::dispatch_from_args,
    )
}

fn run_train_invocation<F>(invocation: ResolvedInvocation, render: &RenderArgs, run: F) -> ExitCode
where
    F: FnOnce(Vec<String>) -> std::result::Result<(), String>,
{
    if let Some(exit) = dry_run_exit(&invocation, render) {
        return exit;
    }
    match run(invocation.argv) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("[ARLE {}] error: {err}", invocation.command);
            ExitCode::FAILURE
        }
    }
}

fn run_train_command<F>(
    command: &'static str,
    resolved: Result<ResolvedInvocation>,
    render: &RenderArgs,
    run: F,
) -> ExitCode
where
    F: FnOnce(Vec<String>) -> std::result::Result<(), String>,
{
    match resolved {
        Ok(invocation) => run_train_invocation(invocation, render, run),
        Err(err) => {
            eprintln!("[ARLE {command}] error: {err:#}");
            ExitCode::FAILURE
        }
    }
}

fn run_passthrough_invocation<F>(
    invocation: ResolvedInvocation,
    render: &RenderArgs,
    run: F,
) -> ExitCode
where
    F: FnOnce(Vec<String>) -> ExitCode,
{
    if let Some(exit) = dry_run_exit(&invocation, render) {
        return exit;
    }
    run(invocation.argv)
}

fn dry_run_exit(invocation: &ResolvedInvocation, render: &RenderArgs) -> Option<ExitCode> {
    render
        .dry_run
        .then(|| print_invocation(invocation, render.json))
}

fn print_invocation(invocation: &ResolvedInvocation, json: bool) -> ExitCode {
    if json {
        println!("{}", serde_json::to_string_pretty(&invocation).unwrap());
    } else {
        println!("command {}", invocation.command);
        if let Some(backend) = &invocation.backend {
            println!("backend {}", backend);
        }
        if let Some(output_dir) = &invocation.output_dir {
            println!("out {}", output_dir);
        }
        if let Some(model) = &invocation.model {
            if let Some(resolved_dir) = &model.resolved_dir {
                println!("model {}", resolved_dir);
            } else {
                println!("model {}", model.source);
            }
            if let Some(config_path) = &model.config_path {
                println!("config {}", config_path);
            }
            if let Some(tokenizer_path) = &model.tokenizer_path {
                println!("tokenizer {}", tokenizer_path);
            }
            if let Some(generation_config_path) = &model.generation_config_path {
                println!("generation_config {}", generation_config_path);
            }
            if let Some(family) = &model.family {
                println!("family {}", family);
            }
        }
        for note in &invocation.notes {
            println!("note {note}");
        }
        println!("argv {}", shell_words(&invocation.argv));
    }
    ExitCode::SUCCESS
}

fn resolve_pretrain_invocation(args: &TrainPretrainArgs) -> Result<ResolvedInvocation> {
    let tokenizer_path = resolve_local_tokenizer_path(&args.tokenizer)?;
    let out_dir = args
        .out
        .clone()
        .unwrap_or_else(|| default_job_output("pretrain", &args.corpus));
    let backend = args
        .backend
        .as_train_backend()
        .unwrap_or_else(default_train_backend);

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

    let mut argv = vec![
        "--corpus".to_string(),
        args.corpus.display().to_string(),
        "--tokenizer".to_string(),
        tokenizer_path.display().to_string(),
        "--out".to_string(),
        out_dir.display().to_string(),
        "--backend".to_string(),
        backend.to_string(),
    ];
    push_opt_family(&mut argv, args.model_family);
    push_opt_value(&mut argv, "--steps", args.steps);
    push_opt_value(&mut argv, "--batch", args.batch);
    push_opt_value(&mut argv, "--seq", args.seq);
    push_opt_value(&mut argv, "--lr", args.lr);
    push_opt_value(&mut argv, "--grad-accum-steps", args.grad_accum_steps);
    push_opt_value(&mut argv, "--log-every", args.log_every);
    push_opt_value(&mut argv, "--save-every", args.save_every);
    push_opt_value(&mut argv, "--eval-every", args.eval_every);
    push_opt_value(&mut argv, "--eval-windows", args.eval_windows);
    push_opt_value(&mut argv, "--eval-frac", args.eval_frac);
    push_opt_path(&mut argv, "--resume-from", args.resume_from.as_deref());
    push_opt_value(&mut argv, "--seed", args.seed);
    push_grad_clip_flags(&mut argv, args.grad_clip, args.no_grad_clip);
    push_opt_save_dtype(&mut argv, args.save_dtype);
    push_opt_value(&mut argv, "--vocab-size", args.vocab_size);
    let explicit_shape = args.preset.is_some() || args.has_shape_overrides();
    if explicit_shape {
        shape.push_shape_flags(&mut argv);
    }
    push_opt_value(&mut argv, "--rms-eps", args.rms_eps);
    push_opt_value(&mut argv, "--rope-theta", args.rope_theta);
    if args.no_tie_embed {
        argv.push("--no-tie-embed".to_string());
    }
    push_opt_token(&mut argv, "--bos-token", args.bos_token.as_deref());
    push_opt_token(&mut argv, "--eos-token", args.eos_token.as_deref());
    push_opt_value(&mut argv, "--bos-token-id", args.bos_token_id);
    push_opt_value(&mut argv, "--eos-token-id", args.eos_token_id);
    push_opt_path(&mut argv, "--metrics-jsonl", args.metrics_jsonl.as_deref());
    push_opt_value(&mut argv, "--serve", args.serve);
    argv.extend(args.extra.extra_args.iter().cloned());

    let mut notes = Vec::new();
    if args.out.is_none() {
        notes.push("out omitted; defaulted under runs/pretrain".to_string());
    }
    if args.preset.is_some() {
        notes.push("applied scratch preset before explicit shape overrides".to_string());
    }
    if !explicit_shape {
        notes.push("scratch shape omitted; underlying pretrain defaults apply".to_string());
    }
    notes.push(format!("resolved tokenizer {}", tokenizer_path.display()));

    Ok(ResolvedInvocation {
        command: "train pretrain",
        argv,
        backend: Some(backend.to_string()),
        output_dir: Some(out_dir.display().to_string()),
        model: None,
        notes,
    })
}

fn resolve_sft_invocation(args: &TrainSftArgs) -> Result<ResolvedInvocation> {
    let model = resolve_model_command(&args.model, args.backend, args.render.dry_run)?;
    let out_dir = args
        .out
        .clone()
        .unwrap_or_else(|| default_job_output("sft", &args.model));

    let mut argv = vec![
        "--model".to_string(),
        model.model_arg.clone(),
        "--data".to_string(),
        args.data.display().to_string(),
        "--out".to_string(),
        out_dir.display().to_string(),
        "--backend".to_string(),
        model.backend.clone(),
    ];
    push_opt_family(&mut argv, args.model_family);
    push_opt_value(&mut argv, "--steps", args.steps);
    push_opt_value(&mut argv, "--batch", args.batch);
    push_opt_value(&mut argv, "--lr", args.lr);
    push_opt_value(&mut argv, "--seq-len", args.seq_len);
    push_opt_value(&mut argv, "--save-every", args.save_every);
    push_opt_value(&mut argv, "--log-every", args.log_every);
    push_opt_value(&mut argv, "--seed", args.seed);
    push_opt_save_dtype(&mut argv, args.save_dtype);
    push_opt_string(&mut argv, "--lr-schedule", args.lr_schedule.as_deref());
    push_opt_value(&mut argv, "--warmup-steps", args.warmup_steps);
    push_opt_value(&mut argv, "--min-lr", args.min_lr);
    push_opt_value(&mut argv, "--grad-accum-steps", args.grad_accum_steps);
    push_opt_path(&mut argv, "--metrics-jsonl", args.metrics_jsonl.as_deref());
    push_opt_path(&mut argv, "--resume-from", args.resume_from.as_deref());
    push_opt_value(&mut argv, "--lora-rank", args.lora_rank);
    push_opt_value(&mut argv, "--lora-alpha", args.lora_alpha);
    push_opt_value(&mut argv, "--serve", args.serve);
    argv.extend(args.extra.extra_args.iter().cloned());

    let mut notes = model.inspection.notes.clone();
    if args.out.is_none() {
        notes.push("out omitted; defaulted under runs/sft".to_string());
    }
    notes.push("config/tokenizer/generation config are auto-loaded from --model".to_string());

    Ok(ResolvedInvocation {
        command: "train sft",
        argv,
        backend: Some(model.backend),
        output_dir: Some(out_dir.display().to_string()),
        model: Some(model.inspection),
        notes,
    })
}

fn resolve_eval_invocation(args: &TrainEvalArgs) -> Result<ResolvedInvocation> {
    let model = resolve_model_command(&args.model, args.backend, args.render.dry_run)?;

    let mut argv = vec![
        "--model-path".to_string(),
        model.model_arg.clone(),
        "--data".to_string(),
        args.data.display().to_string(),
        "--backend".to_string(),
        model.backend.clone(),
    ];
    push_opt_family(&mut argv, args.model_family);
    if let Some(tokenizer) = args.tokenizer.as_deref() {
        let tokenizer_path = resolve_local_tokenizer_path(tokenizer)?;
        argv.push("--tokenizer".to_string());
        argv.push(tokenizer_path.display().to_string());
    }
    push_opt_value(&mut argv, "--seq-len", args.seq_len);
    push_opt_path(&mut argv, "--metrics-jsonl", args.metrics_jsonl.as_deref());
    argv.extend(args.extra.extra_args.iter().cloned());

    let mut notes = model.inspection.notes.clone();
    notes.push(
        "config/tokenizer are auto-loaded from --model unless --tokenizer overrides it".to_string(),
    );

    Ok(ResolvedInvocation {
        command: "train eval",
        argv,
        backend: Some(model.backend),
        output_dir: None,
        model: Some(model.inspection),
        notes,
    })
}

fn resolve_grpo_invocation(args: &TrainGrpoArgs) -> ResolvedInvocation {
    let backend = args
        .backend
        .as_train_backend()
        .unwrap_or_else(default_train_backend);
    let mut argv = vec!["--backend".to_string(), backend.to_string()];
    push_opt_family(&mut argv, args.model_family);
    push_opt_value(&mut argv, "--sft-steps", args.sft_steps);
    push_opt_value(&mut argv, "--grpo-iters", args.grpo_iters);
    push_opt_value(&mut argv, "--save-every", args.save_every);
    push_opt_value(&mut argv, "--batch-prompts", args.batch_prompts);
    push_opt_value(&mut argv, "--group-size", args.group_size);
    push_opt_value(&mut argv, "--seq", args.seq);
    push_opt_value(&mut argv, "--lr", args.lr);
    push_opt_value(&mut argv, "--kl-coef", args.kl_coef);
    push_opt_value(&mut argv, "--temperature", args.temperature);
    push_opt_value(&mut argv, "--seed", args.seed);
    push_opt_value(&mut argv, "--lora-rank", args.lora_rank);
    push_opt_value(&mut argv, "--lora-alpha", args.lora_alpha);
    push_grad_clip_flags(&mut argv, args.grad_clip, args.no_grad_clip);
    push_opt_path(&mut argv, "--metrics-jsonl", args.metrics_jsonl.as_deref());
    push_opt_path(&mut argv, "--save-path", args.save_path.as_deref());
    push_opt_path(&mut argv, "--resume-from", args.resume_from.as_deref());
    push_opt_value(&mut argv, "--serve", args.serve);
    push_opt_value(&mut argv, "--linear-attn-every", args.linear_attn_every);
    argv.extend(args.extra.extra_args.iter().cloned());

    ResolvedInvocation {
        command: "train grpo",
        argv,
        backend: Some(backend.to_string()),
        output_dir: args
            .save_path
            .as_ref()
            .map(|path| path.display().to_string()),
        model: None,
        notes: Vec::new(),
    }
}

fn resolve_multi_turn_invocation(args: &TrainMultiTurnArgs) -> ResolvedInvocation {
    let backend = args
        .backend
        .as_train_backend()
        .unwrap_or_else(default_train_backend);
    let mut argv = vec!["--backend".to_string(), backend.to_string()];
    push_opt_value(&mut argv, "--iters", args.iters);
    push_opt_value(&mut argv, "--group-size", args.group_size);
    push_opt_value(&mut argv, "--agent-tokens", args.agent_tokens);
    push_opt_value(&mut argv, "--obs-tokens", args.obs_tokens);
    push_opt_value(&mut argv, "--turns", args.turns);
    push_opt_value(&mut argv, "--prompt-len", args.prompt_len);
    push_opt_value(&mut argv, "--lr", args.lr);
    push_opt_value(&mut argv, "--kl-coef", args.kl_coef);
    push_opt_value(&mut argv, "--clip-eps", args.clip_eps);
    push_opt_value(&mut argv, "--temperature", args.temperature);
    push_opt_value(&mut argv, "--gamma", args.gamma);
    push_opt_value(&mut argv, "--lora-rank", args.lora_rank);
    push_opt_value(&mut argv, "--lora-alpha", args.lora_alpha);
    push_opt_value(&mut argv, "--seed", args.seed);
    push_opt_value(&mut argv, "--vocab", args.vocab);
    push_opt_value(&mut argv, "--target-range", args.target_range);
    push_opt_value(&mut argv, "--d-model", args.d_model);
    push_opt_value(&mut argv, "--n-layers", args.n_layers);
    push_opt_value(&mut argv, "--n-heads", args.n_heads);
    push_opt_value(&mut argv, "--d-head", args.d_head);
    push_opt_value(&mut argv, "--d-ff", args.d_ff);
    push_opt_value(&mut argv, "--linear-attn-every", args.linear_attn_every);
    push_opt_value(&mut argv, "--eval-every", args.eval_every);
    push_opt_value(&mut argv, "--eval-prompts", args.eval_prompts);
    push_opt_value(&mut argv, "--eval-temperature", args.eval_temperature);
    push_opt_path(&mut argv, "--save-path", args.save_path.as_deref());
    push_opt_path(&mut argv, "--resume-from", args.resume_from.as_deref());
    push_opt_value(&mut argv, "--serve", args.serve);
    push_grad_clip_flags(&mut argv, args.grad_clip, args.no_grad_clip);
    push_opt_path(&mut argv, "--metrics-jsonl", args.metrics_jsonl.as_deref());
    if let Some(objective) = args.objective {
        argv.push("--objective".to_string());
        argv.push(objective.as_train_objective().to_string());
    }
    argv.extend(args.extra.extra_args.iter().cloned());

    ResolvedInvocation {
        command: "train multi-turn",
        argv,
        backend: Some(backend.to_string()),
        output_dir: args
            .save_path
            .as_ref()
            .map(|path| path.display().to_string()),
        model: None,
        notes: Vec::new(),
    }
}

fn train_test_inner(args: &TrainTestArgs, root_dir: &Path) -> Result<TrainTestReport> {
    let backend = args
        .backend
        .as_train_backend()
        .unwrap_or_else(default_train_backend)
        .to_string();
    fs::create_dir_all(root_dir)?;

    let executable = arle_executable()?;
    let corpus = root_dir.join("corpus.txt");
    let tokenizer = root_dir.join("tokenizer.json");
    let raw = root_dir.join("raw_dolly.jsonl");
    let chat = root_dir.join("train.chat.jsonl");
    let pretrain_out = root_dir.join("pretrain");
    let sft_out = root_dir.join("sft");

    fs::write(&corpus, "hello world\n2+2 equals 4\nblue is a color\n")?;
    write_chatml_wordlevel_tokenizer(
        &tokenizer,
        [
            "hello",
            "world",
            "2+2",
            "equals",
            "4",
            "blue",
            "is",
            "a",
            "color",
            "Say",
            "What",
            "Name",
            "one",
            "word.",
            "?",
            "Hello!",
            "Blue.",
            "2+2 equals 4.",
        ],
    )?;
    fs::write(
        &raw,
        concat!(
            "{\"instruction\":\"Say hello in one word.\",\"response\":\"Hello!\"}\n",
            "{\"instruction\":\"What is 2+2?\",\"response\":\"2+2 equals 4.\"}\n"
        ),
    )?;

    let started = Instant::now();
    run_arle_step(
        &executable,
        "convert",
        &[
            "data".to_string(),
            "convert".to_string(),
            "--input".to_string(),
            raw.display().to_string(),
            "--format".to_string(),
            DatasetFormatArg::Dolly.as_train_format().to_string(),
            "--output".to_string(),
            chat.display().to_string(),
        ],
    )?;

    run_arle_step(
        &executable,
        "pretrain",
        &[
            "train".to_string(),
            "pretrain".to_string(),
            "--corpus".to_string(),
            corpus.display().to_string(),
            "--tokenizer".to_string(),
            tokenizer.display().to_string(),
            "--out".to_string(),
            pretrain_out.display().to_string(),
            "--steps".to_string(),
            "2".to_string(),
            "--seq".to_string(),
            "8".to_string(),
            "--hidden".to_string(),
            "32".to_string(),
            "--layers".to_string(),
            "2".to_string(),
            "--heads".to_string(),
            "2".to_string(),
            "--kv-heads".to_string(),
            "2".to_string(),
            "--head-dim".to_string(),
            "16".to_string(),
            "--intermediate".to_string(),
            "64".to_string(),
            "--max-pos".to_string(),
            "16".to_string(),
            "--backend".to_string(),
            backend.clone(),
            "--save-every".to_string(),
            "2".to_string(),
            "--log-every".to_string(),
            "1".to_string(),
        ],
    )?;

    run_arle_step(
        &executable,
        "sft",
        &[
            "train".to_string(),
            "sft".to_string(),
            "--model".to_string(),
            pretrain_out.join("latest").display().to_string(),
            "--data".to_string(),
            chat.display().to_string(),
            "--out".to_string(),
            sft_out.display().to_string(),
            "--steps".to_string(),
            "1".to_string(),
            "--seq-len".to_string(),
            "16".to_string(),
            "--backend".to_string(),
            backend.clone(),
            "--save-every".to_string(),
            "1".to_string(),
            "--log-every".to_string(),
            "1".to_string(),
            "--lora-rank".to_string(),
            "4".to_string(),
            "--lora-alpha".to_string(),
            "8".to_string(),
        ],
    )?;

    let eval = run_arle_step(
        &executable,
        "eval",
        &[
            "train".to_string(),
            "eval".to_string(),
            "--model".to_string(),
            sft_out.join("latest").display().to_string(),
            "--data".to_string(),
            chat.display().to_string(),
            "--seq-len".to_string(),
            "16".to_string(),
            "--backend".to_string(),
            backend.clone(),
        ],
    )?;
    let eval_summary: EvalSummary =
        serde_json::from_str(eval.stdout.trim()).with_context(|| {
            format!(
                "parsing eval smoke JSON from stdout:\n{}",
                printable_output(&eval.stdout)
            )
        })?;
    let kept_artifacts = args.keep_artifacts || args.out_dir.is_some();
    let servable_model_dir = kept_artifacts.then(|| sft_out.join("latest").display().to_string());

    let report = TrainTestReport {
        backend,
        root_dir: root_dir.display().to_string(),
        servable_model_dir,
        wall_secs: started.elapsed().as_secs_f64(),
        steps: ok_train_test_steps(),
        eval_summary: Some(eval_summary),
        kept_artifacts,
    };
    Ok(report)
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
    let (param_count, hidden_size) = match args.model_family.unwrap_or(ModelFamilyArg::Qwen35) {
        ModelFamilyArg::Qwen3 => (
            qwen3_param_count(&shape.qwen3_config(vocab_size)),
            shape.hidden,
        ),
        ModelFamilyArg::Auto | ModelFamilyArg::Qwen35 => (
            qwen35_param_count(&shape.qwen35_config(vocab_size)),
            shape.hidden,
        ),
    };
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
    let mut notes = Vec::new();
    if !allow_download && resolved_dir.is_none() {
        notes.push(
            "model source is not local/cached; dry-run skipped remote resolution".to_string(),
        );
    }
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

fn resolve_model_command(
    source: &Path,
    backend: BackendArg,
    dry_run: bool,
) -> Result<ResolvedModelCommand> {
    let inspection = inspect_model_source(source, !dry_run)?;
    Ok(ResolvedModelCommand {
        model_arg: inspection
            .resolved_dir
            .clone()
            .unwrap_or_else(|| source.display().to_string()),
        backend: backend
            .as_train_backend()
            .unwrap_or_else(default_train_backend)
            .to_string(),
        inspection,
    })
}

fn inspect_resolved_model_dir(model_dir: &Path) -> Result<ModelDirSummary> {
    let config_path = model_dir.join("config.json");
    let family = match resolve_model_family(&config_path, ModelFamily::Auto)? {
        ModelFamily::Qwen3 => "qwen3",
        ModelFamily::Qwen35 => "qwen35",
        ModelFamily::Auto => unreachable!("auto must resolve to a concrete family"),
    };
    match family {
        "qwen3" => {
            let cfg = Qwen3Config::from_json_file(&config_path)?;
            Ok(ModelDirSummary {
                family: "qwen3".to_string(),
                config: ResolvedModelConfig::Qwen3(cfg.clone()),
                config_path: config_path.display().to_string(),
                tokenizer_path: existing_display_path(model_dir.join("tokenizer.json")),
                generation_config_path: existing_display_path(
                    model_dir.join("generation_config.json"),
                ),
                vocab_size: cfg.vocab_size,
                hidden_size: cfg.hidden_size,
                param_count: qwen3_param_count(&cfg),
            })
        }
        "qwen35" => {
            let cfg = Qwen35Config::from_json_file(&config_path)?;
            Ok(ModelDirSummary {
                family: "qwen35".to_string(),
                config: ResolvedModelConfig::Qwen35(cfg.clone()),
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
    infer::hf_hub::resolve_model_path(&source_text)
        .with_context(|| format!("resolving model source {source_text}"))
}

fn resolve_model_dir_local_only(source: &Path) -> Option<PathBuf> {
    let source_text = source.display().to_string();
    infer::hf_hub::resolve_local_model_path(&source_text)
}

fn resolve_local_tokenizer_path(source: &Path) -> Result<PathBuf> {
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
    if let Some(model_dir) = infer::hf_hub::resolve_local_model_path(&source_text) {
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

fn qwen3_param_count(cfg: &Qwen3Config) -> u64 {
    let embed = mul_u64(cfg.vocab_size, cfg.hidden_size);
    let lm_head = if cfg.tie_word_embeddings { 0 } else { embed };
    let attn_q = mul_u64(cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim);
    let attn_k = mul_u64(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim);
    let attn_v = attn_k;
    let attn_o = mul_u64(cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size);
    let attn_norms = mul_u64(2, cfg.head_dim);
    let mlp = mul_u64(cfg.hidden_size, cfg.intermediate_size) * 2
        + mul_u64(cfg.intermediate_size, cfg.hidden_size);
    let layer_norms = mul_u64(2, cfg.hidden_size);
    let per_layer = attn_q + attn_k + attn_v + attn_o + attn_norms + mlp + layer_norms;
    embed
        + lm_head
        + (cfg.num_hidden_layers as u64).saturating_mul(per_layer)
        + cfg.hidden_size as u64
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

fn lora_param_count(config: &ResolvedModelConfig, rank: usize) -> u64 {
    match config {
        ResolvedModelConfig::Qwen3(cfg) => {
            let per_layer = lora_linear(
                cfg.hidden_size,
                cfg.num_attention_heads * cfg.head_dim,
                rank,
            ) + lora_linear(
                cfg.hidden_size,
                cfg.num_key_value_heads * cfg.head_dim,
                rank,
            ) * 2
                + lora_linear(
                    cfg.num_attention_heads * cfg.head_dim,
                    cfg.hidden_size,
                    rank,
                )
                + lora_linear(cfg.hidden_size, cfg.intermediate_size, rank) * 2
                + lora_linear(cfg.intermediate_size, cfg.hidden_size, rank);
            (cfg.num_hidden_layers as u64).saturating_mul(per_layer)
        }
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

fn push_opt_value<T: ToString>(argv: &mut Vec<String>, flag: &str, value: Option<T>) {
    if let Some(value) = value {
        argv.push(flag.to_string());
        argv.push(value.to_string());
    }
}

fn push_opt_string(argv: &mut Vec<String>, flag: &str, value: Option<&str>) {
    if let Some(value) = value {
        argv.push(flag.to_string());
        argv.push(value.to_string());
    }
}

fn push_opt_token(argv: &mut Vec<String>, flag: &str, value: Option<&str>) {
    push_opt_string(argv, flag, value);
}

fn push_opt_path(argv: &mut Vec<String>, flag: &str, value: Option<&Path>) {
    if let Some(value) = value {
        argv.push(flag.to_string());
        argv.push(value.display().to_string());
    }
}

fn push_opt_family(argv: &mut Vec<String>, family: Option<ModelFamilyArg>) {
    if let Some(family) = family {
        argv.push("--model-family".to_string());
        argv.push(family.as_train_family().to_string());
    }
}

fn push_opt_save_dtype(argv: &mut Vec<String>, dtype: Option<SaveDtypeArg>) {
    if let Some(dtype) = dtype {
        argv.push("--save-dtype".to_string());
        argv.push(dtype.as_train_dtype().to_string());
    }
}

fn push_grad_clip_flags(argv: &mut Vec<String>, grad_clip: Option<f32>, no_grad_clip: bool) {
    if no_grad_clip {
        argv.push("--no-grad-clip".to_string());
    } else {
        push_opt_value(argv, "--grad-clip", grad_clip);
    }
}

fn default_job_output(job: &str, seed: &Path) -> PathBuf {
    PathBuf::from("runs")
        .join(job)
        .join(sanitize_name(path_seed_name(seed)))
}

fn default_chat_output_path(input: &Path) -> PathBuf {
    let stem = input
        .file_stem()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .unwrap_or("dataset");
    input.with_file_name(format!("{stem}.chat.jsonl"))
}

fn path_seed_name(seed: &Path) -> String {
    seed.file_name()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .unwrap_or("run")
        .to_string()
}

fn sanitize_name(value: String) -> String {
    let mapped = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '-'
            }
        })
        .collect::<String>();
    let sanitized = mapped.trim_matches('-').to_string();
    if sanitized.is_empty() {
        "run".to_string()
    } else {
        sanitized
    }
}

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    std::env::temp_dir().join(format!("{prefix}-{}-{millis}", std::process::id()))
}

fn shell_words(args: &[String]) -> String {
    args.iter()
        .map(|arg| {
            if arg.contains(' ') {
                format!("{arg:?}")
            } else {
                arg.clone()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn existing_display_path(path: PathBuf) -> Option<String> {
    path.is_file().then(|| path.display().to_string())
}

fn arle_executable() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("CARGO_BIN_EXE_arle").map(PathBuf::from) {
        return Ok(path);
    }
    std::env::current_exe().context("resolve current ARLE executable")
}

fn run_arle_step(
    executable: &Path,
    step: &'static str,
    args: &[String],
) -> Result<CapturedStepOutput> {
    let output = Command::new(executable)
        .args(args)
        .output()
        .with_context(|| format!("spawn smoke step {step}"))?;
    ensure_successful_step(executable, step, args, output)
}

fn ensure_successful_step(
    executable: &Path,
    step: &'static str,
    args: &[String],
    output: Output,
) -> Result<CapturedStepOutput> {
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    if output.status.success() {
        return Ok(CapturedStepOutput { stdout });
    }
    let code = output
        .status
        .code()
        .map(|value| value.to_string())
        .unwrap_or_else(|| "signal".to_string());
    bail!(
        "{step} smoke failed (exit {code})\ncommand: {} {}\nstdout:\n{}\nstderr:\n{}",
        executable.display(),
        shell_words(args),
        printable_output(&stdout),
        printable_output(&stderr),
    );
}

fn printable_output(output: &str) -> &str {
    let trimmed = output.trim();
    if trimmed.is_empty() {
        "<empty>"
    } else {
        trimmed
    }
}

fn ok_train_test_steps() -> Vec<TrainTestStep> {
    vec![
        TrainTestStep {
            name: "convert",
            status: "ok",
        },
        TrainTestStep {
            name: "pretrain",
            status: "ok",
        },
        TrainTestStep {
            name: "sft",
            status: "ok",
        },
        TrainTestStep {
            name: "eval",
            status: "ok",
        },
    ]
}

impl TrainPretrainArgs {
    fn has_shape_overrides(&self) -> bool {
        self.hidden.is_some()
            || self.layers.is_some()
            || self.heads.is_some()
            || self.kv_heads.is_some()
            || self.head_dim.is_some()
            || self.intermediate.is_some()
            || self.max_pos.is_some()
            || self.linear_attn_every.is_some()
    }
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
        } => format!("{chip} ({unified_memory_gb:.1} GB unified)"),
        hardware::GpuInfo::None => "none".to_string(),
    }
}

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

    fn push_shape_flags(&self, argv: &mut Vec<String>) {
        argv.push("--hidden".to_string());
        argv.push(self.hidden.to_string());
        argv.push("--layers".to_string());
        argv.push(self.layers.to_string());
        argv.push("--heads".to_string());
        argv.push(self.heads.to_string());
        argv.push("--kv-heads".to_string());
        argv.push(self.kv_heads.to_string());
        argv.push("--head-dim".to_string());
        argv.push(self.head_dim.to_string());
        argv.push("--intermediate".to_string());
        argv.push(self.intermediate.to_string());
        argv.push("--max-pos".to_string());
        argv.push(self.max_pos.to_string());
        if self.linear_attn_every > 0 {
            argv.push("--linear-attn-every".to_string());
            argv.push(self.linear_attn_every.to_string());
        }
    }

    fn qwen3_config(&self, vocab_size: usize) -> Qwen3Config {
        Qwen3Config {
            hidden_size: self.hidden,
            intermediate_size: self.intermediate,
            num_hidden_layers: self.layers,
            num_attention_heads: self.heads,
            num_key_value_heads: self.kv_heads,
            head_dim: self.head_dim,
            vocab_size,
            rms_norm_eps: 1.0e-6,
            rope_theta: 1_000_000.0,
            tie_word_embeddings: true,
            max_position_embeddings: self.max_pos,
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
struct ResolvedInvocation {
    command: &'static str,
    argv: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    backend: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_dir: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<ModelInspection>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    notes: Vec<String>,
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
struct ResolvedModelCommand {
    inspection: ModelInspection,
    model_arg: String,
    backend: String,
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
    Qwen3(Qwen3Config),
    Qwen35(Qwen35Config),
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

#[derive(Debug, Serialize)]
struct TrainTestReport {
    backend: String,
    root_dir: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    servable_model_dir: Option<String>,
    wall_secs: f64,
    steps: Vec<TrainTestStep>,
    #[serde(skip_serializing_if = "Option::is_none")]
    eval_summary: Option<EvalSummary>,
    kept_artifacts: bool,
}

#[derive(Debug, Clone, Serialize)]
struct TrainTestStep {
    name: &'static str,
    status: &'static str,
}

#[derive(Debug, Clone, Serialize, serde::Deserialize)]
struct EvalSummary {
    loss: f64,
    ppl: f64,
    tokens: usize,
}

#[derive(Debug)]
struct CapturedStepOutput {
    stdout: String,
}

impl SaveDtypeArg {
    fn bytes_per_param(self) -> u64 {
        match self {
            SaveDtypeArg::F32 => 4,
            SaveDtypeArg::Bf16 => 2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        EvalSummary, PretrainPresetArg, ResolvedInvocation, ScratchShape, TrainTestReport,
        TrainTestStep, default_chat_output_path, resolve_pretrain_invocation,
        run_passthrough_invocation, run_train_command, sanitize_name,
    };
    use crate::args::{BackendArg, ExtraArgs, RenderArgs, TrainPretrainArgs};
    use std::{cell::Cell, path::Path, process::ExitCode};

    #[test]
    fn default_chat_output_uses_chat_suffix() {
        let output = default_chat_output_path(Path::new("train.jsonl"));
        assert_eq!(output, Path::new("train.chat.jsonl"));
    }

    #[test]
    fn sanitize_name_collapses_non_ascii_path_chars() {
        assert_eq!(
            sanitize_name("Qwen/Qwen3-0.6B".to_string()),
            "Qwen-Qwen3-0-6B"
        );
    }

    #[test]
    fn sanitize_name_falls_back_when_everything_is_filtered() {
        assert_eq!(sanitize_name("！！！".to_string()), "run");
    }

    #[test]
    fn small_30m_preset_applies_expected_shape() {
        let mut shape = ScratchShape::default();
        shape.apply_preset(PretrainPresetArg::Small30m);
        assert_eq!(shape.hidden, 192);
        assert_eq!(shape.layers, 2);
        assert_eq!(shape.heads, 6);
        assert_eq!(shape.kv_heads, 3);
        assert_eq!(shape.head_dim, 32);
    }

    #[test]
    fn default_pretrain_invocation_preserves_binary_shape_defaults() {
        let temp = tempfile::tempdir().expect("tempdir");
        let tokenizer = temp.path().join("tokenizer.json");
        std::fs::write(&tokenizer, "{}").expect("write tokenizer");
        let invocation = resolve_pretrain_invocation(&base_pretrain_args(&tokenizer))
            .expect("resolve pretrain args");
        assert!(!invocation.argv.iter().any(|arg| arg == "--hidden"));
        assert!(!invocation.argv.iter().any(|arg| arg == "--layers"));
        assert!(
            invocation
                .notes
                .iter()
                .any(|note| note.contains("underlying pretrain defaults apply"))
        );
    }

    #[test]
    fn train_test_report_serializes_public_fields_only() {
        let report = TrainTestReport {
            backend: "metal".to_string(),
            root_dir: "/tmp/arle-train-test".to_string(),
            servable_model_dir: Some("/tmp/arle-train-test/sft/latest".to_string()),
            wall_secs: 1.25,
            steps: vec![
                TrainTestStep {
                    name: "convert",
                    status: "ok",
                },
                TrainTestStep {
                    name: "eval",
                    status: "ok",
                },
            ],
            eval_summary: Some(EvalSummary {
                loss: 1.0,
                ppl: 2.0,
                tokens: 3,
            }),
            kept_artifacts: false,
        };
        let value = serde_json::to_value(report).expect("serialize train test report");
        assert_eq!(value["backend"], "metal");
        assert_eq!(
            value["servable_model_dir"],
            "/tmp/arle-train-test/sft/latest"
        );
        assert_eq!(value["steps"][0]["name"], "convert");
        assert_eq!(value["steps"][1]["status"], "ok");
        assert!(value.get("json").is_none());
    }

    #[test]
    fn train_test_report_omits_model_dir_when_artifacts_are_deleted() {
        let report = TrainTestReport {
            backend: "cpu".to_string(),
            root_dir: "/tmp/arle-train-test".to_string(),
            servable_model_dir: None,
            wall_secs: 1.25,
            steps: vec![TrainTestStep {
                name: "convert",
                status: "ok",
            }],
            eval_summary: None,
            kept_artifacts: false,
        };
        let value = serde_json::to_value(report).expect("serialize train test report");
        assert!(value.get("servable_model_dir").is_none());
    }

    #[test]
    fn train_command_runs_dispatch_for_non_dry_run() {
        let called = Cell::new(false);
        let exit = run_train_command(
            "train eval",
            Ok(test_invocation("train eval", ["--seq-len", "32"])),
            &RenderArgs {
                dry_run: false,
                json: false,
            },
            |argv| {
                called.set(true);
                assert_eq!(argv, vec!["--seq-len".to_string(), "32".to_string()]);
                Ok(())
            },
        );
        assert!(called.get());
        assert_eq!(exit, ExitCode::SUCCESS);
    }

    #[test]
    fn passthrough_invocation_preserves_child_exit_code() {
        let called = Cell::new(false);
        let exit = run_passthrough_invocation(
            test_invocation("data convert", ["--input", "train.jsonl"]),
            &RenderArgs {
                dry_run: false,
                json: false,
            },
            |argv| {
                called.set(true);
                assert_eq!(argv, vec!["--input".to_string(), "train.jsonl".to_string()]);
                ExitCode::from(2)
            },
        );
        assert!(called.get());
        assert_eq!(exit, ExitCode::from(2));
    }

    fn base_pretrain_args(tokenizer: &Path) -> TrainPretrainArgs {
        TrainPretrainArgs {
            corpus: "corpus.txt".into(),
            tokenizer: tokenizer.into(),
            out: None,
            preset: None,
            model_family: None,
            steps: None,
            batch: None,
            seq: None,
            lr: None,
            grad_accum_steps: None,
            log_every: None,
            save_every: None,
            eval_every: None,
            eval_windows: None,
            eval_frac: None,
            resume_from: None,
            seed: None,
            grad_clip: None,
            no_grad_clip: false,
            backend: BackendArg::Auto,
            save_dtype: None,
            vocab_size: None,
            hidden: None,
            layers: None,
            heads: None,
            kv_heads: None,
            head_dim: None,
            intermediate: None,
            max_pos: None,
            rms_eps: None,
            rope_theta: None,
            no_tie_embed: false,
            linear_attn_every: None,
            bos_token: None,
            eos_token: None,
            bos_token_id: None,
            eos_token_id: None,
            metrics_jsonl: None,
            serve: None,
            render: RenderArgs {
                dry_run: false,
                json: false,
            },
            extra: ExtraArgs {
                extra_args: Vec::new(),
            },
        }
    }

    fn test_invocation<const N: usize>(
        command: &'static str,
        argv: [&str; N],
    ) -> ResolvedInvocation {
        ResolvedInvocation {
            command,
            argv: argv.into_iter().map(ToString::to_string).collect(),
            backend: None,
            output_dir: None,
            model: None,
            notes: Vec::new(),
        }
    }
}
