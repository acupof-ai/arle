use anyhow::{Context, Result, anyhow, bail};
use indicatif::ProgressBar;
use qwen35_spec::{LayerType, Qwen35Config};

use super::{
    nll_eval::heldout_nll,
    opd_checkpoint::maybe_save_full_student_checkpoint,
    opd_engine::apply_opd_rollout_engine,
    opd_prompts::{load_opd_prompt_source, parse_prompt_ids},
    opd_runtime::{
        OpdLrSchedule, OpdStepMetric, PromptSampler, apply_tape_dtype, build_opd_store,
        current_grad_norm, kl_direction_arg, kl_mask_arg, opd_logits_window_size_arg,
        opd_progress_style, opd_sft_anchor_arg, opd_step_profile_enabled, opd_summary,
        parse_lora_target_set, print_opd_step_profile, reject_unimplemented_gkd_objectives,
        rollout_sampling_params, trainable_param_ids, validate_train_opd_gkd_args,
    },
    opd_teacher::OpdCliTeacher,
};
#[cfg(feature = "cuda")]
use super::{
    opd_engine::{load_opd_infer_student, maybe_preoffload_infer_student_before_teacher},
    opd_teacher::{load_opd_infer_teacher, maybe_preoffload_infer_teacher_before_steps},
};
use crate::args::{OpdSftAnchorArg, OpdTeacherRuntimeArg, TrainOpdArgs, TrainSelfOpdArgs};

/// Diagnostic (ARLE_OPD_STEP_TRACE): 128-token probe forward through the
/// autograd student, bracketing engine load/offload phases to localize when
/// the weights stop producing finite hidden states.
#[cfg(feature = "cuda")]
fn probe_student_forward(
    student: &train::qwen35::Qwen35Model,
    store: &mut autograd::TensorStore,
    label: &str,
) {
    if std::env::var("ARLE_OPD_STEP_TRACE").is_err() {
        return;
    }
    let keep = train::causal_lm::live_tensor_ids(store);
    let mut tape = autograd::Tape::new();
    tape.set_enabled(false);
    let ids: Vec<u32> = (1..=128).collect();
    let pos: Vec<u32> = (0..128).collect();
    match student.forward_hidden_states(
        store,
        &mut tape,
        &ids,
        &pos,
        train::context_parallel::CpContext::single(),
    ) {
        Ok(hidden) => {
            let sq = store.get(hidden).and_then(|t| {
                t.device_handle
                    .as_ref()
                    .and_then(|h| store.backend().sum_squares(h, &t.shape).ok())
            });
            eprintln!("[probe-forward] {label} hidden_sum_sq={sq:?}");
        }
        Err(err) => eprintln!("[probe-forward] {label} error={err}"),
    }
    store.retain_ids(&keep);
}

/// Startup-fixed VRAM grants for the OPD step's co-resident engines, from ONE
/// probe taken before any engine load. Each grant rides
/// `EngineLoadConfig::memory_budget_bytes` and caps that engine's KV/slot
/// budget for the whole run, so no later engine decision reads instantaneous
/// free VRAM (which co-resident offload/reload churns).
#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
struct OpdVramPlan {
    student_engine_bytes: Option<usize>,
    teacher_engine_bytes: Option<usize>,
}

#[cfg(feature = "cuda")]
impl OpdVramPlan {
    fn probe(backend: &std::sync::Arc<dyn autograd::Backend>, rollout_mem_fraction: f64) -> Self {
        let Some((free, total)) = backend.device_mem_info() else {
            // No CUDA probe (CPU backend): engines keep measured-free behavior.
            return Self {
                student_engine_bytes: None,
                teacher_engine_bytes: None,
            };
        };
        let share = (free as f64 * rollout_mem_fraction) as usize;
        let autograd_reserve = free.saturating_sub(2 * share);
        eprintln!(
            "[opd-vram-plan] free={}MiB total={}MiB student_engine={}MiB teacher_engine={}MiB \
             autograd_reserve={}MiB (rollout_mem_fraction={rollout_mem_fraction})",
            free >> 20,
            total >> 20,
            share >> 20,
            share >> 20,
            autograd_reserve >> 20,
        );
        Self {
            student_engine_bytes: Some(share),
            teacher_engine_bytes: Some(share),
        }
    }
}

pub(super) fn run_opd_from_dirs(args: TrainOpdArgs) -> Result<()> {
    use autograd::{Tape, optim::AdamW};
    use train::{
        lora::LoraConfig,
        opd::{GkdLossConfig, OpdStepConfig, OpdStepInputs, opd_step_with_teacher},
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
    #[cfg(feature = "cuda")]
    probe_student_forward(&student, &mut store, "after_autograd_load");
    let student_params = trainable_param_ids(&student.all_parameter_ids(), &store);
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
    let vram_plan = OpdVramPlan::probe(&train_backend, args.runtime.rollout_mem_fraction);
    // scheduler_config() reserves 1/8 of per_req_cap for generation, so
    // max_prompt_tokens = max_seq_len * 7/8. Compensate by scaling
    // max_seq_len by 8/7 so the prompt cap covers prompt_max_tokens.
    #[cfg(feature = "cuda")]
    let engine_seq = {
        let seq = args.prompt_max_tokens + args.rollout_len + 32;
        (seq * 8 / 7 + 15) / 16 * 16
    };
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
            engine_seq,
            train_backend.clone(),
            cfg.vocab_size,
            &args.runtime,
            vram_plan.student_engine_bytes,
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
                probe_student_forward(&student, &mut store, "after_engine_load");
                maybe_preoffload_infer_student_before_teacher(&infer_student, &train_backend)?;
                probe_student_forward(&student, &mut store, "after_init_offload");
                let teacher = OpdCliTeacher::Infer(load_opd_infer_teacher(
                    teacher_dir,
                    engine_seq,
                    args.runtime.rollout_mem_fraction,
                    train_backend.clone(),
                    cfg.vocab_size,
                    vram_plan.teacher_engine_bytes,
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
        let inputs = OpdStepInputs::new(
            &student,
            &teacher_forward,
            prompt_ids,
            step_cfg.clone(),
            &student_params,
        )
        .gkd(GkdLossConfig {
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
        });
        #[cfg(feature = "cuda")]
        let inputs = inputs.infer_rollout(infer_rollout);
        let outcome = opd_step_with_teacher(
            inputs,
            &mut optimizer,
            &mut store,
            &mut tape,
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

pub(super) fn run_opd_smoke(args: TrainOpdArgs) -> Result<()> {
    use autograd::{Tape, optim::AdamW};
    use train::{
        opd::{GkdLossConfig, GkdSftAnchor, OpdStepConfig, OpdStepInputs, opd_step_with_teacher},
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
        let outcome = opd_step_with_teacher(
            OpdStepInputs::new(
                &student,
                &teacher_forward,
                &prompt_ids,
                step_cfg.clone(),
                &student_params,
            )
            .gkd(GkdLossConfig {
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
            }),
            &mut optimizer,
            &mut store,
            &mut tape,
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

pub(super) fn run_self_opd_from_dir(args: TrainSelfOpdArgs) -> Result<()> {
    use autograd::{Tape, optim::AdamW};
    use train::{
        ema_self_teacher::EmaSelfTeacher,
        lora::LoraConfig,
        opd::{
            GkdLossConfig, GkdSftAnchor, OpdKlMask, OpdStepConfig, OpdStepInputs,
            opd_step_with_teacher,
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

    let student_trainable = trainable_param_ids(&student.all_parameter_ids(), &store);
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
    let seq = prompt_ids.len() + args.rollout_len + 32;
    #[cfg(feature = "cuda")]
    let engine_seq = (seq * 8 / 7 + 15) / 16 * 16;
    #[cfg(feature = "cuda")]
    let infer_student = load_opd_infer_student(
        student_dir,
        engine_seq,
        train_backend.clone(),
        vocab,
        &args.runtime,
        OpdVramPlan::probe(&train_backend, args.runtime.rollout_mem_fraction).student_engine_bytes,
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
            let inputs = OpdStepInputs::new(
                &student,
                &teacher,
                &prompt_ids,
                step_cfg.clone(),
                &student_trainable,
            )
            .gkd(GkdLossConfig {
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
            });
            #[cfg(feature = "cuda")]
            let inputs = inputs.infer_rollout(infer_rollout);
            opd_step_with_teacher(
                inputs,
                &mut optimizer,
                &mut store,
                &mut tape,
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

pub(super) fn run_self_opd_smoke(args: TrainSelfOpdArgs) -> Result<()> {
    use autograd::{Tape, optim::AdamW};
    use train::{
        ema_self_teacher::EmaSelfTeacher,
        lora::LoraConfig,
        opd::{
            GkdLossConfig, GkdSftAnchor, OpdKlMask, OpdStepConfig, OpdStepInputs,
            opd_step_with_teacher,
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
    let student_trainable = trainable_param_ids(&student.all_parameter_ids(), &store);

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
            opd_step_with_teacher(
                OpdStepInputs::new(
                    &student,
                    &teacher,
                    &prompt_ids,
                    step_cfg.clone(),
                    &student_trainable,
                )
                .gkd(GkdLossConfig {
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
                }),
                &mut optimizer,
                &mut store,
                &mut tape,
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
