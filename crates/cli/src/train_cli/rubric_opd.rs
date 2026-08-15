use anyhow::{Result, bail};

use crate::args::TrainRubricOpdArgs;
#[cfg(feature = "cuda")]
use {
    super::{
        model_probe::resolve_local_tokenizer_path,
        opd_checkpoint::{maybe_save_full_student_checkpoint, should_save_step_checkpoint},
        opd_engine::shared_frozen_base_entries,
        opd_runtime::{
            apply_tape_dtype, build_opd_store, log_opd_vram, parse_lora_target_set,
            rollout_sampling_params, trainable_param_ids,
        },
    },
    anyhow::{Context, anyhow},
    autograd::{Tape, TensorId},
    infer_plan::SamplingParams,
    qwen35_spec::Qwen35Config,
    std::{fs, path::Path, time::Instant},
};

#[cfg(not(feature = "cuda"))]
pub(super) fn run_rubric_opd_impl(_args: TrainRubricOpdArgs) -> Result<()> {
    bail!(
        "rubric-opd requires the cuda feature (the Flash judge + rollout engines are \
         CUDA-only). Build with --features cuda,nccl."
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
pub(super) fn run_rubric_opd_impl(args: TrainRubricOpdArgs) -> Result<()> {
    use std::sync::{Arc, Mutex};

    use autograd::optim::AdamW;
    use infer_api::{EngineLoadConfig, LoadedInferenceEngine};
    use train::{
        infer_student::{InferStudent, save_lora_adapters},
        lora::LoraConfig,
        opd::full_batch_ce_writeback_step,
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

    // Shared frozen-base pointers alias engine weight buffers that a student
    // offload frees mid-step — refuse the combination at load time.
    if !args.no_share_frozen_base && train::opd::engine_offload_mode().offloads_student() {
        bail!(
            "--engine-offload student is incompatible with frozen-base sharing; \
             pass --no-share-frozen-base"
        );
    }

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
            // The student is always single-GPU; the judge may be multi-GPU.
            tp_size: Some(1),
            ..EngineLoadConfig::default()
        },
    )
    .with_context(|| format!("load rollout engine from {}", student_dir.display()))?;

    // Train-infer FP8 weight sharing (`--share-frozen-base`): borrow the rollout
    // engine's resident FP8 base pointers, map to the loader's backend-agnostic
    // table, and pass it into the autograd student load so its frozen FP8 base
    // projections import a NON-OWNING view instead of allocating ~27 GB.
    let shared_base_entries: Vec<SharedFrozenBaseEntry> = if !args.no_share_frozen_base {
        shared_frozen_base_entries(&student_engine, "rubric-opd")?
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
    let trainable = trainable_param_ids(&all_params, &store);
    if trainable.is_empty() {
        bail!("rubric-opd student has no trainable (LoRA) parameters; check --lora-target-set");
    }

    let infer_student = InferStudent::new(
        Arc::new(Mutex::new(student_engine)),
        train_backend.clone(),
        vocab,
    )
    .with_lora_merge_fp8(args.lora_merge_fp8);

    // Judge engine (DeepSeek-V4-Flash) — text in, verdict out (own tokenizer).
    // Self-consistency mode loads NO judge (the student majority-votes on its own
    // \boxed answer), freeing the ~35GB judge VRAM; teacher-model is ignored.
    // Multi-GPU models (DSv4, Qwen35 MoE) spawn a separate `arle serve` child
    // process with its own TP env; the parent stays single-GPU for the student.
    // Keep-alive handle: the server must outlive the rounds loop.
    let mut _judge_server: Option<JudgeServer> = None;
    let judge = if args.self_consistency {
        eprintln!(
            "[arle train rubric-opd] self-consistency mode: no judge engine loaded (majority-vote on \\boxed)"
        );
        None
    } else {
        let teacher_str = teacher_dir
            .to_str()
            .ok_or_else(|| anyhow!("teacher path is not valid UTF-8"))?;
        let judge_prompt_cap = (student_seq * 2 + 1024).max(2048);
        let judge_total = judge_prompt_cap + args.max_verdict_tokens;
        let tp_size = crate::serve_multiproc::world_size_from_env();
        let judge_config = EngineLoadConfig {
            num_slots: args.judge_num_slots,
            page_size: 16,
            total_pages: args.judge_num_slots.max(1) * judge_total.div_ceil(16),
            max_prompt_tokens: judge_prompt_cap,
            max_total_tokens: judge_total,
            chunked_prefill_size: Some(judge_prompt_cap),
            tp_size: Some(tp_size),
            ..EngineLoadConfig::default()
        };
        if infer_api::cuda_model_takes_multiproc_serve(teacher_str) {
            if tp_size > 1 {
                eprintln!("[arle train rubric-opd] spawning judge serve child (TP={tp_size})");
                let server =
                    JudgeServer::spawn(teacher_str, tp_size, judge_prompt_cap, judge_total)?;
                let endpoint = server.endpoint().to_string();
                _judge_server = Some(server);
                Some(FlashJudge::new_remote(endpoint, args.max_verdict_tokens))
            } else {
                eprintln!(
                    "[arle train rubric-opd] WARNING: model {} wants multi-GPU but INFER_TP_SIZE=1; loading in-process (may OOM)",
                    teacher_dir.display()
                );
                let judge_engine =
                    LoadedInferenceEngine::load_with_config(teacher_str, true, judge_config)
                        .with_context(|| {
                            format!("load Flash judge from {}", teacher_dir.display())
                        })?;
                Some(FlashJudge::new(
                    Arc::new(Mutex::new(judge_engine)),
                    args.max_verdict_tokens,
                ))
            }
        } else {
            let judge_engine =
                LoadedInferenceEngine::load_with_config(teacher_str, true, judge_config)
                    .with_context(|| format!("load Flash judge from {}", teacher_dir.display()))?;
            Some(FlashJudge::new(
                Arc::new(Mutex::new(judge_engine)),
                args.max_verdict_tokens,
            ))
        }
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
                    full_batch_ce_writeback_step(
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

#[cfg(feature = "cuda")]
struct JudgeServer {
    child: std::process::Child,
    endpoint: String,
}

#[cfg(feature = "cuda")]
impl JudgeServer {
    fn spawn(
        model_path: &str,
        tp_size: usize,
        max_prompt_tokens: usize,
        max_total_tokens: usize,
    ) -> Result<Self> {
        let port = free_port();
        let exe = std::env::current_exe().context("current_exe")?;
        // The student (TP=1) uses the first visible GPU. Pin the judge's TP
        // workers to the next `tp_size` physical GPUs so its rank-0 weight load
        // does not OOM against the student on a shared GPU. The student's GPU
        // is the first entry of the parent's CUDA_VISIBLE_DEVICES, or 0 if unset.
        let student_gpu: usize = std::env::var("CUDA_VISIBLE_DEVICES")
            .ok()
            .and_then(|s| {
                s.split(',')
                    .next()
                    .and_then(|first| first.trim().parse().ok())
            })
            .unwrap_or_else(|| {
                log::warn!("CUDA_VISIBLE_DEVICES parse failed, defaulting judge to GPU 0");
                0
            });
        let judge_gpus: String = (1..=tp_size)
            .map(|i| (student_gpu + i).to_string())
            .collect::<Vec<_>>()
            .join(",");
        let mut child = std::process::Command::new(exe)
            .args([
                "serve",
                "--model-path",
                model_path,
                "--port",
                &port.to_string(),
                "--bind",
                "127.0.0.1",
                "--max-prompt-tokens",
                &max_prompt_tokens.to_string(),
                "--max-total-tokens",
                &max_total_tokens.to_string(),
            ])
            .env("INFER_TP_SIZE", tp_size.to_string())
            .env("CUDA_VISIBLE_DEVICES", judge_gpus)
            .spawn()
            .context("spawn judge serve process")?;
        let endpoint = format!("http://127.0.0.1:{port}");
        let url = format!("{endpoint}/health");
        // Large models (DSv4-Flash ~236B) can take 30+ minutes to load across
        // TP workers. Poll for up to an hour, and fail fast if the child exits.
        for _ in 0..7200 {
            if reqwest::blocking::get(&url).is_ok() {
                return Ok(Self { child, endpoint });
            }
            if let Ok(Some(status)) = child.try_wait() {
                bail!("judge serve child exited with {status} before becoming healthy");
            }
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
        bail!("judge serve at {endpoint} did not become healthy within 1h")
    }

    fn endpoint(&self) -> &str {
        &self.endpoint
    }
}

#[cfg(feature = "cuda")]
impl Drop for JudgeServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[cfg(feature = "cuda")]
fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .expect("bind free port")
        .local_addr()
        .expect("local addr")
        .port()
}
