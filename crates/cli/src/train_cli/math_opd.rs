use anyhow::{Result, bail};

#[cfg(feature = "cuda")]
use super::cc_eval::JsonlSink;

use crate::args::TrainMathOpdArgs;

#[cfg(not(feature = "cuda"))]
pub(super) fn run_math_opd_impl(_args: TrainMathOpdArgs) -> Result<()> {
    bail!(
        "math-opd requires the cuda feature (the in-process rollout engine + writeback are \
         CUDA-only). Build with --features cuda,nccl."
    )
}

/// Math-OPD RFT loop: K samples per problem against the in-process serve,
/// reward = correctness minus a within-group length penalty, GSPO writeback.
/// Trains SHORTER correct reasoning; the grader is boxed-answer exact match
/// after the shared canonicalization. Single-GPU only.
#[cfg(feature = "cuda")]
pub(super) fn run_math_opd_impl(args: TrainMathOpdArgs) -> Result<()> {
    use autograd::optim::AdamW;
    use train::update_strategy::ScoredTrajectory;

    use super::{
        agent_opd::build_value_critic,
        cc_eval::JsonlSink,
        opd_checkpoint::{
            agent_opd_adapter_config, maybe_save_full_student_checkpoint, save_agent_opd_adapters,
            should_save_step_checkpoint,
        },
        opd_engine::{
            AgentOpdServeStudent, load_agent_opd_serve_student, quiesce_and_release_engines,
            sync_and_restore_engines,
        },
        opd_runtime::{log_opd_vram, parse_lora_target_set, validate_online_rollout_temperature},
    };
    use crate::args::{SyncArg, resolve_eval_out_dir};

    let student_dir = args.student_model.as_path();
    let target_set = parse_lora_target_set(&args.lora_target_set)?;
    let lora = train::lora::LoraConfig {
        rank: args.lora_rank,
        alpha: args.lora_alpha,
    };
    let update_preset = args.update_preset();
    validate_online_rollout_temperature(
        update_preset,
        args.update_strategy,
        args.rollout_temperature,
    )?;
    if args.cp_size.max(1) > 1 || args.dp_size.max(1) > 1 {
        bail!("math-opd is single-GPU only (--cp-size/--dp-size > 1 unsupported)");
    }
    let tasks = train::math_harness::load_tasks(&args.dataset, args.task_limit)?;
    let eval_tasks = match args.eval_dataset.as_deref() {
        Some(path) => train::math_harness::load_tasks(path, args.eval_n)?,
        None => Vec::new(),
    };
    let eval_out_dir = resolve_eval_out_dir(
        args.eval_out_dir.as_deref(),
        args.save_lora_adapters.as_deref(),
        args.save_checkpoint.as_deref(),
    );
    let metrics_path = args
        .metrics_out
        .clone()
        .unwrap_or_else(|| eval_out_dir.join("metrics.jsonl"));

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
    } = load_agent_opd_serve_student(&args.serve_args(), lora, target_set, args.serve_port)?;

    let mut optimizer = AdamW::new(args.lr, (0.9, 0.999), 1.0e-8, 0.0);
    let mut value_critic = build_value_critic(
        &update_preset,
        student.config().hidden_size,
        args.value_lr,
        &mut store,
    )?;

    let harness = train::math_harness::MathHarness {
        base_url: format!("http://127.0.0.1:{}", args.serve_port),
        model_id: cc_model_id,
        dump_dir,
        tokenizer: train::cc_harness::load_tokenizer(&student_dir.join("tokenizer.json"))?,
        max_tokens: args.max_tokens,
        agent: ureq::AgentBuilder::new()
            .timeout(std::time::Duration::from_secs(args.cc_timeout))
            .build(),
    };

    let lora_adapter_config = agent_opd_adapter_config(student_dir, target_set, lora);
    let metrics = JsonlSink::new(metrics_path);
    let sync_every_group = matches!(args.sync, SyncArg::EveryGroup);
    let preset_name = clap::ValueEnum::to_possible_value(&args.update_strategy)
        .map_or_else(String::new, |v| v.get_name().to_owned());
    let k = args.samples_per_prompt;
    let g = args.prompts_per_update.max(1);
    let alpha = args.length_penalty;
    let max_update_seq = args.runtime.max_update_seq;

    // Round-0 BASELINE held-out eval BEFORE any training.
    if !eval_tasks.is_empty() {
        let acc = run_math_eval(
            &harness,
            &eval_tasks,
            &metrics,
            "base",
            0,
            args.eval_concurrency,
        )?;
        eprintln!("[arle train math-opd] baseline eval accuracy={acc:.4}");
    }

    let mut policy_version = 0u64;
    for round in 0..args.rounds {
        // Re-acquire the rollout KV pool the previous writeback dropped.
        if let Err(err) = infer_student.ensure_kv_pool() {
            eprintln!("[math-opd] ensure KV pool (round {round}) failed: {err}");
        }
        let mut cap_left = args.writeback_cap.unwrap_or(usize::MAX);
        let mut losses: Vec<f32> = Vec::new();
        let (mut rollouts, mut correct) = (0usize, 0usize);
        let mut reward_sum = 0.0f64;
        let (mut prompt_tokens, mut completion_tokens) = (0u64, 0u64);
        let mut sync_lora_secs = 0.0f64;

        let indices: Vec<usize> = (0..g).map(|j| (round * g + j) % tasks.len()).collect();

        if sync_every_group {
            // Each group trains before the next rolls: strict on-policy per
            // group, one sync per group.
            for (group_id, &task_idx) in indices.iter().enumerate() {
                let behavior_version = policy_version;
                let rolled = roll_one_group(
                    &harness,
                    &tasks[task_idx],
                    &task_idx.to_string(),
                    group_id,
                    round,
                    behavior_version,
                    k,
                    args.rollout_temperature,
                    alpha,
                    &metrics,
                )?;
                rollouts += rolled.samples;
                correct += rolled.correct;
                reward_sum += rolled.reward_sum;
                prompt_tokens += rolled.prompt_tokens;
                completion_tokens += rolled.completion_tokens;

                let mut batch = rolled.trajectories;
                pre_filter_seq(&mut batch, max_update_seq);
                batch.truncate(cap_left);
                cap_left = cap_left.saturating_sub(batch.len());
                if batch.is_empty() {
                    continue;
                }
                quiesce_and_release_engines(&infer_student)?;
                log_opd_vram("math-opd pre-writeback", &train_backend);
                do_update(
                    &batch,
                    1,
                    behavior_version,
                    &update_preset,
                    &student,
                    &all_params,
                    &trainable,
                    &mut optimizer,
                    value_critic.as_mut(),
                    vocab,
                    args.writeback_window,
                    &mut store,
                    &metrics,
                    round,
                    &preset_name,
                    &mut losses,
                )?;
                trim_memory_pool(&mut store);
                sync_lora_secs += sync_and_restore_engines(
                    &infer_student,
                    &mut store,
                    &student.adapter_name_map(),
                    &student.param_name_map(),
                    lora,
                    true,
                )?;
                policy_version += 1;
            }
        } else {
            // All groups roll at ONE policy version, merged into one update.
            let behavior_version = policy_version;
            let mut merged: Vec<ScoredTrajectory> = Vec::new();
            for (group_id, &task_idx) in indices.iter().enumerate() {
                let rolled = roll_one_group(
                    &harness,
                    &tasks[task_idx],
                    &task_idx.to_string(),
                    group_id,
                    round,
                    behavior_version,
                    k,
                    args.rollout_temperature,
                    alpha,
                    &metrics,
                )?;
                rollouts += rolled.samples;
                correct += rolled.correct;
                reward_sum += rolled.reward_sum;
                prompt_tokens += rolled.prompt_tokens;
                completion_tokens += rolled.completion_tokens;
                merged.extend(rolled.trajectories);
            }
            pre_filter_seq(&mut merged, max_update_seq);
            merged.truncate(cap_left);
            cap_left = cap_left.saturating_sub(merged.len());
            if !merged.is_empty() {
                quiesce_and_release_engines(&infer_student)?;
                log_opd_vram("math-opd pre-writeback", &train_backend);
                do_update(
                    &merged,
                    g,
                    behavior_version,
                    &update_preset,
                    &student,
                    &all_params,
                    &trainable,
                    &mut optimizer,
                    value_critic.as_mut(),
                    vocab,
                    args.writeback_window,
                    &mut store,
                    &metrics,
                    round,
                    &preset_name,
                    &mut losses,
                )?;
                trim_memory_pool(&mut store);
                sync_lora_secs += sync_and_restore_engines(
                    &infer_student,
                    &mut store,
                    &student.adapter_name_map(),
                    &student.param_name_map(),
                    lora,
                    true,
                )?;
                policy_version += 1;
            }
        }

        if rollouts > 0 && completion_tokens == 0 {
            bail!(
                "math-opd round {round}: {rollouts} rollouts generated 0 completion tokens — \
                 serve/engine fault, not a weak model"
            );
        }
        let mean_loss = if losses.is_empty() {
            0.0
        } else {
            losses.iter().sum::<f32>() / losses.len() as f32
        };
        eprintln!(
            "[arle train math-opd] round {round}: rollouts={rollouts} correct={correct} \
             accuracy={:.4} reward_mean={:.4} mean_loss={mean_loss:.4}",
            correct as f64 / rollouts.max(1) as f64,
            reward_sum / rollouts.max(1) as f64,
        );
        log_opd_vram(&format!("after round {round} writeback"), &train_backend);

        let is_final_round = round + 1 == args.rounds;
        let do_eval = !eval_tasks.is_empty()
            && (is_final_round || (args.eval_every > 0 && (round + 1) % args.eval_every == 0));
        if do_eval {
            let acc = run_math_eval(
                &harness,
                &eval_tasks,
                &metrics,
                &(round + 1).to_string(),
                round + 1,
                args.eval_concurrency,
            )?;
            eprintln!("[arle train math-opd] round {round}: held-out accuracy={acc:.4}");
        }

        if let Some(adapter_dir) = args.save_lora_adapters.as_deref()
            && should_save_step_checkpoint(round + 1, args.rounds, args.save_every)
        {
            save_agent_opd_adapters(
                adapter_dir,
                &format!("adapters_round{}", round + 1),
                round + 1,
                student_dir,
                &student,
                &mut store,
                &lora_adapter_config,
            )?;
        }
        let mut ckpt_tape = autograd::Tape::new();
        maybe_save_full_student_checkpoint(
            "math-opd",
            args.save_checkpoint.as_deref(),
            args.save_every,
            round + 1,
            args.rounds,
            student_dir,
            &student,
            &mut store,
            &mut ckpt_tape,
        )?;

        metrics.append(&serde_json::json!({
            "kind": "round",
            "round": round,
            "tasks": g,
            "rollouts": rollouts,
            "correct": correct,
            "accuracy": correct as f64 / rollouts.max(1) as f64,
            "reward_mean": reward_sum / rollouts.max(1) as f64,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "mean_train_loss": mean_loss,
            "sync_lora_secs": sync_lora_secs,
            "policy_version": policy_version,
        }));
    }

    serve_thread.shutdown()?;
    eprintln!("[arle train math-opd] done ({} rounds)", args.rounds);
    Ok(())
}

#[cfg(feature = "cuda")]
struct RolledGroup {
    samples: usize,
    correct: usize,
    reward_sum: f64,
    prompt_tokens: u64,
    completion_tokens: u64,
    trajectories: Vec<train::update_strategy::ScoredTrajectory>,
}

/// Roll one task group: K concurrent samples → grade → dump convert →
/// within-group length-shaped rewards → scored trajectories. The reward is
/// computed AFTER convert, so `len` is the engine's own generated-token count.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn roll_one_group(
    harness: &train::math_harness::MathHarness,
    task: &train::math_harness::MathTask,
    task_id: &str,
    group_id: usize,
    round: usize,
    behavior_version: u64,
    k: usize,
    temperature: f32,
    alpha: f32,
    metrics: &JsonlSink,
) -> Result<RolledGroup> {
    let rollout = harness.run_group(task, k, temperature, train::math_harness::next_nonce());
    let passed: Vec<bool> = rollout
        .samples
        .iter()
        .map(|s| train::math_harness::grade(&task.answer, &s.text))
        .collect();
    let windows = train::math_harness::to_windows(task_id, &rollout);
    let records =
        train::cc_convert::convert_cc_dumps(&harness.dump_dir, &harness.tokenizer, &windows)?;

    // Records carry "{task_id}#{sample}#r{seq}" labels; one request per window.
    let prefix = format!("{task_id}#");
    let sample_of = |label: &str| -> Option<usize> {
        label.strip_prefix(&prefix)?.split('#').next()?.parse().ok()
    };
    let correct_lens: Vec<usize> = records
        .iter()
        .filter(|r| sample_of(&r.label).is_some_and(|s| passed[s]))
        .map(|r| r.response_ids.len())
        .collect();
    let len_min = correct_lens.iter().copied().min().unwrap_or(0);
    let len_max = correct_lens.iter().copied().max().unwrap_or(0);
    let span = (len_max - len_min) as f32;

    let mut sample_rewards = vec![0.0f32; rollout.samples.len()];
    let mut trajectories = Vec::with_capacity(records.len());
    for record in records {
        let Some(s) = sample_of(&record.label) else {
            continue;
        };
        let reward = if !passed[s] {
            0.0
        } else if span == 0.0 {
            1.0
        } else {
            1.0 - alpha * (record.response_ids.len() - len_min) as f32 / (span + 1e-6)
        };
        sample_rewards[s] = reward;
        let sample = &rollout.samples[s];
        trajectories.push(train::update_strategy::ScoredTrajectory {
            prompt_ids: record.prompt_ids,
            response_ids: record.response_ids,
            response_mask: record.response_mask,
            reward,
            behavior_logprobs: (!record.gen_logprobs.is_empty()).then_some(record.gen_logprobs),
            group_id,
            truncated: sample.timed_out || sample.capped,
        });
    }

    let correct = passed.iter().filter(|p| **p).count();
    let reward_mean = sample_rewards.iter().sum::<f32>() / sample_rewards.len().max(1) as f32;
    let reward_std = (sample_rewards
        .iter()
        .map(|r| (r - reward_mean).powi(2))
        .sum::<f32>()
        / sample_rewards.len().max(1) as f32)
        .sqrt();
    let zero_variance = sample_rewards.windows(2).all(|w| w[0] == w[1]);
    let prompt_tokens: u64 = rollout.samples.iter().map(|s| s.input_tokens).sum();
    let completion_tokens: u64 = rollout.samples.iter().map(|s| s.output_tokens).sum();
    let think_tokens: u64 = rollout.samples.iter().map(|s| s.think_tokens).sum();
    let answer_tokens: u64 = rollout.samples.iter().map(|s| s.answer_tokens).sum();
    let capped = rollout.samples.iter().filter(|s| s.capped).count();
    if completion_tokens == 0 {
        eprintln!(
            "[math-opd] group {group_id} (task {task_id}) generated 0 completion tokens across {} \
             sample(s) — check the engine log",
            rollout.samples.len()
        );
    }
    metrics.append(&serde_json::json!({
        "kind": "group",
        "round": round,
        "task_id": task_id,
        "group_id": group_id,
        "behavior_version": behavior_version,
        "rewards": sample_rewards,
        "reward_mean": reward_mean,
        "reward_std": reward_std,
        "zero_variance": zero_variance,
        "passed": correct,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "think_tokens": think_tokens,
        "answer_tokens": answer_tokens,
        "capped": capped,
    }));

    Ok(RolledGroup {
        samples: rollout.samples.len(),
        correct,
        reward_sum: f64::from(sample_rewards.iter().sum::<f32>()),
        prompt_tokens,
        completion_tokens,
        trajectories,
    })
}

/// VRAM-wall length guard before writeback (mirrors agent-opd): skipped here
/// == skipped by update() anyway, while keeping an overlong record out of the
/// batch-wide sidecar validation and any subsequent model forward.
#[cfg(feature = "cuda")]
fn pre_filter_seq(
    batch: &mut Vec<train::update_strategy::ScoredTrajectory>,
    max_update_seq: usize,
) {
    if max_update_seq == 0 {
        return;
    }
    batch.retain(|t| {
        let seq = t.prompt_ids.len() + t.response_ids.len();
        let keep = seq <= max_update_seq;
        if !keep {
            eprintln!(
                "[math-opd] SKIP trajectory pre-capture: seq {seq} > max_update_seq \
                 {max_update_seq} (VRAM wall), prompt {} reward {:.2}",
                t.prompt_ids.len(),
                t.reward
            );
        }
        keep
    });
}

#[cfg(feature = "cuda")]
fn trim_memory_pool(store: &mut autograd::TensorStore) {
    if let Err(err) = store.backend().trim_memory_pool() {
        eprintln!("[math-opd] device-pool trim after writeback failed: {err}");
    }
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn do_update<O: autograd::Optimizer>(
    batch: &[train::update_strategy::ScoredTrajectory],
    groups: usize,
    behavior_version: u64,
    preset: &train::update_strategy::UpdatePreset,
    student: &train::qwen35::Qwen35Model,
    all_params: &[autograd::TensorId],
    trainable: &[autograd::TensorId],
    optimizer: &mut O,
    critic: Option<&mut train::opd::ValueCritic>,
    vocab: usize,
    window: usize,
    store: &mut autograd::TensorStore,
    metrics: &JsonlSink,
    round: usize,
    preset_name: &str,
    losses: &mut Vec<f32>,
) -> Result<()> {
    let started = std::time::Instant::now();
    let report = preset.update(
        batch, student, all_params, trainable, optimizer, critic, vocab, window, store,
    )?;
    losses.push(report.loss);
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
        "update_secs": started.elapsed().as_secs_f64(),
        "groups": groups,
        "behavior_version": behavior_version,
    });
    metrics.append(&row);
    if report.trained > 0 && report.tokens == 0 {
        bail!(
            "math-opd round {round} preset {preset_name}: {} trajectories trained 0 tokens — \
             the update is a no-op (--max-update-seq below the corpus?)",
            report.trained
        );
    }
    let clip_frac = report.stats.clip_frac();
    if !report.stats.ratio_max.is_finite() || !clip_frac.is_finite() || clip_frac >= 1.0 {
        bail!(
            "math-opd round {round} preset {preset_name}: degenerate policy/rollout divergence \
             (ratio_max={:.3}, clip_frac={clip_frac:.3}) — clipped to nothing or numerically \
             blown up",
            report.stats.ratio_max
        );
    }
    Ok(())
}

/// Greedy (K=1, temp=0) held-out eval: accuracy + completion-token stats,
/// broken down by `source` when present. No sidecars/logprobs needed.
#[cfg(feature = "cuda")]
fn run_math_eval(
    harness: &train::math_harness::MathHarness,
    tasks: &[train::math_harness::MathTask],
    metrics: &JsonlSink,
    label: &str,
    round: usize,
    concurrency: usize,
) -> Result<f32> {
    use std::sync::atomic::{AtomicUsize, Ordering};

    if tasks.is_empty() {
        return Ok(0.0);
    }
    let concurrency = concurrency.clamp(1, tasks.len());
    let next = AtomicUsize::new(0);
    let (tx, rx) = std::sync::mpsc::channel::<(usize, Option<String>, bool, u64)>();
    std::thread::scope(|scope| {
        for _ in 0..concurrency {
            let next = &next;
            let tx = tx.clone();
            scope.spawn(move || {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(task) = tasks.get(i) else {
                        break;
                    };
                    let rollout =
                        harness.run_group(task, 1, 0.0, train::math_harness::next_nonce());
                    let sample = &rollout.samples[0];
                    let correct = train::math_harness::grade(&task.answer, &sample.text);
                    let _ = tx.send((i, task.source.clone(), correct, sample.output_tokens));
                }
            });
        }
    });
    drop(tx);
    let mut results: Vec<(Option<String>, bool, u64)> =
        (0..tasks.len()).map(|_| (None, false, 0)).collect();
    for (i, source, correct, tokens) in rx {
        results[i] = (source, correct, tokens);
    }
    let total = results.len();
    let correct = results.iter().filter(|(_, c, _)| *c).count();
    let accuracy = correct as f32 / total as f32;
    let mut toks: Vec<u64> = results.iter().map(|(_, _, t)| *t).collect();
    toks.sort_unstable();
    let mean = toks.iter().sum::<u64>() as f64 / total as f64;
    let median = if total % 2 == 1 {
        toks[total / 2] as f64
    } else {
        (toks[total / 2 - 1] + toks[total / 2]) as f64 / 2.0
    };
    let mut by_source: std::collections::BTreeMap<&str, (usize, usize)> =
        std::collections::BTreeMap::new();
    for (source, c, _) in &results {
        let entry = by_source
            .entry(source.as_deref().unwrap_or("unknown"))
            .or_insert((0, 0));
        entry.0 += usize::from(*c);
        entry.1 += 1;
    }
    let by_source_value = serde_json::Value::Object(
        by_source
            .iter()
            .map(|(k, (c, t))| {
                (
                    (*k).to_owned(),
                    serde_json::json!({
                        "correct": c,
                        "total": t,
                        "accuracy": *c as f32 / *t as f32,
                    }),
                )
            })
            .collect(),
    );
    metrics.append(&serde_json::json!({
        "kind": "eval",
        "label": label,
        "round": round,
        "accuracy": accuracy,
        "tasks": total,
        "correct": correct,
        "completion_tokens_mean": mean,
        "completion_tokens_median": median,
        "by_source": by_source_value,
    }));
    eprintln!(
        "[math-opd] eval[{label}]: accuracy={accuracy:.4} ({correct}/{total}) \
         mean_tokens={mean:.0} median_tokens={median:.0}"
    );
    Ok(accuracy)
}
