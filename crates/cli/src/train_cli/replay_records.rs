use std::fs;

use anyhow::{Context, Result, bail};

use crate::args::TrainCcConvertArgs;
#[cfg(feature = "cuda")]
use {
    super::{
        agent_opd::build_value_critic,
        opd_checkpoint::{agent_opd_adapter_config, save_agent_opd_adapters},
        opd_runtime::{apply_tape_dtype, build_opd_store, log_opd_vram, trainable_param_ids},
    },
    crate::args::TrainAgentOpdArgs,
    autograd::TensorId,
    qwen35_spec::Qwen35Config,
    std::path::Path,
};

/// `arle train cc-convert` — backend-independent (no CUDA): dumps → verl-style
/// token records via `train::cc_convert`.
pub(super) fn run_cc_convert_impl(args: TrainCcConvertArgs) -> Result<()> {
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
    record.reward >= 1.0 && record.response_mask.contains(&1)
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
pub(super) fn run_agent_opd_replay(
    args: &TrainAgentOpdArgs,
    records_path: &Path,
    lora: train::lora::LoraConfig,
    target_set: train::lora::LoraTargetSet,
) -> Result<()> {
    use autograd::optim::AdamW;
    use train::{
        ema_self_teacher::EmaSelfTeacher,
        opd::{WritebackLoss, masked_gkd_writeback_step, masked_writeback_step},
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
    let trainable = trainable_param_ids(&all_params, &store);
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
                        masked_gkd_writeback_step(
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
                    masked_writeback_step(
                        WritebackLoss::Ce,
                        &student,
                        all_params.as_slice(),
                        trainable.as_slice(),
                        &mut optimizer,
                        true,
                        &record.prompt_ids,
                        &record.response_ids,
                        &record.response_mask,
                        vocab,
                        args.writeback_window,
                        train::context_parallel::CpContext::from_env(),
                        train::context_parallel::DpContext::from_env(),
                        &mut store,
                    )
                    .map(|(loss, _, _)| loss)
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
