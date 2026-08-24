use anyhow::{Result, bail};

use crate::args::TrainAgentOpdArgs;
#[cfg(feature = "cuda")]
use {
    anyhow::{Context, anyhow},
    autograd::{Tape, TensorId},
    infer_plan::SamplingParams,
    std::{
        path::{Path, PathBuf},
        sync::Arc,
        time::Instant,
    },
    train::swe_dataset::SweTask,
};

#[cfg(feature = "cuda")]
use super::{
    agent_opd_batch::{ReplayBuffer, TaskSelection},
    agent_opd_mesh::{MeshMsgRef, MeshUpdateChannel, run_agent_opd_cp_follower},
    agent_opd_window::{GroupLauncher, PendingGroup},
    cc_eval::{JsonlSink, agent_opd_eval_out_dir, run_cc_eval},
    opd_checkpoint::{
        agent_opd_adapter_config, maybe_save_full_student_checkpoint, save_agent_opd_adapters,
        should_save_step_checkpoint,
    },
    opd_engine::{
        AgentOpdServeStudent, load_agent_opd_serve_student, quiesce_and_release_engines,
        sync_and_restore_engines,
    },
    opd_runtime::{
        PromptSampler, log_opd_vram, parse_lora_target_set, validate_online_rollout_temperature,
    },
    replay_records::run_agent_opd_replay,
};

#[cfg(not(feature = "cuda"))]
pub(super) fn run_agent_opd_impl(_args: TrainAgentOpdArgs) -> Result<()> {
    bail!(
        "agent-opd requires the cuda feature (the in-process rollout engine + CE \
         writeback are CUDA-only). Build with --features cuda,nccl."
    )
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
        optimizer.zero_grad(store, all_params)?;
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

/// Value critic (ValueGae presets): built after the `trainable` filter and not
/// a student param, so the policy optimizer / LoRA sync / adapter save never
/// touch it — fully isolated, own AdamW.
#[cfg(feature = "cuda")]
pub(super) fn build_value_critic(
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

#[cfg(feature = "cuda")]
pub(super) fn run_agent_opd_impl(args: TrainAgentOpdArgs) -> Result<()> {
    use autograd::optim::AdamW;
    use std::collections::VecDeque;

    use train::lora::LoraConfig;

    let student_dir = args.student_model.as_path();
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

    let tasks = load_agent_opd_tasks(dataset, staged_root, args.task_limit)?;
    // Pass-rate selection state: in-memory, this run only (metrics.jsonl has the history).
    let mut selection = args.task_selection.then(|| TaskSelection::new(tasks.len()));

    let eval_tasks = load_agent_opd_eval_tasks(&args, staged_root, &tasks)?;
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

    if args.synthetic_writeback_seq > 0 {
        return run_synthetic_writeback(
            &args,
            &student,
            all_params.as_slice(),
            trainable.as_slice(),
            &mut optimizer,
            vocab,
            &infer_student,
            &train_backend,
            &mut store,
        );
    }

    // cc rollout harness over the serve. --staleness 0 (default): task groups
    // run sequentially — in-flight concurrency = the K samples of one group.
    // --staleness 1: the next group rolls on a background thread while this
    // group trains+merges (Arc'd for that thread's 'static bound).
    let cp = train::context_parallel::CpContext::from_env();

    // The agent gets Bash and no filesystem confinement: as root it read
    // eval.jsonl, matched its own instance_id and printed the test_patch it was
    // scored against (errors/2026-08-24). Refuse to start unless the answers are
    // out of its reach — a readable corpus yields a run that looks normal and
    // measures nothing.
    let rollout_user = train::sandbox::resolve_rollout_user(&args.rollout_user)?;
    let secrets: Vec<&std::path::Path> = [&args.dataset, &args.staged_root, &args.eval_dataset]
        .into_iter()
        .filter_map(|p| p.as_deref())
        .collect();
    train::sandbox::assert_corpus_unreadable(&secrets, rollout_user)?;

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
        rollout_user,
    });

    // Prompts per update (verl shape): G groups roll concurrently under ONE
    // policy version, then a single update trains their merged batch. Keeps
    // G × width sessions in flight so the fleet's slots stay fed and one
    // group's straggler no longer idles the others; staleness stays 0 at any
    // G. G=1 is the per-group loop.
    let launcher = GroupLauncher {
        tasks: &tasks,
        harness: &harness,
        staleness: args.staleness,
        width,
        groups_per_update: args.prompts_per_update.max(1),
    };

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
    // Consecutive zero-edit rounds before the run is called broken: a harness
    // fault wears a weak model's clothes.
    let mut edit_drought_rounds = 0usize;
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
        let mut round_any_edited = false;
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
        let mut pending: VecDeque<(usize, PendingGroup)> = VecDeque::new();
        let mut launched = 0usize;
        launcher.top_up(&mut pending, &mut launched, &work, policy_version);
        let mut pos = 0;
        let mut groups_done = 0usize;
        while pos < work.len() {
            let window: Vec<(usize, PendingGroup)> = pending.drain(..).collect();
            let n = window.len();
            // Boot-ahead (staleness 0): the next window's sandboxes build during
            // this window's rollout + train (CPU only — staleness-free).
            if args.staleness == 0 {
                launcher.top_up(&mut pending, &mut launched, &work, policy_version);
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
            // The remerge is atomic vs this in-flight traffic — see ServeEngine::remerge_student_lora.
            if args.staleness > 0 {
                launcher.top_up(&mut pending, &mut launched, &work, policy_version);
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
                round_any_edited |= group.samples.iter().any(|s| s.edited);
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
                    // A group the preset discards must not spend a budget reserved
                    // for trainable trajectories: a zero-variance group took the
                    // whole round's cap and left the one group carrying signal
                    // truncated to nothing (errors/2026-08-24).
                    if group_trained {
                        batch.truncate(cap_left);
                        cap_left -= batch.len();
                    } else {
                        batch.clear();
                    }
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
                if report.trained > 0 && report.tokens == 0 {
                    bail!(
                        "agent-opd round {round} preset {preset_name}: {} trajectories trained 0 \
                         tokens — the update is a no-op (--max-update-seq below the corpus?)",
                        report.trained
                    );
                }
                let clip_frac = report.stats.clip_frac();
                if !report.stats.ratio_max.is_finite() || !clip_frac.is_finite() || clip_frac >= 1.0
                {
                    bail!(
                        "agent-opd round {round} preset {preset_name}: degenerate policy/rollout \
                         divergence (ratio_max={:.3}, clip_frac={clip_frac:.3}) — clipped to \
                         nothing or numerically blown up",
                        report.stats.ratio_max
                    );
                }
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
            launcher.top_up(&mut pending, &mut launched, &work, policy_version);
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

        // 3, not the monitor's 5: an in-loop abort needs no human-paging margin.
        const EDIT_DROUGHT_ROUNDS: usize = 3;
        if rollouts > 0 && !round_any_edited {
            edit_drought_rounds += 1;
            if edit_drought_rounds >= EDIT_DROUGHT_ROUNDS {
                bail!(
                    "agent-opd: {EDIT_DROUGHT_ROUNDS} consecutive rounds with no edited rollout \
                     (round {round}, {rollouts} rollouts) — harness fault (cc binary missing, \
                     stream aborted, trajectory skip), not a weak model"
                );
            }
        } else {
            edit_drought_rounds = 0;
        }

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

/// SWE-bench-Pro tasks; each task's staged tree is `<staged_root>/<instance_id>/`.
#[cfg(feature = "cuda")]
fn load_agent_opd_tasks(
    dataset: &Path,
    staged_root: &Path,
    task_limit: Option<usize>,
) -> Result<Vec<(Arc<SweTask>, PathBuf)>> {
    let mut tasks_raw = train::swe_dataset::load_swe_tasks(dataset)?;
    if let Some(n) = task_limit {
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
    Ok(tasks)
}

/// Held-out eval tasks (separate from `--dataset`), staged under
/// `--eval-staged-root` (falling back to `--staged-root`). Loaded once up front
/// so the round-0 baseline and every per-round eval reuse the same set.
#[cfg(feature = "cuda")]
fn load_agent_opd_eval_tasks(
    args: &TrainAgentOpdArgs,
    staged_root: &Path,
    tasks: &[(Arc<SweTask>, PathBuf)],
) -> Result<Vec<(Arc<SweTask>, PathBuf)>> {
    match args.eval_dataset.as_deref() {
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
            Ok(eval_tasks)
        }
        None => Ok(Vec::new()),
    }
}

/// Diagnostic: skip the (slow, stochastic) agent rollout and drive ONE masked-CE
/// writeback on a synthetic trajectory of length N, so the writeback's OOM
/// instrumentation fires deterministically. Same `masked_writeback_step` call
/// the round loop makes, with the trained student + optimizer + store.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn run_synthetic_writeback(
    args: &TrainAgentOpdArgs,
    student: &train::qwen35::Qwen35Model,
    all_params: &[TensorId],
    trainable: &[TensorId],
    optimizer: &mut autograd::optim::AdamW,
    vocab: usize,
    infer_student: &train::infer_student::InferStudent,
    train_backend: &std::sync::Arc<dyn autograd::Backend>,
    store: &mut autograd::TensorStore,
) -> Result<()> {
    use train::opd::{WritebackLoss, masked_writeback_step};

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
    log_opd_vram("synthetic-writeback pre-release", train_backend);
    // Mirror the real round closure: release BOTH the inference scratch and
    // the (dead) rollout KV pool before the masked-CE writeback, so the
    // diagnostic reproduces the real writeback's resident floor.
    if let Err(err) = infer_student.release_inference_scratch() {
        eprintln!("[synthetic-writeback] release inference scratch failed: {err}");
    }
    if let Err(err) = infer_student.release_kv_pool() {
        eprintln!("[synthetic-writeback] release KV pool failed: {err}");
    }
    log_opd_vram("synthetic-writeback pre-writeback", train_backend);
    let started = std::time::Instant::now();
    let cp = train::context_parallel::CpContext::from_env();
    let dp = train::context_parallel::DpContext::from_env();
    let fd_target = std::env::var("ARLE_OPD_FD_PARAM").ok();
    // The FD probe reads raw grads, so it must not let the optimizer move them.
    let loss = masked_writeback_step(
        WritebackLoss::Ce,
        student,
        all_params,
        trainable,
        optimizer,
        fd_target.is_none(),
        &prompt_ids,
        &response_ids,
        &response_mask,
        vocab,
        args.writeback_window,
        cp,
        dp,
        store,
    )?
    .0;
    log_opd_vram("synthetic-writeback post-writeback", train_backend);
    if let Some(target) = fd_target {
        synthetic_writeback_fd_probe(
            &target,
            student,
            all_params,
            trainable,
            optimizer,
            &prompt_ids,
            &response_ids,
            &response_mask,
            vocab,
            args.writeback_window,
            cp,
            dp,
            store,
        )?;
    }
    eprintln!(
        "[synthetic-writeback] DONE loss={loss:.6} elapsed={:?}",
        started.elapsed()
    );
    Ok(())
}
