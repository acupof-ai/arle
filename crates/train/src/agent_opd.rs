//! Agentic on-policy distillation (agent-OPD) rollout driver.
//!
//! The student LLM (in-process [`InferStudent`] rollout engine) drives the
//! [`agent::AgentSession`] tool loop — read / write / replace / bash — against a
//! per-task repo sandbox. The student EDITS the tree directly; the reward is
//! execution: `git diff` is the candidate patch, the hidden `test_patch` +
//! `fail_to_pass` tests are run, exit-0 = pass.
//!
//! Passing trajectories are written back as cross-entropy targets via
//! [`crate::opd::masked_writeback_ce_step`]: ONE masked next-token CE over the
//! whole `prompt ++ response` trajectory, windowed over the sequence so only a
//! `[window, vocab]` logits tile is ever live. The verl-style `response_mask`
//! (`1` = LLM-generated, `0` = tool/environment) selects which next-token
//! targets receive loss — tool-result tokens are context the LLM predictions
//! condition on but never a loss target, so the student never learns to
//! hallucinate tool output.
//!
//! This replaces the prior per-assistant `(context_prefix, llm_span)` pair
//! explosion (`tokens_record_to_pairs` → `rubric_writeback_ce_step_batched`),
//! which re-forwarded the growing ~30K-token prefix once per turn and
//! materialized a dense `[seq, vocab]` logits tensor → O(N²) work and
//! `cuda alloc_zeros failed` OOM on long agentic trajectories. The masked
//! single-trajectory CE forwards the trajectory once per window instead.
//! Only the rollout front-end + the reward differ from rubric-OPD; the CE /
//! optimizer machinery is shared. `tokens_record_to_pairs` is retained (still
//! unit-tested) but is no longer the writeback path.
//!
//! The rollout machinery needs the CUDA in-process student, so it is gated
//! behind `feature = "cuda"` (mirroring [`crate::infer_student`]). The pure
//! token-pair explosion below is backend-independent and unit-tested everywhere.

/// Explode a verl-style token record into per-assistant `(prefix, span)` CE
/// pairs. `response_mask[i] == 1` marks an LLM-generated token, `0` a
/// tool/environment token. Each maximal run of `1`s becomes one pair whose
/// completion is that LLM span and whose prompt is `prompt_ids` plus everything
/// in `response_ids` before the span — so masked-out tool tokens only ever
/// appear in a prompt prefix and never receive loss.
pub fn tokens_record_to_pairs(
    prompt_ids: &[u32],
    response_ids: &[u32],
    response_mask: &[u8],
) -> Vec<(Vec<u32>, Vec<u32>)> {
    let mut pairs = Vec::new();
    let n = response_ids.len().min(response_mask.len());
    let mut i = 0;
    while i < n {
        if response_mask[i] != 1 {
            i += 1;
            continue;
        }
        let span_start = i;
        while i < n && response_mask[i] == 1 {
            i += 1;
        }
        let completion = response_ids[span_start..i].to_vec();
        if completion.is_empty() {
            continue;
        }
        let prefix: Vec<u32> = prompt_ids
            .iter()
            .copied()
            .chain(response_ids[..span_start].iter().copied())
            .collect();
        pairs.push((prefix, completion));
    }
    pairs
}

/// Held-out eval pass-rate: fraction of tasks solved (`passed / total`), `0.0`
/// when no tasks. Backend-independent so the aggregation is unit-testable
/// without a GPU; the cuda-gated `AgentEvalReport::pass_rate` delegates here.
pub fn pass_rate(passed: usize, total: usize) -> f32 {
    if total == 0 {
        0.0
    } else {
        passed as f32 / total as f32
    }
}

#[cfg(test)]
mod tests {
    use super::{pass_rate, tokens_record_to_pairs};

    #[test]
    fn pass_rate_aggregates_held_out_pass_fail() {
        assert_eq!(pass_rate(0, 0), 0.0, "no tasks -> 0.0 (no div-by-zero)");
        assert_eq!(pass_rate(0, 2), 0.0, "all fail -> 0.0");
        assert_eq!(pass_rate(2, 2), 1.0, "all pass -> 1.0");
        assert!((pass_rate(1, 2) - 0.5).abs() < 1e-6, "1/2 -> 0.5");
        assert!((pass_rate(1, 3) - 0.333_333).abs() < 1e-5, "1/3 -> ~0.333");
    }

    #[test]
    fn pairs_mask_out_tool_tokens() {
        // prompt = [1,2]; response interleaves LLM (mask 1) and tool (mask 0):
        // [10,11] LLM, [20,21] tool, [30] LLM.
        let prompt = [1u32, 2];
        let resp = [10u32, 11, 20, 21, 30];
        let mask = [1u8, 1, 0, 0, 1];
        let pairs = tokens_record_to_pairs(&prompt, &resp, &mask);
        assert_eq!(pairs.len(), 2, "two LLM spans -> two pairs");
        // span 1: completion = the first LLM run, prefix = prompt only.
        assert_eq!(pairs[0].0, vec![1, 2]);
        assert_eq!(pairs[0].1, vec![10, 11]);
        // span 2: completion = trailing LLM token; prefix = prompt + ALL prior
        // response tokens (incl. the masked tool tokens) so tool tokens are
        // context, never a loss target.
        assert_eq!(pairs[1].0, vec![1, 2, 10, 11, 20, 21]);
        assert_eq!(pairs[1].1, vec![30]);
    }

    #[test]
    fn no_llm_tokens_yields_no_pairs() {
        let pairs = tokens_record_to_pairs(&[1, 2], &[5, 6], &[0, 0]);
        assert!(pairs.is_empty());
    }

    #[test]
    fn ragged_mask_is_clamped() {
        // mask shorter than response: only the covered prefix is considered.
        let pairs = tokens_record_to_pairs(&[1], &[7, 8, 9], &[1, 1]);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].1, vec![7, 8]);
    }
}

#[cfg(feature = "cuda")]
mod cuda_rollout {
    use std::path::PathBuf;

    use agent::{AgentSession, AgentSettings, ToolExecutor, ToolPolicy};
    use anyhow::{Context, Result};
    use chat::{ToolCall, ToolDefinition};
    use serde_json::json;

    use crate::aopd_profile;
    use crate::infer_student::InferStudent;
    use crate::sandbox::{
        SandboxToolExecutor, boot_workdir, diff_workdir, reset_workdir, score_workdir,
    };
    use crate::swe_dataset::{SweTask, agent_system_prompt, agent_user_prompt};

    /// A no-op [`ToolPolicy`]: the trained student emits proper tool calls, so
    /// the deterministic recovery / repair hooks stay off (all trait defaults).
    struct NoPolicy;
    impl ToolPolicy for NoPolicy {}

    /// The 4 atomic coding tools (read / write / replace / bash) —
    /// `builtin_tools()` minus `python`.
    fn coding_tools() -> Vec<ToolDefinition> {
        tools::builtin_tools()
            .into_iter()
            .filter(|t| t.name != "python")
            .map(|t| t.to_definition())
            .collect()
    }

    /// Per-round hyperparameters for the agent-OPD loop.
    #[derive(Clone, Debug)]
    pub struct AgentOpdConfig {
        pub rounds: usize,
        /// Agent rollouts per task per round (best-of-N).
        pub samples_per_prompt: usize,
        /// Max agent turns (tool sub-turns) per rollout.
        pub max_turns: usize,
        /// Max tokens the engine generates per sub-turn.
        pub max_tokens: usize,
        /// Sampling temperature for rollout diversity.
        pub temperature: f32,
        /// Thinking soft-switch for ALL rollouts (train + rescue + eval):
        /// think-on everywhere or off everywhere, never mixed — the
        /// 2026-06-20 precedent (no `<think>`-span masking; the reasoning
        /// transfer is the point).
        pub think: bool,
        /// Budget-asymmetric self-rescue: tasks with ZERO accepted samples
        /// get this many extra rollouts at `rescue_max_tokens`. 0 = off.
        /// This is the bootstrap for regimes where plain rejection sampling
        /// starves (real-repo 0-accept wall).
        pub rescue_samples: usize,
        /// Per-sub-turn token budget for rescue rollouts (thinking needs
        /// room; the normal `max_tokens` stays cheap).
        pub rescue_max_tokens: usize,
        /// Turn budget for rescue rollouts. The binding constraint on hard
        /// real-repo tasks is turns-to-first-edit: at `max_turns`=8 they burn
        /// every turn on on-target investigation and never reach the edit phase
        /// (empty diff → unscored → 0-accept), yet the SAME 27B teacher edits
        /// and passes them at 20 turns (measured A/B 2026-07-08). Rescue tuned
        /// only `max_tokens` before — the wrong axis.
        pub rescue_max_turns: usize,
        /// Legacy CE micro-batch size (inert under masked single-trajectory CE,
        /// which forwards one trajectory per window; retained for CLI back-compat).
        pub writeback_batch: usize,
        /// Optional cap on accepted trajectories trained per round.
        pub writeback_cap: Option<usize>,
        /// Root dir under which per-task sandboxes are staged.
        pub work_root: PathBuf,
        /// `PYTHONPATH` (sandbox-relative) for bash + test runs.
        pub pythonpath: Option<String>,
        /// Bash tool timeout (seconds).
        pub bash_timeout_secs: u64,
        /// Test-run timeout (seconds).
        pub test_timeout_secs: u64,
    }

    /// One round's roll-up.
    #[derive(Clone, Debug, Default, serde::Serialize)]
    pub struct AgentRoundReport {
        pub tasks: usize,
        pub rollouts: usize,
        pub passed: usize,
        pub distinct_passed: usize,
        /// Passing rollouts whose engine emitted no token record (skipped).
        pub no_token_record: usize,
        pub trained_pairs: usize,
        pub mean_train_loss: f32,
        /// Rescue-pass rollouts run on 0-accept tasks (and how many passed).
        pub rescue_rollouts: usize,
        pub rescue_passed: usize,
        // --- T2: observed-only rollout aggregates (never affect accept/train) ---
        /// Summed `AgentTurnResult` counts/timings over ALL rollouts this round.
        pub sum_completion_tokens: u64,
        pub sum_prompt_tokens: u64,
        pub sum_tool_calls: u64,
        pub sum_rollout_secs: f64,
        /// Distinct train tasks with ≥1 passing rollout (train pass-rate numerator).
        pub train_tasks_passed: usize,
        /// Rollout terminal-state histogram (Debug name → count).
        pub terminal_state_counts: std::collections::BTreeMap<String, u32>,
    }

    /// Per-task held-out eval outcome: did the current student (greedy, no
    /// training) produce a patch that passes the hidden tests?
    #[derive(Clone, Debug)]
    pub struct AgentEvalTaskResult {
        pub instance_id: String,
        /// True iff the rollout produced a non-empty diff AND `score_workdir`
        /// reported the hidden tests passing.
        pub passed: bool,
        /// True iff the rollout edited the tree at all (diff non-empty).
        pub edited: bool,
        /// Last line of the test log / a short note (error / no-edit).
        pub note: String,
    }

    /// Held-out eval roll-up: per-task pass/fail plus the aggregate pass-rate.
    #[derive(Clone, Debug, Default)]
    pub struct AgentEvalReport {
        pub tasks: Vec<AgentEvalTaskResult>,
    }

    impl AgentEvalReport {
        pub fn passed(&self) -> usize {
            self.tasks.iter().filter(|t| t.passed).count()
        }
        pub fn edited(&self) -> usize {
            self.tasks.iter().filter(|t| t.edited).count()
        }
        /// Fraction of held-out tasks the student solved (0.0 when no tasks).
        pub fn pass_rate(&self) -> f32 {
            super::pass_rate(self.passed(), self.tasks.len())
        }
    }

    /// EVAL-ONLY pass over a HELD-OUT task set: drive the SAME agent rollout +
    /// `score_workdir` harness with the current student, but train NOTHING — no
    /// writeback, no optimizer step, ONE greedy/low-temp sample per task. The
    /// reward is the same execution signal as the train loop (hidden tests on
    /// `git diff`), so the aggregate `pass_rate` measures whether the model
    /// actually got better at coding (baseline → round-N generalization), not
    /// just train loss.
    ///
    /// `eval_temperature` should be `0.0` (greedy) or a small value; the harness
    /// reads the engine in a `&mut` lock exactly like the train rollout but never
    /// calls back into the optimizer.
    pub fn run_agentic_opd_eval(
        student: &InferStudent,
        tasks: &[(SweTask, PathBuf)],
        cfg: &AgentOpdConfig,
        eval_temperature: f32,
        label: &str,
    ) -> Result<AgentEvalReport> {
        let tool_defs = coding_tools();
        let policy = NoPolicy;
        let settings = AgentSettings {
            max_turns: cfg.max_turns,
            max_tokens: cfg.max_tokens,
            temperature: eval_temperature,
        };

        let mut report = AgentEvalReport::default();
        for (task, staged_tree) in tasks {
            let workdir = boot_workdir(
                &cfg.work_root,
                &task.instance_id,
                staged_tree,
                task.before_repo_set_cmd.as_deref(),
            )
            .with_context(|| format!("boot eval sandbox for {}", task.instance_id))?;
            let executor = SandboxToolExecutor::new(
                workdir.clone(),
                cfg.bash_timeout_secs,
                cfg.pythonpath.clone(),
            );
            let overview = executor.execute(&ToolCall::new(
                "bash",
                json!({ "command": "ls && echo '---' && git log -1 --oneline 2>/dev/null" }),
            ));
            let user_prompt = format!(
                "{}\n\nRepo layout (cwd = repo root):\n{}",
                agent_user_prompt(task, cfg.think),
                overview
            );

            // Single greedy rollout per held-out task (best-of-N would inflate the
            // pass-rate vs the production single-shot the eval is meant to predict).
            reset_workdir(&workdir)
                .with_context(|| format!("reset eval sandbox for {}", task.instance_id))?;
            let mut session =
                AgentSession::with_system_prompt(agent_system_prompt(task, cfg.think));
            let result = {
                let mut guard = student
                    .engine()
                    .lock()
                    .map_err(|e| anyhow::anyhow!("eval engine lock poisoned: {e}"))?;
                session.run_turn(
                    &mut *guard,
                    &user_prompt,
                    &tool_defs,
                    &executor,
                    &policy,
                    settings,
                )
            };
            let (passed, edited, note) = match result {
                Err(e) => (false, false, format!("rollout error: {e}")),
                Ok(result) => {
                    let diff = diff_workdir(&workdir).unwrap_or_default();
                    if diff.trim().is_empty() {
                        (
                            false,
                            false,
                            format!("no edits (turns={})", result.tool_calls_executed),
                        )
                    } else {
                        match score_workdir(
                            &workdir,
                            &task.test_patch,
                            &task.fail_to_pass(),
                            cfg.pythonpath.as_deref(),
                            cfg.test_timeout_secs,
                        ) {
                            Ok((passed, log)) => {
                                (passed, true, log.lines().last().unwrap_or("").to_string())
                            }
                            Err(e) => (false, true, format!("score error: {e}")),
                        }
                    }
                }
            };
            eprintln!(
                "[agent-opd-eval {label}] {} passed={passed} edited={edited} :: {note}",
                task.instance_id
            );
            report.tasks.push(AgentEvalTaskResult {
                instance_id: task.instance_id.clone(),
                passed,
                edited,
                note,
            });
        }

        eprintln!(
            "[agent-opd-eval {label}] held-out pass_rate={:.4} ({}/{} tasks, {} edited)",
            report.pass_rate(),
            report.passed(),
            report.tasks.len(),
            report.edited(),
        );
        Ok(report)
    }

    /// Run the agent-OPD loop. `tasks` pairs each [`SweTask`] with the directory
    /// holding its repo already checked out at `base_commit` (the staged tree).
    /// `train_on_accepted` performs ONE masked-CE writeback step on a single
    /// accepted trajectory `(prompt_ids, response_ids, response_mask)` and
    /// returns the mean CE per masked token — the caller wires
    /// [`crate::opd::masked_writeback_ce_step`].
    ///
    /// Runs ONE round (mirroring `run_rubric_rounds` called with `rounds=1`):
    /// the caller loops over rounds and performs the LoRA→rollout-engine sync
    /// between calls, because the sync and `train_on_accepted` both need
    /// `&mut store` and cannot be live simultaneously.
    pub fn run_agentic_opd_round<T>(
        student: &InferStudent,
        tasks: &[(SweTask, PathBuf)],
        cfg: &AgentOpdConfig,
        round: usize,
        mut train_on_accepted: T,
    ) -> Result<AgentRoundReport>
    where
        T: FnMut(&[u32], &[u32], &[u8]) -> Result<f32>,
    {
        let tool_defs = coding_tools();
        let policy = NoPolicy;
        let settings = AgentSettings {
            max_turns: cfg.max_turns,
            max_tokens: cfg.max_tokens,
            temperature: cfg.temperature,
        };

        {
            let mut report = AgentRoundReport::default();
            // One masked-CE writeback per accepted trajectory: collect the
            // verl-style record `(prompt_ids, response_ids, response_mask)` and
            // train each whole trajectory once (windowed), instead of exploding
            // into O(N²) per-turn `(prefix, completion)` pairs.
            let mut accepted_trajectories: Vec<(Vec<u32>, Vec<u32>, Vec<u8>)> = Vec::new();
            let mut loss_sum = 0.0f32;
            let mut loss_steps = 0usize;

            eprintln!(
                "[dbg-opd] run_agentic_opd_round entered, {} tasks",
                tasks.len()
            );
            for (task, staged_tree) in tasks {
                eprintln!("[dbg-opd] boot_workdir for {}", task.instance_id);
                report.tasks += 1;
                let workdir = aopd_profile::time_try("sandbox_boot", aopd_profile::WALL, || {
                    boot_workdir(
                        &cfg.work_root,
                        &task.instance_id,
                        staged_tree,
                        task.before_repo_set_cmd.as_deref(),
                    )
                })
                .with_context(|| format!("boot sandbox for {}", task.instance_id))?;
                eprintln!("[dbg-opd] boot_workdir done for {}", task.instance_id);
                let executor = SandboxToolExecutor::new(
                    workdir.clone(),
                    cfg.bash_timeout_secs,
                    cfg.pythonpath.clone(),
                );

                // Env-info reflecting the SANDBOX (not the host): a one-shot repo
                // overview prepended to the first user turn so the agent doesn't
                // burn turns rediscovering the layout.
                eprintln!(
                    "[dbg-opd] executor.execute(bash ls) for {} spawner_socket={:?}",
                    task.instance_id,
                    std::env::var("ARLE_SPAWNER_SOCKET").ok()
                );
                let overview = aopd_profile::time("sandbox_overview", aopd_profile::WALL, || {
                    executor.execute(&ToolCall::new(
                        "bash",
                        json!({ "command": "ls && echo '---' && git log -1 --oneline 2>/dev/null" }),
                    ))
                });
                eprintln!("[dbg-opd] overview done len={}", overview.len());
                let user_prompt = format!(
                    "{}\n\nRepo layout (cwd = repo root):\n{}",
                    agent_user_prompt(task, cfg.think),
                    overview
                );

                let mut distinct_passed_this_task = 0usize;
                // Normal samples first; a rescue block below re-samples 0-accept
                // tasks at the bigger token budget.
                let rescue_settings = AgentSettings {
                    max_tokens: cfg.rescue_max_tokens,
                    max_turns: cfg.rescue_max_turns,
                    ..settings
                };
                let total_samples = cfg.samples_per_prompt + cfg.rescue_samples;
                for sample in 0..total_samples {
                    let rescue = sample >= cfg.samples_per_prompt;
                    if rescue && distinct_passed_this_task > 0 {
                        break; // rescue only fires on 0-accept tasks
                    }
                    if rescue {
                        report.rescue_rollouts += 1;
                    }
                    eprintln!("[dbg-opd] sample={sample} rescue={rescue} reset_workdir");
                    aopd_profile::time_try("sandbox_reset", aopd_profile::WALL, || {
                        reset_workdir(&workdir)
                    })
                    .with_context(|| format!("reset sandbox for {}", task.instance_id))?;
                    eprintln!("[dbg-opd] sample={sample} run_turn START");
                    let mut session =
                        AgentSession::with_system_prompt(agent_system_prompt(task, cfg.think));
                    let rollout_t0 = std::time::Instant::now();
                    let result = {
                        let mut guard = student
                            .engine()
                            .lock()
                            .map_err(|e| anyhow::anyhow!("rollout engine lock poisoned: {e}"))?;
                        eprintln!("[dbg-opd] sample={sample} engine locked, calling run_turn");
                        session.run_turn(
                            &mut *guard,
                            &user_prompt,
                            &tool_defs,
                            &executor,
                            &policy,
                            if rescue { rescue_settings } else { settings },
                        )
                    };
                    let rollout_wall = rollout_t0.elapsed().as_secs_f64();
                    report.rollouts += 1;
                    let result = match result {
                        Ok(r) => r,
                        Err(e) => {
                            // Failed rollout wall is unattributable to a sub-turn; bill it
                            // to decode (the engine call is where it died).
                            aopd_profile::record("rollout_decode", aopd_profile::GPU, rollout_wall);
                            eprintln!(
                                "[agent-opd] {} sample {sample}: rollout error: {e}",
                                task.instance_id
                            );
                            continue;
                        }
                    };
                    // Split the rollout wall: per-sub-turn engine decode (GPU-bound,
                    // captured by the agent loop) vs the residual (tool exec — bash /
                    // pytest via the sandbox executor, pure host wall).
                    let decode_secs: f64 = result.sub_turns.iter().map(|st| st.decode_secs).sum();
                    aopd_profile::record("rollout_decode", aopd_profile::GPU, decode_secs);
                    aopd_profile::record(
                        "rollout_tool_exec",
                        aopd_profile::WALL,
                        (rollout_wall - decode_secs).max(0.0),
                    );

                    // T2: observe rollout counts/timings BEFORE the accept path drops
                    // them. Pure aggregation — does not gate accept/train.
                    report.sum_completion_tokens += result.completion_tokens;
                    report.sum_prompt_tokens += result.prompt_tokens;
                    report.sum_tool_calls += result.tool_calls_executed as u64;
                    report.sum_rollout_secs += result.wall_secs;
                    *report
                        .terminal_state_counts
                        .entry(format!("{:?}", result.terminal_state))
                        .or_insert(0) += 1;

                    // DEBUG (temporary): decode the FULL trajectory of sample 0 — every
                    // sub-turn's action — to see whether the student locates the bug and
                    // attempts an edit, or just explores. Case-as-fact, not a guess.
                    if sample == 0 {
                        eprintln!(
                            "[agent-opd-debug] {} terminal={:?} turns={} sub_turns={} final={:?}",
                            task.instance_id,
                            result.terminal_state,
                            result.tool_calls_executed,
                            result.sub_turns.len(),
                            result
                                .text
                                .replace('\n', " ")
                                .chars()
                                .take(220)
                                .collect::<String>(),
                        );
                        for st in &result.sub_turns {
                            eprintln!(
                                "[agent-opd-debug]  turn{} fin={} :: {}",
                                st.index,
                                st.finish_reason,
                                st.completion_text
                                    .replace('\n', " ")
                                    .chars()
                                    .take(300)
                                    .collect::<String>(),
                            );
                        }
                    }

                    let diff = aopd_profile::time("rollout_diff", aopd_profile::WALL, || {
                        diff_workdir(&workdir).unwrap_or_default()
                    });
                    if diff.trim().is_empty() {
                        eprintln!(
                            "[agent-opd] {} sample {sample}: no edits (turns={}, {:?})",
                            task.instance_id, result.tool_calls_executed, result.terminal_state
                        );
                        continue;
                    }

                    let passed = match aopd_profile::time(
                        "score_pytest",
                        aopd_profile::WALL,
                        || {
                            score_workdir(
                                &workdir,
                                &task.test_patch,
                                &task.fail_to_pass(),
                                cfg.pythonpath.as_deref(),
                                cfg.test_timeout_secs,
                            )
                        },
                    ) {
                        Ok((passed, log)) => {
                            eprintln!(
                                "[agent-opd] {} sample {sample}: passed={passed} (turns={}) :: {}",
                                task.instance_id,
                                result.tool_calls_executed,
                                log.lines().last().unwrap_or("")
                            );
                            passed
                        }
                        Err(e) => {
                            eprintln!(
                                "[agent-opd] {} sample {sample}: score error: {e}",
                                task.instance_id
                            );
                            false
                        }
                    };
                    if !passed {
                        continue;
                    }
                    report.passed += 1;
                    if rescue {
                        report.rescue_passed += 1;
                    }
                    distinct_passed_this_task += 1;

                    match &result.tokens {
                        Some(tok) => {
                            accepted_trajectories.push((
                                tok.prompt_ids.clone(),
                                tok.response_ids.clone(),
                                tok.response_mask.clone(),
                            ));
                        }
                        None => {
                            report.no_token_record += 1;
                            eprintln!(
                                "[agent-opd] {} sample {sample}: PASSED but no token record — \
                                 skipped writeback (engine did not populate tokens)",
                                task.instance_id
                            );
                        }
                    }
                }
                if distinct_passed_this_task > 0 {
                    report.distinct_passed += 1;
                    report.train_tasks_passed += 1; // distinct train task with ≥1 pass
                }
            }

            // Optional cap on accepted trajectories trained this round.
            if let Some(cap) = cfg.writeback_cap {
                accepted_trajectories.truncate(cap);
            }
            report.trained_pairs = accepted_trajectories.len();

            for (prompt_ids, response_ids, response_mask) in &accepted_trajectories {
                // `train_on_accepted` = scratch/KV release + masked-CE forward +
                // backward + optimizer step (one accepted trajectory). GPU-bound; the
                // loss D2H forces a device sync so the wall is GPU-inclusive.
                let loss = aopd_profile::time_try("writeback", aopd_profile::GPU, || {
                    train_on_accepted(prompt_ids, response_ids, response_mask)
                })
                .with_context(|| format!("masked CE writeback (round {round})"))?;
                loss_sum += loss;
                loss_steps += 1;
            }
            report.mean_train_loss = if loss_steps > 0 {
                loss_sum / loss_steps as f32
            } else {
                0.0
            };

            eprintln!(
                "[agent-opd] round {round}: tasks={} rollouts={} passed={} distinct={} \
                 no_token_record={} trained_pairs={} mean_loss={:.4} rescue_rollouts={} \
                 rescue_passed={}",
                report.tasks,
                report.rollouts,
                report.passed,
                report.distinct_passed,
                report.no_token_record,
                report.trained_pairs,
                report.mean_train_loss,
                report.rescue_rollouts,
                report.rescue_passed
            );
            Ok(report)
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda_rollout::{
    AgentEvalReport, AgentEvalTaskResult, AgentOpdConfig, AgentRoundReport, run_agentic_opd_eval,
    run_agentic_opd_round,
};
