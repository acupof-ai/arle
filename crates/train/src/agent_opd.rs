//! Agentic on-policy distillation (agent-OPD) rollout driver.
//!
//! The student LLM (in-process [`InferStudent`] rollout engine) drives the
//! [`agent::AgentSession`] tool loop — read / write / replace / bash — against a
//! per-task repo sandbox. The student EDITS the tree directly; the reward is
//! execution: `git diff` is the candidate patch, the hidden `test_patch` +
//! `fail_to_pass` tests are run, exit-0 = pass.
//!
//! Passing trajectories are written back as cross-entropy targets through the
//! SAME path as rubric-OPD ([`crate::opd::rubric_writeback_ce_step_batched`]):
//! each accepted trajectory's verl-style `TokensRecord` (LLM tokens masked from
//! tool/environment tokens via `response_mask`) is exploded into per-assistant
//! `(context_prefix, llm_span)` token pairs. Tool-result tokens always land in
//! a later pair's prefix, so they never receive loss — the student never learns
//! to hallucinate tool output. Only the rollout front-end + the reward differ
//! from rubric-OPD; the writeback / CE / optimizer path is reused unchanged.
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
        let mut prefix = Vec::with_capacity(prompt_ids.len() + span_start);
        prefix.extend_from_slice(prompt_ids);
        prefix.extend_from_slice(&response_ids[..span_start]);
        pairs.push((prefix, completion));
    }
    pairs
}

#[cfg(test)]
mod tests {
    use super::tokens_record_to_pairs;

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

    use super::tokens_record_to_pairs;
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
        /// CE micro-batch size for writeback.
        pub writeback_batch: usize,
        /// Optional cap on accepted `(prompt, completion)` pairs trained per round.
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
    #[derive(Clone, Debug, Default)]
    pub struct AgentRoundReport {
        pub tasks: usize,
        pub rollouts: usize,
        pub passed: usize,
        pub distinct_passed: usize,
        /// Passing rollouts whose engine emitted no token record (skipped).
        pub no_token_record: usize,
        pub trained_pairs: usize,
        pub mean_train_loss: f32,
    }

    /// Run the agent-OPD loop. `tasks` pairs each [`SweTask`] with the directory
    /// holding its repo already checked out at `base_commit` (the staged tree).
    /// `train_on_accepted` performs one CE micro-batch step on `(prompt,
    /// completion)` pairs and returns the mean loss — the caller wires
    /// [`crate::opd::rubric_writeback_ce_step_batched`].
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
        T: FnMut(&[(Vec<u32>, Vec<u32>)]) -> Result<f32>,
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
            let mut accepted_pairs: Vec<(Vec<u32>, Vec<u32>)> = Vec::new();
            let mut loss_sum = 0.0f32;
            let mut loss_steps = 0usize;

            for (task, staged_tree) in tasks {
                report.tasks += 1;
                let workdir = boot_workdir(
                    &cfg.work_root,
                    &task.instance_id,
                    staged_tree,
                    task.before_repo_set_cmd.as_deref(),
                )
                .with_context(|| format!("boot sandbox for {}", task.instance_id))?;
                let executor = SandboxToolExecutor::new(
                    workdir.clone(),
                    cfg.bash_timeout_secs,
                    cfg.pythonpath.clone(),
                );

                // Env-info reflecting the SANDBOX (not the host): a one-shot repo
                // overview prepended to the first user turn so the agent doesn't
                // burn turns rediscovering the layout.
                let overview = executor.execute(&ToolCall::new(
                    "bash",
                    json!({ "command": "ls && echo '---' && git log -1 --oneline 2>/dev/null" }),
                ));
                let user_prompt = format!(
                    "{}\n\nRepo layout (cwd = repo root):\n{}",
                    agent_user_prompt(task),
                    overview
                );

                let mut distinct_passed_this_task = 0usize;
                for sample in 0..cfg.samples_per_prompt {
                    reset_workdir(&workdir)
                        .with_context(|| format!("reset sandbox for {}", task.instance_id))?;

                    let mut session = AgentSession::with_system_prompt(agent_system_prompt(task));
                    let result = {
                        let mut guard = student
                            .engine()
                            .lock()
                            .map_err(|e| anyhow::anyhow!("rollout engine lock poisoned: {e}"))?;
                        session.run_turn(
                            &mut *guard,
                            &user_prompt,
                            &tool_defs,
                            &executor,
                            &policy,
                            settings,
                        )
                    };
                    report.rollouts += 1;
                    let result = match result {
                        Ok(r) => r,
                        Err(e) => {
                            eprintln!(
                                "[agent-opd] {} sample {sample}: rollout error: {e}",
                                task.instance_id
                            );
                            continue;
                        }
                    };

                    let diff = diff_workdir(&workdir).unwrap_or_default();
                    if diff.trim().is_empty() {
                        eprintln!(
                            "[agent-opd] {} sample {sample}: no edits (turns={}, {:?})",
                            task.instance_id, result.tool_calls_executed, result.terminal_state
                        );
                        continue;
                    }

                    let passed = match score_workdir(
                        &workdir,
                        &task.test_patch,
                        &task.fail_to_pass(),
                        cfg.pythonpath.as_deref(),
                        cfg.test_timeout_secs,
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
                    distinct_passed_this_task += 1;

                    match &result.tokens {
                        Some(tok) => {
                            let pairs = tokens_record_to_pairs(
                                &tok.prompt_ids,
                                &tok.response_ids,
                                &tok.response_mask,
                            );
                            accepted_pairs.extend(pairs);
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
                }
            }

            // Optional cap on pairs trained this round.
            if let Some(cap) = cfg.writeback_cap {
                accepted_pairs.truncate(cap);
            }
            report.trained_pairs = accepted_pairs.len();

            for chunk in accepted_pairs.chunks(cfg.writeback_batch.max(1)) {
                let loss = train_on_accepted(chunk)
                    .with_context(|| format!("CE writeback (round {round})"))?;
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
                 no_token_record={} trained_pairs={} mean_loss={:.4}",
                report.tasks,
                report.rollouts,
                report.passed,
                report.distinct_passed,
                report.no_token_record,
                report.trained_pairs,
                report.mean_train_loss
            );
            Ok(report)
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda_rollout::{AgentOpdConfig, AgentRoundReport, run_agentic_opd_round};
