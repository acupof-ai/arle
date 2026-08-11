//! Claude-Code-as-harness rollout driver: sandbox boot → `claude -p` against
//! the in-process serve → pytest scoring → in-memory cc-convert. One scoped
//! thread per sample (cc → score, so a sample scores the moment its cc exits);
//! the NEXT group's boots run on background threads, overlapping the current
//! group's rollout/train (CPU-only, staleness-free). Prompt + cc invocation
//! ported verbatim from `scripts/cc_swe_baseline.py` (plan
//! docs/plans/2026-07-16-agent-rl-unified-infra.md §P2).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, sync_channel};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow};

use crate::cc_convert::{CcRecord, CcWindow, convert_cc_dumps};
use crate::sandbox::{boot_workdir, diff_workdir, run_captured, score_workdir};
use crate::swe_dataset::SweTask;

/// TYPICAL cc SWE-session tokens: sizes the per-stream KV budget and is the
/// rollout generation budget (Dr.GRPO's fixed normalizer). NOT a session cap —
/// long sessions run to [`CC_MAX_SESSION_TOKENS`]; the engine preempts under
/// KV pressure (#162).
pub const CC_SESSION_TOKENS: usize = 22_000;

/// A cc session must be schedulable to at least this many tokens (long-horizon
/// requirement, ckl 2026-07-17): the KV pool is sized to fit ≥1 such session
/// and the serve request caps derive from the pool, never from the typical size.
pub const CC_MAX_SESSION_TOKENS: usize = 200_000;

pub struct CcHarness {
    pub work_root: PathBuf,
    pub dump_dir: PathBuf,
    /// One per rollout engine; samples spread round-robin.
    pub base_urls: Vec<String>,
    pub model_id: String,
    pub cc_timeout_secs: u64,
    pub test_timeout_secs: u64,
    pub pythonpath: Option<String>,
    pub reward_shape: RewardShape,
    pub tokenizer: tokenizers::Tokenizer,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum RewardShape {
    #[default]
    Dense,
    Binary,
    /// 1.0 on full pass, 0.0 on timeout/error, else a small fraction that only
    /// orders attempts inside an all-failing group.
    Anchored,
}

/// A partial's reward tops out here, well below a full pass (1.0), so the
/// pass/fail split always dominates and the fraction only breaks ties.
const REWARD_DENSE_WEIGHT: f32 = 0.3;

impl RewardShape {
    /// Shape the raw pass-fraction `f`. `errored` marks a timeout or harness
    /// failure — a budget artifact, not a signal to learn from.
    #[must_use]
    pub fn apply(self, f: f32, errored: bool) -> f32 {
        match self {
            RewardShape::Dense => f,
            RewardShape::Binary => f32::from(f >= 1.0),
            RewardShape::Anchored if errored => 0.0,
            RewardShape::Anchored if f >= 1.0 => 1.0,
            RewardShape::Anchored => REWARD_DENSE_WEIGHT * f,
        }
    }
}

pub fn load_tokenizer(path: &Path) -> Result<tokenizers::Tokenizer> {
    tokenizers::Tokenizer::from_file(path)
        .map_err(|err| anyhow!("load tokenizer {}: {err}", path.display()))
}

#[derive(Clone, Debug)]
pub struct ScoredSample {
    pub task_id: String,
    pub sample: usize,
    /// Pass ⇔ `reward >= 1.0` (all shapes agree on the pass point).
    pub reward: f32,
    pub edited: bool,
    /// Timeout or scoring failure — budget artifact, marks trajectory truncated.
    pub errored: bool,
    pub note: String,
    pub t_start_ms: u64,
    pub t_end_ms: u64,
    pub cc_turns: Option<u64>,
    pub cc_input_tokens: Option<u64>,
    pub cc_output_tokens: Option<u64>,
}

impl ScoredSample {
    #[must_use]
    pub fn passed(&self) -> bool {
        self.reward >= 1.0
    }
}

pub struct CcGroup {
    pub task_id: String,
    pub samples: Vec<ScoredSample>,
    pub records: Vec<CcRecord>,
}

#[must_use]
pub fn zero_variance(samples: &[ScoredSample]) -> bool {
    samples.windows(2).all(|w| w[0].reward == w[1].reward)
}

/// DAPO dynamic sampling: boot replacement groups until `target` train a
/// nonzero batch. Terminates on the launch cap or token budget.
pub struct RefillBudget {
    pub target: usize,
    pub max_launches: usize,
    pub token_budget: u64,
    pub effective: usize,
    pub discarded: usize,
    pub tokens: u64,
}

impl RefillBudget {
    #[must_use]
    pub fn new(target: usize, max_launches: usize, token_budget: u64) -> Self {
        Self {
            target,
            max_launches: max_launches.max(target),
            token_budget,
            effective: 0,
            discarded: 0,
            tokens: 0,
        }
    }

    pub fn complete(&mut self, committed: usize, trained: bool, tokens: u64) -> bool {
        self.tokens = self.tokens.saturating_add(tokens);
        if trained {
            self.effective += 1;
        } else {
            self.discarded += 1;
        }
        !trained
            && self.effective < self.target
            && committed < self.max_launches
            && (self.token_budget == 0 || self.tokens < self.token_budget)
    }
}

pub struct GroupAssembler {
    k: usize,
    pending: HashMap<String, Vec<ScoredSample>>,
}

impl GroupAssembler {
    #[must_use]
    pub fn new(k: usize) -> Self {
        Self {
            k: k.max(1),
            pending: HashMap::new(),
        }
    }

    pub fn add(&mut self, sample: ScoredSample) -> Option<Vec<ScoredSample>> {
        let task_id = sample.task_id.clone();
        let slot = self.pending.entry(task_id.clone()).or_default();
        slot.push(sample);
        (slot.len() >= self.k).then(|| {
            let mut samples = self.pending.remove(&task_id).expect("slot inserted above");
            samples.sort_unstable_by_key(|s| s.sample);
            samples
        })
    }
}

pub struct BootedGroup {
    task: Arc<SweTask>,
    k: usize,
    /// Dump attribution keys on the request's model tag, so concurrent groups
    /// (`--prompts-per-update` > 1) must not share one.
    nonce: u64,
    rx: Receiver<Result<(usize, PathBuf)>>,
}

impl CcHarness {
    pub fn boot_group(&self, task: &Arc<SweTask>, staged_tree: &Path, k: usize) -> BootedGroup {
        static GROUP_NONCE: AtomicUsize = AtomicUsize::new(0);
        let k = k.max(1);
        let (tx, rx) = sync_channel(k);
        let next = Arc::new(AtomicUsize::new(0));
        for _ in 0..k.min(2) {
            let (tx, next) = (tx.clone(), Arc::clone(&next));
            let (task, staged, root) = (
                Arc::clone(task),
                staged_tree.to_owned(),
                self.work_root.clone(),
            );
            std::thread::spawn(move || {
                loop {
                    let sample = next.fetch_add(1, Ordering::Relaxed);
                    if sample >= k {
                        break;
                    }
                    let name = format!("{}#{sample}", task.instance_id);
                    let booted =
                        boot_workdir(&root, &name, &staged, task.before_repo_set_cmd.as_deref())
                            .with_context(|| format!("boot cc sandbox {name}"))
                            .map(|workdir| (sample, workdir));
                    // Deliver the error before stopping — run_group propagates it.
                    let failed = booted.is_err();
                    if tx.send(booted).is_err() || failed {
                        break;
                    }
                }
            });
        }
        BootedGroup {
            task: Arc::clone(task),
            k,
            nonce: GROUP_NONCE.fetch_add(1, Ordering::Relaxed) as u64,
            rx,
        }
    }

    pub fn run_group(&self, booted: BootedGroup) -> Result<CcGroup> {
        let mut samples = std::thread::scope(|scope| -> Result<Vec<ScoredSample>> {
            let (tx, rx) = sync_channel(booted.k);
            for _ in 0..booted.k {
                let (sample, workdir) = booted.rx.recv().context("cc boot thread died")??;
                let (tx, task) = (tx.clone(), &booted.task);
                scope.spawn(move || {
                    let _ = tx.send(self.run_sample(task, booted.nonce, sample, &workdir));
                });
            }
            drop(tx);
            let mut assembler = GroupAssembler::new(booted.k);
            loop {
                let s = rx.recv().context("cc sample thread died")?;
                eprintln!(
                    "[cc-harness] {}#{}: passed={} reward={:.3} edited={} turns={:?} wall={:.1}s :: {}",
                    s.task_id,
                    s.sample,
                    s.passed(),
                    s.reward,
                    s.edited,
                    s.cc_turns,
                    s.t_end_ms.saturating_sub(s.t_start_ms) as f64 / 1000.0,
                    s.note,
                );
                if let Some(samples) = assembler.add(s) {
                    break Ok(samples);
                }
            }
        })?;

        let windows: Vec<CcWindow> = samples
            .iter()
            .map(|s| CcWindow {
                label: format!("{}#{}", s.task_id, s.sample),
                t_start_ms: s.t_start_ms,
                t_end_ms: s.t_end_ms,
                reward: s.reward,
                errored: s.errored,
                model: Some(sample_model(&self.model_id, booted.nonce, s.sample)),
            })
            .collect();
        let records = convert_cc_dumps(&self.dump_dir, &self.tokenizer, &windows)?;
        // Backfill tokens for timed-out samples: CLI usage is None on the 600s
        // wall, but serve sidecars survive. Use the final turn's (largest prompt)
        // prompt/gen lens; only fill None (CLI usage stays authoritative).
        for s in &mut samples {
            if s.cc_input_tokens.is_some() {
                continue;
            }
            let prefix = format!("{}#{}#r", s.task_id, s.sample);
            let final_turn = records
                .iter()
                .filter(|r| r.label.starts_with(&prefix))
                .max_by_key(|r| r.prompt_ids.len());
            if let Some(r) = final_turn {
                s.cc_input_tokens = Some(r.prompt_ids.len() as u64);
                s.cc_output_tokens = Some(r.response_ids.len() as u64);
            }
        }
        Ok(CcGroup {
            task_id: booted.task.instance_id.clone(),
            samples,
            records,
        })
    }

    fn run_sample(
        &self,
        task: &SweTask,
        nonce: u64,
        sample: usize,
        workdir: &Path,
    ) -> ScoredSample {
        // Per-sample model tag keeps concurrent samples' dumps attributable.
        let model = sample_model(&self.model_id, nonce, sample);
        let mut cmd = Command::new("claude");
        cmd.arg("-p")
            .args(["--model", &model])
            // Sandbox is offline; web calls stall on CC's retry-backoff.
            .args(["--allowedTools", "Bash Read Write Edit Grep Glob"])
            .args(["--output-format", "json", "--dangerously-skip-permissions"])
            .arg(cc_prompt(&task.problem_statement))
            .current_dir(workdir)
            .env(
                "ANTHROPIC_BASE_URL",
                &self.base_urls[sample % self.base_urls.len()],
            )
            .env("ANTHROPIC_API_KEY", "dummy-local")
            // Required under uid 0 for --dangerously-skip-permissions.
            .env("IS_SANDBOX", "1")
            .env("ANTHROPIC_MODEL", &model)
            .env("ANTHROPIC_SMALL_FAST_MODEL", &model)
            .env("DISABLE_TELEMETRY", "1")
            .env("DISABLE_AUTOUPDATER", "1")
            .env("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1");

        let t_start_ms = epoch_ms();
        let spawned = run_captured(cmd, Duration::from_secs(self.cc_timeout_secs));
        let t_end_ms = epoch_ms();
        let (cc, cc_error, cc_timeout) = match spawned {
            Err(err) => (None, Some(format!("cc spawn failed: {err}")), false),
            Ok((_, _, true)) => (None, Some("cc timeout".to_owned()), true),
            Ok((output, code, false)) => {
                let (cc, err) = parse_cc_json(&output, code);
                (cc, err, false)
            }
        };

        // `score_err` = diff/score step failed (not a clean fail). Errored
        // attempts don't carry a learnable reward.
        let (raw_reward, edited, score_err, score_note) = match diff_workdir(workdir) {
            Err(err) => (0.0, false, true, format!("diff error: {err}")),
            Ok(diff) if diff.trim().is_empty() => (0.0, false, false, "no edits".to_owned()),
            Ok(_) => match score_workdir(
                workdir,
                &task.test_patch,
                &task.fail_to_pass(),
                self.pythonpath.as_deref(),
                self.test_timeout_secs,
            ) {
                Ok((reward, log)) => (
                    reward,
                    true,
                    false,
                    log.lines().last().unwrap_or_default().to_owned(),
                ),
                Err(err) => (0.0, true, true, format!("score error: {err}")),
            },
        };
        let errored = cc_timeout || score_err;
        let reward = self.reward_shape.apply(raw_reward, errored);

        let usage = cc.as_ref().and_then(|v| v.get("usage"));
        let get = |v: Option<&serde_json::Value>, key| v.and_then(|v| v.get(key)?.as_u64());
        ScoredSample {
            task_id: task.instance_id.clone(),
            sample,
            reward,
            edited,
            errored,
            note: match cc_error {
                Some(err) => format!("{err}; {score_note}"),
                None => score_note,
            },
            t_start_ms,
            t_end_ms,
            cc_turns: get(cc.as_ref(), "num_turns"),
            cc_input_tokens: get(usage, "input_tokens"),
            cc_output_tokens: get(usage, "output_tokens"),
        }
    }
}

/// `--output-format json`; stderr is folded in, so fall back to the outermost `{…}`.
fn parse_cc_json(output: &[u8], code: Option<i32>) -> (Option<serde_json::Value>, Option<String>) {
    let text = String::from_utf8_lossy(output);
    let parsed: Option<serde_json::Value> = serde_json::from_str(text.trim())
        .ok()
        .or_else(|| serde_json::from_str(&text[text.find('{')?..=text.rfind('}')?]).ok());
    let Some(v) = parsed else {
        let tail: String = text
            .chars()
            .skip(text.chars().count().saturating_sub(300))
            .collect();
        return (
            None,
            Some(format!("non-json cc output rc={code:?}: {tail}")),
        );
    };
    let error = (v["is_error"].as_bool() == Some(true))
        .then(|| v["result"].as_str().unwrap_or("cc is_error").to_owned());
    (Some(v), error)
}

/// CC task prompt — verbatim from `cc_swe_baseline.py::cc_attempt`.
fn cc_prompt(problem_statement: &str) -> String {
    let clipped: String = problem_statement.chars().take(3000).collect();
    format!(
        "Fix a bug in this repository (cwd = repo root).\n\n\
         Problem statement:\n{clipped}\n\n\
         Work in this order and be decisive:\n\
         1. Briefly locate the buggy code (a few reads/greps — do NOT read the whole codebase).\n\
         2. As soon as you have identified the fix, EDIT the source file. Do not keep exploring \
         after you understand the bug — make the edit.\n\
         3. Make the SMALLEST correct change that resolves the issue.\n\n\
         You MUST edit at least one source file before you finish; an answer with no edit is a \
         failure. Do not write or run the hidden tests (they are applied at scoring time). Do \
         not commit."
    )
}

/// `<model>#g<nonce>s<sample>` — distinct per sample for dump attribution.
fn sample_model(model_id: &str, nonce: u64, sample: usize) -> String {
    format!("{model_id}#g{nonce}s{sample}")
}

fn epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}
