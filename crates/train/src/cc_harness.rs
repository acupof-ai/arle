//! Claude-Code-as-harness rollout driver: per (task, sample) sandbox boot →
//! `claude -p` against the in-process serve → pytest scoring → group assembly →
//! in-memory cc-convert. Three bounded thread pools (boot / cc-run / score)
//! carry sample-level flow — a sample scores the moment its cc exits, never
//! waiting for the group. Task groups are sequential (staleness 0): the caller
//! releases one group's cc runs at a time while the NEXT group's boots (CPU
//! only, staleness-free) overlap the current group's rollout and train.
//!
//! Replaces `scripts/cc_run.sh` + `scripts/cc_swe_baseline.py` (P2 of
//! docs/plans/2026-07-16-agent-rl-unified-infra.md); prompt and cc invocation
//! are ported verbatim from `cc_swe_baseline.py::cc_attempt`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow};

use crate::cc_convert::{CcRecord, CcWindow, convert_cc_dumps};
use crate::sandbox::{boot_workdir, diff_workdir, run_captured, score_workdir};
use crate::swe_dataset::SweTask;

/// Harness-wide knobs (one per `agent-opd` run).
pub struct CcHarnessConfig {
    /// Per-sample sandboxes land at `<work_root>/<instance_id>#<sample>`.
    pub work_root: PathBuf,
    /// Serve `--dump-messages-dir` the attempt time-windows attribute against.
    pub dump_dir: PathBuf,
    pub tokenizer_path: PathBuf,
    /// In-process serve origin, e.g. `http://127.0.0.1:8000`.
    pub base_url: String,
    /// Served model id (`claude --model` + `ANTHROPIC_MODEL`).
    pub model_id: String,
    /// cc-run pool width = K concurrent samples of one group.
    pub width: usize,
    pub cc_timeout_secs: u64,
    pub test_timeout_secs: u64,
    /// Scoring `PYTHONPATH` prefix (e.g. `lib` for ansible).
    pub pythonpath: Option<String>,
}

/// One scored rollout sample: cc attempt stats + pytest verdict + dump window.
#[derive(Clone, Debug)]
pub struct ScoredSample {
    pub task_id: String,
    pub sample: usize,
    /// Dense pytest reward in [0,1]; pass ⇔ `reward >= 1.0`.
    pub reward: f32,
    pub passed: bool,
    pub edited: bool,
    pub note: String,
    /// cc attempt wall (epoch ms) — the dump-attribution window.
    pub t_start_ms: u64,
    pub t_end_ms: u64,
    pub cc_turns: Option<u64>,
    pub cc_input_tokens: Option<u64>,
    pub cc_output_tokens: Option<u64>,
    pub cc_error: Option<String>,
}

/// One completed task group: K scored samples + their converted token records
/// (records may be fewer than K — a no-request attempt converts to nothing).
pub struct CcGroup {
    pub task_id: String,
    pub samples: Vec<ScoredSample>,
    pub records: Vec<CcRecord>,
}

/// All rewards equal → the group carries zero advantage signal.
#[must_use]
pub fn zero_variance(samples: &[ScoredSample]) -> bool {
    samples.windows(2).all(|w| w[0].reward == w[1].reward)
}

/// Sample-level group assembly: a group completes at its `k`-th scored sample,
/// regardless of completion order across interleaved tasks.
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

    /// Returns the completed group (sorted by sample index) once its k-th
    /// sample lands; `None` while the group is still filling.
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

struct BootJob {
    task: Arc<SweTask>,
    staged_tree: PathBuf,
    sample: usize,
}

struct Booted {
    task: Arc<SweTask>,
    sample: usize,
    workdir: PathBuf,
}

struct CcDone {
    task: Arc<SweTask>,
    sample: usize,
    workdir: PathBuf,
    t_start_ms: u64,
    t_end_ms: u64,
    cc_turns: Option<u64>,
    cc_input_tokens: Option<u64>,
    cc_output_tokens: Option<u64>,
    cc_error: Option<String>,
}

/// The rollout driver. Owns the three pools; `boot_group` feeds boots ahead,
/// `run_group` releases one group's cc runs and blocks for its scored samples.
pub struct CcHarness {
    cfg: CcHarnessConfig,
    tokenizer: tokenizers::Tokenizer,
    boot_tx: SyncSender<BootJob>,
    booted_rx: Receiver<Result<Booted>>,
    cc_tx: SyncSender<Booted>,
    scored_rx: Receiver<ScoredSample>,
    /// Boot-ahead output for a group not yet released to the cc pool.
    parked: Vec<Booted>,
}

/// Blocking pull from a pool's shared receiver; `None` = channel closed
/// (harness dropped → worker exits).
fn next_job<T>(rx: &Arc<Mutex<Receiver<T>>>) -> Option<T> {
    rx.lock().ok()?.recv().ok()
}

impl CcHarness {
    pub fn new(cfg: CcHarnessConfig) -> Result<Self> {
        let tokenizer = tokenizers::Tokenizer::from_file(&cfg.tokenizer_path)
            .map_err(|err| anyhow!("load tokenizer {}: {err}", cfg.tokenizer_path.display()))?;
        let width = cfg.width.max(1);
        // Bounds: boot lanes hold the in-flight group plus one pre-booted group;
        // the rest hold one group. Bounded so a stalled stage backpressures.
        let (boot_tx, boot_rx) = sync_channel::<BootJob>(2 * width);
        let (booted_tx, booted_rx) = sync_channel::<Result<Booted>>(2 * width);
        let (cc_tx, cc_rx) = sync_channel::<Booted>(width);
        let (done_tx, done_rx) = sync_channel::<CcDone>(width);
        let (scored_tx, scored_rx) = sync_channel::<ScoredSample>(width);

        let boot_rx = Arc::new(Mutex::new(boot_rx));
        for i in 0..2 {
            let rx = Arc::clone(&boot_rx);
            let tx = booted_tx.clone();
            let work_root = cfg.work_root.clone();
            std::thread::Builder::new()
                .name(format!("cc-boot-{i}"))
                .spawn(move || {
                    while let Some(job) = next_job(&rx) {
                        let name = format!("{}#{}", job.task.instance_id, job.sample);
                        let booted = boot_workdir(
                            &work_root,
                            &name,
                            &job.staged_tree,
                            job.task.before_repo_set_cmd.as_deref(),
                        )
                        .with_context(|| format!("boot cc sandbox {name}"))
                        .map(|workdir| Booted {
                            task: job.task,
                            sample: job.sample,
                            workdir,
                        });
                        if tx.send(booted).is_err() {
                            break;
                        }
                    }
                })
                .context("spawn cc-boot worker")?;
        }

        let cc_rx = Arc::new(Mutex::new(cc_rx));
        for i in 0..width {
            let rx = Arc::clone(&cc_rx);
            let tx = done_tx.clone();
            let base_url = cfg.base_url.clone();
            let model_id = cfg.model_id.clone();
            let timeout = cfg.cc_timeout_secs;
            std::thread::Builder::new()
                .name(format!("cc-run-{i}"))
                .spawn(move || {
                    while let Some(booted) = next_job(&rx) {
                        let done = cc_attempt(booted, &base_url, &model_id, timeout);
                        if tx.send(done).is_err() {
                            break;
                        }
                    }
                })
                .context("spawn cc-run worker")?;
        }

        let done_rx = Arc::new(Mutex::new(done_rx));
        for i in 0..2 {
            let rx = Arc::clone(&done_rx);
            let tx = scored_tx.clone();
            let pythonpath = cfg.pythonpath.clone();
            let timeout = cfg.test_timeout_secs;
            std::thread::Builder::new()
                .name(format!("cc-score-{i}"))
                .spawn(move || {
                    while let Some(done) = next_job(&rx) {
                        let scored = score_sample(done, pythonpath.as_deref(), timeout);
                        if tx.send(scored).is_err() {
                            break;
                        }
                    }
                })
                .context("spawn cc-score worker")?;
        }

        Ok(Self {
            cfg,
            tokenizer,
            boot_tx,
            booted_rx,
            cc_tx,
            scored_rx,
            parked: Vec::new(),
        })
    }

    /// Enqueue `k` sandbox boots for `task`. Non-blocking up to two queued
    /// groups; boots overlap the current group's rollout/train (CPU only,
    /// staleness-free — cc runs start only when `run_group` releases them).
    pub fn boot_group(&self, task: &Arc<SweTask>, staged_tree: &Path, k: usize) -> Result<()> {
        for sample in 0..k.max(1) {
            self.boot_tx
                .send(BootJob {
                    task: Arc::clone(task),
                    staged_tree: staged_tree.to_owned(),
                    sample,
                })
                .map_err(|_| anyhow!("cc boot pool is gone"))?;
        }
        Ok(())
    }

    /// Release `task_id`'s booted samples to the cc pool as each boot lands
    /// (sample-level flow, no group barrier), block until all `k` are scored,
    /// then convert the group's serve dumps to token records in-memory. The
    /// caller must have `boot_group`ed the task first.
    pub fn run_group(&mut self, task_id: &str, k: usize) -> Result<CcGroup> {
        let k = k.max(1);
        let mut released = 0usize;
        // Boot-ahead may interleave a later group's boots; park those.
        let mut i = 0;
        while i < self.parked.len() {
            if self.parked[i].task.instance_id == task_id {
                let booted = self.parked.swap_remove(i);
                self.cc_tx
                    .send(booted)
                    .map_err(|_| anyhow!("cc run pool is gone"))?;
                released += 1;
            } else {
                i += 1;
            }
        }
        while released < k {
            let booted = self
                .booted_rx
                .recv()
                .map_err(|_| anyhow!("cc boot pool is gone"))??;
            if booted.task.instance_id == task_id {
                self.cc_tx
                    .send(booted)
                    .map_err(|_| anyhow!("cc run pool is gone"))?;
                released += 1;
            } else {
                self.parked.push(booted);
            }
        }

        let mut assembler = GroupAssembler::new(k);
        let samples = loop {
            let scored = self
                .scored_rx
                .recv()
                .map_err(|_| anyhow!("cc score pool is gone"))?;
            eprintln!(
                "[cc-harness] {}#{}: passed={} reward={:.3} edited={} turns={:?} wall={:.1}s :: {}",
                scored.task_id,
                scored.sample,
                scored.passed,
                scored.reward,
                scored.edited,
                scored.cc_turns,
                (scored.t_end_ms.saturating_sub(scored.t_start_ms)) as f64 / 1000.0,
                scored.cc_error.as_deref().unwrap_or(&scored.note),
            );
            if let Some(samples) = assembler.add(scored) {
                break samples;
            }
        };

        let windows: Vec<CcWindow> = samples
            .iter()
            .map(|s| CcWindow {
                label: format!("{}#{}", s.task_id, s.sample),
                t_start_ms: s.t_start_ms,
                t_end_ms: s.t_end_ms,
                reward: s.reward,
            })
            .collect();
        let records = convert_cc_dumps(&self.cfg.dump_dir, &self.tokenizer, &windows)?;
        Ok(CcGroup {
            task_id: task_id.to_owned(),
            samples,
            records,
        })
    }
}

fn epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

/// CC task prompt — ported verbatim from `cc_swe_baseline.py::cc_attempt`
/// (problem statement clipped to 3000 chars, matching the py slice).
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

/// Spawn one `claude -p` attempt in the sample's workdir via the fork-safe
/// `run_captured` path (spawner-routed when `ARLE_SPAWNER_SOCKET` is set — the
/// parent is CUDA-resident and multithreaded). The process group is killed on
/// timeout. Env goes per-child via `Command::env`, never `setenv`.
fn cc_attempt(booted: Booted, base_url: &str, model_id: &str, timeout_secs: u64) -> CcDone {
    let mut cmd = Command::new("claude");
    cmd.arg("-p")
        .args(["--model", model_id])
        // Allowlist keeps CC off WebFetch/WebSearch/Task — the sandbox is
        // offline and one web call stalls on CC's ~38s retry-backoff.
        .args(["--allowedTools", "Bash Read Write Edit Grep Glob"])
        .args(["--output-format", "json", "--dangerously-skip-permissions"])
        .arg(cc_prompt(&booted.task.problem_statement))
        .current_dir(&booted.workdir)
        .env("ANTHROPIC_BASE_URL", base_url)
        .env("ANTHROPIC_API_KEY", "dummy-local")
        // Mandatory on a root (uid 0) container: --dangerously-skip-permissions
        // refuses under root without it.
        .env("IS_SANDBOX", "1")
        .env("ANTHROPIC_MODEL", model_id)
        .env("ANTHROPIC_SMALL_FAST_MODEL", model_id)
        .env("DISABLE_TELEMETRY", "1")
        .env("DISABLE_AUTOUPDATER", "1")
        .env("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1");

    let t_start_ms = epoch_ms();
    let spawned = run_captured(cmd, Duration::from_secs(timeout_secs));
    let t_end_ms = epoch_ms();
    let (cc, cc_error) = match spawned {
        Err(err) => (None, Some(format!("cc spawn failed: {err}"))),
        Ok((_, _, true)) => (None, Some("cc timeout".to_owned())),
        Ok((output, code, false)) => parse_cc_json(&output, code),
    };
    let usage = cc.as_ref().and_then(|v| v.get("usage"));
    let field_u64 = |v: Option<&serde_json::Value>, key: &str| {
        v.and_then(|v| v.get(key))
            .and_then(serde_json::Value::as_u64)
    };
    CcDone {
        cc_turns: field_u64(cc.as_ref(), "num_turns"),
        cc_input_tokens: field_u64(usage, "input_tokens"),
        cc_output_tokens: field_u64(usage, "output_tokens"),
        cc_error,
        task: booted.task,
        sample: booted.sample,
        workdir: booted.workdir,
        t_start_ms,
        t_end_ms,
    }
}

/// `--output-format json` prints one JSON object on stdout; `run_captured`
/// folds stderr in, so fall back to the outermost `{…}` slice before giving up.
fn parse_cc_json(output: &[u8], code: Option<i32>) -> (Option<serde_json::Value>, Option<String>) {
    let text = String::from_utf8_lossy(output);
    let parsed: Option<serde_json::Value> = serde_json::from_str(text.trim()).ok().or_else(|| {
        let start = text.find('{')?;
        let end = text.rfind('}')?;
        serde_json::from_str(&text[start..=end]).ok()
    });
    match parsed {
        Some(v) => {
            let error = v
                .get("is_error")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
                .then(|| {
                    v.get("result")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("cc is_error")
                        .to_owned()
                });
            (Some(v), error)
        }
        None => {
            let tail: String = text
                .chars()
                .skip(text.chars().count().saturating_sub(300))
                .collect();
            (
                None,
                Some(format!("non-json cc output rc={code:?}: {tail}")),
            )
        }
    }
}

/// Score one finished attempt with the single (Rust) reward definition:
/// non-empty `git diff` gate, then `score_workdir` (errors count as failures,
/// denominator = len(fail_to_pass)). Scoring errors fold into reward 0.
fn score_sample(done: CcDone, pythonpath: Option<&str>, test_timeout_secs: u64) -> ScoredSample {
    let (reward, edited, note) = match diff_workdir(&done.workdir) {
        Err(err) => (0.0, false, format!("diff error: {err}")),
        Ok(diff) if diff.trim().is_empty() => (0.0, false, "no edits".to_owned()),
        Ok(_) => match score_workdir(
            &done.workdir,
            &done.task.test_patch,
            &done.task.fail_to_pass(),
            pythonpath,
            test_timeout_secs,
        ) {
            Ok((reward, log)) => (
                reward,
                true,
                log.lines().last().unwrap_or_default().to_owned(),
            ),
            Err(err) => (0.0, true, format!("score error: {err}")),
        },
    };
    ScoredSample {
        task_id: done.task.instance_id.clone(),
        sample: done.sample,
        reward,
        passed: reward >= 1.0,
        edited,
        note,
        t_start_ms: done.t_start_ms,
        t_end_ms: done.t_end_ms,
        cc_turns: done.cc_turns,
        cc_input_tokens: done.cc_input_tokens,
        cc_output_tokens: done.cc_output_tokens,
        cc_error: done.cc_error,
    }
}

#[cfg(test)]
mod tests {
    use super::{GroupAssembler, ScoredSample, zero_variance};

    fn sample(task: &str, idx: usize, reward: f32) -> ScoredSample {
        ScoredSample {
            task_id: task.to_owned(),
            sample: idx,
            reward,
            passed: reward >= 1.0,
            edited: reward > 0.0,
            note: String::new(),
            t_start_ms: 0,
            t_end_ms: 0,
            cc_turns: None,
            cc_input_tokens: None,
            cc_output_tokens: None,
            cc_error: None,
        }
    }

    /// K samples across interleaved completion order form correct groups; the
    /// zero-variance flag distinguishes signal-free groups.
    #[test]
    fn assembler_forms_groups_across_interleaved_completions() {
        let mut assembler = GroupAssembler::new(2);
        assert!(assembler.add(sample("a", 1, 1.0)).is_none());
        assert!(assembler.add(sample("b", 0, 0.5)).is_none());

        let a = assembler
            .add(sample("a", 0, 1.0))
            .expect("a completes at K");
        assert_eq!(a.iter().map(|s| s.sample).collect::<Vec<_>>(), [0, 1]);
        assert!(a.iter().all(|s| s.task_id == "a"));
        assert!(zero_variance(&a), "equal rewards carry no advantage");

        let b = assembler
            .add(sample("b", 1, 1.0))
            .expect("b completes at K");
        assert_eq!(b.iter().map(|s| s.sample).collect::<Vec<_>>(), [0, 1]);
        assert!(!zero_variance(&b), "0.5 vs 1.0 carries advantage");
    }
}
