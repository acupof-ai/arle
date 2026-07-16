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

/// Run-wide knobs; construct literally. Stateless — boot-ahead state lives in
/// the [`BootedGroup`] values the caller threads between calls.
pub struct CcHarness {
    /// Per-sample sandboxes land at `<work_root>/<instance_id>#<sample>`.
    pub work_root: PathBuf,
    /// Serve `--dump-messages-dir` the attempt time-windows attribute against.
    pub dump_dir: PathBuf,
    /// In-process serve origin, e.g. `http://127.0.0.1:8000`.
    pub base_url: String,
    /// Served model id (`claude --model` + `ANTHROPIC_MODEL`).
    pub model_id: String,
    pub cc_timeout_secs: u64,
    pub test_timeout_secs: u64,
    /// Scoring `PYTHONPATH` prefix (e.g. `lib` for ansible).
    pub pythonpath: Option<String>,
    pub tokenizer: tokenizers::Tokenizer,
}

pub fn load_tokenizer(path: &Path) -> Result<tokenizers::Tokenizer> {
    tokenizers::Tokenizer::from_file(path)
        .map_err(|err| anyhow!("load tokenizer {}: {err}", path.display()))
}

/// One scored rollout sample: cc attempt stats + pytest verdict + dump window.
#[derive(Clone, Debug)]
pub struct ScoredSample {
    pub task_id: String,
    pub sample: usize,
    /// Dense pytest reward in [0,1]; pass ⇔ `reward >= 1.0`.
    pub reward: f32,
    pub edited: bool,
    /// Score log tail, prefixed with the cc error when the attempt failed.
    pub note: String,
    /// cc attempt wall (epoch ms) — the dump-attribution window.
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

/// One completed group: K scored samples + their converted token records
/// (fewer than K when an attempt produced no serve request).
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

    /// The completed group (sorted by sample index) once its k-th sample lands.
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

/// Boot-ahead handle: `k` sandboxes building in the background; consumed by
/// [`CcHarness::run_group`]. Dropping it unwinds the boot threads.
pub struct BootedGroup {
    task: Arc<SweTask>,
    k: usize,
    rx: Receiver<Result<(usize, PathBuf)>>,
}

impl CcHarness {
    /// Start booting `k` sandboxes for `task` on 2 background threads.
    pub fn boot_group(&self, task: &Arc<SweTask>, staged_tree: &Path, k: usize) -> BootedGroup {
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
            rx,
        }
    }

    /// Release each sample to its own cc→score thread as its boot lands (no
    /// group barrier), collect all `k`, convert the group's dumps in-memory.
    pub fn run_group(&self, booted: BootedGroup) -> Result<CcGroup> {
        let samples = std::thread::scope(|scope| -> Result<Vec<ScoredSample>> {
            let (tx, rx) = sync_channel(booted.k);
            for _ in 0..booted.k {
                let (sample, workdir) = booted.rx.recv().context("cc boot thread died")??;
                let (tx, task) = (tx.clone(), &booted.task);
                scope.spawn(move || {
                    let _ = tx.send(self.run_sample(task, sample, &workdir));
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
            })
            .collect();
        let records = convert_cc_dumps(&self.dump_dir, &self.tokenizer, &windows)?;
        Ok(CcGroup {
            task_id: booted.task.instance_id.clone(),
            samples,
            records,
        })
    }

    /// One sample end-to-end: `claude -p` (fork-safe `run_captured`, spawner-
    /// routed under `ARLE_SPAWNER_SOCKET`; env per-child via `Command::env`),
    /// then the single Rust reward definition (`git diff` gate →
    /// `score_workdir`; errors fold into reward 0).
    fn run_sample(&self, task: &SweTask, sample: usize, workdir: &Path) -> ScoredSample {
        let mut cmd = Command::new("claude");
        cmd.arg("-p")
            .args(["--model", &self.model_id])
            // Allowlist keeps CC off WebFetch/WebSearch/Task — the sandbox is
            // offline and one web call stalls on CC's ~38s retry-backoff.
            .args(["--allowedTools", "Bash Read Write Edit Grep Glob"])
            .args(["--output-format", "json", "--dangerously-skip-permissions"])
            .arg(cc_prompt(&task.problem_statement))
            .current_dir(workdir)
            .env("ANTHROPIC_BASE_URL", &self.base_url)
            .env("ANTHROPIC_API_KEY", "dummy-local")
            // Mandatory on a root container: --dangerously-skip-permissions
            // refuses under uid 0 without it.
            .env("IS_SANDBOX", "1")
            .env("ANTHROPIC_MODEL", &self.model_id)
            .env("ANTHROPIC_SMALL_FAST_MODEL", &self.model_id)
            .env("DISABLE_TELEMETRY", "1")
            .env("DISABLE_AUTOUPDATER", "1")
            .env("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1");

        let t_start_ms = epoch_ms();
        let spawned = run_captured(cmd, Duration::from_secs(self.cc_timeout_secs));
        let t_end_ms = epoch_ms();
        let (cc, cc_error) = match spawned {
            Err(err) => (None, Some(format!("cc spawn failed: {err}"))),
            Ok((_, _, true)) => (None, Some("cc timeout".to_owned())),
            Ok((output, code, false)) => parse_cc_json(&output, code),
        };

        let (reward, edited, score_note) = match diff_workdir(workdir) {
            Err(err) => (0.0, false, format!("diff error: {err}")),
            Ok(diff) if diff.trim().is_empty() => (0.0, false, "no edits".to_owned()),
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
                    log.lines().last().unwrap_or_default().to_owned(),
                ),
                Err(err) => (0.0, true, format!("score error: {err}")),
            },
        };

        let usage = cc.as_ref().and_then(|v| v.get("usage"));
        let get = |v: Option<&serde_json::Value>, key| v.and_then(|v| v.get(key)?.as_u64());
        ScoredSample {
            task_id: task.instance_id.clone(),
            sample,
            reward,
            edited,
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

/// `--output-format json` prints one JSON object on stdout; `run_captured`
/// folds stderr in, so fall back to the outermost `{…}` slice.
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

/// CC task prompt — verbatim from `cc_swe_baseline.py::cc_attempt`, problem
/// statement clipped to 3000 chars like the py slice.
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

fn epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

#[cfg(test)]
mod tests {
    use super::{GroupAssembler, ScoredSample, zero_variance};

    fn sample(task: &str, idx: usize, reward: f32) -> ScoredSample {
        ScoredSample {
            task_id: task.to_owned(),
            sample: idx,
            reward,
            edited: reward > 0.0,
            note: String::new(),
            t_start_ms: 0,
            t_end_ms: 0,
            cc_turns: None,
            cc_input_tokens: None,
            cc_output_tokens: None,
        }
    }

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
