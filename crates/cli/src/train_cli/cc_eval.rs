#[cfg(feature = "cuda")]
use {
    crate::args::TrainAgentOpdArgs,
    anyhow::{Context, Result},
    std::{
        fs,
        path::{Path, PathBuf},
    },
};

/// ONE sample per task (K=1 — best-of-N would inflate the pass-rate vs the
/// single-shot production setting). Writes the same `eval_round_{label}.jsonl`
/// shape as the in-house eval pass. Sampling params are CC's own (the harness
/// has no temperature knob).
#[cfg(feature = "cuda")]
pub(super) fn run_cc_eval(
    harness: &train::cc_harness::CcHarness,
    eval_tasks: &[(std::sync::Arc<train::swe_dataset::SweTask>, PathBuf)],
    out_dir: &Path,
    label: &str,
    concurrency: usize,
) -> Result<f32> {
    use std::io::Write;
    use std::sync::atomic::{AtomicUsize, Ordering};

    if eval_tasks.is_empty() {
        return Ok(0.0);
    }
    fs::create_dir_all(out_dir)
        .with_context(|| format!("create eval out dir {}", out_dir.display()))?;
    let out_path = out_dir.join(format!("eval_round_{label}.jsonl"));
    let mut file = fs::File::create(&out_path)
        .with_context(|| format!("create eval out {}", out_path.display()))?;

    // Bounded task-level concurrency: N workers steal task indices and send each
    // result back tagged with its index, so the output order matches the serial
    // path exactly. thread::scope joins all workers before the channel is drained.
    let concurrency = concurrency.clamp(1, eval_tasks.len());
    let next = AtomicUsize::new(0);
    let next = &next; // shared by ref so each `move` worker copies the ref, not the counter
    type EvalOut = Result<(serde_json::Value, bool, bool, f32)>;
    let (tx, rx) = std::sync::mpsc::channel::<(usize, EvalOut)>();
    std::thread::scope(|scope| {
        for _ in 0..concurrency {
            let tx = tx.clone();
            scope.spawn(move || {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some((task, staged)) = eval_tasks.get(i) else {
                        break;
                    };
                    let out = harness
                        .run_group(harness.boot_group(task, staged, 1))
                        .map(|g| {
                            let s = &g.samples[0];
                            let line = serde_json::json!({
                                "instance_id": s.task_id,
                                "passed": s.passed(),
                                "edited": s.edited,
                                "note": s.note,
                                "reward": s.reward,
                            });
                            (line, s.passed(), s.edited, s.reward)
                        });
                    let _ = tx.send((i, out));
                }
            });
        }
    });
    drop(tx); // workers (and their clones) are joined; drop ours so rx ends
    let mut results: Vec<Option<EvalOut>> = (0..eval_tasks.len()).map(|_| None).collect();
    for (i, out) in rx {
        results[i] = Some(out);
    }
    let (mut passed, mut edited, mut dense_sum) = (0usize, 0usize, 0.0f32);
    for out in results {
        let (line, p, e, r) = out.expect("every eval task produced a result")?;
        passed += usize::from(p);
        edited += usize::from(e);
        dense_sum += r;
        writeln!(file, "{line}")?;
    }
    let pass_rate = passed as f32 / eval_tasks.len() as f32;
    let mean_dense = dense_sum / eval_tasks.len() as f32;
    let agg = serde_json::json!({
        "aggregate": true,
        "label": label,
        "pass_rate": pass_rate,
        "mean_dense": mean_dense,
        "passed": passed,
        "edited": edited,
        "tasks": eval_tasks.len(),
    });
    writeln!(file, "{agg}")?;
    file.flush()?;
    eprintln!(
        "[arle train agent-opd] eval[{label}]: held-out pass_rate={pass_rate:.4} mean_dense={mean_dense:.4} ({passed}/{} tasks) -> {}",
        eval_tasks.len(),
        out_path.display(),
    );
    Ok(pass_rate)
}

/// Pure function of args so every fleet rank derives the same shared dump dir.
#[cfg(feature = "cuda")]
pub(super) fn agent_opd_eval_out_dir(args: &TrainAgentOpdArgs) -> PathBuf {
    crate::args::resolve_eval_out_dir(
        args.eval_out_dir.as_deref(),
        args.save_lora_adapters.as_deref(),
        args.save_checkpoint.as_deref(),
    )
}

/// Write failures log and never abort training.
#[cfg(feature = "cuda")]
pub(super) struct JsonlSink {
    path: PathBuf,
}

#[cfg(feature = "cuda")]
impl JsonlSink {
    pub(super) fn new(path: PathBuf) -> Self {
        Self { path }
    }

    pub(super) fn append(&self, row: &serde_json::Value) {
        let write = || -> Result<()> {
            use std::io::Write;
            if let Some(parent) = self.path.parent() {
                fs::create_dir_all(parent)?;
            }
            let mut file = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.path)?;
            writeln!(file, "{row}")?;
            Ok(())
        };
        if let Err(err) = write() {
            eprintln!(
                "[agent-opd] metrics write to {} failed: {err}",
                self.path.display()
            );
        }
    }
}
