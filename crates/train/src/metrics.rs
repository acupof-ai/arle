//! Training metrics sinks — JSONL + stdout + multi fan-out.
//!
//! A `MetricSink` consumes `MetricSample { step, fields }` records. The
//! factory [`open_sink`] picks between JSONL file, stdout, both (via
//! [`MultiSink`]) or [`NullSink`] based on caller flags.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;

/// A single metric observation for a training step.
///
/// `fields` is borrowed so hot paths can emit without allocating a map; the
/// borrow lifetime `'a` ties the slice and the embedded `&str` keys to the
/// caller's stack frame.
pub struct MetricSample<'a> {
    pub step: u64,
    pub fields: &'a [(&'a str, f64)],
}

/// Sink that receives metric samples. `Send` so sinks can be owned by a
/// background writer thread later without changing the trait.
pub trait MetricSink: Send {
    fn emit(&mut self, sample: &MetricSample<'_>);
    fn flush(&mut self) {}
}

/// Sink that drops every sample. Used when metrics are disabled.
pub struct NullSink;

impl MetricSink for NullSink {
    fn emit(&mut self, _: &MetricSample<'_>) {}
}

/// Human-readable stdout sink: `step=N key=value ...` one line per sample.
///
/// Format rules:
/// - keys containing `"lr"` → `{:.3e}` (learning-rate style)
/// - keys containing `"ms"` → `{:.2}` (millisecond latencies)
/// - everything else → `{:.6}`
pub struct StdoutSink;

impl MetricSink for StdoutSink {
    fn emit(&mut self, sample: &MetricSample<'_>) {
        let mut line = format!("step={}", sample.step);
        for (k, v) in sample.fields {
            let formatted = if k.contains("lr") {
                format!(" {}={:.3e}", k, v)
            } else if k.contains("ms") {
                format!(" {}={:.2}", k, v)
            } else {
                format!(" {}={:.6}", k, v)
            };
            line.push_str(&formatted);
        }
        println!("{}", line);
    }

    fn flush(&mut self) {
        let _ = std::io::stdout().flush();
    }
}

/// Append-only JSONL sink. One JSON object per line.
pub struct JsonlSink {
    writer: BufWriter<File>,
}

impl JsonlSink {
    /// Create (truncate) `path` and wrap it in a buffered writer. Returns the
    /// underlying IO error if the parent directory is missing or the file is
    /// not creatable — no implicit `mkdir -p`.
    pub fn create(path: &Path) -> std::io::Result<Self> {
        let file = File::create(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
        })
    }

    /// Open `path` in append mode (creating it if absent) and wrap it in a
    /// buffered writer. Use this for multi-phase binaries that already
    /// truncated `path` at phase 1 and need subsequent phases to extend the
    /// same JSONL file rather than restart it (e.g. `train_grpo`'s SFT →
    /// GRPO handoff).
    pub fn open_append(path: &Path) -> std::io::Result<Self> {
        let file = OpenOptions::new().append(true).create(true).open(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
        })
    }
}

impl MetricSink for JsonlSink {
    fn emit(&mut self, sample: &MetricSample<'_>) {
        // Build a serde_json::Map so keys preserve insertion order and numeric
        // precision is handled by serde_json's float serializer.
        let mut map = serde_json::Map::with_capacity(sample.fields.len() + 1);
        map.insert("step".to_string(), serde_json::Value::from(sample.step));
        for (k, v) in sample.fields {
            // serde_json rejects NaN/Inf; fall back to null so the file stays
            // parseable rather than panicking on a bad sample.
            let value = serde_json::Number::from_f64(*v)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null);
            map.insert((*k).to_string(), value);
        }
        let obj = serde_json::Value::Object(map);
        if let Err(e) = writeln!(self.writer, "{}", obj) {
            eprintln!("[metrics] JsonlSink write failed: {}", e);
        }
    }

    fn flush(&mut self) {
        if let Err(e) = self.writer.flush() {
            eprintln!("[metrics] JsonlSink flush failed: {}", e);
        }
    }
}

impl Drop for JsonlSink {
    fn drop(&mut self) {
        let _ = self.writer.flush();
    }
}

/// Fan-out sink: calls `emit`/`flush` on each inner sink in order.
pub struct MultiSink {
    inner: Vec<Box<dyn MetricSink>>,
}

impl MultiSink {
    pub fn new(inner: Vec<Box<dyn MetricSink>>) -> Self {
        Self { inner }
    }
}

impl MetricSink for MultiSink {
    fn emit(&mut self, sample: &MetricSample<'_>) {
        for sink in &mut self.inner {
            sink.emit(sample);
        }
    }

    fn flush(&mut self) {
        for sink in &mut self.inner {
            sink.flush();
        }
    }
}

/// Build a sink from CLI-style flags.
///
/// - `jsonl_path = Some(p)` adds a [`JsonlSink`] writing to `p`.
/// - `also_stdout = true` adds a [`StdoutSink`].
/// - Neither set → [`NullSink`] (metrics disabled).
/// - Both set → [`MultiSink`] fanning out to both.
pub fn open_sink(
    jsonl_path: Option<&Path>,
    also_stdout: bool,
) -> anyhow::Result<Box<dyn MetricSink>> {
    open_sink_inner(jsonl_path, also_stdout, /* append = */ false)
}

/// Append-mode sibling of [`open_sink`]. Used by multi-phase binaries
/// (e.g. `train_grpo`'s GRPO phase, which runs after `run_sft_phase`
/// already truncated and wrote the JSONL header) so the second phase
/// extends the same file rather than restarting it.
pub fn open_sink_append(
    jsonl_path: Option<&Path>,
    also_stdout: bool,
) -> anyhow::Result<Box<dyn MetricSink>> {
    open_sink_inner(jsonl_path, also_stdout, /* append = */ true)
}

fn open_sink_inner(
    jsonl_path: Option<&Path>,
    also_stdout: bool,
    append: bool,
) -> anyhow::Result<Box<dyn MetricSink>> {
    let mut sinks: Vec<Box<dyn MetricSink>> = Vec::new();
    if let Some(path) = jsonl_path {
        let sink = if append {
            JsonlSink::open_append(path)
        } else {
            JsonlSink::create(path)
        };
        let sink = sink.map_err(|e| {
            anyhow::anyhow!(
                "failed to {} JSONL metrics sink at {}: {}",
                if append { "open" } else { "create" },
                path.display(),
                e
            )
        })?;
        sinks.push(Box::new(sink));
    }
    if also_stdout {
        sinks.push(Box::new(StdoutSink));
    }
    Ok(match sinks.len() {
        0 => Box::new(NullSink),
        1 => sinks.into_iter().next().unwrap(),
        _ => Box::new(MultiSink::new(sinks)),
    })
}
