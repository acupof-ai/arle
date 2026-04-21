//! Training observability sinks — scalar metrics, lifecycle events, stdout,
//! JSONL, and a shared async writer handle for the active training binaries.
//!
//! `MetricSample` keeps the hot path allocation-light for scalar step metrics,
//! while `TrainEvent` carries lower-frequency lifecycle records such as
//! `run_start`, `checkpoint`, `status`, and `run_end`.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{SystemTime, UNIX_EPOCH};

/// A single metric observation for a training step.
///
/// `fields` is borrowed so hot paths can emit without allocating a map; the
/// borrow lifetime `'a` ties the slice and the embedded `&str` keys to the
/// caller's stack frame.
pub struct MetricSample<'a> {
    pub step: u64,
    pub phase: &'a str,
    pub fields: &'a [(&'a str, f64)],
}

/// Generic train-side lifecycle / artifact event.
///
/// Strings, scalars, and booleans are split into separate borrowed slices so
/// callers can stay allocation-light on the foreground path while JSON sinks
/// still flatten everything into one record.
pub struct TrainEvent<'a> {
    pub kind: &'a str,
    pub step: Option<u64>,
    pub strings: &'a [(&'a str, &'a str)],
    pub scalars: &'a [(&'a str, f64)],
    pub bools: &'a [(&'a str, bool)],
}

impl<'a> TrainEvent<'a> {
    pub fn new(kind: &'a str) -> Self {
        Self {
            kind,
            step: None,
            strings: &[],
            scalars: &[],
            bools: &[],
        }
    }
}

/// Sink that receives scalar metric samples and optional lifecycle events.
/// `Send` so sinks can live behind a background worker thread.
pub trait MetricSink: Send {
    fn emit(&mut self, sample: &MetricSample<'_>);
    fn event(&mut self, _: &TrainEvent<'_>) {}
    fn flush(&mut self) {}
}

/// Sink that drops every sample/event. Used when observability is disabled.
pub struct NullSink;

impl MetricSink for NullSink {
    fn emit(&mut self, _: &MetricSample<'_>) {}
}

/// Human-readable stdout sink.
///
/// Metrics: `step=N phase=train key=value ...`
/// Events:  `event=checkpoint step=N key=value ...`
pub struct StdoutSink;

impl MetricSink for StdoutSink {
    fn emit(&mut self, sample: &MetricSample<'_>) {
        let mut line = format!("step={} phase={}", sample.step, sample.phase);
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
        println!("{line}");
    }

    fn event(&mut self, event: &TrainEvent<'_>) {
        let mut line = format!("event={}", event.kind);
        if let Some(step) = event.step {
            line.push_str(&format!(" step={step}"));
        }
        for (k, v) in event.strings {
            line.push_str(&format!(" {}={}", k, v));
        }
        for (k, v) in event.scalars {
            line.push_str(&format!(" {}={:.6}", k, v));
        }
        for (k, v) in event.bools {
            line.push_str(&format!(" {}={}", k, v));
        }
        println!("{line}");
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
    /// same JSONL file rather than restart it.
    pub fn open_append(path: &Path) -> std::io::Result<Self> {
        let file = OpenOptions::new().append(true).create(true).open(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
        })
    }

    fn write_json_map(&mut self, map: serde_json::Map<String, serde_json::Value>) {
        let obj = serde_json::Value::Object(map);
        if let Err(e) = writeln!(self.writer, "{}", obj) {
            eprintln!("[metrics] JsonlSink write failed: {e}");
        }
    }
}

impl MetricSink for JsonlSink {
    fn emit(&mut self, sample: &MetricSample<'_>) {
        let mut map = serde_json::Map::with_capacity(sample.fields.len() + 3);
        map.insert("kind".to_string(), serde_json::Value::from("metric"));
        map.insert("phase".to_string(), serde_json::Value::from(sample.phase));
        map.insert("step".to_string(), serde_json::Value::from(sample.step));
        for (k, v) in sample.fields {
            let value = serde_json::Number::from_f64(*v)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null);
            map.insert((*k).to_string(), value);
        }
        self.write_json_map(map);
    }

    fn event(&mut self, event: &TrainEvent<'_>) {
        let mut map = serde_json::Map::new();
        map.insert("kind".to_string(), serde_json::Value::from(event.kind));
        if let Some(step) = event.step {
            map.insert("step".to_string(), serde_json::Value::from(step));
        }
        for (k, v) in event.strings {
            map.insert((*k).to_string(), serde_json::Value::from(*v));
        }
        for (k, v) in event.scalars {
            let value = serde_json::Number::from_f64(*v)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null);
            map.insert((*k).to_string(), value);
        }
        for (k, v) in event.bools {
            map.insert((*k).to_string(), serde_json::Value::from(*v));
        }
        self.write_json_map(map);
    }

    fn flush(&mut self) {
        if let Err(e) = self.writer.flush() {
            eprintln!("[metrics] JsonlSink flush failed: {e}");
        }
    }
}

impl Drop for JsonlSink {
    fn drop(&mut self) {
        let _ = self.writer.flush();
    }
}

/// Fan-out sink: calls `emit`/`event`/`flush` on each inner sink in order.
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

    fn event(&mut self, event: &TrainEvent<'_>) {
        for sink in &mut self.inner {
            sink.event(event);
        }
    }

    fn flush(&mut self) {
        for sink in &mut self.inner {
            sink.flush();
        }
    }
}

#[derive(Clone)]
struct OwnedMetricSample {
    step: u64,
    phase: String,
    fields: Vec<(String, f64)>,
}

impl OwnedMetricSample {
    fn from_borrowed(sample: &MetricSample<'_>) -> Self {
        Self {
            step: sample.step,
            phase: sample.phase.to_string(),
            fields: sample
                .fields
                .iter()
                .map(|(k, v)| ((*k).to_string(), *v))
                .collect(),
        }
    }

    fn with_borrowed<R>(&self, f: impl FnOnce(&MetricSample<'_>) -> R) -> R {
        let fields = self
            .fields
            .iter()
            .map(|(k, v)| (k.as_str(), *v))
            .collect::<Vec<_>>();
        let sample = MetricSample {
            step: self.step,
            phase: self.phase.as_str(),
            fields: &fields,
        };
        f(&sample)
    }
}

#[derive(Clone)]
struct OwnedTrainEvent {
    kind: String,
    step: Option<u64>,
    strings: Vec<(String, String)>,
    scalars: Vec<(String, f64)>,
    bools: Vec<(String, bool)>,
}

impl OwnedTrainEvent {
    fn from_borrowed(event: &TrainEvent<'_>) -> Self {
        Self {
            kind: event.kind.to_string(),
            step: event.step,
            strings: event
                .strings
                .iter()
                .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
                .collect(),
            scalars: event
                .scalars
                .iter()
                .map(|(k, v)| ((*k).to_string(), *v))
                .collect(),
            bools: event
                .bools
                .iter()
                .map(|(k, v)| ((*k).to_string(), *v))
                .collect(),
        }
    }

    fn with_borrowed<R>(&self, f: impl FnOnce(&TrainEvent<'_>) -> R) -> R {
        let strings = self
            .strings
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect::<Vec<_>>();
        let scalars = self
            .scalars
            .iter()
            .map(|(k, v)| (k.as_str(), *v))
            .collect::<Vec<_>>();
        let bools = self
            .bools
            .iter()
            .map(|(k, v)| (k.as_str(), *v))
            .collect::<Vec<_>>();
        let event = TrainEvent {
            kind: self.kind.as_str(),
            step: self.step,
            strings: &strings,
            scalars: &scalars,
            bools: &bools,
        };
        f(&event)
    }
}

enum WorkerMessage {
    Metric(OwnedMetricSample),
    Event(OwnedTrainEvent),
    Flush(Sender<()>),
}

fn run_worker(rx: Receiver<WorkerMessage>, mut sink: Box<dyn MetricSink>) {
    while let Ok(message) = rx.recv() {
        match message {
            WorkerMessage::Metric(sample) => sample.with_borrowed(|sample| sink.emit(sample)),
            WorkerMessage::Event(event) => event.with_borrowed(|event| sink.event(event)),
            WorkerMessage::Flush(ack) => {
                sink.flush();
                let _ = ack.send(());
            }
        }
    }
    sink.flush();
}

struct SharedSinkInner {
    tx: Mutex<Option<Sender<WorkerMessage>>>,
    join: Mutex<Option<JoinHandle<()>>>,
}

impl Drop for SharedSinkInner {
    fn drop(&mut self) {
        let _ = self.tx.lock().expect("metrics tx lock").take();
        if let Some(join) = self.join.lock().expect("metrics join lock").take() {
            let _ = join.join();
        }
    }
}

/// Cloneable async observability handle.
///
/// Binaries keep one clone to emit lifecycle events while passing another clone
/// into `Trainer` as `Box<dyn MetricSink>`.
#[derive(Clone)]
pub struct SharedSink {
    inner: Arc<SharedSinkInner>,
}

impl SharedSink {
    fn new(inner_sink: Box<dyn MetricSink>) -> Self {
        let (tx, rx) = mpsc::channel();
        let join = thread::Builder::new()
            .name("train-metrics".to_string())
            .spawn(move || run_worker(rx, inner_sink))
            .expect("train metrics worker thread");
        Self {
            inner: Arc::new(SharedSinkInner {
                tx: Mutex::new(Some(tx)),
                join: Mutex::new(Some(join)),
            }),
        }
    }

    pub fn emit_metric(&self, sample: &MetricSample<'_>) {
        if let Some(tx) = self.inner.tx.lock().expect("metrics tx lock").as_ref() {
            let _ = tx.send(WorkerMessage::Metric(OwnedMetricSample::from_borrowed(
                sample,
            )));
        }
    }

    pub fn emit_event(&self, event: &TrainEvent<'_>) {
        if let Some(tx) = self.inner.tx.lock().expect("metrics tx lock").as_ref() {
            let _ = tx.send(WorkerMessage::Event(OwnedTrainEvent::from_borrowed(event)));
        }
    }

    pub fn flush_blocking(&self) {
        let (ack_tx, ack_rx) = mpsc::channel();
        let sent = self
            .inner
            .tx
            .lock()
            .expect("metrics tx lock")
            .as_ref()
            .map(|tx| tx.send(WorkerMessage::Flush(ack_tx)).is_ok())
            .unwrap_or(false);
        if sent {
            let _ = ack_rx.recv();
        }
    }
}

impl MetricSink for SharedSink {
    fn emit(&mut self, sample: &MetricSample<'_>) {
        self.emit_metric(sample);
    }

    fn event(&mut self, event: &TrainEvent<'_>) {
        self.emit_event(event);
    }

    fn flush(&mut self) {
        self.flush_blocking();
    }
}

/// Build a cloneable async sink from CLI-style flags.
pub fn open_shared_sink(
    jsonl_path: Option<&Path>,
    also_stdout: bool,
) -> anyhow::Result<SharedSink> {
    Ok(SharedSink::new(build_sink_inner(
        jsonl_path,
        also_stdout,
        /* append = */ false,
    )?))
}

/// Append-mode sibling of [`open_shared_sink`].
pub fn open_shared_sink_append(
    jsonl_path: Option<&Path>,
    also_stdout: bool,
) -> anyhow::Result<SharedSink> {
    Ok(SharedSink::new(build_sink_inner(
        jsonl_path,
        also_stdout,
        /* append = */ true,
    )?))
}

/// Build a boxed sink for call sites that do not need a retained clone.
pub fn open_sink(
    jsonl_path: Option<&Path>,
    also_stdout: bool,
) -> anyhow::Result<Box<dyn MetricSink>> {
    Ok(Box::new(open_shared_sink(jsonl_path, also_stdout)?))
}

/// Append-mode sibling of [`open_sink`].
pub fn open_sink_append(
    jsonl_path: Option<&Path>,
    also_stdout: bool,
) -> anyhow::Result<Box<dyn MetricSink>> {
    Ok(Box::new(open_shared_sink_append(jsonl_path, also_stdout)?))
}

fn build_sink_inner(
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
        1 => sinks.into_iter().next().expect("single sink"),
        _ => Box::new(MultiSink::new(sinks)),
    })
}

/// Small helper for binary-level `run_id` generation.
pub fn default_run_id(job_kind: &str) -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    format!("{job_kind}-{}-{millis}", std::process::id())
}
