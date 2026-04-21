//! Training observability sinks — scalar metrics, lifecycle events, stdout,
//! JSONL, and a shared async writer handle for the active training binaries.
//!
//! `MetricSample` keeps the hot path allocation-light for scalar step metrics,
//! while `TrainEvent` carries lower-frequency lifecycle records such as
//! `run_start`, `checkpoint`, `status`, and `run_end`.

use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

use opentelemetry::KeyValue;
use opentelemetry::logs::{AnyValue, LogRecord, Logger, LoggerProvider, Severity};
use opentelemetry_otlp::{Protocol, WithExportConfig, WithHttpConfig};
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::logs::{SdkLogger, SdkLoggerProvider};

const DEFAULT_SHARED_BUFFER_CAPACITY: usize = 1024;
const OTLP_SERVICE_NAME_DEFAULT: &str = "agent-infer.train";
const WANDB_MODE_DEFAULT: &str = "offline";

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OtlpLogConfig {
    pub endpoint: String,
    pub service_name: String,
    pub timeout: Option<Duration>,
    pub headers: Vec<(String, String)>,
}

impl OtlpLogConfig {
    pub fn from_env() -> Option<Self> {
        let endpoint = env::var("TRAIN_OTLP_LOGS_ENDPOINT")
            .or_else(|_| env::var("TRAIN_OTLP_ENDPOINT"))
            .or_else(|_| env::var("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT"))
            .or_else(|_| env::var("OTEL_EXPORTER_OTLP_ENDPOINT"))
            .ok()
            .map(|value| normalize_otlp_logs_endpoint(value.trim()))
            .filter(|value| !value.is_empty())?;
        let service_name = env::var("TRAIN_OTLP_SERVICE_NAME")
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| OTLP_SERVICE_NAME_DEFAULT.to_string());
        let timeout = env::var("TRAIN_OTLP_TIMEOUT_MS")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .filter(|value| *value > 0)
            .map(Duration::from_millis);
        let headers = env::var("TRAIN_OTLP_HEADERS")
            .ok()
            .map(|value| parse_header_env(&value))
            .unwrap_or_default();
        Some(Self {
            endpoint,
            service_name,
            timeout,
            headers,
        })
    }
}

/// OpenTelemetry OTLP logs sink backed by the blocking HTTP exporter.
///
/// This keeps the existing Rust-native event stream as the source of truth and
/// mirrors it into vendor-neutral OTLP log records. The exporter is wired into
/// the same background worker thread as the other remote sinks, so training
/// loops never perform network I/O directly.
pub struct OtlpLogSink {
    provider: SdkLoggerProvider,
    logger: SdkLogger,
}

impl OtlpLogSink {
    pub fn new(config: OtlpLogConfig) -> anyhow::Result<Self> {
        let endpoint = normalize_otlp_logs_endpoint(&config.endpoint);
        let mut builder = opentelemetry_otlp::LogExporter::builder()
            .with_http()
            .with_protocol(Protocol::HttpBinary)
            .with_endpoint(endpoint);
        if let Some(timeout) = config.timeout {
            builder = builder.with_timeout(timeout);
        }
        if !config.headers.is_empty() {
            let headers = config.headers.into_iter().collect();
            builder = builder.with_headers(headers);
        }
        let exporter = builder
            .build()
            .map_err(|err| anyhow::anyhow!("otlp log exporter build failed: {err}"))?;
        let provider = SdkLoggerProvider::builder()
            .with_resource(
                Resource::builder_empty()
                    .with_attributes([
                        KeyValue::new("service.name", config.service_name),
                        KeyValue::new("service.namespace", "agent-infer"),
                        KeyValue::new("telemetry.sdk.language", "rust"),
                    ])
                    .build(),
            )
            .with_simple_exporter(exporter)
            .build();
        let logger = provider.logger("train.metrics");
        Ok(Self { provider, logger })
    }

    fn emit_log_record(
        &self,
        body: impl Into<AnyValue>,
        severity: Severity,
        attrs: impl IntoIterator<Item = (String, AnyValue)>,
    ) {
        let mut record = self.logger.create_log_record();
        record.set_severity_number(severity);
        record.set_body(body.into());
        record.add_attributes(attrs);
        self.logger.emit(record);
    }
}

impl MetricSink for OtlpLogSink {
    fn emit(&mut self, sample: &MetricSample<'_>) {
        let mut attrs = Vec::with_capacity(sample.fields.len() + 3);
        attrs.push(("train.kind".to_string(), AnyValue::from("metric")));
        attrs.push((
            "train.phase".to_string(),
            AnyValue::from(sample.phase.to_string()),
        ));
        attrs.push(("train.step".to_string(), AnyValue::from(sample.step as i64)));
        for (key, value) in sample.fields {
            attrs.push((format!("metric.{key}"), AnyValue::from(*value)));
        }
        self.emit_log_record("metric", Severity::Info, attrs);
    }

    fn event(&mut self, event: &TrainEvent<'_>) {
        let mut attrs =
            Vec::with_capacity(event.strings.len() + event.scalars.len() + event.bools.len() + 2);
        attrs.push((
            "train.kind".to_string(),
            AnyValue::from(event.kind.to_string()),
        ));
        if let Some(step) = event.step {
            attrs.push(("train.step".to_string(), AnyValue::from(step as i64)));
        }
        for (key, value) in event.strings {
            attrs.push((format!("event.{key}"), AnyValue::from((*value).to_string())));
        }
        for (key, value) in event.scalars {
            attrs.push((format!("event.{key}"), AnyValue::from(*value)));
        }
        for (key, value) in event.bools {
            attrs.push((format!("event.{key}"), AnyValue::from(*value)));
        }
        let severity = match (
            event.kind,
            event
                .strings
                .iter()
                .find_map(|(key, value)| (*key == "status").then_some(*value)),
        ) {
            ("run_end", Some("failed")) => Severity::Error,
            ("run_end", Some("stopped")) | ("status", _) => Severity::Warn,
            _ => Severity::Info,
        };
        self.emit_log_record(event.kind.to_string(), severity, attrs);
    }

    fn flush(&mut self) {
        let _ = self.provider.force_flush();
    }
}

impl Drop for OtlpLogSink {
    fn drop(&mut self) {
        let _ = self.provider.force_flush();
        let _ = self.provider.shutdown();
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WandbConfig {
    pub project: String,
    pub entity: Option<String>,
    pub name: Option<String>,
    pub notes: Option<String>,
    pub group: Option<String>,
    pub job_type: Option<String>,
    pub run_id: Option<String>,
    pub resume: Option<String>,
    pub mode: String,
    pub dir: Option<PathBuf>,
    pub base_url: Option<String>,
    pub tags: Vec<String>,
    pub helper_program: String,
    pub helper_script: PathBuf,
    pub log_checkpoints: bool,
    pub disable_code: bool,
    pub silent: bool,
}

impl WandbConfig {
    pub fn from_env() -> Option<Self> {
        let project = env::var("TRAIN_WANDB_PROJECT")
            .ok()
            .or_else(|| env::var("WANDB_PROJECT").ok())
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())?;
        let entity = optional_env("TRAIN_WANDB_ENTITY").or_else(|| optional_env("WANDB_ENTITY"));
        let name = optional_env("TRAIN_WANDB_NAME").or_else(|| optional_env("WANDB_NAME"));
        let notes = optional_env("TRAIN_WANDB_NOTES").or_else(|| optional_env("WANDB_NOTES"));
        let group = optional_env("TRAIN_WANDB_GROUP").or_else(|| optional_env("WANDB_RUN_GROUP"));
        let job_type =
            optional_env("TRAIN_WANDB_JOB_TYPE").or_else(|| optional_env("WANDB_JOB_TYPE"));
        let run_id = optional_env("TRAIN_WANDB_RUN_ID").or_else(|| optional_env("WANDB_RUN_ID"));
        let resume = optional_env("TRAIN_WANDB_RESUME").or_else(|| optional_env("WANDB_RESUME"));
        let mode = optional_env("TRAIN_WANDB_MODE")
            .or_else(|| optional_env("WANDB_MODE"))
            .unwrap_or_else(|| WANDB_MODE_DEFAULT.to_string());
        let dir = optional_env("TRAIN_WANDB_DIR")
            .or_else(|| optional_env("WANDB_DIR"))
            .map(PathBuf::from);
        let base_url =
            optional_env("TRAIN_WANDB_BASE_URL").or_else(|| optional_env("WANDB_BASE_URL"));
        let tags = optional_env("TRAIN_WANDB_TAGS")
            .map(|value| parse_csv_env(&value))
            .unwrap_or_default();
        let helper_program =
            optional_env("TRAIN_WANDB_PYTHON").unwrap_or_else(|| "python3".to_string());
        let helper_script = optional_env("TRAIN_WANDB_HELPER")
            .map(PathBuf::from)
            .unwrap_or_else(default_wandb_helper_script);
        let log_checkpoints = optional_env("TRAIN_WANDB_LOG_CHECKPOINTS")
            .map(|value| truthy_env_value(&value))
            .unwrap_or(true);
        let disable_code = optional_env("TRAIN_WANDB_DISABLE_CODE")
            .map(|value| truthy_env_value(&value))
            .unwrap_or(true);
        let silent = optional_env("TRAIN_WANDB_SILENT")
            .map(|value| truthy_env_value(&value))
            .unwrap_or(true);
        Some(Self {
            project,
            entity,
            name,
            notes,
            group,
            job_type,
            run_id,
            resume,
            mode,
            dir,
            base_url,
            tags,
            helper_program,
            helper_script,
            log_checkpoints,
            disable_code,
            silent,
        })
    }
}

/// Optional W&B sidecar sink.
///
/// The hot path stays Rust-only: the foreground emits `MetricSample` /
/// `TrainEvent` into `SharedSink`, and the background worker forwards them to a
/// helper process that uses the official W&B SDK. This mirrors W&B's own
/// best-practice guidance around offline/local buffering and avoids
/// reverse-engineering the `.wandb` datastore format.
pub struct WandbProcessSink {
    child: Child,
    stdin: BufWriter<ChildStdin>,
}

impl WandbProcessSink {
    pub fn new(config: WandbConfig) -> anyhow::Result<Self> {
        if !config.helper_script.is_file() {
            return Err(anyhow::anyhow!(
                "wandb helper script missing at {}",
                config.helper_script.display()
            ));
        }
        let mut command = Command::new(&config.helper_program);
        command
            .arg(&config.helper_script)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .env("WANDB_PROJECT", &config.project)
            .env("WANDB_MODE", &config.mode)
            .env(
                "TRAIN_WANDB_LOG_CHECKPOINTS",
                if config.log_checkpoints { "1" } else { "0" },
            )
            .env(
                "WANDB_DISABLE_CODE",
                if config.disable_code { "true" } else { "false" },
            )
            .env("WANDB_SILENT", if config.silent { "true" } else { "false" });
        if let Some(entity) = &config.entity {
            command.env("WANDB_ENTITY", entity);
        }
        if let Some(name) = &config.name {
            command.env("WANDB_NAME", name);
        }
        if let Some(notes) = &config.notes {
            command.env("WANDB_NOTES", notes);
        }
        if let Some(group) = &config.group {
            command.env("WANDB_RUN_GROUP", group);
        }
        if let Some(job_type) = &config.job_type {
            command.env("WANDB_JOB_TYPE", job_type);
        }
        if let Some(run_id) = &config.run_id {
            command.env("WANDB_RUN_ID", run_id);
        }
        if let Some(resume) = &config.resume {
            command.env("WANDB_RESUME", resume);
        }
        if let Some(dir) = &config.dir {
            command.env("WANDB_DIR", dir);
        }
        if let Some(base_url) = &config.base_url {
            command.env("WANDB_BASE_URL", base_url);
        }
        if !config.tags.is_empty() {
            command.env("TRAIN_WANDB_TAGS", config.tags.join(","));
        }
        let mut child = command.spawn().map_err(|err| {
            anyhow::anyhow!(
                "spawn wandb helper {} via {} failed: {err}",
                config.helper_script.display(),
                config.helper_program
            )
        })?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow::anyhow!("wandb helper stdin unavailable"))?;
        Ok(Self {
            child,
            stdin: BufWriter::new(stdin),
        })
    }

    fn write_json_line(&mut self, value: &serde_json::Value) -> anyhow::Result<()> {
        serde_json::to_writer(&mut self.stdin, value)?;
        self.stdin.write_all(b"\n")?;
        self.stdin.flush()?;
        Ok(())
    }

    fn emit_metric_message(&mut self, sample: &MetricSample<'_>) -> anyhow::Result<()> {
        let fields = sample
            .fields
            .iter()
            .map(|(key, value)| ((*key).to_string(), json_f64(*value)))
            .collect::<serde_json::Map<_, _>>();
        let body = serde_json::json!({
            "type": "metric",
            "step": sample.step,
            "phase": sample.phase,
            "fields": fields,
        });
        self.write_json_line(&body)
    }

    fn emit_event_message(&mut self, event: &TrainEvent<'_>) -> anyhow::Result<()> {
        let strings = event
            .strings
            .iter()
            .map(|(key, value)| ((*key).to_string(), serde_json::Value::from(*value)))
            .collect::<serde_json::Map<_, _>>();
        let scalars = event
            .scalars
            .iter()
            .map(|(key, value)| ((*key).to_string(), json_f64(*value)))
            .collect::<serde_json::Map<_, _>>();
        let bools = event
            .bools
            .iter()
            .map(|(key, value)| ((*key).to_string(), serde_json::Value::from(*value)))
            .collect::<serde_json::Map<_, _>>();
        let body = serde_json::json!({
            "type": "event",
            "kind": event.kind,
            "step": event.step,
            "strings": strings,
            "scalars": scalars,
            "bools": bools,
        });
        self.write_json_line(&body)
    }
}

impl MetricSink for WandbProcessSink {
    fn emit(&mut self, sample: &MetricSample<'_>) {
        if let Err(err) = self.emit_metric_message(sample) {
            eprintln!("[metrics] WandbProcessSink metric export failed: {err}");
        }
    }

    fn event(&mut self, event: &TrainEvent<'_>) {
        if let Err(err) = self.emit_event_message(event) {
            eprintln!("[metrics] WandbProcessSink event export failed: {err}");
        }
    }

    fn flush(&mut self) {
        let _ = self.stdin.flush();
    }
}

impl Drop for WandbProcessSink {
    fn drop(&mut self) {
        let _ = self.write_json_line(&serde_json::json!({ "type": "finish" }));
        let _ = self.stdin.flush();
        let _ = self.child.wait();
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlflowConfig {
    pub tracking_uri: String,
    pub experiment_id: String,
    pub run_name: Option<String>,
    pub auth_token: Option<String>,
    pub upload_artifacts: bool,
    pub artifact_path_prefix: String,
}

impl MlflowConfig {
    pub fn from_env() -> Option<Self> {
        let tracking_uri = env::var("TRAIN_MLFLOW_TRACKING_URI").ok()?;
        let tracking_uri = tracking_uri.trim().trim_end_matches('/').to_string();
        if tracking_uri.is_empty() {
            return None;
        }
        let experiment_id =
            env::var("TRAIN_MLFLOW_EXPERIMENT_ID").unwrap_or_else(|_| "0".to_string());
        let run_name = env::var("TRAIN_MLFLOW_RUN_NAME")
            .ok()
            .map(|name| name.trim().to_string())
            .filter(|name| !name.is_empty());
        let auth_token = env::var("TRAIN_MLFLOW_AUTH_TOKEN")
            .ok()
            .map(|token| token.trim().to_string())
            .filter(|token| !token.is_empty());
        let upload_artifacts = env::var("TRAIN_MLFLOW_UPLOAD_ARTIFACTS")
            .ok()
            .map(|value| matches!(value.trim(), "1" | "true" | "TRUE" | "yes" | "on"))
            .unwrap_or(false);
        let artifact_path_prefix = env::var("TRAIN_MLFLOW_ARTIFACT_PATH_PREFIX")
            .ok()
            .map(|value| value.trim().trim_matches('/').to_string())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "checkpoints".to_string());
        Some(Self {
            tracking_uri,
            experiment_id,
            run_name,
            auth_token,
            upload_artifacts,
            artifact_path_prefix,
        })
    }
}

/// MLflow Tracking sink backed by the REST API.
///
/// The sink is intentionally best-effort: remote failures are logged to stderr
/// and do not bubble back into the foreground training loop. The active
/// binaries already funnel all sink I/O through `SharedSink`, so blocking HTTP
/// calls stay off the hot path.
pub struct MlflowSink {
    config: MlflowConfig,
    run_id: Option<String>,
    agent: ureq::Agent,
}

impl MlflowSink {
    pub fn new(config: MlflowConfig) -> Self {
        Self {
            config,
            run_id: None,
            agent: ureq::AgentBuilder::new().build(),
        }
    }

    fn ensure_run(&mut self, event: Option<&TrainEvent<'_>>) -> anyhow::Result<&str> {
        if self.run_id.is_none() {
            let run_name = self
                .config
                .run_name
                .clone()
                .or_else(|| {
                    event.and_then(|event| {
                        event
                            .strings
                            .iter()
                            .find_map(|(k, v)| (*k == "run_id").then_some((*v).to_string()))
                    })
                })
                .unwrap_or_else(|| default_run_id("mlflow"));
            let mut tags = vec![
                serde_json::json!({"key": "source.name", "value": "agent-infer/train"}),
                serde_json::json!({"key": "mlflow.runName", "value": run_name}),
            ];
            if let Some(event) = event {
                tags.extend(
                    event.strings.iter().map(
                        |(k, v)| serde_json::json!({"key": format!("train.{k}"), "value": *v}),
                    ),
                );
                tags.extend(event.bools.iter().map(|(k, v)| {
                    serde_json::json!({"key": format!("train.{k}"), "value": v.to_string()})
                }));
            }
            let body = serde_json::json!({
                "experiment_id": self.config.experiment_id,
                "start_time": now_ms() as i64,
                "tags": tags,
            });
            let response = self.post_json("api/2.0/mlflow/runs/create", &body)?;
            let value: serde_json::Value = response
                .into_json()
                .map_err(|err| anyhow::anyhow!("mlflow runs/create decode failed: {err}"))?;
            let run_id = value
                .pointer("/run/info/run_id")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| anyhow::anyhow!("mlflow runs/create response missing run_id"))?;
            self.run_id = Some(run_id.to_string());
        }
        Ok(self.run_id.as_deref().expect("run_id set"))
    }

    fn post_json(&self, path: &str, body: &serde_json::Value) -> anyhow::Result<ureq::Response> {
        let url = format!("{}/{}", self.config.tracking_uri, path);
        let mut request = self
            .agent
            .post(&url)
            .set("Content-Type", "application/json");
        if let Some(token) = self.config.auth_token.as_deref() {
            request = request.set("Authorization", &format!("Bearer {token}"));
        }
        request
            .send_json(body)
            .map_err(|err| anyhow::anyhow!("mlflow POST {url} failed: {err}"))
    }

    fn update_run_status(&mut self, status: &str) -> anyhow::Result<()> {
        let Some(run_id) = self.run_id.as_deref() else {
            return Ok(());
        };
        let body = serde_json::json!({
            "run_id": run_id,
            "status": status,
            "end_time": now_ms() as i64,
        });
        self.post_json("api/2.0/mlflow/runs/update", &body)?;
        Ok(())
    }

    fn upload_checkpoint_artifacts(
        &self,
        run_id: &str,
        event: &TrainEvent<'_>,
    ) -> anyhow::Result<()> {
        if !self.config.upload_artifacts {
            return Ok(());
        }
        let Some(root_dir) = event
            .strings
            .iter()
            .find_map(|(k, v)| (*k == "path").then_some(PathBuf::from(v)))
        else {
            return Ok(());
        };
        let checkpoint_name = root_dir
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("checkpoint");
        for (key, filename) in event.strings {
            if !key.starts_with("artifact_") {
                continue;
            }
            let artifact_path = root_dir.join(filename);
            if !artifact_path.is_file() {
                continue;
            }
            let body = fs::read(&artifact_path).map_err(|err| {
                anyhow::anyhow!(
                    "mlflow checkpoint artifact read {} failed: {err}",
                    artifact_path.display()
                )
            })?;
            let remote_path = format!(
                "{}/{}/{}",
                self.config.artifact_path_prefix, checkpoint_name, filename
            );
            self.put_artifact(run_id, &remote_path, &body)?;
        }
        Ok(())
    }

    fn put_artifact(&self, run_id: &str, artifact_path: &str, body: &[u8]) -> anyhow::Result<()> {
        let url = format!(
            "{}/api/2.0/mlflow-artifacts/artifacts/{}?run_id={run_id}",
            self.config.tracking_uri, artifact_path
        );
        let mut request = self
            .agent
            .put(&url)
            .set("Content-Type", "application/octet-stream");
        if let Some(token) = self.config.auth_token.as_deref() {
            request = request.set("Authorization", &format!("Bearer {token}"));
        }
        request
            .send_bytes(body)
            .map_err(|err| anyhow::anyhow!("mlflow PUT {url} failed: {err}"))?;
        Ok(())
    }
}

impl MetricSink for MlflowSink {
    fn emit(&mut self, sample: &MetricSample<'_>) {
        let run_id = match self.ensure_run(None) {
            Ok(run_id) => run_id.to_string(),
            Err(err) => {
                eprintln!("[metrics] MlflowSink ensure_run failed: {err}");
                return;
            }
        };
        let timestamp = now_ms() as i64;
        let metrics = sample
            .fields
            .iter()
            .map(|(k, v)| {
                serde_json::json!({
                    "key": format!("{}.{}", sample.phase, k),
                    "value": *v,
                    "timestamp": timestamp,
                    "step": sample.step as i64,
                })
            })
            .collect::<Vec<_>>();
        let body = serde_json::json!({
            "run_id": run_id,
            "metrics": metrics,
            "params": [],
            "tags": [],
        });
        if let Err(err) = self.post_json("api/2.0/mlflow/runs/log-batch", &body) {
            eprintln!("[metrics] MlflowSink metric export failed: {err}");
        }
    }

    fn event(&mut self, event: &TrainEvent<'_>) {
        let run_id = match self.ensure_run(Some(event)) {
            Ok(run_id) => run_id.to_string(),
            Err(err) => {
                eprintln!("[metrics] MlflowSink ensure_run failed: {err}");
                return;
            }
        };
        let timestamp = now_ms() as i64;
        let metrics = event
            .scalars
            .iter()
            .map(|(k, v)| {
                serde_json::json!({
                    "key": format!("event.{}.{}", event.kind, k),
                    "value": *v,
                    "timestamp": timestamp,
                    "step": event.step.unwrap_or_default() as i64,
                })
            })
            .collect::<Vec<_>>();
        let params = if event.kind == "run_start" {
            event.scalars
                .iter()
                .map(|(k, v)| serde_json::json!({"key": format!("train.{k}"), "value": v.to_string()}))
                .chain(event.bools.iter().map(|(k, v)| {
                    serde_json::json!({"key": format!("train.{k}"), "value": v.to_string()})
                }))
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let tags = event
            .strings
            .iter()
            .map(|(k, v)| serde_json::json!({"key": format!("train.{}.{}", event.kind, k), "value": *v}))
            .collect::<Vec<_>>();
        let body = serde_json::json!({
            "run_id": run_id,
            "metrics": metrics,
            "params": params,
            "tags": tags,
        });
        if let Err(err) = self.post_json("api/2.0/mlflow/runs/log-batch", &body) {
            eprintln!("[metrics] MlflowSink event export failed: {err}");
        }
        if event.kind == "checkpoint" {
            if let Err(err) = self.upload_checkpoint_artifacts(&run_id, event) {
                eprintln!("[metrics] MlflowSink artifact upload failed: {err}");
            }
        }
        if event.kind == "run_end" {
            let status = event
                .strings
                .iter()
                .find_map(|(k, v)| (*k == "status").then_some(*v))
                .map(|status| match status {
                    "completed" => "FINISHED",
                    "stopped" => "KILLED",
                    "failed" => "FAILED",
                    _ => "FINISHED",
                })
                .unwrap_or("FINISHED");
            if let Err(err) = self.update_run_status(status) {
                eprintln!("[metrics] MlflowSink run update failed: {err}");
            }
        }
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
    Flush(mpsc::Sender<()>),
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
    tx: Mutex<Option<SyncSender<WorkerMessage>>>,
    join: Mutex<Option<JoinHandle<()>>>,
    dropped_metrics: AtomicU64,
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
    fn new(inner_sink: Box<dyn MetricSink>, buffer_capacity: usize) -> Self {
        let (tx, rx) = mpsc::sync_channel(buffer_capacity.max(1));
        let join = thread::Builder::new()
            .name("train-metrics".to_string())
            .spawn(move || run_worker(rx, inner_sink))
            .expect("train metrics worker thread");
        Self {
            inner: Arc::new(SharedSinkInner {
                tx: Mutex::new(Some(tx)),
                join: Mutex::new(Some(join)),
                dropped_metrics: AtomicU64::new(0),
            }),
        }
    }

    pub fn emit_metric(&self, sample: &MetricSample<'_>) {
        if let Some(tx) = self.inner.tx.lock().expect("metrics tx lock").as_ref() {
            match tx.try_send(WorkerMessage::Metric(OwnedMetricSample::from_borrowed(
                sample,
            ))) {
                Ok(()) => {}
                Err(TrySendError::Full(_)) => {
                    let dropped = self.inner.dropped_metrics.fetch_add(1, Ordering::AcqRel) + 1;
                    if dropped == 1 || dropped.is_multiple_of(100) {
                        eprintln!(
                            "[metrics] SharedSink queue full; dropped {dropped} metric samples"
                        );
                    }
                }
                Err(TrySendError::Disconnected(_)) => {}
            }
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

    pub fn dropped_metrics(&self) -> u64 {
        self.inner.dropped_metrics.load(Ordering::Acquire)
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
    open_shared_sink_with_extra(
        jsonl_path,
        also_stdout,
        /* append = */ false,
        Vec::new(),
    )
}

/// Append-mode sibling of [`open_shared_sink`].
pub fn open_shared_sink_append(
    jsonl_path: Option<&Path>,
    also_stdout: bool,
) -> anyhow::Result<SharedSink> {
    open_shared_sink_with_extra(
        jsonl_path,
        also_stdout,
        /* append = */ true,
        Vec::new(),
    )
}

pub fn open_shared_sink_with_extra(
    jsonl_path: Option<&Path>,
    also_stdout: bool,
    append: bool,
    extra_sinks: Vec<Box<dyn MetricSink>>,
) -> anyhow::Result<SharedSink> {
    Ok(SharedSink::new(
        build_sink_inner(jsonl_path, also_stdout, append, extra_sinks)?,
        shared_sink_buffer_capacity_from_env(),
    ))
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
    mut extra_sinks: Vec<Box<dyn MetricSink>>,
) -> anyhow::Result<Box<dyn MetricSink>> {
    let mut sinks: Vec<Box<dyn MetricSink>> = Vec::new();
    if let Some(config) = WandbConfig::from_env() {
        sinks.push(Box::new(WandbProcessSink::new(config)?));
    }
    if let Some(config) = OtlpLogConfig::from_env() {
        sinks.push(Box::new(OtlpLogSink::new(config)?));
    }
    if let Some(config) = MlflowConfig::from_env() {
        sinks.push(Box::new(MlflowSink::new(config)));
    }
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
    sinks.append(&mut extra_sinks);
    Ok(match sinks.len() {
        0 => Box::new(NullSink),
        1 => sinks.into_iter().next().expect("single sink"),
        _ => Box::new(MultiSink::new(sinks)),
    })
}

/// Small helper for binary-level `run_id` generation.
pub fn default_run_id(job_kind: &str) -> String {
    let millis = unix_ms();
    format!("{job_kind}-{}-{millis}", std::process::id())
}

fn unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn now_ms() -> u64 {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    millis as u64
}

fn parse_header_env(value: &str) -> Vec<(String, String)> {
    value
        .split(',')
        .filter_map(|pair| {
            let (key, value) = pair.split_once('=')?;
            let key = key.trim();
            let value = value.trim();
            (!key.is_empty() && !value.is_empty()).then(|| (key.to_string(), value.to_string()))
        })
        .collect()
}

fn parse_csv_env(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
        .collect()
}

fn optional_env(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn truthy_env_value(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn default_wandb_helper_script() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts/wandb_sink_helper.py")
}

fn json_f64(value: f64) -> serde_json::Value {
    serde_json::Number::from_f64(value)
        .map(serde_json::Value::Number)
        .unwrap_or(serde_json::Value::Null)
}

fn shared_sink_buffer_capacity_from_env() -> usize {
    env::var("TRAIN_METRICS_BUFFER_CAPACITY")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_SHARED_BUFFER_CAPACITY)
}

fn normalize_otlp_logs_endpoint(value: &str) -> String {
    if value.is_empty() || value.ends_with("/v1/logs") {
        return value.to_string();
    }
    if let Some(scheme_sep) = value.find("://") {
        let host_start = scheme_sep + 3;
        if value[host_start..].find('/').is_none() {
            return format!("{value}/v1/logs");
        }
        if value.ends_with('/') {
            return format!("{}v1/logs", value);
        }
    }
    value.to_string()
}
