//! Tracing config and exporters for infer observability.
//!
//! Reuses the existing `fastrace` instrumentation and adds:
//! - file export in Chrome Trace JSON format
//! - OTLP/HTTP trace export
//! - low-overhead sampling + dynamic level controls for request roots

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};
use fastrace::collector::{
    Config as FastraceConfig, Reporter, SpanContext as FastraceSpanContext, SpanRecord,
};
use log::error;
use opentelemetry::trace::{
    Event as OtelEvent, SpanContext as OtelSpanContext, SpanId as OtelSpanId, SpanKind, Status,
    TraceFlags, TraceId as OtelTraceId, TraceState,
};
use opentelemetry::{InstrumentationScope, KeyValue};
use opentelemetry_otlp::{Protocol, WithExportConfig, WithHttpConfig};
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::trace::{SpanData, SpanEvents, SpanExporter as _, SpanLinks};
use serde::Serialize;

const DEFAULT_TRACE_SERVICE_NAME: &str = "agent-infer.infer";
const DEFAULT_TRACE_REPORT_INTERVAL_MS: u64 = 1_000;
const SAMPLE_SCALE: u64 = 1_000_000;

const ENV_TRACE_LEVEL: &str = "INFER_TRACE_LEVEL";
const ENV_TRACE_SAMPLE_RATE: &str = "INFER_TRACE_SAMPLE_RATE";
const ENV_TRACE_OUTPUT_PATH: &str = "INFER_TRACE_OUTPUT_PATH";
const ENV_TRACE_REPORT_INTERVAL_MS: &str = "INFER_TRACE_REPORT_INTERVAL_MS";
const ENV_TRACE_SLOW_REQUEST_MS: &str = "INFER_TRACE_SLOW_REQUEST_MS";
const ENV_OTLP_TRACES_ENDPOINT: &str = "INFER_OTLP_TRACES_ENDPOINT";
const ENV_TRACE_SERVICE_NAME: &str = "INFER_TRACE_SERVICE_NAME";
const ENV_TRACE_OTLP_TIMEOUT_MS: &str = "INFER_TRACE_OTLP_TIMEOUT_MS";
const ENV_TRACE_OTLP_HEADERS: &str = "INFER_TRACE_OTLP_HEADERS";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum TraceLevel {
    Off = 0,
    Basic = 1,
    Verbose = 2,
}

impl TraceLevel {
    fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::Basic,
            2 => Self::Verbose,
            _ => Self::Off,
        }
    }
}

impl std::fmt::Display for TraceLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Off => f.write_str("off"),
            Self::Basic => f.write_str("basic"),
            Self::Verbose => f.write_str("verbose"),
        }
    }
}

impl FromStr for TraceLevel {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        let normalized = value.trim().to_ascii_lowercase();
        match normalized.as_str() {
            "" => bail!("trace level cannot be empty"),
            "off" | "none" | "disabled" => Ok(Self::Off),
            "basic" | "minimal" | "streaming" => Ok(Self::Basic),
            "verbose" | "full" | "debug" => Ok(Self::Verbose),
            _ => bail!("invalid trace level `{value}`; expected off, basic, or verbose"),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct TraceStartupConfig {
    pub level: Option<String>,
    pub sample_rate: Option<f64>,
    pub report_interval_ms: Option<u64>,
    pub slow_request_ms: Option<u64>,
    pub file_output: Option<PathBuf>,
    pub otlp_endpoint: Option<String>,
    pub otlp_headers: Option<String>,
    pub otlp_timeout_ms: Option<u64>,
    pub service_name: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OtlpTraceConfig {
    pub endpoint: String,
    pub service_name: String,
    pub timeout: Option<Duration>,
    pub headers: Vec<(String, String)>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TracingConfig {
    pub level: TraceLevel,
    pub sample_rate: f64,
    pub report_interval: Duration,
    pub slow_request_threshold: Option<Duration>,
    pub file_output: Option<PathBuf>,
    pub otlp: Option<OtlpTraceConfig>,
}

impl TracingConfig {
    pub fn resolve(startup: TraceStartupConfig) -> Result<Self> {
        let file_output = startup
            .file_output
            .or_else(|| std::env::var_os(ENV_TRACE_OUTPUT_PATH).map(PathBuf::from));
        let otlp_endpoint = startup
            .otlp_endpoint
            .or_else(|| std::env::var(ENV_OTLP_TRACES_ENDPOINT).ok())
            .map(|value| normalize_otlp_traces_endpoint(value.trim()))
            .filter(|value| !value.is_empty());
        let service_name = startup
            .service_name
            .or_else(|| std::env::var(ENV_TRACE_SERVICE_NAME).ok())
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| DEFAULT_TRACE_SERVICE_NAME.to_string());
        let otlp_timeout_ms = startup
            .otlp_timeout_ms
            .or_else(|| env_parse_u64(ENV_TRACE_OTLP_TIMEOUT_MS))
            .filter(|value| *value > 0);
        let otlp_headers = startup
            .otlp_headers
            .or_else(|| std::env::var(ENV_TRACE_OTLP_HEADERS).ok())
            .map(|value| parse_header_env(&value))
            .unwrap_or_default();
        let otlp = otlp_endpoint.map(|endpoint| OtlpTraceConfig {
            endpoint,
            service_name: service_name.clone(),
            timeout: otlp_timeout_ms.map(Duration::from_millis),
            headers: otlp_headers,
        });

        let has_sink = file_output.is_some() || otlp.is_some();
        let level = startup
            .level
            .or_else(|| std::env::var(ENV_TRACE_LEVEL).ok())
            .map(|value| TraceLevel::from_str(&value))
            .transpose()?
            .unwrap_or(if has_sink {
                TraceLevel::Basic
            } else {
                TraceLevel::Off
            });
        let sample_rate = startup
            .sample_rate
            .or_else(|| env_parse_f64(ENV_TRACE_SAMPLE_RATE))
            .unwrap_or(if level == TraceLevel::Off { 0.0 } else { 1.0 });
        if !(0.0..=1.0).contains(&sample_rate) {
            bail!("trace sample rate must be between 0.0 and 1.0; got {sample_rate}");
        }
        if !has_sink && (level != TraceLevel::Off || sample_rate > 0.0) {
            bail!(
                "tracing requested without a sink; set --trace-output-path or --otlp-traces-endpoint"
            );
        }

        let report_interval_ms = startup
            .report_interval_ms
            .or_else(|| env_parse_u64(ENV_TRACE_REPORT_INTERVAL_MS))
            .unwrap_or(DEFAULT_TRACE_REPORT_INTERVAL_MS)
            .max(1);
        let slow_request_threshold = startup
            .slow_request_ms
            .or_else(|| env_parse_u64(ENV_TRACE_SLOW_REQUEST_MS))
            .filter(|value| *value > 0)
            .map(Duration::from_millis);

        Ok(Self {
            level,
            sample_rate,
            report_interval: Duration::from_millis(report_interval_ms),
            slow_request_threshold,
            file_output,
            otlp,
        })
    }

    pub fn has_sink(&self) -> bool {
        self.file_output.is_some() || self.otlp.is_some()
    }

    pub fn summary(&self) -> String {
        let mut sinks = Vec::new();
        if let Some(path) = &self.file_output {
            sinks.push(format!("file={}", path.display()));
        }
        if let Some(otlp) = &self.otlp {
            sinks.push(format!("otlp={}", otlp.endpoint));
        }
        let slow = self
            .slow_request_threshold
            .map(|value| format!(", slow_ms={}", value.as_millis()))
            .unwrap_or_default();
        format!(
            "level={}, sample_rate={:.3}, interval_ms={}, sinks=[{}]{}",
            self.level,
            self.sample_rate,
            self.report_interval.as_millis(),
            sinks.join(", "),
            slow
        )
    }
}

#[derive(Clone, Debug)]
pub struct RequestTraceDecision {
    pub sampled: bool,
    pub level: TraceLevel,
}

impl RequestTraceDecision {
    pub fn effective_level(&self) -> TraceLevel {
        if self.sampled {
            self.level
        } else {
            TraceLevel::Off
        }
    }

    pub fn span_context(&self) -> FastraceSpanContext {
        FastraceSpanContext::random().sampled(self.sampled)
    }
}

#[derive(Debug)]
struct TraceRuntimeState {
    level: AtomicU8,
    sample_rate_scaled: AtomicU64,
    slow_request_ms: AtomicU64,
}

#[derive(Clone, Debug)]
pub struct TraceRuntime {
    shared: Arc<TraceRuntimeState>,
}

impl TraceRuntime {
    fn disabled() -> Self {
        Self {
            shared: Arc::new(TraceRuntimeState {
                level: AtomicU8::new(TraceLevel::Off as u8),
                sample_rate_scaled: AtomicU64::new(0),
                slow_request_ms: AtomicU64::new(0),
            }),
        }
    }

    pub fn level(&self) -> TraceLevel {
        TraceLevel::from_u8(self.shared.level.load(Ordering::Relaxed))
    }

    pub fn set_level(&self, level: TraceLevel) {
        self.shared.level.store(level as u8, Ordering::Relaxed);
    }

    pub fn sample_rate(&self) -> f64 {
        self.shared.sample_rate_scaled.load(Ordering::Relaxed) as f64 / SAMPLE_SCALE as f64
    }

    pub fn set_sample_rate(&self, sample_rate: f64) {
        self.shared.sample_rate_scaled.store(
            encode_sample_rate(sample_rate.clamp(0.0, 1.0)),
            Ordering::Relaxed,
        );
    }

    pub fn slow_request_threshold(&self) -> Option<Duration> {
        let millis = self.shared.slow_request_ms.load(Ordering::Relaxed);
        (millis > 0).then(|| Duration::from_millis(millis))
    }

    pub fn set_slow_request_threshold(&self, threshold: Option<Duration>) {
        self.shared.slow_request_ms.store(
            threshold.map_or(0, |value| {
                value.as_millis().min(u128::from(u64::MAX)) as u64
            }),
            Ordering::Relaxed,
        );
    }

    pub fn apply_config(&self, config: &TracingConfig) {
        self.set_level(config.level);
        self.set_sample_rate(config.sample_rate);
        self.set_slow_request_threshold(config.slow_request_threshold);
    }

    pub fn decide_request(&self, sample_key: impl AsRef<[u8]>) -> RequestTraceDecision {
        let level = self.level();
        let sampled = level != TraceLevel::Off && self.should_sample(sample_key.as_ref());
        RequestTraceDecision { sampled, level }
    }

    pub fn should_sample(&self, sample_key: &[u8]) -> bool {
        let threshold = self.shared.sample_rate_scaled.load(Ordering::Relaxed);
        if threshold == 0 {
            return false;
        }
        if threshold >= SAMPLE_SCALE {
            return true;
        }
        let hash = blake3::hash(sample_key);
        let mut prefix = [0u8; 8];
        prefix.copy_from_slice(&hash.as_bytes()[..8]);
        let value = u64::from_be_bytes(prefix) % SAMPLE_SCALE;
        value < threshold
    }

    pub fn effective_level_for_latency(
        &self,
        sample_key: impl AsRef<[u8]>,
        latency: Duration,
    ) -> TraceLevel {
        let decision = self.decide_request(sample_key);
        if !decision.sampled {
            return TraceLevel::Off;
        }
        if let Some(threshold) = self.slow_request_threshold() {
            if latency >= threshold {
                return TraceLevel::Verbose;
            }
        }
        decision.level
    }
}

static TRACE_RUNTIME: OnceLock<TraceRuntime> = OnceLock::new();

pub fn trace_runtime() -> &'static TraceRuntime {
    TRACE_RUNTIME.get_or_init(TraceRuntime::disabled)
}

#[derive(Clone, Debug)]
pub struct TracingInstallation {
    config: TracingConfig,
    runtime: TraceRuntime,
    reporter_installed: bool,
}

impl TracingInstallation {
    pub fn config(&self) -> &TracingConfig {
        &self.config
    }

    pub fn runtime(&self) -> &TraceRuntime {
        &self.runtime
    }

    pub fn reporter_installed(&self) -> bool {
        self.reporter_installed
    }
}

pub fn configure_global_tracing(startup: TraceStartupConfig) -> Result<TracingInstallation> {
    let config = TracingConfig::resolve(startup)?;
    let runtime = trace_runtime().clone();
    runtime.apply_config(&config);

    let reporter_installed = if config.has_sink() {
        let reporter = build_reporter(&config)?;
        fastrace::set_reporter(
            reporter,
            FastraceConfig::default().report_interval(config.report_interval),
        );
        true
    } else {
        false
    };

    Ok(TracingInstallation {
        config,
        runtime,
        reporter_installed,
    })
}

pub struct FileReporter {
    output_dir: PathBuf,
}

impl FileReporter {
    pub fn new(output_dir: PathBuf) -> Self {
        Self { output_dir }
    }
}

#[derive(Debug)]
struct OtlpReporter {
    exporter: opentelemetry_otlp::SpanExporter,
    scope: InstrumentationScope,
}

impl OtlpReporter {
    fn new(config: OtlpTraceConfig) -> Result<Self> {
        let mut builder = opentelemetry_otlp::SpanExporter::builder()
            .with_http()
            .with_protocol(Protocol::HttpBinary)
            .with_endpoint(config.endpoint.clone());
        if let Some(timeout) = config.timeout {
            builder = builder.with_timeout(timeout);
        }
        if !config.headers.is_empty() {
            builder = builder.with_headers(config.headers.into_iter().collect());
        }
        let mut exporter = builder
            .build()
            .map_err(|err| anyhow!("otlp trace exporter build failed: {err}"))?;
        exporter.set_resource(
            &Resource::builder_empty()
                .with_attributes([
                    KeyValue::new("service.name", config.service_name),
                    KeyValue::new("service.namespace", "agent-infer"),
                    KeyValue::new("telemetry.sdk.language", "rust"),
                ])
                .build(),
        );
        Ok(Self {
            exporter,
            scope: InstrumentationScope::builder("agent-infer.infer.fastrace")
                .with_version(env!("CARGO_PKG_VERSION"))
                .build(),
        })
    }
}

impl Reporter for OtlpReporter {
    fn report(&mut self, spans: Vec<SpanRecord>) {
        if spans.is_empty() {
            return;
        }
        let batch = spans
            .into_iter()
            .map(|span| span_record_to_otlp(span, &self.scope))
            .collect();
        if let Err(err) = pollster::block_on(self.exporter.export(batch)) {
            error!("Failed to export OTLP traces: {err}");
        }
    }
}

impl Drop for OtlpReporter {
    fn drop(&mut self) {
        if let Err(err) = self.exporter.shutdown() {
            error!("Failed to shutdown OTLP trace exporter: {err}");
        }
    }
}

#[derive(Default)]
struct MultiReporter {
    reporters: Vec<Box<dyn Reporter>>,
}

impl MultiReporter {
    fn with_reporter(mut self, reporter: impl Reporter) -> Self {
        self.reporters.push(Box::new(reporter));
        self
    }
}

impl Reporter for MultiReporter {
    fn report(&mut self, spans: Vec<SpanRecord>) {
        if spans.is_empty() || self.reporters.is_empty() {
            return;
        }
        let last = self.reporters.len() - 1;
        let mut owned = Some(spans);
        for (idx, reporter) in self.reporters.iter_mut().enumerate() {
            if idx == last {
                reporter.report(owned.take().expect("final reporter batch must exist"));
            } else {
                reporter.report(
                    owned
                        .as_ref()
                        .expect("shared reporter batch must exist")
                        .clone(),
                );
            }
        }
    }
}

/// Chrome Trace Event Format event.
/// Spec: https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU
#[derive(Serialize)]
struct TraceEvent {
    name: String,
    cat: String,
    ph: &'static str,
    /// Microseconds.
    ts: f64,
    /// Duration in microseconds (complete events only).
    #[serde(skip_serializing_if = "Option::is_none")]
    dur: Option<f64>,
    pid: u64,
    tid: u64,
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    args: HashMap<String, String>,
}

impl Reporter for FileReporter {
    fn report(&mut self, spans: Vec<SpanRecord>) {
        if spans.is_empty() {
            return;
        }

        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();

        for (trace_id, mut events) in spans_to_chrome_events(spans) {
            events.sort_by(|a, b| a.ts.partial_cmp(&b.ts).unwrap_or(std::cmp::Ordering::Equal));

            let filename = format!("{}_{}.json", timestamp_ms, trace_id);
            let path = self.output_dir.join(&filename);

            match serde_json::to_string_pretty(&events) {
                Ok(json) => {
                    if let Err(err) = std::fs::write(&path, json) {
                        error!("Failed to write trace file {}: {}", path.display(), err);
                    }
                }
                Err(err) => error!("Failed to serialize trace: {err}"),
            }
        }
    }
}

fn build_reporter(config: &TracingConfig) -> Result<MultiReporter> {
    let mut reporter = MultiReporter::default();
    if let Some(path) = &config.file_output {
        ensure_trace_dir(path)?;
        reporter = reporter.with_reporter(FileReporter::new(path.clone()));
    }
    if let Some(otlp) = &config.otlp {
        reporter = reporter.with_reporter(OtlpReporter::new(otlp.clone())?);
    }
    Ok(reporter)
}

fn ensure_trace_dir(path: &Path) -> Result<()> {
    std::fs::create_dir_all(path)
        .with_context(|| format!("failed to create trace output directory {}", path.display()))
}

fn spans_to_chrome_events(spans: Vec<SpanRecord>) -> HashMap<String, Vec<TraceEvent>> {
    let mut traces: HashMap<String, Vec<TraceEvent>> = HashMap::new();

    for span in spans {
        let trace_id = format!("{}", span.trace_id);
        let mut args = HashMap::new();
        args.insert("span_id".into(), format!("{}", span.span_id));
        args.insert("parent_id".into(), format!("{}", span.parent_id));
        for (key, value) in &span.properties {
            args.insert(key.to_string(), value.to_string());
        }

        let entry = traces.entry(trace_id).or_default();

        for event in &span.events {
            let mut event_args = HashMap::new();
            for (key, value) in &event.properties {
                event_args.insert(key.to_string(), value.to_string());
            }
            entry.push(TraceEvent {
                name: event.name.to_string(),
                cat: "event".into(),
                ph: "i",
                ts: event.timestamp_unix_ns as f64 / 1000.0,
                dur: None,
                pid: 1,
                tid: 1,
                args: event_args,
            });
        }

        entry.push(TraceEvent {
            name: span.name.into_owned(),
            cat: "span".into(),
            ph: "X",
            ts: span.begin_time_unix_ns as f64 / 1000.0,
            dur: Some(span.duration_ns as f64 / 1000.0),
            pid: 1,
            tid: 1,
            args,
        });
    }

    traces
}

fn span_record_to_otlp(span: SpanRecord, scope: &InstrumentationScope) -> SpanData {
    let start_time = unix_ns_to_system_time(span.begin_time_unix_ns);
    let end_time = unix_ns_to_system_time(span.begin_time_unix_ns.saturating_add(span.duration_ns));
    let attributes = span
        .properties
        .into_iter()
        .map(|(key, value)| KeyValue::new(key.to_string(), value.to_string()))
        .collect();
    let mut events = SpanEvents::default();
    events.events = span
        .events
        .into_iter()
        .map(|event| {
            OtelEvent::new(
                event.name.to_string(),
                unix_ns_to_system_time(event.timestamp_unix_ns),
                event
                    .properties
                    .into_iter()
                    .map(|(key, value)| KeyValue::new(key.to_string(), value.to_string()))
                    .collect(),
                0,
            )
        })
        .collect();

    SpanData {
        span_context: OtelSpanContext::new(
            OtelTraceId::from(span.trace_id.0),
            OtelSpanId::from(span.span_id.0),
            TraceFlags::SAMPLED,
            false,
            TraceState::default(),
        ),
        parent_span_id: OtelSpanId::from(span.parent_id.0),
        parent_span_is_remote: false,
        span_kind: SpanKind::Internal,
        name: span.name,
        start_time,
        end_time,
        attributes,
        dropped_attributes_count: 0,
        events,
        links: SpanLinks::default(),
        status: Status::Unset,
        instrumentation_scope: scope.clone(),
    }
}

fn unix_ns_to_system_time(unix_ns: u64) -> SystemTime {
    UNIX_EPOCH + Duration::from_nanos(unix_ns)
}

fn encode_sample_rate(sample_rate: f64) -> u64 {
    (sample_rate * SAMPLE_SCALE as f64).round() as u64
}

fn env_parse_u64(name: &str) -> Option<u64> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
}

fn env_parse_f64(name: &str) -> Option<f64> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<f64>().ok())
}

fn parse_header_env(value: &str) -> Vec<(String, String)> {
    value
        .split(',')
        .filter_map(|entry| {
            let trimmed = entry.trim();
            if trimmed.is_empty() {
                return None;
            }
            let (key, value) = trimmed.split_once('=')?;
            let key = key.trim();
            let value = value.trim();
            (!key.is_empty()).then(|| (key.to_string(), value.to_string()))
        })
        .collect()
}

fn normalize_otlp_traces_endpoint(value: &str) -> String {
    if value.is_empty() || value.ends_with("/v1/traces") {
        return value.to_string();
    }
    if let Some(scheme_sep) = value.find("://") {
        let host_start = scheme_sep + 3;
        if value[host_start..].find('/').is_none() {
            return format!("{value}/v1/traces");
        }
        if value.ends_with('/') {
            return format!("{value}v1/traces");
        }
    }
    value.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_level_parses_aliases() {
        assert_eq!(TraceLevel::from_str("off").unwrap(), TraceLevel::Off);
        assert_eq!(TraceLevel::from_str("basic").unwrap(), TraceLevel::Basic);
        assert_eq!(TraceLevel::from_str("full").unwrap(), TraceLevel::Verbose);
    }

    #[test]
    fn tracing_config_defaults_to_off_without_sinks() {
        let config = TracingConfig::resolve(TraceStartupConfig::default()).unwrap();
        assert_eq!(config.level, TraceLevel::Off);
        assert!(config.sample_rate.abs() < f64::EPSILON);
        assert!(!config.has_sink());
    }

    #[test]
    fn tracing_config_auto_enables_basic_when_sink_present() {
        let config = TracingConfig::resolve(TraceStartupConfig {
            file_output: Some(PathBuf::from("/tmp/infer-traces")),
            ..TraceStartupConfig::default()
        })
        .unwrap();
        assert_eq!(config.level, TraceLevel::Basic);
        assert!((config.sample_rate - 1.0).abs() < f64::EPSILON);
        assert!(config.has_sink());
    }

    #[test]
    fn tracing_config_rejects_missing_sink_for_enabled_trace() {
        let err = TracingConfig::resolve(TraceStartupConfig {
            level: Some("verbose".into()),
            ..TraceStartupConfig::default()
        })
        .unwrap_err();
        assert!(err.to_string().contains("without a sink"));
    }

    #[test]
    fn trace_runtime_sampling_boundaries_hold() {
        let runtime = TraceRuntime::disabled();
        runtime.set_level(TraceLevel::Basic);
        runtime.set_sample_rate(0.0);
        assert!(!runtime.decide_request("abc").sampled);

        runtime.set_sample_rate(1.0);
        assert!(runtime.decide_request("abc").sampled);
        assert_eq!(
            runtime.decide_request("abc").effective_level(),
            TraceLevel::Basic
        );
    }

    #[test]
    fn normalize_otlp_traces_endpoint_adds_default_path() {
        assert_eq!(
            normalize_otlp_traces_endpoint("http://127.0.0.1:4318"),
            "http://127.0.0.1:4318/v1/traces"
        );
        assert_eq!(
            normalize_otlp_traces_endpoint("http://127.0.0.1:4318/custom"),
            "http://127.0.0.1:4318/custom"
        );
    }

    #[test]
    fn parse_header_env_skips_invalid_segments() {
        assert_eq!(
            parse_header_env("a=1, broken, b = 2 "),
            vec![
                ("a".to_string(), "1".to_string()),
                ("b".to_string(), "2".to_string())
            ]
        );
    }
}
