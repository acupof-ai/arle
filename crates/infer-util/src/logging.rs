//! Unified logging configuration for Rust LLM.
//!
//! Provides consistent log initialization across server and tests.

use colored::Color::{Green, Red, Yellow};
use logforth::diagnostic::ThreadLocalDiagnostic;
use logforth::layout::TextLayout;
use std::sync::Once;

static INIT: Once = Once::new();

#[derive(Debug, Clone)]
struct LoggingConfig {
    /// Log level filter (e.g., "info", "debug", "info,infer=debug").
    /// Falls back to RUST_LOG environment variable if set.
    level: String,
    /// Enable colored output (info=green, warn=yellow, error=red).
    colored: bool,
    /// Fixed prefix prepended to every formatted record (e.g. `"[rank0] "`),
    /// so interleaved multi-process stderr stays attributable.
    prefix: Option<String>,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            colored: true,
            prefix: None,
        }
    }
}

/// Layout decorator: prepends [`LoggingConfig::prefix`] to whatever the
/// wrapped layout formats.
#[derive(Debug)]
struct PrefixedLayout {
    prefix: String,
    inner: Box<dyn logforth::Layout>,
}

impl logforth::Layout for PrefixedLayout {
    fn format(
        &self,
        record: &logforth::record::Record,
        diags: &[Box<dyn logforth::Diagnostic>],
    ) -> Result<Vec<u8>, logforth::Error> {
        let mut out = self.prefix.clone().into_bytes();
        out.extend(self.inner.format(record, diags)?);
        Ok(out)
    }
}

/// Default noisy modules to reduce log spam.
const DEFAULT_NOISY_MODULE_LEVELS: [(&str, &str); 5] = [
    ("h2", "warn"),
    ("hyper", "warn"),
    ("hyper_util", "warn"),
    ("axum", "warn"),
    ("tower", "warn"),
];

fn apply_default_module_levels(mut filter: String) -> String {
    let missing = DEFAULT_NOISY_MODULE_LEVELS
        .iter()
        .filter(|(module, _)| !filter.contains(&format!("{module}=")))
        .map(|(module, level)| format!("{module}={level}"))
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        if !filter.is_empty() {
            filter.push(',');
        }
        filter.push_str(&missing.join(","));
    }
    filter
}

/// Initialize logging with the given configuration.
///
/// This function is idempotent - subsequent calls after the first are no-ops.
/// The RUST_LOG environment variable takes precedence over the configured level.
/// When RUST_LOG is not set, noisy dependency modules default to warn to
/// keep debug output focused on application components.
fn init(config: LoggingConfig) {
    INIT.call_once(|| {
        let LoggingConfig {
            level,
            colored,
            prefix,
        } = config;

        let filter_str =
            std::env::var("RUST_LOG").unwrap_or_else(|_| apply_default_module_levels(level));

        // Parse filter from string using EnvFilterBuilder
        let filter =
            logforth::filter::env_filter::EnvFilterBuilder::from_env_or("RUST_LOG", filter_str)
                .build();

        // No prefix keeps both layouts byte-identical to before (the plain
        // layout IS `Stderr::default()`'s built-in layout).
        let layout: Box<dyn logforth::Layout> = if colored {
            Box::new(
                TextLayout::default()
                    .info_color(Green)
                    .warn_color(Yellow)
                    .error_color(Red),
            )
        } else {
            Box::new(logforth::layout::PlainTextLayout::default())
        };
        let layout = match prefix {
            Some(prefix) => Box::new(PrefixedLayout {
                prefix,
                inner: layout,
            }) as Box<dyn logforth::Layout>,
            None => layout,
        };

        logforth::starter_log::builder()
            .dispatch(|d| {
                d.filter(filter)
                    .diagnostic(ThreadLocalDiagnostic::default())
                    .append(logforth::append::Stderr::default().with_layout(layout))
            })
            .apply();
    });
}

/// Initialize logging to stderr without colors.
///
/// Convenience function for tests use case.
pub fn init_stderr(level: &str) {
    init(LoggingConfig {
        level: level.to_string(),
        colored: false,
        prefix: None,
    });
}

/// Initialize stderr logging (no colors) with a fixed per-record prefix, e.g.
/// `"[rank2] "` — multiproc workers share one terminal, so every record must
/// name its rank to stay attributable.
pub fn init_stderr_with_prefix(level: &str, prefix: &str) {
    init(LoggingConfig {
        level: level.to_string(),
        colored: false,
        prefix: Some(prefix.to_string()),
    });
}

/// Initialize logging with default settings (stderr, colored, "info" level).
pub fn init_default() {
    init(LoggingConfig::default());
}
