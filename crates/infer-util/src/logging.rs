//! Unified logging configuration for Rust LLM.
//!
//! Provides consistent log initialization across server and tests.

use colored::Color::{Green, Red, Yellow};
use logforth::diagnostic::ThreadLocalDiagnostic;
use logforth::layout::TextLayout;
use std::num::NonZeroUsize;
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

/// Default-on rolling file sink, so a hung/killed process still leaves a log
/// on disk instead of only the tmux/terminal scrollback (pod round-7: a hang
/// investigation had no server log to inspect post-mortem because stderr was
/// never redirected). `ARLE_LOG_DIR` overrides the directory (default
/// `"logs"`); set it to `off`/`none`/`""` to disable. One file per process —
/// `prefix` (multiproc rank tag, e.g. `"[rank2] "`) becomes part of the
/// filename so ranks don't clobber each other's file.
fn build_file_append(prefix: &Option<String>) -> Option<Box<dyn logforth::Append>> {
    let dir = std::env::var("ARLE_LOG_DIR").unwrap_or_else(|_| "logs".to_string());
    if matches!(dir.as_str(), "" | "off" | "none") {
        return None;
    }
    let filename = match prefix {
        Some(p) => format!("arle-{}", p.trim().trim_matches(['[', ']', ' '])),
        None => "arle".to_string(),
    };
    // Plain text, never colored — ANSI escapes are noise in a file meant for
    // `grep`/`tail`. Prefix still applies so multi-rank lines stay attributable.
    let layout: Box<dyn logforth::Layout> = match prefix {
        Some(p) => Box::new(PrefixedLayout {
            prefix: p.clone(),
            inner: Box::new(logforth::layout::PlainTextLayout::default()),
        }),
        None => Box::new(logforth::layout::PlainTextLayout::default()),
    };
    match logforth::append::file::FileBuilder::new(&dir, filename)
        .layout(layout)
        .rollover_daily()
        .rollover_size(NonZeroUsize::new(256 * 1024 * 1024).expect("nonzero"))
        .max_log_files(NonZeroUsize::new(14).expect("nonzero"))
        .build()
    {
        Ok(file) => Some(Box::new(file)),
        Err(err) => {
            eprintln!("logging: file sink disabled, failed to open {dir}: {err}");
            None
        }
    }
}

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
        let layout = match &prefix {
            Some(prefix) => Box::new(PrefixedLayout {
                prefix: prefix.clone(),
                inner: layout,
            }) as Box<dyn logforth::Layout>,
            None => layout,
        };
        let file_append = build_file_append(&prefix);

        logforth::starter_log::builder()
            .dispatch(|d| {
                let d = d
                    .filter(filter)
                    .diagnostic(ThreadLocalDiagnostic::default())
                    .append(logforth::append::Stderr::default().with_layout(layout));
                match file_append {
                    Some(file_append) => d.append(file_append),
                    None => d,
                }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// `init()` is process-global (`Once`) — this must be the only test in
    /// this crate that initializes logging, so the file sink it sets up is
    /// observable exactly once.
    #[test]
    fn file_sink_writes_a_log_line_to_disk() {
        let dir = std::env::temp_dir().join(format!("arle-logging-test-{}", std::process::id()));
        // SAFETY: single-threaded test process, set before the one `init()` call below.
        unsafe { std::env::set_var("ARLE_LOG_DIR", &dir) };
        init_stderr("info");
        log::info!("logging file-sink self-check");
        for _ in 0..50 {
            if std::fs::read_dir(&dir).is_ok_and(|mut d| d.next().is_some()) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        let entries: Vec<_> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("ARLE_LOG_DIR {dir:?} was never created: {e}"))
            .filter_map(|e| e.ok())
            .collect();
        assert!(!entries.is_empty(), "no log file written under {dir:?}");
        let contents = std::fs::read_to_string(entries[0].path()).unwrap();
        assert!(
            contents.contains("logging file-sink self-check"),
            "log file missing the test record: {contents:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
