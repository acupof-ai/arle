use colored::Color::{Green, Red, Yellow};
use logforth::diagnostic::ThreadLocalDiagnostic;
use logforth::layout::TextLayout;
use std::num::NonZeroUsize;
use std::sync::Once;

static INIT: Once = Once::new();

#[derive(Debug, Clone)]
struct LoggingConfig {
    /// Log level filter; falls back to `RUST_LOG` if set.
    level: String,
    colored: bool,
    /// Fixed prefix prepended to every record, so interleaved multi-process
    /// stderr stays attributable.
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

/// Modules clamped to `warn` by default to reduce log spam.
const DEFAULT_NOISY_MODULE_LEVELS: [(&str, &str); 5] = [
    ("h2", "warn"),
    ("hyper", "warn"),
    ("hyper_util", "warn"),
    ("axum", "warn"),
    ("tower", "warn"),
];

/// Default-on rolling file sink, so a hung/killed process still leaves a log
/// on disk instead of only the terminal scrollback. `ARLE_LOG_DIR` overrides
/// the directory (default `"logs"`); set to `off`/`none`/`""` to disable.
/// One file per process — `prefix` (multiproc rank tag) becomes part of the
/// filename so ranks don't clobber each other.
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
    // `grep`/`tail`.
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

/// Idempotent — subsequent calls are no-ops.
///
/// `RUST_LOG` takes precedence over `config.level`. When `RUST_LOG` is unset,
/// noisy dependency modules default to `warn` to keep debug output focused on
/// application components.
fn init(config: LoggingConfig) {
    INIT.call_once(|| {
        let LoggingConfig {
            level,
            colored,
            prefix,
        } = config;

        let filter_str =
            std::env::var("RUST_LOG").unwrap_or_else(|_| apply_default_module_levels(level));

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
        // Both sinks share one background thread (bounded queue,
        // drop-on-overflow) so the caller thread — inference included — never
        // blocks on stderr/disk I/O. Formatting stays on the caller (cheap);
        // only the actual write is offloaded.
        let mut async_append = logforth::append::asynchronous::AsyncBuilder::new("arle-log")
            .buffered_lines_limit(Some(8192))
            .overflow_drop_incoming()
            .append(logforth::append::Stderr::default().with_layout(layout));
        if let Some(file_append) = build_file_append(&prefix) {
            async_append = async_append.append(file_append);
        }

        logforth::starter_log::builder()
            .dispatch(|d| {
                d.filter(filter)
                    .diagnostic(ThreadLocalDiagnostic::default())
                    .append(async_append.build())
            })
            .apply();
    });
}

/// For tests.
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
