//! Live tokens-per-second meter for streaming chat turns.
//!
//! Maintains a rolling token count + start-time. Exposes a refresh cadence
//! gate (~250ms wall-clock) and formats live + final status strings.
//!
//! Rendering policy:
//! - The live line is written to stderr during quiet stretches of the
//!   stream (caller invokes `maybe_refresh` only when no new tokens are
//!   arriving at this tick).
//! - Before the caller prints the next token chunk, it invokes
//!   `hide_before_chunk` to erase the status line in place with
//!   `\r\x1b[K` — this keeps streamed text clean on the same row the
//!   status briefly occupied.
//! - On non-TTY stderr, every live operation is a no-op; only the final
//!   summary prints (as plain text on its own line).

use std::io::{self, IsTerminal, Write};
use std::time::{Duration, Instant};

use console::Style;

/// Minimum wall-clock interval between live refreshes.
pub(crate) const REFRESH_INTERVAL: Duration = Duration::from_millis(250);

pub(crate) struct TpsMeter {
    start: Instant,
    tokens: u64,
    last_refresh: Option<Instant>,
    tty: bool,
    live_visible: bool,
}

impl TpsMeter {
    pub(crate) fn new() -> Self {
        Self {
            start: Instant::now(),
            tokens: 0,
            last_refresh: None,
            tty: io::stderr().is_terminal() && io::stdout().is_terminal(),
            live_visible: false,
        }
    }

    /// Increment the token counter (one call per received delta text chunk).
    pub(crate) fn record_chunk(&mut self, chars: usize) {
        // We don't have token boundaries from the stream; count chunks as a
        // rough proxy. This is fine for a UX indicator — the final summary
        // prefers `TokenUsage::completion_tokens` when populated.
        if chars > 0 {
            self.tokens = self.tokens.saturating_add(1);
        }
    }

    /// Should we refresh the live line now?
    fn should_refresh(&self, now: Instant) -> bool {
        if !self.tty {
            return false;
        }
        match self.last_refresh {
            None => true,
            Some(prev) => now.duration_since(prev) >= REFRESH_INTERVAL,
        }
    }

    /// Refresh the live line on stderr if the cadence gate says yes.
    ///
    /// The caller must invoke `hide_before_chunk` before printing any
    /// token text to stdout after calling this — otherwise the status
    /// text will be left-as-prefix on the stdout line.
    pub(crate) fn maybe_refresh(&mut self) {
        let now = Instant::now();
        if !self.should_refresh(now) {
            return;
        }
        self.last_refresh = Some(now);
        let elapsed = now.duration_since(self.start);
        let line = format_live(self.tokens, elapsed);
        let mut stderr = io::stderr();
        // `\r\x1b[K`: CR + clear-to-EOL. Write the status in place; we
        // never advance past the current line, so the next token chunk
        // overwrites from column 0 once `hide_before_chunk` is called.
        let _ = write!(stderr, "\r\x1b[K{line}");
        let _ = stderr.flush();
        self.live_visible = true;
    }

    /// Erase the live line in place so the next stdout write starts
    /// clean. Called from the caller right before printing a token chunk.
    pub(crate) fn hide_before_chunk(&mut self) {
        if !self.live_visible {
            return;
        }
        let mut stderr = io::stderr();
        let _ = write!(stderr, "\r\x1b[K");
        let _ = stderr.flush();
        self.live_visible = false;
    }

    /// Print the final summary to stderr on its own line.
    ///
    /// Uses `final_tokens` when provided (engine-reported completion_tokens),
    /// otherwise the rolling counter.
    pub(crate) fn print_final(&mut self, final_tokens: Option<u64>) {
        self.hide_before_chunk();
        let tokens = final_tokens.unwrap_or(self.tokens);
        let elapsed = self.start.elapsed();
        let line = format_final(tokens, elapsed);
        let mut stderr = io::stderr();
        let _ = writeln!(stderr, "{line}");
        let _ = stderr.flush();
    }
}

fn tps(tokens: u64, elapsed: Duration) -> f64 {
    let secs = elapsed.as_secs_f64();
    if secs <= f64::EPSILON {
        0.0
    } else {
        tokens as f64 / secs
    }
}

/// Live status: `▎ 43 tok/s · 0.8s` (dim gray on TTY).
pub(crate) fn format_live(tokens: u64, elapsed: Duration) -> String {
    let rate = tps(tokens, elapsed);
    let raw = format!("▎ {:.0} tok/s · {:.1}s", rate, elapsed.as_secs_f64());
    Style::new().dim().apply_to(raw).to_string()
}

/// Final summary: `▎ 128 tok / 2.9s · 44.1 tok/s` (dim gray on TTY).
pub(crate) fn format_final(tokens: u64, elapsed: Duration) -> String {
    let rate = tps(tokens, elapsed);
    let raw = format!(
        "▎ {} tok / {:.1}s · {:.1} tok/s",
        tokens,
        elapsed.as_secs_f64(),
        rate
    );
    Style::new().dim().apply_to(raw).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strip_ansi(s: &str) -> String {
        // Tests may run with TERM unset; console may or may not emit
        // escapes. Strip CSI sequences (ESC [ ... <letter>) while keeping
        // multi-byte Unicode chars intact.
        let mut out = String::with_capacity(s.len());
        let mut chars = s.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '\u{1b}' && chars.peek() == Some(&'[') {
                chars.next(); // consume '['
                while let Some(&next) = chars.peek() {
                    chars.next();
                    if next.is_ascii_alphabetic() {
                        break;
                    }
                }
                continue;
            }
            out.push(c);
        }
        out
    }

    #[test]
    fn format_tps_summary_matches_spec() {
        // 128 tok over 2.9s => ~44.1 tok/s
        let line = strip_ansi(&format_final(128, Duration::from_millis(2_900)));
        assert_eq!(line, "▎ 128 tok / 2.9s · 44.1 tok/s");
    }

    #[test]
    fn format_tps_live_matches_spec() {
        // 43 tokens at 1.0s => 43 tok/s, format rate as integer.
        let line = strip_ansi(&format_live(43, Duration::from_millis(1_000)));
        // Elapsed rounds to 1.0s; rate to 43.
        assert_eq!(line, "▎ 43 tok/s · 1.0s");
    }

    #[test]
    fn format_tps_zero_elapsed_is_zero() {
        let line = strip_ansi(&format_live(0, Duration::ZERO));
        assert_eq!(line, "▎ 0 tok/s · 0.0s");
    }

    #[test]
    fn record_chunk_increments_counter() {
        let mut m = TpsMeter::new();
        m.record_chunk(3);
        m.record_chunk(0); // zero-length is ignored
        m.record_chunk(1);
        assert_eq!(m.tokens, 2);
    }
}
