//! Tokens-per-second summary for streaming turns.
//!
//! Maintains a rolling token count + start-time and prints one final status
//! line to stderr at end-of-turn.

use std::io::{self, Write};
use std::time::{Duration, Instant};

use console::Style;

pub(crate) struct TpsMeter {
    start: Instant,
    tokens: u64,
    live_visible: bool,
}

impl TpsMeter {
    pub(crate) fn new() -> Self {
        Self {
            start: Instant::now(),
            tokens: 0,
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
    fn record_chunk_increments_counter() {
        let mut m = TpsMeter::new();
        m.record_chunk(3);
        m.record_chunk(0); // zero-length is ignored
        m.record_chunk(1);
        assert_eq!(m.tokens, 2);
    }
}
