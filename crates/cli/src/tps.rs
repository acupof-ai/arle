//! Tokens-per-second summary for streaming turns.
//!
//! Maintains a rolling token count + start-time and prints one final status
//! line to stderr at end-of-turn. The line surfaces three numbers per turn:
//! input throughput (prompt tokens / TTFT), TTFT itself, and output
//! throughput (completion tokens / elapsed).

use std::io::{self, Write};
use std::time::{Duration, Instant};

use console::Style;

pub(crate) struct TpsMeter {
    start: Instant,
    tokens: u64,
    live_visible: bool,
    /// Wall-clock instant of the first non-empty streamed chunk. Used to
    /// derive TTFT (= first_chunk_at − start) for the final summary.
    first_chunk_at: Option<Instant>,
}

impl TpsMeter {
    pub(crate) fn new() -> Self {
        Self {
            start: Instant::now(),
            tokens: 0,
            live_visible: false,
            first_chunk_at: None,
        }
    }

    /// Increment the token counter (one call per received delta text chunk).
    pub(crate) fn record_chunk(&mut self, chars: usize) {
        // We don't have token boundaries from the stream; count chunks as a
        // rough proxy. This is fine for a UX indicator — the final summary
        // prefers `TokenUsage::completion_tokens` when populated.
        if chars > 0 {
            self.tokens = self.tokens.saturating_add(1);
            if self.first_chunk_at.is_none() {
                self.first_chunk_at = Some(Instant::now());
            }
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
    /// Prefers `external_ttft` when supplied — that's the agent's
    /// engine-token TTFT, which catches turns that opened with a
    /// `<tool_call>` block (zero visible text, but the model still
    /// generated tokens). Falls back to the meter's own visible-text
    /// first-chunk capture when the caller passes `None`.
    pub(crate) fn print_final(
        &mut self,
        prompt_tokens: u64,
        final_tokens: Option<u64>,
        external_ttft: Option<Duration>,
    ) {
        self.hide_before_chunk();
        let completion = final_tokens.unwrap_or(self.tokens);
        let elapsed = self.start.elapsed();
        let ttft =
            external_ttft.or_else(|| self.first_chunk_at.map(|t| t.duration_since(self.start)));
        let line = format_final(prompt_tokens, ttft, completion, elapsed);
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

/// Format a duration tightly: sub-second values render as `420ms`, otherwise
/// as `1.4s`. Picks the unit a reader would pick by hand.
fn fmt_short(d: Duration) -> String {
    let secs = d.as_secs_f64();
    if secs >= 1.0 {
        format!("{secs:.1}s")
    } else {
        format!("{}ms", d.as_millis())
    }
}

/// Final summary line. Layout:
///   `▎ in 1234 tok · ttft 0.42s · 2941 tok/s   out 621 tok / 13.1s · 47.4 tok/s`
/// When prompt-side data is unavailable (no TTFT or zero prompt tokens) the
/// `in` segment collapses, leaving just the legacy `out` half.
pub(crate) fn format_final(
    prompt_tokens: u64,
    ttft: Option<Duration>,
    completion_tokens: u64,
    elapsed: Duration,
) -> String {
    let out_rate = tps(completion_tokens, elapsed);
    let out = format!(
        "out {} tok / {:.1}s · {:.1} tok/s",
        completion_tokens,
        elapsed.as_secs_f64(),
        out_rate,
    );
    let raw = match (prompt_tokens, ttft) {
        (p, Some(t)) if p > 0 => {
            let in_rate = tps(p, t);
            format!(
                "▎ in {} tok · ttft {} · {:.1} tok/s   {}",
                p,
                fmt_short(t),
                in_rate,
                out,
            )
        }
        _ => format!("▎ {}", out),
    };
    Style::new().dim().apply_to(raw).to_string()
}
