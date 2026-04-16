//! Scheduler observability: per-step metrics, latency tracking, and stats export.
//!
//! Lightweight counters updated on the scheduler thread — no locks needed.

use std::time::{Duration, Instant};

/// Rolling window metrics for scheduler observability.
pub struct SchedulerMetrics {
    // ── Step-level timing ──
    pub total_steps: u64,
    pub total_decode_steps: u64,
    pub total_prefill_chunks: u64,

    // ── Latency accumulators (reset periodically) ──
    window_start: Instant,
    window_decode_us: u64,
    window_prefill_us: u64,
    window_steps: u64,
    window_decode_steps: u64,
    window_tokens_generated: u64,
    window_tokens_prefilled: u64,

    // ── Peak tracking ──
    pub peak_batch_size: usize,
    pub peak_step_us: u64,

    // ── Lifetime totals ──
    pub total_tokens_generated: u64,
    pub total_tokens_prefilled: u64,
}

impl Default for SchedulerMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl SchedulerMetrics {
    pub fn new() -> Self {
        Self {
            total_steps: 0,
            total_decode_steps: 0,
            total_prefill_chunks: 0,
            window_start: Instant::now(),
            window_decode_us: 0,
            window_prefill_us: 0,
            window_steps: 0,
            window_decode_steps: 0,
            window_tokens_generated: 0,
            window_tokens_prefilled: 0,
            peak_batch_size: 0,
            peak_step_us: 0,
            total_tokens_generated: 0,
            total_tokens_prefilled: 0,
        }
    }

    /// Record a decode step.
    pub fn record_decode(&mut self, batch_size: usize, duration_us: u64, tokens: u64) {
        self.total_steps += 1;
        self.total_decode_steps += 1;
        self.window_steps += 1;
        self.window_decode_steps += 1;
        self.window_decode_us += duration_us;
        self.window_tokens_generated += tokens;
        self.total_tokens_generated += tokens;
        if batch_size > self.peak_batch_size {
            self.peak_batch_size = batch_size;
        }
        if duration_us > self.peak_step_us {
            self.peak_step_us = duration_us;
        }
    }

    /// Record a prefill chunk.
    pub fn record_prefill(&mut self, chunk_tokens: usize, duration_us: u64) {
        self.total_prefill_chunks += 1;
        self.window_prefill_us += duration_us;
        self.window_tokens_prefilled += chunk_tokens as u64;
        self.total_tokens_prefilled += chunk_tokens as u64;
    }

    /// Get window stats and reset the window. Returns None if window is too short.
    pub fn drain_window(&mut self) -> Option<WindowStats> {
        let elapsed = self.window_start.elapsed();
        if elapsed < Duration::from_secs(1) || self.window_steps == 0 {
            return None;
        }

        let stats = WindowStats {
            elapsed,
            steps: self.window_steps,
            decode_steps: self.window_decode_steps,
            avg_decode_us: self
                .window_decode_us
                .checked_div(self.window_decode_steps)
                .unwrap_or(0),
            avg_prefill_us: self
                .window_prefill_us
                .checked_div(self.total_prefill_chunks)
                .unwrap_or(0),
            tokens_generated: self.window_tokens_generated,
            tokens_prefilled: self.window_tokens_prefilled,
            gen_throughput: self.window_tokens_generated as f64 / elapsed.as_secs_f64(),
            prefill_throughput: self.window_tokens_prefilled as f64 / elapsed.as_secs_f64(),
        };

        // Reset window
        self.window_start = Instant::now();
        self.window_decode_us = 0;
        self.window_prefill_us = 0;
        self.window_steps = 0;
        self.window_decode_steps = 0;
        self.window_tokens_generated = 0;
        self.window_tokens_prefilled = 0;

        Some(stats)
    }
}

/// Snapshot of metrics over a time window.
pub struct WindowStats {
    pub elapsed: Duration,
    pub steps: u64,
    pub decode_steps: u64,
    pub avg_decode_us: u64,
    pub avg_prefill_us: u64,
    pub tokens_generated: u64,
    pub tokens_prefilled: u64,
    pub gen_throughput: f64,
    pub prefill_throughput: f64,
}

impl std::fmt::Display for WindowStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "steps={} decode_avg={:.0}us prefill_avg={:.0}us gen={:.0}tok/s prefill={:.0}tok/s",
            self.steps,
            self.avg_decode_us,
            self.avg_prefill_us,
            self.gen_throughput,
            self.prefill_throughput,
        )
    }
}
