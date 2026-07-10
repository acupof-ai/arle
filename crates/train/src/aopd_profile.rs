//! Per-round stage wall-clock profiler for the agent-OPD loop.
//!
//! Opt-in via the `ARLE_AOPD_PROFILE` env var (any value enables it). Zero-cost
//! when unset: every entry point early-returns on the `OnceLock<bool>` gate
//! before touching the mutex, and [`time`] calls the closure directly.
//!
//! All timings are HOST wall-clock. The GPU stages (rollout decode, masked-CE
//! writeback, LoRA sync, eval) block on the device — each materializes a host
//! value (sampled token, loss, D2H adapter read) that forces a stream sync — so
//! their wall IS GPU-inclusive; instrumenting CUDA events inside the frozen
//! executor would only re-measure the same interval. The `kind` column tags
//! each stage `GPU` / `WALL` / `DISK` so the reader sees where the time lives.
//!
//! The round summary sums every recorded stage against the round wall anchored
//! by [`begin_round`], and prints the residual as an `(untimed/overhead)` row so
//! the table accounts for ~100% of the round.

use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// Stage resource tag, printed verbatim in the `kind` column.
pub const GPU: &str = "GPU";
pub const WALL: &str = "WALL";
pub const DISK: &str = "DISK";

#[derive(Clone, Copy, Default)]
struct Stat {
    calls: u64,
    secs: f64,
}

struct Table {
    /// First-seen insertion order preserved (logical, not alphabetical).
    rows: Vec<(&'static str, &'static str, Stat)>,
    round_start: Option<Instant>,
}

static ENABLED: OnceLock<bool> = OnceLock::new();
static TABLE: OnceLock<Mutex<Table>> = OnceLock::new();

pub fn enabled() -> bool {
    *ENABLED.get_or_init(|| std::env::var_os("ARLE_AOPD_PROFILE").is_some())
}

fn table() -> &'static Mutex<Table> {
    TABLE.get_or_init(|| {
        Mutex::new(Table {
            rows: Vec::new(),
            round_start: None,
        })
    })
}

/// Anchor a fresh round: reset the stage table and start the round wall clock.
pub fn begin_round() {
    if !enabled() {
        return;
    }
    if let Ok(mut t) = table().lock() {
        t.rows.clear();
        t.round_start = Some(Instant::now());
    }
}

/// Accumulate one stage observation. `kind` is one of [`GPU`] / [`WALL`] /
/// [`DISK`]. Prefer [`time`] / [`time_try`] for closure-shaped call sites; use
/// this directly when the elapsed time is derived (e.g. decode wall split out of
/// a returned record).
pub fn record(label: &'static str, kind: &'static str, secs: f64) {
    if !enabled() {
        return;
    }
    if let Ok(mut t) = table().lock() {
        match t.rows.iter_mut().find(|(l, _, _)| *l == label) {
            Some((_, _, stat)) => {
                stat.calls += 1;
                stat.secs += secs;
            }
            None => t.rows.push((label, kind, Stat { calls: 1, secs })),
        }
    }
}

/// Time a closure and accumulate its wall under `label`. Returns the closure's
/// value unchanged. When profiling is off, calls `f` directly (zero overhead).
pub fn time<T>(label: &'static str, kind: &'static str, f: impl FnOnce() -> T) -> T {
    if !enabled() {
        return f();
    }
    let t0 = Instant::now();
    let out = f();
    record(label, kind, t0.elapsed().as_secs_f64());
    out
}

/// [`time`] specialized for `Result`-returning stages, so `?` chains stay flat.
pub fn time_try<T, E>(
    label: &'static str,
    kind: &'static str,
    f: impl FnOnce() -> Result<T, E>,
) -> Result<T, E> {
    time(label, kind, f)
}

/// Print the per-round stage breakdown table and reset the round clock. Columns:
/// `stage | kind | calls | total_ms | ms/call | %round`, with an
/// `(untimed/overhead)` residual row and a `TOTAL` row so the table sums to the
/// round wall.
pub fn print_round(round: usize) {
    if !enabled() {
        return;
    }
    let Ok(t) = table().lock() else {
        eprintln!("[aopd-profile round={round}] failed to lock stage table");
        return;
    };
    let round_ms = t
        .round_start
        .map_or(0.0, |s| s.elapsed().as_secs_f64() * 1000.0);
    let timed_ms: f64 = t.rows.iter().map(|(_, _, s)| s.secs * 1000.0).sum();
    let overhead_ms = (round_ms - timed_ms).max(0.0);

    let pct = |ms: f64| {
        if round_ms > 0.0 {
            ms * 100.0 / round_ms
        } else {
            0.0
        }
    };

    eprintln!(
        "[aopd-profile round={round}] stage breakdown (round wall={round_ms:.1} ms, {} stages)",
        t.rows.len()
    );
    eprintln!(
        "[aopd-profile round={round}]   {:<22} {:<5} {:>6} {:>12} {:>10} {:>8}",
        "stage", "kind", "calls", "total_ms", "ms/call", "%round"
    );
    for (label, kind, stat) in &t.rows {
        let total_ms = stat.secs * 1000.0;
        let per_call = if stat.calls > 0 {
            total_ms / stat.calls as f64
        } else {
            0.0
        };
        eprintln!(
            "[aopd-profile round={round}]   {label:<22} {kind:<5} {:>6} {total_ms:>12.1} {per_call:>10.2} {:>7.1}%",
            stat.calls,
            pct(total_ms)
        );
    }
    eprintln!(
        "[aopd-profile round={round}]   {:<22} {:<5} {:>6} {overhead_ms:>12.1} {:>10} {:>7.1}%",
        "(untimed/overhead)",
        "-",
        "-",
        "-",
        pct(overhead_ms)
    );
    eprintln!(
        "[aopd-profile round={round}]   {:<22} {:<5} {:>6} {round_ms:>12.1} {:>10} {:>7.1}%",
        "TOTAL", "-", "-", "-", 100.0
    );
}
