use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

use anyhow::{Result, anyhow};
use cuda_kernels::prelude::DeviceContext;
use cudarc::driver::sys::CUevent_flags;

#[derive(Clone, Copy, Debug, Default)]
struct Stat {
    calls: u64,
    total_ms: f64,
}

static ENABLED: OnceLock<bool> = OnceLock::new();
static STATS: OnceLock<Mutex<BTreeMap<&'static str, Stat>>> = OnceLock::new();

fn enabled() -> bool {
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("ARLE_DSV4_LINEAR_PROFILE").as_deref(),
            Ok("1" | "true" | "TRUE" | "yes" | "on" | "ON")
        )
    })
}

fn stats() -> &'static Mutex<BTreeMap<&'static str, Stat>> {
    STATS.get_or_init(|| Mutex::new(BTreeMap::new()))
}

pub(crate) fn reset() {
    if !enabled() {
        return;
    }
    if let Ok(mut guard) = stats().lock() {
        guard.clear();
    }
}

pub(crate) fn profile<T>(
    ctx: &DeviceContext,
    label: &'static str,
    f: impl FnOnce() -> Result<T>,
) -> Result<T> {
    if !enabled() {
        return f();
    }

    let start = ctx
        .ctx
        .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(|e| anyhow!("create CUDA profile start event for {label} failed: {e}"))?;
    let stop = ctx
        .ctx
        .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(|e| anyhow!("create CUDA profile stop event for {label} failed: {e}"))?;
    start
        .record(&ctx.stream)
        .map_err(|e| anyhow!("record CUDA profile start event for {label} failed: {e}"))?;
    let result = f();
    stop.record(&ctx.stream)
        .map_err(|e| anyhow!("record CUDA profile stop event for {label} failed: {e}"))?;
    stop.synchronize()
        .map_err(|e| anyhow!("sync CUDA profile stop event for {label} failed: {e}"))?;
    let ms = start
        .elapsed_ms(&stop)
        .map_err(|e| anyhow!("elapsed CUDA profile event for {label} failed: {e}"))?
        as f64;

    if let Ok(mut guard) = stats().lock() {
        let entry = guard.entry(label).or_default();
        entry.calls += 1;
        entry.total_ms += ms;
    }
    result
}

pub(crate) fn print_rank0(tag: &str) {
    // Print on rank 0 OR unset (the multiproc coordinator runs the forward but
    // is not spawned with INFER_TP_RANK). Mirrors stage_profile.
    let rank = std::env::var("INFER_TP_RANK").unwrap_or_default();
    if !enabled() || (!rank.is_empty() && rank != "0") {
        return;
    }
    let Ok(guard) = stats().lock() else {
        eprintln!("[dsv4-linear-profile tag={tag}] failed to lock stats");
        return;
    };
    let total_ms: f64 = guard.values().map(|s| s.total_ms).sum();
    eprintln!(
        "[dsv4-linear-profile tag={tag}] total_ms={total_ms:.3} labels={}",
        guard.len()
    );
    for (label, stat) in guard.iter() {
        let avg_us = if stat.calls == 0 {
            0.0
        } else {
            stat.total_ms * 1000.0 / stat.calls as f64
        };
        let pct = if total_ms > 0.0 {
            stat.total_ms * 100.0 / total_ms
        } else {
            0.0
        };
        eprintln!(
            "[dsv4-linear-profile tag={tag}] {label} calls={} total_ms={:.3} avg_us={avg_us:.3} pct_linear={pct:.2}",
            stat.calls, stat.total_ms
        );
    }
}
