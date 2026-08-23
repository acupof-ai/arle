use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use anyhow::{Result, anyhow};
use cuda_kernels::prelude::DeviceContext;
use cudarc::driver::sys::CUevent_flags;

#[derive(Clone, Copy, Debug, Default)]
struct Stat {
    calls: u64,
    cuda_ms: f64,
    host_ms: f64,
}

static ENABLED: OnceLock<bool> = OnceLock::new();
static ACTIVE: AtomicBool = AtomicBool::new(false);
static STATS: OnceLock<Mutex<BTreeMap<&'static str, Stat>>> = OnceLock::new();

fn enabled() -> bool {
    *ENABLED.get_or_init(|| std::env::var_os("ARLE_DSV4_STAGE_PROFILE").is_some())
}

fn stats() -> &'static Mutex<BTreeMap<&'static str, Stat>> {
    STATS.get_or_init(|| Mutex::new(BTreeMap::new()))
}

pub(crate) fn reset() {
    if !enabled() {
        return;
    }
    ACTIVE.store(false, Ordering::Relaxed);
    if let Ok(mut guard) = stats().lock() {
        guard.clear();
    }
}

pub(crate) fn set_active(active: bool) {
    if enabled() {
        ACTIVE.store(active, Ordering::Relaxed);
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
    let _ = ACTIVE.load(Ordering::Relaxed); // retained for explicit-window callers

    let start = ctx
        .ctx
        .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(|e| anyhow!("create CUDA stage-profile start event for {label} failed: {e}"))?;
    let stop = ctx
        .ctx
        .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(|e| anyhow!("create CUDA stage-profile stop event for {label} failed: {e}"))?;
    start
        .record(&ctx.stream)
        .map_err(|e| anyhow!("record CUDA stage-profile start event for {label} failed: {e}"))?;
    let host_t0 = Instant::now();
    let result = f();
    let host_ms = host_t0.elapsed().as_secs_f64() * 1000.0;
    stop.record(&ctx.stream)
        .map_err(|e| anyhow!("record CUDA stage-profile stop event for {label} failed: {e}"))?;
    stop.synchronize()
        .map_err(|e| anyhow!("sync CUDA stage-profile stop event for {label} failed: {e}"))?;
    let cuda_ms = start
        .elapsed_ms(&stop)
        .map_err(|e| anyhow!("elapsed CUDA stage-profile event for {label} failed: {e}"))?
        as f64;

    if let Ok(mut guard) = stats().lock() {
        let entry = guard.entry(label).or_default();
        entry.calls += 1;
        entry.cuda_ms += cuda_ms;
        entry.host_ms += host_ms;
    }
    result
}

pub(crate) fn print_rank0(tag: &str, timed_tokens: usize, timed_wall_ms: f64) {
    // Print on the visible-output rank: INFER_TP_RANK="0" (explicit) OR unset
    // (the multiproc COORDINATOR is the parent process — it runs the forward +
    // sample but is not spawned with INFER_TP_RANK, so an unset value is rank 0).
    let rank = std::env::var("INFER_TP_RANK").unwrap_or_default();
    if !enabled() || (!rank.is_empty() && rank != "0") {
        return;
    }
    let Ok(guard) = stats().lock() else {
        eprintln!("[dsv4-stage-profile tag={tag}] failed to lock stats");
        return;
    };
    let total_cuda_ms: f64 = guard.values().map(|s| s.cuda_ms).sum();
    let total_host_ms: f64 = guard.values().map(|s| s.host_ms).sum();
    let wall_ms_per_token = if timed_tokens > 0 {
        timed_wall_ms / timed_tokens as f64
    } else {
        0.0
    };
    let cuda_ms_per_token = if timed_tokens > 0 {
        total_cuda_ms / timed_tokens as f64
    } else {
        0.0
    };
    let host_ms_per_token = if timed_tokens > 0 {
        total_host_ms / timed_tokens as f64
    } else {
        0.0
    };
    let unaccounted_cuda_ms = (timed_wall_ms - total_cuda_ms).max(0.0);
    eprintln!(
        "[dsv4-stage-profile tag={tag}] labels={} timed_tokens={} timed_wall_ms={timed_wall_ms:.3} wall_ms_per_token={wall_ms_per_token:.3} total_cuda_ms={total_cuda_ms:.3} cuda_ms_per_token={cuda_ms_per_token:.3} total_host_ms={total_host_ms:.3} host_ms_per_token={host_ms_per_token:.3} unaccounted_cuda_ms={unaccounted_cuda_ms:.3}",
        guard.len(),
        timed_tokens,
    );
    for (label, stat) in guard.iter() {
        let cuda_ms_per_token = if timed_tokens > 0 {
            stat.cuda_ms / timed_tokens as f64
        } else {
            0.0
        };
        let host_ms_per_token = if timed_tokens > 0 {
            stat.host_ms / timed_tokens as f64
        } else {
            0.0
        };
        let pct_wall_cuda = if timed_wall_ms > 0.0 {
            stat.cuda_ms * 100.0 / timed_wall_ms
        } else {
            0.0
        };
        let pct_wall_host = if timed_wall_ms > 0.0 {
            stat.host_ms * 100.0 / timed_wall_ms
        } else {
            0.0
        };
        eprintln!(
            "[dsv4-stage-profile tag={tag}] {label} calls={} cuda_ms={:.3} cuda_ms_per_token={cuda_ms_per_token:.3} pct_wall_cuda={pct_wall_cuda:.2} host_ms={:.3} host_ms_per_token={host_ms_per_token:.3} pct_wall_host={pct_wall_host:.2}",
            stat.calls, stat.cuda_ms, stat.host_ms
        );
    }
}
