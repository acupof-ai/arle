#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{OnceLock, RwLock};

use anyhow::{Result, anyhow};
use cuda_kernels::prelude::DeviceContext;
use cudarc::driver::sys::CUevent_flags;

#[derive(Default)]
pub struct OpStats {
    pub total_cuda_micros: AtomicU64,
    pub count: AtomicU64,
}

static ENABLED: OnceLock<bool> = OnceLock::new();
static STATS: OnceLock<RwLock<HashMap<String, OpStats>>> = OnceLock::new();

fn enabled() -> bool {
    *ENABLED.get_or_init(|| std::env::var_os("ARLE_CUDA_PROFILE").is_some())
}

fn stats() -> &'static RwLock<HashMap<String, OpStats>> {
    STATS.get_or_init(|| RwLock::new(HashMap::new()))
}

pub fn profile_op<T>(
    ctx: &DeviceContext,
    name: &str,
    layer_idx: Option<usize>,
    seq_len: usize,
    f: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let _nvtx = if crate::nvtx::is_enabled() {
        let label = match layer_idx {
            Some(idx) => format!("{name}_layer{idx} seq={seq_len}"),
            None => format!("{name} seq={seq_len}"),
        };
        Some(crate::nvtx::range(&label))
    } else {
        None
    };

    if !enabled() {
        return f();
    }

    let stats_key = match layer_idx {
        Some(idx) => format!("{name}_layer{idx}"),
        None => name.to_string(),
    };

    let start = ctx
        .ctx
        .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(|e| anyhow!("create CUDA profile start event for {stats_key} failed: {e}"))?;
    let stop = ctx
        .ctx
        .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(|e| anyhow!("create CUDA profile stop event for {stats_key} failed: {e}"))?;

    start
        .record(&ctx.stream)
        .map_err(|e| anyhow!("record CUDA profile start event for {stats_key} failed: {e}"))?;

    let result = f();

    stop.record(&ctx.stream)
        .map_err(|e| anyhow!("record CUDA profile stop event for {stats_key} failed: {e}"))?;
    stop.synchronize()
        .map_err(|e| anyhow!("sync CUDA profile stop event for {stats_key} failed: {e}"))?;
    let cuda_ms = start
        .elapsed_ms(&stop)
        .map_err(|e| anyhow!("elapsed CUDA profile event for {stats_key} failed: {e}"))?;
    let cuda_micros = (cuda_ms * 1000.0).round() as u64;

    {
        let read = stats().read().unwrap_or_else(|e| e.into_inner());
        if !read.contains_key(&stats_key) {
            drop(read);
            let mut write = stats().write().unwrap_or_else(|e| e.into_inner());
            write.entry(stats_key.clone()).or_default();
        }
    }
    let read = stats().read().unwrap_or_else(|e| e.into_inner());
    if let Some(entry) = read.get(&stats_key) {
        entry
            .total_cuda_micros
            .fetch_add(cuda_micros, Ordering::Relaxed);
        entry.count.fetch_add(1, Ordering::Relaxed);
    }

    result
}

pub fn get_op_stats() -> Vec<(String, u64, u64)> {
    let read = stats().read().unwrap_or_else(|e| e.into_inner());
    let mut out: Vec<(String, u64, u64)> = read
        .iter()
        .map(|(name, stat)| {
            (
                name.clone(),
                stat.total_cuda_micros.load(Ordering::Relaxed),
                stat.count.load(Ordering::Relaxed),
            )
        })
        .collect();
    out.sort_by_key(|b| std::cmp::Reverse(b.1));
    out
}
