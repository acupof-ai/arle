//! Metal runtime toggles: `--flag` → `EngineLoadConfig.metal` →
//! [`apply_runtime_flags`] once before executor construction. The statics are
//! the single truth — no env reads.

use std::sync::LazyLock;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};

use infer_seam::MetalRuntimeFlags;

static PIPELINE: AtomicBool = AtomicBool::new(true);
static WARMUP: AtomicBool = AtomicBool::new(false);
static PAGED_KV_READ: AtomicBool = AtomicBool::new(true);
static HOST_SAMPLING: AtomicBool = AtomicBool::new(false);
/// Speculative-decode resolver inputs (draft model / depth / accept width).
static SPEC: LazyLock<Mutex<MetalRuntimeFlags>> =
    LazyLock::new(|| Mutex::new(MetalRuntimeFlags::default()));

/// Must run before `MetalExecutor` construction.
pub fn apply_runtime_flags(f: &MetalRuntimeFlags) {
    PIPELINE.store(f.pipeline, Relaxed);
    WARMUP.store(f.warmup, Relaxed);
    PAGED_KV_READ.store(f.paged_kv_read, Relaxed);
    HOST_SAMPLING.store(f.host_sampling, Relaxed);
    *SPEC.lock().expect("metal runtime flags lock") = f.clone();
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(crate) fn pipeline() -> bool {
    PIPELINE.load(Relaxed)
}
#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(crate) fn warmup() -> bool {
    WARMUP.load(Relaxed)
}
#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(crate) fn paged_kv_read() -> bool {
    PAGED_KV_READ.load(Relaxed)
}
#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(crate) fn host_sampling() -> bool {
    HOST_SAMPLING.load(Relaxed)
}
#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(crate) fn spec_flags() -> MetalRuntimeFlags {
    SPEC.lock().expect("metal runtime flags lock").clone()
}
