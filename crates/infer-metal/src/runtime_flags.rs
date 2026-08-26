//! Metal runtime toggles: `--flag` → `EngineLoadConfig.metal` →
//! [`apply_runtime_flags`] once before executor construction. The statics are
//! the single truth — no env reads.

use std::sync::LazyLock;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};

use infer_seam::MetalRuntimeFlags;

static WARMUP: AtomicBool = AtomicBool::new(false);
/// Speculative-decode resolver inputs (draft model / depth / accept width).
static SPEC: LazyLock<Mutex<MetalRuntimeFlags>> =
    LazyLock::new(|| Mutex::new(MetalRuntimeFlags::default()));

/// Must run before `MetalExecutor` construction.
pub fn apply_runtime_flags(f: &MetalRuntimeFlags) {
    WARMUP.store(f.warmup, Relaxed);
    *SPEC.lock().expect("metal runtime flags lock") = f.clone();
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(crate) fn warmup() -> bool {
    WARMUP.load(Relaxed)
}
#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(crate) fn spec_flags() -> MetalRuntimeFlags {
    SPEC.lock().expect("metal runtime flags lock").clone()
}
