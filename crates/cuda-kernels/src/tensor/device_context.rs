//! CUDA device context, pipeline fences, and allocation tracing.

use anyhow::{Result, anyhow, ensure};
use cudarc::driver::{
    CudaContext, CudaEvent, CudaSlice, CudaStream, DevicePtrMut, DeviceRepr, DriverError,
    PinnedHostSlice, ValidAsZeroBits,
};
use std::any::type_name;
use std::collections::BTreeMap;
use std::panic::Location;
use std::sync::{Arc, LazyLock, Mutex, OnceLock};

use crate::ffi;

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct CudaAllocTraceKey {
    pub file: &'static str,
    pub line: u32,
    pub column: u32,
    pub kind: &'static str,
    pub label: &'static str,
    pub type_name: &'static str,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaAllocTraceStats {
    pub calls: u64,
    pub bytes: u64,
}

static CUDA_ALLOC_TRACE_ENABLED: OnceLock<bool> = OnceLock::new();
static CUDA_ALLOC_TRACE: LazyLock<Mutex<BTreeMap<CudaAllocTraceKey, CudaAllocTraceStats>>> =
    LazyLock::new(|| Mutex::new(BTreeMap::new()));

fn cuda_alloc_trace_enabled() -> bool {
    *CUDA_ALLOC_TRACE_ENABLED.get_or_init(|| {
        matches!(
            std::env::var("ARLE_CUDA_ALLOC_TRACE").as_deref(),
            Ok("1" | "true" | "TRUE" | "yes" | "on" | "ON")
        )
    })
}

#[track_caller]
pub(super) fn record_cuda_alloc<T>(kind: &'static str, label: &'static str, len: usize) {
    if !cuda_alloc_trace_enabled() {
        return;
    }
    let location = Location::caller();
    let key = CudaAllocTraceKey {
        file: location.file(),
        line: location.line(),
        column: location.column(),
        kind,
        label,
        type_name: type_name::<T>(),
    };
    let bytes = len.saturating_mul(std::mem::size_of::<T>()) as u64;
    let Ok(mut trace) = CUDA_ALLOC_TRACE.lock() else {
        return;
    };
    let stats = trace.entry(key).or_default();
    stats.calls = stats.calls.saturating_add(1);
    stats.bytes = stats.bytes.saturating_add(bytes);
}

pub trait CudaAllocTraceExt {
    /// Allocate and attribute the call site when `ARLE_CUDA_ALLOC_TRACE=1`.
    ///
    /// # Safety
    ///
    /// Same as [`CudaStream::alloc`]: the returned memory is uninitialized.
    unsafe fn alloc_traced<T: DeviceRepr>(&self, len: usize) -> Result<CudaSlice<T>, DriverError>;
}

impl CudaAllocTraceExt for Arc<CudaStream> {
    #[track_caller]
    unsafe fn alloc_traced<T: DeviceRepr>(&self, len: usize) -> Result<CudaSlice<T>, DriverError> {
        // SAFETY: forwards `CudaStream::alloc`'s uninitialized-memory contract
        // to our own caller (this method is `unsafe` with the same `# Safety`).
        let out = unsafe { self.alloc(len)? };
        record_cuda_alloc::<T>("alloc", "CudaStream::alloc", len);
        Ok(out)
    }
}

/// CUDA device context holding compute stream and optional copy stream.
///
/// Three-stream architecture for overlapping H2D/D2H transfers and NCCL with compute:
/// - `stream` (compute): all GPU kernels, CUDA Graph capture/replay
/// - `copy_stream`: async H2D/D2H transfers, runs concurrently with compute
/// - `comm_stream`: NCCL collectives and P2P exchanges that can overlap independent compute
///
/// Cross-stream sync uses raw CUDA events (not cudarc's automatic tracking,
/// which breaks CUDA Graph capture).
#[derive(Clone)]
pub struct DeviceContext {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub copy_stream: Arc<CudaStream>,
    pub comm_stream: Arc<CudaStream>,
    pub ordinal: u32,
    /// Reusable CUDA events for pipeline fences. A fence returns its event here
    /// on drop; the next fence re-uses it only after `cuEventQuery` confirms
    /// the previous record completed. Eliminates per-fence `cuEventCreate`/
    /// `cuEventDestroy` (160/step at TP=8 decode).
    event_pool: Arc<Mutex<Vec<CudaEvent>>>,
}

/// Logical stream lane used by the serving pipeline.
///
/// Keep this small and CUDA-specific: higher-level scheduler stages should
/// pass fences around, not raw `CudaStream` handles.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CudaPipelineStreamKind {
    /// Main compute stream. Kernels, graph capture/replay, and D2D snapshots
    /// stay here unless a call site explicitly opts into a copy-stream stage.
    Compute,
    /// Dedicated transfer stream for H2D/D2H work that can be ordered with
    /// compute via explicit events.
    Copy,
    /// Dedicated communication stream for NCCL collectives and P2P exchanges.
    Comm,
}

/// Result of a non-blocking CUDA pipeline fence poll.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CudaPipelineFenceStatus {
    Ready,
    NotReady,
}

/// CUDA event fence produced by one pipeline stream and consumed by another.
///
/// The fence owns the CUDA event until every consumer has either enqueued its
/// stream wait or polled/read the result. This makes stage dependencies explicit
/// instead of hiding event creation inside ad hoc helper calls.
pub struct CudaPipelineFence {
    device_ordinal: u32,
    producer: CudaPipelineStreamKind,
    event: Option<CudaEvent>,
    event_pool: Arc<Mutex<Vec<CudaEvent>>>,
}

impl CudaPipelineFence {
    #[must_use]
    pub fn device_ordinal(&self) -> u32 {
        self.device_ordinal
    }

    #[must_use]
    pub fn producer(&self) -> CudaPipelineStreamKind {
        self.producer
    }

    pub fn query(&self) -> Result<CudaPipelineFenceStatus> {
        let event = self.event.as_ref().expect("fence event taken");
        event
            .context()
            .bind_to_thread()
            .map_err(|e| anyhow!("Bind CUDA context before pipeline fence query failed: {e}"))?;
        // SAFETY: `self.event` is a live CudaEvent owned by this fence and its
        // context was bound to the thread above; query has no other effects.
        match unsafe { cudarc::driver::result::event::query(event.cu_event()) } {
            Ok(()) => Ok(CudaPipelineFenceStatus::Ready),
            Err(err) if err.0 == cudarc::driver::sys::CUresult::CUDA_ERROR_NOT_READY => {
                Ok(CudaPipelineFenceStatus::NotReady)
            }
            Err(err) => Err(anyhow!("CUDA pipeline fence query failed: {err}")),
        }
    }
}

impl Drop for CudaPipelineFence {
    fn drop(&mut self) {
        // Return the event to the pool for re-use. The GPU may still be
        // touching it; `record_pipeline_fence` skips in-flight events via
        // cuEventQuery before re-recording.
        if let Some(event) = self.event.take() {
            self.event_pool.lock().unwrap().push(event);
        }
    }
}

/// Parse `INFER_CUDA_DEVICE` (default 0). Selects the device for `DeviceContext::new()`.
pub fn parse_device_ordinal_from_env() -> Result<u32> {
    parse_device_ordinal(std::env::var("INFER_CUDA_DEVICE").ok().as_deref())
}

/// String-pure parse of an `INFER_CUDA_DEVICE`-style ordinal. `None` => 0.
/// Split out from [`parse_device_ordinal_from_env`] so unit tests don't need
/// to mutate the process environment (which races with concurrent tests).
fn parse_device_ordinal(value: Option<&str>) -> Result<u32> {
    match value {
        Some(s) => s.trim().parse::<u32>().map_err(|e| {
            anyhow!("INFER_CUDA_DEVICE must be a non-negative integer, got {s:?}: {e}")
        }),
        None => Ok(0),
    }
}

/// `--cuda-mempool-retain` (default on), set once BEFORE context creation.
static MEMPOOL_RETAIN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

pub fn set_mempool_retain(enabled: bool) {
    MEMPOOL_RETAIN.store(enabled, std::sync::atomic::Ordering::Relaxed);
}
impl DeviceContext {
    /// Default constructor: honours `INFER_CUDA_DEVICE` (default 0).
    /// F1+ multi-GPU rank threads bypass this and call `on_device(ordinal)`.
    pub fn new() -> Result<Self> {
        let ordinal = parse_device_ordinal_from_env()?;
        Self::on_device(ordinal)
    }

    pub fn on_device(ordinal: u32) -> Result<Self> {
        let ctx = CudaContext::new(ordinal as usize)
            .map_err(|e| anyhow!("Failed to create CUDA context on device {ordinal}: {e}"))?;

        // Disable cudarc's automatic event tracking before creating streams.
        // Serving owns cross-stream dependencies explicitly via
        // CudaPipelineFence, which avoids hidden waits in CUDA Graph capture
        // paths while still allowing a dedicated copy stream.
        // SAFETY: called before any stream or buffer exists on this context, so
        // no in-flight cudarc dependency tracking is discarded; all cross-stream
        // ordering is owned explicitly by CudaPipelineFence afterwards.
        unsafe {
            ctx.disable_event_tracking();
        }

        // Make the device `cuMemAllocAsync` pool a true caching allocator: raise the
        // release threshold to MAX so freed blocks are CACHED for reuse instead of
        // returned to the OS at the next stream/context sync. The cudarc default is a
        // 0 threshold, which (with a per-step decode sync) re-allocates every per-step
        // `HiddenStates::uninit`/`alloc_zeros` from the OS — the cuMemAllocAsync churn
        // #29 only fixed for the MoE scratch. PyTorch's caching allocator + SGLang do
        // exactly this. `trim_memory_pool()` still reclaims VRAM explicitly when needed
        // (e.g. weight offload). Best-effort: a failure here is not fatal.
        // Default on; `--cuda-mempool-retain false` restores release-at-sync.
        // Configure the device's default pool on every context creation. `false`
        // must reset a threshold a prior context may have raised on the same device.
        let retain_pool = MEMPOOL_RETAIN.load(std::sync::atomic::Ordering::Relaxed);
        let mut threshold = if retain_pool { u64::MAX } else { 0 };
        // SAFETY: `cu_device()` is the live device of the context created above;
        // attribute calls use pointers to local u64 values that outlive each call.
        unsafe {
            if let Ok(pool) = cudarc::driver::result::device::get_mem_pool(ctx.cu_device()) {
                if let Err(e) = cudarc::driver::result::mem_pool::set_attribute(
                    pool,
                    cudarc::driver::sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                    (&mut threshold as *mut u64).cast::<core::ffi::c_void>(),
                ) {
                    log::warn!("set cuMemAllocAsync release threshold failed (non-fatal): {e}");
                } else {
                    let mut effective = 0u64;
                    match cudarc::driver::result::mem_pool::get_attribute(
                        pool,
                        cudarc::driver::sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                        (&mut effective as *mut u64).cast::<core::ffi::c_void>(),
                    ) {
                        Ok(()) => log::info!(
                            "cuMemAllocAsync pool: requested_retain={retain_pool} \
                             effective_release_threshold={effective} bytes"
                        ),
                        Err(e) => log::warn!(
                            "read cuMemAllocAsync release threshold failed (non-fatal): {e}"
                        ),
                    }
                }
            }
        }

        let stream = ctx
            .new_stream()
            .map_err(|e| anyhow!("Failed to create CUDA stream: {}", e))?;

        let copy_stream = ctx
            .new_stream()
            .map_err(|e| anyhow!("Failed to create CUDA copy stream: {}", e))?;

        let comm_stream = ctx
            .new_stream()
            .map_err(|e| anyhow!("Failed to create CUDA communication stream: {}", e))?;

        // SAFETY: no pointers cross the FFI; `cublas_init` is idempotent,
        // mutex-guarded per device, and requires the current CUDA context to be
        // set — `CudaContext::new(ordinal)` above did exactly that.
        unsafe {
            ffi::cublas_init();
        }

        Ok(Self {
            ctx,
            stream,
            copy_stream,
            comm_stream,
            ordinal,
            event_pool: Arc::new(Mutex::new(Vec::new())),
        })
    }

    pub fn ordinal(&self) -> u32 {
        self.ordinal
    }

    pub fn sm_count(&self) -> usize {
        use cudarc::driver::sys::*;
        let mut count: i32 = 0;
        // SAFETY: writes one i32 through a valid pointer to the local `count`;
        // `cu_device()` is this context's live device handle.
        unsafe {
            cuDeviceGetAttribute(
                &mut count,
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                self.ctx.cu_device(),
            );
        }
        count.max(1) as usize
    }

    pub fn compute_capability(&self) -> (i32, i32) {
        use cudarc::driver::sys::*;
        let mut major: i32 = 0;
        let mut minor: i32 = 0;
        // SAFETY: writes one i32 each through valid pointers to the locals
        // `major`/`minor`; `cu_device()` is this context's live device handle.
        unsafe {
            cuDeviceGetAttribute(
                &mut major,
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                self.ctx.cu_device(),
            );
            cuDeviceGetAttribute(
                &mut minor,
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                self.ctx.cu_device(),
            );
        }
        (major, minor)
    }

    /// True on sm_120 (Blackwell RTX PRO 6000): no Hopper DeepGEMM bridge —
    /// FP8 grouped MoE routes to the CUTLASS sm_120a collective instead.
    #[must_use]
    pub fn is_sm120(&self) -> bool {
        self.compute_capability().0 == 12
    }

    ///
    /// Wraps `cuMemGetInfo`. Used by the OPD engine time-share path to verify
    /// that weight offload actually releases resident VRAM.
    pub fn mem_info_bytes(&self) -> Result<(usize, usize)> {
        self.ctx
            .bind_to_thread()
            .map_err(|e| anyhow!("bind context before mem_get_info failed: {e}"))?;
        cudarc::driver::result::mem_get_info().map_err(|e| anyhow!("cuMemGetInfo failed: {e}"))
    }

    /// Release unused blocks cached in the device's async memory pool back to
    /// the OS so `nvidia-smi` / `mem_info_bytes` reflect freed weight VRAM.
    ///
    /// cudarc allocs through `cuMemAllocAsync` with a 0 release-threshold, so a
    /// dropped `CudaSlice` returns its block to the pool rather than the OS.
    /// After offloading weights we trim the pool so the freed VRAM is genuinely
    /// reclaimed (and observable). Best-effort: a trim failure is not fatal.
    pub fn trim_memory_pool(&self) -> Result<()> {
        self.ctx
            .bind_to_thread()
            .map_err(|e| anyhow!("bind context before mem pool trim failed: {e}"))?;
        // SAFETY: `cu_device()` returns this context's valid device; the
        // default pool for that device is valid for the process lifetime.
        unsafe {
            let pool = cudarc::driver::result::device::get_mem_pool(self.ctx.cu_device())
                .map_err(|e| anyhow!("get device mem pool failed: {e}"))?;
            cudarc::driver::result::mem_pool::trim_to(pool, 0)
                .map_err(|e| anyhow!("cuMemPoolTrimTo(0) failed: {e}"))?;
        }
        Ok(())
    }

    /// Switch the pool's release threshold at runtime. `retain=true` sets
    /// MAX (cache blocks, no event queries — required for graph capture);
    /// `retain=false` sets 0 (release at sync).
    pub fn set_pool_retain(&self, retain: bool) -> Result<()> {
        self.ctx
            .bind_to_thread()
            .map_err(|e| anyhow!("bind context before pool retain switch failed: {e}"))?;
        let mut threshold = if retain { u64::MAX } else { 0 };
        // SAFETY: valid device + pool; threshold is a local that outlives the call.
        unsafe {
            let pool = cudarc::driver::result::device::get_mem_pool(self.ctx.cu_device())
                .map_err(|e| anyhow!("get device mem pool failed: {e}"))?;
            cudarc::driver::result::mem_pool::set_attribute(
                pool,
                cudarc::driver::sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                (&mut threshold as *mut u64).cast::<core::ffi::c_void>(),
            )
            .map_err(|e| anyhow!("set pool release threshold failed: {e}"))?;
        }
        Ok(())
    }

    pub fn sync(&self) -> Result<()> {
        self.stream
            .synchronize()
            .map_err(|e| anyhow!("Sync failed: {}", e))
    }

    pub fn sync_copy(&self) -> Result<()> {
        self.copy_stream
            .synchronize()
            .map_err(|e| anyhow!("Copy stream sync failed: {}", e))
    }

    #[must_use]
    pub fn pipeline_stream(&self, kind: CudaPipelineStreamKind) -> &Arc<CudaStream> {
        match kind {
            CudaPipelineStreamKind::Compute => &self.stream,
            CudaPipelineStreamKind::Copy => &self.copy_stream,
            CudaPipelineStreamKind::Comm => &self.comm_stream,
        }
    }

    pub fn record_pipeline_fence(
        &self,
        producer: CudaPipelineStreamKind,
    ) -> Result<CudaPipelineFence> {
        self.ctx
            .bind_to_thread()
            .map_err(|e| anyhow!("Bind CUDA context before pipeline fence record failed: {e}"))?;
        // cuEventQuery is illegal (and invalidates the capture) while the
        // stream is capturing; allocate fresh and skip the pool probe.
        let stream = self.pipeline_stream(producer);
        // SAFETY: the pipeline stream handle is valid for this context.
        let capturing = unsafe { cudarc::driver::result::stream::is_capturing(stream.cu_stream()) }
            .map(|s| s != cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE)
            .unwrap_or(false);
        let event = {
            let mut pool = self.event_pool.lock().unwrap();
            // Find a completed event to re-use. cuEventQuery on an in-flight
            // record returns NOT_READY — skip it. An empty pool or all-in-flight
            // pool allocates fresh; the pool grows to the max in-flight count
            // and stays there.
            let mut ready = None;
            if !capturing {
                for (i, e) in pool.iter().enumerate() {
                    // SAFETY: the pooled event handle is valid for this context.
                    match unsafe { cudarc::driver::result::event::query(e.cu_event()) } {
                        Ok(()) => {
                            ready = Some(i);
                            break;
                        }
                        Err(err)
                            if err.0 == cudarc::driver::sys::CUresult::CUDA_ERROR_NOT_READY =>
                        {
                            continue;
                        }
                        Err(err) => return Err(anyhow!("CUDA event query in pool failed: {err}")),
                    }
                }
            }
            match ready {
                Some(i) => pool.swap_remove(i),
                None => self
                    .ctx
                    .new_event(None)
                    .map_err(|e| anyhow!("Alloc CUDA pipeline fence failed: {e}"))?,
            }
        };
        event
            .record(stream)
            .map_err(|e| anyhow!("Record CUDA pipeline fence on {producer:?} failed: {e}"))?;
        Ok(CudaPipelineFence {
            device_ordinal: self.ordinal,
            producer,
            event: Some(event),
            event_pool: self.event_pool.clone(),
        })
    }

    pub fn wait_on_pipeline_fence(
        &self,
        fence: &CudaPipelineFence,
        consumer: CudaPipelineStreamKind,
    ) -> Result<()> {
        ensure!(
            fence.device_ordinal == self.ordinal,
            "CUDA pipeline fence device mismatch: fence device {} consumed on device {}",
            fence.device_ordinal,
            self.ordinal
        );
        self.pipeline_stream(consumer)
            .wait(fence.event.as_ref().expect("fence event taken"))
            .map_err(|e| {
                anyhow!(
                    "CUDA pipeline fence wait failed for {consumer:?} waiting on {:?}: {e}",
                    fence.producer
                )
            })
    }

    /// Upload pinned host data into an existing device allocation on the copy stream.
    ///
    /// # Safety
    ///
    /// The caller must ensure `dst` is already valid on the copy stream before
    /// this call. If its allocation or previous writes are on another stream,
    /// order that stream first before this call.
    /// `dst` must stay allocated and must not be read, written, or freed by
    /// another stream until that stream waits on the returned fence. `src` must
    /// be pinned so the async H2D copy has a stable host address.
    pub unsafe fn memcpy_pinned_htod_on_copy_stream<T, Dst>(
        &self,
        src: &PinnedHostSlice<T>,
        dst: &mut Dst,
    ) -> Result<CudaPipelineFence>
    where
        T: DeviceRepr,
        Dst: DevicePtrMut<T>,
    {
        self.ctx
            .bind_to_thread()
            .map_err(|e| anyhow!("Bind CUDA context before copy-stream H2D failed: {e}"))?;
        self.copy_stream
            .memcpy_htod(src, dst)
            .map_err(|e| anyhow!("copy-stream pinned H2D memcpy failed: {e}"))?;
        self.record_pipeline_fence(CudaPipelineStreamKind::Copy)
    }

    /// Record an event on the compute stream and make the communication stream wait for it.
    ///
    /// Use after kernels that produce collective inputs, so NCCL can run on
    /// `comm_stream` without reading incomplete compute-stream data.
    pub fn comm_waits_for_compute(&self) -> Result<()> {
        let fence = self.record_pipeline_fence(CudaPipelineStreamKind::Compute)?;
        self.wait_on_pipeline_fence(&fence, CudaPipelineStreamKind::Comm)
    }

    /// Record an event on the communication stream and make compute wait for it.
    ///
    /// Use before kernels consume collective outputs produced on `comm_stream`.
    pub fn compute_waits_for_comm(&self) -> Result<()> {
        let fence = self.record_pipeline_fence(CudaPipelineStreamKind::Comm)?;
        self.wait_on_pipeline_fence(&fence, CudaPipelineStreamKind::Compute)
    }
}

#[cfg(test)]
mod pipeline_fence_tests {
    use super::*;

    #[test]
    fn pipeline_fence_orders_compute_and_copy_streams() -> Result<()> {
        let ctx = DeviceContext::new()?;

        let compute_done = ctx.record_pipeline_fence(CudaPipelineStreamKind::Compute)?;
        assert_eq!(compute_done.device_ordinal(), ctx.ordinal());
        assert_eq!(compute_done.producer(), CudaPipelineStreamKind::Compute);
        ctx.wait_on_pipeline_fence(&compute_done, CudaPipelineStreamKind::Copy)?;

        let copy_done = ctx.record_pipeline_fence(CudaPipelineStreamKind::Copy)?;
        assert_eq!(copy_done.device_ordinal(), ctx.ordinal());
        assert_eq!(copy_done.producer(), CudaPipelineStreamKind::Copy);
        ctx.wait_on_pipeline_fence(&copy_done, CudaPipelineStreamKind::Compute)?;

        ctx.sync()?;
        ctx.sync_copy()?;
        Ok(())
    }

    #[test]
    fn pinned_copy_stream_h2d_helper_returns_compute_waitable_fence() -> Result<()> {
        let ctx = DeviceContext::new()?;

        let initial = [11_i32, 22, 33, 44];
        // SAFETY: `alloc_pinned` returns a valid pinned buffer of `initial.len()`
        // elements; the buffer is freed when `pinned` is dropped at scope end.
        let mut pinned = unsafe {
            ctx.ctx
                .alloc_pinned::<i32>(initial.len())
                .map_err(|e| anyhow!("Alloc pinned H2D helper source failed: {e}"))?
        };
        pinned.as_mut_slice()?.copy_from_slice(&initial);
        let mut existing = ctx
            .copy_stream
            .alloc_zeros::<i32>(initial.len())
            .map_err(|e| anyhow!("Alloc H2D helper test buffer failed: {e}"))?;

        // SAFETY: `pinned` and `existing` both live until compute waits on the
        // returned fence and reads the uploaded data below.
        let upload_done = unsafe { ctx.memcpy_pinned_htod_on_copy_stream(&pinned, &mut existing)? };
        assert_eq!(upload_done.producer(), CudaPipelineStreamKind::Copy);
        ctx.wait_on_pipeline_fence(&upload_done, CudaPipelineStreamKind::Compute)?;
        let got = ctx.stream.clone_dtoh(&existing)?;
        ctx.sync()?;
        assert_eq!(got.as_slice(), &initial);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_device_ordinal_handles_unset_default_and_invalid() {
        assert_eq!(parse_device_ordinal(None).unwrap(), 0);
        assert_eq!(parse_device_ordinal(Some("3")).unwrap(), 3);
        assert_eq!(parse_device_ordinal(Some("  7 ")).unwrap(), 7);
        assert!(parse_device_ordinal(Some("not-a-number")).is_err());
        assert!(parse_device_ordinal(Some("")).is_err());
    }
}
