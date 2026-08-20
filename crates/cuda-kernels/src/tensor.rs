//! Device tensor types and CUDA context.

use anyhow::{Result, anyhow, bail, ensure};
use cudarc::driver::{
    CudaContext, CudaEvent, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr,
    DriverError, PinnedHostSlice, ValidAsZeroBits,
};
use half::{bf16, f16};
use std::any::type_name;
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::panic::Location;
use std::sync::{Arc, LazyLock, Mutex, OnceLock};

use super::ffi;

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
fn record_cuda_alloc<T>(kind: &'static str, label: &'static str, len: usize) {
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

    /// Allocate zeroed memory and attribute the call site when
    /// `ARLE_CUDA_ALLOC_TRACE=1`.
    fn alloc_zeros_traced<T: DeviceRepr + ValidAsZeroBits>(
        &self,
        len: usize,
    ) -> Result<CudaSlice<T>, DriverError>;
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

    #[track_caller]
    fn alloc_zeros_traced<T: DeviceRepr + ValidAsZeroBits>(
        &self,
        len: usize,
    ) -> Result<CudaSlice<T>, DriverError> {
        // SAFETY: the uninitialized allocation is zeroed by the stream-ordered
        // `memset_zeros` below before the slice is returned, so no caller can
        // observe uninitialized bytes.
        let mut out = unsafe { self.alloc(len)? };
        record_cuda_alloc::<T>("alloc_zeros", "CudaStream::alloc_zeros", len);
        self.memset_zeros(&mut out)?;
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
    /// Separate stream for async H2D/D2H memory copies.
    pub copy_stream: Arc<CudaStream>,
    /// Separate stream for NCCL/communication work that can overlap compute.
    pub comm_stream: Arc<CudaStream>,
    /// CUDA device ordinal this context is bound to.
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

    /// Poll the event without blocking the host.
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

    /// Query the number of streaming multiprocessors on the GPU this context is bound to.
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

    /// Query the CUDA compute capability for the GPU this context is bound to.
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

    /// Query (free, total) device memory in bytes for the bound device.
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

    /// Synchronize compute stream.
    pub fn sync(&self) -> Result<()> {
        self.stream
            .synchronize()
            .map_err(|e| anyhow!("Sync failed: {}", e))
    }

    /// Synchronize copy stream.
    pub fn sync_copy(&self) -> Result<()> {
        self.copy_stream
            .synchronize()
            .map_err(|e| anyhow!("Copy stream sync failed: {}", e))
    }

    /// Return the raw stream that backs a pipeline lane.
    #[must_use]
    pub fn pipeline_stream(&self, kind: CudaPipelineStreamKind) -> &Arc<CudaStream> {
        match kind {
            CudaPipelineStreamKind::Compute => &self.stream,
            CudaPipelineStreamKind::Copy => &self.copy_stream,
            CudaPipelineStreamKind::Comm => &self.comm_stream,
        }
    }

    /// Record a fence on the selected producer stream.
    pub fn record_pipeline_fence(
        &self,
        producer: CudaPipelineStreamKind,
    ) -> Result<CudaPipelineFence> {
        self.ctx
            .bind_to_thread()
            .map_err(|e| anyhow!("Bind CUDA context before pipeline fence record failed: {e}"))?;
        let event = {
            let mut pool = self.event_pool.lock().unwrap();
            // Find a completed event to re-use. cuEventQuery on an in-flight
            // record returns NOT_READY — skip it. An empty pool or all-in-flight
            // pool allocates fresh; the pool grows to the max in-flight count
            // and stays there.
            let mut ready = None;
            for (i, e) in pool.iter().enumerate() {
                match unsafe { cudarc::driver::result::event::query(e.cu_event()) } {
                    Ok(()) => {
                        ready = Some(i);
                        break;
                    }
                    Err(err) if err.0 == cudarc::driver::sys::CUresult::CUDA_ERROR_NOT_READY => {
                        continue;
                    }
                    Err(err) => return Err(anyhow!("CUDA event query in pool failed: {err}")),
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
            .record(self.pipeline_stream(producer))
            .map_err(|e| anyhow!("Record CUDA pipeline fence on {producer:?} failed: {e}"))?;
        Ok(CudaPipelineFence {
            device_ordinal: self.ordinal,
            producer,
            event: Some(event),
            event_pool: self.event_pool.clone(),
        })
    }

    /// Make `consumer` wait for `fence` without blocking the host.
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

fn bf16_safetensor_host_slice(data: &[u8]) -> Result<Cow<'_, [bf16]>> {
    ensure!(
        data.len().is_multiple_of(2),
        "Data length must be even for bf16: got {} bytes",
        data.len()
    );
    // Safetensors are little-endian. If a mmap-backed tensor starts at an
    // unaligned byte offset, casting `u8*` to `bf16*` would be undefined
    // behavior; fall back to a small decode buffer only for that case.
    // SAFETY: bf16 is a 2-byte POD for which every bit pattern is valid;
    // `align_to` itself confines the reinterpret to the correctly-aligned
    // middle, and the unaligned prefix/suffix case falls back to a decode copy.
    let (prefix, aligned, suffix) = unsafe { data.align_to::<bf16>() };
    if prefix.is_empty() && suffix.is_empty() {
        return Ok(Cow::Borrowed(aligned));
    }
    Ok(Cow::Owned(
        data.chunks_exact(2)
            .map(|c| bf16::from_le_bytes([c[0], c[1]]))
            .collect(),
    ))
}

/// 1D device tensor (vector) — stored as bf16.
pub struct DeviceVec {
    pub data: CudaSlice<bf16>,
    pub len: usize,
    /// Debug label describing the tensor's semantic shape (e.g., `norm_weight[hidden]`, `kv_cache[heads,seq,dim]`).
    pub label: &'static str,
}

impl DeviceVec {
    /// Create from host data (bf16)
    pub fn from_host(ctx: &DeviceContext, data: &[bf16]) -> Result<Self> {
        let gpu_data = ctx
            .stream
            .clone_htod(data)
            .map_err(|e| anyhow!("H2D copy failed: {}", e))?;
        Ok(Self {
            data: gpu_data,
            len: data.len(),
            label: "",
        })
    }

    pub fn from_safetensors(ctx: &DeviceContext, data: &[u8]) -> Result<Self> {
        let slice = bf16_safetensor_host_slice(data)?;
        Self::from_host(ctx, slice.as_ref())
    }

    /// Create zeroed tensor
    #[track_caller]
    pub fn zeros(ctx: &DeviceContext, len: usize) -> Result<Self> {
        let gpu_data: CudaSlice<bf16> = ctx
            .stream
            .alloc_zeros(len)
            .map_err(|e| anyhow!("Alloc failed: {}", e))?;
        record_cuda_alloc::<bf16>("alloc_zeros", "DeviceVec::zeros", len);
        Ok(Self {
            data: gpu_data,
            len,
            label: "",
        })
    }

    /// Create an UNINITIALIZED tensor (no zeroing memset).
    ///
    /// # Safety
    /// The buffer holds uninitialized device memory; every element must be
    /// written before it is read.
    #[track_caller]
    pub unsafe fn uninit(ctx: &DeviceContext, len: usize) -> Result<Self> {
        // SAFETY: forwards the uninitialized-memory contract to our caller.
        let gpu_data: CudaSlice<bf16> = unsafe {
            ctx.stream
                .alloc(len)
                .map_err(|e| anyhow!("Alloc failed: {}", e))?
        };
        record_cuda_alloc::<bf16>("alloc", "DeviceVec::uninit", len);
        Ok(Self {
            data: gpu_data,
            len,
            label: "",
        })
    }

    /// Create a tensor filled with bf16 ones (1.0).
    /// Useful for dummy RMSNorm weights (identity normalization).
    pub fn ones(ctx: &DeviceContext, len: usize) -> Result<Self> {
        let host = vec![bf16::ONE; len];
        Self::from_host(ctx, &host)
    }

    /// Move the device buffer to host RAM and free the VRAM (OPD time-share).
    ///
    /// Returns the host bytes plus the device bytes freed; the live buffer is
    /// replaced with a 1-element placeholder.
    pub fn offload_to_host(&mut self, ctx: &DeviceContext) -> Result<(Vec<bf16>, usize)> {
        let host = ctx
            .stream
            .clone_dtoh(&self.data)
            .map_err(|e| anyhow!("offload D2H copy (vec) failed: {e}"))?;
        let freed = host.len() * std::mem::size_of::<bf16>();
        ctx.sync()?;
        self.data = ctx
            .stream
            .alloc_zeros::<bf16>(1)
            .map_err(|e| anyhow!("offload vec placeholder alloc failed: {e}"))?;
        Ok((host, freed))
    }

    /// Restore the device buffer from a host snapshot, re-allocating VRAM.
    pub fn reload_from_host(&mut self, ctx: &DeviceContext, host: &[bf16]) -> Result<()> {
        self.data = ctx
            .stream
            .clone_htod(host)
            .map_err(|e| anyhow!("reload H2D copy (vec) failed: {e}"))?;
        ctx.sync()?;
        Ok(())
    }

    /// Copy to host as f32 (for testing). Exposed publicly so downstream
    /// crates in this workspace (notably `infer`) can use it from their
    /// own test suites, since that would otherwise sit behind the
    /// cuda-kernels `#[cfg(test)]` boundary.
    pub fn to_host(&self, ctx: &DeviceContext) -> Result<Vec<f32>> {
        let host_f16 = ctx
            .stream
            .clone_dtoh(&self.data)
            .map_err(|e| anyhow!("D2H copy failed: {}", e))?;
        ctx.sync()?;
        Ok(host_f16.iter().map(|x| x.to_f32()).collect())
    }
}

impl Clone for DeviceVec {
    fn clone(&self) -> Self {
        Self {
            data: self.data.try_clone().unwrap(),
            len: self.len,
            label: self.label,
        }
    }
}

impl std::fmt::Debug for DeviceVec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.label.is_empty() {
            write!(f, "DeviceVec(len={})", self.len)
        } else {
            write!(f, "DeviceVec({}, len={})", self.label, self.len)
        }
    }
}

/// Explicit storage format for a linear weight matrix.
///
/// This is the Rust-side kernel ABI selector: checkpoint format detection and
/// loader packing set this once, then inference dispatch matches this enum
/// instead of re-interpreting packed buffers through bit-width sentinels.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum WeightFormat {
    /// Dense row-major BF16 weights.
    #[default]
    DenseBf16,
    /// Uniform per-group signed INT8 weights with BF16 scales.
    W8A16,
    /// Uniform per-group packed INT4 weights with BF16 scales.
    W4A16,
    /// Marlin W4 weights with dynamic INT8 activations.
    MarlinW4A8,
    /// Uniform per-group packed INT2 weights with BF16 scales.
    W2A16,
    /// GGUF Q3_K packed superblocks, scales embedded in each 256-wide block.
    GgufQ3K,
    /// GGUF Q4_K packed superblocks, scales embedded in each 256-wide block.
    GgufQ4K,
    /// GGUF Q5_K packed superblocks, scales embedded in each 256-wide block.
    GgufQ5K,
    /// GGUF Q6_K packed superblocks, scales embedded in each 256-wide block.
    GgufQ6K,
    /// TurboQuant packed indices + FP16 group norms + Hadamard signs.
    TurboQuant,
    /// DeepSeek V4 row-major FP8 E4M3 weights with FP8 E8M0 block scales.
    Dsv4Fp8BlockScaled,
    /// DeepSeek V4 row-major packed FP4 E2M1 weights with FP8 E8M0 block scales.
    Dsv4Fp4BlockScaled,
    /// ABI-generic row-major FP8 E4M3 weights with f32 block scales.
    Fp8BlockScaled,
    /// ABI-generic row-major FP8 E4M3 weights with one f32 scale per shard.
    Fp8PerShard,
    /// ABI-generic row-major packed FP4 E2M1 weights with FP8 group scales.
    Fp4E2M1Group,
}

/// Shape/layout constraints expected by the matching CUDA kernels.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WeightKernelAlignment {
    pub weight_layout: &'static str,
    pub scale_layout: &'static str,
    pub k_multiple: usize,
    pub n_multiple: usize,
    pub group_size: usize,
}

impl WeightFormat {
    #[must_use]
    pub fn is_quantized(self) -> bool {
        !matches!(self, Self::DenseBf16)
    }

    pub fn validate_shape(self, rows: usize, cols: usize, group_size: usize) -> Result<()> {
        ensure!(rows > 0, "{self} requires rows > 0");
        ensure!(cols > 0, "{self} requires cols > 0");
        match self {
            Self::DenseBf16 => Ok(()),
            Self::W8A16 | Self::W4A16 | Self::W2A16 | Self::TurboQuant => {
                ensure!(group_size > 0, "{self} requires group_size > 0");
                ensure!(
                    cols.is_multiple_of(group_size),
                    "{self} requires cols % group_size == 0, got cols={cols}, group_size={group_size}"
                );
                Ok(())
            }
            Self::MarlinW4A8 => {
                ensure!(group_size > 0, "{self} requires group_size > 0");
                ensure!(
                    group_size == 128,
                    "{self} currently requires group_size=128, got {group_size}"
                );
                ensure!(
                    cols.is_multiple_of(group_size),
                    "{self} requires cols % group_size == 0, got cols={cols}, group_size={group_size}"
                );
                ensure!(
                    cols.is_multiple_of(128),
                    "{self} requires cols % 128 == 0, got {cols}"
                );
                ensure!(
                    rows.is_multiple_of(256),
                    "{self} requires rows % 256 == 0, got {rows}"
                );
                Ok(())
            }
            Self::GgufQ3K | Self::GgufQ4K | Self::GgufQ5K | Self::GgufQ6K => {
                ensure!(
                    cols.is_multiple_of(256),
                    "{self} requires cols % 256 == 0, got {cols}"
                );
                ensure!(
                    group_size == 256,
                    "{self} requires synthetic group_size=256, got {group_size}"
                );
                Ok(())
            }
            Self::Dsv4Fp8BlockScaled => Ok(()),
            Self::Dsv4Fp4BlockScaled => {
                ensure!(
                    cols.is_multiple_of(2),
                    "{self} requires cols % 2 == 0, got {cols}"
                );
                Ok(())
            }
            Self::Fp8BlockScaled | Self::Fp8PerShard => Ok(()),
            Self::Fp4E2M1Group => {
                ensure!(group_size > 0, "{self} requires group_size > 0");
                ensure!(
                    cols.is_multiple_of(2),
                    "{self} requires cols % 2 == 0 for packed E2M1, got {cols}"
                );
                ensure!(
                    cols.is_multiple_of(group_size),
                    "{self} requires cols % group_size == 0, got cols={cols}, group_size={group_size}"
                );
                Ok(())
            }
        }
    }
}

impl std::fmt::Display for WeightFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DenseBf16 => f.write_str("dense_bf16"),
            Self::W8A16 => f.write_str("w8a16"),
            Self::W4A16 => f.write_str("w4a16"),
            Self::MarlinW4A8 => f.write_str("marlin_w4a8"),
            Self::W2A16 => f.write_str("w2a16"),
            Self::GgufQ3K => f.write_str("gguf_q3_k"),
            Self::GgufQ4K => f.write_str("gguf_q4_k"),
            Self::GgufQ5K => f.write_str("gguf_q5_k"),
            Self::GgufQ6K => f.write_str("gguf_q6_k"),
            Self::TurboQuant => f.write_str("turboquant"),
            Self::Dsv4Fp8BlockScaled => f.write_str("dsv4_fp8_block_scaled"),
            Self::Dsv4Fp4BlockScaled => f.write_str("dsv4_fp4_block_scaled"),
            Self::Fp8BlockScaled => f.write_str("fp8_block_scaled"),
            Self::Fp8PerShard => f.write_str("fp8_per_shard"),
            Self::Fp4E2M1Group => f.write_str("fp4_e2m1_group"),
        }
    }
}

const DSV4_DEEPGEMM_FP8_SCALE_GRAN_M: usize = 128;
const DSV4_DEEPGEMM_FP8_SCALE_GRAN_K: usize = 128;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Dsv4DeepGemmSourceFormat {
    Fp8 = 0,
    Fp4 = 1,
}

impl Dsv4DeepGemmSourceFormat {
    fn from_weight_format(format: WeightFormat) -> Result<Self> {
        match format {
            WeightFormat::Dsv4Fp8BlockScaled => Ok(Self::Fp8),
            WeightFormat::Dsv4Fp4BlockScaled => Ok(Self::Fp4),
            other => Err(anyhow!(
                "DeepSeek V4 DeepGEMM FP8 cache needs raw DSv4 block-scaled weights, got {other}"
            )),
        }
    }
}

/// Resident FP8 E4M3 weight cache plus FP32 block scales in DeepGEMM's SM90
/// grouped-GEMM source layout.
///
/// `weight` is row-major `[rows, cols]` FP8 bytes. `scales` is contiguous
/// `[ceil(rows/128), ceil(cols/128)]` FP32, matching DeepGEMM's Hopper SFB
/// recipe for m-grouped FP8 GEMM.
pub struct Dsv4Fp8DeepGemmWeightCache {
    pub weight: CudaSlice<u8>,
    pub scales: CudaSlice<f32>,
    pub rows: usize,
    pub cols: usize,
    pub scale_rows: usize,
    pub scale_cols: usize,
}

impl Dsv4Fp8DeepGemmWeightCache {
    pub fn uninit(ctx: &DeviceContext, rows: usize, cols: usize) -> Result<Self> {
        let scale_rows = rows.div_ceil(DSV4_DEEPGEMM_FP8_SCALE_GRAN_M);
        let scale_cols = cols.div_ceil(DSV4_DEEPGEMM_FP8_SCALE_GRAN_K);
        let weight_len = rows.checked_mul(cols).ok_or_else(|| {
            anyhow!(
                "DeepSeek V4 DeepGEMM cache weight size overflow: rows={} cols={}",
                rows,
                cols
            )
        })?;
        let scale_len = scale_rows.checked_mul(scale_cols).ok_or_else(|| {
            anyhow!(
                "DeepSeek V4 DeepGEMM cache scale size overflow: rows={} cols={}",
                scale_rows,
                scale_cols
            )
        })?;
        Ok(Self {
            // SAFETY: both buffers start uninitialized by design — every row is
            // written by `dsv4_fill_fp8_deepgemm_weight_cache` (the only
            // producer) before any DeepGEMM launch reads the cache.
            weight: unsafe { ctx.stream.alloc_traced::<u8>(weight_len)? },
            // SAFETY: see `weight` above; filled before first read.
            scales: unsafe { ctx.stream.alloc_traced::<f32>(scale_len)? },
            rows,
            cols,
            scale_rows,
            scale_cols,
        })
    }

    #[must_use]
    pub fn weight_bytes(&self) -> usize {
        self.rows.saturating_mul(self.cols)
    }

    #[must_use]
    pub fn scale_bytes(&self) -> usize {
        self.scale_rows
            .saturating_mul(self.scale_cols)
            .saturating_mul(std::mem::size_of::<f32>())
    }

    pub fn from_dsv4_weight(ctx: &DeviceContext, weight: &DeviceMatrix) -> Result<Self> {
        let mut cache = Self::uninit(ctx, weight.rows, weight.cols)?;
        cache.fill_from_dsv4_weight(ctx, weight, 0)?;
        Ok(cache)
    }

    pub fn from_dsv4_weight_row_range(
        ctx: &DeviceContext,
        weight: &DeviceMatrix,
        row_start: usize,
        rows: usize,
    ) -> Result<Self> {
        let mut cache = Self::uninit(ctx, rows, weight.cols)?;
        cache.fill_from_dsv4_weight_row_range(ctx, weight, row_start, rows, 0)?;
        Ok(cache)
    }

    pub fn from_dsv4_weight_pair_rows(
        ctx: &DeviceContext,
        first: &DeviceMatrix,
        second: &DeviceMatrix,
    ) -> Result<Self> {
        ensure!(
            first.cols == second.cols,
            "DeepSeek V4 DeepGEMM fused cache needs matching K: first={} second={}",
            first.cols,
            second.cols
        );
        ensure!(
            first.rows.is_multiple_of(DSV4_DEEPGEMM_FP8_SCALE_GRAN_M),
            "DeepSeek V4 DeepGEMM fused cache needs first row count aligned to {}, got {}",
            DSV4_DEEPGEMM_FP8_SCALE_GRAN_M,
            first.rows
        );
        let mut cache = Self::uninit(ctx, first.rows + second.rows, first.cols)?;
        cache.fill_from_dsv4_weight(ctx, first, 0)?;
        cache.fill_from_dsv4_weight(ctx, second, first.rows)?;
        Ok(cache)
    }

    pub fn from_fp8_block_scaled_weight(
        ctx: &DeviceContext,
        weight: &DeviceMatrix,
    ) -> Result<Self> {
        let mut cache = Self::uninit(ctx, weight.rows, weight.cols)?;
        cache.fill_from_fp8_block_scaled_weight(ctx, weight, 0)?;
        Ok(cache)
    }

    pub fn from_fp8_block_scaled_weight_pair_rows(
        ctx: &DeviceContext,
        first: &DeviceMatrix,
        second: &DeviceMatrix,
    ) -> Result<Self> {
        ensure!(
            first.cols == second.cols,
            "DeepGEMM FP8 fused cache needs matching K: first={} second={}",
            first.cols,
            second.cols
        );
        ensure!(
            first.rows.is_multiple_of(DSV4_DEEPGEMM_FP8_SCALE_GRAN_M),
            "DeepGEMM FP8 fused cache needs first row count aligned to {}, got {}",
            DSV4_DEEPGEMM_FP8_SCALE_GRAN_M,
            first.rows
        );
        let mut cache = Self::uninit(ctx, first.rows + second.rows, first.cols)?;
        cache.fill_from_fp8_block_scaled_weight(ctx, first, 0)?;
        cache.fill_from_fp8_block_scaled_weight(ctx, second, first.rows)?;
        Ok(cache)
    }

    pub fn fill_from_dsv4_weight(
        &mut self,
        ctx: &DeviceContext,
        weight: &DeviceMatrix,
        dst_row_offset: usize,
    ) -> Result<()> {
        dsv4_fill_fp8_deepgemm_weight_cache(
            ctx,
            weight,
            self,
            dst_row_offset,
            dst_row_offset / DSV4_DEEPGEMM_FP8_SCALE_GRAN_M,
        )
    }

    pub fn fill_from_dsv4_weight_row_range(
        &mut self,
        ctx: &DeviceContext,
        weight: &DeviceMatrix,
        row_start: usize,
        rows: usize,
        dst_row_offset: usize,
    ) -> Result<()> {
        dsv4_fill_fp8_deepgemm_weight_cache_row_range(
            ctx,
            weight,
            self,
            row_start,
            rows,
            dst_row_offset,
            dst_row_offset / DSV4_DEEPGEMM_FP8_SCALE_GRAN_M,
        )
    }

    pub fn fill_from_fp8_block_scaled_weight(
        &mut self,
        ctx: &DeviceContext,
        weight: &DeviceMatrix,
        dst_row_offset: usize,
    ) -> Result<()> {
        ensure!(
            weight.weight_format == WeightFormat::Fp8BlockScaled,
            "DeepGEMM FP8 cache needs FP8 block-scaled weights, got {}",
            weight.weight_format
        );
        ensure!(
            weight.quant_block_m == DSV4_DEEPGEMM_FP8_SCALE_GRAN_M
                && weight.quant_block_k == DSV4_DEEPGEMM_FP8_SCALE_GRAN_K,
            "DeepGEMM FP8 cache needs 128x128 block scales, got {}x{}",
            weight.quant_block_m,
            weight.quant_block_k
        );
        ensure!(
            weight.cols == self.cols,
            "DeepGEMM FP8 cache K mismatch: source={} cache={}",
            weight.cols,
            self.cols
        );
        ensure!(
            dst_row_offset + weight.rows <= self.rows,
            "DeepGEMM FP8 cache row range overflow: offset={} src={} cache={}",
            dst_row_offset,
            weight.rows,
            self.rows
        );
        ensure!(
            dst_row_offset.is_multiple_of(DSV4_DEEPGEMM_FP8_SCALE_GRAN_M),
            "DeepGEMM FP8 cache row offset must be {}-aligned, got {}",
            DSV4_DEEPGEMM_FP8_SCALE_GRAN_M,
            dst_row_offset
        );
        let src_scale_rows = weight.rows.div_ceil(DSV4_DEEPGEMM_FP8_SCALE_GRAN_M);
        let src_scale_cols = weight.cols.div_ceil(DSV4_DEEPGEMM_FP8_SCALE_GRAN_K);
        ensure!(
            weight.quant_scale_rows == src_scale_rows && weight.quant_scale_cols == src_scale_cols,
            "DeepGEMM FP8 cache scale shape {}x{} != expected {}x{}",
            weight.quant_scale_rows,
            weight.quant_scale_cols,
            src_scale_rows,
            src_scale_cols
        );
        ensure!(
            self.scale_cols == src_scale_cols,
            "DeepGEMM FP8 cache scale K mismatch: source={} cache={}",
            src_scale_cols,
            self.scale_cols
        );
        let dst_scale_row_offset = dst_row_offset / DSV4_DEEPGEMM_FP8_SCALE_GRAN_M;
        ensure!(
            dst_scale_row_offset + src_scale_rows <= self.scale_rows,
            "DeepGEMM FP8 cache scale row overflow: offset={} src={} cache={}",
            dst_scale_row_offset,
            src_scale_rows,
            self.scale_rows
        );

        let src_weight = weight
            .qweight_u8
            .as_ref()
            .ok_or_else(|| anyhow!("DeepGEMM FP8 cache source missing FP8 weight bytes"))?;
        let src_scales = weight
            .scale_f32
            .as_ref()
            .ok_or_else(|| anyhow!("DeepGEMM FP8 cache source missing f32 block scales"))?;
        ensure!(
            src_weight.len() == weight.rows * weight.cols,
            "DeepGEMM FP8 cache source weight len {} != expected {}",
            src_weight.len(),
            weight.rows * weight.cols
        );
        ensure!(
            src_scales.len() == src_scale_rows * src_scale_cols,
            "DeepGEMM FP8 cache source scale len {} != expected {}",
            src_scales.len(),
            src_scale_rows * src_scale_cols
        );

        {
            let src = src_weight.slice(0..src_weight.len());
            let weight_start = dst_row_offset * self.cols;
            let weight_end = weight_start + src_weight.len();
            let mut dst = self.weight.slice_mut(weight_start..weight_end);
            ctx.stream
                .memcpy_dtod(&src, &mut dst)
                .map_err(|e| anyhow!("DeepGEMM FP8 cache weight D2D failed: {e}"))?;
        }
        {
            let src = src_scales.slice(0..src_scales.len());
            let scale_start = dst_scale_row_offset * self.scale_cols;
            let scale_end = scale_start + src_scales.len();
            let mut dst = self.scales.slice_mut(scale_start..scale_end);
            ctx.stream
                .memcpy_dtod(&src, &mut dst)
                .map_err(|e| anyhow!("DeepGEMM FP8 cache scale D2D failed: {e}"))?;
        }
        Ok(())
    }
}

fn dsv4_fill_fp8_deepgemm_weight_cache(
    ctx: &DeviceContext,
    src: &DeviceMatrix,
    dst: &mut Dsv4Fp8DeepGemmWeightCache,
    dst_row_offset: usize,
    dst_scale_row_offset: usize,
) -> Result<()> {
    let source_format = Dsv4DeepGemmSourceFormat::from_weight_format(src.weight_format)?;
    ensure!(
        src.cols == dst.cols,
        "DeepSeek V4 DeepGEMM cache K mismatch: source={} cache={}",
        src.cols,
        dst.cols
    );
    ensure!(
        dst_row_offset + src.rows <= dst.rows,
        "DeepSeek V4 DeepGEMM cache row range overflow: offset={} src={} cache={}",
        dst_row_offset,
        src.rows,
        dst.rows
    );
    ensure!(
        dst_row_offset.is_multiple_of(DSV4_DEEPGEMM_FP8_SCALE_GRAN_M),
        "DeepSeek V4 DeepGEMM cache row offset must be {}-aligned, got {}",
        DSV4_DEEPGEMM_FP8_SCALE_GRAN_M,
        dst_row_offset
    );
    ensure!(
        src.dsv4_scale_rows > 0 && src.dsv4_scale_cols > 0,
        "DeepSeek V4 DeepGEMM cache source needs DSv4 block scales"
    );
    let src_scale_rows = src.rows.div_ceil(DSV4_DEEPGEMM_FP8_SCALE_GRAN_M);
    ensure!(
        dst_scale_row_offset + src_scale_rows <= dst.scale_rows,
        "DeepSeek V4 DeepGEMM cache scale row overflow: offset={} src={} cache={}",
        dst_scale_row_offset,
        src_scale_rows,
        dst.scale_rows
    );

    let qweight = src
        .qweight
        .as_ref()
        .ok_or_else(|| anyhow!("DeepSeek V4 DeepGEMM cache source missing raw weight bytes"))?;
    let src_scales = src
        .dsv4_scales
        .as_ref()
        .ok_or_else(|| anyhow!("DeepSeek V4 DeepGEMM cache source missing block scales"))?;
    let rows_i32 = i32::try_from(src.rows)
        .map_err(|_| anyhow!("DeepSeek V4 DeepGEMM cache rows overflow i32"))?;
    let cols_i32 = i32::try_from(src.cols)
        .map_err(|_| anyhow!("DeepSeek V4 DeepGEMM cache cols overflow i32"))?;
    let scale_rows_i32 = i32::try_from(src.dsv4_scale_rows)
        .map_err(|_| anyhow!("DeepSeek V4 DeepGEMM source scale rows overflow i32"))?;
    let scale_cols_i32 = i32::try_from(src.dsv4_scale_cols)
        .map_err(|_| anyhow!("DeepSeek V4 DeepGEMM source scale cols overflow i32"))?;
    let dst_scale_cols_i32 = i32::try_from(dst.scale_cols)
        .map_err(|_| anyhow!("DeepSeek V4 DeepGEMM cache scale cols overflow i32"))?;
    let (src_ptr, _src_guard) = qweight.device_ptr(&ctx.stream);
    let (src_scale_ptr, _src_scale_guard) = src_scales.device_ptr(&ctx.stream);
    let (dst_weight_ptr, _dst_weight_guard) = dst.weight.device_ptr_mut(&ctx.stream);
    let (dst_scale_ptr, _dst_scale_guard) = dst.scales.device_ptr_mut(&ctx.stream);
    // SAFETY: `dst_row_offset + src.rows <= dst.rows` was ensured above, so the
    // offset stays inside `dst.weight` (`dst.rows * dst.cols` bytes).
    let dst_weight_ptr = unsafe { (dst_weight_ptr as *mut u8).add(dst_row_offset * dst.cols) };
    // SAFETY: `dst_scale_row_offset + src_scale_rows <= dst.scale_rows` was
    // ensured above, so the offset stays inside `dst.scales`.
    let dst_scale_ptr =
        unsafe { (dst_scale_ptr as *mut f32).add(dst_scale_row_offset * dst.scale_cols) };
    // SAFETY: src pointers are live CudaSlices pinned by the `_g*` guards with
    // lengths matching the ensured shapes; the kernel writes `src.rows` weight
    // rows and `src_scale_rows` scale rows at the bounded offsets above,
    // stream-ordered on `ctx.stream`.
    unsafe {
        ffi::dsv4_block_scaled_to_fp8_deepgemm_cuda(
            src_ptr as *const u8,
            src_scale_ptr as *const u8,
            dst_weight_ptr,
            dst_scale_ptr,
            rows_i32,
            cols_i32,
            scale_rows_i32,
            scale_cols_i32,
            dst_scale_cols_i32,
            source_format as i32,
            ctx.stream.cu_stream(),
        )
        .result()
        .map_err(|err| anyhow!("DeepSeek V4 DeepGEMM FP8 cache build failed: {err}"))?;
    }
    Ok(())
}

fn dsv4_fill_fp8_deepgemm_weight_cache_row_range(
    ctx: &DeviceContext,
    src: &DeviceMatrix,
    dst: &mut Dsv4Fp8DeepGemmWeightCache,
    src_row_start: usize,
    src_rows: usize,
    dst_row_offset: usize,
    dst_scale_row_offset: usize,
) -> Result<()> {
    let source_format = Dsv4DeepGemmSourceFormat::from_weight_format(src.weight_format)?;
    ensure!(
        src_rows > 0,
        "DeepSeek V4 DeepGEMM row-range cache needs rows > 0"
    );
    ensure!(
        src_row_start + src_rows <= src.rows,
        "DeepSeek V4 DeepGEMM source row range [{}..{}) exceeds rows {}",
        src_row_start,
        src_row_start + src_rows,
        src.rows
    );
    ensure!(
        src.cols == dst.cols,
        "DeepSeek V4 DeepGEMM row-range cache K mismatch: source={} cache={}",
        src.cols,
        dst.cols
    );
    ensure!(
        dst_row_offset + src_rows <= dst.rows,
        "DeepSeek V4 DeepGEMM row-range cache dst row overflow: offset={} rows={} cache={}",
        dst_row_offset,
        src_rows,
        dst.rows
    );
    ensure!(
        src_row_start.is_multiple_of(DSV4_DEEPGEMM_FP8_SCALE_GRAN_M)
            && src_rows.is_multiple_of(DSV4_DEEPGEMM_FP8_SCALE_GRAN_M)
            && dst_row_offset.is_multiple_of(DSV4_DEEPGEMM_FP8_SCALE_GRAN_M),
        "DeepSeek V4 DeepGEMM row-range cache rows must be {}-aligned (src_start={} rows={} dst_offset={})",
        DSV4_DEEPGEMM_FP8_SCALE_GRAN_M,
        src_row_start,
        src_rows,
        dst_row_offset
    );
    ensure!(
        src.dsv4_scale_rows > 0 && src.dsv4_scale_cols > 0,
        "DeepSeek V4 DeepGEMM row-range cache source needs DSv4 block scales"
    );
    let src_scale_row_start = src_row_start / DSV4_DEEPGEMM_FP8_SCALE_GRAN_M;
    let src_scale_rows = src_rows / DSV4_DEEPGEMM_FP8_SCALE_GRAN_M;
    ensure!(
        src_scale_row_start + src_scale_rows <= src.dsv4_scale_rows,
        "DeepSeek V4 DeepGEMM row-range scale source overflow: start={} rows={} source={}",
        src_scale_row_start,
        src_scale_rows,
        src.dsv4_scale_rows
    );
    ensure!(
        dst_scale_row_offset + src_scale_rows <= dst.scale_rows,
        "DeepSeek V4 DeepGEMM row-range cache scale row overflow: offset={} rows={} cache={}",
        dst_scale_row_offset,
        src_scale_rows,
        dst.scale_rows
    );

    let qweight = src
        .qweight
        .as_ref()
        .ok_or_else(|| anyhow!("DeepSeek V4 DeepGEMM row-range source missing raw weight bytes"))?;
    let src_scales = src
        .dsv4_scales
        .as_ref()
        .ok_or_else(|| anyhow!("DeepSeek V4 DeepGEMM row-range source missing block scales"))?;
    let bytes_per_src_row = match source_format {
        Dsv4DeepGemmSourceFormat::Fp8 => src.cols,
        Dsv4DeepGemmSourceFormat::Fp4 => {
            ensure!(
                src.cols.is_multiple_of(2),
                "DeepSeek V4 FP4 DeepGEMM row-range source cols must be even, got {}",
                src.cols
            );
            src.cols / 2
        }
    };
    ensure!(
        qweight.len() == src.rows * bytes_per_src_row,
        "DeepSeek V4 DeepGEMM row-range source weight len {} != expected {}",
        qweight.len(),
        src.rows * bytes_per_src_row
    );
    ensure!(
        src_scales.len() == src.dsv4_scale_rows * src.dsv4_scale_cols,
        "DeepSeek V4 DeepGEMM row-range source scale len {} != expected {}",
        src_scales.len(),
        src.dsv4_scale_rows * src.dsv4_scale_cols
    );
    let rows_i32 = i32::try_from(src_rows)
        .map_err(|_| anyhow!("DeepSeek V4 DeepGEMM row-range rows overflow i32"))?;
    let cols_i32 = i32::try_from(src.cols)
        .map_err(|_| anyhow!("DeepSeek V4 DeepGEMM row-range cols overflow i32"))?;
    let scale_rows_i32 = i32::try_from(src_scale_rows)
        .map_err(|_| anyhow!("DeepSeek V4 DeepGEMM row-range scale rows overflow i32"))?;
    let scale_cols_i32 = i32::try_from(src.dsv4_scale_cols)
        .map_err(|_| anyhow!("DeepSeek V4 DeepGEMM row-range scale cols overflow i32"))?;
    let dst_scale_cols_i32 = i32::try_from(dst.scale_cols)
        .map_err(|_| anyhow!("DeepSeek V4 DeepGEMM row-range cache scale cols overflow i32"))?;
    let (src_ptr, _src_guard) = qweight.device_ptr(&ctx.stream);
    let (src_scale_ptr, _src_scale_guard) = src_scales.device_ptr(&ctx.stream);
    let (dst_weight_ptr, _dst_weight_guard) = dst.weight.device_ptr_mut(&ctx.stream);
    let (dst_scale_ptr, _dst_scale_guard) = dst.scales.device_ptr_mut(&ctx.stream);
    // SAFETY: `src_row_start + src_rows <= src.rows` and the qweight length
    // check above keep this offset inside the source weight buffer.
    let src_ptr = unsafe { (src_ptr as *const u8).add(src_row_start * bytes_per_src_row) };
    // SAFETY: `src_scale_row_start + src_scale_rows <= src.dsv4_scale_rows` was
    // ensured above, keeping the offset inside the source scale buffer.
    let src_scale_ptr =
        unsafe { (src_scale_ptr as *const u8).add(src_scale_row_start * src.dsv4_scale_cols) };
    // SAFETY: `dst_row_offset + src_rows <= dst.rows` was ensured above, so the
    // offset stays inside `dst.weight`.
    let dst_weight_ptr = unsafe { (dst_weight_ptr as *mut u8).add(dst_row_offset * dst.cols) };
    // SAFETY: `dst_scale_row_offset + src_scale_rows <= dst.scale_rows` was
    // ensured above, so the offset stays inside `dst.scales`.
    let dst_scale_ptr =
        unsafe { (dst_scale_ptr as *mut f32).add(dst_scale_row_offset * dst.scale_cols) };
    // SAFETY: all four pointers were bounds-offset above from live CudaSlices
    // pinned by the `_g*` guards; the kernel touches `src_rows` weight rows and
    // `src_scale_rows` scale rows only, stream-ordered on `ctx.stream`.
    unsafe {
        ffi::dsv4_block_scaled_to_fp8_deepgemm_cuda(
            src_ptr,
            src_scale_ptr,
            dst_weight_ptr,
            dst_scale_ptr,
            rows_i32,
            cols_i32,
            scale_rows_i32,
            scale_cols_i32,
            dst_scale_cols_i32,
            source_format as i32,
            ctx.stream.cu_stream(),
        )
        .result()
        .map_err(|err| anyhow!("DeepSeek V4 DeepGEMM FP8 row-range cache build failed: {err}"))?;
    }
    Ok(())
}

/// 2D device tensor (matrix) — stored in row-major order as bf16 unless
/// `weight_format` names an explicit packed layout.
pub struct DeviceMatrix {
    pub data: CudaSlice<bf16>,
    pub rows: usize,
    pub cols: usize,
    pub weight_format: WeightFormat,
    /// INT8 quantized weights (if quantized). When set, `data` is unused.
    pub qweight: Option<CudaSlice<i8>>,
    /// ABI-generic unsigned quantized weights (FP8 bytes or packed FP4 bytes).
    pub qweight_u8: Option<CudaSlice<u8>>,
    /// Merge-requant: pristine FP8 base; `qweight_u8`/`scale_f32` then hold merged bytes.
    pub pristine_fp8: Option<(CudaSlice<u8>, CudaSlice<f32>)>,
    /// Per-group bf16 scales for quantized weights. Shape: [rows, cols/group_size].
    pub qscales: Option<CudaSlice<bf16>>,
    /// ABI-generic FP8 E4M3 scale bytes.
    pub qscale_fp8: Option<CudaSlice<u8>>,
    /// ABI-generic direct f32 scale buffer.
    pub scale_f32: Option<CudaSlice<f32>>,
    /// ABI-generic secondary f32 scale buffer (activation metadata in v1).
    pub scale2_f32: Option<CudaSlice<f32>>,
    /// Number of rows in the ABI-generic scale matrix.
    pub quant_scale_rows: usize,
    /// Number of columns in the ABI-generic scale matrix.
    pub quant_scale_cols: usize,
    /// Weight block rows for block-scaled formats.
    pub quant_block_m: usize,
    /// Weight block columns/group size for block-scaled/group formats.
    pub quant_block_k: usize,
    /// DeepSeek V4 block scales encoded as raw FP8 E8M0 bytes.
    pub dsv4_scales: Option<CudaSlice<u8>>,
    /// Number of rows in the DeepSeek V4 block-scale matrix.
    pub dsv4_scale_rows: usize,
    /// Number of columns in the DeepSeek V4 block-scale matrix.
    pub dsv4_scale_cols: usize,
    /// Quantization group size (0 = not quantized).
    pub group_size: usize,
    /// Marlin-repacked INT4 weights for prefill GEMM (None if not W4 or repack failed).
    pub marlin_packed: Option<CudaSlice<u8>>,
    /// FP16 scales in Marlin layout [K/group_size, N] (transposed from qscales).
    pub marlin_scales: Option<CudaSlice<u16>>,
    /// FP32 per-output-channel scales for the W4A8 Marlin path.
    pub marlin_channel_scales: Option<CudaSlice<f32>>,
    /// Per-128x128-block power of two for the NVFP4 DeepGEMM prefill arm,
    /// `[ceil(rows/128) + 1, ceil(cols/128)]` f32. Its presence is also what
    /// keeps the FP4 source resident: that arm contracts the raw nibbles, so
    /// [`Self::free_quant_source_after_marlin`] may not release them.
    pub fp4_deepgemm_sfb: Option<CudaSlice<f32>>,
    /// Hybrid W4 sidecar: W4A8 packed weights for prefill dispatch.
    pub hybrid_w4a8_qweight: Option<CudaSlice<u8>>,
    /// Hybrid W4 sidecar: W4A8 FP32 per-output-channel scales.
    pub hybrid_w4a8_s_channel: Option<CudaSlice<f32>>,
    /// Hybrid W4 sidecar: W4A8 FP16 per-group scales.
    pub hybrid_w4a8_s_group: Option<CudaSlice<u16>>,
    /// Hybrid W4 sidecar: PF8.2 zero-point preprocessed packed weights for
    /// W4+FP8 prefill GEMM.
    pub hybrid_w4_fp8_qweight: Option<CudaSlice<u8>>,
    // -- TurboQuant packed weight storage (Phase 2: fused dequant at runtime) --
    /// TQ packed indices [rows, packed_cols] u8.
    /// 3-bit uses 4-bit nibble packing (2 per byte), 2-bit uses 4 per byte.
    pub tq_packed: Option<CudaSlice<u8>>,
    /// TQ per-group f16 norms `[rows, cols/group_size]`, stored as u16 on device.
    pub tq_scales: Option<CudaSlice<u16>>,
    /// TQ Hadamard signs `[cols]` i8 (+1/-1), shared across rows.
    pub tq_signs: Option<CudaSlice<i8>>,
    /// TQ Lloyd-Max centroids `[2^bits]` f32, shared across all layers.
    pub tq_centroids: Option<CudaSlice<f32>>,
    /// TQ bit width (2, 3, or 4). 0 = not TQ.
    pub tq_bits: u8,
}

/// Host-resident snapshot of an `Option<CudaSlice<T>>` weight buffer, used by
/// the OPD engine time-share offload to hold idle weights in CPU RAM while the
/// device VRAM is freed. `None` means the source buffer was absent (the buffer
/// stays absent on reload).
type OptHostBuf<T> = Option<Vec<T>>;

/// Host-resident snapshot of every device buffer in a [`DeviceMatrix`].
///
/// Captures the full quant-format-agnostic set of side tensors (dense bf16,
/// INT8/INT4 qweight + scales, Marlin packed/scales, hybrid W4A8/W4-FP8
/// sidecars, TurboQuant packed storage) so offload→reload is bit-exact for any
/// weight format. The scalar shape/format fields are restored from the live
/// `DeviceMatrix` they were detached from, so this snapshot only carries the
/// raw buffer bytes.
pub struct HostMatrixSnapshot {
    data: Vec<bf16>,
    qweight: OptHostBuf<i8>,
    qweight_u8: OptHostBuf<u8>,
    pristine_fp8_qweight: OptHostBuf<u8>,
    pristine_fp8_scales: OptHostBuf<f32>,
    qscales: OptHostBuf<bf16>,
    qscale_fp8: OptHostBuf<u8>,
    scale_f32: OptHostBuf<f32>,
    scale2_f32: OptHostBuf<f32>,
    dsv4_scales: OptHostBuf<u8>,
    marlin_packed: OptHostBuf<u8>,
    marlin_scales: OptHostBuf<u16>,
    marlin_channel_scales: OptHostBuf<f32>,
    fp4_deepgemm_sfb: OptHostBuf<f32>,
    hybrid_w4a8_qweight: OptHostBuf<u8>,
    hybrid_w4a8_s_channel: OptHostBuf<f32>,
    hybrid_w4a8_s_group: OptHostBuf<u16>,
    hybrid_w4_fp8_qweight: OptHostBuf<u8>,
    tq_packed: OptHostBuf<u8>,
    tq_scales: OptHostBuf<u16>,
    tq_signs: OptHostBuf<i8>,
    tq_centroids: OptHostBuf<f32>,
    /// The Marlin buffers were dropped rather than copied; the reload rebuilds
    /// them with [`DeviceMatrix::repack_for_marlin_fp8`].
    marlin_fp8_rebuilt: bool,
    /// Total device bytes this snapshot freed when captured (for accounting).
    freed_bytes: usize,
}

impl HostMatrixSnapshot {
    /// Total device VRAM (bytes) freed by capturing this snapshot.
    #[must_use]
    pub fn freed_bytes(&self) -> usize {
        self.freed_bytes
    }
}

/// Copy an optional device buffer to host and report bytes copied.
fn snapshot_opt_slice<T: DeviceRepr + Clone>(
    ctx: &DeviceContext,
    src: &Option<CudaSlice<T>>,
    freed: &mut usize,
) -> Result<OptHostBuf<T>> {
    match src {
        Some(slice) => {
            let host = ctx
                .stream
                .clone_dtoh(slice)
                .map_err(|e| anyhow!("offload D2H copy failed: {e}"))?;
            *freed += host.len() * std::mem::size_of::<T>();
            Ok(Some(host))
        }
        None => Ok(None),
    }
}

/// Device bytes an optional buffer holds, for the offload accounting when it is
/// freed without a host copy.
fn opt_slice_bytes<T>(src: &Option<CudaSlice<T>>) -> usize {
    src.as_ref().map_or(0, CudaSlice::num_bytes)
}

/// Re-upload an optional host buffer to device.
fn restore_opt_slice<T: DeviceRepr>(
    ctx: &DeviceContext,
    host: &Option<Vec<T>>,
) -> Result<Option<CudaSlice<T>>> {
    match host {
        Some(data) => Ok(Some(
            ctx.stream
                .clone_htod(data.as_slice())
                .map_err(|e| anyhow!("reload H2D copy failed: {e}"))?,
        )),
        None => Ok(None),
    }
}

/// Move a raw `CudaSlice<T>` to host RAM, replacing it with a 1-element
/// placeholder and freeing the VRAM. Returns the host copy and bytes freed.
/// Used for the model's bare `CudaSlice<f32>` weight fields (e.g. SSM A_log,
/// norm weights) that are not wrapped in `DeviceVec`/`DeviceMatrix`.
pub fn offload_raw_slice<T: DeviceRepr + Clone + ValidAsZeroBits>(
    ctx: &DeviceContext,
    slice: &mut CudaSlice<T>,
) -> Result<(Vec<T>, usize)> {
    let host = ctx
        .stream
        .clone_dtoh(slice)
        .map_err(|e| anyhow!("offload D2H copy (raw slice) failed: {e}"))?;
    let freed = host.len() * std::mem::size_of::<T>();
    ctx.sync()?;
    *slice = ctx
        .stream
        .alloc_zeros::<T>(1)
        .map_err(|e| anyhow!("offload raw-slice placeholder alloc failed: {e}"))?;
    Ok((host, freed))
}

/// Restore a raw `CudaSlice<T>` from a host snapshot, re-allocating VRAM.
pub fn reload_raw_slice<T: DeviceRepr>(
    ctx: &DeviceContext,
    slice: &mut CudaSlice<T>,
    host: &[T],
) -> Result<()> {
    *slice = ctx
        .stream
        .clone_htod(host)
        .map_err(|e| anyhow!("reload H2D copy (raw slice) failed: {e}"))?;
    ctx.sync()?;
    Ok(())
}

impl DeviceMatrix {
    /// Raw device pointer to the dense BF16 `data` buffer as a `u64`.
    ///
    /// Used to build the per-expert weight-pointer table (`*const u64`) the MoE
    /// grouped-GEMM kernels consume: each entry is one expert's `DeviceMatrix`
    /// device pointer. Only valid for the dense BF16 path (`data` populated);
    /// quantized formats store weights in the side buffers, not `data`.
    pub fn device_ptr(&self, ctx: &DeviceContext) -> u64 {
        use cudarc::driver::DevicePtr;
        let (ptr, _sync) = self.data.device_ptr(&ctx.stream);
        ptr
    }

    /// Resident FP8 block-scaled weight pointers for read-only foreign borrow
    /// (train-infer weight sharing, `--share-frozen-base`).
    ///
    /// Returns `Some((qweight_u8_ptr, scale_f32_ptr, rows, cols, block_m,
    /// block_k))` ONLY when this matrix is stored as block-scaled FP8 with both
    /// the `qweight_u8` byte buffer and the `scale_f32` scale buffer resident
    /// (the layout `from_fp8_block_scaled` produces). Any other weight format —
    /// or a matrix currently offloaded (placeholder buffers) — yields `None`.
    ///
    /// The returned `u64`s are raw `CUdeviceptr`s into THIS matrix's resident
    /// VRAM; the borrower must keep this `DeviceMatrix` resident (no offload,
    /// no LoRA re-merge replacing the buffers) for the borrow's lifetime.
    /// Pristine FP8 pair once merge-requant split it out, else the live slots.
    ///
    /// The live-slot fallback is format-gated: an `Fp4E2M1Group` matrix also
    /// fills `qweight_u8` (packed nibbles, half the byte count) and `scale_f32`
    /// (a 1-element global scale), and reading those as an FP8 block-scaled pair
    /// walks off the end of both buffers. `pristine_fp8` needs no check — only
    /// `requant_merged_matrix` writes it, and only on the FP8 path.
    pub fn merge_base_fp8(&self) -> Option<(&CudaSlice<u8>, &CudaSlice<f32>)> {
        if let Some((qw, sc)) = self.pristine_fp8.as_ref() {
            return Some((qw, sc));
        }
        if self.weight_format != WeightFormat::Fp8BlockScaled {
            return None;
        }
        self.qweight_u8.as_ref().zip(self.scale_f32.as_ref())
    }

    pub fn fp8_block_scaled_ptrs(
        &self,
        ctx: &DeviceContext,
    ) -> Option<(u64, u64, usize, usize, usize, usize)> {
        use cudarc::driver::DevicePtr;
        if self.weight_format != WeightFormat::Fp8BlockScaled {
            return None;
        }
        let (qweight, scales) = self.merge_base_fp8()?;
        let (wptr, _wsync) = qweight.device_ptr(&ctx.stream);
        let (sptr, _ssync) = scales.device_ptr(&ctx.stream);
        Some((
            wptr,
            sptr,
            self.rows,
            self.cols,
            self.quant_block_m,
            self.quant_block_k,
        ))
    }

    /// Copy the dense BF16 `data` buffer to host as f32 (for testing/training).
    ///
    /// Mirrors [`DeviceVec::to_host`]; only reads the dense `data` field, not
    /// quantized side buffers.
    pub fn to_host(&self, ctx: &DeviceContext) -> Result<Vec<f32>> {
        let host_f16 = ctx
            .stream
            .clone_dtoh(&self.data)
            .map_err(|e| anyhow!("D2H copy failed: {}", e))?;
        ctx.sync()?;
        Ok(host_f16.iter().map(|x| x.to_f32()).collect())
    }

    /// Move every device weight buffer to host RAM and free the VRAM.
    ///
    /// Returns a [`HostMatrixSnapshot`] the caller holds until reload. The
    /// live device buffers are replaced with 1-element placeholders so the
    /// struct stays valid (it must not be forwarded through while offloaded).
    /// Format-agnostic: handles dense, INT8/INT4, Marlin, hybrid W4, and TQ.
    pub fn offload_to_host(&mut self, ctx: &DeviceContext) -> Result<HostMatrixSnapshot> {
        let mut freed = 0usize;
        let data = ctx
            .stream
            .clone_dtoh(&self.data)
            .map_err(|e| anyhow!("offload D2H copy (data) failed: {e}"))?;
        freed += data.len() * std::mem::size_of::<bf16>();

        let (pristine_qweight, pristine_scales) = match self.pristine_fp8.take() {
            Some((qw, sc)) => (Some(qw), Some(sc)),
            None => (None, None),
        };
        // A per-channel FP8 weight that kept its E4M3 source for the DeepGEMM
        // prefill arm holds the same bytes twice; copying both to host doubles
        // the snapshot and the PCIe traffic in each direction.
        let marlin_fp8_rebuilt = self.weight_format == WeightFormat::Fp8BlockScaled
            && self.quant_block_m == 1
            && self.marlin_packed.is_some()
            && self.qweight_u8.is_some()
            && self.scale_f32.is_some();
        let (marlin_packed, marlin_scales) = if marlin_fp8_rebuilt {
            freed += opt_slice_bytes(&self.marlin_packed) + opt_slice_bytes(&self.marlin_scales);
            (None, None)
        } else {
            (
                snapshot_opt_slice(ctx, &self.marlin_packed, &mut freed)?,
                snapshot_opt_slice(ctx, &self.marlin_scales, &mut freed)?,
            )
        };
        let snapshot = HostMatrixSnapshot {
            data,
            qweight: snapshot_opt_slice(ctx, &self.qweight, &mut freed)?,
            qweight_u8: snapshot_opt_slice(ctx, &self.qweight_u8, &mut freed)?,
            pristine_fp8_qweight: snapshot_opt_slice(ctx, &pristine_qweight, &mut freed)?,
            pristine_fp8_scales: snapshot_opt_slice(ctx, &pristine_scales, &mut freed)?,
            qscales: snapshot_opt_slice(ctx, &self.qscales, &mut freed)?,
            qscale_fp8: snapshot_opt_slice(ctx, &self.qscale_fp8, &mut freed)?,
            scale_f32: snapshot_opt_slice(ctx, &self.scale_f32, &mut freed)?,
            scale2_f32: snapshot_opt_slice(ctx, &self.scale2_f32, &mut freed)?,
            dsv4_scales: snapshot_opt_slice(ctx, &self.dsv4_scales, &mut freed)?,
            marlin_packed,
            marlin_scales,
            marlin_channel_scales: snapshot_opt_slice(
                ctx,
                &self.marlin_channel_scales,
                &mut freed,
            )?,
            fp4_deepgemm_sfb: snapshot_opt_slice(ctx, &self.fp4_deepgemm_sfb, &mut freed)?,
            hybrid_w4a8_qweight: snapshot_opt_slice(ctx, &self.hybrid_w4a8_qweight, &mut freed)?,
            hybrid_w4a8_s_channel: snapshot_opt_slice(
                ctx,
                &self.hybrid_w4a8_s_channel,
                &mut freed,
            )?,
            hybrid_w4a8_s_group: snapshot_opt_slice(ctx, &self.hybrid_w4a8_s_group, &mut freed)?,
            hybrid_w4_fp8_qweight: snapshot_opt_slice(
                ctx,
                &self.hybrid_w4_fp8_qweight,
                &mut freed,
            )?,
            tq_packed: snapshot_opt_slice(ctx, &self.tq_packed, &mut freed)?,
            tq_scales: snapshot_opt_slice(ctx, &self.tq_scales, &mut freed)?,
            tq_signs: snapshot_opt_slice(ctx, &self.tq_signs, &mut freed)?,
            tq_centroids: snapshot_opt_slice(ctx, &self.tq_centroids, &mut freed)?,
            marlin_fp8_rebuilt,
            freed_bytes: 0,
        };
        ctx.sync()?;

        // Drop the device buffers (return blocks to the async pool). Replace
        // `data` with a 1-element placeholder so the struct stays well-formed.
        let placeholder = ctx
            .stream
            .alloc_zeros::<bf16>(1)
            .map_err(|e| anyhow!("offload placeholder alloc failed: {e}"))?;
        self.data = placeholder;
        self.qweight = None;
        self.qweight_u8 = None;
        self.qscales = None;
        self.qscale_fp8 = None;
        self.scale_f32 = None;
        self.scale2_f32 = None;
        self.dsv4_scales = None;
        self.marlin_packed = None;
        self.marlin_scales = None;
        self.marlin_channel_scales = None;
        self.fp4_deepgemm_sfb = None;
        self.hybrid_w4a8_qweight = None;
        self.hybrid_w4a8_s_channel = None;
        self.hybrid_w4a8_s_group = None;
        self.hybrid_w4_fp8_qweight = None;
        self.tq_packed = None;
        self.tq_scales = None;
        self.tq_signs = None;
        self.tq_centroids = None;

        Ok(HostMatrixSnapshot {
            freed_bytes: freed,
            ..snapshot
        })
    }

    /// Restore device buffers from a host snapshot, re-allocating VRAM.
    pub fn reload_from_host(
        &mut self,
        ctx: &DeviceContext,
        snapshot: &HostMatrixSnapshot,
    ) -> Result<()> {
        self.data = ctx
            .stream
            .clone_htod(snapshot.data.as_slice())
            .map_err(|e| anyhow!("reload H2D copy (data) failed: {e}"))?;
        self.qweight = restore_opt_slice(ctx, &snapshot.qweight)?;
        self.qweight_u8 = restore_opt_slice(ctx, &snapshot.qweight_u8)?;
        self.pristine_fp8 = match (
            restore_opt_slice(ctx, &snapshot.pristine_fp8_qweight)?,
            restore_opt_slice(ctx, &snapshot.pristine_fp8_scales)?,
        ) {
            (Some(qw), Some(sc)) => Some((qw, sc)),
            _ => None,
        };
        self.qscales = restore_opt_slice(ctx, &snapshot.qscales)?;
        self.qscale_fp8 = restore_opt_slice(ctx, &snapshot.qscale_fp8)?;
        self.scale_f32 = restore_opt_slice(ctx, &snapshot.scale_f32)?;
        self.scale2_f32 = restore_opt_slice(ctx, &snapshot.scale2_f32)?;
        self.dsv4_scales = restore_opt_slice(ctx, &snapshot.dsv4_scales)?;
        self.marlin_packed = restore_opt_slice(ctx, &snapshot.marlin_packed)?;
        self.marlin_scales = restore_opt_slice(ctx, &snapshot.marlin_scales)?;
        self.marlin_channel_scales = restore_opt_slice(ctx, &snapshot.marlin_channel_scales)?;
        self.fp4_deepgemm_sfb = restore_opt_slice(ctx, &snapshot.fp4_deepgemm_sfb)?;
        self.hybrid_w4a8_qweight = restore_opt_slice(ctx, &snapshot.hybrid_w4a8_qweight)?;
        self.hybrid_w4a8_s_channel = restore_opt_slice(ctx, &snapshot.hybrid_w4a8_s_channel)?;
        self.hybrid_w4a8_s_group = restore_opt_slice(ctx, &snapshot.hybrid_w4a8_s_group)?;
        self.hybrid_w4_fp8_qweight = restore_opt_slice(ctx, &snapshot.hybrid_w4_fp8_qweight)?;
        self.tq_packed = restore_opt_slice(ctx, &snapshot.tq_packed)?;
        self.tq_scales = restore_opt_slice(ctx, &snapshot.tq_scales)?;
        self.tq_signs = restore_opt_slice(ctx, &snapshot.tq_signs)?;
        self.tq_centroids = restore_opt_slice(ctx, &snapshot.tq_centroids)?;
        ctx.sync()?;
        if snapshot.marlin_fp8_rebuilt {
            // Costs this weight's load-time repack again, against a host
            // snapshot and a PCIe leg that would otherwise both be twice as big.
            self.repack_for_marlin_fp8(ctx)?;
            ensure!(
                self.marlin_packed.is_some() && self.marlin_scales.is_some(),
                "reload: FP8 per-channel Marlin repack produced no layout for a matrix that \
                 carried one at offload ({}x{})",
                self.rows,
                self.cols
            );
        }
        Ok(())
    }

    /// Create from host data (row-major, bf16)
    pub fn from_host(ctx: &DeviceContext, data: &[bf16], rows: usize, cols: usize) -> Result<Self> {
        assert_eq!(data.len(), rows * cols);
        let gpu_data = ctx
            .stream
            .clone_htod(data)
            .map_err(|e| anyhow!("H2D copy failed: {}", e))?;
        Ok(Self {
            data: gpu_data,
            rows,
            cols,
            weight_format: WeightFormat::DenseBf16,
            qweight: None,
            qweight_u8: None,
            pristine_fp8: None,
            qscales: None,
            qscale_fp8: None,
            scale_f32: None,
            scale2_f32: None,
            quant_scale_rows: 0,
            quant_scale_cols: 0,
            quant_block_m: 0,
            quant_block_k: 0,
            dsv4_scales: None,
            dsv4_scale_rows: 0,
            dsv4_scale_cols: 0,
            group_size: 0,
            marlin_packed: None,
            marlin_scales: None,
            marlin_channel_scales: None,
            fp4_deepgemm_sfb: None,
            hybrid_w4a8_qweight: None,
            hybrid_w4a8_s_channel: None,
            hybrid_w4a8_s_group: None,
            hybrid_w4_fp8_qweight: None,
            tq_packed: None,
            tq_scales: None,
            tq_signs: None,
            tq_centroids: None,
            tq_bits: 0,
        })
    }

    /// Create from INT8 quantized weight + bf16 scales.
    pub fn from_quantized_int8(
        ctx: &DeviceContext,
        qweight_data: &[i8],
        scales_data: &[bf16],
        rows: usize,
        cols: usize,
        group_size: usize,
    ) -> Result<Self> {
        WeightFormat::W8A16.validate_shape(rows, cols, group_size)?;
        ensure!(qweight_data.len() == rows * cols);
        let num_groups = cols / group_size;
        ensure!(scales_data.len() == rows * num_groups);

        let qw = ctx
            .stream
            .clone_htod(qweight_data)
            .map_err(|e| anyhow!("H2D qweight failed: {}", e))?;
        let qs = ctx
            .stream
            .clone_htod(scales_data)
            .map_err(|e| anyhow!("H2D scales failed: {}", e))?;
        // Allocate dummy bf16 data (1 element, unused)
        let dummy = ctx
            .stream
            .alloc_zeros::<bf16>(1)
            .map_err(|e| anyhow!("Alloc dummy: {}", e))?;
        Ok(Self {
            data: dummy,
            rows,
            cols,
            weight_format: WeightFormat::W8A16,
            qweight: Some(qw),
            qweight_u8: None,
            pristine_fp8: None,
            qscales: Some(qs),
            qscale_fp8: None,
            scale_f32: None,
            scale2_f32: None,
            quant_scale_rows: 0,
            quant_scale_cols: 0,
            quant_block_m: 0,
            quant_block_k: 0,
            dsv4_scales: None,
            dsv4_scale_rows: 0,
            dsv4_scale_cols: 0,
            group_size,
            marlin_packed: None,
            marlin_scales: None,
            marlin_channel_scales: None,
            fp4_deepgemm_sfb: None,
            hybrid_w4a8_qweight: None,
            hybrid_w4a8_s_channel: None,
            hybrid_w4a8_s_group: None,
            hybrid_w4_fp8_qweight: None,
            tq_packed: None,
            tq_scales: None,
            tq_signs: None,
            tq_centroids: None,
            tq_bits: 0,
        })
    }

    /// All-default matrix around a dense `data` buffer; `fuse_rows` arms then
    /// set their format-specific fields.
    fn from_parts_dense(data: CudaSlice<bf16>, rows: usize, cols: usize) -> Self {
        Self {
            data,
            rows,
            cols,
            weight_format: WeightFormat::DenseBf16,
            qweight: None,
            qweight_u8: None,
            pristine_fp8: None,
            qscales: None,
            qscale_fp8: None,
            scale_f32: None,
            scale2_f32: None,
            quant_scale_rows: 0,
            quant_scale_cols: 0,
            quant_block_m: 0,
            quant_block_k: 0,
            dsv4_scales: None,
            dsv4_scale_rows: 0,
            dsv4_scale_cols: 0,
            group_size: 0,
            marlin_packed: None,
            marlin_scales: None,
            marlin_channel_scales: None,
            fp4_deepgemm_sfb: None,
            hybrid_w4a8_qweight: None,
            hybrid_w4a8_s_channel: None,
            hybrid_w4a8_s_group: None,
            hybrid_w4_fp8_qweight: None,
            tq_packed: None,
            tq_scales: None,
            tq_signs: None,
            tq_centroids: None,
            tq_bits: 0,
        }
    }

    /// True once [`Self::free_quant_source_after_marlin`] has released the
    /// pre-repack bytes. [`Self::merge_base_fp8`] returns `None` both for a
    /// matrix that was never block-scaled FP8 and for one whose source is gone,
    /// and a caller that reads the second as the first merges into a buffer the
    /// GEMM no longer reads. Ask this before treating `None` as "not quantised".
    ///
    /// It reports only whether the bytes are gone, never whether they are safe
    /// to write: a per-channel FP8 weight the DeepGEMM prefill arm can serve
    /// keeps its source, so this reads false and `merge_base_fp8` hands back a
    /// live pair — one the Marlin layout beside it does not track.
    pub fn quant_source_freed(&self) -> bool {
        self.marlin_packed.is_some()
            && self.qweight_u8.is_none()
            && matches!(
                self.weight_format,
                WeightFormat::Fp4E2M1Group | WeightFormat::Fp8BlockScaled
            )
    }

    /// Release the pre-repack quantized bytes once a Marlin layout exists,
    /// unless another arm still contracts them.
    ///
    /// Marlin holds its own copy of every weight byte it reads, so leaving the
    /// source resident stores the model twice: measured on Qwen3.8-27B-NVFP4,
    /// 40.1 GB resident for a 22.8 GB checkpoint, which halved the KV pool
    /// against the FP8 model on the same card (281,577 vs 593,995 tokens) and
    /// forced slot reuse on a long-agent workload. Frees only the two large
    /// buffers; the scalar scales stay for the accessors that report format.
    ///
    /// The two formats differ because NVFP4 has exactly one arm and per-channel
    /// FP8 has two. Nothing but Marlin can read NVFP4's packed nibbles, so its
    /// source is dead the moment the layout exists. Per-channel FP8 also has
    /// DeepGEMM's native FP8 MMA at prefill M, and that kernel contracts the
    /// plain `[N, K]` E4M3 bytes — Marlin's tile layout is a permutation of them
    /// and cannot stand in. `fp8_deepgemm_available` is the caller's answer to
    /// "can that arm ever fire for this weight, on this engine" — shape and card,
    /// but also whether any forward reaches the arm's M floor and whether this
    /// weight is one a prefill chunk is ever multiplied by. False keeps the
    /// unconditional free.
    ///
    /// Where the source IS released, no route may still ask for it — every M has
    /// to reach a Marlin arm — or the dispatch errors out with
    /// "missing qweight_u8" rather than reading freed memory.
    pub fn free_quant_source_after_marlin(&mut self, fp8_deepgemm_available: bool) {
        // Both, because that is exactly what the routes test before sending a
        // weight to a Marlin arm (`fp4_route` / `fp8_route`). Freeing on
        // `marlin_packed` alone would strand a weight that has a layout but no
        // scales on the scalar arm with nothing to read.
        if self.marlin_packed.is_none() || self.marlin_scales.is_none() {
            return;
        }
        match self.weight_format {
            // `fp4_deepgemm_sfb` is set only when the NVFP4 DeepGEMM prefill arm
            // can serve this weight, and that arm widens the raw nibbles to
            // E4M3 per call — Marlin's tile layout is a permutation of them and
            // cannot stand in.
            WeightFormat::Fp4E2M1Group if self.fp4_deepgemm_sfb.is_none() => {
                self.qweight_u8 = None;
                self.qscale_fp8 = None;
            }
            WeightFormat::Fp8BlockScaled if !fp8_deepgemm_available => {
                self.qweight_u8 = None;
            }
            _ => {}
        }
    }

    /// Row-concatenate two weight matrices (`[a; b]` along output rows) so one
    /// GEMM serves both projections — the decode launch-count lever. Formats
    /// covered: DenseBf16 (`data`), pre-repack W8A16 (`qweight`+`qscales`;
    /// fuse BEFORE `repack_for_marlin_w8a16` so the fused matrix repacks and
    /// frees its INT8 source once), Fp8BlockScaled (`qweight_u8`+`scale_f32`,
    /// needs `a.rows % block_m == 0` so the scale grids stack cleanly).
    pub fn fuse_rows(ctx: &DeviceContext, a: &DeviceMatrix, b: &DeviceMatrix) -> Result<Self> {
        ensure!(
            a.weight_format == b.weight_format && a.cols == b.cols,
            "fuse_rows needs matching format/K: {} [{}x{}] vs {} [{}x{}]",
            a.weight_format,
            a.rows,
            a.cols,
            b.weight_format,
            b.rows,
            b.cols
        );
        fn concat<T: DeviceRepr + ValidAsZeroBits>(
            ctx: &DeviceContext,
            x: &CudaSlice<T>,
            y: &CudaSlice<T>,
        ) -> Result<CudaSlice<T>> {
            let mut out = ctx
                .stream
                .alloc_zeros::<T>(x.len() + y.len())
                .map_err(|e| anyhow!("fuse_rows alloc failed: {e}"))?;
            {
                let src = x.slice(0..x.len());
                let mut dst = out.slice_mut(0..x.len());
                ctx.stream
                    .memcpy_dtod(&src, &mut dst)
                    .map_err(|e| anyhow!("fuse_rows D2D (first) failed: {e}"))?;
            }
            {
                let src = y.slice(0..y.len());
                let mut dst = out.slice_mut(x.len()..x.len() + y.len());
                ctx.stream
                    .memcpy_dtod(&src, &mut dst)
                    .map_err(|e| anyhow!("fuse_rows D2D (second) failed: {e}"))?;
            }
            Ok(out)
        }
        let rows = a.rows + b.rows;
        let mut fused = match a.weight_format {
            WeightFormat::DenseBf16 => {
                ensure!(
                    a.data.len() == a.rows * a.cols && b.data.len() == b.rows * b.cols,
                    "fuse_rows dense data len mismatch"
                );
                let mut m = Self::from_parts_dense(concat(ctx, &a.data, &b.data)?, rows, a.cols);
                m.weight_format = WeightFormat::DenseBf16;
                m
            }
            WeightFormat::W8A16 => {
                ensure!(
                    a.group_size == b.group_size && a.marlin_packed.is_none(),
                    "fuse_rows W8A16 needs matching group_size and pre-repack sources"
                );
                let qa = a
                    .qweight
                    .as_ref()
                    .ok_or_else(|| anyhow!("a missing qweight"))?;
                let qb = b
                    .qweight
                    .as_ref()
                    .ok_or_else(|| anyhow!("b missing qweight"))?;
                let sa = a
                    .qscales
                    .as_ref()
                    .ok_or_else(|| anyhow!("a missing qscales"))?;
                let sb = b
                    .qscales
                    .as_ref()
                    .ok_or_else(|| anyhow!("b missing qscales"))?;
                let mut m = Self::from_parts_dense(
                    ctx.stream
                        .alloc_zeros::<bf16>(1)
                        .map_err(|e| anyhow!("fuse_rows dummy alloc failed: {e}"))?,
                    rows,
                    a.cols,
                );
                m.weight_format = WeightFormat::W8A16;
                m.qweight = Some(concat(ctx, qa, qb)?);
                m.qscales = Some(concat(ctx, sa, sb)?);
                m.group_size = a.group_size;
                m
            }
            WeightFormat::W4A16 => {
                ensure!(
                    a.group_size == b.group_size && a.marlin_packed.is_none(),
                    "fuse_rows W4A16 needs matching group_size and pre-repack sources"
                );
                let qa = a
                    .qweight
                    .as_ref()
                    .ok_or_else(|| anyhow!("a missing qweight"))?;
                let qb = b
                    .qweight
                    .as_ref()
                    .ok_or_else(|| anyhow!("b missing qweight"))?;
                let sa = a
                    .qscales
                    .as_ref()
                    .ok_or_else(|| anyhow!("a missing qscales"))?;
                let sb = b
                    .qscales
                    .as_ref()
                    .ok_or_else(|| anyhow!("b missing qscales"))?;
                let mut m = Self::from_parts_dense(
                    ctx.stream
                        .alloc_zeros::<bf16>(1)
                        .map_err(|e| anyhow!("fuse_rows dummy alloc failed: {e}"))?,
                    rows,
                    a.cols,
                );
                m.weight_format = WeightFormat::W4A16;
                m.qweight = Some(concat(ctx, qa, qb)?);
                m.qscales = Some(concat(ctx, sa, sb)?);
                m.group_size = a.group_size;
                m
            }
            WeightFormat::Fp8BlockScaled => {
                ensure!(
                    a.quant_block_m == b.quant_block_m
                        && a.quant_block_k == b.quant_block_k
                        && a.quant_block_m > 0
                        && a.rows.is_multiple_of(a.quant_block_m),
                    "fuse_rows FP8 needs matching blocks and a.rows % block_m == 0 \
                     (block {}x{}, a.rows {})",
                    a.quant_block_m,
                    a.quant_block_k,
                    a.rows
                );
                ensure!(
                    a.quant_scale_cols == b.quant_scale_cols,
                    "fuse_rows FP8 scale col mismatch"
                );
                let qa = a
                    .qweight_u8
                    .as_ref()
                    .ok_or_else(|| anyhow!("a missing qweight_u8"))?;
                let qb = b
                    .qweight_u8
                    .as_ref()
                    .ok_or_else(|| anyhow!("b missing qweight_u8"))?;
                let sa = a
                    .scale_f32
                    .as_ref()
                    .ok_or_else(|| anyhow!("a missing scale_f32"))?;
                let sb = b
                    .scale_f32
                    .as_ref()
                    .ok_or_else(|| anyhow!("b missing scale_f32"))?;
                let mut m = Self::from_parts_dense(
                    ctx.stream
                        .alloc_zeros::<bf16>(1)
                        .map_err(|e| anyhow!("fuse_rows dummy alloc failed: {e}"))?,
                    rows,
                    a.cols,
                );
                m.weight_format = WeightFormat::Fp8BlockScaled;
                m.qweight_u8 = Some(concat(ctx, qa, qb)?);
                m.scale_f32 = Some(concat(ctx, sa, sb)?);
                m.quant_scale_rows = a.quant_scale_rows + b.quant_scale_rows;
                m.quant_scale_cols = a.quant_scale_cols;
                m.quant_block_m = a.quant_block_m;
                m.quant_block_k = a.quant_block_k;
                m
            }
            WeightFormat::Fp4E2M1Group => {
                ensure!(
                    a.group_size == b.group_size && a.cols == b.cols,
                    "fuse_rows FP4 group shape mismatch"
                );
                let qa = a
                    .qweight_u8
                    .as_ref()
                    .ok_or_else(|| anyhow!("a missing qweight_u8"))?;
                let qb = b
                    .qweight_u8
                    .as_ref()
                    .ok_or_else(|| anyhow!("b missing qweight_u8"))?;
                let qsa = a
                    .qscale_fp8
                    .as_ref()
                    .ok_or_else(|| anyhow!("a missing qscale_fp8"))?;
                let qsb = b
                    .qscale_fp8
                    .as_ref()
                    .ok_or_else(|| anyhow!("b missing qscale_fp8"))?;
                // Global weight/input scales are per-tensor scalars; both parts
                // share the same value in practice (same quant config group).
                let mut m = Self::from_parts_dense(
                    ctx.stream
                        .alloc_zeros::<bf16>(1)
                        .map_err(|e| anyhow!("fuse_rows dummy alloc failed: {e}"))?,
                    rows,
                    a.cols,
                );
                m.weight_format = WeightFormat::Fp4E2M1Group;
                m.qweight_u8 = Some(concat(ctx, qa, qb)?);
                m.qscale_fp8 = Some(concat(ctx, qsa, qsb)?);
                m.scale_f32 = a.scale_f32.clone();
                m.scale2_f32 = a.scale2_f32.clone();
                m.quant_scale_rows = a.quant_scale_rows + b.quant_scale_rows;
                m.quant_scale_cols = a.quant_scale_cols;
                m.quant_block_m = a.quant_block_m;
                m.quant_block_k = a.quant_block_k;
                m.group_size = a.group_size;
                m
            }
            other => bail!("fuse_rows unsupported for weight format {other}"),
        };
        fused.rows = rows;
        Ok(fused)
    }

    /// Create from INT4 packed quantized weight + bf16 scales.
    /// Unpacks INT4 → INT8 at load time for the W8 kernel.
    /// TODO: integrate Marlin kernel for native W4 prefill, AWQ-style GEMV for decode.
    pub fn from_quantized_int4(
        ctx: &DeviceContext,
        packed_data: &[u8],
        scales_data: &[bf16],
        rows: usize,
        cols: usize,
        group_size: usize,
    ) -> Result<Self> {
        WeightFormat::W4A16.validate_shape(rows, cols, group_size)?;
        ensure!(
            cols.is_multiple_of(2),
            "W4A16 requires cols % 2 == 0, got {cols}"
        );
        ensure!(packed_data.len() == rows * cols / 2);
        let num_groups = cols / group_size;
        ensure!(scales_data.len() == rows * num_groups);

        // Upload packed INT4 data directly — native W4 kernel handles nibble extraction
        let qw: CudaSlice<i8> = ctx
            .stream
            // SAFETY: same-size reinterpret of `&[u8]` as `&[i8]` (align 1,
            // every bit pattern valid) for the typed H2D upload.
            .clone_htod(unsafe {
                std::slice::from_raw_parts(packed_data.as_ptr().cast::<i8>(), packed_data.len())
            })
            .map_err(|e| anyhow!("H2D qweight int4 failed: {}", e))?;
        let qs = ctx
            .stream
            .clone_htod(scales_data)
            .map_err(|e| anyhow!("H2D scales failed: {}", e))?;
        let dummy = ctx
            .stream
            .alloc_zeros::<bf16>(1)
            .map_err(|e| anyhow!("Alloc dummy: {}", e))?;
        Ok(Self {
            data: dummy,
            rows,
            cols,
            weight_format: WeightFormat::W4A16,
            qweight: Some(qw),
            qweight_u8: None,
            pristine_fp8: None,
            qscales: Some(qs),
            qscale_fp8: None,
            scale_f32: None,
            scale2_f32: None,
            quant_scale_rows: 0,
            quant_scale_cols: 0,
            quant_block_m: 0,
            quant_block_k: 0,
            dsv4_scales: None,
            dsv4_scale_rows: 0,
            dsv4_scale_cols: 0,
            group_size,
            marlin_packed: None,
            marlin_scales: None,
            marlin_channel_scales: None,
            fp4_deepgemm_sfb: None,
            hybrid_w4a8_qweight: None,
            hybrid_w4a8_s_channel: None,
            hybrid_w4a8_s_group: None,
            hybrid_w4_fp8_qweight: None,
            tq_packed: None,
            tq_scales: None,
            tq_signs: None,
            tq_centroids: None,
            tq_bits: 0,
        })
    }

    /// Create from DeepSeek V4 FP8 E4M3 weights plus FP8 E8M0 block scales.
    pub fn from_dsv4_fp8_block_scaled(
        ctx: &DeviceContext,
        weight_bytes: &[u8],
        scale_bytes: &[u8],
        rows: usize,
        cols: usize,
        scale_rows: usize,
        scale_cols: usize,
    ) -> Result<Self> {
        WeightFormat::Dsv4Fp8BlockScaled.validate_shape(rows, cols, 0)?;
        ensure!(
            weight_bytes.len() == rows * cols,
            "DeepSeek V4 FP8 weight bytes {} != expected {} for rows={rows} cols={cols}",
            weight_bytes.len(),
            rows * cols
        );
        ensure!(
            scale_rows > 0 && scale_cols > 0,
            "DeepSeek V4 FP8 scale shape must be non-empty"
        );
        ensure!(
            scale_bytes.len() == scale_rows * scale_cols,
            "DeepSeek V4 FP8 scale bytes {} != expected {}",
            scale_bytes.len(),
            scale_rows * scale_cols
        );

        let qw: CudaSlice<i8> = ctx
            .stream
            // SAFETY: same-size reinterpret of `&[u8]` as `&[i8]` (align 1,
            // every bit pattern valid) for the typed H2D upload.
            .clone_htod(unsafe {
                std::slice::from_raw_parts(weight_bytes.as_ptr().cast::<i8>(), weight_bytes.len())
            })
            .map_err(|e| anyhow!("H2D DeepSeek V4 FP8 weight failed: {e}"))?;
        let scales = ctx
            .stream
            .clone_htod(scale_bytes)
            .map_err(|e| anyhow!("H2D DeepSeek V4 FP8 scales failed: {e}"))?;
        let dummy = ctx
            .stream
            .alloc_zeros::<bf16>(1)
            .map_err(|e| anyhow!("Alloc dummy: {e}"))?;

        let matrix = Self {
            data: dummy,
            rows,
            cols,
            weight_format: WeightFormat::Dsv4Fp8BlockScaled,
            qweight: Some(qw),
            qweight_u8: None,
            pristine_fp8: None,
            qscales: None,
            qscale_fp8: None,
            scale_f32: None,
            scale2_f32: None,
            quant_scale_rows: 0,
            quant_scale_cols: 0,
            quant_block_m: 0,
            quant_block_k: 0,
            dsv4_scales: Some(scales),
            dsv4_scale_rows: scale_rows,
            dsv4_scale_cols: scale_cols,
            group_size: 0,
            marlin_packed: None,
            marlin_scales: None,
            marlin_channel_scales: None,
            fp4_deepgemm_sfb: None,
            hybrid_w4a8_qweight: None,
            hybrid_w4a8_s_channel: None,
            hybrid_w4a8_s_group: None,
            hybrid_w4_fp8_qweight: None,
            tq_packed: None,
            tq_scales: None,
            tq_signs: None,
            tq_centroids: None,
            tq_bits: 0,
        };
        Ok(matrix)
    }

    /// Create from DeepSeek V4 FP4 E2M1 weights (2 packed per I8 byte) plus FP8
    /// E8M0 block scales. `cols` is the logical (unpacked) K; the stored weight
    /// buffer holds `rows * cols / 2` bytes.
    pub fn from_dsv4_fp4_block_scaled(
        ctx: &DeviceContext,
        weight_bytes: &[u8],
        scale_bytes: &[u8],
        rows: usize,
        cols: usize,
        scale_rows: usize,
        scale_cols: usize,
    ) -> Result<Self> {
        WeightFormat::Dsv4Fp4BlockScaled.validate_shape(rows, cols, 0)?;
        ensure!(
            weight_bytes.len() == rows * (cols / 2),
            "DeepSeek V4 FP4 weight bytes {} != expected {} for rows={rows} cols={cols}",
            weight_bytes.len(),
            rows * (cols / 2)
        );
        ensure!(
            scale_rows > 0 && scale_cols > 0,
            "DeepSeek V4 FP4 scale shape must be non-empty"
        );
        ensure!(
            scale_bytes.len() == scale_rows * scale_cols,
            "DeepSeek V4 FP4 scale bytes {} != expected {}",
            scale_bytes.len(),
            scale_rows * scale_cols
        );

        let qw: CudaSlice<i8> = ctx
            .stream
            // SAFETY: same-size reinterpret of `&[u8]` as `&[i8]` (align 1,
            // every bit pattern valid) for the typed H2D upload.
            .clone_htod(unsafe {
                std::slice::from_raw_parts(weight_bytes.as_ptr().cast::<i8>(), weight_bytes.len())
            })
            .map_err(|e| anyhow!("H2D DeepSeek V4 FP4 weight failed: {e}"))?;
        let scales = ctx
            .stream
            .clone_htod(scale_bytes)
            .map_err(|e| anyhow!("H2D DeepSeek V4 FP4 scales failed: {e}"))?;
        let dummy = ctx
            .stream
            .alloc_zeros::<bf16>(1)
            .map_err(|e| anyhow!("Alloc dummy: {e}"))?;

        let matrix = Self {
            data: dummy,
            rows,
            cols,
            weight_format: WeightFormat::Dsv4Fp4BlockScaled,
            qweight: Some(qw),
            qweight_u8: None,
            pristine_fp8: None,
            qscales: None,
            qscale_fp8: None,
            scale_f32: None,
            scale2_f32: None,
            quant_scale_rows: 0,
            quant_scale_cols: 0,
            quant_block_m: 0,
            quant_block_k: 0,
            dsv4_scales: Some(scales),
            dsv4_scale_rows: scale_rows,
            dsv4_scale_cols: scale_cols,
            group_size: 0,
            marlin_packed: None,
            marlin_scales: None,
            marlin_channel_scales: None,
            fp4_deepgemm_sfb: None,
            hybrid_w4a8_qweight: None,
            hybrid_w4a8_s_channel: None,
            hybrid_w4a8_s_group: None,
            hybrid_w4_fp8_qweight: None,
            tq_packed: None,
            tq_scales: None,
            tq_signs: None,
            tq_centroids: None,
            tq_bits: 0,
        };
        Ok(matrix)
    }

    /// Create from ABI-generic FP8 E4M3 weights plus direct f32 block scales.
    #[allow(clippy::too_many_arguments)]
    pub fn from_fp8_block_scaled(
        ctx: &DeviceContext,
        weight_bytes: &[u8],
        scale_f32: &[f32],
        rows: usize,
        cols: usize,
        block_m: usize,
        block_k: usize,
    ) -> Result<Self> {
        WeightFormat::Fp8BlockScaled.validate_shape(rows, cols, 0)?;
        ensure!(block_m > 0, "Fp8BlockScaled requires block_m > 0");
        ensure!(block_k > 0, "Fp8BlockScaled requires block_k > 0");
        ensure!(
            weight_bytes.len() == rows * cols,
            "FP8 block-scaled weight bytes {} != expected {} for rows={rows} cols={cols}",
            weight_bytes.len(),
            rows * cols
        );
        let scale_rows = rows.div_ceil(block_m);
        let scale_cols = cols.div_ceil(block_k);
        ensure!(
            scale_f32.len() == scale_rows * scale_cols,
            "FP8 block-scaled scales {} != expected {}",
            scale_f32.len(),
            scale_rows * scale_cols
        );

        let qweight = ctx
            .stream
            .clone_htod(weight_bytes)
            .map_err(|e| anyhow!("H2D FP8 block-scaled weight failed: {e}"))?;
        let scales = ctx
            .stream
            .clone_htod(scale_f32)
            .map_err(|e| anyhow!("H2D FP8 block-scaled scales failed: {e}"))?;
        let dummy = ctx
            .stream
            .alloc_zeros::<bf16>(1)
            .map_err(|e| anyhow!("Alloc dummy: {e}"))?;

        Ok(Self {
            data: dummy,
            rows,
            cols,
            weight_format: WeightFormat::Fp8BlockScaled,
            qweight: None,
            qweight_u8: Some(qweight),
            pristine_fp8: None,
            qscales: None,
            qscale_fp8: None,
            scale_f32: Some(scales),
            scale2_f32: None,
            quant_scale_rows: scale_rows,
            quant_scale_cols: scale_cols,
            quant_block_m: block_m,
            quant_block_k: block_k,
            dsv4_scales: None,
            dsv4_scale_rows: 0,
            dsv4_scale_cols: 0,
            group_size: 0,
            marlin_packed: None,
            marlin_scales: None,
            marlin_channel_scales: None,
            fp4_deepgemm_sfb: None,
            hybrid_w4a8_qweight: None,
            hybrid_w4a8_s_channel: None,
            hybrid_w4a8_s_group: None,
            hybrid_w4_fp8_qweight: None,
            tq_packed: None,
            tq_scales: None,
            tq_signs: None,
            tq_centroids: None,
            tq_bits: 0,
        })
    }

    /// Create from ABI-generic FP8 E4M3 weights plus direct f32 per-shard scales.
    pub fn from_fp8_per_shard(
        ctx: &DeviceContext,
        weight_bytes: &[u8],
        scale_f32: &[f32],
        input_scale_f32: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        WeightFormat::Fp8PerShard.validate_shape(rows, cols, 0)?;
        ensure!(
            weight_bytes.len() == rows * cols,
            "FP8 per-shard weight bytes {} != expected {} for rows={rows} cols={cols}",
            weight_bytes.len(),
            rows * cols
        );
        ensure!(
            !scale_f32.is_empty(),
            "FP8 per-shard weight scales must be non-empty"
        );
        ensure!(
            !input_scale_f32.is_empty(),
            "FP8 per-shard input scales must be non-empty"
        );

        let qweight = ctx
            .stream
            .clone_htod(weight_bytes)
            .map_err(|e| anyhow!("H2D FP8 per-shard weight failed: {e}"))?;
        let scales = ctx
            .stream
            .clone_htod(scale_f32)
            .map_err(|e| anyhow!("H2D FP8 per-shard scales failed: {e}"))?;
        let input_scales = ctx
            .stream
            .clone_htod(input_scale_f32)
            .map_err(|e| anyhow!("H2D FP8 per-shard input scales failed: {e}"))?;
        let dummy = ctx
            .stream
            .alloc_zeros::<bf16>(1)
            .map_err(|e| anyhow!("Alloc dummy: {e}"))?;

        Ok(Self {
            data: dummy,
            rows,
            cols,
            weight_format: WeightFormat::Fp8PerShard,
            qweight: None,
            qweight_u8: Some(qweight),
            pristine_fp8: None,
            qscales: None,
            qscale_fp8: None,
            scale_f32: Some(scales),
            scale2_f32: Some(input_scales),
            quant_scale_rows: scale_f32.len(),
            quant_scale_cols: 1,
            quant_block_m: 0,
            quant_block_k: 0,
            dsv4_scales: None,
            dsv4_scale_rows: 0,
            dsv4_scale_cols: 0,
            group_size: 0,
            marlin_packed: None,
            marlin_scales: None,
            marlin_channel_scales: None,
            fp4_deepgemm_sfb: None,
            hybrid_w4a8_qweight: None,
            hybrid_w4a8_s_channel: None,
            hybrid_w4a8_s_group: None,
            hybrid_w4_fp8_qweight: None,
            tq_packed: None,
            tq_scales: None,
            tq_signs: None,
            tq_centroids: None,
            tq_bits: 0,
        })
    }

    /// Create from packed FP4 E2M1 weights plus FP8 group scales and direct f32 globals.
    #[allow(clippy::too_many_arguments)]
    pub fn from_fp4_e2m1_group(
        ctx: &DeviceContext,
        packed_bytes: &[u8],
        scale_fp8: &[u8],
        global_scale_f32: &[f32],
        input_scale_f32: Option<&[f32]>,
        rows: usize,
        logical_cols: usize,
        group_size: usize,
    ) -> Result<Self> {
        WeightFormat::Fp4E2M1Group.validate_shape(rows, logical_cols, group_size)?;
        ensure!(
            packed_bytes.len() == rows * logical_cols / 2,
            "FP4 E2M1 packed bytes {} != expected {} for rows={rows} cols={logical_cols}",
            packed_bytes.len(),
            rows * logical_cols / 2
        );
        let scale_cols = logical_cols / group_size;
        ensure!(
            scale_fp8.len() == rows * scale_cols,
            "FP4 E2M1 group scales {} != expected {}",
            scale_fp8.len(),
            rows * scale_cols
        );
        ensure!(
            !global_scale_f32.is_empty(),
            "FP4 E2M1 global scale must be non-empty"
        );

        let qweight = ctx
            .stream
            .clone_htod(packed_bytes)
            .map_err(|e| anyhow!("H2D FP4 E2M1 weight failed: {e}"))?;
        let qscale = ctx
            .stream
            .clone_htod(scale_fp8)
            .map_err(|e| anyhow!("H2D FP4 E2M1 group scales failed: {e}"))?;
        let global = ctx
            .stream
            .clone_htod(global_scale_f32)
            .map_err(|e| anyhow!("H2D FP4 E2M1 global scales failed: {e}"))?;
        let input = match input_scale_f32 {
            Some(scales) => Some(
                ctx.stream
                    .clone_htod(scales)
                    .map_err(|e| anyhow!("H2D FP4 E2M1 input scales failed: {e}"))?,
            ),
            None => None,
        };
        let dummy = ctx
            .stream
            .alloc_zeros::<bf16>(1)
            .map_err(|e| anyhow!("Alloc dummy: {e}"))?;

        Ok(Self {
            data: dummy,
            rows,
            cols: logical_cols,
            weight_format: WeightFormat::Fp4E2M1Group,
            qweight: None,
            qweight_u8: Some(qweight),
            pristine_fp8: None,
            qscales: None,
            qscale_fp8: Some(qscale),
            scale_f32: Some(global),
            scale2_f32: input,
            quant_scale_rows: rows,
            quant_scale_cols: scale_cols,
            quant_block_m: 1,
            quant_block_k: group_size,
            dsv4_scales: None,
            dsv4_scale_rows: 0,
            dsv4_scale_cols: 0,
            group_size,
            marlin_packed: None,
            marlin_scales: None,
            marlin_channel_scales: None,
            fp4_deepgemm_sfb: None,
            hybrid_w4a8_qweight: None,
            hybrid_w4a8_s_channel: None,
            hybrid_w4a8_s_group: None,
            hybrid_w4_fp8_qweight: None,
            tq_packed: None,
            tq_scales: None,
            tq_signs: None,
            tq_centroids: None,
            tq_bits: 0,
        })
    }

    /// Whether this matrix uses quantized weights.
    pub fn is_quantized(&self) -> bool {
        self.weight_format.is_quantized()
            && (self.qweight.is_some()
                || self.qweight_u8.is_some()
                || self.tq_packed.is_some()
                || self.marlin_packed.is_some())
    }

    /// Whether this matrix's active weight format is dense BF16 (the forward
    /// path reads from `data`). Retired FP8/qweight buffers may still be present
    /// (kept alive for the share-frozen-base student alias); they are not used.
    pub fn is_dense_bf16(&self) -> bool {
        self.weight_format == WeightFormat::DenseBf16
    }

    #[must_use]
    pub fn weight_format(&self) -> WeightFormat {
        self.weight_format
    }

    /// Build the Marlin tensor-core layout for a W8A16 weight: re-encode signed
    /// INT8 → uint8b128 (+128), pack to GPTQ `[K/4, N]` i32, GPU-repack to Marlin
    /// tiles, and transpose+permute the BF16 group scales to `[K/gs, N]` (Marlin's
    /// length-64 `scale_perm`). Stores into `marlin_packed`/`marlin_scales`; the
    /// GEMM (`marlin_w8a16_gemm_cuda`) consumes them, scales stay BF16 (matches the
    /// bf16 kernel). No-op (leaves marlin_* None → scalar fallback) when the shape
    /// isn't Marlin tile-aligned. SM-gated by the caller (Ampere+).
    pub fn repack_for_marlin_w8a16(&mut self, ctx: &DeviceContext) -> Result<()> {
        if self.weight_format != WeightFormat::W8A16
            || self.qweight.is_none()
            || self.qscales.is_none()
            || self.group_size == 0
        {
            return Ok(());
        }
        // Ampere+ only (Marlin uses mma.sync/cp.async). Below sm_80 leave marlin_*
        // None so dispatch keeps the dequant→BF16 / scalar path — the shim would
        // otherwise return NOT_SUPPORTED and fail the load.
        if ctx.compute_capability().0 < 8 {
            return Ok(());
        }
        let n = self.rows; // output dim
        let k = self.cols; // input dim
        // kU8B128 is instantiated only for gs ∈ {32,64,128}; other gs → no-op kernel.
        if !k.is_multiple_of(16)
            || !n.is_multiple_of(64)
            || !k.is_multiple_of(self.group_size)
            || !matches!(self.group_size, 32 | 64 | 128)
        {
            log::warn!(
                "Marlin W8A16 repack skipped: [{n}x{k}] gs={} (need K%16, N%64, gs∈{{32,64,128}}); scalar path",
                self.group_size
            );
            return Ok(());
        }

        // Step 1: signed INT8 [N, K] row-major → uint8b128 GPTQ [K/4, N] i32.
        // element (n,k): u8 = int8+128, packed 4-per-word at bits (k%4)*8.
        let qw = self.qweight.as_ref().unwrap();
        let weight_host: Vec<i8> = ctx
            .stream
            .clone_dtoh(qw)
            .map_err(|e| anyhow!("D2H W8A16 qweight: {}", e))?;
        let gptq_rows = k / 4;
        let mut gptq = vec![0u32; gptq_rows * n];
        for row_n in 0..n {
            for col_k in 0..k {
                let u8v = (i16::from(weight_host[row_n * k + col_k]) + 128) as u32 & 0xFF;
                let gptq_row = col_k / 4;
                let bit_pos = (col_k % 4) * 8;
                gptq[gptq_row * n + row_n] |= u8v << bit_pos;
            }
        }
        // SAFETY: views the live `Vec<u32>` as its byte representation (u8 align 1,
        // len*4 bytes); `gptq` outlives the borrow.
        let gptq_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(gptq.as_ptr().cast::<u8>(), gptq.len() * 4) };
        let gptq_gpu: CudaSlice<u8> = ctx
            .stream
            .clone_htod(gptq_bytes)
            .map_err(|e| anyhow!("H2D W8A16 GPTQ: {}", e))?;

        // Marlin output: [K/16, N*4] i32 = K*N/4 i32 = K*N bytes.
        let mut marlin_gpu: CudaSlice<u8> = ctx
            .stream
            .alloc_zeros(k * n)
            .map_err(|e| anyhow!("Alloc W8A16 Marlin: {}", e))?;

        // Step 2: GPTQ → Marlin tile layout on GPU.
        {
            let (gptq_ptr, _g1) = gptq_gpu.device_ptr(&ctx.stream);
            let (marlin_ptr, _g2) = marlin_gpu.device_ptr_mut(&ctx.stream);
            // SAFETY: both from live CudaSlices pinned by the guards; K*N-byte
            // input / output verified tile-aligned above, stream-ordered.
            unsafe {
                ffi::marlin_gptq_repack_w8a16_cuda(
                    gptq_ptr as *const u32,
                    marlin_ptr as *mut u32,
                    k as i32,
                    n as i32,
                    ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|e| anyhow!("W8A16 Marlin repack failed: {:?}", e))?;
            }
        }

        // Step 3: scales [N, K/gs] bf16 → transpose to [K/gs, N] → Marlin permute.
        // scale_perm is an 8×8 transpose within each 64-column block:
        //   perm[out] = (out%8)*8 + (out/8)   (vLLM get_scale_perms, len 64).
        // Kept BF16 (the bf16 GEMM reinterprets scales as scalar_t2).
        let qs = self.qscales.as_ref().unwrap();
        let scales_host: Vec<bf16> = ctx
            .stream
            .clone_dtoh(qs)
            .map_err(|e| anyhow!("D2H W8A16 scales: {}", e))?;
        let num_groups = k / self.group_size;
        let mut scales_t = vec![bf16::from_f32(0.0); num_groups * n];
        for row_n in 0..n {
            for g in 0..num_groups {
                scales_t[g * n + row_n] = scales_host[row_n * num_groups + g];
            }
        }
        // Permute within each 64-block of the flattened [num_groups, N] array
        // (N%64==0 → blocks align to N-column runs).
        let mut scales_perm = vec![0u16; num_groups * n];
        for block in 0..(num_groups * n / 64) {
            let base = block * 64;
            for out in 0..64 {
                let src = (out % 8) * 8 + (out / 8);
                scales_perm[base + out] = scales_t[base + src].to_bits();
            }
        }
        let scales_gpu: CudaSlice<u16> = ctx
            .stream
            .clone_htod(&scales_perm)
            .map_err(|e| anyhow!("H2D W8A16 Marlin scales: {}", e))?;

        self.marlin_packed = Some(marlin_gpu);
        self.marlin_scales = Some(scales_gpu);
        // Marlin consumes only marlin_packed/marlin_scales; drop the source int8
        // weight + scales to realize the W8A16 VRAM win (else both resident).
        self.qweight = None;
        self.qscales = None;

        Ok(())
    }

    /// Build the Marlin tensor-core layout for a PER-CHANNEL FP8 weight
    /// (compressed-tensors float-quantized: `F8_E4M3` + one BF16 scale per output
    /// row). Marlin's channelwise mode is `group_size = -1` -> `group_blocks = -1`,
    /// already instantiated for `kFE4M3fn` by `BIGGROUP_GET_IF`, so this needs no
    /// new kernel. No-op (leaves `marlin_packed` None -> GEMV fallback) outside the
    /// tile alignment. SM-gated by the caller (Ampere+).
    ///
    /// Two details this format does not share with W8A16:
    ///
    /// **No `+128`.** `kU8B128` stores `int8 + 128` and its dequant subtracts the
    /// bias; the `kFE4M3fn` dequant reads the raw E4M3 byte (sign bit kept, the
    /// 4-bit exponent field shifted into BF16's 8-bit one), so the byte is packed
    /// unchanged.
    ///
    /// **The scale absorbs 2^120.** `dequant_skip_flop` is `!is_int_type`, and
    /// `is_int_type` covers only the kU4/kU8 family, so `kFE4M3fn` takes the
    /// skip-flop arm and the kernel never applies its exponent-rebias multiply.
    /// Shifting an E4M3 exponent (bias 7) into a BF16 field (bias 127) without
    /// rebiasing scales every weight by `2^-120`, and unlike NVFP4 there is no `s2`
    /// global-scale channel to park the correction in — only `kFE2M1f` reads
    /// `scale2_ptr`. So the per-channel scale carries it. The fold overflows only
    /// at `scale >= 255.5`; this checkpoint's channel scales are ~1e-3..1e-1,
    /// landing at 2^110..2^117, and 2^120 is an exact power of two so the fold
    /// shifts the exponent without touching the mantissa.
    ///
    /// Unlike NVFP4, nothing underflows. For `size_bits() == 8 && group_blocks ==
    /// -1` the scale multiplies the FP32 accumulator after the K loop
    /// (`marlin_template.h:1548`), not the BF16 weight fragment before the MMA
    /// (`:1136`, the kFE2M1f site that returned `nonzero 0/256`). The fragment
    /// carries `w * 2^-120` down to 2^-129, inside bf16; the accumulator's 2^-120
    /// and the scale's 2^+120 cancel bit-exactly in f32.
    ///
    /// Leaves the source `qweight_u8` / `scale_f32` in place; the loader calls
    /// [`Self::free_quant_source_after_marlin`] once it knows whether the
    /// DeepGEMM prefill arm still needs them.
    pub fn repack_for_marlin_fp8(&mut self, ctx: &DeviceContext) -> Result<()> {
        if self.weight_format != WeightFormat::Fp8BlockScaled
            || self.qweight_u8.is_none()
            || self.scale_f32.is_none()
        {
            return Ok(());
        }
        if ctx.compute_capability().0 < 8 {
            return Ok(());
        }
        let n = self.rows; // output dim
        let k = self.cols; // input dim
        // Per-channel only: one scale per output row, spanning all of K.
        // `block_m == 1` is the discriminator — a 128x128 block-scaled weight
        // belongs to DeepGEMM and fails it. `block_k >= k` rather than `== k` so a
        // TP shard, whose `cols` is a slice of the K the scale was defined over,
        // still qualifies.
        // K%64, not the repack's K%16: `min_thread_k = 64` and every thread config
        // has thread_k in {64,128} (marlin.cuh:18, gptq_marlin.cuh:115-129), so a K
        // the GEMM's `is_valid_config` rejects would repack cleanly here and then
        // throw on every call. N%64 matches min_thread_n and the repack's tile_n.
        if self.quant_block_m != 1
            || self.quant_block_k < k
            || !k.is_multiple_of(64)
            || !n.is_multiple_of(64)
        {
            return Ok(());
        }

        // Step 1: raw E4M3 [N, K] row-major -> GPTQ [K/4, N] i32, element (n,k) at
        // bit (k%4)*8 of word (k/4)*N + n. No bias: see the note above. Each (n,k)
        // owns exactly one byte, so this is a byte-granular transpose, written
        // straight into the little-endian byte position the u32 packing implies.
        // Blocked: the destination stride is 4*N bytes, and this runs over every
        // per-channel FP8 weight in the model at load.
        let qw = self.qweight_u8.as_ref().unwrap();
        let weight_host: Vec<u8> = ctx
            .stream
            .clone_dtoh(qw)
            .map_err(|e| anyhow!("D2H FP8 qweight: {}", e))?;
        ensure!(
            weight_host.len() == n * k,
            "FP8 per-channel weight is {} bytes, expected {n}*{k}",
            weight_host.len()
        );
        const TILE: usize = 64;
        let mut gptq_bytes = vec![0u8; n * k];
        for n0 in (0..n).step_by(TILE) {
            for k0 in (0..k).step_by(TILE) {
                let k_end = (k0 + TILE).min(k);
                for row_n in n0..(n0 + TILE).min(n) {
                    for (i, &byte) in weight_host[row_n * k + k0..row_n * k + k_end]
                        .iter()
                        .enumerate()
                    {
                        let col_k = k0 + i;
                        gptq_bytes[((col_k / 4) * n + row_n) * 4 + (col_k % 4)] = byte;
                    }
                }
            }
        }
        let gptq_gpu: CudaSlice<u8> = ctx
            .stream
            .clone_htod(&gptq_bytes)
            .map_err(|e| anyhow!("H2D FP8 GPTQ: {}", e))?;

        // Step 2: GPTQ -> Marlin tiles. The repack is a pure 8-bit lane shuffle, so
        // the kU8B128 kernel serves kFE4M3fn unchanged.
        let mut marlin_gpu: CudaSlice<u8> = ctx
            .stream
            .alloc_zeros(k * n)
            .map_err(|e| anyhow!("Alloc FP8 Marlin: {}", e))?;
        {
            let (gptq_ptr, _g1) = gptq_gpu.device_ptr(&ctx.stream);
            let (marlin_ptr, _g2) = marlin_gpu.device_ptr_mut(&ctx.stream);
            // SAFETY: both from live CudaSlices pinned by the guards; K*N bytes in
            // and out, tile alignment checked above, stream-ordered.
            unsafe {
                ffi::marlin_gptq_repack_w8a16_cuda(
                    gptq_ptr as *const u32,
                    marlin_ptr as *mut u32,
                    k as i32,
                    n as i32,
                    ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|e| anyhow!("FP8 Marlin repack failed: {:?}", e))?;
            }
        }

        // Step 3: scales [N] f32 -> [1, N] bf16, folding 2^120, then vLLM's
        // `scale_perm_single` (channelwise) — NOT the length-64 `scale_perm` the
        // grouped path uses.
        let scales_host: Vec<f32> = ctx
            .stream
            .clone_dtoh(self.scale_f32.as_ref().unwrap())
            .map_err(|e| anyhow!("D2H FP8 scales: {}", e))?;
        ensure!(
            scales_host.len() == n,
            "per-channel FP8 needs {n} scales, got {}",
            scales_host.len()
        );
        // 2^120 as a bit pattern, not a decimal literal: the whole no-precision-loss
        // argument is that the fold is a pure exponent shift.
        const SKIP_FLOP_FACTOR: f32 = f32::from_bits(0x7B80_0000);
        // The fold is bit-exact only if the source scale is already bf16-exact,
        // which this checkpoint's weight_scale is. An f32-scale checkpoint takes a
        // coherent per-output-channel 2^-9 bias instead — say so rather than assume.
        if let Some(s) = scales_host
            .iter()
            .find(|&&s| bf16::from_f32(s).to_f32() != s)
        {
            log::warn!(
                "Marlin FP8 scales are not bf16-exact (e.g. {s}); the fold costs up to 2^-9 per channel"
            );
        }
        // bf16 rounds to +inf at scale >= 255.5. Testing the f32 product would let
        // (3.3895e38, 3.4028e38] through and store an infinite channel scale. A
        // scale with no Marlin encoding means skip the weight, not fail the load.
        if scales_host
            .iter()
            .any(|&s| !bf16::from_f32(s * SKIP_FLOP_FACTOR).is_finite())
        {
            log::warn!(
                "Marlin FP8 repack skipped: [{n}x{k}] channel scale overflows bf16 once 2^120 is folded in (limit 255.5); scalar path"
            );
            return Ok(());
        }
        let mut perm_single = [0usize; 32];
        for i in 0..4 {
            for (jj, off) in [0usize, 1, 8, 9, 16, 17, 24, 25].into_iter().enumerate() {
                perm_single[i * 8 + jj] = 2 * i + off;
            }
        }
        let mut scales_perm = vec![0u16; n];
        for block in 0..(n / 32) {
            let base = block * 32;
            for out in 0..32 {
                let v = scales_host[base + perm_single[out]] * SKIP_FLOP_FACTOR;
                scales_perm[base + out] = bf16::from_f32(v).to_bits();
            }
        }
        let scales_gpu: CudaSlice<u16> = ctx
            .stream
            .clone_htod(&scales_perm)
            .map_err(|e| anyhow!("H2D FP8 Marlin scales: {}", e))?;

        self.marlin_packed = Some(marlin_gpu);
        self.marlin_scales = Some(scales_gpu);
        Ok(())
    }

    /// Build the Marlin tensor-core layout for an NVFP4 weight
    /// (compressed-tensors nvfp4-pack-quantized): transpose the packed E2M1
    /// nibbles into GPTQ `[K/8, N]` i32, GPU-repack to Marlin tiles, and
    /// re-encode the FP8 E4M3 group scales into the S0E5M3 form Marlin's FP8
    /// scale dequant expects. `marlin_packed` holds both, concatenated —
    /// `[K*N/2 weight bytes][N*K/16 scale bytes]`, one allocation because the
    /// two are always read together — and `marlin_scales` the single BF16
    /// global scale. No-op (leaves `marlin_packed` None → scalar GEMV fallback)
    /// when the shape or group size is outside the kFE2M1f kernel's
    /// instantiation. SM-gated by the caller (Ampere+).
    ///
    /// The source `qweight_u8` / `qscale_fp8` / `scale_f32` stay resident:
    /// Marlin loses to dequant→cuBLAS above M≈1024 (measured on H20), so the
    /// 2048-token prefill chunk keeps reading them. That costs ~1.13x the
    /// packed-weight VRAM.
    ///
    /// Scale encoding (vLLM `marlin_utils_fp4.nvfp4_marlin_process_scales`):
    /// the kernel's weight dequant leaves a 2^-126 factor and the scale dequant
    /// reads an 8-bit `[E5|M3]` field, so the byte is the high half of
    /// `f16(scale * 2^7) << 1` and the leftover 2^119 is folded into the global
    /// scale. `scale_factor` is a power of two that lifts every scale above the
    /// E5-MSB-set floor; the global scale is divided by it.
    /// Build the per-128x128-block power of two the NVFP4 DeepGEMM prefill arm
    /// divides out of the weight and hands DeepGEMM back as `sfb`.
    ///
    /// The group scale cannot ride inside the E4M3 weight value: this
    /// checkpoint's `weight_scale` reaches E4M3's full 448 and an E2M1 value
    /// reaches 6, so the product reaches 2688 against a 448 ceiling. Dividing
    /// out a per-block power of two puts every block's peak in (224, 448] and
    /// costs nothing — a power of two is exact both ways, and DeepGEMM's `sfb`
    /// multiplies it back into the fp32 accumulator.
    ///
    /// The floor is measured, not assumed: over this checkpoint's 168 NVFP4
    /// scale tensors the widest 128x128 block spans 6.81 binades, which leaves
    /// the smallest folded value at 0.332 against E4M3's 0.0156 normal minimum.
    ///
    /// Caller decides whether the arm is reachable at all (SM tier, DeepGEMM
    /// built and enabled); this only checks that the shape and metadata fit.
    /// Setting the buffer also pins the FP4 source — see
    /// [`Self::free_quant_source_after_marlin`].
    pub fn prepare_fp4_deepgemm_sfb(&mut self, ctx: &DeviceContext) -> Result<()> {
        if self.weight_format != WeightFormat::Fp4E2M1Group
            || self.qweight_u8.is_none()
            || self.qscale_fp8.is_none()
            || self.scale_f32.is_none()
        {
            return Ok(());
        }
        let n = self.rows;
        let k = self.cols;
        // DeepGEMM's dense NT entry wants `k % 128` and `n % 8`; the block
        // reduction additionally needs the 128-column tile to be a whole number
        // of groups.
        if self.group_size == 0
            || !128usize.is_multiple_of(self.group_size)
            || !k.is_multiple_of(128)
            || !n.is_multiple_of(8)
            || self.quant_scale_rows != n
            || self.quant_scale_cols != k / self.group_size
        {
            return Ok(());
        }
        let scales = self.qscale_fp8.as_ref().expect("checked above");
        let global = self.scale_f32.as_ref().expect("checked above");
        let len = (n.div_ceil(128) + 1) * k.div_ceil(128);
        let sfb = ctx
            .stream
            .alloc_zeros::<f32>(len)
            .map_err(|e| anyhow!("NVFP4 DeepGEMM sfb alloc failed: {e}"))?;
        {
            let (scales_ptr, _gs) = scales.device_ptr(&ctx.stream);
            let (global_ptr, _gg) = global.device_ptr(&ctx.stream);
            let (sfb_ptr, _gf) = sfb.device_ptr(&ctx.stream);
            // SAFETY: ptrs from live device allocations sized to the dims passed.
            unsafe {
                crate::ffi::fp4_group_scale_block_pow2_cuda(
                    scales_ptr as *const u8,
                    global_ptr as *const f32,
                    sfb_ptr as *mut f32,
                    n as i32,
                    k as i32,
                    self.group_size as i32,
                    self.quant_scale_cols as i32,
                    ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|e| anyhow!("NVFP4 DeepGEMM sfb kernel failed: {e}"))?;
            }
        }
        self.fp4_deepgemm_sfb = Some(sfb);
        Ok(())
    }

    pub fn repack_for_marlin_fp4(&mut self, ctx: &DeviceContext) -> Result<()> {
        if self.weight_format != WeightFormat::Fp4E2M1Group
            || self.qweight_u8.is_none()
            || self.qscale_fp8.is_none()
            || self.scale_f32.is_none()
        {
            return Ok(());
        }
        if ctx.compute_capability().0 < 8 {
            return Ok(());
        }
        let n = self.rows; // output dim
        let k = self.cols; // input dim
        // kFE2M1f is instantiated only at group_blocks == 1 (group_size 16);
        // the tile grid needs N % 64 and K % 64.
        if self.group_size != 16
            || !k.is_multiple_of(64)
            || !n.is_multiple_of(64)
            || self.quant_scale_rows != n
            || self.quant_scale_cols != k / 16
        {
            log::warn!(
                "Marlin NVFP4 repack skipped: [{n}x{k}] gs={} scales=[{}x{}] (need K%64, N%64, gs=16); scalar path",
                self.group_size,
                self.quant_scale_rows,
                self.quant_scale_cols
            );
            return Ok(());
        }
        let global_host: Vec<f32> = ctx
            .stream
            .clone_dtoh(self.scale_f32.as_ref().unwrap())
            .map_err(|e| anyhow!("D2H NVFP4 global scale: {}", e))?;
        if global_host.len() != 1 {
            log::warn!(
                "Marlin NVFP4 repack skipped: {} global scales, expected 1; scalar path",
                global_host.len()
            );
            return Ok(());
        }

        // Step 1: packed [N, K/2] u8 → GPTQ [K/8, N] i32. Each row's u32 view is
        // already k-major inside the word (nibble k at bit (k%8)*4), so this is a
        // plain transpose of that view — no bit shuffling.
        let qw = self.qweight_u8.as_ref().unwrap();
        let packed_host: Vec<u8> = ctx
            .stream
            .clone_dtoh(qw)
            .map_err(|e| anyhow!("D2H NVFP4 qweight: {}", e))?;
        ensure!(
            packed_host.len() == n * (k / 2),
            "NVFP4 qweight {} bytes, expected {}",
            packed_host.len(),
            n * (k / 2)
        );
        let words_per_row = k / 8;
        let mut gptq = vec![0u32; words_per_row * n];
        for row_n in 0..n {
            let base = row_n * (k / 2);
            for j in 0..words_per_row {
                let b = &packed_host[base + j * 4..base + j * 4 + 4];
                gptq[j * n + row_n] = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            }
        }
        // SAFETY: views the live `Vec<u32>` as its byte representation (u8 align 1,
        // len*4 bytes); `gptq` outlives the borrow.
        let gptq_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(gptq.as_ptr().cast::<u8>(), gptq.len() * 4) };
        let gptq_gpu: CudaSlice<u8> = ctx
            .stream
            .clone_htod(gptq_bytes)
            .map_err(|e| anyhow!("H2D NVFP4 GPTQ: {}", e))?;

        // Marlin weight: [K/16, N*2] i32 = K*N/8 i32 = K*N/2 bytes, then the
        // S0E5M3 scale bytes in the tail of the same allocation.
        let weight_bytes = k * n / 2;
        let mut marlin_gpu: CudaSlice<u8> = ctx
            .stream
            .alloc_zeros(weight_bytes + n * (k / 16))
            .map_err(|e| anyhow!("Alloc NVFP4 Marlin: {}", e))?;
        {
            let (gptq_ptr, _g1) = gptq_gpu.device_ptr(&ctx.stream);
            let (marlin_ptr, _g2) = marlin_gpu.device_ptr_mut(&ctx.stream);
            // SAFETY: both from live CudaSlices pinned by the guards; sizes checked
            // tile-aligned above, stream-ordered.
            unsafe {
                ffi::marlin_gptq_repack_fp4_cuda(
                    gptq_ptr as *const u32,
                    marlin_ptr as *mut u32,
                    k as i32,
                    n as i32,
                    ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|e| anyhow!("NVFP4 Marlin repack failed: {:?}", e))?;
            }
        }

        // Step 2: scales [N, K/16] E4M3 → transpose [K/16, N] → Marlin permute
        // (8×8 transpose inside each 64-run) → the FP8-dequant pair order
        // [0,2,1,3] inside each 4-run → S0E5M3 bytes.
        let scales_host: Vec<u8> = ctx
            .stream
            .clone_dtoh(self.qscale_fp8.as_ref().unwrap())
            .map_err(|e| anyhow!("D2H NVFP4 scales: {}", e))?;
        let num_groups = k / 16;
        ensure!(
            scales_host.len() == n * num_groups,
            "NVFP4 scales {} bytes, expected {}",
            scales_host.len(),
            n * num_groups
        );
        let mut sflat = vec![0f32; num_groups * n];
        for row_n in 0..n {
            for g in 0..num_groups {
                sflat[g * n + row_n] = e4m3_to_f32(scales_host[row_n * num_groups + g]);
            }
        }
        let mut sperm = vec![0f32; num_groups * n];
        for block in 0..(num_groups * n / 64) {
            let base = block * 64;
            for out in 0..64 {
                sperm[base + out] = sflat[base + (out % 8) * 8 + (out / 8)];
            }
        }
        for quad in sperm.chunks_exact_mut(4) {
            quad.swap(1, 2);
        }
        // The S0E5M3 field only spans exponents whose MSB is set, i.e. scale*2^7
        // >= 2; lift everything by the largest power of two that keeps the max
        // inside E4M3 range and divide it back out of the global scale.
        let smax = sperm.iter().fold(0f32, |a, &b| a.max(b)) * 128.0;
        let ceiling = 448.0f32 * 128.0;
        let scale_factor = if smax > 0.0 && smax < ceiling {
            (ceiling / smax).log2().floor().exp2()
        } else {
            1.0
        };
        let mut sbytes = vec![0u8; num_groups * n];
        for (dst, &src) in sbytes.iter_mut().zip(sperm.iter()) {
            let v = src * scale_factor * 128.0;
            let v = if v < 2.0 { 0.0 } else { v };
            *dst = (f16::from_f32(v).to_bits() >> 7) as u8;
        }
        {
            let mut tail = marlin_gpu.slice_mut(weight_bytes..weight_bytes + sbytes.len());
            ctx.stream
                .memcpy_htod(sbytes.as_slice(), &mut tail)
                .map_err(|e| anyhow!("H2D NVFP4 Marlin scales: {}", e))?;
        }

        // Step 3: global scale, pre-multiplied by the 2^119 dequant bias.
        let global = bf16::from_f32(
            (f64::from(global_host[0]) * 2f64.powi(119) / f64::from(scale_factor)) as f32,
        );
        let global_gpu: CudaSlice<u16> = ctx
            .stream
            .clone_htod(&[global.to_bits()])
            .map_err(|e| anyhow!("H2D NVFP4 Marlin global scale: {}", e))?;

        self.marlin_packed = Some(marlin_gpu);
        self.marlin_scales = Some(global_gpu);
        Ok(())
    }

    pub fn from_safetensors(
        ctx: &DeviceContext,
        data: &[u8],
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        if data.len() != rows * cols * std::mem::size_of::<bf16>() {
            return Err(anyhow!(
                "Data length mismatch: expected {} bytes, got {} bytes",
                rows * cols * std::mem::size_of::<bf16>(),
                data.len()
            ));
        }
        let slice = bf16_safetensor_host_slice(data)?;
        let gpu_data = ctx
            .stream
            .clone_htod(slice.as_ref())
            .map_err(|e| anyhow!("H2D copy failed: {}", e))?;
        Ok(Self {
            data: gpu_data,
            rows,
            cols,
            weight_format: WeightFormat::DenseBf16,
            qweight: None,
            qweight_u8: None,
            pristine_fp8: None,
            qscales: None,
            qscale_fp8: None,
            scale_f32: None,
            scale2_f32: None,
            quant_scale_rows: 0,
            quant_scale_cols: 0,
            quant_block_m: 0,
            quant_block_k: 0,
            dsv4_scales: None,
            dsv4_scale_rows: 0,
            dsv4_scale_cols: 0,
            group_size: 0,
            marlin_packed: None,
            marlin_scales: None,
            marlin_channel_scales: None,
            fp4_deepgemm_sfb: None,
            hybrid_w4a8_qweight: None,
            hybrid_w4a8_s_channel: None,
            hybrid_w4a8_s_group: None,
            hybrid_w4_fp8_qweight: None,
            tq_packed: None,
            tq_scales: None,
            tq_signs: None,
            tq_centroids: None,
            tq_bits: 0,
        })
    }

    /// Extract a contiguous range of rows `[row_start..row_end)` as a new `DeviceMatrix`.
    /// The result is an independent copy on the GPU.
    pub fn slice_rows(
        ctx: &DeviceContext,
        src: &DeviceMatrix,
        row_start: usize,
        row_end: usize,
    ) -> Result<Self> {
        assert!(
            row_start < row_end && row_end <= src.rows,
            "slice_rows: invalid range [{}..{}) for matrix with {} rows",
            row_start,
            row_end,
            src.rows,
        );
        let out_rows = row_end - row_start;
        let n = out_rows * src.cols;
        let offset = row_start * src.cols;
        let mut dst: CudaSlice<bf16> = ctx
            .stream
            .alloc_zeros(n)
            .map_err(|e| anyhow!("slice_rows alloc failed: {e}"))?;
        ctx.stream
            .memcpy_dtod(&src.data.slice(offset..offset + n), &mut dst)
            .map_err(|e| anyhow!("slice_rows D2D copy failed: {e}"))?;
        Ok(Self {
            data: dst,
            rows: out_rows,
            cols: src.cols,
            weight_format: WeightFormat::DenseBf16,
            qweight: None,
            qweight_u8: None,
            pristine_fp8: None,
            qscales: None,
            qscale_fp8: None,
            scale_f32: None,
            scale2_f32: None,
            quant_scale_rows: 0,
            quant_scale_cols: 0,
            quant_block_m: 0,
            quant_block_k: 0,
            dsv4_scales: None,
            dsv4_scale_rows: 0,
            dsv4_scale_cols: 0,
            group_size: 0,
            marlin_packed: None,
            marlin_scales: None,
            marlin_channel_scales: None,
            fp4_deepgemm_sfb: None,
            hybrid_w4a8_qweight: None,
            hybrid_w4a8_s_channel: None,
            hybrid_w4a8_s_group: None,
            hybrid_w4_fp8_qweight: None,
            tq_packed: None,
            tq_scales: None,
            tq_signs: None,
            tq_centroids: None,
            tq_bits: 0,
        })
    }
}

/// Batched hidden states: seq_len vectors of dim hidden_dim, stored contiguously.
/// Memory layout: [hidden_dim * seq_len] elements, token i at offset i * hidden_dim.
/// cuBLAS interprets as [hidden_dim, seq_len] column-major.
pub struct HiddenStates {
    pub data: CudaSlice<bf16>,
    pub hidden_dim: usize,
    pub seq_len: usize,
}

impl HiddenStates {
    /// Create zeroed batch
    #[track_caller]
    pub fn zeros(ctx: &DeviceContext, hidden_dim: usize, seq_len: usize) -> Result<Self> {
        let len = hidden_dim * seq_len;
        let data: CudaSlice<bf16> = ctx
            .stream
            .alloc_zeros(len)
            .map_err(|e| anyhow!("Alloc failed: {}", e))?;
        record_cuda_alloc::<bf16>("alloc_zeros", "HiddenStates::zeros", len);
        Ok(Self {
            data,
            hidden_dim,
            seq_len,
        })
    }

    /// Create an uninitialized batch for call sites that immediately overwrite
    /// every element with a CUDA kernel.
    ///
    /// # Safety
    ///
    /// The returned buffer must not be read before all `hidden_dim * seq_len`
    /// elements have been written by a kernel or device copy.
    #[track_caller]
    pub unsafe fn uninit(ctx: &DeviceContext, hidden_dim: usize, seq_len: usize) -> Result<Self> {
        let len = hidden_dim * seq_len;
        // SAFETY: forwards the uninitialized-memory contract to our caller per
        // this method's `# Safety` doc (must be fully written before any read).
        let data: CudaSlice<bf16> = unsafe {
            ctx.stream
                .alloc(len)
                .map_err(|e| anyhow!("Alloc failed: {}", e))?
        };
        record_cuda_alloc::<bf16>("alloc", "HiddenStates::uninit", len);
        Ok(Self {
            data,
            hidden_dim,
            seq_len,
        })
    }

    /// Exact requested device bytes this buffer owns:
    /// `data.len() * size_of::<bf16>()`. Read-only accounting for the DSv4
    /// VRAM ledger.
    pub fn device_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<bf16>()
    }

    /// Copy to host as f32, token-major `[seq_len, hidden_dim]`.
    pub fn to_host(&self, ctx: &DeviceContext) -> Result<Vec<f32>> {
        let host = ctx
            .stream
            .clone_dtoh(&self.data)
            .map_err(|e| anyhow!("D2H copy failed: {}", e))?;
        ctx.sync()?;
        Ok(host.iter().map(|x| x.to_f32()).collect())
    }

    /// Borrowed view over the whole buffer (`seq_len` columns).
    pub fn as_view(&self) -> HiddenStatesView<'_> {
        HiddenStatesView {
            data: self.data.slice(..),
            hidden_dim: self.hidden_dim,
            seq_len: self.seq_len,
        }
    }

    /// Borrowed view of column `r` (`[hidden_dim, 1]`). Same device address +
    /// length the per-row D2D copy would have produced — read-only, zero copy.
    pub fn col(&self, r: usize) -> HiddenStatesView<'_> {
        let w = self.hidden_dim;
        HiddenStatesView {
            data: self.data.slice(r * w..(r + 1) * w),
            hidden_dim: w,
            seq_len: 1,
        }
    }
}

/// Borrowed column view into a contiguous `[hidden_dim, seq_len]` [`HiddenStates`].
/// Feeds the identical device pointer the per-row D2D copy produced → bit-identical
/// reads, zero copy. Read-only.
pub struct HiddenStatesView<'a> {
    pub data: cudarc::driver::CudaView<'a, bf16>,
    pub hidden_dim: usize,
    pub seq_len: usize,
}

impl<'a> HiddenStatesView<'a> {
    /// Reborrow this view at the same span, preserving lifetime `'a`
    /// (cudarc `CudaView::slice(..)` returns `Self`). Lets owned and borrowed
    /// indexer-query sources be unified to one `HiddenStatesView` value.
    pub fn as_self_view(&self) -> HiddenStatesView<'a> {
        HiddenStatesView {
            data: self.data.slice(..),
            hidden_dim: self.hidden_dim,
            seq_len: self.seq_len,
        }
    }
}

/// Cached raw CUDA device pointer for a pre-allocated buffer.
///
/// Avoids per-call overhead of cudarc's `device_ptr()` / `device_ptr_mut()`
/// which perform atomic loads + SyncOnDrop bookkeeping even when event tracking
/// is disabled.
///
/// # Safety invariants
/// - The originating CudaSlice must outlive all uses of this pointer.
/// - The originating CudaSlice must not be reallocated.
/// - Only used from the single inference thread (single CUDA stream).
#[derive(Debug, Clone, Copy)]
pub struct RawDevicePtr<T> {
    ptr: u64,
    _marker: PhantomData<*const T>,
}

// SAFETY: RawDevicePtr is only used from the single inference thread.
unsafe impl<T> Send for RawDevicePtr<T> {}

impl<T> RawDevicePtr<T> {
    /// Get as const pointer for kernel read parameters.
    pub fn as_ptr(self) -> *const T {
        self.ptr as *const T
    }

    /// Get as mut pointer for kernel write parameters.
    pub fn as_mut_ptr(self) -> *mut T {
        self.ptr as *mut T
    }

    /// Reinterpret the device address as a pointer to `U`. The caller asserts
    /// the underlying bytes are a valid `[U]` (e.g. a `CudaSlice<u8>` byte
    /// buffer that actually holds bf16 weights). No allocation; just a u64 view.
    pub fn cast<U>(self) -> RawDevicePtr<U> {
        RawDevicePtr {
            ptr: self.ptr,
            _marker: PhantomData,
        }
    }

    /// Advance the pointer by `count` elements of `T` (`count * size_of::<T>()`
    /// bytes). The caller asserts the result stays within the backing slice.
    pub fn offset_elems(self, count: usize) -> RawDevicePtr<T> {
        RawDevicePtr {
            ptr: self.ptr + (count * std::mem::size_of::<T>()) as u64,
            _marker: PhantomData,
        }
    }
}

/// Extract and cache a raw device pointer from a CudaSlice.
/// Calls device_ptr() once -- amortized over thousands of decode steps.
pub fn cache_ptr<T>(slice: &CudaSlice<T>, ctx: &DeviceContext) -> RawDevicePtr<T> {
    cache_ptr_on(slice, &ctx.stream)
}

/// `cache_ptr` for callers that hold a bare stream (the autograd backend) rather
/// than a full `DeviceContext`. Same one-shot device_ptr extraction.
pub fn cache_ptr_on<T>(slice: &CudaSlice<T>, stream: &CudaStream) -> RawDevicePtr<T> {
    use cudarc::driver::DevicePtr;
    let (ptr, _sync) = slice.device_ptr(stream);
    RawDevicePtr {
        ptr,
        _marker: PhantomData,
    }
}

/// A null [`RawDevicePtr`] — for optional kernel tables the kernel treats as
/// absent (e.g. `expert_indices` when the compact index is the expert index).
pub fn null_raw_ptr<T>() -> RawDevicePtr<T> {
    RawDevicePtr {
        ptr: 0,
        _marker: PhantomData,
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

    #[test]
    fn uniform_quant_formats_require_group_aligned_k() {
        assert!(WeightFormat::W4A16.validate_shape(64, 4096, 128).is_ok());
        assert!(WeightFormat::W4A16.validate_shape(64, 4097, 128).is_err());
        assert!(WeightFormat::W8A16.validate_shape(64, 4096, 0).is_err());
    }

    #[test]
    fn gguf_k_formats_require_256_wide_superblocks() {
        assert!(WeightFormat::GgufQ4K.validate_shape(64, 4096, 256).is_ok());
        assert!(WeightFormat::GgufQ4K.validate_shape(64, 4096, 128).is_err());
        assert!(WeightFormat::GgufQ4K.validate_shape(64, 4100, 256).is_err());
    }

    #[test]
    fn resident_quant_abi_formats_validate_shapes() {
        assert!(
            WeightFormat::Fp8BlockScaled
                .validate_shape(512, 2048, 0)
                .is_ok()
        );
        assert!(
            WeightFormat::Fp8PerShard
                .validate_shape(512, 2048, 0)
                .is_ok()
        );
        assert!(
            WeightFormat::Fp4E2M1Group
                .validate_shape(512, 2048, 16)
                .is_ok()
        );
        assert!(
            WeightFormat::Fp4E2M1Group
                .validate_shape(512, 2049, 16)
                .is_err()
        );
        assert!(
            WeightFormat::Fp4E2M1Group
                .validate_shape(512, 2048, 0)
                .is_err()
        );
    }

    fn copy_matrix_to_host(ctx: &DeviceContext, matrix: &DeviceMatrix) -> Vec<bf16> {
        let host = ctx
            .stream
            .clone_dtoh(&matrix.data)
            .expect("D2H copy failed");
        ctx.sync().expect("CUDA sync failed");
        host
    }

    #[test]
    fn test_device_matrix_from_host_roundtrip() {
        let ctx = DeviceContext::new().expect("Failed to create CUDA context");
        let rows = 2;
        let cols = 3;
        let host = vec![
            bf16::from_f32(-1.5),
            bf16::from_f32(0.0),
            bf16::from_f32(2.25),
            bf16::from_f32(7.0),
            bf16::from_f32(-3.0),
            bf16::from_f32(0.5),
        ];

        let matrix =
            DeviceMatrix::from_host(&ctx, &host, rows, cols).expect("from_host should succeed");

        assert_eq!(matrix.rows, rows);
        assert_eq!(matrix.cols, cols);

        let got = copy_matrix_to_host(&ctx, &matrix);
        assert_eq!(got.len(), host.len());
        for (idx, (actual, expected)) in got.iter().zip(host.iter()).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "roundtrip mismatch at index {}",
                idx
            );
        }
    }

    #[test]
    fn test_device_matrix_from_safetensors_matches_from_host() {
        let ctx = DeviceContext::new().expect("Failed to create CUDA context");
        let rows = 3;
        let cols = 2;
        let host = vec![
            bf16::from_f32(-8.0),
            bf16::from_f32(-0.25),
            bf16::from_f32(1.0),
            bf16::from_f32(3.5),
            bf16::from_f32(9.0),
            bf16::from_f32(10.75),
        ];
        let safetensor_bytes: Vec<u8> = host
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect();

        let from_host =
            DeviceMatrix::from_host(&ctx, &host, rows, cols).expect("from_host should succeed");
        let from_safetensors = DeviceMatrix::from_safetensors(&ctx, &safetensor_bytes, rows, cols)
            .expect("from_safetensors should succeed");

        assert_eq!(from_safetensors.rows, from_host.rows);
        assert_eq!(from_safetensors.cols, from_host.cols);

        let host_out = copy_matrix_to_host(&ctx, &from_host);
        let safetensors_out = copy_matrix_to_host(&ctx, &from_safetensors);
        assert_eq!(host_out.len(), safetensors_out.len());
        for (idx, (a, b)) in host_out.iter().zip(safetensors_out.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "from_safetensors/from_host mismatch at index {}",
                idx
            );
        }
    }
}

/// FP8 E4M3FN → f32 (host twin of `dsv4_decode_fp8_e4m3`; 0x7f/0xff is the
/// format's NaN and is clamped to +/-448, matching the device decoder).
fn e4m3_to_f32(bits: u8) -> f32 {
    let sign = if bits & 0x80 == 0 { 1.0 } else { -1.0 };
    let exp = i32::from((bits >> 3) & 0x0f);
    let mant = f32::from(bits & 0x07);
    if bits & 0x7f == 0x7f {
        return sign * 448.0;
    }
    if exp == 0 {
        return sign * (mant / 8.0) * 2.0_f32.powi(-6);
    }
    sign * (1.0 + mant / 8.0) * 2.0_f32.powi(exp - 7)
}
