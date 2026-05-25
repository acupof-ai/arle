//! Device tensor types and CUDA context.

use anyhow::{Result, anyhow, ensure};
use cudarc::driver::{
    CudaContext, CudaEvent, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr,
    DriverError, PinnedHostSlice, ValidAsZeroBits,
};
use half::bf16;
use std::any::type_name;
use std::cell::Cell;
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

#[derive(Clone, Debug, Default)]
pub struct CudaAllocTraceSnapshot {
    entries: BTreeMap<CudaAllocTraceKey, CudaAllocTraceStats>,
}

#[derive(Clone, Debug)]
pub struct CudaAllocTraceEntry {
    pub key: CudaAllocTraceKey,
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

pub fn cuda_alloc_trace_is_enabled() -> bool {
    cuda_alloc_trace_enabled()
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

pub fn cuda_alloc_trace_snapshot() -> Option<CudaAllocTraceSnapshot> {
    if !cuda_alloc_trace_enabled() {
        return None;
    }
    CUDA_ALLOC_TRACE
        .lock()
        .ok()
        .map(|entries| CudaAllocTraceSnapshot {
            entries: entries.clone(),
        })
}

pub fn cuda_alloc_trace_summary_since(
    start: &CudaAllocTraceSnapshot,
    limit: usize,
) -> Option<Vec<CudaAllocTraceEntry>> {
    if !cuda_alloc_trace_enabled() {
        return None;
    }
    let trace = CUDA_ALLOC_TRACE.lock().ok()?;
    let mut entries = Vec::new();
    for (key, current) in trace.iter() {
        let before = start.entries.get(key).copied().unwrap_or_default();
        let calls = current.calls.saturating_sub(before.calls);
        let bytes = current.bytes.saturating_sub(before.bytes);
        if calls == 0 && bytes == 0 {
            continue;
        }
        entries.push(CudaAllocTraceEntry {
            key: key.clone(),
            calls,
            bytes,
        });
    }
    entries.sort_by(|a, b| {
        b.calls
            .cmp(&a.calls)
            .then_with(|| b.bytes.cmp(&a.bytes))
            .then_with(|| a.key.file.cmp(b.key.file))
            .then_with(|| a.key.line.cmp(&b.key.line))
    });
    entries.truncate(limit);
    Some(entries)
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
        let out = unsafe { self.alloc(len)? };
        record_cuda_alloc::<T>("alloc", "CudaStream::alloc", len);
        Ok(out)
    }

    #[track_caller]
    fn alloc_zeros_traced<T: DeviceRepr + ValidAsZeroBits>(
        &self,
        len: usize,
    ) -> Result<CudaSlice<T>, DriverError> {
        let mut out = unsafe { self.alloc(len)? };
        record_cuda_alloc::<T>("alloc_zeros", "CudaStream::alloc_zeros", len);
        self.memset_zeros(&mut out)?;
        Ok(out)
    }
}

/// CUDA device context holding compute stream and optional copy stream.
///
/// Two-stream architecture for overlapping H2D/D2H transfers with compute:
/// - `stream` (compute): all GPU kernels, CUDA Graph capture/replay
/// - `copy_stream`: async H2D/D2H transfers, runs concurrently with compute
/// - `comm_stream`: communication collectives that can overlap independent compute
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
    event: CudaEvent,
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
        self.event
            .context()
            .bind_to_thread()
            .map_err(|e| anyhow!("Bind CUDA context before pipeline fence query failed: {e}"))?;
        match unsafe { cudarc::driver::result::event::query(self.event.cu_event()) } {
            Ok(()) => Ok(CudaPipelineFenceStatus::Ready),
            Err(err) if err.0 == cudarc::driver::sys::CUresult::CUDA_ERROR_NOT_READY => {
                Ok(CudaPipelineFenceStatus::NotReady)
            }
            Err(err) => Err(anyhow!("CUDA pipeline fence query failed: {err}")),
        }
    }

    /// Convenience wrapper for callers that only need a boolean readiness check.
    pub fn is_ready(&self) -> Result<bool> {
        Ok(matches!(self.query()?, CudaPipelineFenceStatus::Ready))
    }
}

/// Parse `INFER_CUDA_DEVICE` (default 0). Selects the device for `DeviceContext::new()`.
pub fn parse_device_ordinal_from_env() -> Result<u32> {
    parse_device_ordinal(std::env::var("INFER_CUDA_DEVICE").ok().as_deref())
}

thread_local! {
    static DEVICE_ORDINAL_OVERRIDE: Cell<Option<u32>> = const { Cell::new(None) };
}

fn scoped_device_ordinal_override() -> Option<u32> {
    DEVICE_ORDINAL_OVERRIDE.with(Cell::get)
}

fn effective_device_ordinal_for_new() -> Result<u32> {
    scoped_device_ordinal_override()
        .map(Ok)
        .unwrap_or_else(parse_device_ordinal_from_env)
}

struct DeviceOrdinalOverrideReset {
    previous: Option<u32>,
}

impl Drop for DeviceOrdinalOverrideReset {
    fn drop(&mut self) {
        DEVICE_ORDINAL_OVERRIDE.with(|slot| slot.set(self.previous));
    }
}

/// Runs `f` while [`DeviceContext::new`] resolves to `ordinal` on this thread.
///
/// The override is thread-local so multi-worker runtimes can initialize
/// separate CUDA contexts without mutating process-global environment variables.
pub fn with_device_ordinal_override<T>(ordinal: u32, f: impl FnOnce() -> T) -> T {
    let previous = DEVICE_ORDINAL_OVERRIDE.with(|slot| {
        let previous = slot.get();
        slot.set(Some(ordinal));
        previous
    });
    let _reset = DeviceOrdinalOverrideReset { previous };
    f()
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

fn marlin_w4_fp8_prefill_enabled_for_load() -> bool {
    matches!(
        std::env::var("INFER_MARLIN_W4_FP8_PREFILL").as_deref(),
        Ok("1" | "true" | "TRUE" | "yes" | "on" | "ON")
    )
}

impl DeviceContext {
    /// Query available (free) GPU memory in bytes.
    /// Returns `(free_bytes, total_bytes)`.
    pub fn gpu_memory_info() -> Result<(usize, usize)> {
        cudarc::driver::result::mem_get_info()
            .map_err(|e| anyhow!("Failed to query GPU memory: {}", e))
    }

    /// Default constructor: honours `INFER_CUDA_DEVICE` (default 0).
    /// F1+ multi-GPU rank threads bypass this and call `on_device(ordinal)`.
    pub fn new() -> Result<Self> {
        let ordinal = effective_device_ordinal_for_new()?;
        Self::on_device(ordinal)
    }

    pub fn on_device(ordinal: u32) -> Result<Self> {
        let ctx = CudaContext::new(ordinal as usize)
            .map_err(|e| anyhow!("Failed to create CUDA context on device {ordinal}: {e}"))?;

        // Disable cudarc's automatic event tracking before creating streams.
        // Serving owns cross-stream dependencies explicitly via
        // CudaPipelineFence, which avoids hidden waits in CUDA Graph capture
        // paths while still allowing a dedicated copy stream.
        unsafe {
            ctx.disable_event_tracking();
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

        // Initialize cuBLAS handle
        unsafe {
            ffi::cublas_init();
        }

        Ok(Self {
            ctx,
            stream,
            copy_stream,
            comm_stream,
            ordinal,
        })
    }

    pub fn ordinal(&self) -> u32 {
        self.ordinal
    }

    /// Query the number of streaming multiprocessors on the GPU this context is bound to.
    pub fn sm_count(&self) -> usize {
        use cudarc::driver::sys::*;
        let mut count: i32 = 0;
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

    /// Synchronize communication stream.
    pub fn sync_comm(&self) -> Result<()> {
        self.comm_stream
            .synchronize()
            .map_err(|e| anyhow!("Communication stream sync failed: {}", e))
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
        let event = self
            .ctx
            .new_event(None)
            .map_err(|e| anyhow!("Alloc CUDA pipeline fence failed: {e}"))?;
        event
            .record(self.pipeline_stream(producer))
            .map_err(|e| anyhow!("Record CUDA pipeline fence on {producer:?} failed: {e}"))?;
        Ok(CudaPipelineFence {
            device_ordinal: self.ordinal,
            producer,
            event,
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
            .wait(&fence.event)
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
    /// order that stream first, e.g. with [`DeviceContext::copy_waits_for_compute`].
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

    /// Record an event on the compute stream and make the copy stream wait for it.
    ///
    /// Use after GPU kernels finish (e.g. sampling) to ensure the copy stream
    /// sees the results before starting D2H transfer.
    pub fn copy_waits_for_compute(&self) -> Result<()> {
        let fence = self.record_pipeline_fence(CudaPipelineStreamKind::Compute)?;
        self.wait_on_pipeline_fence(&fence, CudaPipelineStreamKind::Copy)
    }

    /// Record an event on the copy stream and make the compute stream wait for it.
    ///
    /// Use after H2D transfer completes to ensure compute kernels see the uploaded data.
    pub fn compute_waits_for_copy(&self) -> Result<()> {
        let fence = self.record_pipeline_fence(CudaPipelineStreamKind::Copy)?;
        self.wait_on_pipeline_fence(&fence, CudaPipelineStreamKind::Compute)
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
        assert!(compute_done.is_ready()?);
        assert!(copy_done.is_ready()?);
        Ok(())
    }

    #[test]
    fn pinned_copy_stream_h2d_helper_returns_compute_waitable_fence() -> Result<()> {
        let ctx = DeviceContext::new()?;

        let initial = [11_i32, 22, 33, 44];
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
        assert!(upload_done.is_ready()?);
        Ok(())
    }
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

    #[allow(clippy::cast_ptr_alignment)]
    pub fn from_safetensors(ctx: &DeviceContext, data: &[u8]) -> Result<Self> {
        if !data.len().is_multiple_of(2) {
            return Err(anyhow!(
                "Data length must be even for bf16: got {} bytes",
                data.len()
            ));
        }
        let len = data.len() / 2;
        // NOTE: This assumes a little-endian host. Safetensors are little-endian.
        // On a big-endian machine, this will be incorrect. A full solution would
        // involve byte-swapping.
        let slice = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<bf16>(), len) };
        Self::from_host(ctx, slice)
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

    /// Create a tensor filled with bf16 ones (1.0).
    /// Useful for dummy RMSNorm weights (identity normalization).
    pub fn ones(ctx: &DeviceContext, len: usize) -> Result<Self> {
        let host = vec![bf16::ONE; len];
        Self::from_host(ctx, &host)
    }

    /// Extract a contiguous sub-range `[start..end)` as a new `DeviceVec`.
    /// The result is an independent copy on the GPU.
    pub fn slice_to_vec(
        ctx: &DeviceContext,
        src: &DeviceVec,
        start: usize,
        end: usize,
    ) -> Result<Self> {
        assert!(
            start < end && end <= src.len,
            "slice_to_vec: invalid range [{}..{}) for vec of len {}",
            start,
            end,
            src.len,
        );
        let len = end - start;
        let mut out = Self::zeros(ctx, len)?;
        let src_view = src.data.slice(start..end);
        ctx.stream
            .memcpy_dtod(&src_view, &mut out.data)
            .map_err(|e| anyhow!("slice_to_vec D2D copy failed: {e}"))?;
        Ok(out)
    }

    /// Attach a debug label describing this tensor's semantic shape/purpose.
    ///
    /// ```ignore
    /// let w = DeviceVec::zeros(&ctx, 4096)?.with_label("norm_weight[hidden]");
    /// ```
    pub fn with_label(mut self, label: &'static str) -> Self {
        self.label = label;
        self
    }

    /// Copy a region of the device buffer to a host slice (D2H).
    ///
    /// `offset` and `len` are in elements (bf16), not bytes.
    /// `dst` must have length >= `len`.
    pub fn copy_region_to_host(
        &self,
        ctx: &DeviceContext,
        offset: usize,
        len: usize,
        dst: &mut [bf16],
    ) -> Result<()> {
        assert!(
            offset + len <= self.len,
            "copy_region_to_host: offset {} + len {} exceeds buffer len {}",
            offset,
            len,
            self.len
        );
        assert!(
            dst.len() >= len,
            "copy_region_to_host: dst len {} < requested len {}",
            dst.len(),
            len
        );
        let view = self.data.slice(offset..offset + len);
        ctx.stream
            .memcpy_dtoh(&view, &mut dst[..len])
            .map_err(|e| anyhow!("D2H region copy failed: {}", e))?;
        Ok(())
    }

    /// Copy from a host slice into a region of the device buffer (H2D).
    ///
    /// `offset` is in elements (bf16). `src.len()` elements are copied
    /// starting at `offset` in the device buffer.
    pub fn copy_region_from_host(
        &mut self,
        ctx: &DeviceContext,
        offset: usize,
        src: &[bf16],
    ) -> Result<()> {
        assert!(
            offset + src.len() <= self.len,
            "copy_region_from_host: offset {} + src len {} exceeds buffer len {}",
            offset,
            src.len(),
            self.len
        );
        let mut view = self.data.slice_mut(offset..offset + src.len());
        ctx.stream
            .memcpy_htod(src, &mut view)
            .map_err(|e| anyhow!("H2D region copy failed: {}", e))?;
        Ok(())
    }

    /// Copy a region within the same device buffer or between buffers (D2D).
    ///
    /// Copies `len` elements from `src_offset` in `src` to `dst_offset` in `self`.
    pub fn copy_region_from_device(
        &mut self,
        ctx: &DeviceContext,
        dst_offset: usize,
        src: &DeviceVec,
        src_offset: usize,
        len: usize,
    ) -> Result<()> {
        assert!(
            src_offset + len <= src.len,
            "copy_region_from_device: src_offset {} + len {} exceeds src len {}",
            src_offset,
            len,
            src.len
        );
        assert!(
            dst_offset + len <= self.len,
            "copy_region_from_device: dst_offset {} + len {} exceeds dst len {}",
            dst_offset,
            len,
            self.len
        );
        let src_view = src.data.slice(src_offset..src_offset + len);
        let mut dst_view = self.data.slice_mut(dst_offset..dst_offset + len);
        ctx.stream
            .memcpy_dtod(&src_view, &mut dst_view)
            .map_err(|e| anyhow!("D2D region copy failed: {}", e))?;
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
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
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

    #[must_use]
    pub fn is_gguf_k_quant(self) -> bool {
        matches!(
            self,
            Self::GgufQ3K | Self::GgufQ4K | Self::GgufQ5K | Self::GgufQ6K
        )
    }

    #[must_use]
    pub fn kernel_alignment(self, group_size: usize) -> WeightKernelAlignment {
        match self {
            Self::DenseBf16 => WeightKernelAlignment {
                weight_layout: "bf16.row_major",
                scale_layout: "none",
                k_multiple: 1,
                n_multiple: 1,
                group_size: 0,
            },
            Self::W8A16 | Self::W4A16 | Self::W2A16 => WeightKernelAlignment {
                weight_layout: "wN.row_major.group_packed",
                scale_layout: "bf16[row, k/group_size]",
                k_multiple: group_size.max(1),
                n_multiple: 1,
                group_size,
            },
            Self::MarlinW4A8 => WeightKernelAlignment {
                weight_layout: "marlin.w4a8.packed",
                scale_layout: "f32[channel] + fp16[group,channel]",
                k_multiple: group_size.max(128),
                n_multiple: 256,
                group_size,
            },
            Self::GgufQ3K | Self::GgufQ4K | Self::GgufQ5K | Self::GgufQ6K => {
                WeightKernelAlignment {
                    weight_layout: "gguf.qk.row_major.superblock256",
                    scale_layout: "embedded.superblock",
                    k_multiple: 256,
                    n_multiple: 1,
                    group_size: 256,
                }
            }
            Self::TurboQuant => WeightKernelAlignment {
                weight_layout: "turboquant.row_major.group_packed",
                scale_layout: "fp16[row, k/group_size]",
                k_multiple: group_size.max(1),
                n_multiple: 1,
                group_size,
            },
            Self::Dsv4Fp8BlockScaled => WeightKernelAlignment {
                weight_layout: "dsv4.fp8_e4m3.row_major",
                scale_layout: "fp8_e8m0[scale_rows, scale_cols]",
                k_multiple: 1,
                n_multiple: 1,
                group_size: 0,
            },
            Self::Dsv4Fp4BlockScaled => WeightKernelAlignment {
                weight_layout: "dsv4.fp4_e2m1.row_major.packed2",
                scale_layout: "fp8_e8m0[scale_rows, scale_cols]",
                k_multiple: 2,
                n_multiple: 1,
                group_size: 0,
            },
        }
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
            weight: unsafe { ctx.stream.alloc_traced::<u8>(weight_len)? },
            scales: unsafe { ctx.stream.alloc_traced::<f32>(scale_len)? },
            rows,
            cols,
            scale_rows,
            scale_cols,
        })
    }

    #[must_use]
    pub fn scale_gran_m(&self) -> usize {
        DSV4_DEEPGEMM_FP8_SCALE_GRAN_M
    }

    #[must_use]
    pub fn scale_gran_k(&self) -> usize {
        DSV4_DEEPGEMM_FP8_SCALE_GRAN_K
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
    let dst_weight_ptr = unsafe { (dst_weight_ptr as *mut u8).add(dst_row_offset * dst.cols) };
    let dst_scale_ptr =
        unsafe { (dst_scale_ptr as *mut f32).add(dst_scale_row_offset * dst.scale_cols) };
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

/// 2D device tensor (matrix) — stored in row-major order as bf16 unless
/// `weight_format` names an explicit packed layout.
pub struct DeviceMatrix {
    pub data: CudaSlice<bf16>,
    pub rows: usize,
    pub cols: usize,
    pub weight_format: WeightFormat,
    /// INT8 quantized weights (if quantized). When set, `data` is unused.
    pub qweight: Option<CudaSlice<i8>>,
    /// Per-group bf16 scales for quantized weights. Shape: [rows, cols/group_size].
    pub qscales: Option<CudaSlice<bf16>>,
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

impl DeviceMatrix {
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
            qscales: None,
            dsv4_scales: None,
            dsv4_scale_rows: 0,
            dsv4_scale_cols: 0,
            group_size: 0,
            marlin_packed: None,
            marlin_scales: None,
            marlin_channel_scales: None,
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
            qscales: Some(qs),
            dsv4_scales: None,
            dsv4_scale_rows: 0,
            dsv4_scale_cols: 0,
            group_size,
            marlin_packed: None,
            marlin_scales: None,
            marlin_channel_scales: None,
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
            qscales: Some(qs),
            dsv4_scales: None,
            dsv4_scale_rows: 0,
            dsv4_scale_cols: 0,
            group_size,
            marlin_packed: None,
            marlin_scales: None,
            marlin_channel_scales: None,
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

        Ok(Self {
            data: dummy,
            rows,
            cols,
            weight_format: WeightFormat::Dsv4Fp8BlockScaled,
            qweight: Some(qw),
            qscales: None,
            dsv4_scales: Some(scales),
            dsv4_scale_rows: scale_rows,
            dsv4_scale_cols: scale_cols,
            group_size: 0,
            marlin_packed: None,
            marlin_scales: None,
            marlin_channel_scales: None,
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

    /// Create from DeepSeek V4 packed FP4 E2M1 weights plus FP8 E8M0 block scales.
    pub fn from_dsv4_fp4_block_scaled(
        ctx: &DeviceContext,
        packed_bytes: &[u8],
        scale_bytes: &[u8],
        rows: usize,
        logical_cols: usize,
        scale_rows: usize,
        scale_cols: usize,
    ) -> Result<Self> {
        WeightFormat::Dsv4Fp4BlockScaled.validate_shape(rows, logical_cols, 0)?;
        ensure!(
            packed_bytes.len() == rows * logical_cols / 2,
            "DeepSeek V4 FP4 packed bytes {} != expected {} for rows={rows} cols={logical_cols}",
            packed_bytes.len(),
            rows * logical_cols / 2
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
            .clone_htod(unsafe {
                std::slice::from_raw_parts(packed_bytes.as_ptr().cast::<i8>(), packed_bytes.len())
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

        Ok(Self {
            data: dummy,
            rows,
            cols: logical_cols,
            weight_format: WeightFormat::Dsv4Fp4BlockScaled,
            qweight: Some(qw),
            qscales: None,
            dsv4_scales: Some(scales),
            dsv4_scale_rows: scale_rows,
            dsv4_scale_cols: scale_cols,
            group_size: 0,
            marlin_packed: None,
            marlin_scales: None,
            marlin_channel_scales: None,
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

    /// Create from prepacked W4A8 Marlin side tensors.
    #[allow(clippy::too_many_arguments)]
    pub fn from_marlin_w4a8(
        ctx: &DeviceContext,
        packed_data: &[u8],
        channel_scales: &[f32],
        group_scales: &[u16],
        rows: usize,
        cols: usize,
        group_size: usize,
    ) -> Result<Self> {
        WeightFormat::MarlinW4A8.validate_shape(rows, cols, group_size)?;
        ensure!(
            packed_data.len() == rows * cols / 2,
            "MarlinW4A8 packed bytes {} != expected {} for rows={rows} cols={cols}",
            packed_data.len(),
            rows * cols / 2
        );
        ensure!(
            channel_scales.len() == rows,
            "MarlinW4A8 channel scales {} != rows {rows}",
            channel_scales.len()
        );
        ensure!(
            group_scales.len() == (cols / group_size) * rows,
            "MarlinW4A8 group scales {} != expected {}",
            group_scales.len(),
            (cols / group_size) * rows
        );

        let packed = ctx
            .stream
            .clone_htod(packed_data)
            .map_err(|e| anyhow!("H2D W4A8 Marlin packed failed: {e}"))?;
        let s_channel = ctx
            .stream
            .clone_htod(channel_scales)
            .map_err(|e| anyhow!("H2D W4A8 channel scales failed: {e}"))?;
        let s_group = ctx
            .stream
            .clone_htod(group_scales)
            .map_err(|e| anyhow!("H2D W4A8 group scales failed: {e}"))?;
        let dummy = ctx
            .stream
            .alloc_zeros::<bf16>(1)
            .map_err(|e| anyhow!("Alloc dummy: {}", e))?;

        Ok(Self {
            data: dummy,
            rows,
            cols,
            weight_format: WeightFormat::MarlinW4A8,
            qweight: None,
            qscales: None,
            dsv4_scales: None,
            dsv4_scale_rows: 0,
            dsv4_scale_cols: 0,
            group_size,
            marlin_packed: Some(packed),
            marlin_scales: Some(s_group),
            marlin_channel_scales: Some(s_channel),
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

    /// Create from a hybrid W4 checkpoint that carries W4A16 decode tensors and
    /// W4A8 Marlin prefill side tensors in the same `DeviceMatrix`.
    #[allow(clippy::too_many_arguments)]
    pub fn from_hybrid_w4_marlin(
        ctx: &DeviceContext,
        w4a16_qweight: &[u8],
        w4a16_scales: &[u16],
        w4a8_qweight: &[u8],
        w4a8_s_channel: &[f32],
        w4a8_s_group: &[u16],
        rows: usize,
        cols: usize,
        group_size: usize,
    ) -> Result<Self> {
        WeightFormat::W4A16.validate_shape(rows, cols, group_size)?;
        WeightFormat::MarlinW4A8.validate_shape(rows, cols, group_size)?;
        let num_groups = cols / group_size;
        ensure!(
            w4a16_qweight.len() == rows * cols / 2,
            "Hybrid W4A16 Marlin packed bytes {} != expected {} for rows={rows} cols={cols}",
            w4a16_qweight.len(),
            rows * cols / 2
        );
        ensure!(
            w4a16_scales.len() == num_groups * rows,
            "Hybrid W4A16 Marlin scales {} != expected {}",
            w4a16_scales.len(),
            num_groups * rows
        );
        ensure!(
            w4a8_qweight.len() == rows * cols / 2,
            "Hybrid W4A8 packed bytes {} != expected {} for rows={rows} cols={cols}",
            w4a8_qweight.len(),
            rows * cols / 2
        );
        ensure!(
            w4a8_s_channel.len() == rows,
            "Hybrid W4A8 channel scales {} != rows {rows}",
            w4a8_s_channel.len()
        );
        ensure!(
            w4a8_s_group.len() == num_groups * rows,
            "Hybrid W4A8 group scales {} != expected {}",
            w4a8_s_group.len(),
            num_groups * rows
        );

        let w4a16_packed = ctx
            .stream
            .clone_htod(w4a16_qweight)
            .map_err(|e| anyhow!("H2D hybrid W4A16 Marlin qweight failed: {e}"))?;
        let w4a16_group = ctx
            .stream
            .clone_htod(w4a16_scales)
            .map_err(|e| anyhow!("H2D hybrid W4A16 Marlin scales failed: {e}"))?;
        let w4a8_packed = ctx
            .stream
            .clone_htod(w4a8_qweight)
            .map_err(|e| anyhow!("H2D hybrid W4A8 Marlin qweight failed: {e}"))?;
        let w4a8_channel = ctx
            .stream
            .clone_htod(w4a8_s_channel)
            .map_err(|e| anyhow!("H2D hybrid W4A8 channel scales failed: {e}"))?;
        let w4a8_group = ctx
            .stream
            .clone_htod(w4a8_s_group)
            .map_err(|e| anyhow!("H2D hybrid W4A8 group scales failed: {e}"))?;
        let w4_fp8_packed = if marlin_w4_fp8_prefill_enabled_for_load() {
            let mut packed = ctx
                .stream
                .alloc_zeros::<u8>(w4a8_qweight.len())
                .map_err(|e| anyhow!("Alloc hybrid W4+FP8 qweight: {e}"))?;
            {
                let (src, _src_guard) = w4a8_packed.device_ptr(&ctx.stream);
                let (dst, _dst_guard) = packed.device_ptr_mut(&ctx.stream);
                unsafe {
                    ffi::marlin_int4_fp8_preprocess_without_zp_cuda(
                        src as *const i32,
                        dst as *mut i32,
                        (w4a8_qweight.len() / std::mem::size_of::<i32>()) as i32,
                        ctx.stream.cu_stream(),
                    )
                    .result()
                    .map_err(|e| anyhow!("PF8.2 hybrid W4 qweight preprocess failed: {e}"))?;
                }
            }
            Some(packed)
        } else {
            None
        };

        let dummy = ctx
            .stream
            .alloc_zeros::<bf16>(1)
            .map_err(|e| anyhow!("Alloc dummy: {}", e))?;

        Ok(Self {
            data: dummy,
            rows,
            cols,
            weight_format: WeightFormat::W4A16,
            qweight: None,
            qscales: None,
            dsv4_scales: None,
            dsv4_scale_rows: 0,
            dsv4_scale_cols: 0,
            group_size,
            marlin_packed: Some(w4a16_packed),
            marlin_scales: Some(w4a16_group),
            marlin_channel_scales: None,
            hybrid_w4a8_qweight: Some(w4a8_packed),
            hybrid_w4a8_s_channel: Some(w4a8_channel),
            hybrid_w4a8_s_group: Some(w4a8_group),
            hybrid_w4_fp8_qweight: w4_fp8_packed,
            tq_packed: None,
            tq_scales: None,
            tq_signs: None,
            tq_centroids: None,
            tq_bits: 0,
        })
    }

    /// Create from Q6_K packed GGUF superblocks.
    ///
    /// Each 256-element superblock is 210 bytes: ql(128)|qh(64)|scales(16×i8)|d(f16).
    /// Per-row byte stride = `(cols/256) * 210`.
    pub fn from_quantized_q6k(
        ctx: &DeviceContext,
        packed_bytes: &[u8],
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        WeightFormat::GgufQ6K.validate_shape(rows, cols, 256)?;
        let expected = rows * cols * 210 / 256;
        ensure!(
            packed_bytes.len() == expected,
            "Q6_K packed size {} != expected {} for rows={} cols={}",
            packed_bytes.len(),
            expected,
            rows,
            cols
        );

        let qw: CudaSlice<i8> = ctx
            .stream
            .clone_htod(unsafe {
                std::slice::from_raw_parts(packed_bytes.as_ptr().cast::<i8>(), packed_bytes.len())
            })
            .map_err(|e| anyhow!("H2D Q6_K packed upload failed: {}", e))?;
        let dummy_scales: CudaSlice<bf16> = ctx
            .stream
            .alloc_zeros::<bf16>(1)
            .map_err(|e| anyhow!("Alloc Q6_K dummy scales: {}", e))?;
        let dummy = ctx
            .stream
            .alloc_zeros::<bf16>(1)
            .map_err(|e| anyhow!("Alloc dummy: {}", e))?;
        Ok(Self {
            data: dummy,
            rows,
            cols,
            weight_format: WeightFormat::GgufQ6K,
            qweight: Some(qw),
            qscales: Some(dummy_scales),
            dsv4_scales: None,
            dsv4_scale_rows: 0,
            dsv4_scale_cols: 0,
            group_size: 256,
            marlin_packed: None,
            marlin_scales: None,
            marlin_channel_scales: None,
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

    /// Create from Q3_K packed GGUF superblocks.
    ///
    /// Each 256-element superblock is 110 bytes: hmask(32)|qs(64)|scales(12)|d(f16).
    /// Per-row byte stride = `(cols/256) * 110 = cols * 55/128`.
    pub fn from_quantized_q3k(
        ctx: &DeviceContext,
        packed_bytes: &[u8],
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        WeightFormat::GgufQ3K.validate_shape(rows, cols, 256)?;
        let expected = rows * cols * 55 / 128; // (cols/256) * 110 per row
        ensure!(
            packed_bytes.len() == expected,
            "Q3_K packed size {} != expected {} for rows={} cols={}",
            packed_bytes.len(),
            expected,
            rows,
            cols
        );

        let qw: CudaSlice<i8> = ctx
            .stream
            .clone_htod(unsafe {
                std::slice::from_raw_parts(packed_bytes.as_ptr().cast::<i8>(), packed_bytes.len())
            })
            .map_err(|e| anyhow!("H2D Q3_K packed upload failed: {}", e))?;
        let dummy_scales: CudaSlice<bf16> = ctx
            .stream
            .alloc_zeros::<bf16>(1)
            .map_err(|e| anyhow!("Alloc Q3_K dummy scales: {}", e))?;
        let dummy = ctx
            .stream
            .alloc_zeros::<bf16>(1)
            .map_err(|e| anyhow!("Alloc dummy: {}", e))?;
        Ok(Self {
            data: dummy,
            rows,
            cols,
            weight_format: WeightFormat::GgufQ3K,
            qweight: Some(qw),
            qscales: Some(dummy_scales),
            dsv4_scales: None,
            dsv4_scale_rows: 0,
            dsv4_scale_cols: 0,
            group_size: 256,
            marlin_packed: None,
            marlin_scales: None,
            marlin_channel_scales: None,
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

    /// Create from Q4_K_M/Q4_K_S packed GGUF superblocks.
    ///
    /// Uploads the raw 144-byte superblock bytes verbatim to the GPU — no BF16
    /// intermediate ever materialises. One row consists of `cols/256` contiguous
    /// superblocks, so the per-row byte stride is `(cols/256)*144 = cols*9/16`.
    ///
    /// `weight_format` is set to `GgufQ4K` so dispatch can distinguish this
    /// embedded-scale superblock layout from uniform-group W4A16. `group_size`
    /// is set to 256 (superblock size) for informational purposes; the kernel
    /// decodes scales per superblock.
    pub fn from_quantized_q4k(
        ctx: &DeviceContext,
        packed_bytes: &[u8],
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        WeightFormat::GgufQ4K.validate_shape(rows, cols, 256)?;
        let expected = rows * cols * 9 / 16; // (cols/256) * 144 per row
        ensure!(
            packed_bytes.len() == expected,
            "Q4_K packed size {} != expected {} for rows={} cols={}",
            packed_bytes.len(),
            expected,
            rows,
            cols
        );

        let qw: CudaSlice<i8> = ctx
            .stream
            .clone_htod(unsafe {
                std::slice::from_raw_parts(packed_bytes.as_ptr().cast::<i8>(), packed_bytes.len())
            })
            .map_err(|e| anyhow!("H2D Q4_K packed upload failed: {}", e))?;
        let dummy_scales: CudaSlice<bf16> = ctx
            .stream
            .alloc_zeros::<bf16>(1)
            .map_err(|e| anyhow!("Alloc Q4_K dummy scales: {}", e))?;
        let dummy = ctx
            .stream
            .alloc_zeros::<bf16>(1)
            .map_err(|e| anyhow!("Alloc dummy: {}", e))?;
        Ok(Self {
            data: dummy,
            rows,
            cols,
            weight_format: WeightFormat::GgufQ4K,
            qweight: Some(qw),
            qscales: Some(dummy_scales),
            dsv4_scales: None,
            dsv4_scale_rows: 0,
            dsv4_scale_cols: 0,
            group_size: 256,
            marlin_packed: None,
            marlin_scales: None,
            marlin_channel_scales: None,
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

    /// Create from Q5_K packed GGUF superblocks.
    ///
    /// Each 256-element superblock is 176 bytes:
    /// d(2)|dmin(2)|scales(12)|qh(32)|qs(128).
    pub fn from_quantized_q5k(
        ctx: &DeviceContext,
        packed_bytes: &[u8],
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        WeightFormat::GgufQ5K.validate_shape(rows, cols, 256)?;
        let expected = rows * cols * 11 / 16; // (cols/256) * 176 per row
        ensure!(
            packed_bytes.len() == expected,
            "Q5_K packed size {} != expected {} for rows={} cols={}",
            packed_bytes.len(),
            expected,
            rows,
            cols
        );

        let qw: CudaSlice<i8> = ctx
            .stream
            .clone_htod(unsafe {
                std::slice::from_raw_parts(packed_bytes.as_ptr().cast::<i8>(), packed_bytes.len())
            })
            .map_err(|e| anyhow!("H2D Q5_K packed upload failed: {}", e))?;
        let dummy_scales: CudaSlice<bf16> = ctx
            .stream
            .alloc_zeros::<bf16>(1)
            .map_err(|e| anyhow!("Alloc Q5_K dummy scales: {}", e))?;
        let dummy = ctx
            .stream
            .alloc_zeros::<bf16>(1)
            .map_err(|e| anyhow!("Alloc dummy: {}", e))?;
        Ok(Self {
            data: dummy,
            rows,
            cols,
            weight_format: WeightFormat::GgufQ5K,
            qweight: Some(qw),
            qscales: Some(dummy_scales),
            dsv4_scales: None,
            dsv4_scale_rows: 0,
            dsv4_scale_cols: 0,
            group_size: 256,
            marlin_packed: None,
            marlin_scales: None,
            marlin_channel_scales: None,
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

    /// Create from INT2 packed quantized weight + bf16 scales.
    /// Weight data is packed: 4 int2 values per byte → [rows, cols/4] bytes.
    pub fn from_quantized_int2(
        ctx: &DeviceContext,
        packed_data: &[u8],
        scales_data: &[bf16],
        rows: usize,
        cols: usize,
        group_size: usize,
    ) -> Result<Self> {
        WeightFormat::W2A16.validate_shape(rows, cols, group_size)?;
        ensure!(
            cols.is_multiple_of(4),
            "W2A16 requires cols % 4 == 0, got {cols}"
        );
        ensure!(packed_data.len() == rows * cols / 4);
        let num_groups = cols / group_size;
        ensure!(scales_data.len() == rows * num_groups);

        // Upload packed data directly (native W2 kernel handles bit extraction)
        let qw: CudaSlice<i8> = ctx
            .stream
            .clone_htod(unsafe {
                std::slice::from_raw_parts(packed_data.as_ptr().cast::<i8>(), packed_data.len())
            })
            .map_err(|e| anyhow!("H2D qweight int2 failed: {}", e))?;
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
            weight_format: WeightFormat::W2A16,
            qweight: Some(qw),
            qscales: Some(qs),
            dsv4_scales: None,
            dsv4_scale_rows: 0,
            dsv4_scale_cols: 0,
            group_size,
            marlin_packed: None,
            marlin_scales: None,
            marlin_channel_scales: None,
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
            && (self.qweight.is_some() || self.tq_packed.is_some() || self.marlin_packed.is_some())
    }

    /// Whether this matrix is plain BF16 with no packed side buffers.
    pub fn is_dense_bf16(&self) -> bool {
        self.weight_format == WeightFormat::DenseBf16
            && self.qweight.is_none()
            && self.tq_packed.is_none()
    }

    #[must_use]
    pub fn weight_format(&self) -> WeightFormat {
        self.weight_format
    }

    /// Whether this matrix has Marlin-repacked weights for fast prefill GEMM.
    pub fn has_marlin(&self) -> bool {
        self.marlin_packed.is_some()
    }

    /// Whether this matrix exposes a W4A8 Marlin runtime path.
    pub fn is_marlin_w4a8(&self) -> bool {
        self.weight_format == WeightFormat::MarlinW4A8 || self.is_hybrid_w4_marlin()
    }

    /// Whether this matrix carries both W4A16 and W4A8 Marlin side tensors.
    pub fn is_hybrid_w4_marlin(&self) -> bool {
        self.hybrid_w4a8_qweight.is_some()
    }

    /// Whether the hybrid matrix has the PF8.2 preprocessed W4 side tensor
    /// needed by the W4+FP8 prefill kernel.
    pub fn has_hybrid_w4_fp8_prefill(&self) -> bool {
        self.hybrid_w4_fp8_qweight.is_some()
    }

    /// Whether this matrix uses TurboQuant packed weight storage.
    pub fn has_tq(&self) -> bool {
        self.tq_packed.is_some()
    }

    /// Create from TurboQuant packed weights on GPU.
    ///
    /// Weights stay packed at runtime; dequant happens in the fused GEMV kernel
    /// (decode) or via bulk dequant + cuBLAS GEMM (prefill).
    #[allow(clippy::too_many_arguments)]
    pub fn from_quantized_tq(
        ctx: &DeviceContext,
        packed: &[u8],
        scales: &[u8], // f16 as raw bytes
        signs: &[i8],
        centroids: &CudaSlice<f32>,
        rows: usize,
        cols: usize,
        group_size: usize,
        bits: u8,
    ) -> Result<Self> {
        let tq_p = ctx
            .stream
            .clone_htod(packed)
            .map_err(|e| anyhow!("H2D tq_packed failed: {}", e))?;
        let tq_s: CudaSlice<u16> = ctx
            .stream
            .clone_htod(unsafe {
                std::slice::from_raw_parts(scales.as_ptr().cast::<u16>(), scales.len() / 2)
            })
            .map_err(|e| anyhow!("H2D tq_scales failed: {}", e))?;
        let tq_sg = ctx
            .stream
            .clone_htod(signs)
            .map_err(|e| anyhow!("H2D tq_signs failed: {}", e))?;
        let tq_c = ctx
            .stream
            .clone_dtod(centroids)
            .map_err(|e| anyhow!("D2D tq_centroids failed: {}", e))?;
        let dummy = ctx
            .stream
            .alloc_zeros::<bf16>(1)
            .map_err(|e| anyhow!("Alloc dummy: {}", e))?;
        Ok(Self {
            data: dummy,
            rows,
            cols,
            weight_format: WeightFormat::TurboQuant,
            qweight: None,
            qscales: None,
            dsv4_scales: None,
            dsv4_scale_rows: 0,
            dsv4_scale_cols: 0,
            group_size,
            marlin_packed: None,
            marlin_scales: None,
            marlin_channel_scales: None,
            hybrid_w4a8_qweight: None,
            hybrid_w4a8_s_channel: None,
            hybrid_w4a8_s_group: None,
            hybrid_w4_fp8_qweight: None,
            tq_packed: Some(tq_p),
            tq_scales: Some(tq_s),
            tq_signs: Some(tq_sg),
            tq_centroids: Some(tq_c),
            tq_bits: bits,
        })
    }

    /// Repack W4 weights to Marlin tile layout for fast prefill.
    /// Our format: [N, K/2] uint8 packed (lo/hi nibble = even/odd elements)
    /// Marlin format: tiled int32 layout optimized for tensor core MMA.
    /// Also transposes scales from [N, K/group_size] bf16 → [K/group_size, N] fp16.
    pub fn repack_for_marlin(&mut self, ctx: &DeviceContext) -> Result<()> {
        if self.weight_format != WeightFormat::W4A16
            || self.qweight.is_none()
            || self.qscales.is_none()
        {
            return Ok(()); // Only for W4
        }
        let n = self.rows; // output dim
        let k = self.cols; // input dim

        // Skip if dimensions not Marlin-compatible (need K%16==0, N%64==0)
        if !k.is_multiple_of(16) || !n.is_multiple_of(64) {
            log::warn!("Marlin repack skipped: [{n}x{k}] not tile-aligned (need K%16==0, N%64==0)");
            return Ok(());
        }

        // Step 1: Convert our [N, K/2] uint8 → GPTQ [K/8, N] int32 on CPU
        let qw = self.qweight.as_ref().unwrap();
        let packed_host: Vec<i8> = ctx
            .stream
            .clone_dtoh(qw)
            .map_err(|e| anyhow!("D2H qweight: {}", e))?;
        let packed: &[u8] = unsafe {
            std::slice::from_raw_parts(packed_host.as_ptr().cast::<u8>(), packed_host.len())
        };

        // GPTQ format: qweight[k/8, n] = 8 nibbles packed into int32
        // bit position (k%8)*4 holds the 4-bit unsigned value for element (k, n)
        let gptq_rows = k / 8;
        let mut gptq = vec![0u32; gptq_rows * n];
        for row_n in 0..n {
            for col_k in 0..k {
                let byte_idx = row_n * (k / 2) + col_k / 2;
                let nibble = if col_k % 2 == 0 {
                    packed[byte_idx] & 0x0F
                } else {
                    packed[byte_idx] >> 4
                };
                let gptq_row = col_k / 8;
                let bit_pos = (col_k % 8) * 4;
                gptq[gptq_row * n + row_n] |= (nibble as u32) << bit_pos;
            }
        }

        // Upload GPTQ weights as raw bytes
        let gptq_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(gptq.as_ptr().cast::<u8>(), gptq.len() * 4) };
        let gptq_gpu: CudaSlice<u8> = ctx
            .stream
            .clone_htod(gptq_bytes)
            .map_err(|e| anyhow!("H2D GPTQ: {}", e))?;

        // Allocate Marlin output buffer (same byte count as GPTQ: K*N/2 bytes)
        let marlin_bytes = k * n / 2;
        let mut marlin_gpu: CudaSlice<u8> = ctx
            .stream
            .alloc_zeros(marlin_bytes)
            .map_err(|e| anyhow!("Alloc Marlin: {}", e))?;

        // Step 2: GPTQ → Marlin repack on GPU
        {
            let (gptq_ptr, _g1) = gptq_gpu.device_ptr(&ctx.stream);
            let (marlin_ptr, _g2) = marlin_gpu.device_ptr_mut(&ctx.stream);
            unsafe {
                ffi::gptq_marlin_repack_cuda(
                    gptq_ptr as *const u32,
                    marlin_ptr as *mut u32,
                    k as i32,
                    n as i32,
                    ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|e| anyhow!("Marlin repack failed: {:?}", e))?;
            }
        }

        // Step 3: Transpose + convert scales [N, K/gs] bf16 → [K/gs, N] fp16
        let qs = self.qscales.as_ref().unwrap();
        let scales_host: Vec<bf16> = ctx
            .stream
            .clone_dtoh(qs)
            .map_err(|e| anyhow!("D2H scales: {}", e))?;
        let num_groups = k / self.group_size;
        let mut scales_fp16 = vec![0u16; num_groups * n];
        for row_n in 0..n {
            for g in 0..num_groups {
                let bf = scales_host[row_n * num_groups + g];
                let f = f32::from(bf);
                let fp16 = half::f16::from_f32(f);
                scales_fp16[g * n + row_n] = fp16.to_bits();
            }
        }
        let scales_gpu: CudaSlice<u16> = ctx
            .stream
            .clone_htod(&scales_fp16)
            .map_err(|e| anyhow!("H2D Marlin scales: {}", e))?;

        self.marlin_packed = Some(marlin_gpu);
        self.marlin_scales = Some(scales_gpu);

        Ok(())
    }

    #[allow(clippy::cast_ptr_alignment)]
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
        // NOTE: This assumes a little-endian host. Safetensors are little-endian.
        // On a big-endian machine, this will be incorrect. A full solution would
        // involve byte-swapping.
        let slice =
            unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<bf16>(), rows * cols) };
        let gpu_data = ctx
            .stream
            .clone_htod(slice)
            .map_err(|e| anyhow!("H2D copy failed: {}", e))?;
        Ok(Self {
            data: gpu_data,
            rows,
            cols,
            weight_format: WeightFormat::DenseBf16,
            qweight: None,
            qscales: None,
            dsv4_scales: None,
            dsv4_scale_rows: 0,
            dsv4_scale_cols: 0,
            group_size: 0,
            marlin_packed: None,
            marlin_scales: None,
            marlin_channel_scales: None,
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
            qscales: None,
            dsv4_scales: None,
            dsv4_scale_rows: 0,
            dsv4_scale_cols: 0,
            group_size: 0,
            marlin_packed: None,
            marlin_scales: None,
            marlin_channel_scales: None,
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

    /// Concatenate multiple matrices vertically (stacking rows).
    /// All matrices must have the same number of columns.
    /// Result has rows = sum of all input rows, cols = shared cols.
    pub fn concat_rows(ctx: &DeviceContext, matrices: &[&DeviceMatrix]) -> Result<Self> {
        assert!(!matrices.is_empty(), "concat_rows: empty input");
        let cols = matrices[0].cols;
        for m in matrices {
            assert_eq!(m.cols, cols, "concat_rows: cols mismatch");
        }
        let total_rows: usize = matrices.iter().map(|m| m.rows).sum();

        // Quantized weights use separate GEMVs (not merged), so skip the
        // expensive bf16 concat — just allocate a 1-element dummy.
        if matrices[0].is_quantized() {
            let dummy = ctx
                .stream
                .alloc_zeros::<bf16>(1)
                .map_err(|e| anyhow!("concat_rows dummy alloc: {e}"))?;
            return Ok(Self {
                data: dummy,
                rows: total_rows,
                cols,
                weight_format: WeightFormat::DenseBf16,
                qweight: None,
                qscales: None,
                dsv4_scales: None,
                dsv4_scale_rows: 0,
                dsv4_scale_cols: 0,
                group_size: 0,
                marlin_packed: None,
                marlin_scales: None,
                marlin_channel_scales: None,
                hybrid_w4a8_qweight: None,
                hybrid_w4a8_s_channel: None,
                hybrid_w4a8_s_group: None,
                hybrid_w4_fp8_qweight: None,
                tq_packed: None,
                tq_scales: None,
                tq_signs: None,
                tq_centroids: None,
                tq_bits: 0,
            });
        }

        let total_elements = total_rows * cols;
        let mut merged: CudaSlice<bf16> = ctx
            .stream
            .alloc_zeros(total_elements)
            .map_err(|e| anyhow!("concat_rows alloc failed: {e}"))?;

        let mut offset = 0usize;
        for m in matrices {
            let n = m.rows * m.cols;
            ctx.stream
                .memcpy_dtod(&m.data, &mut merged.slice_mut(offset..offset + n))
                .map_err(|e| anyhow!("concat_rows D2D copy failed: {e}"))?;
            offset += n;
        }

        Ok(Self {
            data: merged,
            rows: total_rows,
            cols,
            weight_format: WeightFormat::DenseBf16,
            qweight: None,
            qscales: None,
            dsv4_scales: None,
            dsv4_scale_rows: 0,
            dsv4_scale_cols: 0,
            group_size: 0,
            marlin_packed: None,
            marlin_scales: None,
            marlin_channel_scales: None,
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
}

/// Extract and cache a raw device pointer from a CudaSlice.
/// Calls device_ptr() once -- amortized over thousands of decode steps.
pub fn cache_ptr<T>(slice: &CudaSlice<T>, ctx: &DeviceContext) -> RawDevicePtr<T> {
    use cudarc::driver::DevicePtr;
    let (ptr, _sync) = slice.device_ptr(&ctx.stream);
    RawDevicePtr {
        ptr,
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
    fn device_ordinal_override_is_thread_local_and_nested() {
        assert_eq!(scoped_device_ordinal_override(), None);
        let outer = with_device_ordinal_override(2, || {
            assert_eq!(scoped_device_ordinal_override(), Some(2));
            let inner = with_device_ordinal_override(7, || scoped_device_ordinal_override());
            assert_eq!(inner, Some(7));
            scoped_device_ordinal_override()
        });
        assert_eq!(outer, Some(2));
        assert_eq!(scoped_device_ordinal_override(), None);
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
    fn kernel_alignment_names_scale_layout_explicitly() {
        let w4 = WeightFormat::W4A16.kernel_alignment(128);
        assert_eq!(w4.weight_layout, "wN.row_major.group_packed");
        assert_eq!(w4.scale_layout, "bf16[row, k/group_size]");
        assert_eq!(w4.k_multiple, 128);

        let q4k = WeightFormat::GgufQ4K.kernel_alignment(256);
        assert_eq!(q4k.weight_layout, "gguf.qk.row_major.superblock256");
        assert_eq!(q4k.scale_layout, "embedded.superblock");
        assert_eq!(q4k.k_multiple, 256);
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
